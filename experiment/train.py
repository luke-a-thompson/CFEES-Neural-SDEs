"""Training entrypoint for the manifold neural SDE covariance forecaster."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from cyreal.loader import DataLoader
from cyreal.sources import ArraySource
from cyreal.transforms import BatchTransform
from georax import CFEES25, CG2, SPD

from datasets.spd.dataset import CovarianceDataset
from models.nsde import ManifoldNeuralSDE
from results.plots import (
    plot_eigenvalue_spectrum,
    plot_riemannian_distance,
    plot_training_curves,
)

PyTree = Any
LossFn = Callable[[eqx.Module, PyTree, jax.Array, jax.Array], jax.Array]


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    loss_beta: float = 0.0
    seed: int = 0
    loader_backend: str | None = "cpu"
    use_disk_source: bool = False
    prefetch_size: int = 128
    pad_last_batch: bool = True
    drop_last: bool = False


def masked_mean(values: jax.Array, mask: jax.Array) -> jax.Array:
    mask = mask.astype(bool)
    safe_values = jnp.where(mask, values, jnp.zeros_like(values))
    return jnp.sum(safe_values) / jnp.maximum(jnp.sum(mask.astype(values.dtype)), 1.0)


def _replace_masked_spd_examples(values: jax.Array, mask: jax.Array) -> jax.Array:
    mask = mask.astype(bool)
    n = values.shape[-1]
    replacement = jnp.broadcast_to(jnp.eye(n, dtype=values.dtype), values.shape)
    expanded_mask = jnp.reshape(mask, mask.shape + (1,) * (values.ndim - 1))
    return jnp.where(expanded_mask, values, replacement)


def make_loader(
    dataset: Any,
    *,
    batch_size: int,
    use_disk_source: bool = False,
    prefetch_size: int = 128,
    pad_last_batch: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    if use_disk_source:
        source = dataset.make_disk_source(prefetch_size=prefetch_size)
    elif hasattr(dataset, "make_array_source"):
        source = dataset.make_array_source()
    else:
        ordering = getattr(dataset, "ordering", "shuffle")
        source = ArraySource(dataset.as_array_dict(), ordering=ordering)

    return DataLoader(
        [
            source,
            BatchTransform(
                batch_size=batch_size,
                pad_last_batch=pad_last_batch,
                drop_last=drop_last,
            ),
        ]
    )


def _jit_loader_next(
    loader: DataLoader, loader_backend: str | None
) -> Callable[[Any], tuple[PyTree, Any, jax.Array]]:
    if loader_backend is None:
        return jax.jit(loader.next)
    return jax.jit(loader.next, backend=loader_backend)


def make_supervised_mse_loss(
    *,
    input_key: str,
    target_key: str,
    prediction_fn: Callable[[eqx.Module, jax.Array, jax.Array], jax.Array]
    | None = None,
) -> LossFn:
    def default_prediction_fn(
        model: eqx.Module,
        inputs: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        del key
        return model(inputs)

    predict = prediction_fn or default_prediction_fn

    def loss_fn(
        model: eqx.Module,
        batch: PyTree,
        mask: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        inputs = _replace_masked_spd_examples(batch[input_key], mask)
        targets = _replace_masked_spd_examples(batch[target_key], mask)
        predictions = predict(model, inputs, key)
        axes = tuple(range(1, predictions.ndim))
        per_example_loss = jnp.mean((predictions - targets) ** 2, axis=axes)
        return masked_mean(per_example_loss, mask)

    return loss_fn


def _symmetrize(matrix: jax.Array) -> jax.Array:
    return 0.5 * (matrix + jnp.swapaxes(matrix, -1, -2))


def _eigh_clamp(matrix: jax.Array, eps: float = 1e-6) -> tuple[jax.Array, jax.Array]:
    eigenvalues, eigenvectors = jnp.linalg.eigh(_symmetrize(matrix))
    return jnp.clip(eigenvalues, a_min=eps), eigenvectors


def _project_to_spd(matrix: jax.Array, eps: float = 1e-6) -> jax.Array:
    eigenvalues, eigenvectors = _eigh_clamp(matrix, eps)
    return _symmetrize(
        (eigenvectors * eigenvalues) @ jnp.swapaxes(eigenvectors, -1, -2)
    )


def _matrix_sqrt(matrix: jax.Array, eps: float = 1e-6) -> jax.Array:
    eigenvalues, eigenvectors = _eigh_clamp(matrix, eps)
    return (eigenvectors * (eigenvalues**0.5)) @ jnp.swapaxes(eigenvectors, -1, -2)


def _matrix_inv_sqrt(matrix: jax.Array, eps: float = 1e-6) -> jax.Array:
    eigenvalues, eigenvectors = _eigh_clamp(matrix, eps)
    return (eigenvectors * (eigenvalues**-0.5)) @ jnp.swapaxes(eigenvectors, -1, -2)


def inverse_congruence_coords(
    geometry: SPD,
    base: jax.Array,
    point: jax.Array,
    eps: float = 1e-6,
) -> jax.Array:
    base = _project_to_spd(base, eps)
    point = _project_to_spd(point, eps)
    sqrt_base = _matrix_sqrt(base, eps)
    inv_sqrt_base = _matrix_inv_sqrt(base, eps)
    middle = sqrt_base @ point @ sqrt_base
    sqrt_middle = _matrix_sqrt(middle, eps)
    group_element = inv_sqrt_base @ sqrt_middle @ inv_sqrt_base
    eigenvalues, eigenvectors = _eigh_clamp(group_element, eps)
    lift = (eigenvectors * jnp.log(eigenvalues)) @ jnp.swapaxes(eigenvectors, -1, -2)
    return geometry._sym_to_coords(_symmetrize(lift))


def affine_invariant_distance(
    predicted: jax.Array,
    target: jax.Array,
    eps: float = 1e-6,
) -> jax.Array:
    inv_sqrt_target = _matrix_inv_sqrt(target, eps)
    aligned = inv_sqrt_target @ _symmetrize(predicted) @ inv_sqrt_target
    eigenvalues, _ = _eigh_clamp(aligned, eps)
    return jnp.linalg.norm(jnp.log(eigenvalues))


def make_georax_chart_loss(
    *,
    input_key: str,
    target_key: str,
    prediction_fn: Callable[[eqx.Module, jax.Array, jax.Array], jax.Array],
    geometry: SPD,
    base_matrix: jax.Array,
    beta: float = 0.0,
    eps: float = 1e-6,
) -> LossFn:
    def loss_fn(
        model: eqx.Module,
        batch: PyTree,
        mask: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        inputs = _replace_masked_spd_examples(batch[input_key], mask)
        targets = _replace_masked_spd_examples(batch[target_key], mask)
        predictions = prediction_fn(model, inputs, key)
        coord_targets = jax.vmap(
            lambda target: inverse_congruence_coords(geometry, base_matrix, target, eps)
        )(targets)
        coord_predictions = jax.vmap(
            lambda predicted: inverse_congruence_coords(
                geometry, base_matrix, predicted, eps
            )
        )(predictions)
        diff = coord_predictions - coord_targets
        per_example_loss = jnp.mean(diff**2, axis=-1)
        if beta > 0.0:
            distances = jax.vmap(
                lambda predicted, target: affine_invariant_distance(
                    predicted, target, eps
                )
            )(predictions, targets)
            per_example_loss = per_example_loss + beta * (distances**2)
        return masked_mean(per_example_loss, mask)

    return loss_fn


def make_riemannian_distance_metric(
    *,
    input_key: str,
    target_key: str,
    prediction_fn: Callable[[eqx.Module, jax.Array, jax.Array], jax.Array],
    eps: float = 1e-6,
) -> LossFn:
    def metric_fn(
        model: eqx.Module,
        batch: PyTree,
        mask: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        inputs = _replace_masked_spd_examples(batch[input_key], mask)
        targets = _replace_masked_spd_examples(batch[target_key], mask)
        predictions = prediction_fn(model, inputs, key)
        distances = jax.vmap(
            lambda predicted, target: affine_invariant_distance(predicted, target, eps)
        )(predictions, targets)
        return masked_mean(distances, mask)

    return metric_fn


def fit(
    model: eqx.Module,
    *,
    train_dataset: Any,
    loss_fn: LossFn,
    config: TrainConfig,
    val_dataset: Any | None = None,
    val_metric_fn: LossFn | None = None,
    val_metric_name: str = "val_metric",
) -> tuple[eqx.Module, dict[str, list[float]]]:
    train_loader = make_loader(
        train_dataset,
        batch_size=config.batch_size,
        use_disk_source=config.use_disk_source,
        prefetch_size=config.prefetch_size,
        pad_last_batch=config.pad_last_batch,
        drop_last=config.drop_last,
    )
    train_loader_next = _jit_loader_next(train_loader, config.loader_backend)

    val_loader = None
    val_loader_next = None
    if val_dataset is not None:
        val_loader = make_loader(
            val_dataset,
            batch_size=config.batch_size,
            use_disk_source=config.use_disk_source,
            prefetch_size=config.prefetch_size,
            pad_last_batch=config.pad_last_batch,
            drop_last=config.drop_last,
        )
        val_loader_next = _jit_loader_next(val_loader, config.loader_backend)

    @eqx.filter_value_and_grad
    def batch_loss(
        current_model: eqx.Module,
        batch: PyTree,
        mask: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        return loss_fn(current_model, batch, mask, key)

    @eqx.filter_jit
    def train_step(
        current_model: eqx.Module,
        batch: PyTree,
        mask: jax.Array,
        key: jax.Array,
    ) -> tuple[eqx.Module, jax.Array]:
        loss, grads = batch_loss(current_model, batch, mask, key)
        updates = jax.tree_util.tree_map(
            lambda grad: None if grad is None else -config.learning_rate * grad,
            grads,
        )
        new_model = eqx.apply_updates(current_model, updates)
        return new_model, loss

    @eqx.filter_jit
    def eval_step(
        current_model: eqx.Module,
        batch: PyTree,
        mask: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        return loss_fn(current_model, batch, mask, key)

    metric_step = None
    if val_metric_fn is not None:

        @eqx.filter_jit
        def metric_step(
            current_model: eqx.Module,
            batch: PyTree,
            mask: jax.Array,
            key: jax.Array,
        ) -> jax.Array:
            return val_metric_fn(current_model, batch, mask, key)

    key = jax.random.key(config.seed)
    key, train_key = jax.random.split(key)
    train_state = train_loader.init_state(train_key)

    val_state = None
    if val_loader is not None:
        key, val_key = jax.random.split(key)
        val_state = val_loader.init_state(val_key)

    history: dict[str, list[float]] = {
        "train": [],
        "val": [],
        "train_loss": [],
        "val_loss": [],
    }
    if val_metric_fn is not None:
        history[val_metric_name] = []

    best_model = model
    best_score = jnp.inf

    for epoch in range(config.epochs):
        train_loss = 0.0
        for _ in range(train_loader.steps_per_epoch):
            key, step_key = jax.random.split(key)
            batch, train_state, mask = train_loader_next(train_state)
            model, loss = train_step(model, batch, mask, step_key)
            train_loss += float(loss)
        train_loss /= train_loader.steps_per_epoch
        history["train"].append(train_loss)
        history["train_loss"].append(train_loss)

        log_line = f"epoch={epoch + 1}/{config.epochs} train_loss={train_loss:.3e}"

        if (
            val_loader is not None
            and val_loader_next is not None
            and val_state is not None
        ):
            val_loss = 0.0
            val_metric = 0.0
            for _ in range(val_loader.steps_per_epoch):
                key, step_key = jax.random.split(key)
                batch, val_state, mask = val_loader_next(val_state)
                loss = eval_step(model, batch, mask, step_key)
                val_loss += float(loss)
                if metric_step is not None:
                    metric_value = metric_step(model, batch, mask, step_key)
                    val_metric += float(metric_value)
            val_loss /= val_loader.steps_per_epoch
            history["val"].append(val_loss)
            history["val_loss"].append(val_loss)
            log_line += f" val_loss={val_loss:.3e}"

            score = val_loss
            if metric_step is not None:
                val_metric /= val_loader.steps_per_epoch
                history[val_metric_name].append(val_metric)
                log_line += f" {val_metric_name}={val_metric:.6f}"
                score = val_metric

            if score < float(best_score):
                best_score = jnp.asarray(score)
                best_model = model

        print(log_line, flush=True)

    return best_model if val_dataset is not None else model, history


def evaluate(
    model: eqx.Module,
    *,
    dataset: Any,
    loss_fn: LossFn,
    batch_size: int,
    seed: int = 0,
    loader_backend: str | None = "cpu",
    use_disk_source: bool = False,
    prefetch_size: int = 128,
    pad_last_batch: bool = True,
    drop_last: bool = False,
) -> float:
    loader = make_loader(
        dataset,
        batch_size=batch_size,
        use_disk_source=use_disk_source,
        prefetch_size=prefetch_size,
        pad_last_batch=pad_last_batch,
        drop_last=drop_last,
    )
    loader_next = _jit_loader_next(loader, loader_backend)

    @eqx.filter_jit
    def eval_step(
        current_model: eqx.Module,
        batch: PyTree,
        mask: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        return loss_fn(current_model, batch, mask, key)

    key = jax.random.key(seed)
    key, loader_key = jax.random.split(key)
    state = loader.init_state(loader_key)

    total_loss = 0.0
    for _ in range(loader.steps_per_epoch):
        key, step_key = jax.random.split(key)
        batch, state, mask = loader_next(state)
        total_loss += float(eval_step(model, batch, mask, step_key))

    return total_loss / loader.steps_per_epoch


def predict_dataset(
    model: eqx.Module,
    *,
    dataset: Any,
    prediction_fn: Callable[[eqx.Module, jax.Array, jax.Array], jax.Array],
    batch_size: int,
    seed: int = 0,
    loader_backend: str | None = "cpu",
    use_disk_source: bool = False,
    prefetch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    loader = make_loader(
        dataset,
        batch_size=batch_size,
        use_disk_source=use_disk_source,
        prefetch_size=prefetch_size,
        pad_last_batch=True,
        drop_last=False,
    )
    loader_next = _jit_loader_next(loader, loader_backend)

    @eqx.filter_jit
    def predict_step(
        current_model: eqx.Module,
        batch: PyTree,
        key: jax.Array,
    ) -> jax.Array:
        return prediction_fn(current_model, batch["context_spd"], key)

    key = jax.random.key(seed)
    key, loader_key = jax.random.split(key)
    state = loader.init_state(loader_key)

    predicted_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []

    for _ in range(loader.steps_per_epoch):
        key, step_key = jax.random.split(key)
        batch, state, mask = loader_next(state)
        sanitized_batch = {
            **batch,
            "context_spd": _replace_masked_spd_examples(batch["context_spd"], mask),
        }
        predictions = predict_step(model, sanitized_batch, step_key)
        valid_mask = np.asarray(jax.device_get(mask), dtype=bool)
        predicted_batches.append(np.asarray(jax.device_get(predictions))[valid_mask])
        target_batches.append(
            np.asarray(jax.device_get(batch["target_spd"]))[valid_mask]
        )

    return np.concatenate(predicted_batches, axis=0), np.concatenate(
        target_batches, axis=0
    )


def make_batch_prediction_fn() -> Callable[
    [eqx.Module, jax.Array, jax.Array], jax.Array
]:
    def prediction_fn(
        model: eqx.Module,
        inputs: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        sample_keys = jax.random.split(key, inputs.shape[0])
        return jax.vmap(lambda context, sample_key: model(context, sample_key))(
            inputs, sample_keys
        )

    return prediction_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--loss-beta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--loader-backend", type=str, default="cpu")
    parser.add_argument("--use-disk-source", action="store_true")
    parser.add_argument("--prefetch-size", type=int, default=128)
    parser.add_argument("--drop-last", action="store_true")

    parser.add_argument("--context-length", type=int, default=20)
    parser.add_argument("--covariance-window", type=int, default=20)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--disable-shrinkage", action="store_true")
    parser.add_argument("--min-eigenvalue", type=float, default=1e-6)
    parser.add_argument("--tickers", nargs="+", default=None)

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--ctx-dim", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=5)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--solver", type=str, default="cfees25")
    parser.add_argument("--diffusion-scale", type=float, default=1.0)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "nsde_run",
    )
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def _dataset_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "context_length": args.context_length,
        "covariance_window": args.covariance_window,
        "train_fraction": args.train_fraction,
        "val_fraction": args.val_fraction,
        "cache_dir": args.cache_dir,
        "force_refresh": args.force_refresh,
        "shrinkage": not args.disable_shrinkage,
        "min_eigenvalue": args.min_eigenvalue,
    }
    if args.tickers is not None:
        kwargs["tickers"] = tuple(args.tickers)
    return kwargs


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_kwargs = _dataset_kwargs(args)
    train_dataset = CovarianceDataset(split="train", **dataset_kwargs)
    val_dataset = CovarianceDataset(split="val", **dataset_kwargs)
    test_dataset = CovarianceDataset(split="test", **dataset_kwargs)

    metadata = train_dataset.metadata()
    prediction_fn = make_batch_prediction_fn()
    train_targets = jnp.asarray(train_dataset.as_array_dict()["target_spd"])
    geometry = SPD(metadata["n_stocks"])
    base_matrix = _project_to_spd(
        jnp.mean(train_targets, axis=0),
        eps=args.min_eigenvalue,
    )
    loss_fn = make_georax_chart_loss(
        input_key="context_spd",
        target_key="target_spd",
        prediction_fn=prediction_fn,
        geometry=geometry,
        base_matrix=base_matrix,
        beta=args.loss_beta,
        eps=args.min_eigenvalue,
    )
    riemannian_metric_fn = make_riemannian_distance_metric(
        input_key="context_spd",
        target_key="target_spd",
        prediction_fn=prediction_fn,
        eps=args.min_eigenvalue,
    )

    model_key = jax.random.key(args.seed)
    model = ManifoldNeuralSDE(
        n_stocks=metadata["n_stocks"],
        hidden_dim=args.hidden_dim,
        ctx_dim=args.ctx_dim,
        n_steps=args.n_steps,
        dt=args.dt,
        solver=CFEES25(),
        diffusion_scale=args.diffusion_scale,
        key=model_key,
    )

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        loss_beta=args.loss_beta,
        seed=args.seed,
        loader_backend=args.loader_backend,
        use_disk_source=args.use_disk_source,
        prefetch_size=args.prefetch_size,
        drop_last=args.drop_last,
    )

    print(
        "dataset sizes:",
        f"train={len(train_dataset)}",
        f"val={len(val_dataset)}",
        f"test={len(test_dataset)}",
        flush=True,
    )
    print(
        "model config:",
        f"n_stocks={metadata['n_stocks']}",
        f"context_length={metadata['context_length']}",
        f"hidden_dim={args.hidden_dim}",
        f"ctx_dim={args.ctx_dim}",
        f"n_steps={args.n_steps}",
        f"solver={args.solver}",
        f"loss_beta={args.loss_beta}",
        flush=True,
    )
    print(
        "chart base:",
        "projected training mean",
        f"eps={args.min_eigenvalue}",
        flush=True,
    )

    best_model, history = fit(
        model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        loss_fn=loss_fn,
        config=config,
        val_metric_fn=riemannian_metric_fn,
        val_metric_name="val_riemannian_dist",
    )

    test_loss = evaluate(
        best_model,
        dataset=test_dataset,
        loss_fn=loss_fn,
        batch_size=args.batch_size,
        seed=args.seed + 1,
        loader_backend=args.loader_backend,
        use_disk_source=args.use_disk_source,
        prefetch_size=args.prefetch_size,
    )
    test_riemannian = evaluate(
        best_model,
        dataset=test_dataset,
        loss_fn=riemannian_metric_fn,
        batch_size=args.batch_size,
        seed=args.seed + 2,
        loader_backend=args.loader_backend,
        use_disk_source=args.use_disk_source,
        prefetch_size=args.prefetch_size,
    )
    predicted, actual = predict_dataset(
        best_model,
        dataset=test_dataset,
        prediction_fn=prediction_fn,
        batch_size=args.batch_size,
        seed=args.seed + 3,
        loader_backend=args.loader_backend,
        use_disk_source=args.use_disk_source,
        prefetch_size=args.prefetch_size,
    )

    riemannian_distances = np.asarray(
        jax.device_get(
            jax.vmap(affine_invariant_distance)(
                jnp.asarray(predicted),
                jnp.asarray(actual),
            )
        )
    )

    eqx.tree_serialise_leaves(output_dir / "nsde.eqx", best_model)
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        predicted=predicted,
        actual=actual,
        riemannian_distance=riemannian_distances,
    )
    np.save(output_dir / "chart_base.npy", np.asarray(jax.device_get(base_matrix)))
    _save_json(output_dir / "history.json", history)
    _save_json(
        output_dir / "metrics.json",
        {
            "test_loss": float(test_loss),
            "test_riemannian_dist": float(test_riemannian),
        },
    )

    if not args.skip_plots:
        plot_training_curves({"nsde": history}, output_dir / "training_curves.png")
        plot_riemannian_distance(
            {"nsde": riemannian_distances},
            output_dir / "riemannian_distance.png",
        )
        plot_eigenvalue_spectrum(
            predicted=predicted,
            actual=actual,
            save_path=output_dir / "eigenvalue_spectrum.png",
            model_name="NSDE",
        )

    print(f"saved artifacts to {output_dir}", flush=True)
    print(
        f"test_loss={test_loss:.3e} test_riemannian_dist={test_riemannian:.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
