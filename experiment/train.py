"""Training entrypoint for the manifold neural SDE covariance forecaster."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from georax import SPD

from datasets.spd.dataset import CovarianceDataset
from experiment.config import ExperimentConfig, load_config
from experiment.losses import (
    LossFn,
    PyTree,
    affine_invariant_distance,
    make_georax_chart_loss,
    make_riemannian_distance_metric,
    project_to_spd,
    replace_masked_spd_examples,
)
from experiment.factories import make_loader, make_model, make_prediction_fn
from results.plots import (
    plot_eigenvalue_spectrum,
    plot_riemannian_distance,
    plot_training_curves,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def fit(
    model: eqx.Module,
    *,
    loss_fn: LossFn,
    config: ExperimentConfig,
    val_metric_fn: LossFn | None = None,
    val_metric_name: str = "val_metric",
) -> tuple[eqx.Module, dict[str, list[float]]]:
    train_loader = make_loader(config, "train")
    train_loader_next = jax.jit(train_loader.next)

    val_loader = make_loader(config, "val")
    val_loader_next = jax.jit(val_loader.next)

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

    return best_model, history


def evaluate(
    model: eqx.Module,
    *,
    loss_fn: LossFn,
    config: ExperimentConfig,
    seed_offset: int = 0,
) -> float:
    loader = make_loader(config, "test")
    loader_next = jax.jit(loader.next)

    @eqx.filter_jit
    def eval_step(
        current_model: eqx.Module,
        batch: PyTree,
        mask: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        return loss_fn(current_model, batch, mask, key)

    key = jax.random.key(config.seed + seed_offset)
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
    prediction_fn: Callable[[eqx.Module, jax.Array, jax.Array], jax.Array],
    config: ExperimentConfig,
    seed_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    loader = make_loader(config, "test")
    loader_next = jax.jit(loader.next)

    @eqx.filter_jit
    def predict_step(
        current_model: eqx.Module,
        batch: PyTree,
        key: jax.Array,
    ) -> jax.Array:
        return prediction_fn(current_model, batch["context_spd"], key)

    key = jax.random.key(config.seed + seed_offset)
    key, loader_key = jax.random.split(key)
    state = loader.init_state(loader_key)

    predicted_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []

    for _ in range(loader.steps_per_epoch):
        key, step_key = jax.random.split(key)
        batch, state, mask = loader_next(state)
        sanitized_batch = {
            **batch,
            "context_spd": replace_masked_spd_examples(batch["context_spd"], mask),
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


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _model_name(model: eqx.Module) -> str:
    raw_name = getattr(model, "name", model.__class__.__name__)
    if callable(raw_name):
        raw_name = raw_name()
    name = str(raw_name).strip()
    if not name:
        name = model.__class__.__name__
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return sanitized or model.__class__.__name__.lower()


def _make_output_dir(model: eqx.Module) -> Path:
    timestamp = datetime.now().astimezone().strftime("%H%M%S_%Y%m%d")
    return PROJECT_ROOT / "results" / f"{_model_name(model)}_{timestamp}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to experiment config TOML")
    args = parser.parse_args()

    config = load_config(args.config)

    train_dataset = CovarianceDataset(split="train")
    val_dataset = CovarianceDataset(split="val")
    test_dataset = CovarianceDataset(split="test")
    metadata = train_dataset.metadata()
    train_targets = jnp.asarray(train_dataset.as_array_dict()["target_spd"])

    geometry = SPD(metadata["n_stocks"])
    base_matrix = project_to_spd(
        jnp.mean(train_targets, axis=0),
        eps=config.min_eigenvalue,
    )

    model_key = jax.random.key(config.seed)
    model = make_model(config, metadata["n_stocks"], model_key)
    prediction_fn = make_prediction_fn()

    loss_fn = make_georax_chart_loss(
        input_key="context_spd",
        target_key="target_spd",
        prediction_fn=prediction_fn,
        geometry=geometry,
        base_matrix=base_matrix,
        beta=config.loss_beta,
        eps=config.min_eigenvalue,
    )
    riemannian_metric_fn = make_riemannian_distance_metric(
        input_key="context_spd",
        target_key="target_spd",
        prediction_fn=prediction_fn,
        eps=config.min_eigenvalue,
    )

    output_dir = _make_output_dir(model)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        f"hidden_dim={config.hidden_dim}",
        f"ctx_dim={config.ctx_dim}",
        f"n_steps={config.n_steps}",
        f"solver={config.solver}",
        f"loss_beta={config.loss_beta}",
        flush=True,
    )
    print(
        "chart base:",
        "projected training mean",
        f"eps={config.min_eigenvalue}",
        flush=True,
    )

    best_model, history = fit(
        model,
        loss_fn=loss_fn,
        config=config,
        val_metric_fn=riemannian_metric_fn,
        val_metric_name="val_riemannian_dist",
    )

    test_loss = evaluate(best_model, loss_fn=loss_fn, config=config, seed_offset=1)
    test_riemannian = evaluate(
        best_model, loss_fn=riemannian_metric_fn, config=config, seed_offset=2
    )
    predicted, actual = predict_dataset(
        best_model, prediction_fn=prediction_fn, config=config, seed_offset=3
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

    if not config.skip_plots:
        plot_training_curves({"nsde": history}, output_dir / "training_curves.png")
        plot_riemannian_distance(
            {"nsde": riemannian_distances},
            output_dir / "riemannian_distance.png",
        )
        plot_eigenvalue_spectrum(
            predicted=predicted,
            actual=actual,
            save_path=output_dir / "eigenvalue_spectrum.png",
            model_name=model.name,
        )

    print(f"saved artifacts to {output_dir}", flush=True)
    print(
        f"test_loss={test_loss:.3e} test_riemannian_dist={test_riemannian:.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
