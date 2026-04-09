from typing import Callable, Literal

import equinox as eqx
import jax
from cyreal.loader import DataLoader
from cyreal.transforms import BatchTransform

from experiment.config import ExperimentConfig, Experiments


def make_loader(
    config: ExperimentConfig,
    split: Literal["train", "val", "test"],
) -> DataLoader:
    match config.experiment:
        case Experiments.SPD:
            from datasets.spd.dataset import CovarianceDataset

            source = CovarianceDataset(split=split).make_array_source()
            return DataLoader(
                [
                    source,
                    BatchTransform(
                        batch_size=config.batch_size,
                        drop_last=split == "train",
                    ),
                ]
            )
        case _:
            raise ValueError(f"Unsupported experiment: {config.experiment}")


def make_model(
    config: ExperimentConfig,
    n_stocks: int,
    key: jax.Array,
) -> eqx.Module:
    match config.experiment:
        case Experiments.SPD:
            from georax import CFEES25

            from models.nsde import ManifoldNeuralSDE

            return ManifoldNeuralSDE(
                n_stocks=n_stocks,
                hidden_dim=config.hidden_dim,
                ctx_dim=config.ctx_dim,
                n_steps=config.n_steps,
                dt=config.dt,
                solver=CFEES25(),
                diffusion_scale=config.diffusion_scale,
                key=key,
            )
        case _:
            raise ValueError(f"Unsupported experiment: {config.experiment}")


def make_prediction_fn() -> Callable[[eqx.Module, jax.Array, jax.Array], jax.Array]:
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
