"""Cyreal dataset utilities for covariance-matrix forecasting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from cyreal.datasets.dataset_protocol import DatasetProtocol
from cyreal.datasets.utils import to_host_jax_array
from cyreal.sources import ArraySource, DiskSource

from datasets.spd.download import (
    DEFAULT_END,
    DEFAULT_START,
    DEFAULT_TICKERS,
    download_prices,
)
from datasets.spd.preprocessing import (
    compute_log_returns,
    compute_rolling_covariances,
)


@dataclass
class CovarianceDataset(DatasetProtocol):
    """One-step-ahead forecasting dataset for SPD covariance time series.

    Each sample is a dict with:
    - ``context_spd``: ``(context_length, n_stocks, n_stocks)``
    - ``target_spd``: ``(n_stocks, n_stocks)``
    """

    split: Literal["train", "val", "test"] = "train"
    context_length: int = 20
    covariance_window: int = 20
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    tickers: tuple[str, ...] = tuple(DEFAULT_TICKERS)
    start: str = DEFAULT_START
    end: str = DEFAULT_END
    cache_dir: str | None = None
    force_refresh: bool = False
    shrinkage: bool = True
    min_eigenvalue: float = 1e-6
    ordering: Literal["sequential", "shuffle"] = field(init=False)

    def __post_init__(self) -> None:
        self.ordering = "shuffle" if self.split == "train" else "sequential"

        covariances = _load_covariances(
            tickers=list(self.tickers),
            start=self.start,
            end=self.end,
            cache_dir=self.cache_dir,
            force_refresh=self.force_refresh,
            covariance_window=self.covariance_window,
            shrinkage=self.shrinkage,
            min_eigenvalue=self.min_eigenvalue,
        )
        contexts, targets = _prepare_covariance_windows(
            covariances,
            split=self.split,
            context_length=self.context_length,
            train_fraction=self.train_fraction,
            val_fraction=self.val_fraction,
        )

        self._contexts = to_host_jax_array(contexts.astype(np.float32))
        self._targets = to_host_jax_array(targets.astype(np.float32))
        self._n_stocks = int(targets.shape[-1])

    def __len__(self) -> int:
        return int(self._contexts.shape[0])

    def __getitem__(self, index: int) -> dict[str, jax.Array]:
        return {
            "context_spd": self._contexts[index],
            "target_spd": self._targets[index],
        }

    def as_array_dict(self) -> dict[str, jax.Array]:
        return {
            "context_spd": self._contexts,
            "target_spd": self._targets,
        }

    def metadata(self) -> dict[str, int]:
        return {
            "n_stocks": self._n_stocks,
            "vech_dim": self._n_stocks * (self._n_stocks + 1) // 2,
            "context_length": self.context_length,
            "dataset_size": len(self),
        }

    def make_array_source(self) -> ArraySource:
        return ArraySource(self.as_array_dict(), ordering=self.ordering)

    def make_disk_source(self, prefetch_size: int = 128) -> DiskSource:
        return _make_covariance_disk_source(
            contexts=np.asarray(self._contexts),
            targets=np.asarray(self._targets),
            ordering=self.ordering,
            prefetch_size=prefetch_size,
        )


def _load_covariances(
    *,
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: str | None,
    force_refresh: bool,
    covariance_window: int,
    shrinkage: bool,
    min_eigenvalue: float,
) -> np.ndarray:
    prices, _, _ = download_prices(
        tickers=tickers,
        start=start,
        end=end,
        cache_dir=cache_dir,
        force_refresh=force_refresh,
    )
    returns = compute_log_returns(prices)
    return compute_rolling_covariances(
        returns,
        window=covariance_window,
        shrinkage=shrinkage,
        min_eigenvalue=min_eigenvalue,
    )


def _prepare_covariance_windows(
    covariances: np.ndarray,
    *,
    split: Literal["train", "val", "test"],
    context_length: int,
    train_fraction: float,
    val_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    split_covariances = _select_covariance_split(
        covariances,
        split=split,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        context_length=context_length,
    )
    return _make_context_target_pairs(
        split_covariances,
        context_length=context_length,
    )


def _select_covariance_split(
    covariances: np.ndarray,
    *,
    split: Literal["train", "val", "test"],
    train_fraction: float,
    val_fraction: float,
    context_length: int,
) -> np.ndarray:
    n = int(len(covariances))
    if n <= 0:
        raise ValueError("covariances must be non-empty.")
    if context_length <= 0:
        raise ValueError("context_length must be positive.")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1).")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1).")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be < 1.")

    train_end = min(max(int(n * train_fraction), 1), n)
    if val_fraction > 0.0:
        val_end = min(max(int(n * (train_fraction + val_fraction)), train_end + 1), n)
    else:
        val_end = train_end

    if split == "train":
        start, end = 0, train_end
    elif split == "val":
        if val_fraction == 0.0:
            raise ValueError("val_fraction must be > 0 when split='val'.")
        start, end = max(train_end - context_length, 0), val_end
    else:
        start, end = max(val_end - context_length, 0), n

    split_covariances = covariances[start:end]
    if len(split_covariances) <= context_length:
        raise ValueError(
            f"Split '{split}' is too short for context_length={context_length}. "
            f"Got {len(split_covariances)} covariance matrices after splitting."
        )
    return split_covariances


def _make_context_target_pairs(
    covariances: np.ndarray,
    *,
    context_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    num_samples = len(covariances) - context_length
    if num_samples <= 0:
        raise ValueError(
            f"Need more than context_length={context_length} covariance matrices, "
            f"got {len(covariances)}."
        )

    row_idx = np.arange(num_samples)[:, None] + np.arange(context_length)[None, :]
    contexts = covariances[row_idx]  # (num_samples, context_length, N, N)
    targets = covariances[context_length:]  # (num_samples, N, N)
    return contexts, targets


def _make_covariance_disk_source(
    *,
    contexts: np.ndarray,
    targets: np.ndarray,
    ordering: Literal["sequential", "shuffle"],
    prefetch_size: int,
) -> DiskSource:
    contexts_np = np.asarray(contexts)
    targets_np = np.asarray(targets)

    if contexts_np.ndim != 4:
        raise ValueError(
            "contexts must have shape (num_samples, context_length, n_stocks, n_stocks)."
        )
    if targets_np.ndim != 3:
        raise ValueError("targets must have shape (num_samples, n_stocks, n_stocks).")
    if contexts_np.shape[0] != targets_np.shape[0]:
        raise ValueError("contexts and targets must have the same number of samples.")

    def _read_sample(index: int | np.ndarray) -> dict[str, np.ndarray]:
        idx = int(np.asarray(index))
        return {
            "context_spd": contexts_np[idx],
            "target_spd": targets_np[idx],
        }

    sample_spec = {
        "context_spd": jax.ShapeDtypeStruct(
            shape=tuple(int(x) for x in contexts_np.shape[1:]),
            dtype=jnp.float32,
        ),
        "target_spd": jax.ShapeDtypeStruct(
            shape=tuple(int(x) for x in targets_np.shape[1:]),
            dtype=jnp.float32,
        ),
    }

    return DiskSource(
        length=int(contexts_np.shape[0]),
        sample_fn=_read_sample,
        sample_spec=sample_spec,
        ordering=ordering,
        prefetch_size=prefetch_size,
    )
