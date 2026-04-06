"""Preprocessing: log-returns, rolling covariance, Ledoit-Wolf shrinkage."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_log_returns(prices: np.ndarray) -> np.ndarray:
    """Compute log-returns from price array.

    Args:
        prices: (T, N) array of prices.

    Returns:
        returns: (T-1, N) array of log-returns.
    """
    return np.diff(np.log(prices), axis=0)


def _sample_cov(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return mean-centered data X and unbiased sample covariance S."""
    X = returns - returns.mean(axis=0, keepdims=True)
    S = (X.T @ X) / (len(returns) - 1)
    return X, S


def ledoit_wolf_shrinkage(
    returns_window: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Analytical Ledoit-Wolf (2004) shrinkage estimator.

    Shrinks the sample covariance toward scaled identity: F = (tr(S)/n) * I.

    Reference: Ledoit & Wolf (2004), "A well-conditioned estimator for
    large-dimensional covariance matrices", JMVA.

    Args:
        returns_window: (T, N) returns matrix for one window.

    Returns:
        shrunk_cov: (N, N) shrinkage estimator.
        alpha: optimal shrinkage intensity in [0, 1].
    """
    T, N = returns_window.shape
    X, S = _sample_cov(returns_window)

    mu = np.trace(S) / N
    F = mu * np.eye(N)

    delta = S - F
    outer_products = X[:, :, None] * X[:, None, :]  # (T, N, N): x_t x_t^T per row
    beta = np.sum((outer_products - S) ** 2) / T**2

    delta_sq = np.sum(delta**2)
    alpha = min(beta / delta_sq, 1.0) if delta_sq > 0 else 1.0

    return alpha * F + (1.0 - alpha) * S, alpha


def compute_rolling_covariances(
    returns: np.ndarray,
    window: int = 20,
    shrinkage: bool = True,
    min_eigenvalue: float = 1e-6,
) -> np.ndarray:
    """Compute rolling-window covariance matrices.

    Args:
        returns: (T, N) array of log-returns.
        window: rolling window size.
        shrinkage: whether to apply Ledoit-Wolf shrinkage.
        min_eigenvalue: minimum eigenvalue threshold for SPD verification.

    Returns:
        covariances: (T - window + 1, N, N) array of SPD covariance matrices.
    """
    T, N = returns.shape
    n_matrices = T - window + 1
    covariances = np.empty((n_matrices, N, N), dtype=np.float64)

    for t in range(n_matrices):
        window_returns = returns[t : t + window]
        if shrinkage:
            cov, _ = ledoit_wolf_shrinkage(window_returns)
        else:
            _, cov = _sample_cov(window_returns)

        covariances[t] = cov

    verify_spd(covariances, min_eigenvalue)
    logger.info(
        f"Computed {n_matrices} covariance matrices ({N}x{N}), "
        f"shrinkage={'on' if shrinkage else 'off'}"
    )
    return covariances


def verify_spd(matrices: np.ndarray, min_eigenvalue: float = 1e-6) -> None:
    """Verify all matrices are symmetric positive definite.

    Args:
        matrices: (T, N, N) array of matrices.
        min_eigenvalue: minimum acceptable eigenvalue.

    Raises:
        ValueError: if any matrix is not SPD.
    """
    asymmetry = np.max(np.abs(matrices - matrices.transpose(0, 2, 1)))
    if asymmetry > 1e-12:
        raise ValueError(f"Matrices not symmetric: max asymmetry = {asymmetry:.2e}")

    eigenvalues = np.linalg.eigvalsh(matrices)
    min_eig = eigenvalues.min()
    if min_eig < min_eigenvalue:
        raise ValueError(
            f"Matrices not SPD: min eigenvalue = {min_eig:.2e} "
            f"(threshold: {min_eigenvalue:.2e})"
        )
