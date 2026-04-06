"""Plot the geodesic distance from the Fréchet mean over time (the 'true trajectory')."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

from datasets.spd.dataset import CovarianceDataset
from datasets.spd.download import DEFAULT_TICKERS

# ---------------------------------------------------------------------------
# Geometry helpers (affine-invariant metric on SPD)
# ---------------------------------------------------------------------------


def _eigh_clamp(M: np.ndarray, eps: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    n = M.shape[-1]
    lam, Q = eigh(M + eps * np.eye(n))
    return np.maximum(lam, eps), Q


def spd_sqrt_inv_sqrt(S: np.ndarray, eps: float = 1e-10):
    lam, Q = _eigh_clamp(S, eps)
    sqrt_S = (Q * np.sqrt(lam)) @ Q.T
    inv_sqrt_S = (Q * (1.0 / np.sqrt(lam))) @ Q.T
    return sqrt_S, inv_sqrt_S


def logmap(base: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Log_base(target) under the affine-invariant metric."""
    sqrt_B, inv_sqrt_B = spd_sqrt_inv_sqrt(base)
    M = inv_sqrt_B @ target @ inv_sqrt_B
    M = 0.5 * (M + M.T)
    lam, Q = _eigh_clamp(M)
    log_M = (Q * np.log(lam)) @ Q.T
    tangent = sqrt_B @ log_M @ sqrt_B
    return 0.5 * (tangent + tangent.T)


def frechet_mean(
    matrices: np.ndarray, max_iter: int = 100, tol: float = 1e-10
) -> np.ndarray:
    """Karcher mean on SPD via Riemannian gradient descent."""
    M = matrices.mean(axis=0)
    M = 0.5 * (M + M.T)
    for _ in range(max_iter):
        tangents = np.stack([logmap(M, A) for A in matrices])
        V = tangents.mean(axis=0)
        norm_V = np.linalg.norm(V, "fro")
        if norm_V < tol:
            break
        # Exp_M(V)
        sqrt_M, inv_sqrt_M = spd_sqrt_inv_sqrt(M)
        W = inv_sqrt_M @ V @ inv_sqrt_M
        W = 0.5 * (W + W.T)
        lam, Q = eigh(W)
        exp_W = (Q * np.exp(lam)) @ Q.T
        M = sqrt_M @ exp_W @ sqrt_M
        M = 0.5 * (M + M.T)
    return M


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

print("Loading dataset...")
ds_train = CovarianceDataset(split="train")
ds_val = CovarianceDataset(split="val")
ds_test = CovarianceDataset(split="test")

train_targets = np.asarray(ds_train.as_array_dict()["target_spd"], dtype=np.float64)
val_targets = np.asarray(ds_val.as_array_dict()["target_spd"], dtype=np.float64)
test_targets = np.asarray(ds_test.as_array_dict()["target_spd"], dtype=np.float64)

all_targets = np.concatenate([train_targets, val_targets, test_targets], axis=0)
n_train, n_val, n_test = len(train_targets), len(val_targets), len(test_targets)
T = len(all_targets)

print(f"Train: {n_train}  Val: {n_val}  Test: {n_test}  Total: {T}")

# ---------------------------------------------------------------------------
# Fréchet mean (computed on training targets only)
# ---------------------------------------------------------------------------

print("Computing Fréchet mean on training data...")
mu = frechet_mean(train_targets)

# ---------------------------------------------------------------------------
# Geodesic distance from mean over time
# ---------------------------------------------------------------------------

print("Computing tangent-space distances...")
distances = np.array([np.linalg.norm(logmap(mu, A), "fro") for A in all_targets])

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

days = np.arange(T)
train_end = n_train
val_end = n_train + n_val

fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(
    days,
    distances,
    color="black",
    linewidth=0.9,
    label="Geodesic distance from $\\mu_F$",
)

ax.axvspan(0, train_end, alpha=0.07, color="steelblue", label="Train")
ax.axvspan(train_end, val_end, alpha=0.07, color="orange", label="Val")
ax.axvspan(val_end, T, alpha=0.07, color="green", label="Test")

ax.set_xlabel("Trading day")
ax.set_ylabel("Tangent-space coordinate  $\\|\\log_{\\mu}(\\Sigma_t)\\|_F$")
ax.set_title("SPD trajectory in tangent space at Fréchet mean")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)

plt.tight_layout()
out_path = Path(__file__).parent / "spd_frechet_mean.png"
plt.savefig(out_path, dpi=150)
print(f"Saved → {out_path}")
plt.show()
