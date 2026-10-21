"""Space-time covariance estimation, regularisation and diagnostics.

Sample covariance from L training snapshots x_l (rows of X, shape (L, D)):

    R_hat = (1/L) * sum_l x_l x_l^H

Sample support: R_hat is singular for L < D. With L >= D the Reed-Mallett-
Brennan (RMB) result gives the expected SINR loss of sample-matrix MVDR
relative to the clairvoyant filter, E[rho] = (L + 2 - D) / (L + 1), i.e.
~3 dB at L ~ 2D. For a full-rank space-time dimension D = C * L_pulses this
is the central practical problem of full-rank STAP: e.g. 4 channels x 16
pulses = 64 requires ~128 homogeneous range bins. Diagonal loading makes the
matrix invertible but does *not* make it an accurate estimate; diagnostics
(eigenvalues, condition number, effective clutter rank) are reported so
ill-conditioning is never hidden.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


class InsufficientTrainingError(ValueError):
    """Raised when there are too few training snapshots for the requested estimate."""


def sample_covariance(X: np.ndarray, allow_rank_deficient: bool = False) -> np.ndarray:
    """Hermitian sample covariance from rows of X (L, D)."""
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"training matrix must be (L, D), got shape {X.shape}")
    L, D = X.shape
    if L == 0:
        raise InsufficientTrainingError("no training snapshots")
    if not np.all(np.isfinite(X)):
        raise ValueError("training snapshots contain non-finite values")
    if L < D and not allow_rank_deficient:
        raise InsufficientTrainingError(
            f"L={L} training snapshots < dimension D={D}: sample covariance is singular. "
            "Use more training cells, a reduced-dimension method, or allow_rank_deficient=True with loading."
        )
    R = (X.T @ X.conj()) / L  # R[i, j] = mean x_i x_j^*
    return hermitian_part(R)


def hermitian_part(R: np.ndarray) -> np.ndarray:
    return 0.5 * (R + R.conj().T)


def estimate_noise_floor(eigenvalues: np.ndarray, fraction: float = 0.25) -> float:
    """Mean of the smallest ``fraction`` of eigenvalues (heuristic noise-level estimate).

    Note: with small L the smallest sample eigenvalues are biased low.
    """
    ev = np.sort(np.real(eigenvalues))
    k = max(1, int(np.ceil(fraction * ev.size)))
    return float(max(ev[:k].mean(), 0.0))


def loading_level(
    R: np.ndarray, mode: str, value: float, noise_floor: float | None = None, noise_level: float | None = None
) -> float:
    """Absolute diagonal-loading level.

    Modes: ``absolute`` (value), ``trace_relative`` (value * tr(R)/D),
    ``noise_relative`` (value * known/estimated noise variance per element, the
    usual "loading-to-noise ratio"), ``noise_eig_relative`` (value * mean of the
    smallest 25% eigenvalues; biased low for small L, see Marchenko-Pastur).
    """
    D = R.shape[0]
    if mode == "noise_relative":
        if noise_level is None:
            raise ValueError("loading_mode 'noise_relative' requires a noise level")
        return float(value * noise_level)
    if mode == "absolute":
        return float(value)
    if mode == "trace_relative":
        return float(value * np.real(np.trace(R)) / D)
    if mode == "noise_eig_relative":
        nf = noise_floor if noise_floor is not None else estimate_noise_floor(np.linalg.eigvalsh(R))
        return float(value * nf)
    raise ValueError(f"unknown loading mode {mode!r}")


def diagonal_load(R: np.ndarray, level: float) -> np.ndarray:
    if level < 0:
        raise ValueError("loading level must be >= 0")
    return R + level * np.eye(R.shape[0], dtype=R.dtype)


def shrinkage_covariance(R: np.ndarray, rho: float) -> np.ndarray:
    """Convex shrinkage toward a scaled identity: (1-rho) R + rho (tr R / D) I.

    Explicitly labelled option (not used by the baseline unless requested).
    """
    if not 0 <= rho <= 1:
        raise ValueError("rho must be in [0, 1]")
    D = R.shape[0]
    return (1 - rho) * R + rho * (np.real(np.trace(R)) / D) * np.eye(D, dtype=R.dtype)


@dataclass
class CovarianceDiagnostics:
    dimension: int
    num_training: int
    eigenvalues: np.ndarray  # descending, real
    condition_number: float
    noise_floor: float
    effective_rank: int  # eigenvalues > 10 x noise floor (interference subspace estimate)
    rmb_expected_loss_db: float | None
    hermitian_error: float
    min_eigenvalue: float

    def as_dict(self) -> dict:
        return {
            "dimension": self.dimension,
            "num_training": self.num_training,
            "condition_number": self.condition_number,
            "noise_floor": self.noise_floor,
            "effective_rank": self.effective_rank,
            "rmb_expected_loss_db": self.rmb_expected_loss_db,
            "hermitian_error": self.hermitian_error,
            "min_eigenvalue": self.min_eigenvalue,
            "max_eigenvalue": float(self.eigenvalues[0]),
        }


def rmb_expected_loss_db(num_training: int, dimension: int) -> float | None:
    """Expected SINR loss of sample-matrix MVDR (RMB), in dB (negative). None if L < D."""
    L, D = num_training, dimension
    if L < D:
        return None
    return float(10 * np.log10((L + 2 - D) / (L + 1)))


def covariance_diagnostics(R: np.ndarray, num_training: int, rank_threshold: float = 10.0) -> CovarianceDiagnostics:
    herm_err = float(np.max(np.abs(R - R.conj().T))) if R.size else 0.0
    ev = np.linalg.eigvalsh(hermitian_part(R))[::-1]
    nf = estimate_noise_floor(ev)
    pos_min = ev[-1]
    cond = float(ev[0] / pos_min) if pos_min > 0 else float("inf")
    rank = int(np.sum(ev > rank_threshold * nf)) if nf > 0 else int(np.sum(ev > 0))
    return CovarianceDiagnostics(
        dimension=R.shape[0],
        num_training=num_training,
        eigenvalues=ev,
        condition_number=cond,
        noise_floor=nf,
        effective_rank=rank,
        rmb_expected_loss_db=rmb_expected_loss_db(num_training, R.shape[0]),
        hermitian_error=herm_err,
        min_eigenvalue=float(pos_min),
    )


def brennan_rank(num_channels: int, num_pulses: int, beta: float) -> int:
    """Brennan's rule for sidelooking ULA clutter rank: N_ch + (N_p - 1) * beta (rounded)."""
    return int(round(num_channels + (num_pulses - 1) * beta))
