"""Space-time adaptive processing (STAP) for GMTI.

Processing domain (baseline, used consistently)
-----------------------------------------------
STAP operates on **range-compressed** multichannel data: for each range bin r
the space-time snapshot is built from the virtual cube V[c, l, r]
(C virtual channels x L slow-time samples, see :mod:`radar_stap.datacube`)

    x_r[l * C + c] = V[c, l, r]       (D = C * L, pulse-major / channel fastest)

Space-time steering vector for body direction u and slow-time frequency nu:

    s[l*C + c] = exp(-j 2 pi p_c . u / lambda) * exp(j 2 pi nu (l T_slow + t_c))

where t_c is the TDM slot offset of virtual channel c (0 for single-TX data).
For single TX this is s = kron(b(nu), a(u)).

Implemented filters (all normalised to be distortionless, w^H s = 1):

* ``nonadaptive``: tapered space-time matched filter (beamformer + Doppler
  filter), w = (t ⊙ s) / sum(t). Conventional reference.
* ``mvdr``: full-rank sample-matrix MVDR, w = R^-1 s / (s^H R^-1 s), computed
  with a Cholesky solve on the diagonally loaded sample covariance.
* ``efa``: adjacent-bin post-Doppler (extended factored) STAP: per Doppler
  hypothesis, the data are projected onto Q adjacent Doppler filters for every
  channel (dimension r = Q*C << D), MVDR is solved in that subspace, and the
  equivalent full-dimension weight w = T w~ is returned. This is the first
  "efficient STAP" candidate (sample support ~2r instead of ~2D).
* ``pc``: principal-components (eigencanceler) reduced-rank filter,
  w ∝ (I - U_k U_k^H) s, with the k dominant eigenvectors of the sample
  covariance.
* ``dpca``: displaced phase centre antenna (non-adaptive clutter cancellation).
  Channel pairs whose two-way phase centres coincide after ``lag`` pulses are
  subtracted, y = x[l+lag, rear] - x[l, front], which removes stationary clutter
  when v_p * lag * T_slow matches the channel spacing; the differenced data are
  then Doppler filtered. Needs the platform velocity (body frame).

TX policy: hardware experiments use a single TX (t_c = 0, C = num_rx). The TDM
terms (t_c) are kept so legacy multi-TX captures and simulations stay correct;
the capture path only enables multi-TX with ``allow_multi_tx: true``.

New methods register in :data:`METHODS` and implement :class:`StapMethod`.
"""

from __future__ import annotations

import logging
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve

from .config import SimulationConfig, StapConfig, element_positions_m, sim_derived, wavelength_for
from .covariance import (
    InsufficientTrainingError,
    covariance_diagnostics,
    diagonal_load,
    estimate_noise_floor,
    forward_backward_average,
    loading_level,
    sample_covariance,
)
from .datacube import virtual_positions, virtual_time_offsets
from .range_processing import get_window

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Space-time model
# ---------------------------------------------------------------------------


@dataclass
class SpaceTimeModel:
    positions: np.ndarray  # (C, 3) virtual phase-centre positions, body frame [m]
    time_offsets: np.ndarray  # (C,) TDM slot offsets [s]
    num_loops: int  # L slow-time samples per channel
    slow_time_interval: float  # T_slow [s]
    wavelength: float  # [m]

    @property
    def num_channels(self) -> int:
        return int(self.positions.shape[0])

    @property
    def dimension(self) -> int:
        return self.num_channels * self.num_loops

    @classmethod
    def from_config(cls, cfg: SimulationConfig) -> "SpaceTimeModel":
        lam = wavelength_for(cfg.radar)
        rx, tx = element_positions_m(cfg.array, lam)
        d = sim_derived(cfg)
        return cls(
            positions=virtual_positions(rx, tx, cfg.array.tx_schedule),
            time_offsets=virtual_time_offsets(cfg.array.tx_schedule, rx.shape[0], d.chirp_period_s),
            num_loops=d.num_loops,
            slow_time_interval=d.slow_time_interval_s,
            wavelength=lam,
        )

    def spatial(self, direction: np.ndarray) -> np.ndarray:
        """(C,) spatial steering for body direction u."""
        return np.exp(-2j * np.pi * (self.positions @ np.asarray(direction, float)) / self.wavelength)

    def steering(self, direction: np.ndarray, nu_hz) -> np.ndarray:
        """(D, J) space-time steering vectors (D,) if nu is scalar."""
        nu = np.atleast_1d(np.asarray(nu_hz, float))
        a = self.spatial(direction)  # (C,)
        l = np.arange(self.num_loops)
        t = l[:, None] * self.slow_time_interval + self.time_offsets[None, :]  # (L, C)
        s = a[None, None, :] * np.exp(2j * np.pi * nu[:, None, None] * t[None])  # (J, L, C)
        S = s.reshape(nu.size, -1).T
        return S[:, 0] if np.ndim(nu_hz) == 0 else S

    def doppler_grid(self, num_hypotheses: int) -> np.ndarray:
        """nu_j = (j - J/2) / (J T_slow): uniform over the unambiguous interval."""
        j = np.arange(num_hypotheses) - num_hypotheses // 2
        return j / (num_hypotheses * self.slow_time_interval)

    def velocity_of(self, nu_hz):
        return np.asarray(nu_hz) * self.wavelength / 2.0


def normalize_distortionless(W: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Scale columns so that w_j^H s_j = 1."""
    g = np.einsum("dj,dj->j", W.conj(), S)
    if np.any(np.abs(g) < 1e-300) or not np.all(np.isfinite(g)):
        raise FloatingPointError("weight normalisation failed (w^H s ~ 0 or non-finite)")
    return W / g.conj()[None, :]


def _validate_steering(S: np.ndarray, D: int) -> np.ndarray:
    S = np.asarray(S)
    if S.ndim == 1:
        S = S[:, None]
    if S.shape[0] != D:
        raise ValueError(f"steering dimension {S.shape[0]} != {D}")
    if not np.all(np.isfinite(S)):
        raise ValueError("steering vectors contain non-finite values")
    if np.any(np.linalg.norm(S, axis=0) == 0):
        raise ValueError("zero steering vector")
    return S


# ---------------------------------------------------------------------------
# Weight computation primitives
# ---------------------------------------------------------------------------


def mvdr_weights(R: np.ndarray, S: np.ndarray, loading: float = 0.0) -> np.ndarray:
    """MVDR weights W = R^-1 S / diag(S^H R^-1 S) via Cholesky solve (no explicit inverse)."""
    R = np.asarray(R)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError("covariance must be square")
    if not np.all(np.isfinite(R)):
        raise ValueError("covariance contains non-finite values")
    S = _validate_steering(S, R.shape[0])
    Rl = diagonal_load(0.5 * (R + R.conj().T), loading)
    try:
        cf = cho_factor(Rl, lower=True, check_finite=False)
    except LinAlgError as exc:
        raise LinAlgError(
            "covariance (after loading) is not positive definite; increase diagonal loading or training support"
        ) from exc
    Q = cho_solve(cf, S, check_finite=False)
    return normalize_distortionless(Q, S)


def nonadaptive_weights(S: np.ndarray, model: SpaceTimeModel, spatial_taper: str, temporal_taper: str) -> np.ndarray:
    S = _validate_steering(S, model.dimension)
    t = np.kron(get_window(temporal_taper, model.num_loops), get_window(spatial_taper, model.num_channels))
    return normalize_distortionless(t[:, None] * S, S)


def efa_filters(model: SpaceTimeModel, nu_grid: np.ndarray, num_bins: int, taper: str) -> np.ndarray:
    """Doppler filter bank F (J, L, Q): F[j, l, q] = t_l exp(j 2 pi (nu_j + q dnu) l T_slow)."""
    q = np.arange(num_bins) - num_bins // 2
    dnu = 1.0 / (model.num_loops * model.slow_time_interval)
    l = np.arange(model.num_loops)
    t = get_window(taper, model.num_loops)
    f = np.asarray(nu_grid)[:, None] + q[None, :] * dnu  # (J, Q)
    return t[None, :, None] * np.exp(2j * np.pi * f[:, None, :] * l[None, :, None] * model.slow_time_interval)


def _project(X: np.ndarray, F: np.ndarray, C: int) -> np.ndarray:
    """Rows x (T, D) -> reduced z = T_j^H x with T_j = kron(F_j, I_C); returns (J, T, Q*C).

    Implemented as one batched matrix product (BLAS) per call.
    """
    T = X.shape[0]
    J, L, Q = F.shape
    X2 = X.reshape(T, L, C).transpose(1, 0, 2).reshape(L, T * C)  # (L, T*C)
    Z = np.matmul(np.conj(np.swapaxes(F, 1, 2)), X2)  # (J, Q, T*C)
    return Z.reshape(J, Q, T, C).transpose(0, 2, 1, 3).reshape(J, T, Q * C)


# ---------------------------------------------------------------------------
# Method interface
# ---------------------------------------------------------------------------


@dataclass
class WeightResult:
    W: np.ndarray  # (D, J) full-dimension weights, w_j^H s_j = 1
    info: dict = field(default_factory=dict)


class StapMethod(Protocol):
    name: str
    adaptive: bool

    def weights(self, training: np.ndarray | None, S: np.ndarray, nu_grid: np.ndarray) -> WeightResult: ...

    def complexity(self, D: int, L: int, J: int) -> float: ...


@dataclass
class NonAdaptive:
    model: SpaceTimeModel
    spatial_taper: str = "rect"
    temporal_taper: str = "hann"
    name: str = "nonadaptive"
    adaptive: bool = False

    def weights(self, training, S, nu_grid) -> WeightResult:
        return WeightResult(nonadaptive_weights(S, self.model, self.spatial_taper, self.temporal_taper), {})

    def complexity(self, D, L, J) -> float:
        return float(J * D)


@dataclass
class FullRankMVDR:
    model: SpaceTimeModel
    loading_mode: str = "noise_relative"
    loading_value: float = 1.0
    noise_level: float | None = None
    forward_backward: bool = False
    name: str = "mvdr"
    adaptive: bool = True

    def weights(self, training, S, nu_grid) -> WeightResult:
        n, D = training.shape
        if (2 * n if self.forward_backward else n) < D:
            raise InsufficientTrainingError(
                f"{n} training snapshots{' (x2 forward-backward)' if self.forward_backward else ''} < dimension "
                f"D={D}: covariance is singular. Use more training cells/frames or a reduced-dimension method."
            )
        # forward-backward averaging makes R full rank from D/2 rows, so the raw estimate may be deficient
        R = sample_covariance(training, allow_rank_deficient=self.forward_backward)
        if self.forward_backward:
            R, n = forward_backward_average(R), 2 * n
        diag = covariance_diagnostics(R, n)
        level = loading_level(R, self.loading_mode, self.loading_value, diag.noise_floor, self.noise_level)
        W = mvdr_weights(R, S, level)
        return WeightResult(
            W,
            {
                "loading": level,
                "condition_number": diag.condition_number,
                "loaded_condition_number": float(
                    (diag.eigenvalues[0] + level) / max(diag.eigenvalues[-1] + level, 1e-300)
                ),
                "effective_rank": diag.effective_rank,
                "rmb_expected_loss_db": diag.rmb_expected_loss_db,
                "dimension": R.shape[0],
            },
        )

    def complexity(self, D, L, J) -> float:
        return float(L * D**2 + D**3 / 3 + J * D**2)


@dataclass
class AdjacentBinPostDoppler:
    """EFA / adjacent-bin post-Doppler STAP (reduced dimension Q*C per Doppler hypothesis)."""

    model: SpaceTimeModel
    num_bins: int = 3
    temporal_taper: str = "hann"
    loading_mode: str = "noise_relative"
    loading_value: float = 1.0
    noise_level: float | None = None
    name: str = "efa"
    adaptive: bool = True

    forward_backward: bool = False

    def transform(self, nu_grid: np.ndarray) -> np.ndarray:
        """Explicit T_j = kron(F_j, I_C), shape (J, D, Q*C) (for tests / analysis)."""
        F = efa_filters(self.model, nu_grid, self.num_bins, self.temporal_taper)
        C = self.model.num_channels
        return np.stack([np.kron(F[j], np.eye(C)) for j in range(F.shape[0])])

    def _filters(self, nu_grid: np.ndarray) -> np.ndarray:
        key = (tuple(np.round(np.asarray(nu_grid, float), 9)),)
        if getattr(self, "_cache_key", None) != key:
            self._cache_F = efa_filters(self.model, nu_grid, self.num_bins, self.temporal_taper)
            self._cache_key = key
        return self._cache_F

    @staticmethod
    def _unique(nu_grid):
        # Several look angles share one Doppler grid: the reduced covariance depends on
        # Doppler only, so it is estimated once per unique hypothesis.
        nu_u, inv = np.unique(np.asarray(nu_grid, float), return_inverse=True)
        return nu_u, np.asarray(inv).reshape(-1)

    def _rows(self, X):
        return np.concatenate([X, np.conj(X[:, ::-1])]) if self.forward_backward else X

    def prepare(self, X_all: np.ndarray, nu_grid: np.ndarray) -> None:
        """Project every snapshot of the CPI pool once; :meth:`weights_indexed` then only
        selects the training rows (avoids re-projecting overlapping training windows)."""
        nu_u, _ = self._unique(nu_grid)
        F = self._filters(nu_u)
        C = self.model.num_channels
        Z = _project(X_all, F, C)
        Zb = _project(np.conj(X_all[:, ::-1]), F, C) if self.forward_backward else None
        self._prepared = (Z, Zb)

    def weights_indexed(self, idx, S, nu_grid) -> WeightResult:
        Z, Zb = self._prepared
        Zs = Z[:, idx] if Zb is None else np.concatenate([Z[:, idx], Zb[:, idx]], axis=1)
        return self._weights_from_projection(Zs, S, nu_grid)

    def weights(self, training, S, nu_grid) -> WeightResult:
        nu_u, _ = self._unique(nu_grid)
        Z = _project(self._rows(training), self._filters(nu_u), self.model.num_channels)
        return self._weights_from_projection(Z, S, nu_grid)

    def _weights_from_projection(self, Z, S, nu_grid) -> WeightResult:
        C = self.model.num_channels
        nu_u, inv = self._unique(nu_grid)
        F = self._filters(nu_u)  # (J0, L, Q) unique hypotheses
        J0, Lp, Q = F.shape
        r = Q * C
        T = Z.shape[1]
        if T < r:
            raise InsufficientTrainingError(f"EFA needs >= {r} training snapshots, got {T}")
        Rr = np.matmul(np.swapaxes(Z, 1, 2), Z.conj()) / T  # (J0, r, r)
        Rr = 0.5 * (Rr + np.conj(np.swapaxes(Rr, 1, 2)))
        S = _validate_steering(S, self.model.dimension)
        J = S.shape[1]
        Fj = F[inv]  # (J, L, Q): filter bank of each column's Doppler hypothesis
        # reduced steering s~_j = T_j^H s_j
        Sr = np.einsum("jlc,jlq->jqc", S.T.reshape(J, Lp, C), Fj.conj()).reshape(J, r)
        if self.loading_mode == "noise_relative":
            if self.noise_level is None:
                raise ValueError("loading_mode 'noise_relative' requires noise_level")
            # noise in the reduced domain is sigma^2 * ||filter||^2 (filters are not unit norm)
            fn2 = (np.abs(F) ** 2).sum(axis=1).mean(axis=1)  # (J0,)
            levels0 = self.loading_value * self.noise_level * fn2
            conds = None
        else:
            ev = np.linalg.eigvalsh(Rr)
            levels0 = np.array(
                [loading_level(Rr[j], self.loading_mode, self.loading_value, estimate_noise_floor(ev[j])) for j in range(J0)]
            )
            conds = ev[:, -1] / np.maximum(ev[:, 0], 1e-300)
        A = Rr[inv] + levels0[inv][:, None, None] * np.eye(r)[None]
        Qr = np.linalg.solve(A, Sr[..., None])[..., 0]  # (J, r)
        g = np.einsum("jr,jr->j", Sr.conj(), Qr)
        Wr = Qr / g.conj()[:, None]
        W = np.einsum("jlq,jqc->jlc", Fj, Wr.reshape(J, Q, C)).reshape(J, -1).T  # (D, J)
        info = {"reduced_dimension": r, "loading_mean": float(levels0.mean()), "training_samples": int(T)}
        if conds is not None:
            info["condition_number_median"] = float(np.median(conds))
        return WeightResult(normalize_distortionless(W, S), info)

    def complexity(self, D, L, J) -> float:
        r = self.num_bins * self.model.num_channels
        return float(J * (L * D * self.num_bins + L * r**2 + r**3 / 3 + D * self.num_bins))


@dataclass
class PrincipalComponents:
    """Eigencanceler: w ∝ (I - U_k U_k^H) s, U_k = dominant eigenvectors of R_hat."""

    model: SpaceTimeModel
    rank: int | None = None
    threshold: float = 10.0
    forward_backward: bool = False
    name: str = "pc"
    adaptive: bool = True

    def weights(self, training, S, nu_grid) -> WeightResult:
        R = sample_covariance(training, allow_rank_deficient=True)
        if self.forward_backward:
            R = forward_backward_average(R)
        ev, U = np.linalg.eigh(R)
        ev, U = ev[::-1], U[:, ::-1]
        nf = estimate_noise_floor(ev)
        k = self.rank if self.rank is not None else int(np.sum(ev > self.threshold * nf))
        k = min(k, R.shape[0] - 1)
        Uk = U[:, :k]
        W = S - Uk @ (Uk.conj().T @ S)
        return WeightResult(normalize_distortionless(W, S), {"rank": k, "noise_floor": nf})

    def complexity(self, D, L, J) -> float:
        return float(L * D**2 + 4 * D**3 + J * D * (self.rank or D // 4))


def dpca_pairs(model: SpaceTimeModel, platform_velocity_mps, lag: int) -> tuple[list[tuple[int, int]], float, float]:
    """Channel pairs (front, rear) whose two-way phase centres coincide after ``lag`` pulses.

    Effective two-way phase centre of channel c at pulse l: p_c + 2 v (l T_slow + t_c)
    (moving TX and RX by d moves the tx+rx virtual position by 2 d). DPCA needs
    p_front + 2 v t_front = p_rear + 2 v (lag T_slow + t_rear).
    Returns (pairs, worst mismatch as a fraction of the channel spacing, spacing [m]).
    """
    v = np.asarray(platform_velocity_mps, float)
    P, t = model.positions, model.time_offsets
    C = P.shape[0]
    dists = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    spacing = float(np.min(dists[dists > 1e-12])) if C > 1 else float("nan")
    pairs, worst = [], 0.0
    for rear in range(C):
        p_rear_later = P[rear] + 2 * v * (lag * model.slow_time_interval + t[rear])
        miss = np.linalg.norm(P + 2 * v * t[:, None] - p_rear_later[None, :], axis=1)
        miss[rear] = np.inf
        front = int(np.argmin(miss))
        if miss[front] < 0.5 * spacing:
            pairs.append((front, rear))
            worst = max(worst, float(miss[front] / spacing))
    return pairs, (worst if pairs else float("inf")), spacing


@dataclass
class DisplacedPhaseCentre:
    """DPCA: fixed two-pulse clutter cancellation followed by a tapered Doppler filter."""

    model: SpaceTimeModel
    platform_velocity_mps: list | np.ndarray
    lag: int = 1
    max_mismatch: float = 0.25
    temporal_taper: str = "hann"
    name: str = "dpca"
    adaptive: bool = False

    def __post_init__(self):
        L, C = self.model.num_loops, self.model.num_channels
        if not 1 <= self.lag < L:
            raise ValueError(f"dpca_pulse_lag must be in [1, {L - 1}]")
        pairs, mismatch, spacing = dpca_pairs(self.model, self.platform_velocity_mps, self.lag)
        speed = float(np.linalg.norm(self.platform_velocity_mps))
        v_match = spacing / (2 * self.lag * self.model.slow_time_interval)
        if not pairs or mismatch > self.max_mismatch:
            raise ValueError(
                f"DPCA condition not met: platform speed {speed:.2f} m/s, lag {self.lag} -> phase-centre mismatch "
                f"{mismatch:.2f} of the channel spacing (max {self.max_mismatch}). One-element DPCA needs "
                f"~{v_match:.2f} m/s for this lag (speed = spacing / (2 lag T_slow))."
            )
        rows = []
        for l in range(L - self.lag):
            for front, rear in pairs:
                rows.append(((l + self.lag) * C + rear, l * C + front))
        self._delta = np.zeros((len(rows), L * C))
        for i, (plus, minus) in enumerate(rows):
            self._delta[i, plus] = 1.0
            self._delta[i, minus] = -1.0
        self._taper = np.repeat(get_window(self.temporal_taper, L - self.lag), len(pairs))
        self.info = {"pairs_front_rear": pairs, "phase_centre_mismatch": mismatch,
                     "matched_speed_mps": v_match, "platform_speed_mps": speed, "lag": self.lag}

    def weights(self, training, S, nu_grid) -> WeightResult:
        S = _validate_steering(S, self.model.dimension)
        DS = self._delta @ S  # target response in the clutter-cancelled domain
        W = self._delta.T @ (self._taper[:, None] * DS)
        # Hypotheses whose (look angle, Doppler) lies on the stationary-clutter ridge are cancelled
        # together with the clutter (w^H s = 0): DPCA is blind there. Output is set to zero.
        g = np.real(np.einsum("dj,dj->j", W.conj(), S))
        blind = g <= 1e-9 * np.max(np.abs(g))
        Wn = np.zeros_like(W)
        if np.any(~blind):
            Wn[:, ~blind] = normalize_distortionless(W[:, ~blind], S[:, ~blind])
        info = dict(self.info)
        info["blind_hypotheses"] = [int(j) for j in np.flatnonzero(blind)]
        return WeightResult(Wn, info)

    def complexity(self, D, L, J) -> float:
        return float(J * D)


def build_method(
    name: str,
    model: SpaceTimeModel,
    stap: StapConfig,
    noise_level: float | None = None,
    platform_velocity_mps=None,
) -> StapMethod:
    """Instantiate a method. ``noise_level``: per-element noise variance of the snapshots
    (required for loading_mode 'noise_relative'). ``platform_velocity_mps``: body-frame
    platform velocity (required for 'dpca')."""
    fb = bool(getattr(stap, "forward_backward", False))
    if fb and name in ("mvdr", "efa", "pc"):
        check_persymmetric(model)
    if name == "nonadaptive":
        return NonAdaptive(model, stap.spatial_taper, stap.temporal_taper)
    if name == "mvdr":
        return FullRankMVDR(model, stap.loading_mode, stap.loading_value, noise_level, fb)
    if name == "efa":
        return AdjacentBinPostDoppler(
            model, stap.efa_num_bins, stap.temporal_taper, stap.loading_mode, stap.loading_value, noise_level,
            forward_backward=fb,
        )
    if name == "pc":
        return PrincipalComponents(model, stap.pc_rank, forward_backward=fb)
    if name == "dpca":
        if platform_velocity_mps is None or not np.any(np.asarray(platform_velocity_mps, float)):
            raise ValueError("dpca needs a non-zero platform velocity (platform_velocity_mps)")
        return DisplacedPhaseCentre(model, platform_velocity_mps, stap.dpca_pulse_lag, stap.dpca_max_mismatch,
                                    stap.temporal_taper)
    raise ValueError(f"unknown STAP method {name!r}")


METHODS = ("nonadaptive", "mvdr", "efa", "pc", "dpca")


def check_persymmetric(model: SpaceTimeModel, rtol: float = 1e-6) -> None:
    """Raise unless forward-backward averaging is valid: equal TDM offsets (single TX) and
    channels on a line with uniform spacing in index order (the clutter covariance is then
    persymmetric; slow time is uniform by construction)."""
    P, t = np.asarray(model.positions, float), np.asarray(model.time_offsets, float)
    if np.ptp(t) > 0:
        raise ValueError("forward_backward needs a single TX (TDM slot offsets differ between channels)")
    if P.shape[0] > 1:
        step = P[1] - P[0]
        expect = P[0] + np.arange(P.shape[0])[:, None] * step[None, :]
        if np.linalg.norm(step) == 0 or np.max(np.abs(P - expect)) > rtol * max(np.linalg.norm(step), 1e-12):
            raise ValueError("forward_backward needs a uniformly spaced linear array in channel index order")


# ---------------------------------------------------------------------------
# Training selection and GMTI processing over range
# ---------------------------------------------------------------------------


def select_training_bins(
    test_bin: int,
    num_training: int,
    num_guard: int,
    valid: np.ndarray,
) -> np.ndarray:
    """Symmetric sliding-window training selection around ``test_bin``.

    Takes the nearest ``num_training/2`` valid bins on each side outside the
    guard region; if one side runs out (edge of the swath), the remainder is
    taken from the other side. ``valid`` (bool per bin) excludes bins outside
    the processed swath and, in controlled simulations, known target bins.
    """
    idx = np.flatnonzero(np.asarray(valid, bool))
    left = idx[idx < test_bin - num_guard][::-1]
    right = idx[idx > test_bin + num_guard]
    half = num_training // 2
    nl = min(half, left.size)
    nr = min(num_training - nl, right.size)
    nl = min(num_training - nr, left.size)
    if nl + nr < num_training:
        raise InsufficientTrainingError(
            f"bin {test_bin}: only {nl + nr} valid training bins available, {num_training} requested"
        )
    return np.sort(np.concatenate([left[:nl], right[:nr]]).astype(int))


@dataclass
class GmtiResult:
    method: str
    test_bins: np.ndarray
    nu_grid: np.ndarray
    output: np.ndarray  # (n_test, J) complex filter outputs y = w^H x
    weight_seconds: float
    apply_seconds: float
    peak_memory_bytes: int
    complexity_macs: float
    info: list[dict]

    @property
    def power(self) -> np.ndarray:
        return np.abs(self.output) ** 2


def run_gmti(
    snaps: np.ndarray,
    model: SpaceTimeModel,
    method: StapMethod,
    direction: np.ndarray,
    nu_grid: np.ndarray,
    test_bins: np.ndarray,
    num_training: int,
    num_guard: int,
    valid_training: np.ndarray,
    measure_memory: bool = True,
    training_snaps: np.ndarray | None = None,
    backend=None,
    precision: str = "double",
    batch_bins: int = 32,
) -> GmtiResult:
    """Angle-gated GMTI: for each test bin and Doppler hypothesis output y = w^H x.

    ``snaps`` (num_bins, D) space-time snapshots (range-compressed domain).
    Training for each test bin excludes the bin itself and its guard cells.
    ``training_snaps`` (K, num_bins, D), optional: K CPIs whose selected training bins
    are pooled (K * num_training samples per covariance). Defaults to ``snaps`` only.
    ``direction`` may be (3,) or (A, 3): with A look angles the steering matrix holds
    A blocks of len(nu_grid) columns (angle-major) sharing one covariance per test bin;
    the result's ``nu_grid`` is then the tiled grid.
    ``backend``: None = per-bin reference implementation (NumPy/SciPy); an array module
    (numpy or cupy, see :func:`radar_stap.batched.resolve_backend`) = batched MVDR/PC/EFA
    over chunks of ``batch_bins`` test bins in ``precision`` ("double" | "single").
    """
    dirs = np.atleast_2d(np.asarray(direction, float))
    nu_base = np.atleast_1d(np.asarray(nu_grid, float))
    S = np.hstack([model.steering(d, nu_base) for d in dirs])
    nu_grid = np.tile(nu_base, dirs.shape[0])
    D, J = S.shape
    if snaps.shape[1] != D:
        raise ValueError(f"snapshot dimension {snaps.shape[1]} != model dimension {D}")
    pool = snaps[None] if training_snaps is None else np.asarray(training_snaps)
    if pool.ndim != 3 or pool.shape[1:] != snaps.shape:
        raise ValueError(f"training_snaps shape {pool.shape} incompatible with snaps {snaps.shape}")

    def training_rows(tr):
        return pool[:, tr].reshape(-1, D)

    indexed = method.adaptive and hasattr(method, "prepare")
    n_bins = pool.shape[1]

    def row_index(tr):
        return (np.arange(pool.shape[0])[:, None] * n_bins + np.asarray(tr)[None, :]).reshape(-1)

    def adaptive_weights(tr):
        if indexed:
            return method.weights_indexed(row_index(tr), S, nu_grid)
        return method.weights(training_rows(tr), S, nu_grid)

    out = np.zeros((len(test_bins), J), complex)
    infos: list[dict] = []
    t_w = t_a = 0.0
    if backend is not None and method.adaptive and method.name in ("mvdr", "pc", "efa"):
        from .batched import run_adaptive_batched

        trs = [select_training_bins(int(b), num_training, num_guard, valid_training) for b in test_bins]
        out, t_w, infos, device_bytes = run_adaptive_batched(snaps, pool, method, S, nu_grid, test_bins, trs, backend,
                                                             precision, batch_bins)
        return GmtiResult(
            method=method.name, test_bins=np.asarray(test_bins), nu_grid=np.asarray(nu_grid), output=out,
            weight_seconds=t_w, apply_seconds=0.0, peak_memory_bytes=device_bytes,
            complexity_macs=method.complexity(D, num_training * pool.shape[0], J) * len(test_bins), info=infos,
        )
    if indexed:  # per-bin reference path only: project the pool once (the batched path does its own)
        method.prepare(pool.reshape(-1, D), nu_grid)
    if not method.adaptive:
        t0 = time.perf_counter()
        W = method.weights(None, S, nu_grid).W
        t_w += time.perf_counter() - t0
        t0 = time.perf_counter()
        out[:] = snaps[test_bins] @ W.conj()
        t_a += time.perf_counter() - t0
    else:
        for i, b in enumerate(test_bins):
            tr = select_training_bins(int(b), num_training, num_guard, valid_training)
            t0 = time.perf_counter()
            res = adaptive_weights(tr)
            t_w += time.perf_counter() - t0
            t0 = time.perf_counter()
            out[i] = snaps[b] @ res.W.conj()
            t_a += time.perf_counter() - t0
            if i % max(1, len(test_bins) // 8) == 0:
                infos.append({"bin": int(b), **res.info})
    peak = 0
    if measure_memory and len(test_bins):
        # Separate pass: tracemalloc slows allocation, so it must not distort the timing above.
        b = int(test_bins[len(test_bins) // 2])
        tr = select_training_bins(b, num_training, num_guard, valid_training) if method.adaptive else None
        tracemalloc.start()
        try:
            if tr is None:
                method.weights(None, S, nu_grid)
            else:
                adaptive_weights(tr)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
    return GmtiResult(
        method=method.name,
        test_bins=np.asarray(test_bins),
        nu_grid=np.asarray(nu_grid),
        output=out,
        weight_seconds=t_w,
        apply_seconds=t_a,
        peak_memory_bytes=int(peak),  # per weight computation (one test bin)
        complexity_macs=method.complexity(D, num_training * pool.shape[0], J) * (len(test_bins) if method.adaptive else 1),
        info=infos,
    )
