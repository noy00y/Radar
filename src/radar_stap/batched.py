"""Batched STAP weight computation for many test bins at once (NumPy on CPU or CuPy on GPU).

The per-bin implementation in :mod:`radar_stap.stap` is the reference. This module
computes the SAME filters (MVDR, PC, EFA, incl. forward-backward averaging, multi-frame
pooled training and multiple look angles) for a chunk of B test bins with batched
linear algebra, so a GPU can be used (RTX 3060: run in single precision -- consumer
GPUs run float64 at ~1/64 of float32 speed).

Outputs are formed directly as y = w^H x without materialising every weight vector:

* MVDR: q = (R + aI)^-1 s,  y = q^H x / (q^H s)
* PC:   w = s - U_k U_k^H s, y = w^H x / (w^H s)
* EFA:  reduced q~ = (R~ + aI)^-1 s~,  y = q~^H z / (q~^H s~),  z = T^H x

which equals w^H x for the distortionless weights of the reference implementation.
Selected in YAML with ``stap.backend`` (numpy | numpy_batched | cupy | auto),
``stap.precision`` (double | single) and ``stap.gpu_batch_bins``.
"""

from __future__ import annotations

import logging
import math
import time
from types import ModuleType

import numpy as np

from .backend import BackendUnavailable, get_array_module, to_numpy

log = logging.getLogger(__name__)

BACKENDS = ("numpy", "numpy_batched", "cupy", "auto")
PRECISIONS = {"double": (np.complex128, np.float64), "single": (np.complex64, np.float32)}
BATCHED_METHODS = ("mvdr", "pc", "efa")


def resolve_backend(name: str) -> tuple[ModuleType | None, str]:
    """YAML backend name -> (array module or None for the per-bin reference path, name used)."""
    if name == "numpy":
        return None, "numpy"
    if name == "numpy_batched":
        return np, "numpy_batched"
    if name == "cupy":
        return get_array_module("cupy"), "cupy"  # raises BackendUnavailable
    if name == "auto":
        try:
            return get_array_module("cupy"), "cupy"
        except BackendUnavailable as exc:
            log.warning("stap.backend auto: %s -> using the CPU reference path", exc)
            return None, "numpy"
    raise ValueError(f"unknown stap.backend {name!r} (supported: {BACKENDS})")


def _sync(xp) -> None:
    if hasattr(xp, "cuda"):
        xp.cuda.Stream.null.synchronize()


def _noise_floor(ev, xp, fraction: float = 0.25):
    """Mean of the smallest ``fraction`` of (ascending) eigenvalues along the last axis."""
    k = max(1, int(math.ceil(fraction * ev.shape[-1])))
    return xp.maximum(ev[..., :k].mean(axis=-1), 0.0)


def _levels(R, mode: str, value: float, noise_level, xp):
    """Absolute loading level per matrix of R (..., D, D), same rules as covariance.loading_level."""
    D = R.shape[-1]
    if mode == "noise_eig_relative":
        return value * _noise_floor(xp.linalg.eigvalsh(R), xp)
    if mode == "absolute":
        return xp.full(R.shape[:-2], float(value), dtype=R.real.dtype)
    if mode == "trace_relative":
        return value * xp.real(xp.trace(R, axis1=-2, axis2=-1)) / D
    if mode == "noise_relative":
        if noise_level is None:
            raise ValueError("loading_mode 'noise_relative' requires a noise level")
        return xp.full(R.shape[:-2], float(value * noise_level), dtype=R.real.dtype)
    raise ValueError(f"unknown loading mode {mode!r}")


def _covariance(X, forward_backward: bool, xp):
    """(B, N, D) rows -> (B, D, D) Hermitian sample covariance R[i, j] = mean x_i x_j^*."""
    R = xp.matmul(xp.swapaxes(X, 1, 2), xp.conj(X)) / X.shape[1]
    R = 0.5 * (R + xp.conj(xp.swapaxes(R, 1, 2)))
    if forward_backward:
        R = 0.5 * (R + xp.conj(R[:, ::-1, ::-1]))
    return R


def _project(X, F, C: int, xp):
    """Rows (T, D) -> reduced z = T_j^H x, T_j = kron(F_j, I_C); returns (J0, T, Q*C)."""
    T = X.shape[0]
    J0, L, Q = F.shape
    X2 = X.reshape(T, L, C).transpose(1, 0, 2).reshape(L, T * C)
    Z = xp.matmul(xp.conj(xp.swapaxes(F, 1, 2)), X2)  # (J0, Q, T*C)
    return Z.reshape(J0, Q, T, C).transpose(0, 2, 1, 3).reshape(J0, T, Q * C)


# ---------------------------------------------------------------------------
# Per-chunk kernels: X (B, N, D) training rows, x (B, D) test snapshots, S (D, J)
# ---------------------------------------------------------------------------

def mvdr_chunk(X, x, S, method, xp):
    B, N, D = X.shape
    if (2 * N if method.forward_backward else N) < D:
        from .covariance import InsufficientTrainingError

        raise InsufficientTrainingError(f"L={N} training snapshots"
                                        f"{' (x2 forward-backward)' if method.forward_backward else ''} < "
                                        f"dimension D={D}: covariance is singular")
    R = _covariance(X, method.forward_backward, xp)
    lev = _levels(R, method.loading_mode, method.loading_value, method.noise_level, xp)
    A = R + lev[:, None, None] * xp.eye(D, dtype=R.dtype)[None]
    Q = xp.linalg.solve(A, xp.ascontiguousarray(xp.broadcast_to(S, (B,) + S.shape)))  # (B, D, J)
    g = xp.sum(xp.conj(Q) * S[None], axis=1)  # q^H s
    y = xp.einsum("bdj,bd->bj", xp.conj(Q), x) / g
    return y, {"loading_mean": float(to_numpy(lev).mean())}


def pc_chunk(X, x, S, method, xp):
    B, N, D = X.shape
    R = _covariance(X, method.forward_backward, xp)
    ev, U = xp.linalg.eigh(R)  # ascending
    nf = _noise_floor(ev, xp)
    if method.rank is not None:
        k = xp.full((B,), int(method.rank))
    else:
        k = xp.sum(ev > method.threshold * nf[:, None], axis=1)
    k = xp.minimum(k, D - 1)
    keep = xp.arange(D)[None, :] >= (D - k)[:, None]  # the k largest (ascending order)
    UhS = xp.matmul(xp.conj(xp.swapaxes(U, 1, 2)), S[None])  # (B, D, J)
    W = S[None] - xp.matmul(U, keep[:, :, None] * UhS)
    g = xp.sum(xp.conj(W) * S[None], axis=1)
    y = xp.einsum("bdj,bd->bj", xp.conj(W), x) / g
    return y, {"rank_mean": float(to_numpy(k).mean())}


class _EfaState:
    """EFA projections of every pooled snapshot and the reduced steering, built once per call."""

    def __init__(self, method, pool_rows, S, nu_grid, xp, cdtype):
        from .stap import efa_filters

        model = method.model
        self.C = C = model.num_channels
        nu_u, inv = np.unique(np.asarray(nu_grid, float), return_inverse=True)
        self.inv = xp.asarray(np.asarray(inv).reshape(-1))
        F = xp.asarray(efa_filters(model, nu_u, method.num_bins, method.temporal_taper).astype(cdtype))
        self.F = F
        J0, Lp, Q = F.shape
        self.r = Q * C
        self.Z = _project(pool_rows, F, C, xp)  # (J0, N_pool, r)
        self.Zb = _project(xp.conj(pool_rows[:, ::-1]), F, C, xp) if method.forward_backward else None
        J = S.shape[1]
        Fj = F[self.inv]
        self.Sr = xp.einsum("jlc,jlq->jqc", S.T.reshape(J, Lp, C), xp.conj(Fj)).reshape(J, self.r)
        if method.loading_mode == "noise_relative":
            fn2 = (xp.abs(F) ** 2).sum(axis=1).mean(axis=1)
            self.noise_levels = method.loading_value * method.noise_level * fn2  # (J0,)
        else:
            self.noise_levels = None


def efa_chunk(idx, x, st: _EfaState, method, xp):
    """idx (B, N) pooled row indices of each test bin's training set."""
    Zs = st.Z[:, idx]  # (J0, B, N, r)
    if st.Zb is not None:
        Zs = xp.concatenate([Zs, st.Zb[:, idx]], axis=2)
    T = Zs.shape[2]
    if T < st.r:
        from .covariance import InsufficientTrainingError

        raise InsufficientTrainingError(f"EFA needs >= {st.r} training snapshots, got {T}")
    Zs = xp.swapaxes(Zs, 0, 1)  # (B, J0, T, r)
    Rr = xp.matmul(xp.swapaxes(Zs, 2, 3), xp.conj(Zs)) / T  # (B, J0, r, r)
    Rr = 0.5 * (Rr + xp.conj(xp.swapaxes(Rr, 2, 3)))
    if st.noise_levels is not None:
        lev0 = xp.broadcast_to(st.noise_levels[None], Rr.shape[:2])
    else:
        lev0 = _levels(Rr, method.loading_mode, method.loading_value, None, xp)  # (B, J0)
    A = Rr[:, st.inv] + lev0[:, st.inv][..., None, None] * xp.eye(st.r, dtype=Rr.dtype)  # (B, J, r, r)
    B, J = A.shape[:2]
    Qr = xp.linalg.solve(A, xp.ascontiguousarray(xp.broadcast_to(st.Sr[None, ..., None], (B, J, st.r, 1))))[..., 0]
    g = xp.sum(xp.conj(Qr) * st.Sr[None], axis=2)  # (B, J)
    z = xp.swapaxes(_project(x, st.F, st.C, xp), 0, 1)[:, st.inv]  # (B, J, r) test snapshot, reduced
    y = xp.sum(xp.conj(Qr) * z, axis=2) / g
    return y, {"loading_mean": float(to_numpy(lev0).mean()), "reduced_dimension": st.r, "training_samples": int(T)}


# ---------------------------------------------------------------------------

def run_adaptive_batched(snaps, pool, method, S, nu_grid, test_bins, training_lists, xp,
                         precision: str = "double", batch_bins: int = 32):
    """Outputs (n_test, J) for an adaptive method using batched linear algebra on ``xp``.

    ``pool`` (K, R, D) pooled CPIs, ``training_lists`` list of training-bin index arrays
    (one per test bin, equal length). Returns (out, weight_seconds, infos, device_bytes).
    """
    if method.name not in BATCHED_METHODS:
        raise ValueError(f"no batched implementation for {method.name!r}")
    if precision not in PRECISIONS:
        raise ValueError(f"stap.precision must be one of {sorted(PRECISIONS)}")
    cdtype = PRECISIONS[precision][0]
    K, R_bins, D = pool.shape
    t0 = time.perf_counter()
    pool_rows = xp.asarray(np.ascontiguousarray(pool.reshape(-1, D)).astype(cdtype))
    snaps_d = xp.asarray(np.asarray(snaps).astype(cdtype))
    S_d = xp.asarray(np.asarray(S).astype(cdtype))
    tests = np.asarray(test_bins, int)
    rows = np.stack([(np.arange(K)[:, None] * R_bins + np.asarray(tr)[None, :]).reshape(-1)
                     for tr in training_lists])  # (n_test, K*num_training)
    st = _EfaState(method, pool_rows, S_d, nu_grid, xp, cdtype) if method.name == "efa" else None
    outs, infos = [], []
    step = max(1, int(batch_bins))
    for c0 in range(0, tests.size, step):
        sel = slice(c0, c0 + step)
        idx = xp.asarray(rows[sel])
        x = snaps_d[xp.asarray(tests[sel])]
        if method.name == "efa":
            y, info = efa_chunk(idx, x, st, method, xp)
        else:
            X = pool_rows[idx]  # (B, N, D)
            y, info = (mvdr_chunk if method.name == "mvdr" else pc_chunk)(X, x, S_d, method, xp)
        outs.append(y)
        infos.append({"bin": int(tests[c0]), "chunk_bins": int(min(step, tests.size - c0)), **info})
    out = xp.concatenate(outs, axis=0) if outs else xp.zeros((0, S.shape[1]), cdtype)
    _sync(xp)
    seconds = time.perf_counter() - t0
    device_bytes = int(xp.get_default_memory_pool().total_bytes()) if hasattr(xp, "get_default_memory_pool") else 0
    return np.asarray(to_numpy(out), dtype=np.complex128), seconds, infos, device_bytes
