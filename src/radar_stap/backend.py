"""Optional array-backend interface for accelerated (GPU) benchmarking.

The canonical implementation is NumPy/SciPy on the CPU. This module exposes a
small, backend-agnostic batched MVDR kernel so that a CuPy (RTX 3060) version
can be benchmarked and *checked against the CPU result*. CuPy is never required:
requesting it when it is not installed raises :class:`BackendUnavailable`.

FPGA note: nothing here is FPGA code. See README "GPU and FPGA readiness" for
the candidate hardware partition (covariance accumulation, Cholesky/solve,
weight application) and its arithmetic/memory constraints.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np


class BackendUnavailable(RuntimeError):
    pass


def get_array_module(name: str = "numpy") -> ModuleType:
    if name == "numpy":
        return np
    if name == "cupy":
        try:
            import cupy  # type: ignore

            cupy.cuda.runtime.getDeviceCount()
            return cupy
        except Exception as exc:  # ImportError or no CUDA device
            raise BackendUnavailable(f"CuPy backend unavailable: {exc}") from exc
    raise BackendUnavailable(f"unknown backend {name!r} (supported: numpy, cupy)")


def batched_mvdr(X, S, loading: float, xp: ModuleType = np):
    """MVDR weights for B independent training sets.

    X: (B, L, D) training snapshots, S: (D, J) steering, loading: absolute level.
    Returns W: (B, D, J) with w^H s = 1. Uses Cholesky + triangular solves.
    """
    B, L, D = X.shape
    R = xp.matmul(xp.swapaxes(X, 1, 2), xp.conj(X)) / L  # (B, D, D)
    R = 0.5 * (R + xp.conj(xp.swapaxes(R, 1, 2))) + loading * xp.eye(D, dtype=R.dtype)[None]
    Lc = xp.linalg.cholesky(R)  # lower
    Sb = xp.broadcast_to(S, (B,) + S.shape)
    Q = xp.linalg.solve(xp.conj(xp.swapaxes(Lc, 1, 2)), xp.linalg.solve(Lc, Sb))
    g = xp.sum(xp.conj(Sb) * Q, axis=1)  # (B, J)
    return Q / xp.conj(g)[:, None, :]


def to_numpy(a) -> np.ndarray:
    return a.get() if hasattr(a, "get") else np.asarray(a)
