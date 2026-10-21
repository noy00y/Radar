"""Data-cube layout conventions and the physical-channel -> virtual-channel mapping.

Raw / range-compressed hardware-like layout
    X[rx, chirp, sample]  or  Y[rx, chirp, range_bin]
with chirps in transmission order (TDM slots interleaved).

Virtual (STAP / angle) layout
    V[c, l, r]   c = p * M_rx + m   (TX slot p of the schedule, RX m)
                 l = loop index (one chirp per TX slot per loop)
Virtual channel c is sampled at slow time  t = (l * P + p) * T_c,  i.e. TX slots
are *not* simultaneous. Any processing combining virtual channels must account
for the per-channel time offset ``p * T_c`` (see :func:`virtual_time_offsets`);
this package does so exactly in the space-time steering vector rather than
pretending the virtual array is a simultaneous snapshot.

Space-time snapshot stacking (pulse-major, channel fastest):
    x[l * C + c] = V[c, l, r]          (D = C * L)
"""

from __future__ import annotations

import numpy as np


def to_virtual_cube(Y: np.ndarray, tx_schedule: list[int]) -> np.ndarray:
    """(M, N, R) chirp-ordered cube -> (P*M, N/P, R) virtual cube."""
    if Y.ndim != 3:
        raise ValueError(f"expected (rx, chirp, bins) cube, got shape {Y.shape}")
    m, n, r = Y.shape
    p = len(tx_schedule)
    if n % p:
        raise ValueError(f"{n} chirps not divisible by TDM period {p}")
    loops = n // p
    return Y.reshape(m, loops, p, r).transpose(2, 0, 1, 3).reshape(p * m, loops, r)


def from_virtual_cube(V: np.ndarray, num_rx: int, tx_schedule: list[int]) -> np.ndarray:
    """Inverse of :func:`to_virtual_cube`."""
    c, loops, r = V.shape
    p = len(tx_schedule)
    if c != p * num_rx:
        raise ValueError("channel count does not match num_rx * len(tx_schedule)")
    return V.reshape(p, num_rx, loops, r).transpose(1, 2, 0, 3).reshape(num_rx, loops * p, r)


def virtual_positions(rx_pos: np.ndarray, tx_pos: np.ndarray, tx_schedule: list[int]) -> np.ndarray:
    """Two-way phase-centre positions (tx + rx) per virtual channel, shape (C, 3)."""
    rx_pos = np.asarray(rx_pos, float)
    tx_pos = np.asarray(tx_pos, float)
    return np.concatenate([tx_pos[t][None, :] + rx_pos for t in tx_schedule], axis=0)


def virtual_time_offsets(tx_schedule: list[int], num_rx: int, chirp_period_s: float) -> np.ndarray:
    """Slow-time offset of each virtual channel within a loop [s], shape (C,)."""
    return np.repeat(np.arange(len(tx_schedule)) * chirp_period_s, num_rx)


def snapshots(V: np.ndarray) -> np.ndarray:
    """(C, L, R) virtual cube -> (R, D) snapshot matrix, x[l*C + c] = V[c, l, r]."""
    c, loops, r = V.shape
    return np.ascontiguousarray(V.transpose(2, 1, 0).reshape(r, loops * c))


def snapshot_to_matrix(x: np.ndarray, num_channels: int, num_loops: int) -> np.ndarray:
    """D-vector -> (C, L) matrix (inverse stacking), for diagnostics/tests."""
    return np.asarray(x).reshape(num_loops, num_channels).T
