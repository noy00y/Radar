from __future__ import annotations

import numpy as np


def estimate_peak(
    rd_db: np.ndarray,
    range_axis_m: np.ndarray,
    velocity_axis_mps: np.ndarray,
    range_bounds_m: tuple[float, float] | None = None,
    velocity_bounds_mps: tuple[float, float] | None = None,
) -> dict:
    """Return the strongest range-Doppler bin, optionally within bounds."""
    mask = np.ones(rd_db.shape, dtype=bool)
    if range_bounds_m is not None:
        mask &= (range_axis_m[None, :] >= range_bounds_m[0]) & (
            range_axis_m[None, :] <= range_bounds_m[1]
        )
    if velocity_bounds_mps is not None:
        mask &= (velocity_axis_mps[:, None] >= velocity_bounds_mps[0]) & (
            velocity_axis_mps[:, None] <= velocity_bounds_mps[1]
        )
    if not np.any(mask):
        raise ValueError("search bounds contain no range-Doppler bins")
    masked = np.where(mask, rd_db, -np.inf)
    d_idx, r_idx = np.unravel_index(np.argmax(masked), masked.shape)
    return {
        "range_m": float(range_axis_m[r_idx]),
        "radial_velocity_mps": float(velocity_axis_mps[d_idx]),
        "peak_db": float(rd_db[d_idx, r_idx]),
        "range_index": int(r_idx),
        "doppler_index": int(d_idx),
    }
