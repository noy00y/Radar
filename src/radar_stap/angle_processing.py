"""Array steering vectors, conventional beamforming and angle estimation.

Convention (consistent with the simulator and the ``tx_times_conj_rx`` IF):
the two-way path to a far-field scatterer in body-frame direction u (unit) from
a virtual element with phase centre p_c = p_tx + p_rx is ~ 2R - p_c . u, and the
IF phase is +2*pi*tau*f, so the spatial steering vector is

    a_c(u) = exp(-j * 2*pi * (p_c . u) / lambda)

For a 1-D array along body x this depends only on the direction cosine
u_x = sin(theta), where theta is the *cone-complement angle* measured from the
array normal (equal to azimuth when elevation is zero), positive toward +x.

Virtual (TDM) channels are not simultaneous; before combining Doppler-domain
values across TX slots, remove the slot Doppler phase with
:func:`radar_stap.doppler_processing.tdm_phase_compensation`.
"""

from __future__ import annotations

import numpy as np

from .range_processing import get_window, parabolic_peak


def direction_from_angles(angle_deg, elevation_deg=0.0) -> np.ndarray:
    """Body-frame unit vector(s) with u_x = sin(angle), u_z = sin(elevation), u_y >= 0."""
    a = np.deg2rad(np.asarray(angle_deg, float))
    e = np.deg2rad(np.asarray(elevation_deg, float))
    ux = np.sin(a)
    uz = np.sin(e) * np.ones_like(ux)
    uy = np.sqrt(np.clip(1.0 - ux**2 - uz**2, 0.0, None))
    return np.stack([ux, uy, uz], axis=-1)


def array_steering(positions_m: np.ndarray, direction: np.ndarray, wavelength_m: float) -> np.ndarray:
    """a(u) for element positions (C,3) and direction(s) (...,3) -> (..., C)."""
    phase = np.asarray(direction, float) @ np.asarray(positions_m, float).T
    return np.exp(-2j * np.pi * phase / wavelength_m)


def steering_matrix(positions_m: np.ndarray, angles_deg: np.ndarray, wavelength_m: float, elevation_deg=0.0) -> np.ndarray:
    """(C, A) matrix of steering vectors over an angle grid."""
    return array_steering(positions_m, direction_from_angles(angles_deg, elevation_deg), wavelength_m).T


def beamform_spectrum(
    snapshot: np.ndarray,
    positions_m: np.ndarray,
    wavelength_m: float,
    angles_deg: np.ndarray,
    taper: str = "rect",
) -> np.ndarray:
    """Conventional (Bartlett) beamformer power |w(theta)^H x|^2 with w = t*a/sum(t).

    Works for uniform and non-uniform linear arrays (positions are explicit).
    ``snapshot`` shape (C,) or (C, K) (powers averaged over K).
    """
    x = np.asarray(snapshot)
    if x.ndim == 1:
        x = x[:, None]
    A = steering_matrix(positions_m, angles_deg, wavelength_m)
    t = get_window(taper, A.shape[0])
    W = (t[:, None] * A) / t.sum()
    return (np.abs(W.conj().T @ x) ** 2).mean(axis=1)


def estimate_angle(
    snapshot: np.ndarray,
    positions_m: np.ndarray,
    wavelength_m: float,
    grid_deg: np.ndarray | None = None,
    taper: str = "rect",
) -> tuple[float, np.ndarray, np.ndarray]:
    """Peak of the beamformer spectrum with parabolic refinement -> (angle_deg, grid, spectrum)."""
    grid = np.linspace(-80, 80, 1601) if grid_deg is None else np.asarray(grid_deg)
    p = beamform_spectrum(snapshot, positions_m, wavelength_m, grid, taper)
    i = int(np.argmax(p))
    if 0 < i < grid.size - 1:
        fi = parabolic_peak(p, i, log_domain=True)
        ang = float(np.interp(fi, np.arange(grid.size), grid))
    else:
        ang = float(grid[i])
    return ang, grid, p


def angle_from_direction(u_body: np.ndarray) -> float:
    return float(np.rad2deg(np.arcsin(np.clip(u_body[0], -1, 1))))
