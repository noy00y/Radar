"""Slow-time (Doppler) processing and range-Doppler maps.

Doppler processing is applied per *virtual* channel (see :mod:`radar_stap.datacube`)
so that for TDM-MIMO each channel's slow-time sequence uses only chirps from
one TX, sampled every ``T_slow = P * T_c``.

Axis convention (internal ``tx_times_conj_rx`` data): FFT bin frequency nu
(after fftshift) is the slow-time phase rate; radial velocity
``v_r = nu * lambda / 2`` with v_r = dR/dt (positive receding). Unambiguous
interval: nu in [-1/(2 T_slow), 1/(2 T_slow)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .range_processing import get_window


@dataclass(frozen=True)
class DopplerAxis:
    freq_hz: np.ndarray
    velocity_mps: np.ndarray
    nfft: int
    slow_time_interval_s: float
    bin_spacing_hz: float
    window: str
    wavelength_m: float

    @property
    def num_bins(self) -> int:
        return int(self.freq_hz.size)

    def bin_of_velocity(self, v_r: float) -> float:
        """Fractional (fftshifted) bin index for radial velocity v_r, wrapped to the axis."""
        nu = 2 * v_r / self.wavelength_m
        idx = nu / self.bin_spacing_hz + self.nfft // 2
        return float(np.mod(idx, self.nfft))


def doppler_axis(nfft: int, slow_time_interval_s: float, wavelength_m: float, window: str = "hann") -> DopplerAxis:
    k = np.arange(nfft) - nfft // 2
    df = 1.0 / (nfft * slow_time_interval_s)
    f = k * df
    return DopplerAxis(f, f * wavelength_m / 2.0, nfft, slow_time_interval_s, df, window, wavelength_m)


def doppler_fft(
    V: np.ndarray,
    slow_time_interval_s: float,
    wavelength_m: float,
    window: str = "hann",
    nfft: int | None = None,
    axis: int = -2,
) -> tuple[np.ndarray, DopplerAxis]:
    """Window + FFT + fftshift along the slow-time axis (default: V[c, l, r] -> axis -2)."""
    V = np.asarray(V)
    n = V.shape[axis]
    nfft = nfft or n
    w = get_window(window, n)
    shape = [1] * V.ndim
    shape[axis] = n
    Z = np.fft.fftshift(np.fft.fft(V * w.reshape(shape), n=nfft, axis=axis), axes=axis)
    return Z.astype(np.complex64), doppler_axis(nfft, slow_time_interval_s, wavelength_m, window)


def range_doppler_map(Z: np.ndarray, combine: str = "noncoherent") -> np.ndarray:
    """Z[c, doppler, range] -> power map P[range, doppler].

    ``noncoherent``: sum of |Z|^2 over channels (angle-independent detection map).
    """
    if Z.ndim != 3:
        raise ValueError("expected Z[channel, doppler, range]")
    if combine == "noncoherent":
        P = (np.abs(Z) ** 2).sum(axis=0)
    elif combine == "first_channel":
        P = np.abs(Z[0]) ** 2
    else:
        raise ValueError(f"unknown combine mode {combine!r}")
    return P.T.astype(np.float64)


def tdm_phase_compensation(nu_hz: np.ndarray | float, time_offsets_s: np.ndarray) -> np.ndarray:
    """Factors exp(-j 2 pi nu t_c) removing the Doppler phase of TDM slot offsets.

    Returns shape (C,) for scalar nu or (len(nu), C). Multiply Doppler-domain
    channel values by these before angle estimation.
    """
    nu = np.atleast_1d(np.asarray(nu_hz, float))
    f = np.exp(-2j * np.pi * nu[:, None] * np.asarray(time_offsets_s)[None, :])
    return f[0] if np.ndim(nu_hz) == 0 else f
