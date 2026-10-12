from __future__ import annotations

import numpy as np
from .config import RadarConfig


def range_fft(
    data: np.ndarray,
    cfg: RadarConfig,
    n_fft: int | None = None,
    window: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Range FFT along fast time.

    Input shape: (num_chirps, samples_per_chirp)
    Returns:
      spectrum shape (num_chirps, n_fft)
      nonnegative range axis in metres, same length as spectrum
    """
    if data.ndim != 2:
        raise ValueError("data must have shape (num_chirps, samples_per_chirp)")
    if data.shape[1] != cfg.samples_per_chirp:
        raise ValueError("fast-time dimension does not match config")
    n_fft = n_fft or cfg.samples_per_chirp
    x = data
    if window:
        x = x * np.hanning(x.shape[1])[None, :]
    spectrum = np.fft.fft(x, n=n_fft, axis=1)
    # For the chosen model, beat frequencies are expected positive for normal ranges.
    half = n_fft // 2
    spectrum = spectrum[:, :half]
    beat_freq = np.fft.fftfreq(n_fft, d=1.0 / cfg.adc_sample_rate_hz)[:half]
    ranges_m = beat_freq * 299_792_458.0 / (2.0 * cfg.slope_hz_per_s)
    return spectrum, ranges_m


def range_doppler_fft(
    data: np.ndarray,
    cfg: RadarConfig,
    n_range_fft: int | None = None,
    n_doppler_fft: int | None = None,
    window: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute range-Doppler magnitude using range FFT then slow-time FFT.

    Returns magnitude_db, range_axis_m, velocity_axis_mps.
    The velocity-axis sign follows the simulator's stated Doppler convention.
    """
    rspec, ranges_m = range_fft(data, cfg, n_fft=n_range_fft, window=window)
    x = rspec
    if window:
        x = x * np.hanning(x.shape[0])[:, None]
    nd = n_doppler_fft or cfg.num_chirps
    rd = np.fft.fftshift(np.fft.fft(x, n=nd, axis=0), axes=0)
    mag_db = 20.0 * np.log10(np.maximum(np.abs(rd), 1e-12))
    fd = np.fft.fftshift(np.fft.fftfreq(nd, d=cfg.chirp_repetition_interval_s))
    velocity_mps = fd * cfg.wavelength_m / 2.0
    return mag_db, ranges_m, velocity_mps
