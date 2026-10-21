"""Fast-time (range) processing: windowing, range FFT, beat-frequency/range axes.

Range conversion uses R = f_b * c / (2 S) with the configured slope S. The
small range-Doppler coupling term (f_b also contains 2 v_r / lambda, i.e. a
range offset of v_r * f_c / S) is *not* removed by the axis; it is reported by
:func:`expected_range_bin` and accounted for in evaluation.

Normalisation: un-normalised FFT. For white noise of variance sigma^2 per
sample the output variance per bin is sigma^2 * sum(w^2); a tone of amplitude A
exactly on a bin peaks at |A| * sum(w).

Zero padding (``range_nfft > N``) interpolates the spectrum on a finer grid; it
does **not** change the physical resolution c / (2 B_sampled) (times the
window broadening factor).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import windows as sigwin

from .config import ProcessingConfig, RadarConfig, wavelength_for
from .constants import MIXER_RX_CONJ_TX, SPEED_OF_LIGHT


def get_window(name: str, n: int) -> np.ndarray:
    """Symmetric taper of length n."""
    if n < 1:
        raise ValueError("window length must be >= 1")
    if name == "rect":
        return np.ones(n)
    if name == "hann":
        return sigwin.hann(n, sym=True) if n > 2 else np.ones(n)
    if name == "hamming":
        return sigwin.hamming(n, sym=True)
    if name == "blackman":
        return sigwin.blackman(n, sym=True)
    if name == "taylor":
        return sigwin.taylor(n, nbar=4, sll=30, norm=True, sym=True)
    if name == "chebyshev":
        return sigwin.chebwin(n, at=50, sym=True)
    raise ValueError(f"unknown window {name!r}")


@dataclass(frozen=True)
class RangeAxis:
    beat_hz: np.ndarray
    range_m: np.ndarray
    nfft: int
    band: str
    window: str
    coherent_gain: float  # sum(w)
    noise_gain: float  # sum(w^2)
    bin_spacing_m: float
    resolution_m: float  # nominal c/(2B) times window 3-dB broadening factor (approx.)

    @property
    def num_bins(self) -> int:
        return int(self.range_m.size)

    def bin_of_range(self, range_m: float) -> float:
        """Fractional bin index for a beat frequency equal to S*2R/c (no Doppler coupling)."""
        return float(np.interp(range_m, self.range_m, np.arange(self.num_bins)))


_WINDOW_BROADENING = {"rect": 0.886, "hann": 1.44, "hamming": 1.30, "blackman": 1.68, "taylor": 1.2, "chebyshev": 1.4}


def range_axis(radar: RadarConfig, nfft: int, band: str = "positive", window: str = "hann") -> RangeAxis:
    fs = radar.adc_sample_rate_hz
    if radar.sampling == "real":
        f = np.arange(nfft // 2 + 1) * fs / nfft
        band = "positive"
    elif band == "positive":
        f = np.arange(nfft) * fs / nfft
    elif band == "two_sided":
        f = (np.arange(nfft) - nfft // 2) * fs / nfft
    else:
        raise ValueError(f"unknown beat band {band!r}")
    w = get_window(window, radar.num_adc_samples)
    b = radar.chirp_slope_hz_per_s * radar.num_adc_samples / fs
    return RangeAxis(
        beat_hz=f,
        range_m=f * SPEED_OF_LIGHT / (2 * radar.chirp_slope_hz_per_s),
        nfft=nfft,
        band=band,
        window=window,
        coherent_gain=float(w.sum()),
        noise_gain=float((w**2).sum()),
        bin_spacing_m=SPEED_OF_LIGHT * fs / (2 * radar.chirp_slope_hz_per_s * nfft),
        resolution_m=SPEED_OF_LIGHT / (2 * b) * _WINDOW_BROADENING.get(window, 1.0) / 0.886,
    )


def normalize_mixer(x: np.ndarray, radar: RadarConfig) -> np.ndarray:
    """Bring data to the package-internal ``tx_times_conj_rx`` convention.

    For ``rx_times_conj_tx`` complex data this is a conjugation; afterwards
    targets have positive beat frequency and receding targets positive
    slow-time frequency. Real data are unaffected.
    """
    if radar.sampling == "complex" and radar.mixer_convention == MIXER_RX_CONJ_TX:
        return np.conj(x)
    return x


def range_fft(
    x: np.ndarray,
    radar: RadarConfig,
    processing: ProcessingConfig | None = None,
) -> tuple[np.ndarray, RangeAxis]:
    """Window + FFT along the last (fast-time) axis. Returns (Y complex64, axis).

    Input layout ``(..., num_adc_samples)``, e.g. X[rx, chirp, sample].
    """
    processing = processing or ProcessingConfig()
    if x.shape[-1] != radar.num_adc_samples:
        raise ValueError(f"last axis has {x.shape[-1]} samples, expected {radar.num_adc_samples}")
    nfft = processing.range_nfft or radar.num_adc_samples
    x = normalize_mixer(np.asarray(x), radar)
    if processing.remove_dc:
        x = x - x.mean(axis=-1, keepdims=True)
    w = get_window(processing.range_window, radar.num_adc_samples)
    xw = x * w
    if radar.sampling == "real":
        Y = np.fft.rfft(np.real(xw), n=nfft, axis=-1)
    else:
        Y = np.fft.fft(xw, n=nfft, axis=-1)
        if processing.beat_band == "two_sided":
            Y = np.fft.fftshift(Y, axes=-1)
    axis = range_axis(radar, nfft, processing.beat_band, processing.range_window)
    return Y.astype(np.complex64), axis


def range_mask(axis: RangeAxis, range_min_m: float = 0.0, range_max_m: float | None = None) -> np.ndarray:
    m = axis.range_m >= range_min_m
    if range_max_m is not None:
        m &= axis.range_m <= range_max_m
    return m


def range_profile(Y: np.ndarray) -> np.ndarray:
    """Mean power over all leading axes (channels, chirps) -> (bins,)."""
    p = np.abs(Y) ** 2
    return p.reshape(-1, p.shape[-1]).mean(axis=0)


def expected_beat_frequency(range_m: float, radial_velocity_mps: float, radar: RadarConfig) -> float:
    """f_b = S*2R/c + 2 v_r / lambda in the internal (tx_times_conj_rx) convention."""
    lam = wavelength_for(radar)
    return radar.chirp_slope_hz_per_s * 2 * range_m / SPEED_OF_LIGHT + 2 * radial_velocity_mps / lam


def expected_range_bin(range_m: float, radial_velocity_mps: float, radar: RadarConfig, axis: RangeAxis) -> float:
    """Fractional FFT bin where the target peak is expected (includes range-Doppler coupling)."""
    fb = expected_beat_frequency(range_m, radial_velocity_mps, radar)
    step = radar.adc_sample_rate_hz / axis.nfft
    if axis.band == "two_sided":
        return fb / step + axis.nfft // 2
    return fb / step


def apparent_range(range_m: float, radial_velocity_mps: float, radar: RadarConfig) -> float:
    """Range that the FFT axis reports for this target (true range + coupling offset)."""
    return expected_beat_frequency(range_m, radial_velocity_mps, radar) * SPEED_OF_LIGHT / (
        2 * radar.chirp_slope_hz_per_s
    )


def parabolic_peak(values: np.ndarray, index: int, log_domain: bool = True) -> float:
    """Sub-bin peak location by fitting a parabola through 3 points (circular indexing)."""
    n = values.size
    y = np.array([values[(index - 1) % n], values[index], values[(index + 1) % n]], float)
    if log_domain:
        y = np.log(np.maximum(y, np.finfo(float).tiny))
    den = y[0] - 2 * y[1] + y[2]
    if den >= 0 or not np.isfinite(den):
        return float(index)
    return float(index + 0.5 * (y[0] - y[2]) / den)
