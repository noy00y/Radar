"""FMCW linear-FM chirp model, dechirping, and the beat (IF) signal.

Signal model (documented convention)
------------------------------------
During the ramp of one chirp the transmitted analytic signal is

    s_tx(t_f) = exp(j * phi_tx(t_f)),   phi_tx(t_f) = 2*pi*(f_c*t_f + S*t_f^2/2)

where ``t_f`` is time since ramp start, ``f_c`` the RF frequency at ramp start
(TI ``startFreq``) and ``S`` the slope. A point scatterer whose two-way path
length is L(t) produces an echo delayed by tau(t) = L(t)/c:

    s_rx(t_f) = A * s_tx(t_f - tau)

Dechirping with the ``tx_times_conj_rx`` mixer gives the beat signal

    x_IF(t_f) = s_tx(t_f) * conj(s_rx(t_f))
              = conj(A) * exp(j * 2*pi*(f_c*tau + S*t_f*tau - S*tau^2/2))

(exact for a constant tau; we evaluate it with tau = tau(t), i.e. the delay of
the echo arriving at time t, which is exact to first order in v/c).

With L = 2(R0 + v_r t) (monostatic) the instantaneous IF frequency is

    f_b = S*tau + (f_c + S*t_f) * 2 v_r / c  ~=  S*tau + 2 v_r / lambda

i.e. ``f_b = S*tau - f_d`` with the classical Doppler shift
``f_d = -2 v_r / lambda`` (positive for approaching targets).  Across chirps the
phase term ``2*pi*f_c*tau`` advances by ``2*pi*(2 v_r/lambda)*T_c`` per chirp, so
the slow-time phase rate is ``nu = +2 v_r / lambda`` (positive = receding).

The ``rx_times_conj_tx`` mixer conjugates the IF signal; every frequency
(beat and slow-time) flips sign. The receiver must be told which convention
the data follows (verified on hardware with a corner reflector).

This is an FMCW dechirp (stretch) model, not a pulsed-LFM matched-filter model:
range information is in the beat frequency of the *dechirped* samples.
"""

from __future__ import annotations

import numpy as np

from .config import RadarConfig, wavelength_for
from .constants import MIXER_RX_CONJ_TX, MIXER_TX_CONJ_RX, SPEED_OF_LIGHT


def chirp_period(radar: RadarConfig) -> float:
    """Chirp repetition interval T_c = idle + ramp_end (TI timing model)."""
    return radar.idle_time_s + radar.ramp_end_time_s


def fast_time_axis(radar: RadarConfig) -> np.ndarray:
    """ADC sampling instants measured from ramp start [s]: adc_start + k/fs."""
    k = np.arange(radar.num_adc_samples)
    return radar.adc_start_time_s + k / radar.adc_sample_rate_hz


def ramp_start_times(radar: RadarConfig, num_chirps: int | None = None) -> np.ndarray:
    """Absolute ramp-start time of each chirp in the CPI [s]: n*T_c + idle."""
    n = np.arange(radar.num_chirps if num_chirps is None else num_chirps)
    return n * chirp_period(radar) + radar.idle_time_s


def tx_phase(radar: RadarConfig, t_f: np.ndarray) -> np.ndarray:
    """RF phase of the transmitted chirp [rad] at fast time t_f (from ramp start)."""
    t_f = np.asarray(t_f, dtype=float)
    return 2 * np.pi * (radar.carrier_frequency_hz * t_f + 0.5 * radar.chirp_slope_hz_per_s * t_f**2)


def instantaneous_frequency(radar: RadarConfig, t_f: np.ndarray) -> np.ndarray:
    """f_c + S t_f [Hz]."""
    return radar.carrier_frequency_hz + radar.chirp_slope_hz_per_s * np.asarray(t_f)


def baseband_chirp(radar: RadarConfig, t_f: np.ndarray) -> np.ndarray:
    """Transmitted chirp with the carrier removed, exp(j*pi*S*t_f^2) (for plotting)."""
    t_f = np.asarray(t_f, dtype=float)
    return np.exp(1j * np.pi * radar.chirp_slope_hz_per_s * t_f**2)


def if_phase(
    tau: np.ndarray,
    t_f: np.ndarray,
    carrier_hz: float,
    slope_hz_per_s: float,
    mixer: str = MIXER_TX_CONJ_RX,
) -> np.ndarray:
    """Beat-signal phase for delay ``tau`` at fast time ``t_f`` (broadcasting).

    phi = 2*pi*(f_c*tau + S*t_f*tau - S*tau^2/2), negated for ``rx_times_conj_tx``.
    """
    phi = 2 * np.pi * (carrier_hz * tau + slope_hz_per_s * t_f * tau - 0.5 * slope_hz_per_s * tau**2)
    if mixer == MIXER_TX_CONJ_RX:
        return phi
    if mixer == MIXER_RX_CONJ_TX:
        return -phi
    raise ValueError(f"unknown mixer convention {mixer!r}")


def dechirp(tx: np.ndarray, rx: np.ndarray, mixer: str = MIXER_TX_CONJ_RX) -> np.ndarray:
    """Explicit mixer: tx*conj(rx) or rx*conj(tx)."""
    if mixer == MIXER_TX_CONJ_RX:
        return tx * np.conj(rx)
    if mixer == MIXER_RX_CONJ_TX:
        return rx * np.conj(tx)
    raise ValueError(f"unknown mixer convention {mixer!r}")


def beat_signal_brute_force(radar: RadarConfig, tau: float, t_f: np.ndarray) -> np.ndarray:
    """Reference implementation: build tx and delayed rx explicitly, then mix.

    Used by tests to verify :func:`if_phase` (the closed form used by the
    simulator) against the definition. Valid for 0 <= tau <= t_f.
    """
    tx = np.exp(1j * tx_phase(radar, t_f))
    rx = np.exp(1j * tx_phase(radar, np.asarray(t_f) - tau))
    return dechirp(tx, rx, radar.mixer_convention)


def round_trip_delay(range_m: np.ndarray | float) -> np.ndarray | float:
    """Monostatic delay tau = 2R/c."""
    return 2.0 * np.asarray(range_m) / SPEED_OF_LIGHT


def slow_time_frequency(radial_velocity_mps, wavelength_m: float):
    """Slow-time phase rate nu = 2 v_r / lambda (positive receding, tx_times_conj_rx)."""
    return 2.0 * np.asarray(radial_velocity_mps) / wavelength_m


def classical_doppler_shift(radial_velocity_mps, wavelength_m: float):
    """Classical Doppler shift f_d = -2 v_r / lambda (positive approaching)."""
    return -2.0 * np.asarray(radial_velocity_mps) / wavelength_m


def beat_frequency(range_m, radial_velocity_mps, radar: RadarConfig):
    """Expected beat frequency f_b = S*2R/c - f_d = S*2R/c + 2 v_r/lambda (tx_times_conj_rx)."""
    lam = wavelength_for(radar)
    fb = radar.chirp_slope_hz_per_s * round_trip_delay(range_m) + slow_time_frequency(radial_velocity_mps, lam)
    return fb if radar.mixer_convention == MIXER_TX_CONJ_RX else -fb


def complex_noise(rng: np.random.Generator, shape, power: float, dtype=np.complex64) -> np.ndarray:
    """Circular complex Gaussian noise with E|n|^2 = power."""
    scale = np.sqrt(power / 2.0)
    out = rng.standard_normal(shape) * scale + 1j * rng.standard_normal(shape) * scale
    return out.astype(dtype, copy=False)
