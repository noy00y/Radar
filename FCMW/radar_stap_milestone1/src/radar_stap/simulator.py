from __future__ import annotations

import numpy as np
from .config import RadarConfig

C0 = 299_792_458.0


def simulate_if_data(
    cfg: RadarConfig,
    targets: list[dict],
    noise_std: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Simulate complex FMCW dechirped IF data.

    Returns an array shaped (num_chirps, samples_per_chirp), complex128.

    Model assumptions:
    - one transmitter and one receiver channel;
    - ideal linear up-chirp;
    - targets remain at constant range and radial velocity over the CPI;
    - no phase noise, leakage, multipath, ADC quantization, or TX/RX timing effects;
    - positive radial velocity means receding from the radar.

    For each target:
      beat frequency is approximately S*2R/c - f_D
      slow-time phase advances according to f_D = 2v/lambda.
    This sign convention is explicit for the simulator and should be checked
    against the chosen TI configuration/data convention before physical comparison.
    """
    rng = np.random.default_rng(seed)
    n_chirps = cfg.num_chirps
    n_samples = cfg.samples_per_chirp
    fast_t = np.arange(n_samples) / cfg.adc_sample_rate_hz
    slow_t = np.arange(n_chirps) * cfg.chirp_repetition_interval_s

    out = np.zeros((n_chirps, n_samples), dtype=np.complex128)
    lam = cfg.wavelength_m

    for target in targets:
        distance = float(target["range_m"])
        velocity = float(target["radial_velocity_mps"])
        amplitude = float(target.get("amplitude", 1.0))

        delay_s = 2.0 * distance / C0
        doppler_hz = 2.0 * velocity / lam
        beat_hz = cfg.slope_hz_per_s * delay_s - doppler_hz

        fast_phase = 2.0 * np.pi * beat_hz * fast_t
        slow_phase = 2.0 * np.pi * doppler_hz * slow_t
        out += amplitude * np.exp(1j * (slow_phase[:, None] + fast_phase[None, :]))

    if noise_std > 0:
        noise = noise_std / np.sqrt(2.0) * (
            rng.standard_normal(out.shape) + 1j * rng.standard_normal(out.shape)
        )
        out += noise
    return out
