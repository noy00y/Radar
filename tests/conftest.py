"""Shared fixtures: small, fast, deterministic configurations (CPU only, no hardware)."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import copy  # noqa: E402

import pytest  # noqa: E402

from radar_stap.config import deep_merge, simulation_config_from_dict  # noqa: E402

# A compact IWR1843-like waveform: 77 GHz, 30 MHz/us, 10 Msps complex, 128 samples.
#   sampled bandwidth 384 MHz -> 0.39 m resolution; IF-limited range 50 m
#   T_c = 30 us -> +-32 m/s unambiguous
BASE = {
    "seed": 1234,
    "scenario_name": "test",
    "radar": {
        "carrier_frequency_hz": 77.0e9,
        "chirp_slope_hz_per_s": 30.0e12,
        "idle_time_s": 10.0e-6,
        "adc_start_time_s": 4.0e-6,
        "ramp_end_time_s": 20.0e-6,
        "adc_sample_rate_hz": 10.0e6,
        "num_adc_samples": 128,
        "num_chirps": 32,
        "sampling": "complex",
        "mixer_convention": "tx_times_conj_rx",
        "noise_power": 1.0,
    },
    "array": {
        "position_units": "half_wavelength",
        "rx_positions": [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
        "tx_positions": [[0, 0, 0]],
        "tx_schedule": [0],
    },
    "platform": {"position_m": [0.0, 0.0, 0.0], "velocity_mps": [0.0, 0.0, 0.0]},
    "clutter": {"enabled": False},
    "targets": [],
    "processing": {"range_min_m": 1.0, "range_max_m": 48.0},
    "detection": {"guard_cells": [2, 1], "training_cells": [8, 4], "pfa": 1e-6},
    "stap": {"enabled": False, "methods": ["nonadaptive"]},
    "monte_carlo": {"enabled": False},
    "output": {"save_raw": False},
}

# Side-looking moving platform with distributed clutter, D = 4 x 8 = 32.
STAP_SCENE = {
    "radar": {"num_chirps": 8},
    "platform": {"position_m": [0.0, 0.0, 5.0], "velocity_mps": [10.0, 0.0, 0.0]},
    "clutter": {
        "enabled": True,
        "num_scatterers": 800,
        "range_min_m": 8.0,
        "range_max_m": 45.0,
        "azimuth_min_deg": -80.0,
        "azimuth_max_deg": 80.0,
        "cnr_db": 25.0,
        "range_power_exponent": 0.0,
    },
    "targets": [
        # endo-clutter mover: same Doppler as sidelobe clutter near -53 deg, ~1.3 Doppler cells from mainbeam clutter
        {"name": "t1", "range_m": 25.0, "angle_deg": 15.0, "radial_velocity_mps": 8.0, "snr_db": -12.0},
    ],
    "processing": {"range_min_m": 7.0, "range_max_m": 47.0},
    "stap": {
        "enabled": True,
        "look_angle_deg": 15.0,
        "num_training": 64,
        "num_guard": 2,
        "num_doppler_hypotheses": 16,
        "methods": ["nonadaptive", "mvdr", "efa", "pc"],
        "loading_mode": "noise_relative",
        "loading_value": 1.0,
    },
    "monte_carlo": {"enabled": False, "num_trials": 3, "test_halfwidth_bins": 6, "basis_halfwidth_bins": 12},
}


def make_cfg(*overrides: dict, validate: bool = True):
    data = copy.deepcopy(BASE)
    for o in overrides:
        data = deep_merge(data, o)
    return simulation_config_from_dict(data, validate=validate)


@pytest.fixture
def base_cfg():
    return make_cfg()


@pytest.fixture(scope="session")
def stap_scene():
    """Simulated side-looking clutter scene (session-scoped: simulated once)."""
    from radar_stap.experiments import process_cpi
    from radar_stap.simulator import simulate

    cfg = make_cfg(STAP_SCENE)
    sim = simulate(cfg)
    proc = process_cpi(sim.data, cfg)
    return cfg, sim, proc
