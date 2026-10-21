"""Derived quantities, YAML handling and the optional backend interface."""

import numpy as np
import pytest

from radar_stap.backend import BackendUnavailable, batched_mvdr, get_array_module
from radar_stap.config import ConfigError, derive_quantities, sim_derived, wavelength_for
from radar_stap.constants import SPEED_OF_LIGHT
from radar_stap.stap import mvdr_weights

from conftest import make_cfg


def test_derived_quantities_match_formulas():
    cfg = make_cfg({"processing": {"range_nfft": 512}})
    r = cfg.radar
    d = sim_derived(cfg)
    B = r.chirp_slope_hz_per_s * r.num_adc_samples / r.adc_sample_rate_hz
    assert d.sampled_bandwidth_hz == pytest.approx(B)
    assert d.range_resolution_m == pytest.approx(SPEED_OF_LIGHT / (2 * B))
    assert d.range_bin_spacing_m == pytest.approx(d.range_resolution_m * r.num_adc_samples / 512)
    assert d.max_range_if_m == pytest.approx(r.adc_sample_rate_hz * SPEED_OF_LIGHT / (2 * r.chirp_slope_hz_per_s))
    lam = wavelength_for(r)
    tc = r.idle_time_s + r.ramp_end_time_s
    assert d.max_unambiguous_velocity_mps == pytest.approx(lam / (4 * tc))
    assert d.velocity_resolution_mps == pytest.approx(lam / (2 * r.num_chirps * tc))
    assert d.bytes_per_cpi_int16 == 4 * r.num_adc_samples * r.num_chirps * 4
    # centre of the sampled sweep defines the wavelength
    fc = r.carrier_frequency_hz + r.chirp_slope_hz_per_s * (r.adc_start_time_s + r.num_adc_samples / r.adc_sample_rate_hz / 2)
    assert lam == pytest.approx(SPEED_OF_LIGHT / fc)


def test_tdm_derived_slow_time():
    cfg = make_cfg({"radar": {"num_chirps": 48}, "array": {"tx_positions": [[0, 0, 0], [2, 0, 1], [4, 0, 0]],
                                                           "tx_schedule": [0, 1, 2]}})
    d = sim_derived(cfg)
    assert d.num_tx_slots == 3 and d.num_loops == 16
    assert d.slow_time_interval_s == pytest.approx(3 * d.chirp_period_s)


def test_real_sampling_halves_if_band():
    cfg = make_cfg({"radar": {"sampling": "real"}})
    d = derive_quantities(cfg.radar)
    assert d.max_beat_frequency_hz == pytest.approx(cfg.radar.adc_sample_rate_hz / 2)


def test_numeric_strings_in_yaml_accepted_but_garbage_rejected():
    assert make_cfg({"radar": {"carrier_frequency_hz": "77.0e9"}}).radar.carrier_frequency_hz == 77e9
    with pytest.raises(ConfigError):
        make_cfg({"radar": {"carrier_frequency_hz": "seventy-seven"}})


def test_backend_numpy_matches_reference_and_cupy_optional():
    xp = get_array_module("numpy")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((3, 40, 12)) + 1j * rng.standard_normal((3, 40, 12))
    S = rng.standard_normal((12, 5)) + 1j * rng.standard_normal((12, 5))
    W = batched_mvdr(X, S, 0.5, xp)
    for b in range(3):
        ref = mvdr_weights(X[b].T @ X[b].conj() / 40, S, 0.5)
        assert np.allclose(W[b], ref, atol=1e-10)
    try:
        get_array_module("cupy")
    except BackendUnavailable:
        pass  # expected on machines without CUDA
    with pytest.raises(BackendUnavailable):
        get_array_module("tpu")
