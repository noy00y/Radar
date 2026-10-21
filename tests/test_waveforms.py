"""Waveform, dechirp and sign-convention tests against analytical expectations."""

import numpy as np
import pytest

from radar_stap.config import wavelength_for
from radar_stap.constants import MIXER_RX_CONJ_TX, SPEED_OF_LIGHT
from radar_stap.range_processing import parabolic_peak
from radar_stap.simulator import RadarGeometry, ScattererSet, path_lengths, simulate, synthesize
from radar_stap.waveforms import (
    beat_signal_brute_force,
    chirp_period,
    complex_noise,
    fast_time_axis,
    if_phase,
    instantaneous_frequency,
    ramp_start_times,
    round_trip_delay,
    tx_phase,
)

from conftest import make_cfg


def _single(range_m=20.0, v_r=0.0, angle=0.0, snr=20.0, **kw):
    t = {"name": "t", "range_m": range_m, "angle_deg": angle, "snr_db": snr, "phase_deg": 0.0}
    if v_r:
        t["radial_velocity_mps"] = v_r
    return make_cfg({"targets": [t]}, *kw.get("extra", []))


def test_fast_time_axis_length_and_spacing(base_cfg):
    r = base_cfg.radar
    t = fast_time_axis(r)
    assert t.size == r.num_adc_samples
    assert t[0] == pytest.approx(r.adc_start_time_s)
    assert np.allclose(np.diff(t), 1 / r.adc_sample_rate_hz)
    assert chirp_period(r) == pytest.approx(r.idle_time_s + r.ramp_end_time_s)
    assert np.allclose(np.diff(ramp_start_times(r)), chirp_period(r))


def test_chirp_phase_progression_and_slope(base_cfg):
    r = base_cfg.radar
    t = np.linspace(0, r.ramp_end_time_s, 20001)
    f_num = np.gradient(tx_phase(r, t), t) / (2 * np.pi)
    assert np.allclose(f_num[1:-1], instantaneous_frequency(r, t)[1:-1], rtol=1e-9)
    slope = np.polyfit(t, f_num, 1)[0]
    assert slope == pytest.approx(r.chirp_slope_hz_per_s, rel=1e-6)


@pytest.mark.parametrize("tau", [1e-8, 1.3e-7, 2.9e-7])
def test_closed_form_if_phase_matches_explicit_mixing(base_cfg, tau):
    r = base_cfg.radar
    t = fast_time_axis(r)
    ref = beat_signal_brute_force(r, tau, t)
    closed = np.exp(1j * if_phase(tau, t, r.carrier_frequency_hz, r.chirp_slope_hz_per_s, r.mixer_convention))
    assert np.max(np.abs(np.angle(ref * np.conj(closed)))) < 1e-5


def test_mixer_conventions_are_conjugates(base_cfg):
    r = base_cfg.radar
    t = fast_time_axis(r)
    p1 = if_phase(1e-7, t, r.carrier_frequency_hz, r.chirp_slope_hz_per_s)
    p2 = if_phase(1e-7, t, r.carrier_frequency_hz, r.chirp_slope_hz_per_s, MIXER_RX_CONJ_TX)
    assert np.allclose(p1, -p2)


def test_round_trip_delay_from_geometry():
    cfg = _single(range_m=17.0)
    geom = RadarGeometry.from_config(cfg)
    # single element at the origin -> path = 2R exactly
    geom.rx_body = np.zeros((1, 3))
    scat = ScattererSet("t", np.array([[0.0, 17.0, 0.0]]), np.zeros((1, 3)), np.ones(1), np.ones(1))
    L, Ld = path_lengths(scat, geom, ramp_start_times(cfg.radar))
    assert np.allclose(L, 34.0) and np.allclose(Ld, 0.0)
    assert round_trip_delay(17.0) == pytest.approx(34.0 / SPEED_OF_LIGHT)


def _beat_peak_hz(x, fs, pad=16):
    n = x.size * pad
    spec = np.abs(np.fft.fft(x * np.hanning(x.size), n)) ** 2
    k = int(np.argmax(spec))
    return parabolic_peak(spec, k) * fs / n


def test_beat_frequency_equals_slope_times_delay():
    R = 21.3
    cfg = _single(range_m=R)
    sim = simulate(cfg)
    x = sim.components["targets"][0, 0]
    fb = _beat_peak_hz(x, cfg.radar.adc_sample_rate_hz)
    expected = cfg.radar.chirp_slope_hz_per_s * 2 * R / SPEED_OF_LIGHT
    assert fb == pytest.approx(expected, abs=0.02 * cfg.radar.adc_sample_rate_hz / cfg.radar.num_adc_samples)


def test_range_doppler_coupling_in_beat_frequency():
    """f_b = S tau + 2 v_r / lambda: the beat frequency shifts with radial velocity."""
    lam = None
    fbs = {}
    for v in (-20.0, 20.0):
        cfg = _single(range_m=20.0, v_r=v)
        lam = wavelength_for(cfg.radar)
        sim = simulate(cfg)
        fbs[v] = _beat_peak_hz(sim.components["targets"][0, 0], cfg.radar.adc_sample_rate_hz, pad=64)
    shift = fbs[20.0] - fbs[-20.0]
    assert shift == pytest.approx(2 * (2 * 20.0 / lam), rel=0.1)


@pytest.mark.parametrize("mixer,sign", [("tx_times_conj_rx", +1), ("rx_times_conj_tx", -1)])
def test_doppler_sign_and_phase_progression(mixer, sign):
    """Receding target (v_r > 0): slow-time phase advances by 2*pi*(2 v_r/lambda)*T_c per chirp (tx*conj(rx))."""
    v = 4.0
    cfg = _single(range_m=20.0, v_r=v, extra=[{"radar": {"mixer_convention": mixer}}])
    sim = simulate(cfg)
    x = sim.components["targets"][0]  # (chirps, samples)
    X = np.fft.fft(x * np.hanning(x.shape[1]), axis=1)
    k = int(np.argmax(np.abs(X[0])))
    dphi = np.angle(X[1:, k] * np.conj(X[:-1, k]))
    lam = wavelength_for(cfg.radar)
    expected = 2 * np.pi * (2 * v / lam) * chirp_period(cfg.radar)
    expected = np.angle(np.exp(1j * expected))
    assert np.allclose(dphi, sign * expected, atol=2e-3)


def test_seeded_noise_is_deterministic_and_calibrated():
    a = complex_noise(np.random.default_rng(5), (200_000,), 2.5)
    b = complex_noise(np.random.default_rng(5), (200_000,), 2.5)
    assert np.array_equal(a, b)
    assert np.mean(np.abs(a) ** 2) == pytest.approx(2.5, rel=0.02)
    assert abs(np.mean(a.real * a.imag)) < 0.02  # circular


def test_simulation_is_deterministic_and_components_add_up():
    cfg = _single(range_m=12.0, v_r=2.0)
    s1, s2 = simulate(cfg), simulate(cfg)
    assert np.array_equal(s1.data, s2.data)
    total = sum(s1.components.values())
    assert np.allclose(total, s1.data, atol=1e-4)
    s3 = simulate(cfg, noise_seed=99)
    assert np.allclose(s3.components["targets"], s1.components["targets"])
    assert not np.allclose(s3.components["noise"], s1.components["noise"])


def test_synthesize_is_linear_in_coefficients():
    cfg = _single(range_m=12.0)
    geom = RadarGeometry.from_config(cfg)
    scat = ScattererSet("t", np.array([[1.0, 15.0, 0.0], [-2.0, 22.0, 0.0]]), np.zeros((2, 3)), np.ones(2), np.ones(2))
    a = synthesize(scat, cfg, geom, coefficients=np.array([1.0, 0.0]))
    b = synthesize(scat, cfg, geom, coefficients=np.array([0.0, 2.0j]))
    ab = synthesize(scat, cfg, geom, coefficients=np.array([1.0, 2.0j]))
    assert np.allclose(a + b, ab)
