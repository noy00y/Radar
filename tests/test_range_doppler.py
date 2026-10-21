"""Range / Doppler / angle processing validated against known ground truth."""

import numpy as np
import pytest

from radar_stap.angle_processing import estimate_angle
from radar_stap.config import ProcessingConfig, sim_derived, wavelength_for
from radar_stap.constants import SPEED_OF_LIGHT
from radar_stap.datacube import from_virtual_cube, snapshots, to_virtual_cube
from radar_stap.doppler_processing import doppler_fft, tdm_phase_compensation
from radar_stap.experiments import detect_range_doppler, evaluate_rd_detections, process_cpi, truth_cell
from radar_stap.range_processing import get_window, range_fft
from radar_stap.simulator import simulate
from radar_stap.stap import SpaceTimeModel
from radar_stap.angle_processing import direction_from_angles

from conftest import make_cfg


def _targets(*specs):
    out = []
    for i, (r, a, v, snr) in enumerate(specs):
        t = {"name": f"t{i}", "range_m": r, "angle_deg": a, "snr_db": snr}
        if v:
            t["radial_velocity_mps"] = v
        out.append(t)
    return {"targets": out}


def _detect_all(cfg):
    sim = simulate(cfg)
    proc = process_cpi(sim.data, cfg)
    dets, _ = detect_range_doppler(proc, cfg)
    return sim, proc, dets, evaluate_rd_detections(dets, sim.scene.truths, proc, cfg)


def test_single_target_range_velocity_angle():
    cfg = make_cfg(_targets((23.7, 12.0, -6.3, -5.0)))
    d = sim_derived(cfg)
    sim, proc, dets, ev = _detect_all(cfg)
    row = ev["targets"][0]
    assert row["detected"]
    e = row["errors"]
    assert abs(e["range_m"]) < 0.25 * d.range_bin_spacing_m
    assert abs(e["radial_velocity_mps"]) < 0.25 * d.velocity_resolution_mps
    assert abs(e["angle_deg"]) < 1.0
    assert len(dets) == 1  # no false alarms on a noise-only background at Pfa 1e-6


def test_multiple_targets_distinct_cells():
    specs = [(10.0, 0.0, 3.0, -5.0), (18.0, -20.0, -12.0, -5.0), (31.0, 30.0, 0.0, 0.0), (42.0, 5.0, 20.0, -2.0)]
    cfg = make_cfg(_targets(*specs))
    d = sim_derived(cfg)
    _, _, dets, ev = _detect_all(cfg)
    assert all(r["detected"] for r in ev["targets"])
    for r in ev["targets"]:
        assert abs(r["errors"]["range_m"]) < 0.3 * d.range_bin_spacing_m
        assert abs(r["errors"]["radial_velocity_mps"]) < 0.3 * d.velocity_resolution_mps
    assert ev["num_unassociated_detections"] == 0


def test_static_reflector_zero_doppler_for_stationary_radar():
    cfg = make_cfg({"stationary_reflectors": [{"name": "s", "range_m": 15.0, "angle_deg": -10.0, "snr_db": 0.0}]})
    sim = simulate(cfg)
    proc = process_cpi(sim.data, cfg)
    rb, kb = truth_cell(sim.scene.truths[0], cfg, proc)
    k = int(np.argmax(proc.rd_map[int(round(rb))]))
    assert proc.doppler_axis.velocity_mps[k] == 0.0
    assert kb == pytest.approx(proc.doppler_axis.nfft // 2)


def test_axes_consistency():
    cfg = make_cfg({"processing": {"range_nfft": 256, "doppler_nfft": 64}})
    d = sim_derived(cfg)
    sim = simulate(cfg)
    proc = process_cpi(sim.data, cfg)
    r = cfg.radar
    assert proc.range_axis.bin_spacing_m == pytest.approx(SPEED_OF_LIGHT * r.adc_sample_rate_hz / (2 * r.chirp_slope_hz_per_s * 256))
    assert np.allclose(np.diff(proc.range_axis.range_m), proc.range_axis.bin_spacing_m)
    lam = wavelength_for(r)
    dv = np.diff(proc.doppler_axis.velocity_mps)
    assert np.allclose(dv, lam / (2 * 64 * d.slow_time_interval_s))
    # physical velocity resolution uses the number of chirps, not the FFT size
    assert d.velocity_resolution_mps == pytest.approx(lam / (2 * r.num_chirps * d.chirp_period_s))
    assert proc.doppler_axis.velocity_mps.min() == pytest.approx(-d.max_unambiguous_velocity_mps)


def test_zero_padding_interpolates_but_does_not_improve_resolution():
    cfg1 = make_cfg(_targets((20.0, 0.0, 0.0, 30.0)))
    cfg4 = make_cfg(_targets((20.0, 0.0, 0.0, 30.0)), {"processing": {"range_nfft": 1024}})
    widths, peaks = [], []
    for cfg in (cfg1, cfg4):
        sim = simulate(cfg)
        Y, ax = range_fft(sim.components["targets"], cfg.radar, cfg.processing)
        p = np.abs(Y[0, 0]) ** 2
        k = int(np.argmax(p))
        peaks.append(ax.range_m[k])
        above = np.where(p >= p[k] / 2)[0]
        widths.append((above.max() - above.min() + 1) * ax.bin_spacing_m)
    assert peaks[0] == pytest.approx(peaks[1], abs=sim_derived(cfg1).range_bin_spacing_m / 2)
    # -3 dB width in metres: on the coarse grid it is quantised to whole bins; on the fine grid it
    # converges to the Hann mainlobe (~1.44 c/2B). Zero padding must not make it narrower than that.
    res = sim_derived(cfg1).range_resolution_m
    assert widths[1] == pytest.approx(1.44 * res, rel=0.15)
    assert widths[1] >= 1.2 * res


def test_range_fft_noise_and_amplitude_scaling():
    cfg = make_cfg()
    r = cfg.radar
    w = get_window("hann", r.num_adc_samples)
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((200, r.num_adc_samples)) + 1j * rng.standard_normal((200, r.num_adc_samples))) / np.sqrt(2)
    Y, _ = range_fft(x, r, ProcessingConfig())
    assert np.mean(np.abs(Y) ** 2) == pytest.approx((w**2).sum(), rel=0.03)
    # on-bin tone of amplitude A peaks at A * sum(w)
    k0, A = 17, 3.0
    t = np.arange(r.num_adc_samples)
    tone = A * np.exp(2j * np.pi * k0 * t / r.num_adc_samples)
    Yt, _ = range_fft(tone, r, ProcessingConfig())
    assert np.abs(Yt[k0]) == pytest.approx(A * w.sum(), rel=1e-5)


def test_angle_estimation_uniform_and_nonuniform_arrays():
    for rx in ([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], [[0, 0, 0], [1, 0, 0], [3, 0, 0], [4, 0, 0]]):
        for ang in (-40.0, -7.5, 0.0, 22.0, 50.0):
            cfg = make_cfg(_targets((20.0, ang, 0.0, 10.0)), {"array": {"rx_positions": rx}})
            sim = simulate(cfg)
            proc = process_cpi(sim.data, cfg)
            rb, kb = truth_cell(sim.scene.truths[0], cfg, proc)
            snap = proc.Z[:, int(round(kb)), int(round(rb))]
            est, _, _ = estimate_angle(snap, proc.model.positions, proc.model.wavelength)
            assert est == pytest.approx(ang, abs=0.6), (rx, ang)


def test_virtual_cube_roundtrip_and_stacking():
    rng = np.random.default_rng(1)
    Y = rng.standard_normal((4, 12, 5)) + 1j * rng.standard_normal((4, 12, 5))
    sched = [0, 2, 1]
    V = to_virtual_cube(Y, sched)
    assert V.shape == (12, 4, 5)
    # virtual channel p*M+m, loop l == physical chirp l*P+p of RX m
    assert V[2 * 4 + 3, 1, 4] == Y[3, 1 * 3 + 2, 4]
    assert np.array_equal(from_virtual_cube(V, 4, sched), Y)
    X = snapshots(V)
    assert X[4, 1 * 12 + 7] == V[7, 1, 4]


def test_tdm_mimo_velocity_and_angle_need_slot_compensation():
    """2-TX TDM: Doppler on per-TX sequences; angle needs the TX-slot Doppler phase removed."""
    tdm = {"radar": {"num_chirps": 32}, "array": {"tx_positions": [[0, 0, 0], [4, 0, 0]], "tx_schedule": [0, 1]}}
    v, ang = 9.0, 18.0
    cfg = make_cfg(_targets((22.0, ang, v, 0.0)), tdm)
    d = sim_derived(cfg)
    sim, proc, dets, ev = _detect_all(cfg)
    row = ev["targets"][0]
    assert row["detected"]
    assert abs(row["errors"]["radial_velocity_mps"]) < 0.3 * d.velocity_resolution_mps
    assert abs(row["errors"]["angle_deg"]) < 1.0  # compensated (inside detect_range_doppler)
    rb, kb = truth_cell(sim.scene.truths[0], cfg, proc)
    snap = proc.Z[:, int(round(kb)), int(round(rb))]
    raw, _, _ = estimate_angle(snap, proc.model.positions, proc.model.wavelength)
    assert abs(raw - ang) > 3.0  # uncompensated TDM phase biases the angle


def test_simulator_matches_processor_space_time_steering():
    """Correlation between the simulated (exact-geometry) target snapshot and the far-field model."""
    for tdm in (False, True):
        extra = {"array": {"tx_positions": [[0, 0, 0], [4, 0, 0]], "tx_schedule": [0, 1]}} if tdm else {}
        cfg = make_cfg(_targets((25.0, -17.0, 7.0, 10.0)), {"platform": {"velocity_mps": [3.0, 0.0, 0.0]}}, extra)
        sim = simulate(cfg)
        Y, ax = range_fft(sim.components["targets"], cfg.radar, cfg.processing)
        V = to_virtual_cube(Y, cfg.array.tx_schedule)
        t = sim.scene.truths[0]
        rb, _ = truth_cell(t, cfg, process_cpi(sim.data, cfg))
        x = snapshots(V)[int(round(rb))]
        model = SpaceTimeModel.from_config(cfg)
        s = model.steering(direction_from_angles(t.angle_deg), 2 * t.radial_velocity_mps / model.wavelength)
        rho = abs(np.vdot(s, x)) / (np.linalg.norm(s) * np.linalg.norm(x))
        assert rho > 0.995, (tdm, rho)


def test_doppler_fft_shift_convention():
    lam, T = 0.004, 1e-4
    n = np.arange(32)
    nu = 1250.0
    x = np.exp(2j * np.pi * nu * n * T)[None, :, None]
    Z, ax = doppler_fft(x, T, lam, "rect")
    k = int(np.argmax(np.abs(Z[0, :, 0])))
    assert ax.freq_hz[k] == pytest.approx(nu)
    assert ax.velocity_mps[k] == pytest.approx(nu * lam / 2)
    assert np.allclose(tdm_phase_compensation(nu, np.array([0.0, T])), [1.0, np.exp(-2j * np.pi * nu * T)])
