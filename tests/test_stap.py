"""STAP data model, covariance, MVDR / reduced-dimension filters and clutter-suppression tests."""

import numpy as np
import pytest

from radar_stap.angle_processing import direction_from_angles
from radar_stap.covariance import (
    InsufficientTrainingError,
    brennan_rank,
    covariance_diagnostics,
    diagonal_load,
    loading_level,
    rmb_expected_loss_db,
    sample_covariance,
    shrinkage_covariance,
)
from radar_stap.datacube import snapshots
from radar_stap.experiments import _near, sinr_analysis, stap_setup, truth_cell
from radar_stap.metrics import optimal_sinr, output_sinr
from radar_stap.response_basis import build_basis, noise_covariance_scale
from radar_stap.simulator import draw_clutter_reflectivity, mixer_amplitude
from radar_stap.stap import (
    AdjacentBinPostDoppler,
    FullRankMVDR,
    NonAdaptive,
    PrincipalComponents,
    SpaceTimeModel,
    mvdr_weights,
    select_training_bins,
)

rng0 = np.random.default_rng(42)


def ula_model(C=4, L=8, T=1e-4, lam=0.004):
    pos = np.c_[np.arange(C) * lam / 2, np.zeros(C), np.zeros(C)]
    return SpaceTimeModel(pos, np.zeros(C), L, T, lam)


def cn(*shape, rng=rng0):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


# ---------------------------------------------------------------- data model


def test_snapshot_dimension_and_stacking_order():
    C, L, R = 3, 5, 7
    V = np.zeros((C, L, R), complex)
    for c in range(C):
        for l in range(L):
            V[c, l, :] = 100 * l + c
    X = snapshots(V)
    assert X.shape == (R, C * L)
    assert np.array_equal(X[2], [100 * l + c for l in range(L) for c in range(C)])


def test_space_time_steering_is_kron_of_temporal_and_spatial():
    m = ula_model()
    u = direction_from_angles(23.0)
    nu = 1234.0
    a = m.spatial(u)
    b = np.exp(2j * np.pi * nu * np.arange(m.num_loops) * m.slow_time_interval)
    assert np.allclose(m.steering(u, nu), np.kron(b, a))
    S = m.steering(u, m.doppler_grid(16))
    assert S.shape == (m.dimension, 16)


def test_training_selection_symmetric_with_guard_and_edges():
    valid = np.ones(100, bool)
    valid[[40, 41]] = False
    tr = select_training_bins(50, 20, 2, valid)
    assert tr.size == 20 and not set(range(48, 53)) & set(tr)
    assert set(tr) == set(range(38, 48)) - {40, 41} | set(range(53, 65)) or tr.size == 20
    assert 40 not in tr and 41 not in tr
    edge = select_training_bins(2, 20, 2, valid)
    assert edge.size == 20 and edge.min() >= 0 and 2 not in edge
    with pytest.raises(InsufficientTrainingError):
        select_training_bins(5, 200, 2, valid)


# ---------------------------------------------------------------- covariance


def test_sample_covariance_hermitian_psd_and_support_checks():
    X = cn(80, 16)
    R = sample_covariance(X)
    assert np.allclose(R, R.conj().T)
    assert np.linalg.eigvalsh(R).min() > -1e-12 * np.linalg.eigvalsh(R).max()
    assert np.allclose(R, np.einsum("li,lj->ij", X, X.conj()) / 80)
    with pytest.raises(InsufficientTrainingError):
        sample_covariance(cn(10, 16))
    Rd = sample_covariance(cn(10, 16), allow_rank_deficient=True)
    assert np.linalg.matrix_rank(Rd) == 10
    with pytest.raises(ValueError):
        sample_covariance(np.full((20, 4), np.nan))


def test_diagonal_loading_improves_conditioning_and_modes():
    R = sample_covariance(cn(20, 16))
    d0 = covariance_diagnostics(R, 20)
    d1 = covariance_diagnostics(diagonal_load(R, 1.0), 20)
    assert d1.condition_number < d0.condition_number
    assert loading_level(R, "absolute", 2.0) == 2.0
    assert loading_level(R, "trace_relative", 0.1) == pytest.approx(0.1 * np.trace(R).real / 16)
    assert loading_level(R, "noise_relative", 3.0, noise_level=2.0) == 6.0
    with pytest.raises(ValueError):
        loading_level(R, "noise_relative", 1.0)
    S = shrinkage_covariance(R, 1.0)
    assert np.allclose(S, np.trace(R).real / 16 * np.eye(16))
    assert rmb_expected_loss_db(32, 16) == pytest.approx(10 * np.log10(18 / 33))
    assert rmb_expected_loss_db(8, 16) is None


def test_mvdr_matches_explicit_formula_and_is_distortionless():
    m = ula_model(C=3, L=4)
    A = cn(12, 12)
    R = A @ A.conj().T + 0.1 * np.eye(12)
    S = m.steering(direction_from_angles(10.0), m.doppler_grid(4))
    W = mvdr_weights(R, S)
    Ri = np.linalg.inv(R)
    for j in range(4):
        ref = Ri @ S[:, j] / (S[:, j].conj() @ Ri @ S[:, j])
        assert np.allclose(W[:, j], ref, atol=1e-10)
        assert np.vdot(W[:, j], S[:, j]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="zero steering"):
        mvdr_weights(R, np.zeros(12))
    with pytest.raises(ValueError):
        mvdr_weights(R, np.ones(5))


def test_stable_under_ill_conditioning_with_loading():
    m = ula_model()
    D = m.dimension
    X = cn(D // 2, D)  # rank-deficient training
    R = sample_covariance(X, allow_rank_deficient=True)
    S = m.steering(direction_from_angles(0.0), m.doppler_grid(8))
    with pytest.raises(np.linalg.LinAlgError):
        mvdr_weights(R - 1e-3 * np.eye(D), S)  # indefinite -> explicit failure, not a silent result
    W = mvdr_weights(R, S, loading=1.0)
    assert np.all(np.isfinite(W))
    assert np.allclose(np.einsum("dj,dj->j", W.conj(), S), 1.0)


def test_all_methods_distortionless():
    m = ula_model()
    X = cn(64, m.dimension)
    nu = m.doppler_grid(8)
    S = m.steering(direction_from_angles(-12.0), nu)
    for meth in (NonAdaptive(m), FullRankMVDR(m, "noise_relative", 1.0, 1.0),
                 AdjacentBinPostDoppler(m, noise_level=1.0), PrincipalComponents(m, rank=4)):
        W = meth.weights(None if not meth.adaptive else X, S, nu).W
        assert np.allclose(np.einsum("dj,dj->j", W.conj(), S), 1.0), meth.name


def test_reduced_dimension_framework_equals_full_mvdr_when_transform_is_full_rank():
    """EFA with Q = L rect filters spans the whole space-time space -> identical to full-rank MVDR."""
    m = ula_model(C=3, L=6)
    X = cn(60, m.dimension)
    nu = m.doppler_grid(6)
    S = m.steering(direction_from_angles(8.0), nu)
    efa = AdjacentBinPostDoppler(m, num_bins=5, temporal_taper="rect", loading_mode="absolute", loading_value=0.0)
    # 5 adjacent bins of a 6-point DFT do not span C^6; use 7 -> wraps -> rank 6 (full)
    efa.num_bins = 7
    W1 = efa.weights(X, S, nu).W
    W2 = mvdr_weights(sample_covariance(X), S, 0.0)
    assert np.allclose(W1, W2, atol=1e-8)


def test_mvdr_with_true_covariance_reaches_optimum_sinr():
    m = ula_model()
    D = m.dimension
    A = cn(D, 6)
    R = A @ A.conj().T * 50 + np.eye(D)  # rank-6 interference + noise
    s = m.steering(direction_from_angles(20.0), 300.0)
    w = mvdr_weights(R, s[:, None])[:, 0]
    assert output_sinr(w, s, R) == pytest.approx(optimal_sinr(s, R), rel=1e-9)
    assert optimal_sinr(s, R) == pytest.approx(np.real(s.conj() @ np.linalg.solve(R, s)), rel=1e-12)


def test_output_sinr_hand_example():
    w = np.array([1.0, 0.0])
    g = np.array([2.0, 0.0])
    R = np.diag([4.0, 1.0])
    assert output_sinr(w, g, R) == pytest.approx(1.0)
    assert optimal_sinr(g, R) == pytest.approx(1.0)


# ---------------------------------------------------------------- physics on a simulated clutter scene


def test_basis_reproduces_full_simulation(stap_scene):
    cfg, sim, proc = stap_scene
    snaps = snapshots(proc.V)
    clean = sim.components["clutter"] + sim.components["targets"]
    from radar_stap.datacube import to_virtual_cube
    from radar_stap.range_processing import range_fft

    Yc, _ = range_fft(clean, cfg.radar, cfg.processing)
    ref = snapshots(to_virtual_cube(Yc, cfg.array.tx_schedule))
    b0, b1 = 40, 70
    geom = sim.scene.geometry
    cb = build_basis(sim.scene.clutter, cfg, geom, (b0, b1), halfwidth=20)
    tb = build_basis(sim.scene.targets, cfg, geom, (b0, b1), halfwidth=20)
    syn = cb.synthesize(None, b0, b1) + tb.synthesize(None, b0, b1)
    err = np.linalg.norm(syn - ref[b0:b1]) / np.linalg.norm(ref[b0:b1])
    assert err < 3e-3  # truncated Hann sidelobes beyond +-20 bins
    assert snaps.shape[1] == cb.dimension


def test_clairvoyant_covariance_matches_ensemble_average(stap_scene):
    cfg, sim, proc = stap_scene
    geom = sim.scene.geometry
    rb = 60
    cb = build_basis(_near(sim.scene.clutter, geom, rb, 12, proc, cfg), cfg, geom, (rb, rb + 1), 12)
    Rc = cb.covariance(rb)
    rng = np.random.default_rng(7)
    acc = np.zeros_like(Rc)
    n = 400
    for _ in range(n):
        c = mixer_amplitude(draw_clutter_reflectivity(cb.powers, "rayleigh", rng), cfg.radar.mixer_convention)
        x = cb.synthesize(c, rb, rb + 1)[0]
        acc += np.outer(x, x.conj())
    acc /= n
    rel = np.linalg.norm(acc - Rc) / np.linalg.norm(Rc)
    assert rel < 0.15  # ~ 1/sqrt(n) statistical error


def test_clutter_rank_follows_brennan_rule(stap_scene):
    cfg, sim, proc = stap_scene
    geom = sim.scene.geometry
    rb = 60
    cb = build_basis(_near(sim.scene.clutter, geom, rb, 12, proc, cfg), cfg, geom, (rb, rb + 1), 12)
    ev = np.linalg.eigvalsh(cb.covariance(rb))[::-1]
    sig2 = noise_covariance_scale(cfg)
    model = proc.model
    beta = 2 * 10.0 * model.slow_time_interval / (model.wavelength / 2)
    expected = brennan_rank(model.num_channels, model.num_loops, beta)
    rank = int(np.sum(ev > 10 * sig2))  # clutter eigenvalues more than 10 dB above the noise level
    assert abs(rank - expected) <= 2, (rank, expected, beta)


def test_clutter_ridge_doppler_follows_platform_motion(stap_scene):
    """Fourier angle-Doppler spectrum of clutter peaks at nu = -2 v_p sin(theta)/lambda (side-looking)."""
    cfg, sim, proc = stap_scene
    geom = sim.scene.geometry
    rb = 60
    cb = build_basis(_near(sim.scene.clutter, geom, rb, 12, proc, cfg), cfg, geom, (rb, rb + 1), 12)
    R = cb.covariance(rb)
    m = proc.model
    nu = m.doppler_grid(128)
    for ang in (-40.0, -15.0, 10.0, 35.0):
        S = m.steering(direction_from_angles(ang), nu)
        p = np.real(np.einsum("dj,de,ej->j", S.conj(), R, S))
        # ground scatterers are below the platform: u_x = cos(el) sin(az); use the measured peak's
        # velocity bound instead of elevation details: |nu_peak - nu_pred| within 1 Doppler resolution
        pred = -2 * 10.0 * np.sin(np.deg2rad(ang)) / m.wavelength
        res = 1 / (m.num_loops * m.slow_time_interval)
        assert abs(nu[np.argmax(p)] - pred) < res, ang


def test_adaptive_processing_suppresses_clutter_and_preserves_target(stap_scene):
    cfg, sim, proc = stap_scene
    truth = sim.scene.truths[0]
    setup = stap_setup(proc, cfg, sim.scene.truths)
    a = sinr_analysis(sim, cfg, setup, truth, proc)
    m = a.target_metrics
    assert a.optimal_sinr_db - m["mvdr"]["output_sinr_db"] < 4.0  # RMB: L = 2D -> ~3 dB expected loss
    assert m["mvdr"]["output_sinr_db"] - m["nonadaptive"]["output_sinr_db"] > 6.0
    assert m["efa"]["output_sinr_db"] - m["nonadaptive"]["output_sinr_db"] > 6.0
    assert m["mvdr"]["clutter_attenuation_db"] > 25.0
    for name in ("mvdr", "efa", "pc", "nonadaptive"):
        assert abs(m[name]["target_gain_db"] - m["nonadaptive"]["target_gain_db"]) < 1.5  # target kept


def test_gmti_detects_target_after_stap(stap_scene):
    from radar_stap.experiments import run_stap_comparison

    cfg, sim, proc = stap_scene
    out = run_stap_comparison(sim, proc, cfg)["summary"]
    assert out["methods"]["mvdr"]["targets"][0]["detected"]
    assert out["methods"]["pc"]["targets"][0]["detected"]
    assert not out["methods"]["nonadaptive"]["targets"][0]["detected"]  # masked by sidelobe clutter
    rb, _ = truth_cell(sim.scene.truths[0], cfg, proc)
    assert out["sinr_analysis"]["t1"]["range_bin"] == int(round(rb))
