"""The batched (GPU-ready) STAP path must reproduce the per-bin reference implementation."""

import numpy as np
import pytest

from radar_stap.angle_processing import direction_from_angles
from radar_stap.backend import BackendUnavailable
from radar_stap.batched import resolve_backend
from radar_stap.stap import AdjacentBinPostDoppler, FullRankMVDR, PrincipalComponents, SpaceTimeModel, run_gmti

LAM = 3e8 / 77e9


def _scene(C=4, L=8, bins=120, K=2, seed=0):
    m = SpaceTimeModel(np.c_[np.arange(C) * LAM / 2, np.zeros(C), np.zeros(C)], np.zeros(C), L, 70e-6, LAM)
    rng = np.random.default_rng(seed)
    us = np.linspace(-0.9, 0.9, 60)
    A = np.stack([m.steering(np.array([u, np.sqrt(1 - u * u), 0]), -2 * 8.0 * u / LAM) for u in us], 1)
    def cpi():
        g = (rng.standard_normal((bins, us.size)) + 1j * rng.standard_normal((bins, us.size))) * 3
        n = (rng.standard_normal((bins, m.dimension)) + 1j * rng.standard_normal((bins, m.dimension))) / np.sqrt(2)
        return g @ A.T + n
    pool = np.stack([cpi() for _ in range(K)])
    return m, pool[0], pool


@pytest.mark.parametrize("fb", [False, True])
@pytest.mark.parametrize("name", ["mvdr", "pc", "efa"])
def test_numpy_batched_matches_reference(name, fb):
    m, snaps, pool = _scene()
    meth = {"mvdr": FullRankMVDR(m, "noise_eig_relative", 1.0, None, fb),
            "pc": PrincipalComponents(m, None, forward_backward=fb),
            "efa": AdjacentBinPostDoppler(m, 3, "hann", "noise_eig_relative", 1.0, forward_backward=fb)}[name]
    dirs = direction_from_angles(np.array([-20.0, 0.0, 25.0]))
    nu = m.doppler_grid(16)
    valid = np.ones(snaps.shape[0], bool)
    tests = np.arange(40, 80)
    ref = run_gmti(snaps, m, meth, dirs, nu, tests, 64, 2, valid, measure_memory=False, training_snaps=pool)
    bat = run_gmti(snaps, m, meth, dirs, nu, tests, 64, 2, valid, measure_memory=False, training_snaps=pool,
                   backend=np, precision="double", batch_bins=7)
    err = np.max(np.abs(bat.output - ref.output)) / np.max(np.abs(ref.output))
    assert err < 1e-8, err
    single = run_gmti(snaps, m, meth, dirs, nu, tests, 64, 2, valid, measure_memory=False, training_snaps=pool,
                      backend=np, precision="single", batch_bins=16)
    err32 = np.max(np.abs(single.output - ref.output)) / np.max(np.abs(ref.output))
    # float32: MVDR/EFA solve well-loaded systems (~1e-4); PC's output is the residual after
    # subtracting a strong clutter subspace, so eigenvector rounding shows up at ~-45 dB.
    assert err32 < (1e-2 if name == "pc" else 2e-3), err32


def test_forward_backward_mvdr_accepts_half_support():
    """D/2 <= N < D rows: singular raw covariance, full rank after forward-backward averaging."""
    from radar_stap.covariance import InsufficientTrainingError

    m, snaps, pool = _scene()  # D = 32, K = 2 CPIs
    dirs, nu = direction_from_angles(np.array([0.0])), m.doppler_grid(16)
    valid, tests = np.ones(snaps.shape[0], bool), np.arange(40, 80)
    fb = FullRankMVDR(m, "noise_eig_relative", 1.0, None, True)
    ref = run_gmti(snaps, m, fb, dirs, nu, tests, 12, 2, valid, measure_memory=False, training_snaps=pool)  # 24 rows
    bat = run_gmti(snaps, m, fb, dirs, nu, tests, 12, 2, valid, measure_memory=False, training_snaps=pool,
                   backend=np, precision="double")
    assert np.all(np.isfinite(ref.output))
    assert np.max(np.abs(bat.output - ref.output)) / np.max(np.abs(ref.output)) < 1e-8
    with pytest.raises(InsufficientTrainingError):  # without FB the same support is singular
        run_gmti(snaps, m, FullRankMVDR(m, "noise_eig_relative", 1.0, None, False), dirs, nu, tests, 12, 2, valid,
                 measure_memory=False, training_snaps=pool)


def test_resolve_backend():
    assert resolve_backend("numpy") == (None, "numpy")
    assert resolve_backend("numpy_batched")[1] == "numpy_batched"
    xp, used = resolve_backend("auto")
    assert used in ("numpy", "cupy")
    with pytest.raises(ValueError):
        resolve_backend("tpu")
    try:
        import cupy  # noqa: F401
        cupy.cuda.runtime.getDeviceCount()
    except Exception:
        with pytest.raises(BackendUnavailable):
            resolve_backend("cupy")


def test_cupy_matches_reference_if_available():
    try:
        xp, _ = resolve_backend("cupy")
    except BackendUnavailable:
        pytest.skip("CuPy / CUDA not available")
    m, snaps, pool = _scene()
    meth = FullRankMVDR(m, "noise_eig_relative", 1.0, None, True)
    dirs, nu = direction_from_angles(np.array([0.0, 20.0])), m.doppler_grid(16)
    valid, tests = np.ones(snaps.shape[0], bool), np.arange(40, 80)
    ref = run_gmti(snaps, m, meth, dirs, nu, tests, 64, 2, valid, measure_memory=False, training_snaps=pool)
    gpu = run_gmti(snaps, m, meth, dirs, nu, tests, 64, 2, valid, measure_memory=False, training_snaps=pool,
                   backend=xp, precision="single")
    assert np.max(np.abs(gpu.output - ref.output)) / np.max(np.abs(ref.output)) < 2e-3
