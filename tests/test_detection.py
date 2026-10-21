"""CA-CFAR threshold model, empirical P_FA, association and interval metrics."""

import numpy as np
import pytest

from radar_stap.detection import ca_cfar, cfar_alpha, cluster_detections
from radar_stap.metrics import associate, wilson_interval


def test_cfar_alpha_closed_form():
    # N training cells, Pfa: alpha = N (Pfa^(-1/N) - 1); limit N->inf is -ln(Pfa)
    assert cfar_alpha(16, 1e-4) == pytest.approx(16 * (1e-4 ** (-1 / 16) - 1))
    assert cfar_alpha(100000, 1e-3) == pytest.approx(-np.log(1e-3), rel=1e-3)
    with pytest.raises(ValueError):
        cfar_alpha(10, 1.5)


@pytest.mark.parametrize("pfa", [1e-2, 1e-3])
def test_empirical_pfa_matches_design_for_exponential_noise(pfa):
    rng = np.random.default_rng(3)
    trials, hits, cells = 30, 0, 0
    for _ in range(trials):
        P = rng.exponential(5.0, (256, 64))
        cf = ca_cfar(P, [2, 1], [8, 4], pfa, ["shrink", "wrap"])
        hits += int(cf.detections.sum())
        cells += P.size
    lo, hi = wilson_interval(hits, cells, z=3.0)
    assert lo <= pfa <= hi, (hits / cells, pfa)


def test_cfar_detects_strong_target_and_handles_edges():
    rng = np.random.default_rng(0)
    P = rng.exponential(1.0, (100, 32))
    P[50, 10] = 300.0
    cf = ca_cfar(P, [2, 1], [6, 3], 1e-6, ["shrink", "wrap"])
    dets = cluster_detections(cf, P, wrap_axes=(1,))
    assert [d.indices for d in dets] == [(50, 10)]
    full = (2 * 8 + 1) * (2 * 4 + 1) - (2 * 2 + 1) * (2 * 1 + 1)
    assert cf.num_training[50, 10] == full
    assert cf.num_training[0, 10] < full  # shrink edge in range
    assert cf.num_training[50, 0] == full  # wrapped in Doppler
    assert cf.alpha[0, 10] > cf.alpha[50, 10]  # fewer cells -> larger multiplier


def test_cfar_exclusion_mask_and_target_masking():
    rng = np.random.default_rng(1)
    P = rng.exponential(1.0, (60, 1))
    P[30, 0] = 200.0
    P[34, 0] = 200.0  # interfering target inside the training window
    cf = ca_cfar(P, [1, 0], [6, 0], 1e-4)
    masked_thr = cf.threshold[30, 0]
    ex = np.zeros_like(P, bool)
    ex[34, 0] = True
    cf2 = ca_cfar(P, [1, 0], [6, 0], 1e-4, exclude=ex)
    assert cf2.threshold[30, 0] < masked_thr
    assert not cf2.detections[34, 0]


def test_local_max_grouping_keeps_weak_target_next_to_large_region():
    P = np.ones((40, 20))
    P[10:30, 0:6] = 1e4  # large "clutter" block
    P[20, 8] = 500.0  # weak target near the block edge
    P[20, 7] = 300.0
    cf = ca_cfar(P, [1, 0], [5, 0], 1e-3, ["shrink", "wrap"])
    loc = [d.indices for d in cluster_detections(cf, P, wrap_axes=(1,), method="local_max")]
    assert (20, 8) in loc


def test_association_rules():
    truths = [(10.0, 5.0), (40.0, 30.0)]
    dets = [(10.5, 5.4), (11.0, 5.0), (40.2, 1.0), (80.0, 2.0)]
    a = associate(truths, dets, (2, 2), (None, 32))
    assert (0, 0) in a.matches  # closest wins, one-to-one
    # circular Doppler distance between 30 and 1 (period 32) is 3 > gate 2 -> no match
    assert a.missed == [1]
    assert sorted(a.false_alarms) == [1, 2, 3]
    b = associate(truths, dets, (2, 3), (None, 32))
    assert (1, 2) in b.matches


def test_association_circular_axis():
    a = associate([(40.0, 31.0)], [(40.0, 0.5)], (1, 2), (None, 32))
    assert a.matches == [(0, 0)]
    b = associate([(40.0, 31.0)], [(40.0, 0.5)], (1, 2), (None, None))
    assert b.matches == [] and b.missed == [0]


def test_wilson_interval():
    lo, hi = wilson_interval(0, 50)
    assert lo == 0.0 and 0.05 < hi < 0.1
    lo, hi = wilson_interval(25, 50)
    assert lo < 0.5 < hi
