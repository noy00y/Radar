"""Evaluation metrics: association, estimation errors, SINR, P_D / P_FA, runtime.

SINR definitions (range-compressed space-time domain, one range bin)
--------------------------------------------------------------------
* target response g_t (D,): noise/clutter-free snapshot of the target at its
  range bin (simulated separately, so it includes every model effect).
* interference covariance R_i (D, D): clutter + discretes + noise at that bin
  (clairvoyant: built from the scatterer model, see
  :mod:`radar_stap.response_basis`; or a sample estimate).
* input SINR (per element per pulse) = (||g_t||^2 / D) / (tr(R_i) / D)
* output SINR of weight w = |w^H g_t|^2 / (w^H R_i w)
* optimum SINR = g_t^H R_i^{-1} g_t (clairvoyant MVDR with the true response)
* SINR loss = SINR_out / (||g_t||^2 / sigma_n^2_bin) relative to the
  noise-only matched-filter bound; improvement factor IF = SINR_out / SINR_in.
* clutter attenuation = (tr(R_c)/D) / (w^H R_c w) with ||w|| normalised to the
  distortionless constraint used (reported for interpretation only).
"""

from __future__ import annotations

import math
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def db(x) -> Any:
    return 10 * np.log10(np.maximum(np.asarray(x, float), 1e-300))


def output_sinr(w: np.ndarray, g_target: np.ndarray, R_interference: np.ndarray) -> float:
    num = abs(np.vdot(w, g_target)) ** 2
    den = float(np.real(np.vdot(w, R_interference @ w)))
    if den <= 0:
        raise ValueError("interference power w^H R w must be positive")
    return float(num / den)


def optimal_sinr(g_target: np.ndarray, R_interference: np.ndarray) -> float:
    cf = cho_factor(0.5 * (R_interference + R_interference.conj().T), lower=True)
    return float(np.real(np.vdot(g_target, cho_solve(cf, g_target))))


def input_sinr(g_target: np.ndarray, R_interference: np.ndarray) -> float:
    D = g_target.size
    return float((np.vdot(g_target, g_target).real / D) / (np.trace(R_interference).real / D))


def interference_power(w: np.ndarray, R: np.ndarray) -> float:
    return float(np.real(np.vdot(w, R @ w)))


def clutter_attenuation(w: np.ndarray, R_clutter: np.ndarray) -> float:
    """Average per-element clutter power in / clutter power at the filter output (w^H s = 1 scaling)."""
    D = w.size
    return float((np.trace(R_clutter).real / D) / max(interference_power(w, R_clutter), 1e-300))


def target_gain(w: np.ndarray, g_target: np.ndarray) -> float:
    """|w^H g_t|^2 / (||g_t||^2 / D): coherent target gain of the filter (D for an ideal matched filter
    with w^H g_t normalisation equal to 1 and ||w||^2 = 1/D)."""
    D = g_target.size
    return float(abs(np.vdot(w, g_target)) ** 2 / (np.vdot(g_target, g_target).real / D))


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if trials <= 0:
        return (0.0, 1.0)
    p = successes / trials
    den = 1 + z**2 / trials
    c = (p + z**2 / (2 * trials)) / den
    h = z * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / den
    return (max(0.0, c - h), min(1.0, c + h))


@dataclass
class Association:
    matches: list[tuple[int, int]]  # (truth index, detection index)
    missed: list[int]
    false_alarms: list[int]


def associate(
    truth_cells: list[tuple[float, ...]],
    detection_cells: list[tuple[float, ...]],
    gates: tuple[float, ...],
    circular: tuple[int | None, ...] | None = None,
) -> Association:
    """Greedy nearest-neighbour association inside per-axis gates.

    Rule: a detection matches a truth if |d_i - t_i| <= gate_i on every axis
    (axis i measured circularly with period circular[i] if given). Each truth
    and each detection is used at most once, closest pairs (normalised
    distance) first. Unmatched detections are false alarms; unmatched truths
    are misses.
    """
    circular = circular or tuple(None for _ in gates)
    pairs = []
    for ti, t in enumerate(truth_cells):
        for di, d in enumerate(detection_cells):
            ok, dist = True, 0.0
            for ax, (a, b, g) in enumerate(zip(t, d, gates)):
                diff = abs(a - b)
                per = circular[ax]
                if per:
                    diff = min(diff % per, per - diff % per)
                if diff > g:
                    ok = False
                    break
                dist += (diff / g) ** 2 if g > 0 else 0.0
            if ok:
                pairs.append((dist, ti, di))
    pairs.sort()
    used_t, used_d, matches = set(), set(), []
    for _, ti, di in pairs:
        if ti not in used_t and di not in used_d:
            used_t.add(ti)
            used_d.add(di)
            matches.append((ti, di))
    return Association(
        matches=matches,
        missed=[i for i in range(len(truth_cells)) if i not in used_t],
        false_alarms=[i for i in range(len(detection_cells)) if i not in used_d],
    )


def estimation_errors(truth: dict, estimate: dict, keys=("range_m", "radial_velocity_mps", "angle_deg")) -> dict:
    out = {}
    for k in keys:
        tv, ev = truth.get(k), estimate.get(k)
        if tv is not None and ev is not None:
            out[k + "_error"] = float(ev - tv)
    return out


@dataclass
class Measurement:
    seconds: float
    peak_bytes: int
    result: Any


def measure(fn: Callable[[], Any], track_memory: bool = True) -> Measurement:
    """Wall-clock time and peak traced (Python/NumPy) allocation of ``fn()``."""
    if track_memory:
        tracemalloc.start()
    t0 = time.perf_counter()
    try:
        res = fn()
    finally:
        dt = time.perf_counter() - t0
        peak = tracemalloc.get_traced_memory()[1] if track_memory else 0
        if track_memory:
            tracemalloc.stop()
    return Measurement(dt, int(peak), res)
