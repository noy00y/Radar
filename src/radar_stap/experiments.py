"""End-to-end orchestration: conventional processing, detection, STAP comparison, Monte Carlo.

These functions are used by ``scripts/simulate_scene.py``, the notebook and the
integration tests, so the pipeline exists in exactly one place.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .angle_processing import direction_from_angles, estimate_angle
from .config import SimulationConfig, StapConfig, sim_derived, wavelength_for
from .covariance import brennan_rank, covariance_diagnostics, sample_covariance
from .datacube import snapshots, to_virtual_cube
from .detection import Detection, ca_cfar, cluster_detections, refine_to_local_power_peak
from .doppler_processing import DopplerAxis, doppler_fft, range_doppler_map, tdm_phase_compensation
from .metrics import (
    associate,
    clutter_attenuation,
    db,
    input_sinr,
    optimal_sinr,
    output_sinr,
    target_gain,
    wilson_interval,
)
from .range_processing import RangeAxis, expected_range_bin, parabolic_peak, range_fft, range_mask
from .response_basis import ResponseBasis, build_basis, noise_covariance_scale
from .simulator import (
    Scene,
    ScattererSet,
    SimulationResult,
    TargetTruth,
    draw_clutter_reflectivity,
    mixer_amplitude,
)
from .stap import GmtiResult, SpaceTimeModel, build_method, run_gmti, select_training_bins
from .waveforms import complex_noise

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conventional processing
# ---------------------------------------------------------------------------


@dataclass
class ProcessedCPI:
    Y: np.ndarray  # range-compressed, (M, N, bins)
    V: np.ndarray  # virtual cube, (C, L, bins)
    Z: np.ndarray  # range-Doppler per channel, (C, nd, bins)
    rd_map: np.ndarray  # (bins, nd) noncoherent power
    range_axis: RangeAxis
    doppler_axis: DopplerAxis
    range_valid: np.ndarray  # bool (bins,)
    model: SpaceTimeModel


def process_cpi(data: np.ndarray, cfg: SimulationConfig) -> ProcessedCPI:
    """Range FFT -> virtual cube -> Doppler FFT -> noncoherent range-Doppler map."""
    d = sim_derived(cfg)
    Y, raxis = range_fft(data, cfg.radar, cfg.processing)
    V = to_virtual_cube(Y, cfg.array.tx_schedule)
    Z, daxis = doppler_fft(V, d.slow_time_interval_s, d.wavelength_m, cfg.processing.doppler_window, cfg.processing.doppler_nfft)
    rd = range_doppler_map(Z)
    valid = range_mask(raxis, cfg.processing.range_min_m, cfg.processing.range_max_m)
    return ProcessedCPI(Y, V, Z, rd, raxis, daxis, valid, SpaceTimeModel.from_config(cfg))


def truth_cell(truth: TargetTruth, cfg: SimulationConfig, proc: ProcessedCPI) -> tuple[float, float]:
    """Expected (fractional range bin incl. coupling, fractional fftshifted Doppler bin)."""
    rb = expected_range_bin(truth.range_m, truth.radial_velocity_mps, cfg.radar, proc.range_axis)
    return rb, proc.doppler_axis.bin_of_velocity(truth.radial_velocity_mps)


def _velocity_from_bin(frac: float, axis: DopplerAxis) -> float:
    k = frac - axis.nfft // 2
    k = (k + axis.nfft / 2) % axis.nfft - axis.nfft / 2
    return float(k * axis.bin_spacing_hz * axis.wavelength_m / 2)


def detect_range_doppler(proc: ProcessedCPI, cfg: SimulationConfig) -> tuple[list[Detection], object]:
    """2-D CA-CFAR on the noncoherent range-Doppler map + parameter estimation per detection."""
    det_cfg = cfg.detection
    exclude = np.repeat(~proc.range_valid[:, None], proc.rd_map.shape[1], axis=1)
    cf = ca_cfar(
        proc.rd_map,
        det_cfg.guard_cells,
        det_cfg.training_cells,
        det_cfg.pfa,
        [det_cfg.range_edge, det_cfg.doppler_edge],
        exclude=exclude,
    )
    dets = cluster_detections(cf, proc.rd_map, wrap_axes=(1,) if det_cfg.doppler_edge == "wrap" else ())
    lam = wavelength_for(cfg.radar)
    for dt in dets:
        r, k = refine_to_local_power_peak(proc.rd_map, dt.indices)
        rf = parabolic_peak(proc.rd_map[:, k], r) if 0 < r < proc.rd_map.shape[0] - 1 else float(r)
        kf = parabolic_peak(proc.rd_map[r, :], k)
        dt.range_bin_frac, dt.doppler_bin_frac = rf, kf
        apparent = float(np.interp(rf, np.arange(proc.range_axis.num_bins), proc.range_axis.range_m))
        dt.velocity_mps = _velocity_from_bin(kf, proc.doppler_axis)
        # remove range-Doppler coupling (f_b contains 2 v_r / lambda): R = R_apparent - v_r f_c / S
        dt.range_m = apparent - dt.velocity_mps * (2 / lam) * 299_792_458.0 / (2 * cfg.radar.chirp_slope_hz_per_s)
        snap = proc.Z[:, k, r] * tdm_phase_compensation(float(proc.doppler_axis.freq_hz[k]), proc.model.time_offsets)
        dt.angle_deg, _, _ = estimate_angle(snap, proc.model.positions, proc.model.wavelength)
    return dets, cf


def evaluate_rd_detections(
    dets: list[Detection], truths: list[TargetTruth], proc: ProcessedCPI, cfg: SimulationConfig
) -> dict:
    """Associate RD detections with ground truth (targets and reflectors) and report errors."""
    cells_t = [truth_cell(t, cfg, proc) for t in truths]
    cells_d = [(d.range_bin_frac, d.doppler_bin_frac) for d in dets]
    g = cfg.detection
    a = associate(cells_t, cells_d, (g.range_gate_bins, g.doppler_gate_bins), (None, proc.doppler_axis.nfft))
    rows = []
    for ti, t in enumerate(truths):
        match = next((di for tj, di in a.matches if tj == ti), None)
        row = {"name": t.name, "kind": t.kind, "detected": match is not None, "truth": t.as_dict()}
        if match is not None:
            d = dets[match]
            row["estimate"] = {"range_m": d.range_m, "radial_velocity_mps": d.velocity_mps, "angle_deg": d.angle_deg}
            row["errors"] = {
                "range_m": d.range_m - t.range_m,
                "radial_velocity_mps": d.velocity_mps - t.radial_velocity_mps,
                "angle_deg": d.angle_deg - t.angle_deg,
            }
            row["snr_estimate_db"] = d.snr_estimate_db
        rows.append(row)
    return {
        "association_rule": (
            f"greedy nearest neighbour; |range bin - expected bin (incl. coupling)| <= {g.range_gate_bins}, "
            f"circular |Doppler bin - expected| <= {g.doppler_gate_bins}"
        ),
        "num_detections": len(dets),
        "num_unassociated_detections": len(a.false_alarms),
        "targets": rows,
        "note": (
            "Unassociated detections on a moving platform are dominated by the clutter ridge: the "
            "noncoherent RD map has no angle discrimination, which is what STAP addresses."
        ),
    }


# ---------------------------------------------------------------------------
# STAP experiment
# ---------------------------------------------------------------------------


@dataclass
class StapSetup:
    model: SpaceTimeModel
    snaps: np.ndarray  # (bins, D)
    direction: np.ndarray
    nu_grid: np.ndarray
    test_bins: np.ndarray
    valid_training: np.ndarray
    excluded_bins: list[int]
    in_beam_truths: list[TargetTruth]
    truth_cells: list[tuple[float, float]]  # (range bin, hypothesis index) per in-beam truth


def _grid_index(nu: float, nu_grid: np.ndarray, T_slow: float) -> float:
    J = nu_grid.size
    frac = nu * J * T_slow + J // 2
    return float(np.mod(frac, J))


def stap_setup(proc: ProcessedCPI, cfg: SimulationConfig, truths: list[TargetTruth], snaps: np.ndarray | None = None) -> StapSetup:
    s = cfg.stap
    model = proc.model
    snaps = snapshots(proc.V) if snaps is None else snaps
    if np.any(np.abs(model.positions[:, 2]) > 1e-9) or np.any(np.abs(model.positions[:, 1]) > 1e-9):
        log.warning("virtual array is not purely along x: look direction assumes elevation 0")
    direction = direction_from_angles(s.look_angle_deg)
    nu_grid = model.doppler_grid(s.num_doppler_hypotheses)
    nb = snaps.shape[0]
    valid = proc.range_valid.copy()
    excluded = []
    # truths in the look direction (within the association angle gate)
    in_beam = [t for t in truths if t.kind == "target" and abs(t.angle_deg - s.look_angle_deg) <= cfg.detection.angle_gate_deg]
    if s.exclude_known_targets:
        for t in truths:  # oracle: remove all known discrete returns from training
            rb = int(round(expected_range_bin(t.range_m, t.radial_velocity_mps, cfg.radar, proc.range_axis)))
            for b in range(rb - s.exclusion_halfwidth_bins, rb + s.exclusion_halfwidth_bins + 1):
                if 0 <= b < nb:
                    valid[b] = False
                    excluded.append(b)
    rmin = s.test_range_min_m if s.test_range_min_m is not None else cfg.processing.range_min_m
    rmax = s.test_range_max_m if s.test_range_max_m is not None else (cfg.processing.range_max_m or np.inf)
    cand = np.where(range_mask(proc.range_axis, rmin, rmax) & proc.range_valid)[0]
    test = []
    for b in cand:
        try:
            select_training_bins(int(b), s.num_training, s.num_guard, valid)
            test.append(int(b))
        except ValueError:
            pass
    cells = []
    for t in in_beam:
        rb = expected_range_bin(t.range_m, t.radial_velocity_mps, cfg.radar, proc.range_axis)
        nu = 2 * t.radial_velocity_mps / model.wavelength
        cells.append((rb, _grid_index(nu, nu_grid, model.slow_time_interval)))
    return StapSetup(model, snaps, direction, nu_grid, np.asarray(test, int), valid, sorted(set(excluded)), in_beam, cells)


@dataclass
class GmtiDetections:
    result: GmtiResult
    cfar_mask: np.ndarray  # (n_test, J)
    detections: list[Detection]


def gmti_detect(res: GmtiResult, stap_cfg: StapConfig) -> GmtiDetections:
    """1-D CA-CFAR along range for every Doppler hypothesis of the STAP output."""
    s = stap_cfg
    P = res.power
    cf = ca_cfar(P, [s.cfar_guard, 0], [s.cfar_training, 0], s.cfar_pfa, ["shrink", "wrap"])
    dets = cluster_detections(cf, P, wrap_axes=(1,))
    for d in dets:
        i, j = d.indices
        d.range_bin_frac = float(res.test_bins[i])
        d.doppler_bin_frac = float(j)
    return GmtiDetections(res, cf.detections, dets)


def evaluate_gmti(gd: GmtiDetections, setup: StapSetup, cfg: SimulationConfig, proc: ProcessedCPI) -> dict:
    cells_d = [(d.range_bin_frac, d.doppler_bin_frac) for d in gd.detections]
    J = setup.nu_grid.size
    a = associate(setup.truth_cells, cells_d, (cfg.detection.range_gate_bins, 1.5), (None, J))
    hits = []
    for ti, t in enumerate(setup.in_beam_truths):
        m = next((di for tj, di in a.matches if tj == ti), None)
        entry = {"name": t.name, "detected": m is not None}
        if m is not None:
            d = gd.detections[m]
            entry["range_m_estimate"] = float(np.interp(d.range_bin_frac, np.arange(proc.range_axis.num_bins), proc.range_axis.range_m))
            entry["radial_velocity_estimate_mps"] = float(setup.model.velocity_of(setup.nu_grid[int(d.doppler_bin_frac)]))
            entry["output_snr_estimate_db"] = d.snr_estimate_db
        hits.append(entry)
    n_cells = gd.cfar_mask.size
    return {
        "targets": hits,
        "num_detections": len(gd.detections),
        "num_false_alarm_clusters": len(a.false_alarms),
        "num_detection_cells": int(gd.cfar_mask.sum()),
        "num_test_cells": int(n_cells),
    }


@dataclass
class SinrAnalysis:
    target: str
    range_bin: int
    nu_grid: np.ndarray
    sinr_loss_db: dict[str, np.ndarray]  # method -> (J,) SINR loss vs Doppler at look angle (ideal target)
    target_metrics: dict[str, dict]  # method -> metrics for the actual simulated target response
    clutter_eigs: np.ndarray
    sample_eigs: np.ndarray
    brennan_rank: int
    input_sinr_db: float
    optimal_sinr_db: float


def sinr_analysis(
    sim: SimulationResult,
    cfg: SimulationConfig,
    setup: StapSetup,
    truth: TargetTruth,
    proc: ProcessedCPI,
) -> SinrAnalysis:
    """Clairvoyant SINR evaluation at the target's range bin.

    * R_i: exact interference covariance from the scatterer model (clutter +
      reflectors) plus noise (see response_basis).
    * g_t: the simulated target's own range-compressed response.
    * Weights are computed from the *same* training snapshots used by the GMTI run.
    """
    rb = int(round(expected_range_bin(truth.range_m, truth.radial_velocity_mps, cfg.radar, proc.range_axis)))
    geom = sim.scene.geometry
    region = (rb, rb + 1)
    hw = cfg.monte_carlo.basis_halfwidth_bins
    cb = build_basis(_near(sim.scene.clutter, geom, rb, hw, proc, cfg), cfg, geom, region, hw)
    rbasis = build_basis(sim.scene.reflectors, cfg, geom, region, hw)
    D = setup.model.dimension
    sig2 = noise_covariance_scale(cfg)
    Rc = cb.covariance(rb) + rbasis.covariance(rb)
    Ri = Rc + sig2 * np.eye(D)
    ti = sim.scene.targets.names.index(truth.name)
    tb = build_basis(sim.scene.targets.subset([ti]), cfg, geom, region, hw)
    g_t = tb.coefficients[0] * tb.response_at(0, rb)
    S = setup.model.steering(setup.direction, setup.nu_grid)  # (D, J)
    tr = select_training_bins(rb, cfg.stap.num_training, cfg.stap.num_guard, setup.valid_training)
    Xtr = setup.snaps[tr]
    nu_t = 2 * truth.radial_velocity_mps / setup.model.wavelength
    s_t = setup.model.steering(setup.direction, nu_t)
    loss, tm = {}, {}
    for name in cfg.stap.methods:
        meth = build_method(name, setup.model, cfg.stap, noise_covariance_scale(cfg))
        W = meth.weights(Xtr, S, setup.nu_grid).W
        # SINR loss vs Doppler: ideal target s(nu_j) with unit power, normalised by noise-only bound D/sig2
        num = np.abs(np.einsum("dj,dj->j", W.conj(), S)) ** 2
        den = np.real(np.einsum("dj,de,ej->j", W.conj(), Ri, W))
        loss[name] = db(num / den / (D / sig2))
        # actual target: weights at the target's exact Doppler
        wt = meth.weights(Xtr, s_t[:, None], np.array([nu_t])).W[:, 0]
        tm[name] = {
            "output_sinr_db": float(db(output_sinr(wt, g_t, Ri))),
            "improvement_factor_db": float(db(output_sinr(wt, g_t, Ri) / input_sinr(g_t, Ri))),
            "clutter_attenuation_db": float(db(clutter_attenuation(wt, Rc))) if np.trace(Rc).real > 0 else None,
            "target_gain_db": float(db(target_gain(wt, g_t))),
        }
    Xc = Xtr
    samp = np.linalg.eigvalsh(sample_covariance(Xc))[::-1]
    beta = 2 * np.linalg.norm(geom.velocity) * setup.model.slow_time_interval / max(_element_spacing(setup.model), 1e-12)
    return SinrAnalysis(
        target=truth.name,
        range_bin=rb,
        nu_grid=setup.nu_grid,
        sinr_loss_db=loss,
        target_metrics=tm,
        clutter_eigs=np.linalg.eigvalsh(Ri)[::-1],
        sample_eigs=samp,
        brennan_rank=brennan_rank(setup.model.num_channels, setup.model.num_loops, beta),
        input_sinr_db=float(db(input_sinr(g_t, Ri))),
        optimal_sinr_db=float(db(optimal_sinr(g_t, Ri))),
    )


def _element_spacing(model: SpaceTimeModel) -> float:
    x = np.sort(np.unique(np.round(model.positions[:, 0], 12)))
    return float(np.min(np.diff(x))) if x.size > 1 else 0.0


def _near(scat: ScattererSet, geom, center_bin: int, halfwidth: int, proc: ProcessedCPI, cfg: SimulationConfig, b1: int | None = None) -> ScattererSet:
    """Pre-filter scatterers whose range bin lies within the stored halfwidth of a bin region."""
    if scat.size == 0:
        return scat
    b1 = center_bin + 1 if b1 is None else b1
    r, _ = geom.line_of_sight(scat.positions)
    bins = r / proc.range_axis.bin_spacing_m
    margin = halfwidth + 4
    keep = np.where((bins >= center_bin - margin) & (bins < b1 + margin))[0]
    return scat.subset(keep)


def run_stap_comparison(sim: SimulationResult, proc: ProcessedCPI, cfg: SimulationConfig) -> dict:
    """Run every configured STAP method on the same snapshots, training rules and hypotheses."""
    setup = stap_setup(proc, cfg, sim.scene.truths)
    if setup.test_bins.size == 0:
        raise ValueError("no STAP test bins with sufficient training support; check stap.num_training / range limits")
    out = {
        "processing_domain": "range-compressed virtual-channel snapshots, x[l*C+c] = V[c,l,r]",
        "dimension": setup.model.dimension,
        "num_channels": setup.model.num_channels,
        "num_slow_time": setup.model.num_loops,
        "num_training": cfg.stap.num_training,
        "training_to_dimension_ratio": cfg.stap.num_training / setup.model.dimension,
        "look_angle_deg": cfg.stap.look_angle_deg,
        "num_test_bins": int(setup.test_bins.size),
        "num_doppler_hypotheses": int(setup.nu_grid.size),
        "oracle_training_exclusion": cfg.stap.exclude_known_targets,
        "excluded_training_bins": setup.excluded_bins,
        "in_beam_targets": [t.name for t in setup.in_beam_truths],
        "methods": {},
    }
    gmti = {}
    for name in cfg.stap.methods:
        meth = build_method(name, setup.model, cfg.stap, noise_covariance_scale(cfg))
        res = run_gmti(
            setup.snaps, setup.model, meth, setup.direction, setup.nu_grid, setup.test_bins,
            cfg.stap.num_training, cfg.stap.num_guard, setup.valid_training,
        )
        gd = gmti_detect(res, cfg.stap)
        gmti[name] = gd
        ev = evaluate_gmti(gd, setup, cfg, proc)
        out["methods"][name] = {
            "weight_seconds": res.weight_seconds,
            "apply_seconds": res.apply_seconds,
            "peak_traced_memory_mb": res.peak_memory_bytes / 2**20,
            "complexity_complex_macs": res.complexity_macs,
            "diagnostics_sample": _jsonable(res.info[:3]),
            **ev,
        }
    if "mvdr" in gmti:
        ref = gmti["mvdr"].result.power
        for name, gd in gmti.items():
            p = gd.result.power
            c = np.corrcoef(db(ref).ravel(), db(p).ravel())[0, 1]
            out["methods"][name]["log_power_correlation_with_mvdr"] = float(c)
    analyses = []
    for t in setup.in_beam_truths:
        try:
            a = sinr_analysis(sim, cfg, setup, t, proc)
            analyses.append(a)
            out.setdefault("sinr_analysis", {})[t.name] = {
                "range_bin": a.range_bin,
                "input_sinr_db_per_element_pulse": a.input_sinr_db,
                "optimal_sinr_db": a.optimal_sinr_db,
                "methods": a.target_metrics,
                "brennan_clutter_rank": a.brennan_rank,
                "clairvoyant_effective_rank_10dB": int(np.sum(a.clutter_eigs > 10 * a.clutter_eigs[-1])),
            }
        except Exception as exc:  # report, never hide
            log.exception("SINR analysis failed for %s", t.name)
            out.setdefault("sinr_analysis", {})[t.name] = {"error": repr(exc)}
    return {"summary": out, "setup": setup, "gmti": gmti, "sinr": analyses}


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
# Monte Carlo P_D / P_FA
# ---------------------------------------------------------------------------


@dataclass
class MonteCarloResult:
    trials: int
    target: str
    target_snr_db: float
    methods: dict[str, dict] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"trials": self.trials, "target": self.target, "target_snr_db": self.target_snr_db, "methods": self.methods}


def monte_carlo(sim: SimulationResult, proc: ProcessedCPI, cfg: SimulationConfig, trials: int | None = None) -> MonteCarloResult:
    """Empirical P_D and P_FA of the GMTI chain for each STAP method.

    Fixed: scene geometry, look direction, Doppler hypotheses, training rule.
    Redrawn every trial: clutter reflectivities, reflector phases, target phase,
    receiver noise. Each trial is run with the target absent (H0) and present (H1)
    using the same interference realisation.

    * P_D: fraction of H1 trials with a CFAR detection within the association
      gate (range +-gate bins, +-1.5 Doppler hypotheses) of the target cell.
    * P_FA: fraction of tested (range, Doppler-hypothesis) cells declared in
      H0 trials (per-cell, comparable to the design value ``stap.cfar_pfa``).
    """
    mc = cfg.monte_carlo
    trials = trials or mc.num_trials
    setup0 = stap_setup(proc, cfg, sim.scene.truths)
    names = [t.name for t in setup0.in_beam_truths]
    tname = mc.target_name or (names[0] if names else None)
    if tname is None or tname not in names:
        raise ValueError(f"Monte Carlo target {tname!r} must be a target within the look-angle gate; in beam: {names}")
    truth = setup0.in_beam_truths[names.index(tname)]
    cell = setup0.truth_cells[names.index(tname)]
    rb = int(round(cell[0]))
    s = cfg.stap
    hw_test = mc.test_halfwidth_bins
    margin = s.cfar_guard + s.cfar_training
    out_lo, out_hi = rb - hw_test - margin, rb + hw_test + margin + 1  # bins needing STAP output
    need = s.num_training // 2 + s.num_guard + 2 * s.exclusion_halfwidth_bins + 2
    b0, b1 = max(out_lo - need, 0), min(out_hi + need, proc.range_axis.num_bins)
    geom = sim.scene.geometry
    hw = mc.basis_halfwidth_bins
    region = (b0, b1)
    cb = build_basis(_near(sim.scene.clutter, geom, b0, hw, proc, cfg, b1), cfg, geom, region, hw)
    rbz = build_basis(sim.scene.reflectors, cfg, geom, region, hw)
    others = [i for i, n in enumerate(sim.scene.targets.names) if n != tname]
    ob = build_basis(sim.scene.targets.subset(others), cfg, geom, region, hw) if others else None
    ti = sim.scene.targets.names.index(tname)
    tb = build_basis(sim.scene.targets.subset([ti]), cfg, geom, region, hw)
    snr = mc.target_snr_db if mc.target_snr_db is not None else truth.snr_db
    p_t = cfg.radar.noise_power * 10 ** (snr / 10)

    # local (region) setup: valid training / test bins in region coordinates
    valid = setup0.valid_training[b0:b1].copy()
    test_global = np.arange(max(out_lo, b0), min(out_hi, b1))
    test_global = np.array([b for b in test_global if proc.range_valid[b]])
    test_local = []
    for b in test_global:
        try:
            select_training_bins(int(b - b0), s.num_training, s.num_guard, valid)
            test_local.append(int(b - b0))
        except ValueError:
            pass
    test_local = np.asarray(test_local, int)
    if test_local.size == 0:
        raise ValueError("Monte Carlo region has no test bins with full training support")
    eval_mask = np.abs(test_local + b0 - rb) <= hw_test  # rows inside the evaluated test region
    J = setup0.nu_grid.size
    methods = {n: build_method(n, setup0.model, s, noise_covariance_scale(cfg)) for n in s.methods}
    counts = {n: {"hits": 0, "fa_cells_h0": 0, "cells_h0": 0} for n in methods}
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 7919]))
    M, N, K = len(cfg.array.rx_positions), cfg.radar.num_chirps, cfg.radar.num_adc_samples
    for _ in range(trials):
        cc = mixer_amplitude(draw_clutter_reflectivity(cb.powers, cfg.clutter.reflectivity, rng), cfg.radar.mixer_convention)
        interf = cb.synthesize(cc, b0, b1)
        if rbz.size:
            interf += rbz.synthesize(np.sqrt(rbz.powers) * np.exp(2j * np.pi * rng.random(rbz.size)), b0, b1)
        if ob is not None and ob.size:
            interf += ob.synthesize(np.sqrt(ob.powers) * np.exp(2j * np.pi * rng.random(ob.size)), b0, b1)
        noise = complex_noise(rng, (M, N, K), cfg.radar.noise_power, np.complex128)
        Yn, _ = range_fft(noise, cfg.radar, cfg.processing)
        interf += snapshots(to_virtual_cube(Yn, cfg.array.tx_schedule))[b0:b1]
        tgt = tb.synthesize(np.sqrt(p_t) * np.exp(2j * np.pi * rng.random(1)), b0, b1)
        for hyp, snaps in (("h0", interf), ("h1", interf + tgt)):
            for n, meth in methods.items():
                res = run_gmti(snaps, setup0.model, meth, setup0.direction, setup0.nu_grid, test_local,
                               s.num_training, s.num_guard, valid, measure_memory=False)
                cf = ca_cfar(res.power, [s.cfar_guard, 0], [s.cfar_training, 0], s.cfar_pfa, ["shrink", "wrap"])
                mask = cf.detections[eval_mask]
                rows = test_local[eval_mask] + b0
                if hyp == "h0":
                    counts[n]["fa_cells_h0"] += int(mask.sum())
                    counts[n]["cells_h0"] += int(mask.size)
                else:
                    rr, jj = np.nonzero(mask)
                    dr = np.abs(rows[rr] - cell[0])
                    dj = np.abs(jj - cell[1])
                    dj = np.minimum(dj, J - dj)
                    if np.any((dr <= cfg.detection.range_gate_bins) & (dj <= 1.5)):
                        counts[n]["hits"] += 1
    res = MonteCarloResult(trials, tname, snr)
    for n, c in counts.items():
        pd = c["hits"] / trials
        pfa = c["fa_cells_h0"] / max(c["cells_h0"], 1)
        res.methods[n] = {
            "pd": pd,
            "pd_95ci": wilson_interval(c["hits"], trials),
            "pfa": pfa,
            "pfa_95ci": wilson_interval(c["fa_cells_h0"], c["cells_h0"]),
            "pfa_design": s.cfar_pfa,
            "h0_cells_tested": c["cells_h0"],
            "h0_false_alarm_cells": c["fa_cells_h0"],
        }
    return res


ASSUMPTIONS = [
    "Point scatterers with constant velocity over the CPI; path length recomputed per chirp (range migration modelled), "
    "linear delay within a chirp.",
    "No intrinsic clutter motion, multipath, occlusion, IF filter/HPF response, ADC nonlinearity, phase noise or "
    "channel gain/phase errors (array is perfectly calibrated).",
    "Clutter reflectivities are independent complex Gaussian (Rayleigh) per scatterer; total clutter power set by cnr_db; "
    "scatterer power ~ R^-n times a two-way cos^p element pattern.",
    "Processor steering vectors use the far-field narrowband model at the sampled-band centre wavelength; the simulator "
    "uses exact geometry. Their agreement is tested.",
    "Look direction assumes elevation 0 for the steering vector (exact for a linear array along x).",
    "Training-cell exclusion of known targets is an oracle aid when stap.exclude_known_targets is true.",
    "CA-CFAR thresholds assume i.i.d. exponential cells; window correlation and noncoherent channel sums make the RD "
    "detector conservative; clutter edges/ridge make it non-homogeneous.",
]
