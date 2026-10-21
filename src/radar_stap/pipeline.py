"""Simulation workflow used by ``scripts/simulate_scene.py`` (and the integration test)."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from .config import SimulationConfig, config_to_dict, describe_derived, sim_derived
from .constants import RADIAL_VELOCITY_CONVENTION, STACKING_ORDER
from .covariance import sample_covariance
from .experiments import (
    ASSUMPTIONS,
    _jsonable,
    detect_range_doppler,
    evaluate_rd_detections,
    monte_carlo,
    process_cpi,
    run_stap_comparison,
)
from .range_processing import range_profile
from .response_basis import noise_covariance_scale
from .simulator import simulate
from .stap import select_training_bins
from .waveforms import fast_time_axis

log = logging.getLogger(__name__)


def json_dump(obj, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(obj), fh, indent=2, default=lambda o: _jsonable(o) if not isinstance(o, (str, int, float)) else o)


def run_simulation_pipeline(
    cfg: SimulationConfig,
    output_dir: str | Path,
    monte_carlo_trials: int | None = None,
    make_plots: bool = True,
) -> dict:
    """Simulate -> process -> detect -> STAP comparison -> (Monte Carlo) -> save results/plots."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    files: list[str] = []
    d = sim_derived(cfg)
    log.info("derived quantities:\n%s", describe_derived(d))

    t0 = time.perf_counter()
    sim = simulate(cfg)
    timings["simulate_s"] = time.perf_counter() - t0
    if cfg.output.save_raw:
        sim.save_npz(out / "synthetic_cpi.npz", include_components=cfg.output.save_components)
        json_dump({**sim.metadata, "config": config_to_dict(cfg)}, out / "synthetic_cpi_metadata.json")
        files += ["synthetic_cpi.npz", "synthetic_cpi_metadata.json"]

    t0 = time.perf_counter()
    proc = process_cpi(sim.data, cfg)
    dets, _ = detect_range_doppler(proc, cfg)
    rd_eval = evaluate_rd_detections(dets, sim.scene.truths, proc, cfg)
    timings["conventional_processing_s"] = time.perf_counter() - t0

    results: dict = {
        "scenario": cfg.scenario_name,
        "description": cfg.description,
        "conventions": {
            "data_layout": sim.metadata["layout"],
            "mixer": cfg.radar.mixer_convention,
            "radial_velocity": RADIAL_VELOCITY_CONVENTION,
            "angle": "cone-complement angle from array normal, sin(theta) = u_x (body frame), positive toward +x",
            "space_time_stacking": STACKING_ORDER,
        },
        "derived": d.as_dict(),
        "assumptions": ASSUMPTIONS,
        "truth": [t.as_dict() for t in sim.scene.truths],
        "range_migration_bins_max": sim.metadata["range_migration_bins_max"],
        "range_doppler_detection": {
            **rd_eval,
            "cfar": {
                "type": "2-D CA-CFAR, square-law, noncoherent sum over channels",
                "guard_cells": cfg.detection.guard_cells,
                "training_cells": cfg.detection.training_cells,
                "pfa_design": cfg.detection.pfa,
            },
            "detections": [x.as_dict() for x in dets[:50]],
        },
    }

    stap_out = None
    if cfg.stap.enabled:
        t0 = time.perf_counter()
        stap_out = run_stap_comparison(sim, proc, cfg)
        timings["stap_comparison_s"] = time.perf_counter() - t0
        results["stap"] = stap_out["summary"]
        if cfg.monte_carlo.enabled and (monte_carlo_trials is None or monte_carlo_trials > 0):
            t0 = time.perf_counter()
            try:
                mc = monte_carlo(sim, proc, cfg, monte_carlo_trials)
                results["monte_carlo"] = mc.as_dict()
            except ValueError as exc:
                results["monte_carlo"] = {"error": str(exc)}
                log.error("Monte Carlo skipped: %s", exc)
            timings["monte_carlo_s"] = time.perf_counter() - t0

    if make_plots:
        files += _make_plots(cfg, sim, proc, dets, stap_out, results, out)

    results["timings"] = timings
    results["files"] = files + ["results.json"]
    json_dump(results, out / "results.json")
    return results


def _make_plots(cfg, sim, proc, dets, stap_out, results, out: Path) -> list[str]:
    from . import plotting as pl

    dpi = cfg.output.plot_dpi
    files = []
    ra = proc.range_axis
    t_f = fast_time_axis(cfg.radar)
    pl.plot_waveform(cfg.radar, t_f, sim.data[0, 0], out / "waveform_if.png", dpi)
    pl.plot_range_profile(ra.range_m, range_profile(proc.Y), sim.scene.truths, proc.range_valid, out / "range_profile.png", dpi)
    pl.plot_range_doppler(proc.rd_map[proc.range_valid], ra.range_m[proc.range_valid], proc.doppler_axis.velocity_mps,
                          dets, sim.scene.truths, out / "range_doppler.png", dpi)
    files += ["waveform_if.png", "range_profile.png", "range_doppler.png"]
    if stap_out is not None:
        setup = stap_out["setup"]
        model = setup.model
        mid = int(setup.test_bins[len(setup.test_bins) // 2])
        tr = select_training_bins(mid, cfg.stap.num_training, cfg.stap.num_guard, setup.valid_training)
        pl.plot_angle_doppler(sample_covariance(setup.snaps[tr]), model, out / "angle_doppler_clutter.png", dpi)
        files.append("angle_doppler_clutter.png")

        def rng_of(bins):
            return np.interp(np.asarray(bins, float), np.arange(ra.num_bins), ra.range_m)

        pl.plot_gmti_maps(stap_out["gmti"], rng_of, model, setup.truth_cells, out / "gmti_outputs.png", dpi)
        files.append("gmti_outputs.png")
        if stap_out["sinr"]:
            a = stap_out["sinr"][0]
            pl.plot_sinr_loss(a, model, out / "sinr_loss.png", dpi)
            pl.plot_eigenspectrum(a.clutter_eigs, a.sample_eigs, a.brennan_rank, noise_covariance_scale(cfg),
                                  out / "eigenspectrum.png", dpi)
            files += ["sinr_loss.png", "eigenspectrum.png"]
        if "monte_carlo" in results and "methods" in results["monte_carlo"]:
            pl.plot_pd_pfa(results["monte_carlo"], out / "pd_pfa.png", dpi)
            files.append("pd_pfa.png")
    return files


def format_summary(results: dict) -> str:
    """Human-readable console report."""
    lines = [f"Scenario: {results['scenario']}"]
    lines.append("Ground truth:")
    for t in results["truth"]:
        lines.append(
            f"  {t['name']:<14} {t['kind']:<9} R={t['range_m']:7.2f} m  v_r={t['radial_velocity_mps']:+6.2f} m/s  "
            f"angle={t['angle_deg']:+6.1f} deg  SNR(raw)={t['snr_db']:+.1f} dB"
        )
    rd = results["range_doppler_detection"]
    lines.append(f"Range-Doppler CFAR: {rd['num_detections']} detections, {rd['num_unassociated_detections']} unassociated")
    for row in rd["targets"]:
        if row["detected"]:
            e = row["errors"]
            lines.append(
                f"  {row['name']:<14} detected: dR={e['range_m']:+.3f} m  dv={e['radial_velocity_mps']:+.3f} m/s  "
                f"dAngle={e['angle_deg']:+.2f} deg"
            )
        else:
            lines.append(f"  {row['name']:<14} not detected in the noncoherent RD map")
    if "stap" in results:
        s = results["stap"]
        lines.append(
            f"STAP: D={s['dimension']} (C={s['num_channels']} x L={s['num_slow_time']}), training L={s['num_training']} "
            f"({s['training_to_dimension_ratio']:.2f} D), look angle {s['look_angle_deg']} deg, {s['num_test_bins']} test bins"
        )
        for name, m in s["methods"].items():
            det = ", ".join(f"{t['name']}={'Y' if t['detected'] else 'n'}" for t in m["targets"])
            lines.append(
                f"  {name:<12} weights {m['weight_seconds']:.3f}s  detections {m['num_detections']:3d} "
                f"(unassociated {m['num_false_alarm_clusters']:3d})  targets: {det}"
            )
        for tname, a in s.get("sinr_analysis", {}).items():
            if "error" in a:
                lines.append(f"  SINR analysis {tname}: ERROR {a['error']}")
                continue
            lines.append(
                f"  SINR @ {tname}: input {a['input_sinr_db_per_element_pulse']:+.1f} dB/elem-pulse, "
                f"optimum {a['optimal_sinr_db']:+.1f} dB, Brennan rank {a['brennan_clutter_rank']}"
            )
            for name, m in a["methods"].items():
                lines.append(f"     {name:<12} SINR_out {m['output_sinr_db']:+6.1f} dB   IF {m['improvement_factor_db']:+6.1f} dB")
    mc = results.get("monte_carlo")
    if mc and "methods" in mc:
        lines.append(f"Monte Carlo ({mc['trials']} trials, target {mc['target']} at {mc['target_snr_db']:+.1f} dB raw SNR):")
        for name, m in mc["methods"].items():
            lines.append(
                f"  {name:<12} P_D={m['pd']:.2f} [{m['pd_95ci'][0]:.2f},{m['pd_95ci'][1]:.2f}]  "
                f"P_FA={m['pfa']:.1e} [{m['pfa_95ci'][0]:.1e},{m['pfa_95ci'][1]:.1e}] (design {m['pfa_design']:.0e})"
            )
    lines.append("Timings: " + ", ".join(f"{k}={v:.1f}" for k, v in results.get("timings", {}).items()))
    return "\n".join(lines)
