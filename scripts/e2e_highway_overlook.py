#!/usr/bin/env python3
"""End-to-end synthetic test of Experiment 2 (17th-floor Hwy 401 overlook) through the HARDWARE path.

What it does (everything goes through raw DCA1000 .bin files, exactly like real data):
  1. Room calibration: corner reflector at 0 deg and at +27 deg, recorded with the overlook
     waveform, with injected per-RX phase/gain errors -> scripts' calibrate_capture().
  2. Scene: radar 51 m up, tilted 25 deg down, boresight perpendicular to Hwy 401; 12 lanes
     (collector + express, both directions) with cars/trucks moving between frames at
     85-125 km/h; ground clutter, median-barrier posts, a sign gantry and light poles
     (static discretes); same channel errors; quantised 16-bit ADC.
  3. process_capture() with configs/radar_capture_highway_overlook.yaml + the measured
     calibration: decode -> range -> Doppler/CFAR/angle -> angle-Doppler plot -> STAP
     (nonadaptive / MVDR / EFA / PC, multi-frame pooled training, forward-backward,
     censoring, look-angle sweep) and an uncalibrated run for comparison.
  4. Scores everything against ground truth and writes e2e_report.json.

Scene geometry is estimated from the Google Maps screenshot (50 m scale bar) and is an
assumption; update LANES / HEIGHT_M if you measure them.

    python scripts/radar.py e2e-overlook --frames 12 --output-dir outputs/e2e_highway_overlook
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", str(os.cpu_count() or 1))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from radar_stap.adc_loader import AdcFormat, encode_cube  # noqa: E402
from radar_stap.capture import calibrate_capture, load_capture_config, process_capture  # noqa: E402
from radar_stap.config import load_simulation_config  # noqa: E402
from radar_stap.simulator import simulate  # noqa: E402
from radar_stap.waveforms import complex_noise  # noqa: E402

CAP_CFG = ROOT / "configs" / "radar_capture_highway_overlook.yaml"

# ---------------------------------------------------------------------------
# Scene assumptions (world frame: +y = toward the highway, z up, +x along the road = array axis)
# ---------------------------------------------------------------------------
HEIGHT_M = 51.0            # 17 floors x ~3 m
DEPRESSION_DEG = 25.0      # board tilted down (road spans ~13-45 deg depression)
FRAME_PERIOD_S = 0.050     # studio.frame.period_s in the overlook config
SNR_REF_DB = -28.0         # raw per-sample per-channel SNR of a 10 dBsm target at 100 m (link budget:
                           # 12 dBm, ~10 dBi per antenna, NF ~13 dB, 10 MHz) -> ~+17 dB after integration
CLUTTER_CNR_DB = -10.0     # total road/grass clutter per raw sample (sigma0 ~ -20 dB at 13-45 deg grazing)
# (name, horizontal distance from the building [m], travel direction along +x, speed range [km/h])
# Right-hand traffic seen from the north: the near (north) half is westbound = +x here.
LANES = [
    ("N-collector-1", 79, +1, (85, 105)), ("N-collector-2", 83, +1, (85, 105)), ("N-collector-3", 87, +1, (85, 105)),
    ("N-express-1", 108, +1, (100, 125)), ("N-express-2", 112, +1, (100, 125)), ("N-express-3", 116, +1, (100, 125)),
    ("S-express-1", 142, -1, (100, 125)), ("S-express-2", 146, -1, (100, 125)), ("S-express-3", 150, -1, (100, 125)),
    ("S-collector-1", 172, -1, (85, 105)), ("S-collector-2", 176, -1, (85, 105)), ("S-collector-3", 180, -1, (85, 105)),
]
RADAR = np.array([0.0, 0.0, HEIGHT_M])
# injected receiver errors (what calibration must remove)
RX_PHASE_DEG = np.array([0.0, 25.0, -15.0, 40.0])
RX_GAIN_DB = np.array([0.0, -0.8, 0.6, -0.4])


def snr_db(rcs_dbsm: float, pos) -> float:
    r = float(np.linalg.norm(np.asarray(pos) - RADAR))
    return SNR_REF_DB + (rcs_dbsm - 10.0) - 40.0 * math.log10(r / 100.0)


def make_vehicles(rng):
    cars = []
    for name, y, sign, (lo, hi) in LANES:
        for k in range(int(rng.integers(1, 3))):
            truck = rng.random() < 0.15
            cars.append({
                "name": f"{name}-{'truck' if truck else 'car'}{k}",
                "x0": float(rng.uniform(-200, 200)), "y": float(y), "z": 2.0 if truck else 0.8,
                "vx": sign * float(rng.uniform(lo, hi)) / 3.6, "rcs_dbsm": 20.0 if truck else 10.0,
            })
    return cars


def static_discretes():
    out = []
    for x in np.arange(-180, 181, 12):               # median barrier posts
        out.append(("barrier", [float(x), 129.0, 0.8], 0.0))
    for y in (80, 95, 110, 125, 140, 160, 180):      # overhead sign gantry
        out.append(("gantry", [-60.0, float(y), 7.0], 15.0))
    for x in np.arange(-150, 151, 50):               # light poles, north shoulder
        out.append(("pole", [float(x), 72.0, 10.0], 5.0))
    return [{"name": f"{n}{i}", "snr_db": snr_db(rcs, p), "position_m": p} for i, (n, p, rcs) in enumerate(out)]


def truth_of(cars, t, integration_gain_db=0.0):
    bs = np.array([0.0, math.cos(math.radians(DEPRESSION_DEG)), -math.sin(math.radians(DEPRESSION_DEG))])
    rows = []
    for c in cars:
        p = np.array([c["x0"] + c["vx"] * t, c["y"], c["z"]])
        d = p - RADAR
        R = float(np.linalg.norm(d))
        pattern_db = 20 * math.log10(max(float(d @ bs) / R, 1e-3))  # two-way cos element pattern
        rows.append({"name": c["name"], "position_m": p.tolist(), "range_m": R,
                     "radial_velocity_mps": float(c["vx"] * d[0] / R),
                     "angle_deg": float(np.degrees(np.arcsin(d[0] / R))),  # yaw 0: body x = world x
                     "speed_kmh": abs(c["vx"]) * 3.6,
                     "snr_integrated_db": snr_db(c["rcs_dbsm"], p) + pattern_db + integration_gain_db})
    return rows


# ---------------------------------------------------------------------------
# Simulation -> raw DCA1000 bytes
# ---------------------------------------------------------------------------

def sim_cfg(cc, seed, platform, targets, reflectors, clutter):
    w, f = cc.waveform, cc.adc_format
    if not targets:  # the simulation schema wants at least one target
        targets = [{"name": "dummy", "snr_db": -200.0, "position_m": [0.0, 150.0, 0.0]}]
    ov = {
        "scenario_name": "highway_overlook_e2e",
        "seed": seed,
        "radar": {"carrier_frequency_hz": w.carrier_frequency_hz, "chirp_slope_hz_per_s": w.chirp_slope_hz_per_s,
                  "idle_time_s": w.idle_time_s, "adc_start_time_s": w.adc_start_time_s,
                  "ramp_end_time_s": w.ramp_end_time_s, "adc_sample_rate_hz": w.adc_sample_rate_hz,
                  "num_adc_samples": f.num_adc_samples, "num_chirps": f.chirps_per_frame, "sampling": "complex",
                  "mixer_convention": w.mixer_convention, "noise_power": 1.0},
        "array": {"position_units": cc.antenna.position_units, "rx_positions": cc.antenna.rx_positions,
                  "tx_positions": cc.antenna.tx_positions, "tx_schedule": cc.tx_schedule},
        "platform": platform,
        "clutter": clutter,
        "targets": targets,
        "stationary_reflectors": reflectors,
        "processing": {"range_min_m": 0.0, "range_max_m": None, "range_nfft": None, "doppler_nfft": None},
        "stap": {"enabled": False},
        "monte_carlo": {"enabled": False},
    }
    return load_simulation_config(ROOT / "configs" / "simulation.yaml", ov)


def clean_cube(cfg):
    sim = simulate(cfg)
    return sim.data - sim.components["noise"], sim  # (rx, chirps, samples), noise-free


def write_bin(clean_frames, path, cc, rng, noise_lsb=4.0):
    f = cc.adc_format
    fmt = AdcFormat(data_type="complex", adc_bits=16, num_rx=f.num_rx, num_adc_samples=f.num_adc_samples,
                    chirps_per_frame=f.chirps_per_frame, num_frames=len(clean_frames),
                    sign_handling=f.sign_handling, byte_order=f.byte_order, iq_order=f.iq_order,
                    complex_packing=f.complex_packing, channel_layout=f.channel_layout)
    err = 10 ** (RX_GAIN_DB / 20) * np.exp(1j * np.radians(RX_PHASE_DEG))
    scale, lim, clip, cube = noise_lsb / math.sqrt(0.5), 32767, 0, []
    for clean in clean_frames:
        x = clean * err[:, None, None] + complex_noise(rng, clean.shape, 1.0, np.complex128)
        q = np.rint(x.real * scale) + 1j * np.rint(x.imag * scale)
        clip += int(np.sum(np.abs(q.real) > lim) + np.sum(np.abs(q.imag) > lim))
        q = np.clip(q.real, -lim - 1, lim) + 1j * np.clip(q.imag, -lim - 1, lim)
        cube.append(np.transpose(q, (1, 0, 2)))  # (chirps, rx, samples)
    path.write_bytes(encode_cube(np.stack(cube), fmt))
    return clip


def derived_yaml(path, num_frames, extra=None):
    data = {"base": str(CAP_CFG), "adc_format": {"num_frames": num_frames}, "max_frames": None, **(extra or {})}
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return load_capture_config(path)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def wrap_v(v, vmax):
    return (v + vmax) % (2 * vmax) - vmax


def score(frames_dets, truth_frames, vmax, dr, dv, test_rng, angle_key=None, angle_tol=15.0, min_speed=1.5,
          max_angle=67.5, min_snr_db=None, static_mps=0.5):
    """Pd over 'detectable' vehicles; false detections per frame (moving returns not near any
    vehicle); returns at |v| <= static_mps are real static objects and counted separately."""
    hits = total = fa = static = 0
    v_err, a_err = [], []
    for dets, truth in zip(frames_dets, truth_frames):
        detectable = [t for t in truth if test_rng[0] <= t["range_m"] <= test_rng[1]
                      and abs(t["radial_velocity_mps"]) >= min_speed and abs(t["angle_deg"]) <= max_angle
                      and (min_snr_db is None or t["snr_integrated_db"] >= min_snr_db)]
        used = set()
        for t in detectable:
            vt = wrap_v(t["radial_velocity_mps"], vmax)
            best = None
            for i, d in enumerate(dets):
                if i in used or abs(d["range_m"] - t["range_m"]) > dr:
                    continue
                if abs(wrap_v(d["radial_velocity_mps"] - vt, vmax)) > dv:
                    continue
                if angle_key and abs(d[angle_key] - t["angle_deg"]) > angle_tol:
                    continue
                best = i
                break
            total += 1
            if best is not None:
                used.add(best)
                hits += 1
                d = dets[best]
                v_err.append(float(wrap_v(d["radial_velocity_mps"] - vt, vmax)))
                if angle_key:
                    a_err.append(float(d[angle_key] - t["angle_deg"]))
        # any detection not near ANY vehicle (detectable or not) counts as false
        for i, d in enumerate(dets):
            if i in used:
                continue
            if abs(d["radial_velocity_mps"]) <= static_mps:
                static += 1
                continue
            near = any(abs(d["range_m"] - t["range_m"]) <= dr and
                       abs(wrap_v(d["radial_velocity_mps"] - wrap_v(t["radial_velocity_mps"], vmax), vmax)) <= dv
                       for t in truth)
            fa += 0 if near else 1
    n = max(len(frames_dets), 1)
    return {"pd": hits / total if total else None, "detected": hits, "detectable": total,
            "false_detections_per_frame": fa / n, "static_returns_per_frame": static / n,
            "velocity_error_rms_mps": float(np.sqrt(np.mean(np.square(v_err)))) if v_err else None,
            "angle_error_mean_deg": float(np.mean(a_err)) if a_err else None,
            "angle_error_rms_deg": float(np.sqrt(np.mean(np.square(a_err)))) if a_err else None}


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frames", type=int, default=12, help="scene frames (>= stap.training_frames)")
    ap.add_argument("--seed", type=int, default=401)
    ap.add_argument("--output-dir", default=str(ROOT / "outputs" / "e2e_highway_overlook"))
    ap.add_argument("--no-stap", action="store_true", help="skip the (slow) STAP stage")
    ap.add_argument("--backend", default=None, help="override stap.backend (numpy | numpy_batched | cupy | auto)")
    ap.add_argument("--precision", default=None, help="override stap.precision (double | single)")
    ap.add_argument("--methods", nargs="+", default=None, help="override stap.methods")
    args = ap.parse_args(argv)
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", stream=sys.stdout)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    base = load_capture_config(CAP_CFG)
    report = {"assumptions": {"height_m": HEIGHT_M, "depression_deg": DEPRESSION_DEG, "lanes": LANES,
                              "snr_ref_db": SNR_REF_DB, "clutter_cnr_db": CLUTTER_CNR_DB,
                              "rx_phase_error_deg": RX_PHASE_DEG.tolist(), "rx_gain_error_db": RX_GAIN_DB.tolist()}}
    t_start = time.time()

    # ---- 1. room calibration captures (same waveform, injected channel errors) ----------
    room = {"position_m": [0.0, 0.0, 1.0], "velocity_mps": [0.0, 0.0, 0.0], "yaw_deg": 0.0, "depression_deg": 0.0}
    no_clutter = {"enabled": False, "num_scatterers": 1}
    walls = [{"name": "wall", "snr_db": -5.0, "position_m": [3.0, 14.0, 1.0]},
             {"name": "cabinet", "snr_db": -8.0, "position_m": [-4.0, 7.0, 0.5]}]
    cal_range = 10.0  # >= 8 m: with 4.49 MHz/us the beat at 10 m is 0.30 MHz, above the 175 kHz HPF
    cal = {}
    for ang in (0.0, 27.0):
        p = [cal_range * math.sin(math.radians(ang)), cal_range * math.cos(math.radians(ang)), 1.0]
        cfg = sim_cfg(base, args.seed + 1, room, [{"name": "corner_reflector", "snr_db": 12.0, "position_m": p}],
                      walls, no_clutter)
        clean, _ = clean_cube(cfg)
        path = out / f"room_cal_{int(ang)}deg.bin"
        write_bin([clean] * 8, path, base, rng)
        cal[ang] = path
    cc_cal = derived_yaml(out / "cal.yaml", 8)
    r0 = calibrate_capture(cc_cal, cal[0.0], cal_range, 0.0)
    inj = RX_PHASE_DEG - RX_PHASE_DEG[0]
    report["calibration"] = {
        "measured_phase_deg": r0["phase_offset_deg"], "injected_phase_deg": inj.tolist(),
        "phase_error_deg": [float(a - b) for a, b in zip(r0["phase_offset_deg"], inj)],
        "angle_uncalibrated_deg": r0["angle_uncalibrated_deg"], "warnings": r0["warnings"],
        "rx_calibration": r0["rx_calibration"]}
    cc_chk = derived_yaml(out / "cal_check.yaml", 8, {"rx_calibration": r0["rx_calibration"]})
    r27 = calibrate_capture(cc_chk, cal[27.0], cal_range, 27.0)
    report["calibration"]["verify_27deg"] = {"angle_with_calibration_deg": r27.get("angle_with_config_calibration_deg"),
                                            "angle_uncalibrated_deg": r27["angle_uncalibrated_deg"],
                                            "warnings": r27["warnings"]}
    print(f"[cal] phase error vs injected: {np.round(report['calibration']['phase_error_deg'], 2)} deg; "
          f"27 deg check: {r27.get('angle_with_config_calibration_deg', float('nan')):.2f} deg calibrated, "
          f"{r27['angle_uncalibrated_deg']:.2f} deg uncalibrated")

    # ---- 2. highway scene ------------------------------------------------------------
    platform = {"position_m": RADAR.tolist(), "velocity_mps": [0.0, 0.0, 0.0], "yaw_deg": 0.0,
                "depression_deg": DEPRESSION_DEG, "element_pattern_exponent": 1.0}
    clutter = {"enabled": True, "num_scatterers": 3000, "range_min_m": HEIGHT_M + 4.0, "range_max_m": 290.0,
               "azimuth_min_deg": -75.0, "azimuth_max_deg": 75.0, "ground_height_m": 0.0,
               "cnr_db": CLUTTER_CNR_DB, "range_power_exponent": 3.0, "reflectivity": "rayleigh"}
    static_clean, _ = clean_cube(sim_cfg(base, args.seed, platform, [], static_discretes(), clutter))
    cars = make_vehicles(rng)
    f = base.adc_format
    gain_db = 10 * math.log10(f.chirps_per_frame * f.num_adc_samples * f.num_rx) - 3.5  # Hann range+Doppler loss
    frames, truth_frames = [], []
    for fi in range(args.frames):
        t = fi * FRAME_PERIOD_S
        truth = truth_of(cars, t, gain_db)
        tgts = [{"name": c["name"], "snr_db": snr_db(c["rcs_dbsm"], tr["position_m"]), "position_m": tr["position_m"],
                 "velocity_mps": [c["vx"], 0.0, 0.0]} for c, tr in zip(cars, truth)]
        moving, sim = clean_cube(sim_cfg(base, args.seed + 7, platform, tgts, [], {"enabled": False, "num_scatterers": 1}))
        if fi == 0:
            report["simulator_truth_frame0_sample"] = sim.metadata.get("truth", [])[:3]
        frames.append(static_clean + moving)
        truth_frames.append(truth)
    clip = write_bin(frames, out / "scene.bin", base, rng)
    report["scene"] = {"vehicles": len(cars), "frames": args.frames, "clipped_values": clip,
                       "truth_frame0": truth_frames[0]}
    print(f"[scene] {len(cars)} vehicles, {args.frames} frames, clipped {clip}, "
          f"{time.time() - t_start:.0f} s so far")

    # ---- 3. process: uncalibrated (conventional only) and calibrated (full STAP) ----------
    cc_unc = derived_yaml(out / "scene_uncal.yaml", args.frames, {"stap": {"enabled": False}})
    rep_unc = process_capture(cc_unc, out / "scene.bin", out / "uncalibrated", make_plots=False)
    extra = {"rx_calibration": r0["rx_calibration"]}
    stap_ov = {k: v for k, v in (("backend", args.backend), ("precision", args.precision),
                                 ("methods", args.methods)) if v is not None}
    if args.no_stap:
        stap_ov["enabled"] = False
    if stap_ov:
        extra["stap"] = stap_ov
    cc = derived_yaml(out / "scene_cal.yaml", args.frames, extra)
    t0 = time.time()
    rep = process_capture(cc, out / "scene.bin", out / "calibrated", make_plots=True)
    report["processing_seconds"] = time.time() - t0

    # ---- 4. score --------------------------------------------------------------------
    lam = 299792458.0 / cc.waveform.carrier_frequency_hz
    tc = cc.waveform.idle_time_s + cc.waveform.ramp_end_time_s
    vmax = rep["doppler"]["max_unambiguous_velocity_mps"]
    dr = max(3.0, 2.5 * rep["range"]["bin_spacing_m"])
    test_rng = (cc.stap.test_range_min_m or 0.0, cc.stap.test_range_max_m or 1e9)
    rd_cal = [f["detections"] for f in rep["doppler"]["frames"]]
    rd_unc = [f["detections"] for f in rep_unc["doppler"]["frames"]]
    snr_min = 13.0  # ~CFAR threshold at Pfa 1e-4 plus margin
    res = {"range_doppler_calibrated": score(rd_cal, truth_frames, vmax, dr, 1.0, test_rng, "angle_deg", 10.0),
           "range_doppler_calibrated_snr13": score(rd_cal, truth_frames, vmax, dr, 1.0, test_rng, "angle_deg", 10.0,
                                                   min_snr_db=snr_min),
           "range_doppler_uncalibrated": score(rd_unc, truth_frames, vmax, dr, 1.0, test_rng, "angle_deg", 10.0),
           "range_doppler_uncalibrated_snr13": score(rd_unc, truth_frames, vmax, dr, 1.0, test_rng, "angle_deg", 10.0,
                                                     min_snr_db=snr_min)}
    st = rep.get("stap")
    if isinstance(st, dict) and "methods" in st:
        report["stap_setup"] = {k: st[k] for k in ("look_angles_deg", "dimension", "num_frames", "num_test_bins",
                                                  "test_range_m", "training")}
        for name, m in st["methods"].items():
            if "frames" not in m:
                res[f"stap_{name}"] = {"error": m.get("error")}
                continue
            dets = [f["detections"] for f in m["frames"]]
            extra_m = {"weight_seconds_per_frame": m["weight_seconds_per_frame"], "error": m.get("error")}
            res[f"stap_{name}"] = {**score(dets, truth_frames, vmax, dr, 1.0, test_rng, "look_angle_deg", 15.0), **extra_m}
            res[f"stap_{name}_snr13"] = {**score(dets, truth_frames, vmax, dr, 1.0, test_rng, "look_angle_deg", 15.0,
                                                 min_snr_db=snr_min), **extra_m}
    report["results"] = res
    report["capture_files"] = rep["files"]
    report["wavelength_m"], report["chirp_period_s"], report["vmax_mps"] = lam, tc, vmax
    (out / "e2e_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\n{'stage':34s} {'Pd':>6s} {'det/total':>10s} {'FA/frame':>9s} {'static/fr':>9s} {'v rms':>6s} "
          f"{'ang mean':>9s} {'ang rms':>8s}")
    for k, v in res.items():
        if v.get("pd") is None and "error" in v and v["error"]:
            print(f"{k:34s} ERROR {v['error']}")
            continue
        f = lambda x, p=2: "-" if x is None else f"{x:.{p}f}"  # noqa: E731
        print(f"{k:34s} {f(v['pd']):>6s} {str(v['detected']) + '/' + str(v['detectable']):>10s} "
              f"{f(v['false_detections_per_frame'], 1):>9s} {f(v['static_returns_per_frame'], 1):>9s} "
              f"{f(v['velocity_error_rms_mps']):>6s} "
              f"{f(v['angle_error_mean_deg'], 1):>9s} {f(v['angle_error_rms_deg'], 1):>8s}")
    print(f"\nReport: {out / 'e2e_report.json'}  (total {time.time() - t_start:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
