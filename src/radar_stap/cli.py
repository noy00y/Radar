"""Command-line entry points. One command for every experiment pipeline:

    python scripts/radar.py check      --config CFG                         # what the config supports
    python scripts/radar.py calibrate  --config CFG --input BIN --range-m R --angle-deg A   # Exp 1
    python scripts/radar.py process    --config CFG --input BIN [BIN ...] --output-dir OUT  # Exp 1/2/3
    python scripts/radar.py synth-capture --sim-config SCENE.yaml --output-dir OUT     # simulated .bin
    python scripts/radar.py e2e-overlook [--frames N --backend cupy ...]   # Exp 2 rehearsal (synthetic)
    python scripts/radar.py simulate   --config configs/simulation.yaml    # Exp 3 rehearsal (moving platform)
    python scripts/radar.py benchmark  [--backend cupy ...]               # STAP compute / GPU sizing
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


def simulate_main(argv: list[str] | None = None) -> int:
    from .config import ConfigError, load_simulation_config
    from .pipeline import format_summary, run_simulation_pipeline

    ap = argparse.ArgumentParser(description="Simulate an FMCW scene and run range/Doppler/CFAR/STAP evaluation.")
    ap.add_argument("--config", default="configs/simulation.yaml")
    ap.add_argument("--output-dir", default="outputs/simulation")
    ap.add_argument("--seed", type=int, default=None, help="override the scenario seed")
    ap.add_argument("--trials", type=int, default=None, help="Monte Carlo trials (0 disables)")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    _setup_logging(args.verbose)
    try:
        cfg = load_simulation_config(args.config, {"seed": args.seed} if args.seed is not None else None)
    except ConfigError as exc:
        print(f"Configuration error:\n{exc}", file=sys.stderr)
        return 2
    results = run_simulation_pipeline(cfg, args.output_dir, args.trials, not args.no_plots)
    print(format_summary(results))
    print(f"\nResults written to {Path(args.output_dir).resolve()}")
    return 0


def process_capture_main(argv: list[str] | None = None) -> int:
    from .adc_loader import AdcFormatError, CaptureDecodeError
    from .capture import CaptureConfigError, assess_capture_config, load_capture_config, process_capture
    from .config import ConfigError
    from .mmwave_cfg import MmwaveCfgError

    ap = argparse.ArgumentParser(description="Decode and process a DCA1000 raw ADC capture (explicit format only).")
    ap.add_argument("--config", required=True,
                    help="capture YAML: configs/radar_capture_highway_overlook.yaml | radar_capture_vehicle.yaml | "
                         "radar_capture_vehicle_highway.yaml")
    ap.add_argument("--input", nargs="+", required=False,
                    help="one or more raw .bin captures or glob patterns (e.g. bins/overlook_*.bin); with several "
                         "files each result goes to OUTPUT_DIR/<file name without .bin>/")
    ap.add_argument("--output-dir", default="outputs/capture")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--allow-unverified-format", action="store_true",
                    help="decode although capture_format_verified is false (exploration only)")
    ap.add_argument("--check-only", action="store_true", help="only report configuration completeness")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    _setup_logging(args.verbose)
    try:
        cc = load_capture_config(args.config)
    except (ConfigError, CaptureConfigError, MmwaveCfgError, FileNotFoundError) as exc:
        print(f"Capture configuration error:\n{exc}", file=sys.stderr)
        return 2
    assess = assess_capture_config(cc, args.allow_unverified_format)
    print("Processing stages supported by this capture configuration:")
    print(assess.message())
    if args.check_only:
        return 0 if assess.available("decode") else 2
    if not args.input:
        print("error: --input is required unless --check-only is given", file=sys.stderr)
        return 2
    if not assess.available("decode"):
        print(
            "\nRefusing to decode: the binary format is incomplete or unverified. Fill in the missing "
            "fields above in the capture YAML (values come from your capture setup, never from the "
            "simulation config), verify them with a known capture, then set capture_format_verified: true.",
            file=sys.stderr,
        )
        return 2
    import glob

    inputs: list[str] = []
    for p in args.input:  # expand patterns ourselves (Windows shells do not)
        inputs += sorted(glob.glob(p)) or [p]
    multi = len(inputs) > 1
    rc = 0
    for path in inputs:
        out_dir = Path(args.output_dir) / Path(path).stem if multi else Path(args.output_dir)
        if multi:
            print(f"\n=== {path} -> {out_dir}")
        try:
            rep = process_capture(cc, path, out_dir, args.allow_unverified_format, args.max_frames,
                                  not args.no_plots)
        except (CaptureConfigError, AdcFormatError, CaptureDecodeError, FileNotFoundError) as exc:
            print(f"Capture processing failed for {path}:\n{exc}", file=sys.stderr)
            rc = 2
            continue
        print(json.dumps({k: rep[k] for k in ("data_shape_frames_rx_chirps_samples", "decode")}, indent=2))
        for st in ("range", "doppler", "stap"):
            v = rep.get(st)
            print(f"{st}: {'unavailable' if v == 'unavailable' or v is None else 'done'}")
        print(f"Summary written to {out_dir.resolve() / 'capture_summary.json'}")
    return rc


def synth_capture_main(argv: list[str] | None = None) -> int:
    from .config import ConfigError, load_simulation_config
    from .synthetic_capture import make_synthetic_capture

    ap = argparse.ArgumentParser(
        description="Turn a simulated scene (simulation YAML) into a DCA1000-style capture: adc_data.bin + "
                    "capture.yaml + synthetic.cfg + truth.json, ready for `radar.py process`.")
    ap.add_argument("--sim-config", default="configs/simulation.yaml", help="scene / waveform YAML")
    ap.add_argument("--output-dir", default="outputs/synthetic_capture")
    ap.add_argument("--frames", type=int, default=4, help="frames to write (same scene, new noise each frame)")
    ap.add_argument("--noise-lsb", type=float, default=4.0, help="noise std per I/Q component in ADC codes")
    ap.add_argument("--no-stap", action="store_true", help="write capture.yaml with STAP disabled")
    args = ap.parse_args(argv)
    try:
        cfg = load_simulation_config(args.sim_config)
    except (ConfigError, FileNotFoundError) as exc:
        print(f"Configuration error:\n{exc}", file=sys.stderr)
        return 2
    info = make_synthetic_capture(cfg, args.output_dir, args.frames, 16, args.noise_lsb, None, not args.no_stap)
    print(json.dumps({k: info[k] for k in ("bytes", "num_frames", "clipped_values")}, indent=2))
    out = Path(args.output_dir)
    print(f"wrote {out.resolve()}\nnext: python scripts/radar.py process --config {out / 'capture.yaml'} "
          f"--input {out / 'adc_data.bin'} --output-dir {out / 'processed'}")
    return 0


def _run_repo_script(name: str, argv: list[str]) -> int:
    """Run a script from the repository's scripts/ folder as __main__ with the given argv."""
    import runpy

    path = Path(__file__).resolve().parents[2] / "scripts" / name
    if not path.is_file():
        print(f"{name} not found next to the package (needs a repository checkout): {path}", file=sys.stderr)
        return 2
    old = sys.argv
    sys.argv = [str(path), *argv]
    try:
        runpy.run_path(str(path), run_name="__main__")
    except SystemExit as exc:
        return int(exc.code or 0)
    finally:
        sys.argv = old
    return 0


COMMANDS = {
    "check": "capture config completeness / stage report (process --check-only)",
    "calibrate": "Exp 1: rx_calibration from a corner-reflector capture + angle-sign check",
    "process": "Exp 1/2/3: decode + range/Doppler/angle + STAP for a recorded .bin",
    "e2e-overlook": "Exp 2 rehearsal: synthetic Hwy 401 overlook through the hardware path, scored vs truth",
    "simulate": "simulated scene -> range/Doppler/STAP scored vs truth + Monte Carlo (configs/simulation.yaml)",
    "synth-capture": "simulated scene -> DCA1000-style .bin + capture.yaml (then run `process` on it)",
    "benchmark": "STAP weight-computation timing vs dimension (CPU, or --backend cupy on a GPU)",
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in COMMANDS:
        print("usage: radar.py COMMAND [options]   (COMMAND --help for its options)\n")
        for k, v in COMMANDS.items():
            print(f"  {k:13s} {v}")
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "check":
        return process_capture_main([*rest, "--check-only"])
    if cmd == "process":
        return process_capture_main(rest)
    if cmd == "calibrate":
        return calibrate_capture_main(rest)
    if cmd == "simulate":
        return simulate_main(rest)
    if cmd == "synth-capture":
        return synth_capture_main(rest)
    if cmd == "e2e-overlook":
        return _run_repo_script("e2e_highway_overlook.py", rest)
    return _run_repo_script("benchmark_stap.py", rest)


def calibrate_capture_main(argv: list[str] | None = None) -> int:
    from .adc_loader import AdcFormatError, CaptureDecodeError
    from .capture import CaptureConfigError, calibrate_capture, load_capture_config
    from .config import ConfigError
    from .mmwave_cfg import MmwaveCfgError

    ap = argparse.ArgumentParser(description="Per-channel phase/gain calibration from a corner-reflector capture.")
    ap.add_argument("--config", required=True, help="capture YAML used for the reflector capture")
    ap.add_argument("--input", required=True, help="raw .bin of the reflector capture")
    ap.add_argument("--range-m", type=float, required=True, help="tape-measured reflector range [m]")
    ap.add_argument("--angle-deg", type=float, default=0.0, help="known reflector angle (+ = toward RX4) [deg]")
    ap.add_argument("--search-m", type=float, default=1.0, help="search +- this around --range-m")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--allow-unverified-format", action="store_true")
    ap.add_argument("--output", default=None, help="optional JSON file for the full result")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    _setup_logging(args.verbose)
    try:
        cc = load_capture_config(args.config)
        res = calibrate_capture(cc, args.input, args.range_m, args.angle_deg, args.search_m,
                                args.allow_unverified_format, args.max_frames)
    except (ConfigError, CaptureConfigError, MmwaveCfgError, AdcFormatError, CaptureDecodeError,
            FileNotFoundError) as exc:
        print(f"Calibration failed:\n{exc}", file=sys.stderr)
        return 2
    print(f"Reflector at {res['reflector_range_m_measured']:.2f} m, {res['reflector_snr_db']:.1f} dB above median, "
          f"{res['num_frames']} frames")
    print(f"Uncalibrated angle: {res['angle_uncalibrated_deg']:.2f} deg (assumed {args.angle_deg} deg)")
    if "angle_with_config_calibration_deg" in res:
        print(f"Angle with config rx_calibration: {res['angle_with_config_calibration_deg']:.2f} deg "
              f"(error {res['angle_error_with_config_calibration_deg']:+.2f} deg)")
    for c, (ph, g, j) in enumerate(zip(res["phase_offset_deg"], res["gain_offset_db"], res["phase_jitter_deg"])):
        print(f"  ch{c:2d}: phase {ph:+7.1f} deg  gain {g:+5.2f} dB  jitter {j:4.1f} deg")
    for w in res["warnings"]:
        print(f"WARNING: {w}")
    print("\nPaste into the capture YAML:\nrx_calibration: " + json.dumps(res["rx_calibration"]))
    if args.output:
        Path(args.output).write_text(json.dumps(res, indent=2))
    return 1 if res["warnings"] else 0
