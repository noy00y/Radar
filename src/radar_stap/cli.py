"""Command-line entry points (wrapped by the scripts in ``scripts/``)."""

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
    ap.add_argument("--config", default="configs/radar_capture.yaml")
    ap.add_argument("--input", required=False, help="raw .bin capture (header-free DCA1000 stream)")
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
    try:
        rep = process_capture(cc, args.input, args.output_dir, args.allow_unverified_format, args.max_frames,
                              not args.no_plots)
    except (CaptureConfigError, AdcFormatError, CaptureDecodeError, FileNotFoundError) as exc:
        print(f"Capture processing failed:\n{exc}", file=sys.stderr)
        return 2
    print(json.dumps({k: rep[k] for k in ("data_shape_frames_rx_chirps_samples", "decode")}, indent=2))
    for st in ("range", "doppler", "stap"):
        v = rep.get(st)
        print(f"{st}: {'unavailable' if v == 'unavailable' or v is None else 'done'}")
    print(f"Summary written to {Path(args.output_dir).resolve() / 'capture_summary.json'}")
    return 0
