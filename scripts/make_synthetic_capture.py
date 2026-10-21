#!/usr/bin/env python3
"""Generate a synthetic DCA1000-style capture (bin + SDK .cfg + capture YAML) from a simulation config.

    python scripts/make_synthetic_capture.py --sim-config configs/scenarios/iwr1843_tdm.yaml \
        --output-dir outputs/synthetic_capture --frames 2
    python scripts/process_capture.py --config outputs/synthetic_capture/capture.yaml \
        --input outputs/synthetic_capture/adc_data.bin --output-dir outputs/capture_synthetic

This exercises decoding, configuration parsing and processing end-to-end without
hardware. It does NOT validate that a real IWR1843/DCA1000 capture uses the
chosen layout -- that must be verified on hardware (README checklist).
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from radar_stap.config import load_simulation_config  # noqa: E402
from radar_stap.synthetic_capture import make_synthetic_capture  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-config", default="configs/scenarios/iwr1843_tdm.yaml")
    ap.add_argument("--output-dir", default="outputs/synthetic_capture")
    ap.add_argument("--frames", type=int, default=2)
    ap.add_argument("--adc-bits", type=int, default=16, choices=(12, 14, 16))
    ap.add_argument("--noise-lsb", type=float, default=4.0)
    ap.add_argument("--iq-order", choices=("I_first", "Q_first"))
    ap.add_argument("--complex-packing", choices=("iq_interleaved", "two_sample_blocks"))
    ap.add_argument("--channel-layout", choices=("rx_planar_per_chirp", "rx_interleaved_per_sample"))
    ap.add_argument("--sign-handling", choices=("int16_twos_complement", "sign_extend_adc_bits"))
    args = ap.parse_args()
    over = {k: v for k, v in {
        "iq_order": args.iq_order,
        "complex_packing": args.complex_packing,
        "channel_layout": args.channel_layout,
        "sign_handling": args.sign_handling,
    }.items() if v is not None}
    cfg = load_simulation_config(args.sim_config)
    info = make_synthetic_capture(cfg, args.output_dir, args.frames, args.adc_bits, args.noise_lsb, over)
    print(json.dumps({k: info[k] for k in ("bytes", "num_frames", "clipped_values", "adc_format")}, indent=2))
    print(f"wrote {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
