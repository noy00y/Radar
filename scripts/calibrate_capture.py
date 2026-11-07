#!/usr/bin/env python3
"""Per-channel phase/gain calibration from a corner-reflector capture.

Step 1 (measure): reflector straight ahead (or at a measured angle), static scene:

    python scripts/calibrate_capture.py --config configs/radar_capture_vehicle_highway.yaml \
        --input cal_0deg.bin --range-m 5.0 --angle-deg 0

    -> paste the printed ``rx_calibration: [...]`` into the capture YAML.

Step 2 (verify + angle sign): reflector moved to a known side angle, same command with
``--angle-deg 27`` on the new capture. "Angle with config rx_calibration" must match.
Exit code 1 if any warning (weak reflector, unstable phase, reversed angle sign).
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from radar_stap.cli import calibrate_capture_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(calibrate_capture_main())
