#!/usr/bin/env python3
"""Decode and process a DCA1000EVM raw ADC capture.

    python scripts/process_capture.py --config configs/radar_capture.yaml \
        --input PATH_TO_CAPTURE --output-dir outputs/capture

The configuration must fully specify (and mark as verified) the binary format.
Use ``--check-only`` to see which settings are still missing.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from radar_stap.cli import process_capture_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(process_capture_main())
