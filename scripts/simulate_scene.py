#!/usr/bin/env python3
"""Simulate a synthetic FMCW scene and run the full processing / STAP / evaluation chain.

Example (from the repository root):
    python scripts/simulate_scene.py --config configs/simulation.yaml --output-dir outputs/simulation
"""

import os

# Small-matrix STAP is much faster single-threaded; NumPy and SciPy can ship separate
# OpenBLAS builds whose thread pools contend. Respect any user setting.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from radar_stap.cli import simulate_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(simulate_main())
