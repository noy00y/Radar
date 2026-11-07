#!/usr/bin/env python3
"""Single entry point for the three experiment pipelines (run from the repository root).

  Exp 1  calibration (corner reflector at a measured spot: distance + left/right check)
         python scripts/radar.py calibrate --config configs/<experiment>.yaml --input cal_0deg.bin \
             --range-m 10 --angle-deg 0          # -> prints rx_calibration to paste into the YAML
         python scripts/radar.py calibrate --config configs/<experiment>.yaml --input cal_27deg.bin \
             --range-m 10 --angle-deg 27         # verify: angle with calibration must read ~+27
  Exp 2  stationary highway overlook
         python scripts/radar.py check   --config configs/radar_capture_highway_overlook.yaml
         python scripts/radar.py process --config configs/radar_capture_highway_overlook.yaml \
             --input overlook_001.bin --output-dir outputs/overlook_001
         python scripts/radar.py e2e-overlook --frames 12 [--backend cupy]   # synthetic rehearsal
  Exp 3  car STAP (regular speed: radar_capture_vehicle.yaml, fast: radar_capture_vehicle_highway.yaml)
         python scripts/radar.py process --config configs/radar_capture_vehicle_highway.yaml \
             --input car_fast_001.bin --output-dir outputs/car_fast_001
         python scripts/radar.py simulate --config configs/simulation.yaml   # moving-platform rehearsal
  Tools  python scripts/radar.py benchmark [--backend cupy]                 # STAP compute / GPU sizing
"""

import os
import sys
from pathlib import Path

# Small per-bin STAP matrices are much faster single-threaded (NumPy/SciPy OpenBLAS pools contend);
# the synthetic e2e run and GPU benchmark manage their own threading.
if len(sys.argv) < 2 or sys.argv[1] not in ("e2e-overlook", "benchmark"):
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from radar_stap.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
