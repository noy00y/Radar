# 1. Setup (do this once)

Everything is run from a terminal **in the repository root** (the folder that contains
`pyproject.toml`, `scripts/`, `configs/`, `src/`). Every command in these docs assumes that.

## 1.1 On your PC (Windows, where mmWave Studio and the RTX 3060 are)

1. Install Python 3.10 or newer (python.org). Tick "Add python.exe to PATH".
2. Open PowerShell in the repository root (Shift + right-click the folder -> "Open PowerShell window here").
3. Install the code and its dependencies (numpy, scipy, matplotlib, pyyaml, pytest):

   ```powershell
   python -m pip install -e ".[dev]"
   ```

   `-e` means "editable": edits to the code take effect immediately, no reinstall.
4. Optional, for the GPU (makes STAP on the overlook data many times faster):

   ```powershell
   python -m pip install -e ".[gpu]"
   ```

   This installs CuPy for CUDA 12. You need an NVIDIA driver that supports CUDA 12. Nothing else
   changes: the configs say `stap.backend: auto`, which uses the GPU when CuPy works and the CPU otherwise.
5. Check that it works:

   ```powershell
   python scripts/radar.py --help
   python -m pytest -q
   ```

   `--help` prints the command list (below). `pytest` takes ~30-40 s and should end with
   `133 passed, 1 skipped` (the skip is the GPU test; with CuPy installed it runs too).

## 1.2 On Databricks (no radar, no GPU)

Open the notebook `notebooks/run_on_databricks`, attach **Serverless** compute (the shared
group cluster cannot read your home folder) and run the cells top to bottom:

| Cell | What it does |
|---|---|
| Setup | finds the repo, prints the command list |
| Run the test suite | `pytest` on the whole repo |
| Exp 2 rehearsal, quick | synthetic overlook through the real processing chain, no STAP (~1 min) |
| Exp 2 rehearsal, full STAP | same with STAP, in the background (~1 h on CPU) |
| Check background run / Process health | poll the background run |

## 1.3 The one command you need: `scripts/radar.py`

Every task is `python scripts/radar.py <command> [options]`. Add `--help` after any command to
see all of its options (e.g. `python scripts/radar.py process --help`).

| Command | Use it for | Needs a radar? |
|---|---|---|
| `check` | "Is this config complete? What will run?" | no |
| `calibrate` | Experiment 1: corner-reflector calibration + angle-sign check | a capture |
| `process` | Experiments 1-3: decode a `.bin` and run range / Doppler / angle / STAP | a capture |
| `simulate` | make a simulated scene and run STAP on it, scored against the truth | no |
| `synth-capture` | turn a simulated scene into a fake `.bin`, then run `process` on it | no |
| `e2e-overlook` | full Experiment 2 rehearsal on a synthetic highway | no |
| `benchmark` | time the STAP maths on CPU or GPU | no |

`scripts/process_capture.py` and `scripts/calibrate_capture.py` (if still present) are old
shortcuts for `radar.py process` and `radar.py calibrate`; they take the same options.

## 1.4 Files you will touch

| File | What it is | When you edit it |
|---|---|---|
| `scripts/capture_v2_verified_use.lua` | mmWave Studio recording script (see 1.5) | before every recording (3 lines at the top) |
| `configs/radar_capture_highway_overlook.yaml` | Experiment 2 (and its calibration) | paste `rx_calibration` after Exp 1 |
| `configs/radar_capture_vehicle.yaml` | Experiment 3, regular speed (<= ~45 km/h) | `rx_calibration`, `platform_velocity_mps` per drive |
| `configs/radar_capture_vehicle_highway.yaml` | Experiment 3, highway (50-100 km/h) | `rx_calibration`, `platform_velocity_mps` per drive |
| `configs/simulation.yaml` | the default simulated scene | copy it to make your own scenes (see doc 2) |

Each capture config is long, but every line is commented. Its header explains why the waveform was
chosen; its `studio:` section lists the exact values to load in mmWave Studio.

## 1.5 Recording a capture (used by every experiment)

1. In mmWave Studio, connect, load the radar settings from the experiment config's `studio:`
   section, and set the DCA1000 up (as you already do). Set the **frame count** from the table below.
2. Open `scripts/capture_v2_verified_use.lua` and set the three numbers at the top for that recording:

   | Recording | Config (load it in Studio) | Studio frames | `capture_ms` | `expected_bytes` |
   |---|---|---|---|---|
   | calibration, overlook | `radar_capture_highway_overlook.yaml` | 64 | 6000 | 31457280 |
   | calibration, car regular | `radar_capture_vehicle.yaml` | 64 | 9000 | 8388608 |
   | calibration, car highway | `radar_capture_vehicle_highway.yaml` | 64 | 4000 | 3932160 |
   | Exp 2 overlook | `radar_capture_highway_overlook.yaml` | 600 | 35000 | 294912000 |
   | Exp 3 car regular | `radar_capture_vehicle.yaml` | 128 | 16000 | 16777216 |
   | Exp 3 car highway | `radar_capture_vehicle_highway.yaml` | 3000 | 65000 | 184320000 |

   `capture_count` = how many recordings back to back (1 is fine; 3 gives spares).
   `expected_bytes` = frames x chirps x 4 RX x samples x 4 bytes; `capture_ms` = frames x frame period + margin.
3. Run the script in Studio (Lua shell). Each recording must log `size matches expected` in green.
4. The script always writes `bins/stap_capture_001.bin`, `_002.bin`, ... and **overwrites them on the
   next run**. Rename the files right away to say what they are; these docs use names like
   `cal_0deg_001.bin`, `overlook_001.bin`, `car_fast_001.bin`.

(`scripts/capture_v3_untested.lua` does the same with a `name` prefix instead of fixed names; it has
not been tried on the hardware yet.)

Next: [2. Simulation](02_simulation.md) (no hardware) or [3. Calibration](03_calibration.md) (first hardware step).
