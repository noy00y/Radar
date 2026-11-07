# radar-stap

STAP (space-time adaptive processing) for ground moving-target detection with a
**TI IWR1843BOOST + DCA1000EVM** (77 GHz FMCW, 4 RX, single TX), plus a simulator that
produces the same data, so every step can be rehearsed without the radar.

You record raw ADC data in mmWave Studio (`scripts/capture_v2_verified_use.lua`, see
[1.5](docs/01_setup.md#15-recording-a-capture-used-by-every-experiment)), then one program does the rest:

```powershell
python scripts/radar.py <command> --help
```

| Command | Does |
|---|---|
| `check` | is a capture config complete, which stages will run |
| `calibrate` | Experiment 1: channel calibration from a corner reflector |
| `process` | Experiments 1-3: `.bin` -> range / speed / angle / STAP detections (one or many files) |
| `simulate` | simulated scene -> STAP, scored against the truth |
| `synth-capture` | simulated scene -> fake `.bin` + config, for `process` |
| `e2e-overlook` | full Experiment 2 rehearsal on a synthetic highway |
| `benchmark` | STAP timing on CPU / GPU |

## Read in this order

1. [Setup](docs/01_setup.md): install, run the tests, the files you will edit
2. [Simulation](docs/02_simulation.md): make a scene, run STAP on it (no hardware)
3. [Experiment 1: calibration](docs/03_calibration.md): corner reflector, distance and left/right checks
4. [Experiment 2: highway overlook](docs/04_highway_overlook.md): static radar, 17th floor, Hwy 401
5. [Experiment 3: car](docs/05_car.md): side-looking radar on a moving car, regular speed and highway
6. [Processing real `.bin` files](docs/06_processing_real_bins.md): one or many files, options, overrides, GPU, troubleshooting
7. [Reading the outputs](docs/07_reading_outputs.md): every file and JSON field
8. [Code map](docs/08_code_map.md): modules, data flow, terms

## Shortest path for each goal

| Goal | Commands |
|---|---|
| see STAP work, no radar | `python scripts/radar.py simulate --config configs/simulation.yaml --trials 0` |
| calibrate | record, rename to `cal_0deg_001.bin` -> `python scripts/radar.py calibrate --config <config> --input bins/cal_0deg_001.bin --range-m 10 --angle-deg 0` -> paste `rx_calibration` |
| highway overlook | record, rename to `overlook_001.bin` -> `python scripts/radar.py process --config configs/radar_capture_highway_overlook.yaml --input bins/overlook_001.bin --output-dir outputs/overlook_001 --max-frames 12` |
| car | record, rename to `car_fast_001.bin` -> `python scripts/radar.py process --config configs/radar_capture_vehicle_highway.yaml --input bins/car_fast_001.bin --output-dir outputs/car_fast_001 --max-frames 20` |
| many files | `--input "bins/overlook_*.bin"` (one output subfolder per file) |

## Layout

```text
configs/    capture configs (one per experiment) + simulation.yaml (default scene)
scripts/    radar.py (all commands), capture_v2_verified_use.lua (mmWave Studio), e2e_highway_overlook.py, benchmark_stap.py
src/radar_stap/   the package (see docs/08_code_map.md)
tests/      pytest suite (python -m pytest -q)
notebooks/run_on_databricks   tests + Experiment 2 rehearsal on Databricks Serverless
docs/       this documentation
bins/, outputs/   your recordings / results (not code)
```

Sign conventions: radial velocity negative = approaching; angle positive = toward the RX4 end of the board.
