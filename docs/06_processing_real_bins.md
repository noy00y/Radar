# 6. Running the pipeline on real `.bin` files

## 6.1 One file

```powershell
python scripts/radar.py process --config configs/radar_capture_highway_overlook.yaml --input bins/overlook_001.bin --output-dir outputs/overlook_001
```

Always pass the config the capture was **recorded with** (the waveform must match). Run `check`
with the same config first if unsure: it prints which stages will run and what is missing.

| Option | Meaning |
|---|---|
| `--config` | capture YAML (required) |
| `--input` | one or more `.bin` files or patterns (see 6.2) |
| `--output-dir` | where results go (created if missing; default `outputs/capture`) |
| `--max-frames N` | process only the first N frames (default: the config's `max_frames`) |
| `--no-plots` | skip the PNGs (faster) |
| `--allow-unverified-format` | decode even if `capture_format_verified: false` (exploration only) |
| `--check-only` | same as the `check` command |
| `-v` | more logging |

Exit code 0 = done, 2 = a problem (message printed).

## 6.2 Several files at once

```powershell
python scripts/radar.py process --config configs/radar_capture_highway_overlook.yaml --input "bins/overlook_*.bin" --output-dir outputs/overlook
python scripts/radar.py process --config configs/radar_capture_highway_overlook.yaml --input bins/overlook_001.bin bins/overlook_002.bin --output-dir outputs/overlook
```

With more than one file, each one gets its own subfolder named after the file:
`outputs/overlook/overlook_001/`, `outputs/overlook/overlook_002/`, ... Patterns are expanded by
the program, so quoting them works in PowerShell too. If one file fails, the others still run
and the exit code is 2. All files must be from the same config.

## 6.3 What `process` does, in order

Each stage only runs if the config has what it needs (that is what `check` reports):

1. **decode** (`adc_loader.py`): read the raw 16-bit words in the configured layout ->
   `frames x RX x chirps x samples`. Reports dropped bytes and out-of-range values.
2. **range** (`range_processing.py`): window + FFT over the samples of each chirp -> distance.
   Bins outside `processing.range_min_m .. range_max_m` are ignored.
3. **doppler + CFAR + angle** (`doppler_processing.py`, `detection.py`, `angle_processing.py`),
   per frame: FFT over the chirps -> radial speed; `rx_calibration` is applied; CA-CFAR finds cells
   that stand out from their neighbours (`detection` section); each detection gets an angle from the
   4 channels.
4. **stap** (`capture.py` -> `stap.py` / `batched.py`), per frame: for every test range bin, learn
   the clutter from neighbouring range bins (and neighbouring frames, `training_frames`), build an
   adaptive filter per look angle x speed hypothesis, apply it, run CFAR along range on each output.
   Repeated for each method in `stap.methods`.
5. **write** `capture_summary.json` and the plots (see [7](07_reading_outputs.md)).

## 6.4 Changing settings without editing the main config

Make a small YAML that starts from the real config and overrides a few keys (sections merge, lists
replace). Put it next to the config:

```yaml
base: radar_capture_vehicle_highway.yaml
max_frames: 50
stap:
  methods: [nonadaptive, efa]      # only the fast methods
  test_range_max_m: 60.0           # only test bins up to 60 m
```

Common overrides: `stap.enabled: false` (no STAP, quick look), `stap.methods`, `stap.backend:
numpy` (force CPU), `detection.pfa` (lower = fewer false alarms, fewer weak detections),
`processing.range_min_m` / `range_max_m`.

## 6.5 GPU

Install CuPy once (`python -m pip install -e ".[gpu]"`). With `stap.backend: auto` (all configs)
`check` prints `STAP backend: cupy`; otherwise it falls back to the CPU with a warning
`CuPy backend unavailable`. `stap.precision: single` is deliberate (consumer GPUs are slow in
double); results match the CPU to ~1e-6. `stap.gpu_batch_bins` = range bins per GPU batch; lower it
if you run out of GPU memory.

## 6.6 Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Refusing to decode: ... missing` | the config lacks a field or has `capture_format_verified: false`; the message lists what |
| the Lua script says `size was X bytes; expected Y` | Studio frame count differs from the config's `adc_format.num_frames`, or packets were dropped: record again |
| the Lua script says `Could not check the supplied output filename` | Studio may have saved under another name (e.g. `..._Raw_0.bin`): use the file that is actually there |
| `capture has N bytes but 600 frames x ... are required` | the recording is shorter than `adc_format.num_frames` (cut short / dropped data). To use what is there: set `adc_format.num_frames: null` (frames are then counted from the file size) and, if it then complains about a partial frame, `allow_partial_final_frame: true` |
| `decode.warnings` mentions trailing bytes | the file is longer than `num_frames` frames; the extra is ignored (harmless) |
| `decode.out_of_range_values` > 0 | layout/format wrong, or the ADC clipped: check `adc_diagnostics.png`, lower RX gain |
| strongest range at the minimum range | leakage / the window glass: raise `processing.range_min_m` |
| STAP `error: ... covariance is singular` | not enough training: more `num_training` / `training_frames`, or use `efa` |
| `CuPy backend unavailable` | CuPy not installed or no CUDA driver: CPU is used instead |
| very slow | see the cost notes in [4](04_highway_overlook.md) / [5](05_car.md); use `--max-frames`, fewer methods, the GPU |
