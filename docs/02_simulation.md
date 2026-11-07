# 2. Simulation: make a scene, run STAP on it

No radar needed. There are three ways, from simplest to most realistic:

| Way | Command | What you get | Use it to |
|---|---|---|---|
| A | `simulate` | scene -> range/Doppler -> STAP, every method scored against the truth (SINR, P_D) | understand / compare STAP methods |
| B | `synth-capture` then `process` | scene -> a fake DCA1000 `.bin` -> the **same** chain real captures go through | rehearse the real-data workflow |
| C | `e2e-overlook` | a ready-made Hwy 401 scene + calibration, through the real chain, scored | rehearse Experiment 2 end to end |

## 2.1 Make your own scene

A scene is a YAML file. Don't write one from scratch: start from `configs/simulation.yaml`
(side-looking radar on a car at 8 m/s, ground clutter, 3 movers, 1 pole) and list only what you change.

Create `configs/my_scene.yaml`:

```yaml
base: simulation.yaml          # load the default scene first (path is relative to THIS file)
scenario_name: my_scene

platform:
  velocity_mps: [12.0, 0.0, 0.0]   # radar moving 12 m/s (43 km/h) along the array axis

targets:                        # a list REPLACES the whole default list
  - name: cyclist
    range_m: 40.0               # distance from the radar
    angle_deg: 15.0             # + = toward RX4 side of the board, 0 = straight out (boresight)
    height_m: 0.0
    radial_velocity_mps: -2.0   # NEGATIVE = coming closer, positive = moving away
    snr_db: -15.0               # per raw sample per channel (STAP/FFT gain comes on top)

stap:
  look_angle_deg: 15.0          # STAP looks in this direction

monte_carlo:
  target_name: cyclist          # must name one of your targets
```

Rules: maps (sections) are merged key by key into the base; lists are replaced whole. A typo in
a key is an error, not silently ignored.

The sections you will usually change:

| Section | Means | Tangible example |
|---|---|---|
| `radar` | the chirp: slope, timing, samples, chirps per CPI | `num_chirps: 16` = 16 pulses per look |
| `array` | antenna positions (half-wavelength units) and TX schedule | 4 RX in a line, TX1 only |
| `platform` | where the radar is and how it moves | `position_m: [0,0,10]` = 10 m up; `velocity_mps` = car speed |
| `clutter` | the ground: how many scatterers, where, how strong | `cnr_db: 27` = ground 27 dB above noise |
| `targets` | moving things you want to find | name, range, angle, radial speed, SNR |
| `stationary_reflectors` | strong static things (poles, signs) | a pole at 38 m |
| `processing` | FFT windows/sizes, range limits | `range_min_m`, `range_max_m` |
| `detection` | CFAR for the plain range-Doppler map | `pfa: 1e-5` = false-alarm rate per cell |
| `stap` | STAP methods and training | `methods: [nonadaptive, mvdr, efa, pc]`, `num_training: 160` |
| `monte_carlo` | repeat with new noise to estimate P_D vs P_FA | `num_trials: 40` |

Speed limit of the default waveform: +-13.8 m/s radial (anything faster wraps around).

## 2.2 Way A: `simulate`

```powershell
python scripts/radar.py simulate --config configs/my_scene.yaml --output-dir outputs/my_scene
```

Options: `--trials 0` skips the Monte Carlo (much faster), `--trials 100` overrides the count,
`--seed 5` gives a different random scene, `--no-plots` skips the PNGs.
Run time on a 4-core CPU: ~1 min without Monte Carlo (default scene); the Monte Carlo adds several minutes.
The `my_scene.yaml` above was run this way: the plain range-Doppler map misses the cyclist, and
mvdr / efa / pc all detect it (output SINR +15.5 / +14.0 / +15.7 dB vs optimum +17.7 dB).

What happens inside: build the scene -> simulate one CPI of raw IF samples for 4 channels ->
range FFT -> Doppler FFT -> CFAR on the range-Doppler map -> STAP with each method at
`stap.look_angle_deg` -> score every detection against the known truth -> optional Monte Carlo.

The console summary (real output, default scene):

```
Ground truth:
  mover_fast     target    R=  45.00 m  v_r= +3.00 m/s  angle= +20.0 deg  SNR(raw)=-18.0 dB
  mover_slow     target    R=  62.00 m  v_r= -0.90 m/s  angle= +20.0 deg  SNR(raw)=-15.0 dB
  ...
Range-Doppler CFAR: 3 detections, 2 unassociated
  mover_fast     not detected in the noncoherent RD map
  mover_offbeam  detected: dR=+0.010 m  dv=-0.091 m/s  dAngle=+0.05 deg
STAP: D=64 (C=4 x L=16), training L=160 (2.50 D), look angle 20.0 deg, 415 test bins
  nonadaptive  weights 0.000s  detections  29 (unassociated  29)  targets: mover_fast=n, mover_slow=n
  mvdr         weights 32.614s  detections  19 (unassociated  17)  targets: mover_fast=Y, mover_slow=Y
  efa          weights 0.580s  detections  13 (unassociated  11)  targets: mover_fast=Y, mover_slow=Y
  SINR @ mover_fast: input -16.4 dB/elem-pulse, optimum +24.1 dB, Brennan rank 13
     mvdr         SINR_out  +23.0 dB   IF  +39.4 dB
```

How to read it: the plain range-Doppler map misses both movers (buried in ground clutter), the
non-adaptive beam misses them, every adaptive STAP method finds both. `unassociated` = detections
not matching any truth object (false alarms or clutter residue). `SINR_out` close to `optimum`
= the method is near ideal. Full output description: [7. Reading outputs](07_reading_outputs.md).

## 2.3 Way B: simulated `.bin` through the real chain

```powershell
python scripts/radar.py synth-capture --sim-config configs/my_scene.yaml --output-dir outputs/my_scene_cap --frames 4
python scripts/radar.py process --config outputs/my_scene_cap/capture.yaml --input outputs/my_scene_cap/adc_data.bin --output-dir outputs/my_scene_cap/processed
```

`synth-capture` writes four files:

| File | Content |
|---|---|
| `adc_data.bin` | raw 16-bit I/Q in the exact DCA1000 byte layout of your real captures |
| `capture.yaml` | a capture config generated from the scene (same keys as the real configs) |
| `synthetic.cfg` | the waveform as a TI mmWave SDK `.cfg` |
| `truth.json` | where every target really is (to compare with the detections) |

Each frame is the same scene (targets do not move between frames) with fresh noise.
`process` then does exactly what it does for a real capture (see [6](06_processing_real_bins.md)).
Measured: 4 frames of the default scene took 144 s on a 4-core CPU (MVDR ~30 s/frame); the
integrated MVDR output found `mover_fast` at 45.09 m / +2.59 m/s and `mover_slow` at
62.07 m / -1.73 m/s (truth: 45 m / +3.0 and 62 m / -0.9; STAP speeds snap to a 0.86 m/s grid).

## 2.4 Way C: Experiment 2 rehearsal

```powershell
python scripts/radar.py e2e-overlook --frames 12 --no-stap        # ~1 min: calibration + range/Doppler/angle
python scripts/radar.py e2e-overlook --frames 12 --backend cupy   # full STAP on the GPU
```

What it does: (1) simulates two corner-reflector captures (0 deg and +27 deg) with deliberate
channel errors and runs `calibrate` on them; (2) simulates the 17th-floor view of Hwy 401: 12 lanes
of cars/trucks at 85-125 km/h, ground clutter, barrier posts, a sign gantry, poles; (3) writes it
as a `.bin` and runs `process` with `configs/radar_capture_highway_overlook.yaml`, once with and once
without the measured calibration; (4) scores the detections against the truth.

Options: `--frames` (>= 9, the STAP training window), `--seed`, `--output-dir`
(default `outputs/e2e_highway_overlook`), `--no-stap`, `--backend`, `--precision`, `--methods mvdr efa`.
The scene assumptions (height 51 m, 25 deg tilt, lane distances) are constants at the top of
`scripts/e2e_highway_overlook.py`; change them if you measure the real geometry.

Runtime: `--no-stap` ~40 s. Full STAP on a 4-core CPU is slow (MVDR ~0.8 s and PC ~1 s per range bin
per frame at D = 512: hours for 12 frames) -> use `--backend cupy` on the RTX 3060, or
`--methods nonadaptive efa` (EFA ~0.2 s per bin).

Output (`outputs/e2e_highway_overlook/`): `e2e_report.json`, the fake `.bin` files, and the
full `process` outputs in `calibrated/` and `uncalibrated/`. The console table:

```
stage                            Pd  det/total  FA/frame  ...  ang mean  ang rms
range_doppler_calibrated       0.18     36/197       7.8   ...      0.0      1.3
range_doppler_uncalibrated     0.18     36/197       8.4   ...     -3.3      3.7
```

`Pd` = fraction of vehicles found; `det/total` = found / vehicles in the test window (all frames);
`FA/frame` = moving detections not near any vehicle; `ang mean/rms` = angle error in degrees
(calibration removes the -3.3 deg bias). Rows ending `_snr13` only count vehicles strong enough
(>= 13 dB after integration) to be detectable at all: that is the fair P_D. Low raw P_D here is the
link budget (most simulated cars are weak at 150-230 m), not a processing bug.
