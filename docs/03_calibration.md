# 3. Experiment 1: calibration (corner reflector)

**Why.** The 4 receive channels have slightly different phase and gain (PCB traces, mixers).
Uncorrected, every angle is biased by a few degrees and STAP cannot cancel clutter deeply. One strong
reflector at a known spot shows what the 4 channels *should* see; the difference is the correction
(`rx_calibration`: one complex multiplier per channel). The same captures also check **distance**
(is the measured range right?) and **left/right** (is a positive angle really on the side you think?).

**Do it once per experiment config you will use** (the correction is measured with that config's
waveform; each run takes ~2 minutes). Redo it if you remount the board or change RX gain.

## 3.1 What you need

* a corner reflector (trihedral), a tape measure, a tripod;
* an open space: parking lot, long corridor, balcony. Nothing else strong within ~1 m of the reflector's distance;
* the board mounted the way it will be used: RX1 -> RX4 horizontal, facing the reflector.

Distance: **10 m** (works for all three configs). Not closer than 8 m with the overlook config: its
low chirp slope puts closer objects below the receiver's 175 kHz high-pass filter.

## 3.2 Step by step

1. **mmWave Studio**: load the settings of the experiment you are calibrating for (the `studio:`
   section of its config; e.g. [4.1](04_highway_overlook.md#41-mmwave-studio-settings)).
2. **Place the reflector straight ahead** (0 deg), 10.0 m from the board, same height, opening facing the radar.
   Measure from the antenna area of the board to the reflector's inner corner.
3. **Record** ([1.5](01_setup.md#15-recording-a-capture-used-by-every-experiment)): Studio frame
   count 64; in `scripts/capture_v2_verified_use.lua`, e.g. for the overlook config:

   ```lua
   local capture_count = 1
   local capture_ms = 6000
   local expected_bytes = 31457280
   ```

   Run it; the log must say `size matches expected`. Rename `bins/stap_capture_001.bin` to
   `bins/cal_0deg_001.bin`.
4. **Compute the calibration**:

   ```powershell
   python scripts/radar.py calibrate --config configs/radar_capture_highway_overlook.yaml --input bins/cal_0deg_001.bin --range-m 10 --angle-deg 0
   ```

5. **Read the result.** It looks like this (numbers illustrative):

   ```
   Reflector at 10.07 m, 36.5 dB above median, 64 frames
   Uncalibrated angle: -3.30 deg (assumed 0.0 deg)
     ch 0: phase    +0.0 deg  gain +0.00 dB  jitter  0.3 deg
     ch 1: phase   +25.4 deg  gain -0.81 dB  jitter  0.4 deg
     ch 2: phase   -14.8 deg  gain +0.62 dB  jitter  0.4 deg
     ch 3: phase   +40.3 deg  gain -0.38 dB  jitter  0.5 deg

   Paste into the capture YAML:
   rx_calibration: [[1.0, 0.0], [0.89, -0.43], [0.95, 0.25], [0.74, -0.63]]
   ```

   | Line | Pass if | If not |
   |---|---|---|
   | `Reflector at X m` | within ~1 range bin of your tape (overlook 1.3 m, car-highway 0.53 m, car-regular 0.16 m) | the chirp slope in the config is wrong: copy the slope Studio reports into `waveform.chirp_slope_hz_per_s` |
   | `dB above median` | >= 20 dB | move closer, aim the reflector, remove nearby clutter |
   | `jitter` per channel | <= 5 deg | something moved during the capture, or the reflector is too weak |
   | `WARNING:` lines | none | fix the cause above and record again (the command exits with code 1 when there are warnings) |

   The `phase` / `gain` lines are the measured channel errors (relative to channel 0).
6. **Paste** the `rx_calibration: [...]` line into the config, replacing `rx_calibration: null`.
7. **Verify + left/right check.** Move the reflector to **+27 deg**: keep 10 m distance, move it
   toward the **RX4 end** of the board (positive angles are on the RX4 side, see the silkscreen):
   4.54 m sideways and 8.91 m forward. Record again (same settings) and rename the file to
   `bins/cal_27deg_001.bin`, then:

   ```powershell
   python scripts/radar.py calibrate --config configs/radar_capture_highway_overlook.yaml --input bins/cal_27deg_001.bin --range-m 10 --angle-deg 27
   ```

   Because the config now has `rx_calibration`, it also prints
   `Angle with config rx_calibration: 26.9 deg (error -0.10 deg)`.

   * about **+27 deg** (error within +-2 deg): calibration and angle sign are correct. Done.
   * about **-27 deg**: left/right is flipped: either RX4 is on the other side, or the board is mounted
     upside down. Fix the mounting (or note it) and repeat from step 2.
   * anything else: redo step 2-6 (bad first capture).

   Ignore the `rx_calibration` this second run prints: keep the 0 deg one.
8. Repeat 1-7 for the other configs you will use (`radar_capture_vehicle.yaml`,
   `radar_capture_vehicle_highway.yaml`), with their rows in the [1.5](01_setup.md#15-recording-a-capture-used-by-every-experiment) table.

## 3.3 Optional: Doppler sign check

Record ~10 s while someone carries the reflector **toward** the radar, then
`python scripts/radar.py process --config <config> --input <bin> --output-dir outputs/sign_check`.
The walker must show up with **negative** `radial_velocity_mps` in `capture_summary.json`
(convention: negative = approaching). If it is positive, flip `waveform.mixer_convention`
(`tx_times_conj_rx` <-> `rx_times_conj_tx`) in all configs. Experiment 2 also checks this.

Options of `calibrate`: `--search-m 1.0` (search window around `--range-m`), `--max-frames N`,
`--output result.json` (save everything as JSON), `-v` (more logging).
