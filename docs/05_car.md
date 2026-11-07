# 5. Experiment 3: car STAP (regular speed and highway)

**Goal.** The radar moves, so every piece of stationary ground has a radial speed that depends on
its angle: `v_r = -v_car * sin(angle)`. In the angle-Doppler plane the ground becomes a diagonal
stripe, the **clutter ridge**. A slow mover at the same speed as the ground in the beam is
invisible to plain Doppler processing; STAP uses all 4 channels x all chirps together to cancel the
ridge and keep the mover.

## 5.1 Pick the config

| | Regular speed | Highway |
|---|---|---|
| config | `configs/radar_capture_vehicle.yaml` | `configs/radar_capture_vehicle_highway.yaml` |
| drive at | ~30 km/h (max ~45) | 50-100 km/h |
| targets | pedestrians, cyclists, slow cars | other vehicles |
| slope / samples | 18 MHz/us / 512 | 11.0078 MHz/us / 240 |
| chirp period | 70 us -> speed limit +-13.9 m/s | 33.5 us -> +-29.1 m/s |
| range processed | 5-75 m (0.16 m bins) | 4-115 m (0.53 m bins) |
| frame | 16 chirps, 100 ms, 128 frames (~13 s, 17 MB) | 16 chirps, 20 ms, 3000 frames (60 s, 184 MB) |
| STAP | D = 64, look 0 deg, 1 frame of training | D = 64, looks -45..45 deg, forward-backward, 3 frames of training |
| processed by default | all frames | first 200 (4 s) |

The speed limit must cover the car's own speed or the ground wraps: that is why the highway config
uses a faster chirp. Oncoming traffic (closing at up to ~55 m/s) wraps in speed on both configs;
it is still detected, at a wrong speed.

## 5.2 Steps

1. **Calibrate** with the config you will drive with ([3](03_calibration.md)), e.g. in a parking lot.
2. **Mount**: on a roof pole (~3 m above ground), boresight **sideways** (perpendicular to travel),
   RX1 -> RX4 **along** the direction of travel, rigid.
3. **Set the speed** in the config before processing each drive (from GPS, in m/s = km/h / 3.6):

   ```yaml
   platform_velocity_mps: [22.2, 0.0, 0.0]   # 80 km/h
   ```

   Positive x = moving toward the RX4 end. If RX4 points to the back of the car, use a negative value.
   Adaptive STAP does not need this; it is used to draw the predicted ridge on the angle-Doppler plot
   and by `dpca`.
4. **mmWave Studio**: load the config's `studio:` values (TX1 only, slope, samples, 16 loops,
   frame period and count from the table above).
5. **Record** ([1.5](01_setup.md#15-recording-a-capture-used-by-every-experiment)): regular =
   128 frames, `capture_ms = 16000`, `expected_bytes = 16777216`; highway = 3000 frames,
   `capture_ms = 65000`, `expected_bytes = 184320000`. Start once you are at a steady speed on a
   straight road. Rename the file to `car_regular_001.bin` / `car_fast_001.bin` and note the GPS speed.
6. **Check and process** (start small: the first 20 frames):

   ```powershell
   python scripts/radar.py check --config configs/radar_capture_vehicle_highway.yaml
   python scripts/radar.py process --config configs/radar_capture_vehicle_highway.yaml --input bins/car_fast_001.bin --output-dir outputs/car_fast_001 --max-frames 20
   ```

   (Regular speed: same with `radar_capture_vehicle.yaml` and `car_regular_001.bin`.)
   Cost on a 4-core CPU at D = 64 (measured on a simulated capture with ~400 test bins, one look):
   MVDR ~30 s per frame, PC ~3 s, EFA ~1 s, nonadaptive ~0. Then process more frames (`--max-frames 200`, or the GPU).

## 5.3 What "good" looks like

* `angle_doppler_mean.png`: a clear diagonal ridge (ground), matching the predicted ridge line drawn
  from `platform_velocity_mps` if that speed is right. A horizontal line at 0 Doppler instead means the car was not moving.
* `stap_gmti_*.png`: `nonadaptive` is full of ridge residue; `mvdr`/`efa`/`pc` are dark except for
  movers.
* `persistent_detections`: real vehicles appear in several consecutive frames. On a moving
  platform things shift ~1 range bin per frame, so persistence is per cell and short; the
  `integrated` result smears and is less useful here than in Experiment 2.
* A mover is credible if it is found by several adaptive methods at the same range and speed.

## 5.4 Optional: DPCA

`dpca` (displaced phase centre) cancels the ground by subtracting chirps offset by exactly the
distance the car moves between them. It only works at one speed: on the highway config the default
`dpca_pulse_lag: 1` matches 105 km/h; set `dpca_pulse_lag: 2` for 52 km/h. Hold it on cruise control, set
`platform_velocity_mps` to the measured speed, and add `dpca` to `stap.methods` (override file with
`base:` as in [4.2](04_highway_overlook.md#42-steps)). Its `method_info` lists the blind speeds.
