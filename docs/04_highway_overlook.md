# 4. Experiment 2: highway overlook (static radar, 17th floor, Hwy 401)

**Goal.** Prove the whole real-data chain on easy targets before going in a car: many fast cars,
static radar (all ground clutter sits at 0 m/s). You check: decoding, distance, speed sign,
angle sign, calibration, multi-frame STAP.

Config: `configs/radar_capture_highway_overlook.yaml`. Key numbers (all explained in its header):

| | Value | Meaning |
|---|---|---|
| range | 40-290 m processed, bins 1.3 m | road is ~71-226 m slant range |
| speed limit | +-29.1 m/s (105 km/h) radial | faster radial speeds wrap around |
| speed resolution | 0.45 m/s | 128 chirps per frame |
| frames | 600 x 50 ms = 30 s, 295 MB | processed: first 64 by default |
| STAP | D = 4 x 128 = 512, training pooled over 9 frames, 9 look angles -60..60 deg | |

## 4.1 mmWave Studio settings

From the config's `studio:` section:

* Static config: **TX1 only**, RX1-RX4 on.
* Profile: start 77 GHz, slope **4.49 MHz/us** (enter 4.5; copy the value Studio reports into
  `waveform.chirp_slope_hz_per_s`), idle 4 us, ADC start 4 us, ramp end 29.5 us, **240 samples**,
  10000 ksps complex, RX gain 30 dB, HPF 175 / 350 kHz.
* Chirp 0: TX1. Frame: chirp 0-0, **128 loops**, period **50 ms**, **600 frames**, software trigger.
* DCA1000: LVDS 2 lanes, raw mode, Ethernet stream (IPs as in the config).

## 4.2 Steps

1. **Calibrate** with this config ([3](03_calibration.md)); `rx_calibration` must be filled.
2. **Mount**: board on a tripod at an open window/balcony (glass can block 77 GHz by 10-40 dB),
   RX1 -> RX4 parallel to the highway, facing it, tilted **down 25-30 deg**. Nothing may move.
3. **Check the config**:

   ```powershell
   python scripts/radar.py check --config configs/radar_capture_highway_overlook.yaml
   ```

   Every stage must say `[OK ]`. The note "no per-channel phase/gain calibration" means step 1 is missing.
4. **Record** ([1.5](01_setup.md#15-recording-a-capture-used-by-every-experiment)): Studio frame
   count 600; `capture_ms = 35000`, `expected_bytes = 294912000`, `capture_count = 3` (spares).
   Rename `stap_capture_001..003.bin` to `overlook_001..003.bin`.
5. **Quick look first** (range/Doppler/angle only, ~1 min). Make `configs/overlook_quick.yaml`:

   ```yaml
   base: radar_capture_highway_overlook.yaml
   stap:
     enabled: false
   ```

   ```powershell
   python scripts/radar.py process --config configs/overlook_quick.yaml --input bins/overlook_001.bin --output-dir outputs/overlook_001_quick
   ```

6. **Sanity checks** on the quick look (how: [7](07_reading_outputs.md)):
   * `capture_summary.json` -> `decode.bytes_dropped` = 0 and `out_of_range_values` = 0;
   * `range_profile.png`: returns between ~70 and ~230 m (the road), little beyond;
   * `range_doppler_mean.png`: a strong band at 0 m/s (barriers, poles, gantry, road) and car streaks
     away from 0 on both sides; cars getting **closer** have **negative** speed;
   * detections: `angle_deg` mostly within +-15..60 deg (cars right in front move across the beam
     and have ~0 radial speed: they hide in the 0 m/s band; that is expected).
7. **Full STAP**:

   ```powershell
   python scripts/radar.py process --config configs/radar_capture_highway_overlook.yaml --input bins/overlook_001.bin --output-dir outputs/overlook_001 --max-frames 12
   ```

   Cost per frame at D = 512 on a 4-core CPU: MVDR ~0.8 s, PC ~1.1 s, EFA ~0.3 s per range bin,
   ~154 test bins -> ~5 min per frame for all methods. So: start with `--max-frames 12`, use the GPU
   (`pip install -e ".[gpu]"`; `check` then prints `STAP backend: cupy`), or run only the cheap methods
   with an override file:

   ```yaml
   base: radar_capture_highway_overlook.yaml
   stap:
     methods: [nonadaptive, efa]
   ```

   Use `--max-frames` >= 9 so every frame gets the full 9-frame training window; the default when omitted is the config's
   `max_frames: 64`; `--max-frames 600` processes the whole 30 s.
8. **Read the STAP result** ([7.3](07_reading_outputs.md#73-the-stap-block)): per method,
   `persistent_detections` (same cell found in many frames = a car, not noise) and
   `integrated.detections` (power averaged over frames: valid here because the radar is static).

## 4.3 What "good" looks like

* Both traffic directions present, with opposite speed signs on the same side of boresight.
* Lanes appear as clusters of detections at fixed slant ranges, frame after frame.
* The 0 m/s band does not drift between frames (if it does, the mount is moving).
* Adaptive methods (mvdr/efa/pc) show fewer detections at ~0 m/s than `nonadaptive`.
* Speeds above 29.1 m/s radial appear wrapped (e.g. a real +31 m/s shows as about -27 m/s).
