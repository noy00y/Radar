# 8. Code map: what is where, and the words used

## 8.1 Words

| Word | Meaning here |
|---|---|
| chirp | one frequency sweep (33.5 or 70 us); the ADC samples it -> one line of data per RX |
| frame / CPI | a block of consecutive chirps (16 or 128) processed together; one look in time |
| range bin | one distance cell after the range FFT |
| Doppler bin / hypothesis | one radial-speed cell (FFT) / one speed STAP tests for |
| channel | one receive antenna (4, single TX); `C` |
| snapshot | all data of one range bin in one frame: C channels x L chirps = a vector of length **D = C x L** |
| training bins | range bins near the test bin used to learn the clutter (guard bins next to it are skipped) |
| covariance R | D x D matrix learned from training snapshots: "what clutter looks like" |
| look angle | direction STAP points the beam |
| steering vector | what a target at (look angle, speed) would look like in a snapshot |
| loading | small number added to R's diagonal so it can be inverted safely |
| forward-backward | trick that doubles the training samples for a uniform array/pulse train |
| CFAR | detector that compares each cell to the average of its neighbours (constant false-alarm rate) |
| clutter ridge | stationary ground seen from a moving radar: speed depends on angle |

## 8.2 The two pipelines

```text
REAL / SYNTHETIC CAPTURE  (radar.py check | calibrate | process | synth-capture | e2e-overlook)
  capture YAML --load_capture_config--> CaptureConfig --assess_capture_config--> stages OK/missing
  .bin --adc_loader.load_capture_file--> cube (frames, chirps, rx, samples)
       --range_processing.range_fft--> Y (frames, rx, chirps, range bins)
       per frame: datacube.to_virtual_cube -> x rx_calibration -> doppler_processing.doppler_fft
                  -> detection.ca_cfar -> angle_processing.estimate_angle      => doppler.frames
       per frame: capture._capture_stap -> stap.run_gmti (CPU)  or  batched.run_adaptive_batched (GPU)
                  -> experiments.gmti_detect (CFAR along range)                 => stap.methods
       -> plotting.* + capture_summary.json

SIMULATION  (radar.py simulate)
  scene YAML --config.load_simulation_config--> SimulationConfig
       --simulator.simulate--> raw IF data (rx, chirps, samples) + truth
       --experiments.process_cpi / detect_range_doppler--> range-Doppler + CFAR, scored vs truth
       --experiments.run_stap_comparison--> every method, SINR analysis
       --experiments.monte_carlo--> P_D vs P_FA   (response_basis.py makes this fast)
       -> pipeline.py writes results.json + plots
```

## 8.3 Modules (`src/radar_stap/`)

| File | Contains | Touch it when |
|---|---|---|
| `cli.py` | all `radar.py` commands and their options | adding a command / option |
| `capture.py` | capture config dataclasses, `check` logic, `process_capture`, `_capture_stap`, `calibrate_capture` | changing the real-data chain |
| `config.py` | simulation config dataclasses, `base:` loading, validation; `StapConfig` (all `stap:` keys) | adding a config key |
| `adc_loader.py` | DCA1000 decoding (explicit layouts), `encode_cube` (inverse) | a new capture format |
| `mmwave_cfg.py` | reads/writes TI SDK `.cfg` files | using `waveform.mmwave_cfg_file` |
| `range_processing.py` | range FFT, windows, range axis/mask | |
| `doppler_processing.py` | Doppler FFT, range-Doppler map, TDM phase compensation | |
| `angle_processing.py` | steering vectors, angle estimation, look-direction vectors | |
| `datacube.py` | physical -> virtual channels, snapshot stacking (`x[l*C + c]`) | |
| `covariance.py` | sample covariance, loading, forward-backward averaging, diagnostics | |
| `stap.py` | `SpaceTimeModel`, methods (`NonAdaptive`, `FullRankMVDR`, `AdjacentBinPostDoppler` = EFA, `PrincipalComponents`, `DisplacedPhaseCentre`), `build_method`, `run_gmti`, training-bin selection | changing a STAP algorithm |
| `batched.py`, `backend.py` | the same MVDR/EFA/PC for many range bins at once, NumPy or CuPy (GPU) | GPU work |
| `detection.py` | N-D CA-CFAR, clustering, peak refinement | detector changes |
| `simulator.py` | exact-geometry multichannel scene simulator | new scene physics |
| `synthetic_capture.py` | simulated scene -> `.bin` + `capture.yaml` + `truth.json` | |
| `experiments.py`, `response_basis.py`, `metrics.py`, `pipeline.py` | simulation scoring, Monte Carlo, SINR, results.json | |
| `plotting.py` | every PNG | |
| `constants.py`, `waveforms.py` | c, sign conventions, chirp/IF formulas | |

Scripts: `scripts/radar.py` (entry point), `scripts/capture_v2_verified_use.lua` (Studio), `scripts/e2e_highway_overlook.py`
(the `e2e-overlook` scene; edit its constants for real geometry), `scripts/benchmark_stap.py` (`benchmark`).

## 8.4 How STAP is computed for one range bin (what `run_gmti` does)

1. Take the test snapshot `x` (length D) at the test range bin.
2. Take training snapshots from `num_training` bins around it (skipping `num_guard` bins on each side,
   skipping bins dropped by `training_censor_db`), from `training_frames` neighbouring frames.
3. Estimate `R` = average of `x_k x_k^H` (+ forward-backward if on), add loading.
4. For each look angle and each of `num_doppler_hypotheses` speeds: steering vector `s`;
   weights `w = R^-1 s / (s^H R^-1 s)` (MVDR; EFA/PC are cheaper variants); output `y = w^H x`.
5. After all test bins: CFAR along range on `|y|^2` for every (look, speed) -> detections.

Cost grows with D^3 per range bin (MVDR), which is why D = 512 (overlook) is slow on a CPU and
the GPU path batches many range bins.

## 8.5 Tests (`tests/`)

`python -m pytest -q` runs all of them (~35 s). They cover decoding (known answers), the simulator
physics, each STAP method, CPU vs GPU/batched agreement, config validation, the TX policy, and the
command-line flows in these docs (`process`, `simulate`, `synth-capture` + multi-file `process`).
