# 7. Reading the outputs

**Conventions used everywhere:** range = straight-line (slant) distance in m; radial velocity in
m/s, **negative = getting closer**, positive = moving away; angle in deg from boresight,
**positive = toward the RX4 end** of the board; SNR in dB.

## 7.1 `process` output folder

| File | Look at it for |
|---|---|
| `capture_summary.json` | all numbers (below) |
| `adc_diagnostics.png` | histogram of raw ADC codes: should be a smooth bell well inside +-32768. Touching the edges = clipping (lower RX gain); very narrow = signal too weak (raise gain) |
| `range_profile.png` | average power vs distance: peaks are strong reflectors at their distance (corner reflector, walls, road) |
| `range_doppler_frame0.png`, `range_doppler_mean.png` | power vs distance and radial speed (first frame / average). Static things sit at 0 m/s; movers are away from 0 |
| `angle_doppler_frame0.png`, `angle_doppler_mean.png` | where the clutter is in angle vs speed. Static radar: a line at 0 speed across all angles. Moving radar: a diagonal ridge |
| `stap_gmti_frame0.png`, `stap_gmti_integrated.png` | STAP output per method (distance vs speed hypothesis) at the first look angle; bright spots that survive in mvdr/efa/pc = movers |

Plots exist only for stages that ran (`--no-plots` skips them all).

## 7.2 `capture_summary.json`, top to bottom

```text
input                               the .bin that was processed
stages                              what `check` reported (decode/range/doppler/angle/stap: available, missing, notes)
provenance                          where each waveform value came from (yaml or .cfg)
decode                              -> data health, check this first
adc_format                          the layout used to decode
data_shape_frames_rx_chirps_samples e.g. [4, 4, 16, 512] = 4 frames, 4 RX, 16 chirps, 512 samples
files                               the PNGs written
range                               range axis + strongest return
doppler                             speed axis + per-frame detections
stap                                STAP setup + per-method results ("unavailable" if not run)
```

**`decode`** (real example from a simulated capture):

```json
"decode": {"num_frames": 4, "bytes_total": 524288, "bytes_used": 524288, "bytes_dropped": 0,
           "out_of_range_values": 0, "full_scale_fraction": 0.0,
           "channel_mean": [[-0.05, 0.11], ...], "channel_rms": [118.6, 118.5, 118.6, 117.9], "warnings": []}
```

Good: `bytes_dropped` 0, `out_of_range_values` 0 (non-zero = wrong format), `full_scale_fraction`
~0 (non-zero = clipping), the 4 `channel_rms` similar (one very different = a dead/odd channel),
`warnings` empty.

**`range`**: `num_bins`, `bin_spacing_m` (distance per bin), `strongest_range_m` (the biggest
return: in a calibration capture this should be the reflector), `noise_floor_median_db`.

**`doppler`**: `velocity_resolution_mps`, `max_unambiguous_velocity_mps` (speeds beyond +- this wrap),
and `frames`: one entry per frame with up to 100 detections from the plain range-Doppler map:

```json
{"range_m": 36.17, "radial_velocity_mps": 3.66, "snr_estimate_db": 12.4, "angle_deg": -26.3}
```

`snr_estimate_db` = how far the cell is above its CFAR background. `angle_deg` is the 4-channel angle
estimate (only meaningful with `rx_calibration`).

## 7.3 The `stap` block

Setup:

| Key | Meaning |
|---|---|
| `look_angles_deg` | directions tested (a target is reported at the look where it is strongest) |
| `dimension` | D = channels x chirps (adaptive filter size) |
| `num_frames`, `num_test_bins`, `test_range_m` | how much was tested |
| `training` | `range_bins`, `guard_bins`, `frames_pooled`, `samples_per_covariance` (should be >= 2 x D for MVDR), `forward_backward`, `backend` (numpy/cupy), `precision`, `censor_db`, `mean_censored_bins_per_frame` |

Per method (`stap.methods.<name>`):

| Key | Meaning |
|---|---|
| `frames_processed` | frames that ran (fewer than `num_frames` = an error stopped it, see `error`) |
| `total_detections`, `mean_detections_per_frame` | raw counts (all detections, including false alarms) |
| `weight_seconds_per_frame` | compute time |
| `persistent_detections` | the 30 (look, range bin, speed) cells detected most often: `frames_detected`, `fraction_of_frames`, `mean_output_snr_db`. **Best evidence of a real target**: high fraction |
| `integrated` | detections on the output power averaged over all frames (good for a static radar; smeared on a moving one) |
| `frames` | per frame: `num_detections`, and up to 50 `detections` with `range_m`, `radial_velocity_mps`, `look_angle_deg`, `output_snr_db` |
| `method_info` | extra for non-adaptive methods, e.g. DPCA `blind_velocities_mps` |
| `error` | why a method failed (e.g. singular covariance) |

STAP speeds snap to the hypothesis grid: spacing = 2 x `max_unambiguous_velocity_mps` /
`num_doppler_hypotheses` (e.g. 0.86 m/s for 32 hypotheses at +-13.8 m/s). Range-Doppler detections
are interpolated and finer.

**Is it a real target?** Credible if: found in many frames (`fraction_of_frames`), found by more than
one adaptive method at the same range/speed, not at the clutter speed for its angle, and at a
plausible range (e.g. on a road lane). `nonadaptive` is the baseline: adaptive methods should show
fewer clutter detections, not more.

Methods in one line each: `nonadaptive` = fixed beam + Doppler filter (no learning);
`mvdr` = full adaptive filter, best but slowest; `efa` = adaptive over 3 neighbouring Doppler bins
(cheap, nearly as good); `pc` = adaptive using only the strongest clutter directions; `dpca` =
subtract time-shifted channels (one speed only).

## 7.4 `simulate` output folder

`results.json` plus `waveform_if.png` (raw IF signal of one chirp), `range_profile.png`,
`range_doppler.png` (with truth markers), `angle_doppler_clutter.png` (clutter ridge),
`eigenspectrum.png` (clutter rank: how many "directions" clutter occupies), `sinr_loss.png` (how close
each method is to optimum across speed), `gmti_outputs.png` (STAP output maps with truth),
`pd_pfa.png` (Monte Carlo detection probability vs false-alarm rate, if run),
`synthetic_cpi.npz` + metadata (the raw simulated data).

`results.json` keys: `truth` (every object's true range/speed/angle/SNR), `derived` (resolutions,
limits), `range_doppler_detection` (per-target detected yes/no and errors, unassociated count),
`stap` (setup, per-method detections and target hits, `sinr_analysis`: output SINR and improvement
factor `IF` vs optimum), `monte_carlo`, `timings`.

## 7.5 `calibrate` output

See [3.2 step 5](03_calibration.md#32-step-by-step). With `--output cal.json` everything is saved,
including `phase_offset_deg`, `gain_offset_db`, `phase_jitter_deg`, `angle_uncalibrated_deg`,
`rx_calibration`, `warnings`.

## 7.6 `e2e-overlook` output

`e2e_report.json`: `assumptions` (scene), `calibration` (measured vs injected channel errors, 27 deg
check), `scene` (vehicles, truth for frame 0), `stap_setup`, `results` (one score per stage, as in
the console table explained in [2.4](02_simulation.md#24-way-c-experiment-2-rehearsal)), `processing_seconds`.
