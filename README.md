# radar-stap — FMCW radar, GMTI and STAP research pipeline

A reproducible research codebase for **FMCW radar simulation, coherent multichannel clutter
modelling, space-time adaptive processing (STAP), ground-moving-target indication (GMTI) and
computational evaluation**, built around the TI **IWR1843BOOST** radar and **DCA1000EVM** raw-ADC
capture card.

It is designed to be useful *today without the hardware* and to accept genuine DCA1000 captures
once their configuration and binary layout have been verified:

| Today (no hardware) | When the board is connected |
|---|---|
| Physics-consistent multichannel FMCW simulator (exact geometry, coherent clutter ridge) | Same processing chain on decoded captures |
| Range/Doppler/angle processing, CA-CFAR, STAP baselines, Monte Carlo P_D/P_FA | Explicit, verified DCA1000 decoding; mmWave SDK `.cfg` parsing |
| Synthetic DCA1000-style captures (`.bin` + SDK `.cfg` + capture YAML) through the *hardware path* | Stage gating: only stages supported by verified metadata run |

Primary research direction: computationally efficient STAP for GMTI, evaluated against a
conventional full-rank baseline on identical data, training rules and hypotheses.

> **Status.** All software validation below was run on synthetic data. Nothing in this repository
> has been validated against a physical IWR1843BOOST/DCA1000EVM capture. The decoder supports
> explicitly selectable layouts; which one your capture uses must be verified (checklist below).

---

## Contents
1. [Quick start](#1-quick-start)
2. [Repository structure](#2-repository-structure)
3. [Signal and array conventions](#3-signal-and-array-conventions)
4. [Signal model and simulator](#4-signal-model-and-simulator)
5. [Parameters and derived quantities](#5-parameters-and-derived-quantities)
6. [Conventional processing](#6-conventional-processing)
7. [STAP formulation](#7-stap-formulation)
8. [Detection (CFAR)](#8-detection-cfar)
9. [Evaluation metrics](#9-evaluation-metrics)
10. [Results of the default scenario](#10-results-of-the-default-scenario)
11. [Hardware path: DCA1000 decoding](#11-hardware-path-dca1000-decoding)
12. [Hardware validation checklist](#12-hardware-validation-checklist-iwr1843boost--dca1000evm)
13. [Verified facts vs configurable vs unknown](#13-verified-facts-vs-configurable-vs-unknown)
14. [GPU and FPGA readiness](#14-gpu-and-fpga-readiness)
15. [Reproducibility](#15-reproducibility)
16. [Known limitations](#16-known-limitations)
17. [Troubleshooting](#17-troubleshooting)
18. [Glossary](#18-glossary)

---

## 1. Quick start

```bash
pip install -e ".[dev]"                 # numpy, scipy, matplotlib, pyyaml (+ pytest, notebook tools)
pytest                                  # ~30 s on a laptop CPU, no GPU or hardware needed

# Simulation workflow (~50 s, includes a 40-trial Monte Carlo)
python scripts/simulate_scene.py --config configs/simulation.yaml --output-dir outputs/simulation

# Other scenarios
python scripts/simulate_scene.py --config configs/scenarios/stationary_pole.yaml --output-dir outputs/pole
python scripts/simulate_scene.py --config configs/scenarios/vehicle.yaml         --output-dir outputs/vehicle
python scripts/simulate_scene.py --config configs/scenarios/iwr1843_tdm.yaml     --output-dir outputs/tdm

# Hardware path without hardware: synthetic capture -> decode -> process
python scripts/make_synthetic_capture.py --output-dir outputs/synthetic_capture --frames 2
python scripts/process_capture.py --config outputs/synthetic_capture/capture.yaml \
    --input outputs/synthetic_capture/adc_data.bin --output-dir outputs/capture_synthetic

# Real capture (after filling and verifying configs/radar_capture.yaml)
python scripts/process_capture.py --config configs/radar_capture.yaml --check-only
python scripts/process_capture.py --config configs/radar_capture.yaml --input PATH_TO_CAPTURE --output-dir outputs/capture

# Compute benchmark (CPU; add --backend cupy on the RTX 3060 if CuPy is installed)
python scripts/benchmark_stap.py --channels 4 8 --pulses 8 16 32
```

The scripts set `OPENBLAS_NUM_THREADS=1` unless you set it yourself (see Troubleshooting: small
STAP matrices are ~30x *slower* when NumPy's and SciPy's OpenBLAS thread pools contend).

`outputs/simulation/` then contains `results.json` (machine-readable everything), the raw synthetic
CPI (`synthetic_cpi.npz` + metadata JSON), and figures: `waveform_if.png`, `range_profile.png`,
`range_doppler.png`, `angle_doppler_clutter.png`, `eigenspectrum.png`, `sinr_loss.png`,
`gmti_outputs.png`, `pd_pfa.png`.

## 2. Repository structure

```
radar-stap/
├── configs/
│   ├── simulation.yaml            # complete synthetic scenario (default)
│   ├── radar_capture.yaml         # hardware capture template: unknowns are null
│   └── scenarios/                 # stationary_pole, vehicle, iwr1843_tdm (inherit via `base:`)
├── src/radar_stap/
│   ├── constants.py               # c, sign/mixer/stacking conventions
│   ├── config.py                  # typed config, strict YAML loading, validation, derived quantities
│   ├── waveforms.py               # chirp, dechirp, IF phase, Doppler conventions
│   ├── simulator.py               # exact-geometry multichannel scene simulator
│   ├── datacube.py                # physical -> virtual (TDM) channels, snapshot stacking
│   ├── range_processing.py        # windows, range FFT, axes, coupling
│   ├── doppler_processing.py      # Doppler FFT, RD maps, TDM slot compensation
│   ├── angle_processing.py        # steering vectors, beamforming, angle estimation (ULA/NULA)
│   ├── covariance.py              # sample covariance, loading, shrinkage, diagnostics, RMB, Brennan
│   ├── stap.py                    # space-time model, MVDR, EFA, PC, non-adaptive, GMTI runner
│   ├── detection.py               # N-D CA-CFAR, peak grouping
│   ├── metrics.py                 # SINR, association, Wilson intervals, timing/memory
│   ├── response_basis.py          # per-scatterer responses: clairvoyant covariance, fast Monte Carlo
│   ├── experiments.py             # processing/detection/STAP comparison/Monte Carlo orchestration
│   ├── pipeline.py                # simulation workflow + report
│   ├── adc_loader.py              # DCA1000 decoding (explicit formats), UDP packet reassembly
│   ├── mmwave_cfg.py              # TI mmWave SDK .cfg parser/writer
│   ├── capture.py                 # capture config, completeness/stage gating, capture processing
│   ├── synthetic_capture.py       # synthetic DCA1000-style capture generator
│   ├── backend.py                 # optional CuPy backend for batched MVDR benchmarking
│   ├── plotting.py, cli.py
├── scripts/                       # simulate_scene, process_capture, make_synthetic_capture, benchmark_stap
├── tests/                         # 119 tests (unit, physics, hardware-path, integration, notebook)
├── notebooks/exploration.ipynb    # explanatory walk-through using the package
├── pyproject.toml
└── README.md
```

## 3. Signal and array conventions

All conventions are defined once (`constants.py`) and verified by tests.

| Item | Convention |
|---|---|
| Raw / range-compressed layout | `X[rx, chirp, sample]`, chirps in transmission order |
| Mixer | `tx_times_conj_rx`: IF = s_tx · conj(s_rx). `rx_times_conj_tx` supported (data conjugated on ingestion) |
| Radial velocity | `v_r = dR/dt`, **positive = receding** |
| Beat frequency | `f_b = S·τ + 2 v_r/λ` (= `S τ − f_d` with classical Doppler `f_d = −2 v_r/λ`) |
| Slow-time frequency | `ν = +2 v_r / λ` (phase advance per chirp `2π ν T`) |
| Wavelength | at the centre of the *sampled* sweep: `f_c + S (t_adc + N/(2 f_s))` |
| Body frame | +x array axis, +y boresight, +z up; platform yaw about z, depression about x |
| Angle | cone-complement angle θ from the array normal, `sin θ = u_x` (body frame), positive toward +x |
| Spatial steering | `a_c(u) = exp(−j 2π p_c·u / λ)`, `p_c = p_tx + p_rx` (two-way phase centre) |
| Virtual channel (TDM) | `c = p·M + m` (TX slot p of `tx_schedule`, RX m); sampled at `t = (l P + p) T_c` |
| Space-time stacking | `x[l·C + c] = V[c, l, r]` (pulse-major, channel fastest) ⇒ `s = kron(b(ν), a(u))` for single TX |
| FFT normalisation | un-normalised; noise per bin `σ² Σw²`, on-bin tone peak `|A| Σw` |

The mixer convention of real hardware must be **verified** (an approaching reflector must give
negative `v_r`); the IF sign of a given TI capture chain is not assumed.

## 4. Signal model and simulator

**Physical evolution of the signal.** During one chirp the transmitter frequency rises linearly,
`f(t) = f_c + S t`. An echo from range R arrives τ = 2R/c later, so at any instant the received
frequency lags the transmitted one by `S τ`. Mixing the two (dechirping) leaves a sinusoid at that
difference — the beat tone — whose frequency is proportional to range; a range FFT over the ADC
samples of one chirp therefore turns time into range. Over successive chirps the round-trip path
changes by `2 v_r T_c`, so the beat tone's phase advances by `2π (2 v_r/λ) T_c` per chirp; an FFT
across chirps resolves radial velocity. Across antennas, the path differs by `p·u`, giving a
phase gradient that encodes angle. On a moving platform each stationary ground patch also has a
radial velocity `−v_p·u`, so clutter Doppler is tied to angle: clutter energy lies on a ridge in
angle–Doppler space. A slow ground mover can share its Doppler with clutter from another angle;
separating them requires joint space–time filtering (STAP).

**Model.** Transmit phase `φ(t) = 2π(f_c t + S t²/2)`; echo `A·s_tx(t − τ(t))`; IF phase
`2π(f_c τ + S t τ − S τ²/2)` (closed form checked against explicit mixing in the tests). For every
scatterer s, RX m and chirp n the simulator computes the exact bistatic path
`L = |p_s − p_tx(n)| + |p_s − p_rx,m|` at the chirp's ramp start (positions moving with constant
velocities) and its rate, and uses `τ(t_f) = (L + L̇ t_f)/c` inside the chirp. Hence:

* fast-time and slow-time phases come from one delay model (range–Doppler coupling included);
* **range migration** is modelled exactly (path recomputed per chirp); the report lists the
  largest migration in bins (default scene: 0.07 bins over the CPI, i.e. negligible);
* array phase uses the configured element positions, no far-field approximation *in the simulator*
  (the processor's far-field steering agrees with the simulated target snapshot to correlation
  > 0.995 — tested, including TDM);
* **clutter** is thousands of stationary point scatterers on the ground, each with *one* complex
  reflectivity shared by all channels and chirps (coherent space-time structure), Rayleigh
  amplitudes, power ∝ R^−n × two-way cos^p element pattern, normalised to a total clutter-to-noise
  ratio. Platform motion produces the clutter ridge; a stationary platform puts clutter at zero
  Doppler.

Geometries supported by configuration: side-looking airborne/drone (default), stationary
pole-mounted (`scenarios/stationary_pole.yaml`), forward-looking car-mounted with roadside clutter
(`scenarios/vehicle.yaml`), TDM-MIMO virtual arrays (`scenarios/iwr1843_tdm.yaml`).

Approximations (explicit): point scatterers; constant velocities; linear delay within a chirp
(curvature error ≪ λ for these geometries); first order in v/c; no intrinsic clutter motion,
multipath, occlusion/shadowing, polarisation, IF filter/HPF, phase noise, ADC nonlinearity or
channel mismatch; isotropic scatterers; real-sampling mode keeps only the in-phase channel. The
simulation does not claim to reproduce real-world clutter statistics.

## 5. Parameters and derived quantities

`configs/simulation.yaml` defines carrier, slope, idle/ADC-start/ramp-end times, sample rate and
count, chirps per CPI, sampling (complex/real), mixer, noise power, array positions (units of λ/2
or metres), TX positions and TDM schedule, platform position/velocity/orientation, clutter, targets
(position or range+angle+height; velocity vector or radial velocity; per-sample SNR), stationary
reflectors, FFT sizes/windows, CFAR, STAP and Monte Carlo settings, all validated on load
(unknown keys rejected; all errors reported together).

Derived quantities (`config.derive_quantities`), default scenario values in brackets:

| Quantity | Formula | Assumption | Default |
|---|---|---|---|
| Chirp period T_c | idle + ramp_end | TI timing model | 70 µs |
| Sampled bandwidth B | S N / f_s | only sampled sweep contributes | 768 MHz |
| Range resolution | c / 2B | rectangular window; Hann ≈ 1.44× | 0.195 m |
| Range-bin spacing | c f_s / (2 S N_fft) | zero padding shrinks *spacing* only | 0.195 m |
| IF-limited range | f_max c / 2S, f_max = f_s (complex) or f_s/2 (real) | IF filter ignored | 99.9 m |
| ADC-start limit | c t_adc / 2 | echo must arrive before sampling | 899 m |
| Slow-time interval | P T_c (P TDM slots) | per virtual channel | 70 µs |
| Velocity resolution | λ / (2 L T_slow) | L slow-time samples | 1.73 m/s |
| Unambiguous velocity | ± λ / (4 T_slow) | Nyquist in slow time | ±13.8 m/s |
| Range–Doppler coupling | f_c / S per m/s | f_b includes 2v/λ | 5.2 mm per m/s |
| Data per CPI | 2 B × (2 if complex) × N × chirps × RX | int16 words | 128 KiB |

Validation catches e.g. an ADC window ending after the ramp, scenes beyond the IF or ADC-start
limits, TDM schedules that don't divide the CPI, and full-rank STAP with fewer training snapshots
than the space-time dimension.

## 6. Conventional processing

* **Range**: per-chirp DC removal (optional), window, FFT (optional zero padding). Complex data use
  the positive beat band `[0, f_s)` by default (`two_sided` available); real data use `rfft`.
  Detected ranges are corrected for range–Doppler coupling using the estimated velocity.
* **Virtual array**: `datacube.to_virtual_cube` regroups chirps by TX slot. Virtual channels of a
  TDM schedule are *not* simultaneous: the slot offset `p T_c` is modelled exactly in the STAP
  steering vector and removed (`tdm_phase_compensation`) before angle estimation. A test shows
  that skipping this compensation biases angle by several degrees.
* **Doppler**: window, FFT (+zero padding), fftshift per virtual channel; noncoherent sum over
  channels for the RD map.
* **Angle**: Bartlett beamformer with explicit element positions (uniform or non-uniform), peak
  with parabolic refinement.

## 7. STAP formulation

**Processing domain** (one, used consistently): range-compressed virtual-channel data. For range
bin r the snapshot is `x_r ∈ C^D`, `D = C · L` (C virtual channels, L slow-time samples).

**Steering vector**: `s[l C + c] = exp(−j2π p_c·u/λ) · exp(j2π ν (l T_slow + t_c))`.

**Sample covariance**: `R̂ = (1/L_t) Σ x_l x_lᴴ` over training bins chosen symmetrically around
the test bin outside ±`num_guard` bins (falling back to one side at swath edges); in controlled
simulations known target bins can be excluded (`exclude_known_targets`, an *oracle* aid, reported
as such). Hermitian symmetrisation, diagnostics (eigenvalues, condition number before/after loading,
effective rank) and clear errors for insufficient support are built in.

**Sample support.** Reed–Mallett–Brennan: the expected SINR loss of sample-matrix MVDR is
`(L_t + 2 − D)/(L_t + 1)` — about 3 dB at `L_t = 2D`. The default (4 × 16 = 64) uses 160 bins
(2.5 D, expected loss −2.2 dB). The TDM scenario (8 × 16 = 128) would need ≥ 256 homogeneous bins
for full-rank MVDR, which the swath cannot provide; this is exactly the motivation for reduced-
dimension STAP, and the config validator refuses full-rank MVDR with L_t < D.

**Filters** (all distortionless, `wᴴs = 1`):

| Method | Weight | Dimension solved | Notes |
|---|---|---|---|
| `nonadaptive` | `(t ⊙ s)/Σt` | — | tapered beamformer + Doppler filter (baseline) |
| `mvdr` | `R̂_λ⁻¹ s / (sᴴ R̂_λ⁻¹ s)` | D | Cholesky solve, never an explicit inverse; diagonal loading |
| `efa` | `T w̃`, `w̃ = R̃⁻¹ s̃ / (s̃ᴴ R̃⁻¹ s̃)`, `T = kron(F_j, I_C)` | Q·C (=12) | adjacent-bin post-Doppler; Q Hann-tapered Doppler filters per hypothesis |
| `pc` | `(I − U_k U_kᴴ) s`, normalised | eig of D | principal-components eigencanceler, rank from eigenvalue threshold |

Diagonal loading modes: `noise_relative` (value × per-element noise variance; default for
simulation where it is known), `noise_eig_relative` (data-driven; default for captures),
`trace_relative`, `absolute`. Loading makes R̂ invertible; it does not make it accurate.
Shrinkage (`covariance.shrinkage_covariance`) exists as a separately labelled option.

**Research interface.** A method implements `weights(training, S, nu_grid) → WeightResult(W, info)`
and `complexity(D, L, J)`, and is registered in `stap.build_method`. `experiments.run_stap_comparison`
runs every configured method on the same snapshots, training selection and Doppler hypotheses, and
records weight/apply wall time, peak traced memory per weight computation, complexity estimate
(complex MACs), detections, output SINR at each in-beam target, clutter attenuation, target gain,
and agreement with MVDR (correlation of log output maps). Candidate methods for future work
(beamspace, JDL, iterative solvers, structured covariance, low precision) slot into the same
interface; they are **not** implemented.

A test checks the reduced-dimension machinery: with a full-rank transform EFA reproduces full MVDR
to 1e-8.

## 8. Detection (CFAR)

N-D cell-averaging CFAR (`detection.ca_cfar`) with per-axis guard/training cells and per-axis edge
handling (`wrap` for Doppler; `shrink` = use only in-map cells and recompute α from the actual
count). Threshold `α = N(P_FA^(−1/N) − 1)` is exact for i.i.d. exponential (square-law complex
Gaussian) cells; a test confirms the empirical P_FA falls in the 3σ Wilson interval of the design
value on exponential noise.

When it is only approximate: windowed/zero-padded bins are correlated; the noncoherent channel sum
is Gamma-distributed (CA becomes conservative); clutter is non-homogeneous (ridge, edges); targets in
the training window mask each other; STAP weights differ from bin to bin, so the output background
varies along range (the Monte Carlo below measures P_FA ≈ 1.7× design for MVDR).

The default RD detector is **range-only** CA-CFAR per Doppler column: on a moving platform Doppler
training cells next to the clutter band are not homogeneous and a 2-D window masks targets just
outside the band. Detections are grouped by local maxima of the CFAR-normalised statistic (not by
connected components, which let a large clutter region swallow a nearby target), then refined on
raw power before parabolic interpolation. STAP outputs use 1-D CA-CFAR along range per Doppler
hypothesis.

## 9. Evaluation metrics

* **Association rule**: greedy nearest-neighbour inside gates — |range bin − expected bin
  (including coupling)| ≤ `range_gate_bins` and circular |Doppler bin − expected| ≤
  `doppler_gate_bins` (STAP: ±1.5 hypotheses); one-to-one; unmatched detections are false alarms.
* **Estimation errors**: range, radial velocity and angle of associated detections vs ground truth.
* **SINR** (range-compressed space-time domain, one bin): target response `g_t` is the simulated
  target's own noise-free snapshot; interference covariance `R_i` is the *exact* clutter +
  discretes covariance from the scatterer model (`response_basis`) plus noise `σ²Σw² I`.
  Input SINR = (‖g_t‖²/D)/(tr R_i/D) per element-pulse; output SINR = `|wᴴg_t|² / wᴴR_i w`;
  optimum = `g_tᴴ R_i⁻¹ g_t`; improvement factor = out/in; clutter attenuation = (tr R_c/D)/(wᴴR_c w);
  SINR loss vs Doppler normalised to the noise-only matched filter `D/σ²_bin`.
* **P_D / P_FA**: Monte Carlo with fixed geometry and look direction; every trial redraws clutter
  reflectivities, discrete/target phases and noise, and runs H0 (target absent) and H1 (present) on
  the same interference. P_D = fraction of H1 trials with a detection inside the gate; P_FA =
  declared H0 cells / tested cells (per cell, comparable to the design value). 95 % Wilson intervals
  and trial/cell counts are reported. Trials are fast because clutter snapshots are synthesised as
  `Σ c_s g_s` from precomputed per-scatterer responses (validated against the full simulator to
  < 0.3 % error; the ensemble covariance converges to the clairvoyant one — both tested).
* **Runtime / memory**: wall-clock for weight computation and application (timed *without*
  tracemalloc, which distorts timing), peak traced allocation per weight computation, and an
  analytic complex-MAC estimate.

## 10. Results of the default scenario

`python scripts/simulate_scene.py` (seed 20261012; 2-core cloud CPU; numbers will differ slightly
on other BLAS builds). Platform 8 m/s side-looking, 4 RX, 16 chirps, D = 64, 160 training bins.

Conventional chain: the exo-clutter off-beam mover (+12 m/s) is detected with errors
ΔR = +0.010 m, Δv = −0.09 m/s, Δθ = +0.05°. The two in-beam movers lie inside the clutter band
(|v_r| < 8 m/s) and are not detectable in the noncoherent RD map — as expected.

SINR at the in-beam targets (clairvoyant evaluation; optimum is the clairvoyant MVDR bound):

| Target | Input SINR | Optimum | non-adaptive | MVDR | EFA | PC |
|---|---|---|---|---|---|---|
| mover_fast (+3 m/s, Doppler shared with sidelobe clutter) | −16.4 dB | 24.1 dB | 6.4 dB | 23.0 dB | 21.3 dB | 22.8 dB |
| mover_slow (−0.9 m/s, ~1.1 cells from mainbeam clutter) | −10.1 dB | 23.7 dB | −2.3 dB | 22.8 dB | 20.8 dB | 22.5 dB |

Clutter rank: Brennan's rule predicts 13; the clairvoyant eigenspectrum has 13 eigenvalues more
than 10 dB above the noise floor (`eigenspectrum.png`). MVDR loses ~1 dB vs optimum (RMB predicts
2.2 dB on average without loading). EFA trades ~2 dB of SINR (Hann-tapered 16-pulse Doppler filters
widen the mainbeam-clutter notch) for a 12-dimensional problem needing ~24 instead of ~128 training
bins. On this small D its wall time is *not* lower than full-rank MVDR (it solves one 12×12 system
per Doppler hypothesis); the compute advantage appears at larger D — see `benchmark_stap.py`. This
is the kind of trade-off the research interface is meant to expose.

Monte Carlo (40 trials, mover_fast at −30 dB raw SNR, design per-cell P_FA = 1e−3):

| Method | P_D [95 % CI] | P_FA [95 % CI] |
|---|---|---|
| non-adaptive | 0.00 [0.00, 0.09] | 7.1e−3 [6.4e−3, 7.8e−3] |
| MVDR | 0.88 [0.74, 0.95] | 1.7e−3 [1.3e−3, 2.0e−3] |
| EFA | 0.57 [0.42, 0.71] | 2.2e−3 [1.9e−3, 2.7e−3] |
| PC | 0.80 [0.65, 0.90] | 1.7e−3 [1.4e−3, 2.1e−3] |

Interpretation: adaptive processing is clearly better here, but this is one geometry and seed; the
excess P_FA of the adaptive filters comes from bin-to-bin weight variation (CA-CFAR assumes a
homogeneous background), and of the non-adaptive filter from residual clutter heterogeneity. A
strong off-beam target excluded from training (oracle exclusion) leaks through sidelobes of the
adaptive filters at its range (`gmti_outputs.png`, 30 m) — training-data contamination vs exclusion
is a real design question, not solved here. Mainbeam clutter always passes at its own Doppler
(it is in the look direction), producing the vertical stripe in the GMTI maps.

Runtime: simulation 7.7 s (4000 scatterers), STAP comparison 3.8 s for 415 bins × 32 hypotheses ×
4 methods, Monte Carlo 34 s.

## 11. Hardware path: DCA1000 decoding

`configs/radar_capture.yaml` holds only verified or user-supplied settings; unknowns stay `null`.
`process_capture.py` reports completeness per stage and refuses to decode until the binary format is
complete **and** `capture_format_verified: true` (`--allow-unverified-format` exists for
exploration and is recorded in the report).

| Stage | Requires |
|---|---|
| decode | complete `adc_format`, `capture_format_verified` |
| range | + slope, sample rate (mixer unknown → two-sided spectrum) |
| doppler | + carrier, idle/ADC-start/ramp-end times, TX schedule, mixer convention |
| angle | + antenna positions with `antenna.verified: true` (calibration optional but flagged) |
| stap | + `stap.enabled` (loading must be data-driven: no known noise level) |

`adc_format` fields: `data_type` (complex/real), `adc_bits` (12/14/16), `sign_handling`
(`int16_twos_complement` | `sign_extend_adc_bits`), `byte_order`, `iq_order` (I_first/Q_first),
`complex_packing` (`iq_interleaved` A0 B0 A1 B1… | `two_sample_blocks` A0 A1 B0 B1…),
`channel_layout` (`rx_planar_per_chirp` | `rx_interleaved_per_sample`), `num_rx`,
`num_adc_samples`, `chirps_per_frame`, optional `num_frames`. Chirps are consecutive in transmission
order. File size must divide exactly into frames (or `allow_partial_final_frame`). The decoder
reports out-of-range values / unexpected upper bits (a strong hint of wrong packing), full-scale
fraction (clipping), per-channel DC and RMS.

Waveform fields can come from the mmWave SDK `.cfg` used for the capture
(`waveform.mmwave_cfg_file`): `profileCfg`, `chirpCfg`, `frameCfg`, `channelCfg`, `adcCfg` are
parsed (argument order of mmWave SDK 3.x — verify for your SDK); values given in both places must
agree. `adcbufCfg` is reported only as a hint: it describes the on-chip buffer, not necessarily the
file written by the DCA1000/mmWave Studio. Provenance of every filled field is reported.

Packet-level data: `adc_loader.reassemble_udp_packets` reorders raw-mode packets by sequence number
and zero-fills gaps using the byte-count field (4-byte sequence + 6-byte byte count header, little
endian), reporting losses and duplicates. Verify the header against your DCA1000 firmware/tool.

**Adding a format**: add the value to the relevant `SUPPORTED_*` tuple in `adc_loader.py`, implement
it in `_words_to_stream` / `_stream_to_chirp` and `encode_cube`, and add a known-answer vector test.

`make_synthetic_capture.py` writes a quantised capture in any supported layout plus a matching `.cfg`
and capture YAML (format verified *by construction*); tests run it through the full hardware path for
two different layouts and check range/velocity/angle against truth, and show that a wrong I/Q order
moves the target to a mirrored range.

## 12. Hardware validation checklist (IWR1843BOOST + DCA1000EVM)

Do these in order; each step unlocks a stage. Record evidence (files, photos, distances).

1. **Capture chain**: note mmWave Studio / DCA1000 CLI / SDK versions, LVDS lane count, data
   format setting (e.g. complex 1x vs real), and *which* file you process (raw packet log vs
   post-processed `.bin`). Keep the exact `.cfg`/JSON used.
2. **Waveform**: fill `waveform.mmwave_cfg_file` (or the fields). Run `--check-only`; resolve
   conflicts.
3. **Static noise capture** (antennas covered / pointing at absorber): decode with the candidate
   format and `--allow-unverified-format`. Expect zero out-of-range values, near-zero DC (after
   `remove_dc`), similar RMS on all RX, no full-scale values, noise-like spectrum.
4. **Corner reflector at a measured range** (e.g. 3–5 m, boresight): the range peak must appear at
   the measured distance (± a few cm after any fixed range bias). A peak at a mirrored/unrelated
   range indicates wrong I/Q order or packing; garbage across all ranges indicates wrong
   interleaving or sample count.
5. **Channel order**: move the reflector off-boresight to a surveyed angle; with candidate antenna
   positions the angle estimate must have the right sign and magnitude. Swapping RX order flips or
   scrambles it.
6. **Mixer / velocity sign**: walk the reflector toward the radar: this package reports approaching
   motion as **negative** `v_r`. If positive, set `mixer_convention: rx_times_conj_tx`.
7. **TDM schedule**: confirm `tx_schedule` from `chirpCfg`; check that angle with the 8-element
   virtual array is consistent with the 4-RX estimate.
8. **Calibration**: measure per-virtual-channel phase/gain on the boresight reflector (e.g. TI's
   range-bias/RX-phase compensation procedure) and enter `rx_calibration`.
9. Set `capture_format_verified: true` and `antenna.verified: true`; only then enable STAP.

## 13. Verified facts vs configurable vs unknown

| Category | Items |
|---|---|
| Verified *in software* (tests) | sign conventions, beat frequency, Doppler phase progression, range–Doppler coupling, steering-vector agreement with exact geometry, TDM slot effect, decoder known-answer vectors for every supported layout, CFAR P_FA on exponential noise, MVDR/EFA/PC algebra, clutter rank vs Brennan, clairvoyant covariance vs ensemble |
| Hardware facts used (from TI product descriptions; not verified here) | IWR1843: 76–81 GHz FMCW, 3 TX / 4 RX; DCA1000EVM streams raw LVDS ADC data to a PC over Ethernet |
| Configurable (simulation) | everything in `simulation.yaml`; the IWR1843-like waveform and the antenna layout in `scenarios/iwr1843_tdm.yaml` are **nominal, for simulation only** |
| Unknown until verified on your setup | file layout (I/Q order, packing, interleaving, sign handling), mixer/velocity sign, actual element phase centres, channel calibration, IF filter response, ADC bit usage, packet loss behaviour, range bias |

## 14. GPU and FPGA readiness

**GPU (RTX 3060)**: numerical kernels are pure functions of arrays, separate from orchestration and
plotting. `backend.batched_mvdr` is a backend-agnostic batched Cholesky-solve MVDR (NumPy or CuPy);
`scripts/benchmark_stap.py --backend cupy` times it and reports relative error vs the CPU reference
(1e−12 on NumPy). CuPy is optional (`pip install -e ".[gpu]"`); not tested here because no CUDA
device was available.

**FPGA (Artix-7 XC7A100T)** — nothing here is FPGA code. Candidate partition and constraints for a
later implementation (order-of-magnitude estimates for the default D = 64 configuration; device
resources per the Xilinx 7-series tables — verify):

| Block | Dataflow / arithmetic | Notes |
|---|---|---|
| Range FFT (512-pt, per chirp, per RX) | streaming fixed-point (16-bit in, ~24-bit out), window ROM | standard FFT IP; per-chirp latency ≪ T_c |
| Corner turn | write range bins, read slow-time | one CPI = 4 × 16 × 512 × 32-bit complex ≈ 128 KiB (BRAM-feasible; XC7A100T ≈ 600 KB BRAM) |
| Doppler FFT / RD map | 16- or 32-pt FFTs, magnitude, CA-CFAR via running sums | CA-CFAR = sliding-window adders |
| Covariance accumulation | L_t·D² ≈ 0.65 M complex MAC per test bin | sliding-window update (add new bin, subtract old) cuts this by ~L_t/2 |
| Solve | Cholesky + triangular solves (D³/3 ≈ 87 k CMAC) or QR/inverse-QR-RLS systolic array | dynamic range is the issue: fixed-point covariance needs care or block floating point; reduced-dimension (EFA, 12×12) is far friendlier |
| Weight application | D CMAC per hypothesis per bin | trivially parallel |

Total MVDR load ≈ 0.36 G complex MAC per CPI here; with ~240 DSP slices (~4 per complex MAC) at
~200 MHz that is ~30 ms per CPI before sliding-window savings — feasible for non-real-time
offline processing; real-time at the CPI rate favours reduced-dimension STAP.

## 15. Reproducibility

* All randomness derives from `seed` via independent `SeedSequence` streams (geometry, clutter
  reflectivity, target phase, noise, …): changing the noise does not move clutter.
* `results.json` stores the full config, derived quantities, assumptions, truth, metrics and timings;
  `synthetic_cpi_metadata.json` stores the config and truth next to the raw CPI.
* Same seed + same config ⇒ identical data (tested). Timings depend on the machine and BLAS.

## 16. Known limitations

* Simulation fidelity: see Section 4 (no ICM, multipath, channel errors, IF filter, phase noise,
  shadowing; Gaussian clutter only).
* Look direction assumes elevation 0 in the steering vector (exact for a 1-D array along x; the
  IWR1843 TX1 is elevated — use azimuth-only schedules such as `[0, 2]` for STAP).
* Oracle exclusion of known targets from training in controlled experiments; no non-homogeneity
  detector or training-cell censoring yet.
* CA-CFAR only (no OS/GO/SO CFAR); STAP output P_FA exceeds design by ~1.7× here.
* Doppler hypotheses are a grid (straddle loss up to the grid spacing).
* Forward-looking (vehicle) clutter is range-dependent; sliding-window training is then biased
  (no range-dependence compensation implemented). Brennan's rule applies to side-looking arrays.
* Real-sampling mode: simulator and decoder support it; the response-basis Monte Carlo supports
  complex sampling only.
* DCA1000: only header-free raw streams are decoded directly; UDP reassembly is a utility; formats
  beyond the listed ones are not supported.
* Hardware behaviour is unverified.

## 17. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| STAP weights ~30× slower than expected | Two OpenBLAS thread pools (NumPy and SciPy wheels) contending on small matrices: `export OPENBLAS_NUM_THREADS=1` (scripts do this by default) |
| a number in YAML is read as text | PyYAML (YAML 1.1) parses `77.0e9` as a string; the loader accepts numeric strings, but `77.0e+9` is the unambiguous spelling |
| `unknown key(s)` | typo in YAML; keys are strict on purpose |
| `num_training < D: sample covariance is singular` | raise `stap.num_training`, shorten the CPI, or use `efa`/`pc` |
| `InsufficientTrainingError` for some bins | test bins too close to the swath edge or many excluded bins; they are skipped in the GMTI map |
| `not a multiple of the frame size` | wrong `num_adc_samples`/`num_rx`/`chirps_per_frame`/data type, or truncated capture; do not "fix" by guessing |
| many `out_of_range_values` | wrong `sign_handling`/`adc_bits`/packing |
| target at mirrored range / garbage everywhere | I/Q order / interleaving wrong (checklist steps 4–5) |
| velocities have the wrong sign | `mixer_convention` (checklist step 6) |
| angle biased in TDM | slot compensation needs the correct `tx_schedule` and chirp period |
| huge condition numbers | expected for clutter (rank ≪ D); check `loaded_condition_number`, consider more loading |
| plots fail on a headless machine | matplotlib uses the Agg backend in `plotting.py`; the notebook needs `ipykernel` |

## 18. Glossary

* **FMCW** — frequency-modulated continuous wave: the radar transmits linear frequency ramps (chirps).
* **IF / beat signal** — output of mixing transmitted and received chirps; a tone at `S·τ` per scatterer.
* **CPI** — coherent processing interval: the block of chirps processed coherently (here 16 per channel).
* **ADC** — analog-to-digital converter sampling the IF signal (fast time).
* **LVDS** — low-voltage differential signalling: the serial interface carrying raw ADC data off the radar chip.
* **DCA1000** — TI DCA1000EVM, captures LVDS ADC data and streams it to a PC over Ethernet.
* **Range-Doppler map** — power vs range (fast-time FFT) and radial velocity (slow-time FFT).
* **Clutter ridge** — locus of stationary-clutter energy in angle–Doppler space; a line `ν = −2 v_p u_x/λ` for a side-looking array on a moving platform.
* **Steering vector** — expected noise-free array (and space-time) response to a unit target at a given angle (and Doppler).
* **Covariance matrix** — second-order statistics `E[x xᴴ]` of interference snapshots; estimated from training range bins.
* **STAP** — space-time adaptive processing: joint adaptive filtering over channels and pulses to cancel clutter/jamming.
* **GMTI** — ground moving target indication: detecting slow movers against ground clutter, typically from a moving platform.
* **CFAR** — constant false-alarm rate detection: thresholds scaled to a locally estimated background.
* **SINR** — signal-to-interference-plus-noise ratio.
* **P_D** — probability of detection; **P_FA** — probability of false alarm (per cell here).
