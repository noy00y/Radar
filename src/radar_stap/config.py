"""Typed configuration objects, YAML loading, validation and derived quantities.

Design rules
------------
* ``simulation.yaml`` describes a *complete, self-consistent synthetic scenario*.
  It is loaded with :func:`load_simulation_config`.
* Hardware capture settings live in a separate file and are handled by
  :mod:`radar_stap.capture` -- simulation values are never used to decode a
  hardware file.
* Unknown keys are rejected (typos must not silently fall back to defaults).
* A scenario file may contain ``base: other.yaml``; the base is loaded first
  and the scenario's keys are deep-merged on top (lists are replaced, not
  merged).
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import yaml

from .constants import MIXER_CONVENTIONS, MIXER_TX_CONJ_RX, SPEED_OF_LIGHT

log = logging.getLogger(__name__)

VALID_WINDOWS = ("rect", "hann", "hamming", "blackman", "taylor", "chebyshev")


class ConfigError(ValueError):
    """Raised for invalid, inconsistent or incomplete configuration."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RadarConfig:
    """FMCW waveform and receiver parameters (TI ``profileCfg`` naming in comments).

    Chirp n starts its ramp at absolute time ``n*Tc + idle_time_s`` where
    ``Tc = idle_time_s + ramp_end_time_s`` is the chirp repetition interval.
    ADC sample k of that chirp is taken at fast time
    ``t_f = adc_start_time_s + k / adc_sample_rate_hz`` after ramp start.
    """

    carrier_frequency_hz: float  # startFreq: RF frequency at ramp start
    chirp_slope_hz_per_s: float  # freqSlopeConst
    idle_time_s: float  # idleTime
    adc_start_time_s: float  # adcStartTime
    ramp_end_time_s: float  # rampEndTime
    adc_sample_rate_hz: float  # digOutSampleRate
    num_adc_samples: int  # numAdcSamples
    num_chirps: int  # chirps per CPI (all TX slots included)
    sampling: str = "complex"  # "complex" (complex1x-like) or "real"
    mixer_convention: str = MIXER_TX_CONJ_RX
    noise_power: float = 1.0  # complex noise variance per raw ADC sample, per channel


@dataclass
class ArrayConfig:
    """Antenna geometry in the radar body frame.

    Body frame: +x along the (azimuth) array axis, +y boresight, +z up.
    ``tx_schedule`` lists the TX index used by consecutive chirps; it repeats
    through the CPI (``[0]`` = single TX, ``[0, 1]`` = 2-TX TDM-MIMO).
    """

    rx_positions: list[list[float]]
    tx_positions: list[list[float]] = field(default_factory=lambda: [[0.0, 0.0, 0.0]])
    tx_schedule: list[int] = field(default_factory=lambda: [0])
    position_units: str = "half_wavelength"  # or "meters"


@dataclass
class PlatformConfig:
    """Radar platform kinematics (constant velocity over the CPI) in world frame (z up)."""

    position_m: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    velocity_mps: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    yaw_deg: float = 0.0  # rotation of body frame about world z
    depression_deg: float = 0.0  # boresight tilt below horizontal
    element_pattern_exponent: float = 1.0  # two-way amplitude gain = max(cos(off-boresight),0)^p


@dataclass
class ClutterConfig:
    """Distributed stationary ground clutter made of discrete point scatterers."""

    enabled: bool = True
    num_scatterers: int = 2000
    range_min_m: float = 5.0  # slant range
    range_max_m: float = 50.0
    azimuth_min_deg: float = -75.0  # horizontal azimuth relative to boresight
    azimuth_max_deg: float = 75.0
    ground_height_m: float = 0.0
    cnr_db: float = 20.0  # total expected clutter power / noise power per raw sample per channel
    range_power_exponent: float = 3.0  # scatterer power ~ R^-n before normalisation
    reflectivity: str = "rayleigh"  # "rayleigh" (complex Gaussian) or "constant_amplitude"


@dataclass
class TargetConfig:
    """Point target / reflector.

    Position: either ``position_m`` (world) or ``range_m`` + ``angle_deg``
    (array cone angle, sin(angle) = direction cosine along the array axis)
    + ``height_m`` (world z of the scatterer).
    Velocity: either ``velocity_mps`` (world) or ``radial_velocity_mps``
    (relative, dR/dt, positive receding) realised by motion along the
    horizontal line of sight.
    """

    name: str
    snr_db: float  # per raw ADC sample, per channel, relative to noise_power
    position_m: Optional[list[float]] = None
    range_m: Optional[float] = None
    angle_deg: Optional[float] = None
    height_m: float = 0.0
    velocity_mps: Optional[list[float]] = None
    radial_velocity_mps: Optional[float] = None
    phase_deg: Optional[float] = None  # None -> random (seeded)


@dataclass
class ProcessingConfig:
    range_window: str = "hann"
    doppler_window: str = "hann"
    range_nfft: Optional[int] = None  # None -> num_adc_samples
    doppler_nfft: Optional[int] = None  # None -> number of slow-time samples
    range_min_m: float = 0.0
    range_max_m: Optional[float] = None
    beat_band: str = "positive"  # complex: "positive" [0, fs) or "two_sided" [-fs/2, fs/2)
    remove_dc: bool = False  # subtract per-chirp fast-time mean (TI-style DC removal)


@dataclass
class DetectionConfig:
    """2-D cell-averaging CFAR on the (noncoherently combined) range-Doppler map."""

    guard_cells: list[int] = field(default_factory=lambda: [2, 1])  # [range, doppler]
    training_cells: list[int] = field(default_factory=lambda: [8, 4])
    pfa: float = 1e-5
    range_edge: str = "shrink"  # "shrink" (use available cells) or "wrap"
    doppler_edge: str = "wrap"  # Doppler is circular
    range_gate_bins: int = 2  # association gates
    doppler_gate_bins: int = 1
    angle_gate_deg: float = 10.0


@dataclass
class StapConfig:
    enabled: bool = True
    look_angle_deg: float = 0.0
    num_training: int = 128
    num_guard: int = 3  # guard bins either side of the test bin
    loading_mode: str = "noise_relative"  # "noise_relative" | "absolute" | "trace_relative" | "noise_eig_relative"
    loading_value: float = 1.0
    num_doppler_hypotheses: int = 32
    spatial_taper: str = "rect"
    temporal_taper: str = "hann"
    methods: list[str] = field(default_factory=lambda: ["nonadaptive", "mvdr", "efa"])
    efa_num_bins: int = 3
    pc_rank: Optional[int] = None  # principal-components rank; None -> eigenvalues > 10 x noise floor
    exclude_known_targets: bool = True  # oracle exclusion of target bins from training
    exclusion_halfwidth_bins: int = 3
    cfar_guard: int = 2
    cfar_training: int = 12
    cfar_pfa: float = 1e-4
    test_range_min_m: Optional[float] = None
    test_range_max_m: Optional[float] = None


@dataclass
class MonteCarloConfig:
    enabled: bool = True
    num_trials: int = 60
    target_name: Optional[str] = None  # None -> first target
    target_snr_db: Optional[float] = None  # override for Pd evaluation
    test_halfwidth_bins: int = 20
    basis_halfwidth_bins: int = 16


@dataclass
class OutputConfig:
    save_raw: bool = True
    save_components: bool = True
    plot_dpi: int = 120


@dataclass
class SimulationConfig:
    seed: int
    radar: RadarConfig
    array: ArrayConfig
    platform: PlatformConfig = field(default_factory=PlatformConfig)
    clutter: ClutterConfig = field(default_factory=ClutterConfig)
    targets: list[TargetConfig] = field(default_factory=list)
    stationary_reflectors: list[TargetConfig] = field(default_factory=list)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    stap: StapConfig = field(default_factory=StapConfig)
    monte_carlo: MonteCarloConfig = field(default_factory=MonteCarloConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    scenario_name: str = "unnamed"
    description: str = ""


# ---------------------------------------------------------------------------
# Generic dict -> dataclass builder with strict key checking
# ---------------------------------------------------------------------------


def _is_optional(tp: Any) -> tuple[bool, Any]:
    origin = typing.get_origin(tp)
    if origin in (Union, types.UnionType):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1 and len(typing.get_args(tp)) == 2:
            return True, args[0]
    return False, tp


def _coerce(value: Any, tp: Any, path: str) -> Any:
    optional, inner = _is_optional(tp)
    if value is None:
        if optional:
            return None
        raise ConfigError(f"{path}: value is required (got null)")
    tp = inner
    origin = typing.get_origin(tp)
    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping, got {type(value).__name__}")
        return build_dataclass(tp, value, path)
    if origin is list:
        (item_tp,) = typing.get_args(tp)
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
        return [_coerce(v, item_tp, f"{path}[{i}]") for i, v in enumerate(value)]
    if tp is float:
        # PyYAML (YAML 1.1) reads "77.0e9" (no exponent sign) as a string: accept numeric strings.
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                raise ConfigError(f"{path}: expected a number, got {value!r}") from None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected a number, got {value!r}")
        return float(value)
    if tp is int:
        if isinstance(value, bool) or not (
            isinstance(value, int) or (isinstance(value, float) and value.is_integer())
        ):
            raise ConfigError(f"{path}: expected an integer, got {value!r}")
        return int(value)
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected true/false, got {value!r}")
        return value
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string, got {value!r}")
        return value
    return value


def build_dataclass(cls: type, data: dict, path: str = "") -> Any:
    """Build dataclass ``cls`` from ``data``; reject unknown keys, coerce types."""
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ConfigError(
            f"{path or cls.__name__}: unknown key(s) {sorted(unknown)}; allowed: {sorted(names)}"
        )
    kwargs = {}
    for f in dataclasses.fields(cls):
        sub = f"{path}.{f.name}" if path else f.name
        if f.name in data:
            kwargs[f.name] = _coerce(data[f.name], hints[f.name], sub)
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            raise ConfigError(f"{sub}: required key missing")
    return cls(**kwargs)


def deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge; lists and scalars in ``override`` replace ``base``."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml_with_base(path: str | Path, _depth: int = 0) -> dict:
    path = Path(path)
    if _depth > 5:
        raise ConfigError("config 'base' chain too deep (cycle?)")
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    base = data.pop("base", None)
    if base is not None:
        base_data = load_yaml_with_base((path.parent / base).resolve(), _depth + 1)
        data = deep_merge(base_data, data)
    return data


def simulation_config_from_dict(data: dict, validate: bool = True) -> SimulationConfig:
    cfg = build_dataclass(SimulationConfig, data)
    if validate:
        validate_simulation_config(cfg)
    return cfg


def load_simulation_config(path: str | Path, overrides: dict | None = None) -> SimulationConfig:
    """Load + validate a simulation YAML (with optional ``base`` and dict overrides)."""
    data = load_yaml_with_base(path)
    if overrides:
        data = deep_merge(data, overrides)
    return simulation_config_from_dict(data)


def config_to_dict(cfg: Any) -> dict:
    return dataclasses.asdict(cfg)


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DerivedQuantities:
    """Quantities derived from the waveform. See README for the assumptions.

    * ``range_resolution_m`` = c / (2 B_sampled): nominal (rectangular-window)
      resolution set by the *sampled* bandwidth. Windows broaden it.
    * ``range_bin_spacing_m`` = c fs / (2 S nfft): FFT grid spacing; zero
      padding shrinks it but does not improve resolution.
    * ``max_range_if_m``: beat-frequency limit (complex: fs, real: fs/2),
      ignoring IF filter roll-off which is not modelled.
    * ``max_range_adc_start_m``: echoes must arrive before ADC sampling
      starts for the dechirp model of the *same* chirp to be valid.
    * ``max_unambiguous_velocity_mps`` = lambda / (4 T_slow): Nyquist limit
      for the slow-time sampling interval of one virtual channel.
    """

    chirp_period_s: float
    sampling_duration_s: float
    sampled_bandwidth_hz: float
    ramp_bandwidth_hz: float
    center_frequency_hz: float
    wavelength_m: float
    range_resolution_m: float
    range_bin_spacing_m: float
    max_beat_frequency_hz: float
    max_range_if_m: float
    max_range_adc_start_m: float
    num_tx_slots: int
    num_loops: int
    slow_time_interval_s: float
    cpi_duration_s: float
    doppler_bin_spacing_hz: float
    velocity_resolution_mps: float
    max_unambiguous_velocity_mps: float
    range_doppler_coupling_m_per_mps: float
    bytes_per_cpi_int16: int

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def wavelength_for(radar: RadarConfig) -> float:
    """Wavelength at the centre of the *sampled* sweep (used for all phase models)."""
    fc_eff = radar.carrier_frequency_hz + radar.chirp_slope_hz_per_s * (
        radar.adc_start_time_s + radar.num_adc_samples / (2.0 * radar.adc_sample_rate_hz)
    )
    return SPEED_OF_LIGHT / fc_eff


def derive_quantities(
    radar: RadarConfig,
    tx_schedule: list[int] | None = None,
    range_nfft: int | None = None,
    doppler_nfft: int | None = None,
    num_rx: int = 1,
) -> DerivedQuantities:
    S = radar.chirp_slope_hz_per_s
    fs = radar.adc_sample_rate_hz
    K = radar.num_adc_samples
    tc = radar.idle_time_s + radar.ramp_end_time_s
    t_samp = K / fs
    b_samp = S * t_samp
    fc_eff = radar.carrier_frequency_hz + S * (radar.adc_start_time_s + t_samp / 2)
    lam = SPEED_OF_LIGHT / fc_eff
    nfft_r = range_nfft or K
    f_max = fs if radar.sampling == "complex" else fs / 2
    p = len(tx_schedule) if tx_schedule else 1
    loops = radar.num_chirps // p
    t_slow = p * tc
    nd = doppler_nfft or loops
    words = 2 if radar.sampling == "complex" else 1
    return DerivedQuantities(
        chirp_period_s=tc,
        sampling_duration_s=t_samp,
        sampled_bandwidth_hz=b_samp,
        ramp_bandwidth_hz=S * radar.ramp_end_time_s,
        center_frequency_hz=fc_eff,
        wavelength_m=lam,
        range_resolution_m=SPEED_OF_LIGHT / (2 * b_samp),
        range_bin_spacing_m=SPEED_OF_LIGHT * fs / (2 * S * nfft_r),
        max_beat_frequency_hz=f_max,
        max_range_if_m=f_max * SPEED_OF_LIGHT / (2 * S),
        max_range_adc_start_m=SPEED_OF_LIGHT * radar.adc_start_time_s / 2,
        num_tx_slots=p,
        num_loops=loops,
        slow_time_interval_s=t_slow,
        cpi_duration_s=radar.num_chirps * tc,
        doppler_bin_spacing_hz=1.0 / (nd * t_slow),
        velocity_resolution_mps=lam / (2 * loops * t_slow),
        max_unambiguous_velocity_mps=lam / (4 * t_slow),
        range_doppler_coupling_m_per_mps=fc_eff / S,
        bytes_per_cpi_int16=int(2 * words * K * radar.num_chirps * num_rx),
    )


def element_positions_m(array: ArrayConfig, wavelength: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (rx_positions (M,3), tx_positions (P,3)) in metres, body frame."""
    rx = np.asarray(array.rx_positions, dtype=float)
    tx = np.asarray(array.tx_positions, dtype=float)
    if array.position_units == "half_wavelength":
        scale = wavelength / 2.0
    elif array.position_units == "meters":
        scale = 1.0
    else:  # pragma: no cover - validated earlier
        raise ConfigError(f"unknown position_units {array.position_units!r}")
    return rx * scale, tx * scale


def sim_derived(cfg: SimulationConfig) -> DerivedQuantities:
    return derive_quantities(
        cfg.radar,
        cfg.array.tx_schedule,
        cfg.processing.range_nfft,
        cfg.processing.doppler_nfft,
        num_rx=len(cfg.array.rx_positions),
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _check(cond: bool, msg: str, errors: list[str]) -> None:
    if not cond:
        errors.append(msg)


def validate_radar(radar: RadarConfig, errors: list[str]) -> None:
    for name in (
        "carrier_frequency_hz",
        "chirp_slope_hz_per_s",
        "ramp_end_time_s",
        "adc_sample_rate_hz",
        "noise_power",
    ):
        _check(getattr(radar, name) > 0, f"radar.{name} must be > 0", errors)
    _check(radar.idle_time_s >= 0, "radar.idle_time_s must be >= 0", errors)
    _check(radar.adc_start_time_s >= 0, "radar.adc_start_time_s must be >= 0", errors)
    _check(radar.num_adc_samples >= 8, "radar.num_adc_samples must be >= 8", errors)
    _check(radar.num_chirps >= 2, "radar.num_chirps must be >= 2", errors)
    _check(radar.sampling in ("complex", "real"), "radar.sampling must be 'complex' or 'real'", errors)
    _check(
        radar.mixer_convention in MIXER_CONVENTIONS,
        f"radar.mixer_convention must be one of {MIXER_CONVENTIONS}",
        errors,
    )
    if radar.adc_sample_rate_hz > 0:
        end = radar.adc_start_time_s + radar.num_adc_samples / radar.adc_sample_rate_hz
        _check(
            end <= radar.ramp_end_time_s * (1 + 1e-9),
            f"ADC window ends at {end * 1e6:.3f} us, after ramp end {radar.ramp_end_time_s * 1e6:.3f} us "
            "(adc_start + N/fs must be <= ramp_end)",
            errors,
        )
    if not (76e9 <= radar.carrier_frequency_hz <= 81e9):
        log.info(
            "carrier %.2f GHz is outside the 76-81 GHz IWR1843 band (fine for pure simulation)",
            radar.carrier_frequency_hz / 1e9,
        )


def validate_array(array: ArrayConfig, num_chirps: int, errors: list[str]) -> None:
    _check(
        array.position_units in ("half_wavelength", "meters"),
        "array.position_units must be 'half_wavelength' or 'meters'",
        errors,
    )
    for name in ("rx_positions", "tx_positions"):
        pos = getattr(array, name)
        _check(len(pos) >= 1, f"array.{name} must not be empty", errors)
        _check(all(len(p) == 3 for p in pos), f"array.{name} entries must be [x, y, z]", errors)
    _check(len(array.tx_schedule) >= 1, "array.tx_schedule must not be empty", errors)
    ntx = len(array.tx_positions)
    _check(
        all(0 <= t < ntx for t in array.tx_schedule),
        f"array.tx_schedule indices must be in [0, {ntx - 1}]",
        errors,
    )
    if array.tx_schedule:
        _check(
            num_chirps % len(array.tx_schedule) == 0,
            "radar.num_chirps must be a multiple of len(array.tx_schedule)",
            errors,
        )
        _check(
            len(set(array.tx_schedule)) == len(array.tx_schedule),
            "array.tx_schedule must use each TX at most once per TDM period",
            errors,
        )


def _target_ranges(cfg: SimulationConfig) -> list[float]:
    p0 = np.asarray(cfg.platform.position_m, float)
    out = []
    for t in cfg.targets + cfg.stationary_reflectors:
        if t.position_m is not None:
            out.append(float(np.linalg.norm(np.asarray(t.position_m) - p0)))
        elif t.range_m is not None:
            out.append(t.range_m)
    return out


def validate_simulation_config(cfg: SimulationConfig) -> None:
    """Raise :class:`ConfigError` listing *all* problems found."""
    errors: list[str] = []
    validate_radar(cfg.radar, errors)
    validate_array(cfg.array, cfg.radar.num_chirps, errors)
    if errors:
        raise ConfigError("invalid configuration:\n  - " + "\n  - ".join(errors))

    d = sim_derived(cfg)
    pr = cfg.processing
    for w in (pr.range_window, pr.doppler_window, cfg.stap.spatial_taper, cfg.stap.temporal_taper):
        _check(w in VALID_WINDOWS, f"window {w!r} not in {VALID_WINDOWS}", errors)
    _check(
        pr.range_nfft is None or pr.range_nfft >= cfg.radar.num_adc_samples,
        "processing.range_nfft must be >= num_adc_samples",
        errors,
    )
    _check(
        pr.doppler_nfft is None or pr.doppler_nfft >= d.num_loops,
        "processing.doppler_nfft must be >= slow-time samples per channel",
        errors,
    )
    _check(pr.beat_band in ("positive", "two_sided"), "processing.beat_band invalid", errors)
    if cfg.radar.sampling == "real":
        _check(pr.beat_band == "positive", "real sampling supports beat_band 'positive' only", errors)

    rmax_scene = max(_target_ranges(cfg) + ([cfg.clutter.range_max_m] if cfg.clutter.enabled else [0.0]))
    _check(
        rmax_scene < d.max_range_adc_start_m,
        f"scene range {rmax_scene:.1f} m: round-trip delay exceeds adc_start_time "
        f"(limit {d.max_range_adc_start_m:.1f} m); increase adc_start_time_s",
        errors,
    )
    _check(
        rmax_scene < 0.95 * d.max_range_if_m,
        f"scene range {rmax_scene:.1f} m exceeds ~95% of IF-limited range {d.max_range_if_m:.1f} m",
        errors,
    )

    c = cfg.clutter
    if c.enabled:
        _check(c.num_scatterers >= 1, "clutter.num_scatterers must be >= 1", errors)
        _check(0 <= c.range_min_m < c.range_max_m, "clutter range limits invalid", errors)
        _check(c.azimuth_min_deg < c.azimuth_max_deg, "clutter azimuth limits invalid", errors)
        _check(c.reflectivity in ("rayleigh", "constant_amplitude"), "clutter.reflectivity invalid", errors)
        h = abs(cfg.platform.position_m[2] - c.ground_height_m)
        _check(c.range_min_m > h, f"clutter.range_min_m must exceed platform height above ground ({h} m)", errors)

    names = set()
    for t in cfg.targets + cfg.stationary_reflectors:
        _check(t.name not in names, f"duplicate target name {t.name!r}", errors)
        names.add(t.name)
        _check(
            (t.position_m is None) != (t.range_m is None),
            f"target {t.name}: give exactly one of position_m or range_m(+angle_deg)",
            errors,
        )
        if t.range_m is not None:
            _check(t.angle_deg is not None, f"target {t.name}: range_m needs angle_deg", errors)
        _check(
            not (t.velocity_mps is not None and t.radial_velocity_mps is not None),
            f"target {t.name}: give at most one of velocity_mps or radial_velocity_mps",
            errors,
        )
    for t in cfg.stationary_reflectors:
        _check(
            t.velocity_mps is None and t.radial_velocity_mps is None,
            f"stationary reflector {t.name} must not have a velocity",
            errors,
        )

    det = cfg.detection
    _check(0 < det.pfa < 1, "detection.pfa must be in (0,1)", errors)
    _check(len(det.guard_cells) == 2 and len(det.training_cells) == 2, "CFAR cells must be [range, doppler]", errors)

    s = cfg.stap
    D = len(cfg.array.rx_positions) * len(cfg.array.tx_schedule) * d.num_loops
    if s.enabled:
        _check(abs(s.look_angle_deg) < 90, "stap.look_angle_deg must be in (-90, 90)", errors)
        _check(s.num_doppler_hypotheses >= 1, "stap.num_doppler_hypotheses must be >= 1", errors)
        _check(
            s.loading_mode in ("noise_relative", "absolute", "trace_relative", "noise_eig_relative"),
            "stap.loading_mode invalid",
            errors,
        )
        _check(s.loading_value >= 0, "stap.loading_value must be >= 0", errors)
        _check(s.num_training % 2 == 0, "stap.num_training must be even (symmetric window)", errors)
        _check(s.efa_num_bins % 2 == 1, "stap.efa_num_bins must be odd", errors)
        known = {"nonadaptive", "mvdr", "efa", "pc"}
        _check(set(s.methods) <= known, f"stap.methods must be a subset of {sorted(known)}", errors)
        if "mvdr" in s.methods and s.num_training < D:
            errors.append(
                f"stap.num_training={s.num_training} < space-time dimension D={D}: the full-rank "
                "sample covariance is singular. Increase training or drop 'mvdr'."
            )
        elif "mvdr" in s.methods and s.num_training < 2 * D:
            log.warning(
                "full-rank STAP: L=%d < 2D=%d training snapshots; expect > 3 dB average SINR loss (RMB rule)",
                s.num_training,
                2 * D,
            )
        _check(0 < s.cfar_pfa < 1, "stap.cfar_pfa must be in (0,1)", errors)

    for t in cfg.targets:
        if t.radial_velocity_mps is not None and abs(t.radial_velocity_mps) >= d.max_unambiguous_velocity_mps:
            log.warning(
                "target %s radial velocity %.2f m/s exceeds unambiguous +-%.2f m/s (will alias)",
                t.name,
                t.radial_velocity_mps,
                d.max_unambiguous_velocity_mps,
            )
    if errors:
        raise ConfigError("invalid configuration:\n  - " + "\n  - ".join(errors))


def describe_derived(d: DerivedQuantities) -> str:
    lines = [
        f"chirp period            : {d.chirp_period_s * 1e6:.2f} us",
        f"sampled bandwidth       : {d.sampled_bandwidth_hz / 1e6:.1f} MHz (ramp {d.ramp_bandwidth_hz / 1e6:.1f} MHz)",
        f"centre freq / lambda    : {d.center_frequency_hz / 1e9:.3f} GHz / {d.wavelength_m * 1e3:.3f} mm",
        f"range resolution        : {d.range_resolution_m:.3f} m (nominal, rect window)",
        f"range bin spacing       : {d.range_bin_spacing_m:.3f} m",
        f"IF-limited max range    : {d.max_range_if_m:.1f} m",
        f"adc-start-limited range : {d.max_range_adc_start_m:.1f} m",
        f"TX slots / loops        : {d.num_tx_slots} / {d.num_loops}",
        f"slow-time interval      : {d.slow_time_interval_s * 1e6:.2f} us",
        f"velocity resolution     : {d.velocity_resolution_mps:.3f} m/s",
        f"unambiguous velocity    : +-{d.max_unambiguous_velocity_mps:.2f} m/s",
        f"range-Doppler coupling  : {d.range_doppler_coupling_m_per_mps * 1e3:.1f} mm per m/s",
        f"CPI duration            : {d.cpi_duration_s * 1e3:.3f} ms",
        f"raw data per CPI (int16): {d.bytes_per_cpi_int16 / 1024:.1f} KiB",
    ]
    return "\n".join(lines)

