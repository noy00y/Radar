"""Hardware-capture configuration, completeness checks, and offline processing of DCA1000 captures.

Discipline
----------
* The capture configuration (``configs/radar_capture.yaml``) contains only
  verified or explicitly user-supplied settings. Unknown values stay ``null``.
* Waveform values may come from the TI mmWave SDK ``.cfg`` used for the
  capture (``waveform.mmwave_cfg_file``); a value given both in YAML and in the
  ``.cfg`` must agree, otherwise loading fails.
* Processing stages are enabled only if the information they need is present:

  ======== ==============================================================
  decode   complete ``adc_format`` and ``capture_format_verified: true``
  range    + chirp slope, ADC sample rate
  doppler  + carrier, idle/ramp/adc-start times, TX schedule, mixer convention
  angle    + verified antenna positions (``antenna.verified: true``)
  stap     + angle requirements and ``stap.enabled``
  ======== ==============================================================

  Missing per-channel phase calibration does not block angle/STAP but is
  reported as a limitation (IWR1843 channels need calibration).
* Simulation settings are never used to decode or interpret a capture.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .adc_loader import AdcFormat, cube_to_cpis, load_capture_file
from .angle_processing import direction_from_angles, estimate_angle
from .config import (
    ConfigError,
    DetectionConfig,
    ProcessingConfig,
    RadarConfig,
    StapConfig,
    build_dataclass,
    load_yaml_with_base,
    validate_radar,
    wavelength_for,
)
from .constants import MIXER_CONVENTIONS, MIXER_TX_CONJ_RX, SPEED_OF_LIGHT
from .datacube import snapshots, to_virtual_cube, virtual_positions, virtual_time_offsets
from .detection import ca_cfar, cluster_detections, refine_to_local_power_peak
from .doppler_processing import doppler_fft, range_doppler_map, tdm_phase_compensation
from .mmwave_cfg import load_mmwave_cfg
from .range_processing import parabolic_peak, range_fft, range_mask, range_profile
from .stap import SpaceTimeModel, build_method, run_gmti, select_training_bins

log = logging.getLogger(__name__)


class CaptureConfigError(ValueError):
    """Capture configuration is incomplete, unverified or contradictory."""


WAVEFORM_FIELDS = (
    "carrier_frequency_hz",
    "chirp_slope_hz_per_s",
    "idle_time_s",
    "adc_start_time_s",
    "ramp_end_time_s",
    "adc_sample_rate_hz",
)


@dataclass
class CaptureWaveform:
    mmwave_cfg_file: Optional[str] = None
    carrier_frequency_hz: Optional[float] = None
    chirp_slope_hz_per_s: Optional[float] = None
    idle_time_s: Optional[float] = None
    adc_start_time_s: Optional[float] = None
    ramp_end_time_s: Optional[float] = None
    adc_sample_rate_hz: Optional[float] = None
    mixer_convention: Optional[str] = None


@dataclass
class CaptureAntenna:
    verified: bool = False
    position_units: str = "half_wavelength"
    rx_positions: Optional[list[list[float]]] = None
    tx_positions: Optional[list[list[float]]] = None


@dataclass
class CaptureConfig:
    capture_format_verified: bool = False
    adc_format: AdcFormat = field(default_factory=AdcFormat)
    waveform: CaptureWaveform = field(default_factory=CaptureWaveform)
    tx_schedule: Optional[list[int]] = None
    # Project policy: single TX only. TDM-MIMO couples target speed into angle (the per-TX
    # phase compensation needs the true, unaliased Doppler) and divides the speed limit by the
    # number of TX. Multi-TX captures get angle/STAP only with this explicit opt-in (legacy data).
    allow_multi_tx: bool = False
    antenna: CaptureAntenna = field(default_factory=CaptureAntenna)
    rx_calibration: Optional[list[list[float]]] = None  # per virtual channel [re, im] multipliers
    platform_velocity_mps: Optional[list[float]] = None  # informational
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    stap: StapConfig = field(default_factory=lambda: StapConfig(enabled=False, loading_mode="noise_eig_relative"))
    max_frames: Optional[int] = None
    allow_partial_final_frame: bool = False
    notes: str = ""
    provenance: dict = field(default_factory=dict)  # filled by the loader (not a YAML key)


def load_capture_config(path: str | Path) -> CaptureConfig:
    path = Path(path)
    data = load_yaml_with_base(path)
    if "provenance" in data:
        raise ConfigError("'provenance' is filled by the loader and must not be set in YAML")

    allowed = set(CaptureConfig.__dataclass_fields__)
    ignored = {"studio"}
    unknown = sorted(set(data) - allowed - ignored)
    if unknown:
        raise CaptureConfigError(
            f"unknown key(s) {unknown}; allowed: {sorted(allowed)}"
        )
    data = {k: v for k, v in data.items() if k not in ignored}

    cc = build_dataclass(CaptureConfig, data)
    prov = {}
    for f in WAVEFORM_FIELDS:
        if getattr(cc.waveform, f) is not None:
            prov[f"waveform.{f}"] = "yaml"
    if cc.waveform.mmwave_cfg_file:
        cfg_path = Path(cc.waveform.mmwave_cfg_file)
        if not cfg_path.is_absolute():
            cfg_path = (path.parent / cfg_path).resolve()
        derived = load_mmwave_cfg(cfg_path)
        _merge_cli(cc, derived, prov, str(cfg_path))
    cc.provenance = prov
    return cc


def _close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-15)


def _merge_cli(cc: CaptureConfig, d: dict, prov: dict, src: str) -> None:
    conflicts = []

    def put(obj, attr, value, key):
        if value is None:
            return
        cur = getattr(obj, attr)
        if cur is None:
            setattr(obj, attr, value)
            prov[key] = f"mmwave_cfg ({Path(src).name})"
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if not _close(cur, value):
                conflicts.append(f"{key}: yaml={cur} vs cfg={value}")
        elif cur != value:
            conflicts.append(f"{key}: yaml={cur} vs cfg={value}")

    for f in WAVEFORM_FIELDS:
        put(cc.waveform, f, d["waveform"][f], f"waveform.{f}")
    put(cc.adc_format, "num_adc_samples", d["num_adc_samples"], "adc_format.num_adc_samples")
    put(cc.adc_format, "num_rx", d["num_rx"], "adc_format.num_rx")
    put(cc.adc_format, "chirps_per_frame", d["chirps_per_frame"], "adc_format.chirps_per_frame")
    put(cc.adc_format, "adc_bits", d.get("adc_bits"), "adc_format.adc_bits")
    put(cc.adc_format, "data_type", d.get("data_type"), "adc_format.data_type")
    put(cc, "tx_schedule", d["tx_schedule"], "tx_schedule")
    if conflicts:
        raise CaptureConfigError("capture YAML contradicts the mmWave .cfg:\n  - " + "\n  - ".join(conflicts))
    prov["mmwave_cfg_hints"] = d.get("hints", {})


# ---------------------------------------------------------------------------
# Completeness / stage availability
# ---------------------------------------------------------------------------


@dataclass
class CaptureAssessment:
    stages: dict  # stage -> {"available": bool, "missing": [...], "notes": [...]}

    def available(self, stage: str) -> bool:
        return bool(self.stages[stage]["available"])

    def message(self) -> str:
        lines = []
        for st, info in self.stages.items():
            mark = "OK " if info["available"] else "-- "
            lines.append(f"[{mark}] {st}")
            for m in info["missing"]:
                lines.append(f"        missing: {m}")
            for n in info["notes"]:
                lines.append(f"        note: {n}")
        return "\n".join(lines)


_HELP = {
    "adc_format.data_type": "complex|real (adcCfg adcOutputFmt / mmWave Studio 'Complex 1x')",
    "adc_format.adc_bits": "12|14|16 (adcCfg numADCBits)",
    "adc_format.sign_handling": "int16_twos_complement|sign_extend_adc_bits (inspect raw words of a known capture)",
    "adc_format.byte_order": "little|big (DCA1000 captures are normally little-endian; verify)",
    "adc_format.iq_order": "I_first|Q_first (verify with a corner reflector: wrong order mirrors the beat spectrum)",
    "adc_format.complex_packing": "iq_interleaved|two_sample_blocks (LVDS lanes / capture tool dependent)",
    "adc_format.channel_layout": "rx_planar_per_chirp|rx_interleaved_per_sample",
    "adc_format.num_rx": "number of enabled RX (channelCfg rxChannelEn)",
    "adc_format.num_adc_samples": "profileCfg numAdcSamples",
    "adc_format.chirps_per_frame": "(chirpEndIdx-chirpStartIdx+1) * numLoops from frameCfg",
    "capture_format_verified": "set true only after checking a known capture (see README checklist)",
    "waveform.chirp_slope_hz_per_s": "profileCfg freqSlopeConst [MHz/us] * 1e12",
    "waveform.adc_sample_rate_hz": "profileCfg digOutSampleRate [ksps] * 1e3",
    "waveform.carrier_frequency_hz": "profileCfg startFreq [GHz] * 1e9",
    "waveform.idle_time_s": "profileCfg idleTime [us] * 1e-6",
    "waveform.adc_start_time_s": "profileCfg adcStartTime [us] * 1e-6",
    "waveform.ramp_end_time_s": "profileCfg rampEndTime [us] * 1e-6",
    "waveform.mixer_convention": "tx_times_conj_rx|rx_times_conj_tx: verify with an approaching corner reflector",
    "tx_schedule": "TX index of consecutive chirps in one loop (chirpCfg txEnable), e.g. [0, 2]",
    "antenna.verified": "true only after checking element positions against the board documentation",
    "antenna.rx_positions": "RX phase-centre positions [[x,y,z],...] in position_units",
    "antenna.tx_positions": "TX phase-centre positions [[x,y,z],...] indexed by TX number",
}


def assess_capture_config(cc: CaptureConfig, allow_unverified_format: bool = False) -> CaptureAssessment:
    fmt = cc.adc_format
    stages: dict[str, dict] = {}

    def missing_desc(keys):
        return [f"{k} -- {_HELP.get(k, '')}".rstrip(" -") for k in keys]

    dec_missing = [f"adc_format.{f}" for f in fmt.missing_fields()]
    notes = []
    if not cc.capture_format_verified:
        if allow_unverified_format:
            notes.append("format NOT verified; decoding anyway because --allow-unverified-format was given")
        else:
            dec_missing.append("capture_format_verified")
    stages["decode"] = {"available": not dec_missing, "missing": missing_desc(dec_missing), "notes": notes}

    w = cc.waveform
    rng_missing = [f"waveform.{f}" for f in ("chirp_slope_hz_per_s", "adc_sample_rate_hz") if getattr(w, f) is None]
    rng_notes = []
    if w.mixer_convention is None:
        rng_notes.append("mixer_convention unknown: range profile shown two-sided (positive beat side not assumed)")
    elif w.mixer_convention not in MIXER_CONVENTIONS:
        rng_missing.append("waveform.mixer_convention")
    stages["range"] = {
        "available": stages["decode"]["available"] and not rng_missing,
        "missing": missing_desc(rng_missing),
        "notes": rng_notes,
    }

    dop_missing = [f"waveform.{f}" for f in ("carrier_frequency_hz", "idle_time_s", "adc_start_time_s", "ramp_end_time_s")
                   if getattr(w, f) is None]
    if cc.tx_schedule is None:
        dop_missing.append("tx_schedule")
    if w.mixer_convention is None:
        dop_missing.append("waveform.mixer_convention")
    dop_notes = []
    if cc.tx_schedule is not None and fmt.chirps_per_frame and fmt.chirps_per_frame % len(cc.tx_schedule):
        dop_missing.append("tx_schedule (chirps_per_frame is not a multiple of its length)")
    stages["doppler"] = {
        "available": stages["range"]["available"] and not dop_missing,
        "missing": missing_desc(dop_missing),
        "notes": dop_notes,
    }

    a = cc.antenna
    ang_missing = []
    if not a.verified:
        ang_missing.append("antenna.verified")
    if a.rx_positions is None:
        ang_missing.append("antenna.rx_positions")
    elif fmt.num_rx is not None and len(a.rx_positions) != fmt.num_rx:
        ang_missing.append(f"antenna.rx_positions (has {len(a.rx_positions)} entries, num_rx={fmt.num_rx})")
    if a.tx_positions is None:
        ang_missing.append("antenna.tx_positions")
    elif cc.tx_schedule is not None and max(cc.tx_schedule) >= len(a.tx_positions):
        ang_missing.append("antenna.tx_positions (does not cover every TX in tx_schedule)")
    if cc.tx_schedule is not None and len(set(cc.tx_schedule)) > 1 and not cc.allow_multi_tx:
        ang_missing.append(
            f"tx_schedule {cc.tx_schedule}: multi-TX (TDM-MIMO) is disabled by project policy - its angle "
            "depends on the unaliased target speed. Use a single TX (e.g. [0]); for legacy multi-TX "
            "captures set allow_multi_tx: true"
        )
    ang_notes = []
    if cc.rx_calibration is None:
        ang_notes.append("no per-channel phase/gain calibration: angle estimates and STAP null depth are unreliable")
    stages["angle"] = {
        "available": stages["doppler"]["available"] and not ang_missing,
        "missing": missing_desc(ang_missing),
        "notes": ang_notes,
    }
    s = cc.stap
    stap_missing = [] if s.enabled else ["stap.enabled"]
    stap_notes = ["platform velocity not needed by adaptive STAP; used for dpca and the predicted clutter ridge"] if s.enabled else []
    if s.enabled:
        known = {"nonadaptive", "mvdr", "efa", "pc", "dpca"}
        if not set(s.methods) <= known:
            stap_missing.append(f"stap.methods (unknown: {sorted(set(s.methods) - known)})")
        if "dpca" in s.methods and not (cc.platform_velocity_mps and any(cc.platform_velocity_mps)):
            stap_missing.append("platform_velocity_mps (required by dpca)")
        if s.training_frames < 1 or s.training_frames % 2 == 0:
            stap_missing.append("stap.training_frames (must be an odd integer >= 1)")
        if s.training_censor_db is not None and s.training_censor_db <= 0:
            stap_missing.append("stap.training_censor_db (must be > 0 dB or null)")
        if s.look_angles_deg is not None and (not s.look_angles_deg or any(abs(a) >= 90 for a in s.look_angles_deg)):
            stap_missing.append("stap.look_angles_deg (non-empty list of angles in (-90, 90))")
        if s.forward_backward and cc.rx_calibration is None:
            stap_notes.append("forward_backward assumes calibrated channels: uncalibrated phase errors break "
                              "persymmetry (expect extra SINR loss)")
        from .batched import BACKENDS, PRECISIONS, resolve_backend
        from .backend import BackendUnavailable

        if s.backend not in BACKENDS:
            stap_missing.append(f"stap.backend (one of {list(BACKENDS)})")
        elif s.precision not in PRECISIONS:
            stap_missing.append(f"stap.precision (one of {sorted(PRECISIONS)})")
        elif s.gpu_batch_bins < 1:
            stap_missing.append("stap.gpu_batch_bins (must be >= 1)")
        else:
            try:
                _, used = resolve_backend(s.backend)
                stap_notes.append(f"STAP backend: {used}" + (f" ({s.precision} precision, {s.gpu_batch_bins} bins/batch)"
                                                            if used != "numpy" else " (per-bin CPU reference)"))
            except BackendUnavailable as exc:
                stap_missing.append(f"stap.backend cupy: {exc} (install radar-stap[gpu] on a CUDA machine, or use auto)")
        if fmt.num_rx is not None and fmt.chirps_per_frame is not None and cc.tx_schedule:
            D = fmt.num_rx * fmt.chirps_per_frame
            samples = s.num_training * s.training_frames * (2 if s.forward_backward else 1)
            for m in ("mvdr", "pc"):
                if m in s.methods and samples < D:
                    stap_notes.append(f"{m}: {samples} training samples < D={D}: covariance is singular "
                                      "(increase num_training or training_frames)")
                elif m == "mvdr" and m in s.methods and samples < 2 * D:
                    stap_notes.append(f"mvdr: {samples} training samples < 2D={2 * D}: > 3 dB expected loss (RMB)")
    stages["stap"] = {
        "available": stages["angle"]["available"] and not stap_missing,
        "missing": missing_desc(stap_missing),
        "notes": stap_notes,
    }
    return CaptureAssessment(stages)


def capture_radar_config(cc: CaptureConfig) -> RadarConfig:
    """RadarConfig built *only* from capture settings (for the processing functions)."""
    w, f = cc.waveform, cc.adc_format
    radar = RadarConfig(
        carrier_frequency_hz=w.carrier_frequency_hz if w.carrier_frequency_hz is not None else float("nan"),
        chirp_slope_hz_per_s=w.chirp_slope_hz_per_s,
        idle_time_s=w.idle_time_s if w.idle_time_s is not None else float("nan"),
        adc_start_time_s=w.adc_start_time_s if w.adc_start_time_s is not None else 0.0,
        ramp_end_time_s=w.ramp_end_time_s if w.ramp_end_time_s is not None else float("nan"),
        adc_sample_rate_hz=w.adc_sample_rate_hz,
        num_adc_samples=f.num_adc_samples,
        num_chirps=f.chirps_per_frame,
        sampling=f.data_type,
        mixer_convention=w.mixer_convention or MIXER_TX_CONJ_RX,
        noise_power=1.0,
    )
    if all(getattr(w, k) is not None for k in WAVEFORM_FIELDS):
        errs: list[str] = []
        validate_radar(radar, errs)
        if errs:
            raise CaptureConfigError("capture waveform is inconsistent:\n  - " + "\n  - ".join(errs))
    return radar


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------


def process_capture(
    cc: CaptureConfig,
    input_path: str | Path,
    output_dir: str | Path,
    allow_unverified_format: bool = False,
    max_frames: int | None = None,
    make_plots: bool = True,
) -> dict:
    """Decode and process a raw capture with only the stages its configuration supports."""
    from .pipeline import json_dump

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    assess = assess_capture_config(cc, allow_unverified_format)
    if not assess.available("decode"):
        raise CaptureConfigError(
            "cannot decode the capture: required settings are missing or unverified.\n" + assess.message()
        )
    max_frames = max_frames or cc.max_frames
    cube, rep = load_capture_file(input_path, cc.adc_format, cc.allow_partial_final_frame, max_frames)
    cpis = cube_to_cpis(cube)  # (F, M, Nc, K)
    report: dict = {
        "input": str(input_path),
        "stages": assess.stages,
        "provenance": cc.provenance,
        "decode": rep.as_dict(),
        "adc_format": cc.adc_format.as_dict(),
        "data_shape_frames_rx_chirps_samples": list(cpis.shape),
        "files": [],
    }
    files = report["files"]
    if make_plots:
        _plot_adc(cube, cc, out / "adc_diagnostics.png")
        files.append("adc_diagnostics.png")

    if not assess.available("range"):
        report["range"] = "unavailable"
        json_dump(report, out / "capture_summary.json")
        return report

    radar = capture_radar_config(cc)
    proc_cfg = ProcessingConfig(**asdict(cc.processing))
    if cc.waveform.mixer_convention is None and radar.sampling == "complex":
        proc_cfg.beat_band = "two_sided"
    Y, raxis = range_fft(cpis, radar, proc_cfg)  # (F, M, Nc, bins)
    prof = range_profile(Y)
    valid = range_mask(raxis, proc_cfg.range_min_m, proc_cfg.range_max_m)
    pk = int(np.argmax(np.where(valid, prof, 0)))
    report["range"] = {
        "num_bins": raxis.num_bins,
        "bin_spacing_m": raxis.bin_spacing_m,
        "band": raxis.band,
        "strongest_bin": pk,
        "strongest_range_m": float(raxis.range_m[pk]),
        "noise_floor_median_db": float(10 * np.log10(np.median(prof[valid]))) if valid.any() else None,
    }
    if make_plots:
        from . import plotting as pl

        pl.plot_range_profile(raxis.range_m, prof, None, valid, out / "range_profile.png", title="Capture range profile (mean over frames/channels/chirps)")
        files.append("range_profile.png")

    if not assess.available("doppler"):
        report["doppler"] = "unavailable"
        json_dump(report, out / "capture_summary.json")
        return report

    sched = cc.tx_schedule
    tc = radar.idle_time_s + radar.ramp_end_time_s
    lam = wavelength_for(radar)
    t_slow = len(sched) * tc
    cal = None
    if cc.rx_calibration is not None:
        cal = np.array([complex(a, b) for a, b in cc.rx_calibration])
    frames_out = []
    model = None
    if assess.available("angle"):
        units = cc.antenna.position_units
        scale = lam / 2 if units == "half_wavelength" else 1.0
        rx = np.asarray(cc.antenna.rx_positions, float) * scale
        tx = np.asarray(cc.antenna.tx_positions, float) * scale
        pos = virtual_positions(rx, tx, sched)
        offs = virtual_time_offsets(sched, rx.shape[0], tc)
        if cal is not None and cal.size != pos.shape[0]:
            raise CaptureConfigError(f"rx_calibration has {cal.size} entries; {pos.shape[0]} virtual channels expected")
        model = SpaceTimeModel(pos, offs, radar.num_chirps // len(sched), t_slow, lam)
    rd_sum = None
    for fi in range(Y.shape[0]):
        V = to_virtual_cube(Y[fi], sched)
        if cal is not None:
            V = V * cal[:, None, None]
        Z, daxis = doppler_fft(V, t_slow, lam, proc_cfg.doppler_window, proc_cfg.doppler_nfft)
        rd = range_doppler_map(Z)
        rd_sum = rd if rd_sum is None else rd_sum + rd
        exclude = np.repeat(~valid[:, None], rd.shape[1], axis=1)
        det = cc.detection
        cf = ca_cfar(rd, det.guard_cells, det.training_cells, det.pfa, [det.range_edge, det.doppler_edge], exclude)
        dets = cluster_detections(cf, rd, wrap_axes=(1,))
        rows = []
        for d in dets[:100]:
            r, k = refine_to_local_power_peak(rd, d.indices)
            rf = parabolic_peak(rd[:, k], r) if 0 < r < rd.shape[0] - 1 else float(r)
            kf = parabolic_peak(rd[r, :], k)
            v = float((kf - daxis.nfft // 2) * daxis.bin_spacing_hz * lam / 2)
            row = {
                "range_m": float(np.interp(rf, np.arange(raxis.num_bins), raxis.range_m)) - v * (2 / lam) * SPEED_OF_LIGHT / (2 * radar.chirp_slope_hz_per_s),
                "radial_velocity_mps": v,
                "snr_estimate_db": d.snr_estimate_db,
            }
            if model is not None:
                snap = Z[:, k, r] * tdm_phase_compensation(float(daxis.freq_hz[k]), model.time_offsets)
                row["angle_deg"] = estimate_angle(snap, model.positions, lam)[0]
            rows.append(row)
        frames_out.append({"frame": fi, "num_detections": len(dets), "detections": rows})
        if fi == 0 and make_plots:
            from . import plotting as pl

            pl.plot_range_doppler(rd[valid], raxis.range_m[valid], daxis.velocity_mps, None, None,
                                  out / "range_doppler_frame0.png", title="Capture range-Doppler map, frame 0")
            files.append("range_doppler_frame0.png")
    report["doppler"] = {
        "slow_time_interval_s": t_slow,
        "velocity_resolution_mps": lam / (2 * (radar.num_chirps // len(sched)) * t_slow),
        "max_unambiguous_velocity_mps": lam / (4 * t_slow),
        "frames": frames_out,
    }
    if make_plots and Y.shape[0] > 1:
        from . import plotting as pl

        pl.plot_range_doppler(rd_sum[valid], raxis.range_m[valid], daxis.velocity_mps, None, None,
                              out / "range_doppler_mean.png", title="Capture range-Doppler map, mean over frames")
        files.append("range_doppler_mean.png")

    if assess.available("stap"):
        report["stap"] = _capture_stap(Y, sched, cal, model, valid, cc, raxis, out if make_plots else None)
        files.extend(report["stap"].pop("files", []))
    else:
        report["stap"] = "unavailable"
    json_dump(report, out / "capture_summary.json")
    return report


def _capture_stap(Y, sched, cal, model, valid, cc, raxis, out_dir):
    """Angle-gated STAP/GMTI on every frame (CPI) of the capture.

    ``Y`` is the range-compressed capture (frames, rx, chirps, bins). Each frame is one
    CPI and is tested on its own data. Training:

    * range bins around the test bin (guard cells excluded), restricted to ``valid``;
    * pooled over ``stap.training_frames`` consecutive CPIs centred on the test frame
      (samples = num_training * training_frames). Valid while the clutter statistics
      are stationary over the pooling window: static radar, or a moving platform
      whose scene shifts only ~1 range bin per frame;
    * optionally censored: bins more than ``stap.training_censor_db`` above the local
      median power (strong movers / discretes) are dropped from training, which
      protects targets from self-nulling (they are still tested).

    Reported per method: per-frame CFAR detections, persistence of each (range bin,
    Doppler hypothesis) cell across frames, and detections on the noncoherently
    integrated (mean-power over frames) output.
    """
    from collections import defaultdict
    from dataclasses import replace

    from .experiments import gmti_detect

    s = cc.stap
    if s.loading_mode == "noise_relative":
        raise CaptureConfigError("capture STAP has no known noise level: use loading_mode noise_eig_relative/absolute/trace_relative")
    num_frames = int(Y.shape[0])
    angles = [float(a) for a in (s.look_angles_deg or [s.look_angle_deg])]
    directions = direction_from_angles(np.asarray(angles))  # (A, 3)
    direction = directions[0]  # primary look: plots and DPCA blind-speed report
    nu = model.doppler_grid(s.num_doppler_hypotheses)
    J = nu.size
    tmin = -np.inf if s.test_range_min_m is None else s.test_range_min_m
    tmax = np.inf if s.test_range_max_m is None else s.test_range_max_m
    test = []
    for b in np.where(valid)[0]:
        if not tmin <= raxis.range_m[b] <= tmax:
            continue
        try:
            select_training_bins(int(b), s.num_training, s.num_guard, valid)
            test.append(int(b))
        except ValueError:
            pass
    if not test:
        return {"error": "no range bins with enough training support"}
    test = np.asarray(test)

    def range_of(b):
        return np.interp(np.asarray(b, float), np.arange(raxis.num_bins), raxis.range_m)

    def det_rows(dets, angle):
        return [
            {
                "range_m": float(range_of(d.range_bin_frac)),
                "radial_velocity_mps": float(model.velocity_of(nu[int(d.doppler_bin_frac)])),
                "look_angle_deg": angle,
                "output_snr_db": d.snr_estimate_db,
            }
            for d in dets[:50]
        ]

    dv = float(abs(model.velocity_of(nu[1] - nu[0]))) if J > 1 else 0.0

    def merge_looks(rows):
        """One target is seen by neighbouring look angles: keep the strongest look per
        (range, velocity) cell, which also gives a coarse angle estimate."""
        kept = []
        for r in sorted(rows, key=lambda r: -r["output_snr_db"]):
            if not any(abs(r["range_m"] - k["range_m"]) <= 1.5 * raxis.bin_spacing_m
                       and abs(r["radial_velocity_mps"] - k["radial_velocity_mps"]) <= 1.5 * dv for k in kept):
                kept.append(r)
        return kept

    methods, build_errors = {}, {}
    for name in s.methods:
        try:
            methods[name] = build_method(name, model, s, platform_velocity_mps=cc.platform_velocity_mps)
        except ValueError as exc:
            build_errors[name] = str(exc)
            log.warning("STAP method %s not run: %s", name, exc)
    state = {
        name: {"frames": [], "weight_s": 0.0, "power_sum": [None] * len(angles), "last_res": None, "first": None,
               "cells": defaultdict(list), "error": None}
        for name in methods
    }
    K = int(s.training_frames)
    from .batched import resolve_backend

    xp, backend_used = resolve_backend(s.backend)
    cache: dict[int, np.ndarray] = {}

    def frame_snaps(k):
        if k not in cache:
            V = to_virtual_cube(Y[k], sched)
            if cal is not None:
                V = V * cal[:, None, None]
            cache[k] = snapshots(V)
        return cache[k]

    censored = []
    for fi in range(num_frames):
        k0 = min(max(0, fi - K // 2), max(0, num_frames - K))
        window = list(range(k0, min(num_frames, k0 + K)))
        for k in [k for k in cache if k < k0]:
            del cache[k]
        snaps = frame_snaps(fi)
        pool = np.stack([frame_snaps(k) for k in window]) if len(window) > 1 else None
        train_mask = valid
        if s.training_censor_db is not None:
            power = (np.abs(snaps[None] if pool is None else pool) ** 2).mean(axis=(0, 2))
            train_mask = censor_training_bins(power, valid, s.num_training, s.num_guard, s.training_censor_db)
        censored.append(int(valid.sum() - train_mask.sum()))
        Xv = snaps[valid]
        R_frame = (Xv.conj().T @ Xv) / max(Xv.shape[0], 1)   # swath-averaged space-time covariance
        if fi == 0:
            R_first = R_frame
            R_sum = np.zeros_like(R_frame)
        R_sum += R_frame
        for name, meth in methods.items():
            st = state[name]
            if st["error"] is not None:
                continue
            try:
                res = run_gmti(snaps, model, meth, directions, nu, test, s.num_training, s.num_guard, train_mask,
                               measure_memory=(fi == 0), training_snaps=pool, backend=xp,
                               precision=s.precision, batch_bins=s.gpu_batch_bins)
            except (ValueError, FloatingPointError) as exc:
                st["error"] = f"frame {fi}: {exc}"
                continue
            st["weight_s"] += res.weight_seconds
            st["last_res"] = res
            rows, n_raw = [], 0
            for ai, ang in enumerate(angles):
                res_a = replace(res, output=res.output[:, ai * J:(ai + 1) * J], nu_grid=nu)
                gd = gmti_detect(res_a, s)
                st["power_sum"][ai] = res_a.power if st["power_sum"][ai] is None else st["power_sum"][ai] + res_a.power
                if fi == 0 and ai == 0:
                    st["first"] = gd
                for d in gd.detections:
                    st["cells"][(ai, int(round(d.range_bin_frac)), int(d.doppler_bin_frac))].append(d.snr_estimate_db)
                n_raw += len(gd.detections)
                rows += det_rows(gd.detections, ang)
            merged = merge_looks(rows)
            st["frames"].append({"frame": fi, "num_detections": len(merged), "num_detections_all_looks": n_raw,
                                 "detections": merged[:50]})
        if (fi + 1) % 8 == 0 or fi + 1 == num_frames:
            log.info("STAP: processed %d/%d frames", fi + 1, num_frames)

    res_all, gmti_first, gmti_int = {name: {"error": e} for name, e in build_errors.items()}, {}, {}
    for name, st in state.items():
        n_ok = len(st["frames"])
        if n_ok == 0:
            res_all[name] = {"error": st["error"]}
            continue
        # Noncoherent integration over CPIs (valid for a static radar; on a moving platform
        # targets and clutter migrate in range between frames, so this map smears).
        # The integrated power is Gamma- not exponential-distributed: CA-CFAR is conservative here.
        rows_int = []
        for ai, ang in enumerate(angles):
            res_int = replace(st["last_res"], output=np.sqrt(st["power_sum"][ai] / n_ok), nu_grid=nu)
            gd_int = gmti_detect(res_int, s)
            if ai == 0:
                gmti_int[name] = gd_int
            rows_int += det_rows(gd_int.detections, ang)
        rows_int = merge_looks(rows_int)
        if st["first"] is not None:
            gmti_first[name] = st["first"]
        counts = [f["num_detections"] for f in st["frames"]]
        persistent = sorted(st["cells"].items(), key=lambda kv: -len(kv[1]))[:30]
        res_all[name] = {
            "frames_processed": n_ok,
            "total_detections": int(sum(counts)),
            "mean_detections_per_frame": float(np.mean(counts)),
            "weight_seconds_total": st["weight_s"],
            "weight_seconds_per_frame": st["weight_s"] / n_ok,
            "persistent_detections": [
                {
                    "range_m": float(range_of(b)),
                    "radial_velocity_mps": float(model.velocity_of(nu[j])),
                    "look_angle_deg": angles[ai],
                    "frames_detected": len(v),
                    "fraction_of_frames": len(v) / n_ok,
                    "mean_output_snr_db": float(np.mean(v)),
                }
                for (ai, b, j), v in persistent
            ],
            "integrated": {"num_detections": len(rows_int), "detections": rows_int[:50]},
            "frames": st["frames"],
        }
        if getattr(methods[name], "adaptive", True) is False and getattr(methods[name], "info", None) is not None:
            info = dict(methods[name].weights(None, model.steering(direction, nu), nu).info)
            if "blind_hypotheses" in info:
                info["blind_velocities_mps"] = [float(model.velocity_of(nu[j])) for j in info.pop("blind_hypotheses")]
            res_all[name]["method_info"] = info
        if st["error"] is not None:
            res_all[name]["error"] = st["error"]

    files = []
    if out_dir is not None:
        from . import plotting as pl

        vp = cc.platform_velocity_mps
        pl.plot_angle_doppler(R_first, model, Path(out_dir) / "angle_doppler_frame0.png", platform_velocity_mps=vp,
                              title="Angle-Doppler spectrum, frame 0 (swath average)")
        files.append("angle_doppler_frame0.png")
        if num_frames > 1:
            pl.plot_angle_doppler(R_sum / num_frames, model, Path(out_dir) / "angle_doppler_mean.png",
                                  platform_velocity_mps=vp,
                                  title=f"Angle-Doppler spectrum, mean of {num_frames} frames (swath average)")
            files.append("angle_doppler_mean.png")
        if gmti_first:
            pl.plot_gmti_maps(gmti_first, range_of, model, None, Path(out_dir) / "stap_gmti_frame0.png")
            files.append("stap_gmti_frame0.png")
        if gmti_int and num_frames > 1:
            pl.plot_gmti_maps(gmti_int, range_of, model, None, Path(out_dir) / "stap_gmti_integrated.png")
            files.append("stap_gmti_integrated.png")
    return {
        "look_angle_deg": angles[0],
        "look_angles_deg": angles,
        "dimension": model.dimension,
        "num_frames": num_frames,
        "num_test_bins": int(test.size),
        "test_range_m": [float(range_of(test.min())), float(range_of(test.max()))],
        "training": {
            "range_bins": s.num_training,
            "guard_bins": s.num_guard,
            "frames_pooled": K,
            "samples_per_covariance": s.num_training * min(K, num_frames) * (2 if s.forward_backward else 1),
            "forward_backward": s.forward_backward,
            "backend": backend_used,
            "precision": s.precision if backend_used != "numpy" else "double",
            "censor_db": s.training_censor_db,
            "mean_censored_bins_per_frame": float(np.mean(censored)) if censored else 0.0,
        },
        "methods": res_all,
        "files": files,
    }


def capture_space_time_model(cc: CaptureConfig, radar: RadarConfig) -> SpaceTimeModel:
    """Space-time model (virtual positions, TDM offsets, loops, T_slow) from capture settings."""
    sched = cc.tx_schedule
    tc = radar.idle_time_s + radar.ramp_end_time_s
    lam = wavelength_for(radar)
    scale = lam / 2 if cc.antenna.position_units == "half_wavelength" else 1.0
    rx = np.asarray(cc.antenna.rx_positions, float) * scale
    tx = np.asarray(cc.antenna.tx_positions, float) * scale
    return SpaceTimeModel(virtual_positions(rx, tx, sched), virtual_time_offsets(sched, rx.shape[0], tc),
                          radar.num_chirps // len(sched), len(sched) * tc, lam)


def calibrate_capture(
    cc: CaptureConfig,
    input_path: str | Path,
    reflector_range_m: float,
    reflector_angle_deg: float = 0.0,
    search_halfwidth_m: float = 1.0,
    allow_unverified_format: bool = False,
    max_frames: int | None = None,
    min_reflector_snr_db: float = 20.0,
    max_phase_jitter_deg: float = 5.0,
) -> dict:
    """Per-virtual-channel complex calibration from a capture of one dominant static reflector.

    The reflector (corner reflector, static scene) at known range and angle gives, per
    virtual channel c, the zero-Doppler response x_c at its range bin (mean over loops
    and frames). The ideal response is the spatial steering a_c(angle); the calibration
    multiplier is cal_c = a_c / x_c, referenced to channel 0 phase and unit mean gain.
    Applying ``V * cal`` (what process_capture does) then makes the reflector look ideal.

    If the config already has ``rx_calibration``, the angle measured with it is also
    reported: use a second capture at a different known angle to verify the calibration
    and the angle sign (the reported angle must match ``reflector_angle_deg``).
    """
    assess = assess_capture_config(cc, allow_unverified_format)
    if not assess.available("angle"):
        raise CaptureConfigError("calibration needs the decode/range/doppler/angle settings:\n" + assess.message())
    cube, _ = load_capture_file(input_path, cc.adc_format, cc.allow_partial_final_frame, max_frames or cc.max_frames)
    radar = capture_radar_config(cc)
    proc_cfg = ProcessingConfig(**asdict(cc.processing))
    Y, raxis = range_fft(cube_to_cpis(cube), radar, proc_cfg)
    model = capture_space_time_model(cc, radar)
    valid = range_mask(raxis, proc_cfg.range_min_m, proc_cfg.range_max_m)
    win = np.abs(raxis.range_m - reflector_range_m) <= search_halfwidth_m
    if not win.any():
        raise CaptureConfigError(f"no range bins within {search_halfwidth_m} m of {reflector_range_m} m")
    # zero-Doppler response per frame, channel, bin: static reflector survives, movers average out
    X = np.stack([to_virtual_cube(Y[f], cc.tx_schedule).mean(axis=1) for f in range(Y.shape[0])])  # (F, C, R)
    pw = (np.abs(X) ** 2).mean(axis=(0, 1))
    cand = np.flatnonzero(win)
    rb = int(cand[np.argmax(pw[cand])])
    ref = valid & ~win
    snr_db = float(10 * np.log10(pw[rb] / np.median(pw[ref]))) if ref.any() else float("nan")
    xf = X[:, :, rb]  # (F, C)
    x = xf.mean(axis=0)
    a = model.spatial(direction_from_angles(reflector_angle_deg))
    cal = a / x
    cal = cal / (cal[0] / abs(cal[0]))
    cal = cal / np.mean(np.abs(cal))
    rel = xf * np.conj(xf[:, :1])  # phase of each channel relative to channel 0, per frame
    jitter = np.rad2deg(np.std(np.angle(rel * np.conj(rel.mean(axis=0))[None]), axis=0))
    warnings = []
    if not snr_db >= min_reflector_snr_db:
        warnings.append(f"reflector only {snr_db:.1f} dB above the median bin (< {min_reflector_snr_db} dB): "
                        "other scatterers bias the calibration")
    if float(jitter.max()) > max_phase_jitter_deg:
        warnings.append(f"channel phase varies by up to {jitter.max():.1f} deg across frames "
                        f"(> {max_phase_jitter_deg}): reflector or radar moved, or SNR too low")
    edge = abs(float(raxis.range_m[rb]) - reflector_range_m) > 0.9 * search_halfwidth_m
    if edge:
        warnings.append("strongest bin is at the edge of the search window: check reflector_range_m")
    out = {
        "reflector_range_m_measured": float(raxis.range_m[rb]),
        "reflector_bin": rb,
        "reflector_snr_db": snr_db,
        "reflector_angle_deg_assumed": reflector_angle_deg,
        "angle_uncalibrated_deg": float(estimate_angle(x, model.positions, model.wavelength)[0]),
        "phase_offset_deg": [float(v) for v in -np.rad2deg(np.angle(cal))],
        "gain_offset_db": [float(v) for v in -20 * np.log10(np.abs(cal))],
        "phase_jitter_deg": [float(v) for v in jitter],
        "rx_calibration": [[round(float(c.real), 6), round(float(c.imag), 6)] for c in cal],
        "num_frames": int(Y.shape[0]),
        "warnings": warnings,
    }
    if cc.rx_calibration is not None:
        cfg_cal = np.array([complex(re, im) for re, im in cc.rx_calibration])
        if cfg_cal.size == x.size:
            ang = float(estimate_angle(x * cfg_cal, model.positions, model.wavelength)[0])
            out["angle_with_config_calibration_deg"] = ang
            out["angle_error_with_config_calibration_deg"] = ang - reflector_angle_deg
            if reflector_angle_deg != 0 and np.sign(ang) != np.sign(reflector_angle_deg):
                warnings.append("angle SIGN is reversed with the config calibration: check antenna positions / "
                                "mixer_convention before trusting any angle")
    return out


def censor_training_bins(power, valid, num_training, num_guard, censor_db):
    """Training mask without bins more than ``censor_db`` above their local median power.

    The local reference is the median over the ~num_training nearest valid bins (excluding
    the bin itself), so the natural fall of clutter power with range is not censored.
    Censoring is skipped for a frame if it would leave too few bins to train any test bin.
    Trade-off: strong clutter discretes are censored too, so they are not nulled.
    """
    idx = np.flatnonzero(valid)
    p = np.asarray(power, float)[idx]
    half = max(num_training // 2, 4)
    thr = 10.0 ** (censor_db / 10.0)
    keep = np.ones(idx.size, bool)
    for i in range(idx.size):
        lo, hi = max(0, i - half), min(idx.size, i + half + 1)
        local = np.delete(p[lo:hi], i - lo)
        if local.size and p[i] > thr * np.median(local):
            keep[i] = False
    if keep.sum() < num_training + 2 * num_guard + 1:
        return valid
    out = np.zeros_like(np.asarray(valid, bool))
    out[idx[keep]] = True
    return out


def _plot_adc(cube, cc, path):
    import matplotlib.pyplot as plt

    from .plotting import _save

    fig, ax = plt.subplots(figsize=(7, 3.5))
    lim = 1 << (cc.adc_format.adc_bits - 1)
    for m in range(cube.shape[2]):
        x = cube[:, :, m, :].ravel()
        ax.hist(np.real(x), bins=128, range=(-lim, lim), histtype="step", label=f"RX{m} I")
    ax.set_yscale("log")
    ax.set_xlabel("ADC code")
    ax.set_ylabel("Count")
    ax.set_title("Raw ADC value histogram (I / real part); check for clipping and DC offset")
    ax.legend(fontsize=7)
    _save(fig, path)
