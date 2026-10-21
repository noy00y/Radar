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
    ang_notes = []
    if cc.rx_calibration is None:
        ang_notes.append("no per-channel phase/gain calibration: angle estimates and STAP null depth are unreliable")
    stages["angle"] = {
        "available": stages["doppler"]["available"] and not ang_missing,
        "missing": missing_desc(ang_missing),
        "notes": ang_notes,
    }
    stap_missing = [] if cc.stap.enabled else ["stap.enabled"]
    stap_notes = ["platform velocity not needed by adaptive STAP; used only for interpretation"] if cc.stap.enabled else []
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
        report["stap"] = _capture_stap(Y[0], sched, cal, model, valid, cc, raxis, out if make_plots else None)
        if make_plots:
            files.append("stap_gmti_frame0.png")
    else:
        report["stap"] = "unavailable"
    json_dump(report, out / "capture_summary.json")
    return report


def _capture_stap(Y0, sched, cal, model, valid, cc, raxis, out_dir):
    from .experiments import gmti_detect

    s = cc.stap
    V = to_virtual_cube(Y0, sched)
    if cal is not None:
        V = V * cal[:, None, None]
    snaps = snapshots(V)
    direction = direction_from_angles(s.look_angle_deg)
    nu = model.doppler_grid(s.num_doppler_hypotheses)
    test = []
    for b in np.where(valid)[0]:
        try:
            select_training_bins(int(b), s.num_training, s.num_guard, valid)
            test.append(int(b))
        except ValueError:
            pass
    if not test:
        return {"error": "no range bins with enough training support"}
    res_all, gmti = {}, {}
    for name in s.methods:
        if s.loading_mode == "noise_relative":
            raise CaptureConfigError("capture STAP has no known noise level: use loading_mode noise_eig_relative/absolute/trace_relative")
        meth = build_method(name, model, s)
        try:
            res = run_gmti(snaps, model, meth, direction, nu, np.asarray(test), s.num_training, s.num_guard, valid)
        except ValueError as exc:
            res_all[name] = {"error": str(exc)}
            continue
        gd = gmti_detect(res, s)
        gmti[name] = gd
        res_all[name] = {
            "num_detections": len(gd.detections),
            "weight_seconds": res.weight_seconds,
            "detections": [
                {
                    "range_m": float(np.interp(d.range_bin_frac, np.arange(raxis.num_bins), raxis.range_m)),
                    "radial_velocity_mps": float(model.velocity_of(nu[int(d.doppler_bin_frac)])),
                    "output_snr_db": d.snr_estimate_db,
                }
                for d in gd.detections[:50]
            ],
        }
    if out_dir is not None and gmti:
        from . import plotting as pl

        pl.plot_gmti_maps(gmti, lambda b: np.interp(np.asarray(b, float), np.arange(raxis.num_bins), raxis.range_m),
                          model, None, Path(out_dir) / "stap_gmti_frame0.png")
    return {"look_angle_deg": s.look_angle_deg, "dimension": model.dimension, "methods": res_all}


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
