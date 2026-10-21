"""Parser for TI mmWave SDK CLI configuration files (``.cfg``), e.g. used with the IWR1843BOOST.

Extracts the *waveform* and *channel* settings (profileCfg, chirpCfg,
frameCfg, channelCfg, adcCfg, adcbufCfg) following the argument order of the
mmWave SDK CLI (verify against the user guide of your SDK version; argument
lists differ between SDK releases for some commands).

What this does NOT determine: the byte layout of the file written by the
DCA1000 (LVDS lane count, I/Q order in the LVDS stream, interleaving written by
mmWave Studio / DCA1000 CLI). ``adcbufCfg`` describes the on-chip ADC buffer and
is reported only as a *hint*; the capture's ``adc_format`` must still be
specified and verified explicitly.

Argument order used (mmWave SDK 3.x):
  profileCfg  id startFreq[GHz] idleTime[us] adcStartTime[us] rampEndTime[us] txOutPower
              txPhaseShifter freqSlopeConst[MHz/us] txStartTime[us] numAdcSamples
              digOutSampleRate[ksps] hpfCornerFreq1 hpfCornerFreq2 rxGain
  chirpCfg    startIdx endIdx profileId startFreqVar freqSlopeVar idleTimeVar adcStartTimeVar txEnableMask
  frameCfg    chirpStartIdx chirpEndIdx numLoops numFrames framePeriodicity[ms] triggerSelect frameTriggerDelay[ms]
  channelCfg  rxChannelEn txChannelEn cascading
  adcCfg      numADCBits(0:12,1:14,2:16) adcOutputFmt(0:real,1:complex1x,2:complex2x)
  adcbufCfg   subFrameIdx adcOutputFmt(0:complex,1:real) sampleSwap chanInterleave chirpThreshold
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


class MmwaveCfgError(ValueError):
    pass


@dataclass
class MmwaveCliConfig:
    profiles: dict[int, dict] = field(default_factory=dict)
    chirps: list[dict] = field(default_factory=list)
    frame: dict | None = None
    channel: dict | None = None
    adc: dict | None = None
    adcbuf: dict | None = None
    unparsed: list[str] = field(default_factory=list)


def _nums(parts: list[str], n: int, cmd: str) -> list[float]:
    if len(parts) < n:
        raise MmwaveCfgError(f"{cmd}: expected at least {n} arguments, got {len(parts)}")
    try:
        return [float(p) for p in parts[:n]]
    except ValueError as exc:
        raise MmwaveCfgError(f"{cmd}: non-numeric argument in {parts}") from exc


def parse_mmwave_cfg(text: str) -> MmwaveCliConfig:
    cfg = MmwaveCliConfig()
    for raw in text.splitlines():
        line = raw.split("%", 1)[0].strip()
        if not line:
            continue
        cmd, *args = line.split()
        if cmd == "profileCfg":
            v = _nums(args, 11, cmd)
            cfg.profiles[int(v[0])] = {
                "start_freq_hz": v[1] * 1e9,
                "idle_time_s": v[2] * 1e-6,
                "adc_start_time_s": v[3] * 1e-6,
                "ramp_end_time_s": v[4] * 1e-6,
                "slope_hz_per_s": v[7] * 1e12,
                "tx_start_time_s": v[8] * 1e-6,
                "num_adc_samples": int(v[9]),
                "sample_rate_hz": v[10] * 1e3,
            }
        elif cmd == "chirpCfg":
            v = _nums(args, 8, cmd)
            cfg.chirps.append(
                {"start": int(v[0]), "end": int(v[1]), "profile": int(v[2]), "tx_mask": int(v[7]),
                 "variations_nonzero": any(x != 0 for x in v[3:7])}
            )
        elif cmd == "frameCfg":
            v = _nums(args, 5, cmd)
            cfg.frame = {"start": int(v[0]), "end": int(v[1]), "loops": int(v[2]), "frames": int(v[3]),
                         "period_s": v[4] * 1e-3}
        elif cmd == "channelCfg":
            v = _nums(args, 2, cmd)
            cfg.channel = {"rx_mask": int(v[0]), "tx_mask": int(v[1])}
        elif cmd == "adcCfg":
            v = _nums(args, 2, cmd)
            cfg.adc = {"bits_code": int(v[0]), "fmt_code": int(v[1])}
        elif cmd == "adcbufCfg":
            off = 1 if len(args) >= 5 else 0  # SDK 3.x has subFrameIdx first
            v = _nums(args, 3 + off, cmd)
            cfg.adcbuf = {"fmt_code": int(v[off]), "sample_swap": int(v[off + 1]), "chan_interleave": int(v[off + 2])}
        else:
            cfg.unparsed.append(cmd)
    return cfg


def _bits(mask: int) -> list[int]:
    return [i for i in range(16) if mask >> i & 1]


def capture_settings_from_cli(cli: MmwaveCliConfig) -> dict:
    """Derive waveform + channel settings. Raises if the config is incomplete or ambiguous."""
    if not cli.profiles:
        raise MmwaveCfgError("no profileCfg found")
    if cli.frame is None or cli.channel is None:
        raise MmwaveCfgError("frameCfg and channelCfg are required")
    used = [c for c in cli.chirps if not (c["end"] < cli.frame["start"] or c["start"] > cli.frame["end"])]
    profiles = {c["profile"] for c in used}
    if len(profiles) != 1:
        raise MmwaveCfgError(f"frame uses {len(profiles)} profiles; a single profile is required")
    if any(c["variations_nonzero"] for c in used):
        raise MmwaveCfgError("per-chirp start-frequency/slope/time variations are not supported")
    prof = cli.profiles[profiles.pop()]
    schedule = []
    for idx in range(cli.frame["start"], cli.frame["end"] + 1):
        c = next((c for c in used if c["start"] <= idx <= c["end"]), None)
        if c is None:
            raise MmwaveCfgError(f"chirp index {idx} of frameCfg has no chirpCfg")
        txs = _bits(c["tx_mask"])
        if len(txs) != 1:
            raise MmwaveCfgError(f"chirp {idx} enables TX {txs}: simultaneous TX (BPM/beamforming) not supported")
        schedule.append(txs[0])
    if len(set(schedule)) != len(schedule):
        raise MmwaveCfgError(f"TX schedule {schedule} repeats a TX within one loop")
    rx = _bits(cli.channel["rx_mask"])
    out = {
        "waveform": {
            "carrier_frequency_hz": prof["start_freq_hz"],
            "chirp_slope_hz_per_s": prof["slope_hz_per_s"],
            "idle_time_s": prof["idle_time_s"],
            "adc_start_time_s": prof["adc_start_time_s"],
            "ramp_end_time_s": prof["ramp_end_time_s"],
            "adc_sample_rate_hz": prof["sample_rate_hz"],
        },
        "num_adc_samples": prof["num_adc_samples"],
        "num_rx": len(rx),
        "rx_enabled": rx,
        "tx_schedule": schedule,
        "chirps_per_frame": len(schedule) * cli.frame["loops"],
        "num_frames_configured": cli.frame["frames"],
        "frame_period_s": cli.frame["period_s"],
        "hints": {},
    }
    if cli.adc is not None:
        out["adc_bits"] = {0: 12, 1: 14, 2: 16}.get(cli.adc["bits_code"])
        out["data_type"] = {0: "real", 1: "complex", 2: "complex"}.get(cli.adc["fmt_code"])
        if cli.adc["fmt_code"] == 2:
            out["hints"]["complex2x"] = "complex2x output: image-band behaviour differs from complex1x; verify"
    if cli.adcbuf is not None:
        out["hints"]["adcbuf"] = {
            "note": "on-chip ADC buffer layout; NOT necessarily the DCA1000 file layout",
            **cli.adcbuf,
        }
    if prof["tx_start_time_s"] < 0:
        out["hints"]["tx_start"] = "negative txStartTime"
    return out


def load_mmwave_cfg(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    return capture_settings_from_cli(parse_mmwave_cfg(p.read_text(encoding="utf-8", errors="replace")))


def write_mmwave_cfg(waveform: dict, num_adc_samples: int, rx_mask: int, tx_schedule: list[int], loops: int,
                     frames: int, frame_period_ms: float, adc_bits: int = 16, complex_out: bool = True) -> str:
    """Emit a minimal SDK-style .cfg for a synthetic capture (consumed by :func:`parse_mmwave_cfg`)."""
    tx_mask = 0
    for t in tx_schedule:
        tx_mask |= 1 << t
    lines = [
        "% synthetic configuration generated by radar_stap (not from hardware)",
        "sensorStop",
        "flushCfg",
        "dfeDataOutputMode 1",
        f"channelCfg {rx_mask} {tx_mask} 0",
        f"adcCfg {({12: 0, 14: 1, 16: 2})[adc_bits]} {1 if complex_out else 0}",
        f"adcbufCfg -1 {0 if complex_out else 1} 1 1 1",
        "profileCfg 0 {:.6f} {:.3f} {:.3f} {:.3f} 0 0 {:.6f} 1 {} {:.1f} 0 0 30".format(
            waveform["carrier_frequency_hz"] / 1e9,
            waveform["idle_time_s"] * 1e6,
            waveform["adc_start_time_s"] * 1e6,
            waveform["ramp_end_time_s"] * 1e6,
            waveform["chirp_slope_hz_per_s"] / 1e12,
            num_adc_samples,
            waveform["adc_sample_rate_hz"] / 1e3,
        ),
    ]
    for i, t in enumerate(tx_schedule):
        lines.append(f"chirpCfg {i} {i} 0 0 0 0 0 {1 << t}")
    lines.append(f"frameCfg 0 {len(tx_schedule) - 1} {loops} {frames} {frame_period_ms:.3f} 1 0")
    lines.append("sensorStart")
    return "\n".join(lines) + "\n"
