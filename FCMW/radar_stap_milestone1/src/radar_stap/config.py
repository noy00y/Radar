from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True)
class RadarConfig:
    carrier_frequency_hz: float
    slope_hz_per_s: float
    adc_sample_rate_hz: float
    samples_per_chirp: int
    chirp_repetition_interval_s: float
    num_chirps: int
    num_rx: int = 1

    @property
    def wavelength_m(self) -> float:
        return 299_792_458.0 / self.carrier_frequency_hz

    @property
    def range_resolution_m(self) -> float:
        # Approximation using swept bandwidth over the sampled chirp.
        bandwidth_hz = self.slope_hz_per_s * (
            self.samples_per_chirp / self.adc_sample_rate_hz
        )
        return 299_792_458.0 / (2.0 * bandwidth_hz)

    @property
    def nominal_range_bin_spacing_m(self) -> float:
        # FFT-bin spacing; zero-padding can make plotted bins denser,
        # but does not improve physical resolution.
        return (
            299_792_458.0 * self.adc_sample_rate_hz
            / (2.0 * self.slope_hz_per_s * self.samples_per_chirp)
        )


def load_config(path: str | Path) -> tuple[RadarConfig, dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    r = raw["radar"]
    cfg = RadarConfig(
        carrier_frequency_hz=float(r["carrier_frequency_hz"]),
        slope_hz_per_s=float(r["slope_hz_per_s"]),
        adc_sample_rate_hz=float(r["adc_sample_rate_hz"]),
        samples_per_chirp=int(r["samples_per_chirp"]),
        chirp_repetition_interval_s=float(r["chirp_repetition_interval_s"]),
        num_chirps=int(r["num_chirps"]),
        num_rx=int(r.get("num_rx", 1)),
    )
    return cfg, raw.get("scene", {})
