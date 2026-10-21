"""Range-compressed per-scatterer responses: fast Monte Carlo and clairvoyant covariance.

The range-compressed space-time snapshot at range bin r is *linear* in the
scatterer coefficients:

    x_r = sum_s c_s g_s(r) + n_r

where g_s(r) is the (window + FFT + virtual-channel) response of scatterer s
with unit coefficient, computed from exactly the same raw-signal model as the
simulator (:func:`radar_stap.simulator.unit_raw_responses`). Hence

* with independent zero-mean reflectivities, the exact interference covariance
  is  R(r) = sum_s E|c_s|^2 g_s(r) g_s(r)^H + sigma^2 * sum(w^2) * I;
* new clutter/noise realisations for Monte Carlo only need new c_s and noise,
  not a new raw simulation.

Each scatterer's response is stored only for the ``2*halfwidth+1`` bins around
its peak (window sidelobes beyond that are dropped; with a Hann window and
halfwidth 16 the dropped energy is below about -70 dB of the peak). Only
complex sampling is supported here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SimulationConfig
from .datacube import to_virtual_cube
from .range_processing import range_fft
from .simulator import RadarGeometry, ScattererSet, mixer_amplitude, unit_raw_responses


@dataclass
class ResponseBasis:
    start_bins: np.ndarray  # (S,) first stored bin per scatterer
    responses: np.ndarray  # (S, 2W+1, D) complex
    powers: np.ndarray  # (S,) E|c_s|^2
    coefficients: np.ndarray  # (S,) the scene's actual coefficients c_s
    halfwidth: int
    num_bins: int

    @property
    def size(self) -> int:
        return int(self.start_bins.size)

    @property
    def dimension(self) -> int:
        return int(self.responses.shape[2])

    def synthesize(self, coefficients: np.ndarray | None, b0: int, b1: int) -> np.ndarray:
        """Snapshots (b1-b0, D) for bins [b0, b1) for the given coefficients."""
        c = self.coefficients if coefficients is None else np.asarray(coefficients)
        out = np.zeros((b1 - b0, self.dimension), complex)
        if self.size == 0:
            return out
        for o in range(self.responses.shape[1]):
            bins = self.start_bins + o
            m = (bins >= b0) & (bins < b1)
            if m.any():
                np.add.at(out, bins[m] - b0, c[m, None] * self.responses[m, o])
        return out

    def covariance(self, bin_index: int) -> np.ndarray:
        """sum_s E|c_s|^2 g_s g_s^H at one bin (no noise term)."""
        D = self.dimension
        o = bin_index - self.start_bins
        m = (o >= 0) & (o < self.responses.shape[1])
        if not m.any():
            return np.zeros((D, D), complex)
        G = self.responses[m, o[m]]  # (S', D)
        return (G.T * self.powers[m]) @ G.conj()

    def response_at(self, index: int, bin_index: int) -> np.ndarray:
        o = bin_index - self.start_bins[index]
        if not 0 <= o < self.responses.shape[1]:
            return np.zeros(self.dimension, complex)
        return self.responses[index, o]


def build_basis(
    scat: ScattererSet,
    cfg: SimulationConfig,
    geom: RadarGeometry | None = None,
    bin_region: tuple[int, int] | None = None,
    halfwidth: int = 16,
    chunk: int = 64,
) -> ResponseBasis:
    """Compute sparse range-compressed responses for scatterers affecting ``bin_region``."""
    if cfg.radar.sampling != "complex":
        raise NotImplementedError("response basis supports complex sampling only")
    geom = geom or RadarGeometry.from_config(cfg)
    W2 = 2 * halfwidth + 1
    starts, resp, keep = [], [], []
    nb = None
    for s0 in range(0, scat.size, chunk):
        idx = np.arange(s0, min(s0 + chunk, scat.size))
        raw = unit_raw_responses(scat.subset(idx), cfg, geom)  # (cs, M, N, K)
        Y, axis = range_fft(raw, cfg.radar, cfg.processing)  # (cs, M, N, nb)
        nb = Y.shape[-1]
        if W2 > nb:
            raise ValueError("halfwidth too large for the number of range bins")
        for i, s in enumerate(idx):
            V = to_virtual_cube(Y[i], cfg.array.tx_schedule)  # (C, L, nb)
            pk = int(np.argmax((np.abs(V) ** 2).sum(axis=(0, 1))))
            st = int(np.clip(pk - halfwidth, 0, nb - W2))
            if bin_region is not None and (st + W2 <= bin_region[0] or st >= bin_region[1]):
                continue
            starts.append(st)
            keep.append(s)
            resp.append(V[:, :, st : st + W2].transpose(2, 1, 0).reshape(W2, -1))
    keep = np.asarray(keep, int)
    if nb is None:
        nb = cfg.processing.range_nfft or cfg.radar.num_adc_samples
    D = len(cfg.array.rx_positions) * cfg.radar.num_chirps  # C * L = M * N
    coef = mixer_amplitude(scat.amplitudes, cfg.radar.mixer_convention)
    return ResponseBasis(
        start_bins=np.asarray(starts, int),
        responses=np.asarray(resp, np.complex128).reshape(len(starts), W2, D) if resp else np.zeros((0, W2, D)),
        powers=scat.powers[keep] if keep.size else np.zeros(0),
        coefficients=coef[keep] if keep.size else np.zeros(0, complex),
        halfwidth=halfwidth,
        num_bins=int(nb),
    )


def noise_covariance_scale(cfg: SimulationConfig) -> float:
    """Per-element noise variance in the range-compressed domain: sigma^2 * sum(w^2)."""
    from .range_processing import get_window

    w = get_window(cfg.processing.range_window, cfg.radar.num_adc_samples)
    return float(cfg.radar.noise_power * (w**2).sum())
