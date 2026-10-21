"""Cell-averaging CFAR detection (N-D, used as 2-D range-Doppler and 1-D range CFAR).

Threshold model
---------------
Under H0 each square-law cell of a complex-Gaussian (thermal-noise) process is
exponential with unknown mean mu. CA-CFAR estimates mu from N training cells
(excluding guard cells around the cell under test) and declares a detection if

    P_cut > alpha * mean(P_train),   alpha = N * (Pfa^(-1/N) - 1)

which gives *exactly* the design Pfa for i.i.d. exponential cells. The
threshold is only approximate when: training cells are correlated (window
leakage / zero padding / noncoherent sums over channels -> chi-square with
more degrees of freedom, which makes CA conservative), clutter is present or
non-homogeneous (clutter edges, the clutter ridge), or targets fall in the
training window (masking). Results must be interpreted accordingly; the Monte
Carlo evaluation in :mod:`radar_stap.experiments` measures the empirical Pfa.

Edge handling per axis: ``wrap`` (circular, natural for Doppler) or
``shrink`` (use only training cells inside the map; alpha recomputed per cell
from the actual count N).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


def cfar_alpha(num_training: np.ndarray | int, pfa: float) -> np.ndarray:
    """CA-CFAR multiplier for i.i.d. exponential cells (square-law detector)."""
    n = np.asarray(num_training, float)
    if not 0 < pfa < 1:
        raise ValueError("pfa must be in (0, 1)")
    with np.errstate(divide="ignore", invalid="ignore"):
        a = n * (pfa ** (-1.0 / n) - 1.0)
    return np.where(n > 0, a, np.inf)


def _pad(arr: np.ndarray, widths: list[int], modes: list[str], fill: float) -> np.ndarray:
    out = arr
    for ax, (w, m) in enumerate(zip(widths, modes)):
        if w == 0:
            continue
        pw = [(0, 0)] * arr.ndim
        pw[ax] = (w, w)
        if m == "wrap":
            out = np.pad(out, pw, mode="wrap")
        elif m == "shrink":
            out = np.pad(out, pw, mode="constant", constant_values=fill)
        else:
            raise ValueError(f"unknown edge mode {m!r}")
    return out


@dataclass
class CfarResult:
    detections: np.ndarray  # bool mask
    threshold: np.ndarray  # absolute threshold per cell
    noise_estimate: np.ndarray  # mean training power per cell
    num_training: np.ndarray  # training count per cell
    alpha: np.ndarray


def ca_cfar(
    power: np.ndarray,
    guard: list[int] | int,
    train: list[int] | int,
    pfa: float,
    edge_modes: list[str] | str = "shrink",
    exclude: np.ndarray | None = None,
) -> CfarResult:
    """N-D CA-CFAR on a square-law power map.

    ``guard``/``train`` are per-axis half-widths: the window spans
    ``guard + train`` cells either side, the inner ``guard`` region is excluded.
    ``exclude`` marks cells that must not be used as training (and are not tested).
    """
    P = np.asarray(power, float)
    nd = P.ndim
    guard = [guard] * nd if np.isscalar(guard) else list(guard)
    train = [train] * nd if np.isscalar(train) else list(train)
    modes = [edge_modes] * nd if isinstance(edge_modes, str) else list(edge_modes)
    if not (len(guard) == len(train) == len(modes) == nd):
        raise ValueError("guard/train/edge_modes must have one entry per axis")
    if np.any(P < 0) or not np.all(np.isfinite(P)):
        raise ValueError("power map must be finite and non-negative")
    outer = [g + t for g, t in zip(guard, train)]
    kern = np.ones([2 * o + 1 for o in outer])
    inner = tuple(slice(o - g, o + g + 1) for o, g in zip(outer, guard))
    kern[inner] = 0.0
    if kern.sum() == 0:
        raise ValueError("CFAR window has no training cells")
    usable = np.ones_like(P) if exclude is None else (~np.asarray(exclude, bool)).astype(float)
    Pp = _pad(P * usable, outer, modes, 0.0)
    Up = _pad(usable, outer, modes, 0.0)
    s = ndimage.correlate(Pp, kern, mode="constant", cval=0.0)
    n = ndimage.correlate(Up, kern, mode="constant", cval=0.0)
    crop = tuple(slice(o, o + d) for o, d in zip(outer, P.shape))
    s, n = s[crop], np.rint(n[crop])
    mean = np.where(n > 0, s / np.maximum(n, 1), np.inf)
    alpha = cfar_alpha(n, pfa)
    thr = alpha * mean
    det = P > thr
    if exclude is not None:
        det &= ~np.asarray(exclude, bool)
    return CfarResult(det, thr, mean, n.astype(int), alpha)


@dataclass
class Detection:
    indices: tuple[int, ...]  # peak cell
    power: float
    snr_estimate_db: float  # peak power / CFAR noise estimate
    cluster_size: int
    range_m: float | None = None
    velocity_mps: float | None = None
    angle_deg: float | None = None
    range_bin_frac: float | None = None
    doppler_bin_frac: float | None = None

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d["indices"] = [int(i) for i in self.indices]
        return d


def cluster_detections(
    cfar: CfarResult,
    power: np.ndarray,
    wrap_axes: tuple[int, ...] = (),
    method: str = "local_max",
    neighborhood: int = 1,
) -> list[Detection]:
    """Turn the CFAR mask into a detection list.

    ``local_max`` (default, "peak grouping"): every detected cell that is the
    maximum of the CFAR-normalised statistic P / noise_estimate within
    +-``neighborhood`` cells on every axis is a detection. Normalising first
    matters when neighbouring cells come from different filters (STAP Doppler
    hypotheses) or sit on a clutter skirt with a different background level. Two targets in adjacent resolution cells may merge; a weak
    target next to a large detected region (e.g. a clutter edge) is not lost.

    ``connected``: one detection per connected component of the mask (its
    strongest cell); clusters touching across a circular axis are merged.
    Simple, but a target connected to a large clutter region is swallowed.
    """
    mask = cfar.detections
    if not mask.any():
        return []
    P = np.asarray(power, float)
    out = []
    if method == "local_max":
        size = 2 * neighborhood + 1
        modes = ["wrap" if ax in wrap_axes else "nearest" for ax in range(P.ndim)]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(np.isfinite(cfar.noise_estimate) & (cfar.noise_estimate > 0), P / cfar.noise_estimate, 0.0)
        peak = ratio >= ndimage.maximum_filter(ratio, size=size, mode=modes)
        cnt = ndimage.uniform_filter(mask.astype(float), size=size, mode=modes) * size**P.ndim
        for idx in map(tuple, np.argwhere(mask & peak)):
            noise = cfar.noise_estimate[idx]
            out.append(
                Detection(
                    indices=tuple(int(i) for i in idx),
                    power=float(P[idx]),
                    snr_estimate_db=float(10 * np.log10(P[idx] / noise)) if noise > 0 else float("inf"),
                    cluster_size=int(round(cnt[idx])),
                )
            )
    elif method == "connected":
        structure = ndimage.generate_binary_structure(mask.ndim, mask.ndim)
        labels, _ = ndimage.label(mask, structure=structure)
        for ax in wrap_axes:  # merge labels across the wrap boundary
            first = np.take(labels, 0, axis=ax)
            last = np.take(labels, -1, axis=ax)
            for a, b in zip(first.ravel(), last.ravel()):
                if a and b and a != b:
                    labels[labels == b] = a
        for lab in np.unique(labels[labels > 0]):
            cells = np.argwhere(labels == lab)
            vals = P[tuple(cells.T)]
            k = int(np.argmax(vals))
            idx = tuple(int(i) for i in cells[k])
            noise = cfar.noise_estimate[idx]
            out.append(
                Detection(
                    indices=idx,
                    power=float(vals[k]),
                    snr_estimate_db=float(10 * np.log10(vals[k] / noise)) if noise > 0 else float("inf"),
                    cluster_size=int(len(cells)),
                )
            )
    else:
        raise ValueError(f"unknown clustering method {method!r}")
    out.sort(key=lambda d: -d.power)
    return out


def refine_to_local_power_peak(power: np.ndarray, idx: tuple[int, int], wrap_axis: int | None = 1,
                               max_steps: int = 3) -> tuple[int, int]:
    """Hill-climb from ``idx`` to the nearest local maximum of the *raw* power (2-D map).

    Detections are grouped on the CFAR-normalised statistic; parameter
    estimation (parabolic interpolation) must use the raw-power peak.
    """
    r, k = idx
    nr, nk = power.shape
    for _ in range(max_steps):
        best = (power[r, k], r, k)
        for dr in (-1, 0, 1):
            for dk in (-1, 0, 1):
                rr = r + dr
                kk = (k + dk) % nk if wrap_axis == 1 else k + dk
                if 0 <= rr < nr and 0 <= kk < nk and power[rr, kk] > best[0]:
                    best = (power[rr, kk], rr, kk)
        if (best[1], best[2]) == (r, k):
            break
        r, k = best[1], best[2]
    return r, k
