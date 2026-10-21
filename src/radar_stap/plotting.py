"""Diagnostic figures (matplotlib, Agg backend). Plotting is kept out of the numerical kernels."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .metrics import db  # noqa: E402


def _save(fig, path: Path | str | None, dpi: int = 120):
    if path is not None:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_range_profile(range_m, profile, truths=None, valid=None, path=None, dpi=120, title="Range profile"):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(range_m, db(profile), lw=0.9, color="#1f4e79")
    if valid is not None and not valid.all():
        lo, hi = range_m[valid].min(), range_m[valid].max()
        ax.axvspan(range_m.min(), lo, color="0.85")
        ax.axvspan(hi, range_m.max(), color="0.85")
    for t in truths or []:
        ax.axvline(t.range_m, color="#c0392b" if t.kind == "target" else "#7f8c8d", ls="--", lw=0.8)
        ax.text(t.range_m, ax.get_ylim()[1], t.name, rotation=90, va="top", ha="right", fontsize=7)
    ax.set_xlabel("Range [m]")
    ax.set_ylabel("Mean power [dB]")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return _save(fig, path, dpi)


def plot_range_doppler(rd, range_m, velocity_mps, detections=None, truths=None, path=None, dpi=120,
                       title="Range-Doppler map (noncoherent over channels)", dynamic_range_db=60):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    p = db(rd)
    vmax = p.max()
    im = ax.pcolormesh(velocity_mps, range_m, p, shading="nearest", vmin=vmax - dynamic_range_db, vmax=vmax, cmap="viridis")
    fig.colorbar(im, ax=ax, label="Power [dB]")
    for d in detections or []:
        ax.plot(d.velocity_mps, d.range_m, "o", mfc="none", mec="w", ms=7, mew=1.2)
    for t in truths or []:
        ax.plot(t.radial_velocity_mps, t.range_m, "x", color="#e74c3c", ms=8, mew=1.5)
        ax.annotate(t.name, (t.radial_velocity_mps, t.range_m), color="w", fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Radial velocity v_r = dR/dt [m/s] (positive receding)")
    ax.set_ylabel("Range [m]")
    ax.set_title(title + "\n(o = CFAR detection, x = ground truth)")
    return _save(fig, path, dpi)


def plot_angle_doppler(R, model, path=None, dpi=120, title="Angle-Doppler (Fourier) spectrum of training covariance"):
    """P(theta, nu) = s^H R s / (s^H s): shows the clutter ridge."""
    from .angle_processing import direction_from_angles

    ang = np.linspace(-85, 85, 171)
    nu = model.doppler_grid(64)
    P = np.zeros((nu.size, ang.size))
    for i, a in enumerate(ang):
        S = model.steering(direction_from_angles(a), nu)
        P[:, i] = np.real(np.einsum("dj,de,ej->j", S.conj(), R, S)) / model.dimension
    fig, ax = plt.subplots(figsize=(7, 5))
    p = db(P)
    im = ax.pcolormesh(ang, model.velocity_of(nu), p, shading="nearest", vmin=p.max() - 50, vmax=p.max(), cmap="magma")
    fig.colorbar(im, ax=ax, label="Power [dB]")
    ax.set_xlabel("Angle from array normal [deg]")
    ax.set_ylabel("Radial velocity [m/s]")
    ax.set_title(title)
    return _save(fig, path, dpi)


def plot_eigenspectrum(clairvoyant, sample, brennan, noise_level=None, path=None, dpi=120):
    fig, ax = plt.subplots(figsize=(7, 4))
    k = np.arange(1, clairvoyant.size + 1)
    ax.plot(k, db(clairvoyant), "o-", ms=3, label="clairvoyant (scatterer model)")
    ax.plot(k, db(np.maximum(sample, 1e-12)), "s-", ms=3, alpha=0.7, label="sample covariance (training)")
    ax.axvline(brennan + 0.5, color="k", ls="--", lw=0.8, label=f"Brennan rank ~ {brennan}")
    if noise_level is not None:
        ax.axhline(db(noise_level), color="0.5", ls=":", label="noise level")
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue [dB]")
    ax.set_title("Space-time covariance eigenvalues at the target range bin")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, path, dpi)


def plot_sinr_loss(analysis, model, path=None, dpi=120):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    v = model.velocity_of(analysis.nu_grid)
    order = np.argsort(v)
    for name, loss in analysis.sinr_loss_db.items():
        ax.plot(v[order], loss[order], ".-", label=name)
    ax.set_xlabel("Hypothesised radial velocity [m/s]")
    ax.set_ylabel("SINR loss [dB] (rel. noise-only matched filter)")
    ax.set_title(f"SINR loss vs Doppler at the look angle, range bin {analysis.range_bin} ({analysis.target})")
    ax.set_ylim(bottom=max(-70, ax.get_ylim()[0]))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, path, dpi)


def plot_gmti_maps(gmti: dict, range_m_of_bin, model, truths_cells=None, path=None, dpi=120):
    names = list(gmti)
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        gd = gmti[name]
        res = gd.result
        v = model.velocity_of(res.nu_grid)
        o = np.argsort(v)
        p = db(res.power[:, o])
        p -= np.median(p)
        r = range_m_of_bin(res.test_bins)
        im = ax.pcolormesh(v[o], r, p, shading="nearest", vmin=-5, vmax=35, cmap="viridis")
        rr, jj = np.nonzero(gd.cfar_mask)
        ax.plot(v[jj], r[rr], "w.", ms=2)
        for cell in truths_cells or []:
            j = int(round(cell[1])) % res.nu_grid.size
            ax.plot(model.velocity_of(res.nu_grid[j]), range_m_of_bin(np.array([cell[0]]))[0], "rx", ms=9, mew=1.5)
        ax.set_title(name)
        ax.set_xlabel("v_r hypothesis [m/s]")
    axes[0].set_ylabel("Range [m]")
    fig.colorbar(im, ax=axes.tolist(), label="Output power rel. median [dB]")
    fig.suptitle("Angle-gated GMTI output (white = CFAR cells, red x = in-beam truth)")
    return _save(fig, path, dpi)


def plot_pd_pfa(mc: dict, path=None, dpi=120):
    names = list(mc["methods"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, key, label in ((axes[0], "pd", "P_D"), (axes[1], "pfa", "P_FA (per cell)")):
        vals = np.array([mc["methods"][n][key] for n in names])
        ci = np.array([mc["methods"][n][key + "_95ci"] for n in names])
        ax.bar(names, vals, color="#2e86c1", alpha=0.8)
        ax.errorbar(names, vals, yerr=[vals - ci[:, 0], ci[:, 1] - vals], fmt="none", ecolor="k", capsize=4)
        if key == "pfa":
            ax.axhline(mc["methods"][names[0]]["pfa_design"], color="r", ls="--", label="design")
            ax.set_yscale("log")
            ax.legend(fontsize=8)
        ax.set_ylabel(label)
        ax.set_title(f"{label}, {mc['trials']} trials, 95% Wilson CI")
    return _save(fig, path, dpi)


def plot_waveform(radar, t_f, if_signal, path=None, dpi=120):
    from .waveforms import instantaneous_frequency

    fig, axes = plt.subplots(2, 1, figsize=(8, 5))
    axes[0].plot(t_f * 1e6, (instantaneous_frequency(radar, t_f) - radar.carrier_frequency_hz) / 1e6)
    axes[0].set_ylabel("f - f_c [MHz]")
    axes[0].set_title("Transmitted chirp instantaneous frequency (ADC window)")
    axes[1].plot(t_f * 1e6, np.real(if_signal), lw=0.8, label="Re")
    axes[1].plot(t_f * 1e6, np.imag(if_signal), lw=0.8, label="Im")
    axes[1].set_xlabel("Fast time from ramp start [us]")
    axes[1].set_ylabel("IF amplitude")
    axes[1].set_title("Dechirped (beat) signal, channel 0, chirp 0")
    axes[1].legend(fontsize=8)
    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, path, dpi)
