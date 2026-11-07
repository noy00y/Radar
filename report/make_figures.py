#!/usr/bin/env python3
"""Figures for the interim report (report/interim_report.tex).

Everything here is either (a) computed analytically from the two experiment YAML configs,
(b) a measured number recorded in the report, or (c) a copy of a real bench-capture output
produced by scripts/process_capture.py. Nothing is re-simulated.

    python report/make_figures.py          # writes report/figures/*.png and prints derived values
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:  # executed without __file__ (e.g. pasted into a notebook)
    ROOT = Path("/Workspace/Users/khanoza1@cibcpte.com/Repos/test")
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
C0 = 299792458.0

plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.grid": True, "grid.alpha": 0.3,
                     "savefig.dpi": 300, "savefig.bbox": "tight", "legend.fontsize": 7})
COL_W = 3.5  # IEEE single column width [in]


def derived(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    w, f, p = cfg["waveform"], cfg["adc_format"], cfg["processing"]
    S, fs, N = w["chirp_slope_hz_per_s"], w["adc_sample_rate_hz"], f["num_adc_samples"]
    L = f["chirps_per_frame"] // len(cfg["tx_schedule"])
    tc = w["idle_time_s"] + w["ramp_end_time_s"]
    t_slow = len(cfg["tx_schedule"]) * tc
    f_centre = w["carrier_frequency_hz"] + S * (w["adc_start_time_s"] + N / (2 * fs))  # centre of sampled sweep
    lam = C0 / f_centre
    B = S * N / fs
    st = cfg["stap"]
    C = len(cfg["antenna"]["rx_positions"]) * len(cfg["tx_schedule"])
    D = C * L
    samples = st["num_training"] * st.get("training_frames", 1)
    frame_bytes = L * len(cfg["tx_schedule"]) * f["num_rx"] * N * 4
    return {
        "name": path.stem, "S": S, "fs": fs, "N": N, "L": L, "tc": tc, "t_slow": t_slow, "lam": lam, "B": B,
        "range_res": C0 / (2 * B), "bin": C0 * fs / (2 * S * p["range_nfft"]), "r_max": C0 * fs / (2 * S),
        "v_max": lam / (4 * t_slow), "v_res": lam / (2 * L * t_slow), "cpi": L * t_slow, "D": D,
        "samples": samples, "samples_fb": samples * (2 if st.get("forward_backward") else 1),
        "frame_period": cfg["studio"]["frame"]["period_s"], "num_frames": f["num_frames"],
        "frame_bytes": frame_bytes, "total_mb": frame_bytes * f["num_frames"] / 1e6,
        "v_platform": cfg.get("platform_velocity_mps"),
        "looks": st.get("look_angles_deg"), "hyp": st["num_doppler_hypotheses"],
        "test_rng": (st["test_range_min_m"], st["test_range_max_m"]),
    }


EXP1 = derived(ROOT / "configs" / "radar_capture_highway_overlook.yaml")
EXP2 = derived(ROOT / "configs" / "radar_capture_vehicle_highway.yaml")


def print_table():
    keys = [("S [MHz/us]", lambda d: d["S"] / 1e12), ("T_c [us]", lambda d: d["tc"] * 1e6),
            ("lambda [mm]", lambda d: d["lam"] * 1e3), ("B [MHz]", lambda d: d["B"] / 1e6),
            ("range res [m]", lambda d: d["range_res"]), ("bin [m]", lambda d: d["bin"]),
            ("R_max [m]", lambda d: d["r_max"]), ("v_max [m/s]", lambda d: d["v_max"]),
            ("v_res [m/s]", lambda d: d["v_res"]), ("CPI [ms]", lambda d: d["cpi"] * 1e3),
            ("D", lambda d: d["D"]), ("train", lambda d: d["samples"]), ("train FB", lambda d: d["samples_fb"]),
            ("kB/frame", lambda d: d["frame_bytes"] / 1e3), ("MB total", lambda d: d["total_mb"])]
    print(f"{'quantity':16s} {'exp1 overlook':>14s} {'exp2 vehicle':>14s}")
    for k, fn in keys:
        print(f"{k:16s} {fn(EXP1):14.4g} {fn(EXP2):14.4g}")
    vp = np.array([50, 60, 80, 100]) / 3.6
    print("exp2 beta = 4 v_p T_c / lambda:", dict(zip([50, 60, 80, 100], np.round(4 * vp * EXP2["t_slow"] / EXP2["lam"], 3))))


# ---------------------------------------------------------------------------
# Fig. 2: waveform design space
# ---------------------------------------------------------------------------
def fig_waveform():
    lam = EXP1["lam"]
    tc = np.linspace(15e-6, 80e-6, 400)
    fig, ax = plt.subplots(1, 2, figsize=(COL_W * 2, 2.3))
    ax[0].plot(tc * 1e6, lam / (4 * tc) * 3.6, "k", label="1 TX: $v_{max}=\\lambda/4T_c$")
    ax[0].plot(tc * 1e6, lam / (8 * tc) * 3.6, "k--", label="2-TX TDM: $\\lambda/8T_c$")
    ax[0].axhline(100, color="C0", lw=1, label="Exp. 2 platform max (100 km/h)")
    ax[0].axhline(130, color="C1", lw=1, label="Exp. 1 traffic max (130 km/h)")
    ax[0].axvline(EXP1["tc"] * 1e6, color="C3", lw=1, ls=":")
    ax[0].annotate(f"chosen $T_c$ = {EXP1['tc'] * 1e6:.1f} $\\mu$s\n$v_{{max}}$ = {EXP1['v_max'] * 3.6:.0f} km/h",
                   (EXP1["tc"] * 1e6, EXP1["v_max"] * 3.6), xytext=(45, 190), arrowprops={"arrowstyle": "->"})
    ax[0].set(xlabel="Chirp period $T_c$ [$\\mu$s]", ylabel="Unambiguous radial speed [km/h]", ylim=(0, 260),
              title="(a) Speed limit vs chirp period")
    ax[0].legend(loc="upper right")
    t_adc = np.linspace(5e-6, 30e-6, 300)
    for d, lab, c in ((EXP1, "Exp. 1, S = 4.49 MHz/$\\mu$s", "C1"), (EXP2, "Exp. 2, S = 11.01 MHz/$\\mu$s", "C0")):
        ax[1].plot(t_adc * 1e6, C0 / (2 * d["S"] * t_adc), color=c, label=lab)
        ax[1].plot([d["N"] / d["fs"] * 1e6], [d["range_res"]], "o", color=c)
        ax[1].annotate(f"{d['range_res']:.2f} m", (d["N"] / d["fs"] * 1e6, d["range_res"]), xytext=(4, 4),
                       textcoords="offset points")
    ax[1].axvline(24, color="C3", lw=1, ls=":")
    ax[1].text(24.4, 4.2, "24 $\\mu$s ADC window\n(fits $T_c$ = 33.5 $\\mu$s)", fontsize=7)
    ax[1].set(xlabel="ADC sampling window $N/f_s$ [$\\mu$s]", ylabel="Range resolution $c/2B$ [m]",
              ylim=(0, 5), title="(b) Range resolution vs sampling window")
    ax[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "fig_waveform_design.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig. 3: predicted clutter in angle-velocity space
# ---------------------------------------------------------------------------
def fig_clutter_ridge():
    th = np.linspace(-90, 90, 361)
    vmax = EXP2["v_max"]
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    ax.axhspan(vmax, 40, color="0.85")
    ax.axhspan(-40, -vmax, color="0.85")
    ax.text(-85, vmax + 2, "aliased ($|v_r| > v_{max}$)", fontsize=7)
    ax.plot(th, 0 * th, "k", lw=2, label="Exp. 1 (static): all clutter at 0 m/s")
    for kmh, c in ((50, "C2"), (80, "C0"), (100, "C3")):
        vp = kmh / 3.6
        beta = 4 * vp * EXP2["t_slow"] / EXP2["lam"]
        ax.plot(th, -vp * np.sin(np.radians(th)), color=c, label=f"Exp. 2, {kmh} km/h ($\\beta$ = {beta:.2f})")
    ax.axvspan(-45, 45, color="C0", alpha=0.06)
    ax.text(-43, -36, "look-angle sweep $\\pm45^\\circ$", fontsize=7)
    ax.set(xlabel="Angle from boresight $\\theta$ [deg]", ylabel="Clutter radial velocity [m/s]",
           xlim=(-90, 90), ylim=(-40, 40), title="Clutter ridge $v_r = -v_p\\sin\\theta$ (side-looking)")
    ax.legend(loc="upper right", fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "fig_clutter_ridge.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig. 5: sample support (Reed-Mallett-Brennan)
# ---------------------------------------------------------------------------
def fig_rmb():
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    pts = [("Bench (3 TX)", 1536, 64, "C7"), ("Exp. 1", EXP1["D"], EXP1["samples_fb"], "C1"),
           ("Exp. 2", EXP2["D"], EXP2["samples_fb"], "C0")]
    for lab, D, _, c in pts:
        L = np.logspace(np.log10(D), 4, 300)
        ax.semilogx(L, 10 * np.log10((L + 2 - D) / (L + 1)), color=c, label=f"D = {D}")
        ax.axvline(2 * D, color=c, lw=0.6, ls=":")
    for lab, D, L, c in pts:
        if L < D:
            ax.annotate(f"{lab}: {L} samples < D\n(full-rank infeasible)", (D, -9.5), xytext=(70, -8.7), color=c,
                        fontsize=7, arrowprops={"arrowstyle": "->", "color": c})
        else:
            y = 10 * np.log10((L + 2 - D) / (L + 1))
            ax.plot([L], [y], "o", color=c)
            ax.annotate(f"{lab}: {L} samples\n({y:.1f} dB)", (L, y), xytext=(6, -16), textcoords="offset points",
                        color=c, fontsize=7)
    ax.set(xlabel="Training samples $K$ (after multi-frame pooling and FB averaging)",
           ylabel="Expected SINR loss [dB]", ylim=(-10, 0.3), xlim=(50, 1e4),
           title="RMB loss $(K+2-D)/(K+1)$; dotted: $K = 2D$")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig_sample_support.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig. 6: measured STAP weight cost (Exp. 1 sizes)
# ---------------------------------------------------------------------------
# Measured 2026-10-30 on Databricks serverless (4 vCPU, NumPy/OpenBLAS, complex128):
# D = 512, 9 looks x 128 hypotheses = 1152 steering columns, 1152 snapshots (x2 FB).
MEASURED_MS = {"Non-adaptive": 15.6, "EFA (Q=3)": 159.0, "PC (full eigh)": 1411.0, "MVDR (full rank)": 1952.0}
MVDR_BREAKDOWN_MS = {"covariance + FB": 74.0, "eigvalsh (loading,\ndiagnostics)": 472.0, "Cholesky solve\n(1152 columns)": 223.0}
TEST_BINS, FRAMES = 153, 12


def fig_compute():
    fig, ax = plt.subplots(1, 2, figsize=(COL_W * 2, 2.2), gridspec_kw={"width_ratios": [1.3, 1]})
    names, ms = list(MEASURED_MS), np.array(list(MEASURED_MS.values()))
    bars = ax[0].barh(names, ms, color=["C7", "C2", "C1", "C3"])
    ax[0].set_xscale("log")
    for b, m in zip(bars, ms):
        run_min = m * TEST_BINS * FRAMES / 6e4
        ax[0].text(m * 1.15, b.get_y() + b.get_height() / 2, f"{m:.0f} ms/bin  ->  {run_min:.1f} min / 12 frames",
                   va="center", fontsize=7)
    ax[0].set(xlabel="Weight computation per range bin [ms] (log)", xlim=(5, 3e5),
              title=f"(a) Exp. 1 STAP cost, D = {EXP1['D']}, {TEST_BINS} test bins")
    ax[1].bar(range(len(MVDR_BREAKDOWN_MS)), list(MVDR_BREAKDOWN_MS.values()), color="C3")
    ax[1].set_xticks(range(len(MVDR_BREAKDOWN_MS)), list(MVDR_BREAKDOWN_MS), fontsize=6.5)
    ax[1].set(ylabel="Time per bin [ms]", title="(b) MVDR components (timed separately)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_stap_compute.png")
    plt.close(fig)


def copy_bench():
    src = ROOT / "outputs" / "capture_001"
    for name in ("range_profile.png", "range_doppler_mean.png"):
        if (src / name).exists():
            shutil.copy(src / name, FIG / f"bench001_{name}")
        else:
            print(f"WARNING: {src / name} missing", file=sys.stderr)


if __name__ == "__main__":
    print_table()
    fig_waveform()
    fig_clutter_ridge()
    fig_rmb()
    fig_compute()
    copy_bench()
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))
