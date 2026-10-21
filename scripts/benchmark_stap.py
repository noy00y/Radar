#!/usr/bin/env python3
"""Benchmark STAP weight computation vs space-time dimension (CPU; CuPy if available).

    python scripts/benchmark_stap.py --channels 4 8 --pulses 8 16 32 --bins 64
    python scripts/benchmark_stap.py --backend cupy      # RTX 3060, if CuPy is installed

Reports wall time per range bin for full-rank MVDR, EFA (post-Doppler) and the
batched backend kernel, plus agreement of the backend result with the CPU
reference. Synthetic data: random clutter-like low-rank + noise.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from radar_stap.angle_processing import direction_from_angles  # noqa: E402
from radar_stap.backend import BackendUnavailable, batched_mvdr, get_array_module, to_numpy  # noqa: E402
from radar_stap.stap import AdjacentBinPostDoppler, FullRankMVDR, SpaceTimeModel, mvdr_weights  # noqa: E402


def make_data(C, L, n_train, bins, rng):
    lam, T = 0.0039, 70e-6
    model = SpaceTimeModel(np.c_[np.arange(C) * lam / 2, np.zeros(C), np.zeros(C)], np.zeros(C), L, T, lam)
    D = model.dimension
    # clutter ridge: sum of steering vectors along nu = -beta * u / (d T) ... with random amplitudes
    u = np.linspace(-1, 1, 4 * D)
    nu = -0.5 * u / (2 * T)
    A = np.stack([model.steering(np.array([x, np.sqrt(1 - x * x), 0]), f) for x, f in zip(u, nu)], 1)
    g = (rng.standard_normal((bins, n_train, A.shape[1])) + 1j * rng.standard_normal((bins, n_train, A.shape[1]))) * 3
    X = g @ A.T + (rng.standard_normal((bins, n_train, D)) + 1j * rng.standard_normal((bins, n_train, D))) / np.sqrt(2)
    return model, X.astype(np.complex64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--pulses", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--bins", type=int, default=32)
    ap.add_argument("--hypotheses", type=int, default=32)
    ap.add_argument("--backend", default="numpy")
    ap.add_argument("--output", default=None, help="optional JSON output path")
    args = ap.parse_args()
    try:
        xp = get_array_module(args.backend)
    except BackendUnavailable as exc:
        print(exc, file=sys.stderr)
        return 2
    rng = np.random.default_rng(0)
    rows = []
    for C in args.channels:
        for L in args.pulses:
            D = C * L
            n_train = 2 * D
            model, X = make_data(C, L, n_train, args.bins, rng)
            nu = model.doppler_grid(args.hypotheses)
            S = model.steering(direction_from_angles(10.0), nu)
            row = {"C": C, "L": L, "D": D, "training": n_train}
            full = FullRankMVDR(model, "absolute", 1.0)
            t0 = time.perf_counter()
            Wref = [full.weights(X[b], S, nu).W for b in range(args.bins)]
            row["mvdr_cpu_ms_per_bin"] = 1e3 * (time.perf_counter() - t0) / args.bins
            efa = AdjacentBinPostDoppler(model, loading_mode="absolute", loading_value=1.0)
            t0 = time.perf_counter()
            for b in range(args.bins):
                efa.weights(X[b], S, nu)
            row["efa_cpu_ms_per_bin"] = 1e3 * (time.perf_counter() - t0) / args.bins
            Xd, Sd = xp.asarray(X.astype(np.complex128)), xp.asarray(S)
            batched_mvdr(Xd[:1], Sd, 1.0, xp)  # warm-up
            t0 = time.perf_counter()
            Wb = batched_mvdr(Xd, Sd, 1.0, xp)
            if hasattr(xp, "cuda"):
                xp.cuda.Stream.null.synchronize()
            row[f"batched_{args.backend}_ms_per_bin"] = 1e3 * (time.perf_counter() - t0) / args.bins
            Wb = to_numpy(Wb)
            ref = np.stack([mvdr_weights(
                (X[b].T.astype(complex) @ X[b].conj().astype(complex)) / n_train, S, 1.0) for b in range(args.bins)])
            row["batched_rel_error_vs_cpu"] = float(np.linalg.norm(Wb - ref) / np.linalg.norm(ref))
            del Wref
            rows.append(row)
            print(json.dumps(row))
    if args.output:
        Path(args.output).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
