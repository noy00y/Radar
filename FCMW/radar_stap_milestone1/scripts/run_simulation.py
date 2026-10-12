from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np

from radar_stap.config import load_config
from radar_stap.simulator import simulate_if_data
from radar_stap.processing import range_doppler_fft
from radar_stap.metrics import estimate_peak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/simulation.yaml")
    parser.add_argument("--output", default="range_doppler.png")
    args = parser.parse_args()

    cfg, scene = load_config(args.config)
    targets = scene["targets"]
    data = simulate_if_data(
        cfg, targets, noise_std=scene.get("noise_std", 0.0),
        seed=scene.get("seed", 0)
    )
    rd_db, ranges_m, velocities_mps = range_doppler_fft(data, cfg)

    print(f"Simulated data shape: {data.shape} (chirps, fast-time samples)")
    print(f"Wavelength: {cfg.wavelength_m*1e3:.3f} mm")
    print(f"Approx. range resolution: {cfg.range_resolution_m:.3f} m")
    print(f"Unambiguous velocity magnitude (nominal): "
          f"{cfg.wavelength_m/(4*cfg.chirp_repetition_interval_s):.3f} m/s")
    for target in targets:
        r = target["range_m"]
        v = target["radial_velocity_mps"]
        peak = estimate_peak(
            rd_db, ranges_m, velocities_mps,
            range_bounds_m=(max(0, r-5), r+5),
            velocity_bounds_mps=(v-4, v+4),
        )
        print(f"Target truth R={r:.2f} m, v={v:.2f} m/s -> "
              f"peak R={peak['range_m']:.2f} m, v={peak['radial_velocity_mps']:.2f} m/s")

    plt.figure(figsize=(10, 6))
    plt.imshow(
        rd_db,
        origin="lower",
        aspect="auto",
        extent=[ranges_m[0], ranges_m[-1], velocities_mps[0], velocities_mps[-1]],
        cmap="viridis",
    )
    plt.xlabel("Range (m)")
    plt.ylabel("Radial velocity (m/s)")
    plt.title("Synthetic FMCW Range-Doppler Map")
    plt.colorbar(label="Magnitude (dB, arbitrary scale)")
    plt.tight_layout()
    plt.savefig(args.output, dpi=160)
    print(f"Saved plot: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
