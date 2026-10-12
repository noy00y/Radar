import numpy as np

from radar_stap.config import RadarConfig
from radar_stap.simulator import simulate_if_data
from radar_stap.processing import range_doppler_fft
from radar_stap.metrics import estimate_peak


def test_single_target_peak_is_near_ground_truth():
    cfg = RadarConfig(
        carrier_frequency_hz=77e9,
        slope_hz_per_s=50e12,
        adc_sample_rate_hz=20e6,
        samples_per_chirp=256,
        chirp_repetition_interval_s=100e-6,
        num_chirps=128,
    )
    target = {"range_m": 15.0, "radial_velocity_mps": 3.0, "amplitude": 1.0}
    data = simulate_if_data(cfg, [target], noise_std=0.0)
    rd_db, ranges_m, velocities_mps = range_doppler_fft(data, cfg)
    peak = estimate_peak(
        rd_db, ranges_m, velocities_mps,
        range_bounds_m=(10.0, 20.0),
        velocity_bounds_mps=(-5.0, 8.0),
    )
    assert abs(peak["range_m"] - target["range_m"]) < 1.0
    assert abs(peak["radial_velocity_mps"] - target["radial_velocity_mps"]) < 0.5


def test_simulator_shape_and_complex_dtype():
    cfg = RadarConfig(
        carrier_frequency_hz=77e9,
        slope_hz_per_s=50e12,
        adc_sample_rate_hz=20e6,
        samples_per_chirp=128,
        chirp_repetition_interval_s=100e-6,
        num_chirps=32,
    )
    data = simulate_if_data(cfg, [{"range_m": 10, "radial_velocity_mps": 0}], seed=1)
    assert data.shape == (32, 128)
    assert np.iscomplexobj(data)
