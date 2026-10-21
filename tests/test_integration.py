"""End-to-end tests: simulation workflow, configuration validation, hardware-capture workflow."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from radar_stap.capture import CaptureConfigError, assess_capture_config, load_capture_config, process_capture
from radar_stap.cli import process_capture_main
from radar_stap.config import ConfigError, load_simulation_config
from radar_stap.pipeline import run_simulation_pipeline
from radar_stap.synthetic_capture import make_synthetic_capture

from conftest import STAP_SCENE, make_cfg

ROOT = Path(__file__).resolve().parents[1]


def test_end_to_end_simulation_pipeline(tmp_path):
    cfg = make_cfg(STAP_SCENE, {
        "targets": STAP_SCENE["targets"] + [
            {"name": "fast", "range_m": 35.0, "angle_deg": -30.0, "radial_velocity_mps": 25.0, "snr_db": -5.0}
        ],
        # range-only CFAR per Doppler column: Doppler training cells next to the clutter band are not homogeneous
        "detection": {"guard_cells": [2, 0], "training_cells": [10, 0]},
        "processing": {"doppler_nfft": 32},
        "monte_carlo": {"enabled": True, "num_trials": 3, "target_name": "t1"},
        "output": {"save_raw": True},
    })
    res = run_simulation_pipeline(cfg, tmp_path, make_plots=True)
    on_disk = json.loads((tmp_path / "results.json").read_text())
    assert on_disk["scenario"] == "test"
    for f in on_disk["files"]:
        assert (tmp_path / f).is_file(), f
    # exo-clutter mover detected by the conventional RD chain with accurate estimates
    fast = next(r for r in res["range_doppler_detection"]["targets"] if r["name"] == "fast")
    assert fast["detected"]
    assert abs(fast["errors"]["range_m"]) < 0.2
    assert abs(fast["errors"]["radial_velocity_mps"]) < 0.5
    assert abs(fast["errors"]["angle_deg"]) < 2.0
    # STAP comparison and Monte Carlo present with sane values
    st = res["stap"]
    assert st["dimension"] == 32 and set(st["methods"]) == {"nonadaptive", "mvdr", "efa", "pc"}
    mc = res["monte_carlo"]
    assert mc["trials"] == 3
    for m in mc["methods"].values():
        assert 0.0 <= m["pd"] <= 1.0 and m["h0_cells_tested"] > 0
    npz = np.load(tmp_path / "synthetic_cpi.npz")
    assert npz["data"].shape == (4, 8, 128)


def test_simulate_script_runs(tmp_path):
    """The command-line script runs from the repository root (small config written to tmp)."""
    data = yaml.safe_load((ROOT / "configs" / "simulation.yaml").read_text())
    data["clutter"]["num_scatterers"] = 300
    data["monte_carlo"]["enabled"] = False
    data["stap"]["methods"] = ["nonadaptive", "efa"]
    cfgp = tmp_path / "sim.yaml"
    cfgp.write_text(yaml.safe_dump(data))
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    r = subprocess.run([sys.executable, "scripts/simulate_scene.py", "--config", str(cfgp), "--output-dir",
                        str(tmp_path / "out"), "--no-plots"], cwd=ROOT, capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "Ground truth" in r.stdout and (tmp_path / "out" / "results.json").is_file()


@pytest.mark.parametrize(
    "override,msg",
    [
        ({"radar": {"ramp_end_time_s": 10e-6}}, "after ramp end"),
        ({"radar": {"adc_start_time_s": 0.05e-6},
          "targets": [{"name": "a", "snr_db": 0, "range_m": 20.0, "angle_deg": 0.0}]}, "adc_start_time"),
        ({"radar": {"carrier_frequency_hz": -1}}, "carrier_frequency_hz must be > 0"),
        ({"radar": {"num_chirps": 33}, "array": {"tx_schedule": [0, 0]}}, "at most once"),
        ({"stap": {"enabled": True, "methods": ["mvdr"], "num_training": 64}, "radar": {"num_chirps": 32}}, "singular"),
        ({"targets": [{"name": "a", "snr_db": 0, "range_m": 10.0}]}, "needs angle_deg"),
    ],
)
def test_config_validation_errors(override, msg):
    with pytest.raises(ConfigError, match=msg):
        make_cfg(override)


def test_unknown_config_key_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("base: " + str(ROOT / "configs" / "simulation.yaml") + "\nradar:\n  carier_frequency_hz: 1\n")
    with pytest.raises(ConfigError, match="unknown key"):
        load_simulation_config(p)


def test_shipped_configs_are_valid():
    for p in [ROOT / "configs" / "simulation.yaml", *sorted((ROOT / "configs" / "scenarios").glob("*.yaml"))]:
        load_simulation_config(p)


def test_capture_template_is_incomplete_and_refuses_to_decode(tmp_path, capsys):
    cc = load_capture_config(ROOT / "configs" / "radar_capture.yaml")
    a = assess_capture_config(cc)
    assert not a.available("decode")
    assert any("adc_format.iq_order" in m or "adc_format.data_type" in m for m in a.stages["decode"]["missing"])
    dummy = tmp_path / "x.bin"
    dummy.write_bytes(b"\x00" * 1024)
    with pytest.raises(CaptureConfigError, match="cannot decode"):
        process_capture(cc, dummy, tmp_path / "out")
    rc = process_capture_main(["--config", str(ROOT / "configs" / "radar_capture.yaml"), "--input", str(dummy),
                               "--output-dir", str(tmp_path / "out")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "Refusing to decode" in err


def test_unverified_format_needs_explicit_override(tmp_path):
    cfg = make_cfg({"radar": {"num_chirps": 8}, "targets": [{"name": "a", "range_m": 20.0, "angle_deg": 0.0, "snr_db": 10.0}]})
    make_synthetic_capture(cfg, tmp_path, num_frames=1, stap_enabled=False)
    data = yaml.safe_load((tmp_path / "capture.yaml").read_text())
    data["capture_format_verified"] = False
    (tmp_path / "capture.yaml").write_text(yaml.safe_dump(data))
    cc = load_capture_config(tmp_path / "capture.yaml")
    assert not assess_capture_config(cc).available("decode")
    assert assess_capture_config(cc, allow_unverified_format=True).available("decode")


def test_capture_yaml_conflicting_with_cfg_is_rejected(tmp_path):
    cfg = make_cfg({"radar": {"num_chirps": 8}})
    make_synthetic_capture(cfg, tmp_path, num_frames=1, stap_enabled=False)
    data = yaml.safe_load((tmp_path / "capture.yaml").read_text())
    data["waveform"]["chirp_slope_hz_per_s"] = 31e12
    (tmp_path / "capture.yaml").write_text(yaml.safe_dump(data))
    with pytest.raises(CaptureConfigError, match="contradicts"):
        load_capture_config(tmp_path / "capture.yaml")


@pytest.mark.parametrize("packing,layout", [("two_sample_blocks", "rx_planar_per_chirp"),
                                            ("iq_interleaved", "rx_interleaved_per_sample")])
def test_synthetic_capture_roundtrip_through_hardware_path(tmp_path, packing, layout):
    """Simulated scene -> quantised DCA1000-style file + SDK .cfg -> decode -> process -> truth."""
    tdm = {
        "radar": {"num_chirps": 32},
        "array": {"tx_positions": [[0, 0, 0], [4, 0, 0]], "tx_schedule": [0, 1]},
        "targets": [
            {"name": "a", "range_m": 14.0, "angle_deg": 10.0, "radial_velocity_mps": -6.0, "snr_db": -5.0},
            {"name": "b", "range_m": 33.0, "angle_deg": -25.0, "radial_velocity_mps": 4.0, "snr_db": -5.0},
        ],
    }
    cfg = make_cfg(tdm)
    info = make_synthetic_capture(cfg, tmp_path / "cap", num_frames=2, adc_bits=14, noise_lsb=3.0,
                                  fmt_overrides={"complex_packing": packing, "channel_layout": layout},
                                  stap_enabled=False)
    assert info["clipped_values"] == 0
    cc = load_capture_config(tmp_path / "cap" / "capture.yaml")
    assert cc.provenance["waveform.chirp_slope_hz_per_s"].startswith("mmwave_cfg")
    assert cc.tx_schedule == [0, 1]
    rep = process_capture(cc, tmp_path / "cap" / "adc_data.bin", tmp_path / "out", make_plots=False)
    assert rep["data_shape_frames_rx_chirps_samples"] == [2, 4, 32, 128]
    assert rep["decode"]["out_of_range_values"] == 0
    dets = rep["doppler"]["frames"][0]["detections"]
    for t in info["truth"]:
        best = min(dets, key=lambda d: abs(d["range_m"] - t["range_m"]) + abs(d["radial_velocity_mps"] - t["radial_velocity_mps"]))
        assert abs(best["range_m"] - t["range_m"]) < 0.2, t["name"]
        assert abs(best["radial_velocity_mps"] - t["radial_velocity_mps"]) < 0.5, t["name"]
        assert abs(best["angle_deg"] - t["angle_deg"]) < 2.0, t["name"]


def test_wrong_iq_order_is_visible_in_results(tmp_path):
    """Decoding with the wrong I/Q order mirrors the beat spectrum: the target is no longer found at its range."""
    cfg = make_cfg({"radar": {"num_chirps": 8},
                    "targets": [{"name": "a", "range_m": 12.0, "angle_deg": 0.0, "snr_db": 10.0}]})
    make_synthetic_capture(cfg, tmp_path, num_frames=1, stap_enabled=False)
    good = load_capture_config(tmp_path / "capture.yaml")
    rep = process_capture(good, tmp_path / "adc_data.bin", tmp_path / "o1", make_plots=False)
    assert abs(rep["range"]["strongest_range_m"] - 12.0) < 0.3
    data = yaml.safe_load((tmp_path / "capture.yaml").read_text())
    data["adc_format"]["iq_order"] = "Q_first"
    (tmp_path / "capture.yaml").write_text(yaml.safe_dump(data))
    bad = load_capture_config(tmp_path / "capture.yaml")
    rep = process_capture(bad, tmp_path / "adc_data.bin", tmp_path / "o2", make_plots=False)
    assert abs(rep["range"]["strongest_range_m"] - 12.0) > 5.0


def test_process_capture_script_cli(tmp_path):
    cfg = make_cfg({"radar": {"num_chirps": 8},
                    "targets": [{"name": "a", "range_m": 20.0, "angle_deg": 0.0, "snr_db": 10.0}]})
    make_synthetic_capture(cfg, tmp_path / "cap", num_frames=1, stap_enabled=False)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    r = subprocess.run([sys.executable, "scripts/process_capture.py", "--config", str(tmp_path / "cap" / "capture.yaml"),
                        "--input", str(tmp_path / "cap" / "adc_data.bin"), "--output-dir", str(tmp_path / "out")],
                       cwd=ROOT, capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert (tmp_path / "out" / "capture_summary.json").is_file()


def test_notebook_executes():
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    nb = nbformat.read(ROOT / "notebooks" / "exploration.ipynb", as_version=4)
    nbclient.NotebookClient(nb, timeout=600, kernel_name="python3",
                            resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "error"]
    assert not errors
