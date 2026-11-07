"""Single-TX project policy: multi-TX captures need an explicit opt-in for angle/STAP."""

from pathlib import Path

import pytest
import yaml

from radar_stap.capture import assess_capture_config, load_capture_config

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_CONFIGS = ["radar_capture_vehicle.yaml", "radar_capture_highway_overlook.yaml",
                      "radar_capture_vehicle_highway.yaml"]


@pytest.mark.parametrize("name", EXPERIMENT_CONFIGS)
def test_experiment_configs_are_single_tx(name):
    cc = load_capture_config(ROOT / "configs" / name)
    assert len(set(cc.tx_schedule)) == 1 and not cc.allow_multi_tx
    assert len(cc.antenna.tx_positions) == 1


def _bench_variant(tmp_path, allow):
    # a 3-TX TDM variant of a shipped config (the legacy 3-TX room config is no longer shipped)
    data = yaml.safe_load((ROOT / "configs" / "radar_capture_vehicle.yaml").read_text())
    data["tx_schedule"] = [0, 1, 2]
    data["antenna"]["tx_positions"] = [[0, 0, 0], [2, 0, 1], [4, 0, 0]]
    data["adc_format"]["chirps_per_frame"] = 48
    data["rx_calibration"] = None
    data["allow_multi_tx"] = allow
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(data))
    return assess_capture_config(load_capture_config(p), allow_unverified_format=True)


def test_multi_tx_blocks_angle_and_stap_without_opt_in(tmp_path):
    a = _bench_variant(tmp_path, False)
    assert a.available("doppler")
    assert not a.available("angle") and not a.available("stap")
    assert any("multi-TX" in m for m in a.stages["angle"]["missing"])


def test_legacy_opt_in_keeps_angle(tmp_path):
    a = _bench_variant(tmp_path, True)
    assert not any("multi-TX" in m for m in a.stages["angle"]["missing"])
