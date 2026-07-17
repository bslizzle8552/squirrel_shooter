from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from squirrel_shooter.config import ConfigError, load_config
from conftest import PROJECT_ROOT


def test_loads_camera_config(tmp_path: Path) -> None:
    config_path = tmp_path / "camera.yaml"
    raw = yaml.safe_load((PROJECT_ROOT / "config/default.yaml").read_text(encoding="utf-8"))
    raw["camera"]["device_index"] = 2
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_config(config_path)

    assert config.camera.device_index == 2
    assert config.camera.requested_width == 1280
    assert config.camera.requested_height == 720
    assert config.camera.requested_fps == 30.0
    assert config.camera.output_directory == Path("captures")
    assert config.dashboard.host == "0.0.0.0"
    assert config.dashboard.port == 5000
    assert config.shared_camera.reconnect_enabled is True
    assert config.runtime.headless is False
    assert config.night_mode.pause_recording_and_classifier is True
    assert config.night_mode.enter_consecutive_frames == 5
    assert config.night_mode.exit_consecutive_frames == 10
    assert config.motion.min_blob_area == 500
    assert config.motion.inclusion_zone.enabled is True
    assert config.motion.inclusion_zone.polygon[:2] == ((0.0, 0.26), (1.0, 0.26))
    assert config.motion.persistence.frames == 5
    assert config.motion.candidate_filter.require_coherent_small_motion is True


def test_rejects_missing_camera_setting(tmp_path: Path) -> None:
    config_path = tmp_path / "camera.yaml"
    config_path.write_text("camera:\n  device_index: 0\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="Missing camera setting"):
        load_config(config_path)


def test_rejects_boolean_device_index(tmp_path: Path) -> None:
    config_path = tmp_path / "camera.yaml"
    raw = yaml.safe_load((PROJECT_ROOT / "config/default.yaml").read_text(encoding="utf-8"))
    raw["camera"]["device_index"] = False
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="device_index"):
        load_config(config_path)


def test_rejects_invalid_roi_and_even_blur_kernel(tmp_path: Path) -> None:
    raw = yaml.safe_load((PROJECT_ROOT / "config/default.yaml").read_text(encoding="utf-8"))
    raw["motion"]["roi"].update(enabled=True, x=0.8, width=0.5)
    config_path = tmp_path / "bad-roi.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="ROI|roi"):
        load_config(config_path)

    raw["motion"]["roi"].update(x=0.0, width=1.0)
    raw["motion"]["blur_kernel"] = 4
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="odd"):
        load_config(config_path)


def test_rejects_invalid_dashboard_port(tmp_path: Path) -> None:
    raw = yaml.safe_load((PROJECT_ROOT / "config/default.yaml").read_text(encoding="utf-8"))
    raw["dashboard"]["port"] = 70000
    config_path = tmp_path / "bad-dashboard.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="dashboard.port"):
        load_config(config_path)
