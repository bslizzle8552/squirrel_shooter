from __future__ import annotations

from pathlib import Path

import pytest

from squirrel_shooter.config import ConfigError, load_config


def test_loads_camera_config(tmp_path: Path) -> None:
    config_path = tmp_path / "camera.yaml"
    config_path.write_text(
        """
camera:
  device_index: 2
  requested_width: 1280
  requested_height: 720
  requested_fps: 30
  output_directory: captures
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.camera.device_index == 2
    assert config.camera.requested_width == 1280
    assert config.camera.requested_height == 720
    assert config.camera.requested_fps == 30.0
    assert config.camera.output_directory == Path("captures")


def test_rejects_missing_camera_setting(tmp_path: Path) -> None:
    config_path = tmp_path / "camera.yaml"
    config_path.write_text("camera:\n  device_index: 0\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="Missing camera setting"):
        load_config(config_path)


def test_rejects_boolean_device_index(tmp_path: Path) -> None:
    config_path = tmp_path / "camera.yaml"
    config_path.write_text(
        """
camera:
  device_index: false
  requested_width: 1280
  requested_height: 720
  requested_fps: 30
  output_directory: captures
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="device_index"):
        load_config(config_path)
