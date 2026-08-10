from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from squirrel_shooter.config import ConfigError, load_config
from squirrel_shooter.pan_tilt import PanTiltConfig
from squirrel_shooter.manual_control import ManualControlConfig
from squirrel_shooter.valve import ValveConfig
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
    assert config.runtime.opencv_threads == 1
    assert config.night_mode.pause_recording_and_classifier is True
    assert config.night_mode.enter_consecutive_frames == 5
    assert config.night_mode.exit_consecutive_frames == 10
    assert config.pan_tilt.i2c_address == 0x40
    assert config.pan_tilt.pan_channel == 0
    assert config.pan_tilt.tilt_channel == 1
    assert config.pan_tilt.pan_min == 30
    assert config.pan_tilt.pan_center == 85
    assert config.pan_tilt.pan_max == 150
    assert config.pan_tilt.tilt_min == 70
    assert config.pan_tilt.tilt_center == 85
    assert config.pan_tilt.tilt_max == 150
    assert config.pan_tilt.park_pan == 85
    assert config.pan_tilt.park_tilt == 82
    assert config.manual_control.servo_enabled is True
    assert config.manual_control.allowed_step_degrees == (1, 3, 5)
    assert config.manual_control.default_step_degrees == 3
    assert config.manual_control.fire_pulse_seconds == 0.25
    assert config.manual_control.fire_cooldown_seconds == 10.0
    assert config.manual_control.recording.enabled is True
    assert config.manual_control.recording.pre_roll_seconds == 2.0
    assert config.manual_control.recording.post_roll_seconds == 5.0
    assert config.manual_control.recording.target_fps == 12.0
    assert config.manual_control.recording.zoom_factor == 2.0
    assert (config.manual_control.recording.crop_center_x, config.manual_control.recording.crop_center_y) == (640, 360)
    assert config.manual_control.recording.save_full_frame_clip is True
    assert config.manual_control.recording.clip_codec == "MJPG"
    assert config.valve == ValveConfig(enabled=True, gpio_pin=24, active_high=True)
    assert config.motion.min_blob_area == 500
    assert config.motion.target_fps == 10.0
    assert config.motion.inclusion_zone.enabled is True
    assert config.motion.inclusion_zone.polygon == (
        (0.0, 0.569476),
        (0.103858, 0.552392),
        (0.183976, 0.461276),
        (0.282493, 0.287016),
        (0.292582, 0.212984),
        (1.0, 0.218679),
        (1.0, 1.0),
        (0.0, 1.0),
    )
    assert config.motion.persistence.frames == 5
    assert config.motion.candidate_filter.require_coherent_small_motion is True
    assert config.motion.candidate_filter.ignore_localized_lighting_changes is True
    assert config.motion.candidate_filter.localized_lighting_minimum_luminance_delta == 8.0
    assert config.motion.candidate_filter.localized_lighting_minimum_background_luminance == 8.0
    assert config.motion.candidate_filter.localized_lighting_maximum_chromaticity_delta == 0.10
    assert config.motion.candidate_filter.localized_lighting_minimum_fraction == 0.85


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


def test_rejects_reversed_pan_tilt_limits(tmp_path: Path) -> None:
    raw = yaml.safe_load((PROJECT_ROOT / "config/default.yaml").read_text(encoding="utf-8"))
    raw["pan_tilt"].update(pan_min=150, pan_center=90, pan_max=30)
    config_path = tmp_path / "bad-pan-tilt.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="pan limits"):
        load_config(config_path)


def test_older_config_without_pan_tilt_section_uses_safe_defaults(tmp_path: Path) -> None:
    raw = yaml.safe_load((PROJECT_ROOT / "config/default.yaml").read_text(encoding="utf-8"))
    del raw["pan_tilt"]
    config_path = tmp_path / "legacy.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_config(config_path)

    assert config.pan_tilt == PanTiltConfig()


def test_older_config_without_manual_hardware_sections_uses_safe_defaults(tmp_path: Path) -> None:
    raw = yaml.safe_load((PROJECT_ROOT / "config/default.yaml").read_text(encoding="utf-8"))
    del raw["manual_control"]
    del raw["valve"]
    config_path = tmp_path / "legacy.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_config(config_path)

    assert config.manual_control == ManualControlConfig()
    assert config.valve == ValveConfig()


def test_valve_cannot_be_enabled_without_a_gpio_pin(tmp_path: Path) -> None:
    raw = yaml.safe_load((PROJECT_ROOT / "config/default.yaml").read_text(encoding="utf-8"))
    raw["valve"]["enabled"] = True
    raw["valve"]["gpio_pin"] = None
    config_path = tmp_path / "unsafe-valve.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="gpio_pin"):
        load_config(config_path)


@pytest.mark.parametrize(
    "updates",
    [
        {"zoom_factor": 1.0},
        {"target_fps": 0},
        {"crop_center_x": None, "crop_center_y": 360},
        {"clip_codec": "too-long"},
    ],
)
def test_rejects_invalid_manual_fire_recording_configuration(tmp_path: Path, updates: dict[str, object]) -> None:
    raw = yaml.safe_load((PROJECT_ROOT / "config/default.yaml").read_text(encoding="utf-8"))
    raw["manual_control"]["recording"].update(updates)
    config_path = tmp_path / "bad-recording.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="manual_control|recording|zoom_factor|crop_center|clip_codec"):
        load_config(config_path)
