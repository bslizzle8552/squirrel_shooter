"""Load and validate the small YAML configuration used by camera commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path("config/default.yaml")


class ConfigError(ValueError):
    """Raised when a configuration file is missing or invalid."""


@dataclass(frozen=True)
class CameraConfig:
    """Requested camera settings; the camera may negotiate different values."""

    device_index: int
    requested_width: int
    requested_height: int
    requested_fps: float
    output_directory: Path


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    camera: CameraConfig
    source_path: Path


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"camera.{field_name} must be a positive integer")
    return value


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"camera.{field_name} must be a non-negative integer")
    return value


def _positive_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ConfigError(f"camera.{field_name} must be a positive number")
    return float(value)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load the project YAML file and return validated settings."""

    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise ConfigError(
            f"Configuration file not found: {config_path}. "
            "Run the command from the repository root or pass --config."
        )

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse YAML in {config_path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Could not read {config_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError("Configuration must be a YAML mapping")

    camera = raw.get("camera")
    if not isinstance(camera, dict):
        raise ConfigError("Configuration must contain a 'camera' mapping")

    required = {
        "device_index",
        "requested_width",
        "requested_height",
        "requested_fps",
        "output_directory",
    }
    missing = sorted(required - camera.keys())
    if missing:
        raise ConfigError(f"Missing camera setting(s): {', '.join(missing)}")

    output_directory = camera["output_directory"]
    if not isinstance(output_directory, str) or not output_directory.strip():
        raise ConfigError("camera.output_directory must be a non-empty path")

    return AppConfig(
        camera=CameraConfig(
            device_index=_non_negative_int(camera["device_index"], "device_index"),
            requested_width=_positive_int(camera["requested_width"], "requested_width"),
            requested_height=_positive_int(camera["requested_height"], "requested_height"),
            requested_fps=_positive_number(camera["requested_fps"], "requested_fps"),
            output_directory=Path(output_directory).expanduser(),
        ),
        source_path=config_path.resolve(),
    )
