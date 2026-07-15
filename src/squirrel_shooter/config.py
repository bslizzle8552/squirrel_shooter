"""Load and validate the YAML configuration used by camera and vision services."""

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
    device_index: int
    requested_width: int
    requested_height: int
    requested_fps: float
    output_directory: Path


@dataclass(frozen=True)
class RoiConfig:
    enabled: bool
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class DebugOutputConfig:
    directory: Path
    min_interval_seconds: float
    foreground_mask: bool
    cleaned_mask: bool
    annotated_frame: bool
    roi_visualization: bool
    rejected_candidate_frame: bool


@dataclass(frozen=True)
class MotionConfig:
    enabled: bool
    processing_width: int
    learning_frames: int
    history: int
    variance_threshold: float
    detect_shadows: bool
    blur_kernel: int
    morphology_kernel: int
    open_iterations: int
    close_iterations: int
    min_blob_area: float
    max_blob_area: float
    persistence_frames: int
    persistence_max_distance: float
    cooldown_seconds: float
    lighting_change_percent: float
    recent_event_limit: int
    roi: RoiConfig
    debug: DebugOutputConfig


@dataclass(frozen=True)
class StorageConfig:
    max_event_captures: int
    max_debug_images: int
    max_log_files: int


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    directory: Path


@dataclass(frozen=True)
class HealthConfig:
    camera_stale_seconds: float
    detector_stale_seconds: float


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig
    motion: MotionConfig
    storage: StorageConfig
    logging: LoggingConfig
    health: HealthConfig
    source_path: Path


def _mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"Configuration must contain a '{key}' mapping")
    return value


def _required(mapping: dict[str, Any], fields: set[str], prefix: str) -> None:
    missing = sorted(fields - mapping.keys())
    if missing:
        raise ConfigError(f"Missing {prefix} setting(s): {', '.join(missing)}")


def _bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{field} must be true or false")
    return value


def _int(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if minimum == 1 else f"at least {minimum}"
        raise ConfigError(f"{field} must be a {qualifier} integer")
    return value


def _number(
    value: Any,
    field: str,
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
    inclusive_minimum: bool = True,
) -> float:
    invalid = isinstance(value, bool) or not isinstance(value, (int, float))
    numeric = float(value) if not invalid else 0.0
    if invalid or (numeric < minimum if inclusive_minimum else numeric <= minimum):
        raise ConfigError(f"{field} must be greater than {'or equal to ' if inclusive_minimum else ''}{minimum}")
    if maximum is not None and numeric > maximum:
        raise ConfigError(f"{field} must be at most {maximum}")
    return numeric


def _path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-empty path")
    return Path(value).expanduser()


def _odd(value: Any, field: str) -> int:
    result = _int(value, field, minimum=1)
    if result % 2 == 0:
        raise ConfigError(f"{field} must be an odd integer")
    return result


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load the project YAML file and return fully validated settings."""

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

    camera = _mapping(raw, "camera")
    _required(camera, {"device_index", "requested_width", "requested_height", "requested_fps", "output_directory"}, "camera")
    motion = _mapping(raw, "motion")
    roi = _mapping(motion, "roi")
    debug = _mapping(motion, "debug_outputs")
    storage = _mapping(raw, "storage")
    logging_config = _mapping(raw, "logging")
    health = _mapping(raw, "health")

    required_motion = {
        "enabled", "processing_width", "learning_frames", "history",
        "variance_threshold", "detect_shadows", "blur_kernel",
        "morphology_kernel", "open_iterations", "close_iterations",
        "min_blob_area", "max_blob_area", "persistence_frames",
        "persistence_max_distance", "cooldown_seconds",
        "lighting_change_percent", "recent_event_limit", "roi", "debug_outputs",
    }
    _required(motion, required_motion, "motion")
    _required(roi, {"enabled", "x", "y", "width", "height"}, "motion.roi")
    _required(
        debug,
        {"directory", "min_interval_seconds", "foreground_mask", "cleaned_mask", "annotated_frame", "roi_visualization", "rejected_candidate_frame"},
        "motion.debug_outputs",
    )
    _required(storage, {"max_event_captures", "max_debug_images", "max_log_files"}, "storage")
    _required(logging_config, {"level", "directory"}, "logging")
    _required(health, {"camera_stale_seconds", "detector_stale_seconds"}, "health")

    roi_config = RoiConfig(
        enabled=_bool(roi["enabled"], "motion.roi.enabled"),
        x=_number(roi["x"], "motion.roi.x", maximum=1.0),
        y=_number(roi["y"], "motion.roi.y", maximum=1.0),
        width=_number(roi["width"], "motion.roi.width", minimum=0.0, maximum=1.0, inclusive_minimum=False),
        height=_number(roi["height"], "motion.roi.height", minimum=0.0, maximum=1.0, inclusive_minimum=False),
    )
    if roi_config.x + roi_config.width > 1.0 or roi_config.y + roi_config.height > 1.0:
        raise ConfigError("motion.roi rectangle must fit inside the normalized frame")

    level = logging_config["level"]
    if not isinstance(level, str) or level.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ConfigError("logging.level must be DEBUG, INFO, WARNING, or ERROR")

    min_area = _number(motion["min_blob_area"], "motion.min_blob_area", inclusive_minimum=False)
    max_area = _number(motion["max_blob_area"], "motion.max_blob_area", inclusive_minimum=False)
    if max_area <= min_area:
        raise ConfigError("motion.max_blob_area must be greater than motion.min_blob_area")

    return AppConfig(
        camera=CameraConfig(
            device_index=_int(camera["device_index"], "camera.device_index"),
            requested_width=_int(camera["requested_width"], "camera.requested_width", minimum=1),
            requested_height=_int(camera["requested_height"], "camera.requested_height", minimum=1),
            requested_fps=_number(camera["requested_fps"], "camera.requested_fps", inclusive_minimum=False),
            output_directory=_path(camera["output_directory"], "camera.output_directory"),
        ),
        motion=MotionConfig(
            enabled=_bool(motion["enabled"], "motion.enabled"),
            processing_width=_int(motion["processing_width"], "motion.processing_width", minimum=1),
            learning_frames=_int(motion["learning_frames"], "motion.learning_frames", minimum=1),
            history=_int(motion["history"], "motion.history", minimum=1),
            variance_threshold=_number(motion["variance_threshold"], "motion.variance_threshold", inclusive_minimum=False),
            detect_shadows=_bool(motion["detect_shadows"], "motion.detect_shadows"),
            blur_kernel=_odd(motion["blur_kernel"], "motion.blur_kernel"),
            morphology_kernel=_odd(motion["morphology_kernel"], "motion.morphology_kernel"),
            open_iterations=_int(motion["open_iterations"], "motion.open_iterations"),
            close_iterations=_int(motion["close_iterations"], "motion.close_iterations"),
            min_blob_area=min_area,
            max_blob_area=max_area,
            persistence_frames=_int(motion["persistence_frames"], "motion.persistence_frames", minimum=1),
            persistence_max_distance=_number(motion["persistence_max_distance"], "motion.persistence_max_distance", inclusive_minimum=False),
            cooldown_seconds=_number(motion["cooldown_seconds"], "motion.cooldown_seconds"),
            lighting_change_percent=_number(motion["lighting_change_percent"], "motion.lighting_change_percent", maximum=100.0, inclusive_minimum=False),
            recent_event_limit=_int(motion["recent_event_limit"], "motion.recent_event_limit", minimum=1),
            roi=roi_config,
            debug=DebugOutputConfig(
                directory=_path(debug["directory"], "motion.debug_outputs.directory"),
                min_interval_seconds=_number(debug["min_interval_seconds"], "motion.debug_outputs.min_interval_seconds"),
                foreground_mask=_bool(debug["foreground_mask"], "motion.debug_outputs.foreground_mask"),
                cleaned_mask=_bool(debug["cleaned_mask"], "motion.debug_outputs.cleaned_mask"),
                annotated_frame=_bool(debug["annotated_frame"], "motion.debug_outputs.annotated_frame"),
                roi_visualization=_bool(debug["roi_visualization"], "motion.debug_outputs.roi_visualization"),
                rejected_candidate_frame=_bool(debug["rejected_candidate_frame"], "motion.debug_outputs.rejected_candidate_frame"),
            ),
        ),
        storage=StorageConfig(
            max_event_captures=_int(storage["max_event_captures"], "storage.max_event_captures", minimum=1),
            max_debug_images=_int(storage["max_debug_images"], "storage.max_debug_images", minimum=1),
            max_log_files=_int(storage["max_log_files"], "storage.max_log_files", minimum=1),
        ),
        logging=LoggingConfig(level=level.upper(), directory=_path(logging_config["directory"], "logging.directory")),
        health=HealthConfig(
            camera_stale_seconds=_number(health["camera_stale_seconds"], "health.camera_stale_seconds", inclusive_minimum=False),
            detector_stale_seconds=_number(health["detector_stale_seconds"], "health.detector_stale_seconds", inclusive_minimum=False),
        ),
        source_path=config_path.resolve(),
    )
