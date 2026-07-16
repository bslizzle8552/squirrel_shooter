"""Load and validate camera, watcher, reporting, and retention settings."""

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
    camera_mode_if_known: str = "unknown"
    ir_mode_if_explicitly_detected_or_configured: str = "unknown"
    low_fps_threshold: float = 15.0
    reopen_after_failed_reads: int = 10
    reopen_delay_seconds: float = 2.0


@dataclass(frozen=True)
class SharedCameraConfig:
    reconnect_enabled: bool
    maximum_consecutive_read_failures: int
    reconnect_delay_seconds: float
    consumer_wait_timeout_seconds: float
    annotated_frame_stale_seconds: float


@dataclass(frozen=True)
class RuntimeConfig:
    headless: bool
    shutdown_timeout_seconds: float


@dataclass(frozen=True)
class DashboardConfig:
    enabled: bool
    host: str
    port: int
    stream_fps: float
    jpeg_quality: int
    status_refresh_interval_seconds: float


@dataclass(frozen=True)
class RoiConfig:
    enabled: bool
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class InclusionZoneConfig:
    enabled: bool
    polygon: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class StartupWarmupConfig:
    seconds: float
    minimum_frames: int


@dataclass(frozen=True)
class PersistenceConfig:
    frames: int
    max_centroid_distance_pixels: float
    maximum_gap_seconds: float
    cooldown_seconds: float


@dataclass(frozen=True)
class GroupingConfig:
    enabled: bool
    max_horizontal_gap_pixels: int
    max_vertical_gap_pixels: int
    expanded_box_margin_pixels: int
    max_centroid_distance_pixels: float
    direction_similarity_degrees: float
    speed_similarity_ratio: float
    maximum_components_per_group: int


@dataclass(frozen=True)
class GlobalRejectionConfig:
    enabled: bool
    max_frame_motion_percent: float
    max_zone_motion_percent: float
    recovery_seconds: float
    log_rejected_global_events: bool
    save_debug_snapshot: bool
    exposure_luminance_delta: float
    ir_colorfulness_delta: float
    obstruction_motion_percent: float


@dataclass(frozen=True)
class EventLifecycleConfig:
    pre_event_seconds: float
    post_event_seconds: float
    maximum_event_seconds: float
    clip_codec: str


@dataclass(frozen=True)
class ClassificationConfig:
    tiny_max_frame_percent: float
    small_animal_max_frame_percent: float
    medium_animal_max_frame_percent: float
    person_min_height_percent: float
    large_object_min_frame_percent: float
    flicker_min_components: int
    stationary_speed_pixels_per_second: float
    slow_speed_pixels_per_second: float
    fast_speed_pixels_per_second: float


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
    warmup: StartupWarmupConfig
    inclusion_zone: InclusionZoneConfig
    persistence: PersistenceConfig
    grouping: GroupingConfig
    global_rejection: GlobalRejectionConfig
    event_lifecycle: EventLifecycleConfig
    classification: ClassificationConfig


@dataclass(frozen=True)
class StorageConfig:
    max_event_captures: int
    max_debug_images: int
    max_log_files: int


@dataclass(frozen=True)
class RetentionConfig:
    maximum_storage_megabytes: float
    maximum_event_age_days: float
    maximum_event_count: int | None


@dataclass(frozen=True)
class ReportingConfig:
    directory: Path
    thumbnail_width: int
    rebuild_on_clean_shutdown: bool


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    directory: Path
    event_csv: str = "events.csv"
    event_jsonl: str = "events.jsonl"
    rejection_jsonl: str = "rejections.jsonl"
    sessions_directory: str = "sessions"
    maximum_active_log_megabytes: float = 100.0
    retained_log_rotations: int = 5


@dataclass(frozen=True)
class HealthConfig:
    camera_stale_seconds: float
    detector_stale_seconds: float


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig
    shared_camera: SharedCameraConfig
    runtime: RuntimeConfig
    dashboard: DashboardConfig
    motion: MotionConfig
    storage: StorageConfig
    retention: RetentionConfig
    reporting: ReportingConfig
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
        raise ConfigError(f"{field} must be an integer of at least {minimum}")
    return value


def _number(value: Any, field: str, *, minimum: float = 0.0, maximum: float | None = None, exclusive: bool = False) -> float:
    invalid = isinstance(value, bool) or not isinstance(value, (int, float))
    number = float(value) if not invalid else minimum
    if invalid or (number <= minimum if exclusive else number < minimum):
        relation = "greater than" if exclusive else "at least"
        raise ConfigError(f"{field} must be {relation} {minimum}")
    if maximum is not None and number > maximum:
        raise ConfigError(f"{field} must be at most {maximum}")
    return number


def _path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-empty path")
    return Path(value).expanduser()


def _odd(value: Any, field: str) -> int:
    result = _int(value, field, minimum=1)
    if result % 2 == 0:
        raise ConfigError(f"{field} must be an odd integer")
    return result


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be non-empty text")
    return value.strip()


def _percent(mapping: dict[str, Any], key: str, prefix: str) -> float:
    return _number(mapping[key], f"{prefix}.{key}", maximum=100.0)


def _zone(raw: dict[str, Any]) -> InclusionZoneConfig:
    points = raw.get("polygon")
    if not isinstance(points, list) or len(points) < 3:
        raise ConfigError("motion.inclusion_zone.polygon must contain at least three points")
    normalized: list[tuple[float, float]] = []
    for index, point in enumerate(points):
        if not isinstance(point, list) or len(point) != 2:
            raise ConfigError(f"motion.inclusion_zone.polygon point {index} must be [x, y]")
        normalized.append((_number(point[0], f"zone point {index} x", maximum=1.0), _number(point[1], f"zone point {index} y", maximum=1.0)))
    return InclusionZoneConfig(_bool(raw.get("enabled"), "motion.inclusion_zone.enabled"), tuple(normalized))


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load a complete configuration and reject unsafe or ambiguous values."""

    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise ConfigError(f"Configuration file not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Could not read configuration: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("Configuration must be a YAML mapping")

    camera = _mapping(raw, "camera")
    _required(camera, {"device_index", "requested_width", "requested_height", "requested_fps", "output_directory"}, "camera")
    motion = _mapping(raw, "motion")
    shared_camera = _mapping(raw, "shared_camera")
    runtime = _mapping(raw, "runtime")
    dashboard = _mapping(raw, "dashboard")
    roi = _mapping(motion, "roi")
    debug = _mapping(motion, "debug_outputs")
    warmup = _mapping(motion, "startup_warmup")
    persistence = _mapping(motion, "persistence")
    grouping = _mapping(motion, "grouping")
    global_rejection = _mapping(motion, "global_rejection")
    lifecycle = _mapping(motion, "event_lifecycle")
    classification = _mapping(motion, "provisional_classification")
    storage = _mapping(raw, "storage")
    retention = _mapping(raw, "retention")
    reporting = _mapping(raw, "reporting")
    logging_raw = _mapping(raw, "logging")
    health = _mapping(raw, "health")

    roi_config = RoiConfig(
        _bool(roi.get("enabled"), "motion.roi.enabled"),
        _number(roi.get("x"), "motion.roi.x", maximum=1.0),
        _number(roi.get("y"), "motion.roi.y", maximum=1.0),
        _number(roi.get("width"), "motion.roi.width", maximum=1.0, exclusive=True),
        _number(roi.get("height"), "motion.roi.height", maximum=1.0, exclusive=True),
    )
    if roi_config.x + roi_config.width > 1.0 or roi_config.y + roi_config.height > 1.0:
        raise ConfigError("motion.roi rectangle must fit inside the normalized frame")
    min_area = _number(motion.get("min_blob_area"), "motion.min_blob_area", exclusive=True)
    max_area = _number(motion.get("max_blob_area"), "motion.max_blob_area", exclusive=True)
    if max_area <= min_area:
        raise ConfigError("motion.max_blob_area must be greater than motion.min_blob_area")
    clip_codec = _text(lifecycle.get("clip_codec"), "motion.event_lifecycle.clip_codec")
    if len(clip_codec) != 4:
        raise ConfigError("motion.event_lifecycle.clip_codec must contain four characters")

    level = _text(logging_raw.get("level"), "logging.level").upper()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ConfigError("logging.level must be DEBUG, INFO, WARNING, or ERROR")
    maximum_count = retention.get("maximum_event_count")
    if maximum_count is not None:
        maximum_count = _int(maximum_count, "retention.maximum_event_count", minimum=1)

    motion_config = MotionConfig(
        enabled=_bool(motion.get("enabled"), "motion.enabled"),
        processing_width=_int(motion.get("processing_width"), "motion.processing_width", minimum=1),
        learning_frames=_int(motion.get("learning_frames"), "motion.learning_frames", minimum=1),
        history=_int(motion.get("history"), "motion.history", minimum=1),
        variance_threshold=_number(motion.get("variance_threshold"), "motion.variance_threshold", exclusive=True),
        detect_shadows=_bool(motion.get("detect_shadows"), "motion.detect_shadows"),
        blur_kernel=_odd(motion.get("blur_kernel"), "motion.blur_kernel"),
        morphology_kernel=_odd(motion.get("morphology_kernel"), "motion.morphology_kernel"),
        open_iterations=_int(motion.get("open_iterations"), "motion.open_iterations"),
        close_iterations=_int(motion.get("close_iterations"), "motion.close_iterations"),
        min_blob_area=min_area,
        max_blob_area=max_area,
        persistence_frames=_int(motion.get("persistence_frames"), "motion.persistence_frames", minimum=1),
        persistence_max_distance=_number(motion.get("persistence_max_distance"), "motion.persistence_max_distance", exclusive=True),
        cooldown_seconds=_number(motion.get("cooldown_seconds"), "motion.cooldown_seconds"),
        lighting_change_percent=_percent(motion, "lighting_change_percent", "motion"),
        recent_event_limit=_int(motion.get("recent_event_limit"), "motion.recent_event_limit", minimum=1),
        roi=roi_config,
        debug=DebugOutputConfig(
            _path(debug.get("directory"), "motion.debug_outputs.directory"),
            _number(debug.get("min_interval_seconds"), "motion.debug_outputs.min_interval_seconds"),
            *(_bool(debug.get(name), f"motion.debug_outputs.{name}") for name in ("foreground_mask", "cleaned_mask", "annotated_frame", "roi_visualization", "rejected_candidate_frame")),
        ),
        warmup=StartupWarmupConfig(_number(warmup.get("seconds"), "motion.startup_warmup.seconds"), _int(warmup.get("minimum_frames"), "motion.startup_warmup.minimum_frames", minimum=1)),
        inclusion_zone=_zone(_mapping(motion, "inclusion_zone")),
        persistence=PersistenceConfig(
            _int(persistence.get("frames"), "motion.persistence.frames", minimum=1),
            _number(persistence.get("max_centroid_distance_pixels"), "motion.persistence.max_centroid_distance_pixels", exclusive=True),
            _number(persistence.get("maximum_gap_seconds"), "motion.persistence.maximum_gap_seconds", exclusive=True),
            _number(persistence.get("cooldown_seconds"), "motion.persistence.cooldown_seconds"),
        ),
        grouping=GroupingConfig(
            _bool(grouping.get("enabled"), "motion.grouping.enabled"),
            _int(grouping.get("max_horizontal_gap_pixels"), "motion.grouping.max_horizontal_gap_pixels"),
            _int(grouping.get("max_vertical_gap_pixels"), "motion.grouping.max_vertical_gap_pixels"),
            _int(grouping.get("expanded_box_margin_pixels"), "motion.grouping.expanded_box_margin_pixels"),
            _number(grouping.get("max_centroid_distance_pixels"), "motion.grouping.max_centroid_distance_pixels", exclusive=True),
            _number(grouping.get("direction_similarity_degrees"), "motion.grouping.direction_similarity_degrees", maximum=180.0),
            _number(grouping.get("speed_similarity_ratio"), "motion.grouping.speed_similarity_ratio", minimum=1.0),
            _int(grouping.get("maximum_components_per_group"), "motion.grouping.maximum_components_per_group", minimum=1),
        ),
        global_rejection=GlobalRejectionConfig(
            _bool(global_rejection.get("enabled"), "motion.global_rejection.enabled"),
            _percent(global_rejection, "max_frame_motion_percent", "motion.global_rejection"),
            _percent(global_rejection, "max_zone_motion_percent", "motion.global_rejection"),
            _number(global_rejection.get("recovery_seconds"), "motion.global_rejection.recovery_seconds"),
            _bool(global_rejection.get("log_rejected_global_events"), "motion.global_rejection.log_rejected_global_events"),
            _bool(global_rejection.get("save_debug_snapshot"), "motion.global_rejection.save_debug_snapshot"),
            _number(global_rejection.get("exposure_luminance_delta"), "motion.global_rejection.exposure_luminance_delta"),
            _number(global_rejection.get("ir_colorfulness_delta"), "motion.global_rejection.ir_colorfulness_delta"),
            _percent(global_rejection, "obstruction_motion_percent", "motion.global_rejection"),
        ),
        event_lifecycle=EventLifecycleConfig(
            _number(lifecycle.get("pre_event_seconds"), "motion.event_lifecycle.pre_event_seconds"),
            _number(lifecycle.get("post_event_seconds"), "motion.event_lifecycle.post_event_seconds"),
            _number(lifecycle.get("maximum_event_seconds"), "motion.event_lifecycle.maximum_event_seconds", exclusive=True),
            clip_codec,
        ),
        classification=ClassificationConfig(
            *(_number(classification.get(name), f"motion.provisional_classification.{name}") for name in (
                "tiny_max_frame_percent", "small_animal_max_frame_percent", "medium_animal_max_frame_percent", "person_min_height_percent", "large_object_min_frame_percent"
            )),
            _int(classification.get("flicker_min_components"), "motion.provisional_classification.flicker_min_components", minimum=2),
            *(_number(classification.get(name), f"motion.provisional_classification.{name}") for name in (
                "stationary_speed_pixels_per_second", "slow_speed_pixels_per_second", "fast_speed_pixels_per_second"
            )),
        ),
    )
    thresholds = motion_config.classification
    if not (
        thresholds.tiny_max_frame_percent
        <= thresholds.small_animal_max_frame_percent
        <= thresholds.medium_animal_max_frame_percent
        <= thresholds.large_object_min_frame_percent
    ):
        raise ConfigError("provisional classification area thresholds must increase from tiny through large")
    if not (
        thresholds.stationary_speed_pixels_per_second
        <= thresholds.slow_speed_pixels_per_second
        <= thresholds.fast_speed_pixels_per_second
    ):
        raise ConfigError("provisional classification speed thresholds must be ordered stationary, slow, fast")
    if motion_config.event_lifecycle.maximum_event_seconds <= motion_config.event_lifecycle.post_event_seconds:
        raise ConfigError("maximum_event_seconds must be greater than post_event_seconds")

    dashboard_port = _int(dashboard.get("port"), "dashboard.port", minimum=1)
    if dashboard_port > 65535:
        raise ConfigError("dashboard.port must be at most 65535")
    dashboard_quality = _int(dashboard.get("jpeg_quality"), "dashboard.jpeg_quality", minimum=1)
    if dashboard_quality > 100:
        raise ConfigError("dashboard.jpeg_quality must be at most 100")

    return AppConfig(
        camera=CameraConfig(
            _int(camera.get("device_index"), "camera.device_index"),
            _int(camera.get("requested_width"), "camera.requested_width", minimum=1),
            _int(camera.get("requested_height"), "camera.requested_height", minimum=1),
            _number(camera.get("requested_fps"), "camera.requested_fps", exclusive=True),
            _path(camera.get("output_directory"), "camera.output_directory"),
            _text(camera.get("camera_mode_if_known", "unknown"), "camera.camera_mode_if_known"),
            _text(camera.get("ir_mode_if_explicitly_detected_or_configured", "unknown"), "camera.ir_mode_if_explicitly_detected_or_configured"),
            _number(camera.get("low_fps_threshold", 15.0), "camera.low_fps_threshold", exclusive=True),
            _int(camera.get("reopen_after_failed_reads", 10), "camera.reopen_after_failed_reads", minimum=1),
            _number(camera.get("reopen_delay_seconds", 2.0), "camera.reopen_delay_seconds"),
        ),
        shared_camera=SharedCameraConfig(
            _bool(shared_camera.get("reconnect_enabled"), "shared_camera.reconnect_enabled"),
            _int(shared_camera.get("maximum_consecutive_read_failures"), "shared_camera.maximum_consecutive_read_failures", minimum=1),
            _number(shared_camera.get("reconnect_delay_seconds"), "shared_camera.reconnect_delay_seconds"),
            _number(shared_camera.get("consumer_wait_timeout_seconds"), "shared_camera.consumer_wait_timeout_seconds", exclusive=True),
            _number(shared_camera.get("annotated_frame_stale_seconds"), "shared_camera.annotated_frame_stale_seconds", exclusive=True),
        ),
        runtime=RuntimeConfig(
            _bool(runtime.get("headless"), "runtime.headless"),
            _number(runtime.get("shutdown_timeout_seconds"), "runtime.shutdown_timeout_seconds", exclusive=True),
        ),
        dashboard=DashboardConfig(
            _bool(dashboard.get("enabled"), "dashboard.enabled"),
            _text(dashboard.get("host"), "dashboard.host"),
            dashboard_port,
            _number(dashboard.get("stream_fps"), "dashboard.stream_fps", exclusive=True),
            dashboard_quality,
            _number(dashboard.get("status_refresh_interval_seconds"), "dashboard.status_refresh_interval_seconds", exclusive=True),
        ),
        motion=motion_config,
        storage=StorageConfig(*(_int(storage.get(name), f"storage.{name}", minimum=1) for name in ("max_event_captures", "max_debug_images", "max_log_files"))),
        retention=RetentionConfig(
            _number(retention.get("maximum_storage_megabytes"), "retention.maximum_storage_megabytes", exclusive=True),
            _number(retention.get("maximum_event_age_days"), "retention.maximum_event_age_days", exclusive=True),
            maximum_count,
        ),
        reporting=ReportingConfig(
            _path(reporting.get("directory"), "reporting.directory"),
            _int(reporting.get("thumbnail_width"), "reporting.thumbnail_width", minimum=64),
            _bool(reporting.get("rebuild_on_clean_shutdown"), "reporting.rebuild_on_clean_shutdown"),
        ),
        logging=LoggingConfig(
            level,
            _path(logging_raw.get("directory"), "logging.directory"),
            *(_text(logging_raw.get(name), f"logging.{name}") for name in ("event_csv", "event_jsonl", "rejection_jsonl", "sessions_directory")),
            _number(logging_raw.get("maximum_active_log_megabytes", 100.0), "logging.maximum_active_log_megabytes", exclusive=True),
            _int(logging_raw.get("retained_log_rotations", 5), "logging.retained_log_rotations", minimum=1),
        ),
        health=HealthConfig(
            _number(health.get("camera_stale_seconds"), "health.camera_stale_seconds", exclusive=True),
            _number(health.get("detector_stale_seconds"), "health.detector_stale_seconds", exclusive=True),
        ),
        source_path=config_path.resolve(),
    )
