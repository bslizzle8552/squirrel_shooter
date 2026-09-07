"""Flask dashboard for camera evidence and explicitly gated manual control."""

from __future__ import annotations

import argparse
import hmac
import json
import logging
import math
import os
import secrets
import threading
from copy import deepcopy
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Any

from flask import Flask, abort, jsonify, redirect, render_template, request, send_file, url_for
from werkzeug.exceptions import HTTPException

from .camera_service import CameraService, CameraStatus
from .classifier import CLASSIFICATION_VIEWS, VOC_LABELS, ClassifierEvidenceStore
from .config import DEFAULT_CONFIG_PATH, AppConfig, load_config
from .event_report import load_events
from .manual_control import (
    ControlError,
    ControlUnavailableError,
    FireCooldownError,
    ManualControlService,
    build_manual_control_service,
    display_click_to_frame_pixel,
)
from .motion_runtime import MotionProcessingService
from .runtime_provenance import build_runtime_provenance
from .vision_service import VisionService, VisionStatus


LOGGER = logging.getLogger(__name__)
SUPPORTED_CAPTURE_SUFFIXES = frozenset({".jpg", ".jpeg"})
RECENT_EVENT_LIMIT = 5
CAPTURE_COUNT_CACHE_SECONDS = 30.0
CAPTURES_PER_PAGE = 24
EVENTS_PER_PAGE = 20
REVIEW_QUEUE_INITIAL = 10
REVIEW_QUEUE_API_LIMIT = 100
APPLICATION_MODE = "shared-camera-motion-watch"


@dataclass(frozen=True)
class CaptureImage:
    filename: str
    timestamp: str


def list_capture_images(directory: Path) -> list[CaptureImage]:
    """List supported captures newest-first without caching directory contents."""

    try:
        candidates = [
            path for path in directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_CAPTURE_SUFFIXES
            and _resolve_capture_path(directory, path.name) is not None
        ]
    except OSError:
        return []

    def sort_key(path: Path) -> tuple[float, str]:
        try:
            return path.stat().st_mtime, path.name.lower()
        except OSError:
            return 0.0, path.name.lower()

    images: list[CaptureImage] = []
    for path in sorted(candidates, key=sort_key, reverse=True):
        try:
            timestamp = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
        except OSError:
            continue
        images.append(CaptureImage(path.name, timestamp.strftime("%b %d, %Y at %I:%M:%S %p")))
    return images


def read_cpu_temperature() -> float | None:
    temperature_path = Path("/sys/class/thermal/thermal_zone0/temp")
    try:
        value = float(temperature_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    return value / 1000.0 if value > 1000 else value


def _camera_status_dict(status: CameraStatus, app_config: AppConfig) -> dict[str, Any]:
    stale = status.online and status.last_frame_age_seconds is not None and status.last_frame_age_seconds > app_config.health.camera_stale_seconds
    state = "STALE" if stale else ("ONLINE" if status.online else "OFFLINE")
    device_index = app_config.camera.device_index
    return {
        "state": state,
        "online": state == "ONLINE",
        "alive": state == "ONLINE" and status.thread_alive,
        "configured_device": {"index": device_index, "label": f"/dev/video{device_index}"},
        "resolution": {"width": status.width, "height": status.height, "label": f"{status.width}x{status.height}"},
        "fps": round(status.fps, 1),
        "error": status.error,
        "last_frame_received": status.last_frame_at,
        "last_frame_age_seconds": None if status.last_frame_age_seconds is None else round(status.last_frame_age_seconds, 2),
        "frames_received": status.frames_received,
        "thread_alive": status.thread_alive,
        "reported_fps": round(status.reported_fps, 1),
        "read_failures": status.read_failures,
        "reconnects": status.reconnects,
        "camera_open_count": status.camera_open_count,
        "annotated_frames": status.annotated_frames,
        "dashboard_viewers": status.dashboard_viewers,
        "dashboard_stream_fps": round(status.dashboard_stream_fps, 1),
        "dashboard_frames_encoded": status.dashboard_frames_encoded,
        "pre_roll_frames_buffered": status.pre_roll_frames_buffered,
        "pre_roll_buffer_fps": round(status.pre_roll_buffer_fps, 1),
        "pre_roll_target_fps": round(status.pre_roll_target_fps, 1),
        "pre_roll_frames_encoded": status.pre_roll_frames_encoded,
        "pre_roll_frames_copied": status.pre_roll_frames_copied,
        "pre_roll_frames_reused": status.pre_roll_frames_reused,
        "capture_read_average_ms": round(status.capture_read_average_ms, 2),
        "frame_publish_average_ms": round(status.frame_publish_average_ms, 2),
        "pre_roll_copy_average_ms": round(status.pre_roll_copy_average_ms, 2),
        "published_frame_copy_average_ms": round(status.published_frame_copy_average_ms, 2),
        "shared_frame_borrows": status.shared_frame_borrows,
        "dashboard_encode_average_ms": round(status.dashboard_encode_average_ms, 2),
        "dashboard_last_jpeg_bytes": status.dashboard_last_jpeg_bytes,
        "dashboard_estimated_egress_mbps": round(status.dashboard_estimated_egress_mbps, 3),
        "dashboard_encode_failures": status.dashboard_encode_failures,
        "dashboard_encoder_alive": status.dashboard_encoder_alive,
        "dashboard_encode_error": status.dashboard_encode_error,
        "capture_thread_cpu_percent": round(status.capture_thread_cpu_percent, 1),
        "last_annotated_frame": status.last_annotated_at,
        "annotated_frame_stale": status.annotated_frame_age_seconds is None
        or status.annotated_frame_age_seconds > app_config.shared_camera.annotated_frame_stale_seconds,
    }


def _vision_status_dict(status: VisionStatus, config: AppConfig) -> dict[str, Any]:
    fresh = status.last_detector_age_seconds is None or status.last_detector_age_seconds <= config.health.detector_stale_seconds
    data = asdict(status)
    data["processing_fps"] = round(status.processing_fps, 1)
    data["last_detector_age_seconds"] = None if status.last_detector_age_seconds is None else round(status.last_detector_age_seconds, 2)
    data["alive"] = status.thread_alive and fresh
    data.setdefault("global_motion_rejections", 0)
    data.setdefault("active_events", 0)
    data.setdefault("current_groups", ())
    data.setdefault("last_event_summary", None)
    data.setdefault("target_fps", config.motion.target_fps)
    data.setdefault("detector_average_ms", 0.0)
    data.setdefault("annotation_average_ms", 0.0)
    data.setdefault("motion_thread_cpu_percent", 0.0)
    data.setdefault("annotations_rendered", 0)
    data.setdefault("idle_annotations_skipped", 0)
    return data


def _resolve_under(directory: Path, relative_path: str) -> Path | None:
    try:
        root = directory.resolve()
        candidate = (root / relative_path).resolve(strict=True)
    except OSError:
        return None
    if not candidate.is_file() or not candidate.is_relative_to(root):
        return None
    return candidate


def _dashboard_events(
    events: list[dict[str, Any]],
    config: AppConfig,
    *,
    summary_only: bool = False,
) -> list[dict[str, Any]]:
    output_root = config.camera.output_directory.resolve()
    prepared: list[dict[str, Any]] = []
    for event in events:
        item = (
            {
                field: event.get(field)
                for field in (
                    "event_id",
                    "status",
                    "capture_method",
                    "start_timestamp",
                    "end_timestamp",
                    "duration",
                    "provisional_category",
                    "source_event_id",
                    "track_id",
                    "classifier_label",
                    "classifier_confidence",
                    "snapshot_path",
                    "clip_path",
                    "full_frame_clip_path",
                )
            }
            if summary_only
            else dict(event)
        )
        event_directory: Path | None = None
        for field in ("snapshot_path", "clip_path", "full_frame_clip_path"):
            try:
                source_path = Path(str(event.get(field, ""))).resolve()
                item[f"{field}_relative"] = source_path.relative_to(output_root).as_posix()
                event_directory = source_path.parent
            except (OSError, ValueError):
                item[f"{field}_relative"] = None
        classification: dict[str, Any] = {}
        if event_directory is not None:
            try:
                loaded = json.loads((event_directory / "classification.json").read_text(encoding="utf-8"))
                classification = loaded if isinstance(loaded, dict) else {}
            except (OSError, json.JSONDecodeError):
                pass
        capture_method = event.get("capture_method")
        if capture_method == "manual_fire":
            item["display_label"] = "Manual fire"
            item["classification_status"] = "unclassified"
            item["classification_label_source"] = None
        elif capture_method == "auto_fire":
            auto_label = event.get("classifier_label")
            item["display_label"] = (
                f"Auto fire · {str(auto_label).title()}" if auto_label else "Auto fire"
            )
            item["classification_status"] = "auto_fire"
            item["classification_label_source"] = "automatic"
        else:
            item["display_label"] = classification.get("display_label", "Unclassified")
            item["classification_status"] = classification.get(
                "classification_status", "unclassified"
            )
            item["classification_label_source"] = classification.get("label_source")
        item["motion_label"] = event.get("provisional_category", "unclassified_motion")
        prepared.append(item)
    return prepared


def _review_item_payload(item: dict[str, Any]) -> dict[str, Any]:
    """Trim a classifier evidence record to the fields the Pi console polls."""

    item_id = str(item.get("item_id", ""))
    snapshot_relative = item.get("event_snapshot_relative")
    clip_relative = item.get("event_clip_relative")
    original_frame_relative = item.get("event_original_frame_relative")
    return {
        "item_id": item_id,
        "event_id": item.get("event_id"),
        "display_label": item.get("display_label", "Unclassified"),
        "classification_status": item.get("classification_status", "unclassified"),
        "label_source": item.get("label_source"),
        "top_label": item.get("top_label"),
        "top_confidence": item.get("top_confidence"),
        "review_suggestion_label": item.get("review_suggestion_label"),
        "review_suggestion_confidence": item.get("review_suggestion_confidence"),
        "model_suggestion": item.get("model_suggestion"),
        "human_label": item.get("human_label"),
        "human_verified": item.get("human_verified", False),
        "training_label": item.get("training_label"),
        "training_dataset_status": item.get("training_dataset_status"),
        "latency_ms": item.get("latency_ms"),
        "frame_number": item.get("frame_number"),
        "error": item.get("error"),
        "classifier_timestamp": item.get("classifier_timestamp"),
        "image_url": url_for("classifier_file", item_id=item_id),
        "event_snapshot_url": url_for("event_file", relative_path=snapshot_relative) if snapshot_relative else None,
        "event_clip_url": url_for("event_file", relative_path=clip_relative) if clip_relative else None,
        "event_original_frame_url": (
            url_for("event_file", relative_path=original_frame_relative)
            if original_frame_relative
            else None
        ),
    }


def _safe_capture_filename(filename: str) -> bool:
    path = Path(filename)
    return bool(filename) and path.name == filename and filename not in {".", ".."} and path.suffix.lower() in SUPPORTED_CAPTURE_SUFFIXES


def _resolve_capture_path(directory: Path, filename: str) -> Path | None:
    if not _safe_capture_filename(filename):
        return None
    try:
        root = directory.resolve()
        candidate = (root / filename).resolve(strict=True)
    except OSError:
        return None
    if not candidate.is_file() or not candidate.is_relative_to(root):
        return None
    return candidate


def create_app(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    app_config: AppConfig | None = None,
    camera_service: CameraService | None = None,
    vision_service: VisionService | None = None,
    motion_service: MotionProcessingService | None = None,
    manual_control_service: ManualControlService | None = None,
    runtime_provenance: dict[str, Any] | None = None,
    temperature_reader: Callable[[], float | None] = read_cpu_temperature,
    start_camera: bool = True,
    start_vision: bool = True,
) -> Flask:
    """Build the dashboard around already-created shared services."""

    app_config = app_config or load_config(config_path)
    app = Flask(__name__)
    demo_mode = os.environ.get("SQUIRREL_DEMO", "").strip() == "1"
    if camera_service is None:
        raise ValueError("create_app requires the shared camera runtime")
    vision = motion_service or vision_service
    if vision is None:
        raise ValueError("create_app requires the shared motion processor")
    camera = camera_service
    # Cache the construction-time snapshot. GETs must not re-read a newer tree.
    provenance = (
        deepcopy(runtime_provenance)
        if runtime_provenance is not None else build_runtime_provenance(app_config)
    )
    classifier_store = (
        motion_service.classifier_store
        if motion_service is not None and hasattr(motion_service, "classifier_store")
        else ClassifierEvidenceStore(app_config)
    )
    classifier_store.prepare()
    classifier_review_token = secrets.token_urlsafe(32)
    manual_control_token = secrets.token_urlsafe(32)
    shared_manual_control = (
        getattr(motion_service, "manual_control", None)
        if motion_service is not None
        else None
    )
    manual_control = manual_control_service or shared_manual_control
    if manual_control is None:
        manual_control = build_manual_control_service(
            app_config.pan_tilt,
            app_config.manual_control,
            app_config.valve,
            camera_service=camera,
            output_directory=app_config.camera.output_directory,
        )
    started_at = monotonic()
    capture_count_lock = threading.Lock()
    cached_capture_count = 0
    capture_count_cached_at: float | None = None
    app.extensions.update(
        camera_service=camera,
        vision_service=vision,
        motion_service=motion_service,
        squirrel_config=app_config,
        temperature_reader=temperature_reader,
        classifier_store=classifier_store,
        classifier_review_token=classifier_review_token,
        manual_control_service=manual_control,
        manual_control_token=manual_control_token,
        runtime_provenance=provenance,
    )

    if start_camera:
        camera.start()
    if start_vision:
        vision.start()

    def legacy_capture_count() -> int:
        """Avoid rescanning every legacy root capture on each status poll."""

        nonlocal cached_capture_count, capture_count_cached_at
        now = monotonic()
        with capture_count_lock:
            if (
                capture_count_cached_at is None
                or now - capture_count_cached_at >= CAPTURE_COUNT_CACHE_SECONDS
            ):
                cached_capture_count = len(list_capture_images(app_config.camera.output_directory))
                capture_count_cached_at = monotonic()
            return cached_capture_count

    def page_status() -> tuple[dict[str, Any], dict[str, Any], float | None, float]:
        camera_status = _camera_status_dict(camera.status(), app_config)
        detector_status = _vision_status_dict(vision.status(), app_config)
        try:
            temperature = temperature_reader()
            temperature = None if temperature is None else round(temperature, 1)
            if temperature is None:
                LOGGER.warning("Pi temperature unavailable", extra={"structured_data": {"event": "pi_temperature_failure"}})
        except Exception as exc:
            temperature = None
            LOGGER.warning("Pi temperature read failed", extra={"structured_data": {"event": "pi_temperature_failure", "error": str(exc)}})
        return camera_status, detector_status, temperature, monotonic() - started_at

    def classifier_status() -> dict[str, Any]:
        if motion_service is not None and hasattr(motion_service, "classifier"):
            return asdict(motion_service.classifier.status())
        return {
            "enabled": app_config.classifier.enabled,
            "thread_alive": False,
            "queue_depth": 0,
            "evidence_counts": classifier_store.counts(),
        }

    def auto_fire_status() -> dict[str, Any]:
        """Return one lightweight, fail-safe snapshot for every dashboard surface."""

        settings = app_config.auto_fire
        fallback: dict[str, Any] = {
            "enabled": settings.enabled,
            "state": "BLOCKED" if settings.enabled else "DISABLED",
            "candidates_evaluated": 0,
            "accepted": 0,
            "rejected": 0,
            "rejection_counts": {},
            "last_decision": None,
            "last_reason": None,
            "last_classification": None,
            "last_confidence": None,
            "last_event_id": None,
            "last_track_id": None,
            "cooldown_remaining_seconds": 0.0,
            "shots_in_rolling_window": 0,
            "max_shots_per_hour": settings.max_shots_per_hour,
            "remaining_shots_in_window": settings.max_shots_per_hour,
            "rate_limit_persistent": False,
            "rate_limit_state_error": None,
        }
        service = getattr(motion_service, "auto_fire", None)
        status_reader = getattr(service, "status", None)
        if not callable(status_reader):
            return fallback
        try:
            current = status_reader()
        except Exception as exc:
            fallback["state"] = "STATUS ERROR" if settings.enabled else "DISABLED"
            fallback["rate_limit_state_error"] = f"status_unavailable:{type(exc).__name__}"
            return fallback
        if not isinstance(current, dict):
            fallback["state"] = "STATUS ERROR" if settings.enabled else "DISABLED"
            fallback["rate_limit_state_error"] = "status_unavailable:invalid_payload"
            return fallback
        for key in (
            "enabled",
            "state",
            "candidates_evaluated",
            "accepted",
            "rejected",
            "rejection_counts",
            "last_reason",
            "last_classification",
            "last_confidence",
            "last_event_id",
            "last_track_id",
            "cooldown_remaining_seconds",
            "shots_in_rolling_window",
            "max_shots_per_hour",
            "remaining_shots_in_window",
            "rate_limit_persistent",
            "rate_limit_state_error",
        ):
            if key in current:
                fallback[key] = current[key]
        decision = current.get("last_decision")
        if isinstance(decision, dict):
            accepted = decision.get("accepted")
            if isinstance(accepted, bool):
                fallback["last_decision"] = "accepted" if accepted else "rejected"
            fallback["last_reason"] = decision.get("reason", fallback["last_reason"])
            fallback["last_classification"] = decision.get(
                "classifier_label", fallback["last_classification"]
            )
            fallback["last_confidence"] = decision.get(
                "classifier_confidence", fallback["last_confidence"]
            )
            fallback["last_event_id"] = decision.get("event_id", fallback["last_event_id"])
            fallback["last_track_id"] = decision.get("track_id", fallback["last_track_id"])
        elif decision is not None:
            fallback["last_decision"] = decision
        persistence = current.get("persistence")
        if isinstance(persistence, dict):
            fallback["rate_limit_persistent"] = bool(persistence.get("path"))
            fallback["rate_limit_state_error"] = persistence.get("error")
        shots = fallback["shots_in_rolling_window"]
        maximum = fallback["max_shots_per_hour"]
        if "remaining_shots_in_window" not in current and isinstance(shots, int) and isinstance(maximum, int):
            fallback["remaining_shots_in_window"] = max(0, maximum - shots)
        if "state" not in current:
            cooldown = fallback["cooldown_remaining_seconds"]
            if not fallback["enabled"]:
                fallback["state"] = "DISABLED"
            elif fallback["rate_limit_state_error"]:
                fallback["state"] = "BLOCKED"
            elif current.get("engagement_pending") is True:
                fallback["state"] = "ENGAGING"
            elif cooldown is None:
                fallback["state"] = "BLOCKED"
            elif isinstance(cooldown, (int, float)) and cooldown > 0:
                fallback["state"] = "COOLDOWN"
            else:
                fallback["state"] = "IDLE"
        return fallback

    @app.get("/")
    def dashboard() -> str:
        events = _dashboard_events(
            vision.recent_events()[:RECENT_EVENT_LIMIT],
            app_config,
            summary_only=True,
        )
        camera_data, detector, temperature, uptime = page_status()
        auto_fire = auto_fire_status()
        review_overview = classifier_store.overview()
        review_counts = {view: len(items) for view, items in review_overview.items()}
        return render_template(
            "dashboard.html",
            camera=camera_data,
            detector=detector,
            auto_fire=auto_fire,
            cpu_temperature=temperature,
            events=events,
            application_mode=APPLICATION_MODE,
            uptime_seconds=uptime,
            status_refresh_ms=round(app_config.dashboard.status_refresh_interval_seconds * 1000),
            review_items=[_review_item_payload(item) for item in review_overview["review"][:REVIEW_QUEUE_INITIAL]],
            review_counts=review_counts,
            review_token=classifier_review_token,
            demo_mode=demo_mode,
        )

    @app.get("/video-feed")
    @app.get("/video_feed")
    def video_feed() -> Any:
        return app.response_class(
            vision.mjpeg_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )

    def require_manual_control_token() -> None:
        supplied = request.headers.get("X-Control-Token", "")
        if not supplied or not hmac.compare_digest(supplied, manual_control_token):
            abort(403)

    def manual_control_status() -> dict[str, object]:
        status = manual_control.status()
        camera_status = camera.status()
        status.update(
            camera_online=camera_status.online,
            camera_frame_width=camera_status.width,
            camera_frame_height=camera_status.height,
        )
        return status

    def manual_control_error(exc: Exception) -> tuple[Any, int]:
        if isinstance(exc, FireCooldownError):
            return jsonify(error=str(exc), control=manual_control_status()), 409
        if isinstance(exc, ControlUnavailableError):
            return jsonify(error=str(exc), control=manual_control_status()), 503
        if isinstance(exc, (ControlError, ValueError, TypeError)):
            return jsonify(error=str(exc), control=manual_control_status()), 400
        raise exc

    @app.get("/manual-control")
    def manual_control_page() -> str:
        return render_template(
            "manual_control.html",
            control=manual_control_status(),
            control_token=manual_control_token,
            demo_mode=demo_mode,
        )

    @app.get("/api/manual-control")
    def api_manual_control() -> Any:
        return jsonify(control=manual_control_status())

    @app.post("/api/manual-control/move")
    def api_manual_control_move() -> Any:
        require_manual_control_token()
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="A JSON request body is required", control=manual_control_status()), 400
        try:
            manual_control.move(str(payload.get("direction", "")), payload.get("step"))
        except Exception as exc:
            return manual_control_error(exc)
        return jsonify(control=manual_control_status())

    @app.post("/api/manual-control/aim")
    def api_manual_control_aim() -> Any:
        require_manual_control_token()
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="A JSON request body is required", control=manual_control_status()), 400
        camera_status = camera.status()
        try:
            if not camera_status.online:
                raise ControlUnavailableError("Camera must be online before selecting an aim target")
            pixel_x, pixel_y = display_click_to_frame_pixel(
                payload.get("display_x"),
                payload.get("display_y"),
                payload.get("display_width"),
                payload.get("display_height"),
                camera_status.width,
                camera_status.height,
            )
            aim = manual_control.aim_at_pixel(pixel_x, pixel_y)
        except Exception as exc:
            return manual_control_error(exc)
        return jsonify(target=asdict(aim), control=manual_control_status())

    @app.post("/api/manual-control/calibration/edit/start")
    def api_manual_control_calibration_edit_start() -> Any:
        require_manual_control_token()
        try:
            manual_control.enter_calibration_edit_mode()
        except Exception as exc:
            return manual_control_error(exc)
        return jsonify(control=manual_control_status())

    @app.post("/api/manual-control/calibration/edit/stop")
    def api_manual_control_calibration_edit_stop() -> Any:
        require_manual_control_token()
        try:
            manual_control.exit_calibration_edit_mode()
        except Exception as exc:
            return manual_control_error(exc)
        return jsonify(control=manual_control_status())

    @app.post("/api/manual-control/fire")
    def api_manual_control_fire() -> Any:
        require_manual_control_token()
        try:
            held_position = manual_control.fire()
        except Exception as exc:
            return manual_control_error(exc)
        return jsonify(held_position=held_position, control=manual_control_status())

    @app.post("/api/manual-control/park")
    def api_manual_control_park() -> Any:
        require_manual_control_token()
        try:
            manual_control.park()
        except Exception as exc:
            return manual_control_error(exc)
        return jsonify(control=manual_control_status())

    @app.post("/api/manual-control/calibration")
    def api_manual_control_calibration() -> Any:
        require_manual_control_token()
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="A JSON request body is required", control=manual_control_status()), 400
        try:
            point = manual_control.save_active_calibration_point()
        except Exception as exc:
            return manual_control_error(exc)
        return jsonify(calibration_point=asdict(point), control=manual_control_status())

    @app.post("/api/manual-control/calibration/active")
    def api_manual_control_active_calibration() -> Any:
        require_manual_control_token()
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="A JSON request body is required", control=manual_control_status()), 400
        try:
            position = manual_control.select_calibration_point_for_edit(payload.get("point"))
        except Exception as exc:
            return manual_control_error(exc)
        return jsonify(
            moved_to_saved_aim=position is not None,
            commanded_position=None if position is None else asdict(position),
            control=manual_control_status(),
        )

    @app.post("/api/manual-control/calibration/pixel")
    def api_manual_control_calibration_pixel() -> Any:
        require_manual_control_token()
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="A JSON request body is required", control=manual_control_status()), 400
        camera_status = camera.status()
        try:
            if not camera_status.online:
                raise ControlUnavailableError("Camera must be online before selecting a calibration pixel")
            pixel_x, pixel_y = display_click_to_frame_pixel(
                payload.get("display_x"),
                payload.get("display_y"),
                payload.get("display_width"),
                payload.get("display_height"),
                camera_status.width,
                camera_status.height,
            )
            point = manual_control.set_active_calibration_pixel(
                pixel_x,
                pixel_y,
                frame_width=camera_status.width,
                frame_height=camera_status.height,
            )
        except Exception as exc:
            return manual_control_error(exc)
        return jsonify(calibration_point=asdict(point), control=manual_control_status())

    @app.get("/captures")
    def captures() -> str:
        all_captures = list_capture_images(app_config.camera.output_directory)
        page = max(request.args.get("page", default=1, type=int) or 1, 1)
        total_pages = max(1, math.ceil(len(all_captures) / CAPTURES_PER_PAGE))
        if page > total_pages and all_captures:
            abort(404)
        start = (page - 1) * CAPTURES_PER_PAGE
        return render_template("captures.html", captures=all_captures[start : start + CAPTURES_PER_PAGE], page=page, total_pages=total_pages, total_captures=len(all_captures), demo_mode=demo_mode)

    @app.get("/captures/<path:filename>")
    def capture_file(filename: str) -> Any:
        capture_path = _resolve_capture_path(app_config.camera.output_directory, filename)
        if capture_path is None:
            abort(404)
        return send_file(capture_path, conditional=True)

    @app.get("/events")
    def events() -> str:
        saved_events = load_events(app_config.camera.output_directory / "events")
        newest_events = list(reversed(saved_events))
        page = max(request.args.get("page", default=1, type=int) or 1, 1)
        total_pages = max(1, math.ceil(len(newest_events) / EVENTS_PER_PAGE))
        if page > total_pages and newest_events:
            abort(404)
        start = (page - 1) * EVENTS_PER_PAGE
        page_events = _dashboard_events(
            newest_events[start : start + EVENTS_PER_PAGE],
            app_config,
            summary_only=True,
        )
        return render_template(
            "events.html",
            events=page_events,
            page=page,
            total_pages=total_pages,
            total_events=len(newest_events),
            demo_mode=demo_mode,
        )

    @app.get("/files/<path:relative_path>")
    def output_file(relative_path: str) -> Any:
        path = _resolve_under(app_config.camera.output_directory, relative_path)
        if path is None:
            abort(404)
        return send_file(path, conditional=True)

    @app.get("/events/<path:relative_path>")
    def event_file(relative_path: str) -> Any:
        path = _resolve_under(app_config.camera.output_directory / "events", relative_path)
        if path is None:
            abort(404)
        return send_file(path, conditional=True)

    @app.get("/classifier-review")
    def classifier_review() -> str:
        state = request.args.get("state", "review")
        if state not in CLASSIFICATION_VIEWS:
            abort(404)
        training = classifier_store.training_summary()
        return render_template(
            "classifier_review.html",
            items=classifier_store.list_items(state),
            counts=classifier_store.counts(),
            selected_state=state,
            review_token=classifier_review_token,
            audit_log_filename=app_config.classifier.audit_log_filename,
            training=training,
            training_label_suggestions=classifier_store.training_label_suggestions(),
            classifier_labels=frozenset(VOC_LABELS[1:]),
            message=request.args.get("message"),
            demo_mode=demo_mode,
        )

    @app.post("/classifier-review/<item_id>/<decision>")
    def classifier_decision(item_id: str, decision: str) -> Any:
        supplied_token = request.form.get("review_token", "")
        if not hmac.compare_digest(supplied_token, classifier_review_token):
            abort(403)
        return_state = request.form.get("return_state")
        if return_state not in CLASSIFICATION_VIEWS:
            return_state = None
        try:
            if decision == "retry":
                if motion_service is None or not hasattr(motion_service, "classifier"):
                    abort(400)
                queued = motion_service.classifier.retry(item_id)
                message = "Classification retry queued" if queued else "Classifier is busy; retry remains in Errors"
                if request.form.get("format") == "json":
                    return jsonify(ok=bool(queued), message=message, item_id=item_id)
                return redirect(url_for("classifier_review", state=return_state or "errors", message=message))
            approval_label = request.form.get("custom_label", "").strip() or request.form.get("approval_label")
            record = classifier_store.review(item_id, decision, approval_label)
        except KeyError:
            abort(404)
        except (OSError, ValueError):
            abort(400)
        message = f"Event labeled {record['display_label']}"
        if record.get("training_dataset_status") == "included":
            message += f"; verified {record['training_label']} sample saved for training"
        if request.form.get("format") == "json":
            return jsonify(
                ok=True,
                message=message,
                item_id=item_id,
                classification_status=record["classification_status"],
                display_label=record["display_label"],
                training_label=record.get("training_label"),
                training_dataset_status=record.get("training_dataset_status"),
            )
        destination = "errors" if record["classification_status"] == "unclassified" else record["classification_status"]
        return redirect(url_for("classifier_review", state=return_state or destination, message=message))

    @app.post("/classifier-review/bulk")
    def classifier_bulk_decision() -> Any:
        supplied_token = request.form.get("review_token", "")
        if not hmac.compare_digest(supplied_token, classifier_review_token):
            abort(403)
        return_state = request.form.get("return_state", "review")
        if return_state not in CLASSIFICATION_VIEWS:
            return_state = "review"
        item_ids = list(dict.fromkeys(request.form.getlist("item_ids")))
        if not item_ids:
            return redirect(url_for("classifier_review", state=return_state, message="Select at least one event"))

        bulk_action = request.form.get("bulk_action")
        if bulk_action not in {"confirm-model", "approve", "unknown", "false-positive"}:
            abort(400)
        approval_label = request.form.get("custom_label", "").strip() or request.form.get("approval_label")
        updated = 0
        failed = 0
        for item_id in item_ids:
            try:
                classifier_store.review(item_id, bulk_action, approval_label)
                updated += 1
            except (KeyError, OSError, ValueError):
                failed += 1

        message = f"Updated {updated} event{'s' if updated != 1 else ''}"
        if failed:
            message += f"; {failed} could not be updated"
        return redirect(url_for("classifier_review", state=return_state, message=message))

    @app.get("/classifier-files/<item_id>")
    def classifier_file(item_id: str) -> Any:
        try:
            path = classifier_store.input_path(item_id)
        except (KeyError, ValueError):
            abort(404)
        return send_file(path, conditional=True)

    @app.get("/logs/<path:relative_path>")
    def log_file(relative_path: str) -> Any:
        path = _resolve_under(app_config.logging.directory, relative_path)
        if path is None:
            abort(404)
        return send_file(path, conditional=True)

    @app.get("/reports/latest")
    def latest_report() -> Any:
        path = _resolve_under(app_config.reporting.directory, "latest-report.html")
        if path is None:
            abort(404)
        return send_file(path, conditional=True)

    @app.get("/reports/<path:relative_path>")
    def report_file(relative_path: str) -> Any:
        path = _resolve_under(app_config.reporting.directory, relative_path)
        if path is None:
            abort(404)
        return send_file(path, conditional=True)

    @app.get("/api/status")
    def api_status() -> Any:
        camera_data, detector, temperature, uptime = page_status()
        events = vision.recent_events()
        return jsonify(
            application_mode=APPLICATION_MODE,
            runtime_provenance=provenance,
            application_uptime_seconds=round(uptime, 1),
            camera=camera_data,
            detector=detector,
            classifier=classifier_status(),
            auto_fire=auto_fire_status(),
            manual_recording=manual_control_status().get("recording"),
            cpu_temperature_c=temperature,
            total_events=detector["accepted_events"],
            total_snapshots=legacy_capture_count() + len(events),
            last_event_time=detector["last_event"],
            last_snapshot_time=detector["last_snapshot"],
        )

    @app.get("/api/health")
    def api_health() -> Any:
        camera_data, detector, _, uptime = page_status()
        return jsonify(
            application_uptime_seconds=round(uptime, 1),
            camera_alive=camera_data["alive"],
            detector_alive=detector["alive"],
            last_frame_received=camera_data["last_frame_received"],
            last_detector_update=detector["last_detector_update"],
            last_event=detector["last_event"],
            last_snapshot=detector["last_snapshot"],
            processing_fps=detector["processing_fps"],
            frames_processed=detector["frames_processed"],
            candidates_seen=detector["candidates_seen"],
            accepted_events=detector["accepted_events"],
            rejected_events=detector["rejected_events"],
            snapshots_saved=detector["snapshots_saved"],
            global_motion_rejections=detector["global_motion_rejections"],
            camera_read_failures=camera_data["read_failures"],
            camera_reconnects=camera_data["reconnects"],
            camera_open_count=camera_data["camera_open_count"],
            dashboard_viewers=camera_data["dashboard_viewers"],
            dashboard_stream_fps=camera_data["dashboard_stream_fps"],
            dashboard_frames_encoded=camera_data["dashboard_frames_encoded"],
            dashboard_last_jpeg_bytes=camera_data["dashboard_last_jpeg_bytes"],
            dashboard_estimated_egress_mbps=camera_data["dashboard_estimated_egress_mbps"],
            dashboard_encode_failures=camera_data["dashboard_encode_failures"],
            dashboard_encoder_alive=camera_data["dashboard_encoder_alive"],
            dashboard_encode_error=camera_data["dashboard_encode_error"],
            pre_roll_frames_buffered=camera_data["pre_roll_frames_buffered"],
            pre_roll_buffer_fps=camera_data["pre_roll_buffer_fps"],
            pre_roll_frames_reused=camera_data["pre_roll_frames_reused"],
            shared_frame_borrows=camera_data["shared_frame_borrows"],
            capture_fps=camera_data["fps"],
            capture_read_average_ms=camera_data["capture_read_average_ms"],
            published_frame_copy_average_ms=camera_data["published_frame_copy_average_ms"],
            capture_thread_cpu_percent=camera_data["capture_thread_cpu_percent"],
            active_events=detector["active_events"],
            detector_target_fps=detector["target_fps"],
            detector_average_ms=detector["detector_average_ms"],
            annotation_average_ms=detector["annotation_average_ms"],
            motion_thread_cpu_percent=detector["motion_thread_cpu_percent"],
            annotations_rendered=detector["annotations_rendered"],
            idle_annotations_skipped=detector["idle_annotations_skipped"],
            last_error=detector["last_error"] or camera_data["error"],
            capture_directory_writable=detector["capture_directory_writable"],
            camera_state=camera_data["state"],
            detector_state=detector["state"],
            classifier=classifier_status(),
            auto_fire=auto_fire_status(),
            manual_recording=manual_control_status().get("recording"),
        )

    @app.get("/api/recent-events")
    @app.get("/api/events")
    def api_recent_events() -> Any:
        recent = vision.recent_events()
        requested_limit = request.args.get("limit", type=int)
        if requested_limit is not None:
            limit = min(max(requested_limit, 1), app_config.motion.recent_event_limit)
            recent = recent[:limit]
        summary_only = request.args.get("summary", "").strip().lower() in {"1", "true", "yes"}
        events = _dashboard_events(recent, app_config, summary_only=summary_only)
        for event in events:
            snapshot = event.get("snapshot_path_relative")
            clip = event.get("clip_path_relative")
            event["snapshot_url"] = url_for("output_file", relative_path=snapshot) if snapshot else None
            event["clip_url"] = url_for("output_file", relative_path=clip) if clip else None
        return jsonify(events=events, count=len(events))

    @app.get("/api/classifier-review")
    def api_classifier_review() -> Any:
        state = request.args.get("state", "review")
        if state not in CLASSIFICATION_VIEWS:
            abort(404)
        limit = request.args.get("limit", default=REVIEW_QUEUE_API_LIMIT, type=int) or REVIEW_QUEUE_API_LIMIT
        limit = min(max(limit, 1), 500)
        overview = classifier_store.overview()
        counts = {view: len(items) for view, items in overview.items()}
        items = [_review_item_payload(item) for item in overview[state][:limit]]
        return jsonify(state=state, counts=counts, items=items, limit=limit)

    @app.errorhandler(Exception)
    def report_flask_error(exc: Exception) -> Any:
        if isinstance(exc, HTTPException):
            return exc
        LOGGER.error("Flask request failed", extra={"structured_data": {"event": "flask_error", "error": str(exc)}}, exc_info=True)
        return jsonify(error="Internal server error"), 500

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the private motion diagnostics dashboard")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    return parser


def main() -> int:
    from .app import main as combined_main

    return combined_main()


if __name__ == "__main__":
    raise SystemExit(main())
