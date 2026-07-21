"""Read-only Flask dashboard for camera, motion, events, and diagnostics."""

from __future__ import annotations

import argparse
import hmac
import json
import logging
import math
import os
import secrets
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Any

from flask import Flask, abort, jsonify, redirect, render_template, request, send_file, url_for
from werkzeug.exceptions import HTTPException

from .camera_service import CameraService, CameraStatus
from .classifier import CLASSIFICATION_VIEWS, ClassifierEvidenceStore
from .config import DEFAULT_CONFIG_PATH, AppConfig, load_config
from .event_report import load_events
from .motion_runtime import MotionProcessingService
from .vision_service import VisionService, VisionStatus


LOGGER = logging.getLogger(__name__)
SUPPORTED_CAPTURE_SUFFIXES = frozenset({".jpg", ".jpeg"})
RECENT_EVENT_LIMIT = 5
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


def _dashboard_events(events: list[dict[str, Any]], config: AppConfig) -> list[dict[str, Any]]:
    output_root = config.camera.output_directory.resolve()
    prepared: list[dict[str, Any]] = []
    for event in events:
        item = dict(event)
        event_directory: Path | None = None
        for field in ("snapshot_path", "clip_path"):
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
        item["display_label"] = classification.get("display_label", "Unclassified")
        item["classification_status"] = classification.get("classification_status", "unclassified")
        item["classification_label_source"] = classification.get("label_source")
        item["motion_label"] = event.get("provisional_category", "unclassified_motion")
        prepared.append(item)
    return prepared


def _review_item_payload(item: dict[str, Any]) -> dict[str, Any]:
    """Trim a classifier evidence record to the fields the Pi console polls."""

    item_id = str(item.get("item_id", ""))
    snapshot_relative = item.get("event_snapshot_relative")
    clip_relative = item.get("event_clip_relative")
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
    temperature_reader: Callable[[], float | None] = read_cpu_temperature,
    start_camera: bool = True,
    start_vision: bool = True,
) -> Flask:
    """Build a read-only dashboard around already-created shared services."""

    app_config = app_config or load_config(config_path)
    app = Flask(__name__)
    demo_mode = os.environ.get("SQUIRREL_DEMO", "").strip() == "1"
    if camera_service is None:
        raise ValueError("create_app requires the shared camera runtime")
    vision = motion_service or vision_service
    if vision is None:
        raise ValueError("create_app requires the shared motion processor")
    camera = camera_service
    classifier_store = (
        motion_service.classifier_store
        if motion_service is not None and hasattr(motion_service, "classifier_store")
        else ClassifierEvidenceStore(app_config)
    )
    classifier_store.prepare()
    classifier_review_token = secrets.token_urlsafe(32)
    started_at = monotonic()
    app.extensions.update(
        camera_service=camera,
        vision_service=vision,
        motion_service=motion_service,
        squirrel_config=app_config,
        temperature_reader=temperature_reader,
        classifier_store=classifier_store,
        classifier_review_token=classifier_review_token,
    )

    if start_camera:
        camera.start()
    if start_vision:
        vision.start()
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

    @app.get("/")
    def dashboard() -> str:
        events = _dashboard_events(vision.recent_events(), app_config)
        camera_data, detector, temperature, uptime = page_status()
        review_overview = classifier_store.overview()
        review_counts = {view: len(items) for view, items in review_overview.items()}
        return render_template(
            "dashboard.html",
            camera=camera_data,
            detector=detector,
            cpu_temperature=temperature,
            events=events[:RECENT_EVENT_LIMIT],
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
        all_events = _dashboard_events(list(reversed(saved_events)), app_config)
        page = max(request.args.get("page", default=1, type=int) or 1, 1)
        total_pages = max(1, math.ceil(len(all_events) / EVENTS_PER_PAGE))
        if page > total_pages and all_events:
            abort(404)
        start = (page - 1) * EVENTS_PER_PAGE
        return render_template(
            "events.html",
            events=all_events[start : start + EVENTS_PER_PAGE],
            page=page,
            total_pages=total_pages,
            total_events=len(all_events),
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
        captures = list_capture_images(app_config.camera.output_directory)
        events = vision.recent_events()
        return jsonify(
            application_mode=APPLICATION_MODE,
            application_uptime_seconds=round(uptime, 1),
            camera=camera_data,
            detector=detector,
            classifier=classifier_status(),
            cpu_temperature_c=temperature,
            total_events=detector["accepted_events"],
            total_snapshots=len(captures) + len(events),
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
            active_events=detector["active_events"],
            last_error=detector["last_error"] or camera_data["error"],
            capture_directory_writable=detector["capture_directory_writable"],
            camera_state=camera_data["state"],
            detector_state=detector["state"],
            classifier=classifier_status(),
        )

    @app.get("/api/recent-events")
    @app.get("/api/events")
    def api_recent_events() -> Any:
        events = _dashboard_events(vision.recent_events(), app_config)
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
