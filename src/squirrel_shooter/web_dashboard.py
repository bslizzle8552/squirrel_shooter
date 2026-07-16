"""Read-only Flask dashboard for camera, motion, events, and diagnostics."""

from __future__ import annotations

import argparse
import logging
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Any

from flask import Flask, abort, jsonify, render_template, request, send_file
from werkzeug.exceptions import HTTPException

from .camera_service import CameraService, CameraStatus
from .config import DEFAULT_CONFIG_PATH, AppConfig, load_config
from .event_report import load_events
from .motion_runtime import MotionProcessingService
from .vision_service import VisionService, VisionStatus


LOGGER = logging.getLogger(__name__)
SUPPORTED_CAPTURE_SUFFIXES = frozenset({".jpg", ".jpeg"})
RECENT_EVENT_LIMIT = 5
CAPTURES_PER_PAGE = 24
EVENTS_PER_PAGE = 20
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
        for field in ("snapshot_path", "clip_path"):
            try:
                item[f"{field}_relative"] = Path(str(event.get(field, ""))).resolve().relative_to(output_root).as_posix()
            except (OSError, ValueError):
                item[f"{field}_relative"] = None
        prepared.append(item)
    return prepared


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
    if camera_service is None:
        raise ValueError("create_app requires the shared camera runtime")
    vision = motion_service or vision_service
    if vision is None:
        raise ValueError("create_app requires the shared motion processor")
    camera = camera_service
    started_at = monotonic()
    app.extensions.update(
        camera_service=camera,
        vision_service=vision,
        motion_service=motion_service,
        squirrel_config=app_config,
        temperature_reader=temperature_reader,
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

    @app.get("/")
    def dashboard() -> str:
        events = _dashboard_events(vision.recent_events(), app_config)
        camera_data, detector, temperature, uptime = page_status()
        return render_template(
            "dashboard.html",
            camera=camera_data,
            detector=detector,
            cpu_temperature=temperature,
            events=events[:RECENT_EVENT_LIMIT],
            application_mode=APPLICATION_MODE,
            uptime_seconds=uptime,
            status_refresh_ms=round(app_config.dashboard.status_refresh_interval_seconds * 1000),
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
        return render_template("captures.html", captures=all_captures[start : start + CAPTURES_PER_PAGE], page=page, total_pages=total_pages, total_captures=len(all_captures))

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
        )

    @app.get("/api/recent-events")
    @app.get("/api/events")
    def api_recent_events() -> Any:
        events = vision.recent_events()
        return jsonify(events=events, count=len(events))

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
