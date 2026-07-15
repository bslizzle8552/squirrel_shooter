"""Private, observational Flask dashboard for the shared camera service."""

from __future__ import annotations

import argparse
import atexit
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, render_template, request, send_file

from .camera_service import CameraService, CameraStatus
from .config import DEFAULT_CONFIG_PATH, AppConfig, load_config


SUPPORTED_CAPTURE_SUFFIXES = frozenset({".jpg", ".jpeg"})
RECENT_CAPTURE_LIMIT = 12
CAPTURES_PER_PAGE = 24
APPLICATION_MODE = "observation-only"


@dataclass(frozen=True)
class CaptureImage:
    """A gallery-safe image entry derived from the configured directory."""

    filename: str
    timestamp: str


def list_capture_images(directory: Path) -> list[CaptureImage]:
    """List supported captures newest-first without caching directory contents."""

    try:
        candidates = [
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_CAPTURE_SUFFIXES
            and _resolve_capture_path(directory, path.name) is not None
        ]
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return []

    def sort_key(path: Path) -> tuple[float, str]:
        try:
            modified = path.stat().st_mtime
        except OSError:
            modified = 0.0
        return modified, path.name.lower()

    images: list[CaptureImage] = []
    for path in sorted(candidates, key=sort_key, reverse=True):
        try:
            timestamp = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
        except OSError:
            continue
        images.append(
            CaptureImage(
                filename=path.name,
                timestamp=timestamp.strftime("%b %d, %Y at %I:%M:%S %p"),
            )
        )
    return images


def read_cpu_temperature() -> float | None:
    """Read the Pi CPU temperature from Linux sysfs when it is available."""

    temperature_path = Path("/sys/class/thermal/thermal_zone0/temp")
    try:
        value = float(temperature_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    return value / 1000.0 if value > 1000 else value


def _camera_status_dict(status: CameraStatus, app_config: AppConfig) -> dict[str, Any]:
    device_index = app_config.camera.device_index
    return {
        "online": status.online,
        "configured_device": {
            "index": device_index,
            "label": f"/dev/video{device_index}",
        },
        "resolution": {
            "width": status.width,
            "height": status.height,
            "label": f"{status.width}×{status.height}",
        },
        "fps": round(status.fps, 1),
        "error": status.error,
    }


def _safe_capture_filename(filename: str) -> bool:
    path = Path(filename)
    return (
        bool(filename)
        and path.name == filename
        and filename not in {".", ".."}
        and path.suffix.lower() in SUPPORTED_CAPTURE_SUFFIXES
    )


def _resolve_capture_path(directory: Path, filename: str) -> Path | None:
    """Resolve one supported capture while containing symlinks to the gallery root."""

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
    camera_service: CameraService | None = None,
    temperature_reader: Callable[[], float | None] = read_cpu_temperature,
    start_camera: bool = True,
) -> Flask:
    """Build the dashboard with injectable hardware boundaries for testing."""

    app_config: AppConfig = load_config(config_path)
    app = Flask(__name__)
    service = camera_service or CameraService(app_config.camera)
    app.extensions["camera_service"] = service
    app.extensions["squirrel_config"] = app_config
    app.extensions["temperature_reader"] = temperature_reader

    if start_camera:
        service.start()
    if camera_service is None:
        atexit.register(service.stop)

    def page_status() -> tuple[dict[str, Any], float | None]:
        status = _camera_status_dict(service.status(), app_config)
        temperature = temperature_reader()
        if temperature is not None:
            temperature = round(temperature, 1)
        return status, temperature

    @app.get("/")
    def dashboard() -> str:
        captures = list_capture_images(app_config.camera.output_directory)
        status, temperature = page_status()
        return render_template(
            "dashboard.html",
            camera=status,
            cpu_temperature=temperature,
            captures=captures[:RECENT_CAPTURE_LIMIT],
            total_captures=len(captures),
            application_mode=APPLICATION_MODE,
        )

    @app.get("/video-feed")
    def video_feed() -> Any:
        return app.response_class(
            service.mjpeg_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )

    @app.get("/captures")
    def captures() -> str:
        all_captures = list_capture_images(app_config.camera.output_directory)
        page = request.args.get("page", default=1, type=int) or 1
        page = max(page, 1)
        total_pages = max(1, math.ceil(len(all_captures) / CAPTURES_PER_PAGE))
        if page > total_pages and all_captures:
            abort(404)
        start = (page - 1) * CAPTURES_PER_PAGE
        return render_template(
            "captures.html",
            captures=all_captures[start : start + CAPTURES_PER_PAGE],
            page=page,
            total_pages=total_pages,
            total_captures=len(all_captures),
        )

    @app.get("/captures/<path:filename>")
    def capture_file(filename: str) -> Any:
        capture_path = _resolve_capture_path(app_config.camera.output_directory, filename)
        if capture_path is None:
            abort(404)
        return send_file(capture_path, conditional=True)

    @app.get("/api/status")
    def api_status() -> Any:
        status, temperature = page_status()
        return jsonify(
            camera=status,
            cpu_temperature_c=temperature,
            capture_count=len(list_capture_images(app_config.camera.output_directory)),
            application_mode=APPLICATION_MODE,
        )

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the private camera dashboard")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    app = create_app(args.config)
    service: CameraService = app.extensions["camera_service"]
    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    finally:
        service.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
