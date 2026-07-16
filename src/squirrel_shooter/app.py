"""Combined shared-camera motion watcher and private Tailscale dashboard."""

from __future__ import annotations

import argparse
import logging
import threading
from dataclasses import replace
from pathlib import Path
from time import monotonic
from typing import Any

import cv2
from werkzeug.serving import BaseWSGIServer, make_server

from .camera_preview import display_available
from .camera_service import CameraService
from .config import AppConfig, ConfigError, DEFAULT_CONFIG_PATH, load_config
from .diagnostics import configure_logging
from .motion_runtime import MotionProcessingService
from .web_dashboard import create_app


LOGGER = logging.getLogger(__name__)
WINDOW_TITLE = "Squirrel Squirter shared runtime (q quit, s still, e test event, r report)"


class ApplicationRuntime:
    """Coordinate one camera owner and its motion consumer."""

    def __init__(
        self,
        config: AppConfig,
        *,
        camera: CameraService | None = None,
        motion: MotionProcessingService | None = None,
    ) -> None:
        self.config = config
        self.camera = camera or CameraService(
            config.camera,
            shared_settings=config.shared_camera,
            jpeg_quality=config.dashboard.jpeg_quality,
            encode_jpeg=True,
        )
        self.motion = motion or MotionProcessingService(self.camera, config)
        self._lock = threading.Lock()
        self._started = False

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
        self.camera.start()
        try:
            self.motion.start()
        except Exception:
            self.camera.stop(timeout=self.config.runtime.shutdown_timeout_seconds)
            with self._lock:
                self._started = False
            raise

    def stop(self) -> None:
        with self._lock:
            if not self._started:
                return
            self._started = False
        timeout = self.config.runtime.shutdown_timeout_seconds
        self.motion.stop(timeout=timeout)
        self.camera.stop(timeout=timeout)

    def status(self) -> dict[str, Any]:
        return {"camera": self.camera.status(), "motion": self.motion.status()}


class DashboardServer:
    """A stoppable threaded Werkzeug server used by the combined application."""

    def __init__(self, flask_app: Any, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._server: BaseWSGIServer = make_server(host, port, flask_app, threaded=True)
        self.port = self._server.server_port
        self._thread = threading.Thread(target=self._serve, name="squirrel-dashboard", daemon=True)
        self.error: str | None = None

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._server.shutdown()
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=timeout)
        self._server.server_close()

    def _serve(self) -> None:
        try:
            self._server.serve_forever()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            LOGGER.error(
                "Dashboard server failed; motion processing remains active",
                extra={"structured_data": {"event": "dashboard_server_failure", "error": self.error}},
                exc_info=True,
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run motion watching and the private dashboard from one shared camera")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--headless", action="store_true", default=None, help="disable the local OpenCV preview")
    parser.add_argument("--no-dashboard", action="store_true", help="run the shared watcher without the HTTP dashboard")
    parser.add_argument("--host", help="override dashboard.host")
    parser.add_argument("--port", type=int, help="override dashboard.port")
    return parser


def _apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    dashboard = config.dashboard
    if args.no_dashboard:
        dashboard = replace(dashboard, enabled=False)
    if args.host:
        dashboard = replace(dashboard, host=args.host)
    if args.port is not None:
        if not 1 <= args.port <= 65535:
            raise ConfigError("--port must be between 1 and 65535")
        dashboard = replace(dashboard, port=args.port)
    runtime = config.runtime if args.headless is None else replace(config.runtime, headless=args.headless)
    return replace(config, dashboard=dashboard, runtime=runtime)


def run(config: AppConfig) -> int:
    """Run until Ctrl+C or local q, then shut every subsystem down in order."""

    configure_logging(config.logging, config.storage.max_log_files)
    runtime = ApplicationRuntime(config)
    server: DashboardServer | None = None
    runtime.start()
    try:
        if config.dashboard.enabled:
            flask_app = create_app(
                app_config=config,
                camera_service=runtime.camera,
                motion_service=runtime.motion,
                start_camera=False,
                start_vision=False,
            )
            server = DashboardServer(flask_app, config.dashboard.host, config.dashboard.port)
            server.start()
            print(f"Dashboard listening on http://{config.dashboard.host}:{server.port}")
        else:
            print("Dashboard disabled; motion processing is still active.")
    except Exception:
        runtime.stop()
        raise
    show_preview = not config.runtime.headless and display_available()
    if not show_preview:
        print("Headless shared runtime started. Press Ctrl+C to stop safely.")
    clean = False
    last_server_error: str | None = None
    idle_wait = threading.Event()
    try:
        while True:
            if server is not None and server.error and server.error != last_server_error:
                print(f"Dashboard error: {server.error}. Motion watching remains active.")
                last_server_error = server.error
            if show_preview:
                frame = runtime.camera.latest_annotated_frame()
                if frame is not None:
                    cv2.imshow(WINDOW_TITLE, frame)
                key = cv2.waitKey(50) & 0xFF
                if key == ord("q"):
                    clean = True
                    break
                if key == ord("s"):
                    path = runtime.motion.save_manual_still()
                    print(f"Saved manual still: {path.resolve()}" if path else "No annotated frame is available yet.")
                if key == ord("e"):
                    print("Forced test event queued." if runtime.motion.request_forced_event() else "No current candidate is available.")
                if key == ord("r"):
                    paths = runtime.motion.rebuild_report()
                    print("Rebuilt report: " + ", ".join(str(path.resolve()) for path in paths))
            else:
                idle_wait.wait(0.5)
    except KeyboardInterrupt:
        clean = True
        print("Stopping combined application cleanly.")
    finally:
        if show_preview:
            cv2.destroyAllWindows()
        if server is not None:
            server.stop(timeout=config.runtime.shutdown_timeout_seconds)
        runtime.stop()
        LOGGER.info(
            "Combined application shutdown complete",
            extra={"structured_data": {"event": "combined_shutdown", "clean": clean, "finished_at_monotonic": monotonic()}},
        )
    return 0 if clean else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = _apply_overrides(load_config(args.config), args)
        return run(config)
    except ConfigError as exc:
        print(f"Configuration error: {exc}")
        return 2
    except Exception as exc:
        LOGGER.error("Combined application failed", exc_info=True)
        print(f"Application error: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
