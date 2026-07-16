"""Single-owner camera runtime shared by motion processing and the dashboard."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Any

import cv2
import numpy as np

from .camera_common import FrameRateMeter, capture_dimensions, open_camera
from .config import CameraConfig, SharedCameraConfig


LOGGER = logging.getLogger(__name__)


def is_raspberry_pi() -> bool:
    """Return whether the host identifies itself as Raspberry Pi hardware."""

    for model_path in (Path("/proc/device-tree/model"), Path("/sys/firmware/devicetree/base/model")):
        try:
            if "raspberry pi" in model_path.read_text(encoding="utf-8").lower():
                return True
        except OSError:
            continue
    return False


@dataclass(frozen=True)
class CameraStatus:
    """A thread-safe snapshot of the shared camera's current state."""

    online: bool
    width: int
    height: int
    fps: float
    error: str | None
    last_frame_at: str | None = None
    last_frame_age_seconds: float | None = None
    frames_received: int = 0
    thread_alive: bool = False
    reported_fps: float = 0.0
    read_failures: int = 0
    reconnects: int = 0
    camera_open_count: int = 0
    annotated_frames: int = 0
    last_annotated_at: str | None = None
    annotated_frame_age_seconds: float | None = None


@dataclass(frozen=True)
class FramePacket:
    """One raw frame published by the sole camera-reading thread."""

    sequence: int
    frame: np.ndarray
    received_at: str
    received_monotonic: float = 0.0


class CameraService:
    """Open the camera once, reconnect safely, and publish raw/annotated frames."""

    def __init__(
        self,
        settings: CameraConfig,
        *,
        shared_settings: SharedCameraConfig | None = None,
        capture_factory: Callable[[CameraConfig], Any] = open_camera,
        platform_checker: Callable[[], bool] = is_raspberry_pi,
        jpeg_quality: int = 80,
        encode_jpeg: bool = True,
    ) -> None:
        self.settings = settings
        self.shared_settings = shared_settings or SharedCameraConfig(
            reconnect_enabled=True,
            maximum_consecutive_read_failures=settings.reopen_after_failed_reads,
            reconnect_delay_seconds=settings.reopen_delay_seconds,
            consumer_wait_timeout_seconds=1.0,
            annotated_frame_stale_seconds=3.0,
        )
        self._capture_factory = capture_factory
        self._platform_checker = platform_checker
        self._jpeg_quality = jpeg_quality
        self._encode_jpeg = encode_jpeg
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture: Any | None = None
        self._online = False
        self._width = settings.requested_width
        self._height = settings.requested_height
        self._fps = 0.0
        self._reported_fps = 0.0
        self._error: str | None = "Camera has not started"
        self._latest_frame: np.ndarray | None = None
        self._latest_raw_jpeg: bytes | None = None
        self._latest_annotated_frame: np.ndarray | None = None
        self._latest_annotated_jpeg: bytes | None = None
        self._sequence = 0
        self._annotated_sequence = 0
        self._last_annotated_source_sequence = -1
        self._last_frame_at: str | None = None
        self._last_frame_monotonic: float | None = None
        self._last_annotated_at: str | None = None
        self._last_annotated_monotonic: float | None = None
        self._frames_received = 0
        self._annotated_frames = 0
        self._read_failures = 0
        self._reconnects = 0
        self._camera_open_count = 0

    @property
    def stopped(self) -> bool:
        return self._stop_event.is_set()

    def start(self) -> None:
        """Start one camera owner; repeated starts never open another handle."""

        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._online = False
            self._error = None
            self._latest_frame = None
            self._latest_raw_jpeg = None
            self._latest_annotated_frame = None
            self._latest_annotated_jpeg = None
            self._last_frame_at = None
            self._last_frame_monotonic = None
            self._last_annotated_at = None
            self._last_annotated_monotonic = None
            self._thread = threading.Thread(target=self._capture_loop, name="squirrel-camera", daemon=True)
            self._thread.start()
        LOGGER.info("Shared camera runtime started", extra={"structured_data": {"event": "camera_runtime_started"}})

    def stop(self, timeout: float = 3.0) -> None:
        """Stop the whole camera runtime and release its only capture handle."""

        self._stop_event.set()
        with self._condition:
            capture = self._capture
            self._condition.notify_all()
        if capture is not None:
            try:
                capture.release()
            except Exception:
                LOGGER.exception("Camera release failed during shutdown")
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        with self._condition:
            if thread is not None and thread.is_alive():
                self._error = "Camera thread did not stop before the shutdown timeout"
            self._online = False
            self._condition.notify_all()
        LOGGER.info("Shared camera runtime stopped", extra={"structured_data": {"event": "camera_runtime_stopped"}})

    def status(self) -> CameraStatus:
        with self._condition:
            now = monotonic()
            raw_age = None if self._last_frame_monotonic is None else max(0.0, now - self._last_frame_monotonic)
            annotated_age = None if self._last_annotated_monotonic is None else max(0.0, now - self._last_annotated_monotonic)
            return CameraStatus(
                self._online,
                self._width,
                self._height,
                self._fps,
                self._error,
                self._last_frame_at,
                raw_age,
                self._frames_received,
                self._capture_thread_alive(),
                self._reported_fps,
                self._read_failures,
                self._reconnects,
                self._camera_open_count,
                self._annotated_frames,
                self._last_annotated_at,
                annotated_age,
            )

    def latest_frame(self, *, copy: bool = True) -> np.ndarray | None:
        with self._condition:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy() if copy else self._latest_frame

    def latest_annotated_frame(self, *, copy: bool = True) -> np.ndarray | None:
        with self._condition:
            if self._latest_annotated_frame is None:
                return None
            return self._latest_annotated_frame.copy() if copy else self._latest_annotated_frame

    def wait_for_frame(self, after_sequence: int, timeout: float | None = None) -> FramePacket | None:
        """Wait for a new raw frame without reading or reopening the camera."""

        wait_timeout = self.shared_settings.consumer_wait_timeout_seconds if timeout is None else timeout
        with self._condition:
            self._condition.wait_for(
                lambda: self._sequence > after_sequence or self._stop_event.is_set(),
                timeout=wait_timeout,
            )
            if self._sequence <= after_sequence or self._latest_frame is None or self._last_frame_at is None:
                return None
            return FramePacket(
                self._sequence,
                self._latest_frame.copy(),
                self._last_frame_at,
                self._last_frame_monotonic or monotonic(),
            )

    def publish_annotated(self, source_sequence: int, frame: np.ndarray) -> bool:
        """Publish motion annotations for the dashboard; never touch the camera."""

        encoded, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
        if not encoded:
            raise RuntimeError("OpenCV could not encode an annotated dashboard frame")
        now_iso = datetime.now().astimezone().isoformat(timespec="milliseconds")
        now = monotonic()
        with self._condition:
            if source_sequence < self._last_annotated_source_sequence:
                return False
            self._latest_annotated_frame = frame.copy()
            self._latest_annotated_jpeg = jpeg.tobytes()
            self._last_annotated_source_sequence = source_sequence
            self._annotated_sequence += 1
            self._annotated_frames += 1
            self._last_annotated_at = now_iso
            self._last_annotated_monotonic = now
            self._condition.notify_all()
        return True

    def mjpeg_frames(
        self,
        *,
        maximum_fps: float | None = None,
        annotated_only: bool = False,
    ) -> Iterator[bytes]:
        """Yield MJPEG frames, optionally waiting for watcher annotations only."""

        sequence = -1
        last_yield = 0.0
        interval = 0.0 if maximum_fps is None else 1.0 / maximum_fps
        while not self._stop_event.is_set():
            def frame_ready() -> bool:
                if self._stop_event.is_set():
                    return True
                if self._latest_annotated_jpeg is not None:
                    return self._annotated_sequence != sequence
                return not annotated_only and self._latest_raw_jpeg is not None and self._sequence != sequence

            with self._condition:
                self._condition.wait_for(
                    frame_ready,
                    timeout=self.shared_settings.consumer_wait_timeout_seconds,
                )
                if self._stop_event.is_set():
                    return
                if self._annotated_sequence:
                    current_sequence = self._annotated_sequence
                    jpeg = self._latest_annotated_jpeg
                elif annotated_only:
                    continue
                else:
                    current_sequence = self._sequence
                    jpeg = self._latest_raw_jpeg
                if jpeg is None or current_sequence == sequence:
                    continue
                sequence = current_sequence
            remaining = interval - (monotonic() - last_yield)
            if remaining > 0 and self._stop_event.wait(remaining):
                return
            last_yield = monotonic()
            yield b"--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-cache\r\n\r\n" + jpeg + b"\r\n"

    def _capture_thread_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _set_offline(self, error: str) -> None:
        with self._condition:
            self._online = False
            self._fps = 0.0
            self._error = error
            self._condition.notify_all()

    def _capture_loop(self) -> None:
        if not self._platform_checker():
            self._set_offline("Camera capture is disabled because this host is not a Raspberry Pi")
            return
        first_open = True
        while not self._stop_event.is_set():
            capture: Any | None = None
            try:
                capture = self._capture_factory(self.settings)
                meter = FrameRateMeter()
                width, height, reported_fps = capture_dimensions(capture)
                with self._condition:
                    self._capture = capture
                    self._camera_open_count += 1
                    if not first_open:
                        self._reconnects += 1
                    if width > 0 and height > 0:
                        self._width, self._height = width, height
                    self._reported_fps = reported_fps
                    self._error = None
                    self._condition.notify_all()
                first_open = False
                LOGGER.info(
                    "Shared camera opened",
                    extra={"structured_data": {"event": "camera_opened", "width": self._width, "height": self._height, "reported_fps": reported_fps, "open_count": self._camera_open_count}},
                )
                consecutive_failures = 0
                while not self._stop_event.is_set():
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        if self._stop_event.is_set():
                            break
                        consecutive_failures += 1
                        with self._condition:
                            self._read_failures += 1
                            self._error = f"Camera read failed ({consecutive_failures} consecutive)"
                            self._condition.notify_all()
                        if consecutive_failures >= self.shared_settings.maximum_consecutive_read_failures:
                            raise RuntimeError("Camera stopped returning frames")
                        if self._stop_event.wait(0.05):
                            break
                        continue
                    consecutive_failures = 0
                    height, width = frame.shape[:2]
                    fps = meter.update()
                    jpeg_bytes: bytes | None = None
                    if self._encode_jpeg:
                        encoded, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
                        if encoded:
                            jpeg_bytes = jpeg.tobytes()
                    received_at = datetime.now().astimezone().isoformat(timespec="milliseconds")
                    received_monotonic = monotonic()
                    with self._condition:
                        self._online = True
                        self._width = width
                        self._height = height
                        self._fps = fps
                        self._error = None
                        self._latest_frame = frame.copy()
                        self._latest_raw_jpeg = jpeg_bytes
                        self._sequence += 1
                        self._last_frame_at = received_at
                        self._last_frame_monotonic = received_monotonic
                        self._frames_received += 1
                        self._condition.notify_all()
            except Exception as exc:
                if not self._stop_event.is_set():
                    self._set_offline(str(exc))
                    LOGGER.error(
                        "Shared camera is offline; reconnect will be attempted: %s",
                        exc,
                        extra={"structured_data": {"event": "camera_failure", "error": str(exc)}},
                        exc_info=True,
                    )
            finally:
                if capture is not None:
                    try:
                        capture.release()
                    except Exception:
                        LOGGER.exception("Camera release failed")
                with self._condition:
                    if self._capture is capture:
                        self._capture = None
                    self._online = False
                    self._condition.notify_all()
            if not self.shared_settings.reconnect_enabled or self._stop_event.wait(self.shared_settings.reconnect_delay_seconds):
                break
