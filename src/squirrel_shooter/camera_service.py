"""Single-owner camera service shared by the web stream and future detectors."""

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
from .config import CameraConfig


LOGGER = logging.getLogger(__name__)


def is_raspberry_pi() -> bool:
    """Return whether the host identifies itself as Raspberry Pi hardware."""

    model_paths = (
        Path("/proc/device-tree/model"),
        Path("/sys/firmware/devicetree/base/model"),
    )
    for model_path in model_paths:
        try:
            if "raspberry pi" in model_path.read_text(encoding="utf-8").lower():
                return True
        except OSError:
            continue
    return False


@dataclass(frozen=True)
class CameraStatus:
    """A thread-safe snapshot of the camera's current state."""

    online: bool
    width: int
    height: int
    fps: float
    error: str | None
    last_frame_at: str | None = None
    last_frame_age_seconds: float | None = None
    frames_received: int = 0
    thread_alive: bool = False


@dataclass(frozen=True)
class FramePacket:
    """One shared frame and its monotonically increasing sequence number."""

    sequence: int
    frame: np.ndarray
    received_at: str


class CameraService:
    """Own one physical camera and publish its latest frame to many consumers."""

    def __init__(
        self,
        settings: CameraConfig,
        *,
        capture_factory: Callable[[CameraConfig], Any] = open_camera,
        platform_checker: Callable[[], bool] = is_raspberry_pi,
        jpeg_quality: int = 80,
        encode_jpeg: bool = True,
    ) -> None:
        self.settings = settings
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
        self._error: str | None = "Camera has not started"
        self._latest_frame: np.ndarray | None = None
        self._latest_jpeg: bytes | None = None
        self._sequence = 0
        self._last_frame_at: str | None = None
        self._last_frame_monotonic: float | None = None
        self._frames_received = 0

    def start(self) -> None:
        """Start capture once; repeated calls do not open another camera."""

        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._online = False
            self._error = None
            self._latest_frame = None
            self._latest_jpeg = None
            self._last_frame_at = None
            self._last_frame_monotonic = None
            self._thread = threading.Thread(
                target=self._capture_loop,
                name="squirrel-camera",
                daemon=True,
            )
            self._thread.start()
        LOGGER.info("Camera worker started", extra={"structured_data": {"event": "camera_worker_started"}})

    def stop(self, timeout: float = 3.0) -> None:
        """Stop capture and release the physical camera."""

        self._stop_event.set()
        with self._condition:
            capture = self._capture
            self._condition.notify_all()
        if capture is not None:
            capture.release()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        LOGGER.info("Camera worker stopped", extra={"structured_data": {"event": "camera_worker_stopped"}})

    def status(self) -> CameraStatus:
        """Return current health and negotiated stream details."""

        with self._condition:
            age = None if self._last_frame_monotonic is None else max(0.0, monotonic() - self._last_frame_monotonic)
            return CameraStatus(
                online=self._online,
                width=self._width,
                height=self._height,
                fps=self._fps,
                error=self._error,
                last_frame_at=self._last_frame_at,
                last_frame_age_seconds=age,
                frames_received=self._frames_received,
                thread_alive=self._capture_thread_alive(),
            )

    def latest_frame(self, *, copy: bool = True) -> np.ndarray | None:
        """Return the latest raw frame for the future detection pipeline."""

        with self._condition:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy() if copy else self._latest_frame

    def wait_for_frame(self, after_sequence: int, timeout: float = 1.0) -> FramePacket | None:
        """Wait for a new shared frame without opening or reading the camera again."""

        with self._condition:
            self._condition.wait_for(
                lambda: self._sequence > after_sequence
                or self._stop_event.is_set()
                or (self._thread is not None and not self._thread.is_alive()),
                timeout=timeout,
            )
            if self._sequence <= after_sequence or self._latest_frame is None or self._last_frame_at is None:
                return None
            return FramePacket(self._sequence, self._latest_frame.copy(), self._last_frame_at)

    def mjpeg_frames(self) -> Iterator[bytes]:
        """Yield the shared JPEG frame whenever the camera publishes a new one."""

        sequence = -1
        while not self._stop_event.is_set():
            with self._condition:
                self._condition.wait_for(
                    lambda: self._sequence != sequence
                    or self._stop_event.is_set()
                    or not self._capture_thread_alive(),
                    timeout=1.0,
                )
                if self._stop_event.is_set():
                    return
                jpeg = self._latest_jpeg
                if jpeg is None or self._sequence == sequence:
                    sequence = self._sequence
                    if not self._capture_thread_alive():
                        return
                    continue
                sequence = self._sequence

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-cache\r\n\r\n"
                + jpeg
                + b"\r\n"
            )

    def _capture_thread_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _capture_loop(self) -> None:
        capture: Any | None = None
        meter = FrameRateMeter()
        try:
            if not self._platform_checker():
                raise RuntimeError(
                    "Camera capture is disabled because this host is not a Raspberry Pi"
                )
            capture = self._capture_factory(self.settings)
            with self._condition:
                self._capture = capture
                width, height, _ = capture_dimensions(capture)
                if width > 0 and height > 0:
                    self._width, self._height = width, height
            LOGGER.info(
                "Camera opened",
                extra={"structured_data": {"event": "camera_opened", "width": self._width, "height": self._height, "reported_fps": capture_dimensions(capture)[2]}},
            )

            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError("Camera opened but stopped returning frames")

                height, width = frame.shape[:2]
                fps = meter.update()
                jpeg_bytes: bytes | None = None
                if self._encode_jpeg:
                    encoded, jpeg = cv2.imencode(
                        ".jpg",
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
                    )
                    if not encoded:
                        LOGGER.warning("OpenCV could not encode a camera frame as JPEG")
                        continue
                    jpeg_bytes = jpeg.tobytes()

                received_at = datetime.now().astimezone().isoformat(timespec="milliseconds")
                received_monotonic = monotonic()

                with self._condition:
                    self._online = True
                    self._width = width
                    self._height = height
                    self._fps = fps
                    self._error = None
                    self._latest_frame = frame
                    self._latest_jpeg = jpeg_bytes
                    self._sequence += 1
                    self._last_frame_at = received_at
                    self._last_frame_monotonic = received_monotonic
                    self._frames_received += 1
                    self._condition.notify_all()
                if self._frames_received == 30:
                    LOGGER.info(
                        "Camera capture rate measured",
                        extra={"structured_data": {"event": "actual_camera_fps", "fps": round(fps, 2)}},
                    )
        except Exception as exc:  # Keep the web server alive when camera I/O fails.
            if not self._stop_event.is_set():
                LOGGER.error(
                    "Camera is offline: %s",
                    exc,
                    extra={"structured_data": {"event": "camera_failure", "error": str(exc)}},
                    exc_info=True,
                )
                with self._condition:
                    self._error = str(exc)
        finally:
            if capture is not None:
                capture.release()
            with self._condition:
                self._capture = None
                self._online = False
                self._fps = 0.0
                self._condition.notify_all()
