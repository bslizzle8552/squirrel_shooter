"""Single-owner camera service shared by the web stream and future detectors."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
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


class CameraService:
    """Own one physical camera and publish its latest frame to many consumers."""

    def __init__(
        self,
        settings: CameraConfig,
        *,
        capture_factory: Callable[[CameraConfig], Any] = open_camera,
        platform_checker: Callable[[], bool] = is_raspberry_pi,
        jpeg_quality: int = 80,
    ) -> None:
        self.settings = settings
        self._capture_factory = capture_factory
        self._platform_checker = platform_checker
        self._jpeg_quality = jpeg_quality
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

    def start(self) -> None:
        """Start capture once; repeated calls do not open another camera."""

        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._error = None
            self._thread = threading.Thread(
                target=self._capture_loop,
                name="squirrel-camera",
                daemon=True,
            )
            self._thread.start()

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

    def status(self) -> CameraStatus:
        """Return current health and negotiated stream details."""

        with self._condition:
            return CameraStatus(
                online=self._online,
                width=self._width,
                height=self._height,
                fps=self._fps,
                error=self._error,
            )

    def latest_frame(self, *, copy: bool = True) -> np.ndarray | None:
        """Return the latest raw frame for the future detection pipeline."""

        with self._condition:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy() if copy else self._latest_frame

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

            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError("Camera opened but stopped returning frames")

                height, width = frame.shape[:2]
                fps = meter.update()
                encoded, jpeg = cv2.imencode(
                    ".jpg",
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
                )
                if not encoded:
                    LOGGER.warning("OpenCV could not encode a camera frame as JPEG")
                    continue

                with self._condition:
                    self._online = True
                    self._width = width
                    self._height = height
                    self._fps = fps
                    self._error = None
                    self._latest_frame = frame
                    self._latest_jpeg = jpeg.tobytes()
                    self._sequence += 1
                    self._condition.notify_all()
        except Exception as exc:  # Keep the web server alive when camera I/O fails.
            if not self._stop_event.is_set():
                LOGGER.warning("Camera is offline: %s", exc)
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
