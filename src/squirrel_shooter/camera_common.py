"""Shared OpenCV helpers for preview, recording, and diagnostics."""

from __future__ import annotations

from datetime import datetime
from time import perf_counter

import cv2
import numpy as np

from .config import CameraConfig


class CameraOpenError(RuntimeError):
    """Raised when OpenCV cannot open or read the selected camera."""


class FrameRateMeter:
    """Small smoothed FPS meter suitable for overlay text."""

    def __init__(self, smoothing: float = 0.2) -> None:
        self._smoothing = smoothing
        self._last_time: float | None = None
        self._fps = 0.0

    @property
    def fps(self) -> float:
        return self._fps

    def update(self, now: float | None = None) -> float:
        current = perf_counter() if now is None else now
        if self._last_time is not None:
            elapsed = current - self._last_time
            if elapsed > 0:
                instant_fps = 1.0 / elapsed
                self._fps = (
                    instant_fps
                    if self._fps == 0.0
                    else (self._smoothing * instant_fps)
                    + ((1.0 - self._smoothing) * self._fps)
                )
        self._last_time = current
        return self._fps


def open_camera(settings: CameraConfig) -> cv2.VideoCapture:
    """Open the configured camera and request the configured video mode."""

    capture = cv2.VideoCapture(settings.device_index)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, settings.requested_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.requested_height)
    capture.set(cv2.CAP_PROP_FPS, settings.requested_fps)

    if not capture.isOpened():
        capture.release()
        raise CameraOpenError(
            f"OpenCV could not open camera device index {settings.device_index}. "
            "Run 'python -m squirrel_shooter.camera_diagnostic' and check "
            "the /dev/video* devices before changing config/default.yaml."
        )
    return capture


def capture_dimensions(capture: cv2.VideoCapture) -> tuple[int, int, float]:
    """Return the width, height, and FPS actually reported by the camera."""

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    return width, height, fps


def annotate_frame(frame: np.ndarray, fps: float) -> np.ndarray:
    """Return a copy of a frame with timestamp, resolution, and FPS overlays."""

    annotated = frame.copy()
    height, width = annotated.shape[:2]
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = (timestamp, f"{width}x{height} | {fps:5.1f} FPS")

    cv2.rectangle(annotated, (8, 8), (390, 68), (0, 0, 0), thickness=-1)
    for line_number, line in enumerate(lines):
        cv2.putText(
            annotated,
            line,
            (16, 31 + (line_number * 27)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated
