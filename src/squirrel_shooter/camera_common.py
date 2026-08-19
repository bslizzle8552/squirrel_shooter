"""Shared OpenCV helpers for preview, recording, and diagnostics."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from .config import CameraConfig


LOGGER = logging.getLogger(__name__)

_V4L2_CTL_TIMEOUT_SECONDS = 5.0
_SOURCE_FPS_TOLERANCE = 0.1
_V4L2_FORMAT_PATTERN = re.compile(r"^\s*\[\d+\]:\s*'(?P<fourcc>[^']+)'", re.MULTILINE)
_V4L2_SIZE_PATTERN = re.compile(
    r"^\s*Size:\s+Discrete\s+(?P<width>\d+)x(?P<height>\d+)",
    re.MULTILINE,
)
_V4L2_FPS_PATTERN = re.compile(
    r"^\s*Interval:\s+Discrete\s+.*?\((?P<fps>\d+(?:\.\d+)?)\s+fps\)",
    re.MULTILINE,
)


class CameraOpenError(RuntimeError):
    """Raised when OpenCV cannot open or read the selected camera."""


@dataclass(frozen=True)
class SupportedCameraMode:
    """One discrete V4L2 pixel format and resolution with advertised cadences."""

    fourcc: str
    width: int
    height: int
    fps: tuple[float, ...]


@dataclass(frozen=True)
class CameraModeProbe:
    """Read-only result from ``v4l2-ctl --list-formats-ext``."""

    device: str
    utility_available: bool
    modes: tuple[SupportedCameraMode, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class SourceModeRequest:
    """How the pre-open V4L2 source request was handled."""

    method: str
    utility_available: bool
    applied: bool
    error: str | None = None


def _is_linux_v4l2_host() -> bool:
    return sys.platform.startswith("linux")


def _v4l2_device(device_index: int) -> str:
    if isinstance(device_index, bool) or not isinstance(device_index, int) or device_index < 0:
        raise ValueError("camera device index must be a non-negative integer")
    return f"/dev/video{device_index}"


def _unique_fps(values: list[float]) -> tuple[float, ...]:
    unique: list[float] = []
    for value in values:
        if not any(abs(value - existing) <= 0.0005 for existing in unique):
            unique.append(value)
    return tuple(unique)


def parse_v4l2_supported_modes(output: str) -> tuple[SupportedCameraMode, ...]:
    """Parse discrete modes from read-only ``v4l2-ctl --list-formats-ext`` output."""

    if not isinstance(output, str):
        raise TypeError("v4l2-ctl output must be text")

    modes: list[SupportedCameraMode] = []
    current_fourcc: str | None = None
    current_size: tuple[int, int] | None = None
    current_fps: list[float] = []

    def finish_size() -> None:
        nonlocal current_size, current_fps
        if current_fourcc is not None and current_size is not None:
            modes.append(
                SupportedCameraMode(
                    current_fourcc,
                    current_size[0],
                    current_size[1],
                    _unique_fps(current_fps),
                )
            )
        current_size = None
        current_fps = []

    for line in output.splitlines():
        format_match = _V4L2_FORMAT_PATTERN.match(line)
        if format_match is not None:
            finish_size()
            current_fourcc = format_match.group("fourcc").strip().upper()
            continue
        size_match = _V4L2_SIZE_PATTERN.match(line)
        if size_match is not None:
            finish_size()
            current_size = (
                int(size_match.group("width")),
                int(size_match.group("height")),
            )
            continue
        fps_match = _V4L2_FPS_PATTERN.match(line)
        if fps_match is not None and current_size is not None:
            current_fps.append(float(fps_match.group("fps")))
    finish_size()
    return tuple(modes)


def probe_supported_camera_modes(device_index: int) -> CameraModeProbe:
    """Read V4L2's advertised discrete modes without changing device state."""

    device = _v4l2_device(device_index)
    if not _is_linux_v4l2_host():
        return CameraModeProbe(
            device,
            False,
            error="v4l2-ctl probing is only available on Linux",
        )
    utility = shutil.which("v4l2-ctl")
    if utility is None:
        return CameraModeProbe(
            device,
            False,
            error="v4l2-ctl is not installed; supported modes were not probed",
        )
    command = [utility, "--device", device, "--list-formats-ext"]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=_V4L2_CTL_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CameraModeProbe(
            device,
            True,
            error=f"v4l2-ctl mode probe failed: {type(exc).__name__}: {exc}",
        )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic output"
        return CameraModeProbe(
            device,
            True,
            error=f"v4l2-ctl mode probe exited {completed.returncode}: {detail}",
        )
    try:
        modes = parse_v4l2_supported_modes(completed.stdout)
    except (TypeError, ValueError) as exc:
        return CameraModeProbe(
            device,
            True,
            error=f"v4l2-ctl mode output could not be parsed: {exc}",
        )
    return CameraModeProbe(device, True, modes=modes)


def supports_requested_camera_mode(
    modes: tuple[SupportedCameraMode, ...] | list[SupportedCameraMode],
    settings: CameraConfig,
    *,
    fourcc: str = "MJPG",
) -> bool:
    """Return whether an advertised discrete mode contains the exact request."""

    requested_fourcc = fourcc.upper()
    for mode in modes:
        if (
            mode.fourcc == requested_fourcc
            and mode.width == settings.requested_width
            and mode.height == settings.requested_height
            and any(abs(fps - settings.requested_fps) <= _SOURCE_FPS_TOLERANCE for fps in mode.fps)
        ):
            return True
    return False


def _request_v4l2_source_mode(settings: CameraConfig) -> SourceModeRequest:
    """Ask V4L2 for the exact source mode before OpenCV starts streaming."""

    if not _is_linux_v4l2_host():
        return SourceModeRequest("opencv", False, False)
    utility = shutil.which("v4l2-ctl")
    if utility is None:
        LOGGER.info(
            "v4l2-ctl is unavailable; using the explicit OpenCV V4L2 property fallback",
            extra={
                "structured_data": {
                    "event": "camera_v4l2_ctl_unavailable",
                    "device_index": settings.device_index,
                }
            },
        )
        return SourceModeRequest("opencv_v4l2", False, False)
    device = _v4l2_device(settings.device_index)
    format_request = (
        f"width={settings.requested_width},height={settings.requested_height},"
        "pixelformat=MJPG"
    )
    command = [
        utility,
        "--device",
        device,
        "--set-fmt-video",
        format_request,
        "--set-parm",
        f"{settings.requested_fps:g}",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=_V4L2_CTL_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        detail = f"{type(exc).__name__}: {exc}"
        LOGGER.warning(
            "v4l2-ctl source-mode request failed; using OpenCV V4L2 properties: %s",
            detail,
            extra={
                "structured_data": {
                    "event": "camera_v4l2_source_request_failed",
                    "device": device,
                    "error": detail,
                }
            },
        )
        return SourceModeRequest("opencv_v4l2", True, False, detail)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic output"
        LOGGER.warning(
            "v4l2-ctl source-mode request exited %d; using OpenCV V4L2 properties: %s",
            completed.returncode,
            detail,
            extra={
                "structured_data": {
                    "event": "camera_v4l2_source_request_rejected",
                    "device": device,
                    "returncode": completed.returncode,
                    "error": detail,
                }
            },
        )
        return SourceModeRequest("opencv_v4l2", True, False, detail)
    LOGGER.info(
        "Requested V4L2 source mode before streaming: device=%s mode=%dx%d MJPG at %.2f FPS",
        device,
        settings.requested_width,
        settings.requested_height,
        settings.requested_fps,
        extra={
            "structured_data": {
                "event": "camera_v4l2_source_requested",
                "device": device,
                "width": settings.requested_width,
                "height": settings.requested_height,
                "fourcc": "MJPG",
                "fps": settings.requested_fps,
            }
        },
    )
    return SourceModeRequest("v4l2_ctl", True, True)


def _open_video_capture(settings: CameraConfig) -> Any:
    if _is_linux_v4l2_host():
        backend = getattr(cv2, "CAP_V4L2", getattr(cv2, "CAP_V4L", None))
        if backend is not None:
            try:
                capture = cv2.VideoCapture(settings.device_index, backend)
            except (TypeError, cv2.error) as exc:
                LOGGER.warning(
                    "Explicit OpenCV V4L2 backend open failed; trying the default backend: %s",
                    exc,
                )
            else:
                if capture.isOpened():
                    return capture
                capture.release()
                LOGGER.warning(
                    "Explicit OpenCV V4L2 backend did not open the camera; trying the default backend"
                )
    return cv2.VideoCapture(settings.device_index)


def capture_backend_name(capture: Any) -> str:
    """Return OpenCV's actual backend name when the build exposes it."""

    getter = getattr(capture, "getBackendName", None)
    if not callable(getter):
        return "unknown"
    try:
        value = getter()
    except Exception:
        return "unknown"
    return value if isinstance(value, str) and value.strip() else "unknown"


def source_mode_mismatches(
    settings: CameraConfig,
    width: int,
    height: int,
    reported_fps: float,
    fourcc: str,
) -> tuple[str, ...]:
    """Return requested source fields that were not verified from the backend."""

    mismatches: list[str] = []
    if fourcc.upper() != "MJPG":
        mismatches.append("fourcc")
    if width != settings.requested_width or height != settings.requested_height:
        mismatches.append("resolution")
    if reported_fps <= 0 or abs(reported_fps - settings.requested_fps) > _SOURCE_FPS_TOLERANCE:
        mismatches.append("fps")
    return tuple(mismatches)


def _set_capture_property(capture: Any, property_id: int, value: float, name: str) -> bool:
    try:
        return bool(capture.set(property_id, value))
    except Exception as exc:
        LOGGER.warning("Camera backend rejected the %s request: %s", name, exc)
        return False


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
    """Open the configured camera and request MJPEG plus the configured mode."""

    source_request = _request_v4l2_source_mode(settings)
    capture = _open_video_capture(settings)
    if not capture.isOpened():
        capture.release()
        raise CameraOpenError(
            f"OpenCV could not open camera device index {settings.device_index}. "
            "Run 'python -m squirrel_shooter.camera_diagnostic' and check "
            "the /dev/video* devices before changing config/default.yaml."
        )

    mjpeg_requested = _set_capture_property(
        capture,
        cv2.CAP_PROP_FOURCC,
        float(cv2.VideoWriter_fourcc(*"MJPG")),
        "MJPEG format",
    )
    if not mjpeg_requested:
        LOGGER.warning("Camera did not accept the MJPEG request; continuing with its available format")
    width_requested = _set_capture_property(
        capture,
        cv2.CAP_PROP_FRAME_WIDTH,
        float(settings.requested_width),
        "frame-width",
    )
    height_requested = _set_capture_property(
        capture,
        cv2.CAP_PROP_FRAME_HEIGHT,
        float(settings.requested_height),
        "frame-height",
    )
    fps_requested = _set_capture_property(
        capture,
        cv2.CAP_PROP_FPS,
        float(settings.requested_fps),
        "source-cadence",
    )
    # Keep the backend queue shallow. This does not manufacture a lower source
    # cadence; it prevents a slow consumer from working through stale frames.
    # Some OpenCV backends do not expose this control, so rejection is measured
    # and reported rather than treated as proof that freshness is unavailable.
    buffer_size_requested = _set_capture_property(
        capture,
        cv2.CAP_PROP_BUFFERSIZE,
        1.0,
        "one-frame buffer",
    )

    width, height, reported_fps = capture_dimensions(capture)
    fourcc = capture_fourcc(capture)
    backend = capture_backend_name(capture)
    mismatch_fields = source_mode_mismatches(
        settings,
        width,
        height,
        reported_fps,
        fourcc,
    )
    LOGGER.info(
        "Camera initialized: backend=%s width=%d height=%d reported_fps=%.2f fourcc=%s",
        backend,
        width,
        height,
        reported_fps,
        fourcc,
        extra={
            "structured_data": {
                "event": "camera_initialized",
                "width": width,
                "height": height,
                "reported_fps": reported_fps,
                "fourcc": fourcc,
                "backend": backend,
                "mjpeg_request_accepted": mjpeg_requested,
                "width_request_accepted": width_requested,
                "height_request_accepted": height_requested,
                "fps_request_accepted": fps_requested,
                "buffer_size_request_accepted": buffer_size_requested,
                "requested_fps": settings.requested_fps,
                "reported_fps_delta": reported_fps - settings.requested_fps,
                "source_request_method": source_request.method,
                "source_request_applied": source_request.applied,
                "source_mode_matches_request": not mismatch_fields,
                "source_mode_mismatch_fields": list(mismatch_fields),
            }
        },
    )
    if mismatch_fields:
        LOGGER.warning(
            "Camera source cadence request was not confirmed: requested=%dx%d MJPG at %.2f FPS; "
            "actual=%dx%d %s at %.2f FPS; mismatches=%s",
            settings.requested_width,
            settings.requested_height,
            settings.requested_fps,
            width,
            height,
            fourcc,
            reported_fps,
            ",".join(mismatch_fields),
            extra={
                "structured_data": {
                    "event": "camera_source_mode_mismatch",
                    "requested_width": settings.requested_width,
                    "requested_height": settings.requested_height,
                    "requested_fourcc": "MJPG",
                    "requested_fps": settings.requested_fps,
                    "actual_width": width,
                    "actual_height": height,
                    "actual_fourcc": fourcc,
                    "reported_fps": reported_fps,
                    "fps_request_accepted": fps_requested,
                    "source_request_method": source_request.method,
                    "source_request_applied": source_request.applied,
                    "mismatch_fields": list(mismatch_fields),
                }
            },
        )
    return capture


def capture_dimensions(capture: cv2.VideoCapture) -> tuple[int, int, float]:
    """Return the width, height, and FPS actually reported by the camera."""

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    return width, height, fps


def capture_fourcc(capture: cv2.VideoCapture) -> str:
    """Return the camera's actual FOURCC as readable text when possible."""

    try:
        value = int(capture.get(cv2.CAP_PROP_FOURCC))
    except Exception:
        return "unknown"
    if value <= 0:
        return "unknown"
    text = "".join(chr((value >> (8 * index)) & 0xFF) for index in range(4))
    return text if all(character.isprintable() and not character.isspace() for character in text) else str(value)


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
