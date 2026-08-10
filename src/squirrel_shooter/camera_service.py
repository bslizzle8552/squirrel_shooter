"""Single-owner camera runtime shared by motion processing and the dashboard."""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import monotonic, perf_counter
from typing import Any

import cv2
import numpy as np

from .camera_common import FrameRateMeter, capture_dimensions, capture_fourcc, open_camera
from .config import CameraConfig, SharedCameraConfig
from .performance import AverageTimer, ThreadCpuMeter
from .thread_names import set_current_thread_name


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
    dashboard_viewers: int = 0
    dashboard_stream_fps: float = 0.0
    dashboard_frames_encoded: int = 0
    pre_roll_frames_buffered: int = 0
    pre_roll_buffer_fps: float = 0.0
    pre_roll_target_fps: float = 0.0
    pre_roll_frames_encoded: int = 0
    pre_roll_frames_copied: int = 0
    capture_read_average_ms: float = 0.0
    frame_publish_average_ms: float = 0.0
    pre_roll_copy_average_ms: float = 0.0
    capture_thread_cpu_percent: float = 0.0


@dataclass(frozen=True)
class FramePacket:
    """One raw frame published by the sole camera-reading thread."""

    sequence: int
    frame: np.ndarray
    received_at: str
    received_monotonic: float = 0.0


@dataclass(frozen=True)
class BufferedFrameInfo:
    """Timing metadata for one compact frame in the rolling event buffer."""

    sequence: int
    received_at: str
    received_monotonic: float


@dataclass(frozen=True)
class _BufferedRawFrame:
    sequence: int
    frame: np.ndarray
    received_at: str
    received_monotonic: float


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
        frame_buffer_seconds: float = 0.0,
        frame_buffer_fps: float = 12.0,
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
        if frame_buffer_seconds < 0:
            raise ValueError("frame_buffer_seconds must be zero or greater")
        if frame_buffer_fps <= 0:
            raise ValueError("frame_buffer_fps must be greater than zero")
        self._frame_buffer_seconds = float(frame_buffer_seconds)
        self._frame_buffer_fps = float(frame_buffer_fps)
        self._frame_buffer_interval = 1.0 / self._frame_buffer_fps
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._mjpeg_thread: threading.Thread | None = None
        self._capture: Any | None = None
        self._online = False
        self._width = settings.requested_width
        self._height = settings.requested_height
        self._fps = 0.0
        self._reported_fps = 0.0
        self._error: str | None = "Camera has not started"
        self._latest_frame: np.ndarray | None = None
        self._latest_annotated_frame: np.ndarray | None = None
        self._mjpeg: bytes | None = None
        self._mjpeg_sequence = 0
        self._mjpeg_source: tuple[str, int] | None = None
        self._mjpeg_maximum_fps = 0.0
        self._mjpeg_annotated_only = True
        self._dashboard_viewers = 0
        self._dashboard_stream_fps = 0.0
        self._dashboard_frames_encoded = 0
        self._dashboard_meter = FrameRateMeter()
        self._frame_buffer: deque[_BufferedRawFrame] = deque()
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
        self._next_buffer_monotonic: float | None = None
        self._pre_roll_buffer_fps = 0.0
        self._pre_roll_frames_encoded = 0
        self._pre_roll_frames_copied = 0
        self._pre_roll_meter = FrameRateMeter()
        self._capture_read_timer = AverageTimer()
        self._frame_publish_timer = AverageTimer()
        self._pre_roll_copy_timer = AverageTimer()
        self._capture_cpu = ThreadCpuMeter()

    @property
    def stopped(self) -> bool:
        return self._stop_event.is_set()

    @property
    def frame_buffer_seconds(self) -> float:
        return self._frame_buffer_seconds

    @property
    def has_dashboard_viewers(self) -> bool:
        with self._condition:
            return self._dashboard_viewers > 0

    def start(self) -> None:
        """Start one camera owner; repeated starts never open another handle."""

        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._online = False
            self._error = None
            self._latest_frame = None
            self._latest_annotated_frame = None
            self._mjpeg = None
            self._mjpeg_sequence = 0
            self._mjpeg_source = None
            self._mjpeg_annotated_only = True
            self._dashboard_viewers = 0
            self._dashboard_stream_fps = 0.0
            self._dashboard_frames_encoded = 0
            self._dashboard_meter = FrameRateMeter()
            self._frame_buffer.clear()
            self._next_buffer_monotonic = None
            self._pre_roll_buffer_fps = 0.0
            self._pre_roll_frames_encoded = 0
            self._pre_roll_frames_copied = 0
            self._pre_roll_meter = FrameRateMeter()
            self._capture_read_timer = AverageTimer()
            self._frame_publish_timer = AverageTimer()
            self._pre_roll_copy_timer = AverageTimer()
            self._capture_cpu = ThreadCpuMeter()
            self._last_frame_at = None
            self._last_frame_monotonic = None
            self._last_annotated_at = None
            self._last_annotated_monotonic = None
            self._thread = threading.Thread(target=self._capture_loop, name="camera-capture", daemon=True)
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
        mjpeg_thread = self._mjpeg_thread
        if mjpeg_thread is not None and mjpeg_thread is not threading.current_thread():
            mjpeg_thread.join(timeout=timeout)
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
                self._dashboard_viewers,
                self._dashboard_stream_fps,
                self._dashboard_frames_encoded,
                len(self._frame_buffer),
                self._pre_roll_buffer_fps,
                self._frame_buffer_fps if self._frame_buffer_seconds > 0 else 0.0,
                self._pre_roll_frames_encoded,
                self._pre_roll_frames_copied,
                self._capture_read_timer.average_ms,
                self._frame_publish_timer.average_ms,
                self._pre_roll_copy_timer.average_ms,
                self._capture_cpu.percent,
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

    def buffered_frame_metadata(
        self,
        since_monotonic: float,
        *,
        until_monotonic: float | None = None,
        after_sequence: int = -1,
    ) -> list[BufferedFrameInfo]:
        """Return lightweight timing data without copying buffered images."""

        with self._condition:
            return [
                BufferedFrameInfo(item.sequence, item.received_at, item.received_monotonic)
                for item in self._frame_buffer
                if item.received_monotonic >= since_monotonic
                and (until_monotonic is None or item.received_monotonic <= until_monotonic)
                and item.sequence > after_sequence
            ]

    def buffered_frames(
        self,
        since_monotonic: float,
        *,
        until_monotonic: float | None = None,
        after_sequence: int = -1,
    ) -> Iterator[FramePacket]:
        """Yield copies from the rolling raw pre-event buffer."""

        with self._condition:
            buffered = [
                item
                for item in self._frame_buffer
                if item.received_monotonic >= since_monotonic
                and (until_monotonic is None or item.received_monotonic <= until_monotonic)
                and item.sequence > after_sequence
            ]
        for item in buffered:
            yield FramePacket(item.sequence, item.frame.copy(), item.received_at, item.received_monotonic)

    def publish_annotated(self, source_sequence: int, frame: np.ndarray) -> bool:
        """Publish motion annotations without encoding work when nobody is viewing."""

        now_iso = datetime.now().astimezone().isoformat(timespec="milliseconds")
        now = monotonic()
        with self._condition:
            if source_sequence < self._last_annotated_source_sequence:
                return False
            self._latest_annotated_frame = frame.copy()
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
        """Yield frames from one shared encoder that sleeps when there are no viewers."""

        requested_fps = float(maximum_fps or self.settings.requested_fps)
        if requested_fps <= 0:
            raise ValueError("maximum_fps must be greater than zero")
        sequence = -1
        with self._condition:
            if self._dashboard_viewers == 0:
                self._dashboard_meter = FrameRateMeter()
            self._dashboard_viewers += 1
            self._mjpeg_annotated_only = self._mjpeg_annotated_only and annotated_only
            if self._mjpeg_maximum_fps <= 0:
                self._mjpeg_maximum_fps = requested_fps
            else:
                self._mjpeg_maximum_fps = min(self._mjpeg_maximum_fps, requested_fps)
            if self._mjpeg_thread is None or not self._mjpeg_thread.is_alive():
                self._mjpeg_thread = threading.Thread(
                    target=self._mjpeg_encode_loop,
                    name="mjpeg-encoder",
                    daemon=True,
                )
                self._mjpeg_thread.start()
            self._condition.notify_all()
        try:
            while not self._stop_event.is_set():
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._mjpeg_sequence > sequence or self._stop_event.is_set(),
                        timeout=self.shared_settings.consumer_wait_timeout_seconds,
                    )
                    if self._stop_event.is_set():
                        return
                    if self._mjpeg_sequence <= sequence or self._mjpeg is None:
                        continue
                    sequence = self._mjpeg_sequence
                    jpeg = self._mjpeg
                yield b"--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-cache\r\n\r\n" + jpeg + b"\r\n"
        finally:
            with self._condition:
                self._dashboard_viewers = max(0, self._dashboard_viewers - 1)
                if self._dashboard_viewers == 0:
                    self._mjpeg_maximum_fps = 0.0
                    self._mjpeg_annotated_only = True
                    self._dashboard_stream_fps = 0.0
                self._condition.notify_all()

    def _mjpeg_encode_loop(self) -> None:
        set_current_thread_name("mjpeg-encoder")
        last_encoded_at = 0.0
        while not self._stop_event.is_set():
            with self._condition:
                def frame_ready() -> bool:
                    if self._stop_event.is_set():
                        return True
                    if self._dashboard_viewers <= 0:
                        return False
                    source = self._current_mjpeg_source()
                    return source is not None and source[0] != self._mjpeg_source

                self._condition.wait_for(frame_ready)
                if self._stop_event.is_set():
                    return
                maximum_fps = self._mjpeg_maximum_fps
            interval = 1.0 / maximum_fps if maximum_fps > 0 else 0.0
            remaining = interval - (monotonic() - last_encoded_at)
            if remaining > 0 and self._stop_event.wait(remaining):
                return
            with self._condition:
                source = self._current_mjpeg_source()
                if self._dashboard_viewers <= 0 or source is None or source[0] == self._mjpeg_source:
                    continue
                source_key, frame = source
            encoded, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
            if not encoded:
                LOGGER.warning("OpenCV could not encode a dashboard frame")
                continue
            last_encoded_at = monotonic()
            with self._condition:
                self._mjpeg = jpeg.tobytes()
                self._mjpeg_source = source_key
                self._mjpeg_sequence += 1
                self._dashboard_frames_encoded += 1
                self._dashboard_stream_fps = self._dashboard_meter.update(last_encoded_at)
                self._condition.notify_all()

    def _current_mjpeg_source(self) -> tuple[tuple[str, int], np.ndarray] | None:
        if self._latest_annotated_frame is not None:
            return ("annotated", self._annotated_sequence), self._latest_annotated_frame
        if not self._mjpeg_annotated_only and self._encode_jpeg and self._latest_frame is not None:
            return ("raw", self._sequence), self._latest_frame
        return None

    def _capture_thread_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _set_offline(self, error: str) -> None:
        with self._condition:
            self._online = False
            self._fps = 0.0
            self._error = error
            self._condition.notify_all()

    def _capture_loop(self) -> None:
        set_current_thread_name("camera-capture")
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
                fourcc = capture_fourcc(capture)
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
                    "Shared camera opened: width=%d height=%d reported_fps=%.2f fourcc=%s",
                    self._width,
                    self._height,
                    reported_fps,
                    fourcc,
                    extra={"structured_data": {"event": "camera_opened", "width": self._width, "height": self._height, "reported_fps": reported_fps, "fourcc": fourcc, "open_count": self._camera_open_count}},
                )
                consecutive_failures = 0
                while not self._stop_event.is_set():
                    read_started = perf_counter()
                    ok, frame = capture.read()
                    self._capture_read_timer.add(perf_counter() - read_started)
                    self._capture_cpu.update()
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
                    received_at = datetime.now().astimezone().isoformat(timespec="milliseconds")
                    received_monotonic = monotonic()
                    buffered_frame: np.ndarray | None = None
                    should_buffer = (
                        self._frame_buffer_seconds > 0
                        and (
                            self._next_buffer_monotonic is None
                            or received_monotonic >= self._next_buffer_monotonic
                        )
                    )
                    if should_buffer:
                        if self._next_buffer_monotonic is None:
                            self._next_buffer_monotonic = received_monotonic + self._frame_buffer_interval
                        else:
                            while self._next_buffer_monotonic <= received_monotonic:
                                self._next_buffer_monotonic += self._frame_buffer_interval
                        copy_started = perf_counter()
                        buffered_frame = frame.copy()
                        self._pre_roll_copy_timer.add(perf_counter() - copy_started)
                    publish_started = perf_counter()
                    with self._condition:
                        self._online = True
                        self._width = width
                        self._height = height
                        self._fps = fps
                        self._error = None
                        self._latest_frame = frame.copy()
                        self._sequence += 1
                        self._last_frame_at = received_at
                        self._last_frame_monotonic = received_monotonic
                        self._frames_received += 1
                        if self._frame_buffer_seconds > 0 and buffered_frame is not None:
                            self._pre_roll_frames_copied += 1
                            self._pre_roll_buffer_fps = self._pre_roll_meter.update(received_monotonic)
                            self._frame_buffer.append(
                                _BufferedRawFrame(
                                    self._sequence,
                                    buffered_frame,
                                    received_at,
                                    received_monotonic,
                                )
                            )
                            cutoff = received_monotonic - self._frame_buffer_seconds
                            while self._frame_buffer and self._frame_buffer[0].received_monotonic < cutoff:
                                self._frame_buffer.popleft()
                        self._condition.notify_all()
                    self._frame_publish_timer.add(perf_counter() - publish_started)
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
