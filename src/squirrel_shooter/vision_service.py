"""Motion worker, annotated stream, event snapshots, and health accounting."""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Any

import cv2
import numpy as np

from .camera_service import CameraService
from .config import AppConfig
from .detection import DetectionResult, DetectorState, MotionCandidate, MotionDetector, annotate_detection
from .diagnostics import cleanup_oldest
from .files import timestamped_output_path


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VisionStatus:
    state: str
    enabled: bool
    processing_fps: float
    blob_count: int
    persistence_count: int
    frames_processed: int
    candidates_seen: int
    accepted_events: int
    rejected_events: int
    snapshots_saved: int
    last_detector_update: str | None
    last_detector_age_seconds: float | None
    last_event: str | None
    last_snapshot: str | None
    last_error: str | None
    thread_alive: bool
    capture_directory_writable: bool


class VisionService:
    """Consume shared frames without ever opening a physical camera."""

    def __init__(
        self,
        camera: CameraService,
        config: AppConfig,
        *,
        detector: MotionDetector | None = None,
        image_writer: Callable[[str, np.ndarray], bool] = cv2.imwrite,
        jpeg_quality: int = 82,
    ) -> None:
        self.camera = camera
        self.config = config
        self.detector = detector or MotionDetector(config.motion)
        self._image_writer = image_writer
        self._jpeg_quality = jpeg_quality
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest_jpeg: bytes | None = None
        self._stream_sequence = 0
        self._state = self.detector.state.value
        self._processing_fps = 0.0
        self._blob_count = 0
        self._persistence = 0
        self._frames_processed = 0
        self._candidates_seen = 0
        self._accepted_events = 0
        self._rejected_events = 0
        self._snapshots_saved = 0
        self._last_detector_update: str | None = None
        self._last_detector_monotonic: float | None = None
        self._last_event: str | None = None
        self._last_snapshot: str | None = None
        self._last_error: str | None = None
        self._capture_directory_writable = False
        self._recent_events: deque[MotionCandidate] = deque(maxlen=config.motion.recent_event_limit)
        self._last_debug_write = 0.0
        self._last_logged_state: str | None = None

    def start(self) -> None:
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._capture_directory_writable = self._prepare_capture_directory()
            self._thread = threading.Thread(target=self._run, name="squirrel-vision", daemon=True)
            self._thread.start()
        LOGGER.info(
            "Motion detector worker started",
            extra={"structured_data": {"event": "detector_startup", "enabled": self.config.motion.enabled, "learning_frames": self.config.motion.learning_frames}},
        )
        if self.detector.state is DetectorState.LEARNING:
            LOGGER.info(
                "Motion detector is learning the background",
                extra={"structured_data": {"event": "detector_warmup", "learning_frames": self.config.motion.learning_frames}},
            )

    def stop(self, timeout: float = 3.0) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        LOGGER.info("Motion detector worker stopped", extra={"structured_data": {"event": "detector_shutdown"}})

    def status(self) -> VisionStatus:
        with self._condition:
            age = None if self._last_detector_monotonic is None else max(0.0, monotonic() - self._last_detector_monotonic)
            return VisionStatus(
                state=self._state,
                enabled=self.config.motion.enabled,
                processing_fps=self._processing_fps,
                blob_count=self._blob_count,
                persistence_count=self._persistence,
                frames_processed=self._frames_processed,
                candidates_seen=self._candidates_seen,
                accepted_events=self._accepted_events,
                rejected_events=self._rejected_events,
                snapshots_saved=self._snapshots_saved,
                last_detector_update=self._last_detector_update,
                last_detector_age_seconds=age,
                last_event=self._last_event,
                last_snapshot=self._last_snapshot,
                last_error=self._last_error,
                thread_alive=self._thread is not None and self._thread.is_alive(),
                capture_directory_writable=self._capture_directory_writable,
            )

    def recent_events(self) -> list[dict[str, Any]]:
        with self._condition:
            return [candidate.as_dict() for candidate in reversed(self._recent_events)]

    def mjpeg_frames(self) -> Iterator[bytes]:
        sequence = -1
        while not self._stop_event.is_set():
            with self._condition:
                self._condition.wait_for(
                    lambda: self._stream_sequence != sequence or self._stop_event.is_set(),
                    timeout=1.0,
                )
                if self._stop_event.is_set():
                    return
                jpeg = self._latest_jpeg
                if jpeg is None or self._stream_sequence == sequence:
                    continue
                sequence = self._stream_sequence
            yield b"--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-cache\r\n\r\n" + jpeg + b"\r\n"

    def _run(self) -> None:
        camera_sequence = -1
        try:
            while not self._stop_event.is_set():
                packet = self.camera.wait_for_frame(camera_sequence, timeout=0.5)
                if packet is None:
                    continue
                camera_sequence = packet.sequence
                try:
                    result = self.detector.process(packet.frame)
                    annotated = annotate_detection(packet.frame, result, self.config.motion)
                    self._handle_result(packet.frame, annotated, result)
                except Exception as exc:
                    self._record_detector_exception(exc)
                    error_result = DetectionResult(
                        DetectorState.ERROR,
                        (),
                        np.zeros(packet.frame.shape[:2], dtype=np.uint8),
                        np.zeros(packet.frame.shape[:2], dtype=np.uint8),
                        0.0,
                        0,
                    )
                    self._publish(annotate_detection(packet.frame, error_result, self.config.motion))
        except Exception as exc:
            with self._condition:
                self._state = DetectorState.ERROR.value
                self._last_error = str(exc)
                self._condition.notify_all()
            LOGGER.error(
                "Vision worker thread failed",
                extra={"structured_data": {"event": "thread_failure", "thread": "squirrel-vision", "error": str(exc)}},
                exc_info=True,
            )

    def _handle_result(self, frame: np.ndarray, annotated: np.ndarray, result: DetectionResult) -> None:
        now_iso = datetime.now().astimezone().isoformat(timespec="milliseconds")
        updated_candidates: list[MotionCandidate] = []
        accepted_count = 0
        rejected_count = 0
        snapshots_saved = 0
        last_event: str | None = None
        last_snapshot: str | None = None
        with self._condition:
            self._last_error = None
        if result.state.value != self._last_logged_state:
            LOGGER.info(
                "Detector state changed",
                extra={"structured_data": {"event": "detector_state", "state": result.state.value}},
            )
            self._last_logged_state = result.state.value
        if result.lighting_reset:
            LOGGER.warning(
                "Large frame change reset background learning",
                extra={"structured_data": {"event": "lighting_reset", "threshold_percent": self.config.motion.lighting_change_percent}},
            )
        for candidate in result.candidates:
            updated = candidate
            if candidate.accepted:
                accepted_count += 1
                last_event = candidate.timestamp
                updated = self._save_event_snapshot(frame, result, candidate)
                if updated.snapshot_saved:
                    snapshots_saved += 1
                    last_snapshot = candidate.timestamp
                LOGGER.info("Motion event accepted", extra={"structured_data": {"event": "accepted_event", **updated.as_dict()}})
            else:
                rejected_count += 1
                LOGGER.debug("Motion candidate rejected", extra={"structured_data": {"event": "rejected_candidate", **candidate.as_dict()}})
                if candidate.reason == "cooldown":
                    LOGGER.debug("Motion event suppressed by cooldown", extra={"structured_data": {"event": "cooldown", **candidate.as_dict()}})
            updated_candidates.append(updated)

        with self._condition:
            self._state = result.state.value
            self._processing_fps = result.processing_fps
            self._blob_count = result.blob_count
            self._persistence = result.persistence
            self._frames_processed += 1
            self._candidates_seen += len(result.candidates)
            self._accepted_events += accepted_count
            self._rejected_events += rejected_count
            self._snapshots_saved += snapshots_saved
            if last_event is not None:
                self._last_event = last_event
            if last_snapshot is not None:
                self._last_snapshot = last_snapshot
            self._last_detector_update = now_iso
            self._last_detector_monotonic = monotonic()
            self._recent_events.extend(updated_candidates)

        self._publish(annotated)
        self._write_debug_outputs(frame, annotated, result)

    def _save_event_snapshot(
        self,
        frame: np.ndarray,
        result: DetectionResult,
        candidate: MotionCandidate,
    ) -> MotionCandidate:
        path = timestamped_output_path(self.config.camera.output_directory, "event", "jpg")
        snapshot = annotate_detection(frame, result, self.config.motion, event_timestamp=candidate.timestamp)
        try:
            if not self._capture_directory_writable:
                raise OSError("capture directory is not writable")
            if not self._image_writer(str(path), snapshot):
                raise OSError("OpenCV returned false while writing JPEG")
            cleanup_oldest(self.config.camera.output_directory, "event-*.jpg", self.config.storage.max_event_captures, LOGGER)
            LOGGER.info("Event snapshot saved", extra={"structured_data": {"event": "snapshot_success", "filename": path.name}})
            return replace(candidate, snapshot_saved=True, snapshot_filename=path.name)
        except Exception as exc:
            self._last_error = str(exc)
            LOGGER.error(
                "Event snapshot failed",
                extra={"structured_data": {"event": "snapshot_failure", "filename": path.name, "error": str(exc)}},
                exc_info=True,
            )
            return replace(candidate, snapshot_saved=False, snapshot_filename=None)

    def _publish(self, annotated: np.ndarray) -> None:
        encoded, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
        if not encoded:
            raise RuntimeError("OpenCV could not encode annotated stream frame")
        with self._condition:
            self._latest_jpeg = jpeg.tobytes()
            self._stream_sequence += 1
            self._condition.notify_all()

    def _prepare_capture_directory(self) -> bool:
        directory = self.config.camera.output_directory
        probe = directory / ".squirrel-write-check"
        try:
            directory.mkdir(parents=True, exist_ok=True)
            probe.write_bytes(b"")
            probe.unlink()
            return True
        except OSError as exc:
            LOGGER.error(
                "Capture directory is not writable",
                extra={"structured_data": {"event": "capture_directory_error", "directory": str(directory), "error": str(exc)}},
            )
            return False

    def _record_detector_exception(self, exc: Exception) -> None:
        with self._condition:
            self._state = DetectorState.ERROR.value
            self._last_error = str(exc)
            self._last_detector_update = datetime.now().astimezone().isoformat(timespec="milliseconds")
            self._last_detector_monotonic = monotonic()
        LOGGER.error(
            "Motion detector frame failed",
            extra={"structured_data": {"event": "detector_exception", "error": str(exc)}},
            exc_info=True,
        )

    def _write_debug_outputs(self, frame: np.ndarray, annotated: np.ndarray, result: DetectionResult) -> None:
        debug = self.config.motion.debug
        enabled = {
            "foreground": (debug.foreground_mask, result.raw_mask),
            "cleaned": (debug.cleaned_mask, result.cleaned_mask),
            "annotated": (debug.annotated_frame, annotated),
            "roi": (debug.roi_visualization, annotated),
            "rejected": (debug.rejected_candidate_frame and any(not item.accepted for item in result.candidates), annotated),
        }
        if not any(flag for flag, _ in enabled.values()):
            return
        current = monotonic()
        if current - self._last_debug_write < debug.min_interval_seconds:
            return
        self._last_debug_write = current
        try:
            debug.directory.mkdir(parents=True, exist_ok=True)
            for prefix, (should_write, image) in enabled.items():
                if should_write:
                    path = timestamped_output_path(debug.directory, prefix, "jpg")
                    if not self._image_writer(str(path), image):
                        raise OSError(f"OpenCV could not write {path.name}")
            cleanup_oldest(debug.directory, "*.jpg", self.config.storage.max_debug_images, LOGGER)
        except Exception as exc:
            with self._condition:
                self._last_error = str(exc)
            LOGGER.error(
                "Debug image output failed",
                extra={"structured_data": {"event": "debug_image_failure", "error": str(exc)}},
                exc_info=True,
            )
