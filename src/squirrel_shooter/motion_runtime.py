"""Threaded motion/event consumer for the single shared camera runtime."""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Any, Callable

import cv2
import numpy as np

from .camera_service import CameraService, FramePacket
from .classifier import ClassifierEvidenceStore, EventClassifier
from .config import AppConfig
from .diagnostics import cleanup_oldest
from .event_report import generate_reports, load_events
from .event_storage import EventLogWriter, EventRecorder, RollingFrameBuffer, SessionLog, enforce_retention, recover_incomplete_events
from .files import timestamped_output_path
from .watch_detection import MotionWatcherDetector, WatchDetectionResult, annotate_watch_frame


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MotionRuntimeStatus:
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
    global_motion_rejections: int
    active_events: int
    last_detector_update: str | None
    last_detector_age_seconds: float | None
    last_event: str | None
    last_snapshot: str | None
    last_error: str | None
    thread_alive: bool
    capture_directory_writable: bool
    current_groups: tuple[dict[str, Any], ...]
    last_event_summary: dict[str, Any] | None


class MotionProcessingService:
    """Consume shared frames and own all detector, event, and report lifecycle state."""

    def __init__(
        self,
        camera: CameraService,
        config: AppConfig,
        *,
        detector: MotionWatcherDetector | None = None,
        classifier_service: EventClassifier | None = None,
        classifier_store: ClassifierEvidenceStore | None = None,
        video_writer_factory: Callable[..., Any] = cv2.VideoWriter,
        image_writer: Callable[[str, np.ndarray], bool] = cv2.imwrite,
    ) -> None:
        self.camera = camera
        self.config = config
        self.detector = detector or MotionWatcherDetector(config.motion)
        self.classifier_store = classifier_store or ClassifierEvidenceStore(config)
        self.classifier = classifier_service or EventClassifier(config.classifier, self.classifier_store)
        self._video_writer_factory = video_writer_factory
        self._image_writer = image_writer
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._finalized = False
        self._state = "STARTING"
        self._processing_fps = 0.0
        self._blob_count = 0
        self._persistence_count = 0
        self._frames_processed = 0
        self._candidates_seen = 0
        self._accepted_events = 0
        self._rejected_events = 0
        self._snapshots_saved = 0
        self._global_motion_rejections = 0
        self._active_events = 0
        self._last_detector_update: str | None = None
        self._last_detector_monotonic: float | None = None
        self._last_event: str | None = None
        self._last_snapshot: str | None = None
        self._last_error: str | None = None
        self._current_groups: tuple[dict[str, Any], ...] = ()
        self._last_event_summary: dict[str, Any] | None = None
        self._latest_frame: np.ndarray | None = None
        self._latest_annotated: np.ndarray | None = None
        self._latest_result: WatchDetectionResult | None = None
        self._latest_sequence = -1
        self._force_event_requested = False
        self._classifier_event_frames: dict[int, int] = {}
        self._classifier_submitted: set[int] = set()
        self._recent_events: deque[dict[str, Any]] = deque(maxlen=config.motion.recent_event_limit)
        self._logs: EventLogWriter | None = None
        self._session: SessionLog | None = None
        self._recorder: EventRecorder | None = None
        self._prebuffer = RollingFrameBuffer(config.motion.event_lifecycle.pre_event_seconds)
        self._last_rejection: str | None = None
        self._last_session_save = monotonic()
        self._last_camera_read_failures = 0
        self._camera_metadata: dict[str, Any] = {
            "requested_width": config.camera.requested_width,
            "requested_height": config.camera.requested_height,
            "requested_fps": config.camera.requested_fps,
            "actual_width": config.camera.requested_width,
            "actual_height": config.camera.requested_height,
            "camera_reported_fps": 0.0,
            "measured_camera_fps": 0.0,
            "camera_mode_if_known": config.camera.camera_mode_if_known,
            "ir_mode_if_explicitly_detected_or_configured": config.camera.ir_mode_if_explicitly_detected_or_configured,
            "low_fps_observed": False,
        }

    def start(self) -> None:
        """Start consuming frames; this method never starts or opens the camera."""

        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._finalized = False
            self._prepare_outputs()
            self.classifier.start()
            self._thread = threading.Thread(target=self._run, name="squirrel-motion", daemon=True)
            self._thread.start()
        LOGGER.info("Shared motion processor started", extra={"structured_data": {"event": "motion_runtime_started"}})

    def stop(self, timeout: float = 10.0) -> None:
        """Finish events/logs/reports without releasing the shared camera."""

        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        if thread is None or not thread.is_alive():
            self._finalize(clean=True)
        else:
            with self._condition:
                self._last_error = "Motion thread did not stop before the shutdown timeout"
        self.classifier.stop(timeout=timeout)
        LOGGER.info("Shared motion processor stopped", extra={"structured_data": {"event": "motion_runtime_stopped"}})

    def status(self) -> MotionRuntimeStatus:
        with self._condition:
            age = None if self._last_detector_monotonic is None else max(0.0, monotonic() - self._last_detector_monotonic)
            return MotionRuntimeStatus(
                self._state,
                self.config.motion.enabled,
                self._processing_fps,
                self._blob_count,
                self._persistence_count,
                self._frames_processed,
                self._candidates_seen,
                self._accepted_events,
                self._rejected_events,
                self._snapshots_saved,
                self._global_motion_rejections,
                self._active_events,
                self._last_detector_update,
                age,
                self._last_event,
                self._last_snapshot,
                self._last_error,
                self._thread is not None and self._thread.is_alive(),
                self.config.camera.output_directory.exists(),
                self._current_groups,
                None if self._last_event_summary is None else dict(self._last_event_summary),
            )

    def status_dict(self) -> dict[str, Any]:
        data = asdict(self.status())
        data["processing_fps"] = round(float(data["processing_fps"]), 1)
        age = data["last_detector_age_seconds"]
        data["last_detector_age_seconds"] = None if age is None else round(float(age), 2)
        data["alive"] = bool(data["thread_alive"] and (age is None or age <= self.config.health.detector_stale_seconds))
        return data

    def recent_events(self) -> list[dict[str, Any]]:
        with self._condition:
            return [dict(event) for event in reversed(self._recent_events)]

    def mjpeg_frames(self):  # type: ignore[no-untyped-def]
        return self.camera.mjpeg_frames(maximum_fps=self.config.dashboard.stream_fps)

    def request_forced_event(self) -> bool:
        """Queue a local test event; no dashboard route exposes this method."""

        with self._condition:
            if not self._current_groups:
                return False
            self._force_event_requested = True
            return True

    def save_manual_still(self) -> Path | None:
        with self._condition:
            frame = None if self._latest_annotated is None else self._latest_annotated.copy()
        if frame is None:
            return None
        directory = self.config.camera.output_directory / "manual"
        directory.mkdir(parents=True, exist_ok=True)
        path = timestamped_output_path(directory, "manual-still", "jpg")
        if not self._image_writer(str(path), frame):
            return None
        cleanup_oldest(directory, "*.jpg", self.config.storage.max_event_captures, LOGGER)
        return path

    def rebuild_report(self) -> tuple[Path, Path, Path]:
        return generate_reports(self.config)

    def _prepare_outputs(self) -> None:
        self.config.camera.output_directory.mkdir(parents=True, exist_ok=True)
        self._logs = EventLogWriter(self.config)
        requested = {
            "requested_width": self.config.camera.requested_width,
            "requested_height": self.config.camera.requested_height,
            "requested_fps": self.config.camera.requested_fps,
            "camera_mode_if_known": self.config.camera.camera_mode_if_known,
            "ir_mode_if_explicitly_detected_or_configured": self.config.camera.ir_mode_if_explicitly_detected_or_configured,
        }
        self._session = SessionLog(self.config, requested)
        recovered = recover_incomplete_events(self.config.camera.output_directory / "events")
        if recovered:
            self._session.data["recovered_incomplete_events"] = [str(path) for path in recovered]
        existing = load_events(self.config.camera.output_directory / "events")
        self._recent_events.extend(existing[-self.config.motion.recent_event_limit :])
        try:
            generate_reports(self.config)
        except Exception as exc:
            self._session.data["exception_details"].append(f"Startup report generation failed: {exc}")
        self._recorder = EventRecorder(
            self.config,
            self._logs,
            self._camera_metadata,
            video_writer_factory=self._video_writer_factory,
            image_writer=self._image_writer,
        )
        self._session.save()

    def _run(self) -> None:
        camera_sequence = -1
        clean = False
        try:
            while not self._stop_event.is_set():
                try:
                    packet = self.camera.wait_for_frame(camera_sequence)
                except Exception as exc:
                    self._record_error("Shared frame wait failed", exc)
                    self._stop_event.wait(0.2)
                    continue
                self._sync_camera_failures()
                if packet is None:
                    continue
                camera_sequence = packet.sequence
                try:
                    self._process_packet(packet)
                except Exception as exc:
                    self._record_error("Motion frame processing failed", exc)
                    try:
                        self.camera.publish_annotated(packet.sequence, packet.frame)
                    except Exception:
                        LOGGER.exception("Could not publish raw fallback after motion failure")
            clean = True
        except Exception as exc:
            self._record_error("Motion processor thread failed", exc)
        finally:
            self._finalize(clean=clean)

    def _process_packet(self, packet: FramePacket) -> None:
        now = packet.received_monotonic or monotonic()
        camera_status = self.camera.status()
        measured_fps = camera_status.fps
        self._camera_metadata.update(
            actual_width=camera_status.width,
            actual_height=camera_status.height,
            camera_reported_fps=camera_status.reported_fps,
            measured_camera_fps=measured_fps,
            low_fps_observed=0 < measured_fps < self.config.camera.low_fps_threshold,
        )
        if self._session is not None:
            self._session.data["camera_open_result"] = "success"
            self._session.data["actual_camera_mode"] = {
                key: self._camera_metadata[key]
                for key in ("actual_width", "actual_height", "camera_reported_fps", "camera_mode_if_known", "ir_mode_if_explicitly_detected_or_configured")
            }
            self._session.sample_fps(measured_fps)
        result = self.detector.process(packet.frame, now=now)
        annotated = annotate_watch_frame(packet.frame, result, measured_fps=measured_fps)
        self._handle_rejection(result, measured_fps)
        self._handle_events(packet, result, annotated, now, measured_fps)
        self.camera.publish_annotated(packet.sequence, annotated)
        self._prebuffer.append(now, annotated)
        now_iso = datetime.now().astimezone().isoformat(timespec="milliseconds")
        groups = tuple(group.as_dict() for group in result.groups)
        with self._condition:
            self._state = result.state.value
            self._processing_fps = result.measured_processing_fps
            self._blob_count = len(result.groups)
            self._persistence_count = max((group.persistence_count for group in result.groups), default=0)
            self._frames_processed += 1
            self._candidates_seen += len(result.groups)
            self._current_groups = groups
            self._latest_frame = packet.frame.copy()
            self._latest_annotated = annotated.copy()
            self._latest_result = result
            self._latest_sequence = packet.sequence
            self._last_detector_update = now_iso
            self._last_detector_monotonic = monotonic()
            self._last_error = None
            self._condition.notify_all()
        if self._session is not None:
            self._session.increment("raw_contours", result.raw_contour_count)
            self._session.increment("grouped_candidates", len(result.groups))
            if monotonic() - self._last_session_save >= 10.0:
                self._session.save()
                self._last_session_save = monotonic()

    def _handle_rejection(self, result: WatchDetectionResult, measured_fps: float) -> None:
        if result.global_motion.reason and result.state.value == "GLOBAL_RECOVERY":
            reason = result.global_motion.reason
            if self._last_rejection != reason:
                rejection = {
                    "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                    "reason": reason,
                    **result.global_motion.as_dict(),
                    "measured_camera_fps": measured_fps,
                    "low_fps_observed": self._camera_metadata["low_fps_observed"],
                    "ir_mode_if_explicitly_detected_or_configured": self._camera_metadata["ir_mode_if_explicitly_detected_or_configured"],
                }
                if self.config.motion.global_rejection.log_rejected_global_events and self._logs is not None:
                    self._logs.append_rejection(rejection)
                if self._session is not None:
                    self._session.reject(reason)
                self._global_motion_rejections += 1
                self._rejected_events += 1
                self._last_rejection = reason
                if self.config.motion.global_rejection.save_debug_snapshot:
                    directory = self.config.camera.output_directory / "rejections"
                    directory.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(timestamped_output_path(directory, reason, "jpg")), result.cleaned_mask)
                    cleanup_oldest(directory, "*.jpg", self.config.storage.max_debug_images, LOGGER)
        elif result.state.value == "READY":
            self._last_rejection = None

    def _handle_events(self, packet: FramePacket, result: WatchDetectionResult, annotated: np.ndarray, now: float, measured_fps: float) -> None:
        if self._recorder is None:
            return
        groups_by_track = {group.track_id: group for group in result.groups}
        with self._condition:
            force_requested = self._force_event_requested
            self._force_event_requested = False
        forced_track = None
        if force_requested:
            available = [group for group in result.groups if group.track_id not in self._recorder.active]
            if available:
                forced_track = max(available, key=lambda item: item.foreground_pixels).track_id
        for group in result.groups:
            should_begin = group.newly_confirmed or group.track_id == forced_track
            if should_begin and group.track_id not in self._recorder.active:
                event = self._recorder.begin(
                    group.track_id,
                    group,
                    packet.frame,
                    annotated,
                    self._prebuffer.frames(),
                    now=now,
                    measured_fps=measured_fps,
                )
                with self._condition:
                    self._accepted_events += 1
                    self._last_event = event.start_timestamp
                    self._last_event_summary = {
                        "event_id": event.event_id,
                        "start_timestamp": event.start_timestamp,
                        "provisional_category": group.provisional_category,
                        "movement_attributes": list(group.movement_attributes),
                        "status": "recording",
                    }
                if self._session is not None:
                    self._session.increment("confirmed_events")
                self._classifier_event_frames[group.track_id] = 1
                self._submit_classifier_if_due(event.event_id, event.directory, group, packet.frame)
            elif group.track_id in self._recorder.active:
                self._recorder.update(group.track_id, group, annotated, now=now)
                self._classifier_event_frames[group.track_id] = self._classifier_event_frames.get(group.track_id, 1) + 1
                event = self._recorder.active[group.track_id]
                self._submit_classifier_if_due(event.event_id, event.directory, group, packet.frame)
        for track_id, event in list(self._recorder.active.items()):
            if track_id not in groups_by_track:
                self._recorder.update(track_id, None, annotated, now=now)
            if self._recorder.should_finish(event, now):
                self._record_completed_event(self._recorder.finish(track_id, now=now))
                self._classifier_event_frames.pop(track_id, None)
                self._classifier_submitted.discard(track_id)
                active = {item.directory for item in self._recorder.active.values()}
                actions = enforce_retention(self.config.camera.output_directory / "events", self.config.retention, active_directories=active)
                if self._session is not None:
                    self._session.data["retention_actions"].extend(actions)
        with self._condition:
            self._active_events = len(self._recorder.active)

    def _submit_classifier_if_due(
        self,
        event_id: str,
        event_directory: Path,
        group: Any,
        frame: np.ndarray,
    ) -> None:
        track_id = int(group.track_id)
        frame_number = self._classifier_event_frames.get(track_id, 0)
        if track_id in self._classifier_submitted or frame_number != self.config.classifier.event_frame_number:
            return
        self._classifier_submitted.add(track_id)
        self.classifier.submit(event_id, event_directory, frame_number, frame, group.bounding_box)

    def _record_completed_event(self, record: dict[str, Any]) -> None:
        with self._condition:
            self._recent_events.append(record)
            self._last_event_summary = dict(record)
            self._last_snapshot = record.get("end_timestamp")
            self._snapshots_saved += 1

    def _sync_camera_failures(self) -> None:
        status = self.camera.status()
        new_failures = max(0, status.read_failures - self._last_camera_read_failures)
        if new_failures and self._session is not None:
            self._session.increment("dropped_or_failed_frame_reads", new_failures)
            self._session.increment("camera_read_errors", new_failures)
        self._last_camera_read_failures = status.read_failures

    def _record_error(self, message: str, exc: Exception) -> None:
        detail = f"{message}: {type(exc).__name__}: {exc}"
        with self._condition:
            self._state = "ERROR"
            self._last_error = detail
            self._condition.notify_all()
        if self._session is not None:
            self._session.data["exception_details"].append(detail)
            self._session.save()
        LOGGER.error(message, extra={"structured_data": {"event": "motion_runtime_error", "error": str(exc)}}, exc_info=True)

    def _finalize(self, *, clean: bool) -> None:
        with self._condition:
            if self._finalized:
                return
            self._finalized = True
        now = monotonic()
        if self._recorder is not None:
            for record in self._recorder.finish_all(now=now):
                self._record_completed_event(record)
            with self._condition:
                self._active_events = 0
        actions = enforce_retention(self.config.camera.output_directory / "events", self.config.retention)
        if self._session is not None:
            self._sync_camera_failures()
            self._session.data["retention_actions"].extend(actions)
            if self._session.data.get("camera_open_result") == "not_attempted":
                self._session.data["camera_open_result"] = "failed"
            if self.config.reporting.rebuild_on_clean_shutdown and clean:
                try:
                    generate_reports(self.config)
                except Exception as exc:
                    self._session.data["exception_details"].append(f"Report generation failed: {exc}")
            self._session.finish(clean=clean)
