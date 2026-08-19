"""Threaded motion/event consumer for the single shared camera runtime."""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from time import monotonic, perf_counter
from typing import Any, Callable, Iterable

import cv2
import numpy as np

from .auto_fire import (
    AutoFireDetection,
    AutoFireService,
    AutoFireTargetAssociation,
    AutoFireTargetSnapshot,
)
from .camera_service import CameraService, FramePacket
from .classifier import ClassifierDetection, ClassifierEvidenceStore, ClassifierTask, EventClassifier
from .config import AppConfig
from .diagnostics import cleanup_oldest
from .event_report import generate_reports, load_events
from .event_storage import (
    EventLogWriter,
    EventRecorder,
    EventStorageManager,
    SessionLog,
    recover_incomplete_events,
)
from .files import timestamped_output_path
from .frame_selection import BestEventFrameSelector
from .manual_control import ManualControlService
from .performance import AverageTimer, ThreadCpuMeter, TimingDistribution
from .target_association import (
    AssociationDecision,
    AssociationPolicy,
    TargetObservation,
    evaluate_reacquisition,
)
from .thread_names import set_current_thread_name
from .watch_detection import MotionWatcherDetector, WatchDetectionResult, annotate_watch_frame


LOGGER = logging.getLogger(__name__)


class _UnavailableAutoFireError(RuntimeError):
    reason = "hardware_not_ready"


class _UnavailableAutoFireCoordinator:
    """Fail-closed placeholder when the shared physical coordinator was not built."""

    @staticmethod
    def cooldown_remaining_seconds() -> float:
        return 0.0

    @staticmethod
    def automatic_engage(*_args: object, **_kwargs: object) -> object:
        raise _UnavailableAutoFireError("The shared physical coordinator is unavailable")


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
    night_mode_paused: bool = False
    night_mode_evidence: str | None = None
    target_fps: float = 0.0
    detector_average_ms: float = 0.0
    annotation_average_ms: float = 0.0
    motion_thread_cpu_percent: float = 0.0
    annotations_rendered: int = 0
    idle_annotations_skipped: int = 0
    auto_fire: dict[str, Any] = field(default_factory=dict)
    motion_loop_start_interval_timing: dict[str, object] = field(default_factory=dict)
    motion_loop_duration_timing: dict[str, object] = field(default_factory=dict)
    event_storage: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _BufferedDetection:
    timestamp: float
    frame: np.ndarray
    result: WatchDetectionResult
    measured_fps: float


@dataclass(frozen=True)
class _CoastingAutoTarget:
    last_snapshot: AutoFireTargetSnapshot
    lost_at_monotonic: float


class _DetectionFrameBuffer:
    """Retain raw event lead-in frames and render overlays only if an event begins."""

    def __init__(self, duration_seconds: float) -> None:
        self.duration_seconds = duration_seconds
        self._frames: deque[_BufferedDetection] = deque()

    def append(self, timestamp: float, frame: np.ndarray, result: WatchDetectionResult, measured_fps: float) -> None:
        # CameraService gives this worker an immutable borrowed frame whose
        # ndarray remains alive through normal Python reference ownership.
        self._frames.append(_BufferedDetection(timestamp, frame, result, measured_fps))
        cutoff = timestamp - self.duration_seconds
        while self._frames and self._frames[0].timestamp < cutoff:
            self._frames.popleft()

    def rendered_frames(
        self,
        render: Callable[[np.ndarray, WatchDetectionResult, float], np.ndarray],
    ) -> Iterable[tuple[float, np.ndarray]]:
        frames = tuple(self._frames)
        return (
            (item.timestamp, render(item.frame, item.result, item.measured_fps))
            for item in frames
        )

    def clear(self) -> None:
        self._frames.clear()

    def __len__(self) -> int:
        return len(self._frames)


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
        manual_control_service: ManualControlService | None = None,
        auto_fire_service: AutoFireService | None = None,
        video_writer_factory: Callable[..., Any] = cv2.VideoWriter,
        image_writer: Callable[[str, np.ndarray], bool] = cv2.imwrite,
    ) -> None:
        self.camera = camera
        self.config = config
        self.detector = detector or MotionWatcherDetector(config.motion)
        self.classifier_store = classifier_store or ClassifierEvidenceStore(config)
        self._video_writer_factory = video_writer_factory
        self._image_writer = image_writer
        self._condition = threading.Condition()
        self._live_auto_targets: dict[tuple[str, int], AutoFireTargetSnapshot] = {}
        self._coasting_auto_targets: dict[tuple[str, int], _CoastingAutoTarget] = {}
        self._expired_auto_targets: dict[tuple[str, int], AutoFireTargetAssociation] = {}
        association_distance = self.config.auto_fire.reacquisition_max_centroid_distance_pixels
        association_area_ratio = self.config.auto_fire.reacquisition_max_area_ratio
        self._target_association_policy = AssociationPolicy(
            maximum_elapsed_seconds=self.config.auto_fire.track_loss_grace_seconds,
            base_prediction_error_pixels=association_distance,
            maximum_prediction_error_pixels=max(200.0, association_distance * 2.0),
            strong_proximity_pixels=min(60.0, association_distance),
            ordinary_area_ratio=association_area_ratio,
            posture_change_area_ratio=max(4.0, association_area_ratio),
        )
        self.manual_control = manual_control_service
        coordinator = manual_control_service or _UnavailableAutoFireCoordinator()
        self.auto_fire = auto_fire_service or AutoFireService(
            config.auto_fire,
            coordinator,
            self._auto_fire_target,
            self._auto_fire_night_mode,
        )
        self.classifier = classifier_service or EventClassifier(
            config.classifier,
            self.classifier_store,
            result_handler=self._handle_classifier_result,
            evidence_result_handler=self._handle_classifier_evidence_result,
        )
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
        explicit_ir = config.camera.ir_mode_if_explicitly_detected_or_configured.strip().lower()
        self._night_mode_paused = bool(
            config.night_mode.pause_recording_and_classifier
            and explicit_ir in {"night", "night_vision", "ir", "infrared", "on", "enabled", "true"}
        )
        self._night_mode_evidence: str | None = "explicit_camera_setting" if self._night_mode_paused else None
        self._night_mono_frames = 0
        self._night_color_frames = 0
        self._force_event_requested = False
        self._classifier_selectors: dict[str, BestEventFrameSelector] = {}
        self._classifier_clip_offsets: dict[str, int] = {}
        self._classifier_storage_leases: dict[str, str] = {}
        self._live_group_holds: dict[int, tuple[float, Any]] = {}
        self._recent_events: deque[dict[str, Any]] = deque(maxlen=config.motion.recent_event_limit)
        self._logs: EventLogWriter | None = None
        self._session: SessionLog | None = None
        self._recorder: EventRecorder | None = None
        self._event_storage: EventStorageManager | None = None
        self._prebuffer = _DetectionFrameBuffer(config.motion.event_lifecycle.pre_event_seconds)
        self._detector_timer = AverageTimer()
        self._annotation_timer = AverageTimer()
        self._motion_cpu = ThreadCpuMeter()
        self._motion_loop_start_interval = TimingDistribution()
        self._motion_loop_duration = TimingDistribution()
        self._last_motion_loop_started: float | None = None
        self._annotations_rendered = 0
        self._idle_annotations_skipped = 0
        self._last_rejection: str | None = None
        self._last_session_save = monotonic()
        self._last_camera_read_failures = 0
        self._error_throttle: OrderedDict[str, tuple[float, int]] = OrderedDict()
        self._suppressed_error_count = 0
        self._event_storage_last_error: str | None = None
        self._event_storage_submission_rejections = 0
        self._camera_metadata: dict[str, Any] = {
            "source_camera": f"opencv_device_{config.camera.device_index}",
            "camera_device_index": config.camera.device_index,
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
        self.classifier.set_paused(self._night_mode_paused)

    def start(self) -> None:
        """Start consuming frames; this method never starts or opens the camera."""

        self.auto_fire.start_accepting()
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._finalized = False
            self._error_throttle.clear()
            self._suppressed_error_count = 0
            self._event_storage_last_error = None
            self._event_storage_submission_rejections = 0
            self._motion_loop_start_interval.reset()
            self._motion_loop_duration.reset()
            self._last_motion_loop_started = None
            self._classifier_storage_leases.clear()
            self._prepare_outputs()
            self.classifier.start()
            self._thread = threading.Thread(target=self._run, name="motion-detect", daemon=True)
            self._thread.start()
        LOGGER.info("Shared motion processor started", extra={"structured_data": {"event": "motion_runtime_started"}})

    def stop(self, timeout: float = 10.0) -> None:
        """Finish events/logs/reports without releasing the shared camera."""

        self.auto_fire.begin_shutdown()
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
                self._night_mode_paused,
                self._night_mode_evidence,
                self.config.motion.target_fps,
                self._detector_timer.average_ms,
                self._annotation_timer.average_ms,
                self._motion_cpu.percent,
                self._annotations_rendered,
                self._idle_annotations_skipped,
                self.auto_fire.status(),
                self._motion_loop_start_interval.snapshot(),
                self._motion_loop_duration.snapshot(),
                self._event_storage_status(),
            )

    def status_dict(self) -> dict[str, Any]:
        data = asdict(self.status())
        data["processing_fps"] = round(float(data["processing_fps"]), 1)
        age = data["last_detector_age_seconds"]
        data["last_detector_age_seconds"] = None if age is None else round(float(age), 2)
        data["alive"] = bool(data["thread_alive"] and (age is None or age <= self.config.health.detector_stale_seconds))
        return data

    def _event_storage_status(self) -> dict[str, Any]:
        manager = self._event_storage
        recorder = self._recorder
        data: dict[str, Any] = {}
        if manager is not None:
            data.update(asdict(manager.status()))
        writer_statuses = None if recorder is None else getattr(recorder, "active_writer_statuses", None)
        if callable(writer_statuses):
            data["active_writers"] = list(writer_statuses())
        else:
            data["active_writers"] = []
        data["runtime_last_error"] = self._event_storage_last_error
        data["submission_rejections"] = self._event_storage_submission_rejections
        return data

    def recent_events(self) -> list[dict[str, Any]]:
        with self._condition:
            return [dict(event) for event in reversed(self._recent_events)]

    def _auto_fire_target(
        self,
        event_id: str,
        track_id: int,
    ) -> AutoFireTargetSnapshot | AutoFireTargetAssociation | None:
        key = (event_id, track_id)
        with self._condition:
            target = self._live_auto_targets.get(key)
            if target is not None:
                return target
            coasting = self._coasting_auto_targets.get(key)
            if coasting is not None:
                return AutoFireTargetAssociation(
                    "coasting",
                    coasting.lost_at_monotonic
                    + self.config.auto_fire.track_loss_grace_seconds,
                )
            return self._expired_auto_targets.get(key)

    def _auto_fire_night_mode(self) -> bool:
        with self._condition:
            return self._night_mode_paused

    def _handle_classifier_result(
        self,
        task: ClassifierTask,
        detections: list[ClassifierDetection],
        error: str | None,
        _record: dict[str, Any],
    ) -> None:
        if task.context != "auto_fire_live_event":
            return
        self.auto_fire.handle_classification(
            event_id=task.event_id,
            track_id=task.track_id,  # type: ignore[arg-type]
            classified_observation_monotonic=task.target_observed_monotonic,  # type: ignore[arg-type]
            detections=tuple(AutoFireDetection(item.label, item.confidence) for item in detections),
            error=error,
        )

    def _handle_classifier_evidence_result(
        self,
        task: ClassifierTask,
        status: str,
        _record: dict[str, Any] | None,
        error: str | None,
    ) -> None:
        """Release retention ownership only after evidence persistence terminates."""

        if task.context != "completed_event":
            return
        manager = self._event_storage
        with self._condition:
            lease_id = self._classifier_storage_leases.pop(task.event_id, None)
        if lease_id is not None and manager is not None:
            manager.release_lease(lease_id)
        if status != "persisted":
            LOGGER.error(
                "Classifier evidence did not persist for completed event %s: %s",
                task.event_id,
                error or status,
                extra={
                    "structured_data": {
                        "event": "classifier_evidence_persistence_failed",
                        "event_id": task.event_id,
                        "status": status,
                        "error": error,
                    }
                },
            )

    def mjpeg_frames(self):  # type: ignore[no-untyped-def]
        return self.camera.mjpeg_frames(
            maximum_fps=self.config.dashboard.stream_fps,
            annotated_only=True,
        )

    def request_forced_event(self) -> bool:
        """Queue a local test event; no dashboard route exposes this method."""

        with self._condition:
            if self._night_mode_paused or not self._current_groups:
                return False
            self._force_event_requested = True
            return True

    def save_manual_still(self) -> Path | None:
        frame = self.camera.latest_annotated_frame()
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
        self._camera_metadata["session_id"] = self._session.data["session_id"]
        recovered = recover_incomplete_events(self.config.camera.output_directory / "events")
        if recovered:
            self._session.data["recovered_incomplete_events"] = [str(path) for path in recovered]
        existing = load_events(self.config.camera.output_directory / "events")
        self._recent_events.extend(existing[-self.config.motion.recent_event_limit :])
        try:
            generate_reports(self.config)
        except Exception as exc:
            self._session.add_exception(f"Startup report generation failed: {exc}")
        self._recorder = EventRecorder(
            self.config,
            self._logs,
            self._camera_metadata,
            video_writer_factory=self._video_writer_factory,
            image_writer=self._image_writer,
        )
        self._event_storage = EventStorageManager(
            self._recorder,
            self.config.camera.output_directory / "events",
            self.config.retention,
        )
        self._event_storage.start()
        self._session.save()

    def _run(self) -> None:
        set_current_thread_name("motion-detect")
        camera_sequence = -1
        next_frame_at = 0.0
        frame_interval = 1.0 / self.config.motion.target_fps
        consecutive_failures = 0
        clean = False
        try:
            while not self._stop_event.is_set():
                remaining = next_frame_at - monotonic()
                if remaining > 0 and self._stop_event.wait(remaining):
                    break
                loop_started = monotonic()
                if self._last_motion_loop_started is not None:
                    self._motion_loop_start_interval.add(
                        loop_started - self._last_motion_loop_started
                    )
                self._last_motion_loop_started = loop_started
                try:
                    self._poll_event_storage()
                    try:
                        packet = self.camera.wait_for_frame(camera_sequence, copy=False)
                    except Exception as exc:
                        consecutive_failures += 1
                        self._record_error("Shared frame wait failed", exc)
                        self._stop_event.wait(self._failure_backoff_seconds(consecutive_failures))
                        continue
                    self._sync_camera_failures()
                    if packet is None:
                        continue
                    camera_sequence = packet.sequence
                    next_frame_at = monotonic() + frame_interval
                    try:
                        self._process_packet(packet)
                        self._motion_cpu.update()
                        if consecutive_failures:
                            self._flush_suppressed_errors("Motion processing recovered")
                        consecutive_failures = 0
                    except Exception as exc:
                        consecutive_failures += 1
                        self._record_error("Motion frame processing failed", exc)
                        if self._display_requested():
                            try:
                                self.camera.publish_annotated(packet.sequence, packet.frame, copy=False)
                            except Exception:
                                LOGGER.exception("Could not publish raw fallback after motion failure")
                        self._stop_event.wait(self._failure_backoff_seconds(consecutive_failures))
                finally:
                    self._motion_loop_duration.add(monotonic() - loop_started)
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
        detector_started = perf_counter()
        result = self.detector.process(packet.frame, now=now)
        self._detector_timer.add(perf_counter() - detector_started)
        processing_fps = result.measured_processing_fps
        annotated: np.ndarray | None = None

        def render_current() -> np.ndarray:
            nonlocal annotated
            if annotated is None:
                annotated = self._render_annotation(packet.frame, result, processing_fps)
            return annotated

        self._update_night_mode(result, now)
        if not self._night_mode_paused:
            self._handle_rejection(result, measured_fps)
            self._handle_events(packet, result, render_current, now, processing_fps)
            self._prebuffer.append(now, packet.frame, result, processing_fps)
        if self._display_requested():
            visible = render_current()
            live_annotated = (
                self._annotate_night_pause(visible)
                if self._night_mode_paused
                else self._add_live_box_holds(visible, result.groups, now)
            )
            self.camera.publish_annotated(packet.sequence, live_annotated, copy=False)
        elif annotated is None:
            self._idle_annotations_skipped += 1
        now_iso = datetime.now().astimezone().isoformat(timespec="milliseconds")
        groups = tuple(group.as_dict() for group in result.groups)
        with self._condition:
            self._state = "NIGHT_PAUSED" if self._night_mode_paused else result.state.value
            self._processing_fps = result.measured_processing_fps
            self._blob_count = len(result.groups)
            self._persistence_count = max((group.persistence_count for group in result.groups), default=0)
            self._frames_processed += 1
            self._candidates_seen += len(result.groups)
            self._current_groups = groups
            self._last_detector_update = now_iso
            self._last_detector_monotonic = monotonic()
            self._last_error = None
            self._condition.notify_all()
        if self._session is not None:
            self._session.increment("raw_contours", result.raw_contour_count)
            self._session.increment("grouped_candidates", len(result.groups))
            if monotonic() - self._last_session_save >= self.config.runtime.telemetry_interval_seconds:
                self._session.save()
                self._last_session_save = monotonic()

    def _update_night_mode(self, result: WatchDetectionResult, now: float) -> None:
        settings = self.config.night_mode
        if not settings.pause_recording_and_classifier:
            return
        monochrome = result.global_motion.colorfulness <= settings.monochrome_colorfulness_threshold
        if monochrome:
            self._night_mono_frames += 1
            self._night_color_frames = 0
        else:
            self._night_color_frames += 1
            self._night_mono_frames = 0

        transition_to_mono = (
            result.global_motion.reason == "probable_ir_mode_switch" and monochrome
        )
        should_pause = transition_to_mono or self._night_mono_frames >= settings.enter_consecutive_frames
        if not self._night_mode_paused and should_pause:
            self._night_mode_paused = True
            self._night_mode_evidence = (
                "probable_ir_mode_switch" if transition_to_mono else "sustained_monochrome_frames"
            )
            self.classifier.set_paused(True)
            self._prebuffer.clear()
            self._live_group_holds.clear()
            if self._recorder is not None:
                self._request_all_event_finalizations(now=now, notes="night vision pause")
            self._classifier_selectors.clear()
            self._classifier_clip_offsets.clear()
            with self._condition:
                self._active_events = 0
                self._force_event_requested = False
                self._live_auto_targets.clear()
                self._coasting_auto_targets.clear()
                self._expired_auto_targets.clear()
            self.auto_fire.notify_target_state_changed()
            LOGGER.info(
                "Night vision detected; event recording and classifier paused",
                extra={"structured_data": {"event": "night_mode_paused", "evidence": self._night_mode_evidence}},
            )
        elif self._night_mode_paused and self._night_color_frames >= settings.exit_consecutive_frames:
            self._night_mode_paused = False
            self._night_mode_evidence = "sustained_color_return"
            self.classifier.set_paused(False)
            self._prebuffer.clear()
            self._live_group_holds.clear()
            LOGGER.info(
                "Day color returned; event recording and classifier resumed",
                extra={"structured_data": {"event": "night_mode_resumed"}},
            )

    @staticmethod
    def _annotate_night_pause(annotated: np.ndarray) -> np.ndarray:
        output = annotated.copy()
        message = "NIGHT VISION - RECORDING AND CLASSIFIER PAUSED"
        cv2.rectangle(output, (0, 0), (output.shape[1], 42), (12, 12, 12), -1)
        cv2.putText(
            output,
            message,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (80, 210, 255),
            2,
            cv2.LINE_AA,
        )
        return output

    def _add_live_box_holds(
        self,
        annotated: np.ndarray,
        groups: tuple[Any, ...] | list[Any],
        now: float,
    ) -> np.ndarray:
        """Keep a last-known box visible during the tracker's short gap window."""

        visible_track_ids = {int(group.track_id) for group in groups}
        hold_seconds = self.config.motion.persistence.maximum_gap_seconds
        for group in groups:
            self._live_group_holds[int(group.track_id)] = (now + hold_seconds, group)
        expired = [track_id for track_id, (until, _) in self._live_group_holds.items() if until < now]
        for track_id in expired:
            del self._live_group_holds[track_id]
        hidden_track_ids = [track_id for track_id in self._live_group_holds if track_id not in visible_track_ids]
        if not hidden_track_ids:
            return annotated
        held = annotated.copy()
        for track_id in hidden_track_ids:
            _, group = self._live_group_holds[track_id]
            x, y, width, height = group.bounding_box
            color = (0, 170, 255)
            cv2.rectangle(held, (x, y), (x + width, y + height), color, 3)
            cv2.putText(
                held,
                f"last seen: {group.provisional_category}",
                (x, max(22, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )
        return held

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

    def _display_requested(self) -> bool:
        return not self.config.runtime.headless or bool(getattr(self.camera, "has_dashboard_viewers", False))

    def _render_annotation(
        self,
        frame: np.ndarray,
        result: WatchDetectionResult,
        measured_fps: float,
    ) -> np.ndarray:
        started = perf_counter()
        annotated = annotate_watch_frame(frame, result, measured_fps=measured_fps)
        self._annotation_timer.add(perf_counter() - started)
        self._annotations_rendered += 1
        return annotated

    def _handle_events(
        self,
        packet: FramePacket,
        result: WatchDetectionResult,
        get_annotated: Callable[[], np.ndarray],
        now: float,
        measured_fps: float,
    ) -> None:
        if self._night_mode_paused or self._recorder is None:
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
                pre_event_frame_count = len(self._prebuffer)
                pre_event_frames = self._prebuffer.rendered_frames(self._render_annotation)
                event = self._recorder.begin(
                    group.track_id,
                    group,
                    packet.frame,
                    get_annotated(),
                    pre_event_frames,
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
                if self.config.classifier.enabled:
                    if self.config.auto_fire.enabled:
                        self._update_live_auto_target(
                            event.event_id,
                            group,
                            result.groups,
                            packet.frame,
                            now,
                            frame_sequence=packet.sequence,
                        )
                        self._submit_live_auto_fire_classification(event, group, packet, now)
                    else:
                        selector = BestEventFrameSelector(
                            fallback_frame_number=self.config.classifier.fallback_event_frame_number,
                            minimum_motion_area=self.config.motion.min_blob_area,
                            selection_mode=self.config.classifier.frame_selection,
                        )
                        self._classifier_selectors[event.event_id] = selector
                        self._classifier_clip_offsets[event.event_id] = pre_event_frame_count
                        self._consider_classifier_frame(selector, group, packet.frame)
            elif group.track_id in self._recorder.active:
                self._recorder.update(group.track_id, group, get_annotated(), now=now)
                event = self._recorder.active[group.track_id]
                if self.config.auto_fire.enabled:
                    self._update_live_auto_target(
                        event.event_id,
                        group,
                        result.groups,
                        packet.frame,
                        now,
                        frame_sequence=packet.sequence,
                    )
                selector = self._classifier_selectors.get(event.event_id)
                if selector is not None:
                    self._consider_classifier_frame(selector, group, packet.frame)
        for track_id, event in list(self._recorder.active.items()):
            if track_id not in groups_by_track:
                if self.config.auto_fire.enabled:
                    self._handle_missing_auto_target(
                        event.event_id,
                        track_id,
                        result.groups,
                        now,
                        frame_sequence=packet.sequence,
                    )
                self._recorder.update(track_id, None, get_annotated(), now=now)
                selector = self._classifier_selectors.get(event.event_id)
                if selector is not None:
                    self._consider_classifier_frame(selector, None, packet.frame)
            if self._recorder.should_finish(event, now):
                if self._request_event_finalization(track_id, now=now):
                    with self._condition:
                        self._live_auto_targets.pop((event.event_id, track_id), None)
                        self._coasting_auto_targets.pop((event.event_id, track_id), None)
                        self._expired_auto_targets.pop((event.event_id, track_id), None)
                    self.auto_fire.notify_target_state_changed()
        with self._condition:
            self._active_events = len(self._recorder.active)

    def _request_event_finalization(
        self,
        track_id: int,
        *,
        now: float,
        notes: str = "",
    ) -> bool:
        recorder = self._recorder
        if recorder is None:
            return False
        manager = self._event_storage
        if manager is None:
            # Unit-level injected recorders retain their historical synchronous
            # contract. Production setup always constructs the storage manager.
            self._record_completed_event(recorder.finish(track_id, now=now, notes=notes))
            return True
        submission = manager.request_finalize(track_id, now=now, notes=notes)
        if submission.accepted:
            return True
        self._event_storage_submission_rejections += 1
        self._note_event_storage_failure(
            f"Event finalization submission rejected for track {track_id}: {submission.reason}"
        )
        return False

    def _request_all_event_finalizations(self, *, now: float, notes: str) -> bool:
        recorder = self._recorder
        if recorder is None:
            return True
        if self._event_storage is None:
            for record in recorder.finish_all(now=now, notes=notes):
                self._record_completed_event(record)
            return True
        all_accepted = True
        for track_id in tuple(recorder.active):
            if self._request_event_finalization(track_id, now=now, notes=notes):
                continue
            all_accepted = False
            abandon = getattr(recorder, "abandon", None)
            if callable(abandon):
                try:
                    abandon(
                        track_id,
                        reason=f"finalization unavailable during {notes}",
                    )
                except KeyError:
                    pass
        return all_accepted

    def _poll_event_storage(self) -> None:
        manager = self._event_storage
        if manager is None:
            return
        request_retention = False
        for result in manager.poll_results(
            maximum=2,
            claim_finalize_directories=True,
        ):
            if result.kind == "finalize":
                request_retention = True
                if result.record is not None:
                    self._record_completed_event(
                        result.record,
                        storage_lease_id=result.lease_id,
                    )
                elif result.event_id:
                    self._classifier_selectors.pop(result.event_id, None)
                    self._classifier_clip_offsets.pop(result.event_id, None)
                    if result.lease_id is not None:
                        manager.release_lease(result.lease_id)
                if not result.success:
                    self._note_event_storage_failure(
                        f"Event {result.event_id or 'unknown'} evidence finalization failed: "
                        f"{result.error or 'unknown error'}"
                    )
            elif result.success:
                if self._session is not None:
                    self._session.add_retention_actions(list(result.retention_actions))
            else:
                self._note_event_storage_failure(
                    f"Event retention failed: {result.error or 'unknown error'}"
                )
        if request_retention:
            submission = manager.request_retention()
            if not submission.accepted and submission.reason != "manager_not_running":
                self._event_storage_submission_rejections += 1
                self._note_event_storage_failure(
                    f"Event retention submission rejected: {submission.reason}"
                )

    def _note_event_storage_failure(self, detail: str) -> None:
        repeated = detail == self._event_storage_last_error
        self._event_storage_last_error = detail
        if repeated:
            return
        if self._session is not None:
            self._session.add_exception(detail)
        LOGGER.error(
            detail,
            extra={"structured_data": {"event": "event_storage_failure", "error": detail}},
        )

    def _update_live_auto_target(
        self,
        event_id: str,
        group: Any,
        groups: tuple[Any, ...] | list[Any],
        frame: np.ndarray,
        observed_monotonic: float,
        frame_sequence: int | None = None,
    ) -> None:
        centroid_x, centroid_y = group.centroid
        frame_height, frame_width = frame.shape[:2]
        pixel_x = max(0, round(float(centroid_x)))
        pixel_y = max(0, round(float(centroid_y)))
        key = (event_id, int(group.track_id))
        with self._condition:
            previous_live = self._live_auto_targets.get(key)
            coasting = self._coasting_auto_targets.get(key)
            expired = key in self._expired_auto_targets
        if expired:
            return
        velocity = (0.0, 0.0)
        if previous_live is not None:
            elapsed_since_previous = observed_monotonic - previous_live.observed_monotonic
            if elapsed_since_previous > 0.0:
                velocity = (
                    (pixel_x - previous_live.pixel_x) / elapsed_since_previous,
                    (pixel_y - previous_live.pixel_y) / elapsed_since_previous,
                )
        snapshot = AutoFireTargetSnapshot(
            event_id=event_id,
            track_id=int(group.track_id),
            observed_monotonic=observed_monotonic,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            bounding_box=tuple(int(value) for value in group.bounding_box),  # type: ignore[arg-type]
            frame_width=int(frame_width),
            frame_height=int(frame_height),
            confirmed=bool(group.confirmed),
            event_eligible=bool(group.event_eligible),
            provisional_category=str(group.provisional_category),
            frame_sequence=frame_sequence,
            velocity=velocity,
        )
        if coasting is None:
            with self._condition:
                self._live_auto_targets[key] = snapshot
                self._expired_auto_targets.pop(key, None)
            return

        elapsed = observed_monotonic - coasting.lost_at_monotonic
        association = self._evaluate_target_reacquisition(
            coasting.last_snapshot,
            groups,
            observed_monotonic,
        )
        if elapsed > self.config.auto_fire.track_loss_grace_seconds:
            self._record_reacquisition_diagnostic(
                snapshot.track_id,
                coasting,
                groups,
                observed_monotonic,
                "timeout",
                association,
                frame_sequence=frame_sequence,
            )
            self._expire_auto_target(key, coasting, "reacquisition_timeout", observed_monotonic)
            return
        if association.state == "ambiguous":
            self._record_reacquisition_diagnostic(
                snapshot.track_id,
                coasting,
                groups,
                observed_monotonic,
                "ambiguous",
                association,
                frame_sequence=frame_sequence,
            )
            self._expire_auto_target(key, coasting, "reacquisition_ambiguous", observed_monotonic)
            return
        if (
            association.state != "accepted"
            or association.selected is None
            or association.selected.track_id != snapshot.track_id
        ):
            self._record_reacquisition_diagnostic(
                snapshot.track_id,
                coasting,
                groups,
                observed_monotonic,
                "incompatible",
                association,
                frame_sequence=frame_sequence,
            )
            self._expire_auto_target(key, coasting, "reacquisition_incompatible", observed_monotonic)
            return

        self._record_reacquisition_diagnostic(
            snapshot.track_id,
            coasting,
            groups,
            observed_monotonic,
            "reacquired",
            association,
            frame_sequence=frame_sequence,
        )

        with self._condition:
            self._coasting_auto_targets.pop(key, None)
            self._expired_auto_targets.pop(key, None)
            self._live_auto_targets[key] = snapshot
        LOGGER.info(
            "AUTO_FIRE target reacquired event=%s track=%s after=%.3fs old=(%d,%d) new=(%d,%d)",
            snapshot.event_id,
            snapshot.track_id,
            elapsed,
            coasting.last_snapshot.pixel_x,
            coasting.last_snapshot.pixel_y,
            snapshot.pixel_x,
            snapshot.pixel_y,
            extra={
                "structured_data": {
                    "event": "auto_fire_target_reacquired",
                    "event_id": snapshot.event_id,
                    "track_id": snapshot.track_id,
                    "source_frame_sequence": snapshot.frame_sequence,
                    "reacquisition_seconds": elapsed,
                    "association_reason": association.reason,
                    "old_centroid": {
                        "x": coasting.last_snapshot.pixel_x,
                        "y": coasting.last_snapshot.pixel_y,
                    },
                    "new_centroid": {"x": snapshot.pixel_x, "y": snapshot.pixel_y},
                }
            },
        )
        self.auto_fire.notify_target_state_changed()

    def _handle_missing_auto_target(
        self,
        event_id: str,
        track_id: int,
        groups: tuple[Any, ...] | list[Any],
        observed_monotonic: float,
        frame_sequence: int | None = None,
    ) -> None:
        key = (event_id, track_id)
        with self._condition:
            live = self._live_auto_targets.pop(key, None)
            coasting = self._coasting_auto_targets.get(key)
            expired = key in self._expired_auto_targets
            if live is not None:
                coasting = _CoastingAutoTarget(live, observed_monotonic)
                self._coasting_auto_targets[key] = coasting
        if live is not None:
            LOGGER.info(
                "AUTO_FIRE target entered bounded coasting event=%s track=%s grace=%.3fs last=(%d,%d)",
                event_id,
                track_id,
                self.config.auto_fire.track_loss_grace_seconds,
                live.pixel_x,
                live.pixel_y,
                extra={
                    "structured_data": {
                        "event": "auto_fire_target_coasting",
                        "event_id": event_id,
                        "track_id": track_id,
                        "target_last_seen_monotonic": live.observed_monotonic,
                        "target_last_seen_frame_sequence": live.frame_sequence,
                        "source_frame_sequence": frame_sequence,
                        "coasting_started_monotonic": observed_monotonic,
                        "grace_seconds": self.config.auto_fire.track_loss_grace_seconds,
                        "last_centroid": {"x": live.pixel_x, "y": live.pixel_y},
                    }
                },
            )
            self.auto_fire.notify_target_state_changed()
        if coasting is None or expired:
            return
        association = self._evaluate_target_reacquisition(
            coasting.last_snapshot,
            groups,
            observed_monotonic,
        )
        if association.state == "ambiguous":
            self._record_reacquisition_diagnostic(
                track_id,
                coasting,
                groups,
                observed_monotonic,
                "ambiguous",
                association,
                frame_sequence=frame_sequence,
            )
            self._expire_auto_target(key, coasting, "reacquisition_ambiguous", observed_monotonic)
            return
        if association.reason == "tracker_identity_changed":
            self._record_reacquisition_diagnostic(
                track_id,
                coasting,
                groups,
                observed_monotonic,
                "different_tracker_id",
                association,
                frame_sequence=frame_sequence,
            )
            self._expire_auto_target(key, coasting, "reacquisition_incompatible", observed_monotonic)
            return
        if observed_monotonic - coasting.lost_at_monotonic > self.config.auto_fire.track_loss_grace_seconds:
            self._record_reacquisition_diagnostic(
                track_id,
                coasting,
                groups,
                observed_monotonic,
                "timeout",
                association,
                frame_sequence=frame_sequence,
            )
            self._expire_auto_target(key, coasting, "reacquisition_timeout", observed_monotonic)
            return
        self._record_reacquisition_diagnostic(
            track_id,
            coasting,
            groups,
            observed_monotonic,
            "coasting",
            association,
            frame_sequence=frame_sequence,
        )

    def _evaluate_target_reacquisition(
        self,
        previous: AutoFireTargetSnapshot,
        groups: tuple[Any, ...] | list[Any],
        observed_monotonic: float,
    ) -> AssociationDecision:
        reference = TargetObservation(
            track_id=previous.track_id,
            observed_monotonic=previous.observed_monotonic,
            centroid=(float(previous.pixel_x), float(previous.pixel_y)),
            bounding_box=previous.bounding_box,
            velocity=tuple(float(value) for value in previous.velocity),
            confirmed=previous.confirmed,
            event_eligible=previous.event_eligible,
            provisional_category=previous.provisional_category,
            grouping_confidence=1.0,
        )
        candidates = tuple(
            TargetObservation(
                track_id=int(group.track_id),
                observed_monotonic=observed_monotonic,
                centroid=tuple(float(value) for value in group.centroid),  # type: ignore[arg-type]
                bounding_box=tuple(int(value) for value in group.bounding_box),  # type: ignore[arg-type]
                confirmed=bool(getattr(group, "confirmed", False)),
                event_eligible=bool(getattr(group, "event_eligible", False)),
                provisional_category=str(
                    getattr(group, "provisional_category", "unclassified_motion")
                ),
                grouping_confidence=float(getattr(group, "grouping_confidence", 1.0)),
            )
            for group in groups
        )
        return evaluate_reacquisition(
            reference,
            candidates,
            policy=self._target_association_policy,
        )

    def _record_reacquisition_diagnostic(
        self,
        track_id: int,
        coasting: _CoastingAutoTarget,
        groups: tuple[Any, ...] | list[Any],
        observed_monotonic: float,
        decision: str,
        association: AssociationDecision,
        *,
        frame_sequence: int | None = None,
    ) -> None:
        if self._recorder is None:
            return
        previous = coasting.last_snapshot
        previous_area = previous.bounding_box[2] * previous.bounding_box[3]
        candidates: list[dict[str, Any]] = []
        for group, evidence in zip(groups, association.evidence, strict=True):
            centroid = tuple(float(value) for value in group.centroid)
            box = tuple(int(value) for value in group.bounding_box)
            area = box[2] * box[3]
            facts = evidence.as_dict()
            candidates.append(
                {
                    **facts,
                    "track_id": int(group.track_id),
                    "centroid": {"x": round(centroid[0], 2), "y": round(centroid[1], 2)},
                    "bounding_box": {"x": box[0], "y": box[1], "width": box[2], "height": box[3]},
                    "bounding_box_area": area,
                    "distance_from_last_centroid_pixels": round(
                        evidence.last_centroid_distance_pixels,
                        2,
                    ),
                    "area_ratio": round(evidence.area_ratio, 3),
                    "compatible": evidence.selectable,
                    "rejection_reason": None if evidence.selectable else evidence.reason,
                    "persistence_count": int(getattr(group, "persistence_count", 0)),
                    "confirmed": bool(getattr(group, "confirmed", False)),
                    "event_eligible": bool(getattr(group, "event_eligible", False)),
                    "provisional_category": str(getattr(group, "provisional_category", "unknown")),
                    "grouping_confidence": float(getattr(group, "grouping_confidence", 1.0)),
                }
            )
        self._recorder.record_reacquisition_diagnostic(
            track_id,
            {
                "observed_monotonic": observed_monotonic,
                "seconds_since_coasting_started": round(observed_monotonic - coasting.lost_at_monotonic, 3),
                "source_frame_sequence": frame_sequence,
                "reference_mode": "velocity_projected_centroid",
                "reference_centroid": {"x": previous.pixel_x, "y": previous.pixel_y},
                "reference_velocity_pixels_per_second": {
                    "x": previous.velocity[0],
                    "y": previous.velocity[1],
                },
                "reference_frame_sequence": previous.frame_sequence,
                "reference_bounding_box_area": previous_area,
                "association_policy": asdict(self._target_association_policy),
                "association_state": association.state,
                "association_reason": association.reason,
                "selected_track_id": (
                    None if association.selected is None else association.selected.track_id
                ),
                "candidate_count": len(candidates),
                "decision": decision,
                "candidates": candidates,
            },
        )

    def _expire_auto_target(
        self,
        key: tuple[str, int],
        coasting: _CoastingAutoTarget,
        state: str,
        observed_monotonic: float,
    ) -> None:
        association = AutoFireTargetAssociation(state)
        with self._condition:
            self._live_auto_targets.pop(key, None)
            self._coasting_auto_targets.pop(key, None)
            self._expired_auto_targets[key] = association
        LOGGER.info(
            "AUTO_FIRE target reacquisition failed reason=%s event=%s track=%s elapsed=%.3fs",
            state,
            key[0],
            key[1],
            observed_monotonic - coasting.lost_at_monotonic,
            extra={
                "structured_data": {
                    "event": "auto_fire_target_reacquisition_failed",
                    "reason": f"target_{state}",
                    "event_id": key[0],
                    "track_id": key[1],
                    "target_last_seen_monotonic": coasting.last_snapshot.observed_monotonic,
                    "coasting_started_monotonic": coasting.lost_at_monotonic,
                    "failed_at_monotonic": observed_monotonic,
                }
            },
        )
        self.auto_fire.notify_target_state_changed()

    def _submit_live_auto_fire_classification(
        self,
        event: Any,
        group: Any,
        packet: FramePacket,
        observed_monotonic: float,
    ) -> None:
        centroid_x, centroid_y = group.centroid
        width, height = int(group.bounding_box[2]), int(group.bounding_box[3])
        self.classifier.submit(
            event.event_id,
            event.directory,
            1,
            packet.frame,
            tuple(int(value) for value in group.bounding_box),
            selection_method="qualified_live_target",
            selected_motion_bounding_box_area=width * height,
            total_event_frames_considered=1,
            context="auto_fire_live_event",
            track_id=int(group.track_id),
            frame_sequence=getattr(packet, "sequence", None),
            target_pixel=(max(0, round(float(centroid_x))), max(0, round(float(centroid_y)))),
            target_observed_monotonic=observed_monotonic,
            target_provisional_category=str(group.provisional_category),
            target_confirmed=bool(group.confirmed),
            target_event_eligible=bool(group.event_eligible),
        )

    @staticmethod
    def _consider_classifier_frame(
        selector: BestEventFrameSelector,
        group: Any | None,
        frame: np.ndarray | None,
    ) -> None:
        frame_number = selector.total_frames_considered + 1
        motion_area = None
        if group is not None:
            motion_area = getattr(group, "contour_area", None)
            if motion_area is None:
                motion_area = getattr(group, "foreground_pixels", None)
        selector.consider(
            frame_number,
            frame,
            None if group is None else group.bounding_box,
            None if motion_area is None else float(motion_area),
        )

    def _submit_completed_event(
        self,
        record: dict[str, Any],
        *,
        storage_lease_id: str | None = None,
    ) -> bool:
        event_id = str(record.get("event_id", ""))
        selector = self._classifier_selectors.pop(event_id, None)
        clip_offset = self._classifier_clip_offsets.pop(event_id, 0)
        if selector is None or self._night_mode_paused:
            return False
        clip_path = Path(str(record.get("clip_path", "")))
        selected = selector.select(
            lambda frame_number: self._load_event_clip_frame(clip_path, clip_offset + frame_number - 1)
        )
        if selected is None:
            LOGGER.warning(
                "No readable frame was available for event classification: %s",
                event_id,
                extra={"structured_data": {"event": "classifier_frame_unavailable", "event_id": event_id}},
            )
            return False
        if selected.method != "best":
            LOGGER.warning(
                "Classifier frame fallback selected: event_id=%s frame=%d method=%s",
                event_id,
                selected.frame_number,
                selected.method,
                extra={
                    "structured_data": {
                        "event": "classifier_frame_fallback",
                        "event_id": event_id,
                        "selected_event_frame_index": selected.frame_number,
                        "frame_selection_method": selected.method,
                        "total_event_frames_considered": selected.total_frames_considered,
                    }
                },
            )
        event_directory = Path(str(record.get("snapshot_path", ""))).parent
        manager = self._event_storage
        lease_id = storage_lease_id
        if lease_id is None and manager is not None:
            lease_id = manager.acquire_lease(event_directory)
        if lease_id is not None:
            with self._condition:
                prior_lease_id = self._classifier_storage_leases.get(event_id)
                self._classifier_storage_leases[event_id] = lease_id
            if (
                prior_lease_id is not None
                and prior_lease_id != lease_id
                and manager is not None
            ):
                manager.release_lease(prior_lease_id)
        try:
            accepted = self.classifier.submit(
                event_id,
                event_directory,
                selected.frame_number,
                selected.frame,
                selected.bounding_box,
                selection_method=selected.method,
                selected_motion_bounding_box_area=selected.bounding_box_area,
                total_event_frames_considered=selected.total_frames_considered,
            )
        except Exception:
            self._release_classifier_storage_lease(event_id, lease_id)
            raise
        if not accepted:
            self._release_classifier_storage_lease(event_id, lease_id)
        return bool(accepted)

    def _release_classifier_storage_lease(
        self,
        event_id: str,
        expected_lease_id: str | None,
    ) -> None:
        if expected_lease_id is None:
            return
        with self._condition:
            current = self._classifier_storage_leases.get(event_id)
            if current != expected_lease_id:
                return
            lease_id = self._classifier_storage_leases.pop(event_id)
        manager = self._event_storage
        if manager is not None:
            manager.release_lease(lease_id)

    @staticmethod
    def _load_event_clip_frame(clip_path: Path, zero_based_frame_number: int) -> np.ndarray | None:
        if not clip_path.is_file() or zero_based_frame_number < 0:
            return None
        capture = cv2.VideoCapture(str(clip_path))
        try:
            if not capture.isOpened():
                return None
            capture.set(cv2.CAP_PROP_POS_FRAMES, zero_based_frame_number)
            ok, frame = capture.read()
            return frame if ok else None
        finally:
            capture.release()

    def _record_completed_event(
        self,
        record: dict[str, Any],
        *,
        storage_lease_id: str | None = None,
    ) -> None:
        classifier_owns_lease = False
        try:
            classifier_owns_lease = self._submit_completed_event(
                record,
                storage_lease_id=storage_lease_id,
            )
            snapshot_path = record.get("snapshot_path")
            event_directory = (
                Path(snapshot_path).parent
                if isinstance(snapshot_path, str) and snapshot_path
                else None
            )
            classification = (
                None
                if event_directory is None
                else self.classifier_store.reconcile_completed_event(event_directory)
            )
            if classification is not None and event_directory is not None:
                record = {
                    **record,
                    "classification_path": str(event_directory / "classification.json"),
                    "classifier_input_path": classification.get("input_image_path"),
                    "original_frame_path": classification.get("original_frame_path"),
                    "predicted_class": classification.get("top_label"),
                    "prediction_confidence": classification.get("top_confidence"),
                    "classification_status": classification.get("classification_status"),
                    "classification_error": classification.get("error"),
                }
            with self._condition:
                self._recent_events.append(record)
                self._last_event_summary = dict(record)
                self._last_snapshot = record.get("end_timestamp")
                self._snapshots_saved += 1
        finally:
            if storage_lease_id is not None and not classifier_owns_lease:
                manager = self._event_storage
                if manager is not None:
                    manager.release_lease(storage_lease_id)

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
        now = monotonic()
        signature = f"{message}|{type(exc).__name__}|{exc}"
        prior = self._error_throttle.get(signature)
        if prior is not None and now - prior[0] < self.config.runtime.telemetry_interval_seconds:
            self._error_throttle[signature] = (prior[0], prior[1] + 1)
            self._error_throttle.move_to_end(signature)
            self._suppressed_error_count += 1
            return
        persisted_detail = detail
        repeated = 0 if prior is None else prior[1]
        if repeated:
            persisted_detail += f" ({repeated} identical errors suppressed)"
            self._suppressed_error_count = max(0, self._suppressed_error_count - repeated)
        if self._session is not None:
            self._session.add_exception(persisted_detail)
            self._session.save()
        LOGGER.error(
            message,
            extra={
                "structured_data": {
                    "event": "motion_runtime_error",
                    "error": str(exc),
                    "identical_errors_suppressed": repeated,
                }
            },
            exc_info=True,
        )
        self._error_throttle[signature] = (now, 0)
        self._error_throttle.move_to_end(signature)
        while len(self._error_throttle) > 16:
            self._error_throttle.popitem(last=False)

    def _flush_suppressed_errors(self, context: str, *, save_session: bool = True) -> None:
        count = self._suppressed_error_count
        if count <= 0:
            return
        detail = f"{context}: {count} repeated motion errors were suppressed"
        if self._session is not None:
            self._session.add_exception(detail)
            if save_session:
                self._session.save()
        LOGGER.warning(
            detail,
            extra={
                "structured_data": {
                    "event": "motion_runtime_errors_suppressed",
                    "count": count,
                    "context": context,
                }
            },
        )
        self._error_throttle = OrderedDict(
            (signature, (persisted_at, 0))
            for signature, (persisted_at, _) in self._error_throttle.items()
        )
        self._suppressed_error_count = 0

    @staticmethod
    def _failure_backoff_seconds(consecutive_failures: int) -> float:
        return min(5.0, 0.25 * (2 ** min(max(0, consecutive_failures - 1), 5)))

    def _finalize(self, *, clean: bool) -> None:
        with self._condition:
            if self._finalized:
                return
            self._finalized = True
        now = monotonic()
        manager = self._event_storage
        if not self._request_all_event_finalizations(now=now, notes="orderly shutdown"):
            clean = False
        if manager is not None:
            retention_submission = manager.request_retention()
            if not retention_submission.accepted:
                clean = False
                self._event_storage_submission_rejections += 1
                self._note_event_storage_failure(
                    f"Shutdown retention submission rejected: {retention_submission.reason}"
                )
        with self._condition:
            self._active_events = 0
            self._live_auto_targets.clear()
            self._coasting_auto_targets.clear()
            self._expired_auto_targets.clear()
        self.auto_fire.notify_target_state_changed()
        if manager is not None:
            storage_status = manager.stop()
            self._poll_event_storage()
            timed_out = storage_status.thread_alive or bool(storage_status.pending_finalizations)
            storage_failed = bool(
                storage_status.failed_finalizations
                or storage_status.failed_retention_runs
                or storage_status.completion_results_dropped
            )
            if timed_out or storage_failed:
                clean = False
                if timed_out:
                    manager.request_stop_pending_writers()
                    self._note_event_storage_failure(
                        "Event storage did not finish before the bounded shutdown deadline; "
                        "incomplete evidence was preserved"
                    )
                else:
                    self._note_event_storage_failure(
                        "Event storage completed shutdown work with recorded evidence or retention failures"
                    )
        self._flush_suppressed_errors("Motion processor stopped", save_session=False)
        if self._session is not None:
            self._sync_camera_failures()
            if self._session.data.get("camera_open_result") == "not_attempted":
                self._session.data["camera_open_result"] = "failed"
            if self.config.reporting.rebuild_on_clean_shutdown and clean:
                try:
                    generate_reports(self.config)
                except Exception as exc:
                    self._session.add_exception(f"Report generation failed: {exc}")
            self._session.finish(clean=clean)
