"""Background evidence recording for accepted manual and automatic fire events."""

from __future__ import annotations

import json
import logging
import math
import os
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

import cv2
import numpy as np

from .thread_names import set_current_thread_name


LOGGER = logging.getLogger(__name__)
MAX_QUEUED_RECORDINGS = 2
_QUEUE_STOP = object()


@dataclass(frozen=True)
class ManualFireRecordingConfig:
    """Small, conservative recording configuration for manual shots."""

    enabled: bool = True
    pre_roll_seconds: float = 2.0
    post_roll_seconds: float = 5.0
    target_fps: float = 12.0
    zoom_factor: float = 2.0
    crop_center_x: int | None = 640
    crop_center_y: int | None = 360
    save_full_frame_clip: bool = True
    clip_codec: str = "MJPG"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be true or false")
        if not isinstance(self.save_full_frame_clip, bool):
            raise ValueError("save_full_frame_clip must be true or false")
        for name, value, allow_zero in (
            ("pre_roll_seconds", self.pre_roll_seconds, True),
            ("post_roll_seconds", self.post_roll_seconds, False),
            ("target_fps", self.target_fps, False),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
                or (not allow_zero and value == 0)
            ):
                qualifier = "zero or greater" if allow_zero else "greater than zero"
                raise ValueError(f"{name} must be a finite number {qualifier}")
        if (
            isinstance(self.zoom_factor, bool)
            or not isinstance(self.zoom_factor, (int, float))
            or not math.isfinite(float(self.zoom_factor))
            or self.zoom_factor <= 1.0
        ):
            raise ValueError("zoom_factor must be a finite number greater than 1")
        for name, value in (("crop_center_x", self.crop_center_x), ("crop_center_y", self.crop_center_y)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be null or a non-negative integer")
        if (self.crop_center_x is None) != (self.crop_center_y is None):
            raise ValueError("crop_center_x and crop_center_y must either both be set or both be null")
        if not isinstance(self.clip_codec, str) or len(self.clip_codec) != 4:
            raise ValueError("clip_codec must contain four characters")


@dataclass(frozen=True)
class ManualFireEvent:
    """Physical-control facts captured before the recorder is allowed to run."""

    event_id: str
    timestamp: str
    fire_started_monotonic: float
    fire_completed_monotonic: float
    pan: float
    tilt: float
    fire_pulse_seconds: float
    crop_center_x: int | None
    crop_center_y: int | None
    crop_center_source: str
    event_type: str = "manual_fire"
    evidence: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.event_type not in {"manual_fire", "auto_fire"}:
            raise ValueError("event_type must be manual_fire or auto_fire")
        if not isinstance(self.evidence, dict):
            raise ValueError("evidence must be a dictionary")


@dataclass(frozen=True)
class CropBounds:
    x: int
    y: int
    width: int
    height: int


class ManualFireFrameSource(Protocol):
    def buffered_frame_metadata(
        self,
        since_monotonic: float,
        *,
        until_monotonic: float | None = None,
        after_sequence: int = -1,
    ) -> list[Any]:
        ...

    def buffered_frames(
        self,
        since_monotonic: float,
        *,
        until_monotonic: float | None = None,
        after_sequence: int = -1,
        copy: bool = True,
    ) -> Iterable[Any]:
        ...

    def wait_for_frame(
        self,
        after_sequence: int,
        timeout: float | None = None,
        *,
        copy: bool = True,
    ) -> Any | None:
        ...


class ManualFireRecordingSink(Protocol):
    def record(self, event: ManualFireEvent) -> None:
        ...

    def close(self) -> None:
        ...

    def status(self) -> dict[str, object]:
        ...


class LegacyFireRecordingAdapter:
    """Phase 2 boundary preserving the accepted-shot recorder's exact contract.

    FIRE retains its existing pre/post reservation, shot identity, clean full
    frame and aim derivative behavior. It does not create a manual recording
    reason or acquire another camera. Migrate this sink only with FIRE parity.
    """

    def __init__(self, sink: ManualFireRecordingSink) -> None:
        self._sink = sink

    def record(self, event: ManualFireEvent) -> None:
        self._sink.record(event)

    def close(self) -> None:
        self._sink.close()

    def status(self) -> dict[str, object]:
        return self._sink.status()


@dataclass
class _RecordingReservation:
    event: ManualFireEvent
    requested_start_monotonic: float
    recording_start_monotonic: float
    deadline_monotonic: float
    available_pre_roll_seconds: float
    post_roll_target_seconds: float
    max_packets: int
    packets_by_sequence: dict[int, Any]
    last_sequence: int
    collection_error: str | None = None


def new_manual_fire_event_id(now: datetime | None = None) -> str:
    current = now or datetime.now().astimezone()
    return current.strftime("manual-fire-%Y%m%d-%H%M%S-%f")


def new_auto_fire_event_id(now: datetime | None = None) -> str:
    current = now or datetime.now().astimezone()
    return current.strftime("auto-fire-%Y%m%d-%H%M%S-%f")


def calculate_crop_bounds(
    frame_width: int,
    frame_height: int,
    center_x: int,
    center_y: int,
    zoom_factor: float,
) -> CropBounds:
    """Return an in-frame crop with exactly the source aspect ratio."""

    for name, value in (("frame_width", frame_width), ("frame_height", frame_height)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (("center_x", center_x), ("center_y", center_y)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    if (
        isinstance(zoom_factor, bool)
        or not isinstance(zoom_factor, (int, float))
        or not math.isfinite(float(zoom_factor))
        or zoom_factor <= 1.0
    ):
        raise ValueError("zoom_factor must be a finite number greater than 1")

    divisor = math.gcd(frame_width, frame_height)
    aspect_width = frame_width // divisor
    aspect_height = frame_height // divisor
    units = max(
        1,
        int(
            min(
                frame_width / (float(zoom_factor) * aspect_width),
                frame_height / (float(zoom_factor) * aspect_height),
            )
        ),
    )
    crop_width = min(frame_width, aspect_width * units)
    crop_height = min(frame_height, aspect_height * units)
    clamped_x = min(max(center_x, 0), frame_width - 1)
    clamped_y = min(max(center_y, 0), frame_height - 1)
    left = min(max(round(clamped_x - crop_width / 2), 0), frame_width - crop_width)
    top = min(max(round(clamped_y - crop_height / 2), 0), frame_height - crop_height)
    return CropBounds(left, top, crop_width, crop_height)


def crop_and_zoom(frame: np.ndarray, bounds: CropBounds) -> np.ndarray:
    """Crop one frame and restore the original playback dimensions."""

    if frame.ndim < 2:
        raise ValueError("frame must contain image dimensions")
    frame_height, frame_width = frame.shape[:2]
    if (
        bounds.x < 0
        or bounds.y < 0
        or bounds.width <= 0
        or bounds.height <= 0
        or bounds.x + bounds.width > frame_width
        or bounds.y + bounds.height > frame_height
    ):
        raise ValueError("crop bounds must fit inside the source frame")
    cropped = frame[bounds.y : bounds.y + bounds.height, bounds.x : bounds.x + bounds.width]
    return cv2.resize(cropped, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)


class ManualFireRecorder:
    """Reserve shared-camera evidence immediately and encode it off the fire path."""

    def __init__(
        self,
        camera: ManualFireFrameSource,
        output_directory: Path,
        config: ManualFireRecordingConfig,
        *,
        video_writer_factory: Callable[..., Any] = cv2.VideoWriter,
        image_writer: Callable[[str, np.ndarray], bool] = cv2.imwrite,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.camera = camera
        self.output_directory = output_directory
        self.config = config
        self._video_writer_factory = video_writer_factory
        self._image_writer = image_writer
        self._clock = clock
        self._closed = False
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._collector_queue: queue.Queue[object] = queue.Queue(maxsize=MAX_QUEUED_RECORDINGS)
        self._encoder_queue: queue.Queue[object] = queue.Queue(maxsize=MAX_QUEUED_RECORDINGS)
        self._outstanding = 0
        self._submissions_in_progress = 0
        self._collecting = 0
        self._encoding = False
        self._completed = 0
        self._truncated = 0
        self._failed = 0
        self._rejected = 0
        self._last_frames_written = 0
        self._last_output_fps = 0.0
        self._last_processing_seconds = 0.0
        self._last_recording_status: str | None = None
        self._last_pre_roll_gap_seconds = 0.0
        self._last_post_roll_gap_seconds = 0.0
        self._last_error: str | None = None
        self._collector_threads = tuple(
            threading.Thread(
                target=self._collector_worker,
                name=f"manual-collector-{index + 1}",
                daemon=True,
            )
            for index in range(MAX_QUEUED_RECORDINGS)
        )
        self._encoder_thread = threading.Thread(
            target=self._encoder_worker,
            name="manual-recorder",
            daemon=True,
        )
        for thread in self._collector_threads:
            thread.start()
        self._encoder_thread.start()

    def record(self, event: ManualFireEvent) -> None:
        """Reserve one accepted event immediately without encoding in the caller."""

        if not self.config.enabled:
            return
        with self._condition:
            if self._closed:
                self._record_submission_failure_locked("Manual fire recorder is closed")
                raise RuntimeError(self._last_error)
            if self._outstanding >= MAX_QUEUED_RECORDINGS:
                self._record_submission_failure_locked("Manual fire recording queue is full")
                raise RuntimeError(self._last_error)
            self._outstanding += 1
            self._submissions_in_progress += 1
        try:
            reservation = self._reserve_recording(event)
            self._collector_queue.put_nowait(reservation)
        except Exception as exc:
            with self._condition:
                self._outstanding = max(0, self._outstanding - 1)
                self._record_submission_failure_locked(
                    "Manual fire evidence reservation failed: "
                    f"{type(exc).__name__}: {exc}"
                )
            raise RuntimeError(self._last_error) from exc
        finally:
            with self._condition:
                self._submissions_in_progress = max(0, self._submissions_in_progress - 1)
                self._condition.notify_all()

    def status(self) -> dict[str, object]:
        """Return lightweight event-recorder activity and output metrics."""

        with self._lock:
            active = self._collecting > 0 or self._encoding
            return {
                "enabled": self.config.enabled,
                "active": active,
                "queued": self._outstanding,
                "outstanding": self._outstanding,
                "collecting": self._collecting,
                "encoding": self._encoding,
                "collector_queue_depth": self._collector_queue.qsize(),
                "collector_queue_capacity": MAX_QUEUED_RECORDINGS,
                "encoder_queue_depth": self._encoder_queue.qsize(),
                "encoder_queue_capacity": MAX_QUEUED_RECORDINGS,
                "completed": self._completed,
                "truncated": self._truncated,
                "failed": self._failed,
                "rejected": self._rejected,
                "target_fps": self.config.target_fps,
                "last_frames_written": self._last_frames_written,
                "last_output_fps": round(self._last_output_fps, 3),
                "last_processing_seconds": round(self._last_processing_seconds, 3),
                "last_recording_status": self._last_recording_status,
                "last_pre_roll_gap_seconds": round(self._last_pre_roll_gap_seconds, 3),
                "last_post_roll_gap_seconds": round(self._last_post_roll_gap_seconds, 3),
                "last_error": self._last_error,
            }

    def _record_submission_failure_locked(self, detail: str) -> None:
        self._failed += 1
        self._rejected += 1
        self._last_error = detail

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.wait_for(lambda: self._submissions_in_progress == 0)
        self._collector_queue.join()
        for _ in self._collector_threads:
            self._collector_queue.put(_QUEUE_STOP)
        for thread in self._collector_threads:
            thread.join()
        self._encoder_queue.join()
        self._encoder_queue.put(_QUEUE_STOP)
        self._encoder_thread.join()

    def _reserve_recording(self, event: ManualFireEvent) -> _RecordingReservation:
        requested_window_seconds = self.config.pre_roll_seconds + self.config.post_roll_seconds
        requested_start = event.fire_started_monotonic - self.config.pre_roll_seconds
        pre_roll_candidates = self._valid_packets(
            self.camera.buffered_frames(
                requested_start,
                until_monotonic=event.fire_started_monotonic,
                copy=False,
            )
        )
        available_pre_roll_seconds = min(
            self.config.pre_roll_seconds,
            max(0.0, event.fire_started_monotonic - pre_roll_candidates[0].received_monotonic)
            if pre_roll_candidates
            else 0.0,
        )
        recording_start = event.fire_started_monotonic - available_pre_roll_seconds
        post_roll_target_seconds = max(
            self.config.post_roll_seconds,
            requested_window_seconds - available_pre_roll_seconds,
        )
        deadline = event.fire_started_monotonic + post_roll_target_seconds
        capture_until = min(deadline, max(event.fire_started_monotonic, self._clock()))
        initial_candidates = self._valid_packets(
            self.camera.buffered_frames(
                recording_start,
                until_monotonic=capture_until,
                copy=False,
            )
        )
        candidates = self._valid_packets((*pre_roll_candidates, *initial_candidates))
        maximum_packets = max(
            2,
            math.ceil(max(0.001, deadline - recording_start) * self.config.target_fps) + 2,
        )
        selected = self._select_packets(
            candidates,
            start_monotonic=recording_start,
            end_monotonic=deadline,
            maximum_packets=maximum_packets,
        )
        last_sequence = max((packet.sequence for packet in candidates), default=-1)
        return _RecordingReservation(
            event=event,
            requested_start_monotonic=requested_start,
            recording_start_monotonic=recording_start,
            deadline_monotonic=deadline,
            available_pre_roll_seconds=available_pre_roll_seconds,
            post_roll_target_seconds=post_roll_target_seconds,
            max_packets=maximum_packets,
            packets_by_sequence={packet.sequence: packet for packet in selected},
            last_sequence=last_sequence,
        )

    def _collector_worker(self) -> None:
        set_current_thread_name("manual-collector")
        while True:
            item = self._collector_queue.get()
            try:
                if item is _QUEUE_STOP:
                    return
                reservation = item
                if not isinstance(reservation, _RecordingReservation):
                    continue
                with self._lock:
                    self._collecting += 1
                try:
                    self._collect_post_roll(reservation)
                except Exception as exc:
                    reservation.collection_error = f"{type(exc).__name__}: {exc}"
                    LOGGER.exception(
                        "Manual fire post-roll collection failed: event_id=%s",
                        reservation.event.event_id,
                    )
                finally:
                    with self._lock:
                        self._collecting = max(0, self._collecting - 1)
                try:
                    self._encoder_queue.put_nowait(reservation)
                except queue.Full:
                    self._persist_failed_reservation(
                        reservation,
                        "Manual fire encoder queue is full",
                    )
                    with self._lock:
                        self._outstanding = max(0, self._outstanding - 1)
            finally:
                self._collector_queue.task_done()

    def _collect_post_roll(self, reservation: _RecordingReservation) -> None:
        while self._clock() < reservation.deadline_monotonic:
            remaining = reservation.deadline_monotonic - self._clock()
            try:
                packet = self.camera.wait_for_frame(
                    reservation.last_sequence,
                    timeout=min(1.0, max(0.01, remaining)),
                    copy=False,
                )
            except Exception as exc:
                reservation.collection_error = f"{type(exc).__name__}: {exc}"
                break
            if packet is None:
                continue
            valid = self._valid_packets((packet,))
            if not valid:
                continue
            reservation.last_sequence = max(reservation.last_sequence, valid[0].sequence)
            combined = (*reservation.packets_by_sequence.values(), valid[0])
            reservation.packets_by_sequence = {
                item.sequence: item
                for item in self._select_packets(
                    combined,
                    start_monotonic=reservation.recording_start_monotonic,
                    end_monotonic=reservation.deadline_monotonic,
                    maximum_packets=reservation.max_packets,
                )
            }

        try:
            fallback = self._valid_packets(
                self.camera.buffered_frames(
                    reservation.recording_start_monotonic,
                    until_monotonic=reservation.deadline_monotonic,
                    copy=False,
                )
            )
        except Exception as exc:
            if reservation.collection_error is None:
                reservation.collection_error = f"{type(exc).__name__}: {exc}"
            fallback = []
        combined = (*reservation.packets_by_sequence.values(), *fallback)
        reservation.packets_by_sequence = {
            item.sequence: item
            for item in self._select_packets(
                combined,
                start_monotonic=reservation.recording_start_monotonic,
                end_monotonic=reservation.deadline_monotonic,
                maximum_packets=reservation.max_packets,
            )
        }

    def _encoder_worker(self) -> None:
        set_current_thread_name("manual-recorder")
        while True:
            item = self._encoder_queue.get()
            try:
                if item is _QUEUE_STOP:
                    return
                reservation = item
                if not isinstance(reservation, _RecordingReservation):
                    continue
                with self._lock:
                    self._encoding = True
                try:
                    self._record_safely(reservation)
                finally:
                    with self._lock:
                        self._encoding = False
                        self._outstanding = max(0, self._outstanding - 1)
            finally:
                self._encoder_queue.task_done()

    def _record_safely(self, reservation: _RecordingReservation) -> None:
        processing_started = time.perf_counter()
        event = reservation.event
        directory = self._event_directory(event)
        LOGGER.info(
            "%s event recording started: event_id=%s",
            event.event_type.replace("_", " ").title(),
            event.event_id,
            extra={
                "structured_data": {
                    "event": f"{event.event_type}_recording_started",
                    "event_id": event.event_id,
                }
            },
        )
        try:
            directory.mkdir(parents=True, exist_ok=False)
            frames_written, output_fps, recording_status, coverage = self._record_event(
                reservation,
                directory,
            )
            with self._lock:
                self._completed += 1
                if recording_status == "truncated":
                    self._truncated += 1
                self._last_frames_written = frames_written
                self._last_output_fps = output_fps
                self._last_recording_status = recording_status
                self._last_pre_roll_gap_seconds = float(coverage["pre_roll_gap_seconds"])
                self._last_post_roll_gap_seconds = float(coverage["post_roll_gap_seconds"])
                self._last_error = (
                    None
                    if recording_status == "success"
                    else "Recording window was truncated: "
                    + ", ".join(str(item) for item in coverage["truncation_reasons"])
                )
        except Exception as exc:
            with self._lock:
                self._failed += 1
                self._last_recording_status = "error"
                self._last_error = f"{type(exc).__name__}: {exc}"
            LOGGER.error(
                "%s recording failed: event_id=%s error=%s",
                event.event_type.replace("_", " ").title(),
                event.event_id,
                exc,
                extra={
                    "structured_data": {
                        "event": f"{event.event_type}_recording_failed",
                        "event_id": event.event_id,
                        "error": str(exc),
                    }
                },
                exc_info=True,
            )
            try:
                directory.mkdir(parents=True, exist_ok=True)
                self._write_metadata(
                    directory / "event.json",
                    self._base_metadata(event)
                    | self._coverage_metadata(
                        reservation,
                        tuple(reservation.packets_by_sequence.values()),
                    )
                    | {
                        "status": "recording_failed",
                        "recording_status": "error",
                        "recording_error": str(exc),
                    },
                )
            except Exception:
                LOGGER.exception("Could not persist failed manual fire recording metadata: %s", event.event_id)
        finally:
            with self._lock:
                self._last_processing_seconds = time.perf_counter() - processing_started

    def _record_event(
        self,
        reservation: _RecordingReservation,
        directory: Path,
    ) -> tuple[int, float, str, dict[str, Any]]:
        event = reservation.event
        packets = sorted(
            reservation.packets_by_sequence.values(),
            key=lambda item: (item.received_monotonic, item.sequence),
        )
        if not packets:
            raise OSError("No shared-camera frames were available for the accepted fire")
        usable_sample = next(
            (
                packet.frame
                for packet in packets
                if isinstance(packet.frame, np.ndarray) and packet.frame.ndim >= 2
            ),
            None,
        )
        if usable_sample is None:
            raise OSError("No consistent shared-camera frames were available for the accepted fire")
        frame_height, frame_width = usable_sample.shape[:2]
        consistent_packets = [
            packet
            for packet in packets
            if isinstance(packet.frame, np.ndarray)
            and packet.frame.ndim >= 2
            and packet.frame.shape[:2] == (frame_height, frame_width)
        ]
        if not consistent_packets:
            raise OSError("No consistent shared-camera frames were available for the accepted fire")
        inconsistent_frames = len(packets) - len(consistent_packets)
        packets = consistent_packets
        requested_x = event.crop_center_x
        requested_y = event.crop_center_y
        if requested_x is None or requested_y is None:
            requested_x, requested_y = frame_width // 2, frame_height // 2
        requested_x = min(max(requested_x, 0), frame_width - 1)
        requested_y = min(max(requested_y, 0), frame_height - 1)
        bounds = calculate_crop_bounds(frame_width, frame_height, requested_x, requested_y, self.config.zoom_factor)
        LOGGER.info(
            "%s recording crop center: x=%d y=%d source=%s zoom=%.2f",
            event.event_type.replace("_", " ").title(),
            requested_x,
            requested_y,
            event.crop_center_source,
            self.config.zoom_factor,
        )

        filename_prefix = event.event_type
        full_path = directory / f"{filename_prefix}_full.avi"
        zoom_path = directory / f"{filename_prefix}_zoom.avi"
        full_incomplete = directory / f"{filename_prefix}_full.incomplete.avi"
        zoom_incomplete = directory / f"{filename_prefix}_zoom.incomplete.avi"
        recording_window_seconds = max(
            0.001,
            reservation.deadline_monotonic - reservation.recording_start_monotonic,
        )
        output_fps = self._output_fps(len(packets), recording_window_seconds)
        fourcc = cv2.VideoWriter_fourcc(*self.config.clip_codec)
        full_writer = None
        zoom_writer = None
        seen_sequences: set[int] = set()
        first_frame_at: float | None = None
        last_frame_at: float | None = None
        frames_written = 0
        snapshot_frame: np.ndarray | None = None
        snapshot_distance = math.inf

        def write_packet(packet: Any) -> None:
            nonlocal first_frame_at, last_frame_at, frames_written
            nonlocal snapshot_frame, snapshot_distance
            if packet.sequence in seen_sequences:
                return
            seen_sequences.add(packet.sequence)
            if packet.frame.shape[:2] != (frame_height, frame_width):
                return
            if full_writer is not None:
                full_writer.write(packet.frame)
            zoom_writer.write(crop_and_zoom(packet.frame, bounds))
            first_frame_at = packet.received_monotonic if first_frame_at is None else first_frame_at
            last_frame_at = packet.received_monotonic
            frames_written += 1
            distance = abs(packet.received_monotonic - event.fire_started_monotonic)
            if distance < snapshot_distance:
                snapshot_distance = distance
                snapshot_frame = packet.frame.copy()

        try:
            if self.config.save_full_frame_clip:
                full_writer = self._open_writer(full_incomplete, fourcc, output_fps, (frame_width, frame_height))
            zoom_writer = self._open_writer(zoom_incomplete, fourcc, output_fps, (frame_width, frame_height))
            for packet in packets:
                write_packet(packet)
        finally:
            if full_writer is not None:
                full_writer.release()
            if zoom_writer is not None:
                zoom_writer.release()
        if self.config.save_full_frame_clip:
            os.replace(full_incomplete, full_path)
        os.replace(zoom_incomplete, zoom_path)

        if snapshot_frame is None or first_frame_at is None or last_frame_at is None or frames_written == 0:
            raise OSError("No consistent shared-camera frames were available for the accepted fire")
        snapshot_path = directory / "snapshot.jpg"
        if not self._image_writer(str(snapshot_path), crop_and_zoom(snapshot_frame, bounds)):
            raise OSError(f"Could not write {snapshot_path}")
        role_prefix = event.event_type
        coverage = self._coverage_metadata(
            reservation,
            packets,
            extra_reasons=("inconsistent_frame_geometry",) if inconsistent_frames else (),
        )
        recording_status = "success" if not coverage["truncation_reasons"] else "truncated"
        metadata = self._base_metadata(event) | {
            "status": "complete" if recording_status == "success" else "recording_truncated",
            "recording_status": recording_status,
            "end_timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "duration": round(recording_window_seconds, 3),
            "captured_frame_span_seconds": round(max(0.0, last_frame_at - first_frame_at), 3),
            "timing_basis": "monotonic_elapsed_time",
            "frames_written": frames_written,
            "output_fps": round(output_fps, 3),
            "source_width": frame_width,
            "source_height": frame_height,
            "output_width": frame_width,
            "output_height": frame_height,
            "crop_center_pixel": {"x": requested_x, "y": requested_y, "source": event.crop_center_source},
            "crop_bounds": asdict(bounds),
            "zoom_factor": self.config.zoom_factor,
            "clip_codec": self.config.clip_codec,
            "file_format": "AVI",
            "snapshot_path": str(snapshot_path),
            "snapshot_file_role": f"{role_prefix}_zoom_review_frame",
            "clip_path": str(zoom_path),
            "clip_file_role": f"{role_prefix}_zoom_replay",
            "recording_filename": zoom_path.name,
            "recording_path": str(zoom_path),
            "full_frame_filename": full_path.name if self.config.save_full_frame_clip else None,
            "full_frame_clip_path": str(full_path) if self.config.save_full_frame_clip else None,
            "full_frame_file_role": f"{role_prefix}_full_frame_evidence" if self.config.save_full_frame_clip else None,
        } | coverage
        self._write_metadata(directory / "event.json", metadata)
        LOGGER.info("%s zoom recording saved: %s", event.event_type.replace("_", " ").title(), zoom_path)
        if self.config.save_full_frame_clip:
            LOGGER.info("%s recording saved: %s", event.event_type.replace("_", " ").title(), full_path)
        return frames_written, output_fps, recording_status, coverage

    def _persist_failed_reservation(self, reservation: _RecordingReservation, detail: str) -> None:
        directory = self._event_directory(reservation.event)
        try:
            directory.mkdir(parents=True, exist_ok=True)
            self._write_metadata(
                directory / "event.json",
                self._base_metadata(reservation.event)
                | self._coverage_metadata(reservation, tuple(reservation.packets_by_sequence.values()))
                | {
                    "status": "recording_failed",
                    "recording_status": "error",
                    "recording_error": detail,
                },
            )
        except Exception:
            LOGGER.exception("Could not persist failed manual fire recording metadata: %s", reservation.event.event_id)
        with self._lock:
            self._failed += 1
            self._last_recording_status = "error"
            self._last_error = detail

    def _coverage_metadata(
        self,
        reservation: _RecordingReservation,
        packets: Iterable[Any],
        *,
        extra_reasons: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        event = reservation.event
        ordered = sorted(
            self._valid_packets(packets),
            key=lambda item: (item.received_monotonic, item.sequence),
        )
        pre_roll_packets = [
            packet for packet in ordered if packet.received_monotonic <= event.fire_started_monotonic
        ]
        post_roll_packets = [
            packet for packet in ordered if packet.received_monotonic >= event.fire_started_monotonic
        ]
        actual_pre_roll_seconds = min(
            self.config.pre_roll_seconds,
            max(0.0, event.fire_started_monotonic - pre_roll_packets[0].received_monotonic)
            if pre_roll_packets
            else 0.0,
        )
        actual_post_roll_seconds = min(
            reservation.post_roll_target_seconds,
            max(0.0, post_roll_packets[-1].received_monotonic - event.fire_started_monotonic)
            if post_roll_packets
            else 0.0,
        )
        pre_roll_gap_seconds = max(0.0, self.config.pre_roll_seconds - actual_pre_roll_seconds)
        post_roll_gap_seconds = max(
            0.0,
            reservation.post_roll_target_seconds - actual_post_roll_seconds,
        )
        tolerance = max(0.05, 1.5 / self.config.target_fps)
        captured_span = (
            max(0.0, ordered[-1].received_monotonic - ordered[0].received_monotonic)
            if ordered
            else 0.0
        )
        reasons = list(extra_reasons)
        if pre_roll_gap_seconds > tolerance:
            reasons.append("missing_pre_roll")
        if post_roll_gap_seconds > tolerance:
            reasons.append("missing_post_roll")
        if len(ordered) < 2 or captured_span <= 0.0:
            reasons.append("insufficient_frame_span")
        if reservation.collection_error is not None:
            reasons.append("collection_error")
        reasons = list(dict.fromkeys(reasons))
        return {
            "evidence_window_reserved_at_submission": True,
            "pre_roll_requested_seconds": self.config.pre_roll_seconds,
            "post_roll_requested_seconds": self.config.post_roll_seconds,
            "post_roll_target_seconds": round(reservation.post_roll_target_seconds, 3),
            "pre_roll_available": actual_pre_roll_seconds > 0.0,
            "actual_pre_roll_seconds": round(actual_pre_roll_seconds, 3),
            "actual_post_roll_seconds": round(actual_post_roll_seconds, 3),
            "pre_roll_gap_seconds": round(pre_roll_gap_seconds, 3),
            "post_roll_gap_seconds": round(post_roll_gap_seconds, 3),
            "reserved_frame_count": len(ordered),
            "truncation_reasons": reasons,
            "collection_error": reservation.collection_error,
        }

    def _valid_packets(self, packets: Iterable[Any]) -> list[Any]:
        valid: dict[int, Any] = {}
        for packet in packets:
            try:
                sequence = packet.sequence
                received = float(packet.received_monotonic)
            except (AttributeError, TypeError, ValueError):
                continue
            if (
                isinstance(sequence, bool)
                or not isinstance(sequence, int)
                or not math.isfinite(received)
                or received < 0.0
            ):
                continue
            valid[sequence] = packet
        return sorted(valid.values(), key=lambda item: (item.received_monotonic, item.sequence))

    def _select_packets(
        self,
        packets: Iterable[Any],
        *,
        start_monotonic: float,
        end_monotonic: float,
        maximum_packets: int,
    ) -> list[Any]:
        candidates = [
            packet
            for packet in self._valid_packets(packets)
            if start_monotonic <= packet.received_monotonic <= end_monotonic
        ]
        if not candidates:
            return []
        interval = 1.0 / self.config.target_fps
        selected: list[Any] = []
        next_sample_at = start_monotonic
        for packet in candidates:
            if len(selected) >= maximum_packets:
                break
            if not selected or packet.received_monotonic + 1e-9 >= next_sample_at:
                selected.append(packet)
                next_sample_at = packet.received_monotonic + interval
        latest = candidates[-1]
        if latest.sequence not in {packet.sequence for packet in selected} and len(selected) < maximum_packets:
            selected.append(latest)
        return sorted(selected, key=lambda item: (item.received_monotonic, item.sequence))

    @staticmethod
    def _output_fps(frame_count: int, recording_window_seconds: float) -> float:
        if frame_count <= 0 or recording_window_seconds <= 0:
            raise ValueError("A positive frame count and recording window are required")
        return min(120.0, max(0.01, frame_count / recording_window_seconds))

    def _open_writer(self, path: Path, fourcc: int, fps: float, size: tuple[int, int]) -> Any:
        writer = self._video_writer_factory(str(path), fourcc, fps, size)
        if hasattr(writer, "isOpened") and not writer.isOpened():
            writer.release()
            raise OSError(f"OpenCV could not open manual fire clip {path}")
        return writer

    def _event_directory(self, event: ManualFireEvent) -> Path:
        day = event.timestamp[:10]
        return self.output_directory / "events" / day / event.event_id

    @staticmethod
    def _base_metadata(event: ManualFireEvent) -> dict[str, Any]:
        evidence = dict(event.evidence)
        source_event_id = evidence.pop("event_id", None)
        if source_event_id is not None:
            evidence["source_event_id"] = source_event_id
        core = {
            "schema_version": 1,
            "event_id": event.event_id,
            "event_type": event.event_type,
            "capture_method": event.event_type,
            "start_timestamp": event.timestamp,
            "pan": event.pan,
            "tilt": event.tilt,
            "fire_pulse_seconds": event.fire_pulse_seconds,
            "provisional_category": event.event_type,
        }
        # Hardware-authoritative facts and the recorder's own identity always
        # win over caller-supplied evidence. A source motion ``event_id`` is
        # retained under the unambiguous ``source_event_id`` key instead.
        return evidence | core

    @staticmethod
    def _write_metadata(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
        os.replace(temporary, path)
