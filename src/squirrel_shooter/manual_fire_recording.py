"""Background evidence recording for accepted supervised manual fire events."""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

import cv2
import numpy as np

from .thread_names import set_current_thread_name


LOGGER = logging.getLogger(__name__)
MAX_QUEUED_RECORDINGS = 2


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


def new_manual_fire_event_id(now: datetime | None = None) -> str:
    current = now or datetime.now().astimezone()
    return current.strftime("manual-fire-%Y%m%d-%H%M%S-%f")


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
    """Collect shared-camera frames and encode one manual event off the fire path."""

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
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="manual-fire-recorder")
        self._closed = False
        self._lock = threading.Lock()
        self._queued = 0
        self._active = False
        self._completed = 0
        self._failed = 0
        self._rejected = 0
        self._last_frames_written = 0
        self._last_output_fps = 0.0
        self._last_processing_seconds = 0.0
        self._last_error: str | None = None

    def record(self, event: ManualFireEvent) -> None:
        """Queue one accepted event without doing video work in the caller."""

        if not self.config.enabled:
            return
        with self._lock:
            if self._closed:
                self._record_submission_failure_locked("Manual fire recorder is closed")
                raise RuntimeError(self._last_error)
            if self._queued >= MAX_QUEUED_RECORDINGS:
                self._record_submission_failure_locked("Manual fire recording queue is full")
                raise RuntimeError(self._last_error)
            self._queued += 1
            try:
                self._executor.submit(self._record_safely, event)
            except Exception as exc:
                self._queued -= 1
                self._record_submission_failure_locked(
                    f"Manual fire recording submission failed: {type(exc).__name__}: {exc}"
                )
                raise

    def status(self) -> dict[str, object]:
        """Return lightweight event-recorder activity and output metrics."""

        with self._lock:
            return {
                "enabled": self.config.enabled,
                "active": self._active,
                "queued": self._queued,
                "completed": self._completed,
                "failed": self._failed,
                "rejected": self._rejected,
                "target_fps": self.config.target_fps,
                "last_frames_written": self._last_frames_written,
                "last_output_fps": round(self._last_output_fps, 3),
                "last_processing_seconds": round(self._last_processing_seconds, 3),
                "last_error": self._last_error,
            }

    def _record_submission_failure_locked(self, detail: str) -> None:
        self._failed += 1
        self._rejected += 1
        self._last_error = detail

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)

    def _record_safely(self, event: ManualFireEvent) -> None:
        set_current_thread_name("manual-recorder")
        processing_started = time.perf_counter()
        with self._lock:
            self._queued = max(0, self._queued - 1)
            self._active = True
        directory = self._event_directory(event)
        LOGGER.info(
            "Manual fire event recording started: event_id=%s",
            event.event_id,
            extra={"structured_data": {"event": "manual_fire_recording_started", "event_id": event.event_id}},
        )
        try:
            directory.mkdir(parents=True, exist_ok=False)
            frames_written, output_fps = self._record_event(event, directory)
            with self._lock:
                self._completed += 1
                self._last_frames_written = frames_written
                self._last_output_fps = output_fps
                self._last_error = None
        except Exception as exc:
            with self._lock:
                self._failed += 1
                self._last_error = f"{type(exc).__name__}: {exc}"
            LOGGER.error(
                "Manual fire recording failed: event_id=%s error=%s",
                event.event_id,
                exc,
                extra={"structured_data": {"event": "manual_fire_recording_failed", "event_id": event.event_id, "error": str(exc)}},
                exc_info=True,
            )
            try:
                directory.mkdir(parents=True, exist_ok=True)
                self._write_metadata(
                    directory / "event.json",
                    self._base_metadata(event)
                    | {"status": "recording_failed", "recording_status": "error", "recording_error": str(exc)},
                )
            except Exception:
                LOGGER.exception("Could not persist failed manual fire recording metadata: %s", event.event_id)
        finally:
            with self._lock:
                self._active = False
                self._last_processing_seconds = time.perf_counter() - processing_started

    def _record_event(self, event: ManualFireEvent, directory: Path) -> tuple[int, float]:
        requested_window_seconds = self.config.pre_roll_seconds + self.config.post_roll_seconds
        requested_start = event.fire_started_monotonic - self.config.pre_roll_seconds
        initial_pre_roll = self.camera.buffered_frame_metadata(
            requested_start,
            until_monotonic=event.fire_started_monotonic,
        )
        available_pre_roll_seconds = min(
            self.config.pre_roll_seconds,
            max(
                0.0,
                event.fire_started_monotonic - initial_pre_roll[0].received_monotonic,
            )
            if initial_pre_roll
            else 0.0,
        )
        recording_start = event.fire_started_monotonic - available_pre_roll_seconds
        actual_post_roll_seconds = max(
            self.config.post_roll_seconds,
            requested_window_seconds - available_pre_roll_seconds,
        )
        deadline = event.fire_started_monotonic + actual_post_roll_seconds
        sequence = initial_pre_roll[-1].sequence if initial_pre_roll else -1
        packets_by_sequence = {
            packet.sequence: packet
            for packet in self.camera.buffered_frames(
                recording_start,
                until_monotonic=event.fire_started_monotonic,
                copy=False,
            )
        }
        next_sample_at = event.fire_started_monotonic
        sample_interval = 1.0 / self.config.target_fps
        while self._clock() < deadline:
            remaining = deadline - self._clock()
            packet = self.camera.wait_for_frame(
                sequence,
                timeout=min(1.0, max(0.01, remaining)),
                copy=False,
            )
            if packet is not None:
                sequence = max(sequence, packet.sequence)
                if packet.received_monotonic <= deadline and packet.received_monotonic >= next_sample_at:
                    packets_by_sequence[packet.sequence] = packet
                    while next_sample_at <= packet.received_monotonic:
                        next_sample_at += sample_interval

        # The fallback preserves frames supplied by simpler/test sources and
        # fills any short capture gap that is still inside the rolling buffer.
        for packet in self.camera.buffered_frames(
            recording_start,
            until_monotonic=deadline,
            copy=False,
        ):
            packets_by_sequence.setdefault(packet.sequence, packet)
        packets = sorted(
            packets_by_sequence.values(),
            key=lambda item: (item.received_monotonic, item.sequence),
        )
        if not packets:
            raise OSError("No shared-camera frames were available for the accepted fire")
        first_packet = packets[0]
        sample = first_packet.frame
        frame_height, frame_width = sample.shape[:2]
        requested_x = event.crop_center_x
        requested_y = event.crop_center_y
        if requested_x is None or requested_y is None:
            requested_x, requested_y = frame_width // 2, frame_height // 2
        requested_x = min(max(requested_x, 0), frame_width - 1)
        requested_y = min(max(requested_y, 0), frame_height - 1)
        bounds = calculate_crop_bounds(frame_width, frame_height, requested_x, requested_y, self.config.zoom_factor)
        LOGGER.info(
            "Manual fire recording crop center: x=%d y=%d source=%s zoom=%.2f",
            requested_x,
            requested_y,
            event.crop_center_source,
            self.config.zoom_factor,
        )

        full_path = directory / "manual_fire_full.avi"
        zoom_path = directory / "manual_fire_zoom.avi"
        full_incomplete = directory / "manual_fire_full.incomplete.avi"
        zoom_incomplete = directory / "manual_fire_zoom.incomplete.avi"
        recording_window_seconds = max(0.001, deadline - recording_start)
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
        metadata = self._base_metadata(event) | {
            "status": "complete",
            "recording_status": "success",
            "end_timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "duration": round(recording_window_seconds, 3),
            "captured_frame_span_seconds": round(max(0.0, last_frame_at - first_frame_at), 3),
            "pre_roll_requested_seconds": self.config.pre_roll_seconds,
            "post_roll_requested_seconds": self.config.post_roll_seconds,
            "pre_roll_available": available_pre_roll_seconds > 0,
            "actual_pre_roll_seconds": round(available_pre_roll_seconds, 3),
            "actual_post_roll_seconds": round(actual_post_roll_seconds, 3),
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
            "snapshot_file_role": "manual_fire_zoom_review_frame",
            "clip_path": str(zoom_path),
            "clip_file_role": "manual_fire_zoom_replay",
            "recording_filename": zoom_path.name,
            "recording_path": str(zoom_path),
            "full_frame_filename": full_path.name if self.config.save_full_frame_clip else None,
            "full_frame_clip_path": str(full_path) if self.config.save_full_frame_clip else None,
            "full_frame_file_role": "manual_fire_full_frame_evidence" if self.config.save_full_frame_clip else None,
        }
        self._write_metadata(directory / "event.json", metadata)
        LOGGER.info("Manual fire zoom recording saved: %s", zoom_path)
        if self.config.save_full_frame_clip:
            LOGGER.info("Manual fire recording saved: %s", full_path)
        return frames_written, output_fps

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
        return {
            "schema_version": 1,
            "event_id": event.event_id,
            "event_type": "manual_fire",
            "capture_method": "manual_fire",
            "start_timestamp": event.timestamp,
            "pan": event.pan,
            "tilt": event.tilt,
            "fire_pulse_seconds": event.fire_pulse_seconds,
            "provisional_category": "manual_fire",
        }

    @staticmethod
    def _write_metadata(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
        os.replace(temporary, path)
