"""Crash-aware event recording, append-only logs, sessions, and retention."""

from __future__ import annotations

import csv
import json
import math
import os
import queue
import secrets
import shutil
import subprocess
import threading
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta
from itertools import chain
from pathlib import Path
from time import monotonic, perf_counter
from typing import Any, Callable, Iterable

import cv2
import numpy as np

from .config import AppConfig, RetentionConfig
from .performance import TimingDistribution
from .watch_detection import GroupedCandidate


EVENT_FIELDS = (
    "event_id", "track_id", "session_id", "capture_method", "status", "recording_status",
    "recording_error", "start_timestamp", "end_timestamp", "duration", "frames_written", "frames_dropped",
    "snapshot_path", "snapshot_file_role", "clip_path", "clip_file_role",
    "provisional_category", "movement_attributes", "heuristic_score", "minimum_area", "maximum_area",
    "average_area", "foreground_pixel_coverage", "frame_coverage", "inclusion_zone_coverage",
    "maximum_width", "maximum_height", "starting_centroid", "ending_centroid", "total_centroid_travel",
    "average_pixel_speed", "peak_pixel_speed", "inclusion_zone_status", "component_count",
    "grouping_confidence", "source_camera", "camera_device_index", "measured_camera_fps",
    "camera_reported_fps", "requested_width", "requested_height", "requested_fps",
    "actual_width", "actual_height", "camera_mode_if_known", "ir_mode_if_explicitly_detected_or_configured",
    "low_fps_observed", "software_version", "git_commit_sha", "notes", "human_review_label",
    "human_review_notes",
)

SESSION_DETAIL_LIMIT = 100
REACQUISITION_DIAGNOSTIC_LIMIT = 12
EVENT_CLIP_QUEUE_LIMIT = 8
EVENT_CLIP_STOP_TIMEOUT_SECONDS = 2.0
EVENT_CLIP_NO_PROGRESS_SECONDS = 2.0
EVENT_STORAGE_QUEUE_LIMIT = 8
EVENT_STORAGE_COMPLETION_LIMIT = 32
EVENT_GROUP_SAMPLE_LIMIT = 120
EVENT_COMPONENT_SAMPLE_LIMIT = 12


@dataclass(frozen=True)
class ClipWriteResult:
    """Outcome of one bounded producer attempt."""

    accepted: bool
    degraded: bool
    reason: str


@dataclass(frozen=True)
class ClipWriterStatus:
    """Low-cost writer progress and backpressure telemetry."""

    state: str
    thread_alive: bool
    stop_requested: bool
    queue_depth: int
    queue_capacity: int
    queue_high_water: int
    frames_enqueued: int
    frames_written: int
    frames_dropped: int
    write_in_progress: bool
    last_progress_age_seconds: float
    error: str | None
    join_timed_out: bool
    output_fps: float | None
    first_supplied_monotonic: float | None
    last_supplied_monotonic: float | None
    supplied_duration_seconds: float
    encoded_duration_seconds: float | None
    timestamp_regressions: int
    blocking_put_attempts: int
    blocked_put_seconds: float
    nonblocking_enqueue_timing: dict[str, object]
    write_timing: dict[str, object]


@dataclass(frozen=True)
class _ClipFrame:
    timestamp: float | None
    frame: np.ndarray


class _AsyncClipWriter:
    """Serialize clip frames without blocking the motion detector on encoding."""

    def __init__(
        self,
        writer: Any,
        initial_frames: Iterable[np.ndarray | tuple[float, np.ndarray]],
        *,
        output_fps: float | None = None,
        queue_limit: int = EVENT_CLIP_QUEUE_LIMIT,
        no_progress_seconds: float = EVENT_CLIP_NO_PROGRESS_SECONDS,
    ) -> None:
        if queue_limit <= 0:
            raise ValueError("queue_limit must be greater than zero")
        if no_progress_seconds <= 0:
            raise ValueError("no_progress_seconds must be greater than zero")
        if output_fps is not None and (not math.isfinite(output_fps) or output_fps <= 0):
            raise ValueError("output_fps must be a finite positive number")
        self._writer = writer
        self._initial_frames = initial_frames
        self._queue: queue.Queue[_ClipFrame] = queue.Queue(maxsize=queue_limit)
        self._queue_limit = queue_limit
        self._no_progress_seconds = no_progress_seconds
        self._output_fps = output_fps
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._error: str | None = None
        self._finished = False
        self._initializing = True
        self._join_timed_out = False
        self._queue_high_water = 0
        self._frames_enqueued = 0
        self._frames_dropped = 0
        self._write_in_progress = False
        self._first_supplied_monotonic: float | None = None
        self._last_supplied_monotonic: float | None = None
        self._last_timeline_frame: np.ndarray | None = None
        self._timestamp_regressions = 0
        self._nonblocking_enqueue_timing = TimingDistribution()
        self._write_timing = TimingDistribution()
        self._last_progress_monotonic = monotonic()
        self.frames_written = 0
        self._thread = threading.Thread(target=self._run, name="event-clip-writer", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            for item in self._initial_frames:
                if isinstance(item, tuple):
                    timestamp, frame = item
                    supplied = _ClipFrame(float(timestamp), frame)
                else:
                    supplied = _ClipFrame(None, item)
                self._write_supplied_frame(supplied)
            with self._state_lock:
                self._initializing = False
            while not self._stop_event.is_set() or not self._queue.empty():
                try:
                    item = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                self._write_supplied_frame(item)
        except Exception as exc:
            with self._state_lock:
                self._error = f"{type(exc).__name__}: {exc}"
            self._stop_event.set()
        finally:
            with self._state_lock:
                self._initializing = False
            try:
                self._writer.release()
            except Exception as exc:
                with self._state_lock:
                    if self._error is None:
                        self._error = f"{type(exc).__name__}: {exc}"
            with self._state_lock:
                self._finished = True
                self._last_progress_monotonic = monotonic()

    def _write_frame(self, frame: np.ndarray) -> None:
        with self._state_lock:
            self._write_in_progress = True
        started = perf_counter()
        try:
            self._writer.write(frame)
            is_opened = getattr(self._writer, "isOpened", None)
            if callable(is_opened) and not is_opened():
                raise OSError("event clip writer closed while writing")
            with self._state_lock:
                self.frames_written += 1
                self._last_progress_monotonic = monotonic()
        finally:
            self._write_timing.add(perf_counter() - started)
            with self._state_lock:
                self._write_in_progress = False

    def _write_supplied_frame(self, item: _ClipFrame) -> None:
        timestamp = item.timestamp
        if timestamp is None or self._output_fps is None:
            self._write_frame(item.frame)
            return
        with self._state_lock:
            first = self._first_supplied_monotonic
            previous = self._last_supplied_monotonic
            previous_frame = self._last_timeline_frame
            frames_written = self.frames_written
        if first is None:
            self._write_frame(item.frame)
            with self._state_lock:
                self._first_supplied_monotonic = timestamp
                self._last_supplied_monotonic = timestamp
                self._last_timeline_frame = item.frame
            return
        normalized = timestamp
        if previous is not None and normalized < previous:
            normalized = previous
            with self._state_lock:
                self._timestamp_regressions += 1
        supplied_duration = max(0.0, normalized - first)
        desired_frames = max(1, round(supplied_duration * self._output_fps))
        needed = max(0, desired_frames - frames_written)
        for index in range(needed):
            output = item.frame if index == needed - 1 or previous_frame is None else previous_frame
            self._write_frame(output)
        with self._state_lock:
            self._last_supplied_monotonic = normalized
            self._last_timeline_frame = item.frame

    def write(self, frame: np.ndarray, *, timestamp: float | None = None) -> ClipWriteResult:
        """Keep the newest evidence without ever waiting for queue capacity."""

        started = perf_counter()
        with self._state_lock:
            if self._error is not None:
                return self._finish_enqueue(
                    started,
                    ClipWriteResult(False, True, "writer_failed"),
                )
            if self._finished or self._stop_event.is_set():
                return self._finish_enqueue(
                    started,
                    ClipWriteResult(False, True, "writer_closed"),
                )
        try:
            self._queue.put_nowait(_ClipFrame(timestamp, frame))
            reason = "accepted"
            degraded = False
        except queue.Full:
            # Prefer current evidence over an older frame that has not reached
            # the encoder. The event records the loss explicitly.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            else:
                with self._state_lock:
                    self._frames_dropped += 1
            try:
                self._queue.put_nowait(_ClipFrame(timestamp, frame))
                reason = "queue_full_dropped_oldest"
                degraded = True
            except queue.Full:
                with self._state_lock:
                    self._frames_dropped += 1
                return self._finish_enqueue(
                    started,
                    ClipWriteResult(False, True, "queue_full_dropped_newest"),
                )
        with self._state_lock:
            self._frames_enqueued += 1
            self._queue_high_water = max(self._queue_high_water, self._queue.qsize())
        return self._finish_enqueue(started, ClipWriteResult(True, degraded, reason))

    def _finish_enqueue(self, started: float, result: ClipWriteResult) -> ClipWriteResult:
        self._nonblocking_enqueue_timing.add(perf_counter() - started)
        return result

    def request_stop(self) -> None:
        """Ask the writer to drain queued frames and stop without waiting."""

        self._stop_event.set()

    def release(self, timeout: float = EVENT_CLIP_STOP_TIMEOUT_SECONDS) -> ClipWriterStatus:
        """Request completion and wait no longer than ``timeout`` seconds."""

        if timeout < 0:
            raise ValueError("timeout must be zero or greater")
        self.request_stop()
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            with self._state_lock:
                self._join_timed_out = True
        return self.status()

    def status(self) -> ClipWriterStatus:
        now = monotonic()
        with self._state_lock:
            error = self._error
            finished = self._finished
            initializing = self._initializing
            join_timed_out = self._join_timed_out
            last_progress = self._last_progress_monotonic
            queue_high_water = self._queue_high_water
            frames_enqueued = self._frames_enqueued
            frames_written = self.frames_written
            frames_dropped = self._frames_dropped
            write_in_progress = self._write_in_progress
            first_supplied = self._first_supplied_monotonic
            last_supplied = self._last_supplied_monotonic
            timestamp_regressions = self._timestamp_regressions
        alive = self._thread.is_alive()
        depth = self._queue.qsize()
        progress_age = max(0.0, now - last_progress)
        supplied_duration = (
            0.0
            if first_supplied is None or last_supplied is None
            else max(0.0, last_supplied - first_supplied)
        )
        encoded_duration = (
            None if self._output_fps is None else frames_written / self._output_fps
        )
        stalled = alive and (
            initializing or write_in_progress or depth > 0 or self._stop_event.is_set()
        ) and (
            join_timed_out or progress_age >= self._no_progress_seconds
        )
        if error is not None:
            state = "failed"
        elif stalled:
            state = "stalled"
        elif not alive and not finished:
            state = "dead"
        elif finished:
            state = "finished"
        elif self._stop_event.is_set():
            state = "draining"
        else:
            state = "running"
        return ClipWriterStatus(
            state=state,
            thread_alive=alive,
            stop_requested=self._stop_event.is_set(),
            queue_depth=depth,
            queue_capacity=self._queue_limit,
            queue_high_water=queue_high_water,
            frames_enqueued=frames_enqueued,
            frames_written=frames_written,
            frames_dropped=frames_dropped,
            write_in_progress=write_in_progress,
            last_progress_age_seconds=progress_age,
            error=error,
            join_timed_out=join_timed_out,
            output_fps=self._output_fps,
            first_supplied_monotonic=first_supplied,
            last_supplied_monotonic=last_supplied,
            supplied_duration_seconds=supplied_duration,
            encoded_duration_seconds=encoded_duration,
            timestamp_regressions=timestamp_regressions,
            blocking_put_attempts=0,
            blocked_put_seconds=0.0,
            nonblocking_enqueue_timing=self._nonblocking_enqueue_timing.snapshot(),
            write_timing=self._write_timing.snapshot(),
        )


def new_event_id(when: datetime | None = None) -> str:
    local = when or datetime.now().astimezone()
    return f"{local.strftime('%Y%m%d-%H%M%S-%f')[:-3]}-{secrets.token_hex(3)}"


def software_metadata(repository: Path | None = None) -> tuple[str, str]:
    try:
        from importlib.metadata import version
        software_version = version("squirrel-shooter")
    except Exception:
        software_version = "unknown"
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository, check=True, capture_output=True, text=True, timeout=3
        ).stdout.strip()
    except Exception:
        sha = "unknown"
    return software_version, sha


class RollingFrameBuffer:
    """A monotonic-time buffer whose duration adapts naturally to any FPS."""

    def __init__(self, duration_seconds: float) -> None:
        self.duration_seconds = duration_seconds
        self._frames: deque[tuple[float, np.ndarray]] = deque()

    def append(self, timestamp: float, frame: np.ndarray) -> None:
        self._frames.append((timestamp, frame.copy()))
        cutoff = timestamp - self.duration_seconds
        while self._frames and self._frames[0][0] < cutoff:
            self._frames.popleft()

    def frames(self) -> list[tuple[float, np.ndarray]]:
        return list(self._frames)

    def clear(self) -> None:
        self._frames.clear()

    def __len__(self) -> int:
        return len(self._frames)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class EventLogWriter:
    """Append completed events/rejections and flush each record immediately."""

    def __init__(self, config: AppConfig) -> None:
        self.directory = config.logging.directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.directory / config.logging.event_csv
        self.jsonl_path = self.directory / config.logging.event_jsonl
        self.rejection_path = self.directory / config.logging.rejection_jsonl
        self.maximum_log_bytes = int(config.logging.maximum_active_log_megabytes * 1024 * 1024)
        self.retained_rotations = config.logging.retained_log_rotations
        self._csv_fsync_timing = TimingDistribution()
        self._jsonl_fsync_timing = TimingDistribution()
        self._all_fsync_timing = TimingDistribution()

    def append_event(self, event: dict[str, Any]) -> None:
        self._rotate_csv_if_schema_changed()
        if self._at_limit(self.csv_path) or self._at_limit(self.jsonl_path):
            self._rotate(self.csv_path)
            self._rotate(self.jsonl_path)
        row = {field: self._csv_value(event.get(field, "")) for field in EVENT_FIELDS}
        write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            self._timed_fsync(handle, self._csv_fsync_timing)
        self._append_jsonl(self.jsonl_path, event)

    def _rotate_csv_if_schema_changed(self) -> None:
        """Preserve an older CSV before adding columns to the active schema."""

        try:
            if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
                return
            with self.csv_path.open(newline="", encoding="utf-8") as handle:
                header = next(csv.reader(handle), None)
        except OSError:
            # Let the normal append surface the filesystem failure without
            # destructively replacing the original log.
            return
        if header != list(EVENT_FIELDS):
            self._rotate(self.csv_path)

    def append_rejection(self, rejection: dict[str, Any]) -> None:
        if self._at_limit(self.rejection_path):
            self._rotate(self.rejection_path)
        self._append_jsonl(self.rejection_path, rejection)

    def _at_limit(self, path: Path) -> bool:
        try:
            return path.stat().st_size >= self.maximum_log_bytes
        except OSError:
            return False

    def _rotate(self, path: Path) -> None:
        oldest = path.with_name(f"{path.name}.{self.retained_rotations}")
        oldest.unlink(missing_ok=True)
        for number in range(self.retained_rotations - 1, 0, -1):
            source = path.with_name(f"{path.name}.{number}")
            if source.exists():
                os.replace(source, path.with_name(f"{path.name}.{number + 1}"))
        if path.exists():
            os.replace(path, path.with_name(f"{path.name}.1"))

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str, separators=(",", ":")) + "\n")
            handle.flush()
            self._timed_fsync(handle, self._jsonl_fsync_timing)

    def _timed_fsync(self, handle: Any, distribution: TimingDistribution) -> None:
        started = perf_counter()
        try:
            os.fsync(handle.fileno())
        finally:
            elapsed = perf_counter() - started
            distribution.add(elapsed)
            self._all_fsync_timing.add(elapsed)

    def fsync_timing(self) -> dict[str, object]:
        """Return CSV, JSONL, and combined durable-flush tail telemetry."""

        return {
            "csv": self._csv_fsync_timing.snapshot(),
            "jsonl": self._jsonl_fsync_timing.snapshot(),
            "all": self._all_fsync_timing.snapshot(),
        }

    @staticmethod
    def _csv_value(value: Any) -> Any:
        if isinstance(value, (list, tuple, dict)):
            return json.dumps(value, separators=(",", ":"))
        return value


class SessionLog:
    """An atomically refreshed session summary that begins as unclean."""

    def __init__(self, config: AppConfig, camera: dict[str, Any]) -> None:
        session_id = new_event_id()
        directory = config.logging.directory / config.logging.sessions_directory
        directory.mkdir(parents=True, exist_ok=True)
        old_sessions = sorted(directory.glob("session-*.json"), key=lambda path: (path.stat().st_mtime, path.name))
        for old_session in old_sessions[: max(0, len(old_sessions) - config.storage.max_log_files + 1)]:
            old_session.unlink(missing_ok=True)
        self.path = directory / f"session-{session_id}.json"
        self.data: dict[str, Any] = {
            "session_id": session_id,
            "startup_time": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "shutdown_time": None,
            "camera_open_result": "not_attempted",
            "requested_camera_mode": camera,
            "actual_camera_mode": {},
            "average_measured_fps": 0.0,
            "minimum_measured_fps": 0.0,
            "maximum_measured_fps": 0.0,
            "dropped_or_failed_frame_reads": 0,
            "raw_contours": 0,
            "grouped_candidates": 0,
            "confirmed_events": 0,
            "rejected_by_filter": {},
            "global_motion_rejections": 0,
            "camera_read_errors": 0,
            "retention_actions": [],
            "retention_action_count": 0,
            "exception_details": [],
            "exception_count": 0,
            "clean_shutdown": False,
        }
        self._fps_sample_count = 0
        self._fps_sample_sum = 0.0
        self._fps_sample_minimum: float | None = None
        self._fps_sample_maximum: float | None = None
        self.save()

    def save(self) -> None:
        if self._fps_sample_count:
            self.data["average_measured_fps"] = self._fps_sample_sum / self._fps_sample_count
            self.data["minimum_measured_fps"] = self._fps_sample_minimum
            self.data["maximum_measured_fps"] = self._fps_sample_maximum
        _atomic_json(self.path, self.data)

    def increment(self, field: str, amount: int = 1) -> None:
        self.data[field] = int(self.data.get(field, 0)) + amount

    def reject(self, reason: str) -> None:
        counts = self.data["rejected_by_filter"]
        counts[reason] = counts.get(reason, 0) + 1
        self.increment("global_motion_rejections")

    def sample_fps(self, fps: float) -> None:
        if fps > 0:
            self._fps_sample_count += 1
            self._fps_sample_sum += fps
            self._fps_sample_minimum = fps if self._fps_sample_minimum is None else min(self._fps_sample_minimum, fps)
            self._fps_sample_maximum = fps if self._fps_sample_maximum is None else max(self._fps_sample_maximum, fps)

    def add_retention_actions(self, actions: Iterable[dict[str, Any]]) -> None:
        """Retain recent action detail while keeping a lifetime aggregate count."""

        added = list(actions)
        if not added:
            return
        self.data["retention_action_count"] = int(self.data.get("retention_action_count", 0)) + len(added)
        details = self.data["retention_actions"]
        details.extend(added)
        del details[:-SESSION_DETAIL_LIMIT]

    def add_exception(self, detail: str) -> None:
        """Retain recent exception detail without growing the session forever."""

        self.data["exception_count"] = int(self.data.get("exception_count", 0)) + 1
        details = self.data["exception_details"]
        details.append(detail)
        del details[:-SESSION_DETAIL_LIMIT]

    def finish(self, *, clean: bool, exception: str | None = None) -> None:
        if exception:
            self.add_exception(exception)
        self.data["shutdown_time"] = datetime.now().astimezone().isoformat(timespec="milliseconds")
        self.data["clean_shutdown"] = clean
        self.save()


def _append_head_tail(items: list[Any], item: Any, limit: int) -> None:
    """Retain the beginning and most recent tail with a hard representation cap."""

    if len(items) < limit:
        items.append(item)
        return
    del items[limit // 2]
    items.append(item)


@dataclass
class EventGroupSummary:
    """Exact aggregate summary plus bounded forensic samples for one event."""

    total_count: int = 0
    category_counts: dict[str, int] = field(default_factory=dict)
    movement_attributes: set[str] = field(default_factory=set)
    minimum_area: float | None = None
    maximum_area: float = 0.0
    area_total: float = 0.0
    frame_coverage: float = 0.0
    inclusion_zone_coverage: float = 0.0
    maximum_width: int = 0
    maximum_height: int = 0
    starting_centroid: dict[str, Any] | None = None
    ending_centroid: dict[str, Any] | None = None
    ending_recent_centroid_path: list[list[float]] = field(default_factory=list)
    total_centroid_travel: float = 0.0
    pixel_speed_total: float = 0.0
    peak_pixel_speed: float = 0.0
    movement_direction: str = "stationary"
    aspect_ratio: float = 0.0
    mostly_stationary: bool = False
    coherent_motion: bool = False
    dispersed_motion: bool = False
    touched_inclusion_zone_boundary: bool = False
    component_count: int = 0
    grouping_confidence_total: float = 0.0
    heuristic_score: float = 0.0
    samples: list[dict[str, Any]] = field(default_factory=list)
    component_samples: list[dict[str, Any]] = field(default_factory=list)

    def add(self, group: GroupedCandidate) -> None:
        sample = group.as_dict()
        sample_index = self.total_count
        self.total_count += 1
        area = float(sample["combined_foreground_pixel_area"])
        category = str(sample["provisional_category"])
        self.category_counts[category] = self.category_counts.get(category, 0) + 1
        self.movement_attributes.update(str(item) for item in sample["movement_attributes"])
        self.minimum_area = area if self.minimum_area is None else min(self.minimum_area, area)
        self.maximum_area = max(self.maximum_area, area)
        self.area_total += area
        self.frame_coverage = max(self.frame_coverage, float(sample["frame_percentage_covered"]))
        self.inclusion_zone_coverage = max(
            self.inclusion_zone_coverage,
            float(sample["inclusion_zone_percentage_covered"]),
        )
        self.maximum_width = max(self.maximum_width, int(sample["total_width"]))
        self.maximum_height = max(self.maximum_height, int(sample["total_height"]))
        centroid = dict(sample["grouped_centroid"])
        if self.starting_centroid is None:
            self.starting_centroid = centroid
        self.ending_centroid = centroid
        self.ending_recent_centroid_path = list(sample.pop("recent_centroid_path", []))
        self.total_centroid_travel = max(self.total_centroid_travel, float(sample["travel_distance"]))
        self.pixel_speed_total += float(sample["average_pixel_speed"])
        self.peak_pixel_speed = max(self.peak_pixel_speed, float(sample["peak_pixel_speed"]))
        self.movement_direction = str(sample["direction"])
        self.aspect_ratio = float(sample["aspect_ratio"])
        self.mostly_stationary = self.mostly_stationary or bool(sample["mostly_stationary"])
        self.coherent_motion = self.coherent_motion or bool(sample["coherent_motion"])
        self.dispersed_motion = self.dispersed_motion or bool(sample["dispersed_motion"])
        self.touched_inclusion_zone_boundary = (
            self.touched_inclusion_zone_boundary
            or bool(sample["touched_inclusion_zone_boundary"])
        )
        self.component_count = max(self.component_count, int(sample["component_count"]))
        self.grouping_confidence_total += float(sample["grouping_confidence"])
        self.heuristic_score = max(self.heuristic_score, float(sample["heuristic_score"]))
        component_blobs = sample.pop("component_blobs")
        sample["sample_index"] = sample_index
        _append_head_tail(self.samples, sample, EVENT_GROUP_SAMPLE_LIMIT)
        _append_head_tail(
            self.component_samples,
            {"sample_index": sample_index, "component_blobs": component_blobs},
            EVENT_COMPONENT_SAMPLE_LIMIT,
        )


@dataclass
class ActiveEvent:
    event_id: str
    track_id: int
    directory: Path
    marker: Path
    snapshot_path: Path
    clip_incomplete_path: Path
    clip_path: Path
    start_monotonic: float
    start_timestamp: str
    last_motion_at: float
    writer: Any
    group_summary: EventGroupSummary = field(default_factory=EventGroupSummary)
    reacquisition_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    frames_written: int = 0
    latest_snapshot: np.ndarray | None = None
    recording_status: str = "recording"
    recording_error: str | None = None
    writer_telemetry: dict[str, Any] = field(default_factory=dict)


class EventRecorder:
    """Write clips/events with an incomplete marker until atomic finalization."""

    def __init__(
        self,
        config: AppConfig,
        logs: EventLogWriter,
        camera_metadata: dict[str, Any],
        *,
        video_writer_factory: Callable[..., Any] = cv2.VideoWriter,
        image_writer: Callable[[str, np.ndarray], bool] = cv2.imwrite,
    ) -> None:
        self.config = config
        self.logs = logs
        self.camera_metadata = camera_metadata
        self._video_writer_factory = video_writer_factory
        self._image_writer = image_writer
        self.active: dict[int, ActiveEvent] = {}
        self._active_lock = threading.RLock()
        self.software_version, self.git_sha = software_metadata(Path.cwd())

    def begin(
        self,
        track_id: int,
        group: GroupedCandidate,
        frame: np.ndarray,
        annotated: np.ndarray,
        pre_event_frames: Iterable[tuple[float, np.ndarray]],
        *,
        now: float,
        measured_fps: float,
    ) -> ActiveEvent:
        event_id = new_event_id()
        today = datetime.now().astimezone().strftime("%Y-%m-%d")
        directory = self.config.camera.output_directory / "events" / today / event_id
        directory.mkdir(parents=True, exist_ok=False)
        marker = directory / ".incomplete"
        marker.write_text("event is still being written\n", encoding="utf-8")
        clip_incomplete = directory / "clip.incomplete.avi"
        height, width = frame.shape[:2]
        del measured_fps
        output_fps = self.config.motion.target_fps
        fourcc = cv2.VideoWriter_fourcc(*self.config.motion.event_lifecycle.clip_codec)
        raw_writer = self._video_writer_factory(str(clip_incomplete), fourcc, output_fps, (width, height))
        if hasattr(raw_writer, "isOpened") and not raw_writer.isOpened():
            marker.unlink(missing_ok=True)
            raise OSError(f"OpenCV could not open event clip {clip_incomplete}")
        event = ActiveEvent(
            event_id, track_id, directory, marker, directory / "snapshot.jpg", clip_incomplete, directory / "clip.avi",
            now, datetime.now().astimezone().isoformat(timespec="milliseconds"), now, raw_writer,
        )
        event_frame = self._with_event_id(annotated, event_id)
        initial_frames = (
            (timestamp, self._with_event_id(buffered, event_id))
            for timestamp, buffered in pre_event_frames
        )
        event.writer = _AsyncClipWriter(
            raw_writer,
            chain(initial_frames, ((now, event_frame),)),
            output_fps=output_fps,
        )
        event.latest_snapshot = event_frame.copy()
        event.group_summary.add(group)
        with self._active_lock:
            self.active[track_id] = event
        return event

    def update(self, track_id: int, group: GroupedCandidate | None, annotated: np.ndarray, *, now: float) -> None:
        with self._active_lock:
            event = self.active[track_id]
        event_frame = self._with_event_id(annotated, event.event_id)
        outcome = event.writer.write(event_frame, timestamp=now)
        if outcome.accepted:
            event.frames_written += 1
        if outcome.degraded and event.recording_status != "failed":
            event.recording_status = "degraded"
            event.recording_error = outcome.reason
        if not outcome.accepted and outcome.reason in {"writer_closed", "writer_failed"}:
            event.recording_status = "failed"
            event.recording_error = outcome.reason
        if group is not None:
            event.last_motion_at = now
            event.group_summary.add(group)
            event.latest_snapshot = event_frame.copy()

    def record_reacquisition_diagnostic(self, track_id: int, diagnostic: dict[str, Any]) -> None:
        with self._active_lock:
            event = self.active.get(track_id)
        if event is None:
            return
        event.reacquisition_diagnostics.append(diagnostic)
        del event.reacquisition_diagnostics[:-REACQUISITION_DIAGNOSTIC_LIMIT]

    def should_finish(self, event: ActiveEvent, now: float) -> bool:
        lifecycle = self.config.motion.event_lifecycle
        return now - event.last_motion_at >= lifecycle.post_event_seconds or now - event.start_monotonic >= lifecycle.maximum_event_seconds

    def finish(self, track_id: int, *, now: float, notes: str = "") -> dict[str, Any]:
        event = self.detach_for_finalization(track_id)
        return self.finalize_detached(event, now=now, notes=notes)

    def detach_for_finalization(self, track_id: int) -> ActiveEvent:
        """Remove one event from live updates using only an in-memory operation."""

        with self._active_lock:
            return self.active.pop(track_id)

    def restore_active(self, event: ActiveEvent) -> None:
        """Restore an event when a bounded finalization queue rejects it."""

        with self._active_lock:
            if event.track_id in self.active:
                raise ValueError(f"track {event.track_id} is already active")
            self.active[event.track_id] = event

    def active_directories(self) -> set[Path]:
        """Snapshot live event directories for serialized retention."""

        with self._active_lock:
            directories = tuple(event.directory for event in self.active.values())
        return {directory.resolve() for directory in directories}

    def active_writer_statuses(self) -> tuple[dict[str, Any], ...]:
        """Expose bounded per-event writer health without filesystem work."""

        with self._active_lock:
            active = tuple(self.active.values())
        return tuple(
            {
                "event_id": event.event_id,
                "track_id": event.track_id,
                **asdict(event.writer.status()),
            }
            for event in active
        )

    def abandon(self, track_id: int, *, reason: str) -> ActiveEvent:
        """Stop accepting frames and preserve an incomplete event without waiting."""

        event = self.detach_for_finalization(track_id)
        event.recording_status = "failed"
        event.recording_error = reason
        event.writer.request_stop()
        return event

    def finalize_detached(self, event: ActiveEvent, *, now: float, notes: str = "") -> dict[str, Any]:
        """Finalize a detached event; callers may run this on a storage worker."""

        writer_status = event.writer.release()
        event.frames_written = writer_status.frames_written
        event.writer_telemetry = asdict(writer_status)
        if writer_status.state in {"dead", "failed", "stalled"}:
            event.recording_status = "failed"
            event.recording_error = writer_status.error or (
                "event clip writer did not stop before its bounded timeout"
            )
        elif writer_status.frames_dropped > 0:
            event.recording_status = "degraded"
            event.recording_error = (
                event.recording_error
                or f"event clip dropped {writer_status.frames_dropped} frame(s) under backpressure"
            )
        else:
            event.recording_status = "success"
            event.recording_error = None
        if event.recording_status != "failed" and event.clip_incomplete_path.exists():
            os.replace(event.clip_incomplete_path, event.clip_path)
        if event.latest_snapshot is None or not self._image_writer(str(event.snapshot_path), event.latest_snapshot):
            raise OSError(f"Could not write {event.snapshot_path}")
        record = self._build_record(event, now, notes)
        _atomic_json(event.directory / "event.json", record)
        self.logs.append_event(record)
        if event.recording_status != "failed":
            event.marker.unlink(missing_ok=True)
        return record

    def finish_all(self, *, now: float, notes: str = "orderly shutdown") -> list[dict[str, Any]]:
        completed: list[dict[str, Any]] = []
        with self._active_lock:
            active_track_ids = tuple(self.active)
        for track_id in active_track_ids:
            try:
                completed.append(self.finish(track_id, now=now, notes=notes))
            except Exception:
                event = self.active.pop(track_id, None)
                if event is not None:
                    event.writer.release()
        return completed

    def _build_record(self, event: ActiveEvent, now: float, notes: str) -> dict[str, Any]:
        summary = event.group_summary
        category = max(summary.category_counts, key=summary.category_counts.get)
        end_timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
        clip_complete = event.recording_status != "failed" and event.clip_path.exists()
        record: dict[str, Any] = {
            "schema_version": 2,
            "status": "complete" if event.recording_status != "failed" else "recording_failed",
            "recording_status": event.recording_status,
            "recording_error": event.recording_error,
            "event_id": event.event_id,
            "track_id": event.track_id,
            "capture_method": "automatic_motion_event",
            "start_timestamp": event.start_timestamp,
            "end_timestamp": end_timestamp,
            "duration": round(max(0.0, now - event.start_monotonic), 3),
            "snapshot_path": str(event.snapshot_path),
            "snapshot_file_role": "annotated_review_frame",
            "clip_path": str(event.clip_path) if clip_complete else None,
            "clip_file_role": "annotated_review_clip" if clip_complete else None,
            "clip_incomplete_path": (
                str(event.clip_incomplete_path) if event.clip_incomplete_path.exists() else None
            ),
            "provisional_category": category,
            "classification_disclaimer": "Heuristic size/motion label only; not species recognition.",
            "movement_attributes": sorted(summary.movement_attributes),
            "heuristic_score": summary.heuristic_score,
            "minimum_area": summary.minimum_area,
            "maximum_area": summary.maximum_area,
            "average_area": summary.area_total / summary.total_count,
            "foreground_pixel_coverage": summary.maximum_area,
            "frame_coverage": summary.frame_coverage,
            "inclusion_zone_coverage": summary.inclusion_zone_coverage,
            "maximum_width": summary.maximum_width,
            "maximum_height": summary.maximum_height,
            "starting_centroid": summary.starting_centroid,
            "ending_centroid": summary.ending_centroid,
            "ending_recent_centroid_path": summary.ending_recent_centroid_path,
            "total_centroid_travel": summary.total_centroid_travel,
            "average_pixel_speed": summary.pixel_speed_total / summary.total_count,
            "peak_pixel_speed": summary.peak_pixel_speed,
            "movement_direction": summary.movement_direction,
            "aspect_ratio": summary.aspect_ratio,
            "mostly_stationary": summary.mostly_stationary,
            "coherent_motion": summary.coherent_motion,
            "dispersed_motion": summary.dispersed_motion,
            "touched_inclusion_zone_boundary": summary.touched_inclusion_zone_boundary,
            "inclusion_zone_status": "inside",
            "component_count": summary.component_count,
            "grouping_confidence": summary.grouping_confidence_total / summary.total_count,
            "components": [item["component_blobs"] for item in summary.component_samples],
            "component_sample_indices": [item["sample_index"] for item in summary.component_samples],
            "group_samples": summary.samples,
            "group_sample_count": summary.total_count,
            "group_samples_retained": len(summary.samples),
            "group_samples_omitted": summary.total_count - len(summary.samples),
            "group_sample_storage": {
                "policy": "first_and_recent_tail",
                "sample_limit": EVENT_GROUP_SAMPLE_LIMIT,
                "component_sample_limit": EVENT_COMPONENT_SAMPLE_LIMIT,
                "component_samples_retained": len(summary.component_samples),
                "aggregates_cover_all_samples": True,
            },
            "frames_written": event.frames_written,
            "frames_dropped": int(event.writer_telemetry.get("frames_dropped", 0)),
            "writer_telemetry": event.writer_telemetry,
            **self.camera_metadata,
            "software_version": self.software_version,
            "git_commit_sha": self.git_sha,
            "notes": notes,
            "human_review_label": "",
            "human_review_notes": "",
            "reacquisition_diagnostics": event.reacquisition_diagnostics,
        }
        return record

    @staticmethod
    def _with_event_id(frame: np.ndarray, event_id: str) -> np.ndarray:
        annotated = frame.copy()
        height, width = annotated.shape[:2]
        cv2.rectangle(annotated, (8, max(0, height - 38)), (min(width - 8, 520), height - 8), (0, 0, 0), -1)
        cv2.putText(annotated, f"event {event_id}", (16, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        return annotated


@dataclass(frozen=True)
class EventStorageSubmission:
    """Immediate, nonblocking response from the storage manager."""

    accepted: bool
    kind: str
    reason: str
    event_id: str | None = None
    track_id: int | None = None
    directory: Path | None = None


@dataclass(frozen=True)
class EventStorageResult:
    """One completed finalization or retention operation."""

    kind: str
    success: bool
    degraded: bool = False
    event_id: str | None = None
    track_id: int | None = None
    directory: Path | None = None
    record: dict[str, Any] | None = None
    retention_actions: tuple[dict[str, Any], ...] = ()
    error: str | None = None
    lease_id: str | None = None


@dataclass(frozen=True)
class EventStorageManagerStatus:
    """Bounded queue and worker progress telemetry."""

    thread_alive: bool
    stopping: bool
    join_timed_out: bool
    queue_depth: int
    queue_capacity: int
    queue_high_water: int
    pending_finalizations: int
    pinned_directories: int
    leased_directories: int
    retention_pending: bool
    completed_results_waiting: int
    completed_finalizations: int
    failed_finalizations: int
    completed_retention_runs: int
    failed_retention_runs: int
    completion_results_dropped: int
    last_progress_age_seconds: float
    last_error: str | None
    finalization_timing: dict[str, object]
    retention_timing: dict[str, object]
    event_log_fsync_timing: dict[str, object]


@dataclass(frozen=True)
class _FinalizeEventWork:
    event: ActiveEvent
    now: float
    notes: str


@dataclass(frozen=True)
class _RetentionWork:
    active_directories: frozenset[Path]
    now: datetime | None


class EventStorageManager:
    """Serialize finalization and retention away from the motion producer.

    Submission methods never wait for queue capacity. Motion-runtime integration
    submits ending events, polls results, and uses ``protected_directories`` for
    every maintenance decision.
    """

    def __init__(
        self,
        recorder: EventRecorder,
        events_root: Path,
        retention: RetentionConfig,
        *,
        queue_limit: int = EVENT_STORAGE_QUEUE_LIMIT,
        completion_limit: int = EVENT_STORAGE_COMPLETION_LIMIT,
        retention_runner: Callable[..., list[dict[str, Any]]] | None = None,
    ) -> None:
        if queue_limit <= 0:
            raise ValueError("queue_limit must be greater than zero")
        if completion_limit <= 0:
            raise ValueError("completion_limit must be greater than zero")
        self.recorder = recorder
        self.events_root = events_root
        self.retention = retention
        self._retention_runner = retention_runner or enforce_retention
        self._work: queue.Queue[_FinalizeEventWork | _RetentionWork] = queue.Queue(maxsize=queue_limit)
        self._completion_limit = completion_limit
        self._results: deque[EventStorageResult] = deque(maxlen=completion_limit)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._pending_directories: set[Path] = set()
        self._pending_events: dict[Path, ActiveEvent] = {}
        self._pins: dict[Path, int] = {}
        self._leases: dict[str, Path] = {}
        self._retention_pending = False
        self._queue_high_water = 0
        self._completed_finalizations = 0
        self._failed_finalizations = 0
        self._completed_retention_runs = 0
        self._failed_retention_runs = 0
        self._completion_results_dropped = 0
        self._finalization_timing = TimingDistribution()
        self._retention_timing = TimingDistribution()
        self._last_progress_monotonic = monotonic()
        self._last_error: str | None = None
        self._thread_join_timed_out = False

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread_join_timed_out = False
            self._last_progress_monotonic = monotonic()
            self._thread = threading.Thread(
                target=self._run,
                name="event-storage",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float = EVENT_CLIP_STOP_TIMEOUT_SECONDS) -> EventStorageManagerStatus:
        if timeout < 0:
            raise ValueError("timeout must be zero or greater")
        with self._lock:
            self._stop_event.set()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        if thread is not None and thread.is_alive():
            with self._lock:
                self._thread_join_timed_out = True
                self._last_error = "event storage worker did not stop before its bounded timeout"
        return self.status()

    def request_finalize(self, track_id: int, *, now: float, notes: str = "") -> EventStorageSubmission:
        """Detach and enqueue one event without filesystem work or waiting."""

        with self._lock:
            if not self._worker_available_locked():
                return EventStorageSubmission(False, "finalize", "manager_not_running", track_id=track_id)
            try:
                event = self.recorder.detach_for_finalization(track_id)
            except KeyError:
                return EventStorageSubmission(False, "finalize", "event_not_active", track_id=track_id)
            directory = self._normalize(event.directory)
            self._pending_directories.add(directory)
            self._pending_events[directory] = event
            try:
                self._work.put_nowait(_FinalizeEventWork(event, now, notes))
            except queue.Full:
                self._pending_directories.discard(directory)
                self._pending_events.pop(directory, None)
                self.recorder.restore_active(event)
                return EventStorageSubmission(
                    False,
                    "finalize",
                    "finalization_queue_full",
                    event.event_id,
                    track_id,
                    event.directory,
                )
            self._queue_high_water = max(self._queue_high_water, self._work.qsize())
        return EventStorageSubmission(True, "finalize", "queued", event.event_id, track_id, event.directory)

    def request_retention(
        self,
        *,
        active_directories: Iterable[Path] = (),
        now: datetime | None = None,
    ) -> EventStorageSubmission:
        """Coalesce retention work and never scan from the caller."""

        work = _RetentionWork(
            frozenset(self._normalize(path) for path in active_directories),
            now,
        )
        with self._lock:
            if not self._worker_available_locked():
                return EventStorageSubmission(False, "retention", "manager_not_running")
            if self._retention_pending:
                return EventStorageSubmission(True, "retention", "already_pending")
            self._retention_pending = True
            try:
                self._work.put_nowait(work)
            except queue.Full:
                self._retention_pending = False
                return EventStorageSubmission(False, "retention", "storage_queue_full")
            self._queue_high_water = max(self._queue_high_water, self._work.qsize())
        return EventStorageSubmission(True, "retention", "queued")

    def pin(self, directory: Path) -> None:
        """Protect a directory while another evidence owner still needs it."""

        normalized = self._normalize(directory)
        with self._lock:
            self._pins[normalized] = self._pins.get(normalized, 0) + 1

    def unpin(self, directory: Path) -> None:
        normalized = self._normalize(directory)
        with self._lock:
            count = self._pins.get(normalized, 0)
            if count <= 1:
                self._pins.pop(normalized, None)
            else:
                self._pins[normalized] = count - 1

    def acquire_lease(self, directory: Path) -> str:
        """Create an independently releasable retention lease."""

        normalized = self._normalize(directory)
        with self._lock:
            return self._acquire_lease_locked(normalized)

    def release_lease(self, lease_id: str) -> bool:
        """Release a lease exactly once; repeated releases are harmless."""

        with self._lock:
            return self._leases.pop(lease_id, None) is not None

    def pending_directories(self) -> set[Path]:
        with self._lock:
            return set(self._pending_directories)

    def request_stop_pending_writers(self) -> int:
        """Signal detached writers without joining or waiting for storage work."""

        with self._lock:
            events = tuple(self._pending_events.values())
        for event in events:
            event.writer.request_stop()
        return len(events)

    def protected_directories(self) -> set[Path]:
        """Return every directory still owned by runtime or an unpolled result."""

        active = self.recorder.active_directories()
        with self._lock:
            pending = set(self._pending_directories)
            pinned = set(self._pins)
            leased = set(self._leases.values())
            result_directories = tuple(
                result.directory
                for result in self._results
                if result.kind == "finalize" and result.directory is not None
            )
        return active | pending | pinned | leased | {
            self._normalize(directory) for directory in result_directories
        }

    def poll_results(
        self,
        maximum: int | None = None,
        *,
        claim_finalize_directories: bool = False,
    ) -> list[EventStorageResult]:
        """Poll results, optionally replacing result protection with atomic leases."""

        if maximum is not None and maximum < 0:
            raise ValueError("maximum must be zero or greater")
        with self._lock:
            count = len(self._results) if maximum is None else min(maximum, len(self._results))
            results: list[EventStorageResult] = []
            for _ in range(count):
                result = self._results.popleft()
                if (
                    claim_finalize_directories
                    and result.kind == "finalize"
                    and result.directory is not None
                ):
                    lease_id = self._acquire_lease_locked(self._normalize(result.directory))
                    result = replace(result, lease_id=lease_id)
                results.append(result)
            return results

    def status(self) -> EventStorageManagerStatus:
        thread = self._thread
        logs = getattr(self.recorder, "logs", None)
        fsync_timing = getattr(logs, "fsync_timing", None)
        if callable(fsync_timing):
            event_log_fsync_timing = fsync_timing()
        else:
            event_log_fsync_timing = {
                name: TimingDistribution().snapshot()
                for name in ("csv", "jsonl", "all")
            }
        with self._lock:
            return EventStorageManagerStatus(
                thread_alive=thread is not None and thread.is_alive(),
                stopping=self._stop_event.is_set(),
                join_timed_out=self._thread_join_timed_out,
                queue_depth=self._work.qsize(),
                queue_capacity=self._work.maxsize,
                queue_high_water=self._queue_high_water,
                pending_finalizations=len(self._pending_directories),
                pinned_directories=len(self._pins),
                leased_directories=len(self._leases),
                retention_pending=self._retention_pending,
                completed_results_waiting=len(self._results),
                completed_finalizations=self._completed_finalizations,
                failed_finalizations=self._failed_finalizations,
                completed_retention_runs=self._completed_retention_runs,
                failed_retention_runs=self._failed_retention_runs,
                completion_results_dropped=self._completion_results_dropped,
                last_progress_age_seconds=max(0.0, monotonic() - self._last_progress_monotonic),
                last_error=self._last_error,
                finalization_timing=self._finalization_timing.snapshot(),
                retention_timing=self._retention_timing.snapshot(),
                event_log_fsync_timing=event_log_fsync_timing,
            )

    def _worker_available_locked(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive() and not self._stop_event.is_set()

    def _acquire_lease_locked(self, directory: Path) -> str:
        while True:
            lease_id = secrets.token_hex(12)
            if lease_id not in self._leases:
                self._leases[lease_id] = directory
                return lease_id

    def _run(self) -> None:
        while not self._stop_event.is_set() or not self._work.empty():
            try:
                work = self._work.get(timeout=0.1)
            except queue.Empty:
                continue
            if isinstance(work, _FinalizeEventWork):
                self._run_finalize(work)
            else:
                self._run_retention(work)
            with self._lock:
                self._last_progress_monotonic = monotonic()

    def _run_finalize(self, work: _FinalizeEventWork) -> None:
        event = work.event
        directory = self._normalize(event.directory)
        started = perf_counter()
        try:
            record = self.recorder.finalize_detached(event, now=work.now, notes=work.notes)
            recording_status = str(record.get("recording_status", "failed"))
            success = record.get("status") == "complete" and recording_status in {"success", "degraded"}
            result = EventStorageResult(
                kind="finalize",
                success=success,
                degraded=recording_status == "degraded",
                event_id=event.event_id,
                track_id=event.track_id,
                directory=event.directory,
                record=record,
                error=None if success else str(record.get("recording_error") or "event recording failed"),
            )
            with self._lock:
                if success:
                    self._completed_finalizations += 1
                    self._last_error = None
                else:
                    self._failed_finalizations += 1
                    self._last_error = result.error
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            event.writer.request_stop()
            result = EventStorageResult(
                kind="finalize",
                success=False,
                event_id=event.event_id,
                track_id=event.track_id,
                directory=event.directory,
                error=error,
            )
            with self._lock:
                self._failed_finalizations += 1
                self._last_error = error
        finally:
            self._finalization_timing.add(perf_counter() - started)
            with self._lock:
                self._pending_directories.discard(directory)
                self._pending_events.pop(directory, None)
        self._append_result(result)

    def _run_retention(self, work: _RetentionWork) -> None:
        started = perf_counter()
        try:
            protected = self.protected_directories() | set(work.active_directories)
            actions = self._retention_runner(
                self.events_root,
                self.retention,
                active_directories=protected,
                now=work.now,
            )
            result = EventStorageResult(
                kind="retention",
                success=True,
                retention_actions=tuple(actions),
            )
            with self._lock:
                self._completed_retention_runs += 1
                self._last_error = None
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            result = EventStorageResult(kind="retention", success=False, error=error)
            with self._lock:
                self._failed_retention_runs += 1
                self._last_error = error
        finally:
            self._retention_timing.add(perf_counter() - started)
            with self._lock:
                self._retention_pending = False
        self._append_result(result)

    def _append_result(self, result: EventStorageResult) -> None:
        with self._lock:
            if len(self._results) == self._completion_limit:
                self._completion_results_dropped += 1
            self._results.append(result)

    @staticmethod
    def _normalize(directory: Path) -> Path:
        # Lexical normalization keeps producer-facing submit/pin operations free
        # from filesystem resolution calls that can stall on unhealthy storage.
        return Path(os.path.abspath(os.fspath(directory)))


def recover_incomplete_events(events_root: Path) -> list[Path]:
    """Preserve interrupted folders, record recovery, and make them inspectable."""

    recovered: list[Path] = []
    if not events_root.exists():
        return recovered
    for marker in events_root.rglob(".incomplete"):
        directory = marker.parent
        event_json = directory / "event.json"
        if not event_json.exists():
            payload = {
                "schema_version": 1,
                "status": "interrupted_recovered",
                "event_id": directory.name,
                "recovered_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                "notes": "Writer stopped before this event could be completed; files were preserved for review.",
            }
            _atomic_json(event_json, payload)
        os.replace(marker, directory / ".recovered-incomplete")
        recovered.append(directory)
    return recovered


def _directory_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def enforce_retention(events_root: Path, config: RetentionConfig, *, active_directories: set[Path] | None = None, now: datetime | None = None) -> list[dict[str, Any]]:
    """Delete oldest complete events first; never touch active or recovered folders."""

    active = {path.resolve() for path in (active_directories or set())}
    candidates: list[tuple[datetime, Path, int]] = []
    if not events_root.exists():
        return []
    for event_json in events_root.rglob("event.json"):
        directory = event_json.parent
        if directory.resolve() in active or (directory / ".incomplete").exists() or (directory / ".recovered-incomplete").exists():
            continue
        try:
            payload = json.loads(event_json.read_text(encoding="utf-8"))
            if payload.get("status") != "complete":
                continue
            stamp = datetime.fromisoformat(payload["start_timestamp"])
            candidates.append((stamp, directory, _directory_size(directory)))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    candidates.sort(key=lambda item: (item[0], str(item[1])))
    current = now or datetime.now().astimezone()
    cutoff = current - timedelta(days=config.maximum_event_age_days)
    total = sum(size for _, _, size in candidates)
    maximum_bytes = int(config.maximum_storage_megabytes * 1024 * 1024)
    actions: list[dict[str, Any]] = []
    while candidates:
        too_old = candidates[0][0] < cutoff
        too_many = config.maximum_event_count is not None and len(candidates) > config.maximum_event_count
        too_large = total > maximum_bytes
        if not (too_old or too_many or too_large):
            break
        stamp, directory, size = candidates.pop(0)
        reason = "maximum_age" if too_old else ("maximum_count" if too_many else "maximum_storage")
        shutil.rmtree(directory)
        total -= size
        actions.append({"event_id": directory.name, "deleted_at": current.isoformat(timespec="milliseconds"), "reason": reason, "bytes": size, "start_timestamp": stamp.isoformat()})
    return actions
