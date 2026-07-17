"""Crash-aware event recording, append-only logs, sessions, and retention."""

from __future__ import annotations

import csv
import json
import os
import secrets
import shutil
import subprocess
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np

from .config import AppConfig, RetentionConfig
from .watch_detection import GroupedCandidate


EVENT_FIELDS = (
    "event_id", "start_timestamp", "end_timestamp", "duration", "snapshot_path", "clip_path",
    "provisional_category", "movement_attributes", "heuristic_score", "minimum_area", "maximum_area",
    "average_area", "foreground_pixel_coverage", "frame_coverage", "inclusion_zone_coverage",
    "maximum_width", "maximum_height", "starting_centroid", "ending_centroid", "total_centroid_travel",
    "average_pixel_speed", "peak_pixel_speed", "inclusion_zone_status", "component_count",
    "grouping_confidence", "measured_camera_fps", "camera_reported_fps", "requested_width", "requested_height", "requested_fps",
    "actual_width", "actual_height", "camera_mode_if_known", "ir_mode_if_explicitly_detected_or_configured",
    "low_fps_observed", "software_version", "git_commit_sha", "notes", "human_review_label",
    "human_review_notes",
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

    def append_event(self, event: dict[str, Any]) -> None:
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
            os.fsync(handle.fileno())
        self._append_jsonl(self.jsonl_path, event)

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

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

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
            "exception_details": [],
            "clean_shutdown": False,
        }
        self._fps_samples: list[float] = []
        self.save()

    def save(self) -> None:
        if self._fps_samples:
            self.data["average_measured_fps"] = sum(self._fps_samples) / len(self._fps_samples)
            self.data["minimum_measured_fps"] = min(self._fps_samples)
            self.data["maximum_measured_fps"] = max(self._fps_samples)
        _atomic_json(self.path, self.data)

    def increment(self, field: str, amount: int = 1) -> None:
        self.data[field] = int(self.data.get(field, 0)) + amount

    def reject(self, reason: str) -> None:
        counts = self.data["rejected_by_filter"]
        counts[reason] = counts.get(reason, 0) + 1
        self.increment("global_motion_rejections")

    def sample_fps(self, fps: float) -> None:
        if fps > 0:
            self._fps_samples.append(fps)

    def finish(self, *, clean: bool, exception: str | None = None) -> None:
        if exception:
            self.data["exception_details"].append(exception)
        self.data["shutdown_time"] = datetime.now().astimezone().isoformat(timespec="milliseconds")
        self.data["clean_shutdown"] = clean
        self.save()


@dataclass
class ActiveEvent:
    event_id: str
    directory: Path
    marker: Path
    snapshot_path: Path
    clip_incomplete_path: Path
    clip_path: Path
    start_monotonic: float
    start_timestamp: str
    last_motion_at: float
    writer: Any
    group_samples: list[dict[str, Any]] = field(default_factory=list)
    frames_written: int = 0
    latest_snapshot: np.ndarray | None = None


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
        reported_fps = float(self.camera_metadata.get("camera_reported_fps", 0) or 0)
        output_fps = measured_fps if measured_fps > 0 else (reported_fps if reported_fps > 0 else 1.0)
        fourcc = cv2.VideoWriter_fourcc(*self.config.motion.event_lifecycle.clip_codec)
        writer = self._video_writer_factory(str(clip_incomplete), fourcc, output_fps, (width, height))
        if hasattr(writer, "isOpened") and not writer.isOpened():
            marker.unlink(missing_ok=True)
            raise OSError(f"OpenCV could not open event clip {clip_incomplete}")
        event = ActiveEvent(
            event_id, directory, marker, directory / "snapshot.jpg", clip_incomplete, directory / "clip.avi",
            now, datetime.now().astimezone().isoformat(timespec="milliseconds"), now, writer,
        )
        for _, buffered in pre_event_frames:
            writer.write(self._with_event_id(buffered, event_id))
            event.frames_written += 1
        event_frame = self._with_event_id(annotated, event_id)
        writer.write(event_frame)
        event.frames_written += 1
        event.latest_snapshot = event_frame.copy()
        event.group_samples.append(group.as_dict())
        self.active[track_id] = event
        return event

    def update(self, track_id: int, group: GroupedCandidate | None, annotated: np.ndarray, *, now: float) -> None:
        event = self.active[track_id]
        event_frame = self._with_event_id(annotated, event.event_id)
        event.writer.write(event_frame)
        event.frames_written += 1
        if group is not None:
            event.last_motion_at = now
            event.group_samples.append(group.as_dict())
            event.latest_snapshot = event_frame.copy()

    def should_finish(self, event: ActiveEvent, now: float) -> bool:
        lifecycle = self.config.motion.event_lifecycle
        return now - event.last_motion_at >= lifecycle.post_event_seconds or now - event.start_monotonic >= lifecycle.maximum_event_seconds

    def finish(self, track_id: int, *, now: float, notes: str = "") -> dict[str, Any]:
        event = self.active.pop(track_id)
        event.writer.release()
        if event.clip_incomplete_path.exists():
            os.replace(event.clip_incomplete_path, event.clip_path)
        if event.latest_snapshot is None or not self._image_writer(str(event.snapshot_path), event.latest_snapshot):
            raise OSError(f"Could not write {event.snapshot_path}")
        record = self._build_record(event, now, notes)
        _atomic_json(event.directory / "event.json", record)
        self.logs.append_event(record)
        event.marker.unlink(missing_ok=True)
        return record

    def finish_all(self, *, now: float, notes: str = "orderly shutdown") -> list[dict[str, Any]]:
        completed: list[dict[str, Any]] = []
        for track_id in list(self.active):
            try:
                completed.append(self.finish(track_id, now=now, notes=notes))
            except Exception:
                event = self.active.pop(track_id, None)
                if event is not None:
                    event.writer.release()
        return completed

    def _build_record(self, event: ActiveEvent, now: float, notes: str) -> dict[str, Any]:
        samples = event.group_samples
        first, last = samples[0], samples[-1]
        areas = [float(item["combined_foreground_pixel_area"]) for item in samples]
        category_counts: dict[str, int] = {}
        for item in samples:
            category = str(item["provisional_category"])
            category_counts[category] = category_counts.get(category, 0) + 1
        category = max(category_counts, key=category_counts.get)
        attributes = sorted({attribute for item in samples for attribute in item["movement_attributes"]})
        end_timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
        record: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "event_id": event.event_id,
            "start_timestamp": event.start_timestamp,
            "end_timestamp": end_timestamp,
            "duration": round(max(0.0, now - event.start_monotonic), 3),
            "snapshot_path": str(event.snapshot_path),
            "clip_path": str(event.clip_path),
            "provisional_category": category,
            "classification_disclaimer": "Heuristic size/motion label only; not species recognition.",
            "movement_attributes": attributes,
            "heuristic_score": max(float(item["heuristic_score"]) for item in samples),
            "minimum_area": min(areas),
            "maximum_area": max(areas),
            "average_area": sum(areas) / len(areas),
            "foreground_pixel_coverage": max(float(item["combined_foreground_pixel_area"]) for item in samples),
            "frame_coverage": max(float(item["frame_percentage_covered"]) for item in samples),
            "inclusion_zone_coverage": max(float(item["inclusion_zone_percentage_covered"]) for item in samples),
            "maximum_width": max(int(item["total_width"]) for item in samples),
            "maximum_height": max(int(item["total_height"]) for item in samples),
            "starting_centroid": first["grouped_centroid"],
            "ending_centroid": last["grouped_centroid"],
            "total_centroid_travel": max(float(item["travel_distance"]) for item in samples),
            "average_pixel_speed": sum(float(item["average_pixel_speed"]) for item in samples) / len(samples),
            "peak_pixel_speed": max(float(item["peak_pixel_speed"]) for item in samples),
            "movement_direction": last["direction"],
            "aspect_ratio": last["aspect_ratio"],
            "mostly_stationary": any(bool(item["mostly_stationary"]) for item in samples),
            "coherent_motion": any(bool(item["coherent_motion"]) for item in samples),
            "dispersed_motion": any(bool(item["dispersed_motion"]) for item in samples),
            "touched_inclusion_zone_boundary": any(bool(item["touched_inclusion_zone_boundary"]) for item in samples),
            "inclusion_zone_status": "inside",
            "component_count": max(int(item["component_count"]) for item in samples),
            "grouping_confidence": sum(float(item["grouping_confidence"]) for item in samples) / len(samples),
            "components": [item["component_blobs"] for item in samples],
            "group_samples": samples,
            "frames_written": event.frames_written,
            **self.camera_metadata,
            "software_version": self.software_version,
            "git_commit_sha": self.git_sha,
            "notes": notes,
            "human_review_label": "",
            "human_review_notes": "",
        }
        return record

    @staticmethod
    def _with_event_id(frame: np.ndarray, event_id: str) -> np.ndarray:
        annotated = frame.copy()
        height, width = annotated.shape[:2]
        cv2.rectangle(annotated, (8, max(0, height - 38)), (min(width - 8, 520), height - 8), (0, 0, 0), -1)
        cv2.putText(annotated, f"event {event_id}", (16, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        return annotated


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
