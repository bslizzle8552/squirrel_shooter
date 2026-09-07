"""Bounded clean-camera recording. This module has no physical-control authority.

One collector borrows the existing camera's raw packets; one encoder owns file
I/O. Manual and automatic reasons share a logical session. No detector is wired
here. The legacy FIRE recorder remains a separate compatibility path in Phase 2.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import re
import shutil
import threading
import uuid
from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Callable, Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class RecordingConfig:
    enabled: bool = False  # Old configs without the section keep old behavior.
    pre_roll_seconds: float = 2.0
    manual_duration_seconds: float = 30.0
    automatic_tail_seconds: float = 3.0
    maximum_segment_seconds: float = 60.0
    maximum_session_seconds: float = 600.0
    observation_max_age_seconds: float = 3.0
    target_fps: float = 12.0
    codec: str = "MJPG"
    queue_capacity: int = 8
    shutdown_timeout_seconds: float = 2.0
    minimum_free_megabytes: float = 256.0
    storage_budget_megabytes: float = 1024.0
    maximum_buffer_megabytes: float = 128.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("recording.enabled must be boolean")
        for key, value in asdict(self).items():
            if key in {"enabled", "codec", "queue_capacity"}:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"recording.{key} must be finite")
            if value < 0 or (value == 0 and key not in {"pre_roll_seconds", "minimum_free_megabytes"}):
                raise ValueError(f"recording.{key} is out of range")
        if not isinstance(self.queue_capacity, int) or isinstance(self.queue_capacity, bool) or not 1 <= self.queue_capacity <= 32:
            raise ValueError("recording.queue_capacity must be 1..32")
        if self.pre_roll_seconds > 2 or not 1 <= self.target_fps <= 30:
            raise ValueError("recording pre-roll must be <=2s and FPS must be 1..30")
        if self.maximum_session_seconds > 3600 or max(self.manual_duration_seconds, self.maximum_segment_seconds) > self.maximum_session_seconds:
            raise ValueError("recording durations exceed bounded session policy")
        if not isinstance(self.codec, str) or self.codec != "MJPG":
            raise ValueError("recording.codec must be MJPG for the validated AVI path")
        if self.maximum_buffer_megabytes > 256 or self.shutdown_timeout_seconds > 10:
            raise ValueError("recording buffer must be <=256MiB and shutdown wait <=10s")


class RawCamera(Protocol):
    def wait_for_frame(self, after_sequence: int, timeout: float, *, copy: bool) -> Any: ...
    def buffered_frames(self, since_monotonic: float, *, until_monotonic: float, copy: bool) -> Any: ...


class RecordingError(RuntimeError):
    pass


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_video(path: Path, width: int, height: int, count: int) -> dict[str, Any]:
    """Decode a finalized *file*, never a camera; count/geometry must match."""
    if not path.is_file() or path.stat().st_size == 0 or count < 1:
        raise RecordingError("empty_output")
    capture = cv2.VideoCapture(str(path))
    decoded = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if image is None or image.shape[:2] != (height, width):
                raise RecordingError("decoded_geometry_mismatch")
            decoded += 1
    finally:
        capture.release()
    if decoded != count:
        raise RecordingError("undecodable_or_incomplete_output")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"decoded_frames": decoded, "sha256": digest, "size_bytes": path.stat().st_size}


@dataclass
class _Session:
    id: str
    started: float
    wall: str
    maximum_until: float
    manual_until: float | None = None
    automatic: dict[str, dict[str, Any]] = field(default_factory=dict)
    identities: dict[str, dict[str, Any]] = field(default_factory=dict)
    timeline: deque = field(default_factory=lambda: deque(maxlen=128))
    timeline_omitted: int = 0
    start_reasons: list[dict[str, Any]] = field(default_factory=list)
    state: str = "capturing"
    stop_reason: str | None = None
    ended_monotonic: float | None = None
    error: str | None = None
    queued: int = 0
    queue_dropped: int = 0
    source_skipped: int = 0
    source_frames: int = 0
    written_frames: int = 0
    pre_roll_frames: int = 0
    last_sequence: int = -1
    read_sequence: int = -1
    next_sample: float = 0.0
    prepared: bool = False
    pre_packets: deque = field(default_factory=deque)
    manifest_started: bool = False
    segments: list[dict[str, Any]] = field(default_factory=list)


class RecordingService:
    """Session policy is lock-protected; encoding/disk work never holds that lock."""

    def __init__(self, camera: RawCamera, config: RecordingConfig, directory: Path, *,
                 clock: Callable[[], float] = monotonic,
                 writer_factory: Callable[..., Any] = cv2.VideoWriter,
                 validator: Callable[..., dict[str, Any]] = validate_video,
                 disk_usage: Callable[..., Any] = shutil.disk_usage) -> None:
        self.camera, self.config, self.directory = camera, config, directory
        self._clock, self._writer_factory, self._validator, self._disk_usage = clock, writer_factory, validator, disk_usage
        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue(maxsize=config.queue_capacity)
        self._sessions: list[_Session] = []  # At most two pending logical sessions.
        self._active: _Session | None = None
        self._last: dict[str, Any] | None = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._collector: threading.Thread | None = None
        self._encoder: threading.Thread | None = None
        self._ready = False
        self._closed = False
        self._last_error: str | None = None
        self._shutdown_timed_out = False
        self._storage_bytes = 0
        self._buffered_bytes = 0
        self._recovered_count = 0
        self._writer: Any = None
        self._segment: dict[str, Any] | None = None
        self._segment_session: _Session | None = None

    def start(self) -> None:
        with self._lock:
            if self._encoder is not None or self._closed or not self.config.enabled:
                return
            self._encoder = threading.Thread(target=self._encode_loop, name="clean-record-encode", daemon=True)
            self._collector = threading.Thread(target=self._collect_loop, name="clean-record-collect", daemon=True)
            self._encoder.start()
            self._collector.start()

    def _time(self) -> float:
        value = self._clock()
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise RecordingError("invalid_monotonic_clock")
        return value

    def _event(self, session: _Session, action: str, now: float, **fields: Any) -> None:
        entry = {"action": action, "monotonic": now, "wall": _utc(), **fields}
        if not session.start_reasons and action in {"manual_start_or_extend", "automatic_extend"}:
            session.start_reasons.append(entry)
        if len(session.timeline) == session.timeline.maxlen:
            session.timeline_omitted += 1
        session.timeline.append(entry)

    def _ensure_session(self, now: float) -> _Session:
        if not self.config.enabled or not self._ready or self._closed or self._last_error:
            raise RecordingError(self._last_error or "recording_unavailable")
        self._expire(now)
        if self._active is None:
            if len(self._sessions) >= 2:
                raise RecordingError("finalization_backpressure")
            session = _Session(uuid.uuid4().hex, now, _utc(), now + self.config.maximum_session_seconds)
            self._sessions.append(session)
            self._active = session
        return self._active

    def record_manual(self) -> dict[str, Any]:
        now = self._time()
        with self._lock:
            session = self._ensure_session(now)
            session.manual_until = min(session.maximum_until, now + self.config.manual_duration_seconds)
            self._event(session, "manual_start_or_extend", now, deadline=session.manual_until)
        self._wake.set()
        return self.status()

    def stop_manual(self) -> dict[str, Any]:
        now = self._time()
        with self._lock:
            if self._active:
                self._active.manual_until = None
                self._event(self._active, "manual_stop", now)
                self._expire(now, "manual_stop")
        self._wake.set()
        return self.status()

    def extend_automatic_recording(self, *, event_id: str, observed_monotonic: float,
                                   reason: str, visit_id: str | None = None) -> dict[str, Any]:
        now = self._time()
        for value in (event_id, reason, visit_id if visit_id is not None else "unspecified"):
            if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value) is None:
                raise RecordingError("invalid_source_identity")
        if isinstance(observed_monotonic, bool) or not isinstance(observed_monotonic, (int, float)) or not math.isfinite(observed_monotonic):
            raise RecordingError("invalid_observation_time")
        if not 0 <= now - observed_monotonic <= self.config.observation_max_age_seconds:
            raise RecordingError("stale_or_future_observation")
        deadline = observed_monotonic + self.config.automatic_tail_seconds
        if deadline <= now:
            raise RecordingError("observation_tail_expired")
        key = json.dumps([event_id, visit_id, reason])
        with self._lock:
            session = self._ensure_session(now)
            previous = session.identities.get(key)
            if previous and observed_monotonic <= previous["observed_monotonic"]:
                raise RecordingError("duplicate_or_regressed_observation")
            if key not in session.identities and len(session.identities) >= 64:
                raise RecordingError("source_identity_limit")
            entry = {"event_id": event_id, "visit_id": visit_id, "reason": reason,
                     "observed_monotonic": observed_monotonic, "until": min(deadline, session.maximum_until)}
            session.automatic[key] = entry
            session.identities[key] = entry
            self._event(session, "automatic_extend", now, **entry)
        self._wake.set()
        return self.status()

    def _finish(self, session: _Session, reason: str, now: float) -> None:
        session.state, session.stop_reason = "finalizing", reason
        session.ended_monotonic = now
        self._event(session, "finalizing", now, reason=reason)
        if self._active is session:
            self._active = None

    def _expire(self, now: float, reason: str = "deadlines_expired") -> None:
        session = self._active
        if session is None:
            return
        prior_deadline = max([session.manual_until or 0, *[v["until"] for v in session.automatic.values()]])
        if now < session.started:
            session.error = "monotonic_clock_regressed"
            self._finish(session, session.error, now)
            return
        if session.manual_until is not None and now >= session.manual_until:
            session.manual_until = None
            self._event(session, "manual_deadline", now)
        for key in list(session.automatic):
            if now >= session.automatic[key]["until"]:
                self._event(session, "automatic_deadline", now, **session.automatic.pop(key))
        if now >= session.maximum_until or (session.manual_until is None and not session.automatic):
            self._finish(session, "maximum_session" if now >= session.maximum_until else reason,
                         min(now, session.maximum_until, prior_deadline or now) if reason == "deadlines_expired" else now)

    def _summary(self, session: _Session) -> dict[str, Any]:
        return {"schema_version": 1, "session_id": session.id, "status": session.state,
                "file_role": "clean_authoritative", "parent_session_id": None,
                "start_monotonic": session.started, "start_wall": session.wall,
                "maximum_until": session.maximum_until, "manual_until": session.manual_until,
                "automatic_reasons": list(session.automatic.values()), "source_identities": list(session.identities.values()),
                "timeline": list(session.timeline), "timeline_omitted": session.timeline_omitted,
                "start_reasons": session.start_reasons,
                "stop_reason": session.stop_reason, "error": session.error,
                "end_monotonic": session.ended_monotonic,
                "source_frames": session.source_frames, "written_frames": session.written_frames,
                "queue_dropped": session.queue_dropped, "skipped_source_sequences": session.source_skipped,
                "pre_roll_frames": session.pre_roll_frames, "segments": session.segments,
                "target_fps": self.config.target_fps, "codec": self.config.codec,
                "timestamp_basis": "camera receipt after read; not sensor exposure",
                "retention_protected": True, "retention_policy": "no automatic deletion; quota admission",
                "derivatives": [], "fire_recording": "legacy compatibility path unchanged"}

    def status(self) -> dict[str, Any]:
        with self._lock:
            session = self._active or (self._sessions[-1] if self._sessions else None)
            result = deepcopy(self._summary(session) if session else self._last or {})
            result.update(enabled=self.config.enabled, ready=self._ready and not self._closed,
                          active=self._active is not None, queue_depth=self._queue.qsize(),
                          queue_capacity=self.config.queue_capacity, pending_sessions=len(self._sessions),
                          storage_bytes=self._storage_bytes, recovered_sessions=self._recovered_count,
                          buffered_bytes=self._buffered_bytes, maximum_buffer_bytes=int(self.config.maximum_buffer_megabytes * 1024**2),
                          last_error=self._last_error, shutdown_timed_out=self._shutdown_timed_out)
            result["manual_remaining_seconds"] = max(0.0, (session.manual_until or 0) - self._clock()) if session else 0
            result["automatic_active"] = bool(session and session.automatic)
            return result

    def _admit(self, session: _Session, packet: Any, *, pre_roll: bool = False) -> None:
        image = packet.frame
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3
                or image.shape[2] != 3 or min(image.shape[:2]) < 2
                or not math.isfinite(packet.received_monotonic)):
            raise RecordingError("invalid_raw_camera_packet")
        with self._lock:
            if session.state != "capturing":
                return
            session.read_sequence = max(session.read_sequence, packet.sequence)
            if packet.sequence <= session.last_sequence:
                return
            if not pre_roll and packet.received_monotonic < session.started:
                return
            if packet.received_monotonic < session.next_sample:
                return
            if session.last_sequence >= 0:
                session.source_skipped += max(0, packet.sequence - session.last_sequence - 1)
            session.last_sequence = packet.sequence
            interval = 1 / self.config.target_fps
            session.next_sample = (packet.received_monotonic + interval if session.next_sample == 0
                                   else session.next_sample + (math.floor((packet.received_monotonic - session.next_sample) / interval) + 1) * interval)
            session.source_frames += 1
            if pre_roll:
                session.pre_roll_frames += 1
            if self._buffered_bytes + image.nbytes > self.config.maximum_buffer_megabytes * 1024**2:
                session.queue_dropped += 1
                if pre_roll:
                    session.pre_roll_frames -= 1
                return
            # A private bounded copy protects encoder ownership, including injected
            # providers. Production camera buffers are additionally read-only.
            private = image.copy()
            private.flags.writeable = False
            payload = (session, packet.sequence, packet.received_monotonic, packet.received_at, packet.generation, private)
            try:
                if pre_roll:
                    session.pre_packets.append(payload)  # <= pre-roll FPS * 2s +1
                else:
                    self._queue.put_nowait(payload)
                session.queued += 1
                self._buffered_bytes += private.nbytes
            except queue.Full:
                session.queue_dropped += 1

    def _collect_loop(self) -> None:
        while not self._stop.is_set():
            session = None
            try:
                now = self._time()
                with self._lock:
                    self._expire(now)
                    session = self._active
                if session is None:
                    self._wake.wait(0.05)
                    self._wake.clear()
                    continue
                if not session.prepared:
                    # Short bounded references only. Extensions never repeat this.
                    pre = deque(self.camera.buffered_frames(
                        session.started - self.config.pre_roll_seconds,
                        until_monotonic=session.started, copy=False),
                        maxlen=math.ceil(self.config.pre_roll_seconds * self.config.target_fps) + 1)
                    for packet in pre:
                        self._admit(session, packet, pre_roll=True)
                    session.prepared = True
                packet = self.camera.wait_for_frame(session.read_sequence, timeout=0.05, copy=False)
                if packet is not None:
                    with self._lock:
                        current = self._active is session
                        deadline = max([session.manual_until or 0, *[v["until"] for v in session.automatic.values()]])
                    if current and packet.received_monotonic <= deadline:
                        self._admit(session, packet)
            except Exception as exc:
                with self._lock:
                    # Release already admitted pre-roll even if preparation failed.
                    if session is not None:
                        session.prepared = True
                    if self._active:
                        self._active.error = type(exc).__name__ + ": " + str(exc)
                        self._finish(self._active, "capture_error", self._clock())
                self._stop.wait(0.05)

    def _manifest(self, session: _Session) -> None:
        with self._lock:
            payload = deepcopy(self._summary(session))
        path = self.directory / session.id
        path.mkdir(parents=True, exist_ok=True)
        _atomic_json(path / "session.json", payload)

    def _storage_check(self) -> None:
        self._storage_bytes = sum(p.stat().st_size for p in self.directory.rglob("*") if p.is_file())
        if self._storage_bytes >= self.config.storage_budget_megabytes * 1024**2:
            raise RecordingError("recording_storage_budget_exhausted")
        if self._disk_usage(self.directory).free < self.config.minimum_free_megabytes * 1024**2:
            raise RecordingError("insufficient_free_disk")

    def _recover(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        for path in self.directory.glob("*/session.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("status") in {"capturing", "finalizing"}:
                    data.update(status="interrupted", recovery_status="recovered_metadata_only",
                                recovery_wall=_utc(), retention_protected=True,
                                recovered_files=[{"file": p.name, "size_bytes": p.stat().st_size}
                                                 for p in path.parent.glob("*.avi")],
                                error="prior_process_interrupted; media retained without claiming validation")
                    _atomic_json(path, data)
                    self._recovered_count += 1
            except (OSError, ValueError, AttributeError):
                # Corrupt sidecars remain evidence, never deleted or relabeled clean.
                continue
        self._storage_check()

    def _open_segment(self, session: _Session, packet: tuple) -> None:
        if len(session.segments) >= 128:
            raise RecordingError("segment_count_limit")
        self._storage_check()
        _, seq, timestamp, wall, generation, image = packet
        number = len(session.segments)
        name = f"segment-{number:04d}.incomplete.avi"
        segment = {"segment_index": number, "session_id": session.id, "file": name,
                   "file_role": "clean_authoritative", "status": "capturing", "generation": generation,
                   "width": image.shape[1], "height": image.shape[0], "target_fps": self.config.target_fps,
                   "codec": self.config.codec, "first_source_sequence": seq, "last_source_sequence": seq,
                   "first_capture_monotonic": timestamp, "last_capture_monotonic": timestamp,
                   "first_wall": wall, "last_wall": wall, "written_frames": 0, "parent_session_id": session.id}
        with self._lock:
            session.segments.append(segment)
        self._segment, self._segment_session = segment, session
        self._manifest(session)  # Durable identity precedes the first encoded byte.
        self._writer = self._writer_factory(str(self.directory / session.id / name),
                                           cv2.VideoWriter_fourcc(*self.config.codec), self.config.target_fps,
                                           (segment["width"], segment["height"]))
        if not self._writer.isOpened():
            raise RecordingError("video_writer_unavailable")

    def _close_segment(self, reason: str) -> None:
        segment, session, writer = self._segment, self._segment_session, self._writer
        self._segment = self._segment_session = self._writer = None
        if segment is None or session is None:
            return
        with self._lock:
            segment.update(status="finalizing", finalization_reason=reason)
        path = self.directory / session.id / segment["file"]
        try:
            if writer is not None:
                writer.release()
            validation = self._validator(path, segment["width"], segment["height"], segment["written_frames"])
            capture_span = segment["last_capture_monotonic"] - segment["first_capture_monotonic"]
            encoded_span = max(0, segment["written_frames"] - 1) / self.config.target_fps
            degraded = abs(capture_span - encoded_span) > max(0.5, 2 / self.config.target_fps)
            final = path.with_name(path.name.replace(".incomplete", ""))
            os.replace(path, final)
            with self._lock:
                segment.update(validation)
                segment.update(capture_span_seconds=capture_span, encoded_span_seconds=encoded_span,
                               status="degraded" if degraded else "complete", file=final.name)
        except Exception as exc:
            with self._lock:
                segment.update(status="error", error=type(exc).__name__ + ": " + str(exc))
                # Preserve the initiating write failure; validation may only be
                # reporting the empty/truncated file that failure left behind.
                session.error = session.error or segment["error"]
                if session.state == "capturing":
                    self._finish(session, "validation_error", self._clock())
        self._manifest(session)

    def _write_packet(self, packet: tuple) -> None:
        session, seq, timestamp, wall, generation, image = packet
        if session.error:
            return
        segment = self._segment
        if segment is not None:
            reason = ("new_session" if self._segment_session is not session else
                      "camera_generation_or_geometry" if generation != segment["generation"] or image.shape[:2] != (segment["height"], segment["width"]) else
                      "maximum_segment" if timestamp - segment["first_capture_monotonic"] >= self.config.maximum_segment_seconds else None)
            if reason:
                self._close_segment(reason)
                if session.error:
                    return
        if self._segment is None:
            self._open_segment(session, packet)
        # No overlay, resize, crop, FIRE marker, or semantic label can reach here.
        self._writer.write(image)
        with self._lock:
            self._segment.update(last_source_sequence=seq, last_capture_monotonic=timestamp, last_wall=wall,
                                 written_frames=self._segment["written_frames"] + 1)
            session.written_frames += 1

    def _encode_loop(self) -> None:
        try:
            self._recover()
            with self._lock:
                self._ready = True
            checkpoint = monotonic()
            while True:
                with self._lock:
                    sessions = list(self._sessions)
                for session in sessions:
                    if not session.manifest_started:
                        self._manifest(session)
                        session.manifest_started = True
                with self._lock:
                    packet = next((s.pre_packets.popleft() for s in sessions if s.prepared and s.pre_packets), None)
                from_queue = packet is None
                if from_queue:
                    try:
                        packet = self._queue.get(timeout=0.05)
                    except queue.Empty:
                        packet = None
                if packet:
                    session = packet[0]
                    try:
                        self._write_packet(packet)
                    except Exception as exc:
                        with self._lock:
                            session.error = type(exc).__name__ + ": " + str(exc)
                            if session.state == "capturing":
                                self._finish(session, "write_error", self._clock())
                        self._close_segment("write_error")
                    finally:
                        with self._lock:
                            session.queued -= 1
                            self._buffered_bytes -= packet[-1].nbytes
                        if from_queue:
                            self._queue.task_done()
                for session in sessions:
                    with self._lock:
                        done = session.state == "finalizing" and session.queued == 0
                    if done:
                        if self._segment_session is session:
                            self._close_segment(session.stop_reason or "finalized")
                        with self._lock:
                            if not session.written_frames:
                                session.error = session.error or "no_source_frames"
                            missing_tail = bool(session.segments and session.ended_monotonic is not None
                                                and session.ended_monotonic - session.segments[-1]["last_capture_monotonic"] > max(0.5, 2 / self.config.target_fps))
                            if missing_tail:
                                self._event(session, "missing_tail", session.ended_monotonic,
                                            seconds=session.ended_monotonic - session.segments[-1]["last_capture_monotonic"])
                            session.state = "error" if session.error else "interrupted" if session.stop_reason == "shutdown" else (
                                "degraded" if missing_tail or session.queue_dropped or any(s["status"] != "complete" for s in session.segments) else "complete")
                        self._manifest(session)
                        with self._lock:
                            self._last = deepcopy(self._summary(session))
                            self._sessions.remove(session)
                if monotonic() - checkpoint >= 1:
                    self._storage_check()
                    for session in sessions:
                        self._manifest(session)
                    checkpoint = monotonic()
                with self._lock:
                    if self._stop.is_set() and not self._sessions:
                        break
        except Exception as exc:
            with self._lock:
                self._last_error = type(exc).__name__ + ": " + str(exc)
                self._ready = False
                failed_sessions = list(self._sessions)
                for session in failed_sessions:
                    session.error = self._last_error
                    if session.state == "capturing":
                        self._finish(session, "storage_error", self._clock())
                    session.state = "error"
                    session.queue_dropped += session.queued
                    session.pre_packets.clear()
                    session.queued = 0
                while True:
                    try:
                        self._queue.get_nowait()
                        self._queue.task_done()
                    except queue.Empty:
                        break
                self._buffered_bytes = 0
            # Best effort only; a failed disk must not block or affect control.
            try:
                self._close_segment("storage_error")
            except Exception:
                pass
            for session in failed_sessions:
                try:
                    self._manifest(session)
                except Exception:
                    pass  # Prior checkpoint and media remain recoverable.
            with self._lock:
                if failed_sessions:
                    self._last = deepcopy(self._summary(failed_sessions[-1]))
                self._sessions.clear()

    def stop(self) -> None:
        with self._lock:
            self._closed = True
            if self._active:
                self._finish(self._active, "shutdown", self._clock())
        self._stop.set()
        self._wake.set()
        deadline = monotonic() + self.config.shutdown_timeout_seconds
        for thread in (self._collector, self._encoder):
            if thread and thread is not threading.current_thread():
                thread.join(max(0.0, deadline - monotonic()))
        self._shutdown_timed_out = bool(self._encoder and self._encoder.is_alive())
