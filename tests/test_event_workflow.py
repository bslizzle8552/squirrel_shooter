from __future__ import annotations

import csv
import json
import logging
import threading
import time
from logging.handlers import RotatingFileHandler
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import write_test_config
from squirrel_shooter.config import load_config
from squirrel_shooter.event_report import generate_reports
from squirrel_shooter.event_storage import (
    EVENT_COMPONENT_SAMPLE_LIMIT,
    EVENT_FIELDS,
    EVENT_GROUP_SAMPLE_LIMIT,
    EventStorageManager,
    EventLogWriter,
    EventRecorder,
    RollingFrameBuffer,
    SessionLog,
    _AsyncClipWriter,
    enforce_retention,
    new_event_id,
    recover_incomplete_events,
)
from squirrel_shooter.diagnostics import RoutineAccessFilter, configure_logging
from squirrel_shooter.watch_detection import GroupedCandidate, MotionComponent


def configured(tmp_path: Path):
    config = load_config(write_test_config(tmp_path))
    return replace(
        config,
        reporting=replace(config.reporting, directory=tmp_path / "captures" / "reports"),
        retention=replace(config.retention, maximum_storage_megabytes=1, maximum_event_age_days=30, maximum_event_count=2),
    )


def candidate() -> GroupedCandidate:
    part = MotionComponent((10, 12, 20, 15), 250, 280, (20, 19), ((10, 12), (30, 12), (30, 27), (10, 27)))
    return GroupedCandidate(
        (part,), (10, 12, 20, 15), (20, 19), 250, 280, 2.8, 2.8, 1.0,
        track_id=7, persistence_count=3, path=((10, 19), (20, 19)), duration=1.0,
        travel_distance=10, average_speed=10, peak_speed=12, direction="right", coherent_motion=True,
        provisional_category="small_animal_candidate", movement_attributes=("slow", "coherent_travel"),
        heuristic_score=.8, confirmed=True, newly_confirmed=True,
    )


class FakeWriter:
    def __init__(self, path: str, *_: object) -> None:
        self.path = Path(path)
        self.path.write_bytes(b"AVI")
        self.frames = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def write(self, frame: np.ndarray) -> None:
        assert frame.ndim == 3
        self.frames += 1

    def release(self) -> None:
        self.released = True


class BlockingWriter(FakeWriter):
    def __init__(self, path: str, gate: threading.Event, *_: object) -> None:
        super().__init__(path)
        self.gate = gate
        self.write_started = threading.Event()

    def write(self, frame: np.ndarray) -> None:
        self.write_started.set()
        self.gate.wait(timeout=5.0)
        super().write(frame)


class FailingWriter(FakeWriter):
    def write(self, frame: np.ndarray) -> None:
        raise OSError("injected codec failure")


def wait_until(predicate, timeout: float = 2.0) -> bool:  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


def camera_metadata() -> dict[str, object]:
    return {
        "session_id": "session-one",
        "source_camera": "opencv_device_0",
        "camera_device_index": 0,
        "measured_camera_fps": 9.9,
        "requested_width": 1280,
        "requested_height": 720,
        "requested_fps": 30,
        "actual_width": 1280,
        "actual_height": 720,
        "camera_mode_if_known": "unknown",
        "ir_mode_if_explicitly_detected_or_configured": "unknown",
        "low_fps_observed": True,
    }


def test_event_ids_are_unique_and_timestamped() -> None:
    when = datetime(2026, 7, 16, 12, 34, 56, tzinfo=timezone.utc)
    first, second = new_event_id(when), new_event_id(when)
    assert first.startswith("20260716-123456")
    assert first != second


def test_pre_event_buffer_uses_elapsed_time_at_10_and_25_fps() -> None:
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    for fps in (10, 25):
        buffer = RollingFrameBuffer(2.0)
        for index in range(int(3 * fps) + 1):
            buffer.append(index / fps, frame)
        timestamps = [stamp for stamp, _ in buffer.frames()]
        assert 1.95 <= timestamps[-1] - timestamps[0] <= 2.01
        assert len(buffer) in {int(2 * fps), int(2 * fps) + 1}


def test_event_json_csv_and_jsonl_are_completed_and_flushed(tmp_path: Path) -> None:
    config = configured(tmp_path)
    logs = EventLogWriter(config)
    writers: list[FakeWriter] = []

    def factory(path: str, *args: object) -> FakeWriter:
        writer = FakeWriter(path, *args)
        writers.append(writer)
        return writer

    recorder = EventRecorder(config, logs, camera_metadata(), video_writer_factory=factory)
    frame = np.zeros((72, 128, 3), dtype=np.uint8)
    active = recorder.begin(7, candidate(), frame, frame, [(0.0, frame), (0.1, frame)], now=1.0, measured_fps=9.9)
    recorder.update(7, candidate(), frame, now=1.1)
    record = recorder.finish(7, now=4.2)
    assert record["track_id"] == 7
    assert record["provisional_category"] == "small_animal_candidate"
    assert record["ir_mode_if_explicitly_detected_or_configured"] == "unknown"
    assert record["session_id"] == "session-one"
    assert record["capture_method"] == "automatic_motion_event"
    assert record["source_camera"] == "opencv_device_0"
    assert record["snapshot_file_role"] == "annotated_review_frame"
    assert record["clip_file_role"] == "annotated_review_clip"
    assert record["components"][0][0]["contour"]
    assert not active.marker.exists() and active.clip_path.exists() and active.snapshot_path.exists()
    event = json.loads((active.directory / "event.json").read_text(encoding="utf-8"))
    assert event["status"] == "complete" and event["human_review_label"] == ""
    with logs.csv_path.open(newline="", encoding="utf-8") as handle:
        logged = list(csv.DictReader(handle))[0]
        assert logged["event_id"] == active.event_id
        assert logged["track_id"] == "7"
    assert json.loads(logs.jsonl_path.read_text(encoding="utf-8").splitlines()[0])["event_id"] == active.event_id
    assert writers[0].released
    fsync_timing = logs.fsync_timing()
    assert fsync_timing["csv"]["total_count"] == 1
    assert fsync_timing["jsonl"]["total_count"] == 1
    assert fsync_timing["all"]["total_count"] == 2


def test_103410_slow_prebuffer_does_not_recreate_recording_blind_spot(tmp_path: Path) -> None:
    config = configured(tmp_path)
    logs = EventLogWriter(config)
    recorder = EventRecorder(
        config,
        logs,
        camera_metadata(),
        video_writer_factory=lambda path, *_args: FakeWriter(path),
    )
    frame = np.zeros((72, 128, 3), dtype=np.uint8)
    rendering_started = threading.Event()
    release_rendering = threading.Event()
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "field_events" / "20260818-103410-613-77abc4.json")
        .read_text(encoding="utf-8")
    )
    event_started = float(fixture["previous"]["observed_monotonic"])

    def slow_prebuffer():
        rendering_started.set()
        release_rendering.wait(timeout=2.0)
        yield event_started - 0.1, frame

    started = time.monotonic()
    active = recorder.begin(
        7,
        candidate(),
        frame,
        frame,
        slow_prebuffer(),
        now=event_started,
        measured_fps=9.9,
    )
    elapsed = time.monotonic() - started

    assert rendering_started.wait(timeout=0.5)
    assert elapsed < 0.25
    next_detector_started = time.monotonic()
    recorder.update(7, candidate(), frame, now=event_started + 0.1)
    assert time.monotonic() - next_detector_started < 0.1
    release_rendering.set()
    record = recorder.finish(7, now=event_started + 0.1)
    assert record["frames_written"] == 2
    assert record["writer_telemetry"]["supplied_duration_seconds"] == pytest.approx(0.2)
    assert active.clip_path.exists()


def test_event_clip_encoded_duration_tracks_supplied_monotonic_timeline(tmp_path: Path) -> None:
    config = configured(tmp_path)
    logs = EventLogWriter(config)
    writers: list[FakeWriter] = []

    def factory(path: str, *_args: object) -> FakeWriter:
        writer = FakeWriter(path)
        writers.append(writer)
        return writer

    recorder = EventRecorder(config, logs, camera_metadata(), video_writer_factory=factory)
    frame = np.zeros((72, 128, 3), dtype=np.uint8)
    start = 100.0
    recorder.begin(
        7,
        candidate(),
        frame,
        frame,
        ((start, frame), (start + 0.08, frame)),
        now=start + 0.2,
        measured_fps=25.0,
    )
    for timestamp in (start + 0.37, start + 0.61, start + 1.2):
        recorder.update(7, candidate(), frame, now=timestamp)

    record = recorder.finish(7, now=start + 1.2)
    telemetry = record["writer_telemetry"]
    supplied = float(telemetry["supplied_duration_seconds"])
    encoded = float(telemetry["encoded_duration_seconds"])
    output_fps = float(telemetry["output_fps"])

    assert output_fps == config.motion.target_fps
    assert supplied == pytest.approx(1.2)
    assert encoded == pytest.approx(writers[0].frames / output_fps)
    assert abs(encoded - supplied) <= 1.0 / output_fps
    assert telemetry["timestamp_regressions"] == 0


def test_event_metadata_bounds_samples_but_aggregates_every_observation(tmp_path: Path) -> None:
    config = configured(tmp_path)
    logs = EventLogWriter(config)
    recorder = EventRecorder(
        config,
        logs,
        camera_metadata(),
        video_writer_factory=lambda path, *_args: FakeWriter(path),
    )
    frame = np.zeros((72, 128, 3), dtype=np.uint8)
    recorder.begin(7, candidate(), frame, frame, (), now=1.0, measured_fps=10.0)
    for index in range(300):
        recorder.update(7, candidate(), frame, now=1.0 + ((index + 1) / 1000.0))

    record = recorder.finish(7, now=1.301)

    assert record["group_sample_count"] == 301
    assert record["group_samples_retained"] == EVENT_GROUP_SAMPLE_LIMIT
    assert record["group_samples_omitted"] == 301 - EVENT_GROUP_SAMPLE_LIMIT
    assert len(record["components"]) == EVENT_COMPONENT_SAMPLE_LIMIT
    assert record["group_samples"][0]["sample_index"] == 0
    assert record["group_samples"][-1]["sample_index"] == 300
    assert all("component_blobs" not in sample for sample in record["group_samples"])
    assert all("recent_centroid_path" not in sample for sample in record["group_samples"])
    assert record["average_area"] == 280.0
    assert record["group_sample_storage"]["aggregates_cover_all_samples"] is True


def test_event_writer_full_queue_keeps_newest_frame_without_blocking(tmp_path: Path) -> None:
    gate = threading.Event()
    raw = BlockingWriter(str(tmp_path / "blocked.avi"), gate)
    writer = _AsyncClipWriter(raw, (), queue_limit=2, no_progress_seconds=0.05)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    assert writer.write(frame).accepted
    assert raw.write_started.wait(timeout=0.5)
    assert writer.write(frame).accepted
    assert writer.write(frame).accepted
    started = time.monotonic()
    outcome = writer.write(frame)
    elapsed = time.monotonic() - started

    assert elapsed < 0.1
    assert outcome.accepted and outcome.degraded
    assert outcome.reason == "queue_full_dropped_oldest"
    status = writer.status()
    assert status.queue_high_water == 2
    assert status.frames_dropped == 1
    assert status.blocking_put_attempts == 0
    assert status.blocked_put_seconds == 0.0
    assert status.nonblocking_enqueue_timing["total_count"] == 4
    assert wait_until(lambda: writer.status().state == "stalled", timeout=0.5)

    gate.set()
    completed = writer.release(timeout=1.0)
    assert completed.state == "finished"
    assert completed.frames_written == 3


def test_event_writer_stop_timeout_is_bounded_and_observable(tmp_path: Path) -> None:
    gate = threading.Event()
    raw = BlockingWriter(str(tmp_path / "stalled.avi"), gate)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    writer = _AsyncClipWriter(raw, (frame,), no_progress_seconds=0.01)
    assert raw.write_started.wait(timeout=0.5)

    started = time.monotonic()
    status = writer.release(timeout=0.05)
    elapsed = time.monotonic() - started

    assert elapsed < 0.2
    assert status.state == "stalled"
    assert status.thread_alive is True
    assert status.join_timed_out is True

    gate.set()
    assert wait_until(lambda: not writer.status().thread_alive)
    assert writer.release(timeout=0.2).state == "finished"


def test_event_writer_detects_hung_inflight_write_with_empty_queue(tmp_path: Path) -> None:
    gate = threading.Event()
    raw = BlockingWriter(str(tmp_path / "inflight-stall.avi"), gate)
    writer = _AsyncClipWriter(raw, (), queue_limit=1, no_progress_seconds=0.01)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    assert writer.write(frame).accepted
    assert raw.write_started.wait(timeout=0.5)
    assert wait_until(lambda: writer.status().state == "stalled", timeout=0.5)
    status = writer.status()

    assert status.queue_depth == 0
    assert status.write_in_progress is True
    assert status.thread_alive is True
    gate.set()
    assert writer.release(timeout=1.0).state == "finished"


def test_event_queue_degradation_is_persisted_truthfully(tmp_path: Path) -> None:
    config = configured(tmp_path)
    logs = EventLogWriter(config)
    gate = threading.Event()
    writers: list[BlockingWriter] = []

    def factory(path: str, *args: object) -> BlockingWriter:
        writer = BlockingWriter(path, gate, *args)
        writers.append(writer)
        return writer

    recorder = EventRecorder(config, logs, camera_metadata(), video_writer_factory=factory)
    frame = np.zeros((72, 128, 3), dtype=np.uint8)
    active = recorder.begin(7, candidate(), frame, frame, (), now=1.0, measured_fps=9.9)
    assert writers[0].write_started.wait(timeout=0.5)

    for index in range(12):
        recorder.update(7, candidate(), frame, now=1.1 + (index / 10))

    gate.set()
    record = recorder.finish(7, now=4.2)
    persisted = json.loads((active.directory / "event.json").read_text(encoding="utf-8"))

    assert record["status"] == "complete"
    assert record["recording_status"] == "degraded"
    assert record["frames_dropped"] > 0
    assert record["writer_telemetry"]["queue_high_water"] == 8
    assert record["writer_telemetry"]["blocking_put_attempts"] == 0
    assert record["writer_telemetry"]["blocked_put_seconds"] == 0.0
    assert record["writer_telemetry"]["nonblocking_enqueue_timing"]["total_count"] == 12
    assert persisted["recording_status"] == "degraded"
    assert not active.marker.exists()
    assert active.clip_path.exists()


def test_event_writer_failure_keeps_partial_marker_and_failed_metadata(tmp_path: Path) -> None:
    config = configured(tmp_path)
    logs = EventLogWriter(config)
    recorder = EventRecorder(
        config,
        logs,
        camera_metadata(),
        video_writer_factory=lambda path, *_args: FailingWriter(path),
    )
    frame = np.zeros((72, 128, 3), dtype=np.uint8)
    active = recorder.begin(7, candidate(), frame, frame, (), now=1.0, measured_fps=9.9)
    assert wait_until(lambda: active.writer.status().state == "failed")

    recorder.update(7, candidate(), frame, now=1.1)
    record = recorder.finish(7, now=4.2)
    persisted = json.loads((active.directory / "event.json").read_text(encoding="utf-8"))

    assert record["status"] == "recording_failed"
    assert record["recording_status"] == "failed"
    assert "injected codec failure" in str(record["writer_telemetry"]["error"])
    assert record["clip_path"] is None
    assert active.marker.exists()
    assert persisted["status"] == "recording_failed"


def test_event_reacquisition_diagnostics_are_bounded(tmp_path: Path) -> None:
    config = configured(tmp_path)
    logs = EventLogWriter(config)
    recorder = EventRecorder(
        config,
        logs,
        camera_metadata(),
        video_writer_factory=lambda path, *_args: FakeWriter(path),
    )
    frame = np.zeros((72, 128, 3), dtype=np.uint8)
    active = recorder.begin(7, candidate(), frame, frame, (), now=1.0, measured_fps=9.9)

    for sequence in range(15):
        recorder.record_reacquisition_diagnostic(7, {"sequence": sequence})

    record = recorder.finish(7, now=4.2)
    assert [item["sequence"] for item in record["reacquisition_diagnostics"]] == list(range(3, 15))
    persisted = json.loads((active.directory / "event.json").read_text(encoding="utf-8"))
    assert persisted["reacquisition_diagnostics"] == record["reacquisition_diagnostics"]


def test_rejection_and_session_logs_capture_required_counters(tmp_path: Path) -> None:
    config = configured(tmp_path)
    logs = EventLogWriter(config)
    logs.append_rejection({"timestamp": "2026-07-16T12:00:00-04:00", "reason": "excessive_zone_motion"})
    session = SessionLog(config, {"requested_width": 1280})
    session.data["camera_open_result"] = "success"
    session.increment("raw_contours", 4)
    session.increment("grouped_candidates", 2)
    session.increment("confirmed_events")
    session.reject("excessive_zone_motion")
    session.sample_fps(9.8)
    session.sample_fps(10.1)
    session.finish(clean=True)
    payload = json.loads(session.path.read_text(encoding="utf-8"))
    assert payload["clean_shutdown"] is True
    assert payload["average_measured_fps"] == 9.95
    assert payload["rejected_by_filter"]["excessive_zone_motion"] == 1
    assert json.loads(logs.rejection_path.read_text(encoding="utf-8"))["reason"] == "excessive_zone_motion"


def test_session_metrics_and_details_remain_bounded_for_long_runs(tmp_path: Path) -> None:
    session = SessionLog(configured(tmp_path), {"requested_width": 1280})

    for index in range(10_000):
        session.sample_fps(9.0 + (index % 3))
    session.add_retention_actions({"event_id": str(index)} for index in range(150))
    for index in range(150):
        session.add_exception(f"failure-{index}")
    session.finish(clean=True)

    payload = json.loads(session.path.read_text(encoding="utf-8"))
    assert not hasattr(session, "_fps_samples")
    assert session._fps_sample_count == 10_000
    assert payload["minimum_measured_fps"] == 9.0
    assert payload["maximum_measured_fps"] == 11.0
    assert payload["retention_action_count"] == 150
    assert len(payload["retention_actions"]) == 100
    assert payload["retention_actions"][0]["event_id"] == "50"
    assert payload["exception_count"] == 150
    assert len(payload["exception_details"]) == 100
    assert payload["exception_details"][0] == "failure-50"


def test_completed_log_files_rotate_without_deleting_active_log(tmp_path: Path) -> None:
    config = configured(tmp_path)
    config = replace(config, logging=replace(config.logging, maximum_active_log_megabytes=0.000001, retained_log_rotations=2))
    logs = EventLogWriter(config)
    logs.append_event({"event_id": "first"})
    logs.append_event({"event_id": "second"})
    assert logs.csv_path.exists() and logs.jsonl_path.exists()
    assert logs.csv_path.with_name(logs.csv_path.name + ".1").exists()
    assert logs.jsonl_path.with_name(logs.jsonl_path.name + ".1").exists()


def test_event_csv_schema_upgrade_rotates_legacy_header_before_appending(tmp_path: Path) -> None:
    logs = EventLogWriter(configured(tmp_path))
    legacy_fields = [field for field in EVENT_FIELDS if field != "track_id"]
    logs.csv_path.parent.mkdir(parents=True, exist_ok=True)
    with logs.csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=legacy_fields)
        writer.writeheader()
        writer.writerow({"event_id": "legacy-event", "session_id": "legacy-session"})

    logs.append_event({"event_id": "new-event", "track_id": 7, "session_id": "new-session"})

    rotated = logs.csv_path.with_name(logs.csv_path.name + ".1")
    with rotated.open(newline="", encoding="utf-8") as handle:
        legacy_rows = list(csv.DictReader(handle))
    with logs.csv_path.open(newline="", encoding="utf-8") as handle:
        current_reader = csv.DictReader(handle)
        current_rows = list(current_reader)

    assert legacy_rows[0]["event_id"] == "legacy-event"
    assert "track_id" not in legacy_rows[0]
    assert current_reader.fieldnames == list(EVENT_FIELDS)
    assert current_rows[0]["event_id"] == "new-event"
    assert current_rows[0]["track_id"] == "7"
    assert current_rows[0]["session_id"] == "new-session"


def test_application_log_is_bounded_and_only_successful_routine_access_is_suppressed(tmp_path: Path) -> None:
    config = configured(tmp_path)
    root = logging.getLogger()
    existing = set(root.handlers)
    werkzeug = logging.getLogger("werkzeug")
    previous_werkzeug_level = werkzeug.level
    existing_werkzeug_filters = set(werkzeug.filters)

    try:
        path = configure_logging(config.logging, max_log_files=3)
        created = [handler for handler in root.handlers if handler not in existing]
        rotating = next(handler for handler in created if isinstance(handler, RotatingFileHandler))

        assert path is not None and path.exists()
        assert rotating.maxBytes == int(config.logging.maximum_active_log_megabytes * 1024 * 1024)
        assert rotating.backupCount == config.logging.retained_log_rotations
        assert werkzeug.level == logging.INFO
        access_filter = next(item for item in werkzeug.filters if isinstance(item, RoutineAccessFilter))
        routine_ok = logging.LogRecord(
            "werkzeug",
            logging.INFO,
            "",
            0,
            '127.0.0.1 - - [date] "GET /api/status HTTP/1.1" 200 -',
            (),
            None,
        )
        rejected_control = logging.LogRecord(
            "werkzeug",
            logging.INFO,
            "",
            0,
            '127.0.0.1 - - [date] "POST /api/manual-control/fire HTTP/1.1" 403 -',
            (),
            None,
        )
        manual_poll = logging.LogRecord(
            "werkzeug",
            logging.INFO,
            "",
            0,
            '127.0.0.1 - - [date] "GET /api/manual-control HTTP/1.1" 200 -',
            (),
            None,
        )
        failed_poll = logging.LogRecord(
            "werkzeug",
            logging.INFO,
            "",
            0,
            '127.0.0.1 - - [date] "GET /api/status HTTP/1.1" 500 -',
            (),
            None,
        )
        assert access_filter.filter(routine_ok) is False
        assert access_filter.filter(manual_poll) is False
        assert access_filter.filter(rejected_control) is True
        assert access_filter.filter(failed_poll) is True
    finally:
        for handler in list(root.handlers):
            if handler not in existing:
                root.removeHandler(handler)
                handler.close()
        for access_filter in list(werkzeug.filters):
            if access_filter not in existing_werkzeug_filters:
                werkzeug.removeFilter(access_filter)
        werkzeug.setLevel(previous_werkzeug_level)


def test_incomplete_event_recovery_preserves_files_and_marks_folder(tmp_path: Path) -> None:
    directory = tmp_path / "events" / "2026-07-16" / "interrupted-event"
    directory.mkdir(parents=True)
    (directory / ".incomplete").write_text("active", encoding="utf-8")
    (directory / "clip.incomplete.avi").write_bytes(b"partial")
    recovered = recover_incomplete_events(tmp_path / "events")
    assert recovered == [directory]
    assert (directory / ".recovered-incomplete").exists()
    assert json.loads((directory / "event.json").read_text(encoding="utf-8"))["status"] == "interrupted_recovered"
    assert (directory / "clip.incomplete.avi").exists()


def make_complete_event(root: Path, name: str, when: datetime, size: int = 10) -> Path:
    directory = root / when.strftime("%Y-%m-%d") / name
    directory.mkdir(parents=True)
    (directory / "payload.bin").write_bytes(b"x" * size)
    (directory / "event.json").write_text(json.dumps({"status": "complete", "start_timestamp": when.isoformat(), "event_id": name}), encoding="utf-8")
    return directory


def test_retention_deletes_oldest_complete_events_only(tmp_path: Path) -> None:
    config = configured(tmp_path).retention
    root = tmp_path / "events"
    now = datetime(2026, 7, 16, tzinfo=timezone.utc)
    old = make_complete_event(root, "old", now - timedelta(days=40))
    middle = make_complete_event(root, "middle", now - timedelta(days=2))
    newest = make_complete_event(root, "newest", now - timedelta(days=1))
    protected = root / "2026-07-16" / "recovered"
    protected.mkdir(parents=True)
    (protected / ".recovered-incomplete").write_text("", encoding="utf-8")
    (protected / "event.json").write_text(json.dumps({"status": "complete", "start_timestamp": (now - timedelta(days=100)).isoformat()}), encoding="utf-8")
    actions = enforce_retention(root, config, active_directories={middle}, now=now)
    assert not old.exists()
    assert middle.exists() and newest.exists() and protected.exists()
    assert actions[0]["event_id"] == "old"


def test_storage_manager_finalizes_off_thread_and_exposes_pending_pins(tmp_path: Path) -> None:
    config = configured(tmp_path)
    logs = EventLogWriter(config)
    gate = threading.Event()
    writers: list[BlockingWriter] = []

    def factory(path: str, *args: object) -> BlockingWriter:
        writer = BlockingWriter(path, gate, *args)
        writers.append(writer)
        return writer

    recorder = EventRecorder(config, logs, camera_metadata(), video_writer_factory=factory)
    frame = np.zeros((72, 128, 3), dtype=np.uint8)
    active = recorder.begin(7, candidate(), frame, frame, (), now=1.0, measured_fps=9.9)
    assert writers[0].write_started.wait(timeout=0.5)
    manager = EventStorageManager(
        recorder,
        config.camera.output_directory / "events",
        config.retention,
    )
    manager.start()
    external_pin = tmp_path / "classifier-still-writing"
    manager.pin(external_pin)

    started = time.monotonic()
    submission = manager.request_finalize(7, now=4.2)
    elapsed = time.monotonic() - started

    assert submission.accepted
    assert elapsed < 0.1
    assert 7 not in recorder.active
    assert active.directory.resolve() in manager.pending_directories()
    assert external_pin.resolve() in manager.protected_directories()

    gate.set()
    assert wait_until(lambda: manager.status().completed_finalizations == 1)
    results = manager.poll_results()
    assert len(results) == 1
    assert results[0].kind == "finalize" and results[0].success
    assert results[0].lease_id is None
    assert results[0].record is not None
    assert results[0].record["recording_status"] == "success"
    assert manager.pending_directories() == set()
    status = manager.status()
    assert status.leased_directories == 0
    assert status.event_log_fsync_timing["csv"]["total_count"] == 1
    assert status.event_log_fsync_timing["jsonl"]["total_count"] == 1
    assert status.event_log_fsync_timing["all"]["total_count"] == 2
    manager.unpin(external_pin)
    assert external_pin.resolve() not in manager.protected_directories()
    assert manager.stop(timeout=1.0).thread_alive is False


def test_storage_manager_contains_finalizer_failure_and_preserves_marker(tmp_path: Path) -> None:
    config = configured(tmp_path)
    logs = EventLogWriter(config)
    recorder = EventRecorder(
        config,
        logs,
        camera_metadata(),
        video_writer_factory=lambda path, *_args: FakeWriter(path),
        image_writer=lambda _path, _frame: False,
    )
    frame = np.zeros((72, 128, 3), dtype=np.uint8)
    active = recorder.begin(7, candidate(), frame, frame, (), now=1.0, measured_fps=10.0)
    manager = EventStorageManager(recorder, config.camera.output_directory / "events", config.retention)
    manager.start()

    assert manager.request_finalize(7, now=1.1).accepted
    assert wait_until(lambda: manager.status().failed_finalizations == 1)
    result = manager.poll_results()[0]

    assert result.kind == "finalize" and result.success is False
    assert "Could not write" in str(result.error)
    assert active.marker.exists()
    assert manager.pending_directories() == set()
    assert manager.stop(timeout=1.0).thread_alive is False


def test_storage_manager_hung_retention_has_bounded_stop_and_queue_rejection(tmp_path: Path) -> None:
    config = configured(tmp_path)
    logs = EventLogWriter(config)
    recorder = EventRecorder(
        config,
        logs,
        camera_metadata(),
        video_writer_factory=lambda path, *_args: FakeWriter(path),
    )
    frame = np.zeros((72, 128, 3), dtype=np.uint8)
    recorder.begin(7, candidate(), frame, frame, (), now=1.0, measured_fps=10.0)
    recorder.begin(8, candidate(), frame, frame, (), now=1.0, measured_fps=10.0)
    retention_started = threading.Event()
    release_retention = threading.Event()

    def slow_retention(*_args: object, **_kwargs: object) -> list[dict[str, object]]:
        retention_started.set()
        release_retention.wait(timeout=5.0)
        return []

    manager = EventStorageManager(
        recorder,
        config.camera.output_directory / "events",
        config.retention,
        queue_limit=1,
        retention_runner=slow_retention,
    )
    manager.start()
    assert manager.request_retention().accepted
    assert retention_started.wait(timeout=0.5)
    assert manager.request_finalize(7, now=1.1).accepted

    started = time.monotonic()
    rejected = manager.request_finalize(8, now=1.1)
    assert time.monotonic() - started < 0.1
    assert rejected.accepted is False and rejected.reason == "finalization_queue_full"
    assert 8 in recorder.active

    stop_started = time.monotonic()
    stalled = manager.stop(timeout=0.05)
    assert time.monotonic() - stop_started < 0.2
    assert stalled.thread_alive and stalled.join_timed_out
    assert manager.request_stop_pending_writers() == 1

    release_retention.set()
    assert manager.stop(timeout=1.0).thread_alive is False
    recorder.finish(8, now=1.2)
    status = manager.status()
    assert status.queue_high_water == 1
    assert status.retention_timing["total_count"] == 1
    assert status.finalization_timing["total_count"] == 1


def test_storage_manager_retention_never_deletes_pinned_directory(tmp_path: Path) -> None:
    config = configured(tmp_path)
    logs = EventLogWriter(config)
    recorder = EventRecorder(
        config,
        logs,
        camera_metadata(),
        video_writer_factory=lambda path, *_args: FakeWriter(path),
    )
    root = config.camera.output_directory / "events"
    now = datetime(2026, 7, 16, tzinfo=timezone.utc)
    old = make_complete_event(root, "pinned-old", now - timedelta(days=40))
    manager = EventStorageManager(recorder, root, config.retention)
    manager.pin(old)
    manager.start()

    assert manager.request_retention(now=now).accepted
    assert wait_until(lambda: manager.status().completed_retention_runs == 1)
    first = manager.poll_results()
    assert first[0].kind == "retention" and first[0].success
    assert old.exists()

    manager.unpin(old)
    assert manager.request_retention(now=now).accepted
    assert wait_until(lambda: manager.status().completed_retention_runs == 2)
    second = manager.poll_results()
    assert second[0].retention_actions[0]["event_id"] == "pinned-old"
    assert not old.exists()
    assert manager.stop(timeout=1.0).thread_alive is False


def test_storage_result_claim_atomically_leases_directory_across_retention(
    tmp_path: Path,
) -> None:
    config = configured(tmp_path)
    logs = EventLogWriter(config)
    recorder = EventRecorder(
        config,
        logs,
        camera_metadata(),
        video_writer_factory=lambda path, *_args: FakeWriter(path),
    )
    frame = np.zeros((72, 128, 3), dtype=np.uint8)
    active = recorder.begin(7, candidate(), frame, frame, (), now=1.0, measured_fps=10.0)
    manager = EventStorageManager(
        recorder,
        config.camera.output_directory / "events",
        config.retention,
    )
    manager.start()

    assert manager.request_finalize(7, now=1.1).accepted
    assert wait_until(lambda: manager.status().completed_finalizations == 1)
    claimed = manager.poll_results(claim_finalize_directories=True)

    assert len(claimed) == 1
    assert claimed[0].directory == active.directory
    assert claimed[0].lease_id is not None
    assert manager.status().leased_directories == 1
    assert active.directory.resolve() in manager.protected_directories()

    future = datetime.now().astimezone() + timedelta(days=40)
    assert manager.request_retention(now=future).accepted
    assert wait_until(lambda: manager.status().completed_retention_runs == 1)
    retention_while_leased = manager.poll_results()
    assert retention_while_leased[0].retention_actions == ()
    assert active.directory.exists()

    assert manager.release_lease(claimed[0].lease_id)
    assert manager.status().leased_directories == 0
    assert manager.request_retention(now=future).accepted
    assert wait_until(lambda: manager.status().completed_retention_runs == 2)
    retention_after_release = manager.poll_results()
    assert retention_after_release[0].retention_actions[0]["event_id"] == active.event_id
    assert not active.directory.exists()
    assert manager.stop(timeout=1.0).thread_alive is False


def test_report_and_review_csv_generation_preserve_human_labels(tmp_path: Path) -> None:
    config = configured(tmp_path)
    event_dir = config.camera.output_directory / "events" / "2026-07-16" / "event-one"
    event_dir.mkdir(parents=True)
    snapshot = event_dir / "snapshot.jpg"
    clip = event_dir / "clip.avi"
    cv2.imwrite(str(snapshot), np.zeros((20, 30, 3), dtype=np.uint8))
    clip.write_bytes(b"AVI")
    payload = {
        "status": "complete", "event_id": "event-one", "start_timestamp": "2026-07-16T08:15:00-04:00",
        "end_timestamp": "2026-07-16T08:15:04-04:00", "duration": 4, "snapshot_path": str(snapshot), "clip_path": str(clip),
        "provisional_category": "small_animal_candidate", "movement_attributes": ["moderate", "coherent_travel"],
        "total_centroid_travel": 80, "average_pixel_speed": 20, "component_count": 2, "grouping_confidence": .85,
        "measured_camera_fps": 9.9,
    }
    (event_dir / "event.json").write_text(json.dumps(payload), encoding="utf-8")
    html_path, markdown_path, review_path = generate_reports(config)
    html_text = html_path.read_text(encoding="utf-8")
    assert "Chronological event gallery" in html_text and "small_animal_candidate" in html_text
    assert "not species recognition" in html_text
    assert "Events by provisional category" in markdown_path.read_text(encoding="utf-8")
    rows = list(csv.DictReader(review_path.open(encoding="utf-8-sig")))
    rows[0]["human_review_label"] = "squirrel"
    rows[0]["human_review_notes"] = "human review only"
    with review_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    generate_reports(config)
    preserved = list(csv.DictReader(review_path.open(encoding="utf-8-sig")))[0]
    assert preserved["human_review_label"] == "squirrel"
    assert preserved["human_review_notes"] == "human review only"
