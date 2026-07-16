from __future__ import annotations

import csv
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

from conftest import write_test_config
from squirrel_shooter.config import load_config
from squirrel_shooter.event_report import generate_reports
from squirrel_shooter.event_storage import (
    EventLogWriter,
    EventRecorder,
    RollingFrameBuffer,
    SessionLog,
    enforce_retention,
    new_event_id,
    recover_incomplete_events,
)
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


def camera_metadata() -> dict[str, object]:
    return {
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
    assert record["provisional_category"] == "small_animal_candidate"
    assert record["ir_mode_if_explicitly_detected_or_configured"] == "unknown"
    assert record["components"][0][0]["contour"]
    assert not active.marker.exists() and active.clip_path.exists() and active.snapshot_path.exists()
    event = json.loads((active.directory / "event.json").read_text(encoding="utf-8"))
    assert event["status"] == "complete" and event["human_review_label"] == ""
    with logs.csv_path.open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle))[0]["event_id"] == active.event_id
    assert json.loads(logs.jsonl_path.read_text(encoding="utf-8").splitlines()[0])["event_id"] == active.event_id
    assert writers[0].released


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


def test_completed_log_files_rotate_without_deleting_active_log(tmp_path: Path) -> None:
    config = configured(tmp_path)
    config = replace(config, logging=replace(config.logging, maximum_active_log_megabytes=0.000001, retained_log_rotations=2))
    logs = EventLogWriter(config)
    logs.append_event({"event_id": "first"})
    logs.append_event({"event_id": "second"})
    assert logs.csv_path.exists() and logs.jsonl_path.exists()
    assert logs.csv_path.with_name(logs.csv_path.name + ".1").exists()
    assert logs.jsonl_path.with_name(logs.jsonl_path.name + ".1").exists()


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
