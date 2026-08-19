from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from squirrel_shooter.camera_service import FramePacket
from squirrel_shooter.manual_fire_recording import (
    ManualFireEvent,
    ManualFireRecorder,
    ManualFireRecordingConfig,
    calculate_crop_bounds,
    crop_and_zoom,
)


@pytest.mark.parametrize(
    ("center_x", "center_y"),
    [(0, 360), (1279, 360), (640, 0), (640, 719)],
)
def test_crop_bounds_clamp_at_every_frame_edge(center_x: int, center_y: int) -> None:
    bounds = calculate_crop_bounds(1280, 720, center_x, center_y, 2.0)

    assert 0 <= bounds.x <= 1280 - bounds.width
    assert 0 <= bounds.y <= 720 - bounds.height
    assert (bounds.width, bounds.height) == (640, 360)
    assert bounds.width * 720 == bounds.height * 1280


def test_crop_zoom_preserves_source_dimensions_and_aspect_ratio() -> None:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    bounds = calculate_crop_bounds(1280, 720, 640, 360, 2.0)

    zoomed = crop_and_zoom(frame, bounds)

    assert zoomed.shape == frame.shape
    assert bounds.width / bounds.height == 16 / 9


class FakeCamera:
    def __init__(self, frames: list[FramePacket]) -> None:
        self.frames = frames

    def buffered_frame_metadata(
        self,
        since_monotonic: float,
        *,
        until_monotonic: float | None = None,
        after_sequence: int = -1,
    ) -> list[FramePacket]:
        return [
            frame
            for frame in self.frames
            if frame.received_monotonic >= since_monotonic
            and (until_monotonic is None or frame.received_monotonic <= until_monotonic)
            and frame.sequence > after_sequence
        ]

    def buffered_frames(
        self,
        since_monotonic: float,
        *,
        until_monotonic: float | None = None,
        after_sequence: int = -1,
        copy: bool = True,
    ) -> list[FramePacket]:
        del copy
        return [
            frame
            for frame in self.frames
            if frame.received_monotonic >= since_monotonic
            and (until_monotonic is None or frame.received_monotonic <= until_monotonic)
            and frame.sequence > after_sequence
        ]

    def wait_for_frame(
        self,
        after_sequence: int,
        timeout: float | None = None,
        *,
        copy: bool = True,
    ) -> None:
        del after_sequence, timeout, copy
        return None

    def status(self) -> SimpleNamespace:
        return SimpleNamespace(fps=10.0, reported_fps=10.0)


class FakeWriter:
    def __init__(self, path: str, written: dict[str, list[np.ndarray]]) -> None:
        self.path = path
        self.written = written.setdefault(path, [])
        Path(path).write_bytes(b"open")

    def isOpened(self) -> bool:
        return True

    def write(self, frame: np.ndarray) -> None:
        self.written.append(frame.copy())

    def release(self) -> None:
        return None


def manual_event() -> ManualFireEvent:
    return ManualFireEvent(
        event_id="manual-fire-test",
        timestamp="2026-08-09T19:32:00.000-04:00",
        fire_started_monotonic=2.0,
        fire_completed_monotonic=2.25,
        pan=70.0,
        tilt=88.0,
        fire_pulse_seconds=0.25,
        crop_center_x=1,
        crop_center_y=1,
        crop_center_source="calibrated_target_pixel",
    )


def test_recorder_associates_full_and_zoom_clips_with_one_manual_event(tmp_path: Path) -> None:
    frames = [
        FramePacket(index, np.full((36, 64, 3), index * 30, dtype=np.uint8), "stamp", float(index))
        for index in range(3)
    ]
    written: dict[str, list[np.ndarray]] = {}

    def writer_factory(path: str, _fourcc: int, _fps: float, _size: tuple[int, int]) -> FakeWriter:
        return FakeWriter(path, written)

    def image_writer(path: str, _frame: np.ndarray) -> bool:
        Path(path).write_bytes(b"jpeg")
        return True

    recorder = ManualFireRecorder(
        FakeCamera(frames),
        tmp_path,
        ManualFireRecordingConfig(pre_roll_seconds=2.0, post_roll_seconds=0.01, crop_center_x=32, crop_center_y=18),
        video_writer_factory=writer_factory,
        image_writer=image_writer,
        clock=lambda: 10.0,
    )
    recorder.record(manual_event())
    recorder.close()

    directory = tmp_path / "events" / "2026-08-09" / "manual-fire-test"
    metadata = json.loads((directory / "event.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "complete"
    assert metadata["recording_status"] == "success"
    assert metadata["event_type"] == "manual_fire"
    assert metadata["pre_roll_available"] is True
    assert metadata["timing_basis"] == "monotonic_elapsed_time"
    assert Path(metadata["clip_path"]).parent == directory
    assert Path(metadata["full_frame_clip_path"]).parent == directory
    assert Path(metadata["clip_path"]).name == "manual_fire_zoom.avi"
    assert Path(metadata["full_frame_clip_path"]).name == "manual_fire_full.avi"
    assert metadata["zoom_factor"] == 2.0
    assert metadata["crop_center_pixel"]["source"] == "calibrated_target_pixel"
    assert len(written) == 2
    assert all(len(frames_written) == 3 for frames_written in written.values())
    assert all(frame.shape == (36, 64, 3) for frames_written in written.values() for frame in frames_written)


def test_recorder_writes_auto_fire_files_and_protects_record_identity(tmp_path: Path) -> None:
    frames = [
        FramePacket(index, np.full((36, 64, 3), index * 30, dtype=np.uint8), "stamp", float(index))
        for index in range(3)
    ]
    written: dict[str, list[np.ndarray]] = {}

    def writer_factory(path: str, _fourcc: int, _fps: float, _size: tuple[int, int]) -> FakeWriter:
        return FakeWriter(path, written)

    def image_writer(path: str, _frame: np.ndarray) -> bool:
        Path(path).write_bytes(b"jpeg")
        return True

    event = replace(
        manual_event(),
        event_id="auto-fire-test",
        event_type="auto_fire",
        evidence={
            "event_id": "motion-event-source",
            "event_type": "unsafe-override",
            "capture_method": "unsafe-override",
            "pan": 999,
            "classifier_label": "dog",
            "classifier_confidence": 0.91,
            "track_id": 7,
            "target_pixel_x": 640,
            "target_pixel_y": 360,
            "calculated_pan": 70.0,
            "calculated_tilt": 88.0,
            "cooldown_seconds": 5.0,
            "safe_bound_result": "inside_calibrated_area",
            "interpolation_result": "success",
        },
    )
    recorder = ManualFireRecorder(
        FakeCamera(frames),
        tmp_path,
        ManualFireRecordingConfig(
            pre_roll_seconds=2.0,
            post_roll_seconds=0.01,
            crop_center_x=32,
            crop_center_y=18,
        ),
        video_writer_factory=writer_factory,
        image_writer=image_writer,
        clock=lambda: 10.0,
    )

    recorder.record(event)
    recorder.close()

    directory = tmp_path / "events" / "2026-08-09" / "auto-fire-test"
    metadata = json.loads((directory / "event.json").read_text(encoding="utf-8"))
    assert metadata["event_id"] == "auto-fire-test"
    assert metadata["source_event_id"] == "motion-event-source"
    assert metadata["event_type"] == "auto_fire"
    assert metadata["capture_method"] == "auto_fire"
    assert metadata["provisional_category"] == "auto_fire"
    assert metadata["pan"] == 70.0
    assert metadata["classifier_label"] == "dog"
    assert metadata["classifier_confidence"] == 0.91
    assert metadata["track_id"] == 7
    assert metadata["cooldown_seconds"] == 5.0
    assert metadata["safe_bound_result"] == "inside_calibrated_area"
    assert Path(metadata["clip_path"]).name == "auto_fire_zoom.avi"
    assert Path(metadata["full_frame_clip_path"]).name == "auto_fire_full.avi"
    assert metadata["snapshot_file_role"] == "auto_fire_zoom_review_frame"
    assert metadata["clip_file_role"] == "auto_fire_zoom_replay"
    assert metadata["full_frame_file_role"] == "auto_fire_full_frame_evidence"
    assert len(written) == 2


def test_recorder_rejects_a_third_recording_while_two_real_slots_are_outstanding(
    tmp_path: Path,
) -> None:
    frames = [
        FramePacket(index, np.full((36, 64, 3), index, dtype=np.uint8), "stamp", float(index))
        for index in range(3)
    ]
    encoder_blocked = threading.Event()
    release_encoder = threading.Event()
    written: dict[str, list[np.ndarray]] = {}

    class BlockingWriter(FakeWriter):
        def write(self, frame: np.ndarray) -> None:
            if "queue-slot-one" in self.path and not encoder_blocked.is_set():
                encoder_blocked.set()
                assert release_encoder.wait(5.0)
            super().write(frame)

    def writer_factory(path: str, _fourcc: int, _fps: float, _size: tuple[int, int]) -> FakeWriter:
        return BlockingWriter(path, written)

    def image_writer(path: str, _frame: np.ndarray) -> bool:
        Path(path).write_bytes(b"jpeg")
        return True

    recorder = ManualFireRecorder(
        FakeCamera(frames),
        tmp_path,
        ManualFireRecordingConfig(
            pre_roll_seconds=2.0,
            post_roll_seconds=0.01,
            crop_center_x=32,
            crop_center_y=18,
        ),
        video_writer_factory=writer_factory,
        image_writer=image_writer,
        clock=lambda: 10.0,
    )
    try:
        recorder.record(replace(manual_event(), event_id="queue-slot-one"))
        assert encoder_blocked.wait(2.0)
        recorder.record(replace(manual_event(), event_id="queue-slot-two"))

        with pytest.raises(RuntimeError, match="queue is full"):
            recorder.record(replace(manual_event(), event_id="queue-slot-three"))

        status = recorder.status()
        assert status["outstanding"] == 2
        assert status["collector_queue_capacity"] == 2
        assert status["encoder_queue_capacity"] == 2
        assert status["failed"] == 1
        assert status["rejected"] == 1
        assert status["last_error"] == "Manual fire recording queue is full"
    finally:
        release_encoder.set()
        recorder.close()

    status = recorder.status()
    assert status["outstanding"] == 0
    assert status["completed"] == 2


def test_encoder_backlog_over_ten_point_four_seconds_cannot_evict_reserved_window(
    tmp_path: Path,
) -> None:
    first_window = [
        FramePacket(0, np.full((36, 64, 3), 8, dtype=np.uint8), "stamp", 8.0),
        FramePacket(1, np.full((36, 64, 3), 10, dtype=np.uint8), "stamp", 10.0),
        FramePacket(2, np.full((36, 64, 3), 15, dtype=np.uint8), "stamp", 15.0),
    ]
    second_pre_roll = [
        FramePacket(10, np.full((36, 64, 3), 18, dtype=np.uint8), "stamp", 18.0),
        FramePacket(11, np.full((36, 64, 3), 20, dtype=np.uint8), "stamp", 20.0),
    ]
    second_post_roll = [
        FramePacket(12, np.full((36, 64, 3), 21, dtype=np.uint8), "stamp", 21.0),
        FramePacket(13, np.full((36, 64, 3), 23, dtype=np.uint8), "stamp", 23.0),
        FramePacket(14, np.full((36, 64, 3), 25, dtype=np.uint8), "stamp", 25.0),
    ]

    class BacklogCamera(FakeCamera):
        def __init__(self) -> None:
            super().__init__([*first_window, *second_pre_roll])
            self.now = 15.0
            self.post_roll_collected = threading.Event()

        def wait_for_frame(
            self,
            after_sequence: int,
            timeout: float | None = None,
            *,
            copy: bool = True,
        ) -> FramePacket | None:
            del copy
            packet = next((item for item in second_post_roll if item.sequence > after_sequence), None)
            if packet is None:
                self.now += timeout or 0.0
                return None
            self.frames.append(packet)
            self.now = packet.received_monotonic
            if packet is second_post_roll[-1]:
                self.post_roll_collected.set()
            return packet

    camera = BacklogCamera()
    encoder_blocked = threading.Event()
    release_encoder = threading.Event()
    written: dict[str, list[np.ndarray]] = {}

    class BlockingWriter(FakeWriter):
        def write(self, frame: np.ndarray) -> None:
            if "backlog-first" in self.path and not encoder_blocked.is_set():
                encoder_blocked.set()
                assert release_encoder.wait(5.0)
            super().write(frame)

    def writer_factory(path: str, _fourcc: int, _fps: float, _size: tuple[int, int]) -> FakeWriter:
        return BlockingWriter(path, written)

    def image_writer(path: str, _frame: np.ndarray) -> bool:
        Path(path).write_bytes(b"jpeg")
        return True

    config = ManualFireRecordingConfig(
        pre_roll_seconds=2.0,
        post_roll_seconds=5.0,
        target_fps=1.0,
        crop_center_x=32,
        crop_center_y=18,
    )
    first_event = replace(
        manual_event(),
        event_id="backlog-first",
        fire_started_monotonic=10.0,
        fire_completed_monotonic=10.25,
    )
    second_event = replace(
        manual_event(),
        event_id="backlog-second",
        fire_started_monotonic=20.0,
        fire_completed_monotonic=20.25,
    )
    recorder = ManualFireRecorder(
        camera,
        tmp_path,
        config,
        video_writer_factory=writer_factory,
        image_writer=image_writer,
        clock=lambda: camera.now,
    )
    try:
        recorder.record(first_event)
        assert encoder_blocked.wait(2.0)

        camera.now = 20.0
        recorder.record(second_event)
        assert camera.post_roll_collected.wait(2.0)

        camera.frames.clear()
        camera.now = second_event.fire_started_monotonic + 10.5
        assert camera.now - second_event.fire_started_monotonic > 10.4
    finally:
        release_encoder.set()
        recorder.close()

    directory = tmp_path / "events" / "2026-08-09" / second_event.event_id
    metadata = json.loads((directory / "event.json").read_text(encoding="utf-8"))
    second_clip_frames = [
        frames_written
        for path, frames_written in written.items()
        if second_event.event_id in path
    ]
    assert metadata["status"] == "complete"
    assert metadata["recording_status"] == "success"
    assert metadata["evidence_window_reserved_at_submission"] is True
    assert metadata["actual_pre_roll_seconds"] == 2.0
    assert metadata["actual_post_roll_seconds"] == 5.0
    assert all(len(frames_written) == 5 for frames_written in second_clip_frames)


def test_recorder_collects_post_roll_after_the_short_pre_roll_buffer(tmp_path: Path) -> None:
    pre_roll = [FramePacket(0, np.zeros((36, 64, 3), dtype=np.uint8), "stamp", 9.0)]
    future = [
        FramePacket(index, np.full((36, 64, 3), index, dtype=np.uint8), "stamp", timestamp)
        for index, timestamp in enumerate((10.0, 10.5, 11.0), start=1)
    ]

    class CollectingCamera(FakeCamera):
        def __init__(self) -> None:
            super().__init__(pre_roll)
            self.now = 10.0

        def wait_for_frame(
            self,
            after_sequence: int,
            timeout: float | None = None,
            *,
            copy: bool = True,
        ) -> FramePacket | None:
            del timeout, copy
            packet = next((item for item in future if item.sequence > after_sequence), None)
            if packet is not None:
                self.now = packet.received_monotonic
            return packet

    camera = CollectingCamera()
    written: dict[str, list[np.ndarray]] = {}

    def writer_factory(path: str, _fourcc: int, _fps: float, _size: tuple[int, int]) -> FakeWriter:
        return FakeWriter(path, written)

    def image_writer(path: str, _frame: np.ndarray) -> bool:
        Path(path).write_bytes(b"jpeg")
        return True

    event = replace(manual_event(), event_id="collected-post-roll", fire_started_monotonic=10.0)
    recorder = ManualFireRecorder(
        camera,
        tmp_path,
        ManualFireRecordingConfig(
            pre_roll_seconds=1.0,
            post_roll_seconds=1.0,
            target_fps=2.0,
            crop_center_x=32,
            crop_center_y=18,
        ),
        video_writer_factory=writer_factory,
        image_writer=image_writer,
        clock=lambda: camera.now,
    )

    recorder.record(event)
    recorder.close()

    directory = tmp_path / "events" / "2026-08-09" / event.event_id
    metadata = json.loads((directory / "event.json").read_text(encoding="utf-8"))
    assert metadata["frames_written"] == 4
    assert metadata["duration"] == 2.0
    assert all(len(frames) == 4 for frames in written.values())


@pytest.mark.parametrize(
    ("timestamps", "expected_post_roll"),
    [
        ([8.0, 9.5, 11.0, 14.9], 4.9),
        ([8.0 + index * 0.5 for index in range(15)], 5.0),
    ],
)
def test_recording_playback_duration_uses_elapsed_time_at_variable_frame_rates(
    tmp_path: Path,
    timestamps: list[float],
    expected_post_roll: float,
) -> None:
    frames = [
        FramePacket(index, np.full((36, 64, 3), index, dtype=np.uint8), "stamp", timestamp)
        for index, timestamp in enumerate(timestamps)
    ]
    writer_fps: list[float] = []

    def writer_factory(path: str, _fourcc: int, fps: float, _size: tuple[int, int]) -> FakeWriter:
        writer_fps.append(fps)
        return FakeWriter(path, {})

    def image_writer(path: str, _frame: np.ndarray) -> bool:
        Path(path).write_bytes(b"jpeg")
        return True

    event = replace(
        manual_event(),
        event_id=f"variable-fps-{len(timestamps)}",
        fire_started_monotonic=10.0,
        fire_completed_monotonic=10.25,
    )
    recorder = ManualFireRecorder(
        FakeCamera(frames),
        tmp_path,
        ManualFireRecordingConfig(pre_roll_seconds=2.0, post_roll_seconds=5.0, crop_center_x=32, crop_center_y=18),
        video_writer_factory=writer_factory,
        image_writer=image_writer,
        clock=lambda: 20.0,
    )
    recorder.record(event)
    recorder.close()

    directory = tmp_path / "events" / "2026-08-09" / event.event_id
    metadata = json.loads((directory / "event.json").read_text(encoding="utf-8"))
    assert metadata["duration"] == 7.0
    assert metadata["actual_pre_roll_seconds"] == 2.0
    assert metadata["actual_post_roll_seconds"] == expected_post_roll
    assert metadata["frames_written"] == len(timestamps)
    assert metadata["output_fps"] == pytest.approx(len(timestamps) / 7.0, abs=0.001)
    assert writer_fps == pytest.approx([len(timestamps) / 7.0, len(timestamps) / 7.0])
    assert len(timestamps) / writer_fps[0] == pytest.approx(7.0)


def test_missing_pre_roll_extends_post_roll_to_seven_elapsed_seconds(tmp_path: Path) -> None:
    timestamps = [10.0, 12.0, 14.0, 16.9]
    frames = [
        FramePacket(index, np.full((36, 64, 3), index, dtype=np.uint8), "stamp", timestamp)
        for index, timestamp in enumerate(timestamps)
    ]
    writer_fps: list[float] = []

    def writer_factory(path: str, _fourcc: int, fps: float, _size: tuple[int, int]) -> FakeWriter:
        writer_fps.append(fps)
        return FakeWriter(path, {})

    def image_writer(path: str, _frame: np.ndarray) -> bool:
        Path(path).write_bytes(b"jpeg")
        return True

    event = replace(
        manual_event(),
        event_id="no-pre-roll",
        fire_started_monotonic=10.0,
        fire_completed_monotonic=10.25,
    )
    recorder = ManualFireRecorder(
        FakeCamera(frames),
        tmp_path,
        ManualFireRecordingConfig(pre_roll_seconds=2.0, post_roll_seconds=5.0, crop_center_x=32, crop_center_y=18),
        video_writer_factory=writer_factory,
        image_writer=image_writer,
        clock=lambda: 20.0,
    )
    recorder.record(event)
    recorder.close()

    directory = tmp_path / "events" / "2026-08-09" / event.event_id
    metadata = json.loads((directory / "event.json").read_text(encoding="utf-8"))
    assert metadata["duration"] == 7.0
    assert metadata["pre_roll_available"] is False
    assert metadata["actual_pre_roll_seconds"] == 0.0
    assert metadata["actual_post_roll_seconds"] == 6.9
    assert metadata["pre_roll_gap_seconds"] == 2.0
    assert metadata["post_roll_gap_seconds"] == 0.1
    assert metadata["status"] == "recording_truncated"
    assert metadata["recording_status"] == "truncated"
    assert metadata["truncation_reasons"] == ["missing_pre_roll"]
    assert writer_fps == pytest.approx([len(timestamps) / 7.0, len(timestamps) / 7.0])


def test_partial_window_is_encoded_but_truthfully_marked_truncated(tmp_path: Path) -> None:
    frames = [
        FramePacket(0, np.zeros((36, 64, 3), dtype=np.uint8), "stamp", 0.0),
        FramePacket(1, np.ones((36, 64, 3), dtype=np.uint8), "stamp", 2.0),
        FramePacket(2, np.full((36, 64, 3), 2, dtype=np.uint8), "stamp", 3.0),
    ]
    written: dict[str, list[np.ndarray]] = {}

    def writer_factory(path: str, _fourcc: int, _fps: float, _size: tuple[int, int]) -> FakeWriter:
        return FakeWriter(path, written)

    def image_writer(path: str, _frame: np.ndarray) -> bool:
        Path(path).write_bytes(b"jpeg")
        return True

    event = replace(manual_event(), event_id="partial-window")
    recorder = ManualFireRecorder(
        FakeCamera(frames),
        tmp_path,
        ManualFireRecordingConfig(
            pre_roll_seconds=2.0,
            post_roll_seconds=5.0,
            target_fps=1.0,
            crop_center_x=32,
            crop_center_y=18,
        ),
        video_writer_factory=writer_factory,
        image_writer=image_writer,
        clock=lambda: 20.0,
    )
    recorder.record(event)
    recorder.close()

    directory = tmp_path / "events" / "2026-08-09" / event.event_id
    metadata = json.loads((directory / "event.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "recording_truncated"
    assert metadata["recording_status"] == "truncated"
    assert metadata["pre_roll_gap_seconds"] == 0.0
    assert metadata["post_roll_gap_seconds"] == 4.0
    assert metadata["truncation_reasons"] == ["missing_post_roll"]
    assert metadata["frames_written"] == 3
    assert all(len(frames_written) == 3 for frames_written in written.values())
    assert recorder.status()["truncated"] == 1
    assert recorder.status()["failed"] == 0


def test_one_usable_frame_is_truncated_instead_of_claimed_complete_or_failed(
    tmp_path: Path,
) -> None:
    frame = FramePacket(1, np.ones((36, 64, 3), dtype=np.uint8), "stamp", 2.0)
    written: dict[str, list[np.ndarray]] = {}

    def writer_factory(path: str, _fourcc: int, _fps: float, _size: tuple[int, int]) -> FakeWriter:
        return FakeWriter(path, written)

    def image_writer(path: str, _frame: np.ndarray) -> bool:
        Path(path).write_bytes(b"jpeg")
        return True

    event = replace(manual_event(), event_id="one-frame-window")
    recorder = ManualFireRecorder(
        FakeCamera([frame]),
        tmp_path,
        ManualFireRecordingConfig(
            pre_roll_seconds=2.0,
            post_roll_seconds=5.0,
            target_fps=1.0,
            crop_center_x=32,
            crop_center_y=18,
        ),
        video_writer_factory=writer_factory,
        image_writer=image_writer,
        clock=lambda: 20.0,
    )
    recorder.record(event)
    recorder.close()

    directory = tmp_path / "events" / "2026-08-09" / event.event_id
    metadata = json.loads((directory / "event.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "recording_truncated"
    assert metadata["recording_status"] == "truncated"
    assert metadata["frames_written"] == 1
    assert metadata["pre_roll_gap_seconds"] == 2.0
    assert metadata["post_roll_gap_seconds"] == 7.0
    assert metadata["truncation_reasons"] == [
        "missing_pre_roll",
        "missing_post_roll",
        "insufficient_frame_span",
    ]
    assert len(written) == 2
    assert all(len(frames_written) == 1 for frames_written in written.values())
    assert recorder.status()["completed"] == 1
    assert recorder.status()["truncated"] == 1
    assert recorder.status()["failed"] == 0


def test_recorder_persists_error_status_without_successful_clip(tmp_path: Path) -> None:
    recorder = ManualFireRecorder(
        FakeCamera([]),
        tmp_path,
        ManualFireRecordingConfig(post_roll_seconds=0.01),
        clock=lambda: 10.0,
    )
    recorder.record(manual_event())
    recorder.close()

    directory = tmp_path / "events" / "2026-08-09" / "manual-fire-test"
    metadata = json.loads((directory / "event.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "recording_failed"
    assert metadata["recording_status"] == "error"
    assert metadata["pre_roll_gap_seconds"] == 2.0
    assert metadata["post_roll_gap_seconds"] == pytest.approx(2.01)
    assert not (directory / "manual_fire_zoom.avi").exists()


def test_close_drains_both_queues_stops_workers_and_is_idempotent(tmp_path: Path) -> None:
    frames = [
        FramePacket(index, np.full((36, 64, 3), index, dtype=np.uint8), "stamp", float(index))
        for index in range(3)
    ]
    written: dict[str, list[np.ndarray]] = {}

    def writer_factory(path: str, _fourcc: int, _fps: float, _size: tuple[int, int]) -> FakeWriter:
        return FakeWriter(path, written)

    def image_writer(path: str, _frame: np.ndarray) -> bool:
        Path(path).write_bytes(b"jpeg")
        return True

    recorder = ManualFireRecorder(
        FakeCamera(frames),
        tmp_path,
        ManualFireRecordingConfig(
            pre_roll_seconds=2.0,
            post_roll_seconds=0.01,
            crop_center_x=32,
            crop_center_y=18,
        ),
        video_writer_factory=writer_factory,
        image_writer=image_writer,
        clock=lambda: 10.0,
    )
    recorder.record(replace(manual_event(), event_id="shutdown-one"))
    recorder.record(replace(manual_event(), event_id="shutdown-two"))

    recorder.close()
    recorder.close()

    status = recorder.status()
    assert status["active"] is False
    assert status["outstanding"] == 0
    assert status["collector_queue_depth"] == 0
    assert status["encoder_queue_depth"] == 0
    assert status["completed"] == 2
    assert not recorder._encoder_thread.is_alive()
    assert all(not thread.is_alive() for thread in recorder._collector_threads)
    with pytest.raises(RuntimeError, match="recorder is closed"):
        recorder.record(replace(manual_event(), event_id="shutdown-too-late"))
