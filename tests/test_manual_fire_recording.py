from __future__ import annotations

import json
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

    def buffered_frames(self, since_monotonic: float, *, after_sequence: int = -1) -> list[FramePacket]:
        return [
            frame
            for frame in self.frames
            if frame.received_monotonic >= since_monotonic and frame.sequence > after_sequence
        ]

    def wait_for_frame(self, after_sequence: int, timeout: float | None = None) -> None:
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
        fire_started_monotonic=1.0,
        fire_completed_monotonic=1.25,
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
    assert Path(metadata["clip_path"]).parent == directory
    assert Path(metadata["full_frame_clip_path"]).parent == directory
    assert Path(metadata["clip_path"]).name == "manual_fire_zoom.avi"
    assert Path(metadata["full_frame_clip_path"]).name == "manual_fire_full.avi"
    assert metadata["zoom_factor"] == 2.0
    assert metadata["crop_center_pixel"]["source"] == "calibrated_target_pixel"
    assert len(written) == 2
    assert all(len(frames_written) == 3 for frames_written in written.values())
    assert all(frame.shape == (36, 64, 3) for frames_written in written.values() for frame in frames_written)


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
    assert not (directory / "manual_fire_zoom.avi").exists()
