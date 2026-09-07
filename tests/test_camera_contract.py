"""Camera ownership/provenance contracts using no physical camera or model."""

from __future__ import annotations

import ast
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from squirrel_shooter.camera_service import CameraService
from squirrel_shooter.config import CameraConfig, SharedCameraConfig


class StepCapture:
    """Each read consumes one test-supplied image; release unblocks shutdown."""

    def __init__(self) -> None:
        self.frames: queue.Queue[np.ndarray | None] = queue.Queue()
        self.released = threading.Event()

    def read(self) -> tuple[bool, np.ndarray | None]:
        frame = self.frames.get(timeout=3.0)
        return frame is not None and not self.released.is_set(), frame

    def release(self) -> None:
        if not self.released.is_set():
            self.released.set()
            self.frames.put(None)

    def get(self, property_id: int) -> float:
        return {
            cv2.CAP_PROP_FRAME_WIDTH: 32.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 24.0,
            cv2.CAP_PROP_FPS: 15.0,
            cv2.CAP_PROP_FOURCC: float(cv2.VideoWriter_fourcc(*"MJPG")),
        }.get(property_id, 0.0)

    def getBackendName(self) -> str:  # noqa: N802 - OpenCV API
        return "fake"


def make_camera(tmp_path: Path, captures: list[StepCapture]) -> CameraService:
    unopened = iter(captures)

    def open_capture(_settings: CameraConfig) -> StepCapture:
        capture = next(unopened)
        index = captures.index(capture)
        if index:
            assert captures[index - 1].released.is_set(), "old owner must release before reconnect"
        return capture

    return CameraService(
        CameraConfig(0, 32, 24, 15.0, tmp_path),
        shared_settings=SharedCameraConfig(True, 1, 0.01, 0.1, 1.0),
        capture_factory=open_capture,
        platform_checker=lambda: True,
        encode_jpeg=False,
        frame_buffer_seconds=5.0,
        # Retain each of this test's few discrete inputs, without sleeping.
        frame_buffer_fps=1_000_000,
    )


def await_sequence(camera: CameraService, sequence: int) -> None:
    deadline = time.monotonic() + 1.0
    while camera.current_frame_sequence() < sequence and time.monotonic() < deadline:
        threading.Event().wait(0.001)
    assert camera.current_frame_sequence() == sequence


def test_latest_frame_skips_unconsumed_sources_and_preserves_receipt_identity(tmp_path: Path) -> None:
    capture = StepCapture()
    camera = make_camera(tmp_path, [capture])
    camera.start()
    camera.start()
    try:
        before_receive = time.monotonic()
        capture.frames.put(np.full((24, 32, 3), 1, dtype=np.uint8))
        first = camera.wait_for_frame(0, timeout=1.0, copy=False)
        assert first is not None
        capture.frames.put(np.full((24, 32, 3), 2, dtype=np.uint8))
        capture.frames.put(np.full((24, 32, 3), 3, dtype=np.uint8))
        await_sequence(camera, first.sequence + 2)

        # A slow consumer gets frame 3 directly, not queued frame 2.
        latest = camera.wait_for_frame(first.sequence, timeout=0.0, copy=False)
        copied = camera.wait_for_frame(first.sequence, timeout=0.0)
        assert latest is not None and copied is not None
        assert latest.sequence == first.sequence + 2
        assert np.all(latest.frame == 3)
        assert before_receive <= first.received_monotonic <= latest.received_monotonic <= time.monotonic()
        assert datetime.fromisoformat(latest.received_at).utcoffset() is not None
        assert (copied.sequence, copied.received_at, copied.received_monotonic, copied.generation) == (
            latest.sequence, latest.received_at, latest.received_monotonic, latest.generation
        )
        assert copied.frame is not latest.frame
        copied.frame[:] = 99
        assert np.all(latest.frame == 3)
        assert np.all(first.frame == 1)
        assert first.generation == latest.generation == 1
        assert camera.wait_for_frame(latest.sequence, timeout=0.001) is None
        assert camera.status().camera_open_count == 1

        buffered = list(camera.buffered_frames(0.0, copy=False))
        metadata = camera.buffered_frame_metadata(0.0)
        assert [packet.sequence for packet in buffered] == [1, 2, 3]
        assert [(item.sequence, item.generation, item.received_monotonic) for item in metadata] == [
            (packet.sequence, packet.generation, packet.received_monotonic) for packet in buffered
        ]
        assert buffered[-1].frame is latest.frame
    finally:
        camera.stop(timeout=1.0)
    assert capture.released.is_set()


@pytest.mark.parametrize("new_shape", [(24, 32, 3), (20, 30, 3)])
def test_reconnect_advances_generation_without_reusing_sequence_or_old_provenance(
    tmp_path: Path, new_shape: tuple[int, ...]
) -> None:
    first_capture, second_capture = StepCapture(), StepCapture()
    camera = make_camera(tmp_path, [first_capture, second_capture])
    camera.start()
    try:
        first_capture.frames.put(np.full((24, 32, 3), 1, dtype=np.uint8))
        first = camera.wait_for_frame(0, timeout=1.0, copy=False)
        assert first is not None
        first_capture.frames.put(None)
        deadline = time.monotonic() + 1.0
        while camera.status().camera_open_count < 2 and time.monotonic() < deadline:
            threading.Event().wait(0.001)
        assert camera.status().camera_open_count == 2
        assert first_capture.released.is_set()
        assert camera.wait_for_frame(first.sequence, timeout=0.001) is None

        second_capture.frames.put(np.full(new_shape, 2, dtype=np.uint8))
        second = camera.wait_for_frame(first.sequence, timeout=1.0, copy=False)
        assert second is not None
        assert second.sequence == first.sequence + 1
        assert second.generation == first.generation + 1
        assert second.received_monotonic > first.received_monotonic
        assert second.frame.shape == new_shape
        assert np.all(first.frame == 1)
        status = camera.status()
        assert status.frame_generation == second.generation
        assert (status.width, status.height) == (new_shape[1], new_shape[0])
        assert status.reconnects == 1
        # Existing pre-roll remains evidence; it must not be retagged as new.
        buffered = list(camera.buffered_frames(0.0, copy=False))
        assert [(item.sequence, item.generation) for item in buffered] == [
            (first.sequence, first.generation), (second.sequence, second.generation)
        ]
    finally:
        camera.stop(timeout=1.0)


def test_native_resolution_change_advances_generation_without_reopening(tmp_path: Path) -> None:
    capture = StepCapture()
    camera = make_camera(tmp_path, [capture])
    camera.start()
    try:
        packets = []
        for shape in ((24, 32, 3), (20, 30, 3), (20, 30, 3), (24, 32, 3)):
            capture.frames.put(np.zeros(shape, dtype=np.uint8))
            packet = camera.wait_for_frame(packets[-1].sequence if packets else 0, timeout=1.0)
            assert packet is not None
            packets.append(packet)
            status = camera.status()
            assert status.source_mode_matches_request is (shape == (24, 32, 3))
            assert status.source_mode_mismatch_fields == (() if shape == (24, 32, 3) else ("resolution",))
        assert [packet.sequence for packet in packets] == [1, 2, 3, 4]
        assert [packet.generation for packet in packets] == [1, 2, 2, 3]
        assert camera.status().camera_open_count == 1
    finally:
        camera.stop(timeout=1.0)


def test_service_restart_preserves_sequence_and_advances_generation(tmp_path: Path) -> None:
    captures = [StepCapture(), StepCapture()]
    camera = make_camera(tmp_path, captures)
    packets = []
    for capture in captures:
        camera.start()
        try:
            capture.frames.put(np.zeros((24, 32, 3), dtype=np.uint8))
            packet = camera.wait_for_frame(packets[-1].sequence if packets else 0, timeout=1.0)
            assert packet is not None
            packets.append(packet)
        finally:
            camera.stop(timeout=1.0)
    assert [packet.sequence for packet in packets] == [1, 2]
    assert [packet.generation for packet in packets] == [1, 2]


def test_production_consumers_cannot_construct_another_camera_owner() -> None:
    package = Path(__file__).resolve().parents[1] / "src" / "squirrel_shooter"
    consumers = (
        "motion_runtime", "classifier", "current_scene_safety", "manual_fire_recording",
        "manual_control", "web_dashboard", "vision_service",
    )
    for module in consumers:
        tree = ast.parse((package / f"{module}.py").read_text(encoding="utf-8"))
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            name = call.func.id if isinstance(call.func, ast.Name) else getattr(call.func, "attr", None)
            assert name not in {"CameraService", "open_camera"}, (module, call.lineno, name)
            if name == "VideoCapture":
                # Only the historical completed-clip decoder may read a file;
                # no consumer may add a numeric device or competing live path.
                assert module == "motion_runtime"
                assert len(call.args) == 1 and ast.unparse(call.args[0]) == "str(clip_path)"
