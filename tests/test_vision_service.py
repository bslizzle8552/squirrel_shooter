from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from conftest import write_test_config
from squirrel_shooter.camera_service import FramePacket
from squirrel_shooter.config import load_config
from squirrel_shooter.detection import DetectionResult, DetectorState, MotionCandidate
from squirrel_shooter.diagnostics import cleanup_oldest
from squirrel_shooter.vision_service import VisionService
from squirrel_shooter.web_dashboard import list_capture_images


FRAME = np.zeros((100, 100, 3), dtype=np.uint8)
MASK = np.zeros((100, 100), dtype=np.uint8)


class OneFrameCamera:
    def __init__(self, *, fail: bool = False) -> None:
        self.sent = False
        self.fail = fail

    def wait_for_frame(self, after_sequence: int, timeout: float = 1.0) -> FramePacket | None:
        del after_sequence, timeout
        if self.fail:
            raise RuntimeError("camera frame bus failed")
        if not self.sent:
            self.sent = True
            return FramePacket(1, FRAME.copy(), "2026-07-15T12:00:00-04:00")
        time.sleep(0.005)
        return None


class ResultDetector:
    state = DetectorState.READY

    def __init__(self, result: DetectionResult | None = None, *, fail: bool = False) -> None:
        self.result = result or DetectionResult(DetectorState.READY, (), MASK, MASK, 12.0, 0)
        self.fail = fail

    def process(self, frame: np.ndarray) -> DetectionResult:
        del frame
        if self.fail:
            raise RuntimeError("synthetic detector failure")
        return self.result


def accepted_result() -> DetectionResult:
    candidate = MotionCandidate(
        timestamp="2026-07-15T12:34:56.000-04:00",
        bounding_box=(20, 20, 30, 30),
        center=(35, 35),
        area=841.0,
        persistence=3,
        roi_status="inside",
        accepted=True,
        reason="accepted",
    )
    return DetectionResult(DetectorState.READY, (candidate,), MASK, MASK, 10.0, 3)


def wait_until(predicate: object, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:  # type: ignore[operator]
        time.sleep(0.01)


def test_snapshot_creation_updates_gallery_and_event_record(tmp_path: Path) -> None:
    config = load_config(write_test_config(tmp_path))
    service = VisionService(OneFrameCamera(), config, detector=ResultDetector(accepted_result()))  # type: ignore[arg-type]
    service.start()
    try:
        wait_until(lambda: service.status().snapshots_saved == 1 and bool(service.recent_events()))
        status = service.status()
        events = service.recent_events()
        gallery = list_capture_images(config.camera.output_directory)
        assert status.accepted_events == status.snapshots_saved == 1
        assert len(gallery) == 1 and gallery[0].filename.startswith("event-")
        assert events[0]["snapshot_saved"] is True
        assert events[0]["snapshot_filename"] == gallery[0].filename
    finally:
        service.stop()


def test_snapshot_failure_does_not_stop_detection(tmp_path: Path) -> None:
    config = load_config(write_test_config(tmp_path))
    service = VisionService(
        OneFrameCamera(), config, detector=ResultDetector(accepted_result()), image_writer=lambda path, image: False  # type: ignore[arg-type]
    )
    service.start()
    try:
        wait_until(lambda: service.status().accepted_events == 1 and bool(service.recent_events()))
        status = service.status()
        assert status.thread_alive is True
        assert status.snapshots_saved == 0
        assert service.recent_events()[0]["snapshot_saved"] is False
    finally:
        service.stop()


def test_capture_directory_failure_is_health_visible(tmp_path: Path) -> None:
    config_path = write_test_config(tmp_path)
    (tmp_path / "captures").write_text("not a directory", encoding="utf-8")
    config = load_config(config_path)
    service = VisionService(OneFrameCamera(), config, detector=ResultDetector())  # type: ignore[arg-type]
    service.start()
    try:
        assert service.status().capture_directory_writable is False
    finally:
        service.stop()


def test_debug_images_are_disabled_by_default(tmp_path: Path) -> None:
    config = load_config(write_test_config(tmp_path))
    writes: list[str] = []
    service = VisionService(
        OneFrameCamera(), config, detector=ResultDetector(), image_writer=lambda path, image: writes.append(path) or True  # type: ignore[arg-type]
    )
    service.start()
    try:
        wait_until(lambda: service.status().frames_processed == 1)
        assert writes == []
        assert not config.motion.debug.directory.exists()
    finally:
        service.stop()


def test_detector_exception_is_caught_and_worker_stays_alive(tmp_path: Path) -> None:
    config = load_config(write_test_config(tmp_path))
    service = VisionService(OneFrameCamera(), config, detector=ResultDetector(fail=True))  # type: ignore[arg-type]
    service.start()
    try:
        wait_until(lambda: service.status().state == "ERROR")
        status = service.status()
        assert status.thread_alive is True
        assert status.last_error == "synthetic detector failure"
    finally:
        service.stop()


def test_worker_thread_failure_is_reported(tmp_path: Path) -> None:
    config = load_config(write_test_config(tmp_path))
    service = VisionService(OneFrameCamera(fail=True), config, detector=ResultDetector())  # type: ignore[arg-type]
    service.start()
    wait_until(lambda: not service.status().thread_alive)
    status = service.status()
    assert status.state == "ERROR"
    assert status.last_error == "camera frame bus failed"
    service.stop()


def test_storage_retention_deletes_oldest_first(tmp_path: Path) -> None:
    for index in range(5):
        path = tmp_path / f"event-{index}.jpg"
        path.write_bytes(b"x")
        timestamp = 100.0 + index
        path.touch()
        import os
        os.utime(path, (timestamp, timestamp))
    removed = cleanup_oldest(tmp_path, "event-*.jpg", 2, logging.getLogger("test"))
    assert removed == 3
    assert sorted(path.name for path in tmp_path.glob("*.jpg")) == ["event-3.jpg", "event-4.jpg"]
