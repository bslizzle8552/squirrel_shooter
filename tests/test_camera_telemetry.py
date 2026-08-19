from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np

from squirrel_shooter.camera_service import CameraService
from squirrel_shooter.config import CameraConfig, SharedCameraConfig


class _Capture:
    def __init__(self) -> None:
        self.released = threading.Event()
        self.frame = np.zeros((24, 32, 3), dtype=np.uint8)

    def read(self):  # type: ignore[no-untyped-def]
        if self.released.is_set():
            return False, None
        return True, self.frame

    def release(self) -> None:
        self.released.set()

    def get(self, property_id: int) -> float:
        values = {
            cv2.CAP_PROP_FRAME_WIDTH: 32.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 24.0,
            cv2.CAP_PROP_FPS: 15.0,
            cv2.CAP_PROP_FOURCC: float(cv2.VideoWriter_fourcc(*"MJPG")),
        }
        return values.get(property_id, 0.0)

    def getBackendName(self) -> str:  # noqa: N802 - OpenCV API shape
        return "V4L2"


def test_camera_status_exposes_bounded_physical_tail_timings(tmp_path: Path) -> None:
    capture = _Capture()
    camera = CameraService(
        CameraConfig(0, 32, 24, 15.0, tmp_path),
        shared_settings=SharedCameraConfig(True, 3, 0.01, 0.1, 1.0),
        capture_factory=lambda _settings: capture,
        platform_checker=lambda: True,
        encode_jpeg=False,
    )

    camera.start()
    try:
        first = camera.wait_for_frame(0, timeout=1.0, copy=False)
        assert first is not None
        second = camera.wait_for_frame(first.sequence, timeout=1.0, copy=False)
        assert second is not None
        status = camera.status()
    finally:
        camera.stop(timeout=1.0)

    assert status.capture_read_timing is not None
    assert status.physical_frame_interval_timing is not None
    assert status.frame_publish_timing is not None
    assert status.capture_read_timing["total_count"] >= 2
    assert status.physical_frame_interval_timing["total_count"] >= 1
    assert status.frame_publish_timing["total_count"] >= 1
    assert status.physical_frame_interval_timing["capacity"] == 2048
    assert status.last_frame_age_seconds is not None
    assert status.last_frame_age_seconds < 1.0
    assert second.received_monotonic >= first.received_monotonic
    assert status.source_fourcc == "MJPG"
    assert status.capture_backend == "V4L2"
    assert status.source_mode_matches_request is True
    assert status.source_mode_mismatch_fields == ()


def test_first_physical_frame_corrects_reported_resolution_mismatch(tmp_path: Path) -> None:
    capture = _Capture()
    capture.frame = np.zeros((20, 30, 3), dtype=np.uint8)
    camera = CameraService(
        CameraConfig(0, 32, 24, 15.0, tmp_path),
        shared_settings=SharedCameraConfig(True, 3, 0.01, 0.1, 1.0),
        capture_factory=lambda _settings: capture,
        platform_checker=lambda: True,
        encode_jpeg=False,
    )

    camera.start()
    try:
        assert camera.wait_for_frame(0, timeout=1.0, copy=False) is not None
        status = camera.status()
    finally:
        camera.stop(timeout=1.0)

    assert (status.width, status.height) == (30, 20)
    assert status.source_mode_matches_request is False
    assert status.source_mode_mismatch_fields == ("resolution",)
