from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import cv2
import pytest

from squirrel_shooter.camera_capture import positive_seconds
from squirrel_shooter.camera_common import FrameRateMeter, open_camera
from squirrel_shooter.config import CameraConfig
from squirrel_shooter.files import timestamped_output_path
from squirrel_shooter.modes import DEFAULT_MODE, OperatingMode
from squirrel_shooter.valve import DisabledValveController, ValveState
import squirrel_shooter.camera_common as camera_common


def test_timestamped_output_path_is_predictable() -> None:
    when = datetime(2026, 7, 13, 20, 15, 30, 123456, tzinfo=timezone.utc)

    path = timestamped_output_path(Path("captures"), "camera-test", "avi", when=when)

    assert path == Path("captures/camera-test-20260713-201530-123456.avi")


def test_frame_rate_meter_uses_supplied_times() -> None:
    meter = FrameRateMeter()

    assert meter.update(10.0) == 0.0
    assert meter.update(10.1) == pytest.approx(10.0)


def test_capture_seconds_must_be_positive() -> None:
    assert positive_seconds("2.5") == 2.5
    with pytest.raises(Exception):
        positive_seconds("0")


def test_mjpeg_rejection_does_not_prevent_camera_setup(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    class RejectingCapture:
        def __init__(self) -> None:
            self.set_calls: list[tuple[int, float]] = []

        def isOpened(self) -> bool:  # noqa: N802 - OpenCV API shape
            return True

        def set(self, prop: int, value: float) -> bool:
            self.set_calls.append((prop, value))
            return prop != cv2.CAP_PROP_FOURCC

        def get(self, prop: int) -> float:
            values = {
                cv2.CAP_PROP_FRAME_WIDTH: 1280.0,
                cv2.CAP_PROP_FRAME_HEIGHT: 720.0,
                cv2.CAP_PROP_FPS: 10.0,
                cv2.CAP_PROP_FOURCC: float(cv2.VideoWriter_fourcc(*"YUYV")),
            }
            return values.get(prop, 0.0)

        def release(self) -> None:
            return None

    capture = RejectingCapture()
    monkeypatch.setattr(camera_common.cv2, "VideoCapture", lambda _index: capture)

    opened = open_camera(CameraConfig(0, 1280, 720, 30, Path("captures")))

    assert opened is capture
    assert [prop for prop, _value in capture.set_calls] == [
        cv2.CAP_PROP_FOURCC,
        cv2.CAP_PROP_FRAME_WIDTH,
        cv2.CAP_PROP_FRAME_HEIGHT,
        cv2.CAP_PROP_FPS,
    ]
    assert "continuing with its available format" in caplog.text


def test_defaults_cannot_open_water_valve() -> None:
    valve = DisabledValveController()

    assert DEFAULT_MODE is OperatingMode.CAMERA_TEST
    assert valve.state is ValveState.CLOSED
    with pytest.raises(RuntimeError, match="remains closed"):
        valve.open()
    assert valve.state is ValveState.CLOSED
