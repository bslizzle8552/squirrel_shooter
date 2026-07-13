from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from squirrel_shooter.camera_capture import positive_seconds
from squirrel_shooter.camera_common import FrameRateMeter
from squirrel_shooter.files import timestamped_output_path
from squirrel_shooter.modes import DEFAULT_MODE, OperatingMode
from squirrel_shooter.valve import DisabledValveController, ValveState


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


def test_defaults_cannot_open_water_valve() -> None:
    valve = DisabledValveController()

    assert DEFAULT_MODE is OperatingMode.CAMERA_TEST
    assert valve.state is ValveState.CLOSED
    with pytest.raises(RuntimeError, match="remains closed"):
        valve.open()
    assert valve.state is ValveState.CLOSED
