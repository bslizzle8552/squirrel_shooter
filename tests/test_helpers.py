from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import cv2
import pytest

from squirrel_shooter.camera_capture import positive_seconds
from squirrel_shooter.camera_common import (
    FrameRateMeter,
    open_camera,
    parse_v4l2_supported_modes,
    probe_supported_camera_modes,
    supports_requested_camera_mode,
)
from squirrel_shooter.config import CameraConfig
from squirrel_shooter.files import timestamped_output_path
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
    monkeypatch.setattr(camera_common, "_is_linux_v4l2_host", lambda: False)
    monkeypatch.setattr(camera_common.cv2, "VideoCapture", lambda _index: capture)

    opened = open_camera(CameraConfig(0, 1280, 720, 30, Path("captures")))

    assert opened is capture
    assert [prop for prop, _value in capture.set_calls] == [
        cv2.CAP_PROP_FOURCC,
        cv2.CAP_PROP_FRAME_WIDTH,
        cv2.CAP_PROP_FRAME_HEIGHT,
        cv2.CAP_PROP_FPS,
        cv2.CAP_PROP_BUFFERSIZE,
    ]
    assert "continuing with its available format" in caplog.text
    assert "source cadence request was not confirmed" in caplog.text


_V4L2_MODES = """\
ioctl: VIDIOC_ENUM_FMT
\tType: Video Capture

\t[0]: 'MJPG' (Motion-JPEG, compressed)
\t\tSize: Discrete 1280x720
\t\t\tInterval: Discrete 0.033s (30.000 fps)
\t\t\tInterval: Discrete 0.067s (15.000 fps)
\t\tSize: Discrete 640x480
\t\t\tInterval: Discrete 0.033s (30.000 fps)
\t[1]: 'YUYV' (YUYV 4:2:2)
\t\tSize: Discrete 640x480
\t\t\tInterval: Discrete 0.067s (15.000 fps)
"""


def test_v4l2_supported_mode_parser_preserves_discrete_cadences() -> None:
    modes = parse_v4l2_supported_modes(_V4L2_MODES)

    assert modes == (
        camera_common.SupportedCameraMode("MJPG", 1280, 720, (30.0, 15.0)),
        camera_common.SupportedCameraMode("MJPG", 640, 480, (30.0,)),
        camera_common.SupportedCameraMode("YUYV", 640, 480, (15.0,)),
    )
    settings = CameraConfig(0, 1280, 720, 15.0, Path("captures"))
    assert supports_requested_camera_mode(modes, settings) is True
    assert supports_requested_camera_mode(
        modes,
        CameraConfig(0, 1280, 720, 10.0, Path("captures")),
    ) is False


def test_supported_mode_probe_uses_read_only_list_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=_V4L2_MODES, stderr="")

    monkeypatch.setattr(camera_common, "_is_linux_v4l2_host", lambda: True)
    monkeypatch.setattr(camera_common.shutil, "which", lambda _name: "/usr/bin/v4l2-ctl")
    monkeypatch.setattr(camera_common.subprocess, "run", run)

    result = probe_supported_camera_modes(2)

    assert result.utility_available is True and result.error is None
    assert len(result.modes) == 3
    assert calls[0][0] == [
        "/usr/bin/v4l2-ctl",
        "--device",
        "/dev/video2",
        "--list-formats-ext",
    ]
    assert calls[0][1]["check"] is False
    assert "shell" not in calls[0][1]


def test_supported_mode_probe_is_graceful_when_v4l2_ctl_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(camera_common, "_is_linux_v4l2_host", lambda: True)
    monkeypatch.setattr(camera_common.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        camera_common.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("subprocess must not run without v4l2-ctl"),
    )

    result = probe_supported_camera_modes(0)

    assert result.utility_available is False
    assert result.modes == ()
    assert "not installed" in (result.error or "")


def test_linux_open_requests_exact_v4l2_mode_before_capture_and_verifies_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[object] = []

    class Capture:
        def __init__(self) -> None:
            self.set_calls: list[tuple[int, float]] = []

        def isOpened(self) -> bool:  # noqa: N802 - OpenCV API shape
            return True

        def set(self, prop: int, value: float) -> bool:
            self.set_calls.append((prop, value))
            return True

        def get(self, prop: int) -> float:
            return {
                cv2.CAP_PROP_FRAME_WIDTH: 1280.0,
                cv2.CAP_PROP_FRAME_HEIGHT: 720.0,
                cv2.CAP_PROP_FPS: 15.0,
                cv2.CAP_PROP_FOURCC: float(cv2.VideoWriter_fourcc(*"MJPG")),
            }.get(prop, 0.0)

        def getBackendName(self) -> str:  # noqa: N802 - OpenCV API shape
            return "V4L2"

        def release(self) -> None:
            return None

    capture = Capture()

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        order.append(("command", command, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def video_capture(*args: object) -> Capture:
        order.append(("capture", args))
        return capture

    monkeypatch.setattr(camera_common, "_is_linux_v4l2_host", lambda: True)
    monkeypatch.setattr(camera_common.shutil, "which", lambda _name: "/usr/bin/v4l2-ctl")
    monkeypatch.setattr(camera_common.subprocess, "run", run)
    monkeypatch.setattr(camera_common.cv2, "VideoCapture", video_capture)

    opened = open_camera(CameraConfig(0, 1280, 720, 15.0, Path("captures")))

    assert opened is capture
    assert order[0][0] == "command"
    assert order[0][1] == [
        "/usr/bin/v4l2-ctl",
        "--device",
        "/dev/video0",
        "--set-fmt-video",
        "width=1280,height=720,pixelformat=MJPG",
        "--set-parm",
        "15",
    ]
    assert order[1] == ("capture", (0, cv2.CAP_V4L2))
    assert [prop for prop, _value in capture.set_calls] == [
        cv2.CAP_PROP_FOURCC,
        cv2.CAP_PROP_FRAME_WIDTH,
        cv2.CAP_PROP_FRAME_HEIGHT,
        cv2.CAP_PROP_FPS,
        cv2.CAP_PROP_BUFFERSIZE,
    ]


def test_failed_v4l2_utility_request_uses_explicit_backend_property_fallback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Capture:
        def __init__(self) -> None:
            self.set_calls: list[int] = []

        def isOpened(self) -> bool:  # noqa: N802 - OpenCV API shape
            return True

        def set(self, prop: int, _value: float) -> bool:
            self.set_calls.append(prop)
            return prop != cv2.CAP_PROP_FPS

        def get(self, prop: int) -> float:
            return {
                cv2.CAP_PROP_FRAME_WIDTH: 1280.0,
                cv2.CAP_PROP_FRAME_HEIGHT: 720.0,
                cv2.CAP_PROP_FPS: 30.0,
                cv2.CAP_PROP_FOURCC: float(cv2.VideoWriter_fourcc(*"MJPG")),
            }.get(prop, 0.0)

        def release(self) -> None:
            return None

    capture = Capture()
    capture_args: list[tuple[object, ...]] = []
    monkeypatch.setattr(camera_common, "_is_linux_v4l2_host", lambda: True)
    monkeypatch.setattr(camera_common.shutil, "which", lambda _name: "/usr/bin/v4l2-ctl")
    monkeypatch.setattr(
        camera_common.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="unsupported frame interval",
        ),
    )
    monkeypatch.setattr(
        camera_common.cv2,
        "VideoCapture",
        lambda *args: capture_args.append(args) or capture,
    )

    open_camera(CameraConfig(0, 1280, 720, 15.0, Path("captures")))

    assert capture_args == [(0, cv2.CAP_V4L2)]
    assert cv2.CAP_PROP_FPS in capture.set_calls
    assert "unsupported frame interval" in caplog.text
    assert "source cadence request was not confirmed" in caplog.text


def test_absent_utility_and_unavailable_explicit_backend_fall_back_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO")

    class ClosedCapture:
        released = False

        def isOpened(self) -> bool:  # noqa: N802 - OpenCV API shape
            return False

        def release(self) -> None:
            self.released = True

    class OpenCapture:
        def isOpened(self) -> bool:  # noqa: N802 - OpenCV API shape
            return True

        def set(self, _prop: int, _value: float) -> bool:
            return True

        def get(self, prop: int) -> float:
            return {
                cv2.CAP_PROP_FRAME_WIDTH: 1280.0,
                cv2.CAP_PROP_FRAME_HEIGHT: 720.0,
                cv2.CAP_PROP_FPS: 15.0,
                cv2.CAP_PROP_FOURCC: float(cv2.VideoWriter_fourcc(*"MJPG")),
            }.get(prop, 0.0)

        def release(self) -> None:
            return None

    closed = ClosedCapture()
    opened_capture = OpenCapture()
    calls: list[tuple[object, ...]] = []

    def video_capture(*args: object) -> object:
        calls.append(args)
        return closed if len(calls) == 1 else opened_capture

    monkeypatch.setattr(camera_common, "_is_linux_v4l2_host", lambda: True)
    monkeypatch.setattr(camera_common.shutil, "which", lambda _name: None)
    monkeypatch.setattr(camera_common.cv2, "VideoCapture", video_capture)

    result = open_camera(CameraConfig(0, 1280, 720, 15.0, Path("captures")))

    assert result is opened_capture
    assert calls == [(0, cv2.CAP_V4L2), (0,)]
    assert closed.released is True
    assert "v4l2-ctl is unavailable" in caplog.text
    assert "default backend" in caplog.text


def test_disabled_valve_cannot_open_water_valve() -> None:
    valve = DisabledValveController()

    assert valve.state is ValveState.CLOSED
    with pytest.raises(RuntimeError, match="remains closed"):
        valve.open()
    assert valve.state is ValveState.CLOSED
