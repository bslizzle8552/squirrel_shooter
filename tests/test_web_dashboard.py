from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pytest
from flask import Flask

from squirrel_shooter.camera_service import CameraService, CameraStatus
from squirrel_shooter.config import CameraConfig
from squirrel_shooter.web_dashboard import build_parser, create_app, list_capture_images


class OfflineCameraService:
    def __init__(self) -> None:
        self.start_calls = 0

    def start(self) -> None:
        self.start_calls += 1

    def stop(self, timeout: float = 3.0) -> None:
        del timeout

    def status(self) -> CameraStatus:
        return CameraStatus(
            online=False,
            width=1280,
            height=720,
            fps=0.0,
            error="Test camera unavailable",
        )

    def mjpeg_frames(self) -> Iterator[bytes]:
        return iter(())


@pytest.fixture
def dashboard(tmp_path: Path) -> tuple[Flask, Path, OfflineCameraService]:
    capture_directory = tmp_path / "captures"
    capture_directory.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
camera:
  device_index: 0
  requested_width: 1280
  requested_height: 720
  requested_fps: 30
  output_directory: {capture_directory.as_posix()}
""".strip(),
        encoding="utf-8",
    )
    service = OfflineCameraService()
    app = create_app(
        config_path,
        camera_service=service,  # type: ignore[arg-type]
        temperature_reader=lambda: None,
    )
    app.config.update(TESTING=True)
    return app, capture_directory, service


def add_capture(directory: Path, filename: str, modified: float) -> Path:
    path = directory / filename
    path.write_bytes(b"test jpeg")
    os.utime(path, (modified, modified))
    return path


def test_dashboard_loads_when_camera_is_unavailable(
    dashboard: tuple[Flask, Path, OfflineCameraService],
) -> None:
    app, _, service = dashboard

    response = app.test_client().get("/")

    assert response.status_code == 200
    assert b"Squirrel Squirter" in response.data
    assert b"Camera offline" in response.data
    assert b"Detection, aiming, and water controls are not active" in response.data
    assert service.start_calls == 1


def test_status_endpoint_reports_offline_camera(
    dashboard: tuple[Flask, Path, OfflineCameraService],
) -> None:
    app, _, _ = dashboard

    response = app.test_client().get("/api/status")

    assert response.status_code == 200
    assert response.json == {
        "application_mode": "observation-only",
        "camera": {
            "configured_device": {"index": 0, "label": "/dev/video0"},
            "error": "Test camera unavailable",
            "fps": 0.0,
            "online": False,
            "resolution": {"height": 720, "label": "1280×720", "width": 1280},
        },
        "capture_count": 0,
        "cpu_temperature_c": None,
    }


def test_captures_are_sorted_newest_first(tmp_path: Path) -> None:
    add_capture(tmp_path, "old.jpg", 1_700_000_100.0)
    add_capture(tmp_path, "new.jpeg", 1_700_000_300.0)
    add_capture(tmp_path, "middle.JPG", 1_700_000_200.0)

    captures = list_capture_images(tmp_path)

    assert [capture.filename for capture in captures] == [
        "new.jpeg",
        "middle.JPG",
        "old.jpg",
    ]


def test_landing_gallery_is_limited_to_twelve_newest_images(
    dashboard: tuple[Flask, Path, OfflineCameraService],
) -> None:
    app, capture_directory, _ = dashboard
    for index in range(14):
        add_capture(
            capture_directory,
            f"capture-{index:02}.jpg",
            1_700_000_000.0 + index,
        )

    response = app.test_client().get("/")
    page = response.get_data(as_text=True)

    assert response.status_code == 200
    assert page.count('class="capture-card"') == 12
    assert "capture-13.jpg" in page
    assert "capture-02.jpg" in page
    assert "capture-01.jpg" not in page
    assert "capture-00.jpg" not in page


def test_full_captures_page_returns_more_than_landing_page(
    dashboard: tuple[Flask, Path, OfflineCameraService],
) -> None:
    app, capture_directory, _ = dashboard
    for index in range(18):
        add_capture(
            capture_directory,
            f"capture-{index:02}.jpg",
            1_700_000_000.0 + index,
        )

    landing_page = app.test_client().get("/").get_data(as_text=True)
    captures_page = app.test_client().get("/captures").get_data(as_text=True)

    assert landing_page.count('class="capture-card"') == 12
    assert captures_page.count('class="capture-card"') == 18
    assert "capture-00.jpg" in captures_page


def test_empty_capture_directory_is_handled_cleanly(
    dashboard: tuple[Flask, Path, OfflineCameraService],
) -> None:
    app, _, _ = dashboard

    response = app.test_client().get("/captures")

    assert response.status_code == 200
    assert b"No captures yet" in response.data


def test_unsupported_capture_files_are_ignored(
    dashboard: tuple[Flask, Path, OfflineCameraService],
) -> None:
    app, capture_directory, _ = dashboard
    add_capture(capture_directory, "kept.jpg", 1_700_000_002.0)
    add_capture(capture_directory, "ignored.png", 1_700_000_003.0)
    add_capture(capture_directory, "ignored.avi", 1_700_000_004.0)

    response = app.test_client().get("/captures")

    assert response.status_code == 200
    assert b"kept.jpg" in response.data
    assert b"ignored.png" not in response.data
    assert b"ignored.avi" not in response.data


def test_invalid_filename_cannot_escape_capture_directory(
    dashboard: tuple[Flask, Path, OfflineCameraService], tmp_path: Path
) -> None:
    app, capture_directory, _ = dashboard
    add_capture(capture_directory, "allowed.jpg", 1_700_000_001.0)
    (tmp_path / "secret.jpg").write_bytes(b"secret")
    client = app.test_client()

    assert client.get("/captures/allowed.jpg").status_code == 200
    assert client.get("/captures/..%2Fsecret.jpg").status_code == 404
    assert client.get("/captures/..%5Csecret.jpg").status_code == 404
    assert client.get("/captures/ignored.png").status_code == 404


def test_web_app_reuses_one_physical_camera_connection(tmp_path: Path) -> None:
    opened = threading.Event()
    released = threading.Event()
    factory_calls = 0

    class BlockingCapture:
        def isOpened(self) -> bool:
            return True

        def set(self, prop: int, value: float) -> bool:
            del prop, value
            return True

        def get(self, prop: int) -> float:
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 1280.0
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 720.0
            return 30.0

        def read(self) -> tuple[bool, np.ndarray | None]:
            if released.wait(0.02):
                return False, None
            return True, np.zeros((72, 128, 3), dtype=np.uint8)

        def release(self) -> None:
            released.set()

    def capture_factory(settings: CameraConfig) -> BlockingCapture:
        nonlocal factory_calls
        del settings
        factory_calls += 1
        opened.set()
        return BlockingCapture()

    capture_directory = tmp_path / "captures"
    capture_directory.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
camera:
  device_index: 0
  requested_width: 1280
  requested_height: 720
  requested_fps: 30
  output_directory: {capture_directory.as_posix()}
""".strip(),
        encoding="utf-8",
    )
    settings = CameraConfig(0, 1280, 720, 30.0, capture_directory)
    service = CameraService(
        settings,
        capture_factory=capture_factory,
        platform_checker=lambda: True,
    )
    app = create_app(config_path, camera_service=service, temperature_reader=lambda: 47.2)
    app.config.update(TESTING=True)

    try:
        assert opened.wait(1.0)
        client = app.test_client()
        assert client.get("/").status_code == 200
        assert client.get("/api/status").status_code == 200
        first_stream = client.get("/video-feed", buffered=False)
        assert next(first_stream.response).startswith(b"--frame")
        first_stream.close()
        second_stream = client.get("/video-feed", buffered=False)
        assert next(second_stream.response).startswith(b"--frame")
        second_stream.close()
        service.start()
        assert factory_calls == 1
    finally:
        service.stop()


def test_shared_service_publishes_mjpeg_and_raw_frames(tmp_path: Path) -> None:
    released = threading.Event()
    frame = np.full((36, 64, 3), 127, dtype=np.uint8)

    class AvailableCapture:
        def get(self, prop: int) -> float:
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 64.0
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 36.0
            return 20.0

        def read(self) -> tuple[bool, np.ndarray | None]:
            if released.wait(0.01):
                return False, None
            return True, frame.copy()

        def release(self) -> None:
            released.set()

    settings = CameraConfig(0, 1280, 720, 30.0, tmp_path)
    service = CameraService(
        settings,
        capture_factory=lambda _: AvailableCapture(),
        platform_checker=lambda: True,
    )
    service.start()

    try:
        deadline = time.monotonic() + 1.0
        while not service.status().online and time.monotonic() < deadline:
            time.sleep(0.01)

        status = service.status()
        streamed = next(service.mjpeg_frames())
        raw = service.latest_frame()

        assert status.online is True
        assert (status.width, status.height) == (64, 36)
        assert streamed.startswith(b"--frame\r\nContent-Type: image/jpeg")
        assert b"\xff\xd8" in streamed
        assert raw is not None
        assert raw.shape == (36, 64, 3)
    finally:
        service.stop()


def test_non_pi_host_never_opens_a_camera(tmp_path: Path) -> None:
    factory_called = threading.Event()

    def forbidden_factory(settings: CameraConfig) -> None:
        del settings
        factory_called.set()
        raise AssertionError("The camera factory must not run off the Raspberry Pi")

    settings = CameraConfig(0, 1280, 720, 30.0, tmp_path)
    service = CameraService(
        settings,
        capture_factory=forbidden_factory,
        platform_checker=lambda: False,
    )
    service.start()

    deadline = time.monotonic() + 1.0
    while service.status().error is None and time.monotonic() < deadline:
        time.sleep(0.01)

    status = service.status()
    service.stop()

    assert factory_called.is_set() is False
    assert status.online is False
    assert status.error == "Camera capture is disabled because this host is not a Raspberry Pi"


def test_dashboard_cli_defaults_to_required_private_listener() -> None:
    args = build_parser().parse_args([])

    assert args.host == "0.0.0.0"
    assert args.port == 5000
