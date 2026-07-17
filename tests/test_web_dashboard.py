from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pytest
from flask import Flask

from conftest import write_test_config
from squirrel_shooter.camera_service import CameraService, CameraStatus
from squirrel_shooter.config import CameraConfig, load_config
from squirrel_shooter.motion_runtime import MotionRuntimeStatus
from squirrel_shooter.vision_service import VisionService, VisionStatus
from squirrel_shooter.web_dashboard import build_parser, create_app, list_capture_images


class OfflineCameraService:
    def __init__(self, status: CameraStatus | None = None) -> None:
        self.start_calls = 0
        self._status = status or CameraStatus(False, 1280, 720, 0.0, "Test camera unavailable")

    def start(self) -> None:
        self.start_calls += 1

    def stop(self, timeout: float = 3.0) -> None:
        del timeout

    def status(self) -> CameraStatus:
        return self._status

    def wait_for_frame(self, after_sequence: int, timeout: float = 1.0) -> None:
        del after_sequence, timeout
        return None


class StaticVisionService:
    def __init__(
        self,
        status: VisionStatus | MotionRuntimeStatus | None = None,
        events: list[dict[str, object]] | None = None,
    ) -> None:
        self.start_calls = 0
        self._status = status or VisionStatus(
            "LEARNING", True, 0.0, 0, 0, 0, 0, 0, 0, 0,
            None, None, None, None, None, True, True,
        )
        self._events = events or []

    def start(self) -> None:
        self.start_calls += 1

    def stop(self, timeout: float = 3.0) -> None:
        del timeout

    def status(self) -> VisionStatus | MotionRuntimeStatus:
        return self._status

    def recent_events(self) -> list[dict[str, object]]:
        return list(self._events)

    def mjpeg_frames(self) -> Iterator[bytes]:
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\nmock\r\n"


@pytest.fixture
def dashboard(tmp_path: Path) -> tuple[Flask, Path, OfflineCameraService, StaticVisionService]:
    capture_directory = tmp_path / "captures"
    capture_directory.mkdir()
    config_path = write_test_config(tmp_path)
    camera = OfflineCameraService()
    vision = StaticVisionService()
    app = create_app(
        config_path,
        camera_service=camera,  # type: ignore[arg-type]
        vision_service=vision,  # type: ignore[arg-type]
        temperature_reader=lambda: None,
    )
    app.config.update(TESTING=True)
    return app, capture_directory, camera, vision


def add_capture(directory: Path, filename: str, modified: float) -> Path:
    path = directory / filename
    path.write_bytes(b"test jpeg")
    os.utime(path, (modified, modified))
    return path


def test_dashboard_loads_when_camera_is_unavailable(
    dashboard: tuple[Flask, Path, OfflineCameraService, StaticVisionService],
) -> None:
    app, _, camera, vision = dashboard
    response = app.test_client().get("/")
    assert response.status_code == 200
    assert b"Camera offline" in response.data
    assert b"LEARNING background" in response.data
    assert b"No physical outputs" in response.data
    assert response.data.count(b'class="status-card"') == 4
    assert b'id="blob-count"' not in response.data
    assert b"squirrel-squirter-logo.png" in response.data
    assert camera.start_calls == 1
    assert vision.start_calls == 1


def test_status_health_and_recent_events_endpoints(
    dashboard: tuple[Flask, Path, OfflineCameraService, StaticVisionService],
) -> None:
    app, _, _, _ = dashboard
    status = app.test_client().get("/api/status")
    health = app.test_client().get("/api/health")
    recent = app.test_client().get("/api/recent-events")
    assert status.status_code == health.status_code == recent.status_code == 200
    assert status.json["application_mode"] == "shared-camera-motion-watch"
    assert status.json["camera"]["state"] == "OFFLINE"
    assert status.json["detector"]["state"] == "LEARNING"
    assert health.json["camera_alive"] is False
    assert health.json["detector_alive"] is True
    assert health.json["capture_directory_writable"] is True
    assert recent.json == {"count": 0, "events": []}


def test_status_api_explains_filtered_candidate(tmp_path: Path) -> None:
    config_path = write_test_config(tmp_path)
    filtered_group = {
        "provisional_category": "small_animal_candidate",
        "component_count": 1,
        "average_pixel_speed": 3.0,
        "event_eligible": False,
        "event_filter_reason": "small_motion_not_coherent",
    }
    status = MotionRuntimeStatus(
        state="READY",
        enabled=True,
        processing_fps=10.0,
        blob_count=1,
        persistence_count=5,
        frames_processed=10,
        candidates_seen=5,
        accepted_events=0,
        rejected_events=0,
        snapshots_saved=0,
        global_motion_rejections=0,
        active_events=0,
        last_detector_update=None,
        last_detector_age_seconds=0.1,
        last_event=None,
        last_snapshot=None,
        last_error=None,
        thread_alive=True,
        capture_directory_writable=True,
        current_groups=(filtered_group,),
        last_event_summary=None,
    )
    app = create_app(
        config_path,
        camera_service=OfflineCameraService(),  # type: ignore[arg-type]
        vision_service=StaticVisionService(status),  # type: ignore[arg-type]
        temperature_reader=lambda: None,
    )
    app.config.update(TESTING=True)

    response = app.test_client().get("/api/status")
    assert response.status_code == 200
    assert response.json["detector"]["current_groups"][0]["event_filter_reason"] == "small_motion_not_coherent"


def test_stale_camera_is_reported_in_health(tmp_path: Path) -> None:
    config_path = write_test_config(tmp_path)
    camera = OfflineCameraService(CameraStatus(True, 640, 360, 20.0, None, "2026-07-15T12:00:00-04:00", 20.0, 50, True))
    app = create_app(config_path, camera_service=camera, vision_service=StaticVisionService(), temperature_reader=lambda: 45.0)
    app.config.update(TESTING=True)
    response = app.test_client().get("/api/health")
    assert response.json["camera_state"] == "STALE"
    assert response.json["camera_alive"] is False


def test_captures_are_sorted_and_kept_on_separate_archive_page(
    dashboard: tuple[Flask, Path, OfflineCameraService, StaticVisionService],
) -> None:
    app, directory, _, _ = dashboard
    for index in range(14):
        add_capture(directory, f"capture-{index:02}.jpg", 1_700_000_000.0 + index)
    assert [item.filename for item in list_capture_images(directory)][:2] == ["capture-13.jpg", "capture-12.jpg"]
    landing = app.test_client().get("/").get_data(as_text=True)
    full = app.test_client().get("/captures").get_data(as_text=True)
    assert landing.count('class="capture-card"') == 0
    assert full.count('class="capture-card"') == 14
    assert "capture-13.jpg" not in landing and "capture-00.jpg" in full


def test_dashboard_shows_only_five_most_recent_grouped_events(tmp_path: Path) -> None:
    config_path = write_test_config(tmp_path)
    events: list[dict[str, object]] = []
    for index in range(7):
        event_id = f"event-{index}"
        directory = tmp_path / "captures" / "events" / event_id
        directory.mkdir(parents=True)
        snapshot = directory / "snapshot.jpg"
        clip = directory / "clip.avi"
        snapshot.write_bytes(b"jpeg")
        clip.write_bytes(b"avi")
        if index == 0:
            (directory / "classification.json").write_text(
                json.dumps(
                    {
                        "classification_status": "known",
                        "display_label": "Car",
                        "label_source": "automatic",
                    }
                ),
                encoding="utf-8",
            )
        elif index == 1:
            (directory / "classification.json").write_text(
                json.dumps(
                    {
                        "classification_status": "unknown",
                        "display_label": "Unknown",
                        "label_source": None,
                    }
                ),
                encoding="utf-8",
            )
        events.append(
            {
                "event_id": event_id,
                "start_timestamp": f"2026-07-16T17:0{index}:00-04:00",
                "provisional_category": "small_animal_candidate",
                "movement_attributes": ["coherent_travel"],
                "snapshot_path": str(snapshot),
                "clip_path": str(clip),
            }
        )
    app = create_app(
        config_path,
        camera_service=OfflineCameraService(),  # type: ignore[arg-type]
        vision_service=StaticVisionService(events=events),  # type: ignore[arg-type]
        temperature_reader=lambda: 44.0,
    )
    app.config.update(TESTING=True)

    body = app.test_client().get("/").get_data(as_text=True)
    assert body.count('class="capture-card event-card"') == 5
    assert "Event event-0" in body and "Event event-4" in body
    assert "Event event-5" not in body
    assert "Car" in body and "Unknown" in body and "Motion: small animal candidate" in body
    assert "All event pictures and videos" in body and "Standalone pictures" in body
    api_events = app.test_client().get("/api/events").json["events"]
    assert api_events[0]["display_label"] == "Car" and api_events[1]["display_label"] == "Unknown"
    assert api_events[0]["snapshot_url"].endswith("/snapshot.jpg")
    assert api_events[0]["clip_url"].endswith("/clip.avi")
    assert 'id="recent-events-content"' in body
    assert "refreshRecentEvents" in body and ".slice(0, 5)" in body


def test_event_archive_reads_all_saved_events_newest_first(tmp_path: Path) -> None:
    config_path = write_test_config(tmp_path)
    for index in range(21):
        event_id = f"saved-event-{index:02}"
        directory = tmp_path / "captures" / "events" / "2026-07-16" / event_id
        directory.mkdir(parents=True)
        snapshot = directory / "snapshot.jpg"
        clip = directory / "clip.avi"
        snapshot.write_bytes(b"jpeg")
        clip.write_bytes(b"avi")
        (directory / "event.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "event_id": event_id,
                    "start_timestamp": f"2026-07-16T17:{index:02}:00-04:00",
                    "provisional_category": "small_animal_candidate",
                    "movement_attributes": ["coherent_travel"],
                    "snapshot_path": str(snapshot),
                    "clip_path": str(clip),
                }
            ),
            encoding="utf-8",
        )
        if index == 20:
            (directory / "classification.json").write_text(
                json.dumps(
                    {
                        "classification_status": "known",
                        "display_label": "Person",
                        "label_source": "human",
                    }
                ),
                encoding="utf-8",
            )
    app = create_app(
        config_path,
        camera_service=OfflineCameraService(),  # type: ignore[arg-type]
        vision_service=StaticVisionService(),  # type: ignore[arg-type]
        temperature_reader=lambda: 44.0,
    )
    app.config.update(TESTING=True)

    first = app.test_client().get("/events").get_data(as_text=True)
    second = app.test_client().get("/events?page=2").get_data(as_text=True)
    assert first.count('class="capture-card event-card"') == 20
    assert "Event saved-event-20" in first and "Event saved-event-00" not in first
    assert "Person" in first and "Label: human" in first
    assert second.count('class="capture-card event-card"') == 1
    assert "Event saved-event-00" in second
    assert "Generated review report" in first and "picture archive" in first


def test_dashboard_logo_is_packaged_and_served(dashboard: tuple[Flask, Path, OfflineCameraService, StaticVisionService]) -> None:
    app, _, _, _ = dashboard
    response = app.test_client().get("/static/squirrel-squirter-logo.png")
    assert response.status_code == 200
    assert response.content_type == "image/png"
    assert len(response.data) > 1_000_000


def test_empty_unsupported_and_unsafe_captures_are_handled(
    dashboard: tuple[Flask, Path, OfflineCameraService, StaticVisionService], tmp_path: Path
) -> None:
    app, directory, _, _ = dashboard
    assert b"No captures yet" in app.test_client().get("/captures").data
    add_capture(directory, "kept.jpg", 1_700_000_002.0)
    add_capture(directory, "ignored.png", 1_700_000_003.0)
    (tmp_path / "secret.jpg").write_bytes(b"secret")
    page = app.test_client().get("/captures").data
    assert b"kept.jpg" in page and b"ignored.png" not in page
    assert app.test_client().get("/captures/kept.jpg").status_code == 200
    assert app.test_client().get("/captures/..%2Fsecret.jpg").status_code == 404


def test_latest_report_resolves_relative_directory_from_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(write_test_config(tmp_path))
    config = replace(config, reporting=replace(config.reporting, directory=Path("relative-reports")))
    report_directory = tmp_path / "relative-reports"
    report_directory.mkdir()
    (report_directory / "latest-report.html").write_text("<h1>Latest report</h1>", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    app = create_app(
        app_config=config,
        camera_service=OfflineCameraService(),  # type: ignore[arg-type]
        vision_service=StaticVisionService(),  # type: ignore[arg-type]
        start_camera=False,
        start_vision=False,
    )
    app.config.update(TESTING=True)

    response = app.test_client().get("/reports/latest")
    assert response.status_code == 200
    assert b"Latest report" in response.data


def test_multiple_browser_sessions_reuse_one_camera(tmp_path: Path) -> None:
    opened = threading.Event()
    released = threading.Event()
    factory_calls = 0

    class GeneratedCapture:
        def get(self, prop: int) -> float:
            return 128.0 if prop == cv2.CAP_PROP_FRAME_WIDTH else (72.0 if prop == cv2.CAP_PROP_FRAME_HEIGHT else 20.0)

        def read(self) -> tuple[bool, np.ndarray | None]:
            if released.wait(0.005):
                return False, None
            return True, np.zeros((72, 128, 3), dtype=np.uint8)

        def release(self) -> None:
            released.set()

    def capture_factory(settings: CameraConfig) -> GeneratedCapture:
        nonlocal factory_calls
        del settings
        factory_calls += 1
        opened.set()
        return GeneratedCapture()

    config_path = write_test_config(tmp_path, motion__learning_frames=1, motion__processing_width=128)
    config = load_config(config_path)
    camera = CameraService(config.camera, capture_factory=capture_factory, platform_checker=lambda: True, encode_jpeg=False)
    vision = VisionService(camera, config)
    app = create_app(config_path, camera_service=camera, vision_service=vision, temperature_reader=lambda: 47.2)
    app.config.update(TESTING=True)
    try:
        assert opened.wait(1.0)
        deadline = time.monotonic() + 2.0
        while vision.status().frames_processed == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        first = app.test_client().get("/video-feed", buffered=False)
        second = app.test_client().get("/video-feed", buffered=False)
        assert next(first.response).startswith(b"--frame")
        assert next(second.response).startswith(b"--frame")
        first.close()
        second.close()
        camera.start()
        vision.start()
        assert factory_calls == 1
    finally:
        vision.stop()
        camera.stop()


def test_shared_service_publishes_raw_frame_without_real_camera(tmp_path: Path) -> None:
    released = threading.Event()
    frame = np.full((36, 64, 3), 127, dtype=np.uint8)

    class Capture:
        def get(self, prop: int) -> float:
            return 64.0 if prop == cv2.CAP_PROP_FRAME_WIDTH else (36.0 if prop == cv2.CAP_PROP_FRAME_HEIGHT else 20.0)
        def read(self) -> tuple[bool, np.ndarray | None]:
            return (False, None) if released.wait(0.01) else (True, frame.copy())
        def release(self) -> None:
            released.set()

    service = CameraService(CameraConfig(0, 1280, 720, 30.0, tmp_path), capture_factory=lambda _: Capture(), platform_checker=lambda: True)
    service.start()
    try:
        deadline = time.monotonic() + 1.0
        while not service.status().online and time.monotonic() < deadline:
            time.sleep(0.01)
        packet = service.wait_for_frame(-1)
        assert packet is not None and packet.frame.shape == (36, 64, 3)
        assert next(service.mjpeg_frames()).startswith(b"--frame\r\nContent-Type: image/jpeg")
    finally:
        service.stop()


def test_non_pi_host_never_opens_a_camera(tmp_path: Path) -> None:
    called = threading.Event()
    def forbidden(settings: CameraConfig) -> None:
        del settings
        called.set()
        raise AssertionError
    service = CameraService(CameraConfig(0, 1280, 720, 30.0, tmp_path), capture_factory=forbidden, platform_checker=lambda: False)
    service.start()
    deadline = time.monotonic() + 1.0
    while service.status().error is None and time.monotonic() < deadline:
        time.sleep(0.01)
    status = service.status()
    service.stop()
    assert not called.is_set()
    assert status.error == "Camera capture is disabled because this host is not a Raspberry Pi"


def test_dashboard_cli_defaults_to_private_listener() -> None:
    args = build_parser().parse_args([])
    assert args.host == "0.0.0.0" and args.port == 5000
