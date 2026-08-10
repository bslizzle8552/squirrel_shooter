from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import cv2
import numpy as np
import pytest
from flask import Flask

from conftest import write_test_config
from squirrel_shooter.camera_service import CameraService, CameraStatus
from squirrel_shooter.config import CameraConfig, load_config
from squirrel_shooter.manual_control import ManualControlService
from squirrel_shooter.motion_runtime import MotionRuntimeStatus
from squirrel_shooter.pan_tilt import PanTiltController
from squirrel_shooter.valve import GPIOValveController, ValveConfig
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


class ManualControlDriver:
    def configure_channel(self, channel: int, pulse_min_us: int, pulse_max_us: int) -> None:
        del channel, pulse_min_us, pulse_max_us

    def set_angle(self, channel: int, angle: float) -> None:
        del channel, angle

    def release(self, channel: int) -> None:
        del channel

    def cleanup(self) -> None:
        return None


class ManualControlOutput:
    def on(self) -> None:
        return None

    def off(self) -> None:
        return None

    def close(self) -> None:
        return None


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
    assert b"Manual control" in response.data
    assert b'id="stat-fps"' in response.data
    assert b'id="stat-temp"' in response.data
    assert b'id="stat-queue"' in response.data
    assert b'id="stat-mode"' in response.data
    assert b'id="blob-count"' not in response.data
    assert b"squirrel-squirter-logo.png" in response.data
    assert b"console.css" in response.data and b"console.js" in response.data
    assert camera.start_calls == 1
    assert vision.start_calls == 1


def test_manual_control_page_and_api_enforce_token_limits_and_cooldown(tmp_path: Path) -> None:
    config_path = write_test_config(tmp_path)
    config = load_config(config_path)
    now = [100.0]
    pan_tilt = PanTiltController(config.pan_tilt, driver=ManualControlDriver(), sleep=lambda _seconds: None)
    valve = GPIOValveController(ValveConfig(enabled=True, gpio_pin=17), output=ManualControlOutput())
    recordings: list[object] = []
    recorder = SimpleNamespace(record=recordings.append, close=lambda: None)
    controls = ManualControlService(
        config.pan_tilt,
        config.manual_control,
        pan_tilt=pan_tilt,
        valve=valve,
        sleep=lambda _seconds: None,
        clock=lambda: now[0],
        fire_recorder=recorder,  # type: ignore[arg-type]
    )
    app = create_app(
        app_config=config,
        camera_service=OfflineCameraService(CameraStatus(True, 1280, 720, 10.0, None)),  # type: ignore[arg-type]
        vision_service=StaticVisionService(),  # type: ignore[arg-type]
        manual_control_service=controls,
        temperature_reader=lambda: None,
    )
    app.config.update(TESTING=True)
    client = app.test_client()
    token = app.extensions["manual_control_token"]
    headers = {"X-Control-Token": token}

    page = client.get("/manual-control")
    assert page.status_code == 200
    assert b'aria-label="Pan left"' in page.data
    assert b'aria-label="Tilt forward"' in page.data
    assert b'aria-label="Tilt backward"' in page.data
    assert b"<span>Forward</span>" in page.data
    assert b"<span>Backward</span>" in page.data
    assert b"<span>Up</span>" not in page.data
    assert b"<span>Down</span>" not in page.data
    assert b'data-step="1"' in page.data
    assert page.data.count(b'class="step-button') == 3
    assert b'id="fire-button"' in page.data
    assert b'id="fire-button" disabled' not in page.data
    assert b'id="park-button"' in page.data
    assert b"PARK 85" in page.data and b"82" in page.data
    assert b"Rewatch saved manual-fire videos" in page.data
    assert b'id="fire-status">READY<' in page.data
    assert b"Commanded positions only" in page.data
    assert b"startup reference" in page.data
    assert b"Calibration: 0 / 9" in page.data
    assert b'id="active-calibration-point">1<' in page.data
    assert b'id="calibration-image"' in page.data
    assert b'id="calibration-marker"' in page.data
    assert b'id="target-marker"' in page.data
    assert b'id="calibration-geometry-overlay"' in page.data
    assert b'nine numbered anchors' in page.data
    assert b'data-camera-mode="aim"' in page.data
    assert b'data-camera-mode="calibration"' in page.data
    assert b">AIM TARGET</button>" in page.data
    assert b">EDIT CALIBRATION</button>" in page.data
    assert b'id="targeting-status">TARGETING UNAVAILABLE<' in page.data
    assert b'id="target-range">--<' in page.data
    assert b'id="target-cell">--<' in page.data
    assert b'id="pixel-selection-status"' in page.data
    assert b'id="save-calibration"' in page.data
    assert b'id="save-aim-action">SAVE AIM<' in page.data
    assert b"current backend aim for Point 1" in page.data
    assert page.data.count(b"data-calibration-point=") == 9
    assert b"Not started" in page.data
    assert b'id="calibration-confirmation"' in page.data
    assert client.post("/api/manual-control/fire", json={}).status_code == 403
    assert client.post("/api/manual-control/park", json={}).status_code == 403
    assert recordings == []
    parked = client.post("/api/manual-control/park", json={}, headers=headers)
    assert parked.status_code == 200
    assert parked.json["control"]["pan"] == 85
    assert parked.json["control"]["tilt"] == 82
    assert parked.json["control"]["cooldown_remaining_seconds"] == 0
    assert recordings == []
    assert client.post("/api/manual-control/move", json={"direction": "right", "step": 3}).status_code == 403
    assert client.post(
        "/api/manual-control/calibration/pixel",
        json={"display_x": 400, "display_y": 225, "display_width": 800, "display_height": 450},
    ).status_code == 403
    incomplete_aim = client.post(
        "/api/manual-control/aim",
        json={"display_x": 400, "display_y": 225, "display_width": 800, "display_height": 450},
        headers=headers,
    )
    assert incomplete_aim.status_code == 400
    assert b"exactly 9 complete calibration points" in incomplete_aim.data

    fine_move = client.post(
        "/api/manual-control/move",
        json={"direction": "left", "step": 1},
        headers=headers,
    )
    assert fine_move.status_code == 200
    assert fine_move.json["control"]["pan"] == 86
    fine_move_back = client.post(
        "/api/manual-control/move",
        json={"direction": "right", "step": 1},
        headers=headers,
    )
    assert fine_move_back.status_code == 200
    assert fine_move_back.json["control"]["pan"] == 85

    pixel_selected = client.post(
        "/api/manual-control/calibration/pixel",
        json={"point": 9, "display_x": 400, "display_y": 225, "display_width": 800, "display_height": 450},
        headers=headers,
    )
    assert pixel_selected.status_code == 200
    assert pixel_selected.json["calibration_point"] == {
        "point": 1,
        "pixel_x": 640,
        "pixel_y": 360,
        "pan": None,
        "tilt": None,
    }
    assert pixel_selected.json["control"]["completed_calibration_count"] == 0
    assert pixel_selected.json["control"]["camera_frame_width"] == 1280
    assert pixel_selected.json["control"]["camera_frame_height"] == 720
    partial_page = client.get("/manual-control")
    assert b"Calibration: 0 / 9" in partial_page.data
    assert b'class="calibration-point-button active pixel-selected"' in partial_page.data
    assert b'class="calibration-point-button active saved"' not in partial_page.data
    assert b'id="calibration-detail-pixel-x">640<' in partial_page.data
    assert b'id="calibration-detail-pixel-y">360<' in partial_page.data
    assert b'id="calibration-detail-pan">Not saved<' in partial_page.data
    assert b'id="save-aim-action">SAVE AIM<' in partial_page.data

    moved = client.post("/api/manual-control/move", json={"direction": "right", "step": 3}, headers=headers)
    assert moved.status_code == 200
    assert moved.json["control"]["pan"] == 82
    fired = client.post("/api/manual-control/fire", json={}, headers=headers)
    assert fired.status_code == 200
    assert fired.json["control"]["state"] == "COOLDOWN"
    assert fired.json["control"]["pan"] == 85
    assert fired.json["control"]["tilt"] == 82
    assert fired.json["control"]["targeting"]["status"] == "PARKED"
    assert len(recordings) == 1
    assert client.post("/api/manual-control/fire", json={}, headers=headers).status_code == 409
    assert len(recordings) == 1

    during_cooldown = client.post(
        "/api/manual-control/move",
        json={"direction": "right", "step": 3},
        headers=headers,
    )
    assert during_cooldown.status_code == 200
    assert during_cooldown.json["control"]["pan"] == 82
    assert during_cooldown.json["control"]["tilt"] == 82
    adjusted_during_cooldown = client.post(
        "/api/manual-control/move",
        json={"direction": "right", "step": 3},
        headers=headers,
    )
    assert adjusted_during_cooldown.status_code == 200
    assert adjusted_during_cooldown.json["control"]["pan"] == 79
    saved = client.post(
        "/api/manual-control/calibration",
        json={"point": 9, "pan": 30, "tilt": 70, "pixel_x": None, "pixel_y": None},
        headers=headers,
    )
    assert saved.status_code == 200
    assert saved.json["calibration_point"] == {
        "point": 1,
        "pixel_x": 640,
        "pixel_y": 360,
        "pan": 79.0,
        "tilt": 82.0,
    }
    saved_page = client.get("/manual-control")
    assert b"Calibration: 1 / 9" in saved_page.data
    assert b'class="calibration-point-button active saved"' in saved_page.data
    assert b'id="calibration-detail-point">1<' in saved_page.data
    assert b'id="calibration-detail-pixel-x">640<' in saved_page.data
    assert 'id="calibration-detail-pan">79.0°<'.encode("utf-8") in saved_page.data
    assert b'id="save-aim-action">UPDATE AIM<' in saved_page.data

    for point in range(2, 10):
        selected = client.post(
            "/api/manual-control/calibration/active",
            json={"point": point},
            headers=headers,
        )
        assert selected.status_code == 200
        assert selected.json["control"]["active_calibration_point"] == point
        if point == 2:
            missing_pixel = client.post("/api/manual-control/calibration", json={}, headers=headers)
            assert missing_pixel.status_code == 400
            assert b"Click the center" in missing_pixel.data
        row, column = divmod(point - 1, 3)
        selected_pixel = client.post(
            "/api/manual-control/calibration/pixel",
            json={
                "display_x": 400 + column * 100,
                "display_y": 225 + row * 56.25,
                "display_width": 800,
                "display_height": 450,
            },
            headers=headers,
        )
        assert selected_pixel.status_code == 200
        assert selected_pixel.json["calibration_point"]["point"] == point
        response = client.post(
            "/api/manual-control/calibration",
            json={},
            headers=headers,
        )
        assert response.status_code == 200
    complete_page = client.get("/manual-control")
    assert b"Calibration: 9 / 9" in complete_page.data
    assert b"All nine blocks have camera pixels and saved aim" in complete_page.data
    assert b"Desktop AIM TARGET mode is ready" in complete_page.data

    now[0] = 110.0
    before_aim = json.loads((tmp_path / "calibration.json").read_text(encoding="utf-8"))
    outside = client.post(
        "/api/manual-control/aim",
        json={"display_x": 100, "display_y": 100, "display_width": 800, "display_height": 450},
        headers=headers,
    )
    assert outside.status_code == 400
    assert outside.json["control"]["targeting"]["status"] == "OUTSIDE CALIBRATED AREA"
    assert outside.json["control"]["targeting"]["in_range"] is False
    assert outside.json["control"]["targeting"]["cell"] is None
    aimed = client.post(
        "/api/manual-control/aim",
        json={"display_x": 500, "display_y": 281.25, "display_width": 800, "display_height": 450},
        headers=headers,
    )
    assert aimed.status_code == 200
    assert aimed.json["target"]["pixel_x"] == 800
    assert aimed.json["target"]["pixel_y"] == 450
    assert aimed.json["control"]["targeting"]["status"] == "AIM READY"
    assert aimed.json["control"]["targeting"]["in_range"] is True
    assert aimed.json["control"]["targeting"]["cell"] == [1, 2, 4, 5]
    assert aimed.json["control"]["targeting"]["method"] == "inverse bilinear"
    assert len(aimed.json["control"]["targeting"]["geometry"]["anchors"]) == 9
    assert len(aimed.json["control"]["targeting"]["geometry"]["cells"]) == 4
    assert aimed.json["control"]["pan"] == 79
    assert aimed.json["control"]["tilt"] == 82
    assert aimed.json["control"]["cooldown_remaining_seconds"] == 0
    assert json.loads((tmp_path / "calibration.json").read_text(encoding="utf-8")) == before_aim

    moved_again = client.post(
        "/api/manual-control/move",
        json={"direction": "up", "step": 3},
        headers=headers,
    )
    assert moved_again.status_code == 200
    selected_first = client.post(
        "/api/manual-control/calibration/active",
        json={"point": 1},
        headers=headers,
    )
    assert selected_first.status_code == 200
    reclicked = client.post(
        "/api/manual-control/calibration/pixel",
        json={"display_x": 500, "display_y": 225, "display_width": 800, "display_height": 450},
        headers=headers,
    )
    assert reclicked.status_code == 200
    assert reclicked.json["calibration_point"]["pixel_x"] == 800
    assert reclicked.json["calibration_point"]["pixel_y"] == 360
    updated = client.post(
        "/api/manual-control/calibration",
        json={"pixel_x": None, "pixel_y": None},
        headers=headers,
    )
    assert updated.status_code == 200
    assert updated.json["calibration_point"]["pixel_x"] == 800
    assert updated.json["calibration_point"]["pixel_y"] == 360
    assert updated.json["calibration_point"]["tilt"] == 85.0
    assert len(updated.json["control"]["calibration_points"]) == 9
    assert [record["point"] for record in updated.json["control"]["calibration_points"]].count(1) == 1

    manual_script = client.get("/static/manual_control.js")
    manual_style = client.get("/static/manual_control.css")
    assert manual_script.status_code == 200
    assert manual_style.status_code == 200
    assert b"showCalibrationConfirmation" in manual_script.data
    assert b"'UPDATE AIM' : 'SAVE AIM'" in manual_script.data
    assert b"aim " in manual_script.data and b"updated" in manual_script.data and b"saved" in manual_script.data
    assert b"ArrowUp: 'up'" in manual_script.data
    assert b"ArrowDown: 'down'" in manual_script.data
    assert b"ArrowLeft: 'left'" in manual_script.data
    assert b"ArrowRight: 'right'" in manual_script.data
    assert b"display_width: rect.width" in manual_script.data
    assert b"renderPixelMarker" in manual_script.data
    assert b"renderTargetMarker" in manual_script.data
    assert b"cameraMode === 'aim'" in manual_script.data
    assert b"cfg.urls.aim" in manual_script.data
    assert b"cfg.urls.park" in manual_script.data
    assert b"cfg.urls.calibrationPixel" in manual_script.data
    assert b"calibrationPixel" in page.data
    assert b"pollIntervalMs" in page.data and b"1000" in page.data
    assert b'grid-template-areas: "camera aim" "camera fire" "calibration ."' in manual_style.data
    assert b'grid-template-areas: "aim" "fire" "camera"' in manual_style.data
    assert b".calibration-card { display: none; }" in manual_style.data
    assert b".calibration-marker" in manual_style.data
    assert b".target-marker" in manual_style.data
    assert b".save-aim-button" in manual_style.data


def test_manual_control_state_is_shared_across_two_clients_and_save_ignores_stale_point(tmp_path: Path) -> None:
    config = load_config(write_test_config(tmp_path))
    now = [100.0]
    controls = ManualControlService(
        config.pan_tilt,
        config.manual_control,
        pan_tilt=PanTiltController(config.pan_tilt, driver=ManualControlDriver(), sleep=lambda _seconds: None),
        valve=GPIOValveController(ValveConfig(enabled=True, gpio_pin=17), output=ManualControlOutput()),
        sleep=lambda _seconds: None,
        clock=lambda: now[0],
    )
    app = create_app(
        app_config=config,
        camera_service=OfflineCameraService(CameraStatus(True, 1280, 720, 10.0, None)),  # type: ignore[arg-type]
        vision_service=StaticVisionService(),  # type: ignore[arg-type]
        manual_control_service=controls,
        temperature_reader=lambda: None,
    )
    app.config.update(TESTING=True)
    desktop = app.test_client()
    phone = app.test_client()
    headers = {"X-Control-Token": app.extensions["manual_control_token"]}

    assert phone.post("/api/manual-control/calibration/active", json={"point": 4}).status_code == 403
    assert desktop.post(
        "/api/manual-control/calibration/active",
        json={"point": 10},
        headers=headers,
    ).status_code == 400
    selected = desktop.post(
        "/api/manual-control/calibration/active",
        json={"point": 4},
        headers=headers,
    )
    assert selected.status_code == 200
    assert phone.get("/api/manual-control").json["control"]["active_calibration_point"] == 4
    assert b'id="active-calibration-point">4<' in phone.get("/manual-control").data

    pixel_selected = desktop.post(
        "/api/manual-control/calibration/pixel",
        json={"point": 1, "display_x": 400, "display_y": 300, "display_width": 800, "display_height": 600},
        headers=headers,
    )
    assert pixel_selected.status_code == 200
    assert pixel_selected.json["calibration_point"] == {
        "point": 4,
        "pixel_x": 640,
        "pixel_y": 360,
        "pan": None,
        "tilt": None,
    }
    shared_pixel = phone.get("/api/manual-control").json["control"]["calibration_points"]
    assert shared_pixel[0]["pixel_selected"] is True
    assert shared_pixel[0]["complete"] is False

    moved = phone.post(
        "/api/manual-control/move",
        json={"direction": "right", "step": 3},
        headers=headers,
    )
    assert moved.status_code == 200
    assert desktop.get("/api/manual-control").json["control"]["pan"] == 82

    saved = desktop.post(
        "/api/manual-control/calibration",
        json={"point": 1, "pan": 30, "tilt": 70, "pixel_x": None, "pixel_y": None},
        headers=headers,
    )
    assert saved.status_code == 200
    assert saved.json["calibration_point"] == {
        "point": 4,
        "pixel_x": 640,
        "pixel_y": 360,
        "pan": 82.0,
        "tilt": 85.0,
    }

    fired = phone.post("/api/manual-control/fire", json={}, headers=headers)
    assert fired.status_code == 200
    desktop_during_cooldown = desktop.get("/api/manual-control").json["control"]
    assert desktop_during_cooldown["state"] == "COOLDOWN"
    assert desktop_during_cooldown["cooldown_remaining_seconds"] == 10.0
    assert desktop.post("/api/manual-control/fire", json={}, headers=headers).status_code == 409

    now[0] = 104.0
    refreshed_phone = phone.get("/api/manual-control").json["control"]
    assert refreshed_phone["state"] == "COOLDOWN"
    assert refreshed_phone["cooldown_remaining_seconds"] == 6.0


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
    assert body.count('data-kind="recent"') == 5
    assert "Event event-0" in body and "Event event-4" in body
    assert "Event event-5" not in body
    assert "Car" in body and "Unknown" in body and "Motion: small animal candidate" in body
    assert "All event pictures and videos" in body and "Standalone pictures" in body
    api_events = app.test_client().get("/api/events").json["events"]
    assert api_events[0]["display_label"] == "Car" and api_events[1]["display_label"] == "Unknown"
    assert api_events[0]["snapshot_url"].endswith("/snapshot.jpg")
    assert api_events[0]["clip_url"].endswith("/clip.avi")
    assert 'id="queue-list"' in body and 'id="recent-list"' in body
    assert "console.js" in body and "reviewToken" in body


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


def test_event_archive_uses_zoom_clip_as_primary_manual_fire_replay(tmp_path: Path) -> None:
    config_path = write_test_config(tmp_path)
    directory = tmp_path / "captures" / "events" / "2026-08-09" / "manual-fire-test"
    directory.mkdir(parents=True)
    snapshot = directory / "snapshot.jpg"
    zoom = directory / "manual_fire_zoom.avi"
    full = directory / "manual_fire_full.avi"
    for path in (snapshot, zoom, full):
        path.write_bytes(b"evidence")
    (directory / "event.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "event_id": "manual-fire-test",
                "capture_method": "manual_fire",
                "start_timestamp": "2026-08-09T19:32:00-04:00",
                "provisional_category": "manual_fire",
                "snapshot_path": str(snapshot),
                "clip_path": str(zoom),
                "full_frame_clip_path": str(full),
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

    body = app.test_client().get("/events").get_data(as_text=True)
    assert "Manual fire" in body
    assert "manual_fire_zoom.avi" in body
    assert "manual_fire_full.avi" in body
    assert "Full-field clip" in body


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
