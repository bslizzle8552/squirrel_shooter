from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from squirrel_shooter.manual_control import (
    CalibrationStore,
    ControlState,
    FireCooldownError,
    ManualControlConfig,
    ManualControlService,
    display_click_to_frame_pixel,
)
from squirrel_shooter.pan_tilt import PanTiltConfig, PanTiltPosition
from squirrel_shooter.valve import ValveState


class FakePanTilt:
    def __init__(self, config: PanTiltConfig, events: list[str] | None = None) -> None:
        self.config = config
        self.events = events if events is not None else []
        self.moves: list[PanTiltPosition] = []
        self.cleaned = False

    def move_to_smooth(
        self,
        pan_angle: float,
        tilt_angle: float,
        *,
        settling_delay_seconds: float | None = None,
    ) -> PanTiltPosition:
        assert settling_delay_seconds == 0
        position = PanTiltPosition(pan_angle, tilt_angle)
        self.moves.append(position)
        self.events.append("move")
        return position

    def cleanup(self) -> None:
        self.cleaned = True


class FakeValve:
    def __init__(self, events: list[str] | None = None) -> None:
        self._state = ValveState.CLOSED
        self.events = events if events is not None else []
        self.cleaned = False

    @property
    def state(self) -> ValveState:
        return self._state

    def close(self) -> None:
        self.events.append("close")
        self._state = ValveState.CLOSED

    def open(self) -> None:
        self.events.append("open")
        self._state = ValveState.OPEN

    def cleanup(self) -> None:
        self.close()
        self.cleaned = True


def make_service(
    tmp_path: Path,
    *,
    clock=lambda: 100.0,
    sleep=lambda _seconds: None,
    events: list[str] | None = None,
) -> tuple[ManualControlService, FakePanTilt, FakeValve]:
    pan_config = PanTiltConfig(settling_delay_seconds=0.15)
    control_config = ManualControlConfig(calibration_file=tmp_path / "calibration.json")
    pan_tilt = FakePanTilt(pan_config, events)
    valve = FakeValve(events)
    service = ManualControlService(
        pan_config,
        control_config,
        pan_tilt=pan_tilt,
        valve=valve,
        sleep=sleep,
        clock=clock,
    )
    return service, pan_tilt, valve


def test_dpad_tracks_commanded_position_centers_and_clamps(tmp_path: Path) -> None:
    service, pan_tilt, _ = make_service(tmp_path)

    assert service.status()["pan"] == service.status()["tilt"] == 85
    assert service.status()["position_commanded"] is False
    assert service.move("up", 3) == PanTiltPosition(85, 88)
    assert service.move("left", 5) == PanTiltPosition(90, 88)
    assert service.move("right", 5) == PanTiltPosition(85, 88)
    for _ in range(30):
        service.move("left", 5)

    assert pan_tilt.moves[-1] == PanTiltPosition(150, 88)
    for _ in range(30):
        service.move("right", 5)
        service.move("down", 5)

    assert pan_tilt.moves[-1] == PanTiltPosition(30, 70)
    assert service.move("center", 3) == PanTiltPosition(85, 85)
    assert service.status()["pan"] == service.status()["tilt"] == 85
    assert service.status()["position_commanded"] is True


def test_calibration_rejects_uncommanded_startup_reference(tmp_path: Path) -> None:
    service, _, _ = make_service(tmp_path)
    service.set_active_calibration_pixel(640, 360, frame_width=1280, frame_height=720)

    with pytest.raises(RuntimeError, match="Move or center"):
        service.save_active_calibration_point()


def test_display_click_maps_scaled_and_letterboxed_image_to_native_pixel() -> None:
    assert display_click_to_frame_pixel(400, 225, 800, 450, 1280, 720) == (640, 360)
    assert display_click_to_frame_pixel(400, 300, 800, 600, 1280, 720) == (640, 360)

    with pytest.raises(ValueError, match="inside the rendered camera image"):
        display_click_to_frame_pixel(400, 50, 800, 600, 1280, 720)


def test_active_calibration_point_is_shared_service_state(tmp_path: Path) -> None:
    service, _, _ = make_service(tmp_path)

    assert service.status()["active_calibration_point"] == 1
    assert service.select_calibration_point(4) == 4
    assert service.status()["active_calibration_point"] == 4

    with pytest.raises(ValueError, match="integer from 1 to 9"):
        service.select_calibration_point(10)


def test_save_active_point_uses_current_backend_commanded_position(tmp_path: Path) -> None:
    service, _, _ = make_service(tmp_path)
    service.select_calibration_point(4)
    service.set_active_calibration_pixel(834, 261, frame_width=1280, frame_height=720)
    service.move("right", 3)
    service.move("up", 5)

    record = service.save_active_calibration_point()

    assert record.point == 4
    assert record.pixel_x == 834
    assert record.pixel_y == 261
    assert record.pan == 82
    assert record.tilt == 90
    assert CalibrationStore(tmp_path / "calibration.json").load() == [record]


def test_fire_cooldown_is_server_side_and_movement_remains_available(tmp_path: Path) -> None:
    now = [100.0]
    service, _, valve = make_service(tmp_path, clock=lambda: now[0])

    service.fire()
    assert valve.state is ValveState.CLOSED
    assert service.status()["state"] == ControlState.COOLDOWN.value
    assert service.status()["cooldown_remaining_seconds"] == 10.0
    with pytest.raises(FireCooldownError):
        service.fire()

    assert service.move("right", 3) == PanTiltPosition(82, 85)
    assert service.status()["state"] == ControlState.COOLDOWN.value
    now[0] = 110.0
    assert service.status()["state"] == ControlState.IDLE.value
    service.fire()


def test_control_pipeline_moves_settles_then_fires(tmp_path: Path) -> None:
    events: list[str] = []
    delays: list[float] = []
    service: ManualControlService

    def sleep(seconds: float) -> None:
        delays.append(seconds)
        events.append("settle" if seconds == 0.15 else "pulse")

    service, _, _ = make_service(tmp_path, sleep=sleep, events=events)
    result = service.move_and_fire(200, 60)

    assert result == PanTiltPosition(150, 70)
    assert events == ["move", "settle", "close", "open", "pulse", "close"]
    assert delays == [0.15, 0.25]


def test_fire_and_movement_are_serialized(tmp_path: Path) -> None:
    events: list[str] = []
    fire_started = threading.Event()
    release_fire = threading.Event()
    move_started = threading.Event()
    move_finished = threading.Event()
    failures: list[Exception] = []

    def controlled_sleep(seconds: float) -> None:
        if seconds == 0.25:
            events.append("pulse")
            fire_started.set()
            assert release_fire.wait(timeout=1)
        else:
            assert seconds == 0.15
            events.append("settle")

    service, _, _ = make_service(tmp_path, sleep=controlled_sleep, events=events)

    def fire() -> None:
        try:
            service.fire()
        except Exception as exc:
            failures.append(exc)

    def move() -> None:
        move_started.set()
        try:
            service.move("left", 3)
        except Exception as exc:
            failures.append(exc)
        finally:
            move_finished.set()

    fire_thread = threading.Thread(target=fire)
    move_thread = threading.Thread(target=move)
    fire_thread.start()
    assert fire_started.wait(timeout=1)
    move_thread.start()
    assert move_started.wait(timeout=1)
    assert not move_finished.wait(timeout=0.05)
    assert "move" not in events

    release_fire.set()
    fire_thread.join(timeout=1)
    move_thread.join(timeout=1)

    assert not fire_thread.is_alive()
    assert not move_thread.is_alive()
    assert failures == []
    assert events == ["close", "open", "pulse", "close", "move", "settle"]


def test_status_exposes_moving_settling_and_firing_states(tmp_path: Path) -> None:
    observed: list[str] = []
    service, pan_tilt, valve = make_service(tmp_path)
    original_move = pan_tilt.move_to_smooth
    original_open = valve.open

    def inspecting_move(*args: float, **kwargs: float) -> PanTiltPosition:
        observed.append(str(service.status()["state"]))
        return original_move(*args, **kwargs)

    def inspecting_sleep(_seconds: float) -> None:
        observed.append(str(service.status()["state"]))

    def inspecting_open() -> None:
        original_open()
        observed.append(str(service.status()["state"]))

    pan_tilt.move_to_smooth = inspecting_move  # type: ignore[method-assign]
    valve.open = inspecting_open  # type: ignore[method-assign]
    service._sleep = inspecting_sleep
    service.move("right", 3)
    service.fire()

    assert observed == ["MOVING", "SETTLING", "FIRING", "FIRING"]


def test_calibration_store_saves_and_replaces_one_of_nine_points(tmp_path: Path) -> None:
    service, _, _ = make_service(tmp_path)
    service.select_calibration_point(4)
    pixel_only = service.set_active_calibration_pixel(640, 360, frame_width=1280, frame_height=720)

    assert pixel_only.pan is None and pixel_only.tilt is None
    assert CalibrationStore(tmp_path / "calibration.json").load() == [pixel_only]
    assert service.status()["completed_calibration_count"] == 0
    assert service.status()["calibration_points"] == [{
        "point": 4,
        "pixel_x": 640,
        "pixel_y": 360,
        "pan": None,
        "tilt": None,
        "pixel_selected": True,
        "aim_saved": False,
        "complete": False,
    }]

    service.move("right", 3)
    first = service.save_active_calibration_point()
    reclicked = service.set_active_calibration_pixel(700, 400, frame_width=1280, frame_height=720)
    service.move("up", 5)
    replacement = service.save_active_calibration_point()

    assert first.pan == 82
    assert reclicked.pan == 82 and reclicked.tilt == 85
    assert replacement.pixel_x == 700
    assert replacement.pixel_y == 400
    assert replacement.pan == 82 and replacement.tilt == 90
    assert service.status()["completed_calibration_count"] == 1
    records = CalibrationStore(tmp_path / "calibration.json").load()
    assert records == [replacement]
    raw = json.loads((tmp_path / "calibration.json").read_text(encoding="utf-8"))
    assert raw["points"][0] == {
        "point": 4,
        "pixel_x": 700,
        "pixel_y": 400,
        "pan": 82.0,
        "tilt": 90.0,
    }
