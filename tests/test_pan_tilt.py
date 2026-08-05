from __future__ import annotations

from dataclasses import replace

import pytest

from squirrel_shooter.pan_tilt import (
    PanTiltConfig,
    PanTiltController,
    PanTiltPosition,
    clamp_angle,
    generate_coordinated_path,
)
from squirrel_shooter.motion import MotionController


class FakeServoDriver:
    def __init__(self, *, fail_cleanup: bool = False) -> None:
        self.configured: list[tuple[int, int, int]] = []
        self.commands: list[tuple[int, float]] = []
        self.released: list[int] = []
        self.cleanup_calls = 0
        self.fail_cleanup = fail_cleanup

    def configure_channel(self, channel: int, pulse_min_us: int, pulse_max_us: int) -> None:
        self.configured.append((channel, pulse_min_us, pulse_max_us))

    def set_angle(self, channel: int, angle: float) -> None:
        self.commands.append((channel, angle))

    def release(self, channel: int) -> None:
        self.released.append(channel)

    def cleanup(self) -> None:
        self.cleanup_calls += 1
        if self.fail_cleanup:
            raise RuntimeError("cleanup failed")


@pytest.fixture
def config() -> PanTiltConfig:
    return PanTiltConfig(
        movement_speed=100.0,
        fast_acquisition_speed=200.0,
        step_interval_seconds=0.1,
        settling_delay_seconds=0.0,
    )


def test_clamp_angle_enforces_safe_limits() -> None:
    assert clamp_angle(10, 30, 150) == 30
    assert clamp_angle(90, 30, 150) == 90
    assert clamp_angle(170, 30, 150) == 150
    with pytest.raises(ValueError, match="finite"):
        clamp_angle(float("nan"), 30, 150)


def test_coordinated_path_interpolates_both_axes_and_hits_exact_endpoint() -> None:
    start = PanTiltPosition(30, 70)
    target = PanTiltPosition(150, 130)

    path = generate_coordinated_path(start, target, speed=60, step_interval_seconds=0.5)

    assert len(path) == 4
    assert path[0] == PanTiltPosition(60, 85)
    assert path[1] == PanTiltPosition(90, 100)
    assert path[-1] == target
    assert all(
        (point.pan - start.pan) / (target.pan - start.pan)
        == pytest.approx((point.tilt - start.tilt) / (target.tilt - start.tilt))
        for point in path
    )


def test_path_generation_rejects_invalid_speed() -> None:
    position = PanTiltPosition(90, 85)
    with pytest.raises(ValueError, match="speed"):
        generate_coordinated_path(position, position, speed=0, step_interval_seconds=0.1)


def test_startup_configures_channels_without_moving(config: PanTiltConfig) -> None:
    driver = FakeServoDriver()

    controller = PanTiltController(config, driver=driver, sleep=lambda _: None)

    assert driver.configured == [(0, 600, 2400), (1, 600, 2400)]
    assert driver.commands == []
    assert controller.position is None
    assert controller.last_commanded_pan is None
    assert controller.last_commanded_tilt is None


def test_legacy_motion_controller_name_points_to_pan_tilt_controller() -> None:
    assert MotionController is PanTiltController


def test_direct_commands_clamp_and_track_each_axis(config: PanTiltConfig) -> None:
    driver = FakeServoDriver()
    controller = PanTiltController(config, driver=driver, sleep=lambda _: None)

    assert controller.move_pan(-20) == 30
    assert controller.position is None
    assert controller.last_commanded_pan == 30
    assert controller.move_tilt(200) == 150

    assert controller.position == PanTiltPosition(30, 150)
    assert driver.commands == [(0, 30), (1, 150)]


def test_smooth_move_uses_coordinated_writes_and_tracks_final_endpoint(config: PanTiltConfig) -> None:
    driver = FakeServoDriver()
    controller = PanTiltController(config, driver=driver, sleep=lambda _: None)

    result = controller.move_to_smooth(150, 145)

    pan_commands = [angle for channel, angle in driver.commands if channel == config.pan_channel]
    tilt_commands = [angle for channel, angle in driver.commands if channel == config.tilt_channel]
    assert len(pan_commands) == len(tilt_commands) == 7
    assert pan_commands[-1] == 150
    assert tilt_commands[-1] == 145
    assert result == PanTiltPosition(150, 145)
    assert controller.position == result


def test_smooth_move_uses_each_axes_last_command_when_tracking_is_partial(config: PanTiltConfig) -> None:
    driver = FakeServoDriver()
    controller = PanTiltController(config, driver=driver, sleep=lambda _: None)
    controller.move_pan(config.pan_min)
    driver.commands.clear()

    controller.move_to_smooth(config.pan_max, config.tilt_center)

    first_pan = next(angle for channel, angle in driver.commands if channel == config.pan_channel)
    first_tilt = next(angle for channel, angle in driver.commands if channel == config.tilt_channel)
    assert first_pan > config.pan_min
    assert first_pan < config.pan_center
    assert first_tilt == config.tilt_center


def test_fast_move_is_interpolated_and_limit_enforced(config: PanTiltConfig) -> None:
    driver = FakeServoDriver()
    controller = PanTiltController(config, driver=driver, sleep=lambda _: None)

    result = controller.move_to_fast(300, -10)

    assert result == PanTiltPosition(150, 70)
    assert controller.position == result
    assert len(driver.commands) > 2
    assert all(
        config.pan_min <= angle <= config.pan_max
        if channel == config.pan_channel
        else config.tilt_min <= angle <= config.tilt_max
        for channel, angle in driver.commands
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"pan_min": 150, "pan_center": 90, "pan_max": 30}, "pan limits"),
        ({"tilt_min": 70, "tilt_center": 160, "tilt_max": 150}, "tilt limits"),
        ({"pan_pulse_min_us": 2400, "pan_pulse_max_us": 600}, "pulse width"),
        ({"pan_channel": 1, "tilt_channel": 1}, "must be different"),
        ({"park_pan": 10}, "park_pan"),
        ({"movement_speed": 0}, "movement_speed"),
        ({"movement_speed": 100, "fast_acquisition_speed": 90}, "fast_acquisition_speed"),
    ],
)
def test_configuration_rejects_reversed_or_invalid_values(
    changes: dict[str, float | int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(PanTiltConfig(), **changes)


def test_cleanup_parks_releases_when_configured_and_runs_once(config: PanTiltConfig) -> None:
    configured = replace(config, release_pwm_after_movement=True)
    driver = FakeServoDriver()
    controller = PanTiltController(configured, driver=driver, sleep=lambda _: None)
    controller.move_to(120, 120, release=False)

    controller.cleanup()
    controller.cleanup()

    assert controller.position == PanTiltPosition(configured.park_pan, configured.park_tilt)
    assert driver.released[-2:] == [configured.pan_channel, configured.tilt_channel]
    assert driver.cleanup_calls == 1


def test_context_manager_preserves_original_exception_if_cleanup_fails(config: PanTiltConfig) -> None:
    driver = FakeServoDriver(fail_cleanup=True)

    with pytest.raises(ValueError, match="original failure"):
        with PanTiltController(config, driver=driver, sleep=lambda _: None):
            raise ValueError("original failure")
