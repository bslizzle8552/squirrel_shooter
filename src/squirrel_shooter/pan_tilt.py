"""Safe, reusable pan/tilt servo control without valve integration."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Callable, Protocol


LOGGER = logging.getLogger(__name__)


class ServoDriver(Protocol):
    """Small hardware boundary used by the controller and test fakes."""

    def configure_channel(self, channel: int, pulse_min_us: int, pulse_max_us: int) -> None:
        """Configure one servo channel's accepted pulse range."""

    def set_angle(self, channel: int, angle: float) -> None:
        """Command one servo angle."""

    def release(self, channel: int) -> None:
        """Stop PWM output on one channel."""

    def cleanup(self) -> None:
        """Release driver-owned resources, if any."""


@dataclass(frozen=True)
class PanTiltPosition:
    """A commanded pan/tilt position in degrees."""

    pan: float
    tilt: float


@dataclass(frozen=True)
class PanTiltConfig:
    """Mechanically verified limits and conservative movement settings."""

    i2c_address: int = 0x40
    pan_channel: int = 0
    tilt_channel: int = 1
    pan_pulse_min_us: int = 600
    pan_pulse_max_us: int = 2400
    tilt_pulse_min_us: int = 600
    tilt_pulse_max_us: int = 2400
    pan_min: float = 30.0
    pan_center: float = 90.0
    pan_max: float = 150.0
    tilt_min: float = 70.0
    tilt_center: float = 85.0
    tilt_max: float = 150.0
    park_pan: float = 90.0
    park_tilt: float = 85.0
    movement_speed: float = 45.0
    fast_acquisition_speed: float = 120.0
    step_interval_seconds: float = 0.02
    settling_delay_seconds: float = 0.15
    release_pwm_after_movement: bool = False
    park_on_cleanup: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.i2c_address <= 0x7F:
            raise ValueError("i2c_address must be between 0x00 and 0x7f")
        for name, channel in (("pan_channel", self.pan_channel), ("tilt_channel", self.tilt_channel)):
            if isinstance(channel, bool) or not isinstance(channel, int) or not 0 <= channel <= 15:
                raise ValueError(f"{name} must be an integer between 0 and 15")
        if self.pan_channel == self.tilt_channel:
            raise ValueError("pan_channel and tilt_channel must be different")
        self._validate_pulse_range("pan", self.pan_pulse_min_us, self.pan_pulse_max_us)
        self._validate_pulse_range("tilt", self.tilt_pulse_min_us, self.tilt_pulse_max_us)
        self._validate_angle_range("pan", self.pan_min, self.pan_center, self.pan_max)
        self._validate_angle_range("tilt", self.tilt_min, self.tilt_center, self.tilt_max)
        self._validate_angle("park_pan", self.park_pan, self.pan_min, self.pan_max)
        self._validate_angle("park_tilt", self.park_tilt, self.tilt_min, self.tilt_max)
        for name, value in (
            ("movement_speed", self.movement_speed),
            ("fast_acquisition_speed", self.fast_acquisition_speed),
            ("step_interval_seconds", self.step_interval_seconds),
        ):
            if not _is_finite_number(value) or value <= 0:
                raise ValueError(f"{name} must be a finite number greater than zero")
        if self.fast_acquisition_speed < self.movement_speed:
            raise ValueError("fast_acquisition_speed must be at least movement_speed")
        if not _is_finite_number(self.settling_delay_seconds) or self.settling_delay_seconds < 0:
            raise ValueError("settling_delay_seconds must be a finite non-negative number")
        if not isinstance(self.release_pwm_after_movement, bool):
            raise ValueError("release_pwm_after_movement must be true or false")
        if not isinstance(self.park_on_cleanup, bool):
            raise ValueError("park_on_cleanup must be true or false")

    @staticmethod
    def _validate_pulse_range(axis: str, minimum: int, maximum: int) -> None:
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (minimum, maximum)):
            raise ValueError(f"{axis} pulse widths must be integers")
        if minimum <= 0 or maximum <= minimum:
            raise ValueError(f"{axis} pulse width maximum must be greater than its positive minimum")

    @staticmethod
    def _validate_angle_range(axis: str, minimum: float, center: float, maximum: float) -> None:
        if not all(_is_finite_number(value) for value in (minimum, center, maximum)):
            raise ValueError(f"{axis} angles must be finite numbers")
        if not minimum < center < maximum:
            raise ValueError(f"{axis} limits must satisfy minimum < center < maximum")

    @staticmethod
    def _validate_angle(name: str, value: float, minimum: float, maximum: float) -> None:
        if not _is_finite_number(value) or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be within its configured safe limits")


def _is_finite_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def clamp_angle(angle: float, minimum: float, maximum: float) -> float:
    """Clamp one finite angle to an already validated safe range."""

    if not _is_finite_number(angle):
        raise ValueError("servo angle must be a finite number")
    return min(max(float(angle), float(minimum)), float(maximum))


def generate_coordinated_path(
    start: PanTiltPosition,
    target: PanTiltPosition,
    *,
    speed: float,
    step_interval_seconds: float,
) -> tuple[PanTiltPosition, ...]:
    """Interpolate both axes over one duration so they finish together."""

    if not _is_finite_number(speed) or speed <= 0:
        raise ValueError("speed must be a finite number greater than zero")
    if not _is_finite_number(step_interval_seconds) or step_interval_seconds <= 0:
        raise ValueError("step_interval_seconds must be a finite number greater than zero")
    values = (start.pan, start.tilt, target.pan, target.tilt)
    if not all(_is_finite_number(value) for value in values):
        raise ValueError("path positions must contain finite angles")

    pan_delta = target.pan - start.pan
    tilt_delta = target.tilt - start.tilt
    maximum_delta = max(abs(pan_delta), abs(tilt_delta))
    steps = max(1, math.ceil(maximum_delta / (speed * step_interval_seconds)))
    return tuple(
        target
        if step == steps
        else PanTiltPosition(
            pan=start.pan + pan_delta * step / steps,
            tilt=start.tilt + tilt_delta * step / steps,
        )
        for step in range(1, steps + 1)
    )


class ServoKitDriver:
    """Lazy ServoKit adapter so importing this module never requires Pi hardware."""

    def __init__(self, *, i2c_address: int) -> None:
        try:
            from adafruit_servokit import ServoKit
        except ImportError as exc:
            raise RuntimeError(
                "Servo hardware support is not installed. Activate .venv and install the servo optional dependencies."
            ) from exc
        self._kit = ServoKit(channels=16, address=i2c_address)

    def configure_channel(self, channel: int, pulse_min_us: int, pulse_max_us: int) -> None:
        self._kit.servo[channel].set_pulse_width_range(pulse_min_us, pulse_max_us)

    def set_angle(self, channel: int, angle: float) -> None:
        self._kit.servo[channel].angle = angle

    def release(self, channel: int) -> None:
        self._kit.servo[channel].angle = None

    def cleanup(self) -> None:
        return None


class PanTiltController:
    """Limit-enforced two-axis controller with software-only position tracking."""

    def __init__(
        self,
        config: PanTiltConfig | None = None,
        *,
        driver: ServoDriver | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config or PanTiltConfig()
        self._driver = driver or ServoKitDriver(i2c_address=self.config.i2c_address)
        self._sleep = sleep
        self._last_pan: float | None = None
        self._last_tilt: float | None = None
        self._cleaned_up = False
        self._driver.configure_channel(
            self.config.pan_channel,
            self.config.pan_pulse_min_us,
            self.config.pan_pulse_max_us,
        )
        self._driver.configure_channel(
            self.config.tilt_channel,
            self.config.tilt_pulse_min_us,
            self.config.tilt_pulse_max_us,
        )
        LOGGER.info("Pan/tilt controller initialized without commanding movement")

    @property
    def position(self) -> PanTiltPosition | None:
        """Return the last fully commanded two-axis position, never servo feedback."""

        if self._last_pan is None or self._last_tilt is None:
            return None
        return PanTiltPosition(self._last_pan, self._last_tilt)

    @property
    def last_commanded_pan(self) -> float | None:
        return self._last_pan

    @property
    def last_commanded_tilt(self) -> float | None:
        return self._last_tilt

    def move_pan(self, angle: float, *, release: bool | None = None) -> float:
        clamped = clamp_angle(angle, self.config.pan_min, self.config.pan_max)
        self._log_request("pan", angle, clamped, "direct")
        self._set_pan(clamped)
        self._settle_and_maybe_release(release)
        self._log_result()
        return clamped

    def move_tilt(self, angle: float, *, release: bool | None = None) -> float:
        clamped = clamp_angle(angle, self.config.tilt_min, self.config.tilt_max)
        self._log_request("tilt", angle, clamped, "direct")
        self._set_tilt(clamped)
        self._settle_and_maybe_release(release)
        self._log_result()
        return clamped

    def move_to(
        self,
        pan_angle: float,
        tilt_angle: float,
        *,
        release: bool | None = None,
    ) -> PanTiltPosition:
        target = self._clamped_position(pan_angle, tilt_angle)
        self._log_two_axis_request(pan_angle, tilt_angle, target, "direct")
        self._set_position(target)
        self._settle_and_maybe_release(release)
        self._log_result()
        return target

    def move_to_smooth(
        self,
        pan_angle: float,
        tilt_angle: float,
        *,
        speed: float | None = None,
        settling_delay_seconds: float | None = None,
        release: bool | None = None,
    ) -> PanTiltPosition:
        return self._move_interpolated(
            pan_angle,
            tilt_angle,
            speed=self.config.movement_speed if speed is None else speed,
            settling_delay_seconds=settling_delay_seconds,
            release=release,
            mode="smooth",
        )

    def move_to_fast(
        self,
        pan_angle: float,
        tilt_angle: float,
        *,
        speed: float | None = None,
        settling_delay_seconds: float | None = None,
        release: bool | None = None,
    ) -> PanTiltPosition:
        return self._move_interpolated(
            pan_angle,
            tilt_angle,
            speed=self.config.fast_acquisition_speed if speed is None else speed,
            settling_delay_seconds=settling_delay_seconds,
            release=release,
            mode="fast",
        )

    def center(self, *, release: bool | None = None) -> PanTiltPosition:
        return self.move_to_smooth(
            self.config.pan_center,
            self.config.tilt_center,
            release=release,
        )

    def park(self, *, release: bool | None = None) -> PanTiltPosition:
        LOGGER.info("Parking pan/tilt at pan=%.2f tilt=%.2f", self.config.park_pan, self.config.park_tilt)
        return self.move_to_smooth(self.config.park_pan, self.config.park_tilt, release=release)

    def release_pwm(self) -> None:
        LOGGER.info("Releasing pan and tilt PWM; tracked position remains the last commanded position")
        self._driver.release(self.config.pan_channel)
        self._driver.release(self.config.tilt_channel)

    def cleanup(self) -> None:
        """Park when configured, release when configured, and clean the driver once."""

        if self._cleaned_up:
            return
        LOGGER.info("Cleaning up pan/tilt controller")
        try:
            if self.config.park_on_cleanup:
                self.park(release=self.config.release_pwm_after_movement)
            elif self.config.release_pwm_after_movement:
                self.release_pwm()
        except BaseException:
            if self.config.release_pwm_after_movement:
                try:
                    self.release_pwm()
                except Exception:
                    LOGGER.exception("PWM release also failed after a park error")
            self._cleaned_up = True
            try:
                self._driver.cleanup()
            except Exception:
                LOGGER.exception("Driver cleanup also failed after a park error")
            raise
        self._cleaned_up = True
        self._driver.cleanup()

    def __enter__(self) -> PanTiltController:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        try:
            self.cleanup()
        except Exception:
            if exc_type is None:
                raise
            LOGGER.exception("Pan/tilt cleanup failed while preserving the original exception")
        return False

    def _move_interpolated(
        self,
        pan_angle: float,
        tilt_angle: float,
        *,
        speed: float,
        settling_delay_seconds: float | None,
        release: bool | None,
        mode: str,
    ) -> PanTiltPosition:
        target = self._clamped_position(pan_angle, tilt_angle)
        start = PanTiltPosition(
            self.config.pan_center if self._last_pan is None else self._last_pan,
            self.config.tilt_center if self._last_tilt is None else self._last_tilt,
        )
        path = generate_coordinated_path(
            start,
            target,
            speed=speed,
            step_interval_seconds=self.config.step_interval_seconds,
        )
        self._log_two_axis_request(pan_angle, tilt_angle, target, mode)
        LOGGER.info(
            "%s movement path: start pan=%.2f tilt=%.2f, steps=%d, speed=%.2f deg/s",
            mode.capitalize(),
            start.pan,
            start.tilt,
            len(path),
            speed,
        )
        for index, point in enumerate(path):
            self._set_position(point)
            if index < len(path) - 1:
                self._sleep(self.config.step_interval_seconds)
        self._settle_and_maybe_release(release, settling_delay_seconds)
        self._log_result()
        return target

    def _clamped_position(self, pan_angle: float, tilt_angle: float) -> PanTiltPosition:
        return PanTiltPosition(
            clamp_angle(pan_angle, self.config.pan_min, self.config.pan_max),
            clamp_angle(tilt_angle, self.config.tilt_min, self.config.tilt_max),
        )

    def _set_pan(self, angle: float) -> None:
        self._driver.set_angle(self.config.pan_channel, angle)
        self._last_pan = angle

    def _set_tilt(self, angle: float) -> None:
        self._driver.set_angle(self.config.tilt_channel, angle)
        self._last_tilt = angle

    def _set_position(self, position: PanTiltPosition) -> None:
        self._set_pan(position.pan)
        self._set_tilt(position.tilt)

    def _settle_and_maybe_release(
        self,
        release: bool | None,
        settling_delay_seconds: float | None = None,
    ) -> None:
        delay = self.config.settling_delay_seconds if settling_delay_seconds is None else settling_delay_seconds
        if not _is_finite_number(delay) or delay < 0:
            raise ValueError("settling_delay_seconds must be a finite non-negative number")
        if delay:
            self._sleep(float(delay))
        should_release = self.config.release_pwm_after_movement if release is None else release
        if not isinstance(should_release, bool):
            raise ValueError("release must be true, false, or None")
        if should_release:
            self.release_pwm()

    def _log_request(self, axis: str, requested: float, clamped: float, mode: str) -> None:
        LOGGER.info(
            "%s request: requested=%.2f clamped=%.2f mode=%s",
            axis.capitalize(),
            requested,
            clamped,
            mode,
        )

    def _log_two_axis_request(
        self,
        requested_pan: float,
        requested_tilt: float,
        target: PanTiltPosition,
        mode: str,
    ) -> None:
        LOGGER.info(
            "Pan/tilt request: requested=(%.2f, %.2f) clamped=(%.2f, %.2f) mode=%s",
            requested_pan,
            requested_tilt,
            target.pan,
            target.tilt,
            mode,
        )

    def _log_result(self) -> None:
        LOGGER.info(
            "Tracked position after command: pan=%s tilt=%s",
            "uncommanded" if self._last_pan is None else f"{self._last_pan:.2f}",
            "uncommanded" if self._last_tilt is None else f"{self._last_tilt:.2f}",
        )
