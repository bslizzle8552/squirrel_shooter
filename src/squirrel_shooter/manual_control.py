"""Serialized manual aiming, calibration, and firing foundation."""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Protocol

from .pan_tilt import PanTiltConfig, PanTiltController, PanTiltPosition, clamp_angle
from .valve import (
    DisabledValveController,
    GPIOValveController,
    ValveConfig,
    ValveController,
    ValveState,
    pulse_valve,
)


LOGGER = logging.getLogger(__name__)


class ControlState(str, Enum):
    IDLE = "IDLE"
    MOVING = "MOVING"
    SETTLING = "SETTLING"
    FIRING = "FIRING"
    COOLDOWN = "COOLDOWN"


class ControlError(RuntimeError):
    """Base error for a rejected control request."""


class ControlUnavailableError(ControlError):
    """Requested hardware is not configured or could not initialize."""


class FireCooldownError(ControlError):
    """A fire request was rejected by the server-side cooldown."""

    def __init__(self, remaining_seconds: float) -> None:
        self.remaining_seconds = remaining_seconds
        super().__init__(f"Valve cooldown active for {math.ceil(remaining_seconds)} more seconds")


@dataclass(frozen=True)
class ManualControlConfig:
    servo_enabled: bool = False
    default_step_degrees: int = 3
    allowed_step_degrees: tuple[int, ...] = (1, 3, 5)
    fire_pulse_seconds: float = 0.25
    fire_cooldown_seconds: float = 10.0
    calibration_file: Path = Path("config/calibration_points.json")

    def __post_init__(self) -> None:
        if not isinstance(self.servo_enabled, bool):
            raise ValueError("servo_enabled must be true or false")
        if not self.allowed_step_degrees or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in self.allowed_step_degrees
        ):
            raise ValueError("allowed_step_degrees must contain positive integer steps")
        if len(set(self.allowed_step_degrees)) != len(self.allowed_step_degrees):
            raise ValueError("allowed_step_degrees must not contain duplicates")
        if self.default_step_degrees not in self.allowed_step_degrees:
            raise ValueError("default_step_degrees must be one of allowed_step_degrees")
        for name, value in (
            ("fire_pulse_seconds", self.fire_pulse_seconds),
            ("fire_cooldown_seconds", self.fire_cooldown_seconds),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{name} must be a finite number greater than zero")
        if not isinstance(self.calibration_file, Path):
            raise ValueError("calibration_file must be a path")


def display_click_to_frame_pixel(
    display_x: float,
    display_y: float,
    display_width: float,
    display_height: float,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int]:
    """Map an object-fit ``contain`` image click to a native frame pixel."""

    display_values = {
        "display_x": display_x,
        "display_y": display_y,
        "display_width": display_width,
        "display_height": display_height,
    }
    for name, value in display_values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite number")
    if display_width <= 0 or display_height <= 0:
        raise ValueError("Displayed image dimensions must be greater than zero")
    for name, value in (("frame_width", frame_width), ("frame_height", frame_height)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    scale = min(display_width / frame_width, display_height / frame_height)
    rendered_width = frame_width * scale
    rendered_height = frame_height * scale
    offset_x = (display_width - rendered_width) / 2
    offset_y = (display_height - rendered_height) / 2
    image_x = display_x - offset_x
    image_y = display_y - offset_y
    if image_x < 0 or image_y < 0 or image_x > rendered_width or image_y > rendered_height:
        raise ValueError("Click must be inside the rendered camera image")

    pixel_x = min(frame_width - 1, int(image_x / scale))
    pixel_y = min(frame_height - 1, int(image_y / scale))
    return pixel_x, pixel_y


@dataclass(frozen=True)
class CalibrationPoint:
    point: int
    pixel_x: int | None
    pixel_y: int | None
    pan: float | None
    tilt: float | None

    def __post_init__(self) -> None:
        if isinstance(self.point, bool) or not isinstance(self.point, int) or not 1 <= self.point <= 9:
            raise ValueError("Calibration point must be an integer from 1 to 9")
        for name, value in (("pixel_x", self.pixel_x), ("pixel_y", self.pixel_y)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be null or a non-negative integer")
        if (self.pixel_x is None) != (self.pixel_y is None):
            raise ValueError("pixel_x and pixel_y must either both be set or both be null")
        for name, value in (("pan", self.pan), ("tilt", self.tilt)):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be null or a finite angle")
        if (self.pan is None) != (self.tilt is None):
            raise ValueError("pan and tilt must either both be set or both be null")

    @property
    def pixel_selected(self) -> bool:
        return self.pixel_x is not None and self.pixel_y is not None

    @property
    def aim_saved(self) -> bool:
        return self.pan is not None and self.tilt is not None

    @property
    def complete(self) -> bool:
        return self.pixel_selected and self.aim_saved


class CalibrationStore:
    """Small JSON store for the nine physical garden calibration points."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._cache: list[CalibrationPoint] | None = None

    def load(self) -> list[CalibrationPoint]:
        with self._lock:
            return list(self._points_unlocked())

    def save(self, point: CalibrationPoint) -> list[CalibrationPoint]:
        with self._lock:
            points = {item.point: item for item in self._points_unlocked()}
            points[point.point] = point
            ordered = [points[index] for index in sorted(points)]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(f".{self.path.name}.tmp")
            temporary.write_text(
                json.dumps({"points": [asdict(item) for item in ordered]}, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, self.path)
            self._cache = ordered
            return list(ordered)

    def _points_unlocked(self) -> list[CalibrationPoint]:
        if self._cache is None:
            self._cache = self._load_unlocked()
        return self._cache

    def _load_unlocked(self) -> list[CalibrationPoint]:
        if not self.path.exists():
            return []
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        records = raw.get("points") if isinstance(raw, dict) else None
        if not isinstance(records, list):
            raise ValueError("Calibration file must contain a points list")
        points = [CalibrationPoint(**record) for record in records]
        if len({point.point for point in points}) != len(points):
            raise ValueError("Calibration file contains duplicate point numbers")
        return sorted(points, key=lambda point: point.point)


class PanTiltControl(Protocol):
    config: PanTiltConfig

    def move_to_smooth(
        self,
        pan_angle: float,
        tilt_angle: float,
        *,
        settling_delay_seconds: float | None = None,
    ) -> PanTiltPosition:
        ...

    def cleanup(self) -> None:
        ...


class ManualControlService:
    """One lock and one state model for every present and future control path."""

    def __init__(
        self,
        pan_tilt_config: PanTiltConfig,
        control_config: ManualControlConfig,
        *,
        pan_tilt: PanTiltControl | None = None,
        valve: ValveController | None = None,
        calibration_store: CalibrationStore | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        servo_error: str | None = None,
        valve_error: str | None = None,
    ) -> None:
        self.pan_tilt_config = pan_tilt_config
        self.config = control_config
        self._pan_tilt = pan_tilt
        self._valve = valve or DisabledValveController()
        self._calibration = calibration_store or CalibrationStore(control_config.calibration_file)
        self._sleep = sleep
        self._clock = clock
        self._lock = threading.Lock()
        self._transient_state: ControlState | None = None
        self._pan = float(pan_tilt_config.pan_center)
        self._tilt = float(pan_tilt_config.tilt_center)
        self._position_commanded = False
        self._active_calibration_point = 1
        self._last_fire_completed_at: float | None = None
        self._servo_error = servo_error
        self._valve_error = valve_error

    @property
    def servo_available(self) -> bool:
        return self._pan_tilt is not None

    @property
    def valve_available(self) -> bool:
        return not isinstance(self._valve, DisabledValveController)

    def status(self) -> dict[str, object]:
        now = self._clock()
        remaining = self._cooldown_remaining(now)
        state = self._transient_state or (ControlState.COOLDOWN if remaining > 0 else ControlState.IDLE)
        try:
            points = [self._calibration_point_payload(point) for point in self._calibration.load()]
            calibration_error = None
        except (OSError, ValueError, TypeError) as exc:
            points = []
            calibration_error = str(exc)
        return {
            "state": state.value,
            "pan": self._display_angle(self._pan),
            "tilt": self._display_angle(self._tilt),
            "position_commanded": self._position_commanded,
            "pan_min": self.pan_tilt_config.pan_min,
            "pan_max": self.pan_tilt_config.pan_max,
            "tilt_min": self.pan_tilt_config.tilt_min,
            "tilt_max": self.pan_tilt_config.tilt_max,
            "default_step": self.config.default_step_degrees,
            "allowed_steps": list(self.config.allowed_step_degrees),
            "servo_available": self.servo_available,
            "valve_available": self.valve_available,
            "valve_state": self._valve.state.value,
            "cooldown_remaining_seconds": round(remaining, 3),
            "fire_pulse_seconds": self.config.fire_pulse_seconds,
            "active_calibration_point": self._active_calibration_point,
            "calibration_points": points,
            "completed_calibration_count": sum(bool(point["complete"]) for point in points),
            "servo_error": self._servo_error,
            "valve_error": self._valve_error,
            "calibration_error": calibration_error,
        }

    def move(self, direction: str, step_degrees: int) -> PanTiltPosition:
        if direction not in {"up", "down", "left", "right", "center"}:
            raise ValueError("Direction must be up, down, left, right, or center")
        if step_degrees not in self.config.allowed_step_degrees:
            raise ValueError("Step must be one of the configured movement steps")
        if self._pan_tilt is None:
            raise ControlUnavailableError(self._servo_error or "Servo control is disabled")
        with self._lock:
            if direction == "center":
                target = PanTiltPosition(self.pan_tilt_config.pan_center, self.pan_tilt_config.tilt_center)
            else:
                pan_delta = step_degrees if direction == "left" else -step_degrees if direction == "right" else 0
                tilt_delta = step_degrees if direction == "up" else -step_degrees if direction == "down" else 0
                target = PanTiltPosition(
                    clamp_angle(self._pan + pan_delta, self.pan_tilt_config.pan_min, self.pan_tilt_config.pan_max),
                    clamp_angle(self._tilt + tilt_delta, self.pan_tilt_config.tilt_min, self.pan_tilt_config.tilt_max),
                )
            return self._move_locked(target)

    def fire(self) -> None:
        if not self.valve_available:
            raise ControlUnavailableError(self._valve_error or "Valve control is disabled until a GPIO pin is configured")
        with self._lock:
            self._fire_locked()

    def move_and_fire(self, pan: float, tilt: float) -> PanTiltPosition:
        """Shared future targeting pipeline: target, move, settle, fire, cooldown."""

        if self._pan_tilt is None:
            raise ControlUnavailableError(self._servo_error or "Servo control is disabled")
        if not self.valve_available:
            raise ControlUnavailableError(self._valve_error or "Valve control is disabled")
        target = PanTiltPosition(
            clamp_angle(pan, self.pan_tilt_config.pan_min, self.pan_tilt_config.pan_max),
            clamp_angle(tilt, self.pan_tilt_config.tilt_min, self.pan_tilt_config.tilt_max),
        )
        with self._lock:
            position = self._move_locked(target)
            self._fire_locked()
            return position

    def save_calibration_point(
        self,
        point: int,
        *,
        pixel_x: int | None = None,
        pixel_y: int | None = None,
    ) -> CalibrationPoint:
        with self._lock:
            if not self._position_commanded:
                raise ControlError("Move or center the servos before saving a calibration point")
            record = CalibrationPoint(point, pixel_x, pixel_y, self._pan, self._tilt)
            self._calibration.save(record)
            return record

    def select_calibration_point(self, point: int) -> int:
        """Select the one server-side calibration point shared by every client."""

        if isinstance(point, bool) or not isinstance(point, int) or not 1 <= point <= 9:
            raise ValueError("Calibration point must be an integer from 1 to 9")
        with self._lock:
            self._active_calibration_point = point
            return point

    def set_active_calibration_pixel(
        self,
        pixel_x: int,
        pixel_y: int,
        *,
        frame_width: int,
        frame_height: int,
    ) -> CalibrationPoint:
        """Store a native-frame pixel for the backend's active point."""

        for name, value in (("frame_width", frame_width), ("frame_height", frame_height)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(pixel_x, bool) or not isinstance(pixel_x, int) or not 0 <= pixel_x < frame_width:
            raise ValueError("pixel_x must be inside the current camera frame")
        if isinstance(pixel_y, bool) or not isinstance(pixel_y, int) or not 0 <= pixel_y < frame_height:
            raise ValueError("pixel_y must be inside the current camera frame")
        with self._lock:
            existing = next(
                (point for point in self._calibration.load() if point.point == self._active_calibration_point),
                None,
            )
            record = CalibrationPoint(
                self._active_calibration_point,
                pixel_x,
                pixel_y,
                existing.pan if existing is not None else None,
                existing.tilt if existing is not None else None,
            )
            self._calibration.save(record)
            return record

    def save_active_calibration_point(self) -> CalibrationPoint:
        """Save current commanded aim while preserving the active point's pixel."""

        with self._lock:
            if not self._position_commanded:
                raise ControlError("Move or center the servos before saving a calibration point")
            existing = next(
                (point for point in self._calibration.load() if point.point == self._active_calibration_point),
                None,
            )
            if existing is None or not existing.pixel_selected:
                raise ControlError("Click the center of the active calibration block in the camera image before saving")
            record = CalibrationPoint(
                self._active_calibration_point,
                existing.pixel_x,
                existing.pixel_y,
                self._pan,
                self._tilt,
            )
            self._calibration.save(record)
            return record

    def cleanup(self) -> None:
        try:
            self._valve.cleanup()
        finally:
            if self._pan_tilt is not None:
                self._pan_tilt.cleanup()

    def _move_locked(self, target: PanTiltPosition) -> PanTiltPosition:
        if self._valve.state is ValveState.OPEN:
            raise ControlError("Servo movement rejected while the valve is open")
        try:
            self._transient_state = ControlState.MOVING
            result = self._pan_tilt.move_to_smooth(  # type: ignore[union-attr]
                target.pan,
                target.tilt,
                settling_delay_seconds=0,
            )
            self._pan, self._tilt = result.pan, result.tilt
            self._position_commanded = True
            self._transient_state = ControlState.SETTLING
            if self.pan_tilt_config.settling_delay_seconds:
                self._sleep(self.pan_tilt_config.settling_delay_seconds)
            return result
        finally:
            self._transient_state = None

    def _fire_locked(self) -> None:
        remaining = self._cooldown_remaining(self._clock())
        if remaining > 0:
            raise FireCooldownError(remaining)
        self._transient_state = ControlState.FIRING
        try:
            pulse_valve(self._valve, self.config.fire_pulse_seconds, sleep=self._sleep)
        finally:
            self._last_fire_completed_at = self._clock()
            self._transient_state = None

    def _cooldown_remaining(self, now: float) -> float:
        if self._last_fire_completed_at is None:
            return 0.0
        return max(0.0, self.config.fire_cooldown_seconds - (now - self._last_fire_completed_at))

    @staticmethod
    def _display_angle(angle: float) -> int | float:
        return int(angle) if float(angle).is_integer() else round(angle, 2)

    @staticmethod
    def _calibration_point_payload(point: CalibrationPoint) -> dict[str, object]:
        payload: dict[str, object] = asdict(point)
        payload.update(
            pixel_selected=point.pixel_selected,
            aim_saved=point.aim_saved,
            complete=point.complete,
        )
        return payload


def build_manual_control_service(
    pan_tilt_config: PanTiltConfig,
    control_config: ManualControlConfig,
    valve_config: ValveConfig,
) -> ManualControlService:
    """Build hardware boundaries without letting optional hardware break the camera app."""

    pan_tilt: PanTiltController | None = None
    servo_error: str | None = None
    if control_config.servo_enabled:
        try:
            pan_tilt = PanTiltController(pan_tilt_config)
        except Exception as exc:
            servo_error = f"{type(exc).__name__}: {exc}"
            LOGGER.error("Manual servo control unavailable: %s", servo_error, exc_info=True)
    else:
        servo_error = "Set manual_control.servo_enabled to true after hardware verification"
    valve: ValveController = DisabledValveController()
    valve_error: str | None = None
    if valve_config.enabled:
        try:
            valve = GPIOValveController(valve_config)
        except Exception as exc:
            valve_error = f"{type(exc).__name__}: {exc}"
            LOGGER.error("Manual valve control unavailable: %s", valve_error, exc_info=True)
    else:
        valve_error = (
            f"BCM GPIO{valve_config.gpio_pin} is configured; set valve.enabled to true only during supervised testing"
            if valve_config.gpio_pin is not None
            else "Configure a verified valve.gpio_pin, then set valve.enabled to true"
        )
    return ManualControlService(
        pan_tilt_config,
        control_config,
        pan_tilt=pan_tilt,
        valve=valve,
        servo_error=servo_error,
        valve_error=valve_error,
    )
