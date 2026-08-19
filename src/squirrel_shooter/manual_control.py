"""Serialized manual aiming, calibration, and firing foundation."""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Protocol

from .manual_fire_recording import (
    ManualFireEvent,
    ManualFireRecorder,
    ManualFireRecordingConfig,
    ManualFireRecordingSink,
    new_auto_fire_event_id,
    new_manual_fire_event_id,
)
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
    PARKING = "PARKING"
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


class AutomaticEngagementError(ControlError):
    """Typed automatic-engagement rejection or failure."""

    def __init__(self, reason: str, message: str | None = None, *, shot_attempted: bool = False) -> None:
        self.reason = reason
        self.shot_attempted = shot_attempted
        super().__init__(message or reason.replace("_", " "))


@dataclass(frozen=True)
class ManualControlConfig:
    servo_enabled: bool = False
    default_step_degrees: int = 3
    allowed_step_degrees: tuple[int, ...] = (1, 3, 5)
    fire_pulse_seconds: float = 0.25
    fire_cooldown_seconds: float = 10.0
    calibration_file: Path = Path("config/calibration_points.json")
    recording: ManualFireRecordingConfig = field(default_factory=ManualFireRecordingConfig)

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
        if not isinstance(self.recording, ManualFireRecordingConfig):
            raise ValueError("recording must be a ManualFireRecordingConfig")


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


@dataclass(frozen=True)
class InterpolatedAim:
    """One backend-authoritative pixel target and its local grid solution."""

    pixel_x: int
    pixel_y: int
    pan: float
    tilt: float
    cell: tuple[int, int, int, int]
    triangle: None = None


@dataclass(frozen=True)
class AutomaticEngagementResult:
    """Authoritative facts returned after one completed automatic engagement."""

    aim: InterpolatedAim
    event_id: str
    reservation_id: str
    cooldown_seconds: float
    recording_queued: bool
    shot_attempted: bool = True


CALIBRATION_CELLS = (
    ((1, 2, 4, 5), (1, 2, 5, 4)),
    ((2, 3, 5, 6), (2, 3, 6, 5)),
    ((4, 5, 7, 8), (4, 5, 8, 7)),
    ((5, 6, 8, 9), (5, 6, 9, 8)),
)
CALIBRATION_BOUNDARY = (1, 2, 3, 6, 9, 8, 7, 4)


def validate_complete_calibration_grid(points: list[CalibrationPoint]) -> dict[int, CalibrationPoint]:
    """Require one complete, geometrically usable record for each grid point."""

    indexed = {point.point: point for point in points}
    if len(points) != 9 or set(indexed) != set(range(1, 10)):
        completed = sum(point.complete for point in points)
        raise ControlError(
            f"Targeting requires exactly 9 complete calibration points; found {completed} complete record(s)"
        )
    incomplete = [point.point for point in points if not point.complete]
    if incomplete:
        listed = ", ".join(str(point) for point in incomplete)
        raise ControlError(f"Targeting requires pixel and aim values for every point; incomplete: {listed}")
    pixels = {(point.pixel_x, point.pixel_y) for point in points}
    if len(pixels) != 9:
        raise ControlError("Targeting calibration contains duplicate camera pixels")
    boundary = tuple(indexed[number] for number in CALIBRATION_BOUNDARY)
    if not _is_simple_polygon(boundary):
        raise ControlError("Targeting calibration outer boundary is self-intersecting or degenerate")
    for point in points:
        if not _point_in_polygon_or_boundary(float(point.pixel_x), float(point.pixel_y), boundary):
            raise ControlError(f"Targeting calibration point {point.point} falls outside the outer boundary")
    for cell, corner_order in CALIBRATION_CELLS:
        corners = tuple(indexed[number] for number in corner_order)
        if not _is_simple_convex_quadrilateral(corners):
            joined = "-".join(str(number) for number in cell)
            raise ControlError(f"Targeting calibration cell {joined} is self-intersecting or non-convex")
    for point in points:
        if not any(
            _point_in_polygon_or_boundary(float(point.pixel_x), float(point.pixel_y), tuple(indexed[n] for n in order))
            for _cell, order in CALIBRATION_CELLS
        ):
            raise ControlError(f"Targeting calibration point {point.point} falls outside the calibrated cells")
    return indexed


def is_target_in_calibrated_area(points: list[CalibrationPoint], pixel_x: int, pixel_y: int) -> bool:
    """Return whether a native-frame pixel is inside any validated calibration cell."""

    for name, value in (("pixel_x", pixel_x), ("pixel_y", pixel_y)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    indexed = validate_complete_calibration_grid(points)
    return any(
        _point_in_polygon_or_boundary(pixel_x, pixel_y, tuple(indexed[number] for number in corner_order))
        for _cell, corner_order in CALIBRATION_CELLS
    )


def calibration_geometry_payload(points: list[CalibrationPoint]) -> dict[str, object]:
    """Expose the exact native-pixel geometry used by targeting for diagnostics."""

    indexed = validate_complete_calibration_grid(points)

    def pixel(number: int) -> dict[str, int]:
        point = indexed[number]
        return {"point": number, "pixel_x": int(point.pixel_x), "pixel_y": int(point.pixel_y)}

    return {
        "boundary": [pixel(number) for number in CALIBRATION_BOUNDARY],
        "cells": [
            {
                "points": list(cell),
                "corners": [pixel(number) for number in corner_order],
            }
            for cell, corner_order in CALIBRATION_CELLS
        ],
        "anchors": [pixel(number) for number in range(1, 10)],
    }


def interpolate_calibration_target(
    points: list[CalibrationPoint],
    pixel_x: int,
    pixel_y: int,
    pan_tilt_config: PanTiltConfig,
) -> InterpolatedAim:
    """Inverse-map a native pixel and bilinearly interpolate its four-anchor cell."""

    for name, value in (("pixel_x", pixel_x), ("pixel_y", pixel_y)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    indexed = validate_complete_calibration_grid(points)
    for cell, corner_order in CALIBRATION_CELLS:
        corners = tuple(indexed[number] for number in corner_order)
        if not _point_in_polygon_or_boundary(pixel_x, pixel_y, corners):
            continue
        coordinates = _inverse_bilinear_coordinates(pixel_x, pixel_y, corners)
        if coordinates is None:
            joined = "-".join(str(number) for number in cell)
            raise ControlError(f"Target pixel could not be mapped inside calibration cell {joined}")
        horizontal, vertical = coordinates
        pan = _bilinear_value(*(float(point.pan) for point in corners), horizontal, vertical)
        tilt = _bilinear_value(*(float(point.tilt) for point in corners), horizontal, vertical)
        return InterpolatedAim(
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            pan=clamp_angle(pan, pan_tilt_config.pan_min, pan_tilt_config.pan_max),
            tilt=clamp_angle(tilt, pan_tilt_config.tilt_min, pan_tilt_config.tilt_max),
            cell=cell,
        )
    raise ControlError("Target pixel is outside the calibrated area")


def _inverse_bilinear_coordinates(
    pixel_x: float,
    pixel_y: float,
    corners: tuple[CalibrationPoint, ...],
) -> tuple[float, float] | None:
    horizontal = vertical = 0.5
    for _iteration in range(20):
        mapped_x = _bilinear_value(*(float(point.pixel_x) for point in corners), horizontal, vertical)
        mapped_y = _bilinear_value(*(float(point.pixel_y) for point in corners), horizontal, vertical)
        error_x = pixel_x - mapped_x
        error_y = pixel_y - mapped_y
        if max(abs(error_x), abs(error_y)) < 1e-7:
            break
        top_left, top_right, bottom_right, bottom_left = corners
        dx_du = (1 - vertical) * (float(top_right.pixel_x) - float(top_left.pixel_x)) + vertical * (
            float(bottom_right.pixel_x) - float(bottom_left.pixel_x)
        )
        dx_dv = (1 - horizontal) * (float(bottom_left.pixel_x) - float(top_left.pixel_x)) + horizontal * (
            float(bottom_right.pixel_x) - float(top_right.pixel_x)
        )
        dy_du = (1 - vertical) * (float(top_right.pixel_y) - float(top_left.pixel_y)) + vertical * (
            float(bottom_right.pixel_y) - float(bottom_left.pixel_y)
        )
        dy_dv = (1 - horizontal) * (float(bottom_left.pixel_y) - float(top_left.pixel_y)) + horizontal * (
            float(bottom_right.pixel_y) - float(top_right.pixel_y)
        )
        determinant = dx_du * dy_dv - dx_dv * dy_du
        if abs(determinant) < 1e-12:
            return None
        horizontal += (error_x * dy_dv - error_y * dx_dv) / determinant
        vertical += (dx_du * error_y - dy_du * error_x) / determinant
    mapped_x = _bilinear_value(*(float(point.pixel_x) for point in corners), horizontal, vertical)
    mapped_y = _bilinear_value(*(float(point.pixel_y) for point in corners), horizontal, vertical)
    if max(abs(pixel_x - mapped_x), abs(pixel_y - mapped_y)) > 1e-5:
        return None
    if not (-1e-9 <= horizontal <= 1.0 + 1e-9 and -1e-9 <= vertical <= 1.0 + 1e-9):
        return None
    return min(max(horizontal, 0.0), 1.0), min(max(vertical, 0.0), 1.0)


def _bilinear_value(
    top_left: float,
    top_right: float,
    bottom_right: float,
    bottom_left: float,
    horizontal: float,
    vertical: float,
) -> float:
    return (
        (1 - horizontal) * (1 - vertical) * top_left
        + horizontal * (1 - vertical) * top_right
        + horizontal * vertical * bottom_right
        + (1 - horizontal) * vertical * bottom_left
    )


def _is_simple_convex_quadrilateral(points: tuple[CalibrationPoint, ...]) -> bool:
    if len(points) != 4 or not _is_simple_polygon(points):
        return False
    signs: list[bool] = []
    for index in range(4):
        first = points[index]
        second = points[(index + 1) % 4]
        third = points[(index + 2) % 4]
        cross = (float(second.pixel_x) - float(first.pixel_x)) * (
            float(third.pixel_y) - float(second.pixel_y)
        ) - (float(second.pixel_y) - float(first.pixel_y)) * (
            float(third.pixel_x) - float(second.pixel_x)
        )
        if abs(cross) < 1e-9:
            return False
        signs.append(cross > 0)
    return all(sign == signs[0] for sign in signs)


def _is_simple_polygon(points: tuple[CalibrationPoint, ...]) -> bool:
    if len(points) < 3 or abs(_polygon_area(points)) < 1e-9:
        return False
    edges = [(points[index], points[(index + 1) % len(points)]) for index in range(len(points))]
    for first_index, first_edge in enumerate(edges):
        for second_index in range(first_index + 1, len(edges)):
            if second_index in {first_index, first_index + 1} or (
                first_index == 0 and second_index == len(edges) - 1
            ):
                continue
            if _segments_intersect(*first_edge, *edges[second_index]):
                return False
    return True


def _polygon_area(points: tuple[CalibrationPoint, ...]) -> float:
    return 0.5 * sum(
        float(points[index].pixel_x) * float(points[(index + 1) % len(points)].pixel_y)
        - float(points[(index + 1) % len(points)].pixel_x) * float(points[index].pixel_y)
        for index in range(len(points))
    )


def _segments_intersect(
    first_start: CalibrationPoint,
    first_end: CalibrationPoint,
    second_start: CalibrationPoint,
    second_end: CalibrationPoint,
) -> bool:
    def orientation(a: CalibrationPoint, b: CalibrationPoint, c: CalibrationPoint) -> float:
        return (float(b.pixel_x) - float(a.pixel_x)) * (float(c.pixel_y) - float(a.pixel_y)) - (
            float(b.pixel_y) - float(a.pixel_y)
        ) * (float(c.pixel_x) - float(a.pixel_x))

    def on_segment(start: CalibrationPoint, end: CalibrationPoint, point: CalibrationPoint) -> bool:
        return min(float(start.pixel_x), float(end.pixel_x)) <= float(point.pixel_x) <= max(
            float(start.pixel_x), float(end.pixel_x)
        ) and min(float(start.pixel_y), float(end.pixel_y)) <= float(point.pixel_y) <= max(
            float(start.pixel_y), float(end.pixel_y)
        )

    first = orientation(first_start, first_end, second_start)
    second = orientation(first_start, first_end, second_end)
    third = orientation(second_start, second_end, first_start)
    fourth = orientation(second_start, second_end, first_end)
    if first * second < 0 and third * fourth < 0:
        return True
    return (
        (abs(first) < 1e-9 and on_segment(first_start, first_end, second_start))
        or (abs(second) < 1e-9 and on_segment(first_start, first_end, second_end))
        or (abs(third) < 1e-9 and on_segment(second_start, second_end, first_start))
        or (abs(fourth) < 1e-9 and on_segment(second_start, second_end, first_end))
    )


def _point_in_polygon_or_boundary(
    pixel_x: float,
    pixel_y: float,
    points: tuple[CalibrationPoint, ...],
) -> bool:
    inside = False
    for index, first in enumerate(points):
        second = points[(index + 1) % len(points)]
        first_x, first_y = float(first.pixel_x), float(first.pixel_y)
        second_x, second_y = float(second.pixel_x), float(second.pixel_y)
        cross = (pixel_x - first_x) * (second_y - first_y) - (pixel_y - first_y) * (second_x - first_x)
        if abs(cross) < 1e-7 and min(first_x, second_x) - 1e-7 <= pixel_x <= max(first_x, second_x) + 1e-7 and (
            min(first_y, second_y) - 1e-7 <= pixel_y <= max(first_y, second_y) + 1e-7
        ):
            return True
        crosses = (first_y > pixel_y) != (second_y > pixel_y)
        if crosses:
            intersection_x = (second_x - first_x) * (pixel_y - first_y) / (second_y - first_y) + first_x
            if pixel_x < intersection_x:
                inside = not inside
    return inside


class CalibrationStore:
    """Small JSON store for the nine physical garden calibration points."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._cache: list[CalibrationPoint] | None = None
        self._frame_size: tuple[int, int] | None = None
        self._frame_verified_points: set[int] = set()

    def load(self) -> list[CalibrationPoint]:
        with self._lock:
            return list(self._points_unlocked())

    def frame_size(self) -> tuple[int, int] | None:
        """Return native geometry only after all nine pixel anchors verify it."""

        with self._lock:
            self._points_unlocked()
            if self._frame_size is None or not set(range(1, 10)).issubset(
                self._frame_verified_points
            ):
                return None
            return self._frame_size

    def frame_geometry_status(self) -> dict[str, object]:
        """Expose bounded calibration-geometry migration progress for health/UI."""

        with self._lock:
            self._points_unlocked()
            return {
                "width": None if self._frame_size is None else self._frame_size[0],
                "height": None if self._frame_size is None else self._frame_size[1],
                "verified_points": sorted(self._frame_verified_points),
                "complete": self._frame_size is not None
                and set(range(1, 10)).issubset(self._frame_verified_points),
            }

    def save(
        self,
        point: CalibrationPoint,
        *,
        frame_width: int | None = None,
        frame_height: int | None = None,
    ) -> list[CalibrationPoint]:
        with self._lock:
            points = {item.point: item for item in self._points_unlocked()}
            previous = points.get(point.point)
            next_frame_size = self._frame_size
            next_verified_points = set(self._frame_verified_points)
            if (frame_width is None) != (frame_height is None):
                raise ValueError("frame_width and frame_height must be provided together")
            if frame_width is not None and frame_height is not None:
                if (
                    isinstance(frame_width, bool)
                    or not isinstance(frame_width, int)
                    or frame_width <= 0
                    or isinstance(frame_height, bool)
                    or not isinstance(frame_height, int)
                    or frame_height <= 0
                ):
                    raise ValueError("Calibration frame dimensions must be positive integers")
                requested_size = (frame_width, frame_height)
                if next_frame_size is not None and requested_size != next_frame_size:
                    raise ValueError(
                        "Calibration frame dimensions changed; start a new calibration file "
                        "instead of mixing pixel geometries"
                    )
                next_frame_size = requested_size
                next_verified_points.add(point.point)
            elif previous is not None and (
                previous.pixel_x != point.pixel_x or previous.pixel_y != point.pixel_y
            ):
                next_verified_points.discard(point.point)
            points[point.point] = point
            ordered = [points[index] for index in sorted(points)]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(f".{self.path.name}.tmp")
            payload: dict[str, object] = {
                "schema_version": 2,
                "points": [asdict(item) for item in ordered],
            }
            if next_frame_size is not None:
                payload.update(
                    frame_width=next_frame_size[0],
                    frame_height=next_frame_size[1],
                    frame_verified_points=sorted(next_verified_points),
                )
            try:
                temporary.write_text(
                    json.dumps(payload, indent=2) + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary, self.path)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
            self._frame_size = next_frame_size
            self._frame_verified_points = next_verified_points
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
        if not isinstance(raw, dict):
            raise ValueError("Calibration file must contain a JSON object")
        records = raw.get("points")
        if not isinstance(records, list):
            raise ValueError("Calibration file must contain a points list")
        frame_width = raw.get("frame_width")
        frame_height = raw.get("frame_height")
        raw_verified_points = raw.get("frame_verified_points", [])
        if (
            not isinstance(raw_verified_points, list)
            or any(
                isinstance(point, bool) or not isinstance(point, int) or not 1 <= point <= 9
                for point in raw_verified_points
            )
            or len(set(raw_verified_points)) != len(raw_verified_points)
        ):
            raise ValueError("Calibration file frame_verified_points must be unique point numbers 1-9")
        if frame_width is None and frame_height is None:
            if raw_verified_points:
                raise ValueError("Calibration file cannot verify frame points without dimensions")
            loaded_frame_size = None
            loaded_verified_points: set[int] = set()
        elif (
            isinstance(frame_width, bool)
            or not isinstance(frame_width, int)
            or frame_width <= 0
            or isinstance(frame_height, bool)
            or not isinstance(frame_height, int)
            or frame_height <= 0
        ):
            raise ValueError("Calibration file frame dimensions must be positive integers")
        else:
            loaded_frame_size = (frame_width, frame_height)
            loaded_verified_points = set(raw_verified_points)
        points = [CalibrationPoint(**record) for record in records]
        if len({point.point for point in points}) != len(points):
            raise ValueError("Calibration file contains duplicate point numbers")
        points_by_number = {point.point: point for point in points}
        if not loaded_verified_points.issubset(points_by_number):
            raise ValueError("Calibration file verifies frame geometry for a missing point")
        if loaded_frame_size is not None:
            for point_number in loaded_verified_points:
                point = points_by_number[point_number]
                pixel_x, pixel_y = point.pixel_x, point.pixel_y
                if (
                    pixel_x is None
                    or pixel_y is None
                    or pixel_x >= loaded_frame_size[0]
                    or pixel_y >= loaded_frame_size[1]
                ):
                    raise ValueError(
                        f"Calibration Point {point_number} is verified outside its native frame"
                    )
        self._frame_size = loaded_frame_size
        self._frame_verified_points = loaded_verified_points
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
        fire_recorder: ManualFireRecordingSink | None = None,
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
        self._fire_recorder = fire_recorder
        self._lock = threading.Lock()
        self._transient_state: ControlState | None = None
        self._pan = float(pan_tilt_config.pan_center)
        self._tilt = float(pan_tilt_config.tilt_center)
        self._position_commanded = False
        self._active_calibration_point = 1
        self._calibration_edit_active = False
        self._last_fire_completed_at: float | None = None
        self._last_fire_cooldown_seconds = float(control_config.fire_cooldown_seconds)
        self._target_pixel: tuple[int, int] | None = None
        self._target_aim: InterpolatedAim | None = None
        self._target_in_range: bool | None = None
        self._targeting_status: str | None = None
        self._targeting_error: str | None = None
        self._servo_error = servo_error
        self._valve_error = valve_error

    @property
    def servo_available(self) -> bool:
        return self._pan_tilt is not None

    @property
    def valve_available(self) -> bool:
        return not isinstance(self._valve, DisabledValveController)

    def cooldown_remaining_seconds(self) -> float:
        """Return the shared backend cooldown without loading status metadata."""

        return self._cooldown_remaining(self._clock())

    def status(self) -> dict[str, object]:
        now = self._clock()
        remaining = self._cooldown_remaining(now)
        state = self._transient_state or (ControlState.COOLDOWN if remaining > 0 else ControlState.IDLE)
        try:
            calibration_points = self._calibration.load()
            calibration_frame = self._calibration.frame_geometry_status()
            points = [self._calibration_point_payload(point) for point in calibration_points]
            calibration_error = None
        except (OSError, ValueError, TypeError) as exc:
            calibration_points = []
            calibration_frame = {
                "width": None,
                "height": None,
                "verified_points": [],
                "complete": False,
            }
            points = []
            calibration_error = str(exc)
        if calibration_error is not None:
            targeting_readiness_error = f"Calibration data is invalid: {calibration_error}"
        else:
            try:
                validate_complete_calibration_grid(calibration_points)
                targeting_readiness_error = None
            except ControlError as exc:
                targeting_readiness_error = str(exc)
        targeting_enabled = self.servo_available and targeting_readiness_error is None
        recorder_status: dict[str, object] = {"enabled": False, "active": False}
        if self._fire_recorder is not None:
            status_reader = getattr(self._fire_recorder, "status", None)
            if callable(status_reader):
                recorder_status = status_reader()
        return {
            "state": state.value,
            "pan": self._display_angle(self._pan),
            "tilt": self._display_angle(self._tilt),
            "position_commanded": self._position_commanded,
            "pan_min": self.pan_tilt_config.pan_min,
            "pan_max": self.pan_tilt_config.pan_max,
            "tilt_min": self.pan_tilt_config.tilt_min,
            "tilt_max": self.pan_tilt_config.tilt_max,
            "park_pan": self._display_angle(self.pan_tilt_config.park_pan),
            "park_tilt": self._display_angle(self.pan_tilt_config.park_tilt),
            "default_step": self.config.default_step_degrees,
            "allowed_steps": list(self.config.allowed_step_degrees),
            "servo_available": self.servo_available,
            "valve_available": self.valve_available,
            "valve_state": self._valve.state.value,
            "cooldown_remaining_seconds": round(remaining, 3),
            "fire_pulse_seconds": self.config.fire_pulse_seconds,
            "active_calibration_point": self._active_calibration_point,
            "calibration_edit_active": self._calibration_edit_active,
            "calibration_points": points,
            "calibration_frame": calibration_frame,
            "completed_calibration_count": sum(bool(point["complete"]) for point in points),
            "targeting": self._targeting_payload(
                enabled=targeting_enabled,
                readiness_error=targeting_readiness_error,
                calibration_points=calibration_points,
            ),
            "servo_error": self._servo_error,
            "valve_error": self._valve_error,
            "calibration_error": calibration_error,
            "recording": recorder_status,
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
            position = self._move_locked(target)
            self._targeting_status = "MANUAL POSITION"
            self._targeting_error = None
            return position

    def aim_at_pixel(self, pixel_x: int, pixel_y: int) -> InterpolatedAim:
        """Interpolate and move to a camera pixel without activating the valve."""

        for name, value in (("pixel_x", pixel_x), ("pixel_y", pixel_y)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self._pan_tilt is None:
            raise ControlUnavailableError(self._servo_error or "Servo control is disabled")
        with self._lock:
            self._calibration_edit_active = False
            self._target_pixel = (pixel_x, pixel_y)
            self._target_aim = None
            self._target_in_range = None
            self._targeting_error = None
            self._targeting_status = "VALIDATING TARGET"
            try:
                aim = interpolate_calibration_target(
                    self._calibration.load(),
                    pixel_x,
                    pixel_y,
                    self.pan_tilt_config,
                )
            except (ControlError, OSError, ValueError, TypeError) as exc:
                self._target_in_range = False if "outside the calibrated area" in str(exc).lower() else None
                self._targeting_error = str(exc)
                self._targeting_status = (
                    "OUTSIDE CALIBRATED AREA"
                    if "outside the calibrated area" in str(exc).lower()
                    else "TARGET REJECTED"
                )
                LOGGER.warning("Target pixel rejected: x=%d y=%d error=%s", pixel_x, pixel_y, exc)
                raise
            self._target_aim = aim
            self._target_in_range = True
            LOGGER.info(
                "Target pixel: %d,%d; inverse-bilinear pan=%.2f tilt=%.2f; calibration cell=%s",
                aim.pixel_x,
                aim.pixel_y,
                aim.pan,
                aim.tilt,
                "-".join(str(point) for point in aim.cell),
            )
            try:
                self._move_locked(PanTiltPosition(aim.pan, aim.tilt), action="aim")
            except Exception as exc:
                self._targeting_error = str(exc)
                self._targeting_status = "AIM ERROR"
                raise
            self._targeting_status = "AIM READY"
            LOGGER.info("Target movement and settling complete; AIM READY")
            return aim

    def fire(self) -> bool:
        """Pulse manually, parking unless backend calibration edit mode is active.

        Return ``True`` only when calibration mode deliberately held the current
        commanded position after the valve was confirmed closed.
        """

        if not self.valve_available:
            raise ControlUnavailableError(self._valve_error or "Valve control is disabled until a GPIO pin is configured")
        if not self._lock.acquire(blocking=False):
            busy = self._transient_state.value if self._transient_state is not None else "another control action"
            raise ControlError(f"FIRE rejected while {busy} is active; wait for movement and settling to finish")
        try:
            hold_position = self._calibration_edit_active
            self._fire_and_park_locked(park_after_fire=not hold_position)
            return hold_position
        finally:
            self._lock.release()

    def park(self) -> PanTiltPosition:
        """Move to the configured anti-drip rest position without firing."""

        if self._pan_tilt is None:
            raise ControlUnavailableError(self._servo_error or "Servo control is disabled")
        if not self._lock.acquire(blocking=False):
            busy = self._transient_state.value if self._transient_state is not None else "another control action"
            raise ControlError(f"PARK rejected while {busy} is active; wait for the current action to finish")
        try:
            LOGGER.info(
                "Manual PARK requested: pan=%.2f tilt=%.2f",
                self.pan_tilt_config.park_pan,
                self.pan_tilt_config.park_tilt,
            )
            return self._park_locked()
        finally:
            self._lock.release()

    def move_and_fire(self, pan: float, tilt: float) -> PanTiltPosition:
        """Shared future pipeline: target, move, settle, fire, then park during cooldown."""

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
            self._fire_and_park_locked()
            return position

    def automatic_engage(
        self,
        pixel_x: int,
        pixel_y: int,
        *,
        frame_width: int,
        frame_height: int,
        cooldown_seconds: float,
        final_safety_check: Callable[[], bool],
        reserve_actuation: Callable[[], str],
        evidence: dict[str, object],
    ) -> AutomaticEngagementResult:
        """Reserve the shared coordinator and perform one bounded automatic shot.

        Automatic callers are never queued behind another physical action. The
        target is interpolated twice from the same native pixel. The supplied
        final guard runs exactly once, then a durable reservation must succeed
        immediately before the unchanged valve checks/pulse.
        """

        for name, value in (("pixel_x", pixel_x), ("pixel_y", pixel_y)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise AutomaticEngagementError("invalid_request", f"{name} must be a non-negative integer")
        for name, value in (("frame_width", frame_width), ("frame_height", frame_height)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise AutomaticEngagementError(
                    "invalid_request",
                    f"{name} must be a positive integer",
                )
        if pixel_x >= frame_width or pixel_y >= frame_height:
            raise AutomaticEngagementError(
                "invalid_request",
                "Automatic target pixel must be inside the current native camera frame",
            )
        if (
            isinstance(cooldown_seconds, bool)
            or not isinstance(cooldown_seconds, (int, float))
            or not math.isfinite(float(cooldown_seconds))
            or cooldown_seconds <= 0
        ):
            raise AutomaticEngagementError(
                "invalid_request",
                "cooldown_seconds must be a finite number greater than zero",
            )
        if not callable(final_safety_check):
            raise AutomaticEngagementError("invalid_request", "final_safety_check must be callable")
        if not callable(reserve_actuation):
            raise AutomaticEngagementError("invalid_request", "reserve_actuation must be callable")
        if not isinstance(evidence, dict):
            raise AutomaticEngagementError("invalid_request", "evidence must be a dictionary")
        if self._pan_tilt is None:
            raise AutomaticEngagementError(
                "hardware_not_ready",
                self._servo_error or "Servo control is disabled",
            )
        if not self.valve_available:
            raise AutomaticEngagementError(
                "valve_disabled",
                self._valve_error or "Valve control is disabled",
            )
        if self._fire_recorder is None:
            raise AutomaticEngagementError(
                "recording_unavailable",
                "Automatic firing requires the shared fire recorder",
            )
        if not self._lock.acquire(blocking=False):
            raise AutomaticEngagementError(
                "coordinator_busy",
                "Automatic engagement rejected while another physical-control action is active",
            )

        movement_started = False
        park_attempted = False
        shot_attempted = False
        try:
            if self._valve.state is not ValveState.CLOSED:
                raise AutomaticEngagementError(
                    "safety_state_invalid",
                    "Automatic engagement rejected because the valve is not confirmed closed",
                )
            remaining = self._cooldown_remaining(self._clock())
            if remaining > 0:
                raise AutomaticEngagementError(
                    "cooldown_active",
                    f"Automatic engagement rejected during shared cooldown ({remaining:.3f}s remaining)",
                )
            self._validate_auto_calibration_frame_locked(frame_width, frame_height)

            self._target_pixel = (pixel_x, pixel_y)
            self._target_aim = None
            self._target_in_range = None
            self._targeting_status = "VALIDATING AUTO TARGET"
            self._targeting_error = None
            try:
                aim = interpolate_calibration_target(
                    self._calibration.load(),
                    pixel_x,
                    pixel_y,
                    self.pan_tilt_config,
                )
            except (ControlError, OSError, ValueError, TypeError) as exc:
                reason = "outside_safe_bounds" if "outside the calibrated area" in str(exc).lower() else "interpolation_failed"
                self._target_in_range = False if reason == "outside_safe_bounds" else None
                self._targeting_status = "AUTO TARGET REJECTED"
                self._targeting_error = str(exc)
                raise AutomaticEngagementError(reason, str(exc)) from exc
            self._target_aim = aim
            self._target_in_range = True

            movement_started = True
            self._targeting_status = "AUTO MOVING"
            try:
                self._move_locked(PanTiltPosition(aim.pan, aim.tilt), action="automatic")
            except Exception as exc:
                self._targeting_status = "AUTO MOVE ERROR"
                self._targeting_error = str(exc)
                raise AutomaticEngagementError("movement_failed", str(exc)) from exc

            self._targeting_status = "AUTO FINAL SAFETY CHECK"
            self._validate_auto_calibration_frame_locked(frame_width, frame_height)
            try:
                final_aim = interpolate_calibration_target(
                    self._calibration.load(),
                    pixel_x,
                    pixel_y,
                    self.pan_tilt_config,
                )
            except (ControlError, OSError, ValueError, TypeError) as exc:
                reason = "outside_safe_bounds" if "outside the calibrated area" in str(exc).lower() else "interpolation_failed"
                raise AutomaticEngagementError(reason, str(exc)) from exc
            if final_aim != aim:
                raise AutomaticEngagementError(
                    "safety_state_invalid",
                    "Automatic target interpolation changed during movement",
                )
            try:
                final_safe = final_safety_check()
            except Exception as exc:
                raise AutomaticEngagementError(
                    "safety_state_invalid",
                    f"Final automatic safety check failed: {type(exc).__name__}: {exc}",
                ) from exc
            if final_safe is not True:
                raise AutomaticEngagementError(
                    "safety_state_invalid",
                    "Final automatic safety check rejected the engagement",
                )
            if self._valve.state is not ValveState.CLOSED:
                raise AutomaticEngagementError(
                    "safety_state_invalid",
                    "Valve is not confirmed closed immediately before automatic fire",
                )
            remaining = self._cooldown_remaining(self._clock())
            if remaining > 0:
                raise AutomaticEngagementError(
                    "cooldown_active",
                    f"Shared cooldown became active ({remaining:.3f}s remaining)",
                )

            try:
                reservation_id = reserve_actuation()
            except Exception as exc:
                raise AutomaticEngagementError(
                    "safety_state_invalid",
                    f"Durable automatic actuation reservation failed: {type(exc).__name__}: {exc}",
                    shot_attempted=False,
                ) from exc
            if not isinstance(reservation_id, str) or not reservation_id:
                raise AutomaticEngagementError(
                    "safety_state_invalid",
                    "Durable automatic actuation reservation returned an invalid identifier",
                    shot_attempted=False,
                )

            event_evidence = dict(evidence)
            event_evidence.update(
                target_pixel_x=pixel_x,
                target_pixel_y=pixel_y,
                target_frame_width=frame_width,
                target_frame_height=frame_height,
                calculated_pan=final_aim.pan,
                calculated_tilt=final_aim.tilt,
                cooldown_seconds=float(cooldown_seconds),
                safe_bound_result="inside_calibrated_area",
                interpolation_result="success",
                calibration_cell=list(final_aim.cell),
                actuation_reservation_id=reservation_id,
            )
            shot_attempted = True
            try:
                event_id, recording_queued = self._fire_locked(
                    cooldown_seconds=float(cooldown_seconds),
                    event_type="auto_fire",
                    evidence=event_evidence,
                    crop_center=(pixel_x, pixel_y, "auto_fire_target_pixel"),
                )
            except FireCooldownError as exc:
                shot_attempted = False
                raise AutomaticEngagementError("cooldown_active", str(exc), shot_attempted=False) from exc
            except Exception as exc:
                raise AutomaticEngagementError(
                    "valve_failure",
                    f"Automatic valve pulse failed: {type(exc).__name__}: {exc}",
                    shot_attempted=True,
                ) from exc

            park_attempted = True
            try:
                self._park_locked()
            except Exception as exc:
                raise AutomaticEngagementError(
                    "park_failed",
                    str(exc),
                    shot_attempted=True,
                ) from exc
            return AutomaticEngagementResult(
                aim=final_aim,
                event_id=event_id,
                reservation_id=reservation_id,
                cooldown_seconds=float(cooldown_seconds),
                recording_queued=recording_queued,
            )
        except AutomaticEngagementError as exc:
            if movement_started and not park_attempted:
                self._attempt_automatic_park_locked()
            if exc.shot_attempted or not shot_attempted:
                raise
            raise AutomaticEngagementError(exc.reason, str(exc), shot_attempted=True) from exc
        except Exception as exc:
            if movement_started and not park_attempted:
                self._attempt_automatic_park_locked()
            raise AutomaticEngagementError(
                "safety_state_invalid",
                f"Automatic engagement failed: {type(exc).__name__}: {exc}",
                shot_attempted=shot_attempted,
            ) from exc
        finally:
            self._lock.release()

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

    def enter_calibration_edit_mode(self) -> None:
        """Enable the backend policy used by authenticated calibration actions."""

        with self._lock:
            self._calibration_edit_active = True
            self._targeting_status = "EDIT CALIBRATION"
            self._targeting_error = None

    def exit_calibration_edit_mode(self) -> None:
        """Restore normal manual-fire PARK behavior."""

        with self._lock:
            self._calibration_edit_active = False
            self._targeting_status = None
            self._targeting_error = None

    def select_calibration_point_for_edit(self, point: int) -> PanTiltPosition | None:
        """Select a point and command its saved aim without interpolation or firing."""

        if isinstance(point, bool) or not isinstance(point, int) or not 1 <= point <= 9:
            raise ValueError("Calibration point must be an integer from 1 to 9")
        with self._lock:
            self._calibration_edit_active = True
            self._active_calibration_point = point
            self._target_pixel = None
            self._target_aim = None
            self._target_in_range = None
            existing = next(
                (record for record in self._calibration.load() if record.point == point),
                None,
            )
            if existing is None or not existing.aim_saved:
                self._targeting_status = f"POINT {point} SELECTED"
                self._targeting_error = f"Point {point} has no saved Pan/Tilt aim; no movement was commanded"
                return None
            if self._pan_tilt is None:
                raise ControlUnavailableError(self._servo_error or "Servo control is disabled")
            pan, tilt = float(existing.pan), float(existing.tilt)
            if not self.pan_tilt_config.pan_min <= pan <= self.pan_tilt_config.pan_max:
                raise ControlError(
                    f"Point {point} saved pan {pan:g} is outside configured servo limits "
                    f"{self.pan_tilt_config.pan_min:g}-{self.pan_tilt_config.pan_max:g}"
                )
            if not self.pan_tilt_config.tilt_min <= tilt <= self.pan_tilt_config.tilt_max:
                raise ControlError(
                    f"Point {point} saved tilt {tilt:g} is outside configured servo limits "
                    f"{self.pan_tilt_config.tilt_min:g}-{self.pan_tilt_config.tilt_max:g}"
                )
            self._targeting_status = f"MOVING TO POINT {point} SAVED AIM"
            self._targeting_error = None
            try:
                position = self._move_locked(
                    PanTiltPosition(pan, tilt),
                    action="calibration",
                )
            except Exception as exc:
                self._targeting_status = f"POINT {point} MOVE ERROR"
                self._targeting_error = str(exc)
                raise
            self._targeting_status = f"POINT {point} SAVED AIM READY"
            return position

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
            self._calibration.save(
                record,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            self._targeting_error = None
            self._targeting_status = None
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
            self._targeting_error = None
            self._targeting_status = None
            return record

    def cleanup(self) -> None:
        # Serialize teardown with the same physical-control lock so valve,
        # servo, and recorder cleanup can never race an in-flight engagement.
        with self._lock:
            try:
                self._valve.cleanup()
            finally:
                try:
                    if self._pan_tilt is not None:
                        self._pan_tilt.cleanup()
                finally:
                    if self._fire_recorder is not None:
                        self._fire_recorder.close()

    def _move_locked(self, target: PanTiltPosition, *, action: str = "manual") -> PanTiltPosition:
        if self._valve.state is ValveState.OPEN:
            raise ControlError("Servo movement rejected while the valve is open")
        try:
            self._transient_state = ControlState.PARKING if action == "park" else ControlState.MOVING
            if action == "aim":
                self._targeting_status = "MOVING"
            elif action == "park":
                self._targeting_status = "PARKING"
            LOGGER.info(
                "%s movement started: pan=%.2f tilt=%.2f",
                action.capitalize(),
                target.pan,
                target.tilt,
            )
            result = self._pan_tilt.move_to_smooth(  # type: ignore[union-attr]
                target.pan,
                target.tilt,
                settling_delay_seconds=0,
            )
            self._pan, self._tilt = result.pan, result.tilt
            self._position_commanded = True
            if action != "park":
                self._transient_state = ControlState.SETTLING
            if action == "aim":
                self._targeting_status = "SETTLING"
            if self.pan_tilt_config.settling_delay_seconds:
                self._sleep(self.pan_tilt_config.settling_delay_seconds)
            LOGGER.info("%s movement and settling complete", action.capitalize())
            return result
        finally:
            self._transient_state = None

    def _fire_and_park_locked(self, *, park_after_fire: bool = True) -> None:
        self._fire_locked()
        if not park_after_fire:
            self._targeting_status = f"POINT {self._active_calibration_point} CALIBRATION HOLD"
            self._targeting_error = None
            LOGGER.info(
                "Calibration-mode manual fire complete; holding pan=%.2f tilt=%.2f",
                self._pan,
                self._tilt,
            )
            return
        if self._pan_tilt is None:
            self._targeting_status = "PARK UNAVAILABLE"
            LOGGER.warning("Valve pulse completed but PARK is unavailable because servo control is disabled")
            return
        self._park_locked()

    def _attempt_automatic_park_locked(self) -> None:
        """Best-effort PARK after a moved automatic engagement is cancelled."""

        if self._valve.state is not ValveState.CLOSED:
            try:
                self._valve.close()
            except Exception:
                LOGGER.exception("Valve close also failed while cancelling automatic engagement")
        if self._pan_tilt is None or self._valve.state is not ValveState.CLOSED:
            LOGGER.error(
                "Automatic PARK skipped after failure; servo_available=%s valve_state=%s",
                self._pan_tilt is not None,
                self._valve.state.value,
            )
            return
        try:
            self._park_locked()
        except Exception:
            LOGGER.exception("Automatic PARK also failed while preserving the original engagement error")

    def _validate_auto_calibration_frame_locked(
        self,
        frame_width: int,
        frame_height: int,
    ) -> None:
        try:
            calibrated_size = self._calibration.frame_size()
        except (OSError, ValueError, TypeError) as exc:
            raise AutomaticEngagementError(
                "interpolation_failed",
                f"Could not validate calibration frame geometry: {exc}",
            ) from exc
        if calibrated_size is None:
            raise AutomaticEngagementError(
                "calibration_frame_unknown",
                "Automatic engagement requires calibration saved with native frame dimensions",
            )
        if calibrated_size != (frame_width, frame_height):
            raise AutomaticEngagementError(
                "calibration_frame_mismatch",
                "Current camera frame geometry does not match the saved calibration "
                f"({frame_width}x{frame_height} != {calibrated_size[0]}x{calibrated_size[1]})",
            )

    def _park_locked(self) -> PanTiltPosition:
        target = PanTiltPosition(
            clamp_angle(self.pan_tilt_config.park_pan, self.pan_tilt_config.pan_min, self.pan_tilt_config.pan_max),
            clamp_angle(self.pan_tilt_config.park_tilt, self.pan_tilt_config.tilt_min, self.pan_tilt_config.tilt_max),
        )
        LOGGER.info("Park movement started after valve OFF: pan=%.2f tilt=%.2f", target.pan, target.tilt)
        try:
            self._move_locked(target, action="park")
        except Exception as exc:
            self._targeting_status = "PARK ERROR"
            self._targeting_error = f"PARK failed: {exc}"
            LOGGER.exception("PARK failed; valve state=%s", self._valve.state.value)
            raise ControlUnavailableError(self._targeting_error) from exc
        self._targeting_status = "PARKED"
        self._targeting_error = None
        LOGGER.info("Park complete: pan=%.2f tilt=%.2f", self._pan, self._tilt)
        return PanTiltPosition(self._pan, self._tilt)

    def _fire_locked(
        self,
        *,
        cooldown_seconds: float | None = None,
        event_type: str = "manual_fire",
        evidence: dict[str, object] | None = None,
        crop_center: tuple[int | None, int | None, str] | None = None,
    ) -> tuple[str, bool]:
        applied_cooldown = float(
            self.config.fire_cooldown_seconds if cooldown_seconds is None else cooldown_seconds
        )
        if not math.isfinite(applied_cooldown) or applied_cooldown <= 0:
            raise ValueError("cooldown_seconds must be a finite number greater than zero")
        if event_type not in {"manual_fire", "auto_fire"}:
            raise ValueError("event_type must be manual_fire or auto_fire")
        remaining = self._cooldown_remaining(self._clock())
        if remaining > 0:
            raise FireCooldownError(remaining)
        fire_started = self._clock()
        fire_timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
        crop_center_x, crop_center_y, crop_center_source = (
            self._recording_crop_center_locked() if crop_center is None else crop_center
        )
        shot_pan, shot_tilt = self._pan, self._tilt
        self._transient_state = ControlState.FIRING
        self._targeting_status = "FIRING"
        LOGGER.info("Valve pulse started: duration=%.2fs", self.config.fire_pulse_seconds)
        fire_completed: float | None = None
        closure_error: Exception | None = None
        try:
            pulse_valve(self._valve, self.config.fire_pulse_seconds, sleep=self._sleep)
        finally:
            if self._valve.state is not ValveState.CLOSED:
                try:
                    self._valve.close()
                except Exception as exc:
                    closure_error = exc
                    LOGGER.exception("Valve close retry failed after pulse")
            if self._valve.state is ValveState.CLOSED:
                fire_completed = self._clock()
                self._last_fire_completed_at = fire_completed
                self._last_fire_cooldown_seconds = applied_cooldown
            self._transient_state = None
            LOGGER.info(
                "Valve pulse ended; valve state=%s; cooldown_started=%s",
                self._valve.state.value,
                fire_completed is not None,
            )
        if closure_error is not None or fire_completed is None:
            raise ControlError("Valve closure could not be confirmed after pulse") from closure_error
        LOGGER.info("%s pulse completed", event_type.replace("_", " ").title())
        LOGGER.info(
            "%s accepted: pan=%.2f tilt=%.2f pulse=%.2fs cooldown=%.2fs",
            event_type.replace("_", " ").upper(),
            shot_pan,
            shot_tilt,
            self.config.fire_pulse_seconds,
            applied_cooldown,
        )
        event_id = new_auto_fire_event_id() if event_type == "auto_fire" else new_manual_fire_event_id()
        recording_queued = False
        if self._fire_recorder is not None:
            event = ManualFireEvent(
                event_id=event_id,
                timestamp=fire_timestamp,
                fire_started_monotonic=fire_started,
                fire_completed_monotonic=fire_completed,
                pan=shot_pan,
                tilt=shot_tilt,
                fire_pulse_seconds=self.config.fire_pulse_seconds,
                crop_center_x=crop_center_x,
                crop_center_y=crop_center_y,
                crop_center_source=crop_center_source,
                event_type=event_type,
                evidence={} if evidence is None else dict(evidence),
            )
            try:
                self._fire_recorder.record(event)
                recording_queued = True
            except Exception as exc:
                LOGGER.error("%s recording failed to start: %s", event_type, exc, exc_info=True)
        return event_id, recording_queued

    def _recording_crop_center_locked(self) -> tuple[int | None, int | None, str]:
        aim = self._target_aim
        if (
            self._target_pixel is not None
            and aim is not None
            and self._targeting_status == "AIM READY"
            and math.isclose(self._pan, aim.pan, abs_tol=1e-6)
            and math.isclose(self._tilt, aim.tilt, abs_tol=1e-6)
        ):
            return self._target_pixel[0], self._target_pixel[1], "calibrated_target_pixel"
        recording = self.config.recording
        if recording.crop_center_x is not None and recording.crop_center_y is not None:
            return recording.crop_center_x, recording.crop_center_y, "configured_fixed_fallback"
        return None, None, "frame_center_fallback"

    def _cooldown_remaining(self, now: float) -> float:
        if self._last_fire_completed_at is None:
            return 0.0
        return max(0.0, self._last_fire_cooldown_seconds - (now - self._last_fire_completed_at))

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

    def _targeting_payload(
        self,
        *,
        enabled: bool,
        readiness_error: str | None,
        calibration_points: list[CalibrationPoint],
    ) -> dict[str, object]:
        aim = self._target_aim
        pixel_x, pixel_y = self._target_pixel if self._target_pixel is not None else (None, None)
        geometry = None
        if enabled:
            try:
                geometry = calibration_geometry_payload(calibration_points)
            except ControlError:
                geometry = None
        return {
            "enabled": enabled,
            "status": self._targeting_status or ("READY FOR TARGET" if enabled else "TARGETING UNAVAILABLE"),
            "error": self._targeting_error or readiness_error,
            "pixel_x": pixel_x,
            "pixel_y": pixel_y,
            "in_range": self._target_in_range,
            "pan": None if aim is None else self._display_angle(aim.pan),
            "tilt": None if aim is None else self._display_angle(aim.tilt),
            "cell": None if aim is None else list(aim.cell),
            "triangle": None,
            "method": "inverse bilinear" if aim is not None else None,
            "geometry": geometry,
        }


def build_manual_control_service(
    pan_tilt_config: PanTiltConfig,
    control_config: ManualControlConfig,
    valve_config: ValveConfig,
    *,
    camera_service: object | None = None,
    output_directory: Path | None = None,
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
    fire_recorder: ManualFireRecordingSink | None = None
    if control_config.recording.enabled:
        if camera_service is not None and output_directory is not None:
            fire_recorder = ManualFireRecorder(camera_service, output_directory, control_config.recording)  # type: ignore[arg-type]
        else:
            LOGGER.warning("Manual fire recording is enabled but the shared camera/output directory was not provided")
    return ManualControlService(
        pan_tilt_config,
        control_config,
        pan_tilt=pan_tilt,
        valve=valve,
        fire_recorder=fire_recorder,
        servo_error=servo_error,
        valve_error=valve_error,
    )
