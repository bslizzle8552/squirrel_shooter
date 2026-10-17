"""Small, disabled-by-default physical-control configuration for the collector.

Importing this module never initializes a servo driver or GPIO output. Hardware
defaults retain the recorded field configuration; timing ceilings are explicit
software admission bounds, not claims about mechanical accuracy.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from .pan_tilt import PanTiltConfig


HARD_MAXIMUM_PULSE_SECONDS = 0.40
MAXIMUM_SETTLING_SECONDS = 1.0
MAXIMUM_COOLDOWN_SECONDS = 3600.0


def _boolean(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")


def _number(name: str, value: object, *, minimum: float, maximum: float,
            allow_minimum: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be a finite number")
    if value > maximum or (value < minimum if allow_minimum else value <= minimum):
        comparison = "at least" if allow_minimum else "greater than"
        raise ValueError(f"{name} must be {comparison} {minimum} and at most {maximum}")


@dataclass(frozen=True)
class PhysicalControlConfig:
    enabled: bool = False
    valve_enabled: bool = False
    calibration_file: Path = Path("collector-calibration.json")
    pan_tilt: PanTiltConfig = field(default_factory=PanTiltConfig)
    valve_gpio_pin: int = 24
    valve_active_high: bool = True
    pulse_seconds: float = 0.40
    maximum_pulse_seconds: float = HARD_MAXIMUM_PULSE_SECONDS
    cooldown_seconds: float = 10.0

    def __post_init__(self) -> None:
        _boolean("physical_control.enabled", self.enabled)
        _boolean("physical_control.valve_enabled", self.valve_enabled)
        _boolean("physical_control.valve_active_high", self.valve_active_high)
        if not self.valve_active_high:
            raise ValueError("physical_control.valve_active_high must be true: GPIO LOW is closed")
        if (isinstance(self.valve_gpio_pin, bool) or not isinstance(self.valve_gpio_pin, int)
                or not 0 <= self.valve_gpio_pin <= 27):
            raise ValueError("physical_control.valve_gpio_pin must be a BCM GPIO integer from 0 to 27")
        if not isinstance(self.calibration_file, Path) or self.calibration_file == Path("."):
            raise ValueError("physical_control.calibration_file must name a file path")
        if not isinstance(self.pan_tilt, PanTiltConfig):
            raise ValueError("physical_control.pan_tilt must be a PanTiltConfig")
        if (isinstance(self.pan_tilt.i2c_address, bool)
                or not isinstance(self.pan_tilt.i2c_address, int)):
            raise ValueError("physical_control.pan_tilt.i2c_address must be an integer")
        for name in ("pan_min", "pan_center", "pan_max", "tilt_min", "tilt_center",
                     "tilt_max", "park_pan", "park_tilt"):
            _number(f"physical_control.pan_tilt.{name}", getattr(self.pan_tilt, name),
                    minimum=0, maximum=180, allow_minimum=True)
        _number("physical_control.pan_tilt.settling_delay_seconds",
                self.pan_tilt.settling_delay_seconds, minimum=0, maximum=MAXIMUM_SETTLING_SECONDS)
        _number("physical_control.maximum_pulse_seconds", self.maximum_pulse_seconds,
                minimum=0, maximum=HARD_MAXIMUM_PULSE_SECONDS)
        _number("physical_control.pulse_seconds", self.pulse_seconds,
                minimum=0, maximum=self.maximum_pulse_seconds)
        _number("physical_control.cooldown_seconds", self.cooldown_seconds,
                minimum=0, maximum=MAXIMUM_COOLDOWN_SECONDS)


@dataclass(frozen=True)
class AutomaticFireConfig:
    enabled: bool = False
    dry_run: bool = True
    minimum_confidence: float | None = None

    def __post_init__(self) -> None:
        _boolean("automatic_fire.enabled", self.enabled)
        _boolean("automatic_fire.dry_run", self.dry_run)
        if self.minimum_confidence is not None:
            _number("automatic_fire.minimum_confidence", self.minimum_confidence,
                    minimum=0, maximum=1, allow_minimum=True)
        elif self.enabled:
            raise ValueError("automatic_fire.minimum_confidence must be explicitly set before enabling")


def _section(raw: object, name: str, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or any(not isinstance(key, str) for key in raw):
        raise ValueError(f"{name} must be a mapping with string keys")
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} fields: {', '.join(sorted(unknown))}")
    return dict(raw)


def parse_physical_control(raw: object = None) -> PhysicalControlConfig:
    """Parse one section; its caller resolves relative paths against config.parent."""
    if raw is None:
        return PhysicalControlConfig()
    values = _section(raw, "physical_control", {item.name for item in fields(PhysicalControlConfig)})
    if "calibration_file" in values:
        path = values["calibration_file"]
        if not isinstance(path, (str, Path)) or not str(path).strip():
            raise ValueError("physical_control.calibration_file must name a file path")
        values["calibration_file"] = Path(path)
    if "pan_tilt" in values:
        angles = _section(values["pan_tilt"], "physical_control.pan_tilt",
                          {item.name for item in fields(PanTiltConfig)})
        try:
            values["pan_tilt"] = PanTiltConfig(**angles)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"physical_control.pan_tilt: {exc}") from exc
    return PhysicalControlConfig(**values)


def parse_automatic_fire(raw: object = None) -> AutomaticFireConfig:
    """Parse the collector policy, independently of the legacy auto_fire section."""
    if raw is None:
        return AutomaticFireConfig()
    values = _section(raw, "automatic_fire", {item.name for item in fields(AutomaticFireConfig)})
    return AutomaticFireConfig(**values)
