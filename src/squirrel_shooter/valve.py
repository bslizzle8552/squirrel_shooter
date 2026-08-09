"""Fail-safe, centralized water-valve control."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol


class ValveState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"


class ValveController(Protocol):
    """Small hardware boundary used by the control pipeline and test fakes."""

    @property
    def state(self) -> ValveState:
        ...

    def close(self) -> None:
        ...

    def open(self) -> None:
        ...

    def cleanup(self) -> None:
        ...


class DigitalOutput(Protocol):
    """Minimal active-high output used by :class:`GPIOValveController`."""

    def on(self) -> None:
        ...

    def off(self) -> None:
        ...

    def close(self) -> None:
        ...


@dataclass(frozen=True)
class ValveConfig:
    """Valve hardware settings; disabled and pinless is the safe default."""

    enabled: bool = False
    gpio_pin: int | None = None
    active_high: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be true or false")
        if self.active_high is not True:
            raise ValueError("active_high must remain true so the closed valve command is GPIO LOW")
        if self.gpio_pin is not None and (
            isinstance(self.gpio_pin, bool) or not isinstance(self.gpio_pin, int) or not 0 <= self.gpio_pin <= 27
        ):
            raise ValueError("gpio_pin must be null or a BCM GPIO number from 0 to 27")
        if self.enabled and self.gpio_pin is None:
            raise ValueError("gpio_pin must be configured before valve control can be enabled")


class DisabledValveController:
    """Safe default: always closed and incapable of energizing hardware."""

    def __init__(self) -> None:
        self._state = ValveState.CLOSED

    @property
    def state(self) -> ValveState:
        return self._state

    def close(self) -> None:
        self._state = ValveState.CLOSED

    def open(self) -> None:
        self._state = ValveState.CLOSED
        raise RuntimeError("Water control is disabled; the valve remains closed")

    def cleanup(self) -> None:
        self.close()


class GPIOValveController:
    """Normally-closed solenoid output that initializes and cleans up OFF."""

    def __init__(
        self,
        config: ValveConfig,
        *,
        output: DigitalOutput | None = None,
    ) -> None:
        if not config.enabled or config.gpio_pin is None:
            raise ValueError("GPIO valve control requires enabled=true and a configured gpio_pin")
        if output is None:
            try:
                from gpiozero import OutputDevice
            except ImportError as exc:
                raise RuntimeError("Valve hardware support is not installed; install the servo optional dependencies") from exc
            output = OutputDevice(
                config.gpio_pin,
                active_high=config.active_high,
                initial_value=False,
            )
        self._output = output
        self._state = ValveState.CLOSED
        self._cleaned_up = False
        self._output.off()

    @property
    def state(self) -> ValveState:
        return self._state

    def close(self) -> None:
        self._output.off()
        self._state = ValveState.CLOSED

    def open(self) -> None:
        if self._cleaned_up:
            raise RuntimeError("Valve controller has been cleaned up")
        self._output.on()
        self._state = ValveState.OPEN

    def cleanup(self) -> None:
        if self._cleaned_up:
            return
        try:
            self.close()
        finally:
            self._cleaned_up = True
            self._output.close()


def pulse_valve(
    controller: ValveController,
    duration_seconds: float,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Produce one bounded pulse and always attempt to leave the valve closed."""

    if (
        isinstance(duration_seconds, bool)
        or not isinstance(duration_seconds, (int, float))
        or not math.isfinite(float(duration_seconds))
        or duration_seconds <= 0
    ):
        raise ValueError("Valve pulse duration must be a finite number greater than zero")
    controller.close()
    try:
        controller.open()
        sleep(float(duration_seconds))
    finally:
        controller.close()
