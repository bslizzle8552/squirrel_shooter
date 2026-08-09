from __future__ import annotations

import pytest

from squirrel_shooter.valve import GPIOValveController, ValveConfig, ValveState, pulse_valve


class FakeOutput:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def on(self) -> None:
        self.calls.append("on")

    def off(self) -> None:
        self.calls.append("off")

    def close(self) -> None:
        self.calls.append("close")


def test_gpio_valve_starts_low_and_cleanup_leaves_it_low() -> None:
    output = FakeOutput()
    valve = GPIOValveController(ValveConfig(enabled=True, gpio_pin=17), output=output)

    assert valve.state is ValveState.CLOSED
    assert output.calls == ["off"]
    valve.open()
    assert valve.state is ValveState.OPEN
    valve.cleanup()

    assert valve.state is ValveState.CLOSED
    assert output.calls == ["off", "on", "off", "close"]


def test_pulse_always_closes_after_sleep_error() -> None:
    output = FakeOutput()
    valve = GPIOValveController(ValveConfig(enabled=True, gpio_pin=17), output=output)

    with pytest.raises(RuntimeError, match="test interruption"):
        pulse_valve(valve, 0.25, sleep=lambda _seconds: (_ for _ in ()).throw(RuntimeError("test interruption")))

    assert valve.state is ValveState.CLOSED
    assert output.calls[-1] == "off"


def test_enabled_valve_requires_an_explicit_gpio_pin() -> None:
    with pytest.raises(ValueError, match="gpio_pin"):
        ValveConfig(enabled=True, gpio_pin=None)
