"""Fail-safe boundary for future water-valve control.

There is deliberately no GPIO or MOSFET implementation. The only concrete
controller stays closed and refuses requests to open the valve.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol


class ValveState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"


class ValveController(Protocol):
    """Interface for a later, explicitly armed hardware implementation."""

    @property
    def state(self) -> ValveState:
        ...

    def close(self) -> None:
        ...

    def open(self) -> None:
        ...


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
