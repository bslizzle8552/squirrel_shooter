"""Model-independent native-frame and final physical-control contracts.

These types do not certify a species or scene clearance. The operational legacy
MobileNet person-check contract lives in ``legacy_mobilenet_policy``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np


FinalAimAction = Literal["accept", "reaim", "reject"]


@dataclass(frozen=True)
class SceneDetection:
    """One detector result expressed in native full-frame coordinates."""

    label: str
    confidence: float
    bounding_box: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.label, str)
            or not self.label
            or self.label != self.label.strip().lower()
        ):
            raise ValueError("label must be canonical lowercase")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(float(self.confidence))
            or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("confidence must be between zero and one")
        if (
            not isinstance(self.bounding_box, tuple)
            or len(self.bounding_box) != 4
            or any(isinstance(value, bool) or not isinstance(value, int) for value in self.bounding_box)
        ):
            raise ValueError("bounding_box must contain four integers")
        if self.bounding_box[0] < 0 or self.bounding_box[1] < 0:
            raise ValueError("bounding_box origin must be non-negative")
        if self.bounding_box[2] <= 0 or self.bounding_box[3] <= 0:
            raise ValueError("bounding_box dimensions must be positive")


@dataclass(frozen=True)
class SceneFramePacket:
    """A borrowed immutable native frame supplied by the sole camera service."""

    sequence: int
    received_monotonic: float
    frame: np.ndarray

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 0:
            raise ValueError("sequence must be a non-negative integer")
        if (
            isinstance(self.received_monotonic, bool)
            or not isinstance(self.received_monotonic, (int, float))
            or not math.isfinite(float(self.received_monotonic))
            or self.received_monotonic < 0.0
        ):
            raise ValueError("received_monotonic must be finite and non-negative")
        if not isinstance(self.frame, np.ndarray) or self.frame.ndim < 2:
            raise ValueError("frame must be a native image array")
        if self.frame.shape[0] <= 0 or self.frame.shape[1] <= 0:
            raise ValueError("frame dimensions must be positive")

    @property
    def width(self) -> int:
        return int(self.frame.shape[1])

    @property
    def height(self) -> int:
        return int(self.frame.shape[0])


@dataclass(frozen=True)
class FinalAimDecision:
    """Final physical guard outcome with at most one explicit re-aim request."""

    action: FinalAimAction
    reason: str
    pixel_x: int | None = None
    pixel_y: int | None = None

    def __post_init__(self) -> None:
        if self.action not in {"accept", "reaim", "reject"}:
            raise ValueError("unsupported final aim action")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be non-empty")
        if self.action == "reaim":
            for name, value in (("pixel_x", self.pixel_x), ("pixel_y", self.pixel_y)):
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"{name} must be non-negative for reaim")
        elif self.pixel_x is not None or self.pixel_y is not None:
            raise ValueError("only reaim decisions may carry a target pixel")

    @classmethod
    def accept(cls, reason: str = "current_target_consistent") -> FinalAimDecision:
        return cls("accept", reason)

    @classmethod
    def reaim(cls, pixel_x: int, pixel_y: int, reason: str = "bounded_target_movement") -> FinalAimDecision:
        return cls("reaim", reason, pixel_x, pixel_y)

    @classmethod
    def reject(cls, reason: str) -> FinalAimDecision:
        return cls("reject", reason)


__all__ = [
    "FinalAimAction",
    "FinalAimDecision",
    "SceneDetection",
    "SceneFramePacket",
]
