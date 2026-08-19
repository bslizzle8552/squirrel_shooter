"""Fail-closed contracts shared by classifier, auto-fire, and physical control."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np


SceneSafetyStatus = Literal["clear", "person", "ambiguous", "error", "unavailable"]
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
class ScenePersonSafetyResult:
    """Immutable full-scene result; anything except current ``clear`` is unsafe."""

    status: SceneSafetyStatus
    request_id: str
    event_id: str
    track_id: int
    coordinate_space: Literal["native_full_frame"]
    source_sequence: int | None
    source_received_monotonic: float | None
    frame_width: int | None
    frame_height: int | None
    detections: tuple[SceneDetection, ...]
    completed_monotonic: float
    error: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"clear", "person", "ambiguous", "error", "unavailable"}:
            raise ValueError("unsupported scene safety status")
        for name, value in (("request_id", self.request_id), ("event_id", self.event_id)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be non-empty")
        if isinstance(self.track_id, bool) or not isinstance(self.track_id, int) or self.track_id <= 0:
            raise ValueError("track_id must be a positive integer")
        if self.coordinate_space != "native_full_frame":
            raise ValueError("coordinate_space must be native_full_frame")
        if (
            isinstance(self.completed_monotonic, bool)
            or not isinstance(self.completed_monotonic, (int, float))
            or not math.isfinite(float(self.completed_monotonic))
            or self.completed_monotonic < 0.0
        ):
            raise ValueError("completed_monotonic must be finite and non-negative")
        source_values = (
            self.source_sequence,
            self.source_received_monotonic,
            self.frame_width,
            self.frame_height,
        )
        if any(value is None for value in source_values) and not all(value is None for value in source_values):
            raise ValueError("source geometry and timing must be wholly present or absent")
        if not isinstance(self.detections, tuple) or any(
            not isinstance(item, SceneDetection) for item in self.detections
        ):
            raise ValueError("detections must be a tuple of SceneDetection values")
        if self.source_sequence is not None:
            if (
                isinstance(self.source_sequence, bool)
                or not isinstance(self.source_sequence, int)
                or self.source_sequence < 0
            ):
                raise ValueError("source_sequence must be non-negative")
            if (
                isinstance(self.source_received_monotonic, bool)
                or not isinstance(self.source_received_monotonic, (int, float))
                or not math.isfinite(float(self.source_received_monotonic))
                or self.source_received_monotonic < 0.0
            ):
                raise ValueError("source_received_monotonic must be finite and non-negative")
            if (
                isinstance(self.frame_width, bool)
                or not isinstance(self.frame_width, int)
                or self.frame_width <= 0
                or isinstance(self.frame_height, bool)
                or not isinstance(self.frame_height, int)
                or self.frame_height <= 0
            ):
                raise ValueError("source frame dimensions must be positive integers")
            for detection in self.detections:
                x, y, width, height = detection.bounding_box
                if x + width > self.frame_width or y + height > self.frame_height:
                    raise ValueError("scene detection must fit inside the native frame")
            if self.completed_monotonic < self.source_received_monotonic:
                raise ValueError("completion cannot precede the source frame")
        has_person = any(item.label == "person" for item in self.detections)
        if self.status == "person" and not has_person:
            raise ValueError("person status requires a person detection")
        if self.status == "clear" and has_person:
            raise ValueError("clear status cannot contain a person detection")
        if self.status in {"clear", "person", "ambiguous"} and self.source_sequence is None:
            raise ValueError(f"{self.status} status requires a source frame")
        if self.status in {"error", "unavailable", "ambiguous"} and not self.error:
            raise ValueError(f"{self.status} status requires an explanation")
        if self.status in {"clear", "person"} and self.error is not None:
            raise ValueError(f"{self.status} status cannot carry an error")


class ScenePersonSafetyProvider(Protocol):
    """Wiring seam for current, latest-only full-frame safety inference."""

    def check_current_scene(
        self,
        *,
        request_id: str,
        event_id: str,
        track_id: int,
        after_sequence: int | None,
        timeout_seconds: float,
    ) -> ScenePersonSafetyResult:
        ...

    def latest_scene_result(self) -> ScenePersonSafetyResult | None:
        ...


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
    "ScenePersonSafetyProvider",
    "ScenePersonSafetyResult",
    "SceneSafetyStatus",
]
