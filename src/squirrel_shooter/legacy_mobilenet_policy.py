"""LEGACY MobileNet/VOC autonomous semantics, isolated for later retirement.

This module preserves the active classifier's label qualification and person
checks. It is not the future one-class squirrel policy, and no future detector
should fabricate a person-clear result to satisfy it. Native frame provenance
and physical final-aim contracts remain model-independent in ``safety``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol

from .classifier_labels import VOC_LABELS
from .motion_categories import MotionCategory
from .safety import SceneDetection


HUMAN_DENY_LABELS = frozenset({"person"})
_MODEL_LABELS = frozenset(VOC_LABELS[1:])
_UNKNOWN_LABELS = frozenset({"background", "unknown", "unclassified", "no-result", "no_result"})
SceneSafetyStatus = Literal["clear", "person", "ambiguous", "error", "unavailable"]


def validate_allowed_classes(allowed_classes: tuple[str, ...], *, enabled: bool) -> None:
    """Validate the currently selected legacy vocabulary without changing policy."""

    if not isinstance(allowed_classes, tuple):
        raise ValueError("allowed_classes must be a tuple")
    if len(set(allowed_classes)) != len(allowed_classes):
        raise ValueError("allowed_classes must not contain duplicates")
    for label in allowed_classes:
        if not isinstance(label, str) or not label or label != label.strip().lower():
            raise ValueError("allowed_classes must contain canonical lowercase labels")
        if label not in _MODEL_LABELS:
            raise ValueError(f"allowed_classes contains unsupported classifier label: {label}")
        if label in HUMAN_DENY_LABELS:
            raise ValueError(f"allowed_classes must never contain human deny label: {label}")
    if enabled and not allowed_classes:
        raise ValueError("enabled auto-fire requires at least one allowed class")


def is_human_label(label: str) -> bool:
    """Preserve the legacy top-result event veto and full-scene any-person veto."""

    return label in HUMAN_DENY_LABELS


def classification_reason(
    label: str,
    confidence: float,
    *,
    allowed_classes: tuple[str, ...],
    minimum_confidence: float,
) -> str | None:
    """Keep legacy rejection order and its strictly-greater confidence rule."""

    if label in _UNKNOWN_LABELS:
        return "classification_unknown"
    if label not in _MODEL_LABELS:
        return "classification_invalid"
    if confidence <= minimum_confidence:
        return "below_confidence"
    if label not in allowed_classes:
        return "class_not_allowlisted"
    return None


def target_category_reason(category: str) -> str | None:
    """Retain the legacy motion-size veto independently of target identity."""

    return "person_sized_target" if category == MotionCategory.PERSON_SIZED else None


@dataclass(frozen=True)
class ScenePersonSafetyResult:
    """LEGACY MobileNet result; never use squirrel absence to construct ``clear``."""

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
    """LEGACY MobileNet person-check seam, retained until detector migration."""

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


def valid_scene_result_shape(result: ScenePersonSafetyResult) -> bool:
    """Defend the physical boundary even against bypassed/mutated dataclasses."""

    try:
        if not isinstance(result.detections, tuple):
            return False
        if (
            result.status not in {"clear", "person", "ambiguous", "error", "unavailable"}
            or not isinstance(result.request_id, str)
            or not result.request_id
            or not isinstance(result.event_id, str)
            or not result.event_id
            or isinstance(result.track_id, bool)
            or not isinstance(result.track_id, int)
            or result.track_id <= 0
            or result.coordinate_space != "native_full_frame"
            or (
                result.status in {"error", "unavailable", "ambiguous"}
                and (not isinstance(result.error, str) or not result.error)
            )
            or (result.status in {"clear", "person"} and result.error is not None)
        ):
            return False
        if (
            isinstance(result.completed_monotonic, bool)
            or not isinstance(result.completed_monotonic, (int, float))
            or not math.isfinite(float(result.completed_monotonic))
            or result.completed_monotonic < 0.0
        ):
            return False
        source_values = (
            result.source_sequence,
            result.source_received_monotonic,
            result.frame_width,
            result.frame_height,
        )
        if any(value is None for value in source_values):
            return (
                all(value is None for value in source_values)
                and result.status in {"error", "unavailable"}
                and not result.detections
            )
        if (
            isinstance(result.source_sequence, bool)
            or not isinstance(result.source_sequence, int)
            or result.source_sequence < 0
            or isinstance(result.source_received_monotonic, bool)
            or not isinstance(result.source_received_monotonic, (int, float))
            or not math.isfinite(float(result.source_received_monotonic))
            or result.source_received_monotonic < 0.0
            or result.completed_monotonic < result.source_received_monotonic
        ):
            return False
        if (
            isinstance(result.frame_width, bool)
            or not isinstance(result.frame_width, int)
            or result.frame_width <= 0
            or isinstance(result.frame_height, bool)
            or not isinstance(result.frame_height, int)
            or result.frame_height <= 0
        ):
            return False
        for detection in result.detections:
            if not isinstance(detection, SceneDetection):
                return False
            if (
                not isinstance(detection.label, str)
                or not detection.label
                or detection.label != detection.label.strip().lower()
                or isinstance(detection.confidence, bool)
                or not isinstance(detection.confidence, (int, float))
                or not math.isfinite(float(detection.confidence))
                or not 0.0 <= detection.confidence <= 1.0
                or not isinstance(detection.bounding_box, tuple)
                or len(detection.bounding_box) != 4
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in detection.bounding_box
                )
            ):
                return False
            x, y, width, height = detection.bounding_box
            if (
                x < 0
                or y < 0
                or width <= 0
                or height <= 0
                or x + width > result.frame_width
                or y + height > result.frame_height
            ):
                return False
        has_person = any(item.label == "person" for item in result.detections)
        if (result.status == "person") != has_person and (
            result.status == "person" or result.status == "clear"
        ):
            return False
    except (AttributeError, TypeError, ValueError):
        return False
    return True

def scene_has_person(result: ScenePersonSafetyResult) -> bool:
    return result.status == "person" or any(is_human_label(item.label) for item in result.detections)


def scene_status_reason(result: ScenePersonSafetyResult) -> str | None:
    """Interpret legacy statuses only; ``clear`` is never synthesized here."""

    if result.status == "clear":
        return None
    return {
        "ambiguous": "scene_safety_ambiguous",
        "error": "scene_safety_error",
        "unavailable": "scene_safety_unavailable",
    }.get(result.status, "scene_safety_unavailable")
