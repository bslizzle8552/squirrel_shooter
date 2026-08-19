"""Evidence-weighted, fail-closed live-target reacquisition."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Literal


AssociationState = Literal["accepted", "ambiguous", "incompatible"]


@dataclass(frozen=True)
class AssociationPolicy:
    """Conservative geometry bounds used together rather than as lone vetoes."""

    maximum_elapsed_seconds: float = 0.9
    base_prediction_error_pixels: float = 100.0
    maximum_prediction_error_pixels: float = 200.0
    strong_proximity_pixels: float = 60.0
    ordinary_area_ratio: float = 2.5
    posture_change_area_ratio: float = 4.0
    minimum_overlap_for_posture_change: float = 0.10
    maximum_aspect_ratio_change: float = 3.0
    minimum_grouping_confidence: float = 0.5

    def __post_init__(self) -> None:
        positive = (
            "maximum_elapsed_seconds",
            "base_prediction_error_pixels",
            "maximum_prediction_error_pixels",
            "strong_proximity_pixels",
            "ordinary_area_ratio",
            "posture_change_area_ratio",
            "maximum_aspect_ratio_change",
        )
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if self.maximum_prediction_error_pixels < self.base_prediction_error_pixels:
            raise ValueError("maximum_prediction_error_pixels must be at least the base")
        if self.strong_proximity_pixels > self.base_prediction_error_pixels:
            raise ValueError("strong_proximity_pixels must not exceed the base prediction error")
        if self.ordinary_area_ratio < 1.0:
            raise ValueError("ordinary_area_ratio must be at least one")
        if self.posture_change_area_ratio < self.ordinary_area_ratio:
            raise ValueError("posture_change_area_ratio must be at least ordinary_area_ratio")
        for name in ("minimum_overlap_for_posture_change", "minimum_grouping_confidence"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be between zero and one")


@dataclass(frozen=True)
class TargetObservation:
    track_id: int
    observed_monotonic: float
    centroid: tuple[float, float]
    bounding_box: tuple[int, int, int, int]
    velocity: tuple[float, float] = (0.0, 0.0)
    confirmed: bool = True
    event_eligible: bool = True
    provisional_category: str = "small_animal_candidate"
    grouping_confidence: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.track_id, bool) or not isinstance(self.track_id, int) or self.track_id <= 0:
            raise ValueError("track_id must be a positive integer")
        for name, values in (("centroid", self.centroid), ("velocity", self.velocity)):
            if len(values) != 2 or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in values
            ):
                raise ValueError(f"{name} must contain two finite numbers")
        if (
            isinstance(self.observed_monotonic, bool)
            or not isinstance(self.observed_monotonic, (int, float))
            or not math.isfinite(float(self.observed_monotonic))
            or self.observed_monotonic < 0
        ):
            raise ValueError("observed_monotonic must be a finite non-negative number")
        if len(self.bounding_box) != 4 or any(
            isinstance(value, bool) or not isinstance(value, int) for value in self.bounding_box
        ):
            raise ValueError("bounding_box must contain four integers")
        if self.bounding_box[2] <= 0 or self.bounding_box[3] <= 0:
            raise ValueError("bounding_box dimensions must be positive")
        if not isinstance(self.confirmed, bool) or not isinstance(self.event_eligible, bool):
            raise ValueError("confirmed and event_eligible must be booleans")
        if not isinstance(self.provisional_category, str) or not self.provisional_category:
            raise ValueError("provisional_category must be non-empty")
        if (
            isinstance(self.grouping_confidence, bool)
            or not isinstance(self.grouping_confidence, (int, float))
            or not math.isfinite(float(self.grouping_confidence))
            or not 0.0 <= self.grouping_confidence <= 1.0
        ):
            raise ValueError("grouping_confidence must be between zero and one")


@dataclass(frozen=True)
class CandidateEvidence:
    track_id: int
    elapsed_seconds: float
    last_centroid_distance_pixels: float
    predicted_centroid: tuple[float, float]
    prediction_error_pixels: float
    prediction_allowance_pixels: float
    area_ratio: float
    aspect_ratio_change: float
    intersection_over_union: float
    previous_centroid_inside_candidate: bool
    candidate_centroid_inside_previous: bool
    exact_tracker_identity: bool
    confirmed: bool
    event_eligible: bool
    category_safe: bool
    grouping_confident: bool
    selectable: bool
    plausible_competitor: bool
    reason: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AssociationDecision:
    state: AssociationState
    reason: str
    selected: TargetObservation | None
    evidence: tuple[CandidateEvidence, ...]


def evaluate_reacquisition(
    previous: TargetObservation,
    candidates: Iterable[TargetObservation],
    *,
    policy: AssociationPolicy = AssociationPolicy(),
) -> AssociationDecision:
    """Choose only one strong same-track continuation with no plausible rival."""

    parsed = tuple(candidates)
    evidence = tuple(_candidate_evidence(previous, candidate, policy) for candidate in parsed)
    selectable = [
        (candidate, facts)
        for candidate, facts in zip(parsed, evidence, strict=True)
        if facts.selectable
    ]
    competitors = [
        facts
        for facts in evidence
        if facts.plausible_competitor and not facts.selectable
    ]
    if len(selectable) > 1 or (selectable and competitors):
        return AssociationDecision("ambiguous", "multiple_plausible_candidates", None, evidence)
    if len(selectable) == 1:
        candidate, _facts = selectable[0]
        return AssociationDecision("accepted", "strong_same_track_continuity", candidate, evidence)
    if any(facts.plausible_competitor for facts in evidence):
        return AssociationDecision("incompatible", "tracker_identity_changed", None, evidence)
    reason = evidence[0].reason if len(evidence) == 1 else "no_compatible_candidate"
    return AssociationDecision("incompatible", reason, None, evidence)


def _candidate_evidence(
    previous: TargetObservation,
    candidate: TargetObservation,
    policy: AssociationPolicy,
) -> CandidateEvidence:
    elapsed = candidate.observed_monotonic - previous.observed_monotonic
    predicted = (
        previous.centroid[0] + previous.velocity[0] * max(0.0, elapsed),
        previous.centroid[1] + previous.velocity[1] * max(0.0, elapsed),
    )
    speed = math.hypot(*previous.velocity)
    allowance = min(
        policy.maximum_prediction_error_pixels,
        policy.base_prediction_error_pixels + min(
            policy.base_prediction_error_pixels,
            speed * max(0.0, elapsed) * 0.25,
        ),
    )
    last_distance = math.dist(previous.centroid, candidate.centroid)
    prediction_error = math.dist(predicted, candidate.centroid)
    area_ratio = _ratio(_area(previous.bounding_box), _area(candidate.bounding_box))
    aspect_ratio_change = _ratio(_aspect(previous.bounding_box), _aspect(candidate.bounding_box))
    overlap = _intersection_over_union(previous.bounding_box, candidate.bounding_box)
    previous_inside = _inside(previous.centroid, candidate.bounding_box)
    candidate_inside = _inside(candidate.centroid, previous.bounding_box)
    exact_identity = candidate.track_id == previous.track_id
    category_safe = candidate.provisional_category not in {
        "person_sized",
        "large_object_candidate",
        "lighting_change",
    }
    grouping_confident = candidate.grouping_confidence >= policy.minimum_grouping_confidence
    current = 0.0 <= elapsed <= policy.maximum_elapsed_seconds
    predicted_near = prediction_error <= allowance
    shape_bounded = (
        area_ratio <= policy.ordinary_area_ratio
        or (
            area_ratio <= policy.posture_change_area_ratio
            and overlap >= policy.minimum_overlap_for_posture_change
            and (
                prediction_error <= policy.strong_proximity_pixels
                or previous_inside
                or candidate_inside
            )
        )
    )
    aspect_bounded = aspect_ratio_change <= policy.maximum_aspect_ratio_change
    quality = (
        candidate.confirmed
        and candidate.event_eligible
        and category_safe
        and grouping_confident
    )
    geometry_signals = sum(
        (
            prediction_error <= policy.strong_proximity_pixels,
            last_distance <= policy.base_prediction_error_pixels,
            overlap >= policy.minimum_overlap_for_posture_change,
            previous_inside or candidate_inside,
        )
    )
    selectable = bool(
        exact_identity
        and current
        and predicted_near
        and shape_bounded
        and aspect_bounded
        and quality
        and geometry_signals >= 2
    )
    plausible_competitor = bool(
        current
        and predicted_near
        and quality
        and (overlap > 0.0 or last_distance <= policy.base_prediction_error_pixels)
    )
    reason = "accepted"
    if not current:
        reason = "observation_gap"
    elif not quality:
        reason = "candidate_not_confirmed_safe_and_eligible"
    elif not predicted_near:
        reason = "motion_prediction_mismatch"
    elif not shape_bounded:
        reason = "unbounded_shape_change"
    elif not aspect_bounded:
        reason = "unbounded_aspect_change"
    elif geometry_signals < 2:
        reason = "insufficient_geometry_evidence"
    elif not exact_identity:
        reason = "different_tracker_identity"
    return CandidateEvidence(
        candidate.track_id,
        elapsed,
        last_distance,
        predicted,
        prediction_error,
        allowance,
        area_ratio,
        aspect_ratio_change,
        overlap,
        previous_inside,
        candidate_inside,
        exact_identity,
        candidate.confirmed,
        candidate.event_eligible,
        category_safe,
        grouping_confident,
        selectable,
        plausible_competitor,
        reason,
    )


def _area(box: tuple[int, int, int, int]) -> float:
    return float(box[2] * box[3])


def _aspect(box: tuple[int, int, int, int]) -> float:
    return box[2] / box[3]


def _ratio(left: float, right: float) -> float:
    return max(left, right) / max(1e-9, min(left, right))


def _inside(point: tuple[float, float], box: tuple[int, int, int, int]) -> bool:
    x, y, width, height = box
    return x <= point[0] < x + width and y <= point[1] < y + height


def _intersection_over_union(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    left_x2, left_y2 = left[0] + left[2], left[1] + left[3]
    right_x2, right_y2 = right[0] + right[2], right[1] + right[3]
    width = max(0, min(left_x2, right_x2) - max(left[0], right[0]))
    height = max(0, min(left_y2, right_y2) - max(left[1], right[1]))
    intersection = width * height
    union = _area(left) + _area(right) - intersection
    return 0.0 if union <= 0.0 else intersection / union


__all__ = [
    "AssociationDecision",
    "AssociationPolicy",
    "CandidateEvidence",
    "TargetObservation",
    "evaluate_reacquisition",
]
