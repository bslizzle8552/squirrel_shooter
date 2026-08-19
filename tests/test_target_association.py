from __future__ import annotations

import json
from pathlib import Path

import pytest

from squirrel_shooter.target_association import (
    AssociationPolicy,
    TargetObservation,
    evaluate_reacquisition,
)


FIXTURES = Path(__file__).parent / "fixtures" / "field_events"


def _observation(payload: dict[str, object]) -> TargetObservation:
    return TargetObservation(
        track_id=int(payload["track_id"]),
        observed_monotonic=float(payload["observed_monotonic"]),
        centroid=tuple(float(item) for item in payload["centroid"]),  # type: ignore[arg-type]
        bounding_box=tuple(int(item) for item in payload["bounding_box"]),  # type: ignore[arg-type]
        velocity=tuple(float(item) for item in payload["velocity"]),  # type: ignore[arg-type]
        confirmed=bool(payload["confirmed"]),
        event_eligible=bool(payload["event_eligible"]),
        provisional_category=str(payload["provisional_category"]),
        grouping_confidence=float(payload["grouping_confidence"]),
    )


def _field_event(name: str) -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_195417_same_track_posture_change_is_not_rejected_by_area_alone() -> None:
    fixture = _field_event("20260818-195417-725-0d4269.json")
    previous = _observation(fixture["previous"])  # type: ignore[arg-type]
    candidate = _observation(fixture["candidate"])  # type: ignore[arg-type]

    decision = evaluate_reacquisition(previous, [candidate])

    assert decision.state == "accepted"
    assert decision.selected == candidate
    assert decision.evidence[0].area_ratio == pytest.approx(2.89245, rel=1e-3)
    assert decision.evidence[0].intersection_over_union > 0.3
    assert decision.evidence[0].exact_tracker_identity is True


def test_195417_partner_person_merge_remains_fail_closed() -> None:
    fixture = _field_event("20260818-195417-725-0d4269.json")
    previous = _observation(fixture["previous"])  # type: ignore[arg-type]
    merged = TargetObservation(
        3,
        15959.529847694,
        (1080.0, 380.0),
        (880, 210, 400, 450),
        confirmed=True,
        event_eligible=True,
        provisional_category="person_sized",
        grouping_confidence=1.0,
    )

    decision = evaluate_reacquisition(previous, [merged])

    assert decision.state == "incompatible"
    assert decision.selected is None
    assert decision.evidence[0].category_safe is False


def test_195417_partner_crossing_target_makes_real_candidate_ambiguous() -> None:
    fixture = _field_event("20260818-195417-725-0d4269.json")
    previous = _observation(fixture["previous"])  # type: ignore[arg-type]
    real = _observation(fixture["candidate"])  # type: ignore[arg-type]
    crossing = TargetObservation(
        4,
        real.observed_monotonic,
        (1100.0, 370.0),
        (1055, 345, 90, 55),
        confirmed=True,
        event_eligible=True,
        provisional_category="small_animal_candidate",
        grouping_confidence=0.95,
    )

    decision = evaluate_reacquisition(previous, [real, crossing])

    assert decision.state == "ambiguous"
    assert decision.selected is None
    assert sum(item.plausible_competitor for item in decision.evidence) == 2


def test_different_nearby_tracker_is_not_silently_swapped() -> None:
    previous = TargetObservation(9, 10.0, (100.0, 100.0), (80, 80, 40, 40))
    other = TargetObservation(10, 10.1, (104.0, 102.0), (84, 82, 40, 40))

    decision = evaluate_reacquisition(previous, [other])

    assert decision.state == "incompatible"
    assert decision.reason == "tracker_identity_changed"


def test_103410_long_gap_and_fragmented_track_stays_incompatible() -> None:
    fixture = _field_event("20260818-103410-613-77abc4.json")
    previous = _observation(fixture["previous"])  # type: ignore[arg-type]
    candidate = _observation(fixture["candidate"])  # type: ignore[arg-type]

    decision = evaluate_reacquisition(previous, [candidate])

    assert decision.state == "incompatible"
    assert decision.evidence[0].elapsed_seconds == pytest.approx(1.868253919)
    assert decision.evidence[0].reason == "observation_gap"


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        ({"confirmed": False}, "candidate_not_confirmed_safe_and_eligible"),
        ({"event_eligible": False}, "candidate_not_confirmed_safe_and_eligible"),
        ({"grouping_confidence": 0.2}, "candidate_not_confirmed_safe_and_eligible"),
        ({"observed_monotonic": 11.5}, "observation_gap"),
        ({"centroid": (500.0, 500.0)}, "motion_prediction_mismatch"),
        ({"bounding_box": (0, 0, 600, 600)}, "unbounded_shape_change"),
    ],
)
def test_single_weak_or_unsafe_candidate_never_authorizes(
    updates: dict[str, object],
    reason: str,
) -> None:
    previous = TargetObservation(1, 10.0, (100.0, 100.0), (80, 80, 40, 40))
    values: dict[str, object] = {
        "track_id": 1,
        "observed_monotonic": 10.1,
        "centroid": (105.0, 100.0),
        "bounding_box": (82, 80, 45, 40),
        "confirmed": True,
        "event_eligible": True,
        "provisional_category": "small_animal_candidate",
        "grouping_confidence": 1.0,
    }
    values.update(updates)
    candidate = TargetObservation(**values)  # type: ignore[arg-type]

    decision = evaluate_reacquisition(previous, [candidate])

    assert decision.state == "incompatible"
    assert decision.evidence[0].reason == reason


def test_policy_rejects_incoherent_relaxation_bounds() -> None:
    with pytest.raises(ValueError, match="posture_change_area_ratio"):
        AssociationPolicy(ordinary_area_ratio=3.0, posture_change_area_ratio=2.9)
    with pytest.raises(ValueError, match="maximum_prediction_error_pixels"):
        AssociationPolicy(base_prediction_error_pixels=100.0, maximum_prediction_error_pixels=99.0)
