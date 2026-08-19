from __future__ import annotations

import json
from pathlib import Path

import pytest

from squirrel_shooter.target_association import TargetObservation, evaluate_reacquisition


FIXTURES = Path(__file__).parent / "fixtures" / "field_events"


def _observation(payload: dict[str, object]) -> TargetObservation:
    return TargetObservation(
        track_id=int(payload["track_id"]),
        observed_monotonic=float(payload["observed_monotonic"]),
        centroid=tuple(float(item) for item in payload["centroid"]),  # type: ignore[arg-type]
        bounding_box=tuple(int(item) for item in payload["bounding_box"]),  # type: ignore[arg-type]
        velocity=tuple(float(item) for item in payload.get("velocity", (0.0, 0.0))),
        confirmed=bool(payload["confirmed"]),
        event_eligible=bool(payload["event_eligible"]),
        provisional_category=str(payload["provisional_category"]),
        grouping_confidence=float(payload["grouping_confidence"]),
    )


def _field_event(name: str) -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_195417_same_track_posture_change_reacquires_with_combined_evidence() -> None:
    fixture = _field_event("20260818-195417-725-0d4269.json")
    previous = _observation(fixture["previous"])  # type: ignore[arg-type]
    candidate = _observation(fixture["candidate"])  # type: ignore[arg-type]

    decision = evaluate_reacquisition(previous, [candidate])

    assert decision.state == "accepted"
    assert decision.selected == candidate
    assert decision.evidence[0].area_ratio == pytest.approx(2.89245, rel=1e-3)
    assert decision.evidence[0].intersection_over_union > 0.3
    assert decision.evidence[0].exact_tracker_identity is True


def test_different_tracker_nearby_does_not_inherit_classification() -> None:
    previous = TargetObservation(9, 10.0, (100.0, 100.0), (80, 80, 40, 40))
    other = TargetObservation(10, 10.1, (104.0, 102.0), (84, 82, 40, 40))

    decision = evaluate_reacquisition(previous, [other])

    assert decision.state == "incompatible"
    assert decision.reason == "tracker_identity_changed"


def test_plausible_competing_candidate_makes_reacquisition_ambiguous() -> None:
    fixture = _field_event("20260818-195417-725-0d4269.json")
    previous = _observation(fixture["previous"])  # type: ignore[arg-type]
    real = _observation(fixture["candidate"])  # type: ignore[arg-type]
    crossing = TargetObservation(
        4,
        real.observed_monotonic,
        (1100.0, 370.0),
        (1055, 345, 90, 55),
    )

    decision = evaluate_reacquisition(previous, [real, crossing])

    assert decision.state == "ambiguous"
    assert decision.selected is None


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        ({"confirmed": False}, "candidate_not_confirmed_safe_and_eligible"),
        ({"event_eligible": False}, "candidate_not_confirmed_safe_and_eligible"),
        ({"provisional_category": "person_sized"}, "candidate_not_confirmed_safe_and_eligible"),
        ({"observed_monotonic": 11.1}, "observation_gap"),
        ({"bounding_box": (0, 0, 600, 600)}, "unbounded_shape_change"),
    ],
)
def test_weak_stale_or_unsafe_candidate_never_authorizes(
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

    decision = evaluate_reacquisition(
        previous,
        [TargetObservation(**values)],  # type: ignore[arg-type]
    )

    assert decision.state == "incompatible"
    assert decision.evidence[0].reason == reason


def test_recent_velocity_can_support_bounded_motion_beyond_stale_100_pixel_circle() -> None:
    previous = TargetObservation(
        1,
        10.0,
        (100.0, 100.0),
        (20, 80, 160, 40),
        velocity=(240.0, 0.0),
    )
    candidate = TargetObservation(1, 10.5, (220.0, 100.0), (140, 80, 160, 40))

    decision = evaluate_reacquisition(previous, [candidate])

    assert decision.state == "accepted"
    assert decision.evidence[0].last_centroid_distance_pixels == 120.0
    assert decision.evidence[0].prediction_error_pixels == 0.0
