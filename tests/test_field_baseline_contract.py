"""Retained FIELD evidence exercised against the modernization policy boundary.

No media replay, camera construction, DNN execution or real coordinator is used.
The fixture distinguishes observed field evidence from synthetic delivery clocks.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from squirrel_shooter.auto_fire import (
    AutoFireConfig,
    AutoFireDetection,
    AutoFireService,
    AutoFireTargetAssociation,
)
from squirrel_shooter.event_report import load_events


FIXTURE = Path(__file__).parent / "fixtures" / "field_events" / "20260907-072455-726-a33c3e.json"


def field_trace() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class ForbiddenPhysicalCoordinator:
    """Make a policy regression visible without constructing physical control."""

    def __init__(self) -> None:
        self.engagement_calls = 0

    def cooldown_remaining_seconds(self) -> float:
        return 0.0

    def automatic_engage(self, *_args: object, **_kwargs: object) -> object:
        self.engagement_calls += 1
        raise AssertionError("Expired FIELD target reached physical coordination")


@pytest.mark.parametrize("association_state", ["reacquisition_timeout", "coasting"])
def test_late_field_classification_cannot_revive_expired_target(
    tmp_path: Path, association_state: str,
) -> None:
    trace = field_trace()
    event = trace["historical_event"]
    observed = trace["observed"]
    classification = observed["classification"]
    replay = trace["derived_replay"]
    origin = replay["monotonic_origin"]
    source_time = classification["target_observed_monotonic"]
    coasting_at = origin + observed["coasting"]["started_monotonic"] - source_time
    timed_out_at = origin + observed["timeout"]["failed_at_monotonic"] - source_time
    delivered_at = timed_out_at + replay["result_delivery_seconds_after_recorded_timeout"]
    expires_at = coasting_at + observed["coasting"]["grace_seconds"]
    association = AutoFireTargetAssociation(
        association_state,
        expires_monotonic=expires_at if association_state == "coasting" else None,
    )
    target_requests: list[tuple[str, int]] = []

    def expired_target(event_id: str, track_id: int) -> AutoFireTargetAssociation:
        target_requests.append((event_id, track_id))
        return association

    coordinator = ForbiddenPhysicalCoordinator()
    ledger_path = tmp_path / "attempts.json"
    config = AutoFireConfig(
        enabled=True,
        min_confidence=replay["policy_min_confidence"],
        allowed_classes=tuple(replay["policy_allowed_classes"]),
        rate_limit_state_file=ledger_path,
    )
    auto = AutoFireService(
        config,
        coordinator,
        expired_target,
        lambda: False,
        monotonic_clock=lambda: delivered_at,
        wall_clock=lambda: 10_000.0,
    )

    # A class-age/threshold rejection would miss the FIELD regression's cause.
    assert delivered_at > timed_out_at > expires_at
    assert delivered_at - origin < config.classification_max_age_seconds
    assert classification["confidence_from_decision_log"] > config.min_confidence
    assert (
        datetime.fromisoformat(classification["completed_at"])
        > datetime.fromisoformat(observed["timeout"]["timestamp"])
    )
    result = auto.handle_classification(
        event_id=event["event_id"],
        track_id=event["track_id"],
        classified_observation_monotonic=origin,
        classified_frame_sequence=classification["source_frame_sequence"],
        detections=[AutoFireDetection(
            classification["label"], classification["confidence_from_decision_log"],
        )],
    )

    assert result.accepted is False
    assert result.reason == observed["decision"]["reason"] == "target_reacquisition_timeout"
    assert result.classifier_label == "bird"
    assert target_requests and set(target_requests) == {(event["event_id"], event["track_id"])}
    assert coordinator.engagement_calls == 0
    assert auto.status()["classifications_held_for_reacquisition"] == 0
    assert auto.status()["persistence"]["records"] == 0
    assert json.loads(ledger_path.read_text(encoding="utf-8"))["attempts"] == []


def test_legacy_field_event_remains_readable_without_rewriting_prediction_provenance(tmp_path: Path) -> None:
    trace = field_trace()
    historical_event = trace["historical_event"]
    event_directory = tmp_path / historical_event["event_id"]
    event_directory.mkdir()
    event_path = event_directory / "event.json"
    event_path.write_text(json.dumps(historical_event), encoding="utf-8")
    original_bytes = event_path.read_bytes()

    loaded = load_events(tmp_path)

    assert len(loaded) == 1
    assert loaded[0]["git_commit_sha"] == trace["provenance"]["field_reference_revision"]
    assert loaded[0]["predicted_class"] == "bird"
    assert loaded[0]["human_review_label"] == "squirrel"
    assert loaded[0]["track_id"] == historical_event["track_id"]
    assert event_path.read_bytes() == original_bytes
