from __future__ import annotations

import pytest

from squirrel_shooter import legacy_mobilenet_policy as legacy
from squirrel_shooter.safety import SceneDetection


def test_future_squirrel_label_is_not_silently_enabled_in_legacy_autonomy() -> None:
    with pytest.raises(ValueError, match="unsupported classifier label: squirrel"):
        legacy.validate_allowed_classes(("squirrel",), enabled=True)
    assert legacy.classification_reason(
        "squirrel", 1.0, allowed_classes=("dog", "bird"), minimum_confidence=0.75
    ) == "classification_invalid"


@pytest.mark.parametrize(
    ("label", "confidence", "reason"),
    [
        ("unknown", 0.1, "classification_unknown"),
        ("unsupported_label", 0.1, "classification_invalid"),
        ("car", 0.75, "below_confidence"),
        ("car", 0.99, "class_not_allowlisted"),
        ("bird", 0.75, "below_confidence"),
        ("bird", 0.7501, None),
    ],
)
def test_legacy_qualification_retains_rejection_order_and_exact_threshold(
    label: str, confidence: float, reason: str | None
) -> None:
    assert legacy.classification_reason(
        label, confidence, allowed_classes=("dog", "bird"), minimum_confidence=0.75
    ) == reason


def test_legacy_scene_unavailable_cannot_be_interpreted_as_clear() -> None:
    result = legacy.ScenePersonSafetyResult(
        status="unavailable",
        request_id="request",
        event_id="event",
        track_id=1,
        coordinate_space="native_full_frame",
        source_sequence=None,
        source_received_monotonic=None,
        frame_width=None,
        frame_height=None,
        detections=(),
        completed_monotonic=10.0,
        error="classifier_unavailable",
    )

    assert legacy.valid_scene_result_shape(result)
    assert legacy.scene_status_reason(result) == "scene_safety_unavailable"
    assert result.status == "unavailable"


def test_legacy_scene_boundary_rejects_mutated_person_clear_result() -> None:
    result = legacy.ScenePersonSafetyResult(
        status="clear",
        request_id="request",
        event_id="event",
        track_id=1,
        coordinate_space="native_full_frame",
        source_sequence=10,
        source_received_monotonic=9.9,
        frame_width=1280,
        frame_height=720,
        detections=(),
        completed_monotonic=10.0,
    )
    object.__setattr__(result, "detections", (SceneDetection("person", 0.99, (1, 2, 3, 4)),))

    assert not legacy.valid_scene_result_shape(result)
