from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest

from squirrel_shooter.auto_fire import (
    HUMAN_DENY_LABELS,
    AutoFireConfig,
    AutoFireDetection,
    AutoFireService,
    AutoFireTargetAssociation,
    AutoFireTargetSnapshot,
)
from squirrel_shooter.classifier_labels import VOC_LABELS


class Clocks:
    def __init__(self, *, monotonic: float = 100.0, wall: float = 10_000.0) -> None:
        self.monotonic = monotonic
        self.wall = wall

    def monotonic_now(self) -> float:
        return self.monotonic

    def wall_now(self) -> float:
        return self.wall


class TargetSource:
    def __init__(self, target: AutoFireTargetSnapshot | None) -> None:
        self.target = target

    def __call__(self, _event_id: str, _track_id: int) -> AutoFireTargetSnapshot | None:
        return self.target


class AssociationSource:
    def __init__(
        self,
        state: AutoFireTargetSnapshot | AutoFireTargetAssociation | None,
    ) -> None:
        self.state = state

    def __call__(
        self,
        _event_id: str,
        _track_id: int,
    ) -> AutoFireTargetSnapshot | AutoFireTargetAssociation | None:
        return self.state


@dataclass
class CoordinatorResult:
    fired: bool = True
    reason: str | None = None
    recording_queued: bool | None = None


class CoordinatorError(RuntimeError):
    def __init__(self, reason: str | None = None, *, shot_attempted: bool = False) -> None:
        self.reason = reason
        self.shot_attempted = shot_attempted
        super().__init__(reason or "coordinator failed")


class FakeCoordinator:
    def __init__(self) -> None:
        self.cooldown = 0.0
        self.calls: list[dict[str, object]] = []
        self.before_final: Callable[[], None] | None = None
        self.raise_error: Exception | None = None
        self.skip_final = False
        self.result: CoordinatorResult | None = None

    def cooldown_remaining_seconds(self) -> float:
        return self.cooldown

    def automatic_engage(
        self,
        pixel_x: int,
        pixel_y: int,
        *,
        frame_width: int,
        frame_height: int,
        cooldown_seconds: float,
        final_safety_check: Callable[[], bool],
        evidence: dict[str, object],
    ) -> object:
        self.calls.append(
            {
                "pixel": (pixel_x, pixel_y),
                "frame_size": (frame_width, frame_height),
                "cooldown_seconds": cooldown_seconds,
                "evidence": evidence,
            }
        )
        if self.before_final is not None:
            self.before_final()
        if self.raise_error is not None:
            raise self.raise_error
        if self.skip_final:
            return CoordinatorResult()
        final_safe = final_safety_check()
        if self.result is not None:
            return self.result if final_safe else CoordinatorResult(fired=False)
        return CoordinatorResult(fired=final_safe)


def target(
    *,
    event_id: str = "event-one",
    track_id: int = 7,
    observed: float = 99.5,
    pixel: tuple[int, int] = (50, 50),
    box: tuple[int, int, int, int] = (40, 40, 20, 20),
    frame_size: tuple[int, int] = (1280, 720),
    confirmed: bool = True,
    eligible: bool = True,
    category: str = "small_animal_candidate",
) -> AutoFireTargetSnapshot:
    return AutoFireTargetSnapshot(
        event_id,
        track_id,
        observed,
        pixel[0],
        pixel[1],
        box,
        frame_size[0],
        frame_size[1],
        confirmed,
        eligible,
        category,
    )


def config(tmp_path: Path, **changes: object) -> AutoFireConfig:
    values: dict[str, object] = {
        "enabled": True,
        "allowed_classes": ("dog", "bird"),
        "rate_limit_state_file": tmp_path / "auto-fire-state.json",
    }
    values.update(changes)
    return AutoFireConfig(**values)  # type: ignore[arg-type]


def service(
    tmp_path: Path,
    *,
    config_changes: dict[str, object] | None = None,
    current_target: AutoFireTargetSnapshot | None = None,
    night: list[bool] | None = None,
    clocks: Clocks | None = None,
    coordinator: FakeCoordinator | None = None,
) -> tuple[AutoFireService, TargetSource, list[bool], Clocks, FakeCoordinator]:
    clock = clocks or Clocks()
    source = TargetSource(current_target or target())
    night_state = night or [False]
    hardware = coordinator or FakeCoordinator()
    auto = AutoFireService(
        config(tmp_path, **(config_changes or {})),
        hardware,
        source,
        lambda: night_state[0],
        monotonic_clock=clock.monotonic_now,
        wall_clock=clock.wall_now,
    )
    return auto, source, night_state, clock, hardware


def classify(
    auto: AutoFireService,
    *,
    event_id: str = "event-one",
    track_id: int = 7,
    observed: float = 99.5,
    detections: object = (AutoFireDetection("dog", 0.90),),
    error: str | None = None,
):
    return auto.handle_classification(
        event_id=event_id,
        track_id=track_id,
        classified_observation_monotonic=observed,
        detections=detections,  # type: ignore[arg-type]
        error=error,
    )


def test_classifier_vocabulary_is_canonical_and_human_deny_is_explicit() -> None:
    assert VOC_LABELS == (
        "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
        "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
        "sheep", "sofa", "train", "tvmonitor",
    )
    assert HUMAN_DENY_LABELS == {"person"}
    assert "rabbit" not in VOC_LABELS and "squirrel" not in VOC_LABELS


def test_config_defaults_are_disabled_and_validation_is_strict(tmp_path: Path) -> None:
    defaults = AutoFireConfig()
    assert defaults.enabled is False
    assert defaults.min_confidence == 0.75
    assert defaults.allowed_classes == ()
    assert defaults.cooldown_seconds == 5.0
    assert defaults.max_shots_per_event == 1
    assert defaults.minimum_reengagement_seconds == 60.0
    assert defaults.max_shots_per_hour == 6
    assert defaults.classification_max_age_seconds == 3.0
    assert defaults.target_max_age_seconds == 0.75
    assert defaults.track_loss_grace_seconds == 0.9
    assert defaults.reacquisition_max_centroid_distance_pixels == 100.0
    assert defaults.reacquisition_max_area_ratio == 2.5

    with pytest.raises(ValueError, match="requires at least one"):
        AutoFireConfig(enabled=True, rate_limit_state_file=tmp_path / "state.json")
    with pytest.raises(ValueError, match="human deny"):
        config(tmp_path, allowed_classes=("person",))
    with pytest.raises(ValueError, match="unsupported"):
        config(tmp_path, allowed_classes=("rabbit",))
    with pytest.raises(ValueError, match="duplicates"):
        config(tmp_path, allowed_classes=("dog", "dog"))
    with pytest.raises(ValueError, match="min_confidence"):
        config(tmp_path, min_confidence=1.1)
    with pytest.raises(ValueError, match="min_confidence"):
        config(tmp_path, min_confidence=0.69)
    with pytest.raises(ValueError, match="positive integer"):
        config(tmp_path, max_shots_per_hour=True)
    with pytest.raises(ValueError, match="track_loss_grace_seconds"):
        config(tmp_path, track_loss_grace_seconds=3.1)
    with pytest.raises(ValueError, match="reacquisition_max_area_ratio"):
        config(tmp_path, reacquisition_max_area_ratio=0.99)


def test_brief_dropout_holds_classification_then_uses_reacquired_position(tmp_path: Path) -> None:
    clocks = Clocks()
    source = AssociationSource(AutoFireTargetAssociation("coasting", 100.9))
    hardware = FakeCoordinator()
    auto = AutoFireService(
        config(tmp_path),
        hardware,
        source,
        lambda: False,
        monotonic_clock=clocks.monotonic_now,
        wall_clock=clocks.wall_now,
    )
    decisions: list[object] = []
    worker = threading.Thread(target=lambda: decisions.append(classify(auto)))
    worker.start()

    for _ in range(100):
        if auto.status()["classifications_held_for_reacquisition"] == 1:
            break
        worker.join(0.005)
    assert hardware.calls == []
    source.state = target(observed=100.2, pixel=(82, 61), box=(70, 50, 30, 24))
    clocks.monotonic = 100.2
    auto.notify_target_state_changed()
    worker.join(1.0)

    assert not worker.is_alive()
    assert decisions[0].accepted is True  # type: ignore[union-attr]
    assert hardware.calls[0]["pixel"] == (82, 61)
    assert auto.status()["candidates_evaluated"] == 1


def test_classifier_result_while_coasting_does_not_fire_before_reacquisition(tmp_path: Path) -> None:
    clocks = Clocks()
    source = AssociationSource(AutoFireTargetAssociation("coasting", 100.9))
    hardware = FakeCoordinator()
    auto = AutoFireService(
        config(tmp_path), hardware, source, lambda: False,
        monotonic_clock=clocks.monotonic_now, wall_clock=clocks.wall_now,
    )
    worker = threading.Thread(target=lambda: classify(auto))
    worker.start()
    for _ in range(100):
        if auto.status()["classifications_held_for_reacquisition"] == 1:
            break
        worker.join(0.005)

    assert worker.is_alive()
    assert hardware.calls == []
    source.state = AutoFireTargetAssociation("reacquisition_timeout")
    auto.notify_target_state_changed()
    worker.join(1.0)
    assert not worker.is_alive() and hardware.calls == []


@pytest.mark.parametrize(
    ("state", "reason"),
    [
        ("reacquisition_timeout", "target_reacquisition_timeout"),
        ("reacquisition_incompatible", "target_reacquisition_incompatible"),
        ("reacquisition_ambiguous", "target_reacquisition_ambiguous"),
    ],
)
def test_terminal_reacquisition_state_fails_closed(
    tmp_path: Path,
    state: str,
    reason: str,
) -> None:
    clocks = Clocks()
    source = AssociationSource(AutoFireTargetAssociation("coasting", 100.9))
    hardware = FakeCoordinator()
    auto = AutoFireService(
        config(tmp_path), hardware, source, lambda: False,
        monotonic_clock=clocks.monotonic_now, wall_clock=clocks.wall_now,
    )
    decisions: list[object] = []
    worker = threading.Thread(target=lambda: decisions.append(classify(auto)))
    worker.start()
    for _ in range(100):
        if auto.status()["classifications_held_for_reacquisition"] == 1:
            break
        worker.join(0.005)
    source.state = AutoFireTargetAssociation(state)
    auto.notify_target_state_changed()
    worker.join(1.0)

    assert not worker.is_alive()
    assert decisions[0].reason == reason  # type: ignore[union-attr]
    assert hardware.calls == []


def test_reacquisition_does_not_revive_stale_classification_or_target(tmp_path: Path) -> None:
    clocks = Clocks()
    source = AssociationSource(AutoFireTargetAssociation("coasting", 100.9))
    hardware = FakeCoordinator()
    auto = AutoFireService(
        config(tmp_path), hardware, source, lambda: False,
        monotonic_clock=clocks.monotonic_now, wall_clock=clocks.wall_now,
    )
    decisions: list[object] = []
    worker = threading.Thread(target=lambda: decisions.append(classify(auto, observed=97.2)))
    worker.start()
    for _ in range(100):
        if auto.status()["classifications_held_for_reacquisition"] == 1:
            break
        worker.join(0.005)
    clocks.monotonic = 100.7
    source.state = target(observed=100.7)
    auto.notify_target_state_changed()
    worker.join(1.0)
    assert decisions[0].reason == "stale_classification"  # type: ignore[union-attr]

    clocks.monotonic = 200.0
    clocks.wall = 10_100.0
    source.state = AutoFireTargetAssociation("coasting", 200.9)
    second: list[object] = []
    worker = threading.Thread(target=lambda: second.append(classify(auto, event_id="event-two", track_id=8, observed=200.0)))
    worker.start()
    for _ in range(100):
        if auto.status()["classifications_held_for_reacquisition"] == 1:
            break
        worker.join(0.005)
    clocks.monotonic = 200.2
    source.state = target(event_id="event-two", track_id=8, observed=199.0)
    auto.notify_target_state_changed()
    worker.join(1.0)
    assert second[0].reason == "stale_target"  # type: ignore[union-attr]
    assert hardware.calls == []


def test_person_veto_cancels_classification_held_during_dropout(tmp_path: Path) -> None:
    clocks = Clocks()
    source = AssociationSource(AutoFireTargetAssociation("coasting", 100.9))
    hardware = FakeCoordinator()
    auto = AutoFireService(
        config(tmp_path), hardware, source, lambda: False,
        monotonic_clock=clocks.monotonic_now, wall_clock=clocks.wall_now,
    )
    decisions: list[object] = []
    worker = threading.Thread(target=lambda: decisions.append(classify(auto)))
    worker.start()
    for _ in range(100):
        if auto.status()["classifications_held_for_reacquisition"] == 1:
            break
        worker.join(0.005)

    veto = classify(auto, detections=(AutoFireDetection("person", 0.99),))
    worker.join(1.0)
    assert veto.reason == "human_detected"
    assert decisions[0].reason == "human_detected"  # type: ignore[union-attr]
    assert hardware.calls == []


def test_disabled_service_never_calls_coordinator(tmp_path: Path) -> None:
    clock = Clocks()
    source = TargetSource(target())
    hardware = FakeCoordinator()
    auto = AutoFireService(
        AutoFireConfig(rate_limit_state_file=tmp_path / "state.json"),
        hardware,
        source,
        lambda: False,
        monotonic_clock=clock.monotonic_now,
        wall_clock=clock.wall_now,
    )

    decision = classify(auto)

    assert decision.accepted is False and decision.reason == "feature_disabled"
    assert hardware.calls == []


def test_auto_fire_service_adds_no_worker_or_polling_lifecycle(tmp_path: Path) -> None:
    auto, _, _, _, _ = service(tmp_path)

    assert not hasattr(auto, "start")
    assert not hasattr(auto, "stop")
    assert not any(isinstance(value, threading.Thread) for value in vars(auto).values())


def test_shutdown_gate_rejects_new_and_pending_final_checks(tmp_path: Path) -> None:
    hardware = FakeCoordinator()
    auto, _, _, _, hardware = service(tmp_path, coordinator=hardware)
    hardware.before_final = auto.begin_shutdown

    pending = classify(auto)
    after = classify(auto, event_id="event-two", track_id=8)

    assert pending.reason == "service_stopping"
    assert after.reason == "service_stopping"
    auto.start_accepting()
    assert auto.status()["shutting_down"] is False


def test_allowlisted_fresh_target_engages_atomically_and_persists(tmp_path: Path) -> None:
    auto, _, _, _, hardware = service(tmp_path)

    decision = classify(auto)

    assert decision.accepted is True and decision.reason == "accepted"
    assert hardware.calls[0]["pixel"] == (50, 50)
    assert hardware.calls[0]["frame_size"] == (1280, 720)
    assert hardware.calls[0]["cooldown_seconds"] == 5.0
    evidence = hardware.calls[0]["evidence"]
    assert isinstance(evidence, dict)
    assert evidence["event_type"] == "auto_fire"
    assert evidence["source_event_id"] == "event-one" and evidence["track_id"] == 7
    assert evidence["classifier_label"] == "dog" and evidence["classifier_confidence"] == 0.90
    payload = json.loads((tmp_path / "auto-fire-state.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["clock"]["wall_epoch_seconds"] == 10000.0
    assert payload["clock"]["monotonic_seconds"] == 100.0
    assert set(payload["clock"]) == {"boot_id", "monotonic_seconds", "wall_epoch_seconds"}
    assert payload["shots"] == [
        {"accepted_at_epoch_seconds": 10000.0, "event_id": "event-one", "track_id": 7}
    ]
    assert not list(tmp_path.glob(".*.tmp"))
    status = auto.status()
    assert status["accepted"] == 1 and status["shots_in_rolling_window"] == 1
    assert "shots" not in status


@pytest.mark.parametrize(
    ("label", "confidence", "accepted", "reason"),
    [
        ("dog", 0.70, False, "below_confidence"),
        ("cat", 0.70, False, "below_confidence"),
        ("bird", 0.70, False, "below_confidence"),
        ("dog", 0.7001, True, "accepted"),
        ("cat", 0.7001, True, "accepted"),
        ("bird", 0.7001, True, "accepted"),
    ],
)
def test_allowlisted_confidence_must_be_strictly_greater_than_70_percent(
    tmp_path: Path,
    label: str,
    confidence: float,
    accepted: bool,
    reason: str,
) -> None:
    auto, _, _, _, hardware = service(
        tmp_path,
        config_changes={
            "min_confidence": 0.70,
            "allowed_classes": ("dog", "cat", "bird"),
        },
    )

    decision = classify(auto, detections=(AutoFireDetection(label, confidence),))

    assert decision.accepted is accepted
    assert decision.reason == reason
    assert len(hardware.calls) == int(accepted)


@pytest.mark.parametrize(
    ("detections", "error", "reason"),
    [
        (None, None, "classification_missing"),
        ((), None, "classification_unknown"),
        ((AutoFireDetection("unknown", 0.95),), None, "classification_unknown"),
        ((AutoFireDetection("dog", 0.69),), None, "below_confidence"),
        ((AutoFireDetection("cat", 0.95),), None, "class_not_allowlisted"),
        ((AutoFireDetection("squirrel", 0.95),), None, "classification_invalid"),
        ((object(),), None, "classification_invalid"),
        ((AutoFireDetection("dog", 0.95),), "inference failed", "classification_error"),
    ],
)
def test_ambiguous_or_disallowed_classifications_fail_closed(
    tmp_path: Path,
    detections: object,
    error: str | None,
    reason: str,
) -> None:
    auto, _, _, _, hardware = service(tmp_path)

    decision = classify(auto, detections=detections, error=error)

    assert decision.accepted is False and decision.reason == reason
    assert hardware.calls == []


def test_highest_confidence_result_must_itself_be_allowlisted(tmp_path: Path) -> None:
    auto, _, _, _, hardware = service(tmp_path)

    decision = classify(
        auto,
        detections=(AutoFireDetection("cat", 0.91), AutoFireDetection("dog", 0.90)),
    )

    assert decision.reason == "class_not_allowlisted"
    assert hardware.calls == []


def test_100632_top_bird_ignores_secondary_person_for_class_policy(tmp_path: Path) -> None:
    event_id = "20260819-100632-544-762eb7"
    auto, _, _, _, hardware = service(
        tmp_path,
        config_changes={"min_confidence": 0.70},
        current_target=target(event_id=event_id),
    )

    decision = classify(
        auto,
        event_id=event_id,
        detections=(
            AutoFireDetection("bird", 0.71297),
            AutoFireDetection("person", 0.28157),
        ),
    )

    assert decision.accepted is True and decision.reason == "accepted"
    assert hardware.calls


def test_top_person_latches_event_and_overrides_later_animal(tmp_path: Path) -> None:
    auto, _, _, _, hardware = service(tmp_path)

    first = classify(
        auto,
        detections=(AutoFireDetection("person", 0.90), AutoFireDetection("bird", 0.80)),
    )
    second = classify(auto, detections=(AutoFireDetection("dog", 0.99),))

    assert first.reason == second.reason == "human_detected"
    assert hardware.calls == []


def test_below_threshold_animal_with_secondary_person_stays_below_confidence(
    tmp_path: Path,
) -> None:
    auto, _, _, _, hardware = service(
        tmp_path,
        config_changes={"min_confidence": 0.70},
    )

    decision = classify(
        auto,
        detections=(AutoFireDetection("bird", 0.65), AutoFireDetection("person", 0.30)),
    )

    assert decision.accepted is False and decision.reason == "below_confidence"
    assert hardware.calls == []


@pytest.mark.parametrize(
    ("snapshot", "classified_at", "night", "reason"),
    [
        (target(observed=95.0), 99.5, False, "stale_target"),
        (target(confirmed=False), 99.5, False, "target_not_confirmed"),
        (target(eligible=False), 99.5, False, "target_not_event_eligible"),
        (target(category="person_sized"), 99.5, False, "person_sized_target"),
        (target(event_id="different-event"), 99.5, False, "target_association_invalid"),
        (target(), 96.0, False, "stale_classification"),
        (target(), 99.5, True, "night_mode"),
    ],
)
def test_target_association_persistence_freshness_and_daylight_are_required(
    tmp_path: Path,
    snapshot: AutoFireTargetSnapshot,
    classified_at: float,
    night: bool,
    reason: str,
) -> None:
    auto, _, _, _, hardware = service(tmp_path, current_target=snapshot, night=[night])

    decision = classify(auto, observed=classified_at)

    assert decision.accepted is False and decision.reason == reason
    assert hardware.calls == []


def test_final_check_rejects_target_that_moved_out_of_aimed_pixel(tmp_path: Path) -> None:
    hardware = FakeCoordinator()
    auto, source, _, clocks, _ = service(tmp_path, coordinator=hardware)

    def move_target() -> None:
        clocks.monotonic = 100.2
        source.target = target(observed=100.2, pixel=(110, 110), box=(100, 100, 20, 20))

    hardware.before_final = move_target

    decision = classify(auto)

    assert decision.accepted is False and decision.reason == "target_moved_during_aim"
    state = json.loads((tmp_path / "auto-fire-state.json").read_text(encoding="utf-8"))
    assert state["shots"] == []


def test_final_check_rejects_live_camera_geometry_change(tmp_path: Path) -> None:
    hardware = FakeCoordinator()
    auto, source, _, clocks, _ = service(tmp_path, coordinator=hardware)

    def change_frame_geometry() -> None:
        clocks.monotonic = 100.2
        source.target = target(observed=100.2, frame_size=(640, 480))

    hardware.before_final = change_frame_geometry

    decision = classify(auto)

    assert decision.accepted is False and decision.reason == "calibration_frame_mismatch"
    state = json.loads((tmp_path / "auto-fire-state.json").read_text(encoding="utf-8"))
    assert state["shots"] == []


def test_final_check_rechecks_night_and_human_latch(tmp_path: Path) -> None:
    hardware = FakeCoordinator()
    auto, _, night, clocks, _ = service(tmp_path, coordinator=hardware)

    def night_falls() -> None:
        clocks.monotonic = 100.1
        night[0] = True

    hardware.before_final = night_falls
    assert classify(auto).reason == "night_mode"

    second_hardware = FakeCoordinator()
    second, _, _, _, _ = service(tmp_path / "second", coordinator=second_hardware)

    def person_appears() -> None:
        human = classify(second, detections=(AutoFireDetection("person", 0.30),))
        assert human.reason == "human_detected"

    second_hardware.before_final = person_appears
    assert classify(second).reason == "human_detected"


def test_coordinator_must_invoke_final_check_and_reasoned_errors_are_preserved(tmp_path: Path) -> None:
    hardware = FakeCoordinator()
    hardware.skip_final = True
    auto, _, _, _, _ = service(tmp_path, coordinator=hardware)

    first = classify(auto)
    second = classify(auto, event_id="event-two", track_id=8)

    assert first.reason == "safety_state_invalid"
    assert second.reason == "safety_state_invalid"
    assert auto.status()["persistence"]["healthy"] is False  # type: ignore[index]

    reasoned = FakeCoordinator()
    reasoned.raise_error = CoordinatorError("coordinator_busy")
    reasoned_service, _, _, _, _ = service(tmp_path / "reasoned", coordinator=reasoned)
    assert classify(reasoned_service).reason == "coordinator_busy"

    generic = FakeCoordinator()
    generic.raise_error = RuntimeError("boom")
    generic_service, _, _, _, _ = service(tmp_path / "generic", coordinator=generic)
    assert classify(generic_service).reason == "hardware_not_ready"


def test_failed_post_actuation_attempt_is_rate_limited_and_blocks_service(tmp_path: Path) -> None:
    hardware = FakeCoordinator()
    hardware.raise_error = CoordinatorError("park_failed", shot_attempted=True)
    auto, _, _, _, _ = service(tmp_path, coordinator=hardware)

    first = classify(auto)
    second = classify(auto, event_id="event-two", track_id=8)

    assert first.accepted is False and first.reason == "park_failed"
    assert second.reason == "safety_state_invalid"
    persisted = json.loads((tmp_path / "auto-fire-state.json").read_text(encoding="utf-8"))
    assert persisted["shots"] == [
        {"accepted_at_epoch_seconds": 10000.0, "event_id": "event-one", "track_id": 7}
    ]
    status = auto.status()
    assert status["accepted"] == 0
    assert status["rejected"] == 2
    assert status["shots_in_rolling_window"] == 1
    assert status["persistence"]["healthy"] is False  # type: ignore[index]


def test_successful_shot_with_recording_queue_failure_is_counted_then_blocks(
    tmp_path: Path,
) -> None:
    hardware = FakeCoordinator()
    hardware.result = CoordinatorResult(fired=True, recording_queued=False)
    auto, _, _, _, _ = service(tmp_path, coordinator=hardware)

    first = classify(auto)
    second = classify(auto, event_id="event-two", track_id=8)

    assert first.accepted is True and first.reason == "accepted_recording_failed"
    assert second.reason == "safety_state_invalid"
    status = auto.status()
    assert status["accepted"] == 1
    assert status["shots_in_rolling_window"] == 1
    assert "could not queue" in status["persistence"]["error"]  # type: ignore[index,operator]


def test_backend_cooldown_is_checked_before_engagement(tmp_path: Path) -> None:
    hardware = FakeCoordinator()
    hardware.cooldown = 2.25
    auto, _, _, _, _ = service(tmp_path, coordinator=hardware)

    decision = classify(auto)

    assert decision.reason == "cooldown_active"
    assert hardware.calls == []
    assert auto.status()["cooldown_remaining_seconds"] == 2.25


def test_per_event_and_minimum_reengagement_limits_are_persisted(tmp_path: Path) -> None:
    clocks = Clocks()
    auto, source, _, _, hardware = service(
        tmp_path,
        clocks=clocks,
        config_changes={"max_shots_per_event": 2, "minimum_reengagement_seconds": 60.0},
    )
    assert classify(auto).accepted is True

    clocks.monotonic = 130.0
    clocks.wall = 10_030.0
    source.target = target(observed=130.0)
    assert classify(auto, observed=130.0).reason == "minimum_reengagement_delay"

    clocks.monotonic = 161.0
    clocks.wall = 10_061.0
    source.target = target(observed=161.0)
    assert classify(auto, observed=161.0).accepted is True

    clocks.monotonic = 222.0
    clocks.wall = 10_122.0
    source.target = target(observed=222.0)
    assert classify(auto, observed=222.0).reason == "target_already_engaged"
    assert len(hardware.calls) == 2

    restarted, _, _, _, restarted_hardware = service(
        tmp_path,
        clocks=clocks,
        current_target=target(observed=222.0),
        config_changes={"max_shots_per_event": 2, "minimum_reengagement_seconds": 60.0},
    )
    assert classify(restarted, observed=222.0).reason == "target_already_engaged"
    assert restarted_hardware.calls == []


def test_minimum_reengagement_delay_follows_same_track_across_event_ids(tmp_path: Path) -> None:
    clocks = Clocks()
    auto, source, _, _, hardware = service(tmp_path, clocks=clocks)
    assert classify(auto).accepted is True

    clocks.monotonic += 10
    clocks.wall += 10
    source.target = target(event_id="event-two", track_id=7, observed=clocks.monotonic)
    blocked = classify(
        auto,
        event_id="event-two",
        track_id=7,
        observed=clocks.monotonic,
    )

    assert blocked.reason == "minimum_reengagement_delay"
    assert len(hardware.calls) == 1


def test_global_rolling_limit_survives_restart_and_expires_by_wall_time(tmp_path: Path) -> None:
    clocks = Clocks()
    auto, source, _, _, hardware = service(
        tmp_path,
        clocks=clocks,
        config_changes={"max_shots_per_hour": 2},
    )

    assert classify(auto).accepted is True
    clocks.monotonic += 1
    clocks.wall += 1
    source.target = target(event_id="event-two", track_id=8, observed=clocks.monotonic)
    assert classify(auto, event_id="event-two", track_id=8, observed=clocks.monotonic).accepted is True

    clocks.monotonic += 1
    clocks.wall += 1
    source.target = target(event_id="event-three", track_id=9, observed=clocks.monotonic)
    assert classify(auto, event_id="event-three", track_id=9, observed=clocks.monotonic).reason == "global_rate_limit"
    assert len(hardware.calls) == 2

    restarted, restarted_source, _, _, restarted_hardware = service(
        tmp_path,
        clocks=clocks,
        current_target=target(event_id="event-four", track_id=10, observed=clocks.monotonic),
        config_changes={"max_shots_per_hour": 2},
    )
    assert classify(restarted, event_id="event-four", track_id=10, observed=clocks.monotonic).reason == "global_rate_limit"
    assert restarted_hardware.calls == []

    clocks.monotonic += 3601
    clocks.wall += 3601
    restarted_source.target = target(event_id="event-four", track_id=10, observed=clocks.monotonic)
    assert classify(restarted, event_id="event-four", track_id=10, observed=clocks.monotonic).accepted is True


def test_persisted_history_survives_forward_clock_jump_and_correction(tmp_path: Path) -> None:
    original, _, _, _, _ = service(tmp_path)
    assert classify(original).accepted is True
    before = json.loads((tmp_path / "auto-fire-state.json").read_text(encoding="utf-8"))

    jumped_clock = Clocks(monotonic=100.0, wall=100_000.0)
    restarted, source, _, _, hardware = service(tmp_path, clocks=jumped_clock)
    jumped_clock.monotonic = 130.0
    jumped_clock.wall = 10_030.0
    source.target = target(event_id="event-two", track_id=8, observed=130.0)

    decision = classify(restarted, event_id="event-two", track_id=8, observed=130.0)

    assert decision.reason == "safety_state_invalid"
    assert hardware.calls == []
    after = json.loads((tmp_path / "auto-fire-state.json").read_text(encoding="utf-8"))
    assert after == before
    assert after["shots"] == [
        {"accepted_at_epoch_seconds": 10000.0, "event_id": "event-one", "track_id": 7}
    ]
    assert "clock" in restarted.status()["persistence"]["error"]  # type: ignore[index,operator]


def test_runtime_forward_wall_clock_jump_fails_closed_without_a_second_call(tmp_path: Path) -> None:
    clocks = Clocks()
    auto, source, _, _, hardware = service(tmp_path, clocks=clocks)
    assert classify(auto).accepted is True

    clocks.monotonic = 101.0
    clocks.wall = 100_000.0
    source.target = target(event_id="event-two", track_id=8, observed=101.0)
    decision = classify(auto, event_id="event-two", track_id=8, observed=101.0)

    assert decision.reason == "safety_state_invalid"
    assert len(hardware.calls) == 1
    persisted = json.loads((tmp_path / "auto-fire-state.json").read_text(encoding="utf-8"))
    assert persisted["shots"] == [
        {"accepted_at_epoch_seconds": 10000.0, "event_id": "event-one", "track_id": 7}
    ]


def test_runtime_forward_time_sync_rebases_when_shot_history_is_empty(tmp_path: Path) -> None:
    clocks = Clocks()
    auto, source, _, _, hardware = service(tmp_path, clocks=clocks)

    clocks.monotonic = 101.0
    clocks.wall = 13_100.0
    source.target = target(observed=101.0)
    decision = classify(auto, observed=101.0)

    assert decision.accepted is True
    assert len(hardware.calls) == 1
    assert auto.status()["persistence"]["healthy"] is True  # type: ignore[index]
    persisted = json.loads((tmp_path / "auto-fire-state.json").read_text(encoding="utf-8"))
    assert persisted["clock"]["monotonic_seconds"] == 101.0
    assert persisted["clock"]["wall_epoch_seconds"] == 13_100.0
    assert persisted["shots"] == [
        {"accepted_at_epoch_seconds": 13_100.0, "event_id": "event-one", "track_id": 7}
    ]


def test_same_boot_restart_rebases_empty_history_after_forward_time_sync(tmp_path: Path) -> None:
    clocks = Clocks()
    service(tmp_path, clocks=clocks)

    clocks.monotonic = 101.0
    clocks.wall = 13_100.0
    restarted, source, _, _, hardware = service(tmp_path, clocks=clocks)
    source.target = target(observed=101.0)
    decision = classify(restarted, observed=101.0)

    assert decision.accepted is True
    assert len(hardware.calls) == 1
    assert restarted.status()["persistence"]["healthy"] is True  # type: ignore[index]


def test_new_boot_rebases_empty_history_before_time_sync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boot_id = ["boot-one"]
    monkeypatch.setattr(
        AutoFireService,
        "_system_boot_id",
        staticmethod(lambda: boot_id[0]),
    )
    clocks = Clocks()
    service(tmp_path, clocks=clocks)

    boot_id[0] = "boot-two"
    clocks.monotonic = 1.0
    clocks.wall = 9_900.0
    restarted, source, _, _, hardware = service(tmp_path, clocks=clocks)
    source.target = target(observed=1.0)
    decision = classify(restarted, observed=1.0)

    assert decision.accepted is True
    assert len(hardware.calls) == 1
    persisted = json.loads((tmp_path / "auto-fire-state.json").read_text(encoding="utf-8"))
    assert persisted["clock"]["boot_id"] == "boot-two"
    assert persisted["clock"]["monotonic_seconds"] == 1.0
    assert persisted["clock"]["wall_epoch_seconds"] == 9_900.0


def test_runtime_backward_time_sync_rebases_when_shot_history_is_empty(tmp_path: Path) -> None:
    clocks = Clocks()
    auto, source, _, _, hardware = service(tmp_path, clocks=clocks)

    clocks.monotonic = 101.0
    clocks.wall = 9_900.0
    source.target = target(observed=101.0)
    decision = classify(auto, observed=101.0)

    assert decision.accepted is True
    assert len(hardware.calls) == 1
    assert auto.status()["persistence"]["healthy"] is True  # type: ignore[index]


def test_runtime_backward_clock_jump_still_blocks_with_shot_history(tmp_path: Path) -> None:
    clocks = Clocks()
    auto, source, _, _, hardware = service(tmp_path, clocks=clocks)
    assert classify(auto).accepted is True

    clocks.monotonic = 101.0
    clocks.wall = 9_900.0
    source.target = target(event_id="event-two", track_id=8, observed=101.0)
    decision = classify(auto, event_id="event-two", track_id=8, observed=101.0)

    assert decision.reason == "safety_state_invalid"
    assert len(hardware.calls) == 1
    assert auto.status()["persistence"]["healthy"] is False  # type: ignore[index]


def test_corrupt_or_unreadable_rate_state_disables_engagement(tmp_path: Path) -> None:
    state_path = tmp_path / "auto-fire-state.json"
    state_path.write_text("not json", encoding="utf-8")

    auto, _, _, _, hardware = service(tmp_path)
    decision = classify(auto)

    assert decision.reason == "safety_state_invalid"
    assert hardware.calls == []
    persistence = auto.status()["persistence"]
    assert persistence["healthy"] is False  # type: ignore[index]
    assert "Could not read" in persistence["error"]  # type: ignore[index,operator]


def test_unwritable_rate_state_fails_before_coordinator_is_called(tmp_path: Path) -> None:
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("occupied", encoding="utf-8")
    hardware = FakeCoordinator()
    auto, _, _, _, _ = service(
        tmp_path,
        coordinator=hardware,
        config_changes={"rate_limit_state_file": blocked_parent / "state.json"},
    )

    decision = classify(auto)

    assert decision.reason == "safety_state_invalid"
    assert hardware.calls == []
    persistence = auto.status()["persistence"]
    assert persistence["healthy"] is False  # type: ignore[index]
    assert "Could not initialize" in persistence["error"]  # type: ignore[index,operator]


def test_rejection_status_is_bounded_and_contains_reason_counts(tmp_path: Path) -> None:
    auto, _, _, _, _ = service(tmp_path)
    assert classify(auto, detections=None).reason == "classification_missing"
    assert classify(auto, detections=(AutoFireDetection("dog", 0.5),)).reason == "below_confidence"

    status = auto.status()

    assert status["candidates_evaluated"] == 2
    assert status["rejected"] == 2
    assert status["rejection_counts"] == {"below_confidence": 1, "classification_missing": 1}
    assert status["last_decision"]["reason"] == "below_confidence"  # type: ignore[index]
    assert len(json.dumps(status)) < 2048
