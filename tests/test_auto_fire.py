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
from squirrel_shooter.safety import (
    FinalAimDecision,
    SceneDetection,
    ScenePersonSafetyResult,
)


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
    reservation_id: str | None = None
    event_id: str = "auto-fire-test"
    shot_attempted: bool = True


class CoordinatorError(RuntimeError):
    def __init__(self, reason: str | None = None, *, shot_attempted: bool = False) -> None:
        self.reason = reason
        self.shot_attempted = shot_attempted
        super().__init__(reason or "coordinator failed")


class FakeSceneSafetyProvider:
    def __init__(
        self,
        clocks: Clocks,
        target_source: Callable[
            [str, int],
            AutoFireTargetSnapshot | AutoFireTargetAssociation | None,
        ],
    ) -> None:
        self.clocks = clocks
        self.target_source = target_source
        self.calls: list[dict[str, object]] = []
        self.responses: list[
            Callable[[str, str, int, int | None], ScenePersonSafetyResult]
        ] = []
        self.latest: ScenePersonSafetyResult | None = None

    def check_current_scene(
        self,
        *,
        request_id: str,
        event_id: str,
        track_id: int,
        after_sequence: int | None,
        timeout_seconds: float,
    ) -> ScenePersonSafetyResult:
        self.calls.append(
            {
                "request_id": request_id,
                "event_id": event_id,
                "track_id": track_id,
                "after_sequence": after_sequence,
                "timeout_seconds": timeout_seconds,
            }
        )
        if self.responses:
            result = self.responses.pop(0)(request_id, event_id, track_id, after_sequence)
        else:
            current = self.target_source(event_id, track_id)
            source_time = (
                current.observed_monotonic
                if isinstance(current, AutoFireTargetSnapshot)
                else self.clocks.monotonic
            )
            source_sequence = (after_sequence if after_sequence is not None else 0) + 1
            result = ScenePersonSafetyResult(
                status="clear",
                request_id=request_id,
                event_id=event_id,
                track_id=track_id,
                coordinate_space="native_full_frame",
                source_sequence=source_sequence,
                source_received_monotonic=source_time,
                frame_width=1280,
                frame_height=720,
                detections=(),
                completed_monotonic=max(self.clocks.monotonic, source_time),
            )
        self.latest = result
        return result

    def latest_scene_result(self) -> ScenePersonSafetyResult | None:
        return self.latest


class FakeCoordinator:
    def __init__(self) -> None:
        self.cooldown = 0.0
        self.calls: list[dict[str, object]] = []
        self.before_final: Callable[[], None] | None = None
        self.after_reservation: Callable[[str], None] | None = None
        self.raise_error: Exception | None = None
        self.skip_final = False
        self.result: CoordinatorResult | None = None
        self.aim_decisions: list[FinalAimDecision] = []
        self.before_second_final: Callable[[], None] | None = None

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
        final_safety_check: Callable[[], FinalAimDecision],
        reserve_actuation: Callable[[], str],
        post_reservation_safety_check: Callable[[], bool],
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
        if self.raise_error is not None and getattr(self.raise_error, "shot_attempted", False) is not True:
            raise self.raise_error
        if self.skip_final:
            return CoordinatorResult()
        final_decision = final_safety_check()
        self.aim_decisions.append(final_decision)
        if final_decision.action == "reaim":
            if self.before_second_final is not None:
                self.before_second_final()
            final_decision = final_safety_check()
            self.aim_decisions.append(final_decision)
        if final_decision.action != "accept":
            return CoordinatorResult(fired=False, shot_attempted=False)
        reservation_id = reserve_actuation()
        if self.after_reservation is not None:
            self.after_reservation(reservation_id)
        if post_reservation_safety_check() is not True:
            return CoordinatorResult(
                fired=False,
                reservation_id=reservation_id,
                shot_attempted=False,
            )
        if self.raise_error is not None:
            raise self.raise_error
        if self.result is not None:
            self.result.reservation_id = reservation_id
            return self.result
        return CoordinatorResult(reservation_id=reservation_id)


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
    frame_sequence: int | None = None,
    velocity: tuple[float, float] = (0.0, 0.0),
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
        frame_sequence,
        velocity,
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
    scene_provider: FakeSceneSafetyProvider | None = None,
) -> tuple[AutoFireService, TargetSource, list[bool], Clocks, FakeCoordinator]:
    clock = clocks or Clocks()
    source = TargetSource(current_target or target())
    night_state = night or [False]
    hardware = coordinator or FakeCoordinator()
    safety = scene_provider or FakeSceneSafetyProvider(clock, source)
    auto = AutoFireService(
        config(tmp_path, **(config_changes or {})),
        hardware,
        source,
        lambda: night_state[0],
        scene_safety_provider=safety,
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
    frame_sequence: int | None = None,
):
    return auto.handle_classification(
        event_id=event_id,
        track_id=track_id,
        classified_observation_monotonic=observed,
        classified_frame_sequence=frame_sequence,
        detections=detections,  # type: ignore[arg-type]
        error=error,
    )


def scene_result(
    request_id: str,
    event_id: str,
    track_id: int,
    after_sequence: int | None,
    *,
    status: str = "clear",
    source_time: float = 99.5,
    detections: tuple[SceneDetection, ...] = (),
    request_override: str | None = None,
    event_override: str | None = None,
    track_override: int | None = None,
) -> ScenePersonSafetyResult:
    has_source = status not in {"error", "unavailable"}
    return ScenePersonSafetyResult(
        status=status,  # type: ignore[arg-type]
        request_id=request_override or request_id,
        event_id=event_override or event_id,
        track_id=track_override or track_id,
        coordinate_space="native_full_frame",
        source_sequence=((after_sequence or 0) + 1) if has_source else None,
        source_received_monotonic=source_time if has_source else None,
        frame_width=1280 if has_source else None,
        frame_height=720 if has_source else None,
        detections=detections,
        completed_monotonic=max(100.0, source_time),
        error=(f"synthetic_{status}" if status in {"error", "unavailable", "ambiguous"} else None),
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


def test_153131_person_outside_animal_crop_is_a_full_scene_veto(tmp_path: Path) -> None:
    fixture = json.loads(
        Path("tests/fixtures/field_events/20260818-153131-300-99a55b.json").read_text(
            encoding="utf-8"
        )
    )
    clocks = Clocks()
    provider = FakeSceneSafetyProvider(clocks, TargetSource(target()))
    person = SceneDetection("person", 0.88, (900, 40, 220, 640))
    provider.responses.append(
        lambda request, event, track_id, after: scene_result(
            request,
            event,
            track_id,
            after,
            status="person",
            detections=(person,),
        )
    )
    auto, _, _, _, hardware = service(
        tmp_path,
        clocks=clocks,
        scene_provider=provider,
        current_target=target(event_id=fixture["event_id"]),
    )

    decision = classify(
        auto,
        event_id=fixture["event_id"],
        detections=(AutoFireDetection("bird", 0.82),),
    )

    assert fixture["classification"]["full_frame_contains_person"] is True
    assert decision.reason == "human_detected"
    assert hardware.calls and hardware.aim_decisions == [FinalAimDecision.reject("human_detected")]


def test_person_partially_overlapping_crop_is_a_full_scene_veto(tmp_path: Path) -> None:
    clocks = Clocks()
    provider = FakeSceneSafetyProvider(clocks, TargetSource(target()))
    provider.responses.append(
        lambda request, event, track_id, after: scene_result(
            request,
            event,
            track_id,
            after,
            status="person",
            detections=(SceneDetection("person", 0.91, (30, 20, 60, 180)),),
        )
    )
    auto, _, _, _, hardware = service(tmp_path, clocks=clocks, scene_provider=provider)

    decision = classify(auto)

    assert decision.reason == "human_detected"
    assert all(call.action != "accept" for call in hardware.aim_decisions)


def test_person_entering_after_classification_during_aim_vetoes_final_scene(tmp_path: Path) -> None:
    clocks = Clocks()
    provider = FakeSceneSafetyProvider(clocks, TargetSource(target()))
    person_entered = [False]

    def current_scene(request: str, event: str, track_id: int, after: int | None) -> ScenePersonSafetyResult:
        assert person_entered[0] is True
        return scene_result(
            request,
            event,
            track_id,
            after,
            status="person",
            detections=(SceneDetection("person", 0.94, (500, 50, 180, 600)),),
        )

    provider.responses.append(current_scene)
    hardware = FakeCoordinator()
    hardware.before_final = lambda: person_entered.__setitem__(0, True)
    auto, _, _, _, _ = service(
        tmp_path,
        clocks=clocks,
        coordinator=hardware,
        scene_provider=provider,
    )

    decision = classify(auto)

    assert decision.reason == "human_detected"
    assert len(provider.calls) == 1


@pytest.mark.parametrize(
    ("status", "reason", "source_time"),
    [
        ("clear", "stale_scene_safety", 98.0),
        ("ambiguous", "scene_safety_ambiguous", 99.5),
        ("error", "scene_safety_error", 99.5),
        ("unavailable", "scene_safety_unavailable", 99.5),
    ],
)
def test_non_current_or_uncertain_full_scene_result_never_fires(
    tmp_path: Path,
    status: str,
    reason: str,
    source_time: float,
) -> None:
    clocks = Clocks()
    provider = FakeSceneSafetyProvider(clocks, TargetSource(target()))
    provider.responses.append(
        lambda request, event, track_id, after: scene_result(
            request,
            event,
            track_id,
            after,
            status=status,
            source_time=source_time,
        )
    )
    auto, _, _, _, hardware = service(tmp_path, clocks=clocks, scene_provider=provider)

    decision = classify(auto)

    assert decision.reason == reason
    assert hardware.calls and hardware.calls[0]["pixel"] == (50, 50)
    assert hardware.aim_decisions[0].action == "reject"


def test_scene_result_completed_in_the_future_cannot_authorize(tmp_path: Path) -> None:
    clocks = Clocks()
    provider = FakeSceneSafetyProvider(clocks, TargetSource(target()))

    def future_completion(
        request: str,
        event: str,
        track_id: int,
        after: int | None,
    ) -> ScenePersonSafetyResult:
        result = scene_result(request, event, track_id, after)
        object.__setattr__(result, "completed_monotonic", 100.1)
        return result

    provider.responses.append(future_completion)
    auto, _, _, _, _ = service(tmp_path, clocks=clocks, scene_provider=provider)

    assert classify(auto).reason == "stale_scene_safety"


def test_bypass_mutated_scene_detections_fail_closed_without_exception(tmp_path: Path) -> None:
    clocks = Clocks()
    provider = FakeSceneSafetyProvider(clocks, TargetSource(target()))

    def malformed(
        request: str,
        event: str,
        track_id: int,
        after: int | None,
    ) -> ScenePersonSafetyResult:
        result = scene_result(request, event, track_id, after)
        object.__setattr__(result, "detections", [object()])
        return result

    provider.responses.append(malformed)
    auto, _, _, _, _ = service(tmp_path, clocks=clocks, scene_provider=provider)

    assert classify(auto).reason == "scene_safety_invalid"


def test_bypass_mutated_clear_result_with_error_fails_closed(tmp_path: Path) -> None:
    clocks = Clocks()
    provider = FakeSceneSafetyProvider(clocks, TargetSource(target()))

    def malformed(
        request: str,
        event: str,
        track_id: int,
        after: int | None,
    ) -> ScenePersonSafetyResult:
        result = scene_result(request, event, track_id, after)
        object.__setattr__(result, "error", "partial inference failure")
        return result

    provider.responses.append(malformed)
    auto, _, _, _, _ = service(tmp_path, clocks=clocks, scene_provider=provider)

    assert classify(auto).reason == "scene_safety_invalid"


def test_person_arriving_after_reservation_vetoes_before_pulse_and_latches_event(
    tmp_path: Path,
) -> None:
    clocks = Clocks()
    provider = FakeSceneSafetyProvider(clocks, TargetSource(target()))
    hardware = FakeCoordinator()

    def person_arrives(_reservation_id: str) -> None:
        clear = provider.latest
        assert clear is not None
        provider.latest = ScenePersonSafetyResult(
            status="person",
            request_id=clear.request_id,
            event_id=clear.event_id,
            track_id=clear.track_id,
            coordinate_space="native_full_frame",
            source_sequence=clear.source_sequence,
            source_received_monotonic=clear.source_received_monotonic,
            frame_width=clear.frame_width,
            frame_height=clear.frame_height,
            detections=(SceneDetection("person", 0.97, (600, 20, 200, 650)),),
            completed_monotonic=clear.completed_monotonic,
        )

    hardware.after_reservation = person_arrives
    auto, _, _, _, _ = service(
        tmp_path,
        clocks=clocks,
        coordinator=hardware,
        scene_provider=provider,
    )

    first = classify(auto)
    second = classify(auto)

    assert first.reason == second.reason == "human_detected"
    assert first.accepted is second.accepted is False
    payload = json.loads((tmp_path / "auto-fire-state.json").read_text(encoding="utf-8"))
    assert payload["attempts"][0]["state"] == "cancelled"
    assert payload["attempts"][0]["shot_attempted"] is False


@pytest.mark.parametrize("identity", ["request", "event", "track"])
def test_wrong_scene_request_event_or_track_identity_cannot_authorize(
    tmp_path: Path,
    identity: str,
) -> None:
    clocks = Clocks()
    provider = FakeSceneSafetyProvider(clocks, TargetSource(target()))

    def wrong_identity(
        request: str,
        event: str,
        track_id: int,
        after: int | None,
    ) -> ScenePersonSafetyResult:
        return scene_result(
            request,
            event,
            track_id,
            after,
            request_override="wrong-request" if identity == "request" else None,
            event_override="wrong-event" if identity == "event" else None,
            track_override=99 if identity == "track" else None,
        )

    provider.responses.append(wrong_identity)
    auto, _, _, _, _ = service(tmp_path, clocks=clocks, scene_provider=provider)

    assert classify(auto).reason == "scene_safety_identity_mismatch"


def test_target_observation_older_than_new_scene_times_out_fail_closed(tmp_path: Path) -> None:
    clocks = Clocks()
    current = target(frame_sequence=10)
    provider = FakeSceneSafetyProvider(clocks, TargetSource(current))
    auto, _, _, _, hardware = service(
        tmp_path,
        clocks=clocks,
        current_target=current,
        scene_provider=provider,
        config_changes={"scene_safety_timeout_seconds": 0.02},
    )

    decision = classify(auto, frame_sequence=10)

    assert decision.reason == "target_not_current_with_scene"
    assert hardware.aim_decisions[0].action == "reject"


def test_195417_field_target_gets_one_bounded_reaim_then_accepts(tmp_path: Path) -> None:
    fixture = json.loads(
        Path("tests/fixtures/field_events/20260818-195417-725-0d4269.json").read_text(
            encoding="utf-8"
        )
    )
    previous = fixture["previous"]
    candidate = fixture["candidate"]
    clocks = Clocks(monotonic=previous["observed_monotonic"])
    initial = target(
        event_id=fixture["event_id"],
        track_id=previous["track_id"],
        observed=previous["observed_monotonic"],
        pixel=(round(previous["centroid"][0]), round(previous["centroid"][1])),
        box=tuple(previous["bounding_box"]),
    )
    hardware = FakeCoordinator()
    auto, source, _, _, _ = service(
        tmp_path,
        clocks=clocks,
        coordinator=hardware,
        current_target=initial,
        config_changes={"min_confidence": 0.70},
    )

    def move_to_candidate() -> None:
        clocks.monotonic = candidate["observed_monotonic"]
        source.target = target(
            event_id=fixture["event_id"],
            track_id=candidate["track_id"],
            observed=candidate["observed_monotonic"],
            pixel=(round(candidate["centroid"][0]), round(candidate["centroid"][1])),
            box=tuple(candidate["bounding_box"]),
        )

    hardware.before_final = move_to_candidate
    decision = classify(
        auto,
        event_id=fixture["event_id"],
        track_id=previous["track_id"],
        observed=previous["observed_monotonic"],
        detections=(AutoFireDetection("dog", fixture["classification"]["confidence"]),),
    )

    assert decision.accepted is True
    assert [item.action for item in hardware.aim_decisions] == ["reaim", "accept"]
    assert (decision.target_pixel_x, decision.target_pixel_y) == (
        round(candidate["centroid"][0]),
        round(candidate["centroid"][1]),
    )
    final_evidence = hardware.calls[0]["evidence"]
    assert final_evidence["target_bounding_box"] == {
        "x": candidate["bounding_box"][0],
        "y": candidate["bounding_box"][1],
        "width": candidate["bounding_box"][2],
        "height": candidate["bounding_box"][3],
    }


def test_70_pixel_drift_is_not_directly_accepted_and_second_move_rejects(tmp_path: Path) -> None:
    clocks = Clocks()
    initial = target(pixel=(100, 100), box=(50, 50, 150, 100))
    hardware = FakeCoordinator()
    auto, source, _, _, _ = service(
        tmp_path,
        clocks=clocks,
        coordinator=hardware,
        current_target=initial,
    )

    def first_move() -> None:
        clocks.monotonic = 100.1
        source.target = target(
            observed=100.1,
            pixel=(170, 100),
            box=(120, 50, 150, 100),
        )

    def second_move() -> None:
        clocks.monotonic = 100.2
        source.target = target(
            observed=100.2,
            pixel=(240, 100),
            box=(190, 50, 150, 100),
        )

    hardware.before_final = first_move
    hardware.before_second_final = second_move
    decision = classify(auto)

    assert [item.action for item in hardware.aim_decisions] == ["reaim", "reject"]
    assert decision.reason == "target_moved_after_reaim"
    assert decision.accepted is False
    assert (decision.target_pixel_x, decision.target_pixel_y) == (240, 100)
    assert hardware.calls[0]["evidence"]["target_pixel_x"] == 240
    assert hardware.calls[0]["evidence"]["target_pixel_y"] == 100
    assert hardware.calls[0]["evidence"]["target_bounding_box"] == {
        "x": 190,
        "y": 50,
        "width": 150,
        "height": 100,
    }


def test_brief_dropout_holds_classification_then_uses_reacquired_position(tmp_path: Path) -> None:
    clocks = Clocks()
    source = AssociationSource(AutoFireTargetAssociation("coasting", 100.9))
    hardware = FakeCoordinator()
    auto = AutoFireService(
        config(tmp_path),
        hardware,
        source,
        lambda: False,
        scene_safety_provider=FakeSceneSafetyProvider(clocks, source),
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
        scene_safety_provider=FakeSceneSafetyProvider(clocks, source),
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
        scene_safety_provider=FakeSceneSafetyProvider(clocks, source),
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
        scene_safety_provider=FakeSceneSafetyProvider(clocks, source),
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
        scene_safety_provider=FakeSceneSafetyProvider(clocks, source),
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
    assert payload["schema_version"] == 3
    assert payload["clock"]["wall_epoch_seconds"] == 10000.0
    assert payload["clock"]["monotonic_seconds"] == 100.0
    assert set(payload["clock"]) == {"boot_id", "monotonic_seconds", "wall_epoch_seconds"}
    assert len(payload["attempts"]) == 1
    attempt = payload["attempts"][0]
    assert attempt["event_id"] == "event-one" and attempt["track_id"] == 7
    assert attempt["reserved_at_epoch_seconds"] == 10000.0
    assert attempt["reserved_at_monotonic_seconds"] == 100.0
    assert attempt["state"] == "completed" and attempt["outcome"] == "accepted"
    assert attempt["shot_attempted"] is True
    assert attempt["physical_event_id"] == "auto-fire-test"
    assert not list(tmp_path.glob(".*.tmp"))
    status = auto.status()
    assert status["accepted"] == 1 and status["shots_in_rolling_window"] == 1
    assert "shots" not in status


def test_durable_reservation_is_visible_before_coordinator_completion(tmp_path: Path) -> None:
    hardware = FakeCoordinator()
    observed: dict[str, object] = {}
    auto, _, _, _, _ = service(tmp_path, coordinator=hardware)

    def inspect_reservation(reservation_id: str) -> None:
        payload = json.loads((tmp_path / "auto-fire-state.json").read_text(encoding="utf-8"))
        assert payload["schema_version"] == 3
        assert len(payload["attempts"]) == 1
        attempt = payload["attempts"][0]
        assert attempt["reservation_id"] == reservation_id
        assert attempt["state"] == "reserved"
        assert attempt["shot_attempted"] is None
        observed["reservation_id"] = reservation_id

    hardware.after_reservation = inspect_reservation

    decision = classify(auto)

    assert decision.accepted is True
    assert isinstance(observed["reservation_id"], str)
    finalized = json.loads((tmp_path / "auto-fire-state.json").read_text(encoding="utf-8"))
    assert finalized["attempts"][0]["state"] == "completed"
    assert finalized["attempts"][0]["reservation_id"] == observed["reservation_id"]


def test_reservation_persistence_failure_prevents_coordinator_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hardware = FakeCoordinator()
    completed: list[str] = []
    hardware.after_reservation = completed.append
    auto, _, _, _, _ = service(tmp_path, coordinator=hardware)

    def fail_write(_payload: dict[str, object]) -> None:
        raise OSError("simulated disk failure")

    monkeypatch.setattr(auto, "_write_rate_limit_payload", fail_write)

    decision = classify(auto)

    assert decision.accepted is False and decision.reason == "safety_state_invalid"
    assert completed == []
    status = auto.status()
    assert status["persistence"]["healthy"] is False  # type: ignore[index]
    assert status["shots_in_rolling_window"] == 1
    persisted = json.loads((tmp_path / "auto-fire-state.json").read_text(encoding="utf-8"))
    assert persisted["attempts"] == []


def test_post_write_shutdown_veto_finalizes_same_reservation_without_actuation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auto, _, _, _, _ = service(tmp_path)
    original_write = auto._write_rate_limit_payload
    writes = 0

    def write_then_stop(payload: dict[str, object]) -> None:
        nonlocal writes
        writes += 1
        original_write(payload)
        if writes == 1:
            auto.begin_shutdown()

    monkeypatch.setattr(auto, "_write_rate_limit_payload", write_then_stop)

    decision = classify(auto)

    assert decision.reason == "service_stopping"
    assert decision.accepted is False
    payload = json.loads((tmp_path / "auto-fire-state.json").read_text(encoding="utf-8"))
    assert len(payload["attempts"]) == 1
    assert payload["attempts"][0]["state"] == "cancelled"
    assert payload["attempts"][0]["shot_attempted"] is False


def test_post_write_human_veto_finalizes_same_reservation_without_actuation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auto, _, _, _, _ = service(tmp_path)
    original_write = auto._write_rate_limit_payload
    writes = 0

    def write_then_veto(payload: dict[str, object]) -> None:
        nonlocal writes
        writes += 1
        original_write(payload)
        if writes == 1:
            auto._latch_human("event-one")

    monkeypatch.setattr(auto, "_write_rate_limit_payload", write_then_veto)

    decision = classify(auto)

    assert decision.reason == "human_detected"
    assert decision.accepted is False
    payload = json.loads((tmp_path / "auto-fire-state.json").read_text(encoding="utf-8"))
    assert len(payload["attempts"]) == 1
    assert payload["attempts"][0]["state"] == "cancelled"
    assert payload["attempts"][0]["shot_attempted"] is False


def test_abrupt_exit_after_reservation_recovers_as_counted_global_interlock(
    tmp_path: Path,
) -> None:
    hardware = FakeCoordinator()
    clocks = Clocks()
    auto, _, _, _, _ = service(tmp_path, coordinator=hardware, clocks=clocks)

    def terminate_after_reservation(_reservation_id: str) -> None:
        raise SystemExit(99)

    hardware.after_reservation = terminate_after_reservation
    with pytest.raises(SystemExit, match="99"):
        classify(auto)

    interrupted = json.loads((tmp_path / "auto-fire-state.json").read_text(encoding="utf-8"))
    assert interrupted["attempts"][0]["state"] == "reserved"

    restarted, _, _, _, restarted_hardware = service(
        tmp_path,
        clocks=clocks,
        current_target=target(event_id="event-two", track_id=8),
    )
    restarted_decision = classify(restarted, event_id="event-two", track_id=8)

    assert restarted_decision.reason == "unresolved_actuation"
    assert restarted_hardware.calls == []
    status = restarted.status()
    assert status["shots_in_rolling_window"] == 1
    assert status["unresolved_actuation_reservations"] == 1
    assert status["restart_interlock_remaining_seconds"] == 3600.0
    recovered = json.loads((tmp_path / "auto-fire-state.json").read_text(encoding="utf-8"))
    assert recovered["attempts"][0]["state"] == "recovered_uncertain"
    assert recovered["attempts"][0]["outcome"] == "process_restart"


def test_schema_two_history_migrates_without_losing_rate_limit(tmp_path: Path) -> None:
    state_path = tmp_path / "auto-fire-state.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "clock": {
                    "boot_id": None,
                    "monotonic_seconds": 100.0,
                    "wall_epoch_seconds": 10_000.0,
                },
                "shots": [
                    {
                        "event_id": "event-one",
                        "track_id": 7,
                        "accepted_at_epoch_seconds": 10_000.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    auto, _, _, _, hardware = service(tmp_path)

    assert classify(auto).reason == "target_already_engaged"
    assert hardware.calls == []
    migrated = json.loads(state_path.read_text(encoding="utf-8"))
    assert migrated["schema_version"] == 3
    assert len(migrated["attempts"]) == 1
    attempt = migrated["attempts"][0]
    assert attempt["state"] == "legacy_completed"
    assert attempt["reserved_at_epoch_seconds"] == 10_000.0
    assert attempt["shot_attempted"] is True


def test_each_transaction_commit_requests_parent_directory_sync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auto, _, _, _, _ = service(tmp_path)
    synced: list[Path] = []
    monkeypatch.setattr(auto, "_fsync_parent_directory", synced.append)

    assert classify(auto).accepted is True

    assert synced == [tmp_path, tmp_path]


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


def test_100632_qualifying_animal_ignores_secondary_person_for_class_policy(
    tmp_path: Path,
) -> None:
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


def test_strong_animal_ignores_secondary_person_for_class_policy(tmp_path: Path) -> None:
    auto, _, _, _, hardware = service(tmp_path)

    decision = classify(
        auto,
        detections=(AutoFireDetection("dog", 0.90), AutoFireDetection("person", 0.40)),
    )

    assert decision.accepted is True and decision.reason == "accepted"
    assert hardware.calls


def test_top_person_detection_latches_event_and_overrides_later_animal(tmp_path: Path) -> None:
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
    auto, _, _, _, hardware = service(tmp_path)

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
    assert state["attempts"] == []


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
    assert state["attempts"] == []


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
    assert len(persisted["attempts"]) == 1
    attempt = persisted["attempts"][0]
    assert attempt["event_id"] == "event-one" and attempt["track_id"] == 7
    assert attempt["state"] == "failed" and attempt["outcome"] == "park_failed"
    assert attempt["shot_attempted"] is True
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
    assert len(after["attempts"]) == 1
    assert after["attempts"][0]["reserved_at_epoch_seconds"] == 10000.0
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
    assert len(persisted["attempts"]) == 1
    assert persisted["attempts"][0]["reserved_at_epoch_seconds"] == 10000.0


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
    assert len(persisted["attempts"]) == 1
    assert persisted["attempts"][0]["reserved_at_epoch_seconds"] == 13_100.0


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


def test_semantically_invalid_v3_rate_state_disables_engagement(tmp_path: Path) -> None:
    state_path = tmp_path / "auto-fire-state.json"
    auto, _, _, _, _ = service(tmp_path)
    assert classify(auto).accepted is True

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["attempts"][0]["state"] = "reserved"
    state_path.write_text(json.dumps(payload), encoding="utf-8")

    restarted, _, _, _, hardware = service(tmp_path)
    decision = classify(restarted)

    assert decision.reason == "safety_state_invalid"
    assert hardware.calls == []
    persistence = restarted.status()["persistence"]
    assert persistence["healthy"] is False  # type: ignore[index]
    assert "terminal fields" in persistence["error"]  # type: ignore[index,operator]


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
