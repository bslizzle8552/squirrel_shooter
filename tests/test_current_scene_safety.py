from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from squirrel_shooter.camera_service import FramePacket
from squirrel_shooter.current_scene_safety import ClassifierSceneSafetyProvider
from squirrel_shooter.safety import SceneFramePacket, ScenePersonSafetyResult


@dataclass
class FakeClock:
    now: float = 10.0

    def __call__(self) -> float:
        return self.now


class FakeCamera:
    def __init__(self, sequence: int = 7) -> None:
        self.sequence = sequence
        self.wait_calls: list[tuple[int, float | None, bool]] = []
        self.result: FramePacket | None = FramePacket(
            sequence + 1,
            np.zeros((720, 1280, 3), dtype=np.uint8),
            "2026-08-18T20:00:00-04:00",
            10.1,
        )

    def current_frame_sequence(self) -> int:
        return self.sequence

    def wait_for_frame(
        self,
        after_sequence: int,
        timeout: float | None = None,
        *,
        copy: bool = True,
    ) -> FramePacket | None:
        self.wait_calls.append((after_sequence, timeout, copy))
        return self.result


class FakeClassifier:
    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.calls: list[tuple[str, str, int, SceneFramePacket, float]] = []
        self.latest: ScenePersonSafetyResult | None = None
        self.error: Exception | None = None

    def classify_scene(
        self,
        request_id: str,
        event_id: str,
        track_id: int,
        packet: SceneFramePacket,
        *,
        timeout_seconds: float,
    ) -> ScenePersonSafetyResult:
        self.calls.append((request_id, event_id, track_id, packet, timeout_seconds))
        if self.error is not None:
            raise self.error
        self.latest = ScenePersonSafetyResult(
            status="clear",
            request_id=request_id,
            event_id=event_id,
            track_id=track_id,
            coordinate_space="native_full_frame",
            source_sequence=packet.sequence,
            source_received_monotonic=packet.received_monotonic,
            frame_width=packet.width,
            frame_height=packet.height,
            detections=(),
            completed_monotonic=max(self.clock(), packet.received_monotonic),
        )
        return self.latest

    def latest_scene_result(self) -> ScenePersonSafetyResult | None:
        return self.latest


def test_waits_for_frame_newer_than_both_current_camera_and_requested_sequence() -> None:
    clock = FakeClock()
    camera = FakeCamera(sequence=7)
    camera.result = FramePacket(
        12,
        np.zeros((720, 1280, 3), dtype=np.uint8),
        "now",
        10.1,
    )
    classifier = FakeClassifier(clock)
    provider = ClassifierSceneSafetyProvider(camera, classifier, clock=clock)  # type: ignore[arg-type]

    result = provider.check_current_scene(
        request_id="request-1",
        event_id="event-1",
        track_id=3,
        after_sequence=11,
        timeout_seconds=1.5,
    )

    assert camera.wait_calls == [(11, 1.5, False)]
    assert classifier.calls[0][3].sequence == 12
    assert result.status == "clear"
    assert result.source_sequence == 12
    assert provider.latest_scene_result() is result


def test_camera_timeout_returns_structured_unavailable_without_inference() -> None:
    clock = FakeClock()
    camera = FakeCamera()
    camera.result = None
    classifier = FakeClassifier(clock)
    provider = ClassifierSceneSafetyProvider(camera, classifier, clock=clock)  # type: ignore[arg-type]

    result = provider.check_current_scene(
        request_id="request-2",
        event_id="event-2",
        track_id=4,
        after_sequence=5,
        timeout_seconds=0.5,
    )

    assert result.status == "unavailable"
    assert result.error == "current_scene_frame_unavailable"
    assert result.source_sequence is None
    assert classifier.calls == []


def test_classifier_exception_is_a_sourced_fail_closed_error() -> None:
    clock = FakeClock()
    camera = FakeCamera()
    classifier = FakeClassifier(clock)
    classifier.error = RuntimeError("model unavailable")
    provider = ClassifierSceneSafetyProvider(camera, classifier, clock=clock)  # type: ignore[arg-type]

    result = provider.check_current_scene(
        request_id="request-3",
        event_id="event-3",
        track_id=5,
        after_sequence=7,
        timeout_seconds=1.0,
    )

    assert result.status == "error"
    assert result.source_sequence == 8
    assert "model unavailable" in (result.error or "")


@pytest.mark.parametrize("timeout", [0, -1, True, float("nan"), float("inf")])
def test_invalid_timeout_is_rejected(timeout: object) -> None:
    clock = FakeClock()
    provider = ClassifierSceneSafetyProvider(  # type: ignore[arg-type]
        FakeCamera(),
        FakeClassifier(clock),
        clock=clock,
    )

    with pytest.raises(ValueError, match="timeout_seconds"):
        provider.check_current_scene(
            request_id="request",
            event_id="event",
            track_id=1,
            after_sequence=None,
            timeout_seconds=timeout,  # type: ignore[arg-type]
        )
