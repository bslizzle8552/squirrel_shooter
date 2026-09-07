"""LEGACY MobileNet full-frame person checks on the shared camera/classifier.

Retained for current behavior only. The future one-class squirrel architecture
does not require this provider or a replacement human detector.
"""

from __future__ import annotations

import math
from time import monotonic
from typing import Callable, Protocol

from .camera_service import CameraService
from .legacy_mobilenet_policy import ScenePersonSafetyResult
from .safety import SceneFramePacket


class SceneClassifier(Protocol):
    """LEGACY classifier seam used without creating another model owner."""

    def classify_scene(
        self,
        request_id: str,
        event_id: str,
        track_id: int,
        packet: SceneFramePacket,
        *,
        timeout_seconds: float,
    ) -> ScenePersonSafetyResult:
        ...

    def latest_scene_result(self) -> ScenePersonSafetyResult | None:
        ...


class ClassifierSceneSafetyProvider:
    """Wait for a post-settle camera frame, then use the single DNN scheduler."""

    def __init__(
        self,
        camera: CameraService,
        classifier: SceneClassifier,
        *,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._camera = camera
        self._classifier = classifier
        self._clock = clock

    def check_current_scene(
        self,
        *,
        request_id: str,
        event_id: str,
        track_id: int,
        after_sequence: int | None,
        timeout_seconds: float,
    ) -> ScenePersonSafetyResult:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0.0
        ):
            raise ValueError("timeout_seconds must be finite and greater than zero")

        started = self._clock()
        deadline = started + float(timeout_seconds)
        camera_sequence = self._camera.current_frame_sequence()
        minimum_sequence = camera_sequence
        if isinstance(after_sequence, int) and not isinstance(after_sequence, bool):
            minimum_sequence = max(minimum_sequence, after_sequence)
        remaining = deadline - self._clock()
        if remaining <= 0.0:
            return self._failure(
                request_id,
                event_id,
                track_id,
                "scene_frame_deadline_expired",
            )
        try:
            packet = self._camera.wait_for_frame(
                minimum_sequence,
                timeout=remaining,
                copy=False,
            )
        except Exception as exc:
            return self._failure(
                request_id,
                event_id,
                track_id,
                f"scene_frame_error: {type(exc).__name__}: {exc}",
            )
        if packet is None or packet.sequence <= minimum_sequence:
            return self._failure(
                request_id,
                event_id,
                track_id,
                "current_scene_frame_unavailable",
            )

        remaining = deadline - self._clock()
        if remaining <= 0.0:
            return self._failure(
                request_id,
                event_id,
                track_id,
                "scene_inference_deadline_expired",
            )
        scene_packet = SceneFramePacket(
            sequence=packet.sequence,
            received_monotonic=packet.received_monotonic,
            frame=packet.frame,
        )
        try:
            return self._classifier.classify_scene(
                request_id,
                event_id,
                track_id,
                scene_packet,
                timeout_seconds=remaining,
            )
        except Exception as exc:
            return ScenePersonSafetyResult(
                status="error",
                request_id=request_id,
                event_id=event_id,
                track_id=track_id,
                coordinate_space="native_full_frame",
                source_sequence=scene_packet.sequence,
                source_received_monotonic=scene_packet.received_monotonic,
                frame_width=scene_packet.width,
                frame_height=scene_packet.height,
                detections=(),
                completed_monotonic=max(self._clock(), scene_packet.received_monotonic),
                error=f"scene_classifier_error: {type(exc).__name__}: {exc}",
            )

    def latest_scene_result(self) -> ScenePersonSafetyResult | None:
        return self._classifier.latest_scene_result()

    def _failure(
        self,
        request_id: str,
        event_id: str,
        track_id: int,
        error: str,
    ) -> ScenePersonSafetyResult:
        return ScenePersonSafetyResult(
            status="unavailable",
            request_id=request_id,
            event_id=event_id,
            track_id=track_id,
            coordinate_space="native_full_frame",
            source_sequence=None,
            source_received_monotonic=None,
            frame_width=None,
            frame_height=None,
            detections=(),
            completed_monotonic=max(0.0, self._clock()),
            error=error,
        )


__all__ = ["ClassifierSceneSafetyProvider", "SceneClassifier"]
