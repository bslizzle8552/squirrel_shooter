from __future__ import annotations

import numpy as np
import pytest

from squirrel_shooter.safety import (
    FinalAimDecision,
    SceneDetection,
    SceneFramePacket,
    ScenePersonSafetyResult,
)


def test_scene_detection_requires_canonical_native_geometry() -> None:
    with pytest.raises(ValueError, match="canonical lowercase"):
        SceneDetection("Person", 0.9, (1, 2, 3, 4))
    with pytest.raises(ValueError, match="four integers"):
        SceneDetection("person", 0.9, [1, 2, 3, 4])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dimensions must be positive"):
        SceneDetection("person", 0.9, (1, 2, 0, 4))


def test_scene_packet_preserves_native_dimensions_and_source_identity() -> None:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    packet = SceneFramePacket(42, 100.0, frame)

    assert (packet.sequence, packet.width, packet.height) == (42, 1280, 720)
    assert packet.frame is frame


def test_scene_result_rejects_malformed_detection_container_and_time_order() -> None:
    values = {
        "status": "clear",
        "request_id": "request-one",
        "event_id": "event-one",
        "track_id": 7,
        "coordinate_space": "native_full_frame",
        "source_sequence": 42,
        "source_received_monotonic": 100.0,
        "frame_width": 1280,
        "frame_height": 720,
        "detections": (),
        "completed_monotonic": 100.1,
    }
    with pytest.raises(ValueError, match="tuple of SceneDetection"):
        ScenePersonSafetyResult(**{**values, "detections": []})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cannot precede"):
        ScenePersonSafetyResult(**{**values, "completed_monotonic": 99.9})  # type: ignore[arg-type]


def test_scene_result_person_and_clear_status_are_geometry_consistent() -> None:
    person = SceneDetection("person", 0.95, (100, 100, 80, 240))
    values = {
        "request_id": "request-one",
        "event_id": "event-one",
        "track_id": 7,
        "coordinate_space": "native_full_frame",
        "source_sequence": 42,
        "source_received_monotonic": 100.0,
        "frame_width": 1280,
        "frame_height": 720,
        "completed_monotonic": 100.1,
    }
    with pytest.raises(ValueError, match="requires a person"):
        ScenePersonSafetyResult(status="person", detections=(), **values)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cannot contain"):
        ScenePersonSafetyResult(status="clear", detections=(person,), **values)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cannot carry an error"):
        ScenePersonSafetyResult(status="clear", detections=(), error="partial failure", **values)  # type: ignore[arg-type]


def test_final_aim_contract_allows_only_reaim_to_carry_a_pixel() -> None:
    assert FinalAimDecision.accept().action == "accept"
    assert FinalAimDecision.reaim(50, 60).action == "reaim"
    assert FinalAimDecision.reject("unsafe").action == "reject"
    with pytest.raises(ValueError, match="only reaim"):
        FinalAimDecision("accept", "bad", 1, 2)
