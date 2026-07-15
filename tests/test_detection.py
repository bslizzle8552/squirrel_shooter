from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from conftest import write_test_config
from squirrel_shooter.config import MotionConfig, load_config
from squirrel_shooter.detection import DetectorState, MotionDetector


class MaskSubtractor:
    def __init__(self, masks: list[np.ndarray]) -> None:
        self.masks = masks
        self.index = 0

    def apply(self, frame: np.ndarray) -> np.ndarray:
        del frame
        mask = self.masks[min(self.index, len(self.masks) - 1)]
        self.index += 1
        return mask.copy()


def mask_with_box(x: int, y: int, width: int, height: int) -> np.ndarray:
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(mask, (x, y), (x + width - 1, y + height - 1), 255, -1)
    return mask


def detector_config(tmp_path: Path, **changes: object) -> MotionConfig:
    config = load_config(write_test_config(tmp_path)).motion
    base = replace(
        config,
        processing_width=100,
        learning_frames=1,
        blur_kernel=1,
        morphology_kernel=1,
        open_iterations=0,
        close_iterations=0,
        min_blob_area=20,
        max_blob_area=5000,
        persistence_frames=2,
        persistence_max_distance=20,
        lighting_change_percent=80,
    )
    return replace(base, **changes)


def test_learning_mode_suppresses_candidates_and_events(tmp_path: Path) -> None:
    box = mask_with_box(20, 20, 20, 20)
    detector = MotionDetector(detector_config(tmp_path, learning_frames=2), subtractor=MaskSubtractor([box, box, box]))
    first = detector.process(np.zeros((100, 100, 3), dtype=np.uint8), now=0.0)
    second = detector.process(np.zeros((100, 100, 3), dtype=np.uint8), now=0.1)
    assert first.state is second.state is DetectorState.LEARNING
    assert first.candidates == second.candidates == ()


def test_small_blob_and_noise_are_rejected(tmp_path: Path) -> None:
    noise = np.zeros((100, 100), dtype=np.uint8)
    noise[10, 10] = noise[30, 50] = noise[80, 70] = 255
    detector = MotionDetector(detector_config(tmp_path), subtractor=MaskSubtractor([np.zeros_like(noise), noise]))
    detector.process(np.zeros((100, 100, 3), dtype=np.uint8), now=0.0)
    result = detector.process(np.zeros((100, 100, 3), dtype=np.uint8), now=0.1)
    assert result.candidates
    assert all(candidate.reason == "too_small" for candidate in result.candidates)
    assert result.accepted_candidate is None


def test_motion_requires_persistence_then_respects_cooldown(tmp_path: Path) -> None:
    empty = np.zeros((100, 100), dtype=np.uint8)
    box = mask_with_box(25, 25, 20, 20)
    detector = MotionDetector(detector_config(tmp_path, cooldown_seconds=10), subtractor=MaskSubtractor([empty, box, box, box]))
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detector.process(frame, now=0.0)
    first = detector.process(frame, now=1.0)
    accepted = detector.process(frame, now=2.0)
    cooldown = detector.process(frame, now=3.0)
    assert first.candidates[0].reason == "awaiting_persistence"
    assert accepted.accepted_candidate is not None
    assert cooldown.candidates[0].reason == "cooldown"
    assert cooldown.candidates[0].cooldown_remaining == 9.0


def test_roi_and_max_blob_filtering(tmp_path: Path) -> None:
    roi = replace(detector_config(tmp_path).roi, enabled=True, x=0.5, y=0.0, width=0.5, height=1.0)
    config = replace(detector_config(tmp_path), roi=roi, max_blob_area=200)
    empty = np.zeros((100, 100), dtype=np.uint8)
    outside = mask_with_box(5, 20, 10, 10)
    too_large = mask_with_box(60, 20, 30, 30)
    detector = MotionDetector(config, subtractor=MaskSubtractor([empty, outside, too_large]))
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detector.process(frame, now=0.0)
    outside_result = detector.process(frame, now=1.0)
    large_result = detector.process(frame, now=2.0)
    assert outside_result.candidates[0].reason == "outside_roi"
    assert outside_result.candidates[0].roi_status == "outside"
    assert large_result.candidates[0].reason == "too_large"


def test_frame_wide_lighting_change_restarts_learning(tmp_path: Path) -> None:
    empty = np.zeros((100, 100), dtype=np.uint8)
    full = np.full((100, 100), 255, dtype=np.uint8)
    detector = MotionDetector(detector_config(tmp_path, lighting_change_percent=60), subtractor=MaskSubtractor([empty, full]))
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detector.process(frame, now=0.0)
    result = detector.process(frame, now=1.0)
    assert result.lighting_reset is True
    assert result.state is DetectorState.LEARNING
    assert result.candidates == ()
