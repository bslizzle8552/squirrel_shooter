from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from conftest import write_test_config
from squirrel_shooter.config import MotionConfig, load_config
from squirrel_shooter.watch_detection import (
    MotionComponent,
    MotionWatcherDetector,
    WatchState,
    classify_candidate,
    evaluate_event_eligibility,
    group_components,
)


class MaskSubtractor:
    def __init__(self, masks: list[np.ndarray]) -> None:
        self.masks = masks
        self.index = 0

    def apply(self, frame: np.ndarray) -> np.ndarray:
        del frame
        mask = self.masks[min(self.index, len(self.masks) - 1)]
        self.index += 1
        return mask.copy()


def watch_config(tmp_path: Path, *, global_enabled: bool = False, persistence_frames: int = 2) -> MotionConfig:
    config = load_config(write_test_config(tmp_path)).motion
    return replace(
        config,
        processing_width=100,
        blur_kernel=1,
        morphology_kernel=1,
        open_iterations=0,
        close_iterations=0,
        min_blob_area=15,
        max_blob_area=9000,
        warmup=replace(config.warmup, seconds=0, minimum_frames=1),
        persistence=replace(config.persistence, frames=persistence_frames, cooldown_seconds=2.0, max_centroid_distance_pixels=40),
        grouping=replace(config.grouping, max_horizontal_gap_pixels=20, max_vertical_gap_pixels=20, expanded_box_margin_pixels=8, max_centroid_distance_pixels=35),
        global_rejection=replace(config.global_rejection, enabled=global_enabled, max_frame_motion_percent=35, max_zone_motion_percent=45, recovery_seconds=1),
        inclusion_zone=replace(
            config.inclusion_zone,
            enabled=False,
            polygon=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        ),
    )


def box_mask(*boxes: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros((100, 100), dtype=np.uint8)
    for x, y, width, height in boxes:
        cv2.rectangle(mask, (x, y), (x + width - 1, y + height - 1), 255, -1)
    return mask


def component(x: int, y: int, width: int, height: int, velocity: tuple[float, float] = (0, 0)) -> MotionComponent:
    area = width * height
    return MotionComponent((x, y, width, height), area, area, (x + width / 2, y + height / 2), velocity=velocity)


def test_nearby_person_fragments_group_as_one_large_object(tmp_path: Path) -> None:
    config = watch_config(tmp_path)
    parts = [component(20, 10, 18, 45, (5, 0)), component(41, 12, 18, 43, (5, 0)), component(29, 56, 18, 30, (5, 0))]
    groups = group_components(parts, config.grouping, 10_000, 10_000)
    assert len(groups) == 1
    classified = classify_candidate(replace(groups[0], persistence_count=3, coherent_motion=True, average_speed=50), (100, 100), config.classification)
    assert classified.provisional_category in {"large_object", "person_sized"}
    assert len(classified.components) == 3


def test_distant_simultaneous_blobs_remain_separate(tmp_path: Path) -> None:
    grouping = watch_config(tmp_path).grouping
    groups = group_components([component(2, 10, 8, 8), component(85, 70, 8, 8)], grouping, 10_000, 10_000)
    assert len(groups) == 2


def test_nearby_opposite_direction_blobs_remain_separate(tmp_path: Path) -> None:
    grouping = watch_config(tmp_path).grouping
    groups = group_components([component(20, 20, 15, 15, (20, 0)), component(38, 20, 15, 15, (-20, 0))], grouping, 10_000, 10_000)
    assert len(groups) == 2


def test_animal_body_tail_and_fence_fragments_group(tmp_path: Path) -> None:
    grouping = watch_config(tmp_path).grouping
    body_tail = [component(30, 35, 28, 18, (8, 1)), component(58, 38, 18, 8, (8, 1))]
    fence_split = [component(15, 60, 12, 20, (6, 0)), component(30, 60, 12, 20, (6, 0)), component(45, 60, 12, 20, (6, 0))]
    assert len(group_components(body_tail, grouping, 10_000, 10_000)) == 1
    assert len(group_components(fence_split, grouping, 10_000, 10_000)) == 1


def test_no_movement_and_tiny_noise_do_not_confirm(tmp_path: Path) -> None:
    empty = box_mask()
    tiny = box_mask((10, 10, 3, 3))
    detector = MotionWatcherDetector(watch_config(tmp_path), subtractor=MaskSubtractor([empty, tiny]))
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert detector.process(frame, now=0).groups == ()
    assert detector.process(frame, now=0.1).groups == ()


def test_real_mog2_detects_generated_coherent_movement(tmp_path: Path) -> None:
    config = watch_config(tmp_path, persistence_frames=1)
    config = replace(config, warmup=replace(config.warmup, seconds=0, minimum_frames=5))
    detector = MotionWatcherDetector(config)
    background = np.zeros((100, 100, 3), dtype=np.uint8)
    for index in range(20):
        detector.process(background, now=index * 0.1)
    moving = background.copy()
    cv2.rectangle(moving, (25, 30), (50, 50), (255, 255, 255), -1)
    result = detector.process(moving, now=2.1)
    assert result.state is WatchState.READY
    assert result.groups and result.groups[0].confirmed


def test_coherent_motion_persistence_and_time_cooldown(tmp_path: Path) -> None:
    empty = box_mask()
    masks = [empty, box_mask((20, 30, 15, 15)), box_mask((24, 30, 15, 15)), empty, box_mask((40, 30, 15, 15)), box_mask((44, 30, 15, 15)), box_mask((48, 30, 15, 15)), box_mask((50, 30, 15, 15))]
    detector = MotionWatcherDetector(watch_config(tmp_path), subtractor=MaskSubtractor(masks))
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detector.process(frame, now=0)
    assert not detector.process(frame, now=0.1).groups[0].confirmed
    first = detector.process(frame, now=0.2).groups[0]
    assert first.confirmed and first.newly_confirmed and first.average_speed > 0
    detector.process(frame, now=1.0)
    detector.process(frame, now=1.1)
    during_cooldown = detector.process(frame, now=1.2).groups[0]
    assert not during_cooldown.confirmed
    detector.process(frame, now=2.3)
    after_cooldown = detector.process(frame, now=2.4).groups[0]
    assert after_cooldown.confirmed


def test_inclusion_zone_rejects_outside_motion(tmp_path: Path) -> None:
    config = watch_config(tmp_path)
    zone = replace(config.inclusion_zone, enabled=True, polygon=((0.5, 0), (1, 0), (1, 1), (0.5, 1)))
    detector = MotionWatcherDetector(replace(config, inclusion_zone=zone), subtractor=MaskSubtractor([box_mask(), box_mask((5, 30, 15, 15))]))
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detector.process(frame, now=0)
    result = detector.process(frame, now=0.1)
    assert result.groups == ()
    assert result.global_motion.raw_foreground_percent == 0.0
    assert result.global_motion.cleaned_foreground_percent == 0.0


def test_inclusion_zone_keeps_motion_below_sky_cutoff(tmp_path: Path) -> None:
    config = watch_config(tmp_path, persistence_frames=1)
    zone = replace(
        config.inclusion_zone,
        enabled=True,
        polygon=((0.0, 0.26), (1.0, 0.26), (1.0, 1.0), (0.0, 1.0)),
    )
    detector = MotionWatcherDetector(
        replace(config, inclusion_zone=zone),
        subtractor=MaskSubtractor([box_mask(), box_mask((40, 40, 15, 15))]),
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    detector.process(frame, now=0)
    result = detector.process(frame, now=0.1)

    assert len(result.groups) == 1
    assert result.groups[0].bounding_box == (40, 40, 15, 15)


def test_large_lighting_change_and_scene_obstruction_are_rejected(tmp_path: Path) -> None:
    config = watch_config(tmp_path, global_enabled=True)
    full = np.full((100, 100), 255, dtype=np.uint8)
    detector = MotionWatcherDetector(config, subtractor=MaskSubtractor([box_mask(), full, full]))
    dark = np.zeros((100, 100, 3), dtype=np.uint8)
    bright = np.full_like(dark, 255)
    detector.process(dark, now=0)
    lighting = detector.process(bright, now=0.1)
    assert lighting.state is WatchState.GLOBAL_RECOVERY
    assert lighting.global_motion.reason == "probable_exposure_change"
    obstruction_detector = MotionWatcherDetector(config, subtractor=MaskSubtractor([box_mask(), full]))
    obstruction_detector.process(dark, now=0)
    assert obstruction_detector.process(dark, now=0.1).global_motion.reason == "probable_scene_obstruction"


def test_simulated_camera_movement_and_wind_regions_reject(tmp_path: Path) -> None:
    config = watch_config(tmp_path, global_enabled=True)
    boxes = tuple((x, y, 16, 16) for y in (3, 27, 51, 75) for x in (3, 27, 51, 75))
    busy = box_mask(*boxes)
    detector = MotionWatcherDetector(config, subtractor=MaskSubtractor([box_mask(), busy]))
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detector.process(frame, now=0)
    result = detector.process(frame, now=0.1)
    assert result.global_motion.reason == "probable_camera_movement"
    assert result.global_motion.disconnected_regions >= 12


def test_ir_cut_visual_transition_requires_image_evidence(tmp_path: Path) -> None:
    config = watch_config(tmp_path, global_enabled=True)
    full = np.full((100, 100), 255, dtype=np.uint8)
    detector = MotionWatcherDetector(config, subtractor=MaskSubtractor([box_mask(), full]))
    color = np.zeros((100, 100, 3), dtype=np.uint8)
    color[:, :, 2] = 180
    mono = np.full((100, 100, 3), 100, dtype=np.uint8)
    detector.process(color, now=0)
    measurement = detector.process(mono, now=0.1).global_motion
    assert measurement.reason == "probable_ir_mode_switch"
    assert measurement.colorfulness == 0.0
    assert measurement.luminance == 100.0


def test_low_fps_alone_never_implies_ir_and_other_fps_work(tmp_path: Path) -> None:
    mask = box_mask((20, 20, 15, 15))
    for interval in (0.1, 0.04):
        detector = MotionWatcherDetector(watch_config(tmp_path, persistence_frames=1), subtractor=MaskSubtractor([box_mask(), mask]))
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detector.process(frame, now=0)
        result = detector.process(frame, now=interval)
        assert result.groups[0].confirmed
        assert result.global_motion.reason != "probable_ir_mode_switch"


def test_two_small_distant_objects_do_not_trigger_outer_rectangle_rejection(tmp_path: Path) -> None:
    config = watch_config(tmp_path, global_enabled=True, persistence_frames=1)
    mask = box_mask((2, 5, 10, 10), (88, 80, 10, 10))
    detector = MotionWatcherDetector(config, subtractor=MaskSubtractor([box_mask(), mask]))
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detector.process(frame, now=0)
    result = detector.process(frame, now=0.1)
    assert result.global_motion.reason is None
    assert len(result.groups) == 2


def test_legitimate_person_group_below_global_threshold_is_loggable(tmp_path: Path) -> None:
    config = watch_config(tmp_path, global_enabled=True, persistence_frames=1)
    mask = box_mask((35, 10, 12, 50), (49, 10, 12, 50))
    detector = MotionWatcherDetector(config, subtractor=MaskSubtractor([box_mask(), mask]))
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detector.process(frame, now=0)
    result = detector.process(frame, now=0.1)
    assert result.global_motion.reason is None
    assert len(result.groups) == 1 and result.groups[0].confirmed


def test_global_detector_recovers_after_monotonic_pause(tmp_path: Path) -> None:
    config = watch_config(tmp_path, global_enabled=True, persistence_frames=1)
    detector = MotionWatcherDetector(config, subtractor=MaskSubtractor([box_mask(), np.full((100, 100), 255, np.uint8), box_mask(), box_mask((20, 20, 15, 15))]))
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detector.process(frame, now=0)
    assert detector.process(frame, now=0.1).state is WatchState.GLOBAL_RECOVERY
    assert detector.process(frame, now=0.5).state is WatchState.GLOBAL_RECOVERY
    assert detector.process(frame, now=1.2).state is WatchState.READY


def test_provisional_categories_never_claim_squirrel(tmp_path: Path) -> None:
    config = watch_config(tmp_path).classification
    base = group_components([component(10, 10, 2, 2)], watch_config(tmp_path).grouping, 10_000, 10_000)[0]
    tiny = classify_candidate(base, (100, 100), config)
    flicker = classify_candidate(replace(base, components=tuple(component(index * 4, 20, 3, 3) for index in range(6)), dispersed_motion=True), (100, 100), config)
    assert tiny.provisional_category == "tiny_motion"
    assert flicker.provisional_category == "plant_or_shadow_flicker"
    assert "squirrel" not in tiny.provisional_category


def test_candidate_filter_rejects_tiny_and_flickering_motion(tmp_path: Path) -> None:
    config = watch_config(tmp_path)
    tiny_base = group_components([component(10, 10, 2, 2)], config.grouping, 10_000, 10_000)[0]
    tiny = classify_candidate(tiny_base, (100, 100), config.classification)
    flicker = classify_candidate(
        replace(tiny_base, components=tuple(component(index * 4, 20, 3, 3) for index in range(6)), dispersed_motion=True),
        (100, 100),
        config.classification,
    )

    tiny = evaluate_event_eligibility(tiny, config.candidate_filter)
    flicker = evaluate_event_eligibility(flicker, config.candidate_filter)

    assert not tiny.event_eligible and tiny.event_filter_reason == "tiny_motion"
    assert not flicker.event_eligible and flicker.event_filter_reason == "plant_or_shadow_flicker"


def test_small_motion_requires_coherent_travel_before_confirmation(tmp_path: Path) -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    stationary_masks = [box_mask(), *(box_mask((20 + (index % 2), 30, 10, 10)) for index in range(5))]
    stationary = MotionWatcherDetector(
        watch_config(tmp_path, persistence_frames=5),
        subtractor=MaskSubtractor(stationary_masks),
    )
    stationary.process(frame, now=0)
    filtered = None
    for index in range(5):
        filtered = stationary.process(frame, now=0.1 * (index + 1)).groups[0]
    assert filtered is not None
    assert not filtered.confirmed
    assert filtered.event_filter_reason == "small_motion_not_coherent"

    traveling_masks = [box_mask(), *(box_mask((20 + index * 3, 30, 10, 10)) for index in range(5))]
    traveling = MotionWatcherDetector(
        watch_config(tmp_path, persistence_frames=5),
        subtractor=MaskSubtractor(traveling_masks),
    )
    traveling.process(frame, now=0)
    accepted = None
    for index in range(5):
        accepted = traveling.process(frame, now=0.1 * (index + 1)).groups[0]
    assert accepted is not None
    assert accepted.event_eligible and accepted.confirmed and accepted.newly_confirmed
    assert accepted.travel_distance >= 10
