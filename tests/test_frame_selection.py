from __future__ import annotations

import numpy as np

from squirrel_shooter.frame_selection import BestEventFrameSelector


def frame(value: int = 0) -> np.ndarray:
    return np.full((100, 160, 3), value, dtype=np.uint8)


def test_largest_valid_in_frame_motion_candidate_is_selected() -> None:
    selector = BestEventFrameSelector(fallback_frame_number=1, minimum_motion_area=500)
    selector.consider(1, frame(1), (20, 20, 20, 20), 600)
    selector.consider(2, frame(2), (20, 20, 50, 30), 900)
    selector.consider(3, frame(3), (20, 20, 30, 30), 800)

    selected = selector.select()

    assert selected is not None
    assert selected.frame_number == 2
    assert selected.method == "best"
    assert selected.bounding_box_area == 1500
    assert selected.total_frames_considered == 3


def test_large_edge_touching_candidate_does_not_beat_clear_in_frame_candidate() -> None:
    selector = BestEventFrameSelector(fallback_frame_number=1, minimum_motion_area=500)
    selector.consider(1, frame(1), (0, 5, 120, 80), 5000)
    selector.consider(2, frame(2), (30, 25, 35, 25), 700)

    selected = selector.select()

    assert selected is not None
    assert selected.frame_number == 2
    assert selected.bounding_box_area == 875


def test_configured_fallback_is_used_when_no_candidate_can_be_scored() -> None:
    selector = BestEventFrameSelector(fallback_frame_number=2, minimum_motion_area=500)
    selector.consider(1, frame(1), (20, 20, 30, 20), 100)
    selector.consider(2, frame(2), (0, 0, 120, 80), 5000)

    selected = selector.select()

    assert selected is not None
    assert selected.frame_number == 2
    assert selected.method == "configured_fallback"


def test_middle_then_first_fallbacks_work_when_configured_frame_is_unavailable() -> None:
    selector = BestEventFrameSelector(fallback_frame_number=9, minimum_motion_area=500)
    available = {index: frame(index) for index in range(1, 5)}
    for index, event_frame in available.items():
        selector.consider(index, event_frame, None, None)

    middle = selector.select(available.get)
    first = selector.select()

    assert middle is not None
    assert middle.frame_number == 2 and middle.method == "middle_fallback"
    assert np.all(middle.frame == 2)
    assert first is not None
    assert first.frame_number == 1 and first.method == "first_fallback"
