"""Bounded-memory selection of one representative frame from a motion event."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Callable

import numpy as np


BoundingBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class SelectedEventFrame:
    frame_number: int
    frame: np.ndarray
    bounding_box: BoundingBox
    method: str
    bounding_box_area: int | None
    total_frames_considered: int


@dataclass(frozen=True)
class _StoredFrame:
    frame_number: int
    frame: np.ndarray
    bounding_box: BoundingBox
    bounding_box_area: int | None
    motion_area: float | None


class BestEventFrameSelector:
    """Track the best and fallback frames without retaining the whole event."""

    def __init__(
        self,
        *,
        fallback_frame_number: int,
        minimum_motion_area: float,
        selection_mode: str = "best",
    ) -> None:
        self.fallback_frame_number = fallback_frame_number
        self.minimum_motion_area = minimum_motion_area
        self.selection_mode = selection_mode
        self.total_frames_considered = 0
        self._best: _StoredFrame | None = None
        self._configured_fallback: _StoredFrame | None = None
        self._first: _StoredFrame | None = None

    def consider(
        self,
        frame_number: int,
        frame: np.ndarray | None,
        bounding_box: BoundingBox | None,
        motion_area: float | None,
    ) -> None:
        """Consider one event frame and retain only useful representatives."""

        self.total_frames_considered += 1
        if not _valid_frame(frame):
            return
        assert frame is not None
        box = bounding_box or (0, 0, 0, 0)
        box_area = _box_area(box)
        stored = _StoredFrame(
            frame_number,
            frame.copy(),
            box,
            box_area,
            motion_area,
        )
        if self._first is None:
            self._first = stored
        if frame_number == self.fallback_frame_number:
            self._configured_fallback = stored
        if self.selection_mode != "best" or not self._is_valid_candidate(stored):
            return
        if self._best is None or self._rank(stored) > self._rank(self._best):
            self._best = stored

    def select(
        self,
        middle_frame_loader: Callable[[int], np.ndarray | None] | None = None,
    ) -> SelectedEventFrame | None:
        """Return the best frame, then configured, middle, or first fallback."""

        if self._best is not None:
            return self._selected(self._best, "best")
        if self._configured_fallback is not None:
            return self._selected(self._configured_fallback, "configured_fallback")
        middle_number = (self.total_frames_considered + 1) // 2
        if middle_frame_loader is not None and middle_number > 0:
            middle = middle_frame_loader(middle_number)
            if _valid_frame(middle):
                assert middle is not None
                return SelectedEventFrame(
                    middle_number,
                    middle.copy(),
                    (0, 0, 0, 0),
                    "middle_fallback",
                    None,
                    self.total_frames_considered,
                )
        if self._first is not None:
            return self._selected(self._first, "first_fallback")
        return None

    def _is_valid_candidate(self, candidate: _StoredFrame) -> bool:
        if candidate.motion_area is None or candidate.motion_area < self.minimum_motion_area:
            return False
        x, y, width, height = candidate.bounding_box
        frame_height, frame_width = candidate.frame.shape[:2]
        if width <= 0 or height <= 0 or x < 0 or y < 0:
            return False
        if x + width > frame_width or y + height > frame_height:
            return False
        edge_margin = max(2, ceil(min(frame_width, frame_height) * 0.01))
        return (
            x > edge_margin
            and y > edge_margin
            and x + width < frame_width - edge_margin
            and y + height < frame_height - edge_margin
        )

    @staticmethod
    def _rank(candidate: _StoredFrame) -> tuple[int, float, int]:
        return (
            candidate.bounding_box_area or 0,
            candidate.motion_area or 0.0,
            -candidate.frame_number,
        )

    def _selected(self, stored: _StoredFrame, method: str) -> SelectedEventFrame:
        return SelectedEventFrame(
            stored.frame_number,
            stored.frame,
            stored.bounding_box,
            method,
            stored.bounding_box_area,
            self.total_frames_considered,
        )


def _valid_frame(frame: np.ndarray | None) -> bool:
    return (
        isinstance(frame, np.ndarray)
        and frame.ndim in {2, 3}
        and frame.size > 0
        and frame.shape[0] > 0
        and frame.shape[1] > 0
    )


def _box_area(box: BoundingBox) -> int | None:
    _, _, width, height = box
    return width * height if width > 0 and height > 0 else None
