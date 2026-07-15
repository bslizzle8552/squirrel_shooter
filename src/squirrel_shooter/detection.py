"""Lightweight, camera-independent motion detection and visualization."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from time import monotonic
from typing import Any

import cv2
import numpy as np

from .config import MotionConfig


class DetectorState(StrEnum):
    DISABLED = "DISABLED"
    LEARNING = "LEARNING"
    READY = "READY"
    ERROR = "ERROR"


@dataclass(frozen=True)
class MotionCandidate:
    timestamp: str
    bounding_box: tuple[int, int, int, int]
    center: tuple[int, int]
    area: float
    persistence: int
    roi_status: str
    accepted: bool
    reason: str
    snapshot_saved: bool = False
    snapshot_filename: str | None = None
    cooldown_remaining: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        x, y, width, height = self.bounding_box
        return {
            "timestamp": self.timestamp,
            "bounding_box": {"x": x, "y": y, "width": width, "height": height},
            "center": {"x": self.center[0], "y": self.center[1]},
            "area": round(self.area, 1),
            "persistence": self.persistence,
            "roi_status": self.roi_status,
            "accepted": self.accepted,
            "reason": self.reason,
            "snapshot_saved": self.snapshot_saved,
            "snapshot_filename": self.snapshot_filename,
            "cooldown_remaining": round(self.cooldown_remaining, 2),
        }


@dataclass(frozen=True)
class DetectionResult:
    state: DetectorState
    candidates: tuple[MotionCandidate, ...]
    raw_mask: np.ndarray
    cleaned_mask: np.ndarray
    processing_fps: float
    persistence: int
    lighting_reset: bool = False

    @property
    def accepted_candidate(self) -> MotionCandidate | None:
        return next((candidate for candidate in self.candidates if candidate.accepted), None)

    @property
    def blob_count(self) -> int:
        return len(self.candidates)


class MotionDetector:
    """MOG2 detector with learning, cleanup, ROI, persistence, and cooldown."""

    def __init__(self, config: MotionConfig, *, subtractor: Any | None = None) -> None:
        self.config = config
        self._subtractor = subtractor or self._new_subtractor()
        self._learning_count = 0
        self._last_center: tuple[int, int] | None = None
        self._persistence = 0
        self._last_event_at: float | None = None
        self._last_processed_at: float | None = None
        self._processing_fps = 0.0

    @property
    def state(self) -> DetectorState:
        if not self.config.enabled:
            return DetectorState.DISABLED
        if self._learning_count < self.config.learning_frames:
            return DetectorState.LEARNING
        return DetectorState.READY

    @property
    def learning_progress(self) -> tuple[int, int]:
        return min(self._learning_count, self.config.learning_frames), self.config.learning_frames

    def _new_subtractor(self) -> Any:
        return cv2.createBackgroundSubtractorMOG2(
            history=self.config.history,
            varThreshold=self.config.variance_threshold,
            detectShadows=self.config.detect_shadows,
        )

    def reset_background(self) -> None:
        self._subtractor = self._new_subtractor()
        self._learning_count = 0
        self._last_center = None
        self._persistence = 0

    def process(self, frame: np.ndarray, *, now: float | None = None) -> DetectionResult:
        current = monotonic() if now is None else now
        processing_fps = self._update_fps(current)
        target_width = min(self.config.processing_width, frame.shape[1])
        scale = target_width / frame.shape[1]
        target_height = max(1, round(frame.shape[0] * scale))
        reduced = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        empty = np.zeros((target_height, target_width), dtype=np.uint8)

        if not self.config.enabled:
            return DetectionResult(DetectorState.DISABLED, (), empty, empty, processing_fps, 0)

        blurred = cv2.GaussianBlur(
            reduced,
            (self.config.blur_kernel, self.config.blur_kernel),
            0,
        )
        raw_mask = self._subtractor.apply(blurred)
        # MOG2 marks shadows as 127. The 200 threshold retains foreground only.
        _, foreground = cv2.threshold(raw_mask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.morphology_kernel, self.config.morphology_kernel),
        )
        cleaned = foreground
        if self.config.open_iterations:
            cleaned = cv2.morphologyEx(
                cleaned, cv2.MORPH_OPEN, kernel, iterations=self.config.open_iterations
            )
        if self.config.close_iterations:
            cleaned = cv2.morphologyEx(
                cleaned, cv2.MORPH_CLOSE, kernel, iterations=self.config.close_iterations
            )

        if self.state is DetectorState.LEARNING:
            self._learning_count += 1
            self._persistence = 0
            self._last_center = None
            return DetectionResult(DetectorState.LEARNING, (), raw_mask, cleaned, processing_fps, 0)

        roi = self.roi_pixels(target_width, target_height)
        roi_area = max(1, roi[2] * roi[3])
        roi_mask = cleaned[roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]]
        changed_percent = 100.0 * cv2.countNonZero(roi_mask) / roi_area
        if changed_percent >= self.config.lighting_change_percent:
            self.reset_background()
            return DetectionResult(
                DetectorState.LEARNING,
                (),
                raw_mask,
                cleaned,
                processing_fps,
                0,
                lighting_reset=True,
            )

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
        preliminary: list[MotionCandidate] = []
        eligible: list[tuple[int, tuple[int, int]]] = []
        inverse_scale = 1.0 / scale

        for contour in contours:
            area = float(cv2.contourArea(contour))
            x, y, width, height = cv2.boundingRect(contour)
            center = (x + width // 2, y + height // 2)
            inside_roi = self._point_in_rect(center, roi)
            reason = "candidate"
            if not inside_roi:
                reason = "outside_roi"
            elif area < self.config.min_blob_area:
                reason = "too_small"
            elif area > self.config.max_blob_area:
                reason = "too_large"
            index = len(preliminary)
            preliminary.append(
                MotionCandidate(
                    timestamp=timestamp,
                    bounding_box=self._scale_box((x, y, width, height), inverse_scale),
                    center=self._scale_point(center, inverse_scale),
                    area=area,
                    persistence=0,
                    roi_status="inside" if inside_roi else "outside",
                    accepted=False,
                    reason=reason,
                )
            )
            if reason == "candidate":
                eligible.append((index, center))

        if not eligible:
            self._persistence = 0
            self._last_center = None
            return DetectionResult(self.state, tuple(preliminary), raw_mask, cleaned, processing_fps, 0)

        target_index, target_center = max(eligible, key=lambda item: preliminary[item[0]].area)
        if self._last_center is not None and math.dist(target_center, self._last_center) <= self.config.persistence_max_distance:
            self._persistence += 1
        else:
            self._persistence = 1
        self._last_center = target_center

        cooldown_remaining = 0.0
        if self._last_event_at is not None:
            cooldown_remaining = max(0.0, self.config.cooldown_seconds - (current - self._last_event_at))
        accepted = self._persistence >= self.config.persistence_frames and cooldown_remaining == 0.0
        reason = "accepted" if accepted else (
            "cooldown" if cooldown_remaining > 0 else "awaiting_persistence"
        )
        if accepted:
            self._last_event_at = current

        preliminary[target_index] = replace(
            preliminary[target_index],
            persistence=self._persistence,
            accepted=accepted,
            reason=reason,
            cooldown_remaining=cooldown_remaining,
        )
        for index, _ in eligible:
            if index != target_index:
                preliminary[index] = replace(preliminary[index], reason="secondary_candidate")
        return DetectionResult(
            self.state,
            tuple(preliminary),
            raw_mask,
            cleaned,
            processing_fps,
            self._persistence,
        )

    def roi_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        return _roi_pixels(self.config, width, height)

    def _update_fps(self, now: float) -> float:
        if self._last_processed_at is not None and now > self._last_processed_at:
            instant = 1.0 / (now - self._last_processed_at)
            self._processing_fps = instant if self._processing_fps == 0.0 else (0.2 * instant) + (0.8 * self._processing_fps)
        self._last_processed_at = now
        return self._processing_fps

    @staticmethod
    def _point_in_rect(point: tuple[int, int], rect: tuple[int, int, int, int]) -> bool:
        x, y = point
        left, top, width, height = rect
        return left <= x < left + width and top <= y < top + height

    @staticmethod
    def _scale_point(point: tuple[int, int], scale: float) -> tuple[int, int]:
        return round(point[0] * scale), round(point[1] * scale)

    @staticmethod
    def _scale_box(box: tuple[int, int, int, int], scale: float) -> tuple[int, int, int, int]:
        return tuple(round(value * scale) for value in box)  # type: ignore[return-value]


def annotate_detection(
    frame: np.ndarray,
    result: DetectionResult,
    config: MotionConfig,
    *,
    event_timestamp: str | None = None,
) -> np.ndarray:
    """Draw a readable live/debug overlay on a full-resolution frame."""

    annotated = frame.copy()
    height, width = annotated.shape[:2]
    if config.roi.enabled:
        x, y, roi_width, roi_height = _roi_pixels(config, width, height)
        cv2.rectangle(annotated, (x, y), (x + roi_width, y + roi_height), (170, 170, 170), 1)
        cv2.putText(annotated, "ROI", (x + 6, max(18, y + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    for candidate in result.candidates:
        x, y, box_width, box_height = candidate.bounding_box
        center_x, center_y = candidate.center
        cv2.rectangle(annotated, (x, y), (x + box_width, y + box_height), (68, 220, 96), 2)
        cv2.drawMarker(annotated, (center_x, center_y), (68, 220, 96), cv2.MARKER_CROSS, 18, 2)
        label = f"area {candidate.area:.0f} | {candidate.reason}"
        cv2.putText(annotated, label, (x, max(20, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (68, 220, 96), 2, cv2.LINE_AA)

    state_detail = result.state.value
    if result.state is DetectorState.LEARNING:
        state_detail = f"LEARNING (background warm-up)"
    lines = (
        f"Detector: {state_detail} | enabled: {'yes' if config.enabled else 'no'}",
        f"Processing: {result.processing_fps:4.1f} FPS | blobs: {result.blob_count} | persistence: {result.persistence}/{config.persistence_frames}",
    )
    panel_width = min(width - 16, 720)
    cv2.rectangle(annotated, (8, 8), (8 + panel_width, 70), (0, 0, 0), -1)
    for index, line in enumerate(lines):
        cv2.putText(annotated, line, (18, 32 + index * 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

    if result.accepted_candidate is not None:
        cv2.putText(annotated, "MOTION", (18, 108), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (68, 220, 96), 3, cv2.LINE_AA)
    if event_timestamp:
        cv2.rectangle(annotated, (8, height - 42), (min(width - 8, 690), height - 8), (0, 0, 0), -1)
        cv2.putText(annotated, f"Event {event_timestamp} | detector {result.state.value}", (18, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    return annotated


def _roi_pixels(config: MotionConfig, width: int, height: int) -> tuple[int, int, int, int]:
    if not config.roi.enabled:
        return 0, 0, width, height
    x = round(config.roi.x * width)
    y = round(config.roi.y * height)
    right = round((config.roi.x + config.roi.width) * width)
    bottom = round((config.roi.y + config.roi.height) * height)
    return x, y, max(1, right - x), max(1, bottom - y)
