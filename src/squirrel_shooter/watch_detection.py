"""Camera-independent multi-blob motion analysis for the unattended watcher."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from time import monotonic
from typing import Any, Iterable

import cv2
import numpy as np

from .config import CandidateFilterConfig, ClassificationConfig, GroupingConfig, MotionConfig


class WatchState(StrEnum):
    WARMING_UP = "WARMING_UP"
    READY = "READY"
    GLOBAL_RECOVERY = "GLOBAL_RECOVERY"
    DISABLED = "DISABLED"


@dataclass(frozen=True)
class MotionComponent:
    bounding_box: tuple[int, int, int, int]
    contour_area: float
    foreground_pixels: int
    centroid: tuple[float, float]
    contour: tuple[tuple[int, int], ...] = ()
    velocity: tuple[float, float] = (0.0, 0.0)

    def as_dict(self) -> dict[str, Any]:
        x, y, width, height = self.bounding_box
        return {
            "bounding_box": {"x": x, "y": y, "width": width, "height": height},
            "contour_area": round(self.contour_area, 2),
            "foreground_pixel_area": self.foreground_pixels,
            "centroid": {"x": round(self.centroid[0], 2), "y": round(self.centroid[1], 2)},
            "contour": [[x, y] for x, y in self.contour],
            "velocity": {"x": round(self.velocity[0], 2), "y": round(self.velocity[1], 2)},
        }


@dataclass(frozen=True)
class GroupedCandidate:
    components: tuple[MotionComponent, ...]
    bounding_box: tuple[int, int, int, int]
    centroid: tuple[float, float]
    contour_area: float
    foreground_pixels: int
    frame_percent: float
    inclusion_zone_percent: float
    grouping_confidence: float
    track_id: int = 0
    persistence_count: int = 1
    path: tuple[tuple[float, float], ...] = ()
    duration: float = 0.0
    travel_distance: float = 0.0
    average_speed: float = 0.0
    peak_speed: float = 0.0
    direction: str = "stationary"
    coherent_motion: bool = False
    dispersed_motion: bool = False
    touched_zone_boundary: bool = False
    provisional_category: str = "unclassified_motion"
    movement_attributes: tuple[str, ...] = ()
    heuristic_score: float = 0.0
    event_eligible: bool = True
    event_filter_reason: str | None = None
    confirmed: bool = False
    newly_confirmed: bool = False

    @property
    def width(self) -> int:
        return self.bounding_box[2]

    @property
    def height(self) -> int:
        return self.bounding_box[3]

    @property
    def aspect_ratio(self) -> float:
        return self.width / max(1, self.height)

    def as_dict(self) -> dict[str, Any]:
        x, y, width, height = self.bounding_box
        return {
            "track_id": self.track_id,
            "component_blobs": [component.as_dict() for component in self.components],
            "component_count": len(self.components),
            "component_bounding_boxes": [component.as_dict()["bounding_box"] for component in self.components],
            "grouped_bounding_box": {"x": x, "y": y, "width": width, "height": height},
            "grouped_centroid": {"x": round(self.centroid[0], 2), "y": round(self.centroid[1], 2)},
            "recent_centroid_path": [[round(px, 2), round(py, 2)] for px, py in self.path],
            "combined_contour_area": round(self.contour_area, 2),
            "combined_foreground_pixel_area": self.foreground_pixels,
            "total_width": width,
            "total_height": height,
            "frame_percentage_covered": round(self.frame_percent, 4),
            "inclusion_zone_percentage_covered": round(self.inclusion_zone_percent, 4),
            "grouping_confidence": round(self.grouping_confidence, 3),
            "persistence_count": self.persistence_count,
            "duration": round(self.duration, 3),
            "travel_distance": round(self.travel_distance, 2),
            "average_pixel_speed": round(self.average_speed, 2),
            "peak_pixel_speed": round(self.peak_speed, 2),
            "direction": self.direction,
            "aspect_ratio": round(self.aspect_ratio, 3),
            "mostly_stationary": "mostly_stationary" in self.movement_attributes,
            "coherent_motion": self.coherent_motion,
            "dispersed_motion": self.dispersed_motion,
            "touched_inclusion_zone_boundary": self.touched_zone_boundary,
            "provisional_category": self.provisional_category,
            "movement_attributes": list(self.movement_attributes),
            "heuristic_score": round(self.heuristic_score, 3),
            "event_eligible": self.event_eligible,
            "event_filter_reason": self.event_filter_reason,
            "confirmed": self.confirmed,
        }


@dataclass(frozen=True)
class GlobalMotionMeasurement:
    raw_foreground_percent: float
    cleaned_foreground_percent: float
    candidate_region_percent: float
    inclusion_zone_motion_percent: float
    disconnected_regions: int
    reason: str | None = None
    luminance_delta: float = 0.0
    colorfulness_delta: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {name: round(value, 3) if isinstance(value, float) else value for name, value in self.__dict__.items()}


@dataclass(frozen=True)
class WatchDetectionResult:
    state: WatchState
    groups: tuple[GroupedCandidate, ...]
    raw_mask: np.ndarray
    cleaned_mask: np.ndarray
    zone_mask: np.ndarray
    global_motion: GlobalMotionMeasurement
    raw_contour_count: int
    measured_processing_fps: float


@dataclass
class _Track:
    track_id: int
    first_at: float
    last_at: float
    persistence: int
    path: list[tuple[float, float]] = field(default_factory=list)
    speeds: list[float] = field(default_factory=list)
    areas: list[float] = field(default_factory=list)
    confirmed: bool = False


def _box_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int]:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    horizontal = max(0, max(ax, bx) - min(ax + aw, bx + bw))
    vertical = max(0, max(ay, by) - min(ay + ah, by + bh))
    return horizontal, vertical


def _direction_difference(a: tuple[float, float], b: tuple[float, float]) -> float:
    if math.hypot(*a) < 1e-6 or math.hypot(*b) < 1e-6:
        return 0.0
    dot = a[0] * b[0] + a[1] * b[1]
    cosine = max(-1.0, min(1.0, dot / (math.hypot(*a) * math.hypot(*b))))
    return math.degrees(math.acos(cosine))


def _movement_consistent(a: MotionComponent, b: MotionComponent, config: GroupingConfig) -> bool:
    speed_a, speed_b = math.hypot(*a.velocity), math.hypot(*b.velocity)
    if speed_a < 1.0 or speed_b < 1.0:
        return True
    ratio = max(speed_a, speed_b) / max(1e-6, min(speed_a, speed_b))
    return ratio <= config.speed_similarity_ratio and _direction_difference(a.velocity, b.velocity) <= config.direction_similarity_degrees


def group_components(components: Iterable[MotionComponent], config: GroupingConfig, frame_area: int, zone_area: int) -> tuple[GroupedCandidate, ...]:
    """Group fragmented contours only when spatial and movement evidence agree."""

    items = list(components)
    if not items:
        return ()
    if not config.enabled:
        buckets = [[index] for index in range(len(items))]
    else:
        parents = list(range(len(items)))
        sizes = [1] * len(items)

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root and sizes[left_root] + sizes[right_root] <= config.maximum_components_per_group:
                parents[right_root] = left_root
                sizes[left_root] += sizes[right_root]

        for left in range(len(items)):
            for right in range(left + 1, len(items)):
                a, b = items[left], items[right]
                h_gap, v_gap = _box_gap(a.bounding_box, b.bounding_box)
                distance = math.dist(a.centroid, b.centroid)
                spatial = (
                    h_gap <= config.max_horizontal_gap_pixels
                    and v_gap <= config.max_vertical_gap_pixels
                    and distance <= config.max_centroid_distance_pixels
                )
                margin = config.expanded_box_margin_pixels
                expanded = h_gap <= margin * 2 and v_gap <= margin * 2
                if (spatial or expanded) and _movement_consistent(a, b, config):
                    union(left, right)
        grouped: dict[int, list[int]] = {}
        for index in range(len(items)):
            grouped.setdefault(find(index), []).append(index)
        buckets = list(grouped.values())

    results: list[GroupedCandidate] = []
    for bucket in buckets:
        selected = tuple(items[index] for index in bucket[: config.maximum_components_per_group])
        left = min(item.bounding_box[0] for item in selected)
        top = min(item.bounding_box[1] for item in selected)
        right = max(item.bounding_box[0] + item.bounding_box[2] for item in selected)
        bottom = max(item.bounding_box[1] + item.bounding_box[3] for item in selected)
        foreground = sum(item.foreground_pixels for item in selected)
        weight = max(1, foreground)
        centroid = (
            sum(item.centroid[0] * item.foreground_pixels for item in selected) / weight,
            sum(item.centroid[1] * item.foreground_pixels for item in selected) / weight,
        )
        if len(selected) == 1:
            confidence = 1.0
        else:
            distances = [math.dist(a.centroid, b.centroid) for index, a in enumerate(selected) for b in selected[index + 1 :]]
            confidence = max(0.35, min(1.0, 1.0 - (sum(distances) / len(distances)) / max(1.0, config.max_centroid_distance_pixels * 1.5)))
        results.append(GroupedCandidate(
            selected,
            (left, top, right - left, bottom - top),
            centroid,
            sum(item.contour_area for item in selected),
            foreground,
            100.0 * foreground / max(1, frame_area),
            100.0 * foreground / max(1, zone_area),
            confidence,
            dispersed_motion=len(selected) >= 5 and confidence < 0.65,
        ))
    return tuple(results)


def classify_candidate(candidate: GroupedCandidate, frame_shape: tuple[int, int], config: ClassificationConfig) -> GroupedCandidate:
    """Assign an explicitly provisional size/motion label; never a species label."""

    height_percent = 100.0 * candidate.height / max(1, frame_shape[0])
    if candidate.dispersed_motion or (len(candidate.components) >= config.flicker_min_components and not candidate.coherent_motion):
        category = "plant_or_shadow_flicker"
    elif candidate.frame_percent <= config.tiny_max_frame_percent:
        category = "tiny_motion"
    elif height_percent >= config.person_min_height_percent and candidate.aspect_ratio < 1.4:
        category = "person_sized"
    elif candidate.frame_percent >= config.large_object_min_frame_percent:
        category = "large_object"
    elif candidate.frame_percent <= config.small_animal_max_frame_percent:
        category = "small_animal_candidate"
    elif candidate.frame_percent <= config.medium_animal_max_frame_percent:
        category = "medium_animal_candidate"
    else:
        category = "unclassified_motion"

    speed = candidate.average_speed
    if speed <= config.stationary_speed_pixels_per_second:
        speed_attribute = "mostly_stationary"
    elif speed < config.slow_speed_pixels_per_second:
        speed_attribute = "slow"
    elif speed >= config.fast_speed_pixels_per_second:
        speed_attribute = "fast"
    else:
        speed_attribute = "moderate"
    shape_attribute = "dispersed_motion" if candidate.dispersed_motion else ("coherent_travel" if candidate.coherent_motion else None)
    attributes = tuple(item for item in (speed_attribute, shape_attribute) if item)
    score = min(0.95, 0.35 + 0.08 * candidate.persistence_count + 0.25 * candidate.grouping_confidence)
    return replace(candidate, provisional_category=category, movement_attributes=attributes, heuristic_score=score)


def evaluate_event_eligibility(candidate: GroupedCandidate, config: CandidateFilterConfig) -> GroupedCandidate:
    """Keep weak candidates observable while preventing them from creating events."""

    reason: str | None = None
    if config.enabled:
        if config.ignore_tiny_motion and candidate.provisional_category == "tiny_motion":
            reason = "tiny_motion"
        elif config.ignore_plant_or_shadow_flicker and candidate.provisional_category == "plant_or_shadow_flicker":
            reason = "plant_or_shadow_flicker"
        elif candidate.frame_percent < config.minimum_frame_percent:
            reason = "below_minimum_frame_percent"
        elif candidate.provisional_category == "small_animal_candidate":
            if config.require_coherent_small_motion and not candidate.coherent_motion:
                reason = "small_motion_not_coherent"
            elif candidate.travel_distance < config.small_motion_minimum_travel_pixels:
                reason = "small_motion_insufficient_travel"
    return replace(candidate, event_eligible=reason is None, event_filter_reason=reason)


def _direction(path: list[tuple[float, float]]) -> str:
    if len(path) < 2:
        return "stationary"
    dx, dy = path[-1][0] - path[0][0], path[-1][1] - path[0][1]
    if math.hypot(dx, dy) < 5:
        return "stationary"
    horizontal = "right" if dx > 0 else "left"
    vertical = "down" if dy > 0 else "up"
    return horizontal if abs(dx) > abs(dy) * 1.8 else (vertical if abs(dy) > abs(dx) * 1.8 else f"{vertical}-{horizontal}")


class MotionWatcherDetector:
    """MOG2 detector with time warmup, global rejection, grouping, and multi-tracking."""

    def __init__(self, config: MotionConfig, *, subtractor: Any | None = None) -> None:
        self.config = config
        self._subtractor = subtractor or cv2.createBackgroundSubtractorMOG2(
            history=config.history, varThreshold=config.variance_threshold, detectShadows=config.detect_shadows
        )
        self._started_at: float | None = None
        self._frame_count = 0
        self._last_processed_at: float | None = None
        self._processing_fps = 0.0
        self._previous_luminance: float | None = None
        self._previous_colorfulness: float | None = None
        self._recovery_until = 0.0
        self._tracks: dict[int, _Track] = {}
        self._next_track_id = 1
        self._last_confirmation_at: float | None = None
        self._previous_components: list[tuple[tuple[float, float], float]] = []

    def clear_candidates(self) -> None:
        self._tracks.clear()
        self._previous_components.clear()

    def process(self, frame: np.ndarray, *, now: float | None = None) -> WatchDetectionResult:
        current = monotonic() if now is None else now
        if self._started_at is None:
            self._started_at = current
        if self._last_processed_at is not None and current > self._last_processed_at:
            instant = 1.0 / (current - self._last_processed_at)
            self._processing_fps = instant if not self._processing_fps else 0.15 * instant + 0.85 * self._processing_fps
        self._last_processed_at = current

        target_width = min(self.config.processing_width, frame.shape[1])
        scale = target_width / frame.shape[1]
        reduced = cv2.resize(frame, (target_width, max(1, round(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        blurred = cv2.GaussianBlur(reduced, (self.config.blur_kernel, self.config.blur_kernel), 0)
        raw = self._subtractor.apply(blurred)
        _, foreground = cv2.threshold(raw, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.config.morphology_kernel, self.config.morphology_kernel))
        cleaned = foreground
        if self.config.open_iterations:
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=self.config.open_iterations)
        if self.config.close_iterations:
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=self.config.close_iterations)
        zone_mask = self._zone_mask(cleaned.shape)
        measurement = self._measure_global(reduced, raw, cleaned, zone_mask)
        self._frame_count += 1

        if not self.config.enabled:
            state = WatchState.DISABLED
        elif current - self._started_at < self.config.warmup.seconds or self._frame_count < self.config.warmup.minimum_frames:
            state = WatchState.WARMING_UP
            self.clear_candidates()
        elif measurement.reason is not None:
            state = WatchState.GLOBAL_RECOVERY
            self._recovery_until = current + self.config.global_rejection.recovery_seconds
            self.clear_candidates()
        elif current < self._recovery_until:
            state = WatchState.GLOBAL_RECOVERY
            self.clear_candidates()
        else:
            state = WatchState.READY

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if state is not WatchState.READY:
            return WatchDetectionResult(state, (), raw, cleaned, zone_mask, measurement, len(contours), self._processing_fps)

        inverse = 1.0 / scale
        components: list[MotionComponent] = []
        zone_foreground = cv2.bitwise_and(cleaned, zone_mask)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.config.min_blob_area or area > self.config.max_blob_area:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] <= 0:
                continue
            center_reduced = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
            if zone_mask[min(zone_mask.shape[0] - 1, int(center_reduced[1])), min(zone_mask.shape[1] - 1, int(center_reduced[0]))] == 0:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            local = zone_foreground[y : y + height, x : x + width]
            points = tuple((round(point[0][0] * inverse), round(point[0][1] * inverse)) for point in contour[:: max(1, len(contour) // 40)])
            components.append(MotionComponent(
                (round(x * inverse), round(y * inverse), max(1, round(width * inverse)), max(1, round(height * inverse))),
                area * inverse * inverse,
                int(cv2.countNonZero(local) * inverse * inverse),
                (center_reduced[0] * inverse, center_reduced[1] * inverse),
                points,
            ))

        moving_components: list[MotionComponent] = []
        for component in components:
            nearby = [item for item in self._previous_components if current - item[1] <= self.config.persistence.maximum_gap_seconds and math.dist(item[0], component.centroid) <= self.config.persistence.max_centroid_distance_pixels]
            if nearby:
                previous_center, previous_at = min(nearby, key=lambda item: math.dist(item[0], component.centroid))
                elapsed = max(1e-6, current - previous_at)
                velocity = ((component.centroid[0] - previous_center[0]) / elapsed, (component.centroid[1] - previous_center[1]) / elapsed)
                moving_components.append(replace(component, velocity=velocity))
            else:
                moving_components.append(component)
        components = moving_components
        self._previous_components = [(item.centroid, current) for item in components]

        zone_area_original = max(1, int(cv2.countNonZero(zone_mask) * inverse * inverse))
        groups = list(group_components(components, self.config.grouping, frame.shape[0] * frame.shape[1], zone_area_original))
        full_zone = cv2.resize(zone_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        boundary = cv2.subtract(full_zone, cv2.erode(full_zone, np.ones((5, 5), dtype=np.uint8)))
        groups = [replace(group, touched_zone_boundary=self._box_touches_mask(group.bounding_box, boundary)) for group in groups]
        tracked = self._update_tracks(groups, current, frame.shape[:2])
        return WatchDetectionResult(state, tuple(tracked), raw, cleaned, zone_mask, measurement, len(contours), self._processing_fps)

    def _zone_mask(self, shape: tuple[int, int]) -> np.ndarray:
        height, width = shape
        mask = np.zeros(shape, dtype=np.uint8)
        if not self.config.inclusion_zone.enabled:
            mask[:] = 255
            return mask
        points = np.array([[min(width - 1, round(x * width)), min(height - 1, round(y * height))] for x, y in self.config.inclusion_zone.polygon], dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
        return mask

    def _measure_global(self, frame: np.ndarray, raw: np.ndarray, cleaned: np.ndarray, zone_mask: np.ndarray) -> GlobalMotionMeasurement:
        pixels = max(1, raw.size)
        raw_percent = 100.0 * cv2.countNonZero(raw) / pixels
        cleaned_percent = 100.0 * cv2.countNonZero(cleaned) / pixels
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidate_pixels = sum(cv2.contourArea(contour) for contour in contours if cv2.contourArea(contour) >= self.config.min_blob_area)
        candidate_percent = 100.0 * candidate_pixels / pixels
        zone_area = max(1, cv2.countNonZero(zone_mask))
        zone_percent = 100.0 * cv2.countNonZero(cv2.bitwise_and(cleaned, zone_mask)) / zone_area
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        luminance = float(np.mean(gray))
        channels = cv2.split(frame.astype(np.float32))
        colorfulness = float(np.mean(np.abs(channels[2] - channels[1]) + np.abs(channels[1] - channels[0])))
        luminance_delta = 0.0 if self._previous_luminance is None else abs(luminance - self._previous_luminance)
        color_delta = 0.0 if self._previous_colorfulness is None else abs(colorfulness - self._previous_colorfulness)
        previous_color = self._previous_colorfulness
        self._previous_luminance, self._previous_colorfulness = luminance, colorfulness

        reason: str | None = None
        settings = self.config.global_rejection
        over = settings.enabled and (cleaned_percent >= settings.max_frame_motion_percent or zone_percent >= settings.max_zone_motion_percent)
        if over:
            abrupt_mono_transition = previous_color is not None and color_delta >= settings.ir_colorfulness_delta and (colorfulness < 8.0 or previous_color < 8.0)
            if abrupt_mono_transition and luminance_delta >= settings.exposure_luminance_delta * 0.35:
                reason = "probable_ir_mode_switch"
            elif luminance_delta >= settings.exposure_luminance_delta:
                reason = "probable_exposure_change"
            elif cleaned_percent >= settings.obstruction_motion_percent:
                reason = "probable_scene_obstruction"
            elif len(contours) >= 12 and cleaned_percent >= settings.max_frame_motion_percent:
                reason = "probable_camera_movement"
            elif zone_percent >= settings.max_zone_motion_percent:
                reason = "excessive_zone_motion"
            elif cleaned_percent >= settings.max_frame_motion_percent:
                reason = "excessive_frame_motion"
            else:
                reason = "global_motion_unclassified"
        return GlobalMotionMeasurement(raw_percent, cleaned_percent, candidate_percent, zone_percent, len(contours), reason, luminance_delta, color_delta)

    @staticmethod
    def _box_touches_mask(box: tuple[int, int, int, int], mask: np.ndarray) -> bool:
        x, y, width, height = box
        x, y = max(0, x), max(0, y)
        right, bottom = min(mask.shape[1], x + width), min(mask.shape[0], y + height)
        return right > x and bottom > y and cv2.countNonZero(mask[y:bottom, x:right]) > 0

    def _update_tracks(self, groups: list[GroupedCandidate], now: float, frame_shape: tuple[int, int]) -> list[GroupedCandidate]:
        settings = self.config.persistence
        available = set(self._tracks)
        output: list[GroupedCandidate] = []
        for group in sorted(groups, key=lambda item: item.foreground_pixels, reverse=True):
            matches = [
                track for track in self._tracks.values()
                if track.track_id in available
                and now - track.last_at <= settings.maximum_gap_seconds
                and math.dist(track.path[-1], group.centroid) <= settings.max_centroid_distance_pixels
            ]
            if matches:
                track = min(matches, key=lambda item: math.dist(item.path[-1], group.centroid))
                available.remove(track.track_id)
                elapsed = max(1e-6, now - track.last_at)
                speed = math.dist(track.path[-1], group.centroid) / elapsed
                track.speeds.append(speed)
                track.persistence += 1
                track.last_at = now
                track.path.append(group.centroid)
                track.areas.append(group.foreground_pixels)
            else:
                track = _Track(self._next_track_id, now, now, 1, [group.centroid], [], [group.foreground_pixels])
                self._tracks[track.track_id] = track
                self._next_track_id += 1
            path = track.path[-30:]
            segment_distances = [math.dist(path[index - 1], path[index]) for index in range(1, len(path))]
            travel = sum(segment_distances)
            displacement = math.dist(path[0], path[-1]) if len(path) > 1 else 0.0
            coherent = travel > 5 and displacement / travel >= 0.65
            updated = replace(
                group,
                track_id=track.track_id,
                persistence_count=track.persistence,
                path=tuple(path),
                duration=now - track.first_at,
                travel_distance=travel,
                average_speed=sum(track.speeds) / len(track.speeds) if track.speeds else 0.0,
                peak_speed=max(track.speeds, default=0.0),
                direction=_direction(path),
                coherent_motion=coherent,
                dispersed_motion=group.dispersed_motion or (len(group.components) >= 5 and not coherent),
            )
            classified = classify_candidate(updated, frame_shape, self.config.classification)
            filtered = evaluate_event_eligibility(classified, self.config.candidate_filter)
            can_confirm = self._last_confirmation_at is None or now - self._last_confirmation_at >= settings.cooldown_seconds
            newly_confirmed = (
                not track.confirmed
                and track.persistence >= settings.frames
                and can_confirm
                and filtered.event_eligible
            )
            if newly_confirmed:
                track.confirmed = True
                self._last_confirmation_at = now
            output.append(replace(filtered, confirmed=track.confirmed, newly_confirmed=newly_confirmed))
        expired = [track_id for track_id, track in self._tracks.items() if now - track.last_at > settings.maximum_gap_seconds]
        for track_id in expired:
            del self._tracks[track_id]
        return output


def annotate_watch_frame(frame: np.ndarray, result: WatchDetectionResult, *, event_id: str | None = None, measured_fps: float = 0.0) -> np.ndarray:
    """Draw the inclusion polygon, grouped objects, components, paths, and metrics."""

    annotated = frame.copy()
    height, width = frame.shape[:2]
    if result.zone_mask.shape != frame.shape[:2]:
        zone = cv2.resize(result.zone_mask, (width, height), interpolation=cv2.INTER_NEAREST)
    else:
        zone = result.zone_mask
    zone_contours, _ = cv2.findContours(zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(annotated, zone_contours, -1, (255, 180, 0), 2)
    for group in result.groups:
        color = (30, 220, 80) if group.confirmed else ((120, 120, 120) if not group.event_eligible else (0, 190, 255))
        x, y, box_width, box_height = group.bounding_box
        cv2.rectangle(annotated, (x, y), (x + box_width, y + box_height), color, 3)
        for component in group.components:
            cx, cy, cw, ch = component.bounding_box
            cv2.rectangle(annotated, (cx, cy), (cx + cw, cy + ch), (220, 130, 30), 1)
        center = (round(group.centroid[0]), round(group.centroid[1]))
        cv2.circle(annotated, center, 5, color, -1)
        if len(group.path) > 1:
            cv2.polylines(annotated, [np.array(group.path, dtype=np.int32)], False, color, 2)
        label = f"{group.provisional_category} | {','.join(group.movement_attributes)} | {group.average_speed:.1f}px/s"
        cv2.putText(annotated, label, (x, max(22, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)
        filter_detail = f" | FILTERED {group.event_filter_reason}" if group.event_filter_reason else ""
        details = f"area {group.foreground_pixels}px | {group.frame_percent:.2f}% frame | {group.duration:.1f}s | {len(group.components)} blobs | group {group.grouping_confidence:.2f}{filter_detail}"
        cv2.putText(annotated, details, (x, min(height - 12, y + box_height + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1, cv2.LINE_AA)
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [
        timestamp,
        f"{result.state.value} | measured {measured_fps:.1f} FPS | motion {result.global_motion.cleaned_foreground_percent:.1f}%",
    ]
    if event_id:
        lines.append(f"event {event_id}")
    if result.global_motion.reason:
        lines.append(f"REJECTED: {result.global_motion.reason}")
    cv2.rectangle(annotated, (8, 8), (min(width - 8, 760), 18 + 27 * len(lines)), (0, 0, 0), -1)
    for index, line in enumerate(lines):
        cv2.putText(annotated, line, (16, 31 + 25 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return annotated
