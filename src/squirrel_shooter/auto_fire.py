"""Fail-closed, event-driven policy for computer-aided automatic engagement."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from .classifier_labels import VOC_LABELS


LOGGER = logging.getLogger(__name__)
HUMAN_DENY_LABELS = frozenset({"person"})
_MODEL_LABELS = frozenset(VOC_LABELS[1:])
_UNKNOWN_LABELS = frozenset({"background", "unknown", "unclassified", "no-result", "no_result"})
_SAFE_EVENT_ID = re.compile(r"[A-Za-z0-9_-]+")
_RATE_LIMIT_SCHEMA_VERSION = 2
_LEGACY_RATE_LIMIT_SCHEMA_VERSION = 1
_ROLLING_WINDOW_SECONDS = 3600.0
_WALL_CLOCK_JUMP_TOLERANCE_SECONDS = 5.0
_KNOWN_COORDINATOR_REASONS = frozenset(
    {
        "cooldown_active",
        "coordinator_busy",
        "calibration_frame_mismatch",
        "calibration_frame_unknown",
        "hardware_not_ready",
        "invalid_request",
        "interpolation_failed",
        "movement_failed",
        "outside_safe_bounds",
        "park_failed",
        "recording_failed",
        "recording_unavailable",
        "safety_state_invalid",
        "valve_disabled",
        "valve_failure",
    }
)


def _finite_number(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")
    normalized = float(value)
    if minimum is not None and (
        normalized < minimum or (exclusive_minimum and normalized == minimum)
    ):
        qualifier = "greater than" if exclusive_minimum else "at least"
        raise ValueError(f"{name} must be {qualifier} {minimum}")
    if maximum is not None and normalized > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return normalized


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class AutoFireConfig:
    enabled: bool = False
    min_confidence: float = 0.75
    allowed_classes: tuple[str, ...] = ()
    cooldown_seconds: float = 5.0
    max_shots_per_event: int = 1
    minimum_reengagement_seconds: float = 60.0
    max_shots_per_hour: int = 6
    classification_max_age_seconds: float = 3.0
    target_max_age_seconds: float = 0.75
    track_loss_grace_seconds: float = 0.9
    reacquisition_max_centroid_distance_pixels: float = 100.0
    reacquisition_max_area_ratio: float = 2.5
    rate_limit_state_file: Path = Path("captures/auto-fire-rate-limit.json")

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be true or false")
        _finite_number(self.min_confidence, "min_confidence", minimum=0.70, maximum=1.0)
        if not isinstance(self.allowed_classes, tuple):
            raise ValueError("allowed_classes must be a tuple")
        if len(set(self.allowed_classes)) != len(self.allowed_classes):
            raise ValueError("allowed_classes must not contain duplicates")
        for label in self.allowed_classes:
            if not isinstance(label, str) or not label or label != label.strip().lower():
                raise ValueError("allowed_classes must contain canonical lowercase labels")
            if label not in _MODEL_LABELS:
                raise ValueError(f"allowed_classes contains unsupported classifier label: {label}")
            if label in HUMAN_DENY_LABELS:
                raise ValueError(f"allowed_classes must never contain human deny label: {label}")
        if self.enabled and not self.allowed_classes:
            raise ValueError("enabled auto-fire requires at least one allowed class")
        _finite_number(self.cooldown_seconds, "cooldown_seconds", minimum=0.0, exclusive_minimum=True)
        _positive_integer(self.max_shots_per_event, "max_shots_per_event")
        _finite_number(
            self.minimum_reengagement_seconds,
            "minimum_reengagement_seconds",
            minimum=0.0,
        )
        _positive_integer(self.max_shots_per_hour, "max_shots_per_hour")
        _finite_number(
            self.classification_max_age_seconds,
            "classification_max_age_seconds",
            minimum=0.0,
            exclusive_minimum=True,
        )
        _finite_number(
            self.target_max_age_seconds,
            "target_max_age_seconds",
            minimum=0.0,
            exclusive_minimum=True,
        )
        _finite_number(
            self.track_loss_grace_seconds,
            "track_loss_grace_seconds",
            minimum=0.0,
            exclusive_minimum=True,
        )
        _finite_number(
            self.reacquisition_max_centroid_distance_pixels,
            "reacquisition_max_centroid_distance_pixels",
            minimum=0.0,
            exclusive_minimum=True,
        )
        _finite_number(
            self.reacquisition_max_area_ratio,
            "reacquisition_max_area_ratio",
            minimum=1.0,
        )
        if self.track_loss_grace_seconds > self.classification_max_age_seconds:
            raise ValueError(
                "track_loss_grace_seconds must not exceed classification_max_age_seconds"
            )
        if not isinstance(self.rate_limit_state_file, Path) or not self.rate_limit_state_file.name:
            raise ValueError("rate_limit_state_file must be a file path")


@dataclass(frozen=True)
class AutoFireTargetSnapshot:
    event_id: str
    track_id: int
    observed_monotonic: float
    pixel_x: int
    pixel_y: int
    bounding_box: tuple[int, int, int, int]
    frame_width: int
    frame_height: int
    confirmed: bool
    event_eligible: bool
    provisional_category: str

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or _SAFE_EVENT_ID.fullmatch(self.event_id) is None:
            raise ValueError("event_id must be a non-empty safe identifier")
        _positive_integer(self.track_id, "track_id")
        _finite_number(self.observed_monotonic, "observed_monotonic", minimum=0.0)
        for name, value in (("pixel_x", self.pixel_x), ("pixel_y", self.pixel_y)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name, value in (("frame_width", self.frame_width), ("frame_height", self.frame_height)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.pixel_x >= self.frame_width or self.pixel_y >= self.frame_height:
            raise ValueError("target pixel must be inside the native camera frame")
        if (
            not isinstance(self.bounding_box, tuple)
            or len(self.bounding_box) != 4
            or any(isinstance(value, bool) or not isinstance(value, int) for value in self.bounding_box)
        ):
            raise ValueError("bounding_box must contain four integers")
        x, y, width, height = self.bounding_box
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError("bounding_box must be a positive in-frame rectangle")
        if x + width > self.frame_width or y + height > self.frame_height:
            raise ValueError("bounding_box must fit inside the native camera frame")
        if not isinstance(self.confirmed, bool) or not isinstance(self.event_eligible, bool):
            raise ValueError("confirmed and event_eligible must be booleans")
        if not isinstance(self.provisional_category, str) or not self.provisional_category:
            raise ValueError("provisional_category must be a non-empty string")


@dataclass(frozen=True)
class AutoFireTargetAssociation:
    """Non-fireable lifecycle state for an event/track target association."""

    state: str
    expires_monotonic: float | None = None

    def __post_init__(self) -> None:
        if self.state not in {
            "coasting",
            "reacquisition_ambiguous",
            "reacquisition_incompatible",
            "reacquisition_timeout",
        }:
            raise ValueError("unsupported auto-fire target association state")
        if self.state == "coasting":
            _finite_number(
                self.expires_monotonic,
                "expires_monotonic",
                minimum=0.0,
            )
        elif self.expires_monotonic is not None:
            raise ValueError("only a coasting association may have an expiry")


@dataclass(frozen=True)
class AutoFireDetection:
    label: str
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label or self.label != self.label.strip().lower():
            raise ValueError("label must be a canonical lowercase value")
        _finite_number(self.confidence, "confidence", minimum=0.0, maximum=1.0)


class AutoFireCoordinator(Protocol):
    def automatic_engage(
        self,
        pixel_x: int,
        pixel_y: int,
        *,
        frame_width: int,
        frame_height: int,
        cooldown_seconds: float,
        final_safety_check: Callable[[], bool],
        evidence: dict[str, Any],
    ) -> object:
        ...

    def cooldown_remaining_seconds(self) -> float:
        ...


@dataclass(frozen=True)
class AutoFireDecision:
    accepted: bool
    reason: str
    event_id: str | None = None
    track_id: int | None = None
    classifier_label: str | None = None
    classifier_confidence: float | None = None
    target_pixel_x: int | None = None
    target_pixel_y: int | None = None


@dataclass(frozen=True)
class _ShotRecord:
    event_id: str
    track_id: int
    accepted_at_epoch_seconds: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _HeldClassification:
    event_id: str
    track_id: int
    classified_observation_monotonic: float
    detections: tuple[AutoFireDetection, ...]
    error: str | None


TargetProvider = Callable[
    [str, int],
    AutoFireTargetSnapshot | AutoFireTargetAssociation | None,
]
NightModeProvider = Callable[[], bool]


class AutoFireService:
    """Evaluate one already-qualified event without owning any hardware or vision loop."""

    def __init__(
        self,
        config: AutoFireConfig,
        coordinator: AutoFireCoordinator,
        target_provider: TargetProvider,
        night_mode_provider: NightModeProvider,
        *,
        monotonic_clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(config, AutoFireConfig):
            raise TypeError("config must be an AutoFireConfig")
        if not callable(target_provider) or not callable(night_mode_provider):
            raise TypeError("target_provider and night_mode_provider must be callable")
        if not callable(monotonic_clock) or not callable(wall_clock):
            raise TypeError("clocks must be callable")
        self.config = config
        self.coordinator = coordinator
        self._target_provider = target_provider
        self._night_mode_provider = night_mode_provider
        self._monotonic_clock = monotonic_clock
        self._wall_clock = wall_clock
        self._lock = threading.RLock()
        self._reacquisition_condition = threading.Condition(self._lock)
        self._target_state_generation = 0
        self._shots: list[_ShotRecord] = []
        self._state_error: str | None = None
        self._rate_clock_wall = self._read_clock(self._wall_clock)
        self._rate_clock_monotonic = self._read_clock(self._monotonic_clock)
        self._rate_clock_boot_id = self._system_boot_id()
        self._persisted_rate_clock: tuple[float, float, str | None] | None = None
        self._human_denied_events: dict[str, float] = {}
        self._held_classifications: dict[tuple[str, int], _HeldClassification] = {}
        self._pending: tuple[str, int] | None = None
        self._shutting_down = False
        self._candidates_evaluated = 0
        self._accepted = 0
        self._rejected = 0
        self._rejection_counts: dict[str, int] = {}
        self._last_decision: AutoFireDecision | None = None
        self._load_rate_limit_state()
        self._validate_initial_rate_clock()
        self._preflight_rate_limit_state()

    def start_accepting(self) -> None:
        """Allow candidates after a deliberate runtime start or restart."""

        with self._lock:
            self._shutting_down = False

    def begin_shutdown(self) -> None:
        """Reject new work and invalidate any pending final pre-valve check."""

        with self._lock:
            self._shutting_down = True
            self._target_state_generation += 1
            self._reacquisition_condition.notify_all()

    def notify_target_state_changed(self) -> None:
        """Wake a classifier callback waiting for bounded live-target reacquisition."""

        with self._lock:
            self._target_state_generation += 1
            self._reacquisition_condition.notify_all()

    def handle_classification(
        self,
        *,
        event_id: str,
        track_id: int,
        classified_observation_monotonic: float,
        detections: Iterable[AutoFireDetection] | None,
        error: str | None = None,
    ) -> AutoFireDecision:
        """Evaluate one asynchronous classification and synchronously request one safe engagement."""

        return self._evaluate_classification(
            event_id=event_id,
            track_id=track_id,
            classified_observation_monotonic=classified_observation_monotonic,
            detections=detections,
            error=error,
            count_candidate=True,
        )

    def _evaluate_classification(
        self,
        *,
        event_id: str,
        track_id: int,
        classified_observation_monotonic: float,
        detections: Iterable[AutoFireDetection] | None,
        error: str | None,
        count_candidate: bool,
    ) -> AutoFireDecision:
        """Evaluate new or safely resumed classifier evidence."""

        if count_candidate:
            with self._lock:
                self._candidates_evaluated += 1

        if not self._valid_event_and_track(event_id, track_id):
            return self._reject("target_association_invalid")

        parsed, parse_reason = self._parse_detections(detections)
        top = max(parsed, key=lambda item: item.confidence) if parsed else None
        human_detected = any(item.label in HUMAN_DENY_LABELS for item in parsed)
        if human_detected:
            self._latch_human(event_id)

        decision_fields = {
            "event_id": event_id,
            "track_id": track_id,
            "classifier_label": None if top is None else top.label,
            "classifier_confidence": None if top is None else top.confidence,
        }
        if not self.config.enabled:
            return self._reject("feature_disabled", **decision_fields)
        with self._lock:
            if self._shutting_down:
                return self._reject_locked("service_stopping", **decision_fields)
            if self._state_error is not None:
                return self._reject_locked("safety_state_invalid", **decision_fields)
        if parse_reason is not None:
            return self._reject(parse_reason, **decision_fields)
        if human_detected or self._is_human_latched(event_id):
            return self._reject("human_detected", **decision_fields)
        if error is not None:
            if not isinstance(error, str) or not error.strip():
                return self._reject("classification_invalid", **decision_fields)
            return self._reject("classification_error", **decision_fields)
        if top is None:
            return self._reject("classification_unknown", **decision_fields)
        if top.label in _UNKNOWN_LABELS:
            return self._reject("classification_unknown", **decision_fields)
        if top.label not in _MODEL_LABELS:
            return self._reject("classification_invalid", **decision_fields)
        if top.confidence <= self.config.min_confidence:
            return self._reject("below_confidence", **decision_fields)
        if top.label not in self.config.allowed_classes:
            return self._reject("class_not_allowlisted", **decision_fields)

        now_monotonic = self._read_clock(self._monotonic_clock)
        if now_monotonic is None or not self._fresh(
            now_monotonic,
            classified_observation_monotonic,
            self.config.classification_max_age_seconds,
        ):
            return self._reject("stale_classification", **decision_fields)
        night = self._night_mode()
        if night is None:
            return self._reject("safety_state_invalid", **decision_fields)
        if night:
            return self._reject("night_mode", **decision_fields)

        target, target_reason = self._current_target(event_id, track_id, now_monotonic)
        if target is None:
            if target_reason == "target_reacquisition_pending":
                held = _HeldClassification(
                    event_id,
                    track_id,
                    float(classified_observation_monotonic),
                    tuple(parsed),
                    error,
                )
                with self._lock:
                    self._held_classifications[(event_id, track_id)] = held
                LOGGER.info(
                    "AUTO_FIRE classification held for target reacquisition event=%s track=%s class=%s confidence=%.3f",
                    event_id,
                    track_id,
                    top.label,
                    top.confidence,
                    extra={
                        "structured_data": {
                            "event": "auto_fire_classification_held",
                            **decision_fields,
                            "classified_observation_monotonic": classified_observation_monotonic,
                        }
                    },
                )
                return self._wait_for_target_reacquisition(
                    held,
                    decision_fields,
                )
            return self._reject(target_reason, **decision_fields)

        cooldown = self._coordinator_cooldown()
        if cooldown is None:
            return self._reject("hardware_not_ready", **decision_fields)
        if cooldown > 0:
            return self._reject("cooldown_active", **decision_fields)

        now_wall = self._rate_wall_now()
        if now_wall is None:
            return self._reject("safety_state_invalid", **decision_fields)
        with self._lock:
            rate_reason = self._rate_limit_reason_locked(event_id, track_id, now_wall)
            if rate_reason is not None:
                return self._reject_locked(rate_reason, **decision_fields)
            if self._pending is not None:
                return self._reject_locked("coordinator_busy", **decision_fields)
            self._pending = (event_id, track_id)

        aimed_pixel = (target.pixel_x, target.pixel_y)
        evidence = {
            "event_type": "auto_fire",
            "source_event_id": event_id,
            "track_id": track_id,
            "classifier_label": top.label,
            "classifier_confidence": top.confidence,
            "classified_observation_monotonic": float(classified_observation_monotonic),
            "target_observed_monotonic": target.observed_monotonic,
            "target_pixel_x": target.pixel_x,
            "target_pixel_y": target.pixel_y,
            "target_frame_width": target.frame_width,
            "target_frame_height": target.frame_height,
            "target_bounding_box": {
                "x": target.bounding_box[0],
                "y": target.bounding_box[1],
                "width": target.bounding_box[2],
                "height": target.bounding_box[3],
            },
            "cooldown_seconds": self.config.cooldown_seconds,
            "acceptance_reason": "allowlisted_classification",
        }
        LOGGER.info(
            "AUTO_FIRE candidate event=%s track=%s class=%s confidence=%.3f x=%d y=%d",
            event_id,
            track_id,
            top.label,
            top.confidence,
            target.pixel_x,
            target.pixel_y,
            extra={"structured_data": {"event": "auto_fire_candidate", **evidence}},
        )

        final_check_called = False
        final_rejection_reason: str | None = None

        def final_safety_check() -> bool:
            nonlocal final_check_called, final_rejection_reason
            final_check_called = True
            final_rejection_reason = self._final_safety_reason(
                event_id,
                track_id,
                classified_observation_monotonic,
                aimed_pixel,
                (target.frame_width, target.frame_height),
            )
            return final_rejection_reason is None

        try:
            result = self.coordinator.automatic_engage(
                target.pixel_x,
                target.pixel_y,
                frame_width=target.frame_width,
                frame_height=target.frame_height,
                cooldown_seconds=self.config.cooldown_seconds,
                final_safety_check=final_safety_check,
                evidence=evidence,
            )
        except Exception as exc:
            shot_attempted = getattr(exc, "shot_attempted", False) is True
            persistence_error: str | None = None
            with self._lock:
                self._pending = None
                if shot_attempted:
                    attempted_at = self._rate_wall_now()
                    if attempted_at is None:
                        attempted_at = now_wall
                    self._shots.append(_ShotRecord(event_id, track_id, attempted_at))
                    try:
                        self._persist_rate_limit_state_locked()
                    except Exception as persist_exc:
                        persistence_error = f"{type(persist_exc).__name__}: {persist_exc}"
                    failure = f"{type(exc).__name__}: {exc}"
                    self._state_error = (
                        "Automatic engagement failed after a valve actuation was attempted: "
                        f"{failure}"
                    )
                    if persistence_error is not None:
                        self._state_error += f"; rate-limit persistence failed: {persistence_error}"
            if shot_attempted:
                LOGGER.error(
                    "AUTO_FIRE actuation attempt failed event=%s track=%s persistence_error=%s",
                    event_id,
                    track_id,
                    persistence_error,
                    extra={
                        "structured_data": {
                            "event": "auto_fire_actuation_attempt_failed",
                            **evidence,
                            "error": f"{type(exc).__name__}: {exc}",
                            "rate_limit_persisted": persistence_error is None,
                        }
                    },
                )
            reason = final_rejection_reason or self._coordinator_exception_reason(exc)
            return self._reject(reason, **decision_fields, target_pixel_x=target.pixel_x, target_pixel_y=target.pixel_y)

        if not final_check_called:
            with self._lock:
                self._pending = None
                self._state_error = "Coordinator returned without invoking final_safety_check"
            return self._reject(
                "safety_state_invalid",
                **decision_fields,
                target_pixel_x=target.pixel_x,
                target_pixel_y=target.pixel_y,
            )
        unsuccessful_reason = self._unsuccessful_result_reason(result)
        if final_rejection_reason is not None or unsuccessful_reason is not None:
            with self._lock:
                self._pending = None
            return self._reject(
                final_rejection_reason or unsuccessful_reason or "safety_state_invalid",
                **decision_fields,
                target_pixel_x=target.pixel_x,
                target_pixel_y=target.pixel_y,
            )

        recording_queued = getattr(result, "recording_queued", None)
        recording_failed = recording_queued is False
        accepted_at = self._rate_wall_now()
        persistence_error: str | None = None
        with self._lock:
            self._pending = None
            safety_errors: list[str] = []
            if accepted_at is None:
                accepted_at = now_wall
                safety_errors.append("Wall clock failed after an automatic engagement")
            self._shots.append(_ShotRecord(event_id, track_id, accepted_at))
            try:
                self._persist_rate_limit_state_locked()
            except Exception as exc:
                persistence_error = f"{type(exc).__name__}: {exc}"
                safety_errors.append(
                    f"Could not persist accepted automatic engagement: {persistence_error}"
                )
            if recording_failed:
                safety_errors.append(
                    "Accepted automatic engagement could not queue its evidence recording"
                )
            if safety_errors:
                self._state_error = "; ".join(safety_errors)
            decision = AutoFireDecision(
                True,
                "accepted_recording_failed" if recording_failed else "accepted",
                **decision_fields,
                target_pixel_x=target.pixel_x,
                target_pixel_y=target.pixel_y,
            )
            self._record_decision_locked(decision)
        LOGGER.info(
            "AUTO_FIRE completed event=%s track=%s persistence_error=%s",
            event_id,
            track_id,
            persistence_error,
            extra={
                "structured_data": {
                    "event": "auto_fire_completed",
                    **evidence,
                    "recording_queued": recording_queued,
                    "rate_limit_persisted": persistence_error is None,
                }
            },
        )
        return decision

    def _wait_for_target_reacquisition(
        self,
        held: _HeldClassification,
        decision_fields: dict[str, object],
    ) -> AutoFireDecision:
        key = (held.event_id, held.track_id)
        while True:
            with self._lock:
                target_state_generation = self._target_state_generation
                if self._shutting_down:
                    self._held_classifications.pop(key, None)
                    return self._reject_locked("service_stopping", **decision_fields)
                if held.event_id in self._human_denied_events:
                    self._held_classifications.pop(key, None)
                    return self._reject_locked("human_detected", **decision_fields)
            night = self._night_mode()
            if night is None:
                with self._lock:
                    self._held_classifications.pop(key, None)
                return self._reject("safety_state_invalid", **decision_fields)
            if night:
                with self._lock:
                    self._held_classifications.pop(key, None)
                return self._reject("night_mode", **decision_fields)
            try:
                association = self._target_provider(held.event_id, held.track_id)
            except Exception:
                LOGGER.exception("AUTO_FIRE target provider failed while awaiting reacquisition")
                association = None
            if isinstance(association, AutoFireTargetSnapshot):
                with self._lock:
                    self._held_classifications.pop(key, None)
                return self._evaluate_classification(
                    event_id=held.event_id,
                    track_id=held.track_id,
                    classified_observation_monotonic=held.classified_observation_monotonic,
                    detections=held.detections,
                    error=held.error,
                    count_candidate=False,
                )
            if isinstance(association, AutoFireTargetAssociation) and association.state != "coasting":
                reason = {
                    "reacquisition_ambiguous": "target_reacquisition_ambiguous",
                    "reacquisition_incompatible": "target_reacquisition_incompatible",
                    "reacquisition_timeout": "target_reacquisition_timeout",
                }[association.state]
                with self._lock:
                    self._held_classifications.pop(key, None)
                return self._reject(reason, **decision_fields)
            now = self._read_clock(self._monotonic_clock)
            expiry = (
                association.expires_monotonic
                if isinstance(association, AutoFireTargetAssociation)
                else None
            )
            if now is None or expiry is None:
                with self._lock:
                    self._held_classifications.pop(key, None)
                return self._reject("target_association_invalid", **decision_fields)
            remaining = expiry - now
            if remaining <= 0:
                with self._lock:
                    self._held_classifications.pop(key, None)
                return self._reject("target_reacquisition_timeout", **decision_fields)
            with self._reacquisition_condition:
                if target_state_generation != self._target_state_generation:
                    continue
                self._reacquisition_condition.wait(timeout=remaining)

    def status(self) -> dict[str, object]:
        now_wall = self._rate_wall_now()
        cooldown = self._coordinator_cooldown(log_errors=False)
        with self._lock:
            shots_in_window = (
                0
                if now_wall is None
                else sum(
                    shot.accepted_at_epoch_seconds >= now_wall - _ROLLING_WINDOW_SECONDS
                    for shot in self._shots
                )
            )
            return {
                "enabled": self.config.enabled,
                "candidates_evaluated": self._candidates_evaluated,
                "accepted": self._accepted,
                "rejected": self._rejected,
                "rejection_counts": dict(sorted(self._rejection_counts.items())),
                "last_decision": None if self._last_decision is None else asdict(self._last_decision),
                "shots_in_rolling_window": shots_in_window,
                "max_shots_per_hour": self.config.max_shots_per_hour,
                "cooldown_remaining_seconds": None if cooldown is None else round(cooldown, 3),
                "engagement_pending": self._pending is not None,
                "classifications_held_for_reacquisition": len(self._held_classifications),
                "shutting_down": self._shutting_down,
                "persistence": {
                    "healthy": self._state_error is None,
                    "path": str(self.config.rate_limit_state_file),
                    "error": None if self._state_error is None else self._state_error[:256],
                    "records": len(self._shots),
                },
            }

    @staticmethod
    def _valid_event_and_track(event_id: object, track_id: object) -> bool:
        return (
            isinstance(event_id, str)
            and _SAFE_EVENT_ID.fullmatch(event_id) is not None
            and isinstance(track_id, int)
            and not isinstance(track_id, bool)
            and track_id > 0
        )

    @staticmethod
    def _parse_detections(
        detections: Iterable[AutoFireDetection] | None,
    ) -> tuple[list[AutoFireDetection], str | None]:
        if detections is None:
            return [], "classification_missing"
        if isinstance(detections, (str, bytes, dict)):
            return [], "classification_invalid"
        try:
            parsed = list(detections)
        except (TypeError, RuntimeError):
            return [], "classification_invalid"
        if any(not isinstance(item, AutoFireDetection) for item in parsed):
            return [], "classification_invalid"
        return parsed, None

    def _latch_human(self, event_id: str) -> None:
        now = self._read_clock(self._wall_clock)
        with self._lock:
            self._human_denied_events[event_id] = 0.0 if now is None else now
            if len(self._human_denied_events) > 4096:
                ordered = sorted(self._human_denied_events.items(), key=lambda item: item[1])
                self._human_denied_events = dict(ordered[-4096:])
            self._target_state_generation += 1
            self._reacquisition_condition.notify_all()

    def _is_human_latched(self, event_id: str) -> bool:
        with self._lock:
            return event_id in self._human_denied_events

    def _night_mode(self) -> bool | None:
        try:
            value = self._night_mode_provider()
        except Exception:
            LOGGER.exception("AUTO_FIRE night-mode provider failed")
            return None
        return value if isinstance(value, bool) else None

    def _current_target(
        self,
        event_id: str,
        track_id: int,
        now_monotonic: float,
    ) -> tuple[AutoFireTargetSnapshot | None, str]:
        try:
            target = self._target_provider(event_id, track_id)
        except Exception:
            LOGGER.exception("AUTO_FIRE target provider failed")
            return None, "target_association_invalid"
        if isinstance(target, AutoFireTargetAssociation):
            reason = {
                "coasting": "target_reacquisition_pending",
                "reacquisition_ambiguous": "target_reacquisition_ambiguous",
                "reacquisition_incompatible": "target_reacquisition_incompatible",
                "reacquisition_timeout": "target_reacquisition_timeout",
            }[target.state]
            return None, reason
        if target is None:
            return None, "target_association_invalid"
        if not isinstance(target, AutoFireTargetSnapshot):
            return None, "target_association_invalid"
        if target.event_id != event_id or target.track_id != track_id:
            return None, "target_association_invalid"
        if not target.confirmed:
            return None, "target_not_confirmed"
        if not target.event_eligible:
            return None, "target_not_event_eligible"
        if target.provisional_category == "person_sized":
            return None, "person_sized_target"
        if not self._fresh(now_monotonic, target.observed_monotonic, self.config.target_max_age_seconds):
            return None, "stale_target"
        if not self._inside_box(target.pixel_x, target.pixel_y, target.bounding_box):
            return None, "target_association_invalid"
        return target, "accepted"

    def _final_safety_reason(
        self,
        event_id: str,
        track_id: int,
        classified_observation_monotonic: float,
        aimed_pixel: tuple[int, int],
        aimed_frame_size: tuple[int, int],
    ) -> str | None:
        if not self.config.enabled:
            return "feature_disabled"
        with self._lock:
            if self._shutting_down:
                return "service_stopping"
            if self._state_error is not None:
                return "safety_state_invalid"
            if self._pending != (event_id, track_id):
                return "target_association_invalid"
            if event_id in self._human_denied_events:
                return "human_detected"
        night = self._night_mode()
        if night is None:
            return "safety_state_invalid"
        if night:
            return "night_mode"
        if self._rate_wall_now() is None:
            return "safety_state_invalid"
        now = self._read_clock(self._monotonic_clock)
        if now is None:
            return "safety_state_invalid"
        if not self._fresh(now, classified_observation_monotonic, self.config.classification_max_age_seconds):
            return "stale_classification"
        target, reason = self._current_target(event_id, track_id, now)
        if target is None:
            return reason
        if (target.frame_width, target.frame_height) != aimed_frame_size:
            return "calibration_frame_mismatch"
        if not self._inside_box(aimed_pixel[0], aimed_pixel[1], target.bounding_box):
            return "target_moved_during_aim"
        return None

    @staticmethod
    def _fresh(now: float, observed: object, maximum_age: float) -> bool:
        try:
            stamp = _finite_number(observed, "observed_monotonic", minimum=0.0)
        except ValueError:
            return False
        age = now - stamp
        return 0.0 <= age <= maximum_age

    @staticmethod
    def _inside_box(pixel_x: int, pixel_y: int, box: tuple[int, int, int, int]) -> bool:
        x, y, width, height = box
        return x <= pixel_x < x + width and y <= pixel_y < y + height

    def _coordinator_cooldown(self, *, log_errors: bool = True) -> float | None:
        try:
            value = self.coordinator.cooldown_remaining_seconds()
            return _finite_number(value, "cooldown_remaining_seconds", minimum=0.0)
        except Exception:
            if log_errors:
                LOGGER.exception("AUTO_FIRE coordinator cooldown read failed")
            return None

    def _rate_limit_reason_locked(
        self,
        event_id: str,
        track_id: int,
        now_wall: float,
    ) -> str | None:
        event_shots = [shot for shot in self._shots if shot.event_id == event_id]
        if len(event_shots) >= self.config.max_shots_per_event:
            return "target_already_engaged"
        same_target_shots = [
            shot
            for shot in self._shots
            if shot.event_id == event_id or shot.track_id == track_id
        ]
        if same_target_shots:
            elapsed = now_wall - max(
                shot.accepted_at_epoch_seconds for shot in same_target_shots
            )
            if elapsed < self.config.minimum_reengagement_seconds:
                return "minimum_reengagement_delay"
        rolling = sum(
            shot.accepted_at_epoch_seconds >= now_wall - _ROLLING_WINDOW_SECONDS
            for shot in self._shots
        )
        if rolling >= self.config.max_shots_per_hour:
            return "global_rate_limit"
        return None

    @staticmethod
    def _coordinator_exception_reason(exc: Exception) -> str:
        reason = getattr(exc, "reason", None)
        return reason if isinstance(reason, str) and reason in _KNOWN_COORDINATOR_REASONS else "hardware_not_ready"

    @classmethod
    def _unsuccessful_result_reason(cls, result: object) -> str | None:
        if result is False:
            return "safety_state_invalid"
        for attribute in ("accepted", "fired"):
            value = getattr(result, attribute, None)
            if value is False:
                reason = getattr(result, "reason", None)
                return reason if isinstance(reason, str) and reason in _KNOWN_COORDINATOR_REASONS else "safety_state_invalid"
        return None

    @staticmethod
    def _read_clock(clock: Callable[[], float]) -> float | None:
        try:
            value = clock()
            return _finite_number(value, "clock", minimum=0.0)
        except Exception:
            return None

    def _reject(self, reason: str, **fields: object) -> AutoFireDecision:
        with self._lock:
            return self._reject_locked(reason, **fields)

    def _reject_locked(self, reason: str, **fields: object) -> AutoFireDecision:
        decision = AutoFireDecision(False, reason, **fields)  # type: ignore[arg-type]
        self._record_decision_locked(decision)
        LOGGER.info(
            "AUTO_FIRE rejected reason=%s event=%s track=%s",
            reason,
            decision.event_id,
            decision.track_id,
            extra={
                "structured_data": {
                    "event": "auto_fire_rejected",
                    "reason": reason,
                    **asdict(decision),
                }
            },
        )
        return decision

    def _record_decision_locked(self, decision: AutoFireDecision) -> None:
        self._last_decision = decision
        if decision.accepted:
            self._accepted += 1
        else:
            self._rejected += 1
            self._rejection_counts[decision.reason] = self._rejection_counts.get(decision.reason, 0) + 1

    def _load_rate_limit_state(self) -> None:
        path = self.config.rate_limit_state_file
        try:
            if not path.exists():
                return
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("schema_version") not in {
                _LEGACY_RATE_LIMIT_SCHEMA_VERSION,
                _RATE_LIMIT_SCHEMA_VERSION,
            }:
                raise ValueError("unsupported rate-limit state schema")
            raw_shots = payload.get("shots")
            if not isinstance(raw_shots, list):
                raise ValueError("rate-limit state shots must be a list")
            shots: list[_ShotRecord] = []
            for raw in raw_shots:
                if not isinstance(raw, dict) or set(raw) != {
                    "event_id",
                    "track_id",
                    "accepted_at_epoch_seconds",
                }:
                    raise ValueError("rate-limit state contains a malformed shot")
                event_id = raw["event_id"]
                track_id = raw["track_id"]
                accepted_at = raw["accepted_at_epoch_seconds"]
                if not self._valid_event_and_track(event_id, track_id):
                    raise ValueError("rate-limit state contains an invalid event association")
                timestamp = _finite_number(accepted_at, "accepted_at_epoch_seconds", minimum=0.0)
                shots.append(_ShotRecord(event_id, track_id, timestamp))
            self._shots = shots
            if payload["schema_version"] == _RATE_LIMIT_SCHEMA_VERSION:
                raw_clock = payload.get("clock")
                if not isinstance(raw_clock, dict) or set(raw_clock) != {
                    "boot_id",
                    "monotonic_seconds",
                    "wall_epoch_seconds",
                }:
                    raise ValueError("rate-limit state contains malformed clock evidence")
                boot_id = raw_clock["boot_id"]
                if boot_id is not None and (not isinstance(boot_id, str) or not boot_id):
                    raise ValueError("rate-limit state contains an invalid boot identifier")
                self._persisted_rate_clock = (
                    _finite_number(
                        raw_clock["wall_epoch_seconds"],
                        "clock.wall_epoch_seconds",
                        minimum=0.0,
                    ),
                    _finite_number(
                        raw_clock["monotonic_seconds"],
                        "clock.monotonic_seconds",
                        minimum=0.0,
                    ),
                    boot_id,
                )
        except Exception as exc:
            self._state_error = f"Could not read auto-fire rate-limit state: {type(exc).__name__}: {exc}"
            LOGGER.error(
                "AUTO_FIRE rate-limit state is invalid; automatic engagement disabled",
                extra={
                    "structured_data": {
                        "event": "auto_fire_rate_limit_state_invalid",
                        "path": str(path),
                        "error": self._state_error,
                    }
                },
            )

    def _preflight_rate_limit_state(self) -> None:
        """Prove restart-safe state is writable before any enabled engagement."""

        if not self.config.enabled or self._state_error is not None:
            return
        with self._lock:
            try:
                self._persist_rate_limit_state_locked()
            except Exception as exc:
                self._state_error = (
                    "Could not initialize auto-fire rate-limit state: "
                    f"{type(exc).__name__}: {exc}"
                )
                LOGGER.error(
                    "AUTO_FIRE rate-limit state is not writable; automatic engagement disabled",
                    extra={
                        "structured_data": {
                            "event": "auto_fire_rate_limit_state_unwritable",
                            "path": str(self.config.rate_limit_state_file),
                            "error": self._state_error,
                        }
                    },
                )

    def _validate_initial_rate_clock(self) -> None:
        """Reject startup when persisted same-boot clock evidence is inconsistent."""

        if not self.config.enabled or self._state_error is not None:
            return
        with self._lock:
            if self._rate_clock_wall is None or self._rate_clock_monotonic is None:
                self._state_error = "Auto-fire rate-limit clocks were unavailable at startup"
                return
            persisted = self._persisted_rate_clock
            if persisted is None:
                return
            persisted_wall, persisted_monotonic, persisted_boot_id = persisted
            same_known_boot = (
                persisted_boot_id is not None
                and self._rate_clock_boot_id is not None
                and persisted_boot_id == self._rate_clock_boot_id
            )
            boot_changed = (
                persisted_boot_id is not None
                and self._rate_clock_boot_id is not None
                and persisted_boot_id != self._rate_clock_boot_id
            )
            if boot_changed:
                if self._rate_clock_wall < persisted_wall - _WALL_CLOCK_JUMP_TOLERANCE_SECONDS:
                    self._latch_clock_error_locked(
                        "wall clock moved backward across a system restart"
                    )
                return
            if same_known_boot or self._rate_clock_monotonic >= persisted_monotonic:
                self._validate_rate_clock_delta_locked(
                    persisted_wall,
                    persisted_monotonic,
                    self._rate_clock_wall,
                    self._rate_clock_monotonic,
                    context="service startup",
                )
            elif self._rate_clock_wall < persisted_wall - _WALL_CLOCK_JUMP_TOLERANCE_SECONDS:
                self._latch_clock_error_locked(
                    "wall clock moved backward across a system restart"
                )

    def _rate_wall_now(self) -> float | None:
        """Return a wall sample only while it agrees with monotonic elapsed time."""

        wall = self._read_clock(self._wall_clock)
        monotonic = self._read_clock(self._monotonic_clock)
        with self._lock:
            if wall is None or monotonic is None:
                self._latch_clock_error_locked("rate-limit clock became unavailable")
                return None
            previous_wall = self._rate_clock_wall
            previous_monotonic = self._rate_clock_monotonic
            if previous_wall is None or previous_monotonic is None:
                self._latch_clock_error_locked("rate-limit clock had no trusted baseline")
                return None
            if not self._validate_rate_clock_delta_locked(
                previous_wall,
                previous_monotonic,
                wall,
                monotonic,
                context="runtime",
            ):
                return None
            self._rate_clock_wall = wall
            self._rate_clock_monotonic = monotonic
            return wall

    def _validate_rate_clock_delta_locked(
        self,
        previous_wall: float,
        previous_monotonic: float,
        wall: float,
        monotonic: float,
        *,
        context: str,
    ) -> bool:
        monotonic_elapsed = monotonic - previous_monotonic
        if monotonic_elapsed < 0:
            self._latch_clock_error_locked(
                f"monotonic clock moved backward during {context}"
            )
            return False
        expected_wall = previous_wall + monotonic_elapsed
        wall_error = wall - expected_wall
        if abs(wall_error) > _WALL_CLOCK_JUMP_TOLERANCE_SECONDS:
            direction = "forward" if wall_error > 0 else "backward"
            self._latch_clock_error_locked(
                f"wall clock jumped {direction} during {context}"
            )
            return False
        return True

    def _latch_clock_error_locked(self, detail: str) -> None:
        if self._state_error is None:
            self._state_error = f"Auto-fire rate-limit clock is unsafe: {detail}"
            LOGGER.error(
                "AUTO_FIRE rate-limit clock is unsafe; automatic engagement disabled: %s",
                detail,
                extra={
                    "structured_data": {
                        "event": "auto_fire_rate_limit_clock_invalid",
                        "detail": detail,
                    }
                },
            )

    @staticmethod
    def _system_boot_id() -> str | None:
        try:
            value = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
        except (OSError, UnicodeError):
            return None
        return value or None

    def _persist_rate_limit_state_locked(self) -> None:
        path = self.config.rate_limit_state_file
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        payload = {
            "schema_version": _RATE_LIMIT_SCHEMA_VERSION,
            "clock": {
                "boot_id": self._rate_clock_boot_id,
                "monotonic_seconds": self._rate_clock_monotonic,
                "wall_epoch_seconds": self._rate_clock_wall,
            },
            "shots": [shot.as_dict() for shot in self._shots],
        }
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


__all__ = [
    "AutoFireConfig",
    "AutoFireCoordinator",
    "AutoFireDecision",
    "AutoFireDetection",
    "AutoFireService",
    "AutoFireTargetSnapshot",
    "HUMAN_DENY_LABELS",
]
