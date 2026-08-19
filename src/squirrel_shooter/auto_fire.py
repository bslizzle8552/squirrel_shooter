"""Fail-closed, event-driven policy for computer-aided automatic engagement."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from .classifier_labels import VOC_LABELS


LOGGER = logging.getLogger(__name__)
HUMAN_DENY_LABELS = frozenset({"person"})
_MODEL_LABELS = frozenset(VOC_LABELS[1:])
_UNKNOWN_LABELS = frozenset({"background", "unknown", "unclassified", "no-result", "no_result"})
_SAFE_EVENT_ID = re.compile(r"[A-Za-z0-9_-]+")
_RATE_LIMIT_SCHEMA_VERSION = 3
_PREVIOUS_RATE_LIMIT_SCHEMA_VERSION = 2
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


class _ActuationReservationError(RuntimeError):
    """Fail-closed durable-reservation rejection exposed through the coordinator seam."""

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)


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
        reserve_actuation: Callable[[], str],
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
class _ShotAttempt:
    reservation_id: str
    event_id: str
    track_id: int
    reserved_at_epoch_seconds: float
    reserved_at_monotonic_seconds: float | None
    reserved_boot_id: str | None
    state: str
    outcome: str | None = None
    finalized_at_epoch_seconds: float | None = None
    shot_attempted: bool | None = None
    physical_event_id: str | None = None
    recording_queued: bool | None = None

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
        self._attempts: list[_ShotAttempt] = []
        self._state_error: str | None = None
        self._state_io_lock = threading.Lock()
        self._rate_clock_wall = self._read_clock(self._wall_clock)
        self._rate_clock_monotonic = self._read_clock(self._monotonic_clock)
        self._rate_clock_boot_id = self._system_boot_id()
        self._persisted_rate_clock: tuple[float, float, str | None] | None = None
        self._restart_interlock_until_monotonic: float | None = None
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
        self._recover_unresolved_reservations()
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
        reservation_id: str | None = None

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

        def reserve_actuation() -> str:
            nonlocal final_rejection_reason, reservation_id
            if reservation_id is not None:
                return reservation_id
            if not final_check_called or final_rejection_reason is not None:
                final_rejection_reason = final_rejection_reason or "safety_state_invalid"
                raise _ActuationReservationError(
                    final_rejection_reason,
                    "Durable actuation reservation requested before final safety approval",
                )
            try:
                reservation_id = self._reserve_actuation_attempt(event_id, track_id)
            except _ActuationReservationError as exc:
                final_rejection_reason = exc.reason
                raise
            return reservation_id

        try:
            result = self.coordinator.automatic_engage(
                target.pixel_x,
                target.pixel_y,
                frame_width=target.frame_width,
                frame_height=target.frame_height,
                cooldown_seconds=self.config.cooldown_seconds,
                final_safety_check=final_safety_check,
                reserve_actuation=reserve_actuation,
                evidence=evidence,
            )
        except Exception as exc:
            shot_attempted = getattr(exc, "shot_attempted", False) is True
            persistence_error: str | None = None
            if reservation_id is not None:
                try:
                    self._finalize_actuation_attempt(
                        reservation_id,
                        state="failed" if shot_attempted else "cancelled",
                        outcome=self._coordinator_exception_reason(exc),
                        shot_attempted=shot_attempted,
                    )
                except Exception as persist_exc:
                    persistence_error = f"{type(persist_exc).__name__}: {persist_exc}"
            with self._lock:
                self._pending = None
                if shot_attempted:
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
                            "reservation_id": reservation_id,
                            "error": f"{type(exc).__name__}: {exc}",
                            "reservation_finalized": reservation_id is not None
                            and persistence_error is None,
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
        if final_rejection_reason is not None:
            with self._lock:
                self._pending = None
            return self._reject(
                final_rejection_reason,
                **decision_fields,
                target_pixel_x=target.pixel_x,
                target_pixel_y=target.pixel_y,
            )
        if reservation_id is None:
            with self._lock:
                self._pending = None
                self._state_error = "Coordinator returned without a durable actuation reservation"
            return self._reject(
                "safety_state_invalid",
                **decision_fields,
                target_pixel_x=target.pixel_x,
                target_pixel_y=target.pixel_y,
            )
        returned_reservation_id = getattr(result, "reservation_id", None)
        if returned_reservation_id != reservation_id:
            persistence_error: str | None = None
            try:
                self._finalize_actuation_attempt(
                    reservation_id,
                    state="failed",
                    outcome="coordinator_reservation_mismatch",
                    shot_attempted=True,
                )
            except Exception as persist_exc:
                persistence_error = f"{type(persist_exc).__name__}: {persist_exc}"
            with self._lock:
                self._pending = None
                self._state_error = "Coordinator returned an invalid actuation reservation identifier"
                if persistence_error is not None:
                    self._state_error += f"; reservation finalization failed: {persistence_error}"
            return self._reject(
                "safety_state_invalid",
                **decision_fields,
                target_pixel_x=target.pixel_x,
                target_pixel_y=target.pixel_y,
            )
        unsuccessful_reason = self._unsuccessful_result_reason(result)
        if final_rejection_reason is not None or unsuccessful_reason is not None:
            shot_attempted = getattr(result, "shot_attempted", False) is True
            persistence_error: str | None = None
            try:
                self._finalize_actuation_attempt(
                    reservation_id,
                    state="failed" if shot_attempted else "cancelled",
                    outcome=final_rejection_reason or unsuccessful_reason or "safety_state_invalid",
                    shot_attempted=shot_attempted,
                )
            except Exception as persist_exc:
                persistence_error = f"{type(persist_exc).__name__}: {persist_exc}"
            with self._lock:
                self._pending = None
                if persistence_error is not None:
                    self._state_error = (
                        "Could not finalize rejected automatic engagement reservation: "
                        f"{persistence_error}"
                    )
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
        safety_errors: list[str] = []
        if accepted_at is None:
            accepted_at = now_wall
            safety_errors.append("Wall clock failed after an automatic engagement")
        physical_event_id = getattr(result, "event_id", None)
        if not isinstance(physical_event_id, str) or _SAFE_EVENT_ID.fullmatch(physical_event_id) is None:
            physical_event_id = None
            safety_errors.append("Coordinator returned an invalid physical event identifier")
        try:
            self._finalize_actuation_attempt(
                reservation_id,
                state="completed",
                outcome="accepted_recording_failed" if recording_failed else "accepted",
                shot_attempted=True,
                finalized_at_epoch_seconds=accepted_at,
                physical_event_id=physical_event_id,
                recording_queued=recording_queued if isinstance(recording_queued, bool) else None,
            )
        except Exception as exc:
            persistence_error = f"{type(exc).__name__}: {exc}"
            safety_errors.append(
                f"Could not finalize accepted automatic engagement reservation: {persistence_error}"
            )
        with self._lock:
            self._pending = None
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
                    "reservation_id": reservation_id,
                    "physical_event_id": physical_event_id,
                    "recording_queued": recording_queued,
                    "reservation_finalized": persistence_error is None,
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
        now_monotonic = self._read_clock(self._monotonic_clock)
        cooldown = self._coordinator_cooldown(log_errors=False)
        with self._lock:
            shots_in_window = (
                0
                if now_wall is None
                else sum(
                    attempt.reserved_at_epoch_seconds >= now_wall - _ROLLING_WINDOW_SECONDS
                    for attempt in self._attempts
                )
            )
            interlock_remaining = (
                None
                if self._restart_interlock_until_monotonic is None or now_monotonic is None
                else max(0.0, self._restart_interlock_until_monotonic - now_monotonic)
            )
            unresolved = sum(
                attempt.state in {"reserved", "recovered_uncertain"}
                for attempt in self._attempts
            )
            return {
                "enabled": self.config.enabled,
                "candidates_evaluated": self._candidates_evaluated,
                "accepted": self._accepted,
                "rejected": self._rejected,
                "rejection_counts": dict(sorted(self._rejection_counts.items())),
                "last_decision": None if self._last_decision is None else asdict(self._last_decision),
                "shots_in_rolling_window": shots_in_window,
                "counted_attempts_in_rolling_window": shots_in_window,
                "max_shots_per_hour": self.config.max_shots_per_hour,
                "cooldown_remaining_seconds": None if cooldown is None else round(cooldown, 3),
                "engagement_pending": self._pending is not None,
                "classifications_held_for_reacquisition": len(self._held_classifications),
                "shutting_down": self._shutting_down,
                "unresolved_actuation_reservations": unresolved,
                "restart_interlock_remaining_seconds": (
                    None if interlock_remaining is None else round(interlock_remaining, 3)
                ),
                "persistence": {
                    "healthy": self._state_error is None,
                    "path": str(self.config.rate_limit_state_file),
                    "error": None if self._state_error is None else self._state_error[:256],
                    "records": len(self._attempts),
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

    def _reserve_actuation_attempt(self, event_id: str, track_id: int) -> str:
        """Durably count one authorization before the coordinator may touch the valve."""

        with self._state_io_lock:
            now_wall = self._rate_wall_now()
            now_monotonic = self._read_clock(self._monotonic_clock)
            with self._lock:
                if self._shutting_down:
                    raise _ActuationReservationError(
                        "service_stopping",
                        "Automatic engagement stopped before durable reservation",
                    )
                if self._state_error is not None or now_wall is None or now_monotonic is None:
                    raise _ActuationReservationError(
                        "safety_state_invalid",
                        "Automatic rate-limit state is unsafe before durable reservation",
                    )
                if self._pending != (event_id, track_id):
                    raise _ActuationReservationError(
                        "target_association_invalid",
                        "Pending automatic target changed before durable reservation",
                    )
                rate_reason = self._rate_limit_reason_locked(event_id, track_id, now_wall)
                if rate_reason is not None:
                    raise _ActuationReservationError(
                        rate_reason,
                        f"Automatic rate limit changed before durable reservation: {rate_reason}",
                    )
                reservation_id = uuid.uuid4().hex
                attempt = _ShotAttempt(
                    reservation_id=reservation_id,
                    event_id=event_id,
                    track_id=track_id,
                    reserved_at_epoch_seconds=now_wall,
                    reserved_at_monotonic_seconds=now_monotonic,
                    reserved_boot_id=self._rate_clock_boot_id,
                    state="reserved",
                )
                self._attempts.append(attempt)
                payload = self._rate_limit_payload_locked()
            try:
                self._write_rate_limit_payload(payload)
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}"
                with self._lock:
                    self._state_error = (
                        "Could not durably reserve automatic engagement before actuation: "
                        f"{detail}"
                    )
                LOGGER.error(
                    "AUTO_FIRE durable actuation reservation failed event=%s track=%s error=%s",
                    event_id,
                    track_id,
                    detail,
                    extra={
                        "structured_data": {
                            "event": "auto_fire_actuation_reservation_failed",
                            "event_id": event_id,
                            "track_id": track_id,
                            "reservation_id": reservation_id,
                            "error": detail,
                        }
                    },
                )
                raise _ActuationReservationError(
                    "safety_state_invalid",
                    "Durable automatic actuation reservation failed",
                ) from exc
            with self._lock:
                if self._shutting_down:
                    raise _ActuationReservationError(
                        "service_stopping",
                        "Automatic engagement stopped after durable reservation",
                    )
                if self._state_error is not None:
                    raise _ActuationReservationError(
                        "safety_state_invalid",
                        "Automatic safety state changed during durable reservation",
                    )
                if event_id in self._human_denied_events:
                    raise _ActuationReservationError(
                        "human_detected",
                        "Human veto arrived during durable reservation",
                    )
        LOGGER.info(
            "AUTO_FIRE actuation durably reserved event=%s track=%s reservation=%s",
            event_id,
            track_id,
            reservation_id,
            extra={
                "structured_data": {
                    "event": "auto_fire_actuation_reserved",
                    "event_id": event_id,
                    "track_id": track_id,
                    "reservation_id": reservation_id,
                    "reserved_at_epoch_seconds": now_wall,
                }
            },
        )
        return reservation_id

    def _finalize_actuation_attempt(
        self,
        reservation_id: str,
        *,
        state: str,
        outcome: str,
        shot_attempted: bool,
        finalized_at_epoch_seconds: float | None = None,
        physical_event_id: str | None = None,
        recording_queued: bool | None = None,
    ) -> None:
        """Persist the terminal outcome without ever removing the counted reservation."""

        if state not in {"completed", "failed", "cancelled"}:
            raise ValueError("terminal actuation state is invalid")
        if not isinstance(outcome, str) or not outcome:
            raise ValueError("terminal actuation outcome is required")
        with self._state_io_lock:
            finalized_at = finalized_at_epoch_seconds
            if finalized_at is None:
                finalized_at = self._rate_wall_now()
            with self._lock:
                matches = [
                    index
                    for index, attempt in enumerate(self._attempts)
                    if attempt.reservation_id == reservation_id
                ]
                if len(matches) != 1:
                    raise ValueError("actuation reservation is missing or duplicated")
                index = matches[0]
                attempt = self._attempts[index]
                if attempt.state != "reserved":
                    if (
                        attempt.state == state
                        and attempt.outcome == outcome
                        and attempt.shot_attempted is shot_attempted
                    ):
                        return
                    raise ValueError("actuation reservation is already terminal")
                if finalized_at is None:
                    finalized_at = attempt.reserved_at_epoch_seconds
                finalized_at = _finite_number(
                    finalized_at,
                    "finalized_at_epoch_seconds",
                    minimum=0.0,
                )
                finalized_at = max(finalized_at, attempt.reserved_at_epoch_seconds)
                self._attempts[index] = replace(
                    attempt,
                    state=state,
                    outcome=outcome,
                    finalized_at_epoch_seconds=finalized_at,
                    shot_attempted=shot_attempted,
                    physical_event_id=physical_event_id,
                    recording_queued=recording_queued,
                )
                payload = self._rate_limit_payload_locked()
            try:
                self._write_rate_limit_payload(payload)
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}"
                with self._lock:
                    self._state_error = (
                        "Could not durably finalize automatic actuation reservation: "
                        f"{detail}"
                    )
                raise
        LOGGER.info(
            "AUTO_FIRE actuation reservation finalized reservation=%s state=%s outcome=%s",
            reservation_id,
            state,
            outcome,
            extra={
                "structured_data": {
                    "event": "auto_fire_actuation_reservation_finalized",
                    "reservation_id": reservation_id,
                    "state": state,
                    "outcome": outcome,
                    "shot_attempted": shot_attempted,
                    "physical_event_id": physical_event_id,
                    "recording_queued": recording_queued,
                }
            },
        )

    def _rate_limit_reason_locked(
        self,
        event_id: str,
        track_id: int,
        now_wall: float,
    ) -> str | None:
        if self._restart_interlock_until_monotonic is not None:
            now_monotonic = self._read_clock(self._monotonic_clock)
            if now_monotonic is None or now_monotonic < self._restart_interlock_until_monotonic:
                return "unresolved_actuation"
            self._restart_interlock_until_monotonic = None
        event_attempts = [attempt for attempt in self._attempts if attempt.event_id == event_id]
        if len(event_attempts) >= self.config.max_shots_per_event:
            return "target_already_engaged"
        same_target_attempts = [
            attempt
            for attempt in self._attempts
            if attempt.event_id == event_id or attempt.track_id == track_id
        ]
        if same_target_attempts:
            elapsed = now_wall - max(
                attempt.reserved_at_epoch_seconds for attempt in same_target_attempts
            )
            if elapsed < self.config.minimum_reengagement_seconds:
                return "minimum_reengagement_delay"
        rolling = sum(
            attempt.reserved_at_epoch_seconds >= now_wall - _ROLLING_WINDOW_SECONDS
            for attempt in self._attempts
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
                _PREVIOUS_RATE_LIMIT_SCHEMA_VERSION,
                _RATE_LIMIT_SCHEMA_VERSION,
            }:
                raise ValueError("unsupported rate-limit state schema")
            schema_version = payload["schema_version"]
            if schema_version in {
                _LEGACY_RATE_LIMIT_SCHEMA_VERSION,
                _PREVIOUS_RATE_LIMIT_SCHEMA_VERSION,
            }:
                raw_shots = payload.get("shots")
                if not isinstance(raw_shots, list):
                    raise ValueError("rate-limit state shots must be a list")
                attempts: list[_ShotAttempt] = []
                for index, raw in enumerate(raw_shots):
                    if not isinstance(raw, dict) or set(raw) != {
                        "event_id",
                        "track_id",
                        "accepted_at_epoch_seconds",
                    }:
                        raise ValueError("rate-limit state contains a malformed shot")
                    event_id = raw["event_id"]
                    track_id = raw["track_id"]
                    if not self._valid_event_and_track(event_id, track_id):
                        raise ValueError("rate-limit state contains an invalid event association")
                    timestamp = _finite_number(
                        raw["accepted_at_epoch_seconds"],
                        "accepted_at_epoch_seconds",
                        minimum=0.0,
                    )
                    attempts.append(
                        _ShotAttempt(
                            reservation_id=f"legacy-{index + 1}",
                            event_id=event_id,
                            track_id=track_id,
                            reserved_at_epoch_seconds=timestamp,
                            reserved_at_monotonic_seconds=None,
                            reserved_boot_id=None,
                            state="legacy_completed",
                            outcome="legacy_shot_record",
                            finalized_at_epoch_seconds=timestamp,
                            shot_attempted=True,
                        )
                    )
                self._attempts = attempts
            else:
                if set(payload) != {"schema_version", "clock", "attempts"}:
                    raise ValueError("rate-limit state contains unexpected schema-3 fields")
                raw_attempts = payload.get("attempts")
                if not isinstance(raw_attempts, list):
                    raise ValueError("rate-limit state attempts must be a list")
                attempts = [self._parse_persisted_attempt(raw) for raw in raw_attempts]
                reservation_ids = [attempt.reservation_id for attempt in attempts]
                if len(set(reservation_ids)) != len(reservation_ids):
                    raise ValueError("rate-limit state contains duplicate reservation identifiers")
                self._attempts = attempts
            if schema_version in {
                _PREVIOUS_RATE_LIMIT_SCHEMA_VERSION,
                _RATE_LIMIT_SCHEMA_VERSION,
            }:
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

    def _parse_persisted_attempt(self, raw: object) -> _ShotAttempt:
        expected = {
            "reservation_id",
            "event_id",
            "track_id",
            "reserved_at_epoch_seconds",
            "reserved_at_monotonic_seconds",
            "reserved_boot_id",
            "state",
            "outcome",
            "finalized_at_epoch_seconds",
            "shot_attempted",
            "physical_event_id",
            "recording_queued",
        }
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError("rate-limit state contains a malformed actuation attempt")
        reservation_id = raw["reservation_id"]
        event_id = raw["event_id"]
        track_id = raw["track_id"]
        if (
            not isinstance(reservation_id, str)
            or _SAFE_EVENT_ID.fullmatch(reservation_id) is None
        ):
            raise ValueError("rate-limit state contains an invalid reservation identifier")
        if not self._valid_event_and_track(event_id, track_id):
            raise ValueError("rate-limit state contains an invalid event association")
        reserved_at = _finite_number(
            raw["reserved_at_epoch_seconds"],
            "reserved_at_epoch_seconds",
            minimum=0.0,
        )
        reserved_monotonic = raw["reserved_at_monotonic_seconds"]
        if reserved_monotonic is not None:
            reserved_monotonic = _finite_number(
                reserved_monotonic,
                "reserved_at_monotonic_seconds",
                minimum=0.0,
            )
        reserved_boot_id = raw["reserved_boot_id"]
        if reserved_boot_id is not None and (
            not isinstance(reserved_boot_id, str) or not reserved_boot_id
        ):
            raise ValueError("rate-limit state contains an invalid reservation boot identifier")
        state = raw["state"]
        if state not in {
            "reserved",
            "recovered_uncertain",
            "completed",
            "failed",
            "cancelled",
            "legacy_completed",
        }:
            raise ValueError("rate-limit state contains an invalid actuation state")
        outcome = raw["outcome"]
        if outcome is not None and (not isinstance(outcome, str) or not outcome):
            raise ValueError("rate-limit state contains an invalid actuation outcome")
        finalized_at = raw["finalized_at_epoch_seconds"]
        if finalized_at is not None:
            finalized_at = _finite_number(
                finalized_at,
                "finalized_at_epoch_seconds",
                minimum=reserved_at,
            )
        shot_attempted = raw["shot_attempted"]
        if shot_attempted is not None and not isinstance(shot_attempted, bool):
            raise ValueError("rate-limit state contains invalid shot-attempt evidence")
        physical_event_id = raw["physical_event_id"]
        if physical_event_id is not None and (
            not isinstance(physical_event_id, str)
            or _SAFE_EVENT_ID.fullmatch(physical_event_id) is None
        ):
            raise ValueError("rate-limit state contains an invalid physical event identifier")
        recording_queued = raw["recording_queued"]
        if recording_queued is not None and not isinstance(recording_queued, bool):
            raise ValueError("rate-limit state contains invalid recording evidence")
        terminal = state in {"completed", "failed", "cancelled", "legacy_completed"}
        if state == "reserved" and any(
            value is not None
            for value in (outcome, finalized_at, shot_attempted, physical_event_id, recording_queued)
        ):
            raise ValueError("reserved actuation contains terminal fields")
        if state == "recovered_uncertain" and (
            outcome != "process_restart"
            or any(
                value is not None
                for value in (finalized_at, shot_attempted, physical_event_id, recording_queued)
            )
        ):
            raise ValueError("recovered actuation contains invalid uncertainty fields")
        if terminal and (outcome is None or finalized_at is None or shot_attempted is None):
            raise ValueError("terminal actuation is missing required fields")
        if state in {"completed", "failed", "legacy_completed"} and shot_attempted is not True:
            raise ValueError("attempted actuation must remain conservatively counted")
        if state == "cancelled" and shot_attempted is not False:
            raise ValueError("cancelled actuation must record no physical attempt")
        if state == "completed" and physical_event_id is None:
            raise ValueError("completed actuation is missing its physical event identifier")
        return _ShotAttempt(
            reservation_id=reservation_id,
            event_id=event_id,
            track_id=track_id,
            reserved_at_epoch_seconds=reserved_at,
            reserved_at_monotonic_seconds=reserved_monotonic,
            reserved_boot_id=reserved_boot_id,
            state=state,
            outcome=outcome,
            finalized_at_epoch_seconds=finalized_at,
            shot_attempted=shot_attempted,
            physical_event_id=physical_event_id,
            recording_queued=recording_queued,
        )

    def _recover_unresolved_reservations(self) -> None:
        if self._state_error is not None:
            return
        unresolved = [
            index
            for index, attempt in enumerate(self._attempts)
            if attempt.state in {"reserved", "recovered_uncertain"}
        ]
        if not unresolved:
            return
        now_monotonic = self._read_clock(self._monotonic_clock)
        if now_monotonic is None:
            self._state_error = "Could not establish a restart interlock for unresolved actuation"
            return
        with self._lock:
            for index in unresolved:
                attempt = self._attempts[index]
                if attempt.state == "reserved":
                    self._attempts[index] = replace(
                        attempt,
                        state="recovered_uncertain",
                        outcome="process_restart",
                    )
            self._restart_interlock_until_monotonic = now_monotonic + _ROLLING_WINDOW_SECONDS
        LOGGER.error(
            "AUTO_FIRE recovered %d unresolved durable actuation reservation(s); automatic engagement interlocked",
            len(unresolved),
            extra={
                "structured_data": {
                    "event": "auto_fire_unresolved_reservations_recovered",
                    "count": len(unresolved),
                    "interlock_seconds": _ROLLING_WINDOW_SECONDS,
                }
            },
        )

    def _preflight_rate_limit_state(self) -> None:
        """Prove restart-safe state is writable before any enabled engagement."""

        if not self.config.enabled or self._state_error is not None:
            return
        try:
            self._persist_rate_limit_state()
        except Exception as exc:
            with self._lock:
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
                    correction = self._rate_clock_wall - persisted_wall
                    if self._rebaseline_empty_history_clock_locked(
                        "backward",
                        correction,
                        "system restart",
                    ):
                        return
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
            if self._rebaseline_empty_history_clock_locked(direction, wall_error, context):
                return True
            self._latch_clock_error_locked(
                f"wall clock jumped {direction} during {context}"
            )
            return False
        return True

    def _rebaseline_empty_history_clock_locked(
        self,
        direction: str,
        correction_seconds: float,
        context: str,
    ) -> bool:
        if self._attempts:
            return False
        LOGGER.warning(
            "AUTO_FIRE rate-limit clock rebased after empty-history %s correction: %.3fs during %s",
            direction,
            correction_seconds,
            context,
            extra={
                "structured_data": {
                    "event": "auto_fire_rate_limit_clock_rebased",
                    "direction": direction,
                    "correction_seconds": round(correction_seconds, 3),
                    "context": context,
                    "shot_history_count": 0,
                }
            },
        )
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

    def _rate_limit_payload_locked(self) -> dict[str, object]:
        return {
            "schema_version": _RATE_LIMIT_SCHEMA_VERSION,
            "clock": {
                "boot_id": self._rate_clock_boot_id,
                "monotonic_seconds": self._rate_clock_monotonic,
                "wall_epoch_seconds": self._rate_clock_wall,
            },
            "attempts": [attempt.as_dict() for attempt in self._attempts],
        }

    def _persist_rate_limit_state(self) -> None:
        """Serialize state writes without holding the live target/policy lock during I/O."""

        with self._state_io_lock:
            with self._lock:
                payload = self._rate_limit_payload_locked()
            self._write_rate_limit_payload(payload)

    def _write_rate_limit_payload(self, payload: dict[str, object]) -> None:
        path = self.config.rate_limit_state_file
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            self._fsync_parent_directory(path.parent)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def _fsync_parent_directory(directory: Path) -> None:
        """Make the atomic replacement durable on the Linux deployment filesystem."""

        if os.name != "posix":
            return
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(str(directory), flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


__all__ = [
    "AutoFireConfig",
    "AutoFireCoordinator",
    "AutoFireDecision",
    "AutoFireDetection",
    "AutoFireService",
    "AutoFireTargetSnapshot",
    "HUMAN_DENY_LABELS",
]
