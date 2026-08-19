"""Pure progress-health evaluation for the shared application runtime.

This module deliberately owns no threads, hardware, logging, or recovery.  It
turns one explicit progress snapshot into structured findings so the runtime
supervisor can latch a single diagnostic and use the existing shared control
service for any fail-safe action.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Literal


HealthSeverity = Literal["critical", "degraded"]
HealthCondition = Literal["dead", "stale", "stalled"]


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be a finite number")
    return normalized


def _finite_positive(value: object, field_name: str) -> float:
    normalized = _finite_number(value, field_name)
    if normalized <= 0.0:
        raise ValueError(f"{field_name} must be a finite positive number")
    return normalized


def _optional_monotonic(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    return _finite_number(value, field_name)


def _boolean(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _optional_error(value: object, field_name: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string or null")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeHealthThresholds:
    """Bounded progress thresholds, expressed in monotonic-clock seconds."""

    camera_stale_seconds: float = 3.0
    motion_stale_seconds: float = 5.0
    classifier_stale_seconds: float = 5.0
    auto_fire_decision_stale_seconds: float = 5.0
    storage_stale_seconds: float = 2.0
    startup_grace_seconds: float = 10.0

    def __post_init__(self) -> None:
        for field_name in (
            "camera_stale_seconds",
            "motion_stale_seconds",
            "classifier_stale_seconds",
            "auto_fire_decision_stale_seconds",
            "storage_stale_seconds",
            "startup_grace_seconds",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_positive(getattr(self, field_name), field_name),
            )

    def to_dict(self) -> dict[str, float]:
        return {
            "camera_stale_seconds": self.camera_stale_seconds,
            "motion_stale_seconds": self.motion_stale_seconds,
            "classifier_stale_seconds": self.classifier_stale_seconds,
            "auto_fire_decision_stale_seconds": self.auto_fire_decision_stale_seconds,
            "storage_stale_seconds": self.storage_stale_seconds,
            "startup_grace_seconds": self.startup_grace_seconds,
        }


DEFAULT_RUNTIME_HEALTH_THRESHOLDS = RuntimeHealthThresholds()


@dataclass(frozen=True, slots=True, kw_only=True)
class ContinuousWorkerProgress:
    """Progress state for a worker expected to advance continuously."""

    expected: bool
    alive: bool
    last_progress_monotonic: float | None
    last_error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "expected", _boolean(self.expected, "expected"))
        object.__setattr__(self, "alive", _boolean(self.alive, "alive"))
        object.__setattr__(
            self,
            "last_progress_monotonic",
            _optional_monotonic(self.last_progress_monotonic, "last_progress_monotonic"),
        )
        object.__setattr__(self, "last_error", _optional_error(self.last_error, "last_error"))

    def to_dict(self) -> dict[str, object]:
        return {
            "expected": self.expected,
            "alive": self.alive,
            "last_progress_monotonic": self.last_progress_monotonic,
            "last_error": self.last_error,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ActiveWorkProgress:
    """Progress state for a worker that may be legitimately idle.

    ``active_since_monotonic`` should be set when the current queue/in-progress
    period begins.  Staleness is measured from the newer of that timestamp and
    the last observed progress, preventing an old idle timestamp from making
    newly queued work look immediately stale.
    """

    expected: bool
    alive: bool
    queued_count: int
    in_progress: bool
    last_progress_monotonic: float | None
    active_since_monotonic: float | None = None
    last_error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "expected", _boolean(self.expected, "expected"))
        object.__setattr__(self, "alive", _boolean(self.alive, "alive"))
        if isinstance(self.queued_count, bool) or not isinstance(self.queued_count, int):
            raise ValueError("queued_count must be a non-negative integer")
        if self.queued_count < 0:
            raise ValueError("queued_count must be a non-negative integer")
        object.__setattr__(self, "in_progress", _boolean(self.in_progress, "in_progress"))
        object.__setattr__(
            self,
            "last_progress_monotonic",
            _optional_monotonic(self.last_progress_monotonic, "last_progress_monotonic"),
        )
        object.__setattr__(
            self,
            "active_since_monotonic",
            _optional_monotonic(self.active_since_monotonic, "active_since_monotonic"),
        )
        object.__setattr__(self, "last_error", _optional_error(self.last_error, "last_error"))

    @property
    def has_active_work(self) -> bool:
        return self.queued_count > 0 or self.in_progress

    def to_dict(self) -> dict[str, object]:
        return {
            "expected": self.expected,
            "alive": self.alive,
            "queued_count": self.queued_count,
            "in_progress": self.in_progress,
            "has_active_work": self.has_active_work,
            "last_progress_monotonic": self.last_progress_monotonic,
            "active_since_monotonic": self.active_since_monotonic,
            "last_error": self.last_error,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeProgressSnapshot:
    """One immutable, JSON-friendly view of every supervised subsystem."""

    observed_monotonic: float
    runtime_started_monotonic: float
    camera: ContinuousWorkerProgress
    motion: ContinuousWorkerProgress
    classifier: ActiveWorkProgress
    auto_fire_decision: ActiveWorkProgress
    event_writer: ActiveWorkProgress
    event_storage: ActiveWorkProgress

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observed_monotonic",
            _finite_number(self.observed_monotonic, "observed_monotonic"),
        )
        object.__setattr__(
            self,
            "runtime_started_monotonic",
            _finite_number(self.runtime_started_monotonic, "runtime_started_monotonic"),
        )
        if self.observed_monotonic < self.runtime_started_monotonic:
            raise ValueError("observed_monotonic must not precede runtime_started_monotonic")
        for field_name, expected_type in (
            ("camera", ContinuousWorkerProgress),
            ("motion", ContinuousWorkerProgress),
            ("classifier", ActiveWorkProgress),
            ("auto_fire_decision", ActiveWorkProgress),
            ("event_writer", ActiveWorkProgress),
            ("event_storage", ActiveWorkProgress),
        ):
            if not isinstance(getattr(self, field_name), expected_type):
                raise ValueError(f"{field_name} must be {expected_type.__name__}")

    def to_dict(self) -> dict[str, object]:
        return {
            "observed_monotonic": self.observed_monotonic,
            "runtime_started_monotonic": self.runtime_started_monotonic,
            "camera": self.camera.to_dict(),
            "motion": self.motion.to_dict(),
            "classifier": self.classifier.to_dict(),
            "auto_fire_decision": self.auto_fire_decision.to_dict(),
            "event_writer": self.event_writer.to_dict(),
            "event_storage": self.event_storage.to_dict(),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class HealthFinding:
    """One deterministic health finding with no embedded side effects."""

    code: str
    component: str
    severity: HealthSeverity
    condition: HealthCondition
    message: str
    work_active: bool
    age_seconds: float | None
    threshold_seconds: float | None
    last_error: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "component": self.component,
            "severity": self.severity,
            "condition": self.condition,
            "message": self.message,
            "work_active": self.work_active,
            "age_seconds": self.age_seconds,
            "threshold_seconds": self.threshold_seconds,
            "last_error": self.last_error,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeHealthReport:
    """Evaluation result suitable for status output and a latched supervisor."""

    snapshot: RuntimeProgressSnapshot
    thresholds: RuntimeHealthThresholds
    uptime_seconds: float
    startup_grace_remaining_seconds: float
    in_startup_grace: bool
    findings: tuple[HealthFinding, ...]

    @property
    def critical_findings(self) -> tuple[HealthFinding, ...]:
        return tuple(finding for finding in self.findings if finding.severity == "critical")

    @property
    def degraded_findings(self) -> tuple[HealthFinding, ...]:
        return tuple(finding for finding in self.findings if finding.severity == "degraded")

    @property
    def fail_safe_required(self) -> bool:
        return bool(self.critical_findings)

    @property
    def degraded(self) -> bool:
        return bool(self.degraded_findings)

    @property
    def status(self) -> str:
        if self.in_startup_grace:
            return "starting"
        if self.fail_safe_required:
            return "critical"
        if self.degraded:
            return "degraded"
        return "healthy"

    @property
    def diagnostic_key(self) -> str | None:
        """Stable aggregate key that a caller can latch to prevent log storms."""

        if self.fail_safe_required:
            return "runtime_progress_health_critical"
        if self.degraded:
            return "runtime_progress_health_degraded"
        return None

    def to_dict(self) -> dict[str, object]:
        critical = self.critical_findings
        degraded = self.degraded_findings
        return {
            "status": self.status,
            "diagnostic_key": self.diagnostic_key,
            "fail_safe_required": self.fail_safe_required,
            "degraded": self.degraded,
            "in_startup_grace": self.in_startup_grace,
            "uptime_seconds": self.uptime_seconds,
            "startup_grace_remaining_seconds": self.startup_grace_remaining_seconds,
            "thresholds": self.thresholds.to_dict(),
            "progress": self.snapshot.to_dict(),
            "finding_codes": [finding.code for finding in self.findings],
            "findings": [finding.to_dict() for finding in self.findings],
            "critical_findings": [finding.to_dict() for finding in critical],
            "degraded_findings": [finding.to_dict() for finding in degraded],
        }


def _age(now: float, reference: float) -> float:
    # A concurrently sampled worker can report a timestamp just beyond ``now``.
    return max(0.0, now - reference)


def _continuous_finding(
    *,
    component: str,
    progress: ContinuousWorkerProgress,
    now: float,
    runtime_started: float,
    stale_seconds: float,
) -> HealthFinding | None:
    if not progress.expected:
        return None
    if not progress.alive:
        detail = f": {progress.last_error}" if progress.last_error else ""
        return HealthFinding(
            code=f"runtime_{component}_dead",
            component=component,
            severity="critical",
            condition="dead",
            message=f"{component} worker is not alive{detail}",
            work_active=True,
            age_seconds=None,
            threshold_seconds=None,
            last_error=progress.last_error,
        )
    reference = progress.last_progress_monotonic
    progress_age = _age(now, runtime_started if reference is None else reference)
    if progress_age <= stale_seconds:
        return None
    return HealthFinding(
        code=f"runtime_{component}_stale",
        component=component,
        severity="critical",
        condition="stale",
        message=(
            f"{component} worker made no progress for {progress_age:.3f}s "
            f"(limit {stale_seconds:.3f}s)"
        ),
        work_active=True,
        age_seconds=progress_age,
        threshold_seconds=stale_seconds,
        last_error=progress.last_error,
    )


def _active_work_finding(
    *,
    component: str,
    progress: ActiveWorkProgress,
    severity: HealthSeverity,
    stale_condition: Literal["stale", "stalled"],
    now: float,
    runtime_started: float,
    stale_seconds: float,
) -> HealthFinding | None:
    if not progress.expected:
        return None
    if not progress.alive:
        detail = f": {progress.last_error}" if progress.last_error else ""
        return HealthFinding(
            code=f"runtime_{component}_dead",
            component=component,
            severity=severity,
            condition="dead",
            message=f"{component} worker is not alive{detail}",
            work_active=progress.has_active_work,
            age_seconds=None,
            threshold_seconds=None,
            last_error=progress.last_error,
        )
    if not progress.has_active_work:
        return None

    references = [
        timestamp
        for timestamp in (
            progress.last_progress_monotonic,
            progress.active_since_monotonic,
        )
        if timestamp is not None
    ]
    reference = max(references, default=runtime_started)
    progress_age = _age(now, reference)
    if progress_age <= stale_seconds:
        return None
    return HealthFinding(
        code=f"runtime_{component}_{stale_condition}",
        component=component,
        severity=severity,
        condition=stale_condition,
        message=(
            f"{component} active work made no progress for {progress_age:.3f}s "
            f"(limit {stale_seconds:.3f}s)"
        ),
        work_active=True,
        age_seconds=progress_age,
        threshold_seconds=stale_seconds,
        last_error=progress.last_error,
    )


def evaluate_runtime_health(
    snapshot: RuntimeProgressSnapshot,
    thresholds: RuntimeHealthThresholds = DEFAULT_RUNTIME_HEALTH_THRESHOLDS,
) -> RuntimeHealthReport:
    """Evaluate progress without logging, recovery, hardware, or other mutation.

    Dead/stale camera, motion, classifier, and auto-fire decision workers are
    critical.  Event writer/storage failures are reported as degraded so
    evidence failure cannot block tracking.  Queue-backed staleness is checked
    only while work is actually queued or in progress.
    """

    if not isinstance(snapshot, RuntimeProgressSnapshot):
        raise ValueError("snapshot must be RuntimeProgressSnapshot")
    if not isinstance(thresholds, RuntimeHealthThresholds):
        raise ValueError("thresholds must be RuntimeHealthThresholds")

    uptime = snapshot.observed_monotonic - snapshot.runtime_started_monotonic
    grace_remaining = max(0.0, thresholds.startup_grace_seconds - uptime)
    in_startup_grace = uptime < thresholds.startup_grace_seconds
    if in_startup_grace:
        return RuntimeHealthReport(
            snapshot=snapshot,
            thresholds=thresholds,
            uptime_seconds=uptime,
            startup_grace_remaining_seconds=grace_remaining,
            in_startup_grace=True,
            findings=(),
        )

    possible_findings = (
        _continuous_finding(
            component="camera",
            progress=snapshot.camera,
            now=snapshot.observed_monotonic,
            runtime_started=snapshot.runtime_started_monotonic,
            stale_seconds=thresholds.camera_stale_seconds,
        ),
        _continuous_finding(
            component="motion",
            progress=snapshot.motion,
            now=snapshot.observed_monotonic,
            runtime_started=snapshot.runtime_started_monotonic,
            stale_seconds=thresholds.motion_stale_seconds,
        ),
        _active_work_finding(
            component="classifier",
            progress=snapshot.classifier,
            severity="critical",
            stale_condition="stale",
            now=snapshot.observed_monotonic,
            runtime_started=snapshot.runtime_started_monotonic,
            stale_seconds=thresholds.classifier_stale_seconds,
        ),
        _active_work_finding(
            component="auto_fire_decision",
            progress=snapshot.auto_fire_decision,
            severity="critical",
            stale_condition="stale",
            now=snapshot.observed_monotonic,
            runtime_started=snapshot.runtime_started_monotonic,
            stale_seconds=thresholds.auto_fire_decision_stale_seconds,
        ),
        _active_work_finding(
            component="event_writer",
            progress=snapshot.event_writer,
            severity="degraded",
            stale_condition="stalled",
            now=snapshot.observed_monotonic,
            runtime_started=snapshot.runtime_started_monotonic,
            stale_seconds=thresholds.storage_stale_seconds,
        ),
        _active_work_finding(
            component="event_storage",
            progress=snapshot.event_storage,
            severity="degraded",
            stale_condition="stalled",
            now=snapshot.observed_monotonic,
            runtime_started=snapshot.runtime_started_monotonic,
            stale_seconds=thresholds.storage_stale_seconds,
        ),
    )
    findings = tuple(finding for finding in possible_findings if finding is not None)
    return RuntimeHealthReport(
        snapshot=snapshot,
        thresholds=thresholds,
        uptime_seconds=uptime,
        startup_grace_remaining_seconds=grace_remaining,
        in_startup_grace=False,
        findings=findings,
    )
