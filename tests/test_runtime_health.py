from __future__ import annotations

import json

import pytest

from squirrel_shooter.runtime_health import (
    ActiveWorkProgress,
    ContinuousWorkerProgress,
    DEFAULT_RUNTIME_HEALTH_THRESHOLDS,
    RuntimeHealthThresholds,
    RuntimeProgressSnapshot,
    evaluate_runtime_health,
)


def continuous(
    *,
    alive: bool = True,
    progress: float | None = 99.0,
    expected: bool = True,
    error: str | None = None,
) -> ContinuousWorkerProgress:
    return ContinuousWorkerProgress(
        expected=expected,
        alive=alive,
        last_progress_monotonic=progress,
        last_error=error,
    )


def active(
    *,
    alive: bool = True,
    queued: int = 0,
    in_progress: bool = False,
    progress: float | None = 99.0,
    active_since: float | None = None,
    expected: bool = True,
    error: str | None = None,
) -> ActiveWorkProgress:
    return ActiveWorkProgress(
        expected=expected,
        alive=alive,
        queued_count=queued,
        in_progress=in_progress,
        last_progress_monotonic=progress,
        active_since_monotonic=active_since,
        last_error=error,
    )


def snapshot(
    *,
    now: float = 100.0,
    started: float = 80.0,
    camera: ContinuousWorkerProgress | None = None,
    motion: ContinuousWorkerProgress | None = None,
    classifier: ActiveWorkProgress | None = None,
    decision: ActiveWorkProgress | None = None,
    writer: ActiveWorkProgress | None = None,
    storage: ActiveWorkProgress | None = None,
) -> RuntimeProgressSnapshot:
    return RuntimeProgressSnapshot(
        observed_monotonic=now,
        runtime_started_monotonic=started,
        camera=camera or continuous(),
        motion=motion or continuous(),
        classifier=classifier or active(),
        auto_fire_decision=decision or active(),
        event_writer=writer or active(expected=False),
        event_storage=storage or active(),
    )


def test_default_thresholds_are_bounded_and_json_friendly() -> None:
    assert DEFAULT_RUNTIME_HEALTH_THRESHOLDS.to_dict() == {
        "camera_stale_seconds": 3.0,
        "motion_stale_seconds": 5.0,
        "classifier_stale_seconds": 5.0,
        "auto_fire_decision_stale_seconds": 5.0,
        "storage_stale_seconds": 2.0,
        "startup_grace_seconds": 10.0,
    }
    assert json.loads(json.dumps(DEFAULT_RUNTIME_HEALTH_THRESHOLDS.to_dict())) == (
        DEFAULT_RUNTIME_HEALTH_THRESHOLDS.to_dict()
    )


@pytest.mark.parametrize(
    "field_name",
    [
        "camera_stale_seconds",
        "motion_stale_seconds",
        "classifier_stale_seconds",
        "auto_fire_decision_stale_seconds",
        "storage_stale_seconds",
        "startup_grace_seconds",
    ],
)
@pytest.mark.parametrize("invalid", [0, -0.1, True, float("nan"), float("inf"), "5"])
def test_thresholds_reject_values_that_are_not_finite_positive_numbers(
    field_name: str,
    invalid: object,
) -> None:
    kwargs = {field_name: invalid}
    with pytest.raises(ValueError, match=field_name):
        RuntimeHealthThresholds(**kwargs)  # type: ignore[arg-type]


def test_progress_snapshot_and_report_are_explicitly_json_friendly() -> None:
    progress = snapshot(
        classifier=active(queued=2, active_since=98.0, error="retrying"),
        writer=active(expected=True, in_progress=True, active_since=98.5),
    )

    report = evaluate_runtime_health(progress)
    payload = json.loads(json.dumps(report.to_dict(), allow_nan=False))

    assert payload["progress"] == progress.to_dict()
    assert payload["progress"]["classifier"]["has_active_work"] is True
    assert payload["thresholds"] == DEFAULT_RUNTIME_HEALTH_THRESHOLDS.to_dict()
    assert payload["status"] == "healthy"
    assert payload["finding_codes"] == []


def test_startup_grace_is_reported_and_suppresses_findings() -> None:
    progress = snapshot(
        now=109.999,
        started=100.0,
        camera=continuous(alive=False, progress=None),
        motion=continuous(alive=False, progress=None),
        classifier=active(alive=False, queued=1, progress=None),
        decision=active(alive=False, in_progress=True, progress=None),
        writer=active(alive=False, expected=True, queued=1, progress=None),
        storage=active(alive=False, progress=None),
    )

    report = evaluate_runtime_health(progress)

    assert report.status == "starting"
    assert report.in_startup_grace is True
    assert report.startup_grace_remaining_seconds == pytest.approx(0.001)
    assert report.findings == ()
    assert report.fail_safe_required is False
    assert report.diagnostic_key is None


def test_startup_grace_ends_at_exact_boundary() -> None:
    report = evaluate_runtime_health(
        snapshot(
            now=110.0,
            started=100.0,
            camera=continuous(alive=False, progress=None),
            motion=continuous(progress=110.0),
        )
    )

    assert report.in_startup_grace is False
    assert report.startup_grace_remaining_seconds == 0.0
    assert report.status == "critical"
    assert [finding.code for finding in report.findings] == ["runtime_camera_dead"]


@pytest.mark.parametrize(
    ("component", "replacement", "expected_code"),
    [
        ("camera", continuous(alive=False, progress=1.0), "runtime_camera_dead"),
        ("motion", continuous(alive=False, progress=1.0), "runtime_motion_dead"),
        ("classifier", active(alive=False), "runtime_classifier_dead"),
        (
            "decision",
            active(alive=False),
            "runtime_auto_fire_decision_dead",
        ),
    ],
)
def test_dead_critical_workers_require_fail_safe_even_when_queue_workers_are_idle(
    component: str,
    replacement: ContinuousWorkerProgress | ActiveWorkProgress,
    expected_code: str,
) -> None:
    report = evaluate_runtime_health(snapshot(**{component: replacement}))  # type: ignore[arg-type]

    assert report.status == "critical"
    assert report.fail_safe_required is True
    assert report.diagnostic_key == "runtime_progress_health_critical"
    assert [finding.code for finding in report.critical_findings] == [expected_code]
    assert report.findings[0].condition == "dead"
    assert report.findings[0].age_seconds is None


@pytest.mark.parametrize(
    ("component", "replacement", "expected_code", "threshold"),
    [
        ("camera", continuous(progress=96.999), "runtime_camera_stale", 3.0),
        ("motion", continuous(progress=94.999), "runtime_motion_stale", 5.0),
    ],
)
def test_continuous_workers_are_critical_after_progress_limit(
    component: str,
    replacement: ContinuousWorkerProgress,
    expected_code: str,
    threshold: float,
) -> None:
    report = evaluate_runtime_health(snapshot(**{component: replacement}))  # type: ignore[arg-type]

    finding = report.findings[0]
    assert finding.code == expected_code
    assert finding.severity == "critical"
    assert finding.condition == "stale"
    assert finding.age_seconds is not None and finding.age_seconds > threshold
    assert finding.threshold_seconds == threshold
    assert report.fail_safe_required is True


def test_exact_progress_threshold_is_still_healthy() -> None:
    report = evaluate_runtime_health(
        snapshot(
            camera=continuous(progress=97.0),
            motion=continuous(progress=95.0),
        )
    )

    assert report.status == "healthy"
    assert report.findings == ()


@pytest.mark.parametrize("component", ["classifier", "decision", "writer", "storage"])
def test_queue_backed_worker_staleness_is_suppressed_during_legitimate_idle(
    component: str,
) -> None:
    idle = active(expected=True, queued=0, in_progress=False, progress=1.0)

    report = evaluate_runtime_health(snapshot(**{component: idle}))  # type: ignore[arg-type]

    assert report.status == "healthy"
    assert report.findings == ()


@pytest.mark.parametrize(
    ("component", "replacement", "expected_code"),
    [
        (
            "classifier",
            active(queued=1, progress=94.0, active_since=94.0),
            "runtime_classifier_stale",
        ),
        (
            "decision",
            active(in_progress=True, progress=94.0, active_since=94.0),
            "runtime_auto_fire_decision_stale",
        ),
    ],
)
def test_active_classifier_and_decision_staleness_is_critical(
    component: str,
    replacement: ActiveWorkProgress,
    expected_code: str,
) -> None:
    report = evaluate_runtime_health(snapshot(**{component: replacement}))  # type: ignore[arg-type]

    assert [finding.code for finding in report.findings] == [expected_code]
    assert report.findings[0].work_active is True
    assert report.fail_safe_required is True


def test_new_active_period_wins_over_old_idle_progress_timestamp() -> None:
    report = evaluate_runtime_health(
        snapshot(
            classifier=active(
                queued=1,
                progress=1.0,
                active_since=99.0,
            )
        )
    )

    assert report.status == "healthy"
    assert report.findings == ()


def test_newer_worker_progress_wins_over_active_period_start() -> None:
    report = evaluate_runtime_health(
        snapshot(
            classifier=active(
                in_progress=True,
                progress=99.0,
                active_since=80.0,
            )
        )
    )

    assert report.status == "healthy"
    assert report.findings == ()


def test_missing_active_reference_falls_back_conservatively_to_runtime_start() -> None:
    report = evaluate_runtime_health(
        snapshot(
            classifier=active(
                queued=1,
                progress=None,
                active_since=None,
            )
        )
    )

    assert [finding.code for finding in report.findings] == ["runtime_classifier_stale"]


@pytest.mark.parametrize(
    ("component", "replacement", "expected_code"),
    [
        (
            "writer",
            active(expected=True, alive=False, error="encoder exited"),
            "runtime_event_writer_dead",
        ),
        (
            "storage",
            active(alive=False, error="finalizer exited"),
            "runtime_event_storage_dead",
        ),
        (
            "writer",
            active(expected=True, queued=1, progress=97.0, active_since=97.0),
            "runtime_event_writer_stalled",
        ),
        (
            "storage",
            active(in_progress=True, progress=97.0, active_since=97.0),
            "runtime_event_storage_stalled",
        ),
    ],
)
def test_event_evidence_worker_failures_are_degraded_not_fail_safe(
    component: str,
    replacement: ActiveWorkProgress,
    expected_code: str,
) -> None:
    report = evaluate_runtime_health(snapshot(**{component: replacement}))  # type: ignore[arg-type]

    assert report.status == "degraded"
    assert report.fail_safe_required is False
    assert report.degraded is True
    assert report.diagnostic_key == "runtime_progress_health_degraded"
    assert [finding.code for finding in report.degraded_findings] == [expected_code]


def test_mixed_findings_have_one_aggregate_critical_diagnostic_key() -> None:
    report = evaluate_runtime_health(
        snapshot(
            camera=continuous(alive=False, error="read blocked"),
            classifier=active(queued=1, progress=90.0, active_since=90.0),
            writer=active(expected=True, alive=False, error="disk error"),
            storage=active(in_progress=True, progress=90.0, active_since=90.0),
        )
    )

    assert report.status == "critical"
    assert report.diagnostic_key == "runtime_progress_health_critical"
    assert report.fail_safe_required is True
    assert report.degraded is True
    assert [finding.code for finding in report.findings] == [
        "runtime_camera_dead",
        "runtime_classifier_stale",
        "runtime_event_writer_dead",
        "runtime_event_storage_stalled",
    ]
    assert report.to_dict()["finding_codes"] == [
        finding.code for finding in report.findings
    ]
    assert report.findings[0].last_error == "read blocked"
    assert "read blocked" in report.findings[0].message


def test_unexpected_workers_are_ignored_even_if_dead_or_active() -> None:
    report = evaluate_runtime_health(
        snapshot(
            camera=continuous(expected=False, alive=False, progress=None),
            motion=continuous(expected=False, alive=False, progress=None),
            classifier=active(expected=False, alive=False, queued=3, progress=None),
            decision=active(expected=False, alive=False, in_progress=True, progress=None),
            writer=active(expected=False, alive=False, queued=3, progress=None),
            storage=active(expected=False, alive=False, in_progress=True, progress=None),
        )
    )

    assert report.status == "healthy"
    assert report.findings == ()


def test_future_progress_timestamp_is_tolerated_as_zero_age_snapshot_race() -> None:
    report = evaluate_runtime_health(
        snapshot(
            camera=continuous(progress=100.001),
            classifier=active(queued=1, progress=100.001, active_since=100.001),
        )
    )

    assert report.status == "healthy"


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: ContinuousWorkerProgress(
            expected=1,  # type: ignore[arg-type]
            alive=True,
            last_progress_monotonic=1.0,
        ),
        lambda: ContinuousWorkerProgress(
            expected=True,
            alive="yes",  # type: ignore[arg-type]
            last_progress_monotonic=1.0,
        ),
        lambda: ContinuousWorkerProgress(
            expected=True,
            alive=True,
            last_progress_monotonic=float("nan"),
        ),
        lambda: ActiveWorkProgress(
            expected=True,
            alive=True,
            queued_count=-1,
            in_progress=False,
            last_progress_monotonic=1.0,
        ),
        lambda: ActiveWorkProgress(
            expected=True,
            alive=True,
            queued_count=True,  # type: ignore[arg-type]
            in_progress=False,
            last_progress_monotonic=1.0,
        ),
        lambda: ActiveWorkProgress(
            expected=True,
            alive=True,
            queued_count=0,
            in_progress=1,  # type: ignore[arg-type]
            last_progress_monotonic=1.0,
        ),
        lambda: ActiveWorkProgress(
            expected=True,
            alive=True,
            queued_count=0,
            in_progress=False,
            last_progress_monotonic=1.0,
            active_since_monotonic=float("inf"),
        ),
    ],
)
def test_worker_snapshots_reject_ambiguous_or_non_finite_state(constructor: object) -> None:
    with pytest.raises(ValueError):
        constructor()  # type: ignore[operator]


def test_runtime_snapshot_rejects_clock_regression() -> None:
    with pytest.raises(ValueError, match="must not precede"):
        snapshot(now=99.0, started=100.0)


@pytest.mark.parametrize("argument", [None, object()])
def test_evaluator_rejects_unstructured_inputs(argument: object) -> None:
    with pytest.raises(ValueError):
        evaluate_runtime_health(argument)  # type: ignore[arg-type]
