from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from squirrel_shooter.performance import TimingDistribution


def test_timing_distribution_empty_snapshot_has_stable_shape() -> None:
    distribution = TimingDistribution()

    snapshot = distribution.snapshot()

    assert snapshot == {
        "count": 0,
        "total_count": 0,
        "capacity": 2048,
        "p50_ms": 0.0,
        "p90_ms": 0.0,
        "p95_ms": 0.0,
        "p99_ms": 0.0,
        "max_ms": 0.0,
        "anomaly_thresholds_ms": {
            "gte_100ms": 100.0,
            "gte_250ms": 250.0,
            "gte_500ms": 500.0,
            "gte_900ms": 900.0,
        },
        "anomaly_counts": {
            "gte_100ms": 0,
            "gte_250ms": 0,
            "gte_500ms": 0,
            "gte_900ms": 0,
        },
        "total_anomaly_counts": {
            "gte_100ms": 0,
            "gte_250ms": 0,
            "gte_500ms": 0,
            "gte_900ms": 0,
        },
    }


def test_timing_distribution_reports_interpolated_percentiles_and_anomalies() -> None:
    distribution = TimingDistribution(
        capacity=200,
        anomaly_thresholds_seconds={"gte_50ms": 0.050, "gte_100ms": 0.100},
    )
    for milliseconds in range(1, 101):
        distribution.add(milliseconds / 1000.0)

    snapshot = distribution.snapshot()

    assert snapshot["count"] == snapshot["total_count"] == 100
    assert snapshot["p50_ms"] == pytest.approx(50.5)
    assert snapshot["p90_ms"] == pytest.approx(90.1)
    assert snapshot["p95_ms"] == pytest.approx(95.05)
    assert snapshot["p99_ms"] == pytest.approx(99.01)
    assert snapshot["max_ms"] == pytest.approx(100.0)
    assert snapshot["anomaly_counts"] == {"gte_50ms": 51, "gte_100ms": 1}
    assert snapshot["total_anomaly_counts"] == snapshot["anomaly_counts"]


def test_timing_distribution_evicts_old_samples_but_preserves_lifetime_counters() -> None:
    distribution = TimingDistribution(
        capacity=3,
        anomaly_thresholds_seconds={"gte_5ms": 0.005},
    )
    for seconds in (0.010, 0.001, 0.002, 0.003):
        distribution.add(seconds)

    snapshot = distribution.snapshot()

    assert snapshot["count"] == 3
    assert snapshot["total_count"] == 4
    assert snapshot["p50_ms"] == pytest.approx(2.0)
    assert snapshot["max_ms"] == pytest.approx(3.0)
    assert snapshot["anomaly_counts"] == {"gte_5ms": 0}
    assert snapshot["total_anomaly_counts"] == {"gte_5ms": 1}


def test_timing_distribution_is_thread_safe() -> None:
    distribution = TimingDistribution(
        capacity=2048,
        anomaly_thresholds_seconds={"gte_1ms": 0.001},
    )

    def record_batch() -> None:
        for _ in range(250):
            distribution.add(0.001)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(record_batch) for _ in range(8)]
        for future in futures:
            future.result()

    snapshot = distribution.snapshot()
    assert snapshot["count"] == snapshot["total_count"] == 2000
    assert snapshot["p50_ms"] == snapshot["p99_ms"] == snapshot["max_ms"] == 1.0
    assert snapshot["anomaly_counts"] == {"gte_1ms": 2000}
    assert snapshot["total_anomaly_counts"] == {"gte_1ms": 2000}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"capacity": 0}, "capacity"),
        ({"capacity": True}, "capacity"),
        ({"anomaly_thresholds_seconds": {"": 0.1}}, "threshold names"),
        ({"anomaly_thresholds_seconds": {"slow": -0.1}}, "finite non-negative"),
        ({"anomaly_thresholds_seconds": {"slow": float("inf")}}, "finite non-negative"),
    ],
)
def test_timing_distribution_rejects_invalid_configuration(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        TimingDistribution(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("sample", [True, float("nan"), float("inf"), "not-a-number"])
def test_timing_distribution_rejects_invalid_samples(sample: object) -> None:
    distribution = TimingDistribution(anomaly_thresholds_seconds={})

    with pytest.raises(ValueError, match="seconds must be a finite number"):
        distribution.add(sample)  # type: ignore[arg-type]


def test_timing_distribution_clamps_negative_noise_and_resets_atomically() -> None:
    distribution = TimingDistribution(
        capacity=2,
        anomaly_thresholds_seconds={"non_negative": 0.0},
    )
    distribution.add(-0.001)
    before = distribution.snapshot()
    before["anomaly_counts"]["non_negative"] = 99  # type: ignore[index]

    assert distribution.snapshot()["max_ms"] == 0.0
    assert distribution.snapshot()["anomaly_counts"] == {"non_negative": 1}

    distribution.reset()
    after = distribution.snapshot()
    assert after["count"] == after["total_count"] == 0
    assert after["anomaly_counts"] == after["total_anomaly_counts"] == {"non_negative": 0}
