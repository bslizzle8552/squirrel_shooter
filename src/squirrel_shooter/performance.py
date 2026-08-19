"""Tiny dependency-free timing helpers for aggregated runtime observability."""

from __future__ import annotations

import math
import threading
from collections import deque
from collections.abc import Mapping
from time import monotonic, thread_time


DEFAULT_TIMING_SAMPLE_CAPACITY = 2048
DEFAULT_TIMING_ANOMALY_THRESHOLDS_SECONDS: tuple[tuple[str, float], ...] = (
    ("gte_100ms", 0.100),
    ("gte_250ms", 0.250),
    ("gte_500ms", 0.500),
    ("gte_900ms", 0.900),
)


class AverageTimer:
    """Track a cumulative average without retaining individual samples."""

    def __init__(self) -> None:
        self._seconds = 0.0
        self._samples = 0

    def add(self, seconds: float) -> None:
        self._seconds += max(0.0, float(seconds))
        self._samples += 1

    @property
    def average_ms(self) -> float:
        return 0.0 if self._samples == 0 else 1000.0 * self._seconds / self._samples


class TimingDistribution:
    """Retain a bounded timing window and report thread-safe tail statistics.

    Durations and anomaly thresholds are supplied in seconds to match Python's
    monotonic clocks. Snapshots report durations in milliseconds. Percentiles
    and ``anomaly_counts`` describe only the retained window, while
    ``total_count`` and ``total_anomaly_counts`` are lifetime counters since
    construction or the most recent reset.
    """

    def __init__(
        self,
        capacity: int = DEFAULT_TIMING_SAMPLE_CAPACITY,
        *,
        anomaly_thresholds_seconds: Mapping[str, float] | None = None,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        configured_thresholds = (
            dict(DEFAULT_TIMING_ANOMALY_THRESHOLDS_SECONDS)
            if anomaly_thresholds_seconds is None
            else dict(anomaly_thresholds_seconds)
        )
        thresholds: list[tuple[str, float]] = []
        for name, raw_threshold in configured_thresholds.items():
            if not isinstance(name, str) or not name.strip() or name != name.strip():
                raise ValueError("anomaly threshold names must be non-empty trimmed strings")
            if isinstance(raw_threshold, bool):
                raise ValueError(f"anomaly threshold {name!r} must be a finite non-negative number")
            try:
                threshold = float(raw_threshold)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"anomaly threshold {name!r} must be a finite non-negative number"
                ) from exc
            if not math.isfinite(threshold) or threshold < 0.0:
                raise ValueError(f"anomaly threshold {name!r} must be a finite non-negative number")
            thresholds.append((name, threshold))

        self.capacity = capacity
        self._thresholds = tuple(sorted(thresholds, key=lambda item: (item[1], item[0])))
        self._samples: deque[float] = deque(maxlen=capacity)
        self._total_count = 0
        self._total_anomaly_counts = {name: 0 for name, _ in self._thresholds}
        self._lock = threading.Lock()

    def add(self, seconds: float) -> None:
        """Record one duration, clamping negative clock noise to zero."""

        if isinstance(seconds, bool):
            raise ValueError("seconds must be a finite number")
        try:
            sample = float(seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError("seconds must be a finite number") from exc
        if not math.isfinite(sample):
            raise ValueError("seconds must be a finite number")
        sample = max(0.0, sample)
        with self._lock:
            self._samples.append(sample)
            self._total_count += 1
            for name, threshold in self._thresholds:
                if sample >= threshold:
                    self._total_anomaly_counts[name] += 1

    def reset(self) -> None:
        """Clear the rolling window and lifetime counters atomically."""

        with self._lock:
            self._samples.clear()
            self._total_count = 0
            for name in self._total_anomaly_counts:
                self._total_anomaly_counts[name] = 0

    def snapshot(self) -> dict[str, object]:
        """Return one internally consistent, JSON-friendly metrics snapshot."""

        with self._lock:
            samples = tuple(self._samples)
            total_count = self._total_count
            total_anomaly_counts = dict(self._total_anomaly_counts)
        ordered = sorted(samples)
        anomaly_counts = {
            name: sum(sample >= threshold for sample in samples)
            for name, threshold in self._thresholds
        }
        return {
            "count": len(samples),
            "total_count": total_count,
            "capacity": self.capacity,
            "p50_ms": 1000.0 * self._percentile(ordered, 0.50),
            "p90_ms": 1000.0 * self._percentile(ordered, 0.90),
            "p95_ms": 1000.0 * self._percentile(ordered, 0.95),
            "p99_ms": 1000.0 * self._percentile(ordered, 0.99),
            "max_ms": 0.0 if not ordered else 1000.0 * ordered[-1],
            "anomaly_thresholds_ms": {
                name: 1000.0 * threshold for name, threshold in self._thresholds
            },
            "anomaly_counts": anomaly_counts,
            "total_anomaly_counts": total_anomaly_counts,
        }

    @staticmethod
    def _percentile(ordered: list[float], percentile: float) -> float:
        """Return an R-7 linearly interpolated percentile from sorted samples."""

        if not ordered:
            return 0.0
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * percentile
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


class ThreadCpuMeter:
    """Estimate one worker thread's CPU usage over rolling one-second windows."""

    def __init__(self) -> None:
        self._wall = monotonic()
        self._cpu = thread_time()
        self.percent = 0.0

    def update(self) -> None:
        now_wall = monotonic()
        elapsed_wall = now_wall - self._wall
        if elapsed_wall < 1.0:
            return
        now_cpu = thread_time()
        self.percent = max(0.0, 100.0 * (now_cpu - self._cpu) / elapsed_wall)
        self._wall = now_wall
        self._cpu = now_cpu
