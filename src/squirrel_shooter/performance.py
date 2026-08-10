"""Tiny dependency-free timing helpers for aggregated runtime observability."""

from __future__ import annotations

from time import monotonic, thread_time


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
