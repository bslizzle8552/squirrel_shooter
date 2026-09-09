"""Collector-only, sampled monochrome detection. No clock schedule or ML."""
from dataclasses import asdict, dataclass
import math
from time import monotonic
import threading

import numpy as np


@dataclass(frozen=True)
class CollectorNightConfig:
    enabled: bool = False  # Explicitly enabled in the isolated collector config.
    sample_seconds: float = 1.0
    enter_seconds: float = 5.0
    exit_seconds: float = 15.0
    monochrome_p90: float = 3.0
    color_p90: float = 8.0

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('collector_night.enabled must be boolean')
        for key, low, high in [('sample_seconds', 1, 5), ('enter_seconds', 3, 60),
                               ('exit_seconds', 5, 120), ('monochrome_p90', 0, 20), ('color_p90', 1, 64)]:
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f'collector_night.{key} outside bounds')
        if self.color_p90 <= self.monochrome_p90 or self.exit_seconds < self.enter_seconds:
            raise ValueError('collector_night requires separated thresholds and longer daylight confirmation')


class NightVisionMonitor:
    def __init__(self, config, *, clock=monotonic):
        self.config, self._clock = config, clock
        self._lock = threading.Lock()
        self._mode = 'CHECKING' if config.enabled else 'DAY'
        self._candidate = None
        self._since = None
        self._last_sample = None
        self._identity = None
        self._generation = None
        self._shape = None
        self._p90 = None
        self._samples = 0
        self._transitions = 0

    def sample(self, packet):
        """Read at most 64x48 native pixels once/second; never change pixels."""
        with self._lock:
            now = self._clock()
            if not self.config.enabled or (self._last_sample is not None and now - self._last_sample < self.config.sample_seconds):
                return
            identity = (packet.generation, packet.sequence)
            if identity == self._identity or not 0 <= now - packet.received_monotonic <= 2:
                return
            shape = packet.frame.shape
            if len(shape) != 3 or shape[2] != 3 or packet.frame.dtype != np.uint8:
                return
            if (self._generation, self._shape) != (packet.generation, shape):
                self._mode, self._candidate, self._since = 'CHECKING', None, None
            if self._last_sample is not None and now - self._last_sample > self.config.sample_seconds * 2.5:
                self._candidate, self._since = None, None
            self._generation, self._shape, self._identity = packet.generation, shape, identity
            self._last_sample = now
            small = packet.frame[::max(1, math.ceil(shape[0]/48)), ::max(1, math.ceil(shape[1]/64))]
            spread = small.max(axis=2) - small.min(axis=2)
            self._p90 = float(np.percentile(spread, 90))
            self._samples += 1
            candidate = 'NIGHT' if self._p90 <= self.config.monochrome_p90 else 'DAY' if self._p90 >= self.config.color_p90 else None
            if candidate is None or candidate == self._mode:
                self._candidate, self._since = None, None
                return True
            if candidate != self._candidate:
                self._candidate, self._since = candidate, now
            duration = self.config.enter_seconds if candidate == 'NIGHT' else self.config.exit_seconds
            if now - self._since >= duration:
                self._mode = candidate
                self._candidate, self._since = None, None
                self._transitions += 1
            return True

    def status(self):
        with self._lock:
            return dict(mode=self._mode, paused=self._mode != 'DAY',
                        method='sampled_native_channel_spread_p90' if self.config.enabled else 'disabled',
                        hardware_state_available=False, channel_spread_p90=self._p90,
                        candidate=self._candidate, candidate_seconds=0 if self._since is None else max(0, self._clock()-self._since),
                        sample_count=self._samples, transition_count=self._transitions,
                        last_sample_age_seconds=None if self._last_sample is None else self._clock()-self._last_sample,
                        settings=asdict(self.config))
