"""Latest-only recurring inference: one owner, one replaceable pending reference."""
from __future__ import annotations

import math
import threading
from collections import Counter, deque
from dataclasses import asdict, dataclass, replace
from time import monotonic

from .detector import DetectionOutput, DetectorConfig, SquirrelDetectorAdapter


@dataclass(frozen=True)
class DetectorObservation:
    camera_generation: int
    source_sequence: int
    native_width: int
    native_height: int
    source_monotonic: float
    source_wall_time: str
    inference_start: float | None
    inference_end: float | None
    capture_to_result_age: float
    output: DetectionOutput
    dropped_reason: str | None = None
    # Phase 4 association seam: no invented event/track/biological visit identity.
    track_id: str | None = None
    event_id: str | None = None
    visit_id: str | None = None


class LatestFrameInferenceWorker:
    def __init__(self, camera, detector: SquirrelDetectorAdapter, config: DetectorConfig, *, clock=monotonic):
        self.camera, self.detector, self.config = camera, detector, config
        self._clock = clock
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._pending = None
        self._inflight = False
        self._last_sequence = -1
        self._generation = -1
        self._shape = None
        self._revision = 0
        self._next_due = 0.0
        self._result = None
        self._last_drop = None
        self._counts = Counter()
        self._latencies = deque(maxlen=128)
        self._threads = []
        self._state = 'idle' if config.enabled else 'disabled'
        self._backend_status = detector.status()
        self._shutdown_timed_out = False

    def _drop(self, packet, reason, now):
        source_time = packet.received_monotonic
        if not math.isfinite(source_time):
            source_time = None
        self._last_drop = dict(camera_generation=packet.generation, source_sequence=packet.sequence,
                               native_width=packet.frame.shape[1], native_height=packet.frame.shape[0],
                               source_monotonic=source_time, source_wall_time=packet.received_at,
                               model_identity=self._backend_status.get('model_identity'),
                               dropped_reason=reason, observed_monotonic=now)

    def submit(self, packet):
        now = self._clock()
        with self._condition:
            if self._stop.is_set() or not self.config.enabled:
                return False
            if packet.sequence <= self._last_sequence or packet.generation < self._generation:
                self._counts['duplicate_or_regressed_count'] += 1
                return False
            self._last_sequence = packet.sequence
            shape = tuple(packet.frame.shape)
            if len(shape) != 3 or shape[2] != 3 or min(shape[:2]) < 1:
                self._counts['error_unavailable_count'] += 1
                return False
            if self._generation != packet.generation or self._shape != shape:
                self._revision += 1
                if self._generation >= 0:
                    self._counts['source_change_count'] += 1
                self._generation, self._shape = packet.generation, shape
                if self._pending is not None:
                    self._drop(self._pending, 'source_changed', now)
                self._pending = None
                self._result = None
            age = now - packet.received_monotonic
            if not math.isfinite(age) or age < 0 or age > self.config.maximum_source_age_seconds:
                self._counts['stale_before_inference_count'] += 1
                self._drop(packet, 'stale_before_inference', now)
                return False
            if self._pending is not None:
                self._counts['superseded_pending_count'] += 1
                self._drop(self._pending, 'superseded_pending', now)
            if now < self._next_due:
                self._counts['skipped_cadence_count'] += 1
            self._pending = packet
            self._condition.notify_all()
            return True

    def run_once(self):
        """Claim atomically, infer outside the lock; also used by deterministic tests."""
        with self._condition:
            now = self._clock()
            if self._stop.is_set() or self._inflight or self._pending is None or now < self._next_due:
                return False
            packet, self._pending = self._pending, None
            age = now - packet.received_monotonic
            if not math.isfinite(age) or age < 0 or age > self.config.maximum_source_age_seconds:
                self._counts['stale_before_inference_count'] += 1
                self._drop(packet, 'stale_before_inference', now)
                return False
            revision = self._revision
            self._inflight = True
            self._state = 'inferencing'
            start = now
        try:
            output = self.detector.infer(packet.frame)
        except Exception as exc:
            output = DetectionOutput(False, None, error=str(exc))
        end = self._clock()
        with self._condition:
            self._inflight = False
            period = 1 / self.config.target_hz
            # A missed deadline is discarded. Slow inference earns a full quiet interval.
            self._next_due = start + period if end - start < period else end + period
            self._latencies.append(end - start)
            age = end - packet.received_monotonic
            reason = None
            if self._stop.is_set():
                reason = 'shutdown'
            elif revision != self._revision:
                reason = 'source_changed_during_inference'
            elif not math.isfinite(age) or age < 0 or end < start or age > self.config.maximum_result_age_seconds:
                reason = 'stale_after_inference'
                self._counts['stale_after_inference_count'] += 1
            if reason:
                output = replace(output, available=False, detections=(), error=reason)
                self._drop(packet, reason, end)
            if not output.available:
                self._counts['error_unavailable_count'] += 1
            self._backend_status.update(available=output.available, error=output.error)
            observation = DetectorObservation(packet.generation, packet.sequence, packet.frame.shape[1],
                packet.frame.shape[0], packet.received_monotonic, packet.received_at, start, end, age, output, reason)
            if revision == self._revision and not self._stop.is_set():
                self._result = observation
            self._state = 'stopped' if self._stop.is_set() else ('idle' if output.available else 'unavailable')
            self._counts['inference_count'] += 1
            self._condition.notify_all()
        return True

    def start(self):
        with self._condition:
            if self._threads or self._stop.is_set() or not self.config.enabled:
                return
            self._threads = [threading.Thread(target=self._collect, daemon=True, name='detector-latest'),
                             threading.Thread(target=self._infer_loop, daemon=True, name='detector-infer')]
            for thread in self._threads:
                thread.start()

    def _collect(self):
        sequence = -1
        while not self._stop.is_set():
            try:
                packet = self.camera.wait_for_frame(sequence, timeout=.1, copy=False)
                if packet is not None:
                    sequence = max(sequence, packet.sequence)
                    self.submit(packet)
            except Exception:
                with self._condition:
                    self._counts['error_unavailable_count'] += 1
                    self._state = 'camera_unavailable'
                self._stop.wait(.1)

    def _infer_loop(self):
        while not self._stop.is_set():
            if not self.run_once():
                with self._condition:
                    self._condition.wait(timeout=min(.1, max(.001, self._next_due-self._clock())))

    def stop(self):
        self._stop.set()
        with self._condition:
            self._pending = None
            self._result = None
            self._condition.notify_all()
        deadline = monotonic() + self.config.shutdown_timeout_seconds
        for thread in self._threads:
            thread.join(max(0, deadline - monotonic()))
        with self._condition:
            self._shutdown_timed_out = any(t.is_alive() for t in self._threads)
            self._state = 'shutdown_timed_out' if self._shutdown_timed_out else 'stopped'

    def status(self):
        with self._condition:
            result = self._result
            now = self._clock()
            current_age = now - result.source_monotonic if result else None
            if result and (current_age < 0 or current_age > self.config.maximum_result_age_seconds):
                result = replace(result, output=replace(result.output, available=False, detections=(), error='expired'), dropped_reason='expired')
            output = result.output if result else None
            counters = {name:self._counts[name] for name in (
                'skipped_cadence_count', 'superseded_pending_count', 'stale_before_inference_count',
                'stale_after_inference_count', 'error_unavailable_count', 'duplicate_or_regressed_count',
                'source_change_count', 'inference_count')}
            return dict(self._backend_status, **counters, worker_state=self._state,
                        nominal_cadence_hz=self.config.target_hz, in_flight=self._inflight,
                        pending_frame=self._pending is not None,
                        pending_sequence=self._pending.sequence if self._pending else None,
                        last_source_sequence=result.source_sequence if result else None,
                        last_source_age_seconds=current_age,
                        last_inference_latency_seconds=(result.inference_end-result.inference_start) if result else None,
                        last_capture_to_result_age_seconds=result.capture_to_result_age if result else None,
                        last_detection_count=len(output.detections) if output else 0,
                        last_top_squirrel_confidence=max((d.confidence for d in output.detections), default=None) if output else None,
                        result_available=output.available if output else False,
                        last_result=asdict(result) if result else None, last_drop=dict(self._last_drop) if self._last_drop else None,
                        rolling_latency_seconds=list(self._latencies), shutdown_timed_out=self._shutdown_timed_out)
