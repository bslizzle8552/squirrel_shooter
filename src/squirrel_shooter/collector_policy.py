"""Collector-only presentation and evidence policy. No engagement authority."""
from dataclasses import asdict, dataclass
import math
import threading
from time import monotonic

from .recording import RecordingError
from .watch_zone import WatchZoneConfig
from collections import deque


@dataclass(frozen=True)
class CollectorPreviewConfig:
    enabled: bool = True
    maximum_fps: float = 2.0
    jpeg_quality: int = 65
    maximum_width: int = 640

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('collector_preview.enabled must be boolean')
        if (isinstance(self.maximum_fps, bool) or not isinstance(self.maximum_fps, (int, float))
                or not math.isfinite(self.maximum_fps) or not 1 <= self.maximum_fps <= 5):
            raise ValueError('collector_preview.maximum_fps must be 1..5')
        if type(self.jpeg_quality) is not int or not 30 <= self.jpeg_quality <= 85:
            raise ValueError('collector_preview.jpeg_quality must be 30..85')
        if type(self.maximum_width) is not int or not 160 <= self.maximum_width <= 1280:
            raise ValueError('collector_preview.maximum_width must be 160..1280')


@dataclass(frozen=True)
class AutomaticRecordingConfig:
    enabled: bool = False
    squirrel_confidence: float = 0.02  # RECORDING ONLY; never an engagement threshold.
    tail_seconds: float = 10.0

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('automatic_recording.enabled must be boolean')
        for key, low, high in [('squirrel_confidence', 0.000001, 1), ('tail_seconds', 1, 60)]:
            value = getattr(self, key)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not low <= value <= high):
                raise ValueError(f'automatic_recording.{key} outside bounds')


class SquirrelRecordingObserver:
    """Collector policy; exact persistence completes before accepting a trip."""
    def __init__(self, recording, config, maximum_age, *, clock=monotonic, watch_zone=None, require_evidence=False):
        self.recording, self.config, self.maximum_age = recording, config, maximum_age
        self._clock = clock
        self.watch_zone = watch_zone or WatchZoneConfig()
        self.require_evidence = require_evidence
        self._raw = deque(maxlen=128)
        self._outside = 0
        self._save_latencies = deque(maxlen=128)
        self._lock = threading.Lock()
        self._last = None
        self._last_identity = (-1, -1)
        self._count = 0
        self._error = None
        self._paused = False

    def set_paused(self, paused):
        with self._lock:
            self._paused = paused

    def observe(self, observation, packet=None):
        if not self.config.enabled:
            return
        age = self._clock() - observation.source_monotonic
        if (not observation.output.available or observation.dropped_reason
                or not math.isfinite(age) or not 0 <= age <= self.maximum_age):
            return
        boxes = []
        with self._lock:
            if self._paused:
                return
            for d in observation.output.detections:
                zone = self.watch_zone.test_box(d.xyxy, observation.native_width, observation.native_height)
                self._raw.append(dict(confidence=d.confidence,native_box=list(d.xyxy),source_sequence=observation.source_sequence,
                    source_generation=observation.camera_generation,source_monotonic=observation.source_monotonic,**zone))
                if not zone['watch_zone_inside']:
                    self._outside += 1
                elif math.isfinite(d.confidence) and self.config.squirrel_confidence <= d.confidence <= 1:
                    boxes.append(d)
        if not boxes:
            return
        identity = (observation.camera_generation, observation.source_sequence)
        with self._lock:
            if self._paused:
                return
            if identity <= self._last_identity:
                return
            best = max(boxes, key=lambda d: d.confidence)
            zone = self.watch_zone.test_box(best.xyxy, observation.native_width, observation.native_height)
            item = dict(**zone, image_dimensions=[observation.native_width,observation.native_height], model_identity=asdict(observation.output.identity) if observation.output.identity else None,
                        confidence=best.confidence, native_box=list(best.xyxy),
                        source_generation=observation.camera_generation,
                        source_sequence=observation.source_sequence,
                        source_monotonic=observation.source_monotonic,
                        source_wall_time=observation.source_wall_time,
                        capture_to_result_age=observation.capture_to_result_age,
                        recording_only_confidence_threshold=self.config.squirrel_confidence)
            try:
                if self.require_evidence and packet is None:
                    raise RecordingError('exact_trip_source_missing')
                result = self.recording.extend_automatic_recording(
                    event_id='collector-squirrel-observer', visit_id=None, reason='squirrel_detection',
                    observed_monotonic=observation.source_monotonic, observation=item, return_status=False, evidence_packet=packet if self.require_evidence else None)
            except RecordingError as exc:
                self._error = str(exc)
                return
            if self.require_evidence:
                evidence = result.get("evidence")
                if evidence:
                    item = evidence
                    self._save_latencies.append(evidence["save_latency_seconds"])
            self._last_identity = identity
            self._last = item
            self._count += 1
            self._error = None

    def status(self):
        with self._lock:
            age = self._clock() - self._last['source_monotonic'] if self._last else None
            return dict(enabled=self.config.enabled, policy='RECORDING ONLY',
                        paused=self._paused,
                        squirrel_confidence=self.config.squirrel_confidence,
                        tail_seconds=self.config.tail_seconds, qualifying_count=self._count,
                        last_observation=self._last, last_seen_seconds=age,
                        detected_recently=not self._paused and age is not None and 0 <= age <= self.config.tail_seconds,
                        ignored_outside_count=self._outside, raw_diagnostics=list(self._raw),
                        trip_evidence_required=self.require_evidence, trip_save_latency_seconds=list(self._save_latencies),
                        error=self._error)
