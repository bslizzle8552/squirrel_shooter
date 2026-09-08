"""Collector-only presentation and evidence policy. No engagement authority."""
from dataclasses import asdict, dataclass
import math
import threading
from time import monotonic

from .recording import RecordingError


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
    """One worker callback, no disk I/O or biological visit association."""
    def __init__(self, recording, config, maximum_age, *, clock=monotonic):
        self.recording, self.config, self.maximum_age = recording, config, maximum_age
        self._clock = clock
        self._lock = threading.Lock()
        self._last = None
        self._last_identity = (-1, -1)
        self._count = 0
        self._error = None

    def observe(self, observation):
        if not self.config.enabled:
            return
        age = self._clock() - observation.source_monotonic
        if (not observation.output.available or observation.dropped_reason
                or not math.isfinite(age) or not 0 <= age <= self.maximum_age):
            return
        boxes = [d for d in observation.output.detections
                 if math.isfinite(d.confidence) and self.config.squirrel_confidence <= d.confidence <= 1]
        if not boxes:
            return
        identity = (observation.camera_generation, observation.source_sequence)
        with self._lock:
            if identity <= self._last_identity:
                return
            best = max(boxes, key=lambda d: d.confidence)
            item = dict(model_identity=asdict(observation.output.identity) if observation.output.identity else None,
                        confidence=best.confidence, native_box=list(best.xyxy),
                        source_generation=observation.camera_generation,
                        source_sequence=observation.source_sequence,
                        source_monotonic=observation.source_monotonic,
                        source_wall_time=observation.source_wall_time,
                        capture_to_result_age=observation.capture_to_result_age,
                        recording_only_confidence_threshold=self.config.squirrel_confidence)
            try:
                self.recording.extend_automatic_recording(
                    event_id='collector-squirrel-observer', visit_id=None, reason='squirrel_detection',
                    observed_monotonic=observation.source_monotonic, observation=item, return_status=False)
            except RecordingError as exc:
                self._error = str(exc)
                return
            self._last_identity = identity
            self._last = item
            self._count += 1
            self._error = None

    def status(self):
        with self._lock:
            age = self._clock() - self._last['source_monotonic'] if self._last else None
            return dict(enabled=self.config.enabled, policy='RECORDING ONLY',
                        squirrel_confidence=self.config.squirrel_confidence,
                        tail_seconds=self.config.tail_seconds, qualifying_count=self._count,
                        last_observation=self._last, last_seen_seconds=age,
                        detected_recently=age is not None and 0 <= age <= self.config.tail_seconds,
                        error=self._error)
