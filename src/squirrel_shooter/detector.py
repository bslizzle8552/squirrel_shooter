"""One-class detector interface and exact NCNN preprocessing/native geometry."""
from __future__ import annotations

import math
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .model_manifest import ModelIdentity, ModelManifest


@dataclass(frozen=True)
class DetectorConfig:
    enabled: bool = False
    backend: str = 'ncnn'
    manifest_path: str = ''
    manifest_sha256: str = ''
    target_hz: float = 2.0
    maximum_source_age_seconds: float = 0.75
    maximum_result_age_seconds: float = 1.5
    diagnostic_confidence_floor: float = 0.01
    num_threads: int = 2  # Configurable provisional value; no Pi coexistence decision.
    shutdown_timeout_seconds: float = 2.0

    def __post_init__(self):
        if type(self.enabled) is not bool or self.backend != 'ncnn':
            raise ValueError('invalid detector enabled/backend')
        if not isinstance(self.manifest_path, str) or not isinstance(self.manifest_sha256, str):
            raise ValueError('detector manifest fields must be strings')
        for key in ('target_hz', 'maximum_source_age_seconds', 'maximum_result_age_seconds',
                    'diagnostic_confidence_floor', 'shutdown_timeout_seconds'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'invalid detector.{key}')
        if self.target_hz > 30 or self.diagnostic_confidence_floor > 1 or self.shutdown_timeout_seconds > 10:
            raise ValueError('detector settings outside bounds')
        if type(self.num_threads) is not int or not 1 <= self.num_threads <= 4:
            raise ValueError('detector.num_threads must be 1..4')


@dataclass(frozen=True)
class SquirrelBox:
    xyxy: tuple[float, float, float, float]
    confidence: float


@dataclass(frozen=True)
class DetectionOutput:
    available: bool
    identity: ModelIdentity | None
    detections: tuple[SquirrelBox, ...] = ()
    error: str | None = None


class SquirrelDetectorAdapter(Protocol):
    def infer(self, frame: np.ndarray) -> DetectionOutput: ...
    def status(self) -> dict: ...


def preprocess(frame):
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 1:
        raise ValueError('expected native uint8 BGR HWC frame')
    height, width = frame.shape[:2]
    ratio = min(416 / height, 416 / width)
    shape = (int(width * ratio), int(height * ratio))
    if min(shape) < 1:
        raise ValueError('unsupported extreme aspect ratio')
    padded = np.full((416, 416, 3), 114, dtype=np.uint8)
    resized = cv2.resize(frame, shape, interpolation=cv2.INTER_LINEAR)
    padded[:shape[1], :shape[0]] = resized
    return np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32), ratio


def decode(raw, ratio, width, height, floor):
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape != (3549, 6) or not np.isfinite(raw).all():
        raise ValueError('invalid/nonfinite raw output')
    if np.any(raw[:, 4:] < 0) or np.any(raw[:, 4:] > 1):
        raise ValueError('invalid probabilities')
    grids, strides = [], []
    for stride in (8, 16, 32):
        y, x = np.meshgrid(np.arange(416 // stride), np.arange(416 // stride), indexing='ij')
        grids.append(np.stack((x, y), axis=-1).reshape(-1, 2))
        strides.append(np.full((x.size, 1), stride))
    grid = np.concatenate(grids).astype(np.float32)
    stride = np.concatenate(strides).astype(np.float32)
    with np.errstate(over='raise', invalid='raise'):
        center = (raw[:, :2] + grid) * stride
        size = np.exp(raw[:, 2:4]) * stride
        boxes = np.concatenate((center - size / 2, center + size / 2), axis=1)
    if not np.isfinite(boxes).all():
        raise ValueError('nonfinite decoded geometry')
    scores = raw[:, 4] * raw[:, 5]
    indices = np.flatnonzero(scores >= floor)
    indices = indices[np.argsort(-scores[indices], kind='stable')]
    kept = []
    while indices.size:
        i = indices[0]
        kept.append(i)
        rest = indices[1:]
        left = np.maximum(boxes[i, :2], boxes[rest, :2])
        right = np.minimum(boxes[i, 2:], boxes[rest, 2:])
        overlap = np.prod(np.maximum(0, right - left), axis=1)
        area = np.prod(size[rest], axis=1) + np.prod(size[i]) - overlap
        iou = np.divide(overlap, area, out=np.zeros_like(overlap), where=area > 0)
        indices = rest[iou <= 0.65]
    result = []
    for i in kept:
        box = boxes[i] / ratio
        box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
        if box[2] > box[0] and box[3] > box[1]:
            result.append(SquirrelBox(tuple(map(float, box)), float(scores[i])))
    return tuple(result)


class NcnnSquirrelDetector:
    def __init__(self, config: DetectorConfig, *, backend_module=None):
        self.config = config
        self.manifest = None
        self.net = None
        self._lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._state = dict(enabled=config.enabled, configured=bool(config.manifest_path and config.manifest_sha256),
                           backend=config.backend, manifest_status='unchecked', artifact_hash_status='unchecked',
                           available=False, error=None, model_id=None, model_version=None)
        if not config.enabled:
            self._state['error'] = 'disabled'
            return
        try:
            self.manifest = ModelManifest.load(Path(config.manifest_path), config.manifest_sha256)
            self._state.update(manifest_status='valid', artifact_hash_status='valid',
                               model_identity=asdict(self.manifest.identity),
                               **{k:v for k,v in asdict(self.manifest.identity).items() if k in ('model_id', 'model_version')})
            if backend_module is None:
                import ncnn as backend_module
            if backend_module.__version__ != '1.0.20260526':
                raise ValueError('unsupported NCNN runtime version')
            self._backend = backend_module
            net = backend_module.Net()
            net.opt.num_threads = config.num_threads
            net.opt.use_vulkan_compute = False
            net.opt.use_fp16_packed = False
            net.opt.use_fp16_storage = False
            net.opt.use_fp16_arithmetic = False
            net.opt.use_bf16_storage = False
            net.opt.use_int8_inference = False
            if net.load_param(str(self.manifest.param_path)) != 0 or net.load_model(str(self.manifest.bin_path)) != 0:
                raise RuntimeError('NCNN load failed')
            self.net = net
            self._state.update(available=True)
        except Exception as exc:
            self._state.update(error=str(exc), available=False)
            if self.manifest is None:
                self._state.update(manifest_status='rejected', artifact_hash_status='unverified')

    def status(self):
        with self._lock:
            return dict(self._state)

    def infer(self, frame):
        # Serializes even accidental callers outside the single-owner worker.
        with self._infer_lock:
            identity = self.manifest.identity if self.manifest else None
            if self.net is None:
                return DetectionOutput(False, identity, error=self._state['error'])
            try:
                tensor, ratio = preprocess(frame)
                extractor = self.net.create_extractor()
                if extractor.input('in0', self._backend.Mat(tensor).clone()) != 0:
                    raise RuntimeError('NCNN input failed')
                code, output = extractor.extract('out0')
                if code != 0:
                    raise RuntimeError(f'NCNN output failed: {code}')
                boxes = decode(np.array(output, dtype=np.float32), ratio, frame.shape[1], frame.shape[0], self.config.diagnostic_confidence_floor)
                with self._lock:
                    self._state.update(available=True, error=None)
                return DetectionOutput(True, identity, boxes)
            except Exception as exc:
                with self._lock:
                    self._state.update(available=False, error=str(exc))
                return DetectionOutput(False, identity, error=str(exc))
