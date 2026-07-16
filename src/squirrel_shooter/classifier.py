"""Low-load event classifier, durable evidence queue, and human review storage."""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import cv2
import numpy as np

from .config import AppConfig, ClassifierConfig


LOGGER = logging.getLogger(__name__)
VOC_LABELS = (
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)
EVIDENCE_STATES = frozenset({"pending", "accepted", "rejected"})
SAFE_ITEM_ID = re.compile(r"[A-Za-z0-9_-]+")


@dataclass(frozen=True)
class ClassifierDetection:
    label: str
    confidence: float
    bounding_box: tuple[int, int, int, int]

    def as_dict(self) -> dict[str, Any]:
        x, y, width, height = self.bounding_box
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "bounding_box": {"x": x, "y": y, "width": width, "height": height},
        }


@dataclass(frozen=True)
class ClassifierTask:
    event_id: str
    event_directory: Path
    frame_number: int
    image: np.ndarray
    source_bounding_box: tuple[int, int, int, int]
    crop_bounding_box: tuple[int, int, int, int]
    submitted_at: str


@dataclass(frozen=True)
class ClassifierStatus:
    enabled: bool
    thread_alive: bool
    submitted: int
    completed: int
    auto_accepted: int
    queued_for_review: int
    queue_depth: int
    last_latency_ms: float | None
    last_error: str | None


class MobileNetSSDDetector:
    """OpenCV DNN wrapper for the pinned MIT-licensed MobileNet-SSD model."""

    model_name = "MobileNet-SSD VOC0712 bb17b6c"

    def __init__(self, config: ClassifierConfig, *, net: Any | None = None) -> None:
        if net is None:
            missing = [path for path in (config.model_definition, config.model_weights) if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "Classifier model is not installed: " + ", ".join(str(path) for path in missing)
                    + ". Run python -m squirrel_shooter.classifier_setup."
                )
            net = cv2.dnn.readNetFromCaffe(str(config.model_definition), str(config.model_weights))
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.net = net
        self.minimum_confidence = config.detection_confidence

    def classify(self, image: np.ndarray) -> tuple[list[ClassifierDetection], float]:
        started = perf_counter()
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=0.007843,
            size=(300, 300),
            mean=(127.5, 127.5, 127.5),
            swapRB=False,
            crop=False,
        )
        self.net.setInput(blob)
        raw = np.asarray(self.net.forward()).reshape(-1, 7)
        height, width = image.shape[:2]
        detections: list[ClassifierDetection] = []
        for row in raw:
            class_id = int(row[1])
            confidence = float(row[2])
            if confidence < self.minimum_confidence or not 0 < class_id < len(VOC_LABELS):
                continue
            left = max(0, min(width - 1, round(float(row[3]) * width)))
            top = max(0, min(height - 1, round(float(row[4]) * height)))
            right = max(left + 1, min(width, round(float(row[5]) * width)))
            bottom = max(top + 1, min(height, round(float(row[6]) * height)))
            detections.append(ClassifierDetection(VOC_LABELS[class_id], confidence, (left, top, right - left, bottom - top)))
        detections.sort(key=lambda item: item.confidence, reverse=True)
        return detections, (perf_counter() - started) * 1000.0


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class ClassifierEvidenceStore:
    """Persist exact classifier inputs, decisions, and append-only audit records."""

    def __init__(
        self,
        config: AppConfig,
        *,
        image_writer: Callable[[str, np.ndarray], bool] = cv2.imwrite,
    ) -> None:
        self.config = config.classifier
        self.root = self.config.evidence_directory
        self.audit_path = config.logging.directory / self.config.audit_log_filename
        self._image_writer = image_writer
        self._lock = threading.Lock()

    def prepare(self) -> None:
        for state in EVIDENCE_STATES:
            (self.root / state).mkdir(parents=True, exist_ok=True)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)

    def save_classification(
        self,
        task: ClassifierTask,
        detections: list[ClassifierDetection],
        latency_ms: float | None,
        model_name: str,
        *,
        error: str | None = None,
    ) -> dict[str, Any]:
        self.prepare()
        auto = next(
            (
                item for item in detections
                if item.label in self.config.auto_accept_labels and item.confidence >= self.config.auto_accept_confidence
            ),
            None,
        )
        if error:
            state, outcome = "pending", "classifier_error"
        elif auto is not None:
            state, outcome = "accepted", "auto_accepted"
        elif detections:
            state, outcome = "pending", "edge_case"
        else:
            state, outcome = "pending", "negative"
        item_id = task.event_id
        if SAFE_ITEM_ID.fullmatch(item_id) is None:
            raise ValueError("Unsafe classifier evidence id")
        directory = self.root / state
        image_path = directory / f"{item_id}.jpg"
        metadata_path = directory / f"{item_id}.json"
        record = {
            "item_id": item_id,
            "event_id": task.event_id,
            "source_event_directory": str(task.event_directory),
            "classifier_timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "submitted_at": task.submitted_at,
            "frame_number": task.frame_number,
            "source_bounding_box": _box_dict(task.source_bounding_box),
            "crop_bounding_box": _box_dict(task.crop_bounding_box),
            "model": model_name,
            "detections": [item.as_dict() for item in detections],
            "top_label": detections[0].label if detections else None,
            "top_confidence": round(detections[0].confidence, 4) if detections else None,
            "decision_label": auto.label if auto is not None else (detections[0].label if detections else None),
            "decision_confidence": round(auto.confidence, 4) if auto is not None else (
                round(detections[0].confidence, 4) if detections else None
            ),
            "state": state,
            "outcome": outcome,
            "auto_accepted": outcome == "auto_accepted",
            "error": error,
            "latency_ms": None if latency_ms is None else round(latency_ms, 2),
            "image_path": str(image_path),
            "reviewed_at": None,
        }
        with self._lock:
            if not self._image_writer(str(image_path), task.image):
                raise OSError(f"Could not save classifier input image: {image_path}")
            _atomic_json(metadata_path, record)
            _atomic_json(task.event_directory / "classifier.json", record)
            self._append_audit({"action": "classified", **record})
        return record

    def list_items(self, state: str) -> list[dict[str, Any]]:
        if state not in EVIDENCE_STATES:
            raise ValueError("Unknown classifier evidence state")
        directory = self.root / state
        if not directory.exists():
            return []
        items: list[dict[str, Any]] = []
        for path in directory.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    items.append(payload)
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(items, key=lambda item: str(item.get("classifier_timestamp", "")), reverse=True)

    def counts(self) -> dict[str, int]:
        return {state: len(self.list_items(state)) for state in EVIDENCE_STATES}

    def review(self, item_id: str, decision: str) -> dict[str, Any]:
        if SAFE_ITEM_ID.fullmatch(item_id) is None:
            raise ValueError("Unsafe classifier evidence id")
        if decision not in {"approve", "reject"}:
            raise ValueError("Review decision must be approve or reject")
        source = self.root / "pending"
        metadata_path = source / f"{item_id}.json"
        image_path = source / f"{item_id}.jpg"
        with self._lock:
            try:
                record = json.loads(metadata_path.read_text(encoding="utf-8"))
            except FileNotFoundError as exc:
                raise KeyError(item_id) from exc
            target_state = "accepted" if decision == "approve" else "rejected"
            target = self.root / target_state
            target.mkdir(parents=True, exist_ok=True)
            target_image = target / image_path.name
            os.replace(image_path, target_image)
            record.update(
                state=target_state,
                outcome="manual_approved" if decision == "approve" else "manual_rejected",
                image_path=str(target_image),
                reviewed_at=datetime.now().astimezone().isoformat(timespec="milliseconds"),
            )
            target_metadata = target / metadata_path.name
            _atomic_json(target_metadata, record)
            metadata_path.unlink(missing_ok=True)
            event_directory = Path(str(record.get("source_event_directory", "")))
            if event_directory.is_dir():
                _atomic_json(event_directory / "classifier.json", record)
            self._append_audit({"action": f"manual_{decision}", **record})
        return record

    def _append_audit(self, record: dict[str, Any]) -> None:
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


class EventClassifier:
    """Single-worker, bounded-queue classifier that never blocks frame processing."""

    def __init__(
        self,
        config: ClassifierConfig,
        store: ClassifierEvidenceStore,
        *,
        detector_factory: Callable[[], MobileNetSSDDetector] | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self._detector_factory = detector_factory or (lambda: MobileNetSSDDetector(config))
        self._tasks: queue.Queue[ClassifierTask] = queue.Queue(maxsize=config.worker_queue_capacity)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._submitted = 0
        self._completed = 0
        self._auto_accepted = 0
        self._queued_for_review = 0
        self._last_latency_ms: float | None = None
        self._last_error: str | None = None

    def start(self) -> None:
        if not self.config.enabled or (self._thread is not None and self._thread.is_alive()):
            return
        self.store.prepare()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="squirrel-classifier", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)

    def submit(
        self,
        event_id: str,
        event_directory: Path,
        frame_number: int,
        frame: np.ndarray,
        source_bounding_box: tuple[int, int, int, int],
    ) -> bool:
        if not self.config.enabled:
            return False
        crop, crop_box = _candidate_crop(frame, source_bounding_box, self.config.crop_margin_percent)
        task = ClassifierTask(
            event_id,
            event_directory,
            frame_number,
            crop,
            source_bounding_box,
            crop_box,
            datetime.now().astimezone().isoformat(timespec="milliseconds"),
        )
        try:
            self._tasks.put_nowait(task)
        except queue.Full:
            record = self.store.save_classification(
                task, [], None, MobileNetSSDDetector.model_name, error="classifier_queue_full"
            )
            with self._lock:
                self._queued_for_review += 1
                self._last_error = str(record["error"])
            return False
        with self._lock:
            self._submitted += 1
        return True

    def status(self) -> ClassifierStatus:
        with self._lock:
            return ClassifierStatus(
                self.config.enabled,
                self._thread is not None and self._thread.is_alive(),
                self._submitted,
                self._completed,
                self._auto_accepted,
                self._queued_for_review,
                self._tasks.qsize(),
                self._last_latency_ms,
                self._last_error,
            )

    def _run(self) -> None:
        detector: MobileNetSSDDetector | None = None
        detector_error: str | None = None
        try:
            detector = self._detector_factory()
        except Exception as exc:
            detector_error = f"{type(exc).__name__}: {exc}"
            LOGGER.error("Classifier model could not be loaded", extra={"structured_data": {"event": "classifier_load_error", "error": detector_error}})
        while not self._stop_event.is_set() or not self._tasks.empty():
            try:
                task = self._tasks.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if detector is None:
                    detections, latency = [], None
                    error = detector_error or "classifier_unavailable"
                    model_name = MobileNetSSDDetector.model_name
                else:
                    detections, latency = detector.classify(task.image)
                    error = None
                    model_name = detector.model_name
                record = self.store.save_classification(task, detections, latency, model_name, error=error)
                with self._lock:
                    self._completed += 1
                    self._last_latency_ms = latency
                    self._last_error = error
                    if record["auto_accepted"]:
                        self._auto_accepted += 1
                    else:
                        self._queued_for_review += 1
            except Exception as exc:
                with self._lock:
                    self._last_error = f"{type(exc).__name__}: {exc}"
                LOGGER.error("Classifier task failed", extra={"structured_data": {"event": "classifier_task_error", "error": str(exc)}}, exc_info=True)
            finally:
                self._tasks.task_done()


def _candidate_crop(
    frame: np.ndarray,
    bounding_box: tuple[int, int, int, int],
    margin_percent: float,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    frame_height, frame_width = frame.shape[:2]
    x, y, width, height = bounding_box
    x_margin = round(width * margin_percent / 100.0)
    y_margin = round(height * margin_percent / 100.0)
    left = max(0, x - x_margin)
    top = max(0, y - y_margin)
    right = min(frame_width, x + width + x_margin)
    bottom = min(frame_height, y + height + y_margin)
    if right <= left or bottom <= top:
        return frame.copy(), (0, 0, frame_width, frame_height)
    return frame[top:bottom, left:right].copy(), (left, top, right - left, bottom - top)


def _box_dict(box: tuple[int, int, int, int]) -> dict[str, int]:
    x, y, width, height = box
    return {"x": x, "y": y, "width": width, "height": height}
