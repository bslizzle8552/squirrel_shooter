"""Low-load event classifier, durable evidence queue, and human review storage."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import re
import shutil
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
CLASSIFICATION_VIEWS = frozenset({"review", "unknown", "known", "errors", "false_positive"})
LEGACY_APPROVAL_LABELS = frozenset({"car", "person"})
TRAINING_LABEL_SUGGESTIONS = (
    "squirrel",
    "chipmunk",
    "rabbit",
    "bird",
    "raccoon",
    "opossum",
    "groundhog",
    "deer",
    "fox",
    "skunk",
    "cat",
    "dog",
    "person",
    "car",
    "other_animal",
)
SAFE_ITEM_ID = re.compile(r"[A-Za-z0-9_-]+")
SAFE_TRAINING_LABEL = re.compile(r"[a-z][a-z0-9_]{1,39}")
CLASSIFICATION_FILENAME = "classification.json"
CLASSIFIER_INPUT_FILENAME = "classifier-input.jpg"
TRAINING_DATASET_DIRECTORY = "training-dataset"
TRAINING_MANIFEST_FILENAME = "manifest.jsonl"


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
    unknown: int = 0
    errors: int = 0
    paused: bool = False
    skipped_while_paused: int = 0


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


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def normalize_training_label(value: str) -> str:
    """Create one stable, filesystem-safe class name from human-entered truth."""

    normalized = re.sub(r"_+", "_", re.sub(r"[\s-]+", "_", value.strip().lower())).strip("_")
    if not SAFE_TRAINING_LABEL.fullmatch(normalized):
        raise ValueError("Label must be 2-40 characters using letters, numbers, spaces, hyphens, or underscores")
    if normalized in {"unknown", "false_positive", "background"}:
        raise ValueError("Use the dedicated Unknown or False Positive action for this label")
    return normalized


class ClassifierEvidenceStore:
    """Persist exact classifier inputs, decisions, and append-only audit records."""

    def __init__(
        self,
        config: AppConfig,
        *,
        image_writer: Callable[[str, np.ndarray], bool] = cv2.imwrite,
    ) -> None:
        self.config = config.classifier
        self.legacy_root = self.config.evidence_directory
        self.events_root = config.camera.output_directory / "events"
        self.training_root = config.camera.output_directory / TRAINING_DATASET_DIRECTORY
        self.training_samples_root = self.training_root / "samples"
        self.training_manifest_path = self.training_root / TRAINING_MANIFEST_FILENAME
        self.audit_path = config.logging.directory / self.config.audit_log_filename
        self._image_writer = image_writer
        self._lock = threading.Lock()
        self._prepare_lock = threading.Lock()
        self._prepared = False

    def prepare(self) -> None:
        with self._prepare_lock:
            if self._prepared:
                return
            self.events_root.mkdir(parents=True, exist_ok=True)
            self.training_samples_root.mkdir(parents=True, exist_ok=True)
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            self._migrate_legacy_evidence()
            self._rebuild_training_manifest()
            self._prepared = True

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
        automatic_label = next(
            (
                item for item in detections
                if item.label in self.config.auto_accept_labels and item.confidence >= self.config.auto_accept_confidence
            ),
            None,
        )
        weak_known_label = next((item for item in detections if item.label in self.config.auto_accept_labels), None)
        if error:
            status, outcome, display_label, label_source, review_state = (
                "unclassified", "classifier_error", "Classification unavailable", None, "error"
            )
        elif automatic_label is not None:
            status, outcome, display_label, label_source, review_state = (
                "known", "auto_labeled", automatic_label.label.title(), "automatic", "complete"
            )
        elif weak_known_label is not None:
            status, outcome, display_label, label_source, review_state = (
                "review", "needs_review", "Unknown", None, "pending"
            )
        else:
            status, outcome, display_label, label_source, review_state = (
                "unknown", "unknown", "Unknown", None, "not_required"
            )
        item_id = task.event_id
        if SAFE_ITEM_ID.fullmatch(item_id) is None:
            raise ValueError("Unsafe classifier evidence id")
        task.event_directory.mkdir(parents=True, exist_ok=True)
        image_path = task.event_directory / CLASSIFIER_INPUT_FILENAME
        metadata_path = task.event_directory / CLASSIFICATION_FILENAME
        record = {
            "schema_version": 3,
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
            "model_suggestion": detections[0].label if detections else None,
            "review_suggestion_label": weak_known_label.label if weak_known_label is not None else None,
            "review_suggestion_confidence": (
                round(weak_known_label.confidence, 4) if weak_known_label is not None else None
            ),
            "decision_label": automatic_label.label if automatic_label is not None else None,
            "decision_confidence": round(automatic_label.confidence, 4) if automatic_label is not None else None,
            "classification_status": status,
            "display_label": display_label,
            "label_source": label_source,
            "review_state": review_state,
            "outcome": outcome,
            "auto_accepted": outcome == "auto_labeled",
            "approved_label": automatic_label.label if automatic_label is not None else None,
            "human_label": None,
            "human_label_action": None,
            "human_verified": False,
            "training_label": None,
            "training_dataset_status": "not_human_verified",
            "training_sample_relative": None,
            "error": error,
            "latency_ms": None if latency_ms is None else round(latency_ms, 2),
            "input_image_path": str(image_path),
            "reviewed_at": None,
        }
        with self._lock:
            if not self._image_writer(str(image_path), task.image):
                raise OSError(f"Could not save classifier input image: {image_path}")
            _atomic_json(metadata_path, record)
            self._append_audit({"action": "classified", **record})
        return record

    def list_items(self, view: str) -> list[dict[str, Any]]:
        if view not in CLASSIFICATION_VIEWS:
            raise ValueError("Unknown classification view")
        self.prepare()
        items: list[dict[str, Any]] = []
        for path in self.events_root.rglob(CLASSIFICATION_FILENAME):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict) and _classification_view(payload) == view:
                    event_relative = path.parent.relative_to(self.events_root).as_posix()
                    payload["event_snapshot_relative"] = (
                        f"{event_relative}/snapshot.jpg" if (path.parent / "snapshot.jpg").is_file() else None
                    )
                    payload["event_clip_relative"] = (
                        f"{event_relative}/clip.avi" if (path.parent / "clip.avi").is_file() else None
                    )
                    items.append(payload)
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(items, key=lambda item: str(item.get("classifier_timestamp", "")), reverse=True)

    def counts(self) -> dict[str, int]:
        return {view: len(self.list_items(view)) for view in CLASSIFICATION_VIEWS}

    def training_summary(self) -> dict[str, Any]:
        self.prepare()
        samples = self._training_samples(eligible_only=True)
        labels: dict[str, int] = {}
        for sample in samples:
            label = str(sample.get("label", ""))
            if label:
                labels[label] = labels.get(label, 0) + 1
        return {
            "eligible_samples": len(samples),
            "labels": dict(sorted(labels.items())),
            "manifest_relative": f"{TRAINING_DATASET_DIRECTORY}/{TRAINING_MANIFEST_FILENAME}",
        }

    def training_label_suggestions(self) -> tuple[str, ...]:
        self.prepare()
        observed = {
            str(sample.get("label"))
            for sample in self._training_samples(eligible_only=True)
            if sample.get("label") not in {None, "", "background"}
        }
        return tuple(dict.fromkeys((*TRAINING_LABEL_SUGGESTIONS, *sorted(observed))))

    def overview(self) -> dict[str, list[dict[str, Any]]]:
        """Return every view's items from a single evidence-directory scan.

        Polling clients (the Pi console) need items plus per-view counts; one
        rglob pass keeps that refresh cheap compared with per-view listings.
        """
        self.prepare()
        grouped: dict[str, list[dict[str, Any]]] = {view: [] for view in CLASSIFICATION_VIEWS}
        for path in self.events_root.rglob(CLASSIFICATION_FILENAME):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    continue
                view = _classification_view(payload)
                if view not in grouped:
                    continue
                event_relative = path.parent.relative_to(self.events_root).as_posix()
                payload["event_snapshot_relative"] = (
                    f"{event_relative}/snapshot.jpg" if (path.parent / "snapshot.jpg").is_file() else None
                )
                payload["event_clip_relative"] = (
                    f"{event_relative}/clip.avi" if (path.parent / "clip.avi").is_file() else None
                )
                grouped[view].append(payload)
            except (OSError, json.JSONDecodeError):
                continue
        for items in grouped.values():
            items.sort(key=lambda item: str(item.get("classifier_timestamp", "")), reverse=True)
        return grouped

    def get_record(self, item_id: str) -> dict[str, Any]:
        path = self._record_path(item_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise KeyError(item_id) from exc
        if not isinstance(payload, dict):
            raise KeyError(item_id)
        return payload

    def input_path(self, item_id: str) -> Path:
        events_root = self.events_root.resolve()
        try:
            path = (self._record_path(item_id).parent / CLASSIFIER_INPUT_FILENAME).resolve(strict=True)
        except OSError as exc:
            raise KeyError(item_id) from exc
        if not path.is_file() or not path.is_relative_to(events_root):
            raise KeyError(item_id)
        return path

    def record_action(self, action: str, record: dict[str, Any]) -> None:
        self.prepare()
        with self._lock:
            self._append_audit(
                {
                    "action": action,
                    "action_timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                    **record,
                }
            )

    def review(
        self,
        item_id: str,
        decision: str,
        approval_label: str | None = None,
    ) -> dict[str, Any]:
        if SAFE_ITEM_ID.fullmatch(item_id) is None:
            raise ValueError("Unsafe classifier evidence id")
        if decision == "approve":
            if not isinstance(approval_label, str):
                raise ValueError("A corrected label is required")
            label = normalize_training_label(approval_label)
            record = self.get_record(item_id)
            model_label = str(record.get("model_suggestion") or "")
            action = "model_confirmed" if label == model_label else "human_corrected"
        elif decision == "confirm-model":
            record = self.get_record(item_id)
            suggestion = record.get("model_suggestion") or record.get("top_label")
            if not isinstance(suggestion, str) or not suggestion:
                raise ValueError("This event has no model suggestion to confirm")
            label = normalize_training_label(suggestion)
            action = "model_confirmed"
        elif decision in {"unknown", "reject"}:
            label = "unknown"
            action = "marked_unknown"
        elif decision == "false-positive":
            label = "false_positive"
            action = "marked_false_positive"
        else:
            raise ValueError("Unknown classifier review decision")
        return self.set_label(item_id, label, label_action=action)

    def set_label(self, item_id: str, label: str, *, label_action: str = "human_corrected") -> dict[str, Any]:
        if SAFE_ITEM_ID.fullmatch(item_id) is None:
            raise ValueError("Unsafe classifier evidence id")
        normalized_label = label if label in {"unknown", "false_positive"} else normalize_training_label(label)
        metadata_path = self._record_path(item_id)
        with self._lock:
            try:
                record = json.loads(metadata_path.read_text(encoding="utf-8"))
            except FileNotFoundError as exc:
                raise KeyError(item_id) from exc
            status = normalized_label if normalized_label in {"unknown", "false_positive"} else "known"
            display_label = normalized_label.replace("_", " ").title()
            reviewed_at = datetime.now().astimezone().isoformat(timespec="milliseconds")
            training_label = "background" if normalized_label == "false_positive" else (
                None if normalized_label == "unknown" else normalized_label
            )
            record.update(
                schema_version=3,
                classification_status=status,
                display_label=display_label,
                label_source="human",
                review_state="complete",
                outcome="human_labeled",
                approved_label=normalized_label if status == "known" else None,
                decision_label=normalized_label if status == "known" else None,
                decision_confidence=None,
                human_label=normalized_label,
                human_label_action=label_action,
                human_verified=True,
                reviewed_at=reviewed_at,
            )
            if training_label is None:
                self._exclude_training_sample(item_id, "human_marked_unknown", reviewed_at)
                record.update(
                    training_label=None,
                    training_dataset_status="excluded_unknown",
                    training_sample_relative=None,
                )
            else:
                sample_relative = self._write_training_sample(
                    metadata_path,
                    record,
                    training_label,
                    label_action,
                    reviewed_at,
                )
                record.update(
                    training_label=training_label,
                    training_dataset_status="included",
                    training_sample_relative=sample_relative,
                )
            _atomic_json(metadata_path, record)
            self._update_event_truth(metadata_path.parent, record)
            self._append_audit({"action": "human_labeled", **record})
        return record

    def _write_training_sample(
        self,
        metadata_path: Path,
        record: dict[str, Any],
        training_label: str,
        label_action: str,
        labeled_at: str,
    ) -> str:
        source_image = metadata_path.parent / CLASSIFIER_INPUT_FILENAME
        if not source_image.is_file():
            raise OSError(f"Classifier input is missing: {source_image}")
        sample_directory = self.training_samples_root / str(record["item_id"])
        sample_directory.mkdir(parents=True, exist_ok=True)
        image_path = sample_directory / "image.jpg"
        shutil.copy2(source_image, image_path)
        image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
        event: dict[str, Any] = {}
        try:
            loaded_event = json.loads((metadata_path.parent / "event.json").read_text(encoding="utf-8"))
            event = loaded_event if isinstance(loaded_event, dict) else {}
        except (OSError, json.JSONDecodeError):
            pass
        image_relative = image_path.relative_to(self.training_root).as_posix()
        sample = {
            "schema_version": 1,
            "sample_id": record["item_id"],
            "event_id": record.get("event_id"),
            "task": "small_wildlife_image_classification",
            "label": training_label,
            "human_verified": True,
            "training_eligible": True,
            "label_action": label_action,
            "labeled_at": labeled_at,
            "image_relative_path": image_relative,
            "image_sha256": image_hash,
            "source": {
                "classifier_model": record.get("model"),
                "model_suggestion": record.get("model_suggestion"),
                "top_label": record.get("top_label"),
                "top_confidence": record.get("top_confidence"),
                "detections": record.get("detections", []),
                "classifier_frame_number": record.get("frame_number"),
                "source_bounding_box": record.get("source_bounding_box"),
                "crop_bounding_box": record.get("crop_bounding_box"),
                "event_start_timestamp": event.get("start_timestamp"),
                "motion_category": event.get("provisional_category"),
                "movement_attributes": event.get("movement_attributes", []),
            },
        }
        _atomic_json(sample_directory / "sample.json", sample)
        self._rebuild_training_manifest()
        return f"{TRAINING_DATASET_DIRECTORY}/{image_relative}"

    def _exclude_training_sample(self, item_id: str, reason: str, excluded_at: str) -> None:
        sample_path = self.training_samples_root / item_id / "sample.json"
        try:
            sample = json.loads(sample_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._rebuild_training_manifest()
            return
        if isinstance(sample, dict):
            sample.update(training_eligible=False, exclusion_reason=reason, excluded_at=excluded_at)
            _atomic_json(sample_path, sample)
        self._rebuild_training_manifest()

    def _training_samples(self, *, eligible_only: bool) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for path in self.training_samples_root.glob("*/sample.json"):
            try:
                sample = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(sample, dict) and (not eligible_only or sample.get("training_eligible") is True):
                    samples.append(sample)
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(samples, key=lambda item: str(item.get("sample_id", "")))

    def _rebuild_training_manifest(self) -> None:
        samples = self._training_samples(eligible_only=True)
        content = "".join(json.dumps(sample, default=str, sort_keys=True) + "\n" for sample in samples)
        _atomic_text(self.training_manifest_path, content)

    @staticmethod
    def _update_event_truth(event_directory: Path, record: dict[str, Any]) -> None:
        event_path = event_directory / "event.json"
        try:
            event = json.loads(event_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(event, dict):
            return
        event.update(
            human_review_label=record.get("human_label"),
            human_review_notes=f"Classifier review: {record.get('human_label_action')}",
            human_reviewed_at=record.get("reviewed_at"),
            training_label=record.get("training_label"),
            training_sample_relative=record.get("training_sample_relative"),
        )
        _atomic_json(event_path, event)

    def _record_path(self, item_id: str) -> Path:
        if SAFE_ITEM_ID.fullmatch(item_id) is None:
            raise ValueError("Unsafe classifier evidence id")
        self.prepare()
        for path in self.events_root.rglob(CLASSIFICATION_FILENAME):
            if path.parent.name == item_id:
                return path
        raise KeyError(item_id)

    def _migrate_legacy_evidence(self) -> None:
        if not self.legacy_root.exists():
            return
        for legacy_state in ("pending", "accepted", "rejected"):
            for metadata_path in (self.legacy_root / legacy_state).glob("*.json"):
                try:
                    legacy = json.loads(metadata_path.read_text(encoding="utf-8"))
                    item_id = str(legacy.get("event_id") or legacy.get("item_id") or "")
                    if SAFE_ITEM_ID.fullmatch(item_id) is None:
                        continue
                    event_directory = Path(str(legacy.get("source_event_directory", "")))
                    if not event_directory.is_dir():
                        matched_directory = next(
                            (path for path in self.events_root.rglob(item_id) if path.is_dir()),
                            None,
                        )
                        if matched_directory is None:
                            continue
                        event_directory = matched_directory
                    if not event_directory.is_dir() or (event_directory / CLASSIFICATION_FILENAME).exists():
                        continue
                    image_source = Path(str(legacy.get("image_path", "")))
                    if not image_source.is_file():
                        image_source = metadata_path.with_suffix(".jpg")
                    if image_source.is_file():
                        shutil.copy2(image_source, event_directory / CLASSIFIER_INPUT_FILENAME)
                    migrated = _migrate_legacy_record(legacy, event_directory)
                    _atomic_json(event_directory / CLASSIFICATION_FILENAME, migrated)
                    self._append_audit({"action": "legacy_migrated", **migrated})
                except (OSError, ValueError, json.JSONDecodeError, StopIteration):
                    LOGGER.warning("Could not migrate legacy classifier evidence", exc_info=True)

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
        self._unknown = 0
        self._errors = 0
        self._paused = False
        self._skipped_while_paused = 0
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

    def set_paused(self, paused: bool) -> None:
        """Pause CPU-heavy inference without stopping the worker lifecycle."""

        with self._lock:
            self._paused = paused

    def submit(
        self,
        event_id: str,
        event_directory: Path,
        frame_number: int,
        frame: np.ndarray,
        source_bounding_box: tuple[int, int, int, int],
    ) -> bool:
        with self._lock:
            paused = self._paused
        if not self.config.enabled or paused:
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
        return self._enqueue(task)

    def retry(self, item_id: str) -> bool:
        with self._lock:
            paused = self._paused
        if not self.config.enabled or paused:
            return False
        record = self.store.get_record(item_id)
        if record.get("classification_status") != "unclassified":
            raise ValueError("Only classification errors can be retried")
        image = cv2.imread(str(self.store.input_path(item_id)), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError("Saved classifier input could not be read")
        event_directory = self.store._record_path(item_id).parent
        task = ClassifierTask(
            item_id,
            event_directory,
            int(record.get("frame_number", 1)),
            image,
            _dict_box(record.get("source_bounding_box")),
            _dict_box(record.get("crop_bounding_box")),
            datetime.now().astimezone().isoformat(timespec="milliseconds"),
        )
        self.store.record_action("retry_requested", record)
        return self._enqueue(task)

    def _enqueue(self, task: ClassifierTask) -> bool:
        try:
            self._tasks.put_nowait(task)
        except queue.Full:
            record = self.store.save_classification(
                task, [], None, MobileNetSSDDetector.model_name, error="classifier_queue_full"
            )
            with self._lock:
                self._errors += 1
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
                self._unknown,
                self._errors,
                self._paused,
                self._skipped_while_paused,
            )

    def _run(self) -> None:
        detector: MobileNetSSDDetector | None = None
        detector_error: str | None = None
        while not self._stop_event.is_set() or not self._tasks.empty():
            try:
                task = self._tasks.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                while True:
                    with self._lock:
                        paused = self._paused
                    if not paused or self._stop_event.wait(0.2):
                        break
                if paused and self._stop_event.is_set():
                    with self._lock:
                        self._skipped_while_paused += 1
                    continue
                if detector is None:
                    try:
                        detector = self._detector_factory()
                        detector_error = None
                    except Exception as exc:
                        detector_error = f"{type(exc).__name__}: {exc}"
                        LOGGER.error(
                            "Classifier model could not be loaded",
                            extra={"structured_data": {"event": "classifier_load_error", "error": detector_error}},
                        )
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
                    elif record["classification_status"] == "review":
                        self._queued_for_review += 1
                    elif record["classification_status"] == "unknown":
                        self._unknown += 1
                    elif record["classification_status"] == "unclassified":
                        self._errors += 1
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


def _dict_box(value: Any) -> tuple[int, int, int, int]:
    if not isinstance(value, dict):
        return (0, 0, 0, 0)
    return tuple(int(value.get(name, 0)) for name in ("x", "y", "width", "height"))  # type: ignore[return-value]


def _classification_view(record: dict[str, Any]) -> str:
    status = str(record.get("classification_status", "unclassified"))
    return "errors" if status == "unclassified" else status


def _migrate_legacy_record(legacy: dict[str, Any], event_directory: Path) -> dict[str, Any]:
    detections = legacy.get("detections") if isinstance(legacy.get("detections"), list) else []
    known_suggestion = next(
        (
            str(item.get("label"))
            for item in detections
            if isinstance(item, dict) and item.get("label") in LEGACY_APPROVAL_LABELS
        ),
        None,
    )
    approved = str(legacy.get("approved_label") or "").lower()
    outcome = str(legacy.get("outcome") or "")
    error = legacy.get("error")
    if approved in LEGACY_APPROVAL_LABELS and outcome == "manual_approved":
        status, display_label, label_source, review_state = "known", approved.title(), "human", "complete"
        final_label = approved
    elif error or outcome == "classifier_error":
        status, display_label, label_source, review_state = (
            "unclassified", "Classification unavailable", None, "error"
        )
        final_label = None
    elif approved in LEGACY_APPROVAL_LABELS:
        status, display_label, label_source, review_state = "known", approved.title(), "automatic", "complete"
        final_label = approved
    elif outcome == "edge_case" and known_suggestion:
        status, display_label, label_source, review_state = "review", "Unknown", None, "pending"
        final_label = None
    else:
        status, display_label, label_source, review_state = "unknown", "Unknown", None, "not_required"
        final_label = None
    return {
        **legacy,
        "schema_version": 3,
        "source_event_directory": str(event_directory),
        "classification_status": status,
        "display_label": display_label,
        "label_source": label_source,
        "review_state": review_state,
        "model_suggestion": legacy.get("top_label"),
        "decision_label": final_label,
        "approved_label": final_label,
        "human_label": final_label if label_source == "human" else None,
        "human_label_action": "legacy_manual" if label_source == "human" else None,
        "human_verified": label_source == "human",
        "training_label": None,
        "training_dataset_status": "legacy_requires_reconfirmation",
        "training_sample_relative": None,
        "input_image_path": str(event_directory / CLASSIFIER_INPUT_FILENAME),
        "legacy_migrated": True,
    }
