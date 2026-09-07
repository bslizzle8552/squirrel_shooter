"""Low-load event classifier, durable evidence queue, and human review storage."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import queue
import re
import shutil
import threading
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import monotonic, perf_counter
from typing import Any, Callable

import cv2
import numpy as np

from .classifier_labels import VOC_LABELS
from .config import AppConfig, ClassifierConfig
from .performance import TimingDistribution
from .legacy_mobilenet_policy import ScenePersonSafetyResult
from .safety import SceneDetection, SceneFramePacket
from .thread_names import set_current_thread_name
from .media_roles import verified_classifier_source


LOGGER = logging.getLogger(__name__)
CLASSIFICATION_VIEWS = frozenset({"review", "unknown", "known", "errors", "false_positive"})
OVERVIEW_CACHE_SECONDS = 10.0
LEGACY_APPROVAL_LABELS = frozenset({"car", "person"})
TRAINING_LABEL_SUGGESTIONS = (
    "squirrel",
    "person",
    "car",
    "rabbit",
    "deer",
    "other_animal",
    "chipmunk",
    "bird",
    "raccoon",
    "opossum",
    "groundhog",
    "fox",
    "skunk",
    "cat",
    "dog",
)
SAFE_ITEM_ID = re.compile(r"[A-Za-z0-9_-]+")
SAFE_TRAINING_LABEL = re.compile(r"[a-z][a-z0-9_]{1,39}")
CLASSIFICATION_SCHEMA_VERSION = 6
DECISION_DISPATCH_START_WAIT_SECONDS = 0.50
TRAINING_SAMPLE_SCHEMA_VERSION = 2
CLASSIFICATION_FILENAME = "classification.json"
CLASSIFIER_INPUT_FILENAME = "classifier-input.jpg"
ORIGINAL_FRAME_FILENAME = "original-frame.jpg"
TRAINING_DATASET_DIRECTORY = "training-dataset"
TRAINING_MANIFEST_FILENAME = "manifest.jsonl"
CANONICAL_NEGATIVE_LABEL = "background_or_false_positive"
RESERVED_TRAINING_LABELS = frozenset(
    {"unknown", "false_positive", "background", CANONICAL_NEGATIVE_LABEL}
)


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
    selection_method: str = "configured_fallback"
    selected_motion_bounding_box_area: int | None = None
    total_event_frames_considered: int = 1
    original_image: np.ndarray | None = None
    context: str = "completed_event"
    track_id: int | None = None
    frame_sequence: int | None = None
    target_pixel: tuple[int, int] | None = None
    target_observed_monotonic: float | None = None
    target_provisional_category: str | None = None
    target_confirmed: bool = False
    target_event_eligible: bool = False
    enqueued_monotonic: float | None = None
    source_media_role: str = "clean_authoritative"  # submit's raw-frame input contract


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
    inference_fps: float = 0.0
    decision_queue_depth: int = 0
    evidence_queue_depth: int = 0
    evidence_completion_queue_depth: int = 0
    evidence_completion_backpressured: bool = False
    evidence_completion_terminal_error: bool = False
    scene_request_pending: bool = False
    decision_dropped: int = 0
    evidence_dropped: int = 0
    evidence_completion_dropped: int = 0
    decision_thread_alive: bool = False
    evidence_thread_alive: bool = False
    evidence_completion_thread_alive: bool = False
    inference_active: bool = False
    decision_active: bool = False
    evidence_active: bool = False
    evidence_completion_active: bool = False
    inference_last_progress_age_seconds: float | None = None
    decision_last_progress_age_seconds: float | None = None
    evidence_last_progress_age_seconds: float | None = None
    evidence_completion_last_progress_age_seconds: float | None = None
    timing_distributions: dict[str, dict[str, object]] = field(default_factory=dict)
    inference_completed: int = 0
    evidence_persisted: int = 0


@dataclass(frozen=True)
class _ClassifierOutcome:
    task: ClassifierTask
    detections: tuple[ClassifierDetection, ...]
    error: str | None
    latency_ms: float | None
    model_name: str


@dataclass
class _DecisionDeliveryState:
    started: threading.Event = field(default_factory=threading.Event)
    finished: threading.Event = field(default_factory=threading.Event)
    status: str = "queued"
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, status: str, *, started: bool = False, finished: bool = False) -> None:
        with self._lock:
            self.status = status
            if started:
                self.started.set()
            if finished:
                self.finished.set()

    def snapshot(self) -> str:
        with self._lock:
            return self.status


@dataclass(frozen=True)
class _DecisionDispatch:
    outcome: _ClassifierOutcome
    delivery: _DecisionDeliveryState
    enqueued_monotonic: float


@dataclass(frozen=True)
class _EvidenceJob:
    outcome: _ClassifierOutcome
    delivery: _DecisionDeliveryState


@dataclass(frozen=True)
class _EvidenceCompletion:
    task: ClassifierTask
    status: str
    record: dict[str, Any] | None
    error: str | None
    attempts: int = 0


@dataclass
class _SceneInferenceTask:
    request_id: str
    event_id: str
    track_id: int
    packet: SceneFramePacket
    deadline_monotonic: float
    enqueued_monotonic: float
    completed: threading.Event
    expired: threading.Event
    result: ScenePersonSafetyResult | None = None


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
    if normalized in RESERVED_TRAINING_LABELS:
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
        self._overview_cache_lock = threading.Lock()
        self._overview_cache: dict[str, list[dict[str, Any]]] | None = None
        self._overview_cached_at = 0.0

    def prepare(self) -> None:
        with self._prepare_lock:
            if self._prepared:
                return
            self.events_root.mkdir(parents=True, exist_ok=True)
            self.training_samples_root.mkdir(parents=True, exist_ok=True)
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            self._migrate_legacy_evidence()
            self._canonicalize_negative_training_labels()
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
        decision_delivery_status: str | None = None,
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
        original_frame_path = task.event_directory / ORIGINAL_FRAME_FILENAME
        metadata_path = task.event_directory / CLASSIFICATION_FILENAME
        record = {
            "schema_version": CLASSIFICATION_SCHEMA_VERSION,
            "item_id": item_id,
            "event_id": task.event_id,
            "source_event_directory": str(task.event_directory),
            "event_timestamp": None,
            "session_id": None,
            "capture_method": "automatic_motion_event",
            "software_version": None,
            "git_commit_sha": None,
            "source_camera": {},
            "classifier_timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "submitted_at": task.submitted_at,
            "frame_number": task.frame_number,
            "selected_event_frame_index": task.frame_number,
            "frame_selection_method": task.selection_method,
            "selected_motion_bounding_box_area": task.selected_motion_bounding_box_area,
            "total_event_frames_considered": task.total_event_frames_considered,
            "classification_context": task.context,
            "decision_delivery_status": decision_delivery_status,
            "track_id": task.track_id,
            "source_frame_sequence": task.frame_sequence,
            "target_pixel": (
                None
                if task.target_pixel is None
                else {"x": task.target_pixel[0], "y": task.target_pixel[1]}
            ),
            "target_observed_monotonic": task.target_observed_monotonic,
            "target_provisional_category": task.target_provisional_category,
            "target_confirmed": task.target_confirmed,
            "target_event_eligible": task.target_event_eligible,
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
            "input_image_role": "unannotated_target_crop",
            "source_media_role": task.source_media_role if task.selection_method != "middle_fallback" else "unknown_legacy",
            "source_pixel_provenance": (
                "shared_camera_raw_v1" if task.source_media_role == "clean_authoritative" and task.selection_method != "middle_fallback" else None
            ),
            "original_frame_path": None,
            "original_frame_role": "unannotated_full_resolution_context",
            "original_frame_error": None,
            "reviewed_at": None,
        }
        if record["source_pixel_provenance"] is None:
            record.update(input_image_role="unknown_legacy", original_frame_role="unknown_legacy")
        with self._lock:
            # Read finalized event metadata under the same lock used by the
            # completion reconciler. Whichever side wins the race therefore
            # observes and enriches the other side's durable record.
            event = self._load_event_record(task.event_directory)
            record.update(
                event_timestamp=event.get("start_timestamp"),
                session_id=event.get("session_id"),
                capture_method=event.get("capture_method", "automatic_motion_event"),
                software_version=event.get("software_version"),
                git_commit_sha=event.get("git_commit_sha"),
                source_camera={
                    key: event.get(key)
                    for key in (
                        "source_camera",
                        "camera_device_index",
                        "actual_width",
                        "actual_height",
                        "camera_reported_fps",
                        "measured_camera_fps",
                    )
                    if event.get(key) is not None
                },
            )
            if not self._image_writer(str(image_path), task.image):
                raise OSError(f"Could not save classifier input image: {image_path}")
            if task.original_image is not None:
                if self._image_writer(str(original_frame_path), task.original_image):
                    record["original_frame_path"] = str(original_frame_path)
                else:
                    record["original_frame_error"] = f"Could not save original event frame: {original_frame_path}"
            elif original_frame_path.is_file():
                record["original_frame_path"] = str(original_frame_path)
            _atomic_json(metadata_path, record)
            self._invalidate_overview_cache()
            self._update_event_classification_metadata(task.event_directory, record)
            self._append_audit({"action": "classified", **record})
        LOGGER.info(
            "Event classified: event_id=%s frame=%d method=%s motion_box_area=%s frames_considered=%d result=%s confidence=%s",
            task.event_id,
            task.frame_number,
            task.selection_method,
            task.selected_motion_bounding_box_area,
            task.total_event_frames_considered,
            record["top_label"] or record["outcome"],
            record["top_confidence"],
            extra={
                "structured_data": {
                    "event": "event_classified",
                    "event_id": task.event_id,
                    "selected_event_frame_index": task.frame_number,
                    "frame_selection_method": task.selection_method,
                    "selected_motion_bounding_box_area": task.selected_motion_bounding_box_area,
                    "total_event_frames_considered": task.total_event_frames_considered,
                    "classifier_result": record["top_label"] or record["outcome"],
                    "classifier_confidence": record["top_confidence"],
                    "classification_context": task.context,
                    "track_id": task.track_id,
                    "classifier_submitted_at": record["submitted_at"],
                    "classifier_completed_at": record["classifier_timestamp"],
                    "classifier_latency_ms": record["latency_ms"],
                }
            },
        )
        return record

    @staticmethod
    def _load_event_record(event_directory: Path) -> dict[str, Any]:
        try:
            event = json.loads((event_directory / "event.json").read_text(encoding="utf-8"))
            return event if isinstance(event, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _update_event_classification_metadata(event_directory: Path, record: dict[str, Any]) -> None:
        event_path = event_directory / "event.json"
        event = ClassifierEvidenceStore._load_event_record(event_directory)
        if not event:
            return
        event.update(
            classification_path=str(event_directory / CLASSIFICATION_FILENAME),
            classifier_input_path=record.get("input_image_path"),
            classifier_input_role=record.get("input_image_role"),
            original_frame_path=record.get("original_frame_path"),
            original_frame_role=record.get("original_frame_role"),
            predicted_class=record.get("top_label"),
            prediction_confidence=record.get("top_confidence"),
            classification_status=record.get("classification_status"),
            classification_error=record.get("error"),
        )
        _atomic_json(event_path, event)

    def reconcile_completed_event(self, event_directory: Path) -> dict[str, Any] | None:
        """Join an early live classification to metadata finalized with its event."""

        metadata_path = event_directory / CLASSIFICATION_FILENAME
        with self._lock:
            try:
                record = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
            if not isinstance(record, dict):
                return None
            event = self._load_event_record(event_directory)
            if not event:
                return None
            source_camera = {
                key: event.get(key)
                for key in (
                    "source_camera",
                    "camera_device_index",
                    "actual_width",
                    "actual_height",
                    "camera_reported_fps",
                    "measured_camera_fps",
                )
                if event.get(key) is not None
            }
            record.update(
                event_timestamp=event.get("start_timestamp"),
                session_id=event.get("session_id"),
                capture_method=event.get("capture_method", "automatic_motion_event"),
                software_version=event.get("software_version"),
                git_commit_sha=event.get("git_commit_sha"),
                source_camera=source_camera,
            )
            _atomic_json(metadata_path, record)
            self._update_event_classification_metadata(event_directory, record)
            self._invalidate_overview_cache()
            return record

    def list_items(self, view: str) -> list[dict[str, Any]]:
        if view not in CLASSIFICATION_VIEWS:
            raise ValueError("Unknown classification view")
        return deepcopy(self._cached_overview()[view])

    def counts(self) -> dict[str, int]:
        overview = self._cached_overview()
        return {view: len(overview[view]) for view in CLASSIFICATION_VIEWS}

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
            if sample.get("label") not in {None, "", *RESERVED_TRAINING_LABELS}
        }
        return tuple(
            dict.fromkeys(
                (
                    *TRAINING_LABEL_SUGGESTIONS,
                    *VOC_LABELS[1:],
                    *sorted(observed),
                )
            )
        )

    def overview(self) -> dict[str, list[dict[str, Any]]]:
        """Return every view's items from one short-lived cached directory scan.

        Classification and review writes invalidate the cache immediately. The
        ten-second fallback lifetime bounds staleness after an external event
        retention or maintenance change without rereading the microSD on every
        dashboard poll.
        """
        return deepcopy(self._cached_overview())

    def _cached_overview(self) -> dict[str, list[dict[str, Any]]]:
        self.prepare()
        now = perf_counter()
        with self._overview_cache_lock:
            if self._overview_cache is None or now - self._overview_cached_at >= OVERVIEW_CACHE_SECONDS:
                self._overview_cache = self._scan_overview()
                self._overview_cached_at = perf_counter()
            return self._overview_cache

    def _scan_overview(self) -> dict[str, list[dict[str, Any]]]:
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
                payload["event_original_frame_relative"] = (
                    f"{event_relative}/{ORIGINAL_FRAME_FILENAME}"
                    if (path.parent / ORIGINAL_FRAME_FILENAME).is_file()
                    else None
                )
                grouped[view].append(payload)
            except (OSError, json.JSONDecodeError):
                continue
        for items in grouped.values():
            items.sort(key=lambda item: str(item.get("classifier_timestamp", "")), reverse=True)
        return grouped

    def _invalidate_overview_cache(self) -> None:
        with self._overview_cache_lock:
            self._overview_cache = None
            self._overview_cached_at = 0.0

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
            training_label = CANONICAL_NEGATIVE_LABEL if normalized_label == "false_positive" else (
                None if normalized_label == "unknown" else normalized_label
            )
            record.update(
                schema_version=CLASSIFICATION_SCHEMA_VERSION,
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
            elif not verified_classifier_source(record):
                self._exclude_training_sample(item_id, "unverified_media_source", reviewed_at)
                record.update(training_label=training_label, training_dataset_status="excluded_unverified_media",
                              training_sample_relative=None)
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
            self._invalidate_overview_cache()
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
        if not verified_classifier_source(record):
            raise ValueError("Unverified or annotated pixels cannot become training source media")
        source_image = metadata_path.parent / CLASSIFIER_INPUT_FILENAME
        if not source_image.is_file():
            raise OSError(f"Classifier input is missing: {source_image}")
        sample_directory = self.training_samples_root / str(record["item_id"])
        sample_directory.mkdir(parents=True, exist_ok=True)
        image_path = sample_directory / "image.jpg"
        shutil.copy2(source_image, image_path)
        image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
        source_original_frame = metadata_path.parent / ORIGINAL_FRAME_FILENAME
        original_frame_path = sample_directory / ORIGINAL_FRAME_FILENAME
        original_frame_relative: str | None = None
        original_frame_hash: str | None = None
        if source_original_frame.is_file():
            shutil.copy2(source_original_frame, original_frame_path)
            original_frame_relative = original_frame_path.relative_to(self.training_root).as_posix()
            original_frame_hash = hashlib.sha256(original_frame_path.read_bytes()).hexdigest()
        event: dict[str, Any] = {}
        try:
            loaded_event = json.loads((metadata_path.parent / "event.json").read_text(encoding="utf-8"))
            event = loaded_event if isinstance(loaded_event, dict) else {}
        except (OSError, json.JSONDecodeError):
            pass
        image_relative = image_path.relative_to(self.training_root).as_posix()
        sample = {
            "schema_version": TRAINING_SAMPLE_SCHEMA_VERSION,
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
            "image_role": "unannotated_target_crop",
            "source_media_role": "clean_authoritative",
            "source_pixel_provenance": "shared_camera_raw_v1",
            "image_media_role": "derived_crop",
            "original_frame_relative_path": original_frame_relative,
            "original_frame_sha256": original_frame_hash,
            "original_frame_role": "unannotated_full_resolution_context",
            "split_group_event_id": record.get("event_id"),
            "split_group_session_id": event.get("session_id"),
            "source": {
                "capture_method": event.get("capture_method", record.get("capture_method")),
                "source_camera": record.get("source_camera", {}),
                "software_version": event.get("software_version", record.get("software_version")),
                "git_commit_sha": event.get("git_commit_sha", record.get("git_commit_sha")),
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
                "event_snapshot_role": event.get("snapshot_file_role"),
                "event_clip_role": event.get("clip_file_role"),
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

    def _canonicalize_negative_training_labels(self) -> None:
        """Upgrade the two legacy negative labels without losing audit provenance."""

        classification_paths = {
            path.parent.name: path
            for path in self.events_root.rglob(CLASSIFICATION_FILENAME)
        }
        for sample_path in self.training_samples_root.glob("*/sample.json"):
            try:
                sample = json.loads(sample_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(sample, dict) or sample.get("label") not in {"background", "false_positive"}:
                continue
            previous_label = str(sample["label"])
            sample.update(
                schema_version=max(
                    TRAINING_SAMPLE_SCHEMA_VERSION,
                    int(sample.get("schema_version", 0) or 0),
                ),
                label=CANONICAL_NEGATIVE_LABEL,
                label_migrated_from=previous_label,
            )
            _atomic_json(sample_path, sample)

            item_id = str(sample.get("sample_id") or sample_path.parent.name)
            metadata_path = classification_paths.get(item_id)
            if metadata_path is None:
                continue
            try:
                record = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(record, dict) and record.get("training_label") in {"background", "false_positive"}:
                record.update(
                    schema_version=max(
                        CLASSIFICATION_SCHEMA_VERSION,
                        int(record.get("schema_version", 0) or 0),
                    ),
                    training_label=CANONICAL_NEGATIVE_LABEL,
                    training_label_migrated_from=previous_label,
                )
                _atomic_json(metadata_path, record)
                self._update_event_truth(metadata_path.parent, record)

    def _training_samples(self, *, eligible_only: bool) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for path in self.training_samples_root.glob("*/sample.json"):
            try:
                sample = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(sample, dict) and (not eligible_only or (
                    sample.get("training_eligible") is True and verified_classifier_source(sample)
                )):
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
    """One DNN owner with current-scene priority and decoupled downstream work."""

    def __init__(
        self,
        config: ClassifierConfig,
        store: ClassifierEvidenceStore,
        *,
        detector_factory: Callable[[], MobileNetSSDDetector] | None = None,
        result_handler: Callable[
            [ClassifierTask, list[ClassifierDetection], str | None, dict[str, Any]],
            None,
        ]
        | None = None,
        evidence_result_handler: Callable[
            [ClassifierTask, str, dict[str, Any] | None, str | None],
            None,
        ]
        | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self._detector_factory = detector_factory or (lambda: MobileNetSSDDetector(config))
        self._result_handler = result_handler
        self._evidence_result_handler = evidence_result_handler
        self._scheduler_condition = threading.Condition()
        self._live_tasks: deque[ClassifierTask] = deque()
        self._background_tasks: deque[ClassifierTask] = deque()
        self._pending_scene: _SceneInferenceTask | None = None
        self._scene_active = False
        self._decision_queue: queue.Queue[_DecisionDispatch] = queue.Queue(
            maxsize=config.decision_queue_capacity
        )
        self._evidence_queue: queue.Queue[_EvidenceJob] = queue.Queue(
            maxsize=config.evidence_queue_capacity
        )
        self._completion_reserve = max(
            2,
            config.evidence_queue_capacity + config.worker_queue_capacity + 1,
        )
        self._evidence_completion_queue: queue.Queue[_EvidenceCompletion] = queue.Queue(
            maxsize=2 * self._completion_reserve
        )
        self._evidence_completion_overflow: deque[_EvidenceCompletion] = deque(
            maxlen=self._completion_reserve
        )
        self._completion_backpressure = threading.Event()
        self._completion_terminal_error = threading.Event()
        self._stop_event = threading.Event()
        self._downstream_stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._decision_thread: threading.Thread | None = None
        self._evidence_thread: threading.Thread | None = None
        self._evidence_completion_thread: threading.Thread | None = None
        self._decision_done = threading.Event()
        self._evidence_done = threading.Event()
        self._lock = threading.Lock()
        self._submitted = 0
        self._completed = 0
        self._inference_completed = 0
        self._auto_accepted = 0
        self._queued_for_review = 0
        self._unknown = 0
        self._errors = 0
        self._paused = False
        self._skipped_while_paused = 0
        self._decision_dropped = 0
        self._evidence_dropped = 0
        self._evidence_completion_dropped = 0
        self._inference_active = False
        self._decision_active = False
        self._evidence_active = False
        self._evidence_completion_active = False
        self._last_inference_progress_monotonic: float | None = None
        self._last_decision_progress_monotonic: float | None = None
        self._last_evidence_progress_monotonic: float | None = None
        self._last_evidence_completion_progress_monotonic: float | None = None
        self._last_latency_ms: float | None = None
        self._last_error: str | None = None
        self._last_scene_result: ScenePersonSafetyResult | None = None
        self._completion_times: deque[float] = deque()
        self._preprocess_timing = TimingDistribution()
        self._enqueue_timing = TimingDistribution()
        self._dequeue_timing = TimingDistribution()
        self._inference_timing = TimingDistribution()
        self._decision_queue_dwell_timing = TimingDistribution()
        self._decision_handler_timing = TimingDistribution()
        self._evidence_persistence_timing = TimingDistribution()
        self._evidence_completion_handler_timing = TimingDistribution()

    def start(self) -> None:
        if not self.config.enabled or (self._thread is not None and self._thread.is_alive()):
            return
        self.store.prepare()
        self._stop_event.clear()
        self._downstream_stop.clear()
        self._decision_done.clear()
        self._evidence_done.clear()
        self._completion_backpressure.clear()
        self._completion_terminal_error.clear()
        self._decision_thread = threading.Thread(
            target=self._decision_worker,
            name="auto-fire-decision",
            daemon=True,
        )
        self._evidence_thread = threading.Thread(
            target=self._evidence_worker,
            name="classifier-evidence",
            daemon=True,
        )
        self._evidence_completion_thread = threading.Thread(
            target=self._evidence_completion_worker,
            name="classifier-evidence-completion",
            daemon=True,
        )
        self._thread = threading.Thread(target=self._run, name="classifier", daemon=True)
        self._decision_thread.start()
        self._evidence_thread.start()
        self._evidence_completion_thread.start()
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        deadline = monotonic() + max(0.0, timeout)
        self._stop_event.set()
        with self._scheduler_condition:
            self._scheduler_condition.notify_all()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, deadline - monotonic()))
        for downstream in (
            self._decision_thread,
            self._evidence_thread,
            self._evidence_completion_thread,
        ):
            if downstream is not None and downstream is not threading.current_thread():
                downstream.join(timeout=max(0.0, deadline - monotonic()))

    def set_paused(self, paused: bool) -> None:
        """Pause CPU-heavy inference without stopping the worker lifecycle."""

        with self._lock:
            self._paused = paused
        with self._scheduler_condition:
            self._scheduler_condition.notify_all()

    def submit(
        self,
        event_id: str,
        event_directory: Path,
        frame_number: int,
        frame: np.ndarray,
        source_bounding_box: tuple[int, int, int, int],
        *,
        selection_method: str = "configured_fallback",
        selected_motion_bounding_box_area: int | None = None,
        total_event_frames_considered: int = 1,
        context: str = "completed_event",
        track_id: int | None = None,
        frame_sequence: int | None = None,
        target_pixel: tuple[int, int] | None = None,
        target_observed_monotonic: float | None = None,
        target_provisional_category: str | None = None,
        target_confirmed: bool = False,
        target_event_eligible: bool = False,
        source_media_role: str = "clean_authoritative",
    ) -> bool:
        with self._lock:
            paused = self._paused
        if not self.config.enabled or paused or self._stop_event.is_set():
            return False
        preprocess_started = perf_counter()
        crop, crop_box = _candidate_crop(frame, source_bounding_box, self.config.crop_margin_percent)
        self._preprocess_timing.add(perf_counter() - preprocess_started)
        task = ClassifierTask(
            event_id=event_id,
            event_directory=event_directory,
            frame_number=frame_number,
            image=crop,
            source_bounding_box=source_bounding_box,
            crop_bounding_box=crop_box,
            submitted_at=datetime.now().astimezone().isoformat(timespec="milliseconds"),
            selection_method=selection_method,
            selected_motion_bounding_box_area=selected_motion_bounding_box_area,
            total_event_frames_considered=total_event_frames_considered,
            original_image=frame.copy(),
            context=context,
            track_id=track_id,
            frame_sequence=frame_sequence,
            target_pixel=target_pixel,
            target_observed_monotonic=target_observed_monotonic,
            target_provisional_category=target_provisional_category,
            target_confirmed=target_confirmed,
            target_event_eligible=target_event_eligible,
            source_media_role=source_media_role,
            enqueued_monotonic=monotonic(),
        )
        return self._enqueue(task)

    def retry(self, item_id: str) -> bool:
        with self._lock:
            paused = self._paused
        if not self.config.enabled or paused or self._stop_event.is_set():
            return False
        record = self.store.get_record(item_id)
        if record.get("classification_status") != "unclassified":
            raise ValueError("Only classification errors can be retried")
        image = cv2.imread(str(self.store.input_path(item_id)), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError("Saved classifier input could not be read")
        event_directory = self.store._record_path(item_id).parent
        original_image = cv2.imread(str(event_directory / ORIGINAL_FRAME_FILENAME), cv2.IMREAD_COLOR)
        task = ClassifierTask(
            event_id=item_id,
            event_directory=event_directory,
            frame_number=int(record.get("frame_number", 1)),
            image=image,
            source_bounding_box=_dict_box(record.get("source_bounding_box")),
            crop_bounding_box=_dict_box(record.get("crop_bounding_box")),
            submitted_at=datetime.now().astimezone().isoformat(timespec="milliseconds"),
            selection_method=str(record.get("frame_selection_method", "configured_fallback")),
            selected_motion_bounding_box_area=_optional_int(record.get("selected_motion_bounding_box_area")),
            total_event_frames_considered=max(
                1,
                _optional_int(record.get("total_event_frames_considered")) or 1,
            ),
            original_image=original_image,
            context="retry",
            source_media_role="clean_authoritative" if verified_classifier_source(record) else "unknown_legacy",
            enqueued_monotonic=monotonic(),
        )
        self.store.record_action("retry_requested", record)
        return self._enqueue(task)

    def classify_scene(
        self,
        request_id: str,
        event_id: str,
        track_id: int,
        packet: SceneFramePacket,
        *,
        timeout_seconds: float,
    ) -> ScenePersonSafetyResult:
        """Run one highest-priority latest-only full-frame person check."""

        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0.0
        ):
            raise ValueError("timeout_seconds must be finite and greater than zero")
        with self._lock:
            paused = self._paused
        if (
            not self.config.enabled
            or paused
            or self._stop_event.is_set()
            or self._thread is None
            or not self._thread.is_alive()
        ):
            return self._scene_failure(
                request_id,
                event_id,
                track_id,
                packet,
                "unavailable",
                "classifier_unavailable",
            )
        task = _SceneInferenceTask(
            request_id,
            event_id,
            track_id,
            packet,
            monotonic() + float(timeout_seconds),
            monotonic(),
            threading.Event(),
            threading.Event(),
        )
        superseded: _SceneInferenceTask | None = None
        with self._scheduler_condition:
            superseded = self._pending_scene
            self._pending_scene = task
            self._scheduler_condition.notify()
        if superseded is not None:
            superseded.result = self._scene_failure(
                superseded.request_id,
                superseded.event_id,
                superseded.track_id,
                superseded.packet,
                "unavailable",
                "superseded_by_newer_scene_request",
            )
            superseded.completed.set()
        if not task.completed.wait(timeout=float(timeout_seconds)) or task.result is None:
            task.expired.set()
            return self._scene_failure(
                request_id,
                event_id,
                track_id,
                packet,
                "unavailable",
                "scene_inference_timeout",
            )
        return task.result

    def latest_scene_result(self) -> ScenePersonSafetyResult | None:
        with self._lock:
            return self._last_scene_result

    def _enqueue(self, task: ClassifierTask) -> bool:
        enqueue_started = perf_counter()
        if self._completion_backpressure.is_set():
            with self._lock:
                self._last_error = "classifier_completion_backpressure"
            self._queue_rejected_task(
                task,
                "classifier_completion_backpressure",
                evidence_completion_required=False,
            )
            self._enqueue_timing.add(perf_counter() - enqueue_started)
            return False
        live = task.context == "auto_fire_live_event"
        evicted: ClassifierTask | None = None
        accepted = False
        with self._scheduler_condition:
            queued = len(self._live_tasks) + len(self._background_tasks)
            if queued < self.config.worker_queue_capacity:
                (self._live_tasks if live else self._background_tasks).append(task)
                accepted = True
            elif live and self._background_tasks:
                evicted = self._background_tasks.popleft()
                self._live_tasks.append(task)
                accepted = True
            if accepted:
                self._scheduler_condition.notify()
        if evicted is not None:
            self._queue_rejected_task(
                evicted,
                "classifier_preempted_by_live_task",
                evidence_completion_required=True,
            )
        if not accepted:
            self._queue_rejected_task(
                task,
                "classifier_queue_full",
                evidence_completion_required=False,
            )
            self._enqueue_timing.add(perf_counter() - enqueue_started)
            return False
        with self._lock:
            self._submitted += 1
        self._enqueue_timing.add(perf_counter() - enqueue_started)
        return True

    def _queue_rejected_task(
        self,
        task: ClassifierTask,
        error: str,
        *,
        evidence_completion_required: bool,
    ) -> None:
        with self._lock:
            self._errors += 1
            self._last_error = error
        outcome = _ClassifierOutcome(task, (), error, None, MobileNetSSDDetector.model_name)
        self._queue_decision(outcome)
        if evidence_completion_required:
            self._queue_evidence_completion(
                _EvidenceCompletion(task, "dropped_before_inference", None, error)
            )
        self._log_downstream_drop(task, "scheduler", error)

    def _queue_decision(self, outcome: _ClassifierOutcome) -> _DecisionDeliveryState:
        delivery = _DecisionDeliveryState()
        if self._result_handler is None:
            delivery.update("not_configured", started=True, finished=True)
            return delivery
        try:
            self._decision_queue.put_nowait(
                _DecisionDispatch(outcome, delivery, monotonic())
            )
        except queue.Full:
            delivery.update("dropped_queue_full", started=True, finished=True)
            with self._lock:
                self._decision_dropped += 1
                self._last_error = "classifier_decision_queue_full"
            self._log_downstream_drop(
                outcome.task,
                "decision",
                "classifier_decision_queue_full",
            )
        return delivery

    def status(self) -> ClassifierStatus:
        status_now = monotonic()
        with self._scheduler_condition:
            queue_depth = len(self._live_tasks) + len(self._background_tasks)
            scene_pending = self._pending_scene is not None or self._scene_active
        with self._lock:
            self._prune_completion_times_locked(perf_counter())
            return ClassifierStatus(
                enabled=self.config.enabled,
                thread_alive=self._thread is not None and self._thread.is_alive(),
                submitted=self._submitted,
                completed=self._completed,
                auto_accepted=self._auto_accepted,
                queued_for_review=self._queued_for_review,
                queue_depth=queue_depth,
                last_latency_ms=self._last_latency_ms,
                last_error=self._last_error,
                unknown=self._unknown,
                errors=self._errors,
                paused=self._paused,
                skipped_while_paused=self._skipped_while_paused,
                inference_fps=len(self._completion_times) / 60.0,
                decision_queue_depth=self._decision_queue.qsize(),
                evidence_queue_depth=self._evidence_queue.qsize(),
                evidence_completion_queue_depth=(
                    self._evidence_completion_queue.qsize()
                    + len(self._evidence_completion_overflow)
                ),
                evidence_completion_backpressured=self._completion_backpressure.is_set(),
                evidence_completion_terminal_error=self._completion_terminal_error.is_set(),
                scene_request_pending=scene_pending,
                decision_dropped=self._decision_dropped,
                evidence_dropped=self._evidence_dropped,
                evidence_completion_dropped=self._evidence_completion_dropped,
                decision_thread_alive=(
                    self._decision_thread is not None and self._decision_thread.is_alive()
                ),
                evidence_thread_alive=(
                    self._evidence_thread is not None and self._evidence_thread.is_alive()
                ),
                evidence_completion_thread_alive=(
                    self._evidence_completion_thread is not None
                    and self._evidence_completion_thread.is_alive()
                ),
                inference_active=self._inference_active,
                decision_active=self._decision_active,
                evidence_active=self._evidence_active,
                evidence_completion_active=self._evidence_completion_active,
                inference_last_progress_age_seconds=self._progress_age(
                    status_now,
                    self._last_inference_progress_monotonic,
                ),
                decision_last_progress_age_seconds=self._progress_age(
                    status_now,
                    self._last_decision_progress_monotonic,
                ),
                evidence_last_progress_age_seconds=self._progress_age(
                    status_now,
                    self._last_evidence_progress_monotonic,
                ),
                evidence_completion_last_progress_age_seconds=self._progress_age(
                    status_now,
                    self._last_evidence_completion_progress_monotonic,
                ),
                timing_distributions={
                    "producer_preprocess": self._preprocess_timing.snapshot(),
                    "scheduler_enqueue_overhead": self._enqueue_timing.snapshot(),
                    "inference_queue_dwell": self._dequeue_timing.snapshot(),
                    "dnn_inference": self._inference_timing.snapshot(),
                    "decision_queue_dwell": self._decision_queue_dwell_timing.snapshot(),
                    "decision_handler_duration": self._decision_handler_timing.snapshot(),
                    "evidence_persistence": self._evidence_persistence_timing.snapshot(),
                    "evidence_completion_handler": (
                        self._evidence_completion_handler_timing.snapshot()
                    ),
                },
                inference_completed=self._inference_completed,
                evidence_persisted=self._completed,
            )

    @staticmethod
    def _progress_age(now: float, then: float | None) -> float | None:
        return None if then is None else max(0.0, now - then)

    def _set_stage_activity(self, stage: str, active: bool, progressed_at: float) -> None:
        with self._lock:
            if stage == "inference":
                self._inference_active = active
                self._last_inference_progress_monotonic = progressed_at
            elif stage == "decision":
                self._decision_active = active
                self._last_decision_progress_monotonic = progressed_at
            elif stage == "evidence":
                self._evidence_active = active
                self._last_evidence_progress_monotonic = progressed_at
            elif stage == "evidence_completion":
                self._evidence_completion_active = active
                self._last_evidence_completion_progress_monotonic = progressed_at
            else:  # pragma: no cover - internal invariant
                raise ValueError(f"unsupported classifier stage: {stage}")

    def _prune_completion_times_locked(self, now: float) -> None:
        cutoff = now - 60.0
        while self._completion_times and self._completion_times[0] < cutoff:
            self._completion_times.popleft()

    def _append_completion_time_locked(self, completed_at: float) -> None:
        self._completion_times.append(completed_at)
        self._prune_completion_times_locked(completed_at)

    def _record_inference_completion(self, reported_latency_ms: float | None) -> None:
        completed_at = perf_counter()
        with self._lock:
            self._inference_completed += 1
            self._last_latency_ms = reported_latency_ms
            self._append_completion_time_locked(completed_at)

    def _next_work(self) -> ClassifierTask | _SceneInferenceTask | None:
        with self._scheduler_condition:
            self._scheduler_condition.wait_for(
                lambda: self._pending_scene is not None
                or bool(self._live_tasks)
                or bool(self._background_tasks)
                or self._stop_event.is_set(),
                timeout=0.2,
            )
            if self._pending_scene is not None:
                task = self._pending_scene
                self._pending_scene = None
                self._scene_active = True
                return task
            if self._live_tasks:
                return self._live_tasks.popleft()
            if self._background_tasks:
                return self._background_tasks.popleft()
            return None

    def _scheduler_empty(self) -> bool:
        with self._scheduler_condition:
            return (
                self._pending_scene is None
                and not self._live_tasks
                and not self._background_tasks
                and not self._scene_active
            )

    def _run(self) -> None:
        set_current_thread_name("classifier")
        detector: MobileNetSSDDetector | None = None
        detector_error: str | None = None
        try:
            while not self._stop_event.is_set() or not self._scheduler_empty():
                work = self._next_work()
                if work is None:
                    continue
                dequeued_at = monotonic()
                enqueued_at = work.enqueued_monotonic
                if enqueued_at is not None:
                    self._dequeue_timing.add(dequeued_at - enqueued_at)
                self._set_stage_activity("inference", True, dequeued_at)
                try:
                    with self._lock:
                        paused = self._paused
                    if paused:
                        if isinstance(work, _SceneInferenceTask):
                            self._complete_scene(
                                work,
                                self._scene_failure(
                                    work.request_id,
                                    work.event_id,
                                    work.track_id,
                                    work.packet,
                                    "unavailable",
                                    "classifier_paused",
                                ),
                            )
                        else:
                            with self._lock:
                                self._skipped_while_paused += 1
                            self._queue_outcome(
                                _ClassifierOutcome(
                                    work,
                                    (),
                                    "classifier_paused",
                                    None,
                                    MobileNetSSDDetector.model_name,
                                )
                            )
                        continue
                    if detector is None:
                        try:
                            detector = self._detector_factory()
                            detector_error = None
                        except Exception as exc:
                            detector_error = f"{type(exc).__name__}: {exc}"
                            LOGGER.error(
                                "Classifier model could not be loaded",
                                extra={
                                    "structured_data": {
                                        "event": "classifier_load_error",
                                        "error": detector_error,
                                    }
                                },
                            )
                    if isinstance(work, _SceneInferenceTask):
                        if monotonic() > work.deadline_monotonic:
                            self._complete_scene(
                                work,
                                self._scene_failure(
                                    work.request_id,
                                    work.event_id,
                                    work.track_id,
                                    work.packet,
                                    "unavailable",
                                    "scene_inference_deadline_expired",
                                ),
                            )
                            continue
                        if detector is None:
                            self._complete_scene(
                                work,
                                self._scene_failure(
                                    work.request_id,
                                    work.event_id,
                                    work.track_id,
                                    work.packet,
                                    "unavailable",
                                    detector_error or "classifier_unavailable",
                                ),
                            )
                            continue
                        inference_started = perf_counter()
                        reported_latency: float | None = None
                        try:
                            detections, _latency = detector.classify(work.packet.frame)
                            reported_latency = _latency
                            scene_detections = tuple(
                                SceneDetection(
                                    item.label.strip().lower(),
                                    item.confidence,
                                    item.bounding_box,
                                )
                                for item in detections
                            )
                            status = (
                                "person"
                                if any(item.label == "person" for item in scene_detections)
                                else "clear"
                            )
                            result = ScenePersonSafetyResult(
                                status=status,
                                request_id=work.request_id,
                                event_id=work.event_id,
                                track_id=work.track_id,
                                coordinate_space="native_full_frame",
                                source_sequence=work.packet.sequence,
                                source_received_monotonic=work.packet.received_monotonic,
                                frame_width=work.packet.width,
                                frame_height=work.packet.height,
                                detections=scene_detections,
                                completed_monotonic=max(
                                    monotonic(),
                                    work.packet.received_monotonic,
                                ),
                            )
                        except Exception as exc:
                            error = f"{type(exc).__name__}: {exc}"
                            detector = None
                            detector_error = error
                            result = self._scene_failure(
                                work.request_id,
                                work.event_id,
                                work.track_id,
                                work.packet,
                                "error",
                                error,
                            )
                            LOGGER.error(
                                "Full-frame person-safety inference failed",
                                extra={
                                    "structured_data": {
                                        "event": "scene_person_safety_error",
                                        "event_id": work.event_id,
                                        "source_sequence": work.packet.sequence,
                                        "error": error,
                                    }
                                },
                                exc_info=True,
                            )
                        finally:
                            self._inference_timing.add(perf_counter() - inference_started)
                            self._record_inference_completion(reported_latency)
                        self._complete_scene(work, result)
                        continue
                    if detector is None:
                        outcome = _ClassifierOutcome(
                            work,
                            (),
                            detector_error or "classifier_unavailable",
                            None,
                            MobileNetSSDDetector.model_name,
                        )
                    else:
                        inference_started = perf_counter()
                        reported_latency = None
                        try:
                            detections, latency = detector.classify(work.image)
                            reported_latency = latency
                            outcome = _ClassifierOutcome(
                                work,
                                tuple(detections),
                                None,
                                latency,
                                detector.model_name,
                            )
                        except Exception as exc:
                            error = f"{type(exc).__name__}: {exc}"
                            outcome = _ClassifierOutcome(work, (), error, None, detector.model_name)
                            detector = None
                            detector_error = error
                            LOGGER.error(
                                "Classifier inference failed",
                                extra={
                                    "structured_data": {
                                        "event": "classifier_inference_error",
                                        "event_id": work.event_id,
                                        "error": error,
                                    }
                                },
                                exc_info=True,
                            )
                        finally:
                            self._inference_timing.add(perf_counter() - inference_started)
                            self._record_inference_completion(reported_latency)
                    self._queue_outcome(outcome)
                finally:
                    self._set_stage_activity("inference", False, monotonic())
        finally:
            self._downstream_stop.set()

    def _complete_scene(
        self,
        task: _SceneInferenceTask,
        result: ScenePersonSafetyResult,
    ) -> None:
        if (
            task.expired.is_set() or monotonic() >= task.deadline_monotonic
        ) and result.status in {"clear", "person"}:
            result = self._scene_failure(
                task.request_id,
                task.event_id,
                task.track_id,
                task.packet,
                "unavailable",
                "scene_inference_deadline_expired",
            )
        task.result = result
        with self._lock:
            self._last_scene_result = result
        with self._scheduler_condition:
            self._scene_active = False
            self._scheduler_condition.notify_all()
        task.completed.set()

    @staticmethod
    def _scene_failure(
        request_id: str,
        event_id: str,
        track_id: int,
        packet: SceneFramePacket,
        status: str,
        error: str,
    ) -> ScenePersonSafetyResult:
        return ScenePersonSafetyResult(
            status=status,  # type: ignore[arg-type]
            request_id=request_id,
            event_id=event_id,
            track_id=track_id,
            coordinate_space="native_full_frame",
            source_sequence=packet.sequence,
            source_received_monotonic=packet.received_monotonic,
            frame_width=packet.width,
            frame_height=packet.height,
            detections=(),
            completed_monotonic=max(monotonic(), packet.received_monotonic),
            error=error,
        )

    def _queue_outcome(self, outcome: _ClassifierOutcome) -> None:
        delivery = self._queue_decision(outcome)
        try:
            self._evidence_queue.put_nowait(_EvidenceJob(outcome, delivery))
        except queue.Full:
            with self._lock:
                self._evidence_dropped += 1
                self._last_error = "classifier_evidence_queue_full"
            self._log_downstream_drop(outcome.task, "evidence", "classifier_evidence_queue_full")
            self._queue_evidence_completion(
                _EvidenceCompletion(
                    outcome.task,
                    "dropped_queue_full",
                    None,
                    "classifier_evidence_queue_full",
                )
            )

    @staticmethod
    def _log_downstream_drop(task: ClassifierTask, stage: str, reason: str) -> None:
        LOGGER.error(
            "Classifier downstream %s work dropped for event=%s reason=%s",
            stage,
            task.event_id,
            reason,
            extra={
                "structured_data": {
                    "event": "classifier_downstream_drop",
                    "stage": stage,
                    "event_id": task.event_id,
                    "track_id": task.track_id,
                    "classification_context": task.context,
                    "source_frame_sequence": task.frame_sequence,
                    "reason": reason,
                }
            },
        )

    def _decision_worker(self) -> None:
        set_current_thread_name("auto-fire-decision")
        try:
            while not self._downstream_stop.is_set() or not self._decision_queue.empty():
                try:
                    item = self._decision_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                self._decision_queue_dwell_timing.add(
                    monotonic() - item.enqueued_monotonic
                )
                started = perf_counter()
                self._set_stage_activity("decision", True, monotonic())
                item.delivery.update("started", started=True)
                try:
                    status = self._notify_result(item.outcome)
                    item.delivery.update(status, finished=True)
                finally:
                    self._decision_handler_timing.add(perf_counter() - started)
                    self._set_stage_activity("decision", False, monotonic())
                    self._decision_queue.task_done()
        finally:
            self._decision_done.set()

    def _evidence_worker(self) -> None:
        set_current_thread_name("classifier-evidence")
        try:
            while not self._downstream_stop.is_set() or not self._evidence_queue.empty():
                try:
                    item = self._evidence_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                persistence_started = perf_counter()
                self._set_stage_activity("evidence", True, monotonic())
                record: dict[str, Any] | None = None
                completion_status = "failed"
                completion_error: str | None = None
                try:
                    dispatch_started = item.delivery.started.wait(
                        timeout=DECISION_DISPATCH_START_WAIT_SECONDS
                    )
                    if not dispatch_started:
                        raise RuntimeError(
                            "decision_dispatch_not_started_before_evidence_deadline"
                        )
                    delivery_status = item.delivery.snapshot()
                    outcome = item.outcome
                    record = self.store.save_classification(
                        outcome.task,
                        list(outcome.detections),
                        outcome.latency_ms,
                        outcome.model_name,
                        error=outcome.error,
                        decision_delivery_status=delivery_status,
                    )
                    completion_status = "persisted"
                    with self._lock:
                        self._completed += 1
                        self._last_error = outcome.error
                        if record["auto_accepted"]:
                            self._auto_accepted += 1
                        elif record["classification_status"] == "review":
                            self._queued_for_review += 1
                        elif record["classification_status"] == "unknown":
                            self._unknown += 1
                        elif record["classification_status"] == "unclassified":
                            self._errors += 1
                except Exception as exc:
                    completion_error = f"{type(exc).__name__}: {exc}"
                    with self._lock:
                        self._evidence_dropped += 1
                        self._last_error = completion_error
                    LOGGER.error(
                        "Classifier evidence persistence failed",
                        extra={
                            "structured_data": {
                                "event": "classifier_evidence_error",
                                "event_id": item.outcome.task.event_id,
                                "track_id": item.outcome.task.track_id,
                                "classification_context": item.outcome.task.context,
                                "error": completion_error,
                            }
                        },
                        exc_info=True,
                    )
                finally:
                    self._evidence_persistence_timing.add(perf_counter() - persistence_started)
                    self._set_stage_activity("evidence", False, monotonic())
                    self._queue_evidence_completion(
                        _EvidenceCompletion(
                            item.outcome.task,
                            completion_status,
                            record,
                            completion_error,
                        )
                    )
                    self._evidence_queue.task_done()
        finally:
            self._evidence_done.set()

    def _queue_evidence_completion(self, completion: _EvidenceCompletion) -> None:
        if self._evidence_result_handler is None:
            return
        try:
            self._evidence_completion_queue.put_nowait(completion)
            if self._evidence_completion_queue.qsize() >= self._completion_reserve:
                self._completion_backpressure.set()
        except queue.Full:
            capacity_exhausted = False
            with self._lock:
                if len(self._evidence_completion_overflow) < self._completion_reserve:
                    self._evidence_completion_overflow.append(completion)
                    self._last_error = "classifier_evidence_completion_overflow"
                else:  # pragma: no cover - requires violating bounded in-flight invariant
                    self._evidence_completion_dropped += 1
                    self._last_error = "classifier_evidence_completion_capacity_exhausted"
                    capacity_exhausted = True
            self._completion_backpressure.set()
            if capacity_exhausted:
                self._log_downstream_drop(
                    completion.task,
                    "evidence_completion",
                    "classifier_evidence_completion_capacity_exhausted",
                )

    def _evidence_completion_worker(self) -> None:
        set_current_thread_name("classifier-evidence-completion")
        while not (
            self._downstream_stop.is_set()
            and self._evidence_done.is_set()
            and self._evidence_completion_queue.empty()
            and not self._evidence_completion_overflow
        ):
            completion_from_queue = False
            with self._lock:
                completion = (
                    self._evidence_completion_overflow.popleft()
                    if self._evidence_completion_overflow
                    else None
                )
            if completion is None:
                try:
                    completion = self._evidence_completion_queue.get(timeout=0.2)
                    completion_from_queue = True
                except queue.Empty:
                    continue
            handler_started = perf_counter()
            self._set_stage_activity("evidence_completion", True, monotonic())
            try:
                handler = self._evidence_result_handler
                if handler is not None:
                    handler(
                        completion.task,
                        completion.status,
                        completion.record,
                        completion.error,
                    )
            except Exception as exc:
                retrying = completion.attempts < 2
                LOGGER.exception(
                    "Classifier evidence completion handler failed%s",
                    "; retrying" if retrying else "; terminal backpressure engaged",
                    extra={
                        "structured_data": {
                            "event": "classifier_evidence_completion_handler_error",
                            "event_id": completion.task.event_id,
                            "track_id": completion.task.track_id,
                            "classification_context": completion.task.context,
                            "evidence_status": completion.status,
                            "attempt": completion.attempts + 1,
                            "retrying": retrying,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    },
                )
                if retrying:
                    self._queue_evidence_completion(
                        _EvidenceCompletion(
                            completion.task,
                            completion.status,
                            completion.record,
                            completion.error,
                            completion.attempts + 1,
                        )
                    )
                else:
                    with self._lock:
                        self._evidence_completion_dropped += 1
                        self._last_error = "classifier_evidence_completion_handler_failed"
                    self._completion_terminal_error.set()
                    self._completion_backpressure.set()
            finally:
                self._evidence_completion_handler_timing.add(
                    perf_counter() - handler_started
                )
                self._set_stage_activity("evidence_completion", False, monotonic())
                if completion_from_queue:
                    self._evidence_completion_queue.task_done()
                if (
                    self._evidence_completion_queue.qsize() < self._completion_reserve
                    and not self._evidence_completion_overflow
                    and not self._completion_terminal_error.is_set()
                ):
                    self._completion_backpressure.clear()

    def _notify_result(self, outcome: _ClassifierOutcome) -> str:
        handler = self._result_handler
        if handler is None:
            return "not_configured"
        task = outcome.task
        decision_record: dict[str, Any] = {
            "classification_context": task.context,
            "track_id": task.track_id,
            "source_frame_sequence": task.frame_sequence,
            "target_observed_monotonic": task.target_observed_monotonic,
            "target_pixel": (
                None
                if task.target_pixel is None
                else {"x": task.target_pixel[0], "y": task.target_pixel[1]}
            ),
            "evidence_persistence_pending": True,
        }
        try:
            handler(
                task,
                list(outcome.detections),
                outcome.error,
                decision_record,
            )
            return "completed"
        except Exception:
            LOGGER.exception(
                "Classifier result handler failed; evidence persistence remains isolated",
                extra={
                    "structured_data": {
                        "event": "classifier_result_handler_error",
                        "event_id": task.event_id,
                        "classification_context": task.context,
                    }
                },
            )
            return "handler_error"


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


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


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
