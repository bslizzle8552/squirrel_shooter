from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from conftest import write_test_config
from squirrel_shooter.auto_fire import AutoFireTargetAssociation, AutoFireTargetSnapshot
from squirrel_shooter.camera_service import CameraStatus
from squirrel_shooter.classifier import (
    ClassifierDetection,
    ClassifierEvidenceStore,
    ClassifierTask,
    EventClassifier,
    MobileNetSSDDetector,
)
from squirrel_shooter.classifier_setup import ModelFile, install_model
from squirrel_shooter.config import load_config
from squirrel_shooter.motion_runtime import MotionProcessingService
from squirrel_shooter.vision_service import VisionStatus
from squirrel_shooter.web_dashboard import create_app
import squirrel_shooter.classifier_setup as classifier_setup


def classifier_config(tmp_path: Path, **changes: object):
    config = load_config(write_test_config(tmp_path, classifier__enabled=True, **changes))
    return replace(
        config,
        classifier=replace(
            config.classifier,
            model_definition=tmp_path / "deploy.prototxt",
            model_weights=tmp_path / "model.caffemodel",
        ),
    )


def task(tmp_path: Path, event_id: str = "event-one") -> ClassifierTask:
    event_directory = tmp_path / "captures" / "events" / "2026-07-16" / event_id
    event_directory.mkdir(parents=True)
    return ClassifierTask(
        event_id=event_id,
        event_directory=event_directory,
        frame_number=1,
        image=np.full((24, 32, 3), 120, dtype=np.uint8),
        source_bounding_box=(10, 20, 30, 40),
        crop_bounding_box=(4, 12, 42, 56),
        submitted_at="2026-07-16T18:00:00-04:00",
        original_image=np.full((72, 128, 3), 80, dtype=np.uint8),
    )


class FakeNet:
    def __init__(self) -> None:
        self.input_shape: tuple[int, ...] | None = None

    def setInput(self, blob: np.ndarray) -> None:  # noqa: N802 - OpenCV API shape
        self.input_shape = blob.shape

    def forward(self) -> np.ndarray:
        return np.array([[[[0, 15, 0.91, 0.1, 0.2, 0.8, 0.9]]]], dtype=np.float32)


def test_mobilenet_detector_returns_voc_person_detection(tmp_path: Path) -> None:
    config = classifier_config(tmp_path)
    net = FakeNet()
    detector = MobileNetSSDDetector(config.classifier, net=net)

    detections, latency_ms = detector.classify(np.zeros((100, 200, 3), dtype=np.uint8))

    assert net.input_shape == (1, 3, 300, 300)
    assert detections[0].label == "person"
    assert detections[0].confidence == pytest.approx(0.91)
    assert detections[0].bounding_box == (20, 20, 140, 70)
    assert latency_ms >= 0


def test_evidence_store_unifies_known_unknown_and_review_inside_event_folders(tmp_path: Path) -> None:
    config = classifier_config(tmp_path)
    store = ClassifierEvidenceStore(config)
    person_task = task(tmp_path, "person-event")
    unknown_task = task(tmp_path, "unknown-event")
    other_class_task = task(tmp_path, "other-class-event")
    review_task = task(tmp_path, "review-event")

    known = store.save_classification(
        person_task,
        [ClassifierDetection("person", 0.92, (1, 2, 20, 21))],
        185.2,
        "test-model",
    )
    unknown = store.save_classification(unknown_task, [], 172.1, "test-model")
    other_class = store.save_classification(
        other_class_task,
        [ClassifierDetection("dog", 0.91, (2, 3, 12, 14))],
        171.0,
        "test-model",
    )
    review = store.save_classification(
        review_task,
        [ClassifierDetection("car", 0.42, (2, 3, 12, 14))],
        170.0,
        "test-model",
    )

    assert known["classification_status"] == "known" and known["display_label"] == "Person"
    assert known["label_source"] == "automatic" and known["decision_confidence"] == 0.92
    assert known["input_image_role"] == "unannotated_target_crop"
    assert known["original_frame_path"].endswith("original-frame.jpg")
    assert unknown["classification_status"] == "unknown" and unknown["display_label"] == "Unknown"
    assert other_class["classification_status"] == "unknown" and other_class["model_suggestion"] == "dog"
    assert review["classification_status"] == "review" and review["review_suggestion_label"] == "car"
    for classifier_task in (person_task, unknown_task, other_class_task, review_task):
        assert (classifier_task.event_directory / "classifier-input.jpg").is_file()
        assert (classifier_task.event_directory / "original-frame.jpg").is_file()
        assert (classifier_task.event_directory / "classification.json").is_file()
    assert store.counts() == {"known": 1, "unknown": 2, "review": 1, "errors": 0, "false_positive": 0}
    assert store.training_summary()["eligible_samples"] == 0
    audit = config.logging.directory / config.classifier.audit_log_filename
    assert [json.loads(line)["action"] for line in audit.read_text(encoding="utf-8").splitlines()] == [
        "classified",
        "classified",
        "classified",
        "classified",
    ]

    with pytest.raises(ValueError, match="corrected label"):
        store.review("review-event", "approve")
    reviewed = store.review("review-event", "approve", "car")

    assert reviewed["classification_status"] == "known" and reviewed["display_label"] == "Car"
    assert reviewed["human_label"] == "car" and reviewed["label_source"] == "human"
    assert reviewed["schema_version"] == 5
    assert reviewed["training_label"] == "car" and reviewed["training_dataset_status"] == "included"
    assert (config.camera.output_directory / reviewed["training_sample_relative"]).is_file()
    assert store.training_summary()["labels"] == {"car": 1}
    assert json.loads(audit.read_text(encoding="utf-8").splitlines()[-1])["action"] == "human_labeled"

    corrected = store.review("person-event", "unknown")
    assert corrected["classification_status"] == "unknown" and corrected["display_label"] == "Unknown"
    assert corrected["label_source"] == "human" and corrected["human_label"] == "unknown"


def test_evidence_overview_cache_is_shared_copied_and_invalidated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = classifier_config(tmp_path)
    store = ClassifierEvidenceStore(config)
    first_task = task(tmp_path, "first-event")
    store.save_classification(
        first_task,
        [ClassifierDetection("person", 0.92, (1, 2, 20, 21))],
        100.0,
        "test-model",
    )
    scans = 0
    original_scan = store._scan_overview

    def counted_scan() -> dict[str, list[dict[str, object]]]:
        nonlocal scans
        scans += 1
        return original_scan()

    monkeypatch.setattr(store, "_scan_overview", counted_scan)

    overview = store.overview()
    overview["known"][0]["display_label"] = "Tampered"
    overview["known"][0]["detections"][0]["label"] = "tampered"
    assert store.list_items("known")[0]["display_label"] == "Person"
    assert store.list_items("known")[0]["detections"][0]["label"] == "person"
    assert store.counts()["known"] == 1
    assert scans == 1

    second_task = task(tmp_path, "second-event")
    store.save_classification(second_task, [], 110.0, "test-model")
    assert store.counts()["unknown"] == 1
    assert scans == 2

    store.review("second-event", "false-positive")
    assert store.counts()["unknown"] == 0
    assert store.counts()["false_positive"] == 1
    assert scans == 3


def test_classifier_completion_window_is_bounded_when_completion_is_appended(tmp_path: Path) -> None:
    config = classifier_config(tmp_path)
    classifier = EventClassifier(config.classifier, ClassifierEvidenceStore(config))
    classifier._completion_times.extend((1.0, 25.0, 59.0))

    with classifier._lock:
        classifier._append_completion_time_locked(61.1)

    assert list(classifier._completion_times) == [25.0, 59.0, 61.1]


def test_selection_metadata_is_saved_with_classification_result(tmp_path: Path) -> None:
    config = classifier_config(tmp_path)
    store = ClassifierEvidenceStore(config)
    selected_task = replace(
        task(tmp_path, "selected-event"),
        frame_number=4,
        selection_method="best",
        selected_motion_bounding_box_area=1200,
        total_event_frames_considered=7,
    )

    record = store.save_classification(
        selected_task,
        [ClassifierDetection("person", 0.91, (1, 2, 20, 21))],
        123.4,
        "test-model",
    )
    saved = json.loads((selected_task.event_directory / "classification.json").read_text(encoding="utf-8"))

    assert record["selected_event_frame_index"] == 4
    assert saved["frame_selection_method"] == "best"
    assert saved["selected_motion_bounding_box_area"] == 1200
    assert saved["total_event_frames_considered"] == 7
    assert saved["top_label"] == "person" and saved["top_confidence"] == 0.91


def test_live_classification_reads_final_event_metadata_after_lock_race(tmp_path: Path) -> None:
    config = classifier_config(tmp_path)
    store = ClassifierEvidenceStore(config)
    live_task = replace(
        task(tmp_path, "live-race-event"),
        context="auto_fire_live_event",
        track_id=7,
    )

    class SignalingLock:
        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.attempted = threading.Event()

        def __enter__(self) -> None:
            self.attempted.set()
            self.lock.acquire()

        def __exit__(self, *_args: object) -> None:
            self.lock.release()

    gate = SignalingLock()
    gate.lock.acquire()
    store._lock = gate  # type: ignore[assignment]
    result: list[dict[str, object]] = []
    worker = threading.Thread(
        target=lambda: result.append(
            store.save_classification(
                live_task,
                [ClassifierDetection("dog", 0.91, (1, 2, 20, 21))],
                100.0,
                "test-model",
            )
        )
    )
    worker.start()
    assert gate.attempted.wait(timeout=1)
    (live_task.event_directory / "event.json").write_text(
        json.dumps(
            {
                "event_id": live_task.event_id,
                "start_timestamp": "2026-08-11T12:00:00-04:00",
                "session_id": "session-race",
                "capture_method": "automatic_motion_event",
                "software_version": "0.3.0",
                "git_commit_sha": "abc123",
                "source_camera": "opencv_device_0",
                "actual_width": 1280,
                "actual_height": 720,
            }
        ),
        encoding="utf-8",
    )
    gate.lock.release()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert result[0]["event_timestamp"] == "2026-08-11T12:00:00-04:00"
    assert result[0]["session_id"] == "session-race"
    assert result[0]["software_version"] == "0.3.0"
    assert result[0]["source_camera"] == {
        "source_camera": "opencv_device_0",
        "actual_width": 1280,
        "actual_height": 720,
    }


def test_original_frame_write_failure_is_recorded_without_losing_classification(tmp_path: Path) -> None:
    config = classifier_config(tmp_path)

    def image_writer(path: str, _image: np.ndarray) -> bool:
        if path.endswith("original-frame.jpg"):
            return False
        Path(path).write_bytes(b"classifier crop")
        return True

    store = ClassifierEvidenceStore(config, image_writer=image_writer)
    classifier_task = task(tmp_path, "original-write-failure")

    record = store.save_classification(classifier_task, [], 100.0, "test-model")

    assert record["classification_status"] == "unknown"
    assert record["original_frame_path"] is None
    assert "Could not save original event frame" in record["original_frame_error"]
    assert (classifier_task.event_directory / "classifier-input.jpg").is_file()
    assert (classifier_task.event_directory / "classification.json").is_file()


def test_human_truth_builds_durable_current_training_manifest(tmp_path: Path) -> None:
    config = classifier_config(tmp_path)
    store = ClassifierEvidenceStore(config)
    wildlife_task = task(tmp_path, "wildlife-event")
    (wildlife_task.event_directory / "event.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "event_id": wildlife_task.event_id,
                "start_timestamp": "2026-07-18T08:00:00-04:00",
                "session_id": "session-one",
                "capture_method": "automatic_motion_event",
                "source_camera": "opencv_device_0",
                "camera_device_index": 0,
                "actual_width": 1280,
                "actual_height": 720,
                "software_version": "0.3.0",
                "git_commit_sha": "abc123",
                "provisional_category": "small_animal_candidate",
                "movement_attributes": ["coherent_travel"],
                "snapshot_file_role": "annotated_review_frame",
                "clip_file_role": "annotated_review_clip",
            }
        ),
        encoding="utf-8",
    )
    store.save_classification(
        wildlife_task,
        [ClassifierDetection("dog", 0.78, (2, 3, 12, 14))],
        110.0,
        "test-model",
    )

    confirmed = store.review("wildlife-event", "confirm-model")
    assert confirmed["human_label"] == "dog"
    assert confirmed["human_label_action"] == "model_confirmed"
    assert confirmed["human_verified"] is True
    sample_image = config.camera.output_directory / str(confirmed["training_sample_relative"])
    assert sample_image.is_file()
    sample = json.loads((sample_image.parent / "sample.json").read_text(encoding="utf-8"))
    assert sample["label"] == "dog" and sample["training_eligible"] is True
    assert sample["source"]["motion_category"] == "small_animal_candidate"
    assert sample["image_sha256"] == hashlib.sha256(sample_image.read_bytes()).hexdigest()
    original_frame = sample_image.parent / "original-frame.jpg"
    assert original_frame.is_file()
    assert sample["original_frame_sha256"] == hashlib.sha256(original_frame.read_bytes()).hexdigest()
    assert sample["image_role"] == "unannotated_target_crop"
    assert sample["original_frame_role"] == "unannotated_full_resolution_context"
    assert sample["split_group_event_id"] == "wildlife-event"
    assert sample["split_group_session_id"] == "session-one"
    assert sample["source"]["source_camera"] == {
        "source_camera": "opencv_device_0",
        "camera_device_index": 0,
        "actual_width": 1280,
        "actual_height": 720,
    }
    assert sample["source"]["git_commit_sha"] == "abc123"

    corrected = store.review("wildlife-event", "approve", "Eastern Gray-Squirrel")
    assert corrected["human_label"] == "eastern_gray_squirrel"
    manifest_rows = [
        json.loads(line)
        for line in store.training_manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(manifest_rows) == 1
    assert manifest_rows[0]["label"] == "eastern_gray_squirrel"
    assert store.training_summary()["labels"] == {"eastern_gray_squirrel": 1}
    event = json.loads((wildlife_task.event_directory / "event.json").read_text(encoding="utf-8"))
    assert event["human_review_label"] == "eastern_gray_squirrel"
    assert event["training_label"] == "eastern_gray_squirrel"
    assert event["classifier_input_role"] == "unannotated_target_crop"
    assert event["original_frame_role"] == "unannotated_full_resolution_context"
    assert event["predicted_class"] == "dog"

    shutil.rmtree(wildlife_task.event_directory)
    assert sample_image.is_file()
    assert original_frame.is_file()
    assert store.training_manifest_path.is_file()


def test_unknown_is_excluded_and_false_positive_becomes_canonical_negative_training_data(tmp_path: Path) -> None:
    config = classifier_config(tmp_path)
    store = ClassifierEvidenceStore(config)
    uncertain_task = task(tmp_path, "uncertain-event")
    background_task = task(tmp_path, "background-event")
    store.save_classification(uncertain_task, [], 100.0, "test-model")
    store.save_classification(background_task, [], 100.0, "test-model")

    store.review("uncertain-event", "approve", "squirrel")
    excluded = store.review("uncertain-event", "unknown")
    background = store.review("background-event", "false-positive")

    assert excluded["training_dataset_status"] == "excluded_unknown"
    assert background["training_label"] == "background_or_false_positive"
    manifest = [json.loads(line) for line in store.training_manifest_path.read_text(encoding="utf-8").splitlines()]
    assert [sample["label"] for sample in manifest] == ["background_or_false_positive"]


def test_prepare_canonicalizes_legacy_negative_training_labels(tmp_path: Path) -> None:
    config = classifier_config(tmp_path)
    event_directory = config.camera.output_directory / "events" / "2026-07-16" / "legacy-negative"
    event_directory.mkdir(parents=True)
    (event_directory / "event.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "event_id": "legacy-negative",
                "training_label": "background",
            }
        ),
        encoding="utf-8",
    )
    (event_directory / "classification.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "item_id": "legacy-negative",
                "event_id": "legacy-negative",
                "training_label": "background",
                "human_label": "false_positive",
                "human_label_action": "marked_false_positive",
                "reviewed_at": "2026-07-16T18:00:00-04:00",
            }
        ),
        encoding="utf-8",
    )
    sample_directory = config.camera.output_directory / "training-dataset" / "samples" / "legacy-negative"
    sample_directory.mkdir(parents=True)
    (sample_directory / "sample.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sample_id": "legacy-negative",
                "label": "background",
                "training_eligible": True,
            }
        ),
        encoding="utf-8",
    )

    store = ClassifierEvidenceStore(config)
    store.prepare()

    sample = json.loads((sample_directory / "sample.json").read_text(encoding="utf-8"))
    classification = json.loads((event_directory / "classification.json").read_text(encoding="utf-8"))
    event = json.loads((event_directory / "event.json").read_text(encoding="utf-8"))
    manifest = [json.loads(line) for line in store.training_manifest_path.read_text(encoding="utf-8").splitlines()]
    assert sample["label"] == "background_or_false_positive"
    assert sample["label_migrated_from"] == "background"
    assert classification["training_label"] == "background_or_false_positive"
    assert classification["schema_version"] == 5
    assert event["training_label"] == "background_or_false_positive"
    assert [row["label"] for row in manifest] == ["background_or_false_positive"]


def test_legacy_classifier_evidence_is_copied_without_deleting_originals(tmp_path: Path) -> None:
    config = classifier_config(tmp_path)
    event_directory = tmp_path / "captures" / "events" / "2026-07-16" / "legacy-event"
    event_directory.mkdir(parents=True)
    legacy_directory = config.classifier.evidence_directory / "rejected"
    legacy_directory.mkdir(parents=True)
    legacy_image = legacy_directory / "legacy-event.jpg"
    legacy_image.write_bytes(b"legacy classifier image")
    legacy_metadata = legacy_directory / "legacy-event.json"
    legacy_metadata.write_text(
        json.dumps(
            {
                "item_id": "legacy-event",
                "event_id": "legacy-event",
                "source_event_directory": str(event_directory),
                "classifier_timestamp": "2026-07-16T18:00:00-04:00",
                "frame_number": 1,
                "detections": [],
                "top_label": None,
                "outcome": "manual_rejected",
                "image_path": str(legacy_image),
                "error": None,
            }
        ),
        encoding="utf-8",
    )

    store = ClassifierEvidenceStore(config)
    store.prepare()

    migrated = json.loads((event_directory / "classification.json").read_text(encoding="utf-8"))
    assert migrated["classification_status"] == "unknown" and migrated["legacy_migrated"] is True
    assert (event_directory / "classifier-input.jpg").read_bytes() == b"legacy classifier image"
    assert legacy_image.exists() and legacy_metadata.exists()


def test_classifier_worker_is_backgrounded_and_records_one_task(tmp_path: Path) -> None:
    config = classifier_config(tmp_path)
    store = ClassifierEvidenceStore(config)

    class Detector:
        model_name = "fake-detector"

        def classify(self, image: np.ndarray):
            assert image.shape == (26, 32, 3)
            return [ClassifierDetection("car", 0.88, (0, 0, 10, 10))], 12.5

    worker = EventClassifier(config.classifier, store, detector_factory=Detector)  # type: ignore[arg-type]
    event_directory = tmp_path / "captures" / "events" / "worker-event"
    event_directory.mkdir(parents=True)
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    worker.start()
    try:
        assert worker.submit("worker-event", event_directory, 1, frame, (20, 10, 20, 16))
        deadline = time.monotonic() + 2
        while worker.status().completed < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        worker.stop()

    status = worker.status()
    assert status.submitted == status.completed == status.auto_accepted == 1
    assert status.queued_for_review == 0 and status.last_latency_ms == 12.5


def test_classifier_result_handler_receives_exact_live_task_and_is_failure_isolated(tmp_path: Path) -> None:
    config = classifier_config(tmp_path)
    store = ClassifierEvidenceStore(config)
    handled: list[tuple[ClassifierTask, list[ClassifierDetection], str | None, dict[str, object]]] = []

    class Detector:
        model_name = "fake-live-detector"

        def classify(self, _image: np.ndarray):
            return [ClassifierDetection("dog", 0.91, (0, 0, 10, 10))], 9.0

    def handler(
        live_task: ClassifierTask,
        detections: list[ClassifierDetection],
        error: str | None,
        record: dict[str, object],
    ) -> None:
        handled.append((live_task, detections, error, record))
        raise RuntimeError("synthetic handler failure")

    worker = EventClassifier(  # type: ignore[arg-type]
        config.classifier,
        store,
        detector_factory=Detector,
        result_handler=handler,
    )
    event_directory = tmp_path / "captures" / "events" / "live-handler"
    event_directory.mkdir(parents=True)
    worker.start()
    try:
        assert worker.submit(
            "live-handler",
            event_directory,
            1,
            np.zeros((40, 60, 3), dtype=np.uint8),
            (20, 10, 20, 16),
            context="auto_fire_live_event",
            track_id=7,
            frame_sequence=42,
            target_pixel=(30, 18),
            target_observed_monotonic=100.0,
            target_provisional_category="small_animal_candidate",
            target_confirmed=True,
            target_event_eligible=True,
        )
        deadline = time.monotonic() + 2
        while worker.status().completed < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        worker.stop()

    assert worker.status().completed == 1
    assert len(handled) == 1
    live_task, detections, error, record = handled[0]
    assert live_task.context == "auto_fire_live_event" and live_task.track_id == 7
    assert detections[0].label == "dog" and error is None
    assert record["classification_context"] == "auto_fire_live_event"
    assert record["track_id"] == 7 and record["target_pixel"] == {"x": 30, "y": 18}
    assert (event_directory / "classification.json").is_file()


def test_inference_exception_preserves_images_and_unavailable_metadata(tmp_path: Path) -> None:
    config = classifier_config(tmp_path)
    store = ClassifierEvidenceStore(config)

    class Detector:
        model_name = "failing-detector"

        def classify(self, _image: np.ndarray):
            raise RuntimeError("synthetic inference failure")

    worker = EventClassifier(config.classifier, store, detector_factory=Detector)  # type: ignore[arg-type]
    event_directory = tmp_path / "captures" / "events" / "inference-error"
    event_directory.mkdir(parents=True)
    worker.start()
    try:
        assert worker.submit(
            "inference-error",
            event_directory,
            1,
            np.zeros((40, 60, 3), dtype=np.uint8),
            (20, 10, 20, 16),
        )
        deadline = time.monotonic() + 2
        while not (event_directory / "classification.json").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        worker.stop()

    record = store.get_record("inference-error")
    assert record["classification_status"] == "unclassified"
    assert record["outcome"] == "classifier_error"
    assert record["error"] == "RuntimeError: synthetic inference failure"
    assert (event_directory / "classifier-input.jpg").is_file()
    assert (event_directory / "original-frame.jpg").is_file()
    assert worker.status().errors == 1


def test_failed_classification_can_retry_from_saved_input(tmp_path: Path) -> None:
    config = classifier_config(tmp_path)
    store = ClassifierEvidenceStore(config)
    model_available = False

    class Detector:
        model_name = "fake-detector"

        def classify(self, image: np.ndarray):
            assert image.ndim == 3
            return [ClassifierDetection("person", 0.8, (0, 0, 10, 10))], 10.0

    def detector_factory() -> Detector:
        if not model_available:
            raise FileNotFoundError("model unavailable")
        return Detector()

    worker = EventClassifier(config.classifier, store, detector_factory=detector_factory)  # type: ignore[arg-type]
    event_directory = tmp_path / "captures" / "events" / "retry-event"
    event_directory.mkdir(parents=True)
    worker.start()
    try:
        assert worker.submit(
            "retry-event",
            event_directory,
            1,
            np.zeros((40, 60, 3), dtype=np.uint8),
            (20, 10, 20, 16),
        )
        deadline = time.monotonic() + 2
        while not (event_directory / "classification.json").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert store.get_record("retry-event")["classification_status"] == "unclassified"
        model_available = True
        assert worker.retry("retry-event")
        deadline = time.monotonic() + 2
        while store.get_record("retry-event")["classification_status"] != "known" and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        worker.stop()

    assert store.get_record("retry-event")["display_label"] == "Person"
    audit = config.logging.directory / config.classifier.audit_log_filename
    assert [json.loads(line)["action"] for line in audit.read_text(encoding="utf-8").splitlines()] == [
        "classified",
        "retry_requested",
        "classified",
    ]


def test_model_installer_verifies_checksum_before_replacing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    content = b"pinned model bytes"
    monkeypatch.setattr(
        classifier_setup,
        "MODEL_FILES",
        (ModelFile("model.bin", hashlib.sha256(content).hexdigest()),),
    )

    installed = install_model(tmp_path / "model", opener=lambda *_args, **_kwargs: io.BytesIO(content))

    assert installed == [tmp_path / "model" / "model.bin"]
    assert installed[0].read_bytes() == content


def test_motion_submits_best_frame_once_after_event_completes(tmp_path: Path) -> None:
    config = classifier_config(tmp_path)

    class Classifier:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int, dict[str, object]]] = []

        def set_paused(self, _paused: bool) -> None:
            return None

        def submit(
            self,
            event_id: str,
            _directory: Path,
            frame_number: int,
            _frame: np.ndarray,
            _box: tuple[int, int, int, int],
            **metadata: object,
        ) -> bool:
            self.calls.append((event_id, frame_number, metadata))
            return True

    class Recorder:
        def __init__(self) -> None:
            self.active: dict[int, object] = {}
            self.updates = 0

        def begin(self, track_id: int, *_args: object, **_kwargs: object) -> object:
            directory = tmp_path / "captures" / "events" / "event-one"
            directory.mkdir(parents=True, exist_ok=True)
            event = SimpleNamespace(event_id="event-one", directory=directory, start_timestamp="now")
            self.active[track_id] = event
            return event

        def update(self, *_args: object, **_kwargs: object) -> None:
            self.updates += 1

        def should_finish(self, *_args: object, **_kwargs: object) -> bool:
            return self.updates >= 2

        def finish(self, track_id: int, **_kwargs: object) -> dict[str, object]:
            event = self.active.pop(track_id)
            return {
                "event_id": event.event_id,
                "snapshot_path": str(event.directory / "snapshot.jpg"),
                "clip_path": str(event.directory / "clip.avi"),
            }

    classifier = Classifier()
    motion = MotionProcessingService(SimpleNamespace(), config, classifier_service=classifier)  # type: ignore[arg-type]
    motion._recorder = Recorder()  # type: ignore[assignment]
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    packet = SimpleNamespace(frame=frame)
    first = SimpleNamespace(
        track_id=7,
        newly_confirmed=True,
        foreground_pixels=100,
        provisional_category="small_animal_candidate",
        movement_attributes=("coherent_travel",),
        bounding_box=(20, 20, 30, 20),
        contour_area=600.0,
    )
    second = SimpleNamespace(**{**first.__dict__, "newly_confirmed": False, "bounding_box": (20, 20, 45, 25)})
    third = SimpleNamespace(**{**first.__dict__, "newly_confirmed": False, "bounding_box": (20, 20, 35, 25)})

    motion._handle_events(packet, SimpleNamespace(groups=(first,)), lambda: frame, 1.0, 10.0)  # type: ignore[arg-type]
    motion._handle_events(packet, SimpleNamespace(groups=(second,)), lambda: frame, 1.1, 10.0)  # type: ignore[arg-type]
    assert classifier.calls == []
    motion._handle_events(packet, SimpleNamespace(groups=(third,)), lambda: frame, 1.2, 10.0)  # type: ignore[arg-type]

    assert len(classifier.calls) == 1
    event_id, frame_number, metadata = classifier.calls[0]
    assert (event_id, frame_number) == ("event-one", 2)
    assert metadata == {
        "selection_method": "best",
        "selected_motion_bounding_box_area": 1125,
        "total_event_frames_considered": 3,
    }


def test_auto_fire_uses_one_live_qualified_task_with_track_and_freshness_context(tmp_path: Path) -> None:
    config = classifier_config(
        tmp_path,
        auto_fire__enabled=True,
        manual_control__servo_enabled=True,
    )

    class Classifier:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int, dict[str, object]]] = []

        def set_paused(self, _paused: bool) -> None:
            return None

        def submit(
            self,
            event_id: str,
            _directory: Path,
            frame_number: int,
            _frame: np.ndarray,
            _box: tuple[int, int, int, int],
            **metadata: object,
        ) -> bool:
            self.calls.append((event_id, frame_number, metadata))
            return True

    class Recorder:
        def __init__(self) -> None:
            self.active: dict[int, object] = {}
            self.updates = 0

        def begin(self, track_id: int, *_args: object, **_kwargs: object) -> object:
            directory = tmp_path / "captures" / "events" / "event-live"
            directory.mkdir(parents=True, exist_ok=True)
            event = SimpleNamespace(event_id="event-live", directory=directory, start_timestamp="now")
            self.active[track_id] = event
            return event

        def update(self, *_args: object, **_kwargs: object) -> None:
            self.updates += 1

        def should_finish(self, *_args: object, **_kwargs: object) -> bool:
            return self.updates >= 2

        def finish(self, track_id: int, **_kwargs: object) -> dict[str, object]:
            event = self.active.pop(track_id)
            return {
                "event_id": event.event_id,
                "track_id": track_id,
                "snapshot_path": str(event.directory / "snapshot.jpg"),
                "clip_path": str(event.directory / "clip.avi"),
            }

    classifier = Classifier()
    motion = MotionProcessingService(
        SimpleNamespace(),
        config,
        classifier_service=classifier,  # type: ignore[arg-type]
    )
    motion._recorder = Recorder()  # type: ignore[assignment]
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    packet = SimpleNamespace(frame=frame, sequence=42)
    first = SimpleNamespace(
        track_id=7,
        newly_confirmed=True,
        foreground_pixels=100,
        provisional_category="small_animal_candidate",
        movement_attributes=("coherent_travel",),
        bounding_box=(20, 20, 30, 20),
        centroid=(35.2, 30.7),
        contour_area=600.0,
        confirmed=True,
        event_eligible=True,
    )
    second = SimpleNamespace(**{**first.__dict__, "newly_confirmed": False, "centroid": (36.0, 31.0)})

    motion._handle_events(packet, SimpleNamespace(groups=(first,)), lambda: frame, 10.0, 10.0)  # type: ignore[arg-type]
    motion._handle_events(packet, SimpleNamespace(groups=(second,)), lambda: frame, 10.1, 10.0)  # type: ignore[arg-type]
    motion._handle_events(packet, SimpleNamespace(groups=(second,)), lambda: frame, 10.2, 10.0)  # type: ignore[arg-type]

    assert len(classifier.calls) == 1
    event_id, frame_number, metadata = classifier.calls[0]
    assert (event_id, frame_number) == ("event-live", 1)
    assert metadata == {
        "selection_method": "qualified_live_target",
        "selected_motion_bounding_box_area": 600,
        "total_event_frames_considered": 1,
        "context": "auto_fire_live_event",
        "track_id": 7,
        "frame_sequence": 42,
        "target_pixel": (35, 31),
        "target_observed_monotonic": 10.0,
        "target_provisional_category": "small_animal_candidate",
        "target_confirmed": True,
        "target_event_eligible": True,
    }


def _auto_target(event_id: str = "event-live", track_id: int = 7) -> AutoFireTargetSnapshot:
    return AutoFireTargetSnapshot(
        event_id,
        track_id,
        10.0,
        35,
        31,
        (20, 20, 30, 20),
        120,
        80,
        True,
        True,
        "small_animal_candidate",
    )


def _motion_group(
    *,
    track_id: int = 7,
    centroid: tuple[float, float] = (42.0, 34.0),
    bounding_box: tuple[int, int, int, int] = (27, 24, 30, 20),
) -> SimpleNamespace:
    return SimpleNamespace(
        track_id=track_id,
        centroid=centroid,
        bounding_box=bounding_box,
        confirmed=True,
        event_eligible=True,
        provisional_category="small_animal_candidate",
    )


def _auto_motion(tmp_path: Path) -> MotionProcessingService:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = classifier_config(
        tmp_path,
        auto_fire__enabled=True,
        manual_control__servo_enabled=True,
    )
    return MotionProcessingService(SimpleNamespace(), config)  # type: ignore[arg-type]


def test_runtime_coasts_on_brief_loss_and_restores_only_reacquired_live_coordinate(tmp_path: Path) -> None:
    motion = _auto_motion(tmp_path)
    key = ("event-live", 7)
    motion._live_auto_targets[key] = _auto_target()

    motion._handle_missing_auto_target("event-live", 7, (), 10.1)
    association = motion._auto_fire_target("event-live", 7)
    assert isinstance(association, AutoFireTargetAssociation)
    assert association.state == "coasting"
    assert key not in motion._live_auto_targets

    reacquired = _motion_group(centroid=(68.0, 49.0), bounding_box=(53, 39, 30, 20))
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    motion._update_live_auto_target("event-live", reacquired, (reacquired,), frame, 10.5)

    current = motion._auto_fire_target("event-live", 7)
    assert isinstance(current, AutoFireTargetSnapshot)
    assert (current.pixel_x, current.pixel_y) == (68, 49)
    assert current.observed_monotonic == 10.5


def test_runtime_expires_track_after_grace_period(tmp_path: Path) -> None:
    motion = _auto_motion(tmp_path)
    motion._live_auto_targets[("event-live", 7)] = _auto_target()
    motion._handle_missing_auto_target("event-live", 7, (), 10.1)
    motion._handle_missing_auto_target("event-live", 7, (), 11.01)

    association = motion._auto_fire_target("event-live", 7)
    assert isinstance(association, AutoFireTargetAssociation)
    assert association.state == "reacquisition_timeout"


def test_runtime_rejects_wrong_identity_and_ambiguous_reacquisition(tmp_path: Path) -> None:
    wrong = _auto_motion(tmp_path / "wrong")
    wrong._live_auto_targets[("event-live", 7)] = _auto_target()
    wrong._handle_missing_auto_target("event-live", 7, (), 10.1)
    other = _motion_group(track_id=8, centroid=(40.0, 33.0))
    wrong._handle_missing_auto_target("event-live", 7, (other,), 10.2)
    wrong_state = wrong._auto_fire_target("event-live", 7)
    assert isinstance(wrong_state, AutoFireTargetAssociation)
    assert wrong_state.state == "reacquisition_incompatible"

    ambiguous = _auto_motion(tmp_path / "ambiguous")
    ambiguous._live_auto_targets[("event-live", 7)] = _auto_target()
    ambiguous._handle_missing_auto_target("event-live", 7, (), 10.1)
    first = _motion_group(track_id=7, centroid=(40.0, 33.0))
    second = _motion_group(track_id=8, centroid=(55.0, 40.0))
    ambiguous._handle_missing_auto_target("event-live", 7, (first, second), 10.2)
    ambiguous_state = ambiguous._auto_fire_target("event-live", 7)
    assert isinstance(ambiguous_state, AutoFireTargetAssociation)
    assert ambiguous_state.state == "reacquisition_ambiguous"


def test_runtime_records_numeric_reacquisition_gate_diagnostics(tmp_path: Path) -> None:
    motion = _auto_motion(tmp_path)
    recorded: list[tuple[int, dict[str, object]]] = []
    motion._recorder = SimpleNamespace(
        record_reacquisition_diagnostic=lambda track_id, diagnostic: recorded.append((track_id, diagnostic))
    )
    motion._live_auto_targets[("event-live", 7)] = _auto_target()
    far = _motion_group(
        track_id=8,
        centroid=(155.0, 31.0),
        bounding_box=(140, 21, 30, 20),
    )

    motion._handle_missing_auto_target("event-live", 7, (far,), 10.1)

    assert len(recorded) == 1
    track_id, diagnostic = recorded[0]
    assert track_id == 7
    assert diagnostic["reference_mode"] == "last_observed_centroid"
    assert diagnostic["decision"] == "coasting"
    candidate_diagnostic = diagnostic["candidates"][0]  # type: ignore[index]
    assert candidate_diagnostic["track_id"] == 8
    assert candidate_diagnostic["distance_from_last_centroid_pixels"] == 120.0
    assert candidate_diagnostic["area_ratio"] == 1.0
    assert candidate_diagnostic["rejection_reason"] == "centroid_distance"


def test_classifier_rejects_new_work_while_night_mode_is_paused(tmp_path: Path) -> None:
    config = classifier_config(tmp_path)
    classifier = EventClassifier(config.classifier, ClassifierEvidenceStore(config))
    classifier.set_paused(True)

    queued = classifier.submit(
        "night-event",
        tmp_path / "captures" / "events" / "night-event",
        1,
        np.zeros((80, 120, 3), dtype=np.uint8),
        (20, 20, 30, 20),
    )

    assert queued is False
    assert classifier.status().paused is True
    assert classifier.status().queue_depth == 0


def test_classifier_review_page_serves_input_and_requires_token_for_decision(tmp_path: Path) -> None:
    config = classifier_config(tmp_path)
    store = ClassifierEvidenceStore(config)
    store.save_classification(
        task(tmp_path, "review-event"),
        [ClassifierDetection("person", 0.4, (1, 1, 10, 10))],
        100.0,
        "test-model",
    )
    store.save_classification(task(tmp_path, "unknown-event"), [], 100.0, "test-model")
    store.save_classification(task(tmp_path, "false-event"), [], 100.0, "test-model")

    camera = SimpleNamespace(
        start=lambda: None,
        status=lambda: CameraStatus(False, 1280, 720, 0.0, "offline"),
    )
    vision = SimpleNamespace(
        start=lambda: None,
        status=lambda: VisionStatus(
            "READY", True, 10.0, 0, 0, 1, 1, 1, 0, 1,
            None, None, None, None, None, True, True,
        ),
        recent_events=lambda: [],
    )
    app = create_app(
        app_config=config,
        camera_service=camera,  # type: ignore[arg-type]
        vision_service=vision,  # type: ignore[arg-type]
        start_camera=False,
        start_vision=False,
        temperature_reader=lambda: 45.0,
    )
    app.config.update(TESTING=True)
    client = app.test_client()

    page = client.get("/classifier-review")
    assert page.status_code == 200 and b"review-event" in page.data and b"Known-class possibility" in page.data
    assert b"Download clean full frame" in page.data
    assert b'<option value="person" class="classifier-label-option" selected>Person (model guess)</option>' in page.data
    assert b'<option value="dog" class="classifier-label-option">Dog</option>' in page.data
    assert b'<option value="squirrel">Squirrel</option>' in page.data
    assert b"Blue = built into the current classifier" in page.data
    assert b'name="custom_label"' in page.data
    assert b'action="/classifier-review/bulk"' in page.data
    assert client.get("/classifier-files/review-event").status_code == 200
    assert client.post("/classifier-review/review-event/approve").status_code == 403
    token_match = re.search(rb'name="review_token" value="([^"]+)"', page.data)
    assert token_match is not None
    token = token_match.group(1).decode()
    assert client.post(
        "/classifier-review/review-event/approve",
        data={"review_token": token},
    ).status_code == 400
    assert client.post(
        "/classifier-review/review-event/approve",
        data={"review_token": token, "approval_label": "?"},
    ).status_code == 400
    confirmed = client.post(
        "/classifier-review/review-event/confirm-model",
        data={"review_token": token, "return_state": "review"},
    )
    assert confirmed.status_code == 302
    assert "state=review" in confirmed.location
    approved = client.post(
        "/classifier-review/unknown-event/approve",
        data={
            "review_token": token,
            "return_state": "unknown",
            "approval_label": "car",
            "custom_label": "Eastern Gray Squirrel",
        },
    )
    assert approved.status_code == 302
    assert "state=unknown" in approved.location
    known_page = client.get("/classifier-review?state=known").data
    assert b"review-event" in known_page and b"Person" in known_page
    assert b"unknown-event" in known_page and b"Eastern Gray Squirrel" in known_page
    assert b"Training manifest" in known_page and b"training samples" in known_page
    assert client.get("/files/training-dataset/manifest.jsonl").status_code == 200

    response = client.post(
        "/classifier-review/false-event/false-positive",
        data={"review_token": token, "return_state": "unknown"},
    )

    assert response.status_code == 302
    assert "state=unknown" in response.location
    false_positive_page = client.get("/classifier-review?state=false_positive").data
    assert b"false-event" in false_positive_page and b"False Positive" in false_positive_page
    assert store.training_summary()["labels"] == {
        "background_or_false_positive": 1,
        "eastern_gray_squirrel": 1,
        "person": 1,
    }


def test_classifier_review_bulk_actions_keep_current_view_and_use_each_model_guess(tmp_path: Path) -> None:
    config = classifier_config(tmp_path)
    store = ClassifierEvidenceStore(config)
    store.save_classification(
        task(tmp_path, "dog-event"),
        [ClassifierDetection("dog", 0.8, (1, 1, 10, 10))],
        100.0,
        "test-model",
    )
    store.save_classification(
        task(tmp_path, "bird-event"),
        [ClassifierDetection("bird", 0.7, (1, 1, 10, 10))],
        100.0,
        "test-model",
    )

    camera = SimpleNamespace(
        start=lambda: None,
        status=lambda: CameraStatus(False, 1280, 720, 0.0, "offline"),
    )
    vision = SimpleNamespace(
        start=lambda: None,
        status=lambda: VisionStatus(
            "READY", True, 10.0, 0, 0, 1, 1, 1, 0, 1,
            None, None, None, None, None, True, True,
        ),
        recent_events=lambda: [],
    )
    app = create_app(
        app_config=config,
        camera_service=camera,  # type: ignore[arg-type]
        vision_service=vision,  # type: ignore[arg-type]
        start_camera=False,
        start_vision=False,
        temperature_reader=lambda: 45.0,
    )
    app.config.update(TESTING=True)
    client = app.test_client()

    page = client.get("/classifier-review?state=unknown")
    token_match = re.search(rb'name="review_token" value="([^"]+)"', page.data)
    assert token_match is not None
    token = token_match.group(1).decode()
    assert page.data.count(b'name="item_ids"') == 2

    no_selection = client.post(
        "/classifier-review/bulk",
        data={"review_token": token, "return_state": "unknown", "bulk_action": "unknown"},
    )
    assert no_selection.status_code == 302
    assert "state=unknown" in no_selection.location

    response = client.post(
        "/classifier-review/bulk",
        data={
            "review_token": token,
            "return_state": "unknown",
            "bulk_action": "confirm-model",
            "item_ids": ["dog-event", "bird-event"],
        },
    )

    assert response.status_code == 302
    assert "state=unknown" in response.location
    assert store.get_record("dog-event")["training_label"] == "dog"
    assert store.get_record("bird-event")["training_label"] == "bird"
    assert store.training_summary()["labels"] == {"bird": 1, "dog": 1}


def test_classifier_review_json_api_lists_items_and_accepts_decisions(tmp_path: Path) -> None:
    config = classifier_config(tmp_path)
    store = ClassifierEvidenceStore(config)
    store.save_classification(
        task(tmp_path, "review-event"),
        [ClassifierDetection("person", 0.4, (1, 1, 10, 10))],
        100.0,
        "test-model",
    )
    store.save_classification(task(tmp_path, "unknown-event"), [], 100.0, "test-model")

    camera = SimpleNamespace(
        start=lambda: None,
        status=lambda: CameraStatus(False, 1280, 720, 0.0, "offline"),
    )
    vision = SimpleNamespace(
        start=lambda: None,
        status=lambda: VisionStatus(
            "READY", True, 10.0, 0, 0, 1, 1, 1, 0, 1,
            None, None, None, None, None, True, True,
        ),
        recent_events=lambda: [],
    )
    app = create_app(
        app_config=config,
        camera_service=camera,  # type: ignore[arg-type]
        vision_service=vision,  # type: ignore[arg-type]
        start_camera=False,
        start_vision=False,
        temperature_reader=lambda: 45.0,
    )
    app.config.update(TESTING=True)
    client = app.test_client()

    listing = client.get("/api/classifier-review?state=review")
    assert listing.status_code == 200
    payload = listing.json
    assert payload["state"] == "review"
    assert payload["counts"]["review"] == 1
    assert payload["counts"]["unknown"] == 1
    item = payload["items"][0]
    assert item["item_id"] == "review-event"
    assert item["event_id"] == "review-event"
    assert item["image_url"].endswith("/classifier-files/review-event")
    assert item["top_label"] == "person"

    assert client.get("/api/classifier-review?state=bogus").status_code == 404

    page = client.get("/")
    assert b'id="queue-list"' in page.data and b"review-event" in page.data
    token_match = re.search(rb'reviewToken: "([^"]+)"', page.data)
    assert token_match is not None
    token = token_match.group(1).decode()

    decided = client.post(
        "/classifier-review/review-event/approve",
        data={"review_token": token, "approval_label": "car", "format": "json"},
    )
    assert decided.status_code == 200
    assert decided.json["ok"] is True
    assert decided.json["classification_status"] == "known"
    assert decided.json["display_label"] == "Car"
    assert decided.json["training_label"] == "car"
    assert decided.json["training_dataset_status"] == "included"

    updated = client.get("/api/classifier-review?state=review").json
    assert updated["counts"]["review"] == 0
    assert updated["counts"]["known"] == 1
    assert store.training_summary()["labels"] == {"car": 1}


def test_classifier_image_route_resolves_pi_style_relative_capture_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = classifier_config(tmp_path)
    config = replace(
        config,
        camera=replace(config.camera, output_directory=Path("captures")),
        classifier=replace(config.classifier, evidence_directory=Path("captures/classifier")),
        logging=replace(config.logging, directory=Path("captures/logs")),
    )
    monkeypatch.chdir(tmp_path)
    relative_task = ClassifierTask(
        event_id="relative-event",
        event_directory=Path("captures/events/2026-07-16/relative-event"),
        frame_number=1,
        image=np.full((24, 32, 3), 120, dtype=np.uint8),
        source_bounding_box=(1, 2, 20, 18),
        crop_bounding_box=(0, 0, 24, 22),
        submitted_at="2026-07-16T18:00:00-04:00",
    )
    store = ClassifierEvidenceStore(config)
    store.save_classification(relative_task, [], 100.0, "test-model")
    assert store.input_path("relative-event").is_absolute()

    camera = SimpleNamespace(
        start=lambda: None,
        status=lambda: CameraStatus(False, 1280, 720, 0.0, "offline"),
    )
    vision = SimpleNamespace(
        start=lambda: None,
        status=lambda: VisionStatus(
            "READY", True, 10.0, 0, 0, 1, 1, 1, 0, 1,
            None, None, None, None, None, True, True,
        ),
        recent_events=lambda: [],
    )
    app = create_app(
        app_config=config,
        camera_service=camera,  # type: ignore[arg-type]
        vision_service=vision,  # type: ignore[arg-type]
        start_camera=False,
        start_vision=False,
        temperature_reader=lambda: 45.0,
    )
    app.config.update(TESTING=True)

    response = app.test_client().get("/classifier-files/relative-event")
    assert response.status_code == 200 and response.content_type == "image/jpeg"


def test_classifier_config_defaults_to_best_and_uses_legacy_frame_as_fallback(tmp_path: Path) -> None:
    path = write_test_config(tmp_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["classifier"].pop("frame_selection")
    raw["classifier"].pop("fallback_event_frame_number")
    raw["classifier"]["event_frame_number"] = 3
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    config = load_config(path)

    assert config.classifier.frame_selection == "best"
    assert config.classifier.fallback_event_frame_number == 3
