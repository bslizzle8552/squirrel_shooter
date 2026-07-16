from __future__ import annotations

import hashlib
import io
import json
import re
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import write_test_config
from squirrel_shooter.camera_service import CameraStatus
from squirrel_shooter.classifier import (
    ClassifierDetection,
    ClassifierEvidenceStore,
    ClassifierTask,
    EventClassifier,
    MobileNetSSDDetector,
)
from squirrel_shooter.classifier_setup import ModelFile, install_model
from squirrel_shooter.config import ConfigError, load_config
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
    assert unknown["classification_status"] == "unknown" and unknown["display_label"] == "Unknown"
    assert other_class["classification_status"] == "unknown" and other_class["model_suggestion"] == "dog"
    assert review["classification_status"] == "review" and review["review_suggestion_label"] == "car"
    for classifier_task in (person_task, unknown_task, other_class_task, review_task):
        assert (classifier_task.event_directory / "classifier-input.jpg").is_file()
        assert (classifier_task.event_directory / "classification.json").is_file()
    assert store.counts() == {"known": 1, "unknown": 2, "review": 1, "errors": 0, "false_positive": 0}
    audit = config.logging.directory / config.classifier.audit_log_filename
    assert [json.loads(line)["action"] for line in audit.read_text(encoding="utf-8").splitlines()] == [
        "classified",
        "classified",
        "classified",
        "classified",
    ]

    with pytest.raises(ValueError, match="car or person"):
        store.review("review-event", "approve")
    reviewed = store.review("review-event", "approve", "car")

    assert reviewed["classification_status"] == "known" and reviewed["display_label"] == "Car"
    assert reviewed["human_label"] == "car" and reviewed["label_source"] == "human"
    assert json.loads(audit.read_text(encoding="utf-8").splitlines()[-1])["action"] == "human_labeled"

    corrected = store.review("person-event", "unknown")
    assert corrected["classification_status"] == "unknown" and corrected["display_label"] == "Unknown"
    assert corrected["label_source"] == "human" and corrected["human_label"] == "unknown"


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


@pytest.mark.parametrize("configured_frame", [1, 2])
def test_motion_submits_only_configured_event_frame_once(tmp_path: Path, configured_frame: int) -> None:
    config = classifier_config(tmp_path, classifier__event_frame_number=configured_frame)

    class Classifier:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []

        def submit(self, event_id: str, _directory: Path, frame_number: int, _frame: np.ndarray, _box: tuple[int, int, int, int]) -> bool:
            self.calls.append((event_id, frame_number))
            return True

    class Recorder:
        def __init__(self) -> None:
            self.active: dict[int, object] = {}

        def begin(self, track_id: int, *_args: object, **_kwargs: object) -> object:
            directory = tmp_path / "captures" / "events" / "event-one"
            directory.mkdir(parents=True, exist_ok=True)
            event = SimpleNamespace(event_id="event-one", directory=directory, start_timestamp="now")
            self.active[track_id] = event
            return event

        def update(self, *_args: object, **_kwargs: object) -> None:
            return None

        def should_finish(self, *_args: object, **_kwargs: object) -> bool:
            return False

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
    )
    later = SimpleNamespace(**{**first.__dict__, "newly_confirmed": False})

    motion._handle_events(packet, SimpleNamespace(groups=(first,)), frame, 1.0, 10.0)  # type: ignore[arg-type]
    motion._handle_events(packet, SimpleNamespace(groups=(later,)), frame, 1.1, 10.0)  # type: ignore[arg-type]
    motion._handle_events(packet, SimpleNamespace(groups=(later,)), frame, 1.2, 10.0)  # type: ignore[arg-type]

    assert classifier.calls == [("event-one", configured_frame)]


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
        data={"review_token": token, "approval_label": "squirrel"},
    ).status_code == 400
    approved = client.post(
        "/classifier-review/review-event/approve",
        data={"review_token": token, "approval_label": "person"},
    )
    assert approved.status_code == 302
    known_page = client.get("/classifier-review?state=known").data
    assert b"review-event" in known_page and b"Person" in known_page and b"human" in known_page

    response = client.post(
        "/classifier-review/unknown-event/false-positive",
        data={"review_token": token},
    )

    assert response.status_code == 302
    false_positive_page = client.get("/classifier-review?state=false_positive").data
    assert b"unknown-event" in false_positive_page and b"False Positive" in false_positive_page


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


def test_classifier_config_rejects_frame_outside_first_two(tmp_path: Path) -> None:
    path = write_test_config(tmp_path, classifier__event_frame_number=3)
    with pytest.raises(ConfigError, match="event_frame_number must be 1 or 2"):
        load_config(path)
