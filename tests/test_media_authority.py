import json
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from squirrel_shooter.classifier import ClassifierEvidenceStore
from squirrel_shooter.frame_selection import BestEventFrameSelector
from squirrel_shooter.media_roles import MediaRole, media_role
from squirrel_shooter.motion_runtime import MotionProcessingService
from squirrel_shooter.dataset_inventory import build_inventory
from test_classifier import classifier_config, task
from test_pi_dataset import snapshot_fixture


@pytest.mark.parametrize("method,role", [("middle_fallback", "clean_authoritative"), ("configured_fallback", "unknown_legacy"), ("best", "annotated_review")])
def test_unverified_pixels_cannot_be_promoted_by_review(tmp_path, method, role):
    store = ClassifierEvidenceStore(classifier_config(tmp_path))
    value = replace(task(tmp_path, "unverified"), selection_method=method, source_media_role=role)
    saved = store.save_classification(value, [], 1, "fake")
    assert saved["input_image_role"] == "unknown_legacy"
    reviewed = store.set_label("unverified", "squirrel")
    assert reviewed["human_verified"] and reviewed["human_label"] == "squirrel"
    assert reviewed["training_dataset_status"] == "excluded_unverified_media"
    assert not list(store.training_samples_root.glob("*/image.jpg"))
    assert store._training_samples(eligible_only=True) == []


def test_old_unannotated_string_is_not_clean_attestation(tmp_path):
    store = ClassifierEvidenceStore(classifier_config(tmp_path))
    saved = store.save_classification(task(tmp_path, "legacy"), [], 1, "fake")
    path = store._record_path("legacy")
    saved.pop("source_pixel_provenance")
    saved.pop("source_media_role")
    path.write_text(json.dumps(saved), encoding="utf-8")
    result = store.set_label("legacy", "squirrel")
    assert result["training_dataset_status"] == "excluded_unverified_media"
    assert media_role("original-frame.jpg", saved) == MediaRole.UNKNOWN


def test_motion_never_calls_annotated_clip_decoder(tmp_path, monkeypatch):
    config = classifier_config(tmp_path)
    worker = SimpleNamespace(set_paused=lambda _: None,
                             submit=lambda *a, **k: pytest.fail("No clean source available"))
    motion = MotionProcessingService(SimpleNamespace(), config, classifier_service=worker)
    selector = BestEventFrameSelector(fallback_frame_number=1, minimum_motion_area=1)
    selector.consider(1, None, None, None)
    motion._classifier_selectors["old"] = selector
    monkeypatch.setattr(motion, "_load_event_clip_frame", lambda *a: pytest.fail("Annotated fallback used"))
    assert not motion._submit_completed_event({"event_id": "old", "clip_path": "clip.avi"})


def test_decoder_fallback_is_unknown_and_raw_first_is_clean():
    selector = BestEventFrameSelector(fallback_frame_number=50, minimum_motion_area=10000)
    selector.consider(1, np.zeros((12, 16, 3), np.uint8), None, None)
    assert selector.select(lambda _: np.ones((12, 16, 3), np.uint8)).media_role == "unknown_legacy"
    assert selector.select().media_role == "clean_authoritative"


def test_dataset_reports_roles_and_excludes_unverified_training(tmp_path):
    snapshot = snapshot_fixture(tmp_path)
    sample = next(snapshot.glob("raw/project/captures/training-dataset/samples/event-one/sample.json"))
    data = json.loads(sample.read_text())
    data.pop("source_pixel_provenance")
    sample.write_text(json.dumps(data))
    inventory = build_inventory(snapshot)
    assert not inventory["human_verified_training"]
    assert media_role("clip.avi", {"file_role": "clean_authoritative"}) == "annotated_review"
    assert media_role("snapshot.jpg", {"file_role": "clean_authoritative"}) == "annotated_review"


def test_inventory_joins_clean_segments_to_one_session_without_training_promotion(tmp_path):
    snapshot = snapshot_fixture(tmp_path)
    directory = snapshot / "raw/project/captures/recordings" / ("a" * 32)
    directory.mkdir(parents=True)
    segments = []
    for index, status in enumerate(("complete", "error")):
        name = f"segment-{index:04d}.avi"
        (directory / name).write_bytes(b"retained fixture, validation not rerun by inventory")
        segments.append({"file": name, "status": status, "file_role": "clean_authoritative", "sha256": "b" * 64})
    (directory / "session.json").write_text(json.dumps({"session_id": directory.name, "segments": segments}))
    build_inventory(snapshot)
    rows = json.loads((snapshot / "inventory/review_index.json").read_text())
    recordings = [r for r in rows if r["record_kind"] == "clean_recording_segment"]
    assert len(recordings) == 2
    assert {r["event_id"] for r in recordings} == {directory.name}
    assert {r["video_media_role"] for r in recordings} == {"clean_authoritative", "unknown_legacy"}
    assert not any(r["training_eligible"] for r in recordings)
    assert media_role("unrelated.avi", segments[0]) == "unknown_legacy"


def test_review_api_explicit_roles_override_unverified_historical_strings(tmp_path):
    from squirrel_shooter.web_dashboard import create_app, _review_item_payload
    from conftest import write_test_config
    from test_web_dashboard import OfflineCameraService, StaticVisionService
    app = create_app(write_test_config(tmp_path), camera_service=OfflineCameraService(),
                     vision_service=StaticVisionService(), manual_control_service=SimpleNamespace(status=lambda: {}),
                     start_camera=False, start_vision=False)
    with app.test_request_context():
        result = _review_item_payload({"item_id": "legacy", "input_image_role": "unannotated_target_crop"})
    assert result["image_media_role"] == "unknown_legacy"
    assert result["original_frame_media_role"] == "unknown_legacy"
    assert result["snapshot_media_role"] == result["clip_media_role"] == "annotated_review"
