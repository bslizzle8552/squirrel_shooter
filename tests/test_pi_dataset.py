from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import yaml

from squirrel_shooter.dataset_inventory import build_inventory
from squirrel_shooter.pi_data_pull import (
    PiCollector,
    RemoteFile,
    choose_incremental_sources,
    collect_snapshot,
    parse_remote_manifest,
    sanitize_configuration,
)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def snapshot_fixture(tmp_path: Path) -> Path:
    snapshot = tmp_path / "pi_dataset_2026-08-16_210000"
    event = snapshot / "raw" / "project" / "captures" / "events" / "2026-08-16" / "event-one"
    event.mkdir(parents=True)
    write_json(
        event / "event.json",
        {
            "status": "complete",
            "event_id": "event-one",
            "track_id": 7,
            "capture_method": "automatic_motion_event",
            "start_timestamp": "2026-08-16T12:00:00-04:00",
            "measured_camera_fps": 9.5,
            "components": [[{"foreground_pixel_area": 2500}]],
            "group_samples": [{"combined_foreground_pixel_area": 2500}],
        },
    )
    write_json(
        event / "classification.json",
        {
            "event_id": "event-one",
            "track_id": 7,
            "top_label": "dog",
            "top_confidence": 0.91,
            "detections": [
                {"label": "dog", "confidence": 0.91, "bounding_box": {"x": 10, "y": 20, "width": 30, "height": 40}},
                {"label": "bird", "confidence": 0.31, "bounding_box": {"x": 40, "y": 10, "width": 20, "height": 20}},
            ],
            "classification_status": "known",
            "outcome": "auto_labeled",
            "auto_accepted": True,
            "target_pixel": {"x": 320, "y": 240},
            "human_label": "squirrel",
            "human_verified": True,
            "training_label": "squirrel",
        },
    )
    (event / "classifier-input.jpg").write_bytes(b"crop")
    (event / "original-frame.jpg").write_bytes(b"frame")
    (event / "clip.avi").write_bytes(b"video")

    write_json(
        snapshot / "raw" / "project" / "captures" / "events" / "2026-08-16" / "auto-fire-one" / "event.json",
        {
            "status": "complete",
            "event_id": "auto-fire-one",
            "capture_method": "auto_fire",
            "start_timestamp": "2026-08-16T12:00:03-04:00",
            "source_event_id": "event-one",
            "classifier_label": "dog",
            "classifier_confidence": 0.91,
            "target_pixel_x": 320,
            "target_pixel_y": 240,
        },
    )

    sample = snapshot / "raw" / "project" / "captures" / "training-dataset" / "samples" / "event-one"
    sample.mkdir(parents=True)
    write_json(
        sample / "sample.json",
        {
            "sample_id": "event-one",
            "event_id": "event-one",
            "label": "squirrel",
            "human_verified": True,
            "training_eligible": True,
            "source_media_role": "clean_authoritative",
            "source_pixel_provenance": "shared_camera_raw_v1",
            "labeled_at": "2026-08-16T12:05:00-04:00",
            "source": {
                "capture_method": "automatic_motion_event",
                "model_suggestion": "dog",
                "top_label": "dog",
                "top_confidence": 0.91,
                "detections": [{"label": "dog", "confidence": 0.91}],
                "source_bounding_box": {"x": 100, "y": 200, "width": 100, "height": 80},
                "source_camera": {"actual_width": 1280, "actual_height": 720, "camera_mode_if_known": "day"},
                "motion_category": "small_animal_candidate",
            },
        },
    )
    (sample / "image.jpg").write_bytes(b"verified crop")
    (sample / "original-frame.jpg").write_bytes(b"verified frame")
    excluded = snapshot / "raw" / "project" / "captures" / "training-dataset" / "samples" / "excluded-old"
    excluded.mkdir(parents=True)
    write_json(
        excluded / "sample.json",
        {
            "sample_id": "excluded-old",
            "event_id": "excluded-old",
            "label": "rabbit",
            "human_verified": True,
            "training_eligible": False,
            "exclusion_reason": "human_marked_unknown",
        },
    )
    (excluded / "image.jpg").write_bytes(b"excluded crop")

    logs = snapshot / "raw" / "project" / "captures" / "logs"
    logs.mkdir(parents=True)
    (logs / "rejections.jsonl").write_text(
        json.dumps(
            {
                "timestamp": "2026-08-16T12:10:00-04:00",
                "reason": "global_motion",
                "raw_foreground_percent": 56.0,
                "cleaned_foreground_percent": 48.0,
                "candidate_region_percent": 45.0,
                "inclusion_zone_motion_percent": 50.0,
                "disconnected_regions": 30,
                "measured_camera_fps": 8.2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    write_json(
        logs / "sessions" / "session-one.json",
        {
            "raw_contours": 500,
            "grouped_candidates": 40,
            "average_measured_fps": 9.0,
            "rejected_by_filter": {"localized_lighting_change": 12},
            "retention_action_count": 2,
            "retention_actions": [
                {"event_id": "removed-one", "reason": "maximum_age"},
                {"event_id": "removed-two", "reason": "maximum_storage"},
            ],
        },
    )
    (logs / "squirrel-shooter-20260816-120000.jsonl").write_text(
        json.dumps(
            {
                "event": "runtime_performance",
                "capture_fps": 14.8,
                "detection_fps": 9.5,
                "classifier_queue_depth": 1,
                "classifier_inference_fps": 0.2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (logs / "events.jsonl").write_text(
        json.dumps(
            {
                "event_id": "historical-event",
                "track_id": 22,
                "capture_method": "automatic_motion_event",
                "start_timestamp": "2026-07-01T08:00:00-04:00",
                "provisional_category": "small_animal_candidate",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (logs / "classifier.jsonl").write_text(
        json.dumps(
            {
                "action": "classified",
                "event_id": "historical-event",
                "classifier_timestamp": "2026-07-01T08:00:02-04:00",
                "top_label": "bird",
                "top_confidence": 0.44,
                "detections": [{"label": "bird", "confidence": 0.44}],
                "classification_status": "unknown",
                "outcome": "unknown",
            }
        )
        + "\n"
        + json.dumps(
            {
                "action": "human_labeled",
                "event_id": "historical-event",
                "classifier_timestamp": "2026-07-01T08:00:02-04:00",
                "top_label": "bird",
                "top_confidence": 0.44,
                "classification_status": "known",
                "human_label": "squirrel",
                "human_verified": True,
                "training_label": "squirrel",
                "training_dataset_status": "included",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    write_json(
        snapshot / "raw" / "project" / "captures" / "classifier" / "unknown" / "unknown.json",
        {
            "event_id": "unknown-event",
            "classifier_timestamp": "2026-08-16T12:20:00-04:00",
            "classification_status": "unknown",
            "outcome": "unknown",
            "detections": [],
        },
    )
    config = snapshot / "raw" / "config" / "default.sanitized.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(yaml.safe_dump({"classifier": {"auto_accept_confidence": 0.60}}), encoding="utf-8")
    return snapshot


def test_parse_remote_manifest_rejects_traversal_and_preserves_metadata() -> None:
    payload = b"captures/events/a/event.json\x00123\x001723840000.5000000000\x00"
    assert parse_remote_manifest(payload) == [
        RemoteFile("captures/events/a/event.json", 123, "1723840000.5000000000")
    ]

    unsafe = b"../secret\x001\x001.0\x00"
    try:
        parse_remote_manifest(unsafe)
    except ValueError as exc:
        assert "Unsafe" in str(exc)
    else:
        raise AssertionError("path traversal should be rejected")

    unsafe_newline = b"captures/bad\nname.jpg\x001\x001.0\x00"
    try:
        parse_remote_manifest(unsafe_newline)
    except ValueError as exc:
        assert "Unsafe" in str(exc)
    else:
        raise AssertionError("newlines cannot be represented safely in an SFTP batch")


def test_configuration_sanitizer_redacts_secrets_but_preserves_settings() -> None:
    sanitized, redactions = sanitize_configuration(
        {
            "camera": {"requested_width": 1280},
            "remote": {"api_key": "secret", "nested": [{"password": "secret-two"}]},
        }
    )

    assert sanitized["camera"]["requested_width"] == 1280
    assert sanitized["remote"]["api_key"] == "<REDACTED>"
    assert sanitized["remote"]["nested"][0]["password"] == "<REDACTED>"
    assert redactions == ["remote.api_key", "remote.nested.0.password"]


def test_incremental_plan_reuses_only_exact_size_and_mtime_matches(tmp_path: Path) -> None:
    previous = tmp_path / "pi_dataset_old"
    source = previous / "raw" / "project" / "captures" / "events" / "old.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"same")
    manifest = {
        "files": [
            {"relative_path": "captures/events/old.json", "size": 4, "mtime_epoch": "1.0"},
            {"relative_path": "captures/events/changed.json", "size": 3, "mtime_epoch": "1.0"},
        ]
    }
    manifest_path = previous / "inventory" / "remote_files.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    files = [
        RemoteFile("captures/events/old.json", 4, "1.0"),
        RemoteFile("captures/events/changed.json", 4, "2.0"),
        RemoteFile("captures/events/new.json", 5, "3.0"),
    ]

    reusable, transfer = choose_incremental_sources(files, previous)

    assert reusable == {"captures/events/old.json": source}
    assert [item.relative_path for item in transfer] == ["captures/events/changed.json", "captures/events/new.json"]


def test_sftp_batch_keeps_password_authentication_enabled(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    calls: list[tuple[list[str], dict[str, object], Path, str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        batch_path = Path(command[command.index("-b") + 1])
        calls.append((command, kwargs, batch_path, batch_path.read_text(encoding="utf-8")))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("squirrel_shooter.pi_data_pull.subprocess.run", fake_run)
    collector = PiCollector(
        host="100.110.85.40",
        user="bslizzle8552",
        remote_root="/home/bslizzle8552/squirrel_shooter",
    )
    destination = tmp_path / "raw" / "project"

    collector.transfer(
        [RemoteFile("captures/events/example/snapshot.jpg", 123, "1.0")],
        destination,
    )

    assert len(calls) == 1
    command, kwargs, batch_path, batch = calls[0]
    assert command.index("-oBatchMode=no") < command.index("-b")
    assert "-q" not in command
    assert command[-1] == "bslizzle8552@100.110.85.40"
    assert kwargs == {"check": True, "timeout": None}
    assert batch == (
        'get -p "/home/bslizzle8552/squirrel_shooter/captures/events/example/snapshot.jpg" '
        f'"{(destination / "captures/events/example/snapshot.jpg").resolve().as_posix()}"\n'
    )
    assert not batch_path.exists()


def test_inventory_keeps_predictions_separate_from_human_truth(tmp_path: Path) -> None:
    snapshot = snapshot_fixture(tmp_path)

    inventory = build_inventory(snapshot)

    assert inventory["overall"]["event_records"] == 3
    assert inventory["overall"]["event_metadata_files"] == 2
    assert inventory["overall"]["orphan_events_recovered_from_logs"] == 1
    assert inventory["overall"]["event_log_records"] == 1
    assert inventory["overall"]["classifier_records"] == 3
    assert inventory["overall"]["classifier_audit_records"] == 2
    assert inventory["overall"]["orphan_classifier_records_recovered_from_audit"] == 1
    assert inventory["classifier_audit_actions"] == {"classified": 1, "human_labeled": 1}
    assert inventory["retention_evidence"]["reported_action_count_in_retained_sessions"] == 2
    assert inventory["retention_evidence"]["retained_action_reasons"] == {
        "maximum_age": 1,
        "maximum_storage": 1,
    }
    assert inventory["predicted_class_statistics"]["dog"]["classified_records"] == 1
    assert inventory["predicted_class_statistics"]["dog"]["unique_events"] == 1
    assert inventory["human_verified_training"]["squirrel"] == {
        "raw_samples": 1,
        "independent_event_groups": 1,
    }
    assert inventory["discovered_predicted_labels"] == ["bird", "dog"]
    assert inventory["discovered_training_labels"] == ["squirrel"]
    assert "rabbit" not in inventory["human_verified_training"]
    assert inventory["unknown_negative_inventory"]["unknown_classifications"] == 1
    assert inventory["unknown_negative_inventory"]["no_detection_records"] == 1
    with (snapshot / "inventory" / "review_index.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    event_row = next(row for row in rows if row["record_kind"] == "event")
    assert event_row["predicted_label"] == "dog"
    assert event_row["human_label"] == "squirrel"
    assert event_row["shot_state"] == "automatic_shot"
    assert json.loads(event_row["detections_json"])[1]["label"] == "bird"
    shot_row = next(row for row in rows if row["event_id"] == "auto-fire-one")
    assert shot_row["predicted_label"] == "dog"
    assert shot_row["accepted_rejected_state"] == "accepted"
    historical = next(row for row in rows if row["event_id"] == "historical-event")
    assert historical["record_kind"] == "event_log"
    assert historical["predicted_label"] == "bird"
    assert historical["confidence"] == "0.44"
    assert historical["human_label"] == "squirrel"
    excluded_row = next(row for row in rows if row["event_id"] == "excluded-old")
    assert excluded_row["training_eligible"] == "False"
    progress = (snapshot / "inventory" / "classifier_progress.md").read_text(encoding="utf-8")
    assert "predictions are preserved as metadata only" in progress
    assert "Day Night: day (1)" in progress
    wind = (snapshot / "inventory" / "wind_shadow_diagnostic.md").read_text(encoding="utf-8")
    assert "Frame timing/FPS | **AVAILABLE NOW**" in wind
    assert "Classifier queue/load | **AVAILABLE NOW**" in wind
    assert "Dropped/ignored candidates | **PARTIALLY AVAILABLE**" in wind


def test_pi_service_context_and_journal_use_the_user_service_manager(tmp_path: Path) -> None:
    class RecordingCollector(PiCollector):
        def __init__(self) -> None:
            super().__init__(host="pi", user="owner", remote_root="/home/owner/squirrel_shooter")
            self.commands: list[str] = []

        def _run_ssh(self, remote_command: str, *, text: bool = False):  # type: ignore[no-untyped-def]
            self.commands.append(remote_command)

            class Result:
                stdout = "" if text else b""

            return Result()

    collector = RecordingCollector()
    collector.fetch_context()
    collector.fetch_journal(tmp_path / "journal.txt", "30 days ago")

    assert "systemctl --user is-active" in collector.commands[0]
    assert "systemctl --user is-enabled" in collector.commands[0]
    assert collector.commands[1].startswith("journalctl --user -u ")


def test_collection_orchestration_uses_read_only_collector_and_runs_local_inventory(tmp_path: Path) -> None:
    contents = {
        "captures/events/2026-08-16/one/event.json": json.dumps(
            {
                "status": "complete",
                "event_id": "one",
                "capture_method": "manual_fire",
                "start_timestamp": "2026-08-16T21:00:00-04:00",
            }
        ).encode(),
        "captures/events/2026-08-16/one/snapshot.jpg": b"image",
    }

    class FakeCollector:
        target = "pi@example"
        remote_root = "/home/example/squirrel_shooter"

        def __init__(self) -> None:
            self.transfers: list[list[str]] = []

        def list_files(self) -> list[RemoteFile]:
            return [RemoteFile(path, len(payload), "1.0") for path, payload in sorted(contents.items())]

        def transfer(self, files: list[RemoteFile], project: Path) -> None:
            self.transfers.append([item.relative_path for item in files])
            for item in files:
                destination = project / item.relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(contents[item.relative_path])

        def fetch_sanitized_config(self, destination: Path) -> list[str]:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text("classifier:\n  auto_accept_confidence: 0.6\n", encoding="utf-8")
            return []

        def fetch_context(self) -> dict[str, str]:
            return {"hostname": "pi", "git_commit": "abc"}

        def fetch_journal(self, destination: Path, since: str) -> None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(f"journal since {since}\n", encoding="utf-8")

    collector = FakeCollector()
    destination = tmp_path / "analysis" / "pi_dataset_2026-08-16_220000"

    result = collect_snapshot(collector, destination)

    assert len(collector.transfers) == 1
    assert result["inventory"]["overall"]["event_records"] == 1
    assert (destination / "raw" / "project" / "captures" / "events" / "2026-08-16" / "one" / "event.json").is_file()
    assert json.loads((destination / "inventory" / "collection_context.json").read_text())["hostname"] == "pi"
