"""Build offline inventories from a locally copied Raspberry Pi data snapshot."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import yaml


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
VIDEO_SUFFIXES = {".avi", ".h264", ".mkv", ".mov", ".mp4"}
REVIEW_FIELDS = (
    "record_kind",
    "timestamp",
    "source_path",
    "event_id",
    "tracker_id",
    "predicted_label",
    "confidence",
    "image_path",
    "context_image_path",
    "video_path",
    "full_frame_video_path",
    "day_night_state",
    "accepted_rejected_state",
    "shot_state",
    "human_label",
    "human_verified",
    "training_label",
    "training_eligible",
    "classification_status",
    "capture_method",
    "target_x",
    "target_y",
    "detections_json",
    "notes",
)


def _read_json(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"Could not read {path}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"Expected a JSON object in {path}")
        return {}
    return value


def _read_jsonl(path: Path, errors: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        errors.append(f"Could not read {path}: {exc}")
        return records
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"Could not parse {path}:{number}: {exc}")
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _relative(path: Path, snapshot: Path) -> str:
    try:
        return path.relative_to(snapshot).as_posix()
    except ValueError:
        return path.as_posix()


def _first_file(directory: Path, names: Iterable[str]) -> Path | None:
    for name in names:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def _timestamp(*records: dict[str, Any]) -> str:
    keys = ("event_timestamp", "start_timestamp", "timestamp", "classifier_timestamp", "reviewed_at")
    for record in records:
        for key in keys:
            value = record.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def _day_night(event: dict[str, Any], classification: dict[str, Any]) -> str:
    values = (
        event.get("day_night_state"),
        event.get("ir_mode_if_explicitly_detected_or_configured"),
        event.get("camera_mode_if_known"),
        classification.get("day_night_state"),
    )
    for value in values:
        normalized = str(value or "").strip().lower()
        if normalized in {"night", "night_vision", "infrared", "ir", "on", "enabled", "true"}:
            return "night"
        if normalized in {"day", "daylight", "color", "off", "disabled", "false"}:
            return "day"
    return "unknown"


def _decision_state(classification: dict[str, Any]) -> str:
    outcome = str(classification.get("outcome") or "")
    status = str(classification.get("classification_status") or "")
    if classification.get("auto_accepted") is True or outcome == "auto_labeled":
        return "accepted"
    if status == "false_positive" or outcome in {"rejected", "classifier_error"}:
        return "rejected"
    if status in {"review", "unknown", "unclassified"}:
        return status
    return status or outcome or "unknown"


def _shot_state(event: dict[str, Any]) -> str:
    method = str(event.get("capture_method") or event.get("event_type") or "")
    if method == "manual_fire":
        return "manual_shot"
    if method == "auto_fire":
        return "automatic_shot"
    return "no_shot" if method else "unknown"


def _event_row(event_path: Path, snapshot: Path, errors: list[str]) -> tuple[dict[str, Any], set[Path]]:
    directory = event_path.parent
    event = _read_json(event_path, errors)
    classification_path = directory / "classification.json"
    classification = _read_json(classification_path, errors) if classification_path.is_file() else {}
    image = _first_file(directory, ("classifier-input.jpg", "snapshot.jpg", "image.jpg"))
    context = _first_file(directory, ("original-frame.jpg", "snapshot.jpg"))
    video = _first_file(directory, ("auto_fire_zoom.avi", "manual_fire_zoom.avi", "clip.avi"))
    full_video = _first_file(directory, ("auto_fire_full.avi", "manual_fire_full.avi"))
    target = classification.get("target_pixel")
    if not isinstance(target, dict):
        target = event.get("target_pixel") if isinstance(event.get("target_pixel"), dict) else {}
    detections = classification.get("detections")
    if not isinstance(detections, list):
        detections = []
    predicted = classification.get("top_label") or classification.get("model_suggestion") or event.get("classifier_label")
    confidence = classification.get("top_confidence")
    if confidence is None:
        confidence = event.get("classifier_confidence")
    decision_state = _decision_state(classification)
    if event.get("capture_method") in {"manual_fire", "auto_fire"} or event.get("event_type") in {"manual_fire", "auto_fire"}:
        decision_state = "accepted"
    row = {
        "record_kind": "event",
        "timestamp": _timestamp(event, classification),
        "source_path": _relative(event_path, snapshot),
        "event_id": event.get("event_id") or classification.get("event_id") or directory.name,
        "tracker_id": event.get("track_id") if event.get("track_id") is not None else classification.get("track_id"),
        "predicted_label": predicted or "",
        "confidence": confidence if confidence is not None else "",
        "image_path": _relative(image, snapshot) if image else "",
        "context_image_path": _relative(context, snapshot) if context else "",
        "video_path": _relative(video, snapshot) if video else "",
        "full_frame_video_path": _relative(full_video, snapshot) if full_video else "",
        "day_night_state": _day_night(event, classification),
        "accepted_rejected_state": decision_state,
        "shot_state": _shot_state(event),
        "human_label": classification.get("human_label") or event.get("human_review_label") or "",
        "human_verified": bool(classification.get("human_verified", False)),
        "training_label": classification.get("training_label") or event.get("training_label") or "",
        "training_eligible": classification.get("training_dataset_status") == "included",
        "classification_status": classification.get("classification_status") or "",
        "capture_method": event.get("capture_method") or event.get("event_type") or "",
        "target_x": target.get("x", event.get("target_pixel_x", "")),
        "target_y": target.get("y", event.get("target_pixel_y", "")),
        "detections_json": json.dumps(detections, separators=(",", ":")),
        "notes": event.get("notes") or classification.get("error") or "",
        "_event": event,
        "_classification": classification,
    }
    referenced = {path for path in (event_path, classification_path, image, context, video, full_video) if path and path.exists()}
    return row, referenced


def _training_row(sample_path: Path, snapshot: Path, errors: list[str]) -> tuple[dict[str, Any], set[Path]]:
    sample = _read_json(sample_path, errors)
    directory = sample_path.parent
    source = sample.get("source") if isinstance(sample.get("source"), dict) else {}
    image = _first_file(directory, ("image.jpg", "classifier-input.jpg"))
    context = _first_file(directory, ("original-frame.jpg",))
    detections = source.get("detections") if isinstance(source.get("detections"), list) else []
    top = detections[0] if detections and isinstance(detections[0], dict) else {}
    row = {
        "record_kind": "training_sample",
        "timestamp": sample.get("labeled_at") or source.get("event_start_timestamp") or _timestamp(sample),
        "source_path": _relative(sample_path, snapshot),
        "event_id": sample.get("event_id") or sample.get("sample_id") or directory.name,
        "tracker_id": sample.get("track_id", ""),
        "predicted_label": source.get("model_suggestion") or source.get("top_label") or top.get("label") or "",
        "confidence": source.get("top_confidence") if source.get("top_confidence") is not None else top.get("confidence", ""),
        "image_path": _relative(image, snapshot) if image else "",
        "context_image_path": _relative(context, snapshot) if context else "",
        "video_path": "",
        "full_frame_video_path": "",
        "day_night_state": _day_night(source.get("source_camera", {}), source),
        "accepted_rejected_state": "human_verified" if sample.get("human_verified") else "unknown",
        "shot_state": "unknown",
        "human_label": sample.get("label") if sample.get("human_verified") else "",
        "human_verified": bool(sample.get("human_verified", False)),
        "training_label": sample.get("label") or "",
        "training_eligible": bool(sample.get("training_eligible", False)),
        "classification_status": "human_verified" if sample.get("human_verified") else "",
        "capture_method": source.get("capture_method") or "",
        "target_x": "",
        "target_y": "",
        "detections_json": json.dumps(detections, separators=(",", ":")),
        "notes": sample.get("exclusion_reason") or "",
        "_sample": sample,
        "_source": source,
    }
    return row, {path for path in (sample_path, image, context) if path and path.exists()}


def _legacy_classifier_rows(project: Path, snapshot: Path, seen: set[Path], errors: list[str]) -> list[dict[str, Any]]:
    root = project / "captures" / "classifier"
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for metadata_path in root.rglob("*.json"):
        if metadata_path in seen:
            continue
        record = _read_json(metadata_path, errors)
        if not record:
            continue
        image = next((item for item in metadata_path.parent.glob(f"{metadata_path.stem}.*") if item.suffix.lower() in IMAGE_SUFFIXES), None)
        detections = record.get("detections") if isinstance(record.get("detections"), list) else []
        top = detections[0] if detections and isinstance(detections[0], dict) else {}
        rows.append({
            "record_kind": "legacy_classifier",
            "timestamp": _timestamp(record),
            "source_path": _relative(metadata_path, snapshot),
            "event_id": record.get("event_id") or record.get("item_id") or metadata_path.stem,
            "tracker_id": record.get("track_id", ""),
            "predicted_label": record.get("top_label") or record.get("model_suggestion") or top.get("label") or "",
            "confidence": record.get("top_confidence") if record.get("top_confidence") is not None else top.get("confidence", ""),
            "image_path": _relative(image, snapshot) if image else "",
            "context_image_path": "",
            "video_path": "",
            "full_frame_video_path": "",
            "day_night_state": _day_night(record, record),
            "accepted_rejected_state": _decision_state(record),
            "shot_state": "unknown",
            "human_label": record.get("human_label") or "",
            "human_verified": bool(record.get("human_verified", False)),
            "training_label": record.get("training_label") or "",
            "training_eligible": record.get("training_dataset_status") == "included",
            "classification_status": record.get("classification_status") or "",
            "capture_method": record.get("capture_method") or "",
            "target_x": "",
            "target_y": "",
            "detections_json": json.dumps(detections, separators=(",", ":")),
            "notes": record.get("error") or "",
            "_classification": record,
        })
        seen.add(metadata_path)
        if image:
            seen.add(image)
    return rows


def _standalone_rows(project: Path, snapshot: Path, seen: set[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in project.rglob("*"):
        if not path.is_file() or path in seen or path.suffix.lower() not in IMAGE_SUFFIXES | VIDEO_SUFFIXES:
            continue
        is_image = path.suffix.lower() in IMAGE_SUFFIXES
        rows.append({
            "record_kind": "standalone_media",
            "timestamp": datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds"),
            "source_path": _relative(path, snapshot),
            "event_id": path.stem,
            "tracker_id": "",
            "predicted_label": "",
            "confidence": "",
            "image_path": _relative(path, snapshot) if is_image else "",
            "context_image_path": "",
            "video_path": _relative(path, snapshot) if not is_image else "",
            "full_frame_video_path": "",
            "day_night_state": "unknown",
            "accepted_rejected_state": "unclassified",
            "shot_state": "unknown",
            "human_label": "",
            "human_verified": False,
            "training_label": "",
            "training_eligible": "",
            "classification_status": "",
            "capture_method": "standalone_media",
            "target_x": "",
            "target_y": "",
            "detections_json": "[]",
            "notes": "No event/classifier metadata was found beside this retained media file.",
        })
    return rows


def _named_jsonl_records(
    project: Path,
    filename: str,
    errors: list[str],
) -> list[tuple[Path, int, dict[str, Any]]]:
    records: list[tuple[Path, int, dict[str, Any]]] = []
    for path in sorted(project.rglob(f"{filename}*")):
        if not path.is_file():
            continue
        for number, record in enumerate(_read_jsonl(path, errors), start=1):
            records.append((path, number, record))
    return records


def _metadata_only_row(
    record: dict[str, Any],
    *,
    kind: str,
    source_path: str,
) -> dict[str, Any]:
    detections = record.get("detections") if isinstance(record.get("detections"), list) else []
    top = detections[0] if detections and isinstance(detections[0], dict) else {}
    target = record.get("target_pixel") if isinstance(record.get("target_pixel"), dict) else {}
    return {
        "record_kind": kind,
        "timestamp": _timestamp(record),
        "source_path": source_path,
        "event_id": record.get("event_id") or record.get("item_id") or "",
        "tracker_id": record.get("track_id", ""),
        "predicted_label": record.get("top_label") or record.get("model_suggestion") or top.get("label") or "",
        "confidence": (
            record.get("top_confidence")
            if record.get("top_confidence") is not None
            else top.get("confidence", "")
        ),
        "image_path": "",
        "context_image_path": "",
        "video_path": "",
        "full_frame_video_path": "",
        "day_night_state": _day_night(record, record),
        "accepted_rejected_state": _decision_state(record),
        "shot_state": _shot_state(record),
        "human_label": record.get("human_label") or record.get("human_review_label") or "",
        "human_verified": bool(record.get("human_verified", False)),
        "training_label": record.get("training_label") or "",
        "training_eligible": record.get("training_dataset_status") == "included",
        "classification_status": record.get("classification_status") or "",
        "capture_method": record.get("capture_method") or record.get("event_type") or "",
        "target_x": target.get("x", record.get("target_pixel_x", "")),
        "target_y": target.get("y", record.get("target_pixel_y", "")),
        "detections_json": json.dumps(detections, separators=(",", ":")),
        "notes": record.get("notes") or record.get("error") or "",
    }


def _orphan_event_log_rows(
    project: Path,
    snapshot: Path,
    represented_event_ids: set[str],
    errors: list[str],
) -> tuple[list[dict[str, Any]], int]:
    records = _named_jsonl_records(project, "events.jsonl", errors)
    orphaned: dict[str, tuple[Path, int, dict[str, Any]]] = {}
    for path, number, record in records:
        event_id = str(record.get("event_id") or "")
        if event_id and event_id not in represented_event_ids:
            orphaned[event_id] = (path, number, record)
    rows: list[dict[str, Any]] = []
    for event_id, (path, number, record) in sorted(orphaned.items()):
        row = _metadata_only_row(
            record,
            kind="event_log",
            source_path=f"{_relative(path, snapshot)}#record={number}",
        )
        row["event_id"] = event_id
        row["notes"] = "; ".join(
            filter(None, (str(row.get("notes") or ""), "Event media/metadata folder is no longer retained."))
        )
        row["_event"] = record
        rows.append(row)
    return rows, len(records)


def _orphan_classifier_audit_rows(
    project: Path,
    snapshot: Path,
    represented_classifier_ids: set[str],
    errors: list[str],
) -> tuple[list[dict[str, Any]], Counter[str], int]:
    records = _named_jsonl_records(project, "classifier.jsonl", errors)
    actions: Counter[str] = Counter(str(record.get("action") or "unknown") for _, _, record in records)
    candidates: dict[str, list[tuple[Path, int, dict[str, Any]]]] = defaultdict(list)
    for path, number, record in records:
        event_id = str(record.get("event_id") or record.get("item_id") or "")
        if event_id and event_id not in represented_classifier_ids:
            candidates[event_id].append((path, number, record))
    rows: list[dict[str, Any]] = []
    for event_id, items in sorted(candidates.items()):
        classified = [item for item in items if item[2].get("action") in {"classified", "legacy_migrated"}]
        path, number, record = (classified or items)[0]
        record = dict(record)
        reviewed = next((item[2] for item in reversed(items) if item[2].get("human_verified")), None)
        if reviewed is not None:
            for field in (
                "human_label",
                "human_verified",
                "training_label",
                "training_dataset_status",
                "classification_status",
            ):
                if field in reviewed:
                    record[field] = reviewed[field]
        row = _metadata_only_row(
            record,
            kind="classifier_audit",
            source_path=f"{_relative(path, snapshot)}#record={number}",
        )
        row["event_id"] = event_id
        row["notes"] = "; ".join(
            filter(
                None,
                (str(row.get("notes") or ""), "Recovered from retained classifier audit; event folder is not retained."),
            )
        )
        row["_classification"] = record
        rows.append(row)
    return rows, actions, len(records)


def _correlate_shot_events(rows: list[dict[str, Any]]) -> None:
    by_event_id = {
        str(row.get("event_id")): row
        for row in rows
        if row.get("record_kind") == "event" and row.get("event_id")
    }
    for shot in rows:
        event = shot.get("_event") if isinstance(shot.get("_event"), dict) else {}
        if shot.get("shot_state") != "automatic_shot":
            continue
        source_id = str(event.get("source_event_id") or "")
        source = by_event_id.get(source_id)
        if source is not None:
            source["shot_state"] = "automatic_shot"
            note = str(source.get("notes") or "")
            suffix = f"Associated automatic shot recording: {shot.get('event_id')}"
            source["notes"] = f"{note}; {suffix}".strip("; ")


def _config_threshold(snapshot: Path) -> float:
    path = snapshot / "raw" / "config" / "default.sanitized.yaml"
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return float(payload.get("classifier", {}).get("auto_accept_confidence", 0.60))
    except (OSError, TypeError, ValueError, yaml.YAMLError):
        return 0.60


def _class_statistics(rows: list[dict[str, Any]], threshold: float) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _prediction_rows(rows):
        label = str(row.get("predicted_label") or "").strip()
        if label:
            grouped[label].append(row)
    result: dict[str, dict[str, Any]] = {}
    for label, items in sorted(grouped.items()):
        confidences = [float(item["confidence"]) for item in items if _is_number(item.get("confidence"))]
        event_ids = {str(item["event_id"]) for item in items if item.get("event_id") not in {None, ""}}
        tracks = {
            (str(item.get("event_id") or ""), str(item["tracker_id"]))
            for item in items
            if item.get("tracker_id") not in {None, ""}
        }
        result[label] = {
            "classified_records": len(items),
            "usable_images_or_crops": sum(bool(item.get("image_path")) for item in items),
            "confidence_count": len(confidences),
            "confidence_minimum": min(confidences) if confidences else None,
            "confidence_mean": statistics.fmean(confidences) if confidences else None,
            "confidence_median": statistics.median(confidences) if confidences else None,
            "confidence_maximum": max(confidences) if confidences else None,
            "high_confidence_records": sum(value >= threshold for value in confidences),
            "low_confidence_records": sum(value < threshold for value in confidences),
            "accepted_records": sum(item.get("accepted_rejected_state") == "accepted" for item in items),
            "rejected_records": sum(item.get("accepted_rejected_state") == "rejected" for item in items),
            "unique_events": len(event_ids),
            "unique_event_tracks": len(tracks),
        }
    return result


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _human_training_counts(rows: list[dict[str, Any]]) -> tuple[Counter[str], dict[str, set[str]]]:
    canonical = [row for row in rows if row.get("record_kind") == "training_sample"]
    source_rows = canonical or [row for row in rows if row.get("human_verified")]
    raw: Counter[str] = Counter()
    independent: dict[str, set[str]] = defaultdict(set)
    for row in source_rows:
        label = str(row.get("training_label") or "").strip()
        if not label or not row.get("human_verified") or not row.get("training_eligible"):
            continue
        raw[label] += 1
        group = str(row.get("event_id") or row.get("source_path") or "")
        independent[label].add(group)
    return raw, independent


def _prediction_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Count one classifier decision per event, preferring the original event record."""

    chosen: dict[str, dict[str, Any]] = {}
    priority = {
        "event": 5,
        "legacy_classifier": 4,
        "event_log": 3,
        "classifier_audit": 2,
        "training_sample": 1,
    }
    for row in rows:
        source = row.get("_source") if isinstance(row.get("_source"), dict) else {}
        has_classifier_record = (
            bool(row.get("_classification"))
            or row.get("record_kind") == "legacy_classifier"
            or bool(source.get("classifier_model") or source.get("detections") or source.get("top_label"))
        )
        if not has_classifier_record:
            continue
        key = str(row.get("event_id") or row.get("source_path"))
        existing = chosen.get(key)
        if existing is None or priority.get(str(row.get("record_kind")), 0) > priority.get(str(existing.get("record_kind")), 0):
            chosen[key] = row
    return list(chosen.values())


def _unknown_negative_summary(
    classifier_rows: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    threshold: float,
    diagnostics: dict[str, Any],
) -> dict[str, int]:
    def detections(row: dict[str, Any]) -> list[Any]:
        try:
            value = json.loads(str(row.get("detections_json") or "[]"))
        except json.JSONDecodeError:
            return []
        return value if isinstance(value, list) else []

    return {
        "unknown_classifications": sum(row.get("classification_status") == "unknown" for row in classifier_rows),
        "classifier_errors": sum(row.get("classification_status") == "unclassified" for row in classifier_rows),
        "no_detection_records": sum(not detections(row) for row in classifier_rows),
        "low_confidence_records": sum(
            _is_number(row.get("confidence")) and float(row["confidence"]) < threshold
            for row in classifier_rows
        ),
        "human_verified_false_positives": sum(
            row.get("record_kind") == "training_sample"
            and row.get("training_label") == "background_or_false_positive"
            and row.get("training_eligible")
            for row in all_rows
        ),
        "standalone_unclassified_media": sum(row.get("record_kind") == "standalone_media" for row in all_rows),
        "global_motion_rejections": len(diagnostics.get("global_rejections", [])),
    }


def _training_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    canonical = [
        row
        for row in rows
        if row.get("record_kind") == "training_sample"
        and row.get("human_verified")
        and row.get("training_eligible")
    ]
    return canonical or [
        row
        for row in rows
        if row.get("human_verified") and row.get("training_label") and row.get("training_eligible")
    ]


def _squirrel_diversity(rows: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    daylight: Counter[str] = Counter()
    size: Counter[str] = Counter()
    position: Counter[str] = Counter()
    motion: Counter[str] = Counter()
    for row in _training_rows(rows):
        if "squirrel" not in str(row.get("training_label") or ""):
            continue
        daylight[str(row.get("day_night_state") or "unknown")] += 1
        source = row.get("_source") if isinstance(row.get("_source"), dict) else {}
        box = source.get("source_bounding_box") if isinstance(source.get("source_bounding_box"), dict) else {}
        camera = source.get("source_camera") if isinstance(source.get("source_camera"), dict) else {}
        width = camera.get("actual_width") or camera.get("requested_width")
        height = camera.get("actual_height") or camera.get("requested_height")
        if all(_is_number(value) for value in (box.get("x"), box.get("y"), box.get("width"), box.get("height"), width, height)):
            frame_area = float(width) * float(height)
            ratio = float(box["width"]) * float(box["height"]) / max(1.0, frame_area)
            size["small/far"] += ratio < 0.02
            size["medium"] += 0.02 <= ratio < 0.10
            size["large/near"] += ratio >= 0.10
            center_x = (float(box["x"]) + float(box["width"]) / 2.0) / float(width)
            center_y = (float(box["y"]) + float(box["height"]) / 2.0) / float(height)
            horizontal = "left" if center_x < 1 / 3 else ("right" if center_x > 2 / 3 else "center")
            vertical = "top" if center_y < 1 / 3 else ("bottom" if center_y > 2 / 3 else "middle")
            position[f"{vertical}-{horizontal}"] += 1
        category = source.get("motion_category")
        if category:
            motion[str(category)] += 1
    return {"day_night": daylight, "apparent_size": size, "frame_position": position, "motion_category": motion}


def _write_progress_report(path: Path, rows: list[dict[str, Any]]) -> None:
    raw, independent = _human_training_counts(rows)
    squirrel_labels = sorted(label for label in raw if "squirrel" in label)
    squirrel_raw = sum(raw[label] for label in squirrel_labels)
    squirrel_independent = len(set().union(*(independent[label] for label in squirrel_labels))) if squirrel_labels else 0
    negatives = raw.get("background_or_false_positive", 0)
    other = {label: count for label, count in raw.items() if label not in squirrel_labels and label != "background_or_false_positive"}
    diversity = _squirrel_diversity(rows)
    lines = [
        "# Custom-classifier dataset progress",
        "",
        "Current classifier predictions are preserved as metadata only. They are not treated as ground-truth training labels.",
        "",
        "## Counts",
        "",
        f"- Raw retained image files represented in the review index: **{sum(bool(row.get('image_path')) for row in rows)}**",
        f"- Human-verified, training-eligible samples: **{sum(raw.values())}**",
        f"- Human-verified squirrel samples: **{squirrel_raw}**",
        f"- Estimated independent squirrel event groups: **{squirrel_independent}**",
        f"- Human-verified background/false-positive negatives: **{negatives}**",
        "",
        "The independent estimate groups samples by source event ID. It is deliberately more conservative than raw frame count and is not a claim that every group is visually diverse.",
        "",
        "## Human-verified class coverage",
        "",
    ]
    if raw:
        lines.extend(f"- `{label}`: {count} raw / {len(independent[label])} independent event groups" for label, count in sorted(raw.items()))
    else:
        lines.append("- No human-verified training samples were found in this snapshot.")
    lines.extend(["", "## Assessment", ""])
    if not raw:
        lines.append("Dataset readiness cannot be established from predictions alone. Human review and labeling are the next required step.")
    else:
        largest = max(raw.values())
        smallest = min(raw.values())
        imbalance = largest / max(1, smallest)
        lines.append(f"Observed human-verified class imbalance is approximately **{imbalance:.1f}:1** between the largest and smallest retained classes.")
        if squirrel_independent < 20:
            lines.append("Independent squirrel coverage is still sparse; prioritize varied squirrel events rather than many neighboring frames from one encounter.")
        elif squirrel_independent < 100:
            lines.append("There is a meaningful squirrel seed set, but broader appearance, distance, lighting, and seasonal coverage is still needed.")
        else:
            lines.append("Squirrel event count is substantial enough to begin a careful manual diversity audit and grouped train/validation/test planning.")
        if negatives < max(10, squirrel_raw // 2):
            lines.append("Verified background/false-positive coverage is weak relative to squirrel coverage.")
        if other:
            lines.append("Retained verified confuser classes include: " + ", ".join(f"{label} ({count})" for label, count in sorted(other.items())) + ".")
        else:
            lines.append("No verified animal/object confuser classes were found beyond squirrels and background negatives.")
    lines.extend(["", "## Retained squirrel diversity signals", ""])
    for name, counts in diversity.items():
        label = name.replace("_", " ").title()
        if counts:
            lines.append(f"- {label}: " + ", ".join(f"{key} ({value})" for key, value in sorted(counts.items())))
        else:
            lines.append(f"- {label}: not determinable from retained metadata")
    lines.extend([
        "",
        "## Manual review still required",
        "",
        "- Confirm or correct every useful prediction before training.",
        "- Audit near-duplicate frames within each event and split datasets by event/session, never by neighboring frame.",
        "- Review squirrel size/distance, pose, location, daylight/night state, weather, occlusion, and background diversity directly from the media.",
        "- Review common confusers such as birds, rabbits, people, vegetation, shadows, and cars based on what is actually present.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _diagnostic_sources(project: Path, errors: list[str]) -> dict[str, Any]:
    rejection_records: list[dict[str, Any]] = []
    session_records: list[dict[str, Any]] = []
    performance_records: list[dict[str, Any]] = []
    for path in project.rglob("rejections.jsonl*"):
        rejection_records.extend(_read_jsonl(path, errors))
    for path in project.rglob("session-*.json"):
        session_records.append(_read_json(path, errors))
    for path in project.rglob("squirrel-shooter-*.jsonl*"):
        for record in _read_jsonl(path, errors):
            if record.get("event") == "runtime_performance":
                performance_records.append(record)
    return {
        "global_rejections": rejection_records,
        "sessions": [record for record in session_records if record],
        "performance": performance_records,
    }


def _write_wind_report(path: Path, sources: dict[str, Any], event_rows: list[dict[str, Any]]) -> None:
    rejections = sources["global_rejections"]
    sessions = sources["sessions"]
    performance = sources["performance"]
    event_metadata = [row.get("_event", {}) for row in event_rows]
    has_global_area = any("raw_foreground_percent" in item for item in rejections)
    has_event_components = any(item.get("components") or item.get("group_samples") for item in event_metadata)
    has_session_counts = any("raw_contours" in item or "grouped_candidates" in item for item in sessions)
    has_filter_rejections = any(item.get("rejected_by_filter") for item in sessions)
    has_event_tracks = any(row.get("tracker_id") not in {None, ""} for row in event_rows)
    has_fps = any("measured_camera_fps" in item for item in rejections) or any("average_measured_fps" in item for item in sessions)
    has_queue = any("classifier_queue_depth" in item for item in performance)
    rows = [
        ("Raw motion foreground activity", "PARTIALLY AVAILABLE" if has_global_area else "NOT CURRENTLY RETAINED", "Global-rejection samples retain percentages, but no continuous per-frame foreground series is stored."),
        ("Contour/blob counts", "PARTIALLY AVAILABLE" if has_session_counts or has_event_components else "NOT CURRENTLY RETAINED", "Sessions retain aggregates and saved events retain components; ordinary rejected frames are not individually preserved."),
        ("Changed-pixel/foreground area", "PARTIALLY AVAILABLE" if has_global_area or has_event_components else "NOT CURRENTLY RETAINED", "Global rejections and saved motion events provide sampled/aggregate coverage, not every processed frame."),
        ("Motion candidates before tracking", "PARTIALLY AVAILABLE" if has_session_counts or has_event_components else "NOT CURRENTLY RETAINED", "Aggregate raw-contour counts and saved-event components exist, but the full pre-track candidate stream does not."),
        ("Tracker workload", "PARTIALLY AVAILABLE" if has_session_counts or has_event_tracks else "NOT CURRENTLY RETAINED", "Grouped-candidate totals and saved track IDs exist; there is no continuous per-frame tracker workload history."),
        ("Dropped/ignored candidates", "PARTIALLY AVAILABLE" if has_filter_rejections else "NOT CURRENTLY RETAINED", "Session summaries retain rejection counts by filter when present, but individual per-candidate decisions are not durably logged."),
        ("Frame timing/FPS", "AVAILABLE NOW" if has_fps or performance else "NOT CURRENTLY RETAINED", "Event/rejection/session metadata and runtime-performance logs carry camera/detection timing when present."),
        ("Classifier queue/load", "AVAILABLE NOW" if has_queue else "NOT CURRENTLY RETAINED", "The 30-second runtime-performance log records queue depth, inference FPS, and latency when retained."),
    ]
    lines = [
        "# Wind/shadow diagnostic availability",
        "",
        "This report describes evidence actually found in the copied snapshot. It does not infer missing per-frame telemetry.",
        "",
        f"- Global-rejection records found: **{len(rejections)}**",
        f"- Session summaries found: **{len(sessions)}**",
        f"- Runtime-performance samples found: **{len(performance)}**",
        f"- Completed event records found: **{len(event_rows)}**",
        "",
        "| Diagnostic question | Status | What is retained |",
        "|---|---|---|",
    ]
    lines.extend(f"| {name} | **{status}** | {detail} |" for name, status, detail in rows)
    lines.extend([
        "",
        "The current evidence can characterize global wind/shadow overload episodes and their timing, but it cannot reconstruct every rejected contour or candidate. No instrumentation change was made in this pass.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field, "") for field in REVIEW_FIELDS}


def build_inventory(snapshot: Path) -> dict[str, Any]:
    """Analyze one local snapshot and write CSV, JSON, and Markdown reports."""

    snapshot = snapshot.resolve()
    project = snapshot / "raw" / "project"
    if not project.is_dir():
        raise ValueError(f"Snapshot is missing raw/project: {snapshot}")
    inventory_directory = snapshot / "inventory"
    inventory_directory.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    referenced: set[Path] = set()

    event_paths = sorted(project.rglob("event.json"))
    for path in event_paths:
        row, used = _event_row(path, snapshot, errors)
        rows.append(row)
        referenced.update(used)
    event_file_ids = {str(row.get("event_id")) for row in rows if row.get("event_id")}
    event_log_rows, event_log_record_count = _orphan_event_log_rows(
        project,
        snapshot,
        event_file_ids,
        errors,
    )
    rows.extend(event_log_rows)
    sample_paths = sorted(project.rglob("sample.json"))
    for path in sample_paths:
        row, used = _training_row(path, snapshot, errors)
        rows.append(row)
        referenced.update(used)
    rows.extend(_legacy_classifier_rows(project, snapshot, referenced, errors))
    represented_classifier_ids = {
        str(row.get("event_id"))
        for row in _prediction_rows(rows)
        if row.get("event_id")
    }
    audit_rows, classifier_audit_actions, classifier_audit_record_count = _orphan_classifier_audit_rows(
        project,
        snapshot,
        represented_classifier_ids,
        errors,
    )
    metadata_rows = {
        str(row.get("event_id")): row
        for row in rows
        if row.get("record_kind") == "event_log" and row.get("event_id")
    }
    for audit_row in audit_rows:
        event_log_row = metadata_rows.get(str(audit_row.get("event_id") or ""))
        if event_log_row is None:
            audit_row["_recovered_classifier_audit"] = True
            rows.append(audit_row)
            continue
        for field in (
            "predicted_label",
            "confidence",
            "accepted_rejected_state",
            "human_label",
            "human_verified",
            "training_label",
            "training_eligible",
            "classification_status",
            "target_x",
            "target_y",
            "detections_json",
        ):
            if audit_row.get(field) not in {None, "", False}:
                event_log_row[field] = audit_row[field]
        event_log_row["_classification"] = audit_row.get("_classification", {})
        event_log_row["_recovered_classifier_audit"] = True
        event_log_row["notes"] = "; ".join(
            filter(
                None,
                (
                    str(event_log_row.get("notes") or ""),
                    f"Classifier audit source: {audit_row.get('source_path')}",
                ),
            )
        )
    rows.extend(_standalone_rows(project, snapshot, referenced))
    _correlate_shot_events(rows)
    rows.sort(key=lambda item: (str(item.get("timestamp") or ""), str(item.get("source_path") or "")))

    files = [path for path in project.rglob("*") if path.is_file()]
    raw_files = [path for path in (snapshot / "raw").rglob("*") if path.is_file()]
    images = [path for path in files if path.suffix.lower() in IMAGE_SUFFIXES]
    videos = [path for path in files if path.suffix.lower() in VIDEO_SUFFIXES]
    timestamps = [stamp for stamp in (_parse_time(row.get("timestamp")) for row in rows) if stamp is not None]
    threshold = _config_threshold(snapshot)
    class_statistics = _class_statistics(rows, threshold)
    classifier_rows = _prediction_rows(rows)
    raw_training, independent_training = _human_training_counts(rows)
    review_rows = [_public_row(row) for row in rows]

    review_json_path = inventory_directory / "review_index.json"
    review_json_path.write_text(json.dumps(review_rows, indent=2, default=str) + "\n", encoding="utf-8")
    with (inventory_directory / "review_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(review_rows)

    diagnostics = _diagnostic_sources(project, errors)
    retention_actions = [
        action
        for session in diagnostics["sessions"]
        for action in (session.get("retention_actions") or [])
        if isinstance(action, dict)
    ]
    retention_reasons = Counter(
        str(action.get("reason")) for action in retention_actions if action.get("reason")
    )
    retention_action_count = sum(
        int(session.get("retention_action_count") or len(session.get("retention_actions") or []))
        for session in diagnostics["sessions"]
    )
    motion_categories = Counter(
        str(row.get("_event", {}).get("provisional_category"))
        for row in rows
        if isinstance(row.get("_event"), dict) and row.get("_event", {}).get("provisional_category")
    )
    global_rejection_reasons = Counter(
        str(record.get("reason"))
        for record in diagnostics["global_rejections"]
        if record.get("reason")
    )
    inventory = {
        "schema_version": 1,
        "snapshot": str(snapshot),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "overall": {
            "event_records": sum(row["record_kind"] in {"event", "event_log"} for row in rows),
            "event_metadata_files": sum(row["record_kind"] == "event" for row in rows),
            "orphan_events_recovered_from_logs": sum(row["record_kind"] == "event_log" for row in rows),
            "event_log_records": event_log_record_count,
            "review_records": len(rows),
            "images": len(images),
            "videos": len(videos),
            "classifier_records": len(classifier_rows),
            "classifier_audit_records": classifier_audit_record_count,
            "orphan_classifier_records_recovered_from_audit": sum(
                bool(row.get("_recovered_classifier_audit")) for row in rows
            ),
            "total_storage_bytes": sum(path.stat().st_size for path in raw_files),
            "earliest_retained_sample": min(timestamps).isoformat() if timestamps else None,
            "latest_retained_sample": max(timestamps).isoformat() if timestamps else None,
        },
        "predicted_class_statistics": class_statistics,
        "classifier_audit_actions": dict(sorted(classifier_audit_actions.items())),
        "human_verified_training": {
            label: {"raw_samples": count, "independent_event_groups": len(independent_training[label])}
            for label, count in sorted(raw_training.items())
        },
        "confidence_high_threshold": threshold,
        "discovered_predicted_labels": sorted(class_statistics),
        "discovered_human_labels": sorted({str(row["human_label"]) for row in rows if row.get("human_label")}),
        "discovered_training_labels": sorted(raw_training),
        "unknown_negative_inventory": _unknown_negative_summary(classifier_rows, rows, threshold, diagnostics),
        "motion_heuristic_categories": dict(sorted(motion_categories.items())),
        "global_rejection_reasons": dict(sorted(global_rejection_reasons.items())),
        "shot_recordings": {
            "manual": sum(row.get("capture_method") == "manual_fire" for row in rows),
            "automatic": sum(row.get("capture_method") == "auto_fire" for row in rows),
        },
        "day_night_records": dict(sorted(Counter(str(row.get("day_night_state") or "unknown") for row in rows).items())),
        "diagnostic_record_counts": {key: len(value) for key, value in diagnostics.items()},
        "retention_evidence": {
            "reported_action_count_in_retained_sessions": retention_action_count,
            "retained_action_details": len(retention_actions),
            "retained_action_reasons": dict(sorted(retention_reasons.items())),
            "limitation": "Retention logs can prove some removals but cannot recreate deleted media or guarantee a lifetime deletion count after session-log rotation.",
        },
        "errors": errors,
        "ground_truth_policy": "Classifier predictions are metadata only; only explicit human-verified labels count as training truth.",
    }
    (inventory_directory / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    _write_progress_report(inventory_directory / "classifier_progress.md", rows)
    _write_wind_report(
        inventory_directory / "wind_shadow_diagnostic.md",
        diagnostics,
        [row for row in rows if row["record_kind"] == "event"],
    )
    summary = [
        "# Pi dataset inventory",
        "",
        f"- Events: **{inventory['overall']['event_records']}**",
        f"- Images: **{inventory['overall']['images']}**",
        f"- Videos: **{inventory['overall']['videos']}**",
        f"- Classifier records: **{inventory['overall']['classifier_records']}**",
        f"- Total copied project data: **{inventory['overall']['total_storage_bytes']} bytes**",
        f"- Earliest retained sample: **{inventory['overall']['earliest_retained_sample'] or 'unknown'}**",
        f"- Latest retained sample: **{inventory['overall']['latest_retained_sample'] or 'unknown'}**",
        "",
        "Current classifier predictions are preserved as metadata, not assumed to be ground-truth training labels.",
        "",
        "See `review_index.csv`, `review_index.json`, `classifier_progress.md`, and `wind_shadow_diagnostic.md` in this directory.",
        "",
    ]
    (inventory_directory / "README.md").write_text("\n".join(summary), encoding="utf-8")
    return inventory


__all__ = ["build_inventory"]
