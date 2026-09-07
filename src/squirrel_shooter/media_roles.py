"""Explicit pixel provenance; historical filenames alone never prove clean input."""
from enum import StrEnum
from pathlib import Path
from typing import Any


class MediaRole(StrEnum):
    CLEAN = "clean_authoritative"
    REVIEW = "annotated_review"
    CROP = "derived_crop"
    UNKNOWN = "unknown_legacy"


def verified_classifier_source(record: dict[str, Any]) -> bool:
    return (record.get("source_pixel_provenance") == "shared_camera_raw_v1"
            and record.get("source_media_role") == MediaRole.CLEAN
            and record.get("frame_selection_method") != "middle_fallback")


def media_role(filename: str, record: dict[str, Any]) -> MediaRole:
    name = Path(filename).name
    if name in {"snapshot.jpg", "clip.avi"}:
        return MediaRole.REVIEW
    if verified_classifier_source(record):
        if name == "original-frame.jpg":
            return MediaRole.CLEAN
        if name in {"classifier-input.jpg", "image.jpg"}:
            return MediaRole.CROP
    if (record.get("file") == name and record.get("file_role") == MediaRole.CLEAN
            and record.get("sha256") and record.get("status") in {"complete", "degraded"}):
        return MediaRole.CLEAN
    return MediaRole.UNKNOWN
