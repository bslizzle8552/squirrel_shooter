"""Named project phases without any mode-switching side effects."""

from __future__ import annotations

from enum import Enum


class OperatingMode(str, Enum):
    CAMERA_TEST = "camera_test"
    FOOTAGE_COLLECTION = "footage_collection"
    MOTION_DETECTION = "motion_detection"
    DRY_FIRE = "dry_fire"
    WATER_TEST = "water_test"


DEFAULT_MODE = OperatingMode.CAMERA_TEST
