"""Shared legacy motion-size vocabulary, independent of model species labels.

These describe foreground geometry/movement, not semantic identification. Keep
their wire values readable in historical event records during modernization.
"""

from __future__ import annotations

from enum import StrEnum


class MotionCategory(StrEnum):
    PLANT_OR_SHADOW_FLICKER = "plant_or_shadow_flicker"
    TINY_MOTION = "tiny_motion"
    PERSON_SIZED = "person_sized"
    LARGE_OBJECT = "large_object"
    SMALL_ANIMAL_CANDIDATE = "small_animal_candidate"
    MEDIUM_ANIMAL_CANDIDATE = "medium_animal_candidate"
    UNCLASSIFIED_MOTION = "unclassified_motion"
    LIGHTING_CHANGE = "lighting_change"


def legacy_reacquisition_category_safe(category: str) -> bool:
    """Preserve the legacy category gate using the watcher's canonical spelling.

    The old association-only spelling remains rejected for historical callers.
    This policy is a legacy heuristic, not a future squirrel semantic decision.
    """

    if category == "large_object_candidate":
        category = MotionCategory.LARGE_OBJECT
    return category not in {
        MotionCategory.PERSON_SIZED,
        MotionCategory.LARGE_OBJECT,
        MotionCategory.LIGHTING_CHANGE,
    }
