"""Compatibility imports for the permanent pan/tilt motion foundation.

Live camera motion detection remains in ``watch_detection`` and
``motion_runtime``. Importing this module does not construct hardware or move a
servo; hardware is created only when ``PanTiltController`` is instantiated.
"""

from __future__ import annotations

from .pan_tilt import (
    PanTiltConfig,
    PanTiltController,
    PanTiltPosition,
    ServoDriver,
    ServoKitDriver,
    clamp_angle,
    generate_coordinated_path,
)


MotionController = PanTiltController

__all__ = [
    "MotionController",
    "PanTiltConfig",
    "PanTiltController",
    "PanTiltPosition",
    "ServoDriver",
    "ServoKitDriver",
    "clamp_angle",
    "generate_coordinated_path",
]
