"""Future pan/tilt motion boundary.

No GPIO, I2C, PCA9685, or servo library is imported here. Physical motion is
intentionally unavailable until powered hardware can be tested safely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class PanTiltPosition:
    """A future calibrated pan/tilt target in degrees."""

    pan_degrees: float
    tilt_degrees: float


class MotionController(Protocol):
    """Interface to be implemented only during the dry-fire phase."""

    def aim(self, target: PanTiltPosition) -> None:
        """Move to a validated target without controlling water."""
        ...
