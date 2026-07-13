"""Future squirrel-detection boundary.

Phase 3 will place garden-zone motion detection here. This module currently
defines data shapes only and performs no camera, network, or hardware work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class Detection:
    """A future image-space detection result."""

    x: int
    y: int
    width: int
    height: int
    confidence: float


class Detector(Protocol):
    """Interface for future local detection implementations."""

    def detect(self, frame: Any) -> list[Detection]:
        """Return detections without causing physical output."""
        ...
