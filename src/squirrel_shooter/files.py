"""Camera-output filename helpers that do not depend on camera hardware."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def timestamped_output_path(
    output_directory: Path,
    prefix: str,
    suffix: str,
    *,
    when: datetime | None = None,
) -> Path:
    """Build a collision-resistant timestamped output path."""

    timestamp = (when or datetime.now().astimezone()).strftime("%Y%m%d-%H%M%S-%f")
    clean_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    return output_directory / f"{prefix}-{timestamp}{clean_suffix}"
