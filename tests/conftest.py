from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_test_config(tmp_path: Path, **changes: Any) -> Path:
    raw = yaml.safe_load((PROJECT_ROOT / "config/default.yaml").read_text(encoding="utf-8"))
    raw["camera"]["output_directory"] = (tmp_path / "captures").as_posix()
    raw["motion"]["debug_outputs"]["directory"] = (tmp_path / "debug").as_posix()
    raw["logging"]["directory"] = (tmp_path / "logs").as_posix()
    for dotted_key, value in changes.items():
        target = raw
        parts = dotted_key.split("__")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path
