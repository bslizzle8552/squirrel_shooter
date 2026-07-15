"""Structured logging and bounded-file retention helpers."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import LoggingConfig


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(getattr(record, "structured_data", {}))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, separators=(",", ":"))


def configure_logging(config: LoggingConfig, max_log_files: int) -> Path | None:
    """Configure JSON console/file logs; file failures leave console logging active."""

    root = logging.getLogger()
    root.setLevel(getattr(logging, config.level))
    formatter = JsonFormatter()
    if not root.handlers:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        root.addHandler(console)
    try:
        config.directory.mkdir(parents=True, exist_ok=True)
        path = config.directory / f"squirrel-shooter-{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl"
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(formatter)
        root.addHandler(handler)
        cleanup_oldest(config.directory, "squirrel-shooter-*.jsonl", max_log_files, logging.getLogger(__name__))
        return path
    except OSError as exc:
        logging.getLogger(__name__).error(
            "Could not create application log file",
            extra={"structured_data": {"event": "log_directory_error", "error": str(exc)}},
        )
        return None


def cleanup_oldest(directory: Path, pattern: str, limit: int, logger: logging.Logger) -> int:
    """Delete oldest matching regular files first until at most ``limit`` remain."""

    try:
        files = [path for path in directory.glob(pattern) if path.is_file()]
        files.sort(key=lambda path: (path.stat().st_mtime, path.name))
    except OSError as exc:
        logger.warning(
            "Storage retention scan failed",
            extra={"structured_data": {"event": "retention_scan_failure", "directory": str(directory), "error": str(exc)}},
        )
        return 0
    removed = 0
    for path in files[: max(0, len(files) - limit)]:
        try:
            path.unlink()
            removed += 1
            logger.info(
                "Removed oldest retained file",
                extra={"structured_data": {"event": "retention_cleanup", "filename": str(path)}},
            )
        except OSError as exc:
            logger.warning(
                "Could not remove retained file",
                extra={"structured_data": {"event": "retention_delete_failure", "filename": str(path), "error": str(exc)}},
            )
    return removed
