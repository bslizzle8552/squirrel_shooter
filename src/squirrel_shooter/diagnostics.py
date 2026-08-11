"""Structured logging and bounded-file retention helpers."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
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


class RoutineAccessFilter(logging.Filter):
    """Drop successful high-frequency polls while preserving HTTP failures."""

    _paths = (
        "/api/status",
        "/api/events",
        "/api/recent-events",
        "/api/classifier-review",
        "/api/manual-control",
        "/video_feed",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "werkzeug":
            return True
        message = record.getMessage()
        routine_request = any(
            f'"GET {path}' in message or f'"HEAD {path}' in message
            for path in self._paths
        )
        successful = any(f" {status} " in message for status in (200, 204, 304))
        return not (routine_request and successful)


def configure_logging(config: LoggingConfig, max_log_files: int) -> Path | None:
    """Configure JSON console/file logs; file failures leave console logging active."""

    root = logging.getLogger()
    root.setLevel(getattr(logging, config.level))
    # State transitions already have structured application logs. Routine 2xx
    # polling access lines otherwise double-write to this file and journald;
    # retain all failed requests, including rejected physical-control tokens.
    werkzeug = logging.getLogger("werkzeug")
    werkzeug.setLevel(logging.INFO)
    if not any(isinstance(item, RoutineAccessFilter) for item in werkzeug.filters):
        werkzeug.addFilter(RoutineAccessFilter())
    formatter = JsonFormatter()
    if not root.handlers:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        root.addHandler(console)
    try:
        config.directory.mkdir(parents=True, exist_ok=True)
        path = config.directory / f"squirrel-shooter-{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl"
        handler = RotatingFileHandler(
            path,
            maxBytes=max(1, int(config.maximum_active_log_megabytes * 1024 * 1024)),
            backupCount=config.retained_log_rotations,
            encoding="utf-8",
        )
        handler.setFormatter(formatter)
        root.addHandler(handler)
        cleanup_oldest(config.directory, "squirrel-shooter-*.jsonl*", max_log_files, logging.getLogger(__name__))
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
