"""Read-only startup file identity, without hardware or model execution.

These hashes describe files observed at application construction. They do not
attest Python's already-imported bytecode, native libraries, or a live device.
No endpoint refreshes them: a changed checkout after startup must not be
presented as the revision that started the running application.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import AppConfig


PROVENANCE_SCHEMA_VERSION = 1
# A truthful application-process startup observation, not OS process birth time.
PROCESS_START_OBSERVED_AT_UTC = datetime.now(timezone.utc).isoformat()
SOURCE_SUFFIXES = frozenset({".py", ".html", ".js", ".css"})


def _json_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    """Hash bytes once; withhold identity when the file changed during reading."""

    try:
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            after = os.fstat(stream.fileno())
        current = path.stat()
        signature = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if signature(before) != signature(after) or signature(after) != signature(current):
            return {"status": "changed_during_read", "sha256": None}
        return {"status": "present", "sha256": digest, "size_bytes": after.st_size}
    except FileNotFoundError:
        return {"status": "missing", "sha256": None}
    except OSError as exc:
        # Error messages and absolute paths can contain private host details.
        return {"status": "unreadable", "sha256": None, "error_type": type(exc).__name__}


def _git_identity(source_root: Path) -> dict[str, Any]:
    """Observe local refs only; no fetch, status write, remote URL or environment."""

    def read(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "--no-optional-locks", "-C", str(source_root), *args],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None

    revision = read("rev-parse", "--verify", "HEAD")
    branch = read("symbolic-ref", "--quiet", "--short", "HEAD") if revision else None
    # Restrict the status probe to application/configuration source. Never Misc/.
    status = read(
        "status", "--porcelain=v1", "--untracked-files=all", "--",
        "src/squirrel_shooter", "pyproject.toml", "config/default.yaml",
    ) if revision else None
    return {
        "revision": revision,
        "branch": branch,
        "application_worktree_dirty": None if status is None else bool(status),
        "identity_basis": "local_checkout_at_snapshot",
    }


def _source_identity(package_root: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    try:
        # Only this installed package's code/templates; no project-wide scan.
        for path in sorted(package_root.rglob("*")):
            relative = path.relative_to(package_root)
            if "Misc" in relative.parts or "__pycache__" in relative.parts:
                continue
            if path.suffix in SOURCE_SUFFIXES and path.is_file():
                files[relative.as_posix()] = _file_identity(path)
    except OSError as exc:
        return {"status": "unreadable", "sha256": None, "error_type": type(exc).__name__}
    complete = bool(files) and all(item["status"] == "present" for item in files.values())
    return {
        "status": "present" if complete else "incomplete",
        "sha256": _json_digest(files) if complete else None,
        "file_count": len(files),
        "files": files,
        "scope": "package Python, HTML, JavaScript and CSS files",
    }


def build_runtime_provenance(
    config: AppConfig,
    *,
    package_root: Path | None = None,
) -> dict[str, Any]:
    """Capture reusable startup diagnostics; never open a model with a DNN API."""

    try:
        return _build_runtime_provenance(config, package_root=package_root)
    except Exception as exc:
        # Observability must not abort startup after shared resources were built.
        # Never expose exception text, which can include paths or private data.
        return {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "status": "unavailable",
            "observed_at_utc": datetime.now(timezone.utc).isoformat(),
            "process_id": os.getpid(),
            "process_start_observed_at_utc": PROCESS_START_OBSERVED_AT_UTC,
            "process_start_time_basis": "provenance_module_import; not OS process birth",
            "error_type": type(exc).__name__,
        }


def _build_runtime_provenance(
    config: AppConfig,
    *,
    package_root: Path | None,
) -> dict[str, Any]:
    observed_at = datetime.now(timezone.utc).isoformat()
    package_root = package_root or Path(__file__).resolve().parent
    effective = asdict(config)
    effective.pop("source_path", None)
    effective.pop("source_sha256", None)
    config_file = _file_identity(config.source_path)
    parsed_hash = config.source_sha256
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "status": "observed",
        "observed_at_utc": observed_at,
        "process_id": os.getpid(),
        "process_start_observed_at_utc": PROCESS_START_OBSERVED_AT_UTC,
        "process_start_time_basis": "provenance_module_import; not OS process birth",
        "git": _git_identity(package_root.parent.parent),
        "source": _source_identity(package_root),
        "config": {
            "parsed_file_sha256": parsed_hash,
            "file_at_snapshot": config_file,
            "file_matches_parsed": (
                config_file["sha256"] == parsed_hash
                if parsed_hash is not None and config_file["sha256"] is not None else None
            ),
            "effective_sha256": _json_digest(effective),
        },
        "model": {
            "adapter": "legacy_mobilenet_voc",
            "execution_enabled": config.classifier.enabled,
            "definition": _file_identity(config.classifier.model_definition),
            "weights": _file_identity(config.classifier.model_weights),
        },
        "limitations": [
            "Startup file observations do not attest loaded bytecode or native libraries.",
            "Git branch/revision alone does not prove runtime bytes.",
            "Config values, model contents, environment and remote URLs are not exposed.",
            "Model hashes identify selected files, not inference or physical readiness.",
        ],
    }
