"""PC-initiated, read-only collection of retained Squirrel Squirter Pi data."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

import yaml

from .dataset_inventory import build_inventory


DEFAULT_USER = "bslizzle8552"
DEFAULT_REMOTE_ROOT = "/home/bslizzle8552/squirrel_shooter"
DEFAULT_SERVICE = "squirrel-squirter.service"
REMOTE_DATA_PATHS = (
    "captures",
    "logs",  # legacy logs retained by older deployments
    "recordings",  # legacy/standalone recordings named by repository ignore rules
    "data/captures",  # legacy capture layout named by repository ignore rules
    "data/videos",  # legacy video layout named by repository ignore rules
    "debug",  # configured optional detector debug evidence
    "config/calibration_points.json",
)
ROOT_MEDIA_PATTERNS = (
    "camera-test-*.avi",
    "camera-test-*.mp4",
    "camera-still-*.jpg",
    "*.h264",
)
SENSITIVE_KEY = re.compile(
    r"(^|_)(api_?key|authorization|cookie|credential|password|secret|session|token|swid|espn_s2|private_?key)($|_)",
    re.IGNORECASE,
)
SAFE_CONNECTION_VALUE = re.compile(r"^[A-Za-z0-9_.:@-]+$")


@dataclass(frozen=True)
class RemoteFile:
    relative_path: str
    size: int
    mtime_epoch: str


def parse_remote_manifest(payload: bytes) -> list[RemoteFile]:
    """Parse NUL-delimited path/size/mtime triples emitted by GNU find."""

    fields = payload.split(b"\0")
    if fields and fields[-1] == b"":
        fields.pop()
    if len(fields) % 3:
        raise ValueError("Remote file manifest was incomplete")
    files: dict[str, RemoteFile] = {}
    for index in range(0, len(fields), 3):
        relative = fields[index].decode("utf-8", errors="surrogateescape")
        relative = relative[2:] if relative.startswith("./") else relative
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or not path.parts or "\n" in relative or "\r" in relative:
            raise ValueError(f"Unsafe remote manifest path: {relative!r}")
        size = int(fields[index + 1])
        mtime = fields[index + 2].decode("ascii")
        files[path.as_posix()] = RemoteFile(path.as_posix(), size, mtime)
    return sorted(files.values(), key=lambda item: item.relative_path)


def sanitize_configuration(value: Any, *, path: tuple[str, ...] = ()) -> tuple[Any, list[str]]:
    """Redact likely secrets locally while preserving all ordinary settings."""

    redacted: list[str] = []
    if isinstance(value, dict):
        result: dict[Any, Any] = {}
        for key, child in value.items():
            key_text = str(key)
            child_path = (*path, key_text)
            if SENSITIVE_KEY.search(key_text):
                result[key] = "<REDACTED>"
                redacted.append(".".join(child_path))
            else:
                result[key], child_redactions = sanitize_configuration(child, path=child_path)
                redacted.extend(child_redactions)
        return result, redacted
    if isinstance(value, list):
        result_list = []
        for index, child in enumerate(value):
            sanitized, child_redactions = sanitize_configuration(child, path=(*path, str(index)))
            result_list.append(sanitized)
            redacted.extend(child_redactions)
        return result_list, redacted
    return value, redacted


def choose_incremental_sources(
    files: Sequence[RemoteFile],
    previous_snapshot: Path | None,
) -> tuple[dict[str, Path], list[RemoteFile]]:
    """Select unchanged files reusable from an older snapshot by size and mtime."""

    if previous_snapshot is None:
        return {}, list(files)
    manifest_path = previous_snapshot / "inventory" / "remote_files.json"
    try:
        prior_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        prior_records = prior_payload["files"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return {}, list(files)
    prior = {
        str(item["relative_path"]): item
        for item in prior_records
        if isinstance(item, dict) and "relative_path" in item
    }
    reusable: dict[str, Path] = {}
    transfer: list[RemoteFile] = []
    for item in files:
        old = prior.get(item.relative_path)
        source = previous_snapshot / "raw" / "project" / Path(item.relative_path)
        if (
            old
            and int(old.get("size", -1)) == item.size
            and str(old.get("mtime_epoch")) == item.mtime_epoch
            and source.is_file()
            and source.stat().st_size == item.size
        ):
            reusable[item.relative_path] = source
        else:
            transfer.append(item)
    return reusable, transfer


class PiCollector:
    def __init__(
        self,
        *,
        host: str,
        user: str,
        remote_root: str,
        port: int = 22,
        identity_file: Path | None = None,
        service: str = DEFAULT_SERVICE,
    ) -> None:
        for label, value in (("host", host), ("user", user), ("service", service)):
            if not value or not SAFE_CONNECTION_VALUE.fullmatch(value):
                raise ValueError(f"Unsafe or empty {label}: {value!r}")
        if "\n" in remote_root or "\r" in remote_root or not remote_root.startswith("/"):
            raise ValueError("remote_root must be one absolute POSIX path")
        if not 1 <= port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        self.host = host
        self.user = user
        self.remote_root = remote_root.rstrip("/")
        self.port = port
        self.identity_file = identity_file
        self.service = service

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}"

    def _ssh_prefix(self) -> list[str]:
        command = ["ssh", "-o", "ConnectTimeout=20", "-p", str(self.port)]
        if self.identity_file:
            command.extend(("-i", str(self.identity_file)))
        command.append(self.target)
        return command

    def _run_ssh(self, remote_command: str, *, text: bool = False) -> subprocess.CompletedProcess[Any]:
        return subprocess.run(
            [*self._ssh_prefix(), remote_command],
            check=True,
            capture_output=True,
            text=text,
            timeout=None,
        )

    def list_files(self) -> list[RemoteFile]:
        root = shlex.quote(self.remote_root)
        data_paths = " ".join(shlex.quote(item) for item in REMOTE_DATA_PATHS)
        pattern_expression = " -o ".join(f"-name {shlex.quote(pattern)}" for pattern in ROOT_MEDIA_PATTERNS)
        command = (
            f"cd {root} && "
            f"for item in {data_paths}; do "
            "if [ -e \"$item\" ]; then find \"$item\" -type f -printf '%p\\0%s\\0%T@\\0'; fi; "
            "done; "
            f"find . -maxdepth 1 -type f \\( {pattern_expression} \\) -printf '%p\\0%s\\0%T@\\0'"
        )
        result = self._run_ssh(command)
        return parse_remote_manifest(result.stdout)

    def fetch_sanitized_config(self, destination: Path) -> list[str]:
        remote_path = shlex.quote(f"{self.remote_root}/config/default.yaml")
        result = self._run_ssh(f"cat -- {remote_path}", text=True)
        payload = yaml.safe_load(result.stdout) or {}
        sanitized, redactions = sanitize_configuration(payload)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(yaml.safe_dump(sanitized, sort_keys=False), encoding="utf-8")
        return redactions

    def fetch_context(self) -> dict[str, str]:
        root = shlex.quote(self.remote_root)
        service = shlex.quote(self.service)
        command = " ; ".join(
            (
                "printf 'hostname='; hostname",
                "printf 'collected_at='; date --iso-8601=seconds",
                "printf 'timezone='; (timedatectl show -p Timezone --value 2>/dev/null || cat /etc/timezone 2>/dev/null || true)",
                f"printf 'git_branch='; git -C {root} branch --show-current 2>/dev/null || true",
                f"printf 'git_commit='; git -C {root} rev-parse HEAD 2>/dev/null || true",
                f"printf 'service_state='; systemctl --user is-active {service} 2>/dev/null || true",
                f"printf 'service_enabled='; systemctl --user is-enabled {service} 2>/dev/null || true",
                f"printf 'model_definition='; stat -c '%n|%s|%y' {root}/models/mobilenet-ssd/deploy.prototxt 2>/dev/null || true",
                f"printf 'model_weights='; stat -c '%n|%s|%y' {root}/models/mobilenet-ssd/mobilenet_iter_73000.caffemodel 2>/dev/null || true",
            )
        )
        result = self._run_ssh(command, text=True)
        context: dict[str, str] = {}
        for line in result.stdout.splitlines():
            key, separator, value = line.partition("=")
            if separator:
                context[key] = value
        context.update(
            remote_target=self.target,
            remote_root=self.remote_root,
            service=self.service,
        )
        return context

    def fetch_journal(self, destination: Path, since: str) -> None:
        if "\n" in since or "\r" in since:
            raise ValueError("journal since value must be one line")
        service = shlex.quote(self.service)
        command = (
            f"journalctl --user -u {service} --since {shlex.quote(since)} "
            "--no-pager --output=short-iso 2>/dev/null || true"
        )
        result = self._run_ssh(command, text=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(result.stdout, encoding="utf-8")

    @staticmethod
    def _sftp_quote(value: str) -> str:
        return '"' + value.replace("\\", "/").replace('"', '\\"') + '"'

    def transfer(self, files: Sequence[RemoteFile], destination_project: Path) -> None:
        if not files:
            return
        lines: list[str] = []
        for item in files:
            local = destination_project / Path(item.relative_path)
            local.parent.mkdir(parents=True, exist_ok=True)
            remote = str(PurePosixPath(self.remote_root) / PurePosixPath(item.relative_path))
            lines.append(f"get -p {self._sftp_quote(remote)} {self._sftp_quote(str(local.resolve()))}")
        batch_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".sftp", encoding="utf-8", delete=False) as handle:
                handle.write("\n".join(lines) + "\n")
                batch_path = Path(handle.name)
            # OpenSSH's -b option appends BatchMode=yes to its child SSH
            # command. Put BatchMode=no first so Windows OpenSSH can prompt on
            # the console for password/passphrase authentication while SFTP
            # still reads and fail-checks the transfer commands as a batch.
            # OpenSSH uses the first value supplied for each SSH option.
            command = ["sftp", "-oBatchMode=no", "-b", str(batch_path), "-P", str(self.port)]
            if self.identity_file:
                command.extend(("-i", str(self.identity_file)))
            command.append(self.target)
            subprocess.run(command, check=True, timeout=None)
        finally:
            if batch_path is not None:
                batch_path.unlink(missing_ok=True)


def _latest_previous_snapshot(base: Path, current: Path) -> Path | None:
    candidates = [
        path
        for path in base.glob("pi_dataset_*")
        if path.is_dir() and path.resolve() != current.resolve() and (path / "inventory" / "remote_files.json").is_file()
    ]
    return max(candidates, key=lambda path: path.name) if candidates else None


def _copy_reusable(reusable: dict[str, Path], destination_project: Path) -> None:
    for relative, source in reusable.items():
        destination = destination_project / Path(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _changed_files(before: Sequence[RemoteFile], after: Sequence[RemoteFile]) -> list[RemoteFile]:
    old = {item.relative_path: item for item in before}
    return [item for item in after if old.get(item.relative_path) != item]


def _validate_files(files: Sequence[RemoteFile], destination_project: Path) -> list[str]:
    issues: list[str] = []
    for item in files:
        local = destination_project / Path(item.relative_path)
        if not local.is_file():
            issues.append(f"missing: {item.relative_path}")
        elif local.stat().st_size < item.size:
            issues.append(f"short copy: {item.relative_path} ({local.stat().st_size} < {item.size})")
    return issues


def collect_snapshot(
    collector: PiCollector,
    destination: Path,
    *,
    journal_since: str = "30 days ago",
    run_inventory: bool = True,
) -> dict[str, Any]:
    """Collect one snapshot without modifying or analyzing anything on the Pi."""

    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    project = destination / "raw" / "project"
    inventory_directory = destination / "inventory"
    project.mkdir(parents=True)
    inventory_directory.mkdir(parents=True)

    initial_files = collector.list_files()
    previous = _latest_previous_snapshot(destination.parent, destination)
    reusable, transfer = choose_incremental_sources(initial_files, previous)
    _copy_reusable(reusable, project)
    collector.transfer(transfer, project)

    # One cheap second listing catches captures/logs that changed during transfer.
    final_files = collector.list_files()
    changed = _changed_files(initial_files, final_files)
    if changed:
        collector.transfer(changed, project)

    redactions = collector.fetch_sanitized_config(destination / "raw" / "config" / "default.sanitized.yaml")
    context = collector.fetch_context()
    collector.fetch_journal(
        destination / "raw" / "logs" / "service" / "squirrel-squirter-journal.txt",
        journal_since,
    )
    validation_issues = _validate_files(final_files, project)
    manifest = {
        "schema_version": 1,
        "source": {
            "target": collector.target,
            "remote_root": collector.remote_root,
        },
        "files": [asdict(item) for item in final_files],
        "initial_file_count": len(initial_files),
        "final_file_count": len(final_files),
        "reused_from_previous_snapshot": len(reusable),
        "transferred_initially": len(transfer),
        "retransferred_after_change": len(changed),
        "previous_snapshot": str(previous) if previous else None,
        "validation_issues": validation_issues,
    }
    (inventory_directory / "remote_files.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (inventory_directory / "collection_context.json").write_text(json.dumps(context, indent=2) + "\n", encoding="utf-8")
    (inventory_directory / "config_redactions.json").write_text(
        json.dumps({"redacted_keys": redactions}, indent=2) + "\n",
        encoding="utf-8",
    )
    exclusions = {
        "intentionally_excluded": [
            "Git repository objects and ordinary source files already present on the development PC",
            "virtual environments, caches, and unrelated files outside the named project data roots",
            "full classifier model binaries (file metadata is recorded in collection_context.json)",
            "secrets from configuration (likely sensitive keys are redacted on the PC)",
            f"system journal entries older than {journal_since!r}; retained application logs are copied in full",
        ],
        "pi_side_operations": [
            "find/stat-style file listing",
            "read-only SFTP file reads",
            "read-only config and bounded service-journal reads",
            "hostname/timezone/Git/service status queries",
        ],
        "not_performed_on_pi": [
            "hashing the full dataset",
            "video decoding or transcoding",
            "frame extraction, thumbnailing, computer vision, or ML inference",
            "archive creation, deletion, moving, relabeling, or configuration changes",
        ],
    }
    (inventory_directory / "collection_scope.json").write_text(json.dumps(exclusions, indent=2) + "\n", encoding="utf-8")
    result: dict[str, Any] = {"collection": manifest, "context": context}
    if run_inventory:
        result["inventory"] = build_inventory(destination)
    return result


def default_destination() -> Path:
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H%M%S")
    return Path("analysis") / f"pi_dataset_{stamp}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read retained Squirrel Squirter data from the Pi, then inventory the local copy.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect = subparsers.add_parser("collect", help="copy the retained dataset over SSH/SFTP, then inventory it locally")
    collect.add_argument("--host", required=True, help="Pi Tailscale/MagicDNS hostname or IP")
    collect.add_argument("--user", default=DEFAULT_USER)
    collect.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    collect.add_argument("--port", type=int, default=22)
    collect.add_argument("--identity-file", type=Path)
    collect.add_argument("--service", default=DEFAULT_SERVICE)
    collect.add_argument("--destination", type=Path, default=None)
    collect.add_argument("--journal-since", default="30 days ago")
    collect.add_argument("--skip-inventory", action="store_true")
    inventory = subparsers.add_parser("inventory", help="rebuild reports from an already copied local snapshot")
    inventory.add_argument("snapshot", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "inventory":
            inventory = build_inventory(args.snapshot)
            print(f"Inventory written to {args.snapshot.resolve() / 'inventory'}")
            print(json.dumps(inventory["overall"], indent=2))
            return 0
        destination = (args.destination or default_destination()).resolve()
        collector = PiCollector(
            host=args.host,
            user=args.user,
            remote_root=args.remote_root,
            port=args.port,
            identity_file=args.identity_file,
            service=args.service,
        )
        print(f"Read-only source: {collector.target}:{collector.remote_root}")
        print(f"Local snapshot: {destination}")
        result = collect_snapshot(
            collector,
            destination,
            journal_since=args.journal_since,
            run_inventory=not args.skip_inventory,
        )
        print(json.dumps(result.get("inventory", result["collection"]), indent=2))
        return 0
    except (OSError, ValueError, subprocess.SubprocessError, yaml.YAMLError) as exc:
        print(f"Pi dataset collection failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PiCollector",
    "RemoteFile",
    "choose_incremental_sources",
    "collect_snapshot",
    "parse_remote_manifest",
    "sanitize_configuration",
]
