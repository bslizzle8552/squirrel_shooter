from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import write_test_config
from squirrel_shooter.config import load_config
from squirrel_shooter import runtime_provenance as provenance


def make_inputs(tmp_path: Path):
    config = load_config(write_test_config(tmp_path))
    package = tmp_path / "src" / "squirrel_shooter"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    definition = tmp_path / "private-model-definition"
    weights = tmp_path / "private-model-weights"
    definition.write_bytes(b"definition fixture; never executed")
    weights.write_bytes(b"weights fixture; never executed")
    return replace(
        config,
        classifier=replace(config.classifier, model_definition=definition, model_weights=weights),
    ), package


def test_snapshot_identifies_parsed_effective_config_and_selected_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, package = make_inputs(tmp_path)
    monkeypatch.setattr(provenance, "_git_identity", lambda _root: {"revision": "fixture-revision"})
    report = provenance.build_runtime_provenance(config, package_root=package)

    assert report["schema_version"] == 1
    assert report["git"]["revision"] == "fixture-revision"
    assert report["config"]["parsed_file_sha256"] == hashlib.sha256(config.source_path.read_bytes()).hexdigest()
    assert report["config"]["file_matches_parsed"] is True
    assert report["model"]["weights"]["sha256"] == hashlib.sha256(config.classifier.model_weights.read_bytes()).hexdigest()
    assert report["model"]["execution_enabled"] is False
    assert report["source"]["file_count"] == 1
    assert report["process_start_observed_at_utc"].endswith("+00:00")
    serialized = json.dumps(report)
    assert str(tmp_path) not in serialized
    assert "private-model" not in serialized
    assert "never executed" not in serialized


def test_cli_override_changes_effective_hash_without_relabeling_parsed_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, package = make_inputs(tmp_path)
    monkeypatch.setattr(provenance, "_git_identity", lambda _root: {})
    original = provenance.build_runtime_provenance(config, package_root=package)
    overridden = provenance.build_runtime_provenance(
        replace(config, dashboard=replace(config.dashboard, port=config.dashboard.port + 1)),
        package_root=package,
    )
    assert original["config"]["parsed_file_sha256"] == overridden["config"]["parsed_file_sha256"]
    assert original["config"]["effective_sha256"] != overridden["config"]["effective_sha256"]


def test_changed_source_and_config_are_not_mistaken_for_loaded_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, package = make_inputs(tmp_path)
    monkeypatch.setattr(provenance, "_git_identity", lambda _root: {})
    original = provenance.build_runtime_provenance(config, package_root=package)
    config.source_path.write_bytes(config.source_path.read_bytes() + b"\n# changed after load\n")
    (package / "__init__.py").write_text("VALUE = 2\n", encoding="utf-8")
    changed = provenance.build_runtime_provenance(config, package_root=package)

    assert changed["config"]["file_matches_parsed"] is False
    assert changed["config"]["effective_sha256"] == original["config"]["effective_sha256"]
    assert changed["source"]["sha256"] != original["source"]["sha256"]
    assert changed["process_start_observed_at_utc"] == original["process_start_observed_at_utc"]
    assert original["config"]["file_matches_parsed"] is True


def test_missing_models_and_unreadable_file_are_explicit_without_private_error_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, package = make_inputs(tmp_path)
    monkeypatch.setattr(provenance, "_git_identity", lambda _root: {})
    config = replace(config, classifier=replace(config.classifier, model_weights=tmp_path / "absent"))
    report = provenance.build_runtime_provenance(config, package_root=package)
    assert report["model"]["weights"] == {"status": "missing", "sha256": None}

    def deny(*_args, **_kwargs):
        raise PermissionError("private host detail")

    monkeypatch.setattr(Path, "open", deny)
    failure = provenance._file_identity(tmp_path / "secret")
    assert failure == {"status": "unreadable", "sha256": None, "error_type": "PermissionError"}


def test_changing_file_withholds_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "model"
    path.write_bytes(b"old")
    real_digest = hashlib.file_digest

    def changed_digest(stream, algorithm):
        result = real_digest(stream, algorithm)
        path.write_bytes(b"new, longer content")
        return result

    monkeypatch.setattr(provenance.hashlib, "file_digest", changed_digest)
    assert provenance._file_identity(path) == {"status": "changed_during_read", "sha256": None}


@pytest.mark.parametrize("failure", [FileNotFoundError(), subprocess.TimeoutExpired("git", 2)])
def test_git_unavailable_is_nonfatal_and_does_not_claim_clean_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception,
) -> None:
    def fail(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(provenance.subprocess, "run", fail)
    result = provenance._git_identity(tmp_path)
    assert result["revision"] is None
    assert result["branch"] is None
    assert result["application_worktree_dirty"] is None


def test_git_probe_is_local_scoped_and_supports_detached_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    responses = iter([
        SimpleNamespace(returncode=0, stdout="a" * 40 + "\n"),
        SimpleNamespace(returncode=1, stdout=""),
        SimpleNamespace(returncode=0, stdout=" M src/squirrel_shooter/app.py\n"),
    ])

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return next(responses)

    monkeypatch.setattr(provenance.subprocess, "run", run)
    result = provenance._git_identity(tmp_path)
    assert result["revision"] == "a" * 40
    assert result["branch"] is None
    assert result["application_worktree_dirty"] is True
    assert all(call[1]["timeout"] == 2.0 and not call[1].get("shell") for call in calls)
    assert calls[-1][0][-4:] == ["--", "src/squirrel_shooter", "pyproject.toml", "config/default.yaml"]


def test_unexpected_diagnostic_failure_cannot_abort_runtime_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from squirrel_shooter.app import ApplicationRuntime

    config, _ = make_inputs(tmp_path)

    def invalid_git_output(*_args, **_kwargs):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "private output")

    monkeypatch.setattr(provenance.subprocess, "run", invalid_git_output)
    camera, motion, control = object(), object(), object()
    runtime = ApplicationRuntime(config, camera=camera, motion=motion, manual_control=control)

    assert runtime.camera is camera and runtime.motion is motion
    assert runtime.manual_control is control
    assert runtime.provenance["status"] == "unavailable"
    assert runtime.provenance["error_type"] == "UnicodeDecodeError"
    assert "private output" not in json.dumps(runtime.provenance)
