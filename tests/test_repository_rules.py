from __future__ import annotations

from conftest import PROJECT_ROOT


def test_misc_is_ignored_and_declared_off_limits() -> None:
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    agent_rules = (PROJECT_ROOT / "AGENTS.md").read_text(encoding="utf-8")

    assert "/Misc/" in gitignore
    assert "Do not search, inspect, modify, rename, delete, format, refactor, stage, or commit" in agent_rules
    assert "Preserve all existing modified and untracked files under `Misc/` exactly as they are" in agent_rules
