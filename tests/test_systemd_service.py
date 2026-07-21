from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_user_service_runs_combined_app_and_stops_gracefully() -> None:
    service = (PROJECT_ROOT / "deploy/systemd/squirrel-squirter.service").read_text(encoding="utf-8")

    assert "WorkingDirectory=%h/squirrel_shooter" in service
    assert "ExecStart=%h/squirrel_shooter/.venv/bin/python -m squirrel_shooter.app --headless" in service
    assert "Restart=on-failure" in service
    assert "KillSignal=SIGINT" in service
    assert "WantedBy=default.target" in service
    assert "User=" not in service


def test_two_command_scripts_hide_service_setup_and_safe_update_details() -> None:
    start = (PROJECT_ROOT / "start.sh").read_text(encoding="utf-8")
    update = (PROJECT_ROOT / "pull-and-start.sh").read_text(encoding="utf-8")

    assert "sudo loginctl enable-linger" in start
    assert "systemctl --user enable --now squirrel-squirter.service" in start
    assert "git pull --ff-only origin main" in update
    assert '.venv/bin/python -m pip install -e ".[test]"' in update
    assert ".venv/bin/python -m pytest" in update
    assert 'exec "$REPO_DIR/start.sh"' in update
