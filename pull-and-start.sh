#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

update_failed() {
    echo
    echo "Update failed. Squirrel Squirter is stopped."
    echo "Fix the error above, then run: ./start.sh"
}
trap update_failed ERR

systemctl --user stop squirrel-squirter.service 2>/dev/null || true
cd "$REPO_DIR"
CURRENT_BRANCH="$(git branch --show-current)"
if [ -z "$CURRENT_BRANCH" ]; then
  echo "Refusing to update a detached checkout."
  exit 1
fi
git pull --ff-only origin "$CURRENT_BRANCH"
.venv/bin/python -m pip install -e ".[test]"
.venv/bin/python -m pytest

trap - ERR
exec "$REPO_DIR/start.sh"
