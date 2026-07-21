#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_SOURCE="$REPO_DIR/deploy/systemd/squirrel-squirter.service"
SERVICE_TARGET="$HOME/.config/systemd/user/squirrel-squirter.service"

if [[ ! -x "$REPO_DIR/.venv/bin/python" ]]; then
    echo "Virtual environment not found at $REPO_DIR/.venv"
    echo "Create it and install the project before starting Squirrel Squirter."
    exit 1
fi

mkdir -p "$(dirname -- "$SERVICE_TARGET")"
install -m 644 "$SERVICE_SOURCE" "$SERVICE_TARGET"

LINGER_STATUS="$(loginctl show-user "$USER" -p Linger --value 2>/dev/null || true)"
if [[ "$LINGER_STATUS" != "yes" ]]; then
    echo "One-time setup: enabling the service to run after logout and at boot."
    sudo loginctl enable-linger "$USER"
fi

systemctl --user daemon-reload
systemctl --user enable --now squirrel-squirter.service
systemctl --user status squirrel-squirter.service --no-pager

echo
echo "Squirrel Squirter is running. You can close SSH or turn off your laptop."
