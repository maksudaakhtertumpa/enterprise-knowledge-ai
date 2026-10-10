#!/usr/bin/env bash
# Give the web UI a fixed session-sealing secret in web/.env.local (git-ignored), so
# restarting the web server doesn't log you out. Safe to run repeatedly.
set -euo pipefail
cd "$(dirname "$0")/.."
ENV_FILE=web/.env.local
if [[ -f "$ENV_FILE" ]] && grep -qE '^EKA_WEB_SESSION_SECRET=.{32,}' "$ENV_FILE"; then
  exit 0
fi
SECRET=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')
{ echo "# Created by scripts/web-setup.sh. Keep private; never commit."; echo "EKA_WEB_SESSION_SECRET=$SECRET"; } >> "$ENV_FILE"
chmod 600 "$ENV_FILE"
echo "Created $ENV_FILE with a session secret (you stay connected across restarts)."
