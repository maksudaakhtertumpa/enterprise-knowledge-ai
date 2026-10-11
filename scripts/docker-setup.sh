#!/usr/bin/env bash
# One-time setup for `docker compose`: a private session secret for the web container
# (docker/web.env, git-ignored and kept out of images). Safe to run repeatedly.
set -euo pipefail
cd "$(dirname "$0")/.."
ENV_FILE=docker/web.env
if [[ -f "$ENV_FILE" ]] && grep -qE '^EKA_WEB_SESSION_SECRET=.{32,}' "$ENV_FILE"; then
  exit 0
fi
SECRET=$(openssl rand -base64 48 | tr -d '\n=+/' | cut -c1-48)
umask 077
{
  echo "# Created by scripts/docker-setup.sh. Keep private; never commit."
  echo "EKA_WEB_SESSION_SECRET=$SECRET"
} > "$ENV_FILE"
echo "Created $ENV_FILE (web session secret)."
