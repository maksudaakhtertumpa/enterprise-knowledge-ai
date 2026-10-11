#!/usr/bin/env bash
# Take the README screenshots from the real app, in a throwaway workspace.
#
#   ./scripts/screenshots.sh            # real models from .env (Ollama running)
#   ./scripts/screenshots.sh --offline  # stand-ins: checks the script, NOT for the README
#
# Uses its own ports (API 8010, web 3010) and a temporary data folder, so a running
# `make dev` and your own data/ are not touched. Output: docs/screenshots/*.jpg
set -euo pipefail
cd "$(dirname "$0")/.."

API_PORT=8010
WEB_PORT=3010
DATA_DIR="$(mktemp -d "${TMPDIR:-/tmp}/eka-shots-XXXXXX")"
export EKA_DATA_DIR="$DATA_DIR" EKA_LOG_LEVEL=WARNING NEXT_TELEMETRY_DISABLED=1
if [[ "${1:-}" == "--offline" ]]; then
  export EKA_EMBEDDING_BACKEND=hashing EKA_RERANKER_BACKEND=hashing
  export EKA_LLM_BACKEND=extractive EKA_ENTAILMENT_BACKEND=lexical
  echo "Offline stand-ins: these screenshots are NOT real answers. Don't commit them."
fi

cleanup() {
  trap - EXIT INT TERM
  kill 0 2>/dev/null || true
  rm -rf "$DATA_DIR"
}
trap cleanup EXIT INT TERM

KEY="$(python -m app.cli create-workspace "Teratai HR (demo)" | awk '/^API key/ {print $NF}')"

python -m uvicorn app.main:app --port "$API_PORT" --no-server-header &
API_PID=$!
echo "Starting the API (loads the models)…"
for _ in $(seq 1 600); do
  if curl -fsS "http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1; then break; fi
  if ! kill -0 "$API_PID" 2>/dev/null; then echo "The API failed to start."; exit 1; fi
  sleep 1
done

echo "Building and starting the web UI…"
(
  cd web
  npx next build >/dev/null
  EKA_API_URL="http://127.0.0.1:$API_PORT" \
    npx next start --hostname 127.0.0.1 --port "$WEB_PORT" >/dev/null 2>&1
) &
for _ in $(seq 1 300); do
  if curl -fsS "http://127.0.0.1:$WEB_PORT/connect" >/dev/null 2>&1; then break; fi
  sleep 1
done

(cd web && EKA_SHOT_URL="http://127.0.0.1:$WEB_PORT" EKA_SHOT_KEY="$KEY" node scripts/screenshots.mjs)
echo "Done: docs/screenshots/"
