#!/usr/bin/env bash
# Start the API (port 8000) and the web UI (port 3000) together; Ctrl+C stops both.
#
#   ./scripts/dev.sh            # real models (needs `make install-ml` and Ollama)
#   ./scripts/dev.sh --offline  # offline stand-ins: no model download, no Ollama
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${1:-}" == "--offline" ]]; then
  export EKA_EMBEDDING_BACKEND=hashing EKA_RERANKER_BACKEND=hashing EKA_LLM_BACKEND=extractive
  export EKA_ENTAILMENT_BACKEND=lexical
  export EKA_DATA_DIR="${EKA_DATA_DIR:-./data-offline}"   # keep offline vectors apart
  echo "Offline mode: non-semantic stand-ins, data in $EKA_DATA_DIR"
fi

if ! command -v npm >/dev/null; then
  echo "Node.js is needed for the web UI: install the LTS version from https://nodejs.org"; exit 1
fi
if [[ ! -d web/node_modules ]]; then
  echo "Installing web dependencies (first run only)…"
  (cd web && npm ci --no-fund --no-audit)
fi

./scripts/web-setup.sh

python -m uvicorn app.main:app --port 8000 --no-server-header &
API_PID=$!
# Stop everything this script started (the whole process group) on exit or Ctrl+C.
cleanup() { trap - EXIT INT TERM; kill 0 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "Waiting for the API on http://localhost:8000 (first start loads the models)…"
for _ in $(seq 1 600); do
  if curl -fsS http://localhost:8000/health >/dev/null 2>&1; then break; fi
  if ! kill -0 "$API_PID" 2>/dev/null; then echo "The API failed to start (see above)."; exit 1; fi
  sleep 1
done

echo "Open http://localhost:3000 in your browser."
# 127.0.0.1: only this computer can open the UI (not other devices on the Wi-Fi).
(cd web && NEXT_TELEMETRY_DISABLED=1 npx next dev --hostname 127.0.0.1 --port 3000) &
wait $!
