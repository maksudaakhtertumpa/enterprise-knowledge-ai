#!/usr/bin/env bash
# End-to-end check of the running compose stack, through the web UI's own routes.
#
#   docker compose up -d --wait && ./scripts/docker-smoke.sh
#
# Creates a throwaway workspace, connects like a browser (Origin header, cookie),
# uploads a policy document, waits for it to be processed, asks a question, and
# checks the hardening (non-root, headers, ports, docs off, cross-site blocked).
set -euo pipefail
cd "$(dirname "$0")/.."

WEB=http://127.0.0.1:3000
API=http://127.0.0.1:8000
JAR="$(mktemp)"
trap 'rm -f "$JAR"' EXIT
pass=0
ok() { pass=$((pass + 1)); printf '  ok    %s\n' "$1"; }
fail() { printf '  FAIL  %s\n' "$1" >&2; exit 1; }
json() { python3 -c "import json,sys; d=json.load(sys.stdin); print($1)"; }
flavor="$(docker compose exec -T api printenv EKA_IMAGE_FLAVOR)"

echo "[1] Containers"
[[ "$(docker compose exec -T api id -u)" == "10001" ]] && ok "api runs as an unprivileged user" || fail "api user"
[[ "$(docker compose exec -T web id -u)" != "0" ]] && ok "web runs as an unprivileged user" || fail "web user"
if docker compose exec -T api sh -c 'touch /app/x' 2>/dev/null; then fail "api filesystem is writable"; fi
ok "api root filesystem is read-only"

echo "[2] API"
headers="$(curl -fsS -D - -o /dev/null "$API/health")"
grep -qi '^x-content-type-options: nosniff' <<<"$headers" && ok "security headers" || fail "headers"
grep -qi '^server:' <<<"$headers" && fail "server header leaks" || ok "no server header"
[[ "$(curl -s -o /dev/null -w '%{http_code}' "$API/docs")" == "404" ]] && ok "/docs off in production" || fail "/docs"
ready="$(curl -fsS "$API/health/ready" | json 'd["status"]')"
[[ "$ready" == "ready" ]] && ok "readiness: $ready" || fail "readiness $ready"

echo "[3] Web UI session"
KEY="$(docker compose exec -T api python -m app.cli create-workspace "Docker smoke" | awk '/^API key/ {print $NF}')"
[[ ${#KEY} -eq 47 ]] && ok "workspace created" || fail "no key"
code="$(curl -s -o /dev/null -w '%{http_code}' -c "$JAR" -H "Origin: $WEB" -H 'Content-Type: application/json' \
  -d "{\"key\": \"$KEY\"}" "$WEB/api/session")"
[[ "$code" == "200" ]] && ok "connected through the web UI" || fail "session $code"
code="$(curl -s -o /dev/null -w '%{http_code}' -b "$JAR" -H 'Origin: http://evil.example' \
  -H 'Content-Type: application/json' -d '{"query":"x"}' "$WEB/api/eka/search")"
[[ "$code" == "403" ]] && ok "cross-site request blocked" || fail "cross-site $code"

echo "[4] Upload and process"
doc="$(curl -fsS -b "$JAR" -H "Origin: $WEB" \
  -F "file=@eval/datasets/teratai/corpus/leave-policy.txt;type=text/plain" "$WEB/api/eka/documents" | json 'd["id"]')"
ok "uploaded ($doc)"
status=pending
for _ in $(seq 1 180); do
  status="$(curl -fsS -b "$JAR" "$WEB/api/eka/documents/$doc" | json 'd["status"]')"
  [[ "$status" == "ready" || "$status" == "failed" ]] && break
  sleep 1
done
[[ "$status" == "ready" ]] && ok "processed: $status" || fail "document $status"

echo "[5] Ask"
start=$(date +%s)
answer="$(curl -fsS -b "$JAR" -H "Origin: $WEB" -H 'Content-Type: application/json' --max-time 300 \
  -d '{"question": "How many days of annual leave do staff with less than two years get?"}' "$WEB/api/eka/ask")"
seconds=$(($(date +%s) - start))
status="$(json 'd["status"]' <<<"$answer")"
[[ "$status" == "answered" ]] && ok "answered in ${seconds} s" || fail "ask: $status $(json 'd.get("reason")' <<<"$answer")"
cites="$(json 'len(d["citations"])' <<<"$answer")"
[[ "$cites" -ge 1 ]] && ok "$cites verified citation(s)" || fail "no citations"
if [[ "$flavor" == "ml" ]]; then
  json 'd["answer"]' <<<"$answer" | grep -q '14' && ok "answer contains 14 days" || fail "wrong answer"
  json 'd["verification"]["entailment"]["model"]' <<<"$answer" | grep -q 'mDeBERTa' && ok "entailment checked" || fail "entailment"
fi

echo
echo "All $pass Docker smoke checks passed ($flavor image)."
