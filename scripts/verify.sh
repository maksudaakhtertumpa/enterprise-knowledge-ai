#!/usr/bin/env bash
# One-command verification on your machine (written for the MacBook M4).
#
#   ./scripts/verify.sh           # lint + tests + live smoke test + benchmark (real model)
#   ./scripts/verify.sh --quick   # same, but with the offline hashing embedder (no download)
#
# Both finish with the web UI checks (needs Node.js). Browser tests: `make e2e`.
#
# The first full run downloads the embedding model (~1.2 GB), the reranker (~2.3 GB) and
# the entailment model (~1.1 GB) into ~/.cache/huggingface, in the "Prepare models" step.
set -euo pipefail
cd "$(dirname "$0")/.."

QUICK=0
[[ "${1:-}" == "--quick" ]] && QUICK=1

step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

step "Environment"
python - <<'PY'
import platform, sys
assert sys.version_info >= (3, 11), "Python 3.11+ required"
print(f"Python {platform.python_version()} on {platform.platform()} ({platform.machine()})")
try:
    import torch
    mps = torch.backends.mps.is_available()
    print(f"torch {torch.__version__}; Apple GPU (MPS) available: {mps}")
except ImportError:
    print("torch not installed (install the ML extra for real embeddings)")
PY

step "Ruff lint + format check"
ruff check .
ruff format --check .

step "Test suite"
pytest -q

if [[ $QUICK -eq 1 ]]; then
  step "Live smoke test (hashing embedder)"
  python scripts/smoke_test.py --backend hashing
  step "Benchmark (BM25 only — hashing is not semantic)"
  python -m eval.run_retrieval --backend hashing --modes bm25 --label quick-bm25
else
  step "Prepare models (first run downloads ~4.6 GB; also checks Ollama)"
  python -m app.cli prepare-models
  step "Live smoke test (real embedding model)"
  python scripts/smoke_test.py --backend sentence-transformers
  step "Retrieval benchmark"
  python -m eval.run_retrieval
  step "Answer benchmark (local LLM; several minutes)"
  python -m eval.run_answers
  step "Held-out benchmarks (questions never used for tuning)"
  python -m eval.run_retrieval --dataset eval/datasets/teratai-heldout
  python -m eval.run_answers --dataset eval/datasets/teratai-heldout --no-injection
  step "Evaluation report (docs/evaluation/)"
  python -m eval.report
fi

step "Web UI: format, lint, types, unit tests, production build"
if command -v npm >/dev/null; then
  if [[ ! -d web/node_modules ]]; then (cd web && npm ci --no-fund --no-audit); fi
  (cd web && NEXT_TELEMETRY_DISABLED=1 npm run check)
else
  echo "Node.js not found: install the LTS version from https://nodejs.org to check the web UI."
  exit 1
fi

step "All checks passed"
