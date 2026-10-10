#!/usr/bin/env bash
# One-command verification on your machine (written for the MacBook M4).
#
#   ./scripts/verify.sh           # lint + tests + live smoke test + benchmark (real model)
#   ./scripts/verify.sh --quick   # same, but with the offline hashing embedder (no download)
#
# The first full run downloads the embedding model (~1.2 GB) and the reranker (~2.3 GB)
# into ~/.cache/huggingface, in the "Prepare models" step, with progress bars.
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
  step "Prepare models (first run downloads ~3.5 GB; progress bars below)"
  python -m app.cli prepare-models
  step "Live smoke test (real embedding model)"
  python scripts/smoke_test.py --backend sentence-transformers
  step "Benchmark: BM25 vs dense"
  python -m eval.run_retrieval
fi

step "All checks passed"
