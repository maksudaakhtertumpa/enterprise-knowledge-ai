# Enterprise Knowledge AI — Hybrid Search RAG Platform

![CI](https://github.com/maksudaakhtertumpa/enterprise-knowledge-ai/actions/workflows/ci.yml/badge.svg)

Upload company documents, search them with hybrid retrieval (BM25 + dense vectors, fused with RRF and reranked by a cross-encoder), and get answers grounded in the documents with **verified** source citations — or an explicit "insufficient evidence" response.

> **Status: Phase 4 of 8 complete.** Upload PDF/DOCX/TXT into isolated workspaces and search them with two-stage retrieval (dense embeddings, then a cross-encoder reranker), with page and section citations. Measured on a MacBook Air M4: MRR 0.962. Next: grounded answers with verified citations; see the [roadmap](#roadmap). Nothing is described as working until it is.

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full diagram, design decisions and milestone plan.

## Quick start

Requires **Python 3.11+** and Git.

**macOS / Linux** (developed on a MacBook M4; `brew install python@3.12` if needed)
```bash
git clone https://github.com/maksudaakhtertumpa/enterprise-knowledge-ai.git
cd enterprise-knowledge-ai
python3.12 -m venv .venv            # 3.11 also works
source .venv/bin/activate
pip install -r requirements-ml.lock && pip install -e . --no-deps   # includes PyTorch for embeddings
cp .env.example .env
./scripts/verify.sh                         # lint, tests, live smoke test, benchmark
python -m app.cli create-workspace "Demo"   # prints an API key once — copy it
uvicorn app.main:app --reload               # first start downloads the embedding model (~1.2 GB)
```
No GPU or no wish to download a model? Install `requirements-dev.lock` instead and set `EKA_EMBEDDING_BACKEND=hashing` (keyword-like stand-in for development; not semantic).

**Windows (PowerShell)**
```powershell
git clone https://github.com/maksudaakhtertumpa/enterprise-knowledge-ai.git
cd enterprise-knowledge-ai
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.lock; pip install -e . --no-deps
Copy-Item .env.example .env
pytest
python -m app.cli create-workspace "Demo"
uvicorn app.main:app --reload
```

Then open <http://localhost:8000/docs>, click **Authorize**, paste the key, and try uploading a document.

```bash
curl -H "X-API-Key: $EKA_KEY" -F "file=@handbook.pdf" http://localhost:8000/api/v1/documents
curl -H "X-API-Key: $EKA_KEY" "http://localhost:8000/api/v1/documents/<id>/chunks?limit=5"
curl -H "X-API-Key: $EKA_KEY" -H "Content-Type: application/json" \
     -d '{"query": "How many days of annual leave?", "top_k": 3}' \
     http://localhost:8000/api/v1/search
```

## Common commands

| Task | Command |
|---|---|
| Run API (reload) | `uvicorn app.main:app --reload` |
| Tests | `pytest` |
| Coverage | `pytest --cov --cov-report=term-missing` |
| Lint | `ruff check . && ruff format --check .` |
| Auto-format | `ruff format . && ruff check --fix .` |
| Verify everything | `./scripts/verify.sh` (or `--quick` for no model download) |
| Retrieval benchmark | `python -m eval.run_retrieval` |
| Fusion settings sweep | `python -m eval.sweep_fusion` |
| New workspace + key | `python -m app.cli create-workspace "Name"` |
| Re-embed after changing model | `python -m app.cli reindex` (API stopped) |
| List / revoke keys | `python -m app.cli list-workspaces` · `python -m app.cli revoke-key eka_XXXXXXXX` |

(`make run`, `make test`, `make check` etc. on macOS/Linux.)

## API (current)

Feature endpoints require an `X-API-Key` header; each key belongs to one workspace, and a workspace can only ever see its own documents.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness — process is up |
| GET | `/health/ready` | Readiness — data dir, database, vector store, embedding model (503 otherwise) |
| GET | `/api/v1/workspace` | Which workspace the key belongs to |
| POST | `/api/v1/documents` | Upload PDF/DOCX/TXT → `202`, processed in the background |
| GET | `/api/v1/documents` | List (paginated, `?status=` filter) |
| GET | `/api/v1/documents/{id}` | Status, metadata, page/chunk counts, warnings |
| GET | `/api/v1/documents/{id}/chunks` | Chunks with heading, page range and character offsets |
| POST | `/api/v1/documents/{id}/reprocess` | Re-run ingestion |
| DELETE | `/api/v1/documents/{id}` | Delete document, chunks, index entries and stored file |
| POST | `/api/v1/search` | `mode: bm25 \| dense \| hybrid` (default dense) with cross-encoder reranking (default on; `"rerank": false` for ~35 ms results); results with text, score, file, section, pages, offsets, per-retriever and pre-rerank ranks |

All errors share one shape:
```json
{"error": {"code": "not_found", "message": "...", "details": null, "request_id": "..."}}
```
Every response carries an `X-Request-ID` header matching the `request_id` in logs.

### Ingestion at a glance
- **Validation:** size limit enforced before buffering; content must match the extension (magic bytes, DOCX structure); zip-bomb guard; filenames sanitised and never used as paths.
- **Extraction:** PDF text with page numbers and bookmark sections (pypdf); DOCX paragraphs, headings and tables in reading order; TXT with encoding detection and Markdown headings.
- **Chunking:** section-aware, sentence-aligned, configurable size/overlap; every chunk is an exact slice of the stored text (character offsets kept for citations).
- **Duplicates:** SHA-256 per workspace → `409` with the existing document ID.

### Search at a glance
- **BM25:** SQLite FTS5, one index per workspace (isolation and per-tenant BM25 statistics), stemming, accent folding, headings weighted 2×, injection-safe query building.
- **Dense:** `Qwen/Qwen3-Embedding-0.6B` by default, on the Apple-silicon GPU (MPS) when available; ChromaDB with one collection per workspace; chunks embedded with their document title and section.
- **Hybrid:** Reciprocal Rank Fusion of both lists. Uses ranks, not scores, so BM25's unbounded scores and cosine similarities never need normalising. Defaults (k = 10, BM25 weight 0.25) come from a 36-setting sweep on the M4. Every hybrid result reports its rank in each retriever.
- **Reranking:** cross-encoder second stage (`BAAI/bge-reranker-v2-m3`, float16 on the Apple GPU) that reorders the top 10 candidates from any mode, on by default. Each result keeps its pre-rerank rank.
- **Safety:** indexes return IDs only; text and metadata come from SQLite, re-checked for workspace and `ready` status. Changing the embedding model is detected and needs a re-index.
- **Measured on a MacBook Air M4** ([details](eval/README.md)):

  | Pipeline | MRR | Recall@1 | Recall@5 | p50 latency |
  |---|---|---|---|---|
  | BM25 | 0.538 | 0.500 | 0.553 | 0.8 ms |
  | Dense (Qwen3-Embedding-0.6B) | 0.900 | 0.819 | 0.979 | 36 ms |
  | Hybrid RRF (best of 36 settings) | 0.879 | 0.798 | 0.979 | 36 ms |
  | **Dense + cross-encoder rerank (default)** | **0.962** | **0.925** | **1.000** | 335 ms |

  Fusion could only trade identifier precision for paraphrase quality; the reranker improved both (lexical MRR 1.000, paraphrase 0.955) and handles Malay questions over English documents.

## Configuration

All settings are environment variables prefixed `EKA_`; see [.env.example](.env.example). Invalid values stop the app at startup with a message naming the field.

## Roadmap

- [x] **Phase 1** — Project setup, config, structured logging, error handling, health checks, tests, CI
- [x] **Phase 2** — Document ingestion (PDF/DOCX/TXT), structure-aware chunking, dedupe, workspace API keys
- [x] **Phase 3** — BM25 (SQLite FTS5), dense (ChromaDB) and hybrid (RRF) retrieval, benchmark on Apple silicon
- [x] **Phase 4** — Fusion tuning and cross-encoder reranking, defaults chosen from M4 measurements
- [ ] **Phase 5** — Grounded answers, citation verification, prompt-injection defences
- [ ] **Phase 6** — Streamlit UI
- [ ] **Phase 7** — Evaluation (Recall@k, MRR, faithfulness, latency), security hardening
- [ ] **Phase 8** — Docker Compose, AWS deployment, CI/CD

## Project docs
- [Architecture & plan](docs/ARCHITECTURE.md)
- [Model choices for Apple silicon](docs/MODELS.md)
- [Phase 1 notes](docs/phases/phase-1-setup.md)
- [Phase 2 notes](docs/phases/phase-2-ingestion.md)
- [Phase 3 notes](docs/phases/phase-3-retrieval.md)
- [Phase 4 notes](docs/phases/phase-4-reranking.md)
- [Retrieval evaluation](eval/README.md)

## License
MIT
