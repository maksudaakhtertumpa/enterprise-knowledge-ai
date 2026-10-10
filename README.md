# Enterprise Knowledge AI — Hybrid Search RAG Platform

![CI](https://github.com/maksudaakhtertumpa/enterprise-knowledge-ai/actions/workflows/ci.yml/badge.svg)

Upload company documents through a web UI or the API, search them with two-stage retrieval (dense embeddings + cross-encoder reranking, with BM25 and RRF hybrid available), and get answers grounded in the documents with **verified** source citations — or an explicit "insufficient evidence" response.

**Runs entirely on a laptop for $0:** open-source models (Qwen3-Embedding, bge-reranker, `qwen3.5:4b` via Ollama), a FastAPI backend and a Next.js + Tailwind CSS + shadcn/ui web app. No paid APIs, no cloud account needed.

> **Status: Phase 6 of 8 complete** (Next.js web UI with verified citations, checked on a MacBook Air M4). Grounded answers from a local LLM, with every citation checked against the retrieved text, measured on a MacBook Air M4: 29 of 30 questions answered with verified citations, 11 of 12 unanswerable questions refused, 4 of 4 prompt-injection attacks resisted, about 5 s per answer, $0. See the [roadmap](#roadmap). Nothing is described as working until it is.

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full diagram, design decisions and milestone plan.

## Quick start

Requires **Python 3.11+**, **[Node.js](https://nodejs.org) 20.9+** (LTS) for the web UI, Git and, for answers, **[Ollama](https://ollama.com/download)** (free) with `ollama pull qwen3.5:4b`.

**macOS / Linux** (developed on a MacBook M4; `brew install python@3.12` if needed)
```bash
git clone https://github.com/maksudaakhtertumpa/enterprise-knowledge-ai.git
cd enterprise-knowledge-ai
python3.12 -m venv .venv            # 3.11 also works
source .venv/bin/activate
pip install -r requirements-ml.lock && pip install -e . --no-deps   # includes PyTorch for embeddings
cp .env.example .env
ollama pull qwen3.5:4b                      # local LLM for answers (~3.4 GB, free)
make web-install                            # web UI dependencies (exact versions, npm ci)
./scripts/verify.sh                         # lint, tests, live smoke test, benchmarks, web checks
python -m app.cli create-workspace "Demo"   # prints an API key once — copy it
make dev                                    # API + web UI; open http://localhost:3000
```
No GPU or no wish to download models? Install `requirements-dev.lock` instead and set `EKA_EMBEDDING_BACKEND=hashing`, `EKA_RERANKER_BACKEND=hashing` and `EKA_LLM_BACKEND=extractive`. These are offline stand-ins for development: not semantic, not generative.

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
# second PowerShell window:
cd web; npm ci; npx next dev --port 3000
```

### Web UI
Open <http://localhost:3000>, paste the key, upload documents (the policy files in `eval/datasets/teratai/corpus/` make a good demo), then ask questions. `make dev` starts the API and the UI together; `make run` and `make web` start them separately. See [Phase 6 notes](docs/phases/phase-6-web-ui.md).

### API
Open <http://localhost:8000/docs>, click **Authorize**, paste the key, and try uploading a document.

```bash
curl -H "X-API-Key: $EKA_KEY" -F "file=@handbook.pdf" http://localhost:8000/api/v1/documents
curl -H "X-API-Key: $EKA_KEY" "http://localhost:8000/api/v1/documents/<id>/chunks?limit=5"
curl -H "X-API-Key: $EKA_KEY" -H "Content-Type: application/json" \
     -d '{"query": "How many days of annual leave?", "top_k": 3}' \
     http://localhost:8000/api/v1/search
curl -H "X-API-Key: $EKA_KEY" -H "Content-Type: application/json" \
     -d '{"question": "How many days of annual leave do I get?"}' \
     http://localhost:8000/api/v1/ask
```

## Common commands

| Task | Command |
|---|---|
| Run API (reload) | `uvicorn app.main:app --reload` |
| Run web UI (API running) | `make web` (`cd web && npx next dev --port 3000`) |
| API + UI together | `./scripts/dev.sh` (`--offline` for no models / no Ollama) |
| Web UI checks | `make web-check` (Prettier, ESLint, TypeScript, Vitest, build) |
| Web UI browser tests | `make e2e` (Playwright; first `cd web && npx playwright install chromium`) |
| Tests | `pytest` |
| Coverage | `pytest --cov --cov-report=term-missing` |
| Lint | `ruff check . && ruff format --check .` |
| Auto-format | `ruff format . && ruff check --fix .` |
| Verify everything | `./scripts/verify.sh` (or `--quick` for no model download) |
| Retrieval benchmark | `python -m eval.run_retrieval` |
| Fusion settings sweep | `python -m eval.sweep_fusion` |
| Answer benchmark (needs Ollama) | `python -m eval.run_answers` |
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
| POST | `/api/v1/ask` | Grounded answer with verified citations (quote, file, section, pages, offsets), or `insufficient_evidence` with a reason |
| GET | `/api/v1/ask/history` | Question history for this workspace; `/ask/history/{id}` replays an answer |
| GET | `/api/v1/evidence/{chunk_id}` | A cited passage with surrounding document text |
| GET | `/api/v1/analytics?days=30` | Usage: questions, answer rate, refusal reasons, latency p50/p95, citation acceptance, tokens ($0), per-day counts, documents |
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

### Answers at a glance
- **Local LLM:** `qwen3.5:4b` through Ollama, temperature 0, JSON-only replies, explicit 8K context window. $0 per question.
- **Citations can't be invented:** the model cites passages by number with a short quote. Each quote is checked word for word against the passage it was shown, and file names, sections and page numbers are attached from the database. A near-exact quote may differ only in filler words, never in numbers or negations.
- **Unsupported sentences are removed.** If nothing verifiable remains, the answer is `insufficient_evidence` with a reason (no passages, weak retrieval, model declined, nothing verified).
- **Prompt injection:** document text is fenced with a random value per request, treated as data, flagged when it looks like instructions, and can only influence answers through verified quotes. The residual risk, a model quoting a malicious document from the same workspace, is measured by a planted-attack benchmark ([details](docs/phases/phase-5-grounded-qa.md)).
- **Measured on a MacBook Air M4** with `qwen3.5:4b` ([details](eval/README.md)):

  | Answer rate | Fact recall | Citation precision | Correct refusals | Injection attacks resisted | p50 latency | Cost |
  |---|---|---|---|---|---|---|
  | 0.967 | 0.883 | 0.906 | 0.917 | 4 / 4 | 5.4 s | $0 |

  The one wrong answer (a "trick" question) cited a genuine quote but claimed more than it said. Quote verification can't catch that, so an entailment check is on the roadmap.

### Web UI at a glance
- **Stack:** Next.js 16 (App Router), React 19, Tailwind CSS 4, shadcn/ui on Radix UI; Vitest and Playwright tests.
- **Pages:** Ask (chat with citation chips), Search (three-retriever comparison), Documents (drag-and-drop upload, live status, chunks, delete), History (replay answers), Analytics, Connect (backend status).
- **Checkable answers:** each citation chip opens the verified quote highlighted in the passage the model saw, then in its surrounding document text. "How this answer was checked" shows what was verified, removed or rejected.
- **Key never in the browser:** the Next.js server keeps the workspace key in an encrypted, httpOnly, SameSite=Strict cookie and proxies an allowlist of API calls. Untrusted text renders only as React text, behind a strict Content Security Policy.

## Cost

The project is designed to run for **$0**: every model is open-weight and runs locally, retrieval uses SQLite and ChromaDB on disk, the web UI is open-source Next.js running on your machine (telemetry off), and CI uses GitHub Actions' free minutes for public repositories. No API keys, no paid services, no cloud account.

## Configuration

All settings are environment variables prefixed `EKA_`; see [.env.example](.env.example). Invalid values stop the app at startup with a message naming the field.

## Roadmap

- [x] **Phase 1** — Project setup, config, structured logging, error handling, health checks, tests, CI
- [x] **Phase 2** — Document ingestion (PDF/DOCX/TXT), structure-aware chunking, dedupe, workspace API keys
- [x] **Phase 3** — BM25 (SQLite FTS5), dense (ChromaDB) and hybrid (RRF) retrieval, benchmark on Apple silicon
- [x] **Phase 4** — Fusion tuning and cross-encoder reranking, defaults chosen from M4 measurements
- [x] **Phase 5** — Grounded answers (local LLM), citation verification, prompt-injection defences, answer benchmark on Apple silicon
- [x] **Phase 6** — Web UI (Next.js + Tailwind CSS + shadcn/ui): chat with verified citations, search comparison, document management, history, analytics
- [ ] **Phase 7** — Evaluation (Recall@k, MRR, faithfulness, latency), security hardening; sentence-level entailment check for answers (local NLI model)
- [ ] **Phase 8** — Docker Compose, AWS deployment, CI/CD

## Project docs
- [Architecture & plan](docs/ARCHITECTURE.md)
- [Model choices for Apple silicon](docs/MODELS.md)
- [Phase 1 notes](docs/phases/phase-1-setup.md)
- [Phase 2 notes](docs/phases/phase-2-ingestion.md)
- [Phase 3 notes](docs/phases/phase-3-retrieval.md)
- [Phase 4 notes](docs/phases/phase-4-reranking.md)
- [Phase 5 notes](docs/phases/phase-5-grounded-qa.md)
- [Phase 6 notes](docs/phases/phase-6-web-ui.md)
- [Retrieval evaluation](eval/README.md)

## License
MIT
