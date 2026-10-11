# Security notes and threat model

This is a portfolio project that runs on one machine. It has **not** had an
independent security review, and it is not "enterprise-secure". This document says
what it defends against, how, which test checks each control, and what it does not
cover. If you find a problem, please report it privately through GitHub's
*Security → Report a vulnerability* page of this repository rather than in a public
issue.

## What is being protected

| Asset | Where it lives |
|---|---|
| Uploaded documents and their chunks | `data/uploads/`, SQLite (`data/eka.sqlite3`), Chroma (`data/chroma/`) |
| Workspace API keys | SHA-256 hashes in SQLite; the plaintext is shown once at creation |
| Question history (optional, `EKA_ASK_HISTORY`) | SQLite |
| The machine itself | API, web UI, Ollama and models all run locally |

## Trust boundaries

```
Browser ──(cookie, same-site only)──▶ Next.js server :3000 ──(X-API-Key)──▶ FastAPI :8000
                                                                             │
                                  SQLite · Chroma (embedded) · files ◀───────┤
                                  Ollama :11434 (local LLM) ◀────────────────┤
                                  NLI / embedding / reranker models (in process)
```

- **The browser is untrusted.** It never sees the API key: the key is sealed
  (AES-256-GCM) in an `httpOnly`, `SameSite=Strict` cookie that only the Next.js
  server can read.
- **Document text is untrusted.** Anyone who can upload a file controls text that
  later reaches the LLM prompt and the UI.
- **Model output is untrusted.** The LLM can invent passages, quotes and claims.
- Both servers listen on `127.0.0.1` by default. Exposing them on a network needs
  the TLS reverse proxy described for Phase 8.

## Threats and controls

| Threat | Control | Checked by |
|---|---|---|
| Stealing the API key through script in the page (XSS) | Key never sent to the browser; strict CSP on the web UI; untrusted text rendered only as React text nodes, never HTML | `web/e2e/app.spec.ts` (key absent from DOM, cookies, storage; planted `<img onerror>` / `<script>` never runs) |
| Cross-site request forgery | `SameSite=Strict` cookie, plus an `Origin` check on every state-changing request | `routes.test.ts`, e2e "proxy only forwards allowed calls" |
| Using the web server as an open proxy | 13 allowlisted method + path pairs; path segments must be plain or UUIDs | `routes.test.ts` allowlist table, e2e path-traversal cases |
| Reading another workspace's documents | Every query is scoped to the workspace the key resolves to; separate BM25 and vector collections per workspace; IDs from another workspace return 404 | `tests/api/test_auth_and_isolation.py` |
| Guessing keys | 256-bit random keys; one generic error for every failure; failed attempts limited per client address (`EKA_RATE_LIMIT_AUTH_FAILURES_*`) | `test_rate_limits_and_headers.py` |
| One key exhausting the server (LLM calls, uploads, searches) | Per-workspace token buckets with `429` + `Retry-After` (`EKA_RATE_LIMIT_*`) | `test_rate_limits_and_headers.py` |
| Huge or hostile files | Body-size limit enforced before buffering; PDF page cap; DOCX zip-bomb guard; content checked against the claimed type; filenames sanitised | `tests/api/test_documents_api.py`, `tests/ingestion/` |
| Prompt injection in documents | Passages fenced and labelled as untrusted data; instruction-like text flagged; answers must cite verifiable quotes; the injection benchmark checks payloads don't leak | `tests/api/test_ask_api.py`, `eval/datasets/injection` (4 attacks: a smoke test, not a guarantee) |
| Confident but unsupported answers | Citation verification (the quote must exist in the cited passage) and the NLI entailment check (the sentence must follow from it) | `tests/generation/`, `tests/api/test_ask_api.py`, answer benchmark |
| Leaking internals or data in errors and logs | One error envelope, no stack traces, validation errors don't echo input; access log records method, path, status, workspace ID only | `test_api_keys_and_questions_never_reach_the_logs`, `tests/test_errors_and_middleware.py` |
| Clickjacking, MIME sniffing, caching of responses | API sends `X-Content-Type-Options`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, `Cache-Control: no-store`, `CSP: default-src 'none'`; `uvicorn --no-server-header` | `test_every_api_response_carries_security_headers` |
| Interactive API docs exposing the attack surface | `/docs`, `/redoc` and `/openapi.json` are off when `EKA_ENVIRONMENT=production` (override with `EKA_API_DOCS`) | `test_docs_pages_skip_the_csp_and_are_off_in_production` |
| Vulnerable dependencies | Exact pins in lock files; `pip-audit` and `npm audit` in CI; findings below | CI jobs `audit` and `web` |

## Accepted findings from dependency scans

Each was checked against how this project uses the package. Re-check them whenever
the lock files change; new findings fail CI.

| Package | Advisory | Why it is accepted here |
|---|---|---|
| chromadb 1.5.9 | PYSEC-2026-311 (CVE-2026-45829), PYSEC-2026-3813, PYSEC-2026-3814, PYSEC-2026-3815 | All concern Chroma's HTTP **server** (`/api/v2/...` endpoints, `trust_remote_code` embedding configs, its RBAC provider). This project uses Chroma only as an embedded library (`PersistentClient`, `embedding_function=None`) and never starts that server. No fixed release exists yet. Never expose a Chroma server from this deployment. |
| braces 3.0.3 (web, dev only) | GHSA-vfj7-8cjw-p6xm | Pulled in by the ESLint plugin; it only expands glob patterns written in this repository, never user input, and isn't part of the production build. No fixed release exists yet. |

## Known limitations

- **Rate limits are per process.** Buckets live in memory. That fits the single
  uvicorn process this project runs; several workers or replicas would need a shared
  store such as Redis, and a restart resets the counters. A rate-limited upload has
  already been received (up to the size limit) before the 429 is returned.
- **Behind the web UI, every request comes from 127.0.0.1**, so the auth-failure
  limit is shared by everyone using the UI. Behind a real reverse proxy, list it in
  `EKA_TRUSTED_PROXIES` so the client address comes from `X-Forwarded-For`.
- **No TLS locally.** Fine on `127.0.0.1`; anything else needs the Phase 8 reverse
  proxy with HTTPS and HSTS.
- **No user accounts or roles.** A workspace key gives full access to that
  workspace. Keys can be revoked (`python -m app.cli revoke-key`), not scoped.
- **No encryption at rest** beyond the operating system's (use FileVault on macOS).
  Question history is stored in plain text unless `EKA_ASK_HISTORY=false`.
- **Models are pinned by name, not by revision hash**, and are downloaded from
  Hugging Face on first use. None of them is loaded with `trust_remote_code`.
- **Prompt-injection defences are heuristic.** Pattern flagging can be evaded;
  the real protection is that every claim must cite a verifiable quote, and even
  that cannot stop a planted document from being quoted accurately.
- **The entailment model can be wrong both ways.** On the held-out set it removed a
  correct sentence that needed world knowledge (that a mother is "immediate family"),
  and it cannot catch a true sentence that doesn't answer the question. Its
  trade-off is measured in `docs/evaluation/`.
