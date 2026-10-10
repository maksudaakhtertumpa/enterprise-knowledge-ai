# Phase 2 — Document Ingestion, Extraction, Metadata and Chunking

## What was built

```
POST /documents ──► validate ──► store file ──► DB row (pending) ──► 202 Accepted
                                                       │
                                     background job ───┘
                                           │
     extract (PDF/DOCX/TXT) ─► normalise ─► assemble ─► chunk ─► chunks in SQLite (ready)
```

| Area | Files |
|---|---|
| Auth & workspaces | `app/core/security.py`, `app/cli.py` |
| Database | `app/db/models.py`, `app/db/session.py`, `app/repositories/documents.py` |
| Upload safety | `app/services/ingestion/validation.py`, `app/services/storage.py`, `BodySizeLimitMiddleware` in `app/core/middleware.py` |
| Extraction | `app/services/ingestion/extractors/{pdf,docx,txt}.py` |
| Normalisation | `app/services/ingestion/normalize.py` |
| Chunking | `app/services/ingestion/chunking.py` |
| Orchestration | `app/services/ingestion/pipeline.py` |
| API | `app/api/routes/documents.py`, `app/api/routes/workspace.py` |
| Tests | `tests/ingestion/*`, `tests/api/*` — 143 tests total, ~95% line coverage |

### Endpoints (all need `X-API-Key`)
| Method | Path | Notes |
|---|---|---|
| GET | `/api/v1/workspace` | Which workspace this key belongs to |
| POST | `/api/v1/documents` | Multipart upload → `202` + document (status `pending`) |
| GET | `/api/v1/documents?limit&offset&status` | Newest first, paginated, filterable |
| GET | `/api/v1/documents/{id}` | Status, metadata, counts, warnings |
| GET | `/api/v1/documents/{id}/chunks?limit&offset` | Inspect chunks with page/heading/offsets |
| POST | `/api/v1/documents/{id}/reprocess` | Re-run ingestion (e.g. new chunk settings) |
| DELETE | `/api/v1/documents/{id}` | Row, chunks and stored file |

## Design decisions

**Workspace isolation is structural.** The API key resolves to a workspace; route handlers never accept a workspace ID from the client. `DocumentRepository` has no method that reads without a `workspace_id`. Another workspace's document returns **404, not 403** — a 403 would confirm it exists. Duplicate detection is per workspace for the same reason: a cross-workspace 409 would leak that someone else holds the file.

**API keys, hashed with SHA-256.** Keys carry 256 random bits, so fast hashing is correct here (bcrypt/argon2 protect *low-entropy* passwords from guessing). Only the hash and a 12-character display prefix are stored; the CLI shows the key once.

**Never trust the upload.**
- Size: `BodySizeLimitMiddleware` rejects by `Content-Length` before the body is read, and counts streamed bytes for chunked uploads; storage enforces the exact file limit while streaming.
- Type: extension picks the expected type, then the bytes must match (`%PDF-` header; DOCX must be a zip containing `word/document.xml`; TXT must not contain NUL bytes). The client's `Content-Type` header is recorded but never used for decisions.
- Paths: files are stored as `{workspace_id}/{document_id}.{ext}` — the user's filename is display-only and sanitised, so path traversal is impossible by construction.
- Zip bombs: DOCX archives are checked for total uncompressed size and entry count *before* any XML parser runs. python-docx parses with entity resolution disabled (no XXE).
- PDFs: page cap; encrypted files that open with an empty password are read, others fail with a clear message.

**Background processing with honest status.** Upload returns `202` as soon as the file is safely stored; ingestion runs in `BackgroundTasks` with its own DB session. Status moves `pending → processing → ready | failed`. Expected failures (encrypted, scanned/no text, corrupt) give the user an actionable message plus an `error_code`; unexpected crashes are logged with a stack trace and shown only as a generic error. On restart, jobs left `pending/processing` are marked failed ("interrupted — reprocess"), so nothing is stuck forever. *Trade-off:* BackgroundTasks run in the API process; a task queue (RQ/Celery/SQS) is the upgrade when ingestion volume or isolation demands it.

**One canonical text, offsets everywhere.** Extraction yields blocks (paragraph/heading/table + page + heading). These are normalised (NFKC folds PDF ligatures like `ﬁ`; hyphenated line breaks repaired; control/zero-width characters removed) and joined into one string per document, stored as `extracted_text`. Every chunk records `char_start/char_end` into it, and a test enforces `extracted_text[start:end] == chunk.text`. Phase 5's citation verifier depends on this.

**Structure-aware chunking** (the "intelligent" part):
1. A new section always starts a new chunk — at a DOCX/Markdown heading, or wherever a PDF bookmark's section changes. Chunks never mix sections, so `heading` is a meaningful citation label.
2. Within a section, whole sentences (or table rows) are packed up to `chunk_size_tokens`. The sentence splitter knows common abbreviations including Malaysian ones (`Sdn.`, `Bhd.`).
3. Over-long sentences fall back to word windows; a single giant token (base64, a URL) to character windows.
4. Overlap carries whole trailing sentences (≤ `chunk_overlap_tokens`) into the next chunk, never across sections. If a single sentence exceeds the overlap budget, no overlap is added rather than cutting a sentence.

**Page numbers vs headings.** PDFs give 1-based `page_start/page_end`. DOCX and TXT have no fixed pages (Word paginates at render time), so pages are `null` and citations will use the heading. Inventing page numbers would violate the "never invent page numbers" requirement.

**Token counting is approximate in Phase 2** (word + punctuation count, with long strings penalised). Phase 3 replaces it with the embedding model's own tokenizer, and the chunk metadata records which counter produced the chunks (`tokenizer: "approx"`), so reprocessing is traceable.

**Library choices.** `pypdf` (BSD, pure Python) instead of PyMuPDF (faster, but AGPL — it would pull AGPL terms into an MIT project). `python-docx` for DOCX. SQLAlchemy 2.0 ORM so the Postgres move is a URL change plus migrations.

## Measured on the sandbox (not your Mac)
A generated 300-page PDF (626K characters): extraction 1.3 s, chunking 0.12 s, 226 chunks. Three concurrent uploads processed without SQLite lock errors (WAL mode + busy timeout). Your M4 should be faster; re-measure in Phase 7.

## Try it
```bash
python -m app.cli create-workspace "Acme HR"          # copy the printed key
export EKA_KEY=eka_...                                 # your key
uvicorn app.main:app --reload

curl -H "X-API-Key: $EKA_KEY" -F "file=@handbook.pdf" http://localhost:8000/api/v1/documents
curl -H "X-API-Key: $EKA_KEY" http://localhost:8000/api/v1/documents
curl -H "X-API-Key: $EKA_KEY" "http://localhost:8000/api/v1/documents/<id>/chunks?limit=5"
```
Or use `http://localhost:8000/docs` → **Authorize** → paste the key → try the endpoints.

## Likely errors and fixes
| Symptom | Cause | Fix |
|---|---|---|
| `401 unauthorized` | Header missing/misspelled, or key revoked | Header is exactly `X-API-Key`; `python -m app.cli list-workspaces` |
| `415 content_mismatch` | File renamed to a different extension, or `.doc` saved as `.docx` | Re-save in the real format |
| Document `failed`, `error_code: encrypted` | PDF needs a password | Remove the password, re-upload |
| `failed`, `error_code: no_text` | Scanned PDF (images only) | OCR is on the roadmap; use a text PDF for now |
| `409 conflict` | Identical bytes already uploaded | Use `existing_document_id`, or delete the old one |
| `413 payload_too_large` | Over `EKA_MAX_UPLOAD_MB` | Raise the limit in `.env` or split the file |
| `database is locked` | Another process holds a long write | Stop duplicate servers; WAL mode normally prevents this |
| Strange spacing in PDF chunks | PDF stores glyphs, not paragraphs | Inspect `/chunks`; layout-aware extraction is a roadmap item |
| Changed chunk settings had no effect | Existing documents keep old chunks | `POST /documents/{id}/reprocess` |

## Acceptance criteria
- [x] Upload PDF, DOCX, TXT; each reaches `ready` with chunks (tested per format)
- [x] PDF chunks carry correct page ranges; DOCX/TXT carry headings, pages `null`
- [x] Every chunk is an exact slice of the stored text (tested, including randomised documents)
- [x] Chunks never exceed the configured size; every character is covered
- [x] Duplicate upload → 409 with the existing ID; same file allowed in another workspace
- [x] Wrong type, mismatched content, empty, oversize → rejected and nothing left on disk
- [x] Encrypted / scanned / corrupt → `failed` with an actionable message; crashes don't leak internals
- [x] Cross-workspace access → 404 on every endpoint; listing is isolated
- [x] Delete removes row, chunks and file; restart marks interrupted jobs failed
- [x] Lint clean; 143 tests pass on Python 3.11 and 3.12

## Interview talking points
- "I enforce tenant isolation at the repository layer and return 404 rather than 403, so the API can't be used to probe for other tenants' documents — and I have tests that try."
- "Uploads are validated by content, not by extension or Content-Type, with zip-bomb and size checks that happen before parsing."
- "Ingestion is asynchronous with explicit status, actionable error codes, and crash recovery on restart."
- "Every chunk stores character offsets into a canonical text, which is what makes citation verification possible later."
- "My chunker is section-aware and sentence-aligned, and I test its invariants with randomised documents, not just examples."
- "I chose pypdf over PyMuPDF for licensing reasons — PyMuPDF is AGPL."
