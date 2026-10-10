# Phase 3 — Keyword, Semantic and Hybrid Retrieval

## What was built

```
                    ┌──────── SQLite (source of truth: chunks, documents) ───────┐
ingest ─► chunks ──►│  fts_<workspace>   FTS5 BM25 index  (same transaction)      │
                    └─────────────────────────────────────────────────────────────┘
               └──► embed (title + section + text, MPS) ─► Chroma  ws-<workspace>

POST /search ─► bm25:   FTS5 MATCH ─────────┐
              ├ dense:  embed query ─► Chroma ┼─► chunk IDs ─► hydrate from SQLite
              └ hybrid: both lists ───────────┘   (workspace + status=ready filter)
                                                   └─► hybrid: RRF over the *visible* lists
```

| Area | Files |
|---|---|
| Embeddings (MPS / CUDA / CPU, lazy, thread-safe) | `app/services/retrieval/embeddings.py` |
| BM25 (SQLite FTS5, one table per workspace) | `app/services/retrieval/keyword.py` |
| Vector store (ChromaDB, one collection per workspace) | `app/services/retrieval/vector_store.py` |
| Contextual embedding + re-indexing | `app/services/retrieval/indexing.py` |
| Reciprocal Rank Fusion (pure function) | `app/services/retrieval/fusion.py` |
| Search orchestration + hydration + fusion | `app/services/retrieval/search.py` |
| Service container (DI, no globals) | `app/services/container.py` |
| API | `app/api/routes/search.py`, `app/schemas/search.py` |
| Pipeline step 3 (embed before `ready`) | `app/services/ingestion/pipeline.py` |
| Benchmark | `eval/` (dataset, metrics, runner) |
| Verification on your Mac | `scripts/verify.sh`, `scripts/smoke_test.py` |

### API
`POST /api/v1/search` (needs `X-API-Key`)
```json
{"query": "How many days of annual leave?", "mode": "hybrid", "top_k": 5,
 "document_ids": ["optional", "filter"]}
```
Each result carries `rank`, `score`, `chunk_id`, `text`, and `source` = {`document_id`, `filename`, `title`, `file_type`, `heading`, `page_start`, `page_end`, `char_start`, `char_end`}. BM25 and hybrid results also list `matched_terms`. Hybrid results add `retriever_ranks` (e.g. `{"bm25": 3, "dense": 1}`; a missing key means that retriever didn't return the chunk) and `retriever_scores`, so every fused result can be explained. `mode` defaults to `hybrid`. The response includes `embedding_model` and per-stage `timings_ms` (`bm25_ms`, `embed_ms`, `dense_ms`, `hydrate_ms`, `fuse_ms`, `total_ms`).

## Engineering decisions

**1. SQLite stays the source of truth; indexes return only IDs.** FTS5 and Chroma return chunk IDs and scores. Text, page numbers and headings come from one SQLite query that filters on the workspace again and on `status = ready`. Consequences: indexes can never disagree with the database about what a chunk says; a stale index entry (crash between steps) is silently dropped rather than shown; and a document is invisible to search until *both* indexes hold it.

**2. One index per workspace, in both engines.** FTS5 table `fts_<id>` and Chroma collection `ws-<id>`. Isolation then holds by construction, not by remembering a filter. For BM25 there's a second, subtler reason: BM25 statistics (IDF, average length) are computed over the whole table. In a shared table, tenant A's scores would shift when tenant B uploads documents. That hurts ranking quality and creates a side channel: A could probe how common a term is in other tenants' data. A test floods one workspace and checks the other's scores don't move.

**3. FTS5 query safety.** User text is reduced to word tokens, and each is double-quoted (`"annual" OR "leave"`), so FTS5 operators (`NEAR`, `AND`, `column:`, `*`, `^`) can't be injected. Table names are interpolated only after strict hex validation; every value is a bound parameter. Tests throw hostile queries at it.

**4. BM25 tuning choices (not tuned to the benchmark).** Tokenizer `porter unicode61 remove_diacritics 2`: Unicode-aware, accent-folding, English stemming (leaves → leave). Headings weighted 2× body text. Common English function words are dropped from queries only when real terms remain. Trade-off: the Porter stemmer is English-only, so a Malay-heavy corpus may prefer plain `unicode61`.

**5. Contextual embedding.** Chunks are embedded as `"<title> — <section>\n\n<text>"`. A chunk saying "This applies to all grades" is meaningless alone; with its header it sits near questions about the right topic. Only the embedding input changes, not the stored text. BM25 gets the same benefit through its weighted `heading` column.

**6. Apple-silicon embedding.** `EKA_EMBEDDING_DEVICE=auto` resolves to CUDA, then MPS, then CPU. The model loads once at startup (`EKA_EMBEDDING_PRELOAD`), off the event loop. Inference is serialised with a lock, because MPS isn't reliably safe under concurrent calls and background ingestion shares the model with live searches. Documents are embedded in batches (`EKA_EMBEDDING_BATCH_SIZE`). Qwen3-Embedding uses its query instruction automatically (`encode_query`) and left padding (it pools the last token).

**7. Chunks are measured with the embedding model's own tokenizer.** A "400-token chunk" now means 400 tokens of the model that embeds it. At startup the server refuses a configuration where chunks (plus header) exceed the model's context. Models such as `all-MiniLM-L6-v2` (256 tokens) would otherwise silently truncate every chunk.

**8. Model changes are detected, not silently mixed.** Each Chroma collection records the model name and dimension that built it. Querying with a different model returns `409 index_out_of_date` with the fix: stop the server, then `python -m app.cli reindex`. Keyword search keeps working meanwhile.

**9. Two embedder backends.** The real `sentence-transformers` backend comes from the optional `[ml]` extra (PyTorch is a multi-GB install). The `hashing` backend is deterministic feature hashing with no download. It lets CI and 220+ tests exercise the whole dense pipeline (Chroma, filters, isolation, lifecycle) in seconds. It is **not semantic**, and the benchmark report prints a warning whenever it's used.

**10. Privacy.** Search is `POST`, so queries never land in URLs or proxy logs. Logs record the query's length, not its text. Chroma telemetry is disabled, and Chroma's own default embedding function is disabled too: it would silently download a different model.

**11. Hybrid search with Reciprocal Rank Fusion.** `score(d) = Σ w_r / (k + rank_r(d))`, with k = 60 (the original paper's value), 50 candidates from each retriever, and weights 1.0/1.0, all configurable.
- *Why RRF instead of adding scores:* BM25 scores are unbounded and cosine similarities sit in [-1, 1]. Adding them needs normalisation, and min-max normalisation is fragile: one outlier rescales everything. RRF needs only ranks.
- *Agreement is rewarded:* a chunk ranked 2nd by both retrievers beats one ranked 1st by only one.
- *Filter, then fuse:* both lists are hydrated first, and anything not visible (stale entries, documents being processed) is removed. Only then are ranks fused, so a phantom ID can never occupy a rank position.
- *Deterministic ties* (best single rank, then ID), so evaluation runs are reproducible.
- *Explainable:* each result reports its rank and raw score from each retriever. The UI can show "found by keyword #3 and semantic #1", and debugging a bad ranking takes seconds.
- *Known failure mode, measured rather than assumed:* with a weak or noisy second retriever, deep candidate lists let mid-ranked "consensus" chunks outrank a confident single hit. The sandbox run shows this with the hashing stand-in (see `eval/README.md`). Whether it happens with the real model is checked on the Mac, and depth, k and weights are tuned in Phase 4 from those results.
- *Trade-off:* hybrid runs both retrievers, so its latency is roughly BM25 + dense (≈5 ms vs 1.6 ms for BM25 in the sandbox, where dense is cheap). On the Mac, query embedding dominates.

**Trade-offs and upgrade paths.** FTS5 → Postgres `tsvector` or OpenSearch when moving off SQLite. Chroma embedded mode → Chroma server or Qdrant for multi-process writers (the CLI `reindex` must run with the API stopped). One lock around the GPU → a dedicated embedding worker when throughput matters.

## Testing
| Suite | What it proves |
|---|---|
| `tests/retrieval/test_keyword_index.py` | Query sanitisation, hostile queries, ranking, heading boost, stemming/diacritics, idempotent replace, delete, filters, per-tenant isolation and independent BM25 statistics |
| `tests/retrieval/test_vector_store_and_embeddings.py` | Cosine ordering, limits, filters, delete, per-workspace collections, model/dimension mismatch, lazy init, embedder determinism, context-length guard |
| `tests/retrieval/test_fusion.py` | RRF against hand-computed scores; agreement beats a lone first place; k sensitivity; weights (including zero); deterministic ties; duplicates; empty inputs; invalid parameters |
| `tests/api/test_search_api.py` | All three modes through HTTP: source metadata, offsets, filters, top-k cap, validation, auth, cross-workspace isolation (even when another workspace's IDs are passed), delete/reprocess consistency (no orphans, no duplicates), not-ready documents hidden, embedding failure, model change → 409 → reindex; hybrid scores equal the RRF formula over the reported ranks, hybrid survives an empty BM25 list, ranks count only visible chunks, weights and k come from settings |
| `tests/ml/` | The real sentence-transformers path with a tiny local model: device selection (MPS on your Mac), batching, normalisation, tokenizer counting (skipped without PyTorch) |
| `tests/eval/` | Metric definitions, dataset label integrity, runner end to end |

## Benchmark (measured on the MacBook Air M4)
| Mode | MRR | Recall@5 | Recall@10 | p50 ms |
|---|---|---|---|---|
| bm25 | 0.538 | 0.553 | 0.638 | 0.5 |
| dense (Qwen3-Embedding-0.6B, MPS) | **0.900** | **0.979** | **1.000** | 35.1 |
| hybrid (RRF k=60, 50 candidates, equal weights) | 0.794 | 0.851 | 0.915 | 35.5 |

Per category (MRR): lexical 1.000 / 0.882 / 1.000; paraphrase 0.377 / 0.915 / 0.596; Malay 0.000 / 0.900 / 0.900 (bm25 / dense / hybrid). Default fusion recovers BM25's precision on identifiers but costs a lot on paraphrases: weak BM25 lists outvote dense's correct answer (decision 11). Full analysis in `eval/README.md`. Tuning (`python -m eval.sweep_fusion`) and reranking are Phase 4.

## Likely errors and fixes
| Symptom | Cause | Fix |
|---|---|---|
| Startup error: `sentence-transformers is not installed` | Installed without the ML extra | `make install-ml`, or set `EKA_EMBEDDING_BACKEND=hashing` |
| First startup takes minutes | Model download (~1.2 GB) | One-off; cached in `~/.cache/huggingface` |
| `EmbeddingConfigError ... EKA_CHUNK_SIZE_TOKENS` | Model context shorter than chunks | Lower chunk size or choose a longer-context model |
| `409 index_out_of_date` | Model changed after indexing | Stop server → `python -m app.cli reindex` |
| Dense search slow on first query only | Lazy load with `EKA_EMBEDDING_PRELOAD=false` | Set it to `true` |
| `MPS backend out of memory` | Batch too large with other apps open | `EKA_EMBEDDING_BATCH_SIZE=8`, close apps, or `EKA_EMBEDDING_DEVICE=cpu` |
| An MPS op is "not implemented" | PyTorch gap for an operator | `export PYTORCH_ENABLE_MPS_FALLBACK=1` |
| Chroma "database is locked"/readonly | Two processes writing to `data/chroma` | Run CLI reindex only with the API stopped |
| macOS: "Python quit unexpectedly", or python-docx "Package not found" mid-test | File-handle exhaustion (macOS default limit is 256). Fixed: `Services.close()` now closes the Chroma client on shutdown; `tests/test_resource_cleanup.py` guards it | Update to the fixed version; check with `ulimit -n` |
| BM25 finds nothing for a valid question | Vocabulary mismatch (e.g. "vacation" vs "annual leave") | Use `dense` or `hybrid` |
| Hybrid ranks a correct BM25 #1 too low | Noisy second list outvotes it (see decision 11) | Benchmark with `--candidates 20` / `--rrf-k 20`; tune in Phase 4 |

## Acceptance criteria
Verified here (Linux sandbox, Python 3.11 and 3.12):
- [x] BM25 search via FTS5 with source metadata; operators can't be injected
- [x] Dense search via ChromaDB with source metadata and the embedding model reported
- [x] Hybrid search via Reciprocal Rank Fusion: scores match hand-computed values; fusion happens after visibility filtering; per-retriever ranks reported; k, depth and weights configurable and validated
- [x] Workspace isolation in all three modes: separate indexes, re-checked at hydration, tested through the API
- [x] Delete and reprocess keep both indexes consistent (no orphans, no duplicates); not-ready documents hidden
- [x] Model change detected (409, in dense and hybrid) and recoverable (reindex)
- [x] Real sentence-transformers code path tested offline with a tiny model; device auto-selection tested
- [x] Benchmark runs BM25, dense and hybrid on the same dataset (51 labelled queries); metrics and runner tested
- [x] **257 tests pass** with the ML extra, including under macOS's 256-open-file limit (`ulimit -n 256`) (without PyTorch the 6 real-model tests skip); Ruff clean; live smoke test passes with the hashing backend (23 checks) and through the real sentence-transformers path with a tiny local model (25 checks)

Verified on the MacBook Air M4 (macOS 26.3, Python 3.11.9, torch 2.14.1, MPS):
- [x] `./scripts/verify.sh` passes: Ruff, 257 tests, live smoke test with `Qwen3-Embedding-0.6B` on MPS including the semantic dense and hybrid checks
- [x] Benchmark with the real model produced BM25, dense and hybrid results (table above)

**Phase 3: complete.**

## Interview talking points
- "Indexes return only IDs; SQLite hydrates and re-checks permissions. A stale index can't leak data or show half-processed documents."
- "I partitioned the BM25 index per tenant because BM25 statistics are corpus-wide, which is a ranking problem and a side channel. I have a test that proves scores don't move across tenants."
- "Chunks are measured in the embedding model's own tokens, and the server refuses configurations that would silently truncate them."
- "I built an offline hashing embedder so CI exercises the whole vector pipeline without a GPU or a 1 GB download. The benchmark labels it as non-semantic so nobody mistakes it for a result."
- "My benchmark labels sections, not chunks, so it survives chunking changes. I report per-category results because the overall average depends on the query mix."
- "My suite passed on Linux but crashed on my Mac. I traced it to leaked file handles: ChromaDB clients weren't closed on shutdown, and macOS allows only 256 open files. I reproduced it with `ulimit -n 256`, fixed the lifecycle, and added a regression test that fails if handles leak. Running CI on a macOS runner too is why the matrix includes one."
- "On my M4, dense retrieval reached MRR 0.90 against BM25's 0.54. Hybrid with textbook RRF settings scored *worse* than dense alone, 0.79. I report that instead of hiding it, explain the mechanism, and tune from a measured sweep."
- "I fuse with Reciprocal Rank Fusion because BM25 and cosine scores live on incompatible scales; RRF only needs ranks. I filter invisible chunks *before* fusing, and every result reports its rank from each retriever. I also measured RRF's known weakness: a noisy second retriever can outvote a confident first hit. I tune depth, k and weights from benchmark results rather than defaults."
