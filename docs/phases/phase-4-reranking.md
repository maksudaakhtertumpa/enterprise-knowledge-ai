# Phase 4 — Fusion Tuning and Cross-Encoder Reranking

## Starting point: what Phase 3 measured (MacBook Air M4)
| Mode | MRR | lexical | paraphrase | Malay |
|---|---|---|---|---|
| bm25 | 0.538 | **1.000** | 0.377 | 0.000 |
| dense (Qwen3-Embedding-0.6B) | **0.900** | 0.882 | **0.915** | **0.900** |
| hybrid, textbook RRF (k=60, equal weights) | 0.794 | 1.000 | 0.596 | 0.900 |

## Step 1 — Fusion sweep (measured, M4)
`python -m eval.sweep_fusion` scored 36 settings: k ∈ {10, 30, 60}, candidate depth ∈ {5, 10, 20, 50}, BM25 weight ∈ {0.25, 0.5, 1.0}.

| Finding | Evidence |
|---|---|
| **No fusion setting beat dense alone.** | Best hybrid MRR 0.879 (k=10, BM25 weight 0.25) vs dense 0.900 |
| **Every setting trades paraphrase for lexical quality.** | Lexical 0.882 → 0.941–1.000 while paraphrase falls 0.915 → 0.824–0.618. No setting was ≥ the best single retriever in all three categories |
| **Weight and k matter; depth doesn't.** | At weight 0.25, depths 5–50 give identical results. Lower k helps because it lets dense's confident first place count for more |
| **Malay is unaffected.** | 0.900 everywhere: BM25 matches nothing in Malay queries |

**Decisions from these measurements:**
- Default search mode is **dense** (`EKA_SEARCH_DEFAULT_MODE`).
- Hybrid stays available with the sweep's best setting: k=10, BM25 weight 0.25, 20 candidates. Depth is irrelevant, so the cheaper value wins.
- No query-specific tricks such as "boost BM25 when the query looks like a code". With 17 lexical queries, that would fit the test set rather than the problem.
- Caveat: settings were chosen on the same 47 queries they're scored on, so held-out confirmation is Phase 7's job.

## Step 2 — Cross-encoder reranking
```
first stage (bm25 | dense | hybrid) → top 20 → cross-encoder reorders → top k
```
| File | Role |
|---|---|
| `app/services/retrieval/reranker.py` | `CrossEncoderReranker` (bge-reranker-v2-m3, MPS/CUDA/CPU, fp16 on GPU, lock-serialised), `HashingReranker` (offline stand-in), `build_reranker` |
| `app/services/retrieval/context.py` | Title + section + text: the same passage format for embedding and reranking |
| `app/services/retrieval/search.py` | Optional rerank stage after any mode; keeps `retrieval_rank` for every hit |
| `app/api/routes/search.py` | `rerank` request flag (server default `EKA_RERANK_BY_DEFAULT`); `mode` optional (server default) |
| `eval/run_retrieval.py` | Five configurations: `bm25`, `dense`, `hybrid`, `dense+rerank`, `hybrid+rerank` |

### Design decisions
1. **Rerank only the top 20.** Cross-encoder cost is linear in candidates, one model call per pair. Dense already has Recall@10 = 1.000 on the benchmark, so 20 candidates cover everything relevant with margin, and the reranker's only job is ordering. `EKA_RERANK_CANDIDATES` and `--rerank-candidates` tune it.
2. **A flag on any mode, not a separate mode.** `{"mode": "dense", "rerank": true}` makes the two stages independent, so the benchmark can measure each combination, and the server default can change from config alone.
3. **Explainable.** Each reranked hit reports `retrieval_rank` (where retrieval put it) next to its new rank, and `timings_ms.rerank_ms` shows the cost. A result that jumped from #14 to #1 is visible in the API response.
4. **Scores in (0, 1).** A sigmoid on the cross-encoder logit makes scores comparable across queries. Phase 5 can then use a threshold such as "no passage above 0.3 → insufficient evidence".
5. **Half precision on the GPU.** bge-reranker-v2-m3 (568M parameters) takes about 2.2 GB in float32 and about 1.1 GB in float16, which matters on a 16 GB Mac that will also run an LLM in Phase 5. CPU stays float32 (no speed gain there). A test guards the `dtype` argument name, which changed between transformers versions.
6. **Same passage text as embedding.** Title and section header plus body, so both stages judge the same thing.
7. **Deterministic ties.** Equal scores keep first-stage order.
8. **Disabled cleanly.** `EKA_RERANKER_BACKEND=none` returns `503 reranking_disabled` when a request asks for reranking; plain search is unaffected.
9. **Default chosen by measurement.** Shipped off, then switched on (with 10 candidates) after the M4 benchmark showed it was worth the latency. A test pins the measured defaults.

## Testing
| Suite | Covers |
|---|---|
| `tests/retrieval/test_reranker.py` | Overlap scoring, backend selection, lazy loading |
| `tests/api/test_rerank_api.py` | All modes with `rerank=true`: rank renumbering, `retrieval_rank`, sorted scores in [0, 1], promotion from beyond top-k, candidate-pool limit, empty results, default flag, 503 when disabled, workspace isolation, readiness |
| `tests/ml/test_cross_encoder_reranker.py` | Real CrossEncoder path with a tiny model: probabilities, batching, determinism, device and dtype selection, float16 loading |
| `tests/eval/test_benchmark.py` | Five-configuration run, rerank flags recorded, rerank modes rejected without a reranker |
| `scripts/smoke_test.py` | Live server: `dense+rerank` search; with the real model, "when do I have to update my login credentials" must find the password policy |

280 tests pass with the ML extra, including under macOS's 256-open-file limit.

## Results (MacBook Air M4, measured)
| Mode | MRR | Recall@1 | Recall@5 | lexical MRR | paraphrase MRR | Malay MRR | p50 ms |
|---|---|---|---|---|---|---|---|
| bm25 | 0.538 | 0.500 | 0.553 | 1.000 | 0.377 | 0.000 | 0.8 |
| dense | 0.900 | 0.819 | 0.979 | 0.882 | 0.915 | 0.900 | 35.7 |
| hybrid (tuned) | 0.879 | 0.798 | 0.979 | 0.941 | 0.824 | 0.900 | 35.5 |
| dense + rerank 20 | 0.961 | 0.925 | 0.979 | 1.000 | 0.955 | 0.896 | 633.8 |
| hybrid + rerank 20 | 0.961 | 0.925 | 0.979 | 1.000 | 0.955 | 0.896 | 629.6 |
| **dense + rerank 10** | **0.962** | **0.925** | **1.000** | **1.000** | **0.955** | **0.900** | **334.6** |

`prepare-models` on the M4: embedding model ready in 12.8 s on MPS; reranker ready in 17.4 s on MPS in float16, test scores 0.982 (relevant) vs 0.000 (irrelevant). Smoke test: "when do I have to update my login credentials" → the password policy, with no shared keywords (dense + rerank, 149 ms).

### Decisions
| Setting | Value | Evidence |
|---|---|---|
| Default mode | `dense` | Beat every hybrid setting; with reranking, hybrid adds nothing (identical results) |
| Rerank by default | **on** | MRR 0.900 → 0.962; lexical 0.882 → 1.000; paraphrase 0.915 → 0.955 |
| Rerank candidates | **10** | Same or better than 20 on every metric, half the latency, and fixes the one Malay regression 20 caused |
| Fast path | `"rerank": false` | ~35 ms instead of ~335 ms, for autocomplete-style callers |

**Phase 5 note:** the reranker gave a correct answer to a no-shared-keyword paraphrase a score of only 0.122, while ranking it first. A fixed "refuse below 0.5" threshold would reject good answers, so the insufficient-evidence threshold must be calibrated on the benchmark, using the unanswerable queries U01–U04.

## Likely errors and fixes
| Symptom | Cause | Fix |
|---|---|---|
| Long pause at startup on first run | Reranker download (~2.3 GB) | Run `python -m app.cli prepare-models` first: downloads with progress bars and test-loads both models (`verify.sh` does this) |
| Smoke test stuck at "[1] Startup", then crashed and left a server running | Bug, fixed: server output went to an unread pipe; ~64 KB of download progress filled it and froze the server | Fixed: output goes to a log file, its tail is printed on failure, and the server is killed if it won't stop. Clean up an old one with `pkill -f "uvicorn app.main:app"` |
| `503 reranking_disabled` | `EKA_RERANKER_BACKEND=none` | Set it to `cross-encoder`, or don't send `rerank: true` |
| Reranked search much slower than plain | 20 cross-encoder calls per query | Lower `EKA_RERANK_CANDIDATES` (e.g. 10) and re-benchmark |
| `MPS backend out of memory` | Two models plus apps on 16 GB | Keep `EKA_RERANKER_HALF_PRECISION=true`, lower `EKA_RERANKER_BATCH_SIZE`, close apps |
| NaN or identical scores on MPS | Rare fp16 numerical issue on some ops | `EKA_RERANKER_HALF_PRECISION=false` |

## Acceptance criteria
Verified here (Linux sandbox):
- [x] Fusion defaults set from the measured M4 sweep; default search mode `dense`; both configurable
- [x] Cross-encoder reranker on any mode; first-stage rank kept; scores in (0, 1); deterministic ties
- [x] Reranker loaded at startup, in readiness, disableable (503), isolated by workspace
- [x] Real CrossEncoder path tested offline: device and dtype selection, float16 loading, batching
- [x] Benchmark runs five configurations on the same dataset; warnings when stand-ins are used
- [x] 280 tests pass; Ruff clean; live smoke test passes (26 checks)

Verified on the MacBook Air M4:
- [x] `prepare-models`: both models on MPS, reranker in float16 with correct relevance separation (0.982 vs 0.000)
- [x] `./scripts/verify.sh` passes, including the semantic `dense+rerank` smoke check
- [x] Five-configuration benchmark plus a rerank-depth comparison; defaults set from them (dense, rerank on, 10 candidates)

**Phase 4: complete.**

## Interview talking points
- "Textbook hybrid search made my results worse. I swept 36 fusion settings on real hardware: none beat dense alone, and each traded paraphrase quality for identifier precision. So I made dense the default and put a cross-encoder after retrieval instead."
- "Two-stage retrieval: dense retrieval feeds 10 candidates to a cross-encoder. On my M4 that took MRR from 0.900 to 0.962, and reached perfect identifier precision without BM25. Reranking 10 instead of 20 halved latency to 335 ms and was slightly *better*: fewer candidates means fewer distractors."
- "The reranker solved what fusion couldn't. Fusion traded paraphrase quality for identifier precision; reading the question and passage together improved both."
- "Every reranked result reports where retrieval put it, so the reranker's effect is visible per query, not just in an average."
- "I load the reranker in float16 on the GPU to halve its memory on a 16 GB laptop that also has to run an LLM, and a test pins the library argument that controls it."
- "My smoke test froze on the first real run: it piped the server's output and never read it, so download progress bars filled the 64 KB pipe buffer and blocked the server. Classic subprocess deadlock. I send output to a log file now, print its tail on failure, and download models in a separate visible step."
- "Settings tuned on the same queries they're scored on are optimistic. I say so in the report and keep a held-out check for Phase 7."
