# Retrieval evaluation

## Dataset: `teratai`

An **original, fictional** corpus written for this project: eleven internal-policy documents for "Teratai Digital Sdn Bhd", a made-up Malaysian company (leave, travel and expenses, IT security, remote work, code of conduct, data protection, onboarding, benefits, procurement, performance, facilities). 67 sections, about 3,500 words. No real company's documents are used, so it can be published with the repository.

`queries.jsonl` holds 51 labelled queries:

| Category | Count | What it tests | Expected winner |
|---|---|---|---|
| `lexical` | 17 | Exact identifiers, product names, codes: "form FIN-12", "extension 4357", "WPA2" | BM25 |
| `paraphrase` | 22 | Natural questions using different words from the document: "How much vacation time do I get?" → *Annual Leave* | Dense |
| `malay` | 8 | Bahasa Malaysia questions over English documents (cross-lingual) | Dense, if the model is multilingual |
| `unanswerable` | 4 | Nothing in the corpus answers them | Excluded from retrieval metrics; used in Phase 5 for "insufficient evidence" |

**Labels are sections, not chunks.** Each query lists the relevant `(document, section heading)` pairs. A retrieved chunk counts as relevant if it comes from a labelled section, so labels stay valid when chunk size or overlap changes. A test checks that every label points at a real section.

**Honest limitations.** One author wrote both the documents and the queries, so phrasing may be unconsciously easier than real users'. The corpus is small (67 sections), so one query moves Recall by about 2 points. The category mix was chosen to show where each method is strong, so the overall average depends on that mix; the per-category numbers are the meaningful ones. Treat results as a regression benchmark and a teaching tool, not a leaderboard.

## Metrics (section-level)

| Metric | Meaning |
|---|---|
| Recall@k | Share of the relevant sections found in the top k |
| Precision@k | Share of the top k results that are relevant (divides by k) |
| MRR | Mean of 1 / rank of the first relevant result |
| nDCG@10 | Rank-weighted gain; each section counts once |
| p50 / p95 ms | Search latency per query, after one warm-up query |

Definitions and edge cases are in `eval/metrics.py` and tested in `tests/eval/`.

## Running it

```bash
python -m eval.run_retrieval                                   # BM25 vs dense vs hybrid, model from .env
python -m eval.run_retrieval --model BAAI/bge-m3 --label bge-m3  # compare another embedding model
python -m eval.run_retrieval --rrf-k 20 --candidates 20 --label k20-d20   # RRF sensitivity
python -m eval.run_retrieval --backend hashing                   # offline plumbing check
python -m eval.sweep_fusion                                      # 36 hybrid settings, one ingestion
python -m eval.run_retrieval --modes dense dense+rerank --rerank-candidates 10 --label rerank10
```

Each run ingests the corpus through the real ingestion pipeline into a temporary directory. It writes `eval/results/<timestamp>-<label>.json` and a Markdown report. The JSON holds every number, the per-query top 3, machine and model details, the fusion settings and a dataset hash.

## Results so far

### MacBook Air M4 (16 GB), `Qwen/Qwen3-Embedding-0.6B` on MPS, 10 Oct 2026

Measured with `./scripts/verify.sh` (`python -m eval.run_retrieval`). Default settings: 400-token chunks, RRF k = 60, 50 candidates per retriever, equal weights. The result files are written to `eval/results/` on the machine that ran them.

| Mode | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|
| bm25 | 0.500 | 0.553 | 0.638 | 0.538 | 0.559 | 0.5 | 1.0 |
| dense | **0.819** | **0.979** | **1.000** | **0.900** | **0.925** | 35.1 | 45.4 |
| hybrid (RRF, defaults) | 0.734 | 0.851 | 0.915 | 0.794 | 0.822 | 35.5 | 45.6 |

| Category (MRR) | Queries | bm25 | dense | hybrid |
|---|---|---|---|---|
| lexical | 17 | **1.000** | 0.882 | **1.000** |
| paraphrase | 22 | 0.377 | **0.915** | 0.596 |
| malay | 8 | 0.000 | **0.900** | **0.900** |

**What this shows.**
- **Dense retrieval works.** With a multilingual embedding model, every relevant section lands in the top 10, Malay questions find English sections, and paraphrases that BM25 misses entirely are answered at rank 1.
- **BM25 keeps one advantage:** exact identifiers (form codes, product names), where it is perfect and dense is slightly less sure.
- **Default RRF fusion hurts.** It recovers BM25's precision on identifiers but drops paraphrase MRR from 0.915 to 0.596. When BM25 has no real match, it still returns 50 weak keyword hits, and with equal weights and deep lists they outvote dense's confident first place (P02, P06: dense rank 1 → hybrid outside the top 10). The sandbox run predicted this failure mode, and the real model confirms it.
- **Latency:** query embedding dominates (≈35 ms on the M4 GPU); BM25 adds under 1 ms, so hybrid costs the same as dense.

### Fusion sweep, same machine (36 settings)

| Setting | MRR | lexical | paraphrase | Malay |
|---|---|---|---|---|
| dense (reference) | **0.900** | 0.882 | **0.915** | 0.900 |
| best hybrid: k=10, BM25 weight 0.25 (any depth) | 0.879 | 0.941 | 0.824 | 0.900 |
| hybrid k=10, weight 0.5 | 0.862 | 0.971 | 0.765 | 0.900 |
| hybrid k=10, weight 1.0 | 0.824 | 1.000 | 0.660 | 0.900 |
| worst: k=60, depth 50, weight 1.0 | 0.804 | 1.000 | 0.618 | 0.900 |

No setting matched the better single retriever in every category: each one bought lexical precision with paraphrase quality. Candidate depth made no difference at the best weight. Consequences: **dense is the default mode**, hybrid keeps the best setting (k=10, w=0.25, depth 20), and Phase 4 adds a cross-encoder reranker. Settings were chosen on the scored queries, so held-out confirmation comes in Phase 7.

### Reranking, same machine (Phase 4)

`BAAI/bge-reranker-v2-m3` on MPS (float16), applied after first-stage retrieval. Hybrid uses the swept setting (k=10, BM25 weight 0.25).

| Mode | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|
| bm25 | 0.500 | 0.553 | 0.638 | 0.538 | 0.559 | 0.8 | 3.7 |
| dense | 0.819 | 0.979 | 1.000 | 0.900 | 0.925 | 35.7 | 47.7 |
| hybrid (tuned) | 0.798 | 0.979 | 1.000 | 0.879 | 0.909 | 35.5 | 44.9 |
| dense + rerank top 20 | 0.925 | 0.979 | 1.000 | 0.961 | 0.971 | 633.8 | 709.0 |
| hybrid + rerank top 20 | 0.925 | 0.979 | 1.000 | 0.961 | 0.971 | 629.6 | 704.7 |
| **dense + rerank top 10** (default) | **0.925** | **1.000** | **1.000** | **0.962** | **0.971** | 334.6 | 380.5 |

| Category (Recall@5 / MRR) | Queries | dense | dense + rerank 20 | **dense + rerank 10** |
|---|---|---|---|---|
| lexical | 17 | 1.000 / 0.882 | 1.000 / 1.000 | **1.000 / 1.000** |
| paraphrase | 22 | 0.955 / 0.915 | 1.000 / 0.955 | **1.000 / 0.955** |
| malay | 8 | 1.000 / 0.900 | 0.875 / 0.896 | **1.000 / 0.900** |

**What this shows.**
- **The reranker does what fusion could not.** Dense + rerank is perfect on exact identifiers (lexical MRR 1.000, BM25's strength) and better on paraphrases (0.915 → 0.955, dense's strength) at the same time. Fixed-weight fusion could only trade one for the other. Biggest gains: "Car Park C assembly point", "EPF SOCSO EIS", "WPA2", "RM 350 per night hotel" (rank 2 → 1), and "We accidentally emailed a customer list to the wrong person" (rank 8 → 2).
- **With a reranker, BM25 adds nothing here.** Hybrid + rerank equals dense + rerank exactly: both first stages already contain every relevant section, and the reranker decides the order.
- **A smaller pool is better, not just faster.** Reranking 10 candidates matched 20 on MRR (0.962 vs 0.961), and recovered the one Malay query that a distractor from candidates 11–20 had pushed out of the top 5. It costs half the latency (335 vs 634 ms). Dense's Recall@10 of 1.000 is why 10 is enough.
- **Cost:** about 0.3 s per search on the M4 GPU. Fine for question answering, where LLM generation will take seconds; too slow for search-as-you-type, so callers can send `"rerank": false` and get ~35 ms dense results.
- **Caveats:** 47 queries; the +0.06 MRR is about five more questions answered at rank 1. The rerank depth was chosen on these same queries, so the Phase 7 held-out set must confirm it.

**Shipped defaults** (`EKA_SEARCH_DEFAULT_MODE=dense`, `EKA_RERANK_BY_DEFAULT=true`, `EKA_RERANK_CANDIDATES=10`) come from these runs. `tests/test_config.py` pins them so they can't drift without new measurements.

### Linux sandbox (development)
`*-sandbox-hashing` runs: only the BM25 row is meaningful there. Dense and hybrid used the offline hashing stand-in, as the report states.
