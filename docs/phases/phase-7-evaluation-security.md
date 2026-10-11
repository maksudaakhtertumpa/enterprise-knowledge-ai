# Phase 7 — Entailment Check, Held-out Evaluation, Security Hardening

**Budget: $0.** The entailment model (MIT licence), pip-audit and every chart run locally or in GitHub Actions' free minutes. No new paid service, no cloud account.

## What was built

| Area | What | Files |
|---|---|---|
| **Entailment check** | After citation verification, a local NLI model scores whether each answer sentence *follows from* the passages it cites. Below the threshold the sentence is removed (or kept with a warning in `flag` mode); if nothing is left the answer becomes `insufficient_evidence` with reason `claims_not_supported`. Scores are returned per sentence and shown in the UI under "How this answer was checked". | `app/services/generation/entailment.py`, `answer.py`, `verify.py`, `web/src/components/answer-view.tsx` |
| **Held-out question set** | 48 new labelled questions over the same 11 documents (8 keyword, 24 paraphrase, 6 Malay, 10 unanswerable, 6 of them hard negatives), written before any held-out run and never used to choose a setting. | `eval/datasets/teratai-heldout/` |
| **Benchmark upgrades** | `--dataset` for both runners; the answer runner scores every sentence in `flag` mode so one run shows every entailment threshold; per-sentence scores saved for review. | `eval/run_answers.py`, `eval/run_retrieval.py` |
| **Evaluation report** | `python -m eval.report` builds `docs/evaluation/README.md` and five SVG charts from the saved runs. No plotting library: the SVG is generated directly. Runs with offline stand-ins are excluded unless asked for, and then labelled. | `eval/report.py`, `eval/charts.py` |
| **Rate limiting** | Token buckets per workspace for `ask`, `search` and uploads/reprocessing; failed key checks limited per client address. `429` + `Retry-After`, passed through the web UI's proxy. | `app/core/ratelimit.py`, `app/core/security.py` |
| **Security headers** | `nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, `Cache-Control: no-store`, `CSP: default-src 'none'` on every API response (docs pages excepted); `uvicorn --no-server-header`; `/docs` off in production. | `app/core/middleware.py`, `app/main.py` |
| **Dependency audit** | `pip-audit` job in CI (new findings fail the build); `make audit` locally. Accepted findings documented with reasons. | `.github/workflows/ci.yml`, `Makefile` |
| **Threat model** | Assets, trust boundaries, threats → controls → the test that checks each, accepted findings, known limitations. | `docs/SECURITY.md` |

## Engineering decisions

**Why an entailment check at all.** Citation verification proves a quote exists in a passage the model saw. It can't tell whether the sentence *around* the quote follows from it. Phase 5's one wrong answer (U06) cited a real quote about uncertified sick days and claimed it for carrying medical leave forward. An NLI model answers exactly the missing question: given this premise, does this hypothesis follow?

**Model: `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`.** MIT licence, ~280M parameters (fits beside the embedder, reranker and Ollama in 16 GB), trained on NLI in 100+ languages, so Malay answers over English documents can be checked. Float32 on purpose: DeBERTa-v3 is known to overflow in float16. The entailment label is found from the model config, not assumed, and a non-NLI model is rejected at load.

**Premise = sentence windows, not whole passages (measured).** The first version gave the model the whole cited passage (~400 tokens) as the premise. On the development set it removed 26.8 % of sentences at threshold 0.5, including sentences copied word for word from the passage (one scored 0.26). NLI models are trained on short premises and become unsure as premises grow. Following SummaC (Laban et al., 2022), each sentence is now scored against the whole passage *and* every window of 1–3 consecutive sentences in it, keeping the best score. Next run, same threshold: 4.9 % removed, answer rate back from 0.800 to 0.900, and the wrong claims still score low (P05 0.003, U06 0.685). The two runs are separate (the local LLM worded 11 of 42 answers differently), but that variation moves results by about one question; the removal rate fell from 27 % to 5 %, and the same verbatim sentences that scored 0.26 and 0.45 before score above 0.94 now. Cost: about 0.5 s per answer on the M4 (p50 458 ms, p95 1.2 s on the development set).

**Threshold 0.75, chosen on the development set only.** Every threshold from 0.69 to 0.82 gave identical development results (answer rate 0.900, fact recall 0.817, correct refusals 1.000), so 0.75 sits in the middle of that range rather than on an edge. It was fixed before the held-out answer run, which was then run once.

**`remove` stays the default, `flag` is one setting away.** For a policy assistant a missing sentence is cheaper than a confident wrong one, and the development set shows the check catching real errors. The held-out set shows the cost (below), which is why `EKA_ANSWER_ENTAILMENT_MODE=flag` exists.

**Rate limits in process memory.** One uvicorn process serves this project, so an in-memory token bucket is correct and needs no Redis. Buckets are bounded (10,000 keys, least recently used dropped, which can only ever *allow* a request). `X-Forwarded-For` is believed only from `EKA_TRUSTED_PROXIES`, reading right to left so a client can't spoof its address. Successful requests never spend auth tokens, so a valid user isn't locked out by their own traffic.

**Accepted dependency findings, not silenced ones.** pip-audit reports four ChromaDB advisories with no fixed release. All concern Chroma's HTTP server, which this project never runs (embedded `PersistentClient`, `embedding_function=None`). They are ignored by ID in CI with that reason written next to them and in `docs/SECURITY.md`; any other finding fails the build.

## Results (MacBook Air M4, measured 11 Oct 2026)

Full tables and charts: [docs/evaluation/](../evaluation/README.md). Raw runs: `eval/results/20261011-*.json`.

### Retrieval, held-out set (38 scored queries)

| Mode | MRR | Recall@1 | Recall@5 | p50 |
|---|---|---|---|---|
| BM25 | 0.794 | 0.789 | 0.789 | 1.2 ms |
| Dense | 0.961 | 0.921 | 1.000 | 42 ms |
| Hybrid (RRF) | 0.947 | 0.895 | 1.000 | 39 ms |
| **Dense + rerank (default)** | **1.000** | **1.000** | **1.000** | 376 ms |

The configuration chosen on the development set (dense + rerank) is also the best on new questions. The held-out questions turned out **easier for BM25** than the development ones (MRR 0.794 vs 0.538): written by the same author, they share more words with the documents. So this set confirms the *ranking* of methods, not that real users' questions would score this high. Malay is still where BM25 fails (MRR 0.167) and reranking matters (dense 0.833 → 1.000).

### Answers

| Set | Configuration | Answer rate | Fact recall | Citation precision | Correct refusals | p50 / p95 |
|---|---|---|---|---|---|---|
| Development (42 q) | model + citation check | 0.967 | 0.850 | 0.967 | 0.917 | 6.8 / 12.1 s |
| Development | **+ entailment, 0.75** | 0.900 | 0.817 | — | **1.000** | +0.5 s |
| Held-out (40 q) | model + citation check | 0.867 | 0.867 | **1.000** | 0.900 | 5.3 / 7.4 s |
| Held-out | **+ entailment, 0.75** | 0.833 | 0.833 | — | 0.900 | +0.3 s |

What the entailment check did, sentence by sentence:
- **Development:** removed 4 of 41 sentences. Three were genuinely unsupported: U06 (medical-leave carry-forward, the Phase 5 failure), P05 (concert tickets refused "because cash and vouchers are banned", a rule about something else) and a Malay sentence that mistranslated "Finance" as "Kanan". The fourth (P21, Spotify) is debatable: the policy restricts software to a catalogue but doesn't name Spotify.
- **Held-out:** removed 1 of 29 sentences, and it was **correct**: "your mother can use the counselling service as it covers immediate family members". The NLI model doesn't know a mother is immediate family. It caught nothing on this set because the model made no unfaithful claim.
- **Not caught:** the held-out wrong answer (HU02, Singapore hotel limit) said "the limit within Malaysia is RM 350". That is *true and supported*; it just doesn't answer the question. Entailment checks faithfulness, not relevance.
- On the held-out set every answered question had all its facts (fact recall when answered 1.000) and every citation pointed at a labelled section. The misses are refusals: the model declined two Malay questions and one paraphrase, and one answer lost all its citations in verification.

Cost: $0. Latency: the check adds ~0.3–0.5 s to a 5–7 s answer.

**Run-to-run variation.** Two development runs a few minutes apart, same settings and temperature 0, gave different wording for 11 of 42 answers (Ollama's GPU backend isn't bit-for-bit deterministic). Metrics moved by about one question. Treat differences under ~3 points as noise.

## Testing

| Layer | What is checked | Where |
|---|---|---|
| Entailment unit | Marker stripping, lexical stand-in (numbers, negations), sentence windows, best-window scoring, `passage` vs `sentences` | `tests/generation/test_entailment.py` |
| Real NLI code path | A tiny BERT NLI model built on the fly (no download): label lookup, probabilities, non-NLI model rejected | `tests/ml/test_nli_entailment.py` (CI `ml` job, macOS) |
| API | Unsupported sentence removed with `claims_not_supported`; `flag` keeps it with a warning; supported sentences pass; scores in the response | `tests/api/test_ask_api.py` |
| Rate limits | Bucket maths with a fake clock, bounded memory, per-workspace isolation, separate buckets, auth-failure lockout, `Retry-After`, off switch, trusted-proxy parsing | `tests/api/test_rate_limits_and_headers.py` |
| Headers and logs | Security headers on 200/401/404/422/413; docs without CSP and off in production; keys and question text never in log records | same file |
| Benchmarks | Threshold simulation, held-out set well-formed and disjoint from development, every expected fact present in its labelled section, report generator and SVGs | `tests/eval/` |
| Web | Proxy passes `429` + `Retry-After` through; entailment panel in a real browser | `routes.test.ts`, `e2e/app.spec.ts` |

Python: 437 tests. Web: 54 unit tests, 11 browser tests.

## Run it

```bash
cd ~/Projects/enterprise-knowledge-ai
source .venv/bin/activate
pip install -r requirements-ml.lock            # adds sentencepiece (the NLI tokenizer)
python -m app.cli prepare-models               # first run downloads the NLI model (~1.1 GB)
python -m eval.run_answers                     # development set (~6 min)
make heldout                                   # held-out retrieval + answers (~5 min)
python -m eval.report                          # docs/evaluation/README.md + charts
make audit                                     # dependency vulnerability scan
```

## Likely errors and fixes

| Symptom | Cause | Fix |
|---|---|---|
| Startup: `sentence-transformers is not installed` (entailment) | Installed without the ML extra | `pip install -r requirements-ml.lock`, or `EKA_ENTAILMENT_BACKEND=none` |
| `Couldn't instantiate the backend tokenizer` / sentencepiece error | Old environment without `sentencepiece` | `pip install -r requirements-ml.lock` |
| Many correct sentences removed | Premise set to whole passages, or threshold too high | Keep `EKA_ENTAILMENT_GRANULARITY=sentences`; check the threshold table in `docs/evaluation/` |
| `429 rate_limited` while testing | Burst used up | Wait `Retry-After` seconds, raise `EKA_RATE_LIMIT_*`, or `EKA_RATE_LIMIT_ENABLED=false` locally |
| Everyone using the web UI gets "too many failed key attempts" | All UI traffic comes from 127.0.0.1, so it shares the auth-failure bucket | Wait a minute; behind a real proxy set `EKA_TRUSTED_PROXIES` |
| `/docs` returns 404 | `EKA_ENVIRONMENT=production` | `EKA_API_DOCS=true` if you really want it |
| CI `audit` job fails | A new advisory | Upgrade the package (`make lock`); only accept it with a written reason in `docs/SECURITY.md` |
| `python -m eval.report` says no runs | Only offline runs exist | Run the benchmarks with real models, or `--include-standins` (labelled) |

## Acceptance criteria

- [x] Every answer sentence is scored by a local NLI model; unsupported ones are removed (or flagged), with scores in the API response and the UI.
- [x] The premise design (whole passage vs sentence windows) and the threshold were chosen from development-set measurements, then evaluated once on a held-out set.
- [x] Held-out set: new questions, labels checked automatically against the documents, disjoint from the development set.
- [x] Report with charts generated from saved runs only, naming the file behind every number; stand-in runs excluded or labelled.
- [x] Rate limits with `429` + `Retry-After`, per workspace and per client address; tested.
- [x] Security headers on every API response, including errors; docs off in production; tested.
- [x] Dependency audit in CI with documented, reasoned exceptions.
- [x] Threat model naming the test behind each control and the known gaps.
- [x] All tests pass on the M4 and in CI; cost $0.

## Interview talking points

- **"How do you know your answers are faithful?"** Two layers. The citation verifier proves the quote exists in the text the model saw; an NLI model checks the sentence follows from it. I can show the case that motivated it (a real quote attached to a wrong claim) and the tables for what the check catches and what it costs.
- **A measured design change.** Whole-passage premises removed 27 % of sentences, including verbatim ones, because NLI models are trained on short premises. Scoring against sentence windows (the SummaC approach) cut that to 5 % without letting the wrong claims through, far beyond the run-to-run noise, and the same verbatim sentences went from 0.26 to above 0.94.
- **Development vs held-out discipline.** I chose the threshold on one set, picked the middle of a flat region instead of an edge, then ran the held-out set once. The held-out result is less flattering (it removed one correct sentence and caught nothing), and I report it as it came out.
- **Knowing the limits of a metric.** The held-out wrong answer was a *true* sentence that didn't answer the question. Faithfulness and relevance are different properties; one check can't cover both.
- **Honest evaluation.** The held-out set turned out easier for keyword search, which tells me about author bias in my own questions. It confirms which method ranks best, not how well real users would do.
- **Security as engineering, not adjectives.** Every row in the threat model names the test that proves the control. Accepted dependency findings are listed with the reason they don't apply (ChromaDB server bugs in a project that never runs the server), and anything new fails CI.
- **Right-sized infrastructure.** In-memory token buckets are correct for one process; I documented exactly when that stops being true (several workers) and what would replace it.
