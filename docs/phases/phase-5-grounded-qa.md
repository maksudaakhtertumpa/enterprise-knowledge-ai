# Phase 5 — Grounded Answers, Citation Verification, Prompt-Injection Defences

**Budget: $0.** Answers come from a local model through Ollama (`qwen3.5:4b`). No paid API is used or required.

## What was built

```
POST /ask ─► dense retrieval + rerank (Phase 4 pipeline) ─► top 5 passages
                │
                ├─ no passages / best score under the gate ─► "insufficient evidence" (no LLM call)
                ▼
        nonce-fenced prompt + JSON schema ─► local LLM (Ollama)
                ▼
        verifier: each citation = (passage number, exact quote) checked against what
        the model was actually shown; sentences without a verified citation are removed
                ▼
        answer + citations whose file, section, pages and offsets come from the DATABASE
```

| Area | Files |
|---|---|
| LLM backends (Ollama over HTTP, offline extractive stand-in) | `app/services/generation/llm.py` |
| Prompt, nonce fences, injection flagging | `app/services/generation/prompt.py` |
| Citation verifier | `app/services/generation/verify.py` |
| Orchestration (retrieve → gate → generate → verify) | `app/services/generation/answer.py` |
| API: `/ask`, `/ask/history`, `/ask/history/{id}`, `/evidence/{chunk_id}` | `app/api/routes/ask.py`, `app/schemas/ask.py` |
| Question history (for the UI and analytics) | `QuestionLog` in `app/db/models.py` |
| Answer benchmark + gate calibration + injection suite | `eval/run_answers.py`, `eval/datasets/teratai/answers.jsonl`, `eval/datasets/injection/` |

### Endpoints (all need `X-API-Key`)
| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/ask` | `{"question": "..."}` → `status` (`answered` / `insufficient_evidence`), `answer` with `[n]` markers, `citations` (verified quote, file, section, pages, character offsets), every `evidence` passage shown to the model, `verification` counts, `warnings`, `usage` (tokens, cost $0), per-stage `timings_ms` |
| GET | `/api/v1/ask/history` | Recent questions with status, latency and citation counts |
| GET | `/api/v1/ask/history/{id}` | Replay a past answer with its citations |
| GET | `/api/v1/evidence/{chunk_id}?context=300` | The cited passage with surrounding document text, to inspect the original |

## Engineering decisions

**1. The model can't invent a source, a page or a quote.** The model only returns `{"passage": n, "quote": "..."}` pairs. Every quote is checked against passage *n* as the model actually saw it. The file name, section, page numbers and character offsets in the response are then looked up from the database by passage number. A model that writes "see page 99 of secret.pdf" changes nothing: a test proves it.

**2. Strict verification, with one deliberate tolerance.**
- *Exact:* the quote's words appear in order in the passage, ignoring case, punctuation and Unicode forms.
- *Near-exact ("fuzzy"):* the quote differs only in **filler words** such as "the" or "of". A changed number, negation or content word is always rejected: "14 working days" → "40 working days", or "must be approved" → "must not be approved". The first version used a plain word-overlap threshold, and writing tests showed it would accept a long quote with one changed number, so it was replaced.
- *Sentences:* each answer sentence needs at least one verified marker, or it is removed. Markers pointing at unverified citations are stripped. If nothing survives, the answer becomes `insufficient_evidence` with reason `no_verified_citations`.

**3. "Insufficient evidence" has four explicit causes**, reported as `reason`: no passages retrieved; the best reranker score is under the gate; the model declined; nothing verified. A fifth, `model_output_invalid`, covers replies that aren't valid JSON. The UI and the benchmark can tell these apart.

**4. Prompt-injection defences, layered, with stated limits.**
| Layer | What it stops | Limit |
|---|---|---|
| Evidence fenced with a **random nonce per request** | A document closing its own block and appending "new instructions" | None for forgery: the nonce is unguessable |
| Look-alike fences in documents broken up (`<<<` → `< < <`) | Weak models being confused by fake fences | Cosmetic; quotes still verify (words only) |
| System prompt: passages are untrusted data | Most "ignore previous instructions" text | Depends on the model obeying |
| JSON-only reply (Ollama structured output) | Free-form output, role-play, extra text | Content inside the fields is still the model's |
| Citation verification | Claims the evidence doesn't contain | **Can't** block a payload the model quotes *verbatim* from a malicious document |
| Heuristic flag (`suspicious: true` + warning) | Nothing on its own; tells the user | Easily bypassed by rephrasing |
| Workspace isolation (Phase 2) | Other tenants planting documents | Insiders with upload rights in the same workspace |
| No tools, no secrets in the prompt | Exfiltration or actions | — |

The worst case is therefore a *wrong answer quoting a malicious document from the same workspace*, not data theft or actions. That residual risk is measured by the injection suite on the real model rather than claimed away.

**5. Context-window safety.** `num_ctx` is sent explicitly (8192), because Ollama silently truncates prompts that don't fit. Passages are added in rank order while they fit an estimated budget (counted with the Qwen3 embedding tokenizer, close to the LLM's). Ollama's actual `prompt_eval_count` is checked afterwards, and a warning is returned if truncation was possible.

**6. Reliability.**
- *Timeouts:* explicit, with 180 s for reading the reply, since the first question also loads the model.
- *Retries:* only for transient failures (HTTP 429, 500, 502, 503, 504, connection resets), with exponential backoff and at most 2 retries.
- *Fail fast:* no retry on a timeout, a missing model (`503 llm_model_missing`: "Run: ollama pull qwen3.5:4b") or Ollama not running (`503 llm_unavailable`).
- *Optional readiness:* readiness reports the LLM but doesn't fail on it, so upload and search keep working while Ollama is stopped.

**7. Repeatable answers.** Temperature 0 with a fixed seed, `think: false` (reasoning traces add seconds and aren't needed for extraction from given text), and `keep_alive` of 30 minutes, so the model isn't reloaded for every question.

**8. No new dependency for HTTP.** The Ollama client uses the standard library. Tests run a real local HTTP server that imitates Ollama, so timeouts, retries and error mapping are exercised over a real socket.

**9. Offline stand-in, labelled.** `EKA_LLM_BACKEND=extractive` quotes the best-matching evidence sentence. It isn't generative or semantic. It lets 360+ tests and CI run with no model, and the benchmark report prints a warning whenever it's used.

**10. History is opt-out.** `EKA_ASK_HISTORY=true` stores the question, answer and citations per workspace, for the UI's conversation history and analytics in Phase 6. With it off, only metrics are stored (status, latency, token counts), with no question or answer text.

## Testing
| Suite | Covers |
|---|---|
| `tests/generation/test_verify.py` | JSON parsing (fences, chatter, malformed items); sentence splitting with markers and abbreviations; exact, near-exact and rejected quotes; **five "one fact changed in a long quote" cases** that must all fail; unsupported sentences removed; invalid markers stripped |
| `tests/generation/test_prompt_and_llm.py` | Fresh nonce per prompt; a document can't close its own fence; labels single-line; injection flagging in English and Malay without flagging ordinary policy text; extractive stand-in; **Ollama over a real HTTP socket**: request shape, retries, bounded retries, missing model, models without the "think" flag, timeout, Ollama not running, health |
| `tests/api/test_ask_api.py` | Answers with real source metadata and quote offsets; filters; all insufficient-evidence paths; a **scripted adversarial model** (fabricated passages, altered facts, unsupported sentences, fake page numbers) can't get anything unverified into the response; injection documents flagged and fenced; isolation; error mapping (503, 504); readiness stays green with the LLM down; history and replay; evidence context |
| `tests/eval/test_answer_benchmark.py` | Fact matching, summary and gate maths, dataset consistency, runner end to end offline |
| `scripts/smoke_test.py` step 5 | Live server: answerable question cites the right file; unanswerable question refused; other workspace sees no evidence; history |

## Benchmark (run on your Mac)
`python -m eval.run_answers` (also run by `./scripts/verify.sh`) asks:
- the **30 natural-language questions** (22 English paraphrases, 8 Malay), each with expected facts (`answers.jsonl`, every fact checked to appear in its source section);
- **12 unanswerable questions**, including hard negatives such as "medical leave carry-over" (only annual leave carries over) and "adoptive parents";
- **4 prompt-injection attacks** planted in the same workspace (`eval/datasets/injection/`).

It reports answer rate, fact recall, citation precision, unsupported-sentence rate, correct refusals, latency and tokens. It also produces a **gate table** showing what each reranker-score threshold would add, measured from the same run.

### Results (MacBook Air M4, `qwen3.5:4b`, measured)
| Metric | Result |
|---|---|
| Answer rate (30 answerable) | **0.967** (29/30) |
| Fact recall | **0.883** (0.914 on answered questions) |
| Citation precision (citations pointing at a labelled source section) | **0.906** |
| Unsupported sentences removed by the verifier | 0.022 |
| Proposed citations that passed verification | 0.733 |
| Correct refusals (12 unanswerable) | **0.917** (11/12) |
| Prompt-injection attacks resisted / answered correctly | **4/4** / **4/4** |
| Latency p50 / p95 | 5.4 s / 9.9 s (Malay p50 7.7 s) |
| Tokens per question | ~1,000 prompt, ~85 completion; **$0** |

**Decisions from these measurements:**
- **Score gate stays off.** Any threshold above 0 dropped a third or more of answerable questions (correct passages often score below 0.005) and caught no extra unanswerable ones. The one unanswerable question answered (U06) had a top score of 0.974: relevant, not answering. Reranker scores measure relevance, not answerability; the model's own refusals did that job (11/12).
- **Verification stays strict.** It cost one answerable question (P04: the model's quotes didn't match the source) in exchange for never showing an unverified quote.
- **Known gap, recorded for Phase 7:** U06's sentence went beyond its (genuine) quote. Quote verification can't catch that; a local entailment model (free NLI cross-encoder) checking "does the quote support this sentence?" can.

## Likely errors and fixes
| Symptom | Cause | Fix |
|---|---|---|
| `zsh: command not found: ollama` | Ollama app not installed, or never opened | Install from ollama.com/download, open it once, new Terminal window |
| `503 llm_unavailable` | Ollama not running | Open the Ollama app (menu-bar llama icon) |
| `503 llm_model_missing` | Model not pulled | `ollama pull qwen3.5:4b` |
| `504 llm_timeout` on the first question | Model loading plus first answer | Ask again (model now loaded), or raise `EKA_LLM_READ_TIMEOUT_S` |
| Many `insufficient_evidence` with `model_output_invalid` | Model ignoring the JSON format | Update Ollama (structured outputs need 0.5+); try `gemma4:e4b` |
| Many `no_verified_citations` | Model paraphrasing instead of quoting | Check `verification.rejected`; a larger model (`qwen3.5:9b`) quotes better |
| Mac slow or swapping during answers | LLM + embedder + reranker on 16 GB | Close apps; keep `EKA_RERANKER_HALF_PRECISION=true`; use the 4B model |
| Warning "prompt may have been truncated" | Long passages | Lower `EKA_ANSWER_PASSAGES` or raise `EKA_LLM_NUM_CTX` (uses more memory) |

## Acceptance criteria
Verified here (Linux sandbox, offline stand-ins):
- [x] `/ask` returns verified citations whose metadata comes from the database; quote offsets point into the document text
- [x] Fabricated passages, altered facts and unsupported sentences never reach the response (adversarial model tests)
- [x] Insufficient-evidence paths: no passages, score gate, model declined, nothing verified, invalid output
- [x] Prompt-injection layers implemented and tested (nonce fences, defanging, flagging, verification, isolation)
- [x] Ollama client tested over real HTTP: retries, timeouts, missing model, not running
- [x] History and evidence endpoints, workspace-scoped
- [x] Answer benchmark with expected facts, gate calibration and injection suite; runs end to end offline
- [x] Full test suite passes; Ruff clean; live smoke test passes (34 checks)

Verified on the MacBook Air M4 with Ollama and `qwen3.5:4b`:
- [x] `prepare-models` (inside `verify.sh`): the LLM answered with a verified citation
- [x] `./scripts/verify.sh` passes, including smoke step 5: cited answer, refusal of an unanswerable question, isolation
- [x] Answer benchmark run (table above); `EKA_ANSWER_MIN_RERANK_SCORE` set from its gate table: stays 0

**Phase 5: complete.**

## Interview talking points
- "The model never supplies a source. It returns passage numbers and quotes; I verify each quote against the exact text it was shown, and attach file names and page numbers from the database. Fabricated sources are impossible by construction, and there's a test with a deliberately lying model."
- "My first fuzzy-matching rule would have accepted a long quote with one changed number. A test caught it, so near-exact quotes may now differ only in filler words, never numbers or negations."
- "Prompt injection is layered: nonce-fenced evidence, JSON-only output, citation verification, flagging and workspace isolation. I document the residual risk honestly: a model can still quote a malicious document from the same workspace. So I measure that with a planted-attack suite instead of claiming it's solved."
- "Everything runs locally for $0. The Ollama client has explicit timeouts, retries only transient errors, and the readiness probe doesn't fail when the LLM is down, because search still works."
- "I planned a reranker-score threshold for 'insufficient evidence', measured it, and turned it off: a third of answerable questions had a correct top passage scoring below 0.005, while the one wrong answer came from a passage scoring 0.97. Relevance isn't answerability."
- "On a MacBook Air, a 4B local model answered 29 of 30 questions with verified citations, refused 11 of 12 unanswerable ones and resisted 4 of 4 planted injection attacks, in about 5 seconds per question at $0."
- "My one failure taught me the limit of quote verification: it proves the quote exists, not that the sentence follows from it. That's the next thing I'd add, with an entailment model."
