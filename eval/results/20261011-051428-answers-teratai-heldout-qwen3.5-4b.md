# Answer benchmark — teratai-heldout-qwen3.5-4b

- Run: 2026-10-11T05:14:28+00:00 · dataset `teratai-heldout` (hash `c6ff23b40861`) · 40 questions (30 answerable, 10 unanswerable)
- LLM: `qwen3.5:4b` (ollama), context 8192, temperature 0.0
- Retrieval: `Qwen/Qwen3-Embedding-0.6B` + `BAAI/bge-reranker-v2-m3`, 5 passages per question
- Entailment check: `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (nli, premise: sentences windows of up to 3), flag mode (scored, nothing removed)
- Machine: macOS-26.3.1-arm64-arm-64bit · Python 3.11.9 · cost: $0 (local models)

## Overall (score gate off)

| Group | Questions | Answer rate | Fact recall | Fact recall (answered) | Citation precision | Unsupported sentences | Correct refusals | p50 s | p95 s |
|---|---|---|---|---|---|---|---|---|---|
| all | 40 | 0.867 | 0.867 | 1.000 | 1.000 | 0.033 | 0.900 | 5.276 | 7.391 |
| paraphrase | 24 | 0.917 | 0.917 | 1.000 | 1.000 | 0.040 | — | 5.302 | 7.107 |
| malay | 6 | 0.667 | 0.667 | 1.000 | 1.000 | 0.000 | — | 6.945 | 8.486 |
| unanswerable | 10 | — | — | — | — | 0.000 | 0.900 | 5.178 | 6.201 |

Tokens per question: 997 prompt, 59 completion. Citations verified: 0.871 of those proposed. API cost: $0.

## Score gate trade-off (from the same run)

Skipping the LLM when the best reranker score is below a threshold:

| Threshold | Answerable reaching LLM | Answer rate | Correct refusals | LLM calls saved |
|---|---|---|---|---|
| 0.0 | 1.000 | 0.867 | 0.900 | 0.000 |
| 0.005 | 1.000 | 0.867 | 0.900 | 0.125 |
| 0.01 | 0.933 | 0.800 | 0.900 | 0.175 |
| 0.02 | 0.900 | 0.800 | 0.900 | 0.200 |
| 0.03 | 0.833 | 0.767 | 0.900 | 0.250 |
| 0.05 | 0.833 | 0.767 | 1.000 | 0.275 |
| 0.075 | 0.733 | 0.700 | 1.000 | 0.375 |
| 0.1 | 0.733 | 0.700 | 1.000 | 0.375 |
| 0.15 | 0.733 | 0.700 | 1.000 | 0.375 |
| 0.2 | 0.700 | 0.667 | 1.000 | 0.400 |
| 0.3 | 0.633 | 0.600 | 1.000 | 0.450 |
| 0.5 | 0.600 | 0.567 | 1.000 | 0.475 |

## Entailment threshold trade-off (from the same run)

Removing answer sentences whose P(entailment | cited passages) is below a threshold; an answer with no sentence left becomes a refusal:

| Threshold | Answer rate | Fact recall | Correct refusals | Sentences removed |
|---|---|---|---|---|
| 0.0 | 0.867 | 0.867 | 0.900 | 0.000 |
| 0.05 | 0.833 | 0.833 | 0.900 | 0.035 |
| 0.1 | 0.833 | 0.833 | 0.900 | 0.035 |
| 0.2 | 0.833 | 0.833 | 0.900 | 0.035 |
| 0.3 | 0.833 | 0.833 | 0.900 | 0.035 |
| 0.4 | 0.833 | 0.833 | 0.900 | 0.035 |
| 0.5 | 0.833 | 0.833 | 0.900 | 0.035 |
| 0.6 | 0.833 | 0.833 | 0.900 | 0.035 |
| 0.7 | 0.833 | 0.833 | 0.900 | 0.035 |
| 0.8 | 0.833 | 0.833 | 0.900 | 0.035 |
| 0.9 | 0.833 | 0.833 | 0.900 | 0.035 |

Sentences scoring below 0.5 (check by hand whether the model is right):

| Id | Score | Sentence |
|---|---|---|
| HP06 | 0.038 | Yes, your mother can use the free counselling service as it covers immediate family members. |

## Questions to review

| Id | Question | Status | Facts | Answer |
|---|---|---|---|---|
| HP14 | What's the most I can spend on a hotel per night for a trip within Malaysia? | insufficient_evidence | 0.00 | no_verified_citations |
| HP15 | Do I need a receipt for a RM 15 taxi fare? | insufficient_evidence | 0.00 | model_declined |
| HM04 | Berapa lama barang yang hilang disimpan di kaunter penyambut tetamu? | insufficient_evidence | 0.00 | model_declined |
| HM05 | Berapa hari cuti ihsan jika ibu bapa meninggal dunia? | insufficient_evidence | 0.00 | model_declined |
| HU02 | What is the hotel limit per night for a business trip to Singapore? | answered | — | The hotel limit per night for a business trip within Malaysia is RM 350 [1]. |
