# Model selection — MacBook M4, 16 GB unified memory

These are **starting choices**, not final answers. The embedding model is wired in as of Phase 3 and the reranker as of Phase 4 (LLM: Phase 5). `python -m eval.run_retrieval --model <name>` compares embedding models on our own benchmark, and the README will report whichever wins on measured results.

## Recommended stack

| Role | Primary | Size (fp16) | Why | Comparison / fallback |
|---|---|---|---|---|
| Embeddings | `Qwen/Qwen3-Embedding-0.6B` | ~1.2 GB | Apache-2.0, multilingual (English + Malay docs are common in Malaysian companies), 32K context, instruction-aware queries | `BAAI/bge-m3` (MIT, 8K ctx) for the Phase 7 comparison |
| Reranker | `BAAI/bge-reranker-v2-m3` | ~1.1 GB | A true cross-encoder (matches the spec), MIT, multilingual, strong BEIR results for its size | `cross-encoder/ms-marco-MiniLM-L-6-v2` (~90 MB, English-only) for CI and a speed baseline |
| LLM (Ollama) | `qwen3.5:4b` | ~3.3–4 GB download | Leaves headroom for the other two models plus VS Code and a browser | `qwen3.5:9b` (~6.6 GB) for better answers when other apps are closed; `gemma4:e4b` as an alternative family |

All three run on Apple-silicon GPU: sentence-transformers via PyTorch **MPS**, Ollama via Metal.

## Memory budget (approximate)

| Consumer | 4B LLM | 9B LLM |
|---|---|---|
| macOS + VS Code + browser | ~5–6 GB | ~5–6 GB |
| Embedding model | ~1.2 GB | ~1.2 GB |
| Reranker (float16 on MPS) | ~1.1 GB | ~1.1 GB |
| LLM weights + 8K-token KV cache | ~4.5–5 GB | ~7.5–8.5 GB |
| **Total** | **~12–13 GB** ✅ | **~15–17 GB** ⚠️ tight; close other apps |

## Settings that matter on this machine

- **Set the context window explicitly** (`num_ctx` ≈ 8192). If the prompt (question + retrieved chunks) exceeds Ollama's context, it is truncated *silently* — the model never sees some evidence. Phase 5 will count prompt tokens and refuse to send an over-budget prompt.
- **Turn off "thinking" for answers.** Qwen 3.x reasoning traces add latency and tokens; grounded extraction from supplied evidence doesn't need them. (`think: false` in the Ollama API.)
- **Batch embeddings** (e.g. 32 chunks per call) during ingestion; one-at-a-time is far slower on MPS.
- **Rerank ~30 candidates, not 100.** Cross-encoder cost is linear in pairs.
- **Device = auto** → `mps` on your Mac, `cpu` in CI/Docker. Docker on macOS cannot use the Mac GPU, so for the local demo we run models natively and only containerise the API (Phase 8).

## Install
```bash
pip install -r requirements-ml.lock && pip install -e . --no-deps   # PyTorch + sentence-transformers (Phase 3)
brew install ollama && ollama pull qwen3.5:4b                       # Phase 5
```
Embedding and reranker weights download automatically from Hugging Face on first use (cached in `~/.cache/huggingface`). To compare embedding models: `python -m eval.run_retrieval --model BAAI/bge-m3 --label bge-m3`.

## Sources
- Embedding model comparison (2026): https://www.premai.io/blog/best-embedding-models-for-rag-2026-ranked-by-mteb-score-cost-and-self-hosting/
- Open-weight reranker comparison (2026): https://presenc.ai/research/best-open-weight-reranker-models-2026
- Ollama `qwen3.5` tags and sizes: https://ollama.com/library/qwen3.5
- Gemma 4 on a 16 GB M4: https://brain.ansonbiggs.com/Gemma-4-on-M4-Mac-Mini
