# Fusion sweep — qwen3-embedding-0.6b-mps

- Run: 2026-10-10T06:17:38+00:00 · dataset `teratai` (hash `03eb427205b1`) · 47 scored queries
- Embedding: `Qwen/Qwen3-Embedding-0.6B` on `mps`
- Dense weight fixed at 1.0. `≥ best single in all` = hybrid's MRR is at least the better
  of BM25 and dense in every category (the property we want from fusion).

> **Caution:** settings are chosen on the same queries they are scored on, which
> overstates how well they generalise. Prefer a simple setting that holds up in every
> category; confirm on held-out queries (Phase 7).

| Setting | MRR | R@1 | R@5 | lexical MRR | paraphrase MRR | malay MRR | ≥ best single in all |
|---|---|---|---|---|---|---|---|
| bm25 | 0.538 | 0.500 | 0.553 | 1.000 | 0.377 | 0.000 | — |
| dense | 0.900 | 0.819 | 0.979 | 0.882 | 0.915 | 0.900 | — |
| hybrid k=10 cand=5 w_bm25=0.25 | 0.879 | 0.798 | 0.979 | 0.941 | 0.824 | 0.900 |  |
| hybrid k=10 cand=10 w_bm25=0.25 | 0.879 | 0.798 | 0.979 | 0.941 | 0.824 | 0.900 |  |
| hybrid k=10 cand=20 w_bm25=0.25 | 0.879 | 0.798 | 0.979 | 0.941 | 0.824 | 0.900 |  |
| hybrid k=10 cand=50 w_bm25=0.25 | 0.879 | 0.798 | 0.979 | 0.941 | 0.824 | 0.900 |  |
| hybrid k=10 cand=5 w_bm25=0.5 | 0.862 | 0.777 | 0.979 | 0.971 | 0.765 | 0.900 |  |
| hybrid k=10 cand=10 w_bm25=0.5 | 0.862 | 0.777 | 0.979 | 0.971 | 0.765 | 0.900 |  |
| hybrid k=30 cand=5 w_bm25=0.25 | 0.862 | 0.777 | 0.936 | 0.941 | 0.787 | 0.900 |  |
| hybrid k=30 cand=10 w_bm25=0.25 | 0.862 | 0.777 | 0.936 | 0.941 | 0.787 | 0.900 |  |
| hybrid k=30 cand=20 w_bm25=0.25 | 0.861 | 0.777 | 0.936 | 0.941 | 0.785 | 0.900 |  |
| hybrid k=30 cand=50 w_bm25=0.25 | 0.861 | 0.777 | 0.936 | 0.941 | 0.784 | 0.900 |  |
| hybrid k=10 cand=20 w_bm25=0.5 | 0.860 | 0.777 | 0.979 | 0.971 | 0.761 | 0.900 |  |
| hybrid k=10 cand=50 w_bm25=0.5 | 0.860 | 0.777 | 0.979 | 0.971 | 0.761 | 0.900 |  |
| hybrid k=30 cand=5 w_bm25=0.5 | 0.828 | 0.755 | 0.936 | 0.971 | 0.692 | 0.900 |  |
| hybrid k=30 cand=10 w_bm25=0.5 | 0.828 | 0.755 | 0.936 | 0.971 | 0.692 | 0.900 |  |
| hybrid k=60 cand=5 w_bm25=0.5 | 0.828 | 0.755 | 0.936 | 0.971 | 0.691 | 0.900 |  |
| hybrid k=60 cand=10 w_bm25=0.5 | 0.828 | 0.755 | 0.936 | 0.971 | 0.691 | 0.900 |  |
| hybrid k=30 cand=20 w_bm25=0.5 | 0.827 | 0.755 | 0.915 | 0.971 | 0.689 | 0.900 |  |
| hybrid k=10 cand=5 w_bm25=1.0 | 0.824 | 0.755 | 0.936 | 1.000 | 0.660 | 0.900 |  |
| hybrid k=10 cand=10 w_bm25=1.0 | 0.824 | 0.755 | 0.936 | 1.000 | 0.660 | 0.900 |  |
| hybrid k=30 cand=50 w_bm25=0.5 | 0.824 | 0.755 | 0.915 | 0.971 | 0.683 | 0.900 |  |
| hybrid k=60 cand=20 w_bm25=0.5 | 0.822 | 0.755 | 0.894 | 0.971 | 0.679 | 0.900 |  |
| hybrid k=10 cand=20 w_bm25=1.0 | 0.820 | 0.755 | 0.915 | 1.000 | 0.651 | 0.900 |  |
| hybrid k=30 cand=5 w_bm25=1.0 | 0.819 | 0.755 | 0.915 | 1.000 | 0.650 | 0.900 |  |
| hybrid k=30 cand=10 w_bm25=1.0 | 0.819 | 0.755 | 0.915 | 1.000 | 0.650 | 0.900 |  |
| hybrid k=60 cand=5 w_bm25=0.25 | 0.818 | 0.734 | 0.936 | 0.941 | 0.693 | 0.900 |  |
| hybrid k=60 cand=10 w_bm25=0.25 | 0.818 | 0.734 | 0.936 | 0.941 | 0.693 | 0.900 |  |
| hybrid k=60 cand=20 w_bm25=0.25 | 0.818 | 0.734 | 0.936 | 0.941 | 0.692 | 0.900 |  |
| hybrid k=60 cand=50 w_bm25=0.25 | 0.818 | 0.734 | 0.936 | 0.941 | 0.692 | 0.900 |  |
| hybrid k=60 cand=5 w_bm25=1.0 | 0.818 | 0.755 | 0.915 | 1.000 | 0.647 | 0.900 |  |
| hybrid k=60 cand=10 w_bm25=1.0 | 0.818 | 0.755 | 0.915 | 1.000 | 0.647 | 0.900 |  |
| hybrid k=10 cand=50 w_bm25=1.0 | 0.816 | 0.755 | 0.851 | 1.000 | 0.643 | 0.900 |  |
| hybrid k=30 cand=20 w_bm25=1.0 | 0.815 | 0.755 | 0.894 | 1.000 | 0.640 | 0.900 |  |
| hybrid k=60 cand=20 w_bm25=1.0 | 0.813 | 0.755 | 0.894 | 1.000 | 0.636 | 0.900 |  |
| hybrid k=60 cand=50 w_bm25=0.5 | 0.811 | 0.755 | 0.851 | 0.971 | 0.654 | 0.900 |  |
| hybrid k=30 cand=50 w_bm25=1.0 | 0.806 | 0.755 | 0.851 | 1.000 | 0.622 | 0.900 |  |
| hybrid k=60 cand=50 w_bm25=1.0 | 0.804 | 0.755 | 0.851 | 1.000 | 0.618 | 0.900 |  |
