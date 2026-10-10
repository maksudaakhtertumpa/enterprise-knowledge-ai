"""Retrieval metrics, defined at the *section* level.

Chunk IDs change whenever chunking settings change, so relevance is labelled
as (document filename, section heading). A retrieved chunk is relevant if it
comes from a labelled section; a relevant section counts as found if any of
its chunks is retrieved.

For one query with retrieved results r_1..r_n and relevant sections R:
- Recall@k    = |sections of R found in r_1..r_k| / |R|
- Precision@k = |{i <= k : r_i is relevant}| / k        (k, not n: short lists are penalised)
- Hit@k       = 1 if any of r_1..r_k is relevant else 0
- RR          = 1 / rank of the first relevant result (0 if none retrieved)
- nDCG@k      = binary-gain DCG / ideal DCG, where the ideal ranking places
                each relevant section once at the top
Averages over queries give mean Recall@k, Precision@k, Hit@k, MRR, nDCG@k.
"""

import math
import statistics
from collections.abc import Iterable, Sequence

Section = tuple[str, str]  # (document filename, section heading)


def recall_at_k(retrieved: Sequence[Section], relevant: set[Section], k: int) -> float:
    if not relevant:
        raise ValueError("recall is undefined without relevant items")
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: Sequence[Section], relevant: set[Section], k: int) -> float:
    return sum(1 for r in retrieved[:k] if r in relevant) / k


def hit_at_k(retrieved: Sequence[Section], relevant: set[Section], k: int) -> float:
    return 1.0 if any(r in relevant for r in retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: Sequence[Section], relevant: set[Section]) -> float:
    for rank, r in enumerate(retrieved, start=1):
        if r in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[Section], relevant: set[Section], k: int) -> float:
    seen: set[Section] = set()
    dcg = 0.0
    for i, r in enumerate(retrieved[:k]):
        if r in relevant and r not in seen:  # a section earns gain once
            seen.add(r)
            dcg += 1.0 / math.log2(i + 2)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def percentile(values: Iterable[float], pct: float) -> float:
    data = sorted(values)
    if not data:
        return float("nan")
    if len(data) == 1:
        return data[0]
    # statistics.quantiles with n=100 gives the 1st..99th percentiles.
    return statistics.quantiles(data, n=100, method="inclusive")[int(pct) - 1]


def mean(values: Iterable[float]) -> float:
    data = list(values)
    return sum(data) / len(data) if data else float("nan")
