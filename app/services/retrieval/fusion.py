"""Reciprocal Rank Fusion (Cormack, Clarke & Büttcher, SIGIR 2009).

    score(d) = Σ_r  w_r / (k + rank_r(d))      summed over the rankings r that contain d

Why RRF rather than adding the scores together:
- BM25 scores are unbounded (2.1, 14.7, ...) while cosine similarities sit in
  [-1, 1]. Adding them needs normalisation, and min-max normalisation is
  fragile: one outlier rescales everything. RRF uses only *ranks*, so the
  two scales never have to be reconciled.
- A document ranked well by *both* retrievers beats one ranked first by only
  one of them, which is the behaviour we want: agreement is evidence.
- `k` (default 60, the value from the original paper) damps the advantage of
  the very top ranks. Small k = trust each list's top hit more; large k =
  flatter, more consensus-driven.

Ties are broken deterministically (best single rank, then ID), so the same
query always returns the same order — important for tests and evaluation.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class FusedItem:
    id: str
    score: float
    # 1-based rank in each input ranking that contained the item.
    ranks: dict[str, int] = field(default_factory=dict)


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]],
    *,
    k: int = 60,
    weights: Mapping[str, float] | None = None,
) -> list[FusedItem]:
    """Fuse named rankings (best first). Returns items sorted by fused score."""
    if k < 1:
        raise ValueError("k must be >= 1")
    weights = weights or {}
    if any(w < 0 for w in weights.values()):
        raise ValueError("weights must be non-negative")

    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    for name, ids in rankings.items():
        weight = weights.get(name, 1.0)
        seen: set[str] = set()
        for position, item in enumerate(ids, start=1):
            if item in seen:  # a retriever listing an ID twice gets credit once
                continue
            seen.add(item)
            ranks.setdefault(item, {})[name] = position
            scores[item] = scores.get(item, 0.0) + weight / (k + position)

    fused = [FusedItem(id=i, score=s, ranks=ranks[i]) for i, s in scores.items()]
    fused.sort(key=lambda f: (-f.score, min(f.ranks.values()), f.id))
    return fused
