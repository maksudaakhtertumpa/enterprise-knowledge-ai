"""Fusion sweep: try many hybrid (RRF) settings on one ingested corpus.

    python -m eval.sweep_fusion                       # default grid, real model from .env
    python -m eval.sweep_fusion --k 20 60 --candidates 10 50 --bm25-weights 0.5 1.0

Ingests the dataset once, measures BM25 and dense once as baselines, then
evaluates hybrid for every (k, candidates, bm25_weight) combination, with the
dense weight fixed at 1.0. Writes a ranked table to eval/results/.

Caution (written into the report too): choosing settings on the same 47
queries we report on overstates how well they generalise. Prefer a simple
setting that wins in every category over the single best overall number, and
confirm it on held-out queries in Phase 7.
"""

import argparse
import itertools
import json
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.core.security import create_workspace_with_key
from app.db.session import init_db
from app.services.container import build_services
from app.services.retrieval.embeddings import check_compatibility
from app.services.retrieval.search import FusionConfig
from eval.run_retrieval import (
    DEFAULT_DATASET,
    DEFAULT_OUT,
    KS,
    environment,
    evaluate,
    ingest_corpus,
    load_dataset,
)

CATEGORIES = ("lexical", "paraphrase", "malay")


def _row(name: str, summary: dict[str, Any], **params: Any) -> dict[str, Any]:
    overall, by_cat = summary["overall"], summary["by_category"]
    return {
        "name": name,
        **params,
        "mrr": overall["mrr"],
        "recall@1": overall["recall@1"],
        "recall@5": overall["recall@5"],
        **{f"{c}_mrr": by_cat[c]["mrr"] for c in CATEGORIES if c in by_cat},
    }


def render(result: dict[str, Any]) -> str:
    rows = result["rows"]
    base = {r["name"]: r for r in rows if r["name"] in ("bm25", "dense")}
    best_single = {c: max(base[n][f"{c}_mrr"] for n in base) for c in CATEGORIES}
    header = [
        "Setting",
        "MRR",
        "R@1",
        "R@5",
        *[f"{c} MRR" for c in CATEGORIES],
        "≥ best single in all",
    ]
    lines = [
        f"# Fusion sweep — {result['label']}",
        "",
        f"- Run: {result['timestamp']} · dataset `{result['dataset']}` "
        f"(hash `{result['dataset_hash']}`) · {result['scored_queries']} scored queries",
        f"- Embedding: `{result['environment']['embedding'].get('model')}` on "
        f"`{result['environment']['embedding'].get('device', 'n/a')}`",
        "- Dense weight fixed at 1.0. `≥ best single in all` = hybrid's MRR is at least the better",
        "  of BM25 and dense in every category (the property we want from fusion).",
        "",
        "> **Caution:** settings are chosen on the same queries they are scored on, which",
        "> overstates how well they generalise. Prefer a simple setting that holds up in every",
        "> category; confirm on held-out queries (Phase 7).",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    hybrids = sorted(
        (r for r in rows if r["name"] == "hybrid"),
        key=lambda r: (-r["mrr"], -r["recall@1"], r["candidates"], r["rrf_k"]),
    )
    for r in [base["bm25"], base["dense"], *hybrids]:
        if r["name"] == "hybrid":
            label = f"hybrid k={r['rrf_k']} cand={r['candidates']} w_bm25={r['bm25_weight']}"
            ok = all(r[f"{c}_mrr"] >= best_single[c] - 1e-9 for c in CATEGORIES)
            flag = "yes" if ok else ""
        else:
            label, flag = r["name"], "—"
        cells = [
            label,
            f"{r['mrr']:.3f}",
            f"{r['recall@1']:.3f}",
            f"{r['recall@5']:.3f}",
            *[f"{r[f'{c}_mrr']:.3f}" for c in CATEGORIES],
            flag,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--backend", choices=["sentence-transformers", "hashing"])
    parser.add_argument("--model")
    parser.add_argument("--k", type=int, nargs="+", default=[10, 30, 60])
    parser.add_argument("--candidates", type=int, nargs="+", default=[5, 10, 20, 50])
    parser.add_argument("--bm25-weights", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    parser.add_argument("--label")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    files, queries, dataset_hash = load_dataset(args.dataset)
    data_dir = Path(tempfile.mkdtemp(prefix="eka-sweep-"))
    # The sweep never reranks, so don't load a reranker model.
    overrides: dict[str, Any] = {
        "data_dir": data_dir,
        "log_level": "WARNING",
        "reranker_backend": "none",
    }
    if args.backend:
        overrides["embedding_backend"] = args.backend
    if args.model:
        overrides["embedding_model"] = args.model
    settings = Settings(**overrides)

    grid = list(itertools.product(args.k, args.candidates, args.bm25_weights))
    rows: list[dict[str, Any]] = []
    try:
        services = build_services(settings)
        init_db(services.engine)
        print(f"Loading embedder ({settings.embedding_backend}: {settings.embedding_model}) ...")
        services.embedder.load()
        check_compatibility(services.embedder, settings)
        with services.session_factory() as db:
            workspace, _ = create_workspace_with_key(db, "sweep")
            db.commit()
            workspace_id = workspace.id
        print(f"Ingesting {len(files)} documents ...")
        ingest_corpus(services, workspace_id, files)

        baselines = evaluate(services, workspace_id, queries, ["bm25", "dense"], KS)
        scored = baselines["scored_queries"]
        for name in ("bm25", "dense"):
            rows.append(_row(name, baselines["summary"][name]))

        for i, (k, candidates, weight) in enumerate(grid, start=1):
            print(f"  [{i}/{len(grid)}] k={k} candidates={candidates} bm25_weight={weight}")
            fusion = FusionConfig(
                rrf_k=k, candidates=candidates, bm25_weight=weight, dense_weight=1.0
            )
            res = evaluate(services, workspace_id, queries, ["hybrid"], KS, fusion)
            rows.append(
                _row(
                    "hybrid",
                    res["summary"]["hybrid"],
                    rrf_k=k,
                    candidates=candidates,
                    bm25_weight=weight,
                )
            )
        env = environment(services)
    finally:
        if "services" in locals():
            services.close()
        shutil.rmtree(data_dir, ignore_errors=True)

    emb = env["embedding"]
    label = args.label or (
        f"{emb['model'].split('/')[-1]}-{emb.get('device') or 'cpu'}".lower().replace(" ", "-")
    )
    stamp = datetime.now(UTC)
    result = {
        "label": label,
        "timestamp": stamp.isoformat(timespec="seconds"),
        "dataset": args.dataset.name,
        "dataset_hash": dataset_hash,
        "scored_queries": scored,
        "environment": env,
        "rows": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{stamp:%Y%m%d-%H%M%S}-fusion-sweep-{label}"
    (args.out / f"{stem}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    report = render(result)
    (args.out / f"{stem}.md").write_text(report)
    print()
    print(report)
    print(f"Saved {stem}.json and {stem}.md in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
