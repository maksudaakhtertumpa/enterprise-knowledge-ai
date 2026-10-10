"""Retrieval benchmark: BM25, dense, hybrid (RRF) and cross-encoder reranking on labelled data.

Ingests the corpus through the *real* ingestion pipeline into a throwaway
data directory, runs every labelled query in every mode, and writes a JSON
file (all numbers + per-query detail + environment) and a Markdown report.

    # On your Mac, with the real embedding model:
    python -m eval.run_retrieval

    # Another model, for comparison:
    python -m eval.run_retrieval --model BAAI/bge-m3 --label bge-m3

    # RRF sensitivity:
    python -m eval.run_retrieval --rrf-k 20 --label k20

    # Offline plumbing check (NOT a semantic result):
    python -m eval.run_retrieval --backend hashing

Nothing here is tuned to the dataset: chunking, BM25 weights and the model are
the application's defaults unless you pass flags.
"""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.core.security import create_workspace_with_key
from app.db.models import Document, DocumentStatus, FileType
from app.db.session import init_db
from app.services.container import build_services
from app.services.ingestion.pipeline import ingest_document
from app.services.retrieval.embeddings import check_compatibility
from app.services.retrieval.search import SearchMode, SearchService
from app.services.storage import LocalFileStorage
from eval import metrics

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "eval" / "datasets" / "teratai"
DEFAULT_OUT = ROOT / "eval" / "results"
KS = (1, 3, 5, 10)


def load_dataset(path: Path) -> tuple[list[Path], list[dict[str, Any]], str]:
    corpus = sorted((path / "corpus").glob("*.txt"))
    queries = [
        json.loads(line) for line in (path / "queries.jsonl").read_text().splitlines() if line
    ]
    digest = hashlib.sha256()
    for f in [*corpus, path / "queries.jsonl"]:
        digest.update(f.name.encode())
        digest.update(f.read_bytes())
    return corpus, queries, digest.hexdigest()[:12]


def ingest_corpus(services, workspace_id: str, files: list[Path]) -> dict[str, Any]:
    storage = LocalFileStorage(services.settings.uploads_dir)
    t0 = time.perf_counter()
    for path in files:
        doc_id = str(uuid.uuid4())
        with path.open("rb") as fh:
            stored = storage.save(
                fh, workspace_id=workspace_id, document_id=doc_id, ext="txt", max_bytes=10**8
            )
        with services.session_factory() as db:
            db.add(
                Document(
                    id=doc_id,
                    workspace_id=workspace_id,
                    filename=path.name,
                    file_type=FileType.TXT,
                    size_bytes=stored.size_bytes,
                    sha256=stored.sha256,
                    storage_key=stored.key,
                )
            )
            db.commit()
        ingest_document(services, doc_id)
    with services.session_factory() as db:
        docs = db.query(Document).filter_by(workspace_id=workspace_id).all()
        failed = [d.filename for d in docs if d.status is not DocumentStatus.READY]
        if failed:
            raise SystemExit(f"Ingestion failed for: {failed}")
        return {
            "documents": len(docs),
            "chunks": sum(d.chunk_count for d in docs),
            "ingest_seconds": round(time.perf_counter() - t0, 2),
        }


CONFIGS = ("bm25", "dense", "hybrid", "bm25+rerank", "dense+rerank", "hybrid+rerank")


def parse_config(config: str) -> tuple[SearchMode, bool]:
    """'hybrid+rerank' -> (HYBRID, True); 'dense' -> (DENSE, False)."""
    name, _, suffix = str(config).partition("+")
    return SearchMode(name), suffix == "rerank"


def evaluate(services, workspace_id, queries, modes, ks, fusion=None) -> dict[str, Any]:
    """Run every scored query under each config ('bm25', 'dense+rerank', ...)."""
    fusion = fusion or services.settings.fusion
    max_k = max(ks)
    scored = [q for q in queries if q["relevant"]]
    per_query: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}

    def service(db):
        return SearchService(
            db,
            services.embedder,
            services.vectors,
            fusion,
            reranker=services.reranker,
            rerank_candidates=services.settings.rerank_candidates,
        )

    for config in modes:
        mode, rerank = parse_config(config)
        name = f"{mode.value}+rerank" if rerank else mode.value
        rows = []
        # One warm-up query so model/kernel initialisation isn't counted as latency.
        with services.session_factory() as db:
            service(db).search(workspace_id, "warm up", mode=mode, top_k=1, rerank=rerank)
        for q in scored:
            relevant = {(r["doc"], r["section"]) for r in q["relevant"]}
            with services.session_factory() as db:
                outcome = service(db).search(
                    workspace_id, q["query"], mode=mode, top_k=max_k, rerank=rerank
                )
            retrieved = [(h.document.filename, h.chunk.heading or "") for h in outcome.hits]
            row = {
                "id": q["id"],
                "category": q["category"],
                "mode": name,
                "rr": metrics.reciprocal_rank(retrieved, relevant),
                "latency_ms": outcome.timings_ms["total_ms"],
                "top3": [f"{d} > {s}" for d, s in retrieved[:3]],
            }
            for k in ks:
                row[f"recall@{k}"] = metrics.recall_at_k(retrieved, relevant, k)
                row[f"precision@{k}"] = metrics.precision_at_k(retrieved, relevant, k)
                row[f"hit@{k}"] = metrics.hit_at_k(retrieved, relevant, k)
                row[f"ndcg@{k}"] = metrics.ndcg_at_k(retrieved, relevant, k)
            rows.append(row)
        per_query.extend(rows)
        summary[name] = {
            "overall": _aggregate(rows, ks),
            "by_category": {
                cat: _aggregate([r for r in rows if r["category"] == cat], ks)
                for cat in sorted({r["category"] for r in rows})
            },
        }
    return {"summary": summary, "per_query": per_query, "scored_queries": len(scored)}


def _aggregate(rows: list[dict[str, Any]], ks) -> dict[str, float]:
    out = {"queries": len(rows), "mrr": metrics.mean(r["rr"] for r in rows)}
    for k in ks:
        for name in ("recall", "precision", "hit", "ndcg"):
            out[f"{name}@{k}"] = metrics.mean(r[f"{name}@{k}"] for r in rows)
    latencies = [r["latency_ms"] for r in rows]
    out["latency_p50_ms"] = metrics.percentile(latencies, 50)
    out["latency_p95_ms"] = metrics.percentile(latencies, 95)
    return {k: round(v, 4) for k, v in out.items()}


def environment(services) -> dict[str, Any]:
    env = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "embedding": services.embedder.describe(),
        "chunk_size_tokens": services.settings.chunk_size_tokens,
        "chunk_overlap_tokens": services.settings.chunk_overlap_tokens,
        "reranker": services.reranker.describe() if services.reranker else None,
        "rerank_candidates": services.settings.rerank_candidates,
        "fusion": {
            "rrf_k": services.settings.rrf_k,
            "candidates": services.settings.hybrid_candidates,
            "bm25_weight": services.settings.hybrid_bm25_weight,
            "dense_weight": services.settings.hybrid_dense_weight,
        },
    }
    try:
        import torch

        env["torch"] = torch.__version__
    except ImportError:
        env["torch"] = None
    try:
        env["git_commit"] = (
            subprocess.run(  # noqa: S603, S607
                ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
            ).stdout.strip()
            or None
        )
    except OSError:
        env["git_commit"] = None
    return env


def render_markdown(result: dict[str, Any], ks) -> str:
    env, summary = result["environment"], result["results"]["summary"]
    emb = env["embedding"]
    lines = [
        f"# Retrieval benchmark — {result['label']}",
        "",
        f"- Run: {result['timestamp']} · dataset `{result['dataset']}` "
        f"(hash `{result['dataset_hash']}`)",
        f"- Corpus: {result['corpus']['documents']} documents, "
        f"{result['corpus']['chunks']} chunks; "
        f"{result['results']['scored_queries']} scored queries",
        f"- Embedding: `{emb.get('model')}` on `{emb.get('device', 'n/a')}` "
        f"(dim {emb.get('dimension')}); chunks {env['chunk_size_tokens']} tokens, "
        f"overlap {env['chunk_overlap_tokens']}",
        f"- Hybrid: RRF k={env['fusion']['rrf_k']}, {env['fusion']['candidates']} candidates "
        f"per retriever, weights bm25={env['fusion']['bm25_weight']} "
        f"dense={env['fusion']['dense_weight']}",
        f"- Machine: {env['platform']} · Python {env['python']} · torch {env['torch']}",
        "",
    ]
    reranked = any("+rerank" in m for m in summary)
    rr = env.get("reranker") or {}
    if reranked:
        lines.insert(
            -2,
            f"- Reranker: `{rr.get('model')}` on `{rr.get('device', 'n/a')}` "
            f"({rr.get('dtype', 'n/a')}), reordering the top {env['rerank_candidates']} candidates",
        )
    stand_ins = []
    if emb.get("backend") == "hashing" and any(m != "bm25" for m in summary):
        stand_ins.append("the *hashing* embedder (dense and hybrid rows)")
    if reranked and rr.get("backend") == "hashing":
        stand_ins.append("the *overlap* reranker (+rerank rows)")
    if stand_ins:
        lines += [
            f"> **Warning:** this run uses {' and '.join(stand_ins)}, offline stand-ins",
            "> that only match shared words. Those rows are a plumbing check, **not** a",
            "> semantic-search result.",
            "",
        ]
    header = [
        "Mode",
        *[f"Recall@{k}" for k in ks],
        "Precision@1",
        "MRR",
        f"nDCG@{max(ks)}",
        "p50 ms",
        "p95 ms",
    ]
    lines += ["## Overall", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for mode, data in summary.items():
        o = data["overall"]
        cells = [
            mode,
            *[f"{o[f'recall@{k}']:.3f}" for k in ks],
            f"{o['precision@1']:.3f}",
            f"{o['mrr']:.3f}",
            f"{o[f'ndcg@{max(ks)}']:.3f}",
            f"{o['latency_p50_ms']:.1f}",
            f"{o['latency_p95_ms']:.1f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")

    categories = sorted({c for d in summary.values() for c in d["by_category"]})
    lines += [
        "",
        "## By query type (Recall@5 / MRR)",
        "",
        "| Category | Queries | " + " | ".join(summary) + " |",
        "|---|---|" + "---|" * len(summary),
    ]
    for cat in categories:
        n = next(iter(summary.values()))["by_category"][cat]["queries"]
        cells = [
            f"{d['by_category'][cat]['recall@5']:.3f} / {d['by_category'][cat]['mrr']:.3f}"
            for d in summary.values()
        ]
        lines.append(f"| {cat} | {n} | " + " | ".join(cells) + " |")

    # Where the modes disagree: the cases that show what each retriever adds.
    if len(summary) >= 2:
        by_id: dict[str, dict[str, float]] = {}
        for row in result["results"]["per_query"]:
            by_id.setdefault(row["id"], {})[row["mode"]] = row["rr"]
        modes = list(summary)
        spread = sorted(
            by_id.items(), key=lambda kv: (-(max(kv[1].values()) - min(kv[1].values())), kv[0])
        )
        lines += [
            "",
            "## Largest disagreements between modes (reciprocal rank of first relevant hit)",
            "",
            "| Query | " + " | ".join(modes) + " |",
            "|---|" + "---|" * len(modes),
        ]
        queries = {q["id"]: q["query"] for q in result["queries"]}
        for qid, rr in spread[:12]:
            if max(rr.values()) == min(rr.values()):
                break
            cells = " | ".join(f"{rr[m]:.2f}" for m in modes)
            lines.append(f"| {qid}: {queries[qid]} | {cells} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--backend", choices=["sentence-transformers", "hashing"])
    parser.add_argument("--model", help="Embedding model (default: EKA_EMBEDDING_MODEL)")
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["bm25", "dense", "hybrid", "dense+rerank", "hybrid+rerank"],
        choices=list(CONFIGS),
    )
    parser.add_argument("--reranker-backend", choices=["cross-encoder", "hashing", "none"])
    parser.add_argument("--reranker-model", help="Default: EKA_RERANKER_MODEL")
    parser.add_argument("--rerank-candidates", type=int, help="Candidates the reranker reorders")
    parser.add_argument("--rrf-k", type=int, help="RRF k (default: EKA_RRF_K, 60)")
    parser.add_argument("--candidates", type=int, help="Per-retriever depth before fusion")
    parser.add_argument("--label", help="Name for the result files")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    files, queries, dataset_hash = load_dataset(args.dataset)
    data_dir = Path(tempfile.mkdtemp(prefix="eka-bench-"))
    overrides: dict[str, Any] = {"data_dir": data_dir, "log_level": "WARNING"}
    if args.backend:
        overrides["embedding_backend"] = args.backend
    if args.model:
        overrides["embedding_model"] = args.model
    if args.device:
        overrides["embedding_device"] = args.device
    if args.rrf_k:
        overrides["rrf_k"] = args.rrf_k
    if args.candidates:
        overrides["hybrid_candidates"] = args.candidates
    # An offline embedder run is an offline run: use the offline reranker too unless told.
    reranker_backend = args.reranker_backend or ("hashing" if args.backend == "hashing" else None)
    if reranker_backend:
        overrides["reranker_backend"] = reranker_backend
    if args.reranker_model:
        overrides["reranker_model"] = args.reranker_model
    if args.rerank_candidates:
        overrides["rerank_candidates"] = args.rerank_candidates
    settings = Settings(**overrides)
    if any("+rerank" in m for m in args.modes) and settings.reranker_backend == "none":
        parser.error("rerank modes need a reranker (EKA_RERANKER_BACKEND is 'none')")

    try:
        services = build_services(settings)
        init_db(services.engine)
        print(f"Loading embedder ({settings.embedding_backend}: {settings.embedding_model}) ...")
        services.embedder.load()
        check_compatibility(services.embedder, settings)
        if services.reranker and any("+rerank" in m for m in args.modes):
            print(f"Loading reranker ({settings.reranker_backend}: {settings.reranker_model}) ...")
            services.reranker.load()
        with services.session_factory() as db:
            workspace, _ = create_workspace_with_key(db, "benchmark")
            db.commit()
            workspace_id = workspace.id

        print(f"Ingesting {len(files)} documents ...")
        corpus = ingest_corpus(services, workspace_id, files)
        print(f"  {corpus['chunks']} chunks in {corpus['ingest_seconds']} s")

        modes = args.modes
        print(
            f"Running {sum(1 for q in queries if q['relevant'])} queries x {len(modes)} modes ..."
        )
        results = evaluate(services, workspace_id, queries, modes, KS)
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
        "corpus": corpus,
        "environment": env,
        "results": results,
        "queries": [{"id": q["id"], "query": q["query"]} for q in queries],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    # Build names by concatenation: Path.with_suffix() would treat the ".6b-mps"
    # in a label like "qwen3-embedding-0.6b-mps" as an extension and drop it.
    stem = f"{stamp:%Y%m%d-%H%M%S}-{label}"
    json_path, md_path = args.out / f"{stem}.json", args.out / f"{stem}.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    report = render_markdown(result, KS)
    md_path.write_text(report)
    print()
    print(report)
    print(f"Saved {json_path.name} and {md_path.name} in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
