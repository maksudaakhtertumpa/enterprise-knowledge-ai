import json
import re
from pathlib import Path

import pytest

from eval import metrics
from eval.run_retrieval import DEFAULT_DATASET, load_dataset, main

A, B, C, D = ("a.txt", "A"), ("b.txt", "B"), ("c.txt", "C"), ("d.txt", "D")


# --------------------------------------------------------------- metric maths
def test_recall_precision_hit():
    retrieved = [C, A, D, B]
    relevant = {A, B}
    assert metrics.recall_at_k(retrieved, relevant, 1) == 0
    assert metrics.recall_at_k(retrieved, relevant, 2) == 0.5
    assert metrics.recall_at_k(retrieved, relevant, 4) == 1.0
    assert metrics.precision_at_k(retrieved, relevant, 2) == 0.5
    assert metrics.precision_at_k(retrieved, relevant, 10) == 0.2  # divides by k, not len
    assert metrics.hit_at_k(retrieved, relevant, 1) == 0
    assert metrics.hit_at_k(retrieved, relevant, 2) == 1


def test_duplicate_chunks_of_one_section_count_once_for_recall():
    assert metrics.recall_at_k([A, A, A], {A, B}, 3) == 0.5
    assert metrics.precision_at_k([A, A, C], {A, B}, 3) == pytest.approx(2 / 3)


def test_reciprocal_rank():
    assert metrics.reciprocal_rank([C, D, A], {A}) == pytest.approx(1 / 3)
    assert metrics.reciprocal_rank([A], {A}) == 1.0
    assert metrics.reciprocal_rank([C, D], {A}) == 0.0


def test_ndcg():
    assert metrics.ndcg_at_k([A, B], {A, B}, 10) == pytest.approx(1.0)
    assert metrics.ndcg_at_k([C, A], {A}, 10) == pytest.approx(1 / 1.584962500721156)
    assert metrics.ndcg_at_k([A, A], {A, B}, 10) < 1.0  # repeated section earns gain once
    assert metrics.ndcg_at_k([C], {A}, 10) == 0.0


def test_recall_requires_relevant_items():
    with pytest.raises(ValueError):
        metrics.recall_at_k([A], set(), 1)


def test_percentiles():
    values = list(range(1, 101))
    assert metrics.percentile(values, 50) == pytest.approx(50.5)
    assert metrics.percentile([7.0], 95) == 7.0


# --------------------------------------------------------------- dataset sanity
@pytest.mark.parametrize("name", ["teratai", "teratai-heldout"])
def test_dataset_labels_point_at_real_sections(name):
    files, queries, digest = load_dataset(DEFAULT_DATASET.parent / name)
    sections = {f.name: set(re.findall(r"^# (.+)$", f.read_text(), re.MULTILINE)) for f in files}
    assert len(files) >= 10 and len(digest) == 12
    ids = [q["id"] for q in queries]
    assert len(ids) == len(set(ids)), "query ids must be unique"
    for q in queries:
        assert q["category"] in {"lexical", "paraphrase", "malay", "unanswerable"}
        assert bool(q["relevant"]) == (q["category"] != "unanswerable"), q["id"]
        for rel in q["relevant"]:
            assert rel["section"] in sections[rel["doc"]], (q["id"], rel)


# ------------------------------------------------------------------- runner
def test_runner_end_to_end_with_hashing(tmp_path: Path):
    """Full pipeline on the real dataset, offline. Checks output shape, not quality."""
    assert main(["--backend", "hashing", "--out", str(tmp_path), "--label", "t"]) == 0
    (json_file,) = tmp_path.glob("*-t.json")
    (md_file,) = tmp_path.glob("*-t.md")
    result = json.loads(json_file.read_text())
    assert result["corpus"]["documents"] == 11
    summary = result["results"]["summary"]
    assert set(summary) == {"bm25", "dense", "hybrid", "dense+rerank", "hybrid+rerank"}
    assert result["environment"]["fusion"]["rrf_k"] == 10
    assert result["environment"]["reranker"]["backend"] == "hashing"  # offline run stays offline
    overall = summary["bm25"]["overall"]
    for key in ("recall@1", "recall@10", "precision@1", "mrr", "ndcg@10"):
        assert 0.0 <= overall[key] <= 1.0, key
    assert overall["latency_p95_ms"] >= overall["latency_p50_ms"] > 0
    # Exact-identifier queries are BM25's home ground; anything less means a regression.
    assert summary["bm25"]["by_category"]["lexical"]["recall@5"] >= 0.9
    report = md_file.read_text()
    assert "plumbing check, **not**" in report and "*overlap* reranker" in report
    assert "| bm25 |" in report and "| hybrid+rerank |" in report and "By query type" in report


def test_runner_accepts_fusion_flags(tmp_path: Path):
    args = ["--backend", "hashing", "--modes", "hybrid", "--rrf-k", "20", "--candidates", "15"]
    assert main([*args, "--out", str(tmp_path), "--label", "k20"]) == 0
    result = json.loads(next(tmp_path.glob("*-k20.json")).read_text())
    assert result["environment"]["fusion"] == {
        "rrf_k": 20,
        "candidates": 15,
        "bm25_weight": 0.25,
        "dense_weight": 1.0,
    }
    assert set(result["results"]["summary"]) == {"hybrid"}


def test_result_files_keep_dotted_labels(tmp_path: Path):
    assert (
        main(
            [
                "--backend",
                "hashing",
                "--modes",
                "bm25",
                "--out",
                str(tmp_path),
                "--label",
                "qwen3-embedding-0.6b-mps",
            ]
        )
        == 0
    )
    names = sorted(p.name for p in tmp_path.iterdir())
    assert len(names) == 2
    assert all("qwen3-embedding-0.6b-mps" in n for n in names), names


def test_fusion_sweep_runs_grid_and_ranks(tmp_path: Path):
    from eval.sweep_fusion import main as sweep

    args = [
        "--backend",
        "hashing",
        "--k",
        "10",
        "60",
        "--candidates",
        "5",
        "50",
        "--bm25-weights",
        "0.5",
        "--out",
        str(tmp_path),
        "--label",
        "t",
    ]
    assert sweep(args) == 0
    result = json.loads(next(tmp_path.glob("*fusion-sweep-t.json")).read_text())
    hybrids = [r for r in result["rows"] if r["name"] == "hybrid"]
    assert {(r["rrf_k"], r["candidates"]) for r in hybrids} == {
        (10, 5),
        (10, 50),
        (60, 5),
        (60, 50),
    }
    assert {r["name"] for r in result["rows"]} == {"bm25", "dense", "hybrid"}
    report = next(tmp_path.glob("*fusion-sweep-t.md")).read_text()
    assert "Caution" in report and "hybrid k=10 cand=5 w_bm25=0.5" in report


def test_rerank_flags_are_recorded(tmp_path: Path):
    args = ["--backend", "hashing", "--modes", "dense", "dense+rerank", "--rerank-candidates", "7"]
    assert main([*args, "--out", str(tmp_path), "--label", "rr"]) == 0
    result = json.loads(next(tmp_path.glob("*-rr.json")).read_text())
    assert result["environment"]["rerank_candidates"] == 7
    assert set(result["results"]["summary"]) == {"dense", "dense+rerank"}


def test_rerank_mode_without_reranker_is_rejected(tmp_path: Path):
    with pytest.raises(SystemExit):
        main(
            [
                "--backend",
                "hashing",
                "--reranker-backend",
                "none",
                "--modes",
                "dense+rerank",
                "--out",
                str(tmp_path),
            ]
        )
