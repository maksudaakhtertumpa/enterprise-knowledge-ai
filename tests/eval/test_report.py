import json
import xml.etree.ElementTree as ET
from pathlib import Path

from eval import charts
from eval.report import build, is_standin, load_runs, main, pick

NS = {"s": "http://www.w3.org/2000/svg"}


def test_bar_chart_is_valid_accessible_svg():
    svg = charts.bar_chart(
        "MRR <by> mode", ["bm25", "dense"], {"dev": [0.5, 0.9], "held-out": [None, 0.8]}
    )
    root = ET.fromstring(svg)  # well-formed, title escaped
    assert root.find("s:title", NS).text == "MRR <by> mode"
    desc = root.find("s:desc", NS).text
    assert "bm25: dev 50.0%, held-out n/a" in desc  # numbers readable without the picture
    assert len(root.findall("s:rect", NS)) == 1 + 2 + 3  # background, legend keys, 3 bars
    assert "90%" in svg and "n/a" in svg


def test_line_chart_marks_the_configured_threshold():
    svg = charts.line_chart(
        "Trade-off",
        [0.0, 0.5, 0.9],
        {"Answer rate": [1.0, 0.9, 0.4], "Refusals": [0.5, 0.8, None]},
        marker_x=0.5,
        marker_label="configured 0.5",
    )
    root = ET.fromstring(svg)
    assert len(root.findall("s:path", NS)) >= 2  # two lines (+ triangle markers if any)
    assert "configured 0.5" in svg and "stroke-dasharray" in svg


def _retrieval(dataset, modes, stamp, backend="sentence-transformers"):
    summary = {
        m: {
            "overall": {
                "mrr": 0.9,
                "recall@1": 0.8,
                "recall@5": 0.9,
                "recall@10": 1.0,
                "ndcg@10": 0.9,
                "latency_p50_ms": 30.0,
            },
            "by_category": {"paraphrase": {"mrr": 0.85}},
        }
        for m in modes
    }
    return {
        "label": "r",
        "timestamp": stamp,
        "dataset": dataset,
        "dataset_hash": "abc",
        "environment": {"embedding": {"backend": backend, "model": "emb"}, "platform": "mac"},
        "results": {"summary": summary, "scored_queries": 38},
    }


def _answers(dataset, stamp, llm="ollama", entailment="nli"):
    overall = {
        "questions": 40,
        "answer_rate": 0.9,
        "fact_recall": 0.8,
        "citation_precision": 0.9,
        "correct_refusal_rate": 0.8,
        "unsupported_sentence_rate": 0.0,
        "latency_p50_ms": 5000,
        "latency_p95_ms": 9000,
    }
    table = [
        {
            "threshold": t,
            "answer_rate": 0.9,
            "fact_recall": 0.8,
            "correct_refusal_rate": 0.8,
            "sentences_removed": t / 2,
        }
        for t in (0.0, 0.5)
    ]
    return {
        "label": "a",
        "timestamp": stamp,
        "dataset": dataset,
        "dataset_hash": "def",
        "environment": {
            "embedding": {"backend": "sentence-transformers", "model": "emb"},
            "llm": {"backend": llm, "model": "qwen"},
        },
        "entailment": {"backend": entailment, "model": "nli-model"},
        "summary": {"overall": overall},
        "gate": [],
        "entailment_table": table,
        "injection": [{"resisted": True}],
        "rows": [],
    }


def _write(tmp_path: Path, name: str, result: dict) -> None:
    (tmp_path / name).write_text(json.dumps(result))


def test_standins_are_excluded_unless_asked(tmp_path: Path):
    _write(tmp_path, "1.json", _retrieval("teratai", ["dense"], "2026-10-01"))
    _write(tmp_path, "2.json", _retrieval("teratai", ["dense"], "2026-10-02", backend="hashing"))
    _write(tmp_path, "3.json", _answers("teratai", "2026-10-02", llm="extractive"))
    _write(tmp_path, "4.json", _answers("teratai", "2026-10-02", entailment="lexical"))
    _write(tmp_path, "5.json", {"rows": [], "label": "fusion sweep"})  # not a benchmark run
    (tmp_path / "broken.json").write_text("{")
    assert [r["_file"] for r in load_runs(tmp_path, False)] == ["1.json"]
    everything = load_runs(tmp_path, True)
    assert len(everything) == 4 and sum(r["_standin"] for r in everything) == 3
    assert is_standin(_answers("teratai", "x", entailment="lexical"))
    report, _ = build(everything, 0.5)
    assert "offline stand-in, not a quality result" in report


def test_pick_prefers_the_most_complete_then_newest_run(tmp_path: Path):
    _write(tmp_path, "full.json", _retrieval("teratai", ["bm25", "dense", "hybrid"], "2026-10-01"))
    _write(tmp_path, "quick.json", _retrieval("teratai", ["bm25"], "2026-10-05"))
    _write(tmp_path, "full2.json", _retrieval("teratai", ["bm25", "dense", "hybrid"], "2026-10-03"))
    runs = load_runs(tmp_path, False)
    assert pick(runs, "retrieval", "teratai")["_file"] == "full2.json"
    assert pick(runs, "retrieval", "teratai-heldout") is None


def test_report_end_to_end(tmp_path: Path):
    results, out = tmp_path / "results", tmp_path / "out"
    results.mkdir()
    _write(results, "r-dev.json", _retrieval("teratai", ["bm25", "dense"], "2026-10-01"))
    _write(results, "r-ho.json", _retrieval("teratai-heldout", ["bm25", "dense"], "2026-10-01"))
    _write(results, "a-dev.json", _answers("teratai", "2026-10-01"))
    _write(results, "a-ho.json", _answers("teratai-heldout", "2026-10-01"))
    assert main(["--results", str(results), "--out", str(out), "--threshold", "0.5"]) == 0
    names = {p.name for p in out.iterdir()}
    assert names == {
        "README.md",
        "retrieval-mrr.svg",
        "retrieval-categories.svg",
        "answers.svg",
        "entailment-development.svg",
        "entailment-held-out.svg",
    }
    report = (out / "README.md").read_text()
    for needle in ("| held-out | dense |", "`a-ho.json`", "2 of 2 planted", "Entailment threshold"):
        assert needle in report, needle
    assert "offline stand-in" not in report
    assert "configured 0.5" in (out / "entailment-held-out.svg").read_text()
    # The by-type chart describes the held-out set when there is one.
    assert "held-out set" in (out / "retrieval-categories.svg").read_text()


def test_report_without_runs_says_so(tmp_path: Path):
    assert (
        main(["--results", str(tmp_path), "--out", str(tmp_path / "o"), "--threshold", "0.5"]) == 0
    )
    assert "No benchmark runs found" in (tmp_path / "o" / "README.md").read_text()
