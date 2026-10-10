import json
from pathlib import Path

import pytest

from eval.run_answers import (
    DATASET,
    INJECTION,
    contains_phrase,
    fact_recall,
    gate_table,
    main,
    summarize,
)


@pytest.mark.parametrize(
    ("text", "phrase", "expected"),
    [
        ("You get 14 working days [1].", "14", True),
        ("You get 140 days.", "14", False),  # whole words only
        ("Paid on the 25th.", "25th", True),
        ("Contact the D.P.O.", "DPO", False),
        ("Report it to the Data Protection Officer!", "data protection officer", True),
        ("RM80 per month", "RM80", True),
        ("RM 80 per month", "RM80", False),
    ],
)
def test_contains_phrase(text, phrase, expected):
    assert contains_phrase(text, phrase) is expected


def test_fact_recall_partial_credit_and_alternatives():
    facts = [["14"], ["18"], ["22"]]
    assert fact_recall("14 or 18 days [1].", facts) == pytest.approx(2 / 3)
    assert fact_recall("Within two hours [1].", [["two hours", "2 hours"]]) == 1.0
    assert fact_recall(None, facts) == 0.0


def _row(cat, answered, score, facts=1.0, cites=1, on_rel=1):
    return {
        "category": cat,
        "answered": answered,
        "top_rerank_score": score,
        "fact_recall": facts if cat != "unanswerable" else None,
        "citations": cites if answered else 0,
        "citations_on_relevant": on_rel if answered else 0,
        "citations_proposed": cites,
        "sentences_total": 2,
        "sentences_removed": 0,
        "latency_ms": 1000.0,
        "prompt_tokens": 900,
        "completion_tokens": 50,
    }


def test_summary_and_gate_table():
    rows = [
        _row("paraphrase", True, 0.9),
        _row("paraphrase", True, 0.04, facts=0.5),
        _row("unanswerable", True, 0.03),  # model wrongly answered
        _row("unanswerable", False, 0.01),
    ]
    s = summarize(rows)
    assert (s["answer_rate"], s["correct_refusal_rate"], s["fact_recall"]) == (1.0, 0.5, 0.75)
    assert s["cost_usd"] == 0.0
    gate = {g["threshold"]: g for g in gate_table(rows)}
    assert gate[0.0]["correct_refusal_rate"] == 0.5
    # A 0.035 gate isn't in the table, but 0.05 blocks the wrong answer AND loses one answerable.
    assert gate[0.05]["correct_refusal_rate"] == 1.0 and gate[0.05]["answer_rate"] == 0.5
    assert gate[0.03]["llm_calls_saved"] == 0.25


def test_answer_facts_cover_every_answerable_question():
    queries = [json.loads(line) for line in (DATASET / "queries.jsonl").read_text().splitlines()]
    facts = {
        json.loads(line)["id"] for line in (DATASET / "answers.jsonl").read_text().splitlines()
    }
    questions = {q["id"] for q in queries if q["category"] in {"paraphrase", "malay"}}
    assert questions == facts
    assert sum(q["category"] == "unanswerable" for q in queries) == 12


def test_injection_set_is_consistent():
    queries = [json.loads(line) for line in (INJECTION / "queries.jsonl").read_text().splitlines()]
    attacks = {p.name for p in (INJECTION / "attacks").glob("*.txt")}
    assert {q["attack"] for q in queries} == attacks
    for q in queries:
        assert q["payload"] and q["expected"]


def test_runner_end_to_end_offline(tmp_path: Path):
    args = [
        "--backend",
        "hashing",
        "--reranker-backend",
        "hashing",
        "--llm-backend",
        "extractive",
        "--limit",
        "6",
        "--out",
        str(tmp_path),
        "--label",
        "t",
    ]
    assert main(args) == 0
    result = json.loads(next(tmp_path.glob("*answers-t.json")).read_text())
    assert result["summary"]["overall"]["questions"] == 6
    assert len(result["injection"]) == 4
    assert [g["threshold"] for g in result["gate"]][0] == 0.0
    report = next(tmp_path.glob("*answers-t.md")).read_text()
    assert "offline stand-ins" in report and "Prompt-injection resistance" in report
