import json
import re
from pathlib import Path

import pytest

from eval.run_answers import (
    DATASET,
    INJECTION,
    apply_entailment,
    contains_phrase,
    entailment_table,
    fact_recall,
    gate_table,
    load_questions,
    main,
    summarize,
)
from eval.run_retrieval import ROOT, corpus_dir, load_dataset

HELDOUT = ROOT / "eval" / "datasets" / "teratai-heldout"


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


def _ent_row(cat, answered, scores, facts=None, texts=None):
    texts = texts or [f"Sentence {i} says 14 days." for i in range(len(scores))]
    return {
        "category": cat,
        "answered": answered,
        "answer": " ".join(texts) if answered else None,
        "facts": facts if facts is not None else ([["14"]] if cat != "unanswerable" else []),
        "entailment": (
            [{"text": t, "score": sc} for t, sc in zip(texts, scores, strict=True)]
            if answered
            else None
        ),
    }


def test_apply_entailment_removes_weak_sentences_and_refuses_when_none_left():
    row = _ent_row("paraphrase", True, [0.9, 0.2], texts=["Staff get 14 days.", "And a car."])
    assert apply_entailment(row, 0.5) == (True, "Staff get 14 days.")
    assert apply_entailment(row, 0.95) == (False, None)
    assert apply_entailment(row, 0.0) == (True, "Staff get 14 days. And a car.")
    refused = _ent_row("paraphrase", False, [])
    assert apply_entailment(refused, 0.0) == (False, None)
    no_data = {**row, "entailment": None}  # entailment off: the answer stands
    assert apply_entailment(no_data, 0.9) == (True, row["answer"])


def test_entailment_table_trades_answers_for_refusals():
    rows = [
        _ent_row("paraphrase", True, [0.95]),
        _ent_row("paraphrase", True, [0.35]),  # true answer the NLI model doubts
        _ent_row("unanswerable", True, [0.15]),  # wrong answer: weak support
        _ent_row("unanswerable", False, []),
    ]
    table = {t["threshold"]: t for t in entailment_table(rows)}
    assert (table[0.0]["answer_rate"], table[0.0]["correct_refusal_rate"]) == (1.0, 0.5)
    assert table[0.0]["fact_recall"] == 1.0 and table[0.0]["sentences_removed"] == 0.0
    # 0.2 removes only the wrong answer; 0.4 also loses a true one.
    assert (table[0.2]["answer_rate"], table[0.2]["correct_refusal_rate"]) == (1.0, 1.0)
    assert (table[0.4]["answer_rate"], table[0.4]["fact_recall"]) == (0.5, 0.5)
    assert table[0.4]["sentences_removed"] == pytest.approx(2 / 3, abs=1e-4)
    assert entailment_table([{**r, "entailment": None} for r in rows]) == []


@pytest.mark.parametrize("dataset", [DATASET, HELDOUT], ids=["dev", "heldout"])
def test_answer_facts_are_in_the_labelled_sections(dataset):
    """Every expected fact has at least one wording that the labelled sections contain.

    Malay alternatives (and wordings the model may use) are allowed as long as one
    alternative is in the source, so a fact can't be unanswerable by construction.
    """
    questions, answers, files = load_questions(dataset, ["paraphrase", "malay", "unanswerable"])
    text = {f.name: f.read_text() for f in files}

    def section(doc, heading):
        match = re.search(rf"^# {re.escape(heading)}\n(.*?)(?=^# |\Z)", text[doc], re.M | re.S)
        assert match, (doc, heading)
        return match.group(1)

    for q in questions:
        if q["category"] == "unanswerable":
            assert q["id"] not in answers
            continue
        source = " ".join(section(r["doc"], r["section"]) for r in q["relevant"])
        for fact in answers[q["id"]]:
            assert any(contains_phrase(source, alt) for alt in fact), (q["id"], fact)


def test_heldout_set_is_new_and_well_formed():
    dev_files, dev_queries, dev_hash = load_dataset(DATASET)
    files, queries, digest = load_dataset(HELDOUT)
    assert corpus_dir(HELDOUT) == corpus_dir(DATASET).resolve()  # same documents ...
    assert files == [f.resolve() for f in dev_files] and digest != dev_hash  # ... new questions
    dev = {re.sub(r"\W+", " ", q["query"].casefold()).strip() for q in dev_queries}
    ids = [q["id"] for q in queries]
    assert len(ids) == len(set(ids)) and all(i.startswith("H") for i in ids)
    for q in queries:
        assert re.sub(r"\W+", " ", q["query"].casefold()).strip() not in dev, q["id"]
        assert bool(q["relevant"]) == (q["category"] != "unanswerable"), q["id"]
    counts = {c: sum(q["category"] == c for q in queries) for c in {q["category"] for q in queries}}
    assert counts == {"lexical": 8, "paraphrase": 24, "malay": 6, "unanswerable": 10}


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
    # Offline run: lexical entailment stand-in in flag mode, scored per sentence.
    assert result["entailment"]["backend"] == "lexical"
    assert result["entailment_table"][0]["threshold"] == 0.0
    answered = [r for r in result["rows"] if r["answered"]]
    assert answered and all(r["entailment"] for r in answered)
    report = next(tmp_path.glob("*answers-t.md")).read_text()
    assert "offline stand-ins" in report and "Prompt-injection resistance" in report
    assert "Entailment threshold trade-off" in report


def test_runner_on_the_heldout_set_without_entailment(tmp_path: Path):
    args = ["--backend", "hashing", "--llm-backend", "extractive", "--entailment-backend"]
    args += ["none", "--dataset", str(HELDOUT), "--no-injection", "--limit", "3"]
    assert main([*args, "--out", str(tmp_path)]) == 0
    (path,) = tmp_path.glob("*answers-teratai-heldout-*.json")
    result = json.loads(path.read_text())
    assert result["dataset"] == "teratai-heldout" and result["entailment"] is None
    assert result["entailment_table"] == [] and result["injection"] == []
    assert [r["id"] for r in result["rows"]] == ["HP01", "HP02", "HP03"]
