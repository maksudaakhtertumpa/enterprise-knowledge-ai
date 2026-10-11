"""Answer benchmark: grounded Q&A quality, refusals, injection resistance, latency.

    python -m eval.run_answers                     # real models from .env (Ollama on your Mac)
    python -m eval.run_answers --limit 5           # quick look
    python -m eval.run_answers --dataset eval/datasets/teratai-heldout --no-injection
                                                   # held-out questions (report once, don't tune)
    python -m eval.run_answers --backend hashing --reranker-backend hashing \
        --llm-backend extractive                   # offline plumbing check (NOT a quality result)

Ingests the `teratai` corpus through the real pipeline, asks every natural-language
question (paraphrase, Malay and unanswerable; the keyword-style "lexical" queries are
retrieval-only), and scores:

- answer rate on answerable questions, and correct refusals on unanswerable ones;
- fact recall: share of expected facts (answers.jsonl) present in the answer;
- citation precision: share of verified citations pointing at a labelled source section;
- unsupported-sentence rate: share of answer sentences the verifier removed;
- latency and token counts (cost is $0: everything runs locally).

The run uses the score gate *off*, so it measures the model alone; the gate table then
shows, from the same data, what each threshold would add. Choose
EKA_ANSWER_MIN_RERANK_SCORE from that table.

The entailment check runs in *flag* mode: every answer sentence is scored by the NLI
model but nothing is removed, so one run shows what every entailment threshold would
do (answer rate, fact recall, refusals). Choose EKA_ANSWER_ENTAILMENT_THRESHOLD from
that table on the development set, then confirm it once on the held-out set.

Then a second workspace adds the four planted documents in datasets/injection and
checks whether their payloads leak into answers.
"""

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.core.security import create_workspace_with_key
from app.db.models import AnswerStatus
from app.db.session import init_db
from app.services.container import build_services
from app.services.generation.answer import AnswerOutcome, AnswerService
from app.services.generation.entailment import strip_markers
from app.services.generation.verify import normalize_words
from app.services.retrieval.embeddings import check_compatibility
from app.services.retrieval.search import SearchService
from eval import metrics
from eval.run_retrieval import DEFAULT_OUT, ROOT, corpus_dir, environment, ingest_corpus

DATASET = ROOT / "eval" / "datasets" / "teratai"
INJECTION = ROOT / "eval" / "datasets" / "injection"
QUESTION_CATEGORIES = ("paraphrase", "malay", "unanswerable")
GATE_THRESHOLDS = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5)
ENTAILMENT_THRESHOLDS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


# ------------------------------------------------------------------- scoring
def contains_phrase(text: str, phrase: str) -> bool:
    words = " " + " ".join(normalize_words(text)) + " "
    return " " + " ".join(normalize_words(phrase)) + " " in words


def fact_recall(answer: str | None, facts: list[list[str]]) -> float:
    if not facts:
        return 0.0
    if not answer:
        return 0.0
    return sum(any(contains_phrase(answer, alt) for alt in fact) for fact in facts) / len(facts)


def score_row(q: dict[str, Any], facts: list[list[str]], outcome: AnswerOutcome) -> dict[str, Any]:
    relevant = {(r["doc"], r["section"]) for r in q["relevant"]}
    v = outcome.verification
    by_number = {e.number: e for e in outcome.passages}
    citations = []
    if v and outcome.answer:
        for c in v.citations:
            hit = by_number[c.passage].hit
            citations.append((hit.document.filename, hit.chunk.heading or ""))
    top = outcome.passages[0].hit.rerank_score if outcome.passages else None
    answered = outcome.status is AnswerStatus.ANSWERED
    checks = outcome.entailment
    return {
        "id": q["id"],
        "category": q["category"],
        "question": q["query"],
        "status": outcome.status.value,
        "reason": outcome.reason,
        "answer": outcome.answer,
        "top_rerank_score": top,
        "fact_recall": fact_recall(outcome.answer, facts) if relevant else None,
        "citations": len(citations),
        "citations_on_relevant": sum(1 for c in citations if c in relevant),
        "citations_proposed": outcome.citations_proposed,
        "sentences_total": v.sentences_total if v else 0,
        "sentences_removed": v.sentences_removed if v else 0,
        # Reason and quote, so rejected citations can be diagnosed from the JSON.
        "rejected": (
            [{"reason": r.reason, "passage": r.passage, "quote": r.quote[:300]} for r in v.rejected]
            if v
            else []
        ),
        "answered": answered,
        "latency_ms": outcome.timings_ms.get("total_ms", 0.0),
        "generate_ms": outcome.timings_ms.get("generate_ms"),
        "prompt_tokens": outcome.prompt_tokens,
        "completion_tokens": outcome.completion_tokens,
        "evidence_sections": [
            f"{e.hit.document.filename} > {e.hit.chunk.heading or ''}" for e in outcome.passages
        ],
        "facts": facts,
        # Per-sentence P(entailment) from the NLI model (flag mode: nothing removed yet).
        "entailment": (
            [{"text": strip_markers(c.text), "score": c.score} for c in checks.sentences]
            if checks and answered
            else None
        ),
        "entailment_ms": outcome.timings_ms.get("entailment_ms"),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [r for r in rows if r["category"] != "unanswerable"]
    unanswerable = [r for r in rows if r["category"] == "unanswerable"]
    answered = [r for r in answerable if r["answered"]]
    cites = sum(r["citations"] for r in answered)
    sentences = sum(r["sentences_total"] for r in rows)
    out: dict[str, Any] = {
        "questions": len(rows),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "answer_rate": _ratio(len(answered), len(answerable)),
        "fact_recall": metrics.mean(r["fact_recall"] for r in answerable) if answerable else None,
        "fact_recall_when_answered": (
            metrics.mean(r["fact_recall"] for r in answered) if answered else None
        ),
        "citation_precision": _ratio(sum(r["citations_on_relevant"] for r in answered), cites),
        "unsupported_sentence_rate": _ratio(sum(r["sentences_removed"] for r in rows), sentences),
        "citations_verified_rate": _ratio(
            sum(r["citations"] for r in rows), sum(r["citations_proposed"] for r in rows)
        ),
        "correct_refusal_rate": _ratio(
            sum(1 for r in unanswerable if not r["answered"]), len(unanswerable)
        ),
        "latency_p50_ms": metrics.percentile([r["latency_ms"] for r in rows], 50) if rows else None,
        "latency_p95_ms": metrics.percentile([r["latency_ms"] for r in rows], 95) if rows else None,
    }
    tokens = [r for r in rows if r["prompt_tokens"]]
    out["prompt_tokens_mean"] = metrics.mean(r["prompt_tokens"] for r in tokens) if tokens else None
    out["completion_tokens_mean"] = (
        metrics.mean(r["completion_tokens"] or 0 for r in tokens) if tokens else None
    )
    out["cost_usd"] = 0.0
    return out


def gate_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """What each score threshold would do, combined with the model's own decisions."""
    answerable = [r for r in rows if r["category"] != "unanswerable"]
    unanswerable = [r for r in rows if r["category"] == "unanswerable"]
    table = []
    for t in GATE_THRESHOLDS:

        def passes(r, t=t):
            return r["top_rerank_score"] is None or r["top_rerank_score"] >= t

        table.append(
            {
                "threshold": t,
                "answerable_reaching_model": _ratio(
                    sum(passes(r) for r in answerable), len(answerable)
                ),
                "answer_rate": _ratio(
                    sum(passes(r) and r["answered"] for r in answerable), len(answerable)
                ),
                "correct_refusal_rate": _ratio(
                    sum(not passes(r) or not r["answered"] for r in unanswerable),
                    len(unanswerable),
                ),
                "llm_calls_saved": _ratio(sum(not passes(r) for r in rows), len(rows)),
            }
        )
    return table


def apply_entailment(row: dict[str, Any], threshold: float) -> tuple[bool, str | None]:
    """(answered, answer text) if sentences scoring below `threshold` were removed."""
    if not row["answered"]:
        return False, None
    checks = row.get("entailment")
    if not checks:  # no entailment data for this answer: nothing to remove
        return True, row["answer"]
    kept = [c["text"] for c in checks if c["score"] >= threshold]
    return (True, " ".join(kept)) if kept else (False, None)


def entailment_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """What removing sentences below each NLI threshold would do, from one flag-mode run."""
    if not any(r.get("entailment") for r in rows):
        return []
    answerable = [r for r in rows if r["category"] != "unanswerable"]
    unanswerable = [r for r in rows if r["category"] == "unanswerable"]
    sentences = [c for r in rows for c in (r.get("entailment") or [])]
    table = []
    for t in ENTAILMENT_THRESHOLDS:
        outcomes = {id(r): apply_entailment(r, t) for r in rows}
        recall = [fact_recall(outcomes[id(r)][1], r.get("facts") or []) for r in answerable]
        table.append(
            {
                "threshold": t,
                "answer_rate": _ratio(sum(outcomes[id(r)][0] for r in answerable), len(answerable)),
                "fact_recall": metrics.mean(recall) if recall else None,
                "correct_refusal_rate": _ratio(
                    sum(not outcomes[id(r)][0] for r in unanswerable), len(unanswerable)
                ),
                "sentences_removed": _ratio(sum(c["score"] < t for c in sentences), len(sentences)),
            }
        )
    return table


def _ratio(num: float, den: float) -> float | None:
    return round(num / den, 4) if den else None


# ------------------------------------------------------------------ running
def run_questions(services, workspace_id, questions, answers) -> list[dict[str, Any]]:
    rows = []
    for i, q in enumerate(questions, start=1):
        with services.session_factory() as db:
            search = SearchService(
                db,
                services.embedder,
                services.vectors,
                services.settings.fusion,
                reranker=services.reranker,
                rerank_candidates=services.settings.rerank_candidates,
            )
            service = AnswerService(
                db,
                services.settings,
                search,
                services.llm,
                services.embedder.count_tokens,
                entailment=services.entailment,
            )
            outcome = service.answer(workspace_id, q["query"])
        row = score_row(q, answers.get(q["id"], []), outcome)
        rows.append(row)
        mark = "answered" if row["answered"] else f"no answer ({row['reason']})"
        print(f"  [{i}/{len(questions)}] {q['id']} {mark} in {row['latency_ms'] / 1000:.1f} s")
    return rows


def run_injection(services, files: list[Path]) -> list[dict[str, Any]]:
    queries = [json.loads(line) for line in (INJECTION / "queries.jsonl").read_text().splitlines()]
    with services.session_factory() as db:
        workspace, _ = create_workspace_with_key(db, "injection")
        db.commit()
        ws = workspace.id
    attacks = sorted((INJECTION / "attacks").glob("*.txt"))
    ingest_corpus(services, ws, [*files, *attacks])
    rows = []
    for q in queries:
        with services.session_factory() as db:
            search = SearchService(
                db,
                services.embedder,
                services.vectors,
                services.settings.fusion,
                reranker=services.reranker,
                rerank_candidates=services.settings.rerank_candidates,
            )
            outcome = AnswerService(
                db,
                services.settings,
                search,
                services.llm,
                services.embedder.count_tokens,
                entailment=services.entailment,
            ).answer(ws, q["query"])
        attack_hits = [e for e in outcome.passages if e.hit.document.filename == q["attack"]]
        answer = outcome.answer or ""
        leaked = any(contains_phrase(answer, p) for p in q["payload"])
        rows.append(
            {
                "id": q["id"],
                "attack": q["attack"],
                "status": outcome.status.value,
                "reason": outcome.reason,
                "answer": outcome.answer,
                "attack_retrieved": bool(attack_hits),
                "attack_flagged": any(e.suspicious for e in attack_hits),
                "resisted": not leaked,
                "correct": (not leaked) and fact_recall(answer, q["expected"]) == 1.0,
            }
        )
        print(f"  {q['id']} {'resisted' if not leaked else 'LEAKED'} ({outcome.status.value})")
    return rows


def load_questions(
    dataset: Path, categories: list[str], limit: int | None = None
) -> tuple[list[dict[str, Any]], dict[str, list[list[str]]], list[Path]]:
    queries = [json.loads(line) for line in (dataset / "queries.jsonl").read_text().splitlines()]
    questions = [q for q in queries if q["category"] in categories][:limit]
    answers = {
        a["id"]: a["facts"]
        for a in map(json.loads, (dataset / "answers.jsonl").read_text().splitlines())
    }
    return questions, answers, sorted(corpus_dir(dataset).glob("*.txt"))


def dataset_hash(dataset: Path = DATASET) -> str:
    digest = hashlib.sha256()
    paths = [
        *sorted(corpus_dir(dataset).glob("*.txt")),
        dataset / "queries.jsonl",
        dataset / "answers.jsonl",
        *sorted((INJECTION / "attacks").glob("*.txt")),
        INJECTION / "queries.jsonl",
    ]
    for p in paths:
        digest.update(p.name.encode())
        digest.update(p.read_bytes())
    return digest.hexdigest()[:12]


# ------------------------------------------------------------------ report
def _fmt(v: Any, pct: bool = True) -> str:
    if v is None:
        return "—"
    return f"{v:.3f}" if pct else f"{v:.0f}"


def render(result: dict[str, Any]) -> str:
    env, s = result["environment"], result["summary"]
    llm = env["llm"]
    ent = result.get("entailment") or {}
    lines = [
        f"# Answer benchmark — {result['label']}",
        "",
        f"- Run: {result['timestamp']} · dataset `{result.get('dataset', 'teratai')}` "
        f"(hash `{result['dataset_hash']}`) · "
        f"{s['overall']['questions']} questions ({s['overall']['answerable']} answerable, "
        f"{s['overall']['unanswerable']} unanswerable)",
        f"- LLM: `{llm.get('model')}` ({llm.get('backend')}), context {llm.get('num_ctx', '—')}, "
        f"temperature {llm.get('temperature', '—')}",
        f"- Retrieval: `{env['embedding'].get('model')}` + "
        f"`{(env.get('reranker') or {}).get('model')}`, {result['passages']} passages per question",
        f"- Entailment check: `{ent.get('model') or 'off'}` ({ent.get('backend', 'none')}, "
        f"premise: {ent.get('granularity', 'passage')}"
        + (
            f" windows of up to {ent['window_sentences']}"
            if ent.get("granularity") == "sentences"
            else ""
        )
        + "), flag mode (scored, nothing removed)",
        f"- Machine: {env['platform']} · Python {env['python']} · cost: $0 (local models)",
        "",
    ]
    if (
        llm.get("backend") != "ollama"
        or env["embedding"].get("backend") == "hashing"
        or ent.get("backend") == "lexical"
    ):
        lines += [
            "> **Warning:** this run uses offline stand-ins (no real LLM, non-semantic",
            "> retrieval and/or the lexical entailment stand-in). It checks the pipeline,",
            "> **not** answer quality.",
            "",
        ]
    lines += [
        "## Overall (score gate off)",
        "",
        "| Group | Questions | Answer rate | Fact recall | Fact recall (answered) "
        "| Citation precision | Unsupported sentences | Correct refusals | p50 s | p95 s |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, g in [("all", s["overall"]), *s["by_category"].items()]:
        lines.append(
            f"| {name} | {g['questions']} | {_fmt(g['answer_rate'])} | {_fmt(g['fact_recall'])} | "
            f"{_fmt(g['fact_recall_when_answered'])} | {_fmt(g['citation_precision'])} | "
            f"{_fmt(g['unsupported_sentence_rate'])} | {_fmt(g['correct_refusal_rate'])} | "
            f"{_fmt((g['latency_p50_ms'] or 0) / 1000)} | "
            f"{_fmt((g['latency_p95_ms'] or 0) / 1000)} |"
        )
    o = s["overall"]
    lines += [
        "",
        f"Tokens per question: {_fmt(o['prompt_tokens_mean'], False)} prompt, "
        f"{_fmt(o['completion_tokens_mean'], False)} completion. "
        f"Citations verified: {_fmt(o['citations_verified_rate'])} of those proposed. "
        "API cost: $0.",
        "",
        "## Score gate trade-off (from the same run)",
        "",
        "Skipping the LLM when the best reranker score is below a threshold:",
        "",
        "| Threshold | Answerable reaching LLM | Answer rate | Correct refusals "
        "| LLM calls saved |",
        "|---|---|---|---|---|",
    ]
    for g in result["gate"]:
        lines.append(
            f"| {g['threshold']} | {_fmt(g['answerable_reaching_model'])} | "
            f"{_fmt(g['answer_rate'])} | {_fmt(g['correct_refusal_rate'])} | "
            f"{_fmt(g['llm_calls_saved'])} |"
        )
    if result.get("entailment_table"):
        lines += [
            "",
            "## Entailment threshold trade-off (from the same run)",
            "",
            "Removing answer sentences whose P(entailment | cited passages) is below a "
            "threshold; an answer with no sentence left becomes a refusal:",
            "",
            "| Threshold | Answer rate | Fact recall | Correct refusals | Sentences removed |",
            "|---|---|---|---|---|",
        ]
        for g in result["entailment_table"]:
            lines.append(
                f"| {g['threshold']} | {_fmt(g['answer_rate'])} | {_fmt(g['fact_recall'])} | "
                f"{_fmt(g['correct_refusal_rate'])} | {_fmt(g['sentences_removed'])} |"
            )
        low = sorted(
            (
                (c["score"], r["id"], c["text"])
                for r in result["rows"]
                for c in (r.get("entailment") or [])
                if c["score"] < 0.5
            ),
        )
        if low:
            lines += [
                "",
                "Sentences scoring below 0.5 (check by hand whether the model is right):",
                "",
                "| Id | Score | Sentence |",
                "|---|---|---|",
            ]
            for score, qid, text in low[:30]:
                lines.append(f"| {qid} | {score:.3f} | {text.replace('|', '/')[:200]} |")
    if result.get("injection"):
        lines += [
            "",
            "## Prompt-injection resistance",
            "",
            "| Id | Attack | Retrieved | Flagged | Status | Resisted | Correct |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in result["injection"]:
            lines.append(
                f"| {r['id']} | {r['attack']} | {r['attack_retrieved']} | {r['attack_flagged']} | "
                f"{r['status']} | {r['resisted']} | {r['correct']} |"
            )
    misses = [
        r
        for r in result["rows"]
        if (r["category"] != "unanswerable" and (r["fact_recall"] or 0) < 1)
        or (r["category"] == "unanswerable" and r["answered"])
    ]
    if misses:
        lines += [
            "",
            "## Questions to review",
            "",
            "| Id | Question | Status | Facts | Answer |",
            "|---|---|---|---|---|",
        ]
        for r in misses[:25]:
            answer = (r["answer"] or r["reason"] or "").replace("|", "/").replace("\n", " ")[:160]
            facts = "—" if r["fact_recall"] is None else f"{r['fact_recall']:.2f}"
            lines.append(f"| {r['id']} | {r['question']} | {r['status']} | {facts} | {answer} |")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend", choices=["sentence-transformers", "hashing"])
    parser.add_argument("--reranker-backend", choices=["cross-encoder", "hashing", "none"])
    parser.add_argument("--llm-backend", choices=["ollama", "extractive"])
    parser.add_argument("--model", help="Ollama model (default: EKA_OLLAMA_MODEL)")
    parser.add_argument("--entailment-backend", choices=["nli", "lexical", "none"])
    parser.add_argument("--entailment-model", help="Default: EKA_ENTAILMENT_MODEL")
    parser.add_argument(
        "--entailment-granularity",
        choices=["sentences", "passage"],
        help="Default: EKA_ENTAILMENT_GRANULARITY",
    )
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(QUESTION_CATEGORIES),
        choices=list(QUESTION_CATEGORIES),
    )
    parser.add_argument("--limit", type=int, help="Only the first N questions (quick look)")
    parser.add_argument("--no-injection", action="store_true")
    parser.add_argument("--label")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    questions, answers, files = load_questions(args.dataset, args.categories, args.limit)

    data_dir = Path(tempfile.mkdtemp(prefix="eka-answers-"))
    overrides: dict[str, Any] = {
        "data_dir": data_dir,
        "log_level": "WARNING",
        "answer_min_rerank_score": 0.0,  # measure the model alone; gate analysed afterwards
        "ask_history": False,
        "answer_entailment_mode": "flag",  # score every sentence; thresholds analysed afterwards
    }
    # An offline embedder run stays offline: offline reranker and entailment stand-ins
    # unless told otherwise.
    offline = args.backend == "hashing"
    entailment_backend = args.entailment_backend or ("lexical" if offline else None)
    if entailment_backend:
        overrides["entailment_backend"] = entailment_backend
    if entailment_backend == "none":
        overrides["answer_entailment_mode"] = "off"
    for flag, key in [
        (args.backend, "embedding_backend"),
        (args.reranker_backend or ("hashing" if offline else None), "reranker_backend"),
        (args.llm_backend, "llm_backend"),
        (args.model, "ollama_model"),
        (args.entailment_model, "entailment_model"),
        (args.entailment_granularity, "entailment_granularity"),
    ]:
        if flag:
            overrides[key] = flag
    settings = Settings(**overrides)
    if settings.llm_backend == "none":
        parser.error("answering is disabled (EKA_LLM_BACKEND=none)")

    try:
        services = build_services(settings)
        init_db(services.engine)
        print(f"LLM: {services.llm.describe()}")
        ok, detail = services.llm.health()
        if not ok:
            print(f"LLM not ready: {detail}", file=sys.stderr)
            return 2
        services.embedder.load()
        check_compatibility(services.embedder, settings)
        if services.reranker:
            services.reranker.load()
        if services.entailment:
            services.entailment.load()
            print(f"Entailment: {services.entailment.describe()}")
        with services.session_factory() as db:
            workspace, _ = create_workspace_with_key(db, "answers")
            db.commit()
            ws = workspace.id
        print(f"Ingesting {len(files)} documents ...")
        ingest_corpus(services, ws, files)
        print(f"Asking {len(questions)} questions ...")
        rows = run_questions(services, ws, questions, answers)
        injection = []
        if not args.no_injection:
            print("Injection suite ...")
            injection = run_injection(services, files)
        env = environment(services)
        env["llm"] = services.llm.describe()
        entailment = (
            {
                **services.entailment.describe(),
                "granularity": settings.entailment_granularity,
                "window_sentences": settings.entailment_window_sentences,
            }
            if services.entailment
            else None
        )
    finally:
        if "services" in locals():
            services.close()
        shutil.rmtree(data_dir, ignore_errors=True)

    by_category = {
        c: summarize([r for r in rows if r["category"] == c])
        for c in args.categories
        if any(r["category"] == c for r in rows)
    }
    label = args.label or f"{env['llm'].get('model', 'llm')}".replace(":", "-").replace("/", "-")
    if args.dataset.resolve() != DATASET.resolve() and not args.label:
        label = f"{args.dataset.name}-{label}"
    stamp = datetime.now(UTC)
    result = {
        "label": label,
        "timestamp": stamp.isoformat(timespec="seconds"),
        "dataset": args.dataset.name,
        "dataset_hash": dataset_hash(args.dataset),
        "passages": settings.answer_passages,
        "environment": env,
        "entailment": entailment,
        "summary": {"overall": summarize(rows), "by_category": by_category},
        "gate": gate_table(rows),
        "entailment_table": entailment_table(rows),
        "injection": injection,
        "rows": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{stamp:%Y%m%d-%H%M%S}-answers-{label}"
    (args.out / f"{stem}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    report = render(result)
    (args.out / f"{stem}.md").write_text(report)
    print()
    print(report)
    print(f"Saved {stem}.json and {stem}.md in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
