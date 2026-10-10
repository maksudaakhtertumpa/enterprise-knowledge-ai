"""GET /analytics: workspace usage numbers computed from the question log."""

from datetime import UTC, datetime, timedelta

import pytest

from app.api.routes.analytics import percentile
from app.db.models import AnswerStatus, QuestionLog
from tests.api.test_ask_api import ask, upload

API = "/api/v1"


def add_log(app, workspace_id, *, days_ago=0, status=AnswerStatus.ANSWERED, **fields):
    defaults = {
        "latency_ms": 1000.0,
        "model": "qwen3.5:4b",
        "prompt_tokens": 900,
        "completion_tokens": 80,
        "citations_proposed": 2,
        "citations_verified": 2 if status is AnswerStatus.ANSWERED else 0,
        "reason": None if status is AnswerStatus.ANSWERED else "model_declined",
    }
    with app.state.session_factory() as db:
        db.add(
            QuestionLog(
                workspace_id=workspace_id,
                created_at=datetime.now(UTC) - timedelta(days=days_ago),
                status=status,
                **{**defaults, **fields},
            )
        )
        db.commit()


def get(client, auth, **params):
    resp = client.get(f"{API}/analytics", headers=auth, params=params)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_empty_workspace_has_zeroes_and_nulls_not_errors(client, auth):
    body = get(client, auth, days=7)
    assert body["questions"] == {
        "total": 0,
        "answered": 0,
        "insufficient_evidence": 0,
        "answer_rate": None,
        "refusal_reasons": {},
    }
    assert body["latency"] is None
    assert body["citations"]["acceptance_rate"] is None
    assert body["tokens"] == {"prompt": 0, "completion": 0, "cost_usd": 0.0}
    assert len(body["per_day"]) == 7 and all(
        d["answered"] == d["insufficient_evidence"] == 0 for d in body["per_day"]
    )
    assert body["documents"]["total"] == 0
    assert set(body["documents"]["by_status"]) == {"pending", "processing", "ready", "failed"}


def test_counts_rates_latency_and_reasons(client, app, auth, workspace):
    for ms in (1000, 2000, 3000, 4000):
        add_log(app, workspace["id"], latency_ms=ms)
    add_log(app, workspace["id"], status=AnswerStatus.INSUFFICIENT_EVIDENCE, latency_ms=500)
    add_log(
        app,
        workspace["id"],
        status=AnswerStatus.INSUFFICIENT_EVIDENCE,
        reason="no_relevant_passages",
        latency_ms=100,
        model=None,
        prompt_tokens=None,
        completion_tokens=None,
        citations_proposed=0,
    )
    body = get(client, auth)
    q = body["questions"]
    assert (q["total"], q["answered"], q["insufficient_evidence"]) == (6, 4, 2)
    assert q["answer_rate"] == round(4 / 6, 4)
    assert q["refusal_reasons"] == {"model_declined": 1, "no_relevant_passages": 1}
    # latencies 100, 500, 1000, 2000, 3000, 4000
    assert body["latency"]["p50_ms"] == 1500.0
    assert body["latency"]["max_ms"] == 4000.0
    assert body["latency"]["mean_ms"] == round(10600 / 6, 1)
    c = body["citations"]
    assert (c["proposed"], c["verified"]) == (10, 8)
    assert c["acceptance_rate"] == 0.8 and c["avg_per_answer"] == 2.0
    assert body["tokens"]["prompt"] == 5 * 900 and body["tokens"]["cost_usd"] == 0.0
    assert body["models"] == {"qwen3.5:4b": 5}


def test_window_excludes_old_rows_and_buckets_by_day(client, app, auth, workspace):
    add_log(app, workspace["id"], days_ago=0)
    add_log(app, workspace["id"], days_ago=2, status=AnswerStatus.INSUFFICIENT_EVIDENCE)
    add_log(app, workspace["id"], days_ago=40)  # outside a 30-day window
    body = get(client, auth, days=30)
    assert body["questions"]["total"] == 2
    days = body["per_day"]
    assert len(days) == 30 and days[0]["date"] < days[-1]["date"]
    assert days[-1]["answered"] == 1
    assert days[-3]["insufficient_evidence"] == 1
    assert get(client, auth, days=60)["questions"]["total"] == 3


def test_document_stats(client, auth):
    upload(client, auth, "a.txt", b"Annual leave is 14 days.\n")
    upload(client, auth, "b.txt", b"Claims are paid within 30 days.\n")
    docs = get(client, auth)["documents"]
    assert docs["total"] == 2 and docs["by_status"]["ready"] == 2
    assert docs["chunks"] >= 2 and docs["size_bytes"] > 0


def test_real_ask_calls_are_counted(client, auth):
    upload(client, auth, "leave.txt", b"Staff receive 14 working days of annual leave.\n")
    assert ask(client, auth, "How many days of annual leave?").json()["status"] == "answered"
    q = get(client, auth)["questions"]
    assert q["total"] == 1 and q["answered"] == 1


def test_works_without_history_text(app, settings, client, auth):
    settings.ask_history = False
    upload(client, auth, "leave.txt", b"Staff receive 14 working days of annual leave.\n")
    ask(client, auth, "How many days of annual leave?")
    assert get(client, auth)["questions"]["total"] == 1


def test_isolated_per_workspace(client, app, auth, other_auth, workspace):
    add_log(app, workspace["id"])
    assert get(client, auth)["questions"]["total"] == 1
    assert get(client, other_auth)["questions"]["total"] == 0


@pytest.mark.parametrize("days", [0, 366, "x"])
def test_window_is_validated(client, auth, days):
    resp = client.get(f"{API}/analytics", headers=auth, params={"days": days})
    assert resp.status_code == 422


def test_requires_api_key(client):
    assert client.get(f"{API}/analytics").status_code == 401


def test_percentile_matches_linear_interpolation():
    assert percentile([5.0], 0.95) == 5.0
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([10, 1, 4], 0.0) == 1 and percentile([10, 1, 4], 1.0) == 10
    assert percentile(list(range(1, 101)), 0.95) == pytest.approx(95.05)
    with pytest.raises(ValueError):
        percentile([], 0.5)
