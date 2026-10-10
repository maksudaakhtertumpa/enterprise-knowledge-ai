"""End-to-end smoke test against a real running server.

Starts uvicorn on a free port with a throwaway data directory, creates a
workspace, uploads a real PDF, DOCX and TXT, waits for ingestion, checks the
chunks and isolation, runs searches (BM25, dense, hybrid and reranked), asks
grounded questions, then shuts down.

    python scripts/smoke_test.py                 # uses EKA_EMBEDDING_BACKEND from env/.env
    python scripts/smoke_test.py --backend hashing   # no model download (fast, not semantic)

Uses only the standard library plus the project itself, so it runs anywhere
the project is installed. Exit code 0 = all checks passed.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.factories import make_docx, make_pdf  # noqa: E402

LEAVE = (
    "Employees are entitled to fourteen days of annual leave per calendar year. "
    "Leave must be approved by the line manager at least two weeks in advance."
)
CLAIMS = "Travel claims are reimbursed within thirty days when original receipts are attached."

PASSED: list[str] = []


def check(condition: bool, label: str, detail: object = "") -> None:
    if not condition:
        print(f"  FAIL  {label} {detail}")
        raise SystemExit(1)
    PASSED.append(label)
    print(f"  ok    {label}")


def request(base, method, path, key=None, body=None, files=None, timeout=120):
    headers = {}
    data = None
    if key:
        headers["X-API-Key"] = key
    if files:
        boundary = uuid.uuid4().hex
        name, content = files
        data = (
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="file"; filename="{name}"\r\n'
                f"Content-Type: application/octet-stream\r\n\r\n"
            ).encode()
            + content
            + f"\r\n--{boundary}--\r\n".encode()
        )
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    elif body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read() or b"null")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["sentence-transformers", "hashing"])
    parser.add_argument("--keep", action="store_true", help="keep the temp data directory")
    args = parser.parse_args()

    data_dir = Path(tempfile.mkdtemp(prefix="eka-smoke-"))
    env = {**os.environ, "EKA_DATA_DIR": str(data_dir), "EKA_LOG_LEVEL": "WARNING"}
    if args.backend:
        env["EKA_EMBEDDING_BACKEND"] = args.backend
        # Offline embedder -> offline reranker; real embedder -> real cross-encoder.
        env["EKA_RERANKER_BACKEND"] = "hashing" if args.backend == "hashing" else "cross-encoder"
        # Offline run -> extractive stand-in; real run -> Ollama.
        env["EKA_LLM_BACKEND"] = "extractive" if args.backend == "hashing" else "ollama"
    port = free_port()
    base = f"http://127.0.0.1:{port}"

    def cli(*argv: str) -> str:
        out = subprocess.run(  # noqa: S603 - fixed argv
            [sys.executable, "-m", "app.cli", *argv],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout

    print(f"Data dir: {data_dir}")
    key = next(
        line.split()[-1]
        for line in cli("create-workspace", "Smoke A").splitlines()
        if line.startswith("API key")
    )
    other = next(
        line.split()[-1]
        for line in cli("create-workspace", "Smoke B").splitlines()
        if line.startswith("API key")
    )

    # Server output goes to a file, never a pipe: an unread pipe fills up (~64 KB of
    # download progress bars is enough) and then blocks the server forever.
    log_path = data_dir / "server.log"
    log_file = log_path.open("w")
    server = subprocess.Popen(  # noqa: S603
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port)],
        cwd=ROOT,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )

    def server_log_tail(lines: int = 30) -> str:
        log_file.flush()
        text = log_path.read_text(errors="replace").replace("\r", "\n")
        return "\n".join(text.splitlines()[-lines:])

    try:
        print("\n[1] Startup")
        # Models should already be cached by `python -m app.cli prepare-models`;
        # allow time for a first-run download anyway.
        status, body = None, None
        started = time.time()
        deadline = started + 1200
        while time.time() < deadline:
            if server.poll() is not None:
                print(server_log_tail())
                check(False, "server process stays up")
            try:
                status, body = request(base, "GET", "/health/ready", timeout=5)
                if status == 200:
                    break
            except OSError:
                pass
            if int(time.time() - started) % 60 == 0 and time.time() - started > 1:
                print(f"          still starting after {int(time.time() - started)} s ...")
            time.sleep(1)
        if status != 200:
            print("          last server output:\n" + server_log_tail())
        check(status == 200, "readiness probe returns 200", body)
        for c in body["checks"]:
            check(c["ok"], f"component '{c['name']}' ready", c)

        print("\n[2] Uploads (PDF, DOCX, TXT)")
        files = {
            "handbook.pdf": make_pdf(
                [["Leave Policy", LEAVE], ["Claims", CLAIMS]],
                outline={1: "Leave", 2: "Claims"},
                title="Smoke Handbook",
            ),
            "policy.docx": make_docx(
                [("h1", "Remote Work"), ("p", "Staff may work from home two days per week.")],
                title="Remote Work Policy",
            ),
            "notes.txt": b"# IT\n\nPasswords must be rotated every ninety days.\n",
        }
        ids = {}
        for name, content in files.items():
            t0 = time.perf_counter()
            status, body = request(base, "POST", "/api/v1/documents", key, files=(name, content))
            check(status == 202, f"upload {name} accepted", body)
            ids[name] = body["id"]
            for _ in range(600):
                _, doc = request(base, "GET", f"/api/v1/documents/{ids[name]}", key)
                if doc["status"] in ("ready", "failed"):
                    break
                time.sleep(0.5)
            check(doc["status"] == "ready", f"{name} ingested", doc.get("error"))
            print(
                f"          {doc['chunk_count']} chunks, pages={doc['page_count']}, "
                f"{(time.perf_counter() - t0) * 1000:.0f} ms"
            )

        _, chunks = request(base, "GET", f"/api/v1/documents/{ids['handbook.pdf']}/chunks", key)
        pages = {c["page_start"] for c in chunks["items"]}
        check(pages == {1, 2}, "PDF chunks carry page numbers 1 and 2", pages)

        status, body = request(
            base, "POST", "/api/v1/documents", key, files=("again.txt", files["notes.txt"])
        )
        check(status == 409, "duplicate upload rejected with 409")

        print("\n[3] Isolation")
        status, _ = request(base, "GET", f"/api/v1/documents/{ids['notes.txt']}", other)
        check(status == 404, "other workspace gets 404 for our document")

        print("\n[4] Search")
        for mode, rerank, query, expect in [
            ("bm25", False, "ninety days passwords", "notes.txt"),
            ("dense", False, "how often must I change my password", "notes.txt"),
            ("bm25", False, "annual leave approval", "handbook.pdf"),
            ("hybrid", False, "how often must I rotate my password", "notes.txt"),
            ("dense", True, "when do I have to update my login credentials", "notes.txt"),
        ]:
            label = f"{mode}+rerank" if rerank else mode
            t0 = time.perf_counter()
            status, body = request(
                base,
                "POST",
                "/api/v1/search",
                key,
                body={"query": query, "mode": mode, "top_k": 3, "rerank": rerank},
            )
            ms = (time.perf_counter() - t0) * 1000
            check(status == 200 and body["results"], f"{label} search returns results", body)
            top = body["results"][0]
            print(
                f"          {label:12} {query!r} -> {top['source']['filename']} "
                f"(p.{top['source']['page_start']}, score {top['score']:.3f}, {ms:.0f} ms)"
            )
            if rerank:
                check(body["reranked"] and top["retrieval_rank"], "reranked result reports ranks")
            # Anything beyond BM25 depends on meaning; only real models are expected to get it.
            if mode == "bm25" or args.backend != "hashing":
                check(top["source"]["filename"] == expect, f"{label} top hit is {expect}")
        _, body = request(
            base,
            "POST",
            "/api/v1/search",
            other,
            body={"query": "passwords ninety days", "mode": "bm25"},
        )
        check(body["results"] == [], "other workspace's search sees nothing")

        print("\n[5] Ask (grounded answers)")
        t0 = time.perf_counter()
        status, body = request(
            base,
            "POST",
            "/api/v1/ask",
            key,
            body={"question": "How often must passwords be changed?"},
        )
        ms = (time.perf_counter() - t0) * 1000
        check(status == 200, "ask returns 200", body)
        print(f"          answer ({ms / 1000:.1f} s, {body.get('model')}): {body.get('answer')}")
        check(body["status"] == "answered", "answerable question is answered", body.get("reason"))
        files_cited = {c["source"]["filename"] for c in body["citations"]}
        check("notes.txt" in files_cited, "answer cites the password policy", files_cited)
        check(
            all(f"[{c['marker']}]" in body["answer"] for c in body["citations"]),
            "every citation appears in the answer",
        )
        status, body = request(
            base, "POST", "/api/v1/ask", key, body={"question": "Can I bring my dog to the office?"}
        )
        print(f"          unanswerable -> {body['status']} ({body.get('reason')})")
        check(body["status"] == "insufficient_evidence", "unanswerable question is refused")
        _, body = request(
            base,
            "POST",
            "/api/v1/ask",
            other,
            body={"question": "How often must passwords be changed?"},
        )
        check(body["evidence"] == [], "other workspace's question sees no evidence")
        _, body = request(base, "GET", "/api/v1/ask/history", key)
        check(body["total"] == 2, "history records both questions")

        print("\n[6] Analytics")
        status, body = request(base, "GET", "/api/v1/analytics?days=7", key)
        check(status == 200, "analytics returns 200", body)
        q = body["questions"]
        check((q["total"], q["answered"]) == (2, 1), "analytics counts both questions", q)
        check(body["tokens"]["cost_usd"] == 0.0, "cost is $0")

        print("\n[7] Delete")
        status, _ = request(base, "DELETE", f"/api/v1/documents/{ids['notes.txt']}", key)
        check(status == 204, "delete returns 204")
        _, body = request(
            base,
            "POST",
            "/api/v1/search",
            key,
            body={"query": "passwords ninety days", "mode": "bm25"},
        )
        check(
            all(r["source"]["filename"] != "notes.txt" for r in body["results"]),
            "deleted document no longer searchable",
        )
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()  # never leave a server running in the background
            server.wait(timeout=10)
        log_file.close()
        if not args.keep:
            import shutil

            shutil.rmtree(data_dir, ignore_errors=True)

    print(f"\nAll {len(PASSED)} smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
