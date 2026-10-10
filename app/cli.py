"""Admin CLI for workspaces and API keys.

    python -m app.cli create-workspace "Acme HR"
    python -m app.cli list-workspaces
    python -m app.cli issue-key "Acme HR" --label laptop
    python -m app.cli revoke-key eka_AbCdEfGh
    python -m app.cli reindex [--workspace "Acme HR"]   # after changing the embedding model
    python -m app.cli prepare-models                    # download + test models, with progress

Keys are printed exactly once. Only their SHA-256 hash is stored.
"""

import argparse
import sys

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.core.config import get_settings
from app.core.security import create_workspace_with_key, issue_api_key
from app.db.models import ApiKey, Workspace, utcnow
from app.db.session import build_engine, build_session_factory, init_db


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.cli", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("create-workspace", help="Create a workspace and its first API key")
    p.add_argument("name")
    p.add_argument("--label", default="initial")

    sub.add_parser("list-workspaces", help="List workspaces and their active keys")

    p = sub.add_parser("issue-key", help="Issue an additional API key for a workspace")
    p.add_argument("workspace")
    p.add_argument("--label")

    p = sub.add_parser("revoke-key", help="Revoke a key by its displayed prefix")
    p.add_argument("prefix")

    p = sub.add_parser(
        "reindex",
        help="Re-embed all ready documents with the current embedding model "
        "(stop the API server first: the vector store allows one writer process)",
    )
    p.add_argument("--workspace", help="Only this workspace (by name)")

    sub.add_parser(
        "prepare-models",
        help="Download (first run) and test the embedding model and reranker, and check "
        "that Ollama answers with a verified citation",
    )

    args = parser.parse_args(argv)
    if args.command == "prepare-models":
        return _prepare_models()

    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    engine = build_engine(settings.db_url)
    init_db(engine)
    session_factory = build_session_factory(engine)

    with session_factory() as db:
        if args.command == "create-workspace":
            name = args.name.strip()
            if not 1 <= len(name) <= 100:
                print("Workspace name must be 1-100 characters.", file=sys.stderr)
                return 2
            try:
                workspace, key = create_workspace_with_key(db, name, label=args.label)
                db.commit()
            except IntegrityError:
                print(f"A workspace named {name!r} already exists.", file=sys.stderr)
                return 1
            print(f"Workspace: {workspace.name} ({workspace.id})")
            print(f"API key:   {key}")
            print("Store this key now — it cannot be shown again.")

        elif args.command == "list-workspaces":
            for ws in db.scalars(select(Workspace).order_by(Workspace.created_at)):
                active = [k for k in ws.api_keys if k.revoked_at is None]
                prefixes = ", ".join(f"{k.key_prefix}… ({k.label or '-'})" for k in active)
                print(f"{ws.name}  [{ws.id}]  keys: {prefixes or 'none'}")

        elif args.command == "issue-key":
            ws = db.scalar(select(Workspace).where(Workspace.name == args.workspace))
            if ws is None:
                print(f"No workspace named {args.workspace!r}.", file=sys.stderr)
                return 1
            key = issue_api_key(db, ws.id, label=args.label)
            db.commit()
            print(f"API key: {key}")
            print("Store this key now — it cannot be shown again.")

        elif args.command == "reindex":
            return _reindex(settings, db, args.workspace)

        elif args.command == "revoke-key":
            prefix = args.prefix.rstrip("…").strip()
            matches = db.scalars(
                select(ApiKey).where(ApiKey.key_prefix == prefix, ApiKey.revoked_at.is_(None))
            ).all()
            if len(matches) != 1:
                print(
                    f"Expected one active key with prefix {prefix!r}, found {len(matches)}.",
                    file=sys.stderr,
                )
                return 1
            matches[0].revoked_at = utcnow()
            db.commit()
            print(f"Revoked {prefix}…")

    engine.dispose()
    return 0


def _prepare_models() -> int:
    """Load each configured model once, in the foreground, and run one tiny inference.

    The first run downloads weights into ~/.cache/huggingface with visible progress
    bars. Doing this before starting the server means startup is fast and any
    problem (network, disk, GPU) shows up here with a clear message.
    """
    import time

    from app.services.retrieval.embeddings import build_embedder, check_compatibility
    from app.services.retrieval.reranker import build_reranker

    settings = get_settings()
    ok = True

    embedder = build_embedder(settings)
    print(f"Embedding model: {embedder.model_name} ({settings.embedding_backend})")
    t0 = time.perf_counter()
    try:
        embedder.load()
        check_compatibility(embedder, settings)
        vector = embedder.embed_query("How many days of annual leave do I get?")
        info = embedder.describe()
        print(
            f"  ready in {time.perf_counter() - t0:.1f} s on {info.get('device') or 'cpu'}; "
            f"dimension {len(vector)}"
        )
    except Exception as exc:  # noqa: BLE001 - report every failure plainly
        ok = False
        print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)

    reranker = build_reranker(settings)
    if reranker is None:
        print("Reranker: disabled (EKA_RERANKER_BACKEND=none)")
    else:
        print(f"Reranker: {reranker.model_name} ({settings.reranker_backend})")
        t0 = time.perf_counter()
        try:
            reranker.load()
            scores = reranker.score(
                "How many days of annual leave do I get?",
                ["Employees receive 14 days of annual leave.", "Parking is on level B2."],
            )
            info = reranker.describe()
            print(
                f"  ready in {time.perf_counter() - t0:.1f} s on {info.get('device') or 'cpu'} "
                f"({info.get('dtype') or 'n/a'}); test scores "
                f"relevant={scores[0]:.3f} irrelevant={scores[1]:.3f}"
            )
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)

    ok = _check_llm(settings) and ok
    print("All models ready." if ok else "Some models failed to load (see above).")
    return 0 if ok else 1


def _check_llm(settings) -> bool:
    """Ollama running? Model pulled? Can it produce one verified, grounded answer?"""
    import time

    from app.services.generation.llm import LLMError, build_llm
    from app.services.generation.prompt import PromptPassage, build_prompt
    from app.services.generation.verify import ModelOutputError, parse_model_output, verify

    llm = build_llm(settings)
    if llm is None:
        print("LLM: disabled (EKA_LLM_BACKEND=none)")
        return True
    print(f"LLM: {llm.model_name} ({llm.backend})")
    healthy, detail = llm.health()
    if not healthy:
        print(f"  FAILED: {detail}", file=sys.stderr)
        if llm.backend == "ollama":
            print(
                "  Install Ollama (free) from https://ollama.com/download, open it once, then run:"
                f"\n    ollama pull {llm.model_name}",
                file=sys.stderr,
            )
        return False
    passages = [
        PromptPassage(1, "Employees receive 14 days of annual leave per year.", "HR", "Leave"),
        PromptPassage(2, "Parking is available on levels B2 and B3.", "Facilities", "Parking"),
    ]
    question = "How many days of annual leave do employees get?"
    t0 = time.perf_counter()
    try:
        result = llm.generate(build_prompt(question, passages), question, passages)
        checked = verify(parse_model_output(result.text), {p.number: p.text for p in passages})
    except (LLMError, ModelOutputError) as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return False
    seconds = time.perf_counter() - t0
    if not checked.supported:
        print(
            f"  WARNING: test answer had no verified citation ({seconds:.1f} s): "
            f"{result.text[:200]}"
        )
        return True  # the model runs; quality is the benchmark's job
    print(
        f"  ready in {seconds:.1f} s (first call includes loading); test answer: {checked.answer}"
    )
    return True


def _reindex(settings, db, workspace_name: str | None) -> int:
    from app.services.container import build_services
    from app.services.retrieval.embeddings import check_compatibility
    from app.services.retrieval.indexing import reindex_vectors

    workspace_id = None
    if workspace_name:
        workspace_id = db.scalar(select(Workspace.id).where(Workspace.name == workspace_name))
        if workspace_id is None:
            print(f"No workspace named {workspace_name!r}.", file=sys.stderr)
            return 1
    services = build_services(settings)
    print(f"Loading embedding model {services.embedder.model_name} ...")
    services.embedder.load()
    check_compatibility(services.embedder, settings)
    try:
        docs, chunks = reindex_vectors(services, workspace_id)
    finally:
        services.close()
    print(f"Re-embedded {chunks} chunks from {docs} documents with {services.embedder.model_name}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
