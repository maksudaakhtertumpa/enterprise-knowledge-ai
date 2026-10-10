# Phase 1 — Project Setup

## What was built
A FastAPI service skeleton with the cross-cutting pieces every later phase depends on:

| File | Responsibility |
|---|---|
| `app/core/config.py` | Typed settings from env/.env (`EKA_` prefix); invalid config fails at boot |
| `app/core/logging.py` | JSON logs to stdout, each line tagged with the request ID |
| `app/core/middleware.py` | Request-ID (accepts a safe client ID or mints one) + access log with latency |
| `app/core/errors.py` | One error envelope for 4xx/5xx; never echoes inputs or stack traces |
| `app/api/routes/health.py` | `/health` (liveness) and `/health/ready` (readiness, 503 when a dependency fails) |
| `app/main.py` | `create_app(settings)` factory — tests build isolated apps |
| `tests/` | 22 tests: health, readiness failure, error envelope, request-ID sanitising, config validation |
| `.github/workflows/ci.yml` | Lint + tests on Python 3.11 and 3.12 from pinned lock file |

## Design decisions
- **App factory, not a global app.** Tests pass a `Settings` with a temp `data_dir`, so no test touches your real data and tests can't leak state into each other.
- **Pure-ASGI middleware.** Starlette's `BaseHTTPMiddleware` runs the handler in a separate task, so a context variable set there is invisible to the outermost 500 handler. Pure ASGI keeps the request ID on crash responses too — there's a test proving it.
- **Liveness vs readiness.** Docker/AWS restart a container when liveness fails but only stop *routing* to it when readiness fails. Mixing them up causes restart loops when, e.g., the vector DB is briefly down.
- **Validation errors drop `input`.** FastAPI's default 422 echoes the submitted value; for a system handling sensitive documents we don't reflect user data back.
- **Pinned lock files.** `pyproject.toml` declares compatible ranges; `requirements*.lock` pins exact versions so CI, Docker and your laptop install identical packages.
- **Health routes at root, features under `/api/v1`.** Load balancers shouldn't need to know the API version.

## How to test
```bash
pytest                                    # 22 passed
pytest --cov --cov-report=term-missing    # ~99% coverage
uvicorn app.main:app --reload             # then open http://localhost:8000/docs
curl -i http://localhost:8000/health      # note the x-request-id header
curl -i http://localhost:8000/nope        # 404 in the error envelope, same request_id
```

## Likely errors and fixes
| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: app` | Package not installed / venv not active | Activate venv, `pip install -e ".[dev]"` |
| `python` not found / wrong version (Windows) | Multiple Pythons | `py -3.11 -m venv .venv` |
| `Activate.ps1 cannot be loaded` | PowerShell execution policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `ValidationError` at startup | Bad value in `.env` | Error names the field; compare with `.env.example` |
| `EKA_CORS_ORIGINS` error | Lists must be JSON | `EKA_CORS_ORIGINS=["http://localhost:8501"]` |
| `Address already in use` | Port 8000 taken | `--port 8001` |
| CI fails on `ruff format --check` | Unformatted code | `ruff format .` locally, commit |

## Acceptance criteria
- [x] `pytest` passes (22 tests) on Python 3.11 and 3.12
- [x] `ruff check` and `ruff format --check` clean
- [x] `/health` 200; `/health/ready` 200 normally and 503 when the data dir is unusable
- [x] Every response has `x-request-id`; error bodies carry the same ID
- [x] Unhandled exceptions return a generic 500 with no internal detail
- [x] OpenAPI docs served at `/docs`
- [ ] CI green on GitHub (after your first push)

## Interview talking points
- "I separated liveness and readiness probes and can explain why the difference matters in container orchestration."
- "Every request gets a correlation ID that flows into logs and error responses — I sanitise client-supplied IDs to prevent log injection."
- "Config is validated at boot, so a bad deploy fails immediately instead of at the first request."
- "I used an app factory and dependency overrides so tests are hermetic."
- "CI installs from a lock file across two Python versions."
