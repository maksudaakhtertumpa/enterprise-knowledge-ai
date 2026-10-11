# Phase 8 — Docker, CI/CD and AWS Design

**Budget: $0.** Docker Desktop is free for personal and educational use, the images are built from open-source bases, CI uses GitHub Actions' free minutes for public repositories, and AWS is a design document only: no account, no card, nothing created.

## What was built

| Area | What | Files |
|---|---|---|
| **API image** | Multi-stage build on `python:3.12-slim`. Two flavours: `ml` (real models, CPU PyTorch, 2.3 GB) and `lite` (offline stand-ins, 0.9 GB). Unprivileged user (UID 10001), app code read-only and root-owned, health check on `/health/ready`, no `Server` header. Models are mounted, not baked in. | `docker/api.Dockerfile`, `requirements-docker.lock`, `scripts/lock-docker.sh` |
| **Web image** | Next.js standalone server on `node:22-alpine` (331 MB), runs as the `node` user, health check. | `docker/web.Dockerfile`, `web/.dockerignore` |
| **Compose stack** | API + web on a private network; both ports published on `127.0.0.1` only; read-only root filesystems, all Linux capabilities dropped, `no-new-privileges`, memory limits, log rotation; data in a named volume per flavour; the Mac's Hugging Face cache mounted so models aren't downloaded twice; Ollama stays on the Mac and is reached at `host.docker.internal`. | `compose.yaml`, `docker/offline.env`, `scripts/docker-setup.sh`, `.dockerignore` |
| **Smoke test** | 16 checks through the web UI's own routes: non-root users, read-only filesystem, headers, docs off, readiness, workspace, browser-style session, cross-site request blocked, upload, processing, answer with verified citation and entailment check. | `scripts/docker-smoke.sh` |
| **CI** | New `docker` job: builds both images (lite flavour), starts the stack with `--wait`, runs the smoke test, prints image sizes, dumps logs on failure, tears down. | `.github/workflows/ci.yml` |
| **AWS design** | Deployment design (VPC, ALB + ACM, private GPU instance, ECR, SSM, Secrets Manager, CloudWatch, OIDC deploys), costs with sources, the managed-LLM alternative, and the code changes needed before running several replicas. | `docs/AWS.md` |

## Engineering decisions

**The LLM stays outside Docker.** Containers on a Mac run in a Linux virtual machine with no access to the Apple GPU. Ollama keeps running natively on Metal and the API container calls it at `host.docker.internal:11434`. The embedder, reranker and NLI model run on the container's CPU. On a Linux server with an NVIDIA GPU, Ollama would be one more service in the same compose file.

**CPU PyTorch, not PyPI's.** PyPI's Linux PyTorch wheels pull in about 3 GB of NVIDIA CUDA libraries, which are useless in a container without a GPU. The image installs the same version (2.14.1) as the CPU build from `download.pytorch.org`, and every other package from `requirements-docker.lock`. That lock is compiled with `requirements-ml.lock` as a constraint, so the container runs exactly the versions the tests ran with, minus the dev tools.

**Models are mounted, not baked in.** Baking 3.8 GB of model weights into the image would make every rebuild and every pull slower and the image unshareable. The compose file mounts `~/.cache/huggingface` (models already downloaded natively) and sets `HF_HUB_OFFLINE=1`, so a container never reaches out to Hugging Face unexpectedly.

**Least privilege, checked.** Both containers run as unprivileged users, with read-only root filesystems, no Linux capabilities and `no-new-privileges`. Only `/data` (named volume) and `/tmp` (tmpfs) are writable. The ports listen on `127.0.0.1` so the Wi-Fi network can't reach them. The smoke test checks the user IDs and that the app directory can't be written, so the hardening can't silently disappear.

**A bug the first build found.** The first container started and immediately failed with `Permission denied: /app/app/main.py`: the checkout's files were only readable by their owner, and `COPY` keeps permissions, so the unprivileged user couldn't read its own code. The Dockerfile now normalises permissions (`chmod -R a+rX,go-w`). The same root cause had dropped the executable bit from three scripts in the Phase 7 commit; they are fixed in this one.

**A lite flavour for CI.** The `lite` image has no ML libraries and uses the offline stand-ins, so CI can build both images and run the full smoke test in a few minutes without downloading models. It tests the plumbing (containers, network, hardening, routes), not answer quality; quality is measured by the benchmarks.

**Docker-only CI checks rather than a third-party scanner action.** Dependency vulnerabilities are already covered by `pip-audit` and `npm audit` in CI. An image scanner is listed as future work rather than added through a GitHub Action whose tags could change underneath the workflow.

## Measured on the MacBook Air M4 (16 GB)

Docker Desktop 4.94 (Engine 29.8.2), 10 CPUs and 7.7 GiB given to its VM; Ollama 0.40.2 natively.

| | Native (`make dev`, Apple GPU) | Docker (`make docker-up`, CPU in the VM) |
|---|---|---|
| Answer latency, p50 (11 development questions) | ≈ 5–7 s (Phase 7 benchmark) | **11.4 s** (max 21.5 s) |
| Reranking 10 passages, p50 | ≈ 0.3 s | 2.5 s |
| Entailment check, p50 | ≈ 0.5 s | 1.7 s |
| Query embedding | ≈ 35 ms | ≈ 0.5 s |
| LLM generation (Ollama on the GPU in both cases) | ≈ 5–6 s | 6.7 s |
| Ingesting the 11 policy documents | not measured separately | 107 s |
| API container memory during the run | — | 3.6 GiB of the 7 GiB limit |
| Image sizes | — | API ml 2.3 GB · API lite 0.9 GB · web 331 MB |

Docker roughly doubles answer time on this Mac because the retrieval and checking models lose the GPU. The answers are produced by the same models and the same code, so quality is unchanged; the smoke test's real-model run returned the expected "14 working days" with a verified citation and an entailment score. For daily use on the Mac, `make dev` stays the fast path; Docker is the reproducible, portable path and the unit of deployment.

## Run it

```bash
cd ~/Projects/enterprise-knowledge-ai
make docker-up          # build (first time ~3 min) and start; Ollama must be running
make docker-smoke       # 16 end-to-end checks
open http://localhost:3000
docker compose exec api python -m app.cli create-workspace "My team"   # key for the UI
make docker-logs        # follow logs
make docker-down        # stop (documents stay in the eka-data-ml volume)
make docker-offline     # small image, offline stand-ins, no Ollama needed
```

## Likely errors and fixes

| Symptom | Cause | Fix |
|---|---|---|
| `docker: command not found` | Docker Desktop not running, or run from the downloaded disk image | Drag Docker to Applications, eject the disk image, open it from Applications |
| `port is already allocated` (3000 or 8000) | `make dev` (or another container) is using the port | Stop it (Ctrl+C in its terminal) or `make docker-down` |
| API unhealthy; log shows `Permission denied` on `/app/...` | An image built before the permissions fix | `docker compose build --no-cache api` |
| Answers fail with `llm_unavailable` | Ollama isn't running on the Mac | Open the Ollama app; `curl localhost:11434/api/version` |
| `We couldn't connect to 'https://huggingface.co'` | A model isn't in `~/.cache/huggingface` yet | Run `python -m app.cli prepare-models` natively once, or `EKA_HF_OFFLINE=0 make docker-up` |
| `index_out_of_date` after switching flavours | Should not happen: each flavour has its own volume (`eka-data-ml`, `eka-data-lite`) | `docker volume ls`; remove the wrong one with `docker volume rm` |
| API container killed (exit 137) | Out of memory in Docker's VM | Docker Desktop → Settings → Resources: give it more memory, or lower `EKA_API_MEMORY` usage by closing other apps |

## Acceptance criteria

- [x] `make docker-up` starts the full app with real models; `make docker-smoke` passes all 16 checks (measured on the M4).
- [x] Images run as unprivileged users with read-only root filesystems; checked by the smoke test, not just configured.
- [x] Ports are reachable from this computer only; the API has no `Server` header and no public docs.
- [x] The same versions as the tested lock files; no CUDA libraries in a CPU image.
- [x] CI builds both images and runs the end-to-end smoke test on every push.
- [x] AWS design with architecture, security, deploy pipeline, costs with sources, and the changes needed to scale, clearly marked as not deployed.
- [x] $0 spent; no cloud account or card needed.

## Interview talking points

- **Why the LLM isn't in a container on the Mac:** Docker on macOS runs a Linux VM without Metal, so the GPU work stays native. I measured the cost of containerising the rest (answers ≈ 2× slower, retrieval models 5–8× slower on CPU) instead of guessing.
- **Image hygiene:** multi-stage builds, CPU-only PyTorch (no 3 GB of CUDA libraries), locked versions shared with the test suite, models mounted rather than baked in, non-root users, read-only filesystems, dropped capabilities, health checks.
- **Testing the deployment, not just the code:** the smoke test drives the containers through the browser's own routes and asserts the hardening, so a Dockerfile change that runs as root or opens the filesystem fails CI.
- **A real bug from containerising:** file permissions from the checkout made the app unreadable to the unprivileged user; the fix and its root cause (which had also stripped executable bits from scripts) are documented.
- **Cloud design with honest costs:** one GPU instance behind an ALB is the right first deployment; the GPU is about $734/month always on or $221 for office hours, and I listed exactly which components (SQLite, local Chroma, in-memory rate limits, background tasks) must change before running several replicas.
