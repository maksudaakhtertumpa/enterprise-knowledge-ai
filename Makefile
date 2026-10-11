# Shortcuts for macOS/Linux/WSL. On Windows PowerShell, run the commands
# shown on each line directly (see README "Common commands").

.PHONY: install install-ml run web-install web dev web-check e2e test cov lint format check lock smoke bench sweep answers heldout report audit verify docker-up docker-offline docker-smoke docker-logs docker-down

install:            ## App + dev tools (offline hashing embedder only)
	pip install -r requirements-dev.lock && pip install -e . --no-deps

install-ml:         ## App + dev tools + PyTorch / sentence-transformers (real embeddings)
	pip install -r requirements-ml.lock && pip install -e . --no-deps

run:                ## Start the API with auto-reload on http://localhost:8000
	uvicorn app.main:app --reload --port 8000 --no-server-header

web-install:        ## Web UI dependencies (needs Node.js 20.9+; exact versions from package-lock.json)
	cd web && npm ci --no-fund --no-audit
	./scripts/web-setup.sh

web:                ## Start the web UI on http://localhost:3000 (start the API first)
	./scripts/web-setup.sh
	cd web && NEXT_TELEMETRY_DISABLED=1 npx next dev --hostname 127.0.0.1 --port 3000

dev:                ## Start the API and the web UI together (Ctrl+C stops both)
	./scripts/dev.sh

web-check:          ## Web UI: format, lint, types, unit tests, production build
	cd web && NEXT_TELEMETRY_DISABLED=1 npm run check

e2e:                ## Web UI end-to-end tests in a real browser (first: cd web && npx playwright install chromium)
	cd web && NEXT_TELEMETRY_DISABLED=1 npm run e2e

test:               ## Run the test suite
	pytest

cov:                ## Tests with a coverage report
	pytest --cov --cov-report=term-missing

lint:               ## Static checks
	ruff check .
	ruff format --check .

format:             ## Auto-fix formatting and safe lint issues
	ruff format .
	ruff check --fix .

check: lint test    ## What CI runs

smoke:              ## Start a real server and exercise upload/search end to end
	python scripts/smoke_test.py

bench:              ## Retrieval benchmark: BM25 vs dense vs hybrid (writes eval/results/)
	python -m eval.run_retrieval

sweep:              ## Try 36 hybrid fusion settings on one ingested corpus
	python -m eval.sweep_fusion

answers:            ## Answer benchmark: grounded Q&A, refusals, injection (needs Ollama)
	python -m eval.run_answers

heldout:            ## Held-out set: retrieval + answers on questions never used for tuning
	python -m eval.run_retrieval --dataset eval/datasets/teratai-heldout
	python -m eval.run_answers --dataset eval/datasets/teratai-heldout --no-injection

report:             ## Charts + docs/evaluation/README.md from the saved benchmark runs
	python -m eval.report

verify:             ## Everything above, in order
	./scripts/verify.sh

audit:              ## Known-vulnerability scan of pinned Python and web dependencies
	python -m pip install -q pip-audit==2.10.1
	python -m pip_audit --strict --no-deps --disable-pip -r requirements-ml.lock \
		--ignore-vuln PYSEC-2026-311 --ignore-vuln PYSEC-2026-3813 \
		--ignore-vuln PYSEC-2026-3814 --ignore-vuln PYSEC-2026-3815
	cd web && npm audit --omit=dev --audit-level=high

docker-up:          ## API + web UI in Docker (real models; Ollama stays on the Mac) -> http://localhost:3000
	./scripts/docker-setup.sh
	docker compose up -d --build --wait --wait-timeout 900

docker-offline:     ## Same with the small "lite" image and offline stand-ins (no models, no Ollama)
	./scripts/docker-setup.sh
	docker compose --env-file docker/offline.env up -d --build --wait --wait-timeout 300

docker-smoke:       ## End-to-end check of the running Docker stack
	./scripts/docker-smoke.sh

docker-logs:        ## Follow the containers' logs
	docker compose logs -f

docker-down:        ## Stop the containers (your documents stay in the Docker volume)
	docker compose down

lock:               ## Re-pin dependencies (needs uv: pip install uv)
	uv pip compile --universal pyproject.toml --python-version 3.11 -o requirements.lock
	uv pip compile --universal pyproject.toml --extra dev --python-version 3.11 -o requirements-dev.lock
	uv pip compile --universal pyproject.toml --extra dev --extra ml --python-version 3.11 -o requirements-ml.lock
	./scripts/lock-docker.sh
