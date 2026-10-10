# Shortcuts for macOS/Linux/WSL. On Windows PowerShell, run the commands
# shown on each line directly (see README "Common commands").

.PHONY: install install-ml run test cov lint format check lock smoke bench sweep verify

install:            ## App + dev tools (offline hashing embedder only)
	pip install -r requirements-dev.lock && pip install -e . --no-deps

install-ml:         ## App + dev tools + PyTorch / sentence-transformers (real embeddings)
	pip install -r requirements-ml.lock && pip install -e . --no-deps

run:                ## Start the API with auto-reload on http://localhost:8000
	uvicorn app.main:app --reload --port 8000

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

verify:             ## Everything above, in order
	./scripts/verify.sh

lock:               ## Re-pin dependencies (needs uv: pip install uv)
	uv pip compile --universal pyproject.toml --python-version 3.11 -o requirements.lock
	uv pip compile --universal pyproject.toml --extra dev --python-version 3.11 -o requirements-dev.lock
	uv pip compile --universal pyproject.toml --extra dev --extra ml --python-version 3.11 -o requirements-ml.lock
