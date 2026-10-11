# syntax=docker/dockerfile:1.7
# FastAPI backend image.
#
#   FLAVOR=ml   (default) real models: sentence-transformers + CPU PyTorch (2.3 GB image)
#   FLAVOR=lite no ML libraries: offline stand-ins only, for CI and quick checks (0.9 GB)
#
# Models are NOT baked in (several GB): they are read from a mounted Hugging Face
# cache, so a rebuild never re-downloads them. See compose.yaml.

ARG PYTHON_VERSION=3.12

# ------------------------------------------------------------------ build
FROM python:${PYTHON_VERSION}-slim AS build
ARG FLAVOR=ml
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /venv
ENV PATH=/venv/bin:$PATH
WORKDIR /src
COPY requirements.lock requirements-docker.lock ./
# Exact pins, no resolver surprises: every package and version comes from a lock file.
RUN if [ "$FLAVOR" = "ml" ]; then \
      TORCH="$(sed -n 's/^# torch==\([^ ]*\) (CPU build).*/\1/p' requirements-docker.lock)" \
      && test -n "$TORCH" \
      && pip install --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH}" \
      && pip install --no-deps -r requirements-docker.lock; \
    elif [ "$FLAVOR" = "lite" ]; then \
      pip install --no-deps -r requirements.lock; \
    else \
      echo "FLAVOR must be ml or lite" >&2 && exit 1; \
    fi

# ---------------------------------------------------------------- runtime
FROM python:${PYTHON_VERSION}-slim AS runtime
ARG FLAVOR=ml
LABEL org.opencontainers.image.title="enterprise-knowledge-ai-api" \
      org.opencontainers.image.description="Hybrid-search RAG API with verified citations" \
      org.opencontainers.image.source="https://github.com/maksudaakhtertumpa/enterprise-knowledge-ai" \
      org.opencontainers.image.licenses="MIT"
ENV PATH=/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    EKA_DATA_DIR=/data \
    HF_HOME=/models/huggingface \
    EKA_IMAGE_FLAVOR=${FLAVOR}
# An unprivileged user with a fixed UID, so mounted volumes have predictable owners.
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /home/app --create-home app \
    && mkdir -p /data /models/huggingface \
    && chown -R app:app /data /models
COPY --from=build /venv /venv
WORKDIR /app
COPY --chown=root:root app ./app
# Readable by everyone, writable by root only, whatever permissions the checkout had.
RUN chmod -R a+rX,go-w /app
USER app
EXPOSE 8000
# No curl in the slim image; Python's standard library is enough for a health probe.
HEALTHCHECK --interval=15s --timeout=5s --start-period=600s --retries=5 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=4).status == 200 else 1)"]
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--no-server-header", "--timeout-graceful-shutdown", "20"]
