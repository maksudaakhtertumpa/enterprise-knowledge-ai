"""Application settings, loaded from environment variables and an optional .env file.

All configuration flows through this one object so that every other module
receives typed, validated values instead of reading os.environ directly.
"""

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from app.services.retrieval.search import FusionConfig


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="EKA_",
        extra="ignore",
    )

    # --- App ---
    app_name: str = "Enterprise Knowledge AI"
    environment: Literal["development", "test", "production"] = "development"
    debug: bool = False
    api_prefix: str = "/api/v1"

    # --- Logging ---
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = True

    # --- Storage ---
    data_dir: Path = Path("./data")
    # Leave unset to use SQLite inside data_dir; set for Postgres later.
    database_url: str | None = None

    # --- CORS ---
    # Empty by default: the web UI calls the API from its own server (Next.js route
    # handlers), never from the browser, so no browser origin needs access.
    cors_origins: list[str] = Field(default_factory=list)

    # --- Rate limits (token buckets, in process memory; see app/core/ratelimit.py) ---
    # Per workspace: sustained requests per minute, and how many may arrive at once.
    rate_limit_enabled: bool = True
    rate_limit_ask_per_minute: int = Field(default=20, ge=1)
    rate_limit_ask_burst: int = Field(default=5, ge=1)
    rate_limit_search_per_minute: int = Field(default=60, ge=1)
    rate_limit_search_burst: int = Field(default=20, ge=1)
    rate_limit_ingest_per_minute: int = Field(default=60, ge=1)  # uploads + reprocessing
    rate_limit_ingest_burst: int = Field(default=30, ge=1)
    # Per client address: failed API key checks before that address is told to wait.
    rate_limit_auth_failures_per_minute: int = Field(default=10, ge=1)
    rate_limit_auth_failures_burst: int = Field(default=10, ge=1)
    # Reverse proxies (IPs or CIDRs) whose X-Forwarded-For header is believed when
    # identifying the client. Empty: the TCP peer address is the client.
    trusted_proxies: list[str] = Field(default_factory=list)

    # --- API docs ---
    # Swagger UI at /docs and ReDoc at /redoc. Off in production by default.
    api_docs: bool | None = None

    # --- Uploads ---
    max_upload_mb: int = Field(default=20, ge=1, le=200)
    max_pdf_pages: int = Field(default=2000, ge=1)
    # Zip-bomb guard for DOCX (a DOCX is a zip archive).
    max_docx_uncompressed_mb: int = Field(default=150, ge=1)

    # --- Chunking (tokens are counted with the embedding model's tokenizer) ---
    chunk_size_tokens: int = Field(default=400, ge=64, le=4096)
    chunk_overlap_tokens: int = Field(default=60, ge=0)

    # --- Embeddings (see docs/MODELS.md) ---
    # "hashing" is a dependency-free, NON-semantic stand-in for tests and CI.
    embedding_backend: Literal["sentence-transformers", "hashing"] = "sentence-transformers"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_device: Literal["auto", "mps", "cuda", "cpu"] = "auto"
    embedding_batch_size: int = Field(default=32, ge=1, le=512)
    # Load the model at startup so the first search isn't slow; off = load on first use.
    embedding_preload: bool = True

    # --- Search ---
    search_max_top_k: int = Field(default=50, ge=1, le=200)
    # Defaults chosen from measurements on a MacBook Air M4 (eval/README.md):
    # dense alone beat every hybrid setting tried, so it is the default mode.
    search_default_mode: Literal["bm25", "dense", "hybrid"] = "dense"
    # Hybrid = Reciprocal Rank Fusion of BM25 and dense rankings. Best of a 36-setting
    # sweep: k=10, BM25 weight 0.25 (candidate depth made no difference; 20 is cheap).
    rrf_k: int = Field(default=10, ge=1, le=1000)
    hybrid_candidates: int = Field(default=20, ge=1, le=500)
    hybrid_bm25_weight: float = Field(default=0.25, ge=0)
    hybrid_dense_weight: float = Field(default=1.0, ge=0)

    # --- Reranking (cross-encoder second stage) ---
    # "hashing" = offline, NOT semantic (tests/CI); "none" disables reranking.
    reranker_backend: Literal["cross-encoder", "hashing", "none"] = "cross-encoder"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: Literal["auto", "mps", "cuda", "cpu"] = "auto"
    reranker_batch_size: int = Field(default=16, ge=1, le=256)
    reranker_max_length: int = Field(default=512, ge=64, le=8192)
    reranker_half_precision: bool = True
    reranker_preload: bool = True
    # How many first-stage candidates the cross-encoder reorders. Measured on the M4:
    # 10 matched or beat 20 on every metric at about half the latency (335 vs 634 ms).
    rerank_candidates: int = Field(default=10, ge=1, le=200)
    # Whether /search reranks when the request doesn't say. On: dense + rerank raised
    # MRR from 0.900 to 0.962 on the benchmark.
    rerank_by_default: bool = True

    # --- Answer generation (local LLM only: $0, no paid APIs) ---
    # "ollama" = local model via Ollama; "extractive" = offline, NON-generative stand-in
    # that quotes the best-matching sentence (tests/CI, or machines without Ollama);
    # "none" disables /ask.
    llm_backend: Literal["ollama", "extractive", "none"] = "ollama"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:4b"
    # Context window sent explicitly: Ollama silently truncates prompts that don't fit.
    llm_num_ctx: int = Field(default=8192, ge=2048, le=131072)
    llm_max_answer_tokens: int = Field(default=512, ge=64, le=4096)
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_seed: int = 7  # with temperature 0, answers are repeatable
    llm_connect_timeout_s: float = Field(default=5.0, gt=0)
    llm_read_timeout_s: float = Field(default=180.0, gt=0)  # first call also loads the model
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    llm_keep_alive: str = "30m"  # keep the model in memory between questions

    # --- Answering ---
    answer_passages: int = Field(default=5, ge=1, le=20)  # evidence passages given to the LLM
    # Skip the LLM and answer "insufficient evidence" when the best reranker score is
    # below this. 0 = gate off. Measured on the M4 (eval/README.md): OFF is right. Any
    # threshold > 0 dropped a third of answerable questions (relevant passages often
    # score < 0.005) and caught no extra unanswerable ones; the model's own refusals did.
    answer_min_rerank_score: float = Field(default=0.0, ge=0.0, le=1.0)
    # Keep a per-workspace history of questions and answers (shown in the UI).
    ask_history: bool = True

    # --- Entailment check (does each answer sentence follow from its cited passage?) ---
    # "nli" = local NLI cross-encoder; "lexical" = offline, NON-semantic stand-in for
    # tests/CI; "none" = no check.
    entailment_backend: Literal["nli", "lexical", "none"] = "nli"
    entailment_model: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    entailment_device: Literal["auto", "mps", "cuda", "cpu"] = "auto"
    entailment_batch_size: int = Field(default=32, ge=1, le=256)
    entailment_max_length: int = Field(default=512, ge=64, le=2048)
    entailment_preload: bool = True
    # "sentences": best score over the whole cited text and every window of up to
    # `entailment_window_sentences` consecutive sentences in it (NLI models are unsure
    # on long premises). "passage": the whole cited text only.
    entailment_granularity: Literal["sentences", "passage"] = "sentences"
    entailment_window_sentences: int = Field(default=3, ge=1, le=8)
    # remove = drop sentences below the threshold; flag = keep them but mark them;
    # off = don't run the check.
    answer_entailment_mode: Literal["remove", "flag", "off"] = "remove"
    answer_entailment_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("ollama_url")
    @classmethod
    def _ollama_url_shape(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("ollama_url must start with http:// or https://")
        return v.rstrip("/")

    @field_validator("trusted_proxies")
    @classmethod
    def _proxies_are_networks(cls, v: list[str]) -> list[str]:
        from ipaddress import ip_network

        for entry in v:
            ip_network(entry, strict=False)  # ValueError -> validation error
        return v

    @field_validator("api_prefix")
    @classmethod
    def _prefix_shape(cls, v: str) -> str:
        if not v.startswith("/") or v.endswith("/"):
            raise ValueError("api_prefix must start with '/' and not end with '/'")
        return v

    @model_validator(mode="after")
    def _hybrid_weights_not_both_zero(self) -> "Settings":
        if self.hybrid_bm25_weight == 0 and self.hybrid_dense_weight == 0:
            raise ValueError("at least one hybrid weight must be greater than zero")
        return self

    @model_validator(mode="after")
    def _overlap_smaller_than_chunk(self) -> "Settings":
        if self.chunk_overlap_tokens >= self.chunk_size_tokens // 2:
            raise ValueError("chunk_overlap_tokens must be less than half of chunk_size_tokens")
        return self

    @property
    def docs_enabled(self) -> bool:
        return self.api_docs if self.api_docs is not None else self.environment != "production"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def fusion(self) -> "FusionConfig":
        from app.services.retrieval.search import FusionConfig

        return FusionConfig(
            rrf_k=self.rrf_k,
            candidates=self.hybrid_candidates,
            bm25_weight=self.hybrid_bm25_weight,
            dense_weight=self.hybrid_dense_weight,
        )

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def db_url(self) -> str:
        return self.database_url or f"sqlite:///{(self.data_dir / 'eka.sqlite3').as_posix()}"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor; override in tests via app.dependency_overrides."""
    return Settings()
