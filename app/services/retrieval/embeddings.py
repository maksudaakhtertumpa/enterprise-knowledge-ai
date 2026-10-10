"""Embedding models behind one small interface.

- `SentenceTransformerEmbedder`: the real thing. Runs on Apple-silicon GPU
  (MPS), CUDA or CPU. Loaded lazily, guarded by a lock, and inference is
  serialised so background ingestion and live searches never hit the GPU at
  the same time (MPS is not reliably thread-safe).
- `HashingEmbedder`: deterministic feature hashing with no model download.
  It is NOT semantic — it only knows which words overlap. It exists so tests
  and CI can exercise the dense pipeline (Chroma, filters, isolation) quickly
  and offline. Never benchmark semantic search with it.

The chunker counts tokens with the same embedder, so a "400-token chunk" means
400 tokens of the model that will embed it.
"""

import hashlib
import inspect
import logging
import math
import re
import threading
from typing import Any, Protocol

from app.core.config import Settings
from app.services.ingestion.chunking import approx_token_count

logger = logging.getLogger(__name__)


class EmbeddingConfigError(RuntimeError):
    """The configured model can't work with the configured chunking."""


class Embedder(Protocol):
    model_name: str

    @property
    def dimension(self) -> int: ...

    @property
    def max_tokens(self) -> int | None: ...

    @property
    def is_loaded(self) -> bool: ...

    def load(self) -> None: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    def count_tokens(self, text: str) -> int: ...

    def describe(self) -> dict[str, Any]: ...


def resolve_device(preference: str) -> str:
    """auto -> cuda, then mps (Apple silicon), then cpu."""
    import torch

    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 32) -> None:
        self.model_name = model_name
        self.device_preference = device
        self.batch_size = batch_size
        self.device: str | None = None
        self._model: Any = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    # -------------------------------------------------------------- lifecycle
    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - depends on install
                raise EmbeddingConfigError(
                    "sentence-transformers is not installed. Install the ML extra "
                    '(pip install -e ".[ml]") or set EKA_EMBEDDING_BACKEND=hashing.'
                ) from exc

            self.device = resolve_device(self.device_preference)
            kwargs: dict[str, Any] = {}
            if "qwen" in self.model_name.lower():
                # Qwen3-Embedding pools the last token, so padding must be on the left.
                # sentence-transformers 6 renamed `tokenizer_kwargs` to `processor_kwargs`.
                params = inspect.signature(SentenceTransformer.__init__).parameters
                key = "processor_kwargs" if "processor_kwargs" in params else "tokenizer_kwargs"
                kwargs[key] = {"padding_side": "left"}
            logger.info(
                "embedding_model_loading",
                extra={"model": self.model_name, "device": self.device},
            )
            self._model = SentenceTransformer(self.model_name, device=self.device, **kwargs)
            logger.info("embedding_model_loaded", extra=self.describe())

    def _require(self) -> Any:
        if self._model is None:
            self.load()
        return self._model

    # -------------------------------------------------------------- metadata
    @property
    def dimension(self) -> int:
        model = self._require()
        getter = getattr(model, "get_embedding_dimension", None) or (
            model.get_sentence_embedding_dimension
        )
        return int(getter())

    @property
    def max_tokens(self) -> int | None:
        value = getattr(self._require(), "max_seq_length", None)
        return int(value) if value else None

    def describe(self) -> dict[str, Any]:
        return {
            "backend": "sentence-transformers",
            "model": self.model_name,
            "device": self.device,
            "dimension": self.dimension if self.is_loaded else None,
            "max_tokens": self.max_tokens if self.is_loaded else None,
        }

    # ------------------------------------------------------------- inference
    def count_tokens(self, text: str) -> int:
        tokenizer = self._require().tokenizer
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._require()
        encode = getattr(model, "encode_document", model.encode)
        with self._infer_lock:
            vectors = encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        model = self._require()
        # encode_query applies the model's query instruction (Qwen3 uses one).
        encode = getattr(model, "encode_query", model.encode)
        with self._infer_lock:
            vector = encode(
                [text], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
            )[0]
        return vector.tolist()


_WORD = re.compile(r"\w+", re.UNICODE)


class HashingEmbedder:
    """Lexical feature hashing (unigrams + bigrams), L2-normalised. Not semantic."""

    model_name = "hashing-v1"

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def max_tokens(self) -> int | None:
        return None

    @property
    def is_loaded(self) -> bool:
        return True

    def load(self) -> None:
        return None

    def describe(self) -> dict[str, Any]:
        return {"backend": "hashing", "model": self.model_name, "dimension": self._dimension}

    def count_tokens(self, text: str) -> int:
        return approx_token_count(text)

    def _vector(self, text: str) -> list[float]:
        words = _WORD.findall(text.lower())
        features = words + [f"{a} {b}" for a, b in zip(words, words[1:], strict=False)]
        vec = [0.0] * self._dimension
        for feature in features:
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self._dimension
            vec[index] += 1.0 if digest[4] & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(
        settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )


def check_compatibility(embedder: Embedder, settings: Settings) -> None:
    """Fail at startup rather than silently truncating chunks during embedding."""
    limit = embedder.max_tokens
    # Room for the contextual header (title/section) prepended before embedding.
    needed = settings.chunk_size_tokens + 64
    if limit is not None and needed > limit:
        raise EmbeddingConfigError(
            f"Embedding model {embedder.model_name!r} accepts {limit} tokens, but chunks can "
            f"reach {needed} (chunk_size_tokens + header). Lower EKA_CHUNK_SIZE_TOKENS "
            f"to at most {limit - 64} or choose a model with a longer context."
        )
