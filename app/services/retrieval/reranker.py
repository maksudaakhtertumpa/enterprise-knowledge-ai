"""Second-stage reranking with a cross-encoder.

Retrieval (BM25, dense, hybrid) scores the query and each passage
*separately*: fast, but it never looks at them together. A cross-encoder
reads "query [SEP] passage" as one input and scores how well that passage
answers that query. It is far more accurate and far slower (one model call
per candidate), so it only reorders the top N candidates from retrieval:

    retrieve top N (cheap, high recall)  ->  cross-encoder reorders N  ->  return top k

Backends:
- `cross-encoder`: sentence-transformers CrossEncoder, default
  `BAAI/bge-reranker-v2-m3` (multilingual, MIT). Runs on MPS / CUDA / CPU,
  half precision on GPUs to halve memory, inference serialised by a lock.
- `hashing`: query-term overlap with no model. NOT semantic; exists so tests
  and CI can exercise the reranking pipeline offline.
- `none`: reranking disabled; requests asking for it get a clear error.
"""

import logging
import re
import threading
from typing import Any, Protocol

from app.core.config import Settings
from app.services.retrieval.embeddings import inference_lock, resolve_device

logger = logging.getLogger(__name__)


class RerankerUnavailableError(RuntimeError):
    pass


class Reranker(Protocol):
    model_name: str

    @property
    def is_loaded(self) -> bool: ...

    def load(self) -> None: ...

    def score(self, query: str, passages: list[str]) -> list[float]: ...

    def describe(self) -> dict[str, Any]: ...


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        batch_size: int = 16,
        max_length: int = 512,
        half_precision: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device_preference = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.half_precision = half_precision
        self.device: str | None = None
        self.dtype: str | None = None
        self._model: Any = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

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
                import torch
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - depends on install
                raise RerankerUnavailableError(
                    "sentence-transformers is not installed. Install the ML extra or set "
                    "EKA_RERANKER_BACKEND=none (or hashing for tests)."
                ) from exc

            self.device = resolve_device(self.device_preference)
            model_kwargs: dict[str, Any] = {}
            # fp16 halves memory (~2.2 GB -> ~1.1 GB for bge-reranker-v2-m3) with no
            # meaningful ranking change; CPUs gain nothing from it, so keep fp32 there.
            if self.half_precision and self.device in ("mps", "cuda"):
                model_kwargs["dtype"] = torch.float16
            self.dtype = "float16" if model_kwargs else "float32"
            logger.info(
                "reranker_loading",
                extra={"model": self.model_name, "device": self.device, "dtype": self.dtype},
            )
            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=self.max_length,
                model_kwargs=model_kwargs,
                # Sigmoid maps logits to (0, 1): comparable across queries, easy to threshold.
                activation_fn=torch.nn.Sigmoid(),
            )
            logger.info("reranker_loaded", extra=self.describe())

    def describe(self) -> dict[str, Any]:
        return {
            "backend": "cross-encoder",
            "model": self.model_name,
            "device": self.device,
            "dtype": self.dtype,
            "max_length": self.max_length,
        }

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        if self._model is None:
            self.load()
        with inference_lock(self.device, self._infer_lock):
            scores = self._model.predict(
                [(query, p) for p in passages],
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        return [float(s) for s in scores]


_WORD = re.compile(r"\w+", re.UNICODE)


class HashingReranker:
    """Share of query terms present in the passage. Deterministic, NOT semantic."""

    model_name = "overlap-v1"

    @property
    def is_loaded(self) -> bool:
        return True

    def load(self) -> None:
        return None

    def describe(self) -> dict[str, Any]:
        return {"backend": "hashing", "model": self.model_name}

    def score(self, query: str, passages: list[str]) -> list[float]:
        terms = set(_WORD.findall(query.lower()))
        if not terms:
            return [0.0] * len(passages)
        return [len(terms & set(_WORD.findall(p.lower()))) / len(terms) for p in passages]


def build_reranker(settings: Settings) -> Reranker | None:
    if settings.reranker_backend == "none":
        return None
    if settings.reranker_backend == "hashing":
        return HashingReranker()
    return CrossEncoderReranker(
        settings.reranker_model,
        device=settings.reranker_device,
        batch_size=settings.reranker_batch_size,
        max_length=settings.reranker_max_length,
        half_precision=settings.reranker_half_precision,
    )
