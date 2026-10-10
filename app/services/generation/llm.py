"""Language-model backends. Local only: no paid API is used anywhere.

- `OllamaClient`: a local model served by Ollama (default `qwen3.5:4b`). Talks
  to Ollama's HTTP API with the standard library, so there is no extra
  dependency. Timeouts are explicit; transient failures are retried with
  backoff; permanent ones (model not pulled, Ollama not running) fail fast with
  an actionable message.
- `ExtractiveLLM`: deterministic and offline. It does NOT generate text: it
  quotes the evidence sentence that best matches the question. Used by tests
  and CI, and usable on machines without Ollama. Labelled clearly everywhere.

Both return the same JSON shape, so the verifier treats them identically.
Another provider could implement `LLMClient`, but under this project's $0 rule
none is included.
"""

import json
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.core.config import Settings
from app.services.generation.prompt import ANSWER_SCHEMA, BuiltPrompt, PromptPassage

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------- errors
class LLMError(RuntimeError):
    code = "llm_error"
    status_code = 502


class LLMUnavailableError(LLMError):
    code = "llm_unavailable"
    status_code = 503


class LLMModelMissingError(LLMError):
    code = "llm_model_missing"
    status_code = 503


class LLMTimeoutError(LLMError):
    code = "llm_timeout"
    status_code = 504


# ---------------------------------------------------------------------- results
@dataclass(slots=True)
class LLMResult:
    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float = 0.0
    load_ms: float | None = None
    retries: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


class LLMClient(Protocol):
    model_name: str
    backend: str

    def generate(
        self, prompt: BuiltPrompt, question: str, passages: list[PromptPassage]
    ) -> LLMResult: ...

    def health(self) -> tuple[bool, str]: ...

    def describe(self) -> dict[str, Any]: ...


# ----------------------------------------------------------------------- Ollama
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class OllamaClient:
    backend = "ollama"

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        num_ctx: int = 8192,
        max_tokens: int = 512,
        temperature: float = 0.0,
        seed: int = 7,
        connect_timeout_s: float = 5.0,
        read_timeout_s: float = 180.0,
        max_retries: int = 2,
        keep_alive: str = "30m",
        backoff_s: float = 0.5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model
        self.num_ctx = num_ctx
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed
        self.connect_timeout_s = connect_timeout_s
        self.read_timeout_s = read_timeout_s
        self.max_retries = max_retries
        self.keep_alive = keep_alive
        self.backoff_s = backoff_s
        self._think_supported = True  # flips off if the model rejects the "think" flag

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model_name,
            "url": self.base_url,
            "num_ctx": self.num_ctx,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    # -- HTTP -------------------------------------------------------------------
    def _request(self, method: str, path: str, body: dict | None, timeout: float) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(  # noqa: S310 - URL comes from validated settings
            self.base_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return json.loads(resp.read() or b"{}")

    def health(self) -> tuple[bool, str]:
        """Is Ollama running, and is the configured model pulled?"""
        try:
            tags = self._request("GET", "/api/tags", None, timeout=self.connect_timeout_s)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return False, f"Ollama not reachable at {self.base_url} ({type(exc).__name__})"
        names = {m.get("name") for m in tags.get("models", [])} | {
            m.get("model") for m in tags.get("models", [])
        }
        wanted = {self.model_name, f"{self.model_name}:latest"}
        if not names & wanted:
            return (
                False,
                f"model {self.model_name!r} not pulled (run: ollama pull {self.model_name})",
            )
        return True, self.model_name

    def generate(
        self, prompt: BuiltPrompt, question: str, passages: list[PromptPassage]
    ) -> LLMResult:
        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
            "stream": False,
            "format": ANSWER_SCHEMA,
            "keep_alive": self.keep_alive,
            "options": {
                "num_ctx": self.num_ctx,
                "num_predict": self.max_tokens,
                "temperature": self.temperature,
                "seed": self.seed,
            },
        }
        if self._think_supported:
            # Reasoning traces add seconds and tokens; grounded extraction doesn't need them.
            body["think"] = False

        start = time.perf_counter()
        attempt = 0
        while True:
            try:
                data = self._request("POST", "/api/chat", body, timeout=self.read_timeout_s)
                break
            except urllib.error.HTTPError as exc:
                detail = _error_text(exc)
                if exc.code == 404 or "not found" in detail.lower():
                    raise LLMModelMissingError(
                        f"Model {self.model_name!r} is not available in Ollama. "
                        f"Run: ollama pull {self.model_name}"
                    ) from exc
                if exc.code == 400 and "think" in detail.lower() and "think" in body:
                    self._think_supported = False  # older/other models: retry without the flag
                    body.pop("think")
                    continue
                if exc.code in _RETRYABLE_STATUS and attempt < self.max_retries:
                    attempt += 1
                    time.sleep(self.backoff_s * 2 ** (attempt - 1))
                    continue
                raise LLMError(f"Ollama returned HTTP {exc.code}: {detail[:200]}") from exc
            except TimeoutError as exc:  # socket.timeout is an alias since Python 3.10
                # No retry: a request that timed out once would likely time out again.
                raise LLMTimeoutError(
                    f"The model did not answer within {self.read_timeout_s:.0f} s."
                ) from exc
            except urllib.error.URLError as exc:
                reason = exc.reason
                if isinstance(reason, TimeoutError):
                    raise LLMTimeoutError(
                        f"The model did not answer within {self.read_timeout_s:.0f} s."
                    ) from exc
                if isinstance(reason, ConnectionRefusedError):
                    raise LLMUnavailableError(
                        f"Ollama is not running at {self.base_url}. Start the Ollama app."
                    ) from exc
                if attempt < self.max_retries:  # resets, DNS blips: worth one more try
                    attempt += 1
                    time.sleep(self.backoff_s * 2 ** (attempt - 1))
                    continue
                raise LLMUnavailableError(f"Could not reach Ollama: {reason}") from exc

        message = data.get("message") or {}
        return LLMResult(
            text=message.get("content", ""),
            model=data.get("model", self.model_name),
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
            load_ms=round(data["load_duration"] / 1e6, 1) if data.get("load_duration") else None,
            retries=attempt,
            meta={"done_reason": data.get("done_reason")},
        )


def _error_text(exc: urllib.error.HTTPError) -> str:
    try:
        payload = json.loads(exc.read() or b"{}")
        return str(payload.get("error", "")) or exc.reason
    except (ValueError, OSError):
        return str(exc.reason)


# ------------------------------------------------------------------- extractive
_WORD = re.compile(r"\w+", re.UNICODE)
_STOP = frozenset(
    "a an and are as at be by can do does for from how i in is it my of on or our should "
    "that the their this to was we what when where which who why will with you your "
    "berapa saya yang di ke dan untuk apa bila adakah".split()
)
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")  # line breaks too: headings aren't sentences


class ExtractiveLLM:
    """Quotes the single evidence sentence that shares the most words with the question.

    Not a language model and not semantic. It exists so the whole answering
    pipeline (prompt, verification, API, UI) runs offline in tests and CI.
    """

    backend = "extractive"
    model_name = "extractive-v1"

    def describe(self) -> dict[str, Any]:
        return {"backend": self.backend, "model": self.model_name}

    def health(self) -> tuple[bool, str]:
        return True, self.model_name

    def generate(
        self, prompt: BuiltPrompt, question: str, passages: list[PromptPassage]
    ) -> LLMResult:
        start = time.perf_counter()
        terms = {w for w in _WORD.findall(question.casefold()) if w not in _STOP}
        best: tuple[int, int, str] | None = None  # (overlap, -passage number, sentence)
        for p in passages:
            for sentence in _SENT.split(p.text):
                overlap = len(terms & set(_WORD.findall(sentence.casefold())))
                candidate = (overlap, -p.number, sentence.strip())
                if overlap and (best is None or candidate[:2] > best[:2]):
                    best = candidate
        if best is None:
            payload = {"answer": "", "insufficient_evidence": True, "citations": []}
        else:
            number, sentence = -best[1], best[2]
            body = sentence.rstrip(".!? ")
            payload = {
                "answer": f"{body}. [{number}]",
                "insufficient_evidence": False,
                "citations": [{"passage": number, "quote": sentence}],
            }
        return LLMResult(
            text=json.dumps(payload),
            model=self.model_name,
            prompt_tokens=None,
            completion_tokens=None,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
        )


def build_llm(settings: Settings) -> LLMClient | None:
    if settings.llm_backend == "none":
        return None
    if settings.llm_backend == "extractive":
        return ExtractiveLLM()
    return OllamaClient(
        settings.ollama_url,
        settings.ollama_model,
        num_ctx=settings.llm_num_ctx,
        max_tokens=settings.llm_max_answer_tokens,
        temperature=settings.llm_temperature,
        seed=settings.llm_seed,
        connect_timeout_s=settings.llm_connect_timeout_s,
        read_timeout_s=settings.llm_read_timeout_s,
        max_retries=settings.llm_max_retries,
        keep_alive=settings.llm_keep_alive,
    )
