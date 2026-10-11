"""Models on Apple's GPU (MPS) must never run at the same time (Metal aborts the process).

Fake models record how many inferences overlap; real torch is not needed.
"""

import threading
import time

import numpy as np

from app.services.generation.entailment import NLIEntailmentChecker
from app.services.retrieval.embeddings import SentenceTransformerEmbedder, inference_lock
from app.services.retrieval.reranker import CrossEncoderReranker


class Overlap:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def run(self, result):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.02)
        with self.lock:
            self.active -= 1
        return result


def _models(device, overlap):
    class FakeST:
        tokenizer = None

        def encode(self, texts, **_):
            return overlap.run(np.zeros((len(texts), 4)))

    class FakeCE:  # reranker: one relevance score per pair
        def predict(self, pairs, **_):
            return overlap.run(np.zeros(len(pairs)))

    class FakeNLI:  # entailment: probabilities per label
        def predict(self, pairs, **_):
            return overlap.run(np.full((len(pairs), 3), 1 / 3))

    embedder = SentenceTransformerEmbedder("fake", device=device)
    embedder.device, embedder._model = device, FakeST()
    reranker = CrossEncoderReranker("fake", device=device)
    reranker.device, reranker._model = device, FakeCE()
    nli = NLIEntailmentChecker("fake", device=device)
    nli.device, nli._model, nli.entailment_index = device, FakeNLI(), 0
    return [
        lambda: embedder.embed_query("q"),
        lambda: reranker.score("q", ["p"]),
        lambda: nli.entailment([("p", "h")]),
    ]


def _max_overlap(device):
    overlap = Overlap()
    calls = _models(device, overlap) * 4
    errors = []

    def guarded(call):
        try:
            call()
        except Exception as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=guarded, args=(call,)) for call in calls]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    return overlap.max_active


def test_models_on_mps_take_turns():
    assert _max_overlap("mps") == 1


def test_models_elsewhere_may_run_together():
    assert _max_overlap("cpu") > 1


def test_lock_choice():
    own = threading.Lock()
    assert inference_lock("mps", own) is inference_lock("mps", threading.Lock())
    assert inference_lock("cpu", own) is own and inference_lock(None, own) is own
