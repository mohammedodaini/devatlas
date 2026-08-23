"""
devatlas.index.embeddings
=========================
Three encoders feed the index:

1. PROSE dense  — gemini-embedding-001 (docs, docstrings). 1024 dims.
2. CODE dense   — voyage-code-3 (source chunks). 1024 dims.
   WHY TWO MODELS: code and prose have different token distributions;
   voyage-code-3 beats general text embedders by ~13% on code retrieval.
   The two vector spaces are INCOMPATIBLE — you can't compare a Gemini
   vector to a Voyage vector — which is why the Qdrant collection uses
   NAMED vectors and the retriever routes queries per space.
3. SPARSE       — a self-contained BM25 encoder (below). Sparse catches
   exact identifiers ('initialize_agent') that dense models blur.

WHY A HAND-ROLLED BM25 (~60 lines) INSTEAD OF fastembed/SPLADE:
(a) zero model downloads, fully deterministic, trivially debuggable;
(b) you should be able to explain BM25's term-frequency saturation and
    length normalization in an interview — owning the code makes that real;
(c) Qdrant only needs {index: weight} sparse vectors; how they're produced
    is our business. Swap in SPLADE later behind the same interface.

[Gemini/Voyage paths REQUIRE API KEYS; BM25 is fully tested. Both dense
clients are behind the same protocol so a FakeDense encoder drives the
integration tests.]
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from typing import Protocol

# ---------------------------------------------------------------------------
# Dense encoder protocol + implementations
# ---------------------------------------------------------------------------

EMBED_DIM = 1024


class DenseEncoder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class GeminiEncoder:
    """Prose space. Batch-friendly: send up to ~100 texts per request.
    Uses output_dimensionality=1024 to match the code space dims (uniform
    storage; Matryoshka truncation costs little quality)."""

    MODEL = "gemini-embedding-001"

    def __init__(self) -> None:
        from google import genai  # deferred import
        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def encode(self, texts: list[str]) -> list[list[float]]:
        from google.genai import types
        result = self._client.models.embed_content(
            model=self.MODEL,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=EMBED_DIM,
            ),
        )
        return [e.values for e in result.embeddings]

    def encode_query(self, text: str) -> list[float]:
        from google.genai import types
        result = self._client.models.embed_content(
            model=self.MODEL, contents=[text],
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",       # asymmetric: query != doc
                output_dimensionality=EMBED_DIM,
            ),
        )
        return result.embeddings[0].values


class VoyageCodeEncoder:
    """Code space. voyage-code-3, 1024 dims, input_type document/query.
    First 200M tokens are free — the whole LangChain code corpus fits."""

    MODEL = "voyage-code-3"

    def __init__(self) -> None:
        import voyageai  # deferred import
        self._client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

    def encode(self, texts: list[str]) -> list[list[float]]:
        out = self._client.embed(
            texts, model=self.MODEL, input_type="document", output_dimension=EMBED_DIM,
        )
        return out.embeddings

    def encode_query(self, text: str) -> list[float]:
        out = self._client.embed(
            [text], model=self.MODEL, input_type="query", output_dimension=EMBED_DIM,
        )
        return out.embeddings[0]


class FakeDense:
    """Deterministic hash-projection encoder for tests: similar token sets
    -> similar vectors. NOT semantically meaningful — it exists so the
    Qdrant integration test exercises real named-vector plumbing offline."""

    def __init__(self, dim: int = EMBED_DIM) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in _tokenize(text):
            v[hash(tok) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def encode_query(self, text: str) -> list[float]:
        return self._vec(text)


# ---------------------------------------------------------------------------
# BM25 sparse encoder
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+|\d+")


def _tokenize(text: str) -> list[str]:
    """Code-aware tokenization: keeps identifiers intact AND emits their
    snake_case parts, so both 'initialize_agent' (exact) and 'initialize'
    (partial recall) are indexed. This one detail is most of the reason
    sparse search works on code."""
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text.lower()):
        tokens.append(raw)
        if "_" in raw:
            tokens.extend(p for p in raw.split("_") if len(p) > 2)
    return tokens


class BM25Encoder:
    """Classic BM25 (Robertson/Sparck Jones) producing Qdrant sparse vectors.

    Two-pass by design: fit() learns document frequencies over the corpus
    (IDF needs global statistics), then encode_*() produce per-item sparse
    vectors. k1 controls term-frequency saturation (more repeats of a term
    help less and less); b controls length normalization (long files don't
    win just by being long).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.doc_freq: Counter = Counter()
        self.n_docs = 0
        self.avg_len = 1.0
        self.vocab: dict[str, int] = {}

    def fit(self, corpus: list[str]) -> None:
        total_len = 0
        for text in corpus:
            toks = _tokenize(text)
            total_len += len(toks)
            for t in set(toks):
                self.doc_freq[t] += 1
        self.n_docs = len(corpus)
        self.avg_len = (total_len / self.n_docs) if self.n_docs else 1.0
        self.vocab = {t: i for i, t in enumerate(sorted(self.doc_freq))}

    def _idf(self, term: str) -> float:
        df = self.doc_freq.get(term, 0)
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def encode_document(self, text: str) -> tuple[list[int], list[float]]:
        toks = _tokenize(text)
        tf = Counter(toks)
        length = len(toks) or 1
        idx, vals = [], []
        for term, freq in tf.items():
            if term not in self.vocab:
                continue
            score = self._idf(term) * (freq * (self.k1 + 1)) / (
                freq + self.k1 * (1 - self.b + self.b * length / self.avg_len)
            )
            if score > 0:
                idx.append(self.vocab[term])
                vals.append(score)
        return idx, vals

    def encode_query(self, text: str) -> tuple[list[int], list[float]]:
        """Query side: IDF only (standard practice — query term frequency
        is nearly always 1 and saturating it adds nothing)."""
        idx, vals = [], []
        for term in set(_tokenize(text)):
            if term in self.vocab:
                idx.append(self.vocab[term])
                vals.append(self._idf(term))
        return idx, vals
