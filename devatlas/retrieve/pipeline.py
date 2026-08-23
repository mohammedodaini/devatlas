"""
devatlas.retrieve.pipeline
==========================
Query understanding + retrieval orchestration in front of the store.

Version detection: "I'm on langchain 0.2" / "in v0.1.0" / "langchain-core
1.4" -> a payload filter. If no version is stated we DON'T filter (retrieve
across versions) and let the agent surface differences; silently pinning to
'current' would hide exactly the version drift the product exists to expose.

The mapping from a loose user version ("0.2") to an ANCHOR version
("0.2.0") is explicit: chunks only exist at anchor snapshots, so we snap to
the nearest anchor at-or-below the requested version — the code the user
runs on 0.2.7 is the 0.2.0 anchor's API surface plus patches.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from devatlas.index.embeddings import BM25Encoder, DenseEncoder
from devatlas.index.qdrant_store import QdrantStore
from devatlas.schema import Chunk

ANCHOR_VERSIONS = ["0.1.0", "0.2.0", "0.3.0", "1.0.0"]

_VERSION_RE = re.compile(
    r"(?:langchain(?:[-_]core|[-_]community|[-_]classic)?\s*(?:==|\s+v?|\s*version\s*)|(?:^|\s)v)"
    r"(\d+)(?:\.(\d+))?(?:\.(\d+))?",
    re.IGNORECASE,
)
_PACKAGE_RE = re.compile(r"langchain[-_](core|community|classic)", re.IGNORECASE)


@dataclass
class QueryContext:
    raw: str
    target_version: Optional[str] = None   # snapped anchor, e.g. "0.2.0"
    stated_version: Optional[str] = None   # what the user literally said
    package: Optional[str] = None
    rewrites: list[str] = field(default_factory=list)


def _version_key(v: str) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3]) or (0,)


def snap_to_anchor(stated: str) -> Optional[str]:
    """'0.2' or '0.2.7' -> '0.2.0' (nearest anchor at-or-below)."""
    sk = _version_key(stated)
    best = None
    for anchor in ANCHOR_VERSIONS:
        if _version_key(anchor)[:len(sk)] <= sk:
            best = anchor
    # exact-major/minor preference: if stated 0.2.x pick the 0.2 anchor
    for anchor in ANCHOR_VERSIONS:
        if _version_key(anchor)[:2] == sk[:2]:
            return anchor
    return best


def analyze_query(question: str) -> QueryContext:
    ctx = QueryContext(raw=question)
    m = _VERSION_RE.search(question)
    if m:
        parts = [g for g in m.groups() if g is not None]
        ctx.stated_version = ".".join(parts)
        ctx.target_version = snap_to_anchor(ctx.stated_version)
    pm = _PACKAGE_RE.search(question)
    if pm:
        ctx.package = f"langchain-{pm.group(1).lower()}"
    return ctx


class Retriever:
    """Thin orchestration over the store; owns the encoders so callers
    (the agent graph) don't."""

    def __init__(
        self,
        store: QdrantStore,
        prose_encoder: DenseEncoder,
        code_encoder: DenseEncoder,
        bm25: BM25Encoder,
    ) -> None:
        self.store = store
        self.prose = prose_encoder
        self.code = code_encoder
        self.bm25 = bm25

    def retrieve(
        self,
        question: str,
        ctx: Optional[QueryContext] = None,
        limit: int = 10,
    ) -> list[Chunk]:
        ctx = ctx or analyze_query(question)
        return self.store.hybrid_search(
            question,
            self.prose, self.code, self.bm25,
            version=ctx.target_version,
            package=ctx.package,
            limit=limit,
        )
