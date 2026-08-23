"""
devatlas.schema
===============
The data contracts for the whole system. Everything downstream (parsing,
indexing, retrieval, the agent, evaluation) speaks these types.

WHY THIS FILE EXISTS FIRST:
Provenance is the product. A citation-grade expert must say exactly where
every claim came from: which package, which version, which file, which
lines. If you retrofit these fields after embedding 50k chunks, you
re-embed everything. So the schema is locked before any ingestion runs.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Source taxonomy
# ---------------------------------------------------------------------------

class SourceType(str, Enum):
    """Where a chunk came from. Drives trust weighting and retrieval filters.

    v1 scope: SOURCE_CODE, DOCSTRING, DOC_PAGE, MIGRATION_GUIDE.
    RELEASE_NOTE and ISSUE are declared now (schema stability) but only
    populated in v1.1 — adding enum members later is cheap, re-embedding
    because a field was missing is not.
    """
    SOURCE_CODE = "source_code"
    DOCSTRING = "docstring"
    DOC_PAGE = "doc_page"
    MIGRATION_GUIDE = "migration_guide"
    RELEASE_NOTE = "release_note"   # v1.1
    ISSUE = "issue"                 # v1.1


class DeprecationInfo(BaseModel):
    """Mined mechanically from LangChain's @deprecated decorator.

    langchain_core._api.deprecation.deprecated(since=..., removal=...,
    alternative=..., message=...) is the maintainers' own machine-readable
    declaration. We parse it from the AST — deterministic, free, and more
    reliable than any LLM inference about deprecations.
    """
    since: Optional[str] = None        # version when deprecated
    removal: Optional[str] = None      # version when (to be) removed
    alternative: Optional[str] = None  # what to use instead
    alternative_import: Optional[str] = None
    message: Optional[str] = None


# ---------------------------------------------------------------------------
# The chunk — the atomic unit of the index
# ---------------------------------------------------------------------------

class Chunk(BaseModel):
    """One retrievable unit with full provenance.

    Design decisions:
    - `package` + `version` together identify the snapshot. LangChain's
      monorepo versions packages INDEPENDENTLY (langchain-core 1.4.x vs
      langchain 1.3.x), so a bare version number is meaningless.
    - `symbol` is fully qualified (package-qualified) because the same name
      exists in multiple packages (create_react_agent lives in both
      langchain.agents and langgraph.prebuilt).
    - `content` is what gets embedded; `context_blurb` (Anthropic-style
      contextual retrieval) is PREPENDED to content before embedding and
      BM25 indexing, and stored separately so we can re-contextualize
      without re-parsing.
    """
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    package: str                       # e.g. "langchain-core"
    version: str                       # e.g. "0.1.0" — the anchor tag
    source_type: SourceType
    path: str                          # repo-relative or doc URL path
    start_line: Optional[int] = None   # code only
    end_line: Optional[int] = None
    url: str = ""                      # citation target (GitHub blob URL / doc URL)
    symbol: Optional[str] = None       # fully-qualified, e.g. "langchain.agents.initialize_agent"
    title: Optional[str] = None        # docs: header path "Agents > Getting started"
    content: str                       # the raw chunk text
    context_blurb: str = ""            # 50-100 token situating blurb (Part 2)
    deprecated: Optional[DeprecationInfo] = None
    content_hash: str = ""             # for incremental re-indexing
    indexed_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("chunk content must not be empty")
        return v

    def model_post_init(self, __context) -> None:
        # Content-hash for change detection: if the hash is unchanged at the
        # next sync, we skip re-embedding this chunk (incremental indexing).
        if not self.content_hash:
            self.content_hash = hashlib.sha256(
                self.content.encode("utf-8")
            ).hexdigest()[:16]

    @property
    def embed_text(self) -> str:
        """What actually goes to the embedding model / BM25.

        WHY: a chunk saying 'this was deprecated' is useless without knowing
        WHAT and in WHICH version. The blurb restores that context.
        Anthropic measured a 49% reduction in top-20 retrieval failures from
        contextual embeddings + contextual BM25.
        """
        if self.context_blurb:
            return f"{self.context_blurb}\n\n{self.content}"
        return self.content

    @property
    def is_code(self) -> bool:
        """Routes the chunk to the code embedding space vs the prose space.
        Docstrings are natural language -> prose space (voyage-code-3 is for
        actual code; docstring questions are 'what does X do' questions)."""
        return self.source_type == SourceType.SOURCE_CODE


# ---------------------------------------------------------------------------
# Symbol table — input to cross-version diffing (Part 5)
# ---------------------------------------------------------------------------

class SymbolRecord(BaseModel):
    """One public symbol at one (package, version). The version-transition
    table is computed by diffing these across anchor versions."""
    fq_name: str                       # "langchain.agents.initialize_agent"
    package: str
    version: str
    kind: str                          # "function" | "class" | "method"
    signature: str = ""
    docstring_first_line: str = ""
    path: str = ""
    start_line: int = 0
    deprecated: Optional[DeprecationInfo] = None


class TransitionKind(str, Enum):
    ADDED = "added"
    REMOVED = "removed"
    CHANGED = "changed"        # signature changed
    DEPRECATED = "deprecated"


class SymbolTransition(BaseModel):
    """One row of the version-transition table: what happened to a symbol
    between two anchor versions. This is what powers answers like
    'initialize_agent was deprecated in 0.1.0 — use create_agent'."""
    fq_name: str
    package: str
    kind: TransitionKind
    from_version: Optional[str] = None
    to_version: Optional[str] = None
    old_signature: Optional[str] = None
    new_signature: Optional[str] = None
    deprecation: Optional[DeprecationInfo] = None


# ---------------------------------------------------------------------------
# Agent output — structured, enforced with Pydantic
# ---------------------------------------------------------------------------

class Citation(BaseModel):
    """Every factual claim in an answer must map to one of these."""
    symbol: Optional[str] = None
    package: str
    version: str
    url: str
    quote: str = Field(description="verbatim supporting snippet from the chunk")


class ExpertAnswer(BaseModel):
    """The agent's final, structured output.

    WHY STRUCTURED: free-text answers can't be checked. With this model we
    can (a) verify every citation URL exists in the retrieved set,
    (b) post-process symbols against the deprecation table, and
    (c) measure citation accuracy in evaluation.
    """
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    version_warnings: list[str] = Field(default_factory=list)
    insufficient_context: bool = False
    target_version: Optional[str] = None
