"""
devatlas.parse.contextualize
============================
Anthropic-style contextual retrieval: prepend a 50-100 token blurb to each
chunk before embedding + BM25 indexing.

WHY: chunks lose their document context. A chunk reading "this was
deprecated, use the new constructor" is unanswerable without knowing WHAT
was deprecated and in WHICH version. Anthropic measured contextual
embeddings + contextual BM25 cutting top-20 retrieval failures by 49%
(5.7% -> 2.9%) on their corpora.

COST CONTROL:
- Code/docstring chunks get a TEMPLATE blurb (free): we already KNOW the
  package, version, module, symbol, and deprecation status from parsing.
  Spending LLM tokens to restate structured metadata is waste.
- Only doc-page chunks (where the surrounding page genuinely adds meaning)
  get an LLM blurb, generated with Gemini Flash-Lite. At ~$0.10/M input
  this is a few euros for the whole docs corpus.

[LLM PATH REQUIRES GEMINI_API_KEY — the template path and prompt assembly
are fully tested; the API call follows Google's OpenAI-compatible endpoint
you already use in your bootcamp.]
"""

from __future__ import annotations

import os
from typing import Callable, Optional

from devatlas.schema import Chunk, SourceType

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
CONTEXT_MODEL = "gemini-2.5-flash-lite"

_PROMPT = """<page>
{page_text}
</page>

Here is one chunk from the page above:
<chunk>
{chunk_text}
</chunk>

Write 1-2 sentences (max 80 tokens) situating this chunk for a search
index: which product/package and version it concerns, what the page is
about, and what this specific chunk covers. Answer with ONLY the context
sentences, nothing else."""


def template_blurb(chunk: Chunk) -> str:
    """Deterministic blurb for code/docstring chunks — zero LLM cost.

    Everything an embedding needs to disambiguate the chunk is already in
    our provenance metadata; we just serialize it as natural language so
    both the dense model and BM25 can use it.
    """
    parts = [f"From {chunk.package} version {chunk.version}"]
    if chunk.symbol:
        module = ".".join(chunk.symbol.split(".")[:-1])
        parts.append(f"module {module}, defining {chunk.symbol.split('.')[-1]}")
    if chunk.deprecated:
        d = chunk.deprecated
        dep = f"DEPRECATED since {d.since or '?'}"
        if d.removal:
            dep += f", removed in {d.removal}"
        if d.alternative:
            dep += f"; use instead: {d.alternative[:120]}"
        parts.append(dep)
    if chunk.title:
        parts.append(f"section: {chunk.title}")
    return ". ".join(parts) + "."


def contextualize_chunks(
    chunks: list[Chunk],
    page_text_for: Optional[Callable[[Chunk], str]] = None,
    llm_call: Optional[Callable[[str], str]] = None,
) -> list[Chunk]:
    """Fill context_blurb on every chunk.

    llm_call is injected (dependency seam) so tests run without an API key
    and so you can swap models without touching this module.
    """
    for chunk in chunks:
        if chunk.source_type in (SourceType.SOURCE_CODE, SourceType.DOCSTRING):
            chunk.context_blurb = template_blurb(chunk)
        elif llm_call is not None and page_text_for is not None:
            page = page_text_for(chunk)[:24000]  # cap page context
            prompt = _PROMPT.format(page_text=page, chunk_text=chunk.content[:4000])
            try:
                chunk.context_blurb = llm_call(prompt).strip()
            except Exception:
                chunk.context_blurb = template_blurb(chunk)
        else:
            chunk.context_blurb = template_blurb(chunk)
    return chunks


def make_gemini_caller() -> Callable[[str], str]:
    """Returns an llm_call using the OpenAI-compatible Gemini endpoint.
    Kept in one factory so the API surface is swappable and mockable."""
    from openai import OpenAI  # deferred: not needed for template-only runs

    client = OpenAI(
        api_key=os.environ["GEMINI_API_KEY"],
        base_url=GEMINI_BASE_URL,
    )

    def call(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=CONTEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.0,   # blurbs must be deterministic, not creative
        )
        return resp.choices[0].message.content or ""

    return call
