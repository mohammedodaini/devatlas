"""
devatlas.parse.chunker
======================
Turns parsed definitions and markdown docs into Chunks.

CODE CHUNKING — the pragmatic path (v1):
One chunk per function or class; classes over CLASS_SPLIT_LINES are split
into per-method chunks with the class header + docstring prepended so each
method chunk is self-contained. This captures most of the benefit of full
cAST split-then-merge (EMNLP 2025) at a fraction of the complexity.
The invariant we never break: a chunk NEVER ends mid-function — a chunk
that stops inside a body cannot ground an answer and hurts faithfulness.

DOCS CHUNKING:
Header-aware splitting on #/##/### so each chunk is one coherent section,
carrying its full header path ("Agents > Migration > AgentExecutor") as
the title. Sections longer than MAX_DOC_CHARS are split on paragraph
boundaries — never mid-paragraph.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from devatlas.parse.python_parser import ParsedDefinition
from devatlas.schema import Chunk, SourceType

CLASS_SPLIT_LINES = 150   # classes longer than this get per-method chunks
MAX_DOC_CHARS = 4000      # ~1000 tokens; sections beyond this are split


# ---------------------------------------------------------------------------
# Code -> Chunks
# ---------------------------------------------------------------------------

def chunks_from_definitions(
    defs: list[ParsedDefinition],
    *,
    module: str,
    package: str,
    version: str,
    path: str,
    repo_url_base: str,
) -> list[Chunk]:
    """repo_url_base example:
    https://github.com/langchain-ai/langchain/blob/v0.1.0
    """
    chunks: list[Chunk] = []

    def blob_url(start: int, end: int) -> str:
        return f"{repo_url_base}/{path}#L{start}-L{end}"

    for d in defs:
        fq = f"{module}.{d.qualified_name}"
        span = d.end_line - d.start_line + 1

        if d.kind == "class" and span > CLASS_SPLIT_LINES and d.children:
            # Class header chunk: signature + docstring + attribute region
            # (everything before the first method), so questions about the
            # class itself ("what is AgentExecutor?") still retrieve well.
            first_method_line = min(c.start_line for c in d.children)
            header_lines = d.source.split("\n")[: first_method_line - d.start_line]
            header_src = "\n".join(header_lines).rstrip()
            if header_src.strip():
                chunks.append(Chunk(
                    package=package, version=version,
                    source_type=SourceType.SOURCE_CODE,
                    path=path, start_line=d.start_line,
                    end_line=first_method_line - 1,
                    url=blob_url(d.start_line, first_method_line - 1),
                    symbol=fq, content=header_src,
                    deprecated=d.deprecation,
                ))
            # One chunk per method, prefixed with class context so the chunk
            # is self-contained ("which class does invoke() belong to?").
            class_ctx = f"# In class {fq}:\n# {d.signature}\n"
            for m in d.children:
                if m.name.startswith("_") and not m.name.startswith("__"):
                    continue  # private methods: noise, not public API
                chunks.append(Chunk(
                    package=package, version=version,
                    source_type=SourceType.SOURCE_CODE,
                    path=path, start_line=m.start_line, end_line=m.end_line,
                    url=blob_url(m.start_line, m.end_line),
                    symbol=f"{module}.{m.qualified_name}",
                    content=class_ctx + m.source,
                    deprecated=m.deprecation or d.deprecation,
                ))
        else:
            chunks.append(Chunk(
                package=package, version=version,
                source_type=SourceType.SOURCE_CODE,
                path=path, start_line=d.start_line, end_line=d.end_line,
                url=blob_url(d.start_line, d.end_line),
                symbol=fq, content=d.source,
                deprecated=d.deprecation,
            ))

        # Docstrings additionally become PROSE chunks: they answer
        # "what does X do" questions better through the prose embedding
        # space than through the code space.
        if d.docstring and len(d.docstring) > 80:
            chunks.append(Chunk(
                package=package, version=version,
                source_type=SourceType.DOCSTRING,
                path=path, start_line=d.start_line, end_line=d.end_line,
                url=blob_url(d.start_line, d.end_line),
                symbol=fq,
                content=f"{d.signature}\n\n{d.docstring}",
                deprecated=d.deprecation,
            ))
    return chunks


# ---------------------------------------------------------------------------
# Markdown -> Chunks
# ---------------------------------------------------------------------------

_HEADER_RE = re.compile(r"^(#{1,3})\s+(.*)$")


def chunks_from_markdown(
    text: str,
    *,
    package: str,
    version: str,
    url: str,
    path: str,
    source_type: SourceType = SourceType.DOC_PAGE,
) -> list[Chunk]:
    """Header-aware splitting. Each chunk carries the full header breadcrumb
    as `title`, which the contextualizer and citations both use.

    Fenced code blocks are protected: a '#' inside ``` fences is code, not a
    header — naive splitters corrupt exactly the doc pages (code-heavy
    tutorials) that matter most for a library expert.
    """
    lines = text.split("\n")
    header_stack: list[tuple[int, str]] = []   # (level, text)
    sections: list[tuple[str, list[str]]] = [] # (breadcrumb, lines)
    current: list[str] = []
    in_fence = False

    def breadcrumb() -> str:
        return " > ".join(h for _, h in header_stack) or (path or "document")

    def flush() -> None:
        if any(l.strip() for l in current):
            sections.append((breadcrumb(), current.copy()))
        current.clear()

    for line in lines:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            current.append(line)
            continue
        m = None if in_fence else _HEADER_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            while header_stack and header_stack[-1][0] >= level:
                header_stack.pop()
            header_stack.append((level, m.group(2).strip()))
        else:
            current.append(line)
    flush()

    chunks: list[Chunk] = []
    for title, sec_lines in sections:
        body = "\n".join(sec_lines).strip()
        for piece in _split_long(body, MAX_DOC_CHARS):
            chunks.append(Chunk(
                package=package, version=version, source_type=source_type,
                path=path, url=url, title=title, content=piece,
            ))
    return chunks


def _split_long(text: str, max_chars: int) -> list[str]:
    """Split on paragraph boundaries, greedily packing under max_chars."""
    if len(text) <= max_chars:
        return [text]
    paragraphs = text.split("\n\n")
    out, buf, size = [], [], 0
    for p in paragraphs:
        if size + len(p) > max_chars and buf:
            out.append("\n\n".join(buf))
            buf, size = [], 0
        buf.append(p)
        size += len(p) + 2
    if buf:
        out.append("\n\n".join(buf))
    return out


# ---------------------------------------------------------------------------
# Notebook (.ipynb) -> markdown-ish text, then reuse the markdown chunker
# ---------------------------------------------------------------------------

def notebook_to_markdown(nb_json: dict) -> str:
    """Concatenate cells in order; code cells become fenced blocks so the
    fence-protection above applies and cell order (the tutorial narrative)
    is preserved."""
    parts: list[str] = []
    for cell in nb_json.get("cells", []):
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        if cell.get("cell_type") == "markdown":
            parts.append(src)
        elif cell.get("cell_type") == "code":
            parts.append(f"```python\n{src}\n```")
    return "\n\n".join(parts)
