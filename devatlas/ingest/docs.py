"""
devatlas.ingest.docs
====================
Fetches LangChain documentation via the llms.txt manifest.

WHY llms.txt INSTEAD OF CRAWLING:
LangChain publishes https://python.langchain.com/llms.txt — a curated
Markdown index of doc pages, each fetchable as .md. Using it means:
no HTML boilerplate stripping, no sitemap spelunking, no robots.txt risk,
and the maintainers themselves chose what's in it. We filter it because
the manifest is heavily weighted toward LangSmith/platform pages; the OSS
framework subset is what a LangChain expert needs.

[REQUIRES NETWORK to python.langchain.com — the manifest PARSING below is
unit-tested; the HTTP fetching is straightforward but was not run in the
build sandbox. Run `python -m devatlas.ingest.docs --check` first.]
"""

from __future__ import annotations

import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from devatlas.parse.chunker import chunks_from_markdown
from devatlas.schema import Chunk, SourceType

LLMS_TXT_URL = "https://python.langchain.com/llms.txt"
USER_AGENT = "DevAtlas/0.1 (educational project; contact in repo README)"
REQUEST_DELAY_S = 0.6   # ~1.5 req/s: be a polite citizen of someone else's docs

# Only framework pages; drop LangSmith / platform / API-reference noise.
INCLUDE_PATTERNS = (r"/oss/python/", r"/docs/concepts", r"/docs/how_to",
                    r"/docs/tutorials", r"/docs/versions", r"/migrat")
EXCLUDE_PATTERNS = (r"langsmith", r"/api_reference/", r"/platform/")

_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")


@dataclass
class DocPage:
    title: str
    url: str


def parse_llms_txt(text: str) -> list[DocPage]:
    """llms.txt is Markdown: '- [Title](url): description' lines.
    Returns the filtered framework subset, deduplicated, order-preserving."""
    pages: list[DocPage] = []
    seen: set[str] = set()
    for title, url in _LINK_RE.findall(text):
        low = url.lower()
        if any(re.search(p, low) for p in EXCLUDE_PATTERNS):
            continue
        if not any(re.search(p, low) for p in INCLUDE_PATTERNS):
            continue
        if url in seen:
            continue
        seen.add(url)
        pages.append(DocPage(title=title.strip(), url=url))
    return pages


def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", "replace")


def md_url(page_url: str) -> str:
    """Mintlify serves each page as raw Markdown at <url>.md."""
    return page_url if page_url.endswith(".md") else page_url.rstrip("/") + ".md"


def ingest_docs(version_label: str, cache_dir: Path) -> list[Chunk]:
    """Fetch manifest -> fetch each page as .md -> header-aware chunks.

    version_label: current docs describe the CURRENT release line; we label
    doc chunks with that (e.g. "1.3"). Legacy versioned docs (0.1/0.2 sites)
    can be added as separate manifest runs with their own label.

    Pages are cached to disk so re-runs (e.g. after a chunker fix) cost
    zero network — the same incremental principle as the code pipeline.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = _fetch(LLMS_TXT_URL)
    pages = parse_llms_txt(manifest)

    chunks: list[Chunk] = []
    for page in pages:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", page.url)[-120:]
        cached = cache_dir / f"{slug}.md"
        if cached.exists():
            text = cached.read_text(encoding="utf-8")
        else:
            try:
                text = _fetch(md_url(page.url))
            except Exception:
                continue
            cached.write_text(text, encoding="utf-8")
            time.sleep(REQUEST_DELAY_S)

        source_type = (SourceType.MIGRATION_GUIDE
                       if "migrat" in page.url.lower() or "versions" in page.url.lower()
                       else SourceType.DOC_PAGE)
        chunks.extend(chunks_from_markdown(
            text, package="langchain", version=version_label,
            url=page.url, path=page.url, source_type=source_type,
        ))
    return chunks
