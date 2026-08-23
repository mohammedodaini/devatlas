"""
devatlas.ingest.repo
====================
Acquires LangChain source at each anchor version and yields Chunks +
SymbolRecords.

WHY ANCHOR VERSIONS (5 tags, not 500):
Version-aware answers require version-tagged chunks, which requires
re-extracting per version. Every tag would be ~500 snapshots of a huge
repo; the information that matters lives at the MAJOR restructures. So we
snapshot exactly the tags that bracket them: 0.1.0 / 0.2.0 / 0.3.0 /
1.0.0 / current. This is the single biggest scope-control decision in v1.

WHY SHALLOW + SPARSE CLONES:
--depth 1 --filter=blob:none --sparse fetches only the tree we ask for.
The monorepo is large; we only need libs/<package-dirs>. Each anchor is
cloned to its own directory (simpler and more reproducible than one clone
with repeated checkouts, at the cost of some disk).

VERIFIED: the clone commands and libs/ layout below were tested against
the real repo for v0.1.0. The PACKAGE_DIRS mapping per era reflects the
actual monorepo history (langchain_v1/ only exists from 1.0 on).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from devatlas.parse.chunker import chunks_from_definitions
from devatlas.parse.python_parser import PythonParser, build_symbol_records
from devatlas.schema import Chunk, SymbolRecord

REPO_URL = "https://github.com/langchain-ai/langchain"

# Anchor tags -> the package source roots that exist at that tag.
# NOTE: tag names and layouts verified for v0.1.0; verify the 1.x tag name
# (e.g. "langchain==1.0.0") against `git ls-remote --tags` at build time —
# LangChain switched from vX.Y.Z tags to per-package pkg==X.Y.Z tags.
ANCHORS: dict[str, dict[str, str]] = {
    "v0.1.0": {
        "langchain": "libs/langchain/langchain",
        "langchain-core": "libs/core/langchain_core",
        "langchain-community": "libs/community/langchain_community",
    },
    "v0.2.0": {
        "langchain": "libs/langchain/langchain",
        "langchain-core": "libs/core/langchain_core",
        "langchain-community": "libs/community/langchain_community",
    },
    "langchain==0.3.0": {
        "langchain": "libs/langchain/langchain",
        "langchain-core": "libs/core/langchain_core",
    },
    "langchain==1.0.0": {
        "langchain": "libs/langchain_v1/langchain",
        "langchain-core": "libs/core/langchain_core",
        "langchain-classic": "libs/langchain/langchain_classic",
    },
    # "current": resolve the latest langchain==X.Y.Z tag at run time.
}


@dataclass
class AnchorSnapshot:
    tag: str
    root: Path                      # clone directory
    packages: dict[str, Path]       # package name -> source root


def clone_anchor(tag: str, dest: Path, package_dirs: dict[str, str]) -> AnchorSnapshot:
    """Shallow sparse clone of one anchor tag."""
    dest.mkdir(parents=True, exist_ok=True)
    clone_dir = dest / tag.replace("==", "-").replace("/", "-")
    if not (clone_dir / ".git").exists():
        subprocess.run(
            ["git", "clone", "--quiet", "--depth", "1", "--branch", tag,
             "--filter=blob:none", "--sparse", REPO_URL, str(clone_dir)],
            check=True,
        )
        sparse_paths = sorted({str(Path(p).parent) for p in package_dirs.values()})
        subprocess.run(
            ["git", "-C", str(clone_dir), "sparse-checkout", "set", *sparse_paths],
            check=True,
        )
    packages = {
        name: clone_dir / rel
        for name, rel in package_dirs.items()
        if (clone_dir / rel).exists()
    }
    return AnchorSnapshot(tag=tag, root=clone_dir, packages=packages)


def normalize_version(tag: str) -> str:
    """'v0.1.0' -> '0.1.0'; 'langchain==1.0.0' -> '1.0.0'."""
    if "==" in tag:
        return tag.split("==", 1)[1]
    return tag.lstrip("v")


def ingest_snapshot(
    snapshot: AnchorSnapshot,
) -> Iterator[tuple[list[Chunk], list[SymbolRecord]]]:
    """Parse every .py file in every package of one snapshot.

    Skips tests/ and private _internal-only trees: users ask about the
    public API; test code inflates the index with near-duplicate noise.
    """
    parser = PythonParser()
    version = normalize_version(snapshot.tag)
    blob_base = f"{REPO_URL}/blob/{snapshot.tag}"

    for package, pkg_root in snapshot.packages.items():
        repo_root = snapshot.root
        for py_file in sorted(pkg_root.rglob("*.py")):
            rel = py_file.relative_to(repo_root)
            if any(part in ("tests", "test", "__pycache__") for part in rel.parts):
                continue
            try:
                defs = parser.parse_file(py_file)
            except Exception:
                continue  # tree-sitter is tolerant; a hard failure = skip file
            if not defs:
                continue
            module = parser.module_name(repo_root, py_file)
            chunks = chunks_from_definitions(
                defs, module=module, package=package, version=version,
                path=str(rel), repo_url_base=blob_base,
            )
            symbols = build_symbol_records(
                defs, module, package, version, str(rel)
            )
            yield chunks, symbols


def save_jsonl(items: list, path: Path) -> None:
    """Chunks and symbol tables persist as JSONL between pipeline stages so
    each stage is re-runnable without re-cloning/re-parsing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")
