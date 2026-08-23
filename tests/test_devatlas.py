"""
tests/test_devatlas.py
======================
The consolidated offline test suite. Everything here runs WITHOUT API keys:
real tree-sitter parsing, real in-memory Qdrant, fake dense encoders.

Fixture: tests use a small vendored sample (tests/fixtures/) so CI doesn't
need network; during development, point FIXTURE_ROOT at a real clone
(e.g. /tmp/lc010/libs/langchain) to test against actual LangChain source.

Run: pytest tests/ -v
"""

from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from devatlas.agent.graph import build_graph
from devatlas.eval_.harness import load_golden, score_deterministic
from devatlas.index.embeddings import BM25Encoder, FakeDense, _tokenize
from devatlas.index.qdrant_store import QdrantStore
from devatlas.parse.chunker import chunks_from_definitions, chunks_from_markdown
from devatlas.parse.contextualize import contextualize_chunks, template_blurb
from devatlas.parse.python_parser import PythonParser, build_symbol_records
from devatlas.retrieve.pipeline import Retriever, analyze_query, snap_to_anchor
from devatlas.schema import Chunk, Citation, ExpertAnswer, SourceType, SymbolRecord
from devatlas.version.difftable import TransitionTable, apply_version_warnings

SAMPLE = '''\
from langchain_core._api import deprecated

@deprecated(
    "0.1.0",
    alternative=(
        "Use new agent constructor methods like create_react_agent, "
        "create_json_agent, etc."
    ),
    removal="0.2.0",
)
def initialize_agent(tools, llm):
    """Load an agent executor given tools and an LLM."""
    return None

class AgentExecutor:
    """Agent that is using tools."""

    def invoke(self, input):
        """Run the agent loop until finish or limits."""
        return input

def create_agent(model, tools):
    """Create a new-style agent (v1 API)."""
    return model
'''


@pytest.fixture()
def parsed(tmp_path):
    f = tmp_path / "agents.py"
    f.write_text(SAMPLE)
    (tmp_path / "__init__.py").write_text("")
    parser = PythonParser()
    return parser, parser.parse_file(f), f, tmp_path


# -------------------------- parser ---------------------------------------

def test_parser_extracts_definitions(parsed):
    _, defs, _, _ = parsed
    names = {d.qualified_name for d in defs}
    assert {"initialize_agent", "AgentExecutor", "create_agent"} <= names


def test_parser_mines_positional_since_and_concat_alternative(parsed):
    _, defs, _, _ = parsed
    dep = next(d for d in defs if d.name == "initialize_agent").deprecation
    assert dep.since == "0.1.0"
    assert dep.removal == "0.2.0"
    assert "create_react_agent" in dep.alternative
    # concatenated literals must keep the space between parts
    assert "create_react_agent, create_json_agent" in dep.alternative


def test_docstrings_and_methods(parsed):
    _, defs, _, _ = parsed
    cls = next(d for d in defs if d.name == "AgentExecutor")
    assert cls.docstring == "Agent that is using tools."
    assert cls.children[0].qualified_name == "AgentExecutor.invoke"


# -------------------------- chunker --------------------------------------

def test_code_chunks_have_provenance(parsed):
    _, defs, f, root = parsed
    chunks = chunks_from_definitions(
        defs, module="langchain.agents", package="langchain",
        version="0.1.0", path="agents.py", repo_url_base="https://x/blob/v0.1.0")
    code = [c for c in chunks if c.source_type == SourceType.SOURCE_CODE]
    assert all(c.start_line and c.url and c.symbol for c in code)
    dep_chunk = next(c for c in code if "initialize_agent" in c.symbol)
    assert dep_chunk.deprecated.since == "0.1.0"


def test_markdown_chunker_protects_code_fences():
    md = "# Title\ntext\n```python\n# not a header\nx = 1\n```\nmore\n## Sub\nbody"
    chunks = chunks_from_markdown(md, package="langchain", version="1.0.0",
                                  url="https://d", path="p")
    assert len(chunks) == 2                      # fence '#' did NOT split
    assert "# not a header" in chunks[0].content
    assert chunks[1].title == "Title > Sub"      # breadcrumb


# -------------------------- BM25 -----------------------------------------

def test_tokenizer_emits_identifier_and_parts():
    toks = _tokenize("call initialize_agent now")
    assert "initialize_agent" in toks and "initialize" in toks and "agent" in toks


def test_bm25_ranks_exact_identifier_higher():
    docs = ["def initialize_agent(tools): pass",
            "def create_agent(model): pass",
            "completely unrelated text about cooking"]
    enc = BM25Encoder(); enc.fit(docs)
    qi, qv = enc.encode_query("initialize_agent")
    def score(doc):
        di, dv = enc.encode_document(doc)
        d = dict(zip(di, dv))
        return sum(v * d.get(i, 0.0) for i, v in zip(qi, qv))
    scores = [score(d) for d in docs]
    assert scores[0] > scores[1] > 0 and scores[2] == 0


# -------------------------- retrieval ------------------------------------

def test_version_detection():
    assert analyze_query("I'm on langchain 0.2, help").target_version == "0.2.0"
    assert analyze_query("using v0.1.0 initialize_agent").target_version == "0.1.0"
    assert analyze_query("langchain-core 1.4 question").package == "langchain-core"
    assert analyze_query("how do agents work").target_version is None
    assert snap_to_anchor("0.2.7") == "0.2.0"


@pytest.fixture()
def indexed(parsed):
    _, defs, _, _ = parsed
    chunks = chunks_from_definitions(
        defs, module="langchain.agents", package="langchain",
        version="0.1.0", path="agents.py", repo_url_base="https://x/blob/v0.1.0")
    contextualize_chunks(chunks)
    bm25 = BM25Encoder(); bm25.fit([c.embed_text for c in chunks])
    fake = FakeDense()   # default dim == EMBED_DIM: vectors must match the collection schema
    store = QdrantStore(client=QdrantClient(":memory:"))
    store.create_collection(recreate=True)
    store.upsert_chunks(chunks, fake, fake, bm25)
    return store, fake, bm25, defs


def test_hybrid_search_and_version_filter(indexed):
    store, fake, bm25, _ = indexed
    hits = store.hybrid_search("initialize_agent", fake, fake, bm25,
                               version="0.1.0", limit=3)
    assert hits and any("initialize_agent" in (h.symbol or "") for h in hits)
    assert store.hybrid_search("initialize_agent", fake, fake, bm25,
                               version="0.2.0", limit=3) == []


# -------------------------- version diffing ------------------------------

def _symbols(defs, version):
    return build_symbol_records(defs, "langchain.agents", "langchain",
                                version, "agents.py")


def test_transition_table_and_warnings(parsed):
    _, defs, _, _ = parsed
    t1 = _symbols(defs, "0.1.0")
    t2 = [r.model_copy(update={"version": "0.2.0"}) for r in t1
          if "initialize_agent" not in r.fq_name]
    table = TransitionTable().build({"0.1.0": t1, "0.2.0": t2})
    kinds = {t.kind.value for t in table.transitions}
    assert {"removed", "deprecated"} <= kinds

    ans = ExpertAnswer(answer="Sure, call initialize_agent.")
    ans = apply_version_warnings(ans, table, "how to use initialize_agent?")
    joined = " ".join(ans.version_warnings)
    assert "deprecated since 0.1.0" in joined and "removed between" in joined


# -------------------------- agent graph ----------------------------------

def test_graph_happy_path_and_loop_bound(parsed, indexed):
    _, defs, _, _ = parsed
    store, fake, bm25, _ = indexed
    retriever = Retriever(store, fake, fake, bm25)
    table = TransitionTable().build({"0.1.0": _symbols(defs, "0.1.0")})

    calls = {"rewrite": 0}
    def grade_fn(q, ch): return "zzz" not in q
    def generate_fn(q, ch, v):
        return ExpertAnswer(answer="ok", citations=[
            Citation(package="langchain", version="0.1.0", url=ch[0].url, quote="x"),
            Citation(package="langchain", version="0.1.0", url="https://invalid", quote="y")])
    def rewrite_fn(q):
        calls["rewrite"] += 1; return q

    graph = build_graph(retriever, table, grade_fn, generate_fn, rewrite_fn)

    out = graph.invoke({"question": "initialize_agent in v0.1.0?"})
    assert len(out["answer"].citations) == 1          # invalid citation dropped
    assert out["answer"].version_warnings              # warning stamped

    out2 = graph.invoke({"question": "zzz nonsense"})
    assert calls["rewrite"] == 2                       # bounded loop
    assert out2["answer"].insufficient_context         # honest failure


# -------------------------- eval harness ---------------------------------

def test_golden_dataset_loads_and_scores():
    golden = load_golden(Path("data/golden/golden.jsonl"))
    assert len(golden) >= 5
    table = TransitionTable()
    ans = ExpertAnswer(answer="deprecated — use create_react_agent",
                       version_warnings=["x is deprecated"])
    trap = next(g for g in golden if g.category == "deprecated_trap")
    det = score_deterministic([(trap, ans, set())], table)
    assert det.version_correctness == 1.0
