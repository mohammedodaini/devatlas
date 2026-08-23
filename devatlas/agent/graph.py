"""
devatlas.agent.graph
====================
The agentic-RAG loop as a LangGraph StateGraph:

  START -> analyze -> retrieve -> grade --(relevant)--> generate -> postprocess -> END
                          ^          |
                          |     (not relevant, retry < MAX)
                          +---- rewrite

DESIGN DECISIONS:
- BOUNDED loop: retry_count in state, MAX_RETRIES=2. Unbounded
  rewrite->retrieve loops hit GraphRecursionError on out-of-scope
  questions; a bound turns "loop forever" into "answer honestly that we
  don't know" (insufficient_context=True), which grounding requires anyway.
- The LLM is INJECTED as two callables (grade_fn, generate_fn). The graph
  is pure orchestration — testable offline with fakes, swappable between
  Gemini/others without touching the wiring. The graph structure itself is
  VERIFIED in tests with a fake LLM.
- generate returns an ExpertAnswer (Pydantic). Structured output lets the
  postprocess node verify citations against the retrieved set and lets the
  version layer stamp warnings deterministically.

WHY RAW StateGraph AND NOT create_agent: the grade->rewrite loop with a
retry bound and a deterministic postprocess node is custom control flow;
prebuilt agents optimize for tool-calling loops, not this shape.
"""

from __future__ import annotations

import json
import os
from typing import Callable, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from devatlas.retrieve.pipeline import QueryContext, Retriever, analyze_query
from devatlas.schema import Chunk, Citation, ExpertAnswer
from devatlas.version.difftable import TransitionTable, apply_version_warnings

MAX_RETRIES = 2
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GENERATE_MODEL = "gemini-2.5-flash"

GradeFn = Callable[[str, list[Chunk]], bool]
GenerateFn = Callable[[str, list[Chunk], Optional[str]], ExpertAnswer]
RewriteFn = Callable[[str], str]


class AgentState(TypedDict, total=False):
    question: str            # possibly-rewritten working question
    original_question: str   # what the user actually asked
    ctx: QueryContext
    retrieved: list[Chunk]
    relevant: bool
    retry_count: int
    answer: ExpertAnswer


def build_graph(
    retriever: Retriever,
    table: TransitionTable,
    grade_fn: GradeFn,
    generate_fn: GenerateFn,
    rewrite_fn: RewriteFn,
):
    def analyze(state: AgentState) -> AgentState:
        q = state["question"]
        return {
            "original_question": q,
            "ctx": analyze_query(q),
            "retry_count": 0,
        }

    def retrieve(state: AgentState) -> AgentState:
        chunks = retriever.retrieve(state["question"], state.get("ctx"))
        return {"retrieved": chunks}

    def grade(state: AgentState) -> AgentState:
        chunks = state.get("retrieved", [])
        return {"relevant": bool(chunks) and grade_fn(state["question"], chunks)}

    def rewrite(state: AgentState) -> AgentState:
        return {
            "question": rewrite_fn(state["question"]),
            "retry_count": state.get("retry_count", 0) + 1,
        }

    def generate(state: AgentState) -> AgentState:
        ctx = state.get("ctx")
        target = ctx.target_version if ctx else None
        if not state.get("relevant"):
            # Honest failure beats a confident hallucination: this is the
            # grounding contract, and it's what faithfulness measures.
            answer = ExpertAnswer(
                answer=("I could not retrieve sufficiently relevant LangChain "
                        "sources for this question, so I won't guess."),
                insufficient_context=True,
                target_version=target,
            )
        else:
            answer = generate_fn(
                state["original_question"], state["retrieved"], target
            )
            answer.target_version = target
        return {"answer": answer}

    def postprocess(state: AgentState) -> AgentState:
        answer = state["answer"]
        # 1) Deterministic version warnings get the last word (Part 5).
        answer = apply_version_warnings(
            answer, table, state["original_question"]
        )
        # 2) Citation integrity: drop any citation whose URL is not in the
        #    retrieved set — the model cannot invent sources.
        valid_urls = {c.url for c in state.get("retrieved", [])}
        answer.citations = [c for c in answer.citations if c.url in valid_urls]
        return {"answer": answer}

    def route_after_grade(state: AgentState) -> str:
        if state.get("relevant"):
            return "generate"
        if state.get("retry_count", 0) < MAX_RETRIES:
            return "rewrite"
        return "generate"   # exhausted retries -> honest insufficient answer

    g = StateGraph(AgentState)
    g.add_node("analyze", analyze)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("rewrite", rewrite)
    g.add_node("generate", generate)
    g.add_node("postprocess", postprocess)

    g.add_edge(START, "analyze")
    g.add_edge("analyze", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route_after_grade,
                            {"generate": "generate", "rewrite": "rewrite"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", "postprocess")
    g.add_edge("postprocess", END)
    return g.compile()


# ---------------------------------------------------------------------------
# Gemini-backed node functions [REQUIRE GEMINI_API_KEY]
# ---------------------------------------------------------------------------

_GENERATE_SYSTEM = """You are the DevAtlas LangChain Expert. Answer ONLY from
the provided context chunks. Rules:
- Every factual claim must be supported by a chunk; cite it.
- If the context is insufficient, set insufficient_context=true and say so.
- Never invent APIs, parameters, or version facts.
- If the user's version is given, scope the answer to it.
Return JSON matching the schema exactly."""


def make_gemini_nodes() -> tuple[GradeFn, GenerateFn, RewriteFn]:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["GEMINI_API_KEY"], base_url=GEMINI_BASE_URL)

    def _chat(system: str, user: str, json_mode: bool = False, max_tokens: int = 1200) -> str:
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        resp = client.chat.completions.create(
            model=GENERATE_MODEL, temperature=0.0, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            **kwargs,
        )
        return resp.choices[0].message.content or ""

    def grade_fn(question: str, chunks: list[Chunk]) -> bool:
        preview = "\n---\n".join(
            f"[{c.symbol or c.title}] {c.content[:400]}" for c in chunks[:6]
        )
        out = _chat(
            "You judge retrieval relevance. Reply with exactly YES or NO.",
            f"Question: {question}\n\nRetrieved:\n{preview}\n\n"
            "Do these chunks contain information that answers the question?",
            max_tokens=4,
        )
        return out.strip().upper().startswith("Y")

    def generate_fn(question: str, chunks: list[Chunk], version: Optional[str]) -> ExpertAnswer:
        context = "\n\n---\n\n".join(
            f"CHUNK {i}\nsymbol: {c.symbol}\npackage: {c.package} v{c.version}\n"
            f"url: {c.url}\n{c.embed_text[:2500]}"
            for i, c in enumerate(chunks)
        )
        schema_hint = json.dumps(ExpertAnswer.model_json_schema())[:2000]
        raw = _chat(
            _GENERATE_SYSTEM,
            f"Target version: {version or 'unspecified'}\n\nContext:\n{context}\n\n"
            f"Question: {question}\n\nJSON schema:\n{schema_hint}",
            json_mode=True,
        )
        try:
            return ExpertAnswer.model_validate_json(raw)
        except Exception:
            # Model returned malformed JSON: degrade to unstructured-but-honest
            return ExpertAnswer(answer=raw[:2000], insufficient_context=False)

    def rewrite_fn(question: str) -> str:
        return _chat(
            "Rewrite the developer question to be specific and retrieval-"
            "friendly for LangChain documentation search. Keep symbol names. "
            "Reply with only the rewritten question.",
            question, max_tokens=80,
        )

    return grade_fn, generate_fn, rewrite_fn
