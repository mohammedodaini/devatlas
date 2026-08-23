"""
devatlas.eval_.harness
======================
Three evaluation layers, cheapest first:

1. DETERMINISTIC metrics (free, run always, VERIFIED in tests):
   - citation_validity: every citation URL exists in the retrieved set
     (should be 1.0 by construction — the postprocess node enforces it;
     measuring it anyway catches regressions in that enforcement).
   - version_correctness: for deprecation-trap questions, does the answer
     (a) flag the deprecation and (b) name the correct alternative?
     Checked against the transition table, NOT an LLM opinion.
2. RAGAS LLM-judged metrics (faithfulness, context recall) — [REQUIRES
   API KEY]. Targets: faithfulness >= 0.8, context_recall >= 0.8.
3. Judge-bias control: if Gemini generates AND judges, scores inflate
   (self-preference). Periodically re-judge a sample with a different
   model family and report both numbers.

The golden dataset is JSONL, one GoldenQuestion per line, versioned in
git next to the code: the eval set IS part of the system.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from devatlas.schema import ExpertAnswer
from devatlas.version.difftable import TransitionTable


class GoldenQuestion(BaseModel):
    qid: str
    question: str
    category: str            # api_usage | version_migration | debugging | conceptual | deprecated_trap
    target_version: Optional[str] = None
    # ground truth for deterministic checks:
    must_mention: list[str] = Field(default_factory=list)     # substrings the answer must contain
    must_not_recommend: list[str] = Field(default_factory=list)  # deprecated symbols the answer must not endorse
    expected_alternative: Optional[str] = None                # for deprecated_trap
    reference_answer: str = ""                                # for Ragas context_recall
    reference_urls: list[str] = Field(default_factory=list)


def load_golden(path: Path) -> list[GoldenQuestion]:
    return [GoldenQuestion.model_validate_json(line)
            for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Deterministic metrics
# ---------------------------------------------------------------------------

@dataclass
class DeterministicScores:
    citation_validity: float
    version_correctness: float
    refusal_honesty: float     # refused iff it should have
    n: int


def score_deterministic(
    results: list[tuple[GoldenQuestion, ExpertAnswer, set[str]]],
    table: TransitionTable,
) -> DeterministicScores:
    """results: (question, answer, retrieved_urls) triples."""
    cit_ok = ver_ok = ref_ok = 0
    n_cit = n_ver = n_ref = 0

    for gq, ans, retrieved_urls in results:
        # citation validity
        if ans.citations:
            n_cit += 1
            if all(c.url in retrieved_urls for c in ans.citations):
                cit_ok += 1

        # version correctness (deprecation traps only)
        if gq.category == "deprecated_trap":
            n_ver += 1
            text = (ans.answer + " " + " ".join(ans.version_warnings)).lower()
            flagged = bool(ans.version_warnings) or "deprecat" in text
            alt_ok = (gq.expected_alternative is None
                      or gq.expected_alternative.lower() in text)
            clean = all(f"use {bad.lower()}" not in text
                        for bad in gq.must_not_recommend)
            if flagged and alt_ok and clean:
                ver_ok += 1

        # refusal honesty: out-of-scope questions must refuse; in-scope must not
        n_ref += 1
        should_refuse = gq.category == "out_of_scope"
        if ans.insufficient_context == should_refuse:
            ref_ok += 1

    return DeterministicScores(
        citation_validity=cit_ok / n_cit if n_cit else 1.0,
        version_correctness=ver_ok / n_ver if n_ver else 1.0,
        refusal_honesty=ref_ok / n_ref if n_ref else 1.0,
        n=len(results),
    )


# ---------------------------------------------------------------------------
# Ragas harness [REQUIRES GEMINI_API_KEY — interface per ragas >= 0.2]
# ---------------------------------------------------------------------------

def run_ragas(
    rows: list[dict],
    judge_model: str = "gemini-2.5-flash",
) -> dict:
    """rows: [{user_input, response, retrieved_contexts, reference}, ...]

    Kept as a thin adapter: Ragas' API moves; pinning the adapter in ONE
    place means an upgrade touches one function. Returns metric -> score.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    from ragas import EvaluationDataset, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextRecall

    judge = LangchainLLMWrapper(ChatGoogleGenerativeAI(model=judge_model, temperature=0))
    dataset = EvaluationDataset.from_list(rows)
    result = evaluate(dataset, metrics=[Faithfulness(), LLMContextRecall()], llm=judge)
    return dict(result._repr_dict) if hasattr(result, "_repr_dict") else dict(result)


THRESHOLDS = {"faithfulness": 0.8, "context_recall": 0.8,
              "citation_validity": 0.95, "version_correctness": 0.8}


def check_thresholds(scores: dict) -> list[str]:
    """Returns failure messages; empty list == CI green."""
    failures = []
    for metric, minimum in THRESHOLDS.items():
        if metric in scores and scores[metric] < minimum:
            failures.append(f"{metric}={scores[metric]:.3f} < {minimum}")
    return failures


def save_report(scores: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scores, indent=2))
