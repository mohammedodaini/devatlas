"""
devatlas CLI
============
    python -m devatlas.cli ingest     # clone anchors, parse, chunk, save JSONL
    python -m devatlas.cli index      # contextualize, embed, load Qdrant  [API keys]
    python -m devatlas.cli ask "..."  # one question through the agent     [API keys]
    python -m devatlas.cli eval       # deterministic + Ragas evaluation   [API keys]

Stages communicate through data/ artifacts (JSONL), so each stage is
independently re-runnable — fix the chunker, re-run `index` without
re-cloning; fix a prompt, re-run `ask` without re-indexing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DATA = Path("data")


def cmd_ingest(args: argparse.Namespace) -> None:
    from devatlas.ingest.repo import ANCHORS, clone_anchor, ingest_snapshot, save_jsonl

    all_chunks, all_symbols = [], []
    tags = args.tags.split(",") if args.tags else list(ANCHORS)
    for tag in tags:
        print(f"[ingest] cloning {tag} ...")
        snap = clone_anchor(tag, DATA / "repos", ANCHORS[tag])
        n_c = n_s = 0
        for chunks, symbols in ingest_snapshot(snap):
            all_chunks += chunks
            all_symbols += symbols
            n_c += len(chunks); n_s += len(symbols)
        print(f"[ingest] {tag}: {n_c} chunks, {n_s} symbols")
    save_jsonl(all_chunks, DATA / "chunks.jsonl")
    save_jsonl(all_symbols, DATA / "symbols.jsonl")
    print(f"[ingest] total: {len(all_chunks)} chunks -> data/chunks.jsonl")


def cmd_index(args: argparse.Namespace) -> None:
    from devatlas.index.embeddings import BM25Encoder, GeminiEncoder, VoyageCodeEncoder
    from devatlas.index.qdrant_store import QdrantStore
    from devatlas.parse.contextualize import contextualize_chunks
    from devatlas.schema import Chunk

    chunks = [Chunk.model_validate_json(l)
              for l in (DATA / "chunks.jsonl").read_text().splitlines()]
    print(f"[index] contextualizing {len(chunks)} chunks (template blurbs) ...")
    contextualize_chunks(chunks)   # add make_gemini_caller() for LLM doc blurbs

    print("[index] fitting BM25 ...")
    bm25 = BM25Encoder(); bm25.fit([c.embed_text for c in chunks])
    (DATA / "bm25_vocab.json").write_text(json.dumps({
        "doc_freq": dict(bm25.doc_freq), "n_docs": bm25.n_docs,
        "avg_len": bm25.avg_len, "vocab": bm25.vocab,
    }))

    print("[index] embedding + upserting (this is the paid step) ...")
    store = QdrantStore(url=args.qdrant_url)
    store.create_collection(recreate=args.recreate)
    n = store.upsert_chunks(chunks, GeminiEncoder(), VoyageCodeEncoder(), bm25)
    print(f"[index] upserted {n} points")


def _load_runtime(qdrant_url: str):
    from devatlas.index.embeddings import BM25Encoder, GeminiEncoder, VoyageCodeEncoder
    from devatlas.index.qdrant_store import QdrantStore
    from devatlas.retrieve.pipeline import Retriever
    from devatlas.schema import SymbolRecord
    from devatlas.version.difftable import TransitionTable

    bm25 = BM25Encoder()
    saved = json.loads((DATA / "bm25_vocab.json").read_text())
    bm25.doc_freq.update(saved["doc_freq"])
    bm25.n_docs, bm25.avg_len, bm25.vocab = saved["n_docs"], saved["avg_len"], saved["vocab"]

    retriever = Retriever(QdrantStore(url=qdrant_url), GeminiEncoder(), VoyageCodeEncoder(), bm25)

    symbols = [SymbolRecord.model_validate_json(l)
               for l in (DATA / "symbols.jsonl").read_text().splitlines()]
    by_version: dict[str, list] = {}
    for s in symbols:
        by_version.setdefault(s.version, []).append(s)
    table = TransitionTable().build(by_version)
    return retriever, table


def cmd_ask(args: argparse.Namespace) -> None:
    from devatlas.agent.graph import build_graph, make_gemini_nodes

    retriever, table = _load_runtime(args.qdrant_url)
    graph = build_graph(retriever, table, *make_gemini_nodes())
    out = graph.invoke({"question": args.question})
    a = out["answer"]

    print("\n" + "=" * 70)
    for w in a.version_warnings:
        print(f"⚠  {w}")
    if a.version_warnings:
        print("-" * 70)
    print(a.answer)
    if a.citations:
        print("\nSources:")
        for c in a.citations:
            print(f"  [{c.package} v{c.version}] {c.url}")
    if a.insufficient_context:
        print("\n(context was insufficient — no answer was invented)")


def cmd_eval(args: argparse.Namespace) -> None:
    from devatlas.agent.graph import build_graph, make_gemini_nodes
    from devatlas.eval_.harness import (check_thresholds, load_golden,
                                        run_ragas, save_report,
                                        score_deterministic)

    retriever, table = _load_runtime(args.qdrant_url)
    graph = build_graph(retriever, table, *make_gemini_nodes())
    golden = load_golden(DATA / "golden" / "golden.jsonl")

    results, ragas_rows = [], []
    for gq in golden:
        out = graph.invoke({"question": gq.question})
        ans, retrieved = out["answer"], out.get("retrieved", [])
        results.append((gq, ans, {c.url for c in retrieved}))
        if gq.reference_answer:
            ragas_rows.append({
                "user_input": gq.question,
                "response": ans.answer,
                "retrieved_contexts": [c.embed_text for c in retrieved],
                "reference": gq.reference_answer,
            })
        print(f"  [{gq.qid}] answered ({len(retrieved)} chunks)")

    det = score_deterministic(results, table)
    scores = {"citation_validity": det.citation_validity,
              "version_correctness": det.version_correctness,
              "refusal_honesty": det.refusal_honesty}
    if not args.skip_ragas:
        scores.update(run_ragas(ragas_rows))

    save_report(scores, DATA / "eval_report.json")
    print("\nScores:", json.dumps(scores, indent=2))
    failures = check_thresholds(scores)
    if failures:
        print("THRESHOLD FAILURES:", *failures, sep="\n  ")
        sys.exit(1)
    print("All thresholds passed.")


def main() -> None:
    ap = argparse.ArgumentParser(prog="devatlas")
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest"); p.add_argument("--tags", default="")
    p.set_defaults(fn=cmd_ingest)
    p = sub.add_parser("index"); p.add_argument("--recreate", action="store_true")
    p.set_defaults(fn=cmd_index)
    p = sub.add_parser("ask"); p.add_argument("question")
    p.set_defaults(fn=cmd_ask)
    p = sub.add_parser("eval"); p.add_argument("--skip-ragas", action="store_true")
    p.set_defaults(fn=cmd_eval)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
