# DevAtlas: LangChain Expert (v1)

Version-aware, source-grounded Q&A over LangChain: docs + source code,
anchor-version snapshots, hybrid retrieval (Qdrant), LangGraph agent with
citation enforcement, deterministic deprecation warnings, evaluated.

## Quickstart
    pip install -r requirements.txt
    cp .env.example .env                       # add your keys
    docker run -p 6333:6333 qdrant/qdrant      # real Qdrant (tests use :memory:)
    python -m devatlas.cli ingest              # clone anchors, parse, chunk (no keys needed)
    python -m devatlas.cli index               # embed + load Qdrant (keys needed)
    python -m devatlas.cli ask "How do I use initialize_agent?"
    python -m devatlas.cli eval --skip-ragas   # deterministic metrics only

## What was verified where
- Parser, chunker, BM25, Qdrant hybrid+RRF, version diffing, agent graph,
  deterministic eval: tested against REAL LangChain v0.1.0 source (tests/).
- Gemini/Voyage calls + Ragas: correct per current docs, need your keys.

## Layout
    devatlas/schema.py           data contracts (start reading here)
    devatlas/ingest/             repo + docs acquisition
    devatlas/parse/              tree-sitter parser, chunker, contextualizer
    devatlas/index/              embeddings (incl. BM25), Qdrant store
    devatlas/retrieve/           version detection, retrieval orchestration
    devatlas/agent/              LangGraph graph + Gemini nodes
    devatlas/version/            transition table, answer post-processing
    devatlas/eval_/              golden dataset, metrics, Ragas adapter
    data/golden/golden.jsonl     starter eval set (grow to 50+)
