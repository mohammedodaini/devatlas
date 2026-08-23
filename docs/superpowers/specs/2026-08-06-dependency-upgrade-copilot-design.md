# DevAtlas → Dependency Upgrade Copilot — Design

**Date:** 2026-08-06
**Status:** Approved (brainstorming), pending spec review
**Goal:** Repurpose DevAtlas (version-aware, source-grounded LangChain Q&A) into a
portfolio-showcase web app that generates **migration guides** between two versions
of any Python library.

---

## 1. Motivation

DevAtlas's reusable core is not "LangChain Q&A" — it is a rare capability stack:

- **Version-aware** knowledge (anchor snapshots + a deterministic transition table).
- **Source-grounded** answers (tree-sitter parsing of real code, not just docs).
- **Citation enforcement** + **deterministic deprecation warnings**.
- **Evaluated** (golden set + Ragas).

The single strongest differentiator is the **`TransitionTable`**, which already computes
`ADDED / REMOVED / CHANGED / DEPRECATED` transitions between versions with old/new
signatures and deprecation alternatives — deterministically. A migration guide is
largely a *rendering and enrichment* of that table between two chosen versions.

Therefore this repurpose is mostly **generalize + wrap**, not rebuild.

### Success criteria (portfolio-showcase framing)

- A live web demo that renders a rich, cited migration guide instantly for a
  pre-baked example, and can generate one live for an arbitrary pasted repo.
- Preserves DevAtlas's trust story: every breaking change traces to a deterministic
  `SymbolTransition`; the LLM only illustrates, always with a citation.
- A portfolio-grade evaluation metric (breaking-change detection precision/recall).

Commercial viability is explicitly out of scope.

---

## 2. Scope

**In scope (v1):**
- Python-only libraries (reuse existing `tree-sitter-python` parser).
- Output: a **general migration guide** between two versions (`v_from → v_to`) —
  ranked breaking changes, each with a cited before/after snippet and an upgrade checklist.
- Web app: FastAPI backend + vanilla HTML/JS single-page UI.
- Demo strategy: **pre-baked + live** — curated instant guides plus "try your own repo".

**Out of scope (v1):**
- JS/TS or other languages (architected to add later, not built).
- Personalized "what breaks in *my* code" usage-site analysis (designed for as a
  future v2 layer, not built now).
- Auth, persistence beyond process memory, multi-tenant concerns.

---

## 3. Architecture

Three moves on top of the existing pipeline:

1. **Generalize the pipeline** from hardcoded LangChain `ANCHORS` to a per-request
   `Target(repo_url, tags)`. The existing `ingest` (clone tag → `Chunk`s + `SymbolRecord`s)
   and `TransitionTable.build()` (diff versions) are reused with the target injected.
2. **Add a migration-guide synthesizer.** Given the transitions between `v_from → v_to`,
   rank by severity and, per breaking change, use the *existing retriever + an LLM node*
   to produce a cited before/after snippet + explanation. The deterministic table is the
   source of truth; the LLM only illustrates.
3. **Wrap in FastAPI + a small web UI.** Because clone + parse + LLM synthesis takes
   minutes, the API is **job-based**.

### Data flow

```
submit(repo_url, v_from, v_to)
  → validate (tags exist, repo allowed, size cap)
  → clone both tags               [ingest.repo, generalized]
  → parse both                    [parse/*]  → Chunks + SymbolRecords
  → TransitionTable.build({v_from: [...], v_to: [...]})   (single version pair)
  → rank breaking changes by severity          [migrate/guide.py]
  → per change: retrieve source context + LLM before/after + Citation
                                                [migrate/synthesize.py]
  → assemble MigrationGuide (Pydantic)
  → serve JSON → UI renders
```

---

## 4. Components

New modules, each with one clear purpose. All speak the existing `schema.py` contracts.

| Module | Purpose | Depends on |
|---|---|---|
| `devatlas/migrate/target.py` | `Target(repo_url, tags)` contract; replaces hardcoded `ANCHORS`. | schema |
| `devatlas/migrate/guide.py` | `MigrationGuide` + `BreakingChange` Pydantic models; severity ranking; renders `TransitionTable` → guide skeleton (deterministic, no LLM). | schema, version/difftable |
| `devatlas/migrate/synthesize.py` | Per-change enrichment: retrieve context, LLM before/after snippet, attach `Citation`. Marks "no example available" when retrieval is insufficient. | retrieve, agent |
| `devatlas/api/app.py` | FastAPI app: `POST /guides`, `GET /guides/{id}`, `GET /healthz`. | migrate/* |
| `devatlas/api/jobs.py` | In-memory async job store + background runner with explicit state machine. | — |
| `web/index.html` (+ inline JS/CSS) | Single-page UI: form → live progress → rendered guide. | — |

Reused as-is: `Citation`, `SymbolTransition`, `DeprecationInfo`, `Chunk`, `SymbolRecord`,
the retriever, the agent's Gemini node plumbing.

### New Pydantic models (sketch)

```python
class BreakingChange(BaseModel):
    fq_name: str
    kind: TransitionKind           # REMOVED | CHANGED | DEPRECATED
    severity: int                  # ranked: REMOVED > CHANGED > DEPRECATED (tunable)
    old_signature: str | None = None
    new_signature: str | None = None
    alternative: str | None = None          # from DeprecationInfo
    before_snippet: str | None = None        # LLM, cited
    after_snippet: str | None = None         # LLM, cited
    explanation: str = ""
    citations: list[Citation] = []
    example_available: bool = True           # False → grounded example not found

class MigrationGuide(BaseModel):
    package: str
    from_version: str
    to_version: str
    breaking_changes: list[BreakingChange]
    checklist: list[str]
    stats: dict                              # counts by kind, detection metrics
```

---

## 5. API design (job-based)

- `POST /guides` → body `{repo_url, from_version, to_version}` → `201 {job_id}`.
  Pre-baked targets short-circuit to a cached `MigrationGuide` (still returned via a
  completed job for a uniform client contract).
- `GET /guides/{job_id}` → `{state, progress, guide?, error?}`.
- `GET /healthz` → liveness.

**Job state machine:** `pending → cloning → parsing → diffing → synthesizing → done | failed`.
Each state carries a human-readable `progress` string for the UI.

---

## 6. Error handling & trust

- **Deterministic table is authoritative.** Every `BreakingChange` originates from a
  `SymbolTransition` computed from source — never an LLM claim. The LLM writes only the
  illustrative snippet, each carrying a `Citation` (GitHub blob URL + verbatim quote).
- **Insufficient-context honesty.** If retrieval can't ground a snippet, the change still
  appears (from the table) with `example_available = False` — mirroring the existing
  `insufficient_context` flag. No hallucinated examples.
- **Explicit job failure states** with human-readable errors (tag not found, private/blocked
  repo, repo too large, timeout). No silent failures.
- **Guardrails on live input:** validate tag existence *before* the expensive clone; repo
  size cap; per-job timeout; repo-URL allow-pattern.

---

## 7. Testing & evaluation

- **Unit tests** (no network, no keys), mirroring how `difftable` is tested today:
  - `guide.py` severity ranking and table→guide rendering against a **synthetic
    two-version symbol table**.
  - `synthesize.py` "no example available" path when retrieval returns nothing.
- **Golden guide set:** `data/golden_guides/` with a known **Pydantic v1→v2** expected set
  of breaking changes. Assert all deterministic changes are caught → report
  **precision/recall on breaking-change detection** (the portfolio-grade metric).
  Record shape (JSONL, one per line):
  `{package, from_version, to_version, expected_breaking_changes: [{fq_name, kind}]}`.
- **API tests** with FastAPI `TestClient`: full job lifecycle + each error state.

---

## 8. Demo (end to end)

1. Open the app → click **"Pydantic v1 → v2"** (pre-baked, instant).
2. Guide renders: ranked breaking changes, each with a cited source diff + before/after
   snippet, plus deprecation warnings and an upgrade checklist.
3. Paste **any** Python repo + two tags → watch the job progress live → same rich guide.
4. Close with eval numbers: "detects N/N known breaking changes; every claim cited."

---

## 9. Build order (for the implementation plan)

1. `migrate/target.py` + generalize `ingest.repo` to accept a `Target`.
2. `migrate/guide.py` (deterministic guide from `TransitionTable`) + unit tests.
3. `migrate/synthesize.py` (LLM enrichment + citations) + unit tests.
4. `api/jobs.py` + `api/app.py` + `TestClient` tests.
5. `web/index.html` UI.
6. Pre-baked Pydantic v1→v2 guide + `data/golden_guides/` + detection metric.
7. README/demo script update.

---

## 10. Open items / notes

- Git is not yet initialized in this project; the spec is written to disk. Initializing
  git and committing is recommended before implementation (offered to the user).
- `requirements.txt` will gain `fastapi` and an ASGI server (`uvicorn`); no other new
  heavyweight deps (vanilla UI, no React build).
