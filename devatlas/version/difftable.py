"""
devatlas.version.difftable
==========================
The differentiator. Diffs symbol tables across anchor versions into a
transition table, then post-processes agent answers against it.

WHY MECHANICAL DIFFING BEATS LLM INFERENCE:
"When was initialize_agent removed?" has exactly one correct answer, and
it is computable: the symbol exists in the 0.1.0 table and not in the
0.2.0 table -> removed in 0.2.0. An LLM guessing from training data gets
this wrong constantly (it's a long-tail fact that changed over time — the
worst case for parametric memory). We compute it once and attach it to
answers deterministically.

VERIFIED: the diff logic below is exercised in tests against symbol tables
built from real LangChain 0.1.0 source plus synthetic later versions.
"""

from __future__ import annotations

import re
from collections import defaultdict

from devatlas.schema import (
    DeprecationInfo,
    ExpertAnswer,
    SymbolRecord,
    SymbolTransition,
    TransitionKind,
)


def _version_key(v: str) -> tuple:
    """'0.1.0' -> (0,1,0) for correct ordering ('0.10.0' > '0.2.0')."""
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3]) or (0,)


class TransitionTable:
    def __init__(self) -> None:
        self.transitions: list[SymbolTransition] = []
        self._by_symbol: dict[str, list[SymbolTransition]] = defaultdict(list)
        # latest known deprecation info per symbol (from ANY version's table)
        self._deprecations: dict[str, tuple[str, DeprecationInfo]] = {}

    # -- construction ------------------------------------------------------

    def build(self, tables: dict[str, list[SymbolRecord]]) -> "TransitionTable":
        """tables: {version -> symbol records}. Diffs CONSECUTIVE anchors.

        Per consecutive pair (older, newer):
          in older, not newer  -> REMOVED at newer
          in newer, not older  -> ADDED at newer
          in both, signature differs -> CHANGED
        Independently, any record carrying @deprecated metadata registers a
        DEPRECATED transition (the decorator itself says since/removal).
        """
        versions = sorted(tables.keys(), key=_version_key)

        for version in versions:
            for rec in tables[version]:
                if rec.deprecated:
                    self._deprecations[rec.fq_name] = (version, rec.deprecated)

        for older_v, newer_v in zip(versions, versions[1:]):
            older = {r.fq_name: r for r in tables[older_v]}
            newer = {r.fq_name: r for r in tables[newer_v]}
            for name in older.keys() - newer.keys():
                self._add(SymbolTransition(
                    fq_name=name, package=older[name].package,
                    kind=TransitionKind.REMOVED,
                    from_version=older_v, to_version=newer_v,
                    old_signature=older[name].signature,
                    deprecation=older[name].deprecated,
                ))
            for name in newer.keys() - older.keys():
                self._add(SymbolTransition(
                    fq_name=name, package=newer[name].package,
                    kind=TransitionKind.ADDED,
                    from_version=older_v, to_version=newer_v,
                    new_signature=newer[name].signature,
                ))
            for name in older.keys() & newer.keys():
                if _norm_sig(older[name].signature) != _norm_sig(newer[name].signature):
                    self._add(SymbolTransition(
                        fq_name=name, package=newer[name].package,
                        kind=TransitionKind.CHANGED,
                        from_version=older_v, to_version=newer_v,
                        old_signature=older[name].signature,
                        new_signature=newer[name].signature,
                    ))

        for name, (version, dep) in self._deprecations.items():
            self._add(SymbolTransition(
                fq_name=name,
                package=name.split(".")[0],
                kind=TransitionKind.DEPRECATED,
                from_version=dep.since or version,
                to_version=dep.removal,
                deprecation=dep,
            ))
        return self

    def _add(self, t: SymbolTransition) -> None:
        self.transitions.append(t)
        self._by_symbol[t.fq_name].append(t)
        # also index by short name so "initialize_agent" matches
        self._by_symbol[t.fq_name.split(".")[-1]].append(t)

    # -- lookups -----------------------------------------------------------

    def lookup(self, symbol: str) -> list[SymbolTransition]:
        return self._by_symbol.get(symbol, [])

    def warnings_for_text(self, text: str, target_version: str | None) -> list[str]:
        """Scan an answer (or a question) for symbols with known transitions
        and produce human-readable warnings. This runs as answer
        POST-PROCESSING: whatever the LLM wrote, the deterministic layer
        gets the last word on deprecation facts.
        """
        warnings: list[str] = []
        seen: set[str] = set()
        candidates = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", text))
        for name in candidates:
            for t in self._by_symbol.get(name, []):
                key = (t.fq_name, t.kind.value)
                if key in seen:
                    continue
                seen.add(key)
                warnings.append(self._render(t, target_version))
        return [w for w in warnings if w]

    def _render(self, t: SymbolTransition, target_version: str | None) -> str:
        if t.kind == TransitionKind.DEPRECATED:
            msg = f"`{t.fq_name}` is deprecated since {t.from_version}"
            if t.to_version:
                msg += f" (removal: {t.to_version})"
            if t.deprecation and t.deprecation.alternative:
                msg += f". Alternative: {t.deprecation.alternative[:160]}"
            return msg + "."
        if t.kind == TransitionKind.REMOVED:
            return (f"`{t.fq_name}` was removed between {t.from_version} and "
                    f"{t.to_version}; it no longer exists in current versions.")
        if t.kind == TransitionKind.CHANGED:
            return (f"`{t.fq_name}` changed signature between {t.from_version} "
                    f"and {t.to_version} — verify arguments against your version.")
        if t.kind == TransitionKind.ADDED and target_version:
            # Only warn about ADDED when the user targets an OLDER version:
            # "you're on 0.1 but create_agent only exists from 1.0".
            if _version_key(target_version) < _version_key(t.to_version or "999"):
                return (f"`{t.fq_name}` was introduced in {t.to_version} — "
                        f"not available on your version {target_version}.")
        return ""


def _norm_sig(sig: str) -> str:
    return " ".join(sig.split())


def apply_version_warnings(
    answer: ExpertAnswer, table: TransitionTable, question: str
) -> ExpertAnswer:
    """Check BOTH the question and the answer text: the question catches
    'how do I use initialize_agent' even when the model's answer avoids
    naming the deprecated symbol."""
    combined = f"{question}\n{answer.answer}"
    answer.version_warnings = table.warnings_for_text(
        combined, answer.target_version
    )
    return answer
