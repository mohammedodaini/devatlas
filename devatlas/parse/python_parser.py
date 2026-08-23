"""
devatlas.parse.python_parser
============================
Tree-sitter based extraction of functions, classes, methods, docstrings,
and — critically — @deprecated decorator metadata from Python source.

WHY TREE-SITTER (not Python's `ast` module):
1. `ast.parse` must run under an interpreter compatible with the target
   code. LangChain 0.1.0 predates some 3.12 syntax handling and, more
   importantly, we may later index non-Python libraries; tree-sitter gives
   one uniform approach with 100+ grammars.
2. Tree-sitter is error-tolerant: a file with one syntax error still yields
   a mostly-correct tree, so one bad file doesn't kill an ingestion run.
3. It returns exact byte/line ranges — which become our citation line
   numbers for free.

WHY MINE @deprecated MECHANICALLY:
langchain_core's decorator carries since/removal/alternative as literal
keyword arguments. Parsing them from the AST is deterministic and correct;
asking an LLM "is this deprecated?" hallucinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

from devatlas.schema import DeprecationInfo, SymbolRecord

PY_LANGUAGE = Language(tspython.language())


@dataclass
class ParsedDefinition:
    """Intermediate representation of one function/class/method."""
    kind: str                     # "function" | "class" | "method"
    name: str
    qualified_name: str           # module-relative, e.g. "AgentExecutor.invoke"
    signature: str
    docstring: str
    decorators: list[str]
    deprecation: Optional[DeprecationInfo]
    start_line: int               # 1-based, inclusive
    end_line: int
    source: str                   # full source text of the definition
    children: list["ParsedDefinition"] = field(default_factory=list)


class PythonParser:
    def __init__(self) -> None:
        self.parser = Parser(PY_LANGUAGE)

    # -- public API ---------------------------------------------------------

    def parse_file(self, path: Path) -> list[ParsedDefinition]:
        source = path.read_bytes()
        tree = self.parser.parse(source)
        return list(self._walk_definitions(tree.root_node, source, parent_prefix=""))

    def module_name(self, repo_root: Path, file_path: Path) -> str:
        """'libs/core/langchain_core/agents.py' -> 'langchain_core.agents'.

        Heuristic: strip everything up to the first path component that looks
        like a package root (contains __init__.py at that level is the robust
        check; here we take the path after the known libs/<pkg-dir>/ layout).
        """
        rel = file_path.relative_to(repo_root)
        parts = list(rel.with_suffix("").parts)
        # Walk down from repo_root; the module path starts at the SHALLOWEST
        # directory that contains an __init__.py (the package root, e.g.
        # 'langchain' in libs/langchain/langchain/agents/initialize.py).
        current = repo_root
        start = len(parts) - 1  # fallback: bare filename
        for i, part in enumerate(parts[:-1]):
            current = current / part
            if (current / "__init__.py").exists():
                start = i
                break
        parts = parts[start:]
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts)

    # -- internals ----------------------------------------------------------

    def _walk_definitions(
        self, node: Node, source: bytes, parent_prefix: str
    ) -> Iterator[ParsedDefinition]:
        """Yield top-level defs; methods are attached as children of classes.

        Tree-sitter wraps a decorated def in a `decorated_definition` node
        whose last child is the actual function/class node — we must handle
        both shapes or we silently drop every decorated (= every deprecated)
        definition, which is exactly the data we care about most.
        """
        for child in node.children:
            target, decorators = child, []
            if child.type == "decorated_definition":
                decorators = [
                    self._text(d, source)
                    for d in child.children
                    if d.type == "decorator"
                ]
                target = child.children[-1]

            if target.type in ("function_definition", "class_definition"):
                yield self._build_definition(
                    target, decorators, source, parent_prefix,
                    outer_start=child.start_point[0] + 1,
                )

    def _build_definition(
        self,
        node: Node,
        decorators: list[str],
        source: bytes,
        parent_prefix: str,
        outer_start: int,
    ) -> ParsedDefinition:
        name_node = node.child_by_field_name("name")
        name = self._text(name_node, source) if name_node else "<anonymous>"
        qualified = f"{parent_prefix}{name}"
        kind = "class" if node.type == "class_definition" else (
            "method" if parent_prefix else "function"
        )

        # Signature = the def/class header line(s) up to the ':' before the body
        body = node.child_by_field_name("body")
        header_end = body.start_byte if body else node.end_byte
        signature = source[node.start_byte:header_end].decode("utf-8", "replace")
        signature = " ".join(signature.split())  # normalize whitespace
        if signature.endswith(":"):
            signature = signature[:-1]

        docstring = self._extract_docstring(body, source) if body else ""
        deprecation = self._parse_deprecated(decorators)

        children: list[ParsedDefinition] = []
        if kind == "class" and body is not None:
            children = list(
                self._walk_definitions(body, source, parent_prefix=f"{qualified}.")
            )

        return ParsedDefinition(
            kind=kind,
            name=name,
            qualified_name=qualified,
            signature=signature,
            docstring=docstring,
            decorators=decorators,
            deprecation=deprecation,
            start_line=outer_start,                 # include decorators in the span
            end_line=node.end_point[0] + 1,
            source=source[node.start_byte:node.end_byte].decode("utf-8", "replace"),
            children=children,
        )

    def _extract_docstring(self, body: Node, source: bytes) -> str:
        """The docstring is the first statement of the body iff it is a bare
        string expression. Handles both plain and concatenated strings."""
        for stmt in body.children:
            if stmt.type == "expression_statement":
                expr = stmt.children[0] if stmt.children else None
                if expr is not None and expr.type in ("string", "concatenated_string"):
                    raw = self._text(expr, source)
                    return _strip_string_literal(raw)
                return ""
            if stmt.type == "comment":
                continue
            return ""
        return ""

    def _parse_deprecated(self, decorators: list[str]) -> Optional[DeprecationInfo]:
        """Extract since/removal/alternative/message from a @deprecated(...)
        decorator string.

        We re-parse the decorator text with tree-sitter itself rather than
        regexing: the arguments are proper Python literals, and a tiny parse
        of `@deprecated(since="0.1.0", ...)` is exact where regex on nested
        quotes/parens is fragile.
        """
        for dec in decorators:
            head = dec.lstrip("@").split("(", 1)[0].strip()
            if head.split(".")[-1] != "deprecated":
                continue
            info = DeprecationInfo()
            tree = self.parser.parse(dec.lstrip("@").encode())
            src = dec.lstrip("@").encode()
            kwargs, positionals = _collect_call_args(tree.root_node, src)
            # LangChain passes `since` POSITIONALLY: @deprecated("0.1.0", ...).
            # The first positional string argument is the since-version.
            if positionals and "since" not in kwargs:
                info.since = positionals[0]
            for key in ("since", "removal", "alternative", "alternative_import", "message"):
                if key in kwargs:
                    setattr(info, key, kwargs[key])
            return info
        return None

    @staticmethod
    def _text(node: Node, source: bytes) -> str:
        return source[node.start_byte:node.end_byte].decode("utf-8", "replace")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _strip_string_literal(raw: str, keep_ws: bool = False) -> str:
    """'\"\"\"docstring\"\"\"' -> 'docstring' (handles prefixes and quote styles).
    keep_ws=True preserves inner whitespace (needed when joining adjacent
    string literals, where trailing spaces are significant)."""
    s = raw.strip()
    for prefix in ("r", "b", "u", "f", "rb", "br"):
        if s.lower().startswith(prefix) and len(s) > len(prefix) and s[len(prefix)] in "\"'":
            s = s[len(prefix):]
            break
    for quote in ('"""', "'''", '"', "'"):
        if s.startswith(quote) and s.endswith(quote) and len(s) >= 2 * len(quote):
            inner = s[len(quote):-len(quote)]
            return inner if keep_ws else inner.strip()
    return s


def _string_value(node: Node, source: bytes) -> Optional[str]:
    """Resolve a value node to a plain string if it is string-like.

    Handles the three shapes LangChain actually uses:
      "0.1.0"                                  -> string
      ("part one " "part two")                 -> parenthesized concatenation
      "a" "b"                                  -> concatenated_string
    Implicit adjacent-literal concatenation is joined; anything non-string
    returns None so callers can fall back to raw text.
    """
    if node.type == "parenthesized_expression":
        inner = [c for c in node.children if c.type not in ("(", ")")]
        if len(inner) == 1:
            return _string_value(inner[0], source)
        return None
    if node.type == "concatenated_string":
        parts = [
            _strip_string_literal(source[c.start_byte:c.end_byte].decode(), keep_ws=True)
            for c in node.children
            if c.type == "string"
        ]
        return "".join(parts)
    if node.type == "string":
        return _strip_string_literal(source[node.start_byte:node.end_byte].decode())
    return None


def _collect_call_args(root: Node, source: bytes) -> tuple[dict[str, str], list[str]]:
    """Walk a parsed `deprecated(...)` expression.

    Returns (keyword_args, positional_string_args). Only the OUTERMOST call's
    arguments are collected — we stop descending once inside an argument value
    so nested calls don't pollute the result.
    """
    kwargs: dict[str, str] = {}
    positionals: list[str] = []

    def find_call(n: Node) -> Optional[Node]:
        if n.type == "call":
            return n
        for c in n.children:
            found = find_call(c)
            if found is not None:
                return found
        return None

    call = find_call(root)
    if call is None:
        return kwargs, positionals
    args = call.child_by_field_name("arguments")
    if args is None:
        return kwargs, positionals

    for arg in args.children:
        if arg.type == "keyword_argument":
            name_n = arg.child_by_field_name("name")
            value_n = arg.child_by_field_name("value")
            if name_n is not None and value_n is not None:
                key = source[name_n.start_byte:name_n.end_byte].decode()
                val = _string_value(value_n, source)
                if val is None:  # non-string (True, identifiers): keep raw text
                    val = source[value_n.start_byte:value_n.end_byte].decode()
                kwargs[key] = val
        elif arg.type not in ("(", ")", ","):
            val = _string_value(arg, source)
            if val is not None:
                positionals.append(val)

    return kwargs, positionals


# ---------------------------------------------------------------------------
# Symbol table construction (feeds Part 5 diffing)
# ---------------------------------------------------------------------------

def build_symbol_records(
    defs: list[ParsedDefinition],
    module: str,
    package: str,
    version: str,
    path: str,
) -> list[SymbolRecord]:
    """Flatten parsed definitions (incl. methods) into SymbolRecords.

    Private symbols (leading underscore) are skipped: users ask about the
    public API, and diffing private churn produces noise, not signal.
    """
    records: list[SymbolRecord] = []

    def visit(d: ParsedDefinition) -> None:
        if not d.name.startswith("_"):
            records.append(
                SymbolRecord(
                    fq_name=f"{module}.{d.qualified_name}",
                    package=package,
                    version=version,
                    kind=d.kind,
                    signature=d.signature,
                    docstring_first_line=d.docstring.split("\n")[0][:200],
                    path=path,
                    start_line=d.start_line,
                    deprecated=d.deprecation,
                )
            )
        for c in d.children:
            visit(c)

    for d in defs:
        visit(d)
    return records
