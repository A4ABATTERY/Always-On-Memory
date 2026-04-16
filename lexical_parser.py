"""
Lexical Parser Module — Extracts named code identifiers for the Lexical Symbol Index (LSI).

Scope: functions, classes, and UPPER_CASE constants only.
Hardcoded strings, SQL queries, and log messages are explicitly excluded.

Language support:
  - Python: ast module (high-fidelity, SyntaxError-safe)
  - JS/TS/JSX/TSX: regex (top-level declarations only)
  - Go: regex (top-level func/type declarations)
  - Rust: regex (top-level fn/struct/enum declarations)
  - All others: returns empty list (graceful no-op)
"""

import ast
import re
import logging
from typing import List, Dict, Any

log = logging.getLogger("memory-agent.lexical-parser")


def extract_symbols(text: str, file_ext: str) -> List[Dict[str, Any]]:
    """
    Extract named identifiers from source code text.

    Returns a list of dicts with keys:
      name (str), type (str), line_no (int | None), signature (str | None)
    """
    ext = file_ext.lower()
    if ext == ".py":
        return _extract_python(text)
    elif ext in (".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs"):
        return _extract_js_ts(text)
    elif ext == ".go":
        return _extract_go(text)
    elif ext == ".rs":
        return _extract_rust(text)
    return []


# ─── Python ───────────────────────────────────────────────────────────────────

def _extract_python(text: str) -> List[Dict[str, Any]]:
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        log.debug(f"lexical_parser: SyntaxError in Python file — skipping symbol extraction ({e})")
        return []

    symbols = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            try:
                sig = ast.unparse(node.args)
            except Exception:
                sig = None
            symbols.append({
                "name": node.name,
                "type": "function",
                "line_no": node.lineno,
                "signature": sig,
            })
        elif isinstance(node, ast.ClassDef):
            symbols.append({
                "name": node.name,
                "type": "class",
                "line_no": node.lineno,
                "signature": None,
            })
        elif isinstance(node, ast.Assign):
            # Only UPPER_CASE module-level constants (col_offset == 0)
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id.isupper()
                    and target.col_offset == 0
                ):
                    symbols.append({
                        "name": target.id,
                        "type": "constant",
                        "line_no": node.lineno,
                        "signature": None,
                    })
    return symbols


# ─── JavaScript / TypeScript ──────────────────────────────────────────────────

# Patterns target top-level declarations only.
# Known limitations: arrow functions assigned to non-const, decorated generics.
_JS_TS_PATTERNS = [
    # function foo(...) / async function foo(...)
    (re.compile(r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)", re.MULTILINE), "function"),
    # class Foo
    (re.compile(r"^(?:export\s+)?(?:default\s+)?class\s+(\w+)", re.MULTILINE), "class"),
    # const foo = (...) => / const foo = async (...) =>
    (re.compile(r"^(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s*)?\(", re.MULTILINE), "function"),
    # const FOO = value  (UPPER_CASE only)
    (re.compile(r"^(?:export\s+)?const\s+([A-Z][A-Z0-9_]+)\s*=", re.MULTILINE), "constant"),
]

def _extract_js_ts(text: str) -> List[Dict[str, Any]]:
    symbols = []
    lines = text.splitlines()
    for pattern, sym_type in _JS_TS_PATTERNS:
        for m in pattern.finditer(text):
            line_no = text[:m.start()].count("\n") + 1
            symbols.append({
                "name": m.group(1),
                "type": sym_type,
                "line_no": line_no,
                "signature": None,
            })
    # Deduplicate by (name, line_no)
    seen = set()
    deduped = []
    for s in symbols:
        key = (s["name"], s["line_no"])
        if key not in seen:
            seen.add(key)
            deduped.append(s)
    return deduped


# ─── Go ───────────────────────────────────────────────────────────────────────

_GO_PATTERNS = [
    (re.compile(r"^func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(", re.MULTILINE), "function"),
    (re.compile(r"^type\s+(\w+)\s+struct", re.MULTILINE), "class"),
    (re.compile(r"^type\s+(\w+)\s+interface", re.MULTILINE), "class"),
    (re.compile(r"^const\s+([A-Z][A-Z0-9_]+)\b", re.MULTILINE), "constant"),
]

def _extract_go(text: str) -> List[Dict[str, Any]]:
    symbols = []
    for pattern, sym_type in _GO_PATTERNS:
        for m in pattern.finditer(text):
            line_no = text[:m.start()].count("\n") + 1
            symbols.append({"name": m.group(1), "type": sym_type, "line_no": line_no, "signature": None})
    return symbols


# ─── Rust ─────────────────────────────────────────────────────────────────────

_RUST_PATTERNS = [
    (re.compile(r"^(?:pub\s+)?(?:async\s+)?fn\s+(\w+)", re.MULTILINE), "function"),
    (re.compile(r"^(?:pub\s+)?struct\s+(\w+)", re.MULTILINE), "class"),
    (re.compile(r"^(?:pub\s+)?enum\s+(\w+)", re.MULTILINE), "class"),
    (re.compile(r"^(?:pub\s+)?const\s+([A-Z][A-Z0-9_]+)\b", re.MULTILINE), "constant"),
]

def _extract_rust(text: str) -> List[Dict[str, Any]]:
    symbols = []
    for pattern, sym_type in _RUST_PATTERNS:
        for m in pattern.finditer(text):
            line_no = text[:m.start()].count("\n") + 1
            symbols.append({"name": m.group(1), "type": sym_type, "line_no": line_no, "signature": None})
    return symbols
