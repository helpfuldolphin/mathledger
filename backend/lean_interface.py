"""
Helpers for sanitizing MathLedger Lean job statements.

This module centralizes all normalization logic so that job emitters,
workers, and tests agree on the exact ASCII/Unicode forms we use when
talking to Lean, the database, and our deterministic hashers.

Reference: MathLedger Whitepaper §5.2 (Lean Interface Normalization).
"""

from __future__ import annotations

from dataclasses import dataclass
import html
import re
from typing import Optional

from normalization.canon import normalize, normalize_pretty


_LATEX_PATTERN_MAP: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (
        re.compile(pattern, flags=re.IGNORECASE),
        replacement,
    )
    for pattern, replacement in [
        (r"\\(?:to|rightarrow|Rightarrow|implies)", "->"),
        (r"\\(?:leftrightarrow|iff)", "<->"),
        (r"\\(?:land|wedge|bigwedge)", r"/\\"),  # Use raw string for backslash
        (r"\\(?:lor|vee|bigvee)", r"\/"),  # Use raw string for backslash
        (r"\\(?:neg|lnot|not)", "~"),
        (r"\\(?:bot)", "False"),
        (r"\\(?:top)", "True"),
    ]
)

_MISC_LATEX_REWRITES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\\left\W*"), ""),
    (re.compile(r"\\right\W*"), ""),
    (re.compile(r"\\text\{([^}]*)\}"), r"\1"),
)

_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class LeanStatement:
    """Canonical views of a propositional statement."""

    canonical: str          # structural, compact ASCII form used for hashing
    ascii_pretty: str       # human-readable ASCII form (stored in DB/jobs)
    lean: str               # Unicode form emitted to Lean

    def is_empty(self) -> bool:
        return not self.canonical


def _decode_unicode_escapes(text: str) -> str:
    r"""Decode unicode escape sequences like \u2192 to their character form."""
    try:
        return text.encode("utf-8", "surrogatepass").decode("unicode_escape")
    except UnicodeDecodeError:
        return text


def _apply_latex(text: str) -> str:
    for pattern, replacement in _LATEX_PATTERN_MAP:
        text = pattern.sub(replacement, text)
    for pattern, replacement in _MISC_LATEX_REWRITES:
        text = pattern.sub(replacement, text)
    return text


def _ascii_to_unicode(pretty: str) -> str:
    expr = pretty
    # Replace longer tokens first to avoid partial overlaps.
    expr = expr.replace("<->", " ↔ ")
    expr = expr.replace("->", " → ")
    expr = expr.replace("/\\", " ∧ ")
    expr = expr.replace("\\/", " ∨ ")
    expr = expr.replace("~", "¬")
    expr = _WHITESPACE_RE.sub(" ", expr)
    expr = expr.replace("( ", "(").replace(" )", ")")
    expr = expr.replace("¬ ", "¬")
    return expr.strip()


def sanitize_statement(raw: Optional[str]) -> LeanStatement:
    """
    Convert raw statements (possibly containing LaTeX/unicode escapes) into
    canonical ASCII + Unicode Lean-ready forms.
    """
    if raw is None:
        return LeanStatement(canonical="", ascii_pretty="", lean="")

    text = html.unescape(raw)
    text = _decode_unicode_escapes(text)
    text = _apply_latex(text)
    text = text.replace("{", "(").replace("}", ")")
    text = text.replace("\r", " ").replace("\n", " ")
    text = _WHITESPACE_RE.sub(" ", text).strip()

    if not text:
        return LeanStatement(canonical="", ascii_pretty="", lean="")

    canonical = normalize(text)
    if not canonical:
        return LeanStatement(canonical="", ascii_pretty="", lean="")

    pretty = normalize_pretty(text)
    lean_stmt = _ascii_to_unicode(pretty)

    return LeanStatement(canonical=canonical, ascii_pretty=pretty, lean=lean_stmt)

