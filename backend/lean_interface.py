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
from typing import Literal, Optional

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


# Structured Lean failure telemetry -----------------------------------------

LeanFailureKind = Literal["timeout", "type_error", "tactic_failure", "unknown"]


@dataclass(frozen=True, slots=True)
class LeanFailureSignal:
    """
    Structured metadata describing why Lean failed to verify a goal.

    kind:
        timeout         - Lean exceeded its timeout or returned code 124.
        type_error      - Lean reported a type mismatch or elaboration failure.
        tactic_failure  - Lean reached tactic failure / unsolved goals.
        unknown         - Could not classify the stderr payload.

    message:
        Short, canonicalized description of the failure (first stderr line).

    elapsed_ms:
        Wall-clock duration (in milliseconds) for the Lean attempt.
    """

    kind: LeanFailureKind
    message: str
    elapsed_ms: int


_TIMEOUT_HINTS = (
    "timeout",
    "time out",
    "timed out",
    "execution exceeded",
    "[timeout]",
)
_TYPE_ERROR_HINTS = (
    "type mismatch",
    "type expected",
    "has type",
    "failed to synthesize type",
    "invalid type",
)
_TACTIC_FAILURE_HINTS = (
    "tactic failed",
    "no goals to be solved",
    "unsolved goals",
    "state space search failed",
)


def classify_lean_failure(
    stderr: Optional[str],
    returncode: int,
    elapsed_ms: int,
) -> LeanFailureSignal:
    """
    Map Lean stderr text + return code into a LeanFailureSignal.

    Args:
        stderr: Raw stderr captured from Lean/lake build.
        returncode: Process return code.
        elapsed_ms: Execution duration (milliseconds) for observability.
    """
    text = (stderr or "").strip()
    normalized = text.lower()

    kind: LeanFailureKind = "unknown"

    if returncode == 124 or any(hint in normalized for hint in _TIMEOUT_HINTS):
        kind = "timeout"
    elif any(hint in normalized for hint in _TYPE_ERROR_HINTS):
        kind = "type_error"
    elif any(hint in normalized for hint in _TACTIC_FAILURE_HINTS):
        kind = "tactic_failure"

    if not text:
        text = f"lean return code {returncode}"

    first_line = text.splitlines()[0]
    return LeanFailureSignal(kind=kind, message=first_line, elapsed_ms=elapsed_ms)


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

