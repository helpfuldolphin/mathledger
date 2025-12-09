"""
Rule definitions and proof context for the axiom engine.

Provides:
- ProofContext dataclass for proof metadata
- Rule definitions for derivation
- Tautology recognition patterns
- TautologyResult for structured verification outcomes
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class TautologyVerdict(Enum):
    """Verdict for tautology checking."""
    TAUTOLOGY = "tautology"
    NOT_TAUTOLOGY = "not_tautology"
    ABSTAIN = "abstain"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class TautologyResult:
    """
    Structured result from tautology verification.

    Attributes:
        verdict: The verification verdict (TAUTOLOGY, NOT_TAUTOLOGY, ABSTAIN, etc.)
        method: The method used for verification (e.g., 'pattern', 'truth_table')
        duration_ms: Time taken for verification in milliseconds
        formula: The formula that was checked
        error_message: Error message if verdict is ERROR
    """
    verdict: TautologyVerdict
    method: str = "unknown"
    duration_ms: float = 0.0
    formula: str = ""
    error_message: Optional[str] = None

    @property
    def is_tautology(self) -> bool:
        """True if the formula was verified as a tautology."""
        return self.verdict == TautologyVerdict.TAUTOLOGY

    @property
    def is_not_tautology(self) -> bool:
        """True if the formula was verified as not a tautology."""
        return self.verdict == TautologyVerdict.NOT_TAUTOLOGY

    @property
    def is_abstain(self) -> bool:
        """True if verification abstained (timeout, error, etc.)."""
        return self.verdict in (TautologyVerdict.ABSTAIN, TautologyVerdict.TIMEOUT, TautologyVerdict.ERROR)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "verdict": self.verdict.value,
            "method": self.method,
            "duration_ms": self.duration_ms,
            "formula": self.formula,
            "error_message": self.error_message,
            "is_tautology": self.is_tautology,
            "is_not_tautology": self.is_not_tautology,
            "is_abstain": self.is_abstain,
        }


@dataclass
class ProofContext:
    """
    Context information for a proof derivation.
    
    Attributes:
        statement_id: Unique identifier for the statement
        dependencies: List of parent statement IDs
        derivation_rule: Name of the rule used (e.g., 'mp', 'axiom', 'smoke_pl')
        merkle_root: Merkle root of the proof tree
        signature_b64: Base64-encoded Ed25519 signature
    """
    statement_id: str
    dependencies: List[str] = field(default_factory=list)
    derivation_rule: str = "unknown"
    merkle_root: str = ""
    signature_b64: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "statement_id": self.statement_id,
            "dependencies": self.dependencies,
            "derivation_rule": self.derivation_rule,
            "merkle_root": self.merkle_root,
            "signature_b64": self.signature_b64,
        }


# Tautology recognition patterns for common PL-2 forms
_R_AND = r"/\\"
_v = r"([a-z])"
_vv = rf"\({_v}\){_R_AND}\({_v}\)"  # (x/\y)
_assocL = rf"\({_v}\){_R_AND}\({_v}\{_R_AND}{_v}\)"  # (x/\(y/\z))
_assocR = rf"\(\({_v}\{_R_AND}{_v}\){_R_AND}{_v}\)"  # ((x/\y)/\z)

KNOWN_TAUTOLOGY_PATTERNS = [
    rf"^\({_v}\{_R_AND}{_v}\)->\1$",  # (x/\y)->x
    rf"^\({_v}\{_R_AND}{_v}\)->\2$",  # (x/\y)->y
    rf"^{_v}->\({_v}->\1\)$",  # x->(y->x)
    rf"^\({_v}\{_R_AND}\({_v}\{_R_AND}{_v}\)\)->\1$",  # (x/\(y/\z))->x
    rf"^\(\({_v}\{_R_AND}{_v}\)\{_R_AND}{_v}\)->\1$",  # ((x/\y)/\z)->x
    rf"^\({_v}\{_R_AND}{_v}\)->\(\2\{_R_AND}\1\)$",  # (x/\y)->(y/\x)
    rf"^{_v}->{_v}->\1$",  # x->y->x (simplified)
]

KNOWN_TAUTOLOGY_RE = [re.compile(p) for p in KNOWN_TAUTOLOGY_PATTERNS]


def is_known_tautology(norm: str) -> bool:
    """
    Check if normalized formula matches known tautology patterns.
    
    Args:
        norm: Normalized formula string
        
    Returns:
        True if matches a known pattern
    """
    return any(r.match(norm) for r in KNOWN_TAUTOLOGY_RE)


def is_tautology_with_timeout(norm: str, timeout_ms: int = 5) -> TautologyResult:
    """
    Fast check for tautology with timeout for slow path.

    Args:
        norm: Normalized formula string
        timeout_ms: Timeout in milliseconds for slow verification

    Returns:
        TautologyResult with verification outcome
    """
    import time
    start_time = time.perf_counter()

    # Instant check for known schemata
    if is_known_tautology(norm):
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return TautologyResult(
            verdict=TautologyVerdict.TAUTOLOGY,
            method="pattern",
            duration_ms=elapsed_ms,
            formula=norm,
        )

    # Bounded slow path with timeout
    try:
        from normalization.truthtab import is_tautology as slow_tauto

        result = bool(slow_tauto(norm))
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if elapsed_ms > timeout_ms:
            # Timeout hit
            return TautologyResult(
                verdict=TautologyVerdict.TIMEOUT,
                method="truth_table",
                duration_ms=elapsed_ms,
                formula=norm,
            )

        if result:
            return TautologyResult(
                verdict=TautologyVerdict.TAUTOLOGY,
                method="truth_table",
                duration_ms=elapsed_ms,
                formula=norm,
            )
        else:
            return TautologyResult(
                verdict=TautologyVerdict.NOT_TAUTOLOGY,
                method="truth_table",
                duration_ms=elapsed_ms,
                formula=norm,
            )
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return TautologyResult(
            verdict=TautologyVerdict.ERROR,
            method="truth_table",
            duration_ms=elapsed_ms,
            formula=norm,
            error_message=str(e),
        )


@dataclass
class ProofResult:
    """Result of a proof verification."""
    formula: str
    normalized: str
    method: str = "smoke_pl"
    verified: bool = True


# Rule constants
RULE_MODUS_PONENS = "mp"
RULE_AXIOM = "axiom"
RULE_SMOKE_PL = "smoke_pl"
RULE_DERIVED = "derived"

# Logical operators
IMPLIES = "->"
AND = "/\\"
OR = "\\/"
NOT = "~"
