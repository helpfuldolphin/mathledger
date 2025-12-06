"""
Rule definitions and proof context for the axiom engine.

Provides:
- ProofContext dataclass for proof metadata
- Rule definitions for derivation
- Tautology recognition patterns
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


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


def is_tautology_with_timeout(norm: str, timeout_ms: int = 5) -> bool:
    """
    Fast check for tautology with timeout for slow path.
    
    Args:
        norm: Normalized formula string
        timeout_ms: Timeout in milliseconds for slow verification
        
    Returns:
        True if tautology, False otherwise or on timeout
    """
    from backend.repro.determinism import deterministic_unix_timestamp
    
    # Instant check for known schemata
    if is_known_tautology(norm):
        return True

    # Bounded slow path with timeout
    try:
        from backend.logic.truthtab import is_tautology as slow_tauto
        
        start_time = deterministic_unix_timestamp(0)
        result = bool(slow_tauto(norm))
        elapsed_ms = (deterministic_unix_timestamp(0) - start_time) * 1000
        
        if elapsed_ms > timeout_ms:
            # Timeout hit, skip this candidate
            return False
        return result
    except Exception:
        return False


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
