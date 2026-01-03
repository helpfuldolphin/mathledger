"""
Trust Class Definitions for UVIL v0.

Subordinate to fm.tex sec:trust-classes.

Trust Classes:
- FV: Formally Verified - machine-checkable proof (Lean/Z3, Phase II; returns ABSTAINED in v0)
- MV: Mechanically Validated - arithmetic validator (limited coverage: a op b = c for integers)
- PA: Procedurally Attested - user-attested for v0 (human asserts correctness)
- ADV: Advisory - non-authoritative, exploration only

Only FV, MV, PA are authority-bearing and may enter R_t.
ADV is exploration-only and MUST NEVER appear in R_t.
"""

from enum import Enum

__all__ = [
    "TrustClass",
    "Outcome",
    "DEFAULT_SUGGESTED_TRUST_CLASS",
    "AUTHORITY_BEARING_TRUST_CLASSES",
    "is_authority_bearing",
]


class TrustClass(str, Enum):
    """
    Trust classes per FM sec:trust-classes.

    FV, MV, PA are authority-bearing.
    ADV is exploration-only (non-authoritative).
    """

    FV = "FV"  # Formally Verified
    MV = "MV"  # Mechanically Validated
    PA = "PA"  # Procedurally Attested (user-attested for v0)
    ADV = "ADV"  # Advisory (non-authoritative)


class Outcome(str, Enum):
    """Verification outcomes per FM."""

    VERIFIED = "VERIFIED"
    REFUTED = "REFUTED"
    ABSTAINED = "ABSTAINED"


# UX: user must explicitly promote from ADV
DEFAULT_SUGGESTED_TRUST_CLASS = TrustClass.ADV

# Authority-bearing trust classes that may enter R_t
AUTHORITY_BEARING_TRUST_CLASSES = frozenset({TrustClass.FV, TrustClass.MV, TrustClass.PA})


def is_authority_bearing(tc: TrustClass) -> bool:
    """
    Return True if trust class can enter R_t.

    Only FV, MV, PA are authority-bearing.
    ADV MUST NEVER enter R_t.

    Args:
        tc: Trust class to check

    Returns:
        True if authority-bearing, False otherwise
    """
    return tc in AUTHORITY_BEARING_TRUST_CLASSES
