"""
Calibration Experiment Language Constraints — Single Source of Truth

Canonical references:
    docs/system_law/calibration/CAL_EXP_2_LANGUAGE_CONSTRAINTS.md
    docs/system_law/calibration/CAL_EXP_3_LANGUAGE_CONSTRAINTS.md

This module defines forbidden phrases for calibration experiment reporting.
All lint tests and documentation should import from here.

If a phrase sounds impressive, it's probably illegal.

SHADOW MODE — observational only.
"""

import re
from typing import FrozenSet

# =============================================================================
# CAL-EXP-2: Twin/Divergence calibration
# =============================================================================

CAL_EXP_2_FORBIDDEN_PHRASES: FrozenSet[str] = frozenset([
    "divergence eliminated",
    "twin validated",
    "calibration passed",
    "model converged",
    "accuracy improved",
    "system aligned",
    "ready for production",
    "governance approved",
    "monotone improvement achieved",
])

# =============================================================================
# CAL-EXP-3: Uplift/Learning calibration
# =============================================================================

# Mechanism claims — always forbidden (per CAL_EXP_3_UPLIFT_SPEC.md)
# These imply causal attribution without mechanism evidence.
CAL_EXP_3_MECHANISM_CLAIMS: FrozenSet[str] = frozenset([
    "learning works",
    "system improved",
])

CAL_EXP_3_FORBIDDEN_PHRASES: FrozenSet[str] = frozenset([
    # Inherited from CAL-EXP-2 (universal)
    "calibration passed",
    "ready for production",
    "governance approved",
    # CAL-EXP-3 specific — anthropomorphizing/overstating
    "improved intelligence",
    "validated learning",
    "generalization",          # Too strong; implies transfer beyond training
    "learned behavior",        # Implies autonomous adaptation
    "intelligence gain",       # Overstates capability
    "cognitive improvement",   # Anthropomorphizes
]) | CAL_EXP_3_MECHANISM_CLAIMS

# Explicitly ALLOWED phrases for CAL-EXP-3
# These are neutral, measurement-focused terms.
CAL_EXP_3_ALLOWED_PHRASES: FrozenSet[str] = frozenset([
    "measured uplift",
    "observed delta",
    "metric change",
    "performance delta",
    "treatment arm",
    "baseline arm",
])

# =============================================================================
# Combined / Universal
# =============================================================================

ALL_FORBIDDEN_PHRASES: FrozenSet[str] = (
    CAL_EXP_2_FORBIDDEN_PHRASES | CAL_EXP_3_FORBIDDEN_PHRASES
)

# =============================================================================
# Backward compatibility aliases
# =============================================================================

# Legacy alias for CAL-EXP-2 tests
FORBIDDEN_PHRASES: FrozenSet[str] = CAL_EXP_2_FORBIDDEN_PHRASES

# =============================================================================
# Pre-compiled patterns
# =============================================================================

def _build_pattern(phrases: FrozenSet[str]) -> re.Pattern:
    """Build case-insensitive regex pattern from phrase set."""
    return re.compile(
        "|".join(re.escape(phrase) for phrase in sorted(phrases)),
        re.IGNORECASE,
    )

FORBIDDEN_PATTERN: re.Pattern = _build_pattern(CAL_EXP_2_FORBIDDEN_PHRASES)
CAL_EXP_3_FORBIDDEN_PATTERN: re.Pattern = _build_pattern(CAL_EXP_3_FORBIDDEN_PHRASES)
ALL_FORBIDDEN_PATTERN: re.Pattern = _build_pattern(ALL_FORBIDDEN_PHRASES)

# =============================================================================
# Scanning utilities
# =============================================================================

def scan_text_for_violations(
    text: str,
    pattern: re.Pattern = FORBIDDEN_PATTERN,
) -> list[tuple[int, str, str]]:
    """
    Scan text for forbidden phrases.

    Args:
        text: The text to scan.
        pattern: Regex pattern to match against. Defaults to CAL-EXP-2 pattern.

    Returns:
        List of (line_num, phrase, context) tuples for each violation.
    """
    violations = []
    for line_num, line in enumerate(text.splitlines(), start=1):
        for match in pattern.finditer(line):
            violations.append((line_num, match.group(), line.strip()[:80]))
    return violations


def scan_text_cal_exp_3(text: str) -> list[tuple[int, str, str]]:
    """Scan text for CAL-EXP-3 forbidden phrases."""
    return scan_text_for_violations(text, CAL_EXP_3_FORBIDDEN_PATTERN)


def scan_text_all(text: str) -> list[tuple[int, str, str]]:
    """Scan text for ALL forbidden phrases (CAL-EXP-2 + CAL-EXP-3)."""
    return scan_text_for_violations(text, ALL_FORBIDDEN_PATTERN)
