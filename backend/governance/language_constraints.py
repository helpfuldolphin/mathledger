"""
CAL-EXP-2 Language Constraints — Single Source of Truth

Canonical reference:
    docs/system_law/calibration/CAL_EXP_2_LANGUAGE_CONSTRAINTS.md

This module defines the forbidden phrases for CAL-EXP-2 reporting.
All lint tests and documentation should import from here.

SHADOW MODE — observational only.
"""

import re
from typing import FrozenSet

# Canonical forbidden phrases for CAL-EXP-2 reporting.
# These phrases imply validation, production-readiness, or claim inflation
# beyond SHADOW MODE scope.
#
# To add/remove phrases: update this list AND the canonical markdown doc.
FORBIDDEN_PHRASES: FrozenSet[str] = frozenset([
    "divergence eliminated",
    "twin validated",
    "calibration passed",
    "model converged",
    "accuracy improved",
    "system aligned",
    "ready for production",
    "governance approved",
    "monotone improvement achieved",  # Added from EXIT_DECISION doc
])

# Pre-compiled regex for efficient scanning
FORBIDDEN_PATTERN: re.Pattern = re.compile(
    "|".join(re.escape(phrase) for phrase in sorted(FORBIDDEN_PHRASES)),
    re.IGNORECASE,
)


def scan_text_for_violations(text: str) -> list[tuple[int, str, str]]:
    """
    Scan text for forbidden phrases.

    Returns:
        List of (line_num, phrase, context) tuples for each violation.
    """
    violations = []
    for line_num, line in enumerate(text.splitlines(), start=1):
        for match in FORBIDDEN_PATTERN.finditer(line):
            violations.append((line_num, match.group(), line.strip()[:80]))
    return violations
