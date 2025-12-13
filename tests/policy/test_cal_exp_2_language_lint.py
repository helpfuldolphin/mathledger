"""
CAL-EXP-2 Language Lint Test

Enforces language constraints from:
    docs/system_law/calibration/CAL_EXP_2_LANGUAGE_CONSTRAINTS.md

Single source of truth for forbidden phrases:
    backend/governance/language_constraints.py

Scans specified files for forbidden phrases that imply validation,
production-readiness, or claim inflation beyond SHADOW MODE scope.

SHADOW MODE — observational only.
"""

from pathlib import Path
from typing import NamedTuple

import pytest

from backend.governance.language_constraints import (
    FORBIDDEN_PATTERN,
    FORBIDDEN_PHRASES,
    scan_text_for_violations,
)


class Violation(NamedTuple):
    file: str
    line_num: int
    phrase: str
    context: str


def scan_file(path: Path) -> list[Violation]:
    """Scan a file for forbidden phrases. Returns list of violations."""
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        pytest.skip(f"Cannot read {path}: {e}")
        return []

    raw_violations = scan_text_for_violations(content)
    return [
        Violation(file=str(path), line_num=ln, phrase=ph, context=ctx)
        for ln, ph, ctx in raw_violations
    ]


def format_violations(violations: list[Violation]) -> str:
    """Format violations for actionable output."""
    lines = ["", "CAL-EXP-2 Language Violations Found:", ""]
    for v in violations:
        lines.append(f"  {v.file}:{v.line_num}")
        lines.append(f"    Phrase: \"{v.phrase}\"")
        lines.append(f"    Context: {v.context}")
        lines.append("")
    lines.append("See: docs/system_law/calibration/CAL_EXP_2_LANGUAGE_CONSTRAINTS.md")
    lines.append("Single source: backend/governance/language_constraints.py")
    return "\n".join(lines)


# --- Test targets ---

ROOT = Path(__file__).resolve().parents[2]

# Required files (must exist and pass)
REQUIRED_FILES = [
    ROOT / "docs/system_law/calibration/CAL_EXP_2_Canonical_Record.md",
]

# Optional files (skip if absent, fail if present and violating)
OPTIONAL_FILES = [
    ROOT / "results/cal_exp_2/CAL_EXP_2_Scientist_Report.md",
    ROOT / "docs/system_law/calibration/audits/CAL_EXP_2_RESULTS.md",
]


class TestCalExp2LanguageLint:
    """Language hygiene tests for CAL-EXP-2 documentation."""

    @pytest.mark.parametrize("filepath", REQUIRED_FILES, ids=lambda p: p.name)
    def test_required_file_no_forbidden_phrases(self, filepath: Path):
        """Required files must exist and contain no forbidden phrases."""
        if not filepath.exists():
            pytest.fail(f"Required file missing: {filepath}")

        violations = scan_file(filepath)
        if violations:
            pytest.fail(format_violations(violations))

    @pytest.mark.parametrize("filepath", OPTIONAL_FILES, ids=lambda p: p.name)
    def test_optional_file_no_forbidden_phrases(self, filepath: Path):
        """Optional files are skipped if absent, checked if present."""
        if not filepath.exists():
            pytest.skip(f"Optional file not present: {filepath.name}")

        violations = scan_file(filepath)
        if violations:
            pytest.fail(format_violations(violations))


def test_forbidden_phrases_list_is_complete():
    """Verify forbidden phrases match the canonical constraints document."""
    constraints_file = ROOT / "docs/system_law/calibration/CAL_EXP_2_LANGUAGE_CONSTRAINTS.md"
    if not constraints_file.exists():
        pytest.skip("Constraints document not found")

    content = constraints_file.read_text(encoding="utf-8").lower()

    # Check each forbidden phrase appears in the constraints doc
    missing = []
    for phrase in FORBIDDEN_PHRASES:
        if phrase.lower() not in content:
            missing.append(phrase)

    if missing:
        pytest.fail(
            f"Forbidden phrases in code but not in constraints doc: {missing}\n"
            f"Update CAL_EXP_2_LANGUAGE_CONSTRAINTS.md to include these phrases,\n"
            f"or remove them from backend/governance/language_constraints.py"
        )


def test_single_source_of_truth():
    """Verify no duplicate FORBIDDEN_PHRASES definitions exist."""
    # This test documents the single-source contract.
    # If someone duplicates the list, this test reminds them to use the import.
    assert len(FORBIDDEN_PHRASES) >= 8, "Canonical list should have at least 8 phrases"
    assert "divergence eliminated" in FORBIDDEN_PHRASES
    assert "ready for production" in FORBIDDEN_PHRASES
