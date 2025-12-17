"""
CAL-EXP-3 Language Lint Test

Enforces language constraints for uplift/learning calibration experiments.
Maps to claim ladder L0-L5 defined in CAL_EXP_3_UPLIFT_SPEC.md.

Single source of truth:
    backend/governance/language_constraints.py

If a phrase sounds impressive, it's probably illegal.

SHADOW MODE — observational only.
"""

from pathlib import Path
from typing import NamedTuple

import pytest

from backend.governance.language_constraints import (
    CAL_EXP_3_ALLOWED_PHRASES,
    CAL_EXP_3_FORBIDDEN_PHRASES,
    CAL_EXP_3_FORBIDDEN_PATTERN,
    CAL_EXP_3_MECHANISM_CLAIMS,
    scan_text_cal_exp_3,
)


class Violation(NamedTuple):
    file: str
    line_num: int
    phrase: str
    context: str


def scan_file(path: Path) -> list[Violation]:
    """Scan a file for CAL-EXP-3 forbidden phrases."""
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        pytest.skip(f"Cannot read {path}: {e}")
        return []

    raw_violations = scan_text_cal_exp_3(content)
    return [
        Violation(file=str(path), line_num=ln, phrase=ph, context=ctx)
        for ln, ph, ctx in raw_violations
    ]


def format_violation(v: Violation) -> str:
    """Single-line actionable output."""
    return f"{v.file}:{v.line_num}: forbidden phrase \"{v.phrase}\""


def format_violations(violations: list[Violation]) -> str:
    """Format violations for actionable output."""
    lines = ["CAL-EXP-3 Language Violations:"]
    for v in violations:
        lines.append(f"  {format_violation(v)}")
    lines.append("Single source: backend/governance/language_constraints.py")
    return "\n".join(lines)


# --- Test targets ---

ROOT = Path(__file__).resolve().parents[2]

# CAL-EXP-3 canonical docs (spec + plan) — should exist and pass
CAL_EXP_3_CANONICAL_DOCS = [
    ROOT / "docs/system_law/calibration/CAL_EXP_3_UPLIFT_SPEC.md",
    ROOT / "docs/system_law/calibration/CAL_EXP_3_IMPLEMENTATION_PLAN.md",
    ROOT / "docs/system_law/calibration/CAL_EXP_3_AUTHORIZATION.md",
    ROOT / "docs/system_law/calibration/CAL_EXP_3_LANGUAGE_CONSTRAINTS.md",
]

# CAL-EXP-3 docs artifacts (skip if not yet created)
# NOTE: results/ are NOT scanned — they are local-only and not committed.
CAL_EXP_3_DOC_ARTIFACTS = [
    ROOT / "docs/system_law/calibration/CAL_EXP_3_EXPERIMENT_DESIGN.md",
    ROOT / "docs/system_law/calibration/CAL_EXP_3_Canonical_Record.md",
]

# Code surfaces that generate CAL-EXP-3 outputs
CAL_EXP_3_CODE_SURFACES = [
    ROOT / "scripts/run_cal_exp_3.py",
]


def is_self_documenting(context: str) -> bool:
    """Check if the violation is in a self-documenting context (table/example/code)."""
    ctx = context.lower()
    # Table markers: | at start
    if ctx.startswith("|"):
        return True
    # Code block patterns (regex strings, code examples)
    if ctx.strip().startswith(('r"', "r'", '"', "'")):
        return True
    # Documentation keywords that indicate the phrase is being defined/prohibited
    doc_keywords = [
        "forbidden",
        "prohibited",
        "invalid",
        "why invalid",
        "mechanism claim",
        "escalation",
        "alternative",
        "correct alternative",
        "forbidden_patterns",
    ]
    return any(kw in ctx for kw in doc_keywords)


class TestCalExp3CanonicalDocs:
    """Language hygiene for CAL-EXP-3 specification documents."""

    @pytest.mark.parametrize("filepath", CAL_EXP_3_CANONICAL_DOCS, ids=lambda p: p.name)
    def test_canonical_doc_no_forbidden_phrases(self, filepath: Path):
        """CAL-EXP-3 canonical docs must not contain forbidden phrases in claims."""
        if not filepath.exists():
            pytest.skip(f"Canonical doc not yet created: {filepath.name}")

        violations = scan_file(filepath)
        # Filter out violations in self-documenting contexts (tables, examples)
        violations = [v for v in violations if not is_self_documenting(v.context)]

        if violations:
            pytest.fail(format_violations(violations))


class TestCalExp3DocArtifacts:
    """Language hygiene for CAL-EXP-3 doc artifacts (skip if absent)."""

    @pytest.mark.parametrize("filepath", CAL_EXP_3_DOC_ARTIFACTS, ids=lambda p: p.name)
    def test_doc_artifact_no_forbidden_phrases(self, filepath: Path):
        """Doc artifacts are skipped if absent, checked if present."""
        if not filepath.exists():
            pytest.skip(f"Doc artifact not yet created: {filepath.name}")

        violations = scan_file(filepath)
        # Filter out self-documenting contexts
        violations = [v for v in violations if not is_self_documenting(v.context)]
        if violations:
            pytest.fail(format_violations(violations))


class TestCalExp3CodeSurfaces:
    """Language hygiene for CAL-EXP-3 generator code."""

    @pytest.mark.parametrize("filepath", CAL_EXP_3_CODE_SURFACES, ids=lambda p: p.name)
    def test_code_surface_no_forbidden_phrases(self, filepath: Path):
        """Generator code must emit neutral language."""
        if not filepath.exists():
            pytest.skip(f"Code file not found: {filepath.name}")

        violations = scan_file(filepath)
        # Filter out self-documenting contexts (comments, docstrings explaining what's forbidden)
        violations = [v for v in violations if not is_self_documenting(v.context)]
        if violations:
            pytest.fail(format_violations(violations))


class TestCalExp3ForbiddenPhrases:
    """Verify forbidden phrase detection works correctly."""

    @pytest.mark.parametrize("phrase", sorted(CAL_EXP_3_FORBIDDEN_PHRASES))
    def test_forbidden_phrase_detected(self, phrase: str):
        """Each forbidden phrase must be detected by the scanner."""
        text = f"Line 1: OK\nLine 2: The system showed {phrase} after tuning.\nLine 3: OK"
        violations = scan_text_cal_exp_3(text)

        detected_phrases = [v[1].lower() for v in violations]
        assert phrase.lower() in detected_phrases, (
            f"Forbidden phrase '{phrase}' was not detected"
        )

    @pytest.mark.parametrize("phrase", sorted(CAL_EXP_3_ALLOWED_PHRASES))
    def test_allowed_phrase_not_flagged(self, phrase: str):
        """Allowed phrases must NOT be flagged as violations."""
        text = f"The experiment showed {phrase} of 0.02."
        violations = scan_text_cal_exp_3(text)

        detected_phrases = [v[1].lower() for v in violations]
        assert phrase.lower() not in detected_phrases, (
            f"Allowed phrase '{phrase}' was incorrectly flagged"
        )


class TestCalExp3MechanismClaims:
    """Mechanism claims are always forbidden (per UPLIFT_SPEC)."""

    @pytest.mark.parametrize("phrase", sorted(CAL_EXP_3_MECHANISM_CLAIMS))
    def test_mechanism_claim_detected(self, phrase: str):
        """Mechanism claims must be detected as forbidden."""
        text = f"The experiment proves {phrase}."
        violations = scan_text_cal_exp_3(text)

        detected_phrases = [v[1].lower() for v in violations]
        assert phrase.lower() in detected_phrases, (
            f"Mechanism claim '{phrase}' was not detected"
        )

    def test_mechanism_claims_in_forbidden_set(self):
        """All mechanism claims must be in the forbidden phrases set."""
        missing = CAL_EXP_3_MECHANISM_CLAIMS - CAL_EXP_3_FORBIDDEN_PHRASES
        assert not missing, f"Mechanism claims not in forbidden set: {missing}"


class TestCalExp3ClaimLadder:
    """Verify claim ladder alignment."""

    def test_required_forbidden_phrases(self):
        """Verify the canonical list includes required forbidden phrases."""
        required = {"improved intelligence", "validated learning", "generalization"}
        missing = required - CAL_EXP_3_FORBIDDEN_PHRASES

        assert not missing, (
            f"Required forbidden phrases missing: {missing}\n"
            f"Update backend/governance/language_constraints.py"
        )

    def test_measured_uplift_is_allowed(self):
        """L4 claim template 'measured uplift' must be allowed."""
        assert "measured uplift" in CAL_EXP_3_ALLOWED_PHRASES

        text = "The measured uplift was 0.02."
        violations = scan_text_cal_exp_3(text)
        assert len(violations) == 0, f"'measured uplift' flagged: {violations}"

    def test_impressive_phrases_are_forbidden(self):
        """If a phrase sounds impressive, it's probably illegal."""
        impressive = [
            "improved intelligence",
            "validated learning",
            "intelligence gain",
            "cognitive improvement",
            "learned behavior",
            "learning works",
            "system improved",
        ]

        for phrase in impressive:
            assert phrase in CAL_EXP_3_FORBIDDEN_PHRASES, (
                f"Impressive phrase '{phrase}' should be forbidden"
            )
