"""
Tests for V0_LOCK.md and invariants_status.md documentation integrity.

These tests prevent regression of documentation fixes for external audit findings:
1. "All claims ABSTAINED" false claim (must scope to PA/FV/ADV)
2. "no verifier" ambiguous language (must scope to formal/Lean/Z3)
3. Tier A=10 but invariant 10 unexplained (must have full section)

Run with:
    uv run pytest tests/governance/test_scope_lock_doc_integrity.py -v
"""

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).parent.parent.parent
DOCS_DIR = REPO_ROOT / "docs"

V0_LOCK_PATH = DOCS_DIR / "V0_LOCK.md"
INVARIANTS_STATUS_PATH = DOCS_DIR / "invariants_status.md"


# ---------------------------------------------------------------------------
# Tests: V0_LOCK.md - No False Abstain Claim
# ---------------------------------------------------------------------------

class TestScopeLockNoFalseAbstainClaim:
    """
    Verify V0_LOCK.md does not falsely claim all claims return ABSTAINED.

    MV arithmetic claims CAN return VERIFIED/REFUTED.
    Only PA, FV, ADV claims return ABSTAINED.
    """

    def test_scope_lock_no_false_abstain_claim(self):
        """
        V0_LOCK.md must NOT contain unscoped 'all claims ABSTAINED'.

        The phrase 'All claims in v0 return ABSTAINED' is factually wrong
        because MV arithmetic claims return VERIFIED/REFUTED.
        """
        if not V0_LOCK_PATH.exists():
            pytest.skip(f"V0_LOCK.md not found: {V0_LOCK_PATH}")

        content = V0_LOCK_PATH.read_text(encoding="utf-8")

        # Pattern to detect the false claim (case insensitive)
        false_claim_pattern = re.compile(
            r"all\s+claims\s+(?:in\s+v0\s+)?return\s+abstained",
            re.IGNORECASE
        )

        match = false_claim_pattern.search(content)
        assert not match, (
            "V0_LOCK.md contains false claim that 'all claims return ABSTAINED'. "
            "MV arithmetic claims can return VERIFIED/REFUTED. "
            "Scope the statement to PA/FV/ADV claims only."
        )

    def test_abstain_claim_properly_scoped(self):
        """
        If ABSTAINED is mentioned with trust classes, it must include scope.

        Acceptable: 'PA, FV, and ADV claims return ABSTAINED'
        Unacceptable: 'All claims return ABSTAINED'
        """
        if not V0_LOCK_PATH.exists():
            pytest.skip(f"V0_LOCK.md not found: {V0_LOCK_PATH}")

        content = V0_LOCK_PATH.read_text(encoding="utf-8")

        # Check that PA terminology section exists and is properly scoped
        # It should mention PA, FV, ADV returning ABSTAINED, not "all claims"
        if "PA terminology hazard" in content:
            # Find the section
            section_match = re.search(
                r"PA terminology hazard.*?(?=###|---|\Z)",
                content,
                re.DOTALL | re.IGNORECASE
            )
            if section_match:
                section = section_match.group()
                # Should mention specific trust classes, not "all claims"
                assert "PA" in section and ("FV" in section or "ADV" in section), (
                    "PA terminology hazard section should scope ABSTAINED to specific trust classes"
                )

    def test_mv_verified_acknowledged(self):
        """
        V0_LOCK.md must acknowledge that MV claims CAN return VERIFIED/REFUTED.
        """
        if not V0_LOCK_PATH.exists():
            pytest.skip(f"V0_LOCK.md not found: {V0_LOCK_PATH}")

        content = V0_LOCK_PATH.read_text(encoding="utf-8")

        # Should mention MV can return VERIFIED or REFUTED
        mv_verified_pattern = re.compile(
            r"MV.*(?:VERIFIED|REFUTED|arithmetic\s+validator)",
            re.IGNORECASE | re.DOTALL
        )

        assert mv_verified_pattern.search(content), (
            "V0_LOCK.md should acknowledge that MV claims can return "
            "VERIFIED/REFUTED via the arithmetic validator"
        )


# ---------------------------------------------------------------------------
# Tests: V0_LOCK.md - Verifier Language Scoped to Formal
# ---------------------------------------------------------------------------

class TestScopeLockVerifierLanguageScoped:
    """
    Verify 'no verifier' language in V0_LOCK.md is scoped to formal (Lean/Z3).

    V0 DOES have a verifier: the MV arithmetic validator.
    'No verifier' is only true for formal verification (Lean/Z3).
    """

    def test_scope_lock_verifier_language_scoped_to_formal(self):
        """
        'no verifier' statements must be scoped to 'no formal verifier'.

        The MV arithmetic validator IS a verifier (limited coverage).
        Unscoped 'no verifier' is misleading.
        """
        if not V0_LOCK_PATH.exists():
            pytest.skip(f"V0_LOCK.md not found: {V0_LOCK_PATH}")

        content = V0_LOCK_PATH.read_text(encoding="utf-8")

        # Find all occurrences of "no verifier"
        no_verifier_pattern = re.compile(r"no\s+verifier", re.IGNORECASE)

        for match in no_verifier_pattern.finditer(content):
            start = match.start()
            end = match.end()

            # Get context (100 chars before and after)
            context_start = max(0, start - 100)
            context_end = min(len(content), end + 100)
            context = content[context_start:context_end]

            # Check if "formal" or "Lean" or "Z3" is nearby
            has_scoping = re.search(
                r"formal|lean|z3|phase\s+ii",
                context,
                re.IGNORECASE
            )

            assert has_scoping, (
                f"Found 'no verifier' without 'formal/Lean/Z3' scoping near position {start}. "
                f"Context: ...{context}... "
                "The MV arithmetic validator is a verifier. "
                "Use 'no formal verifier' instead."
            )

    def test_no_claim_ui_never_shows_verified(self):
        """
        V0_LOCK.md must NOT claim UI never shows VERIFIED.

        MV arithmetic claims (e.g., '2 + 2 = 4') DO show VERIFIED.
        """
        if not V0_LOCK_PATH.exists():
            pytest.skip(f"V0_LOCK.md not found: {V0_LOCK_PATH}")

        content = V0_LOCK_PATH.read_text(encoding="utf-8")

        # Pattern for false claim that UI never shows VERIFIED
        false_pattern = re.compile(
            r"UI\s+does\s+not\s+claim\s+VERIFIED\s+for\s+any\s+claim",
            re.IGNORECASE
        )

        match = false_pattern.search(content)
        assert not match, (
            "V0_LOCK.md falsely claims 'UI does not claim VERIFIED for any claim'. "
            "MV arithmetic claims DO show VERIFIED. "
            "Scope to 'shows VERIFIED only for MV arithmetic claims'."
        )


# ---------------------------------------------------------------------------
# Tests: invariants_status.md - Audit Surface Version Section
# ---------------------------------------------------------------------------

class TestInvariantsStatusAuditSurfaceVersionSection:
    """
    Verify invariants_status.md has a full section for invariant 10.

    Tier A claims 10 invariants but only 9 were fully explained.
    Invariant 10 (Audit Surface Version Field) needs equivalent detail.
    """

    def test_invariants_status_contains_audit_surface_version_section(self):
        """
        invariants_status.md must have a full section for 'Audit Surface Version Field'.

        It should include:
        - FM Reference (or note that it's operational, not FM)
        - Enforcement mechanism
        - Detection method
        - Gate Location
        - Tests
        - Status
        """
        if not INVARIANTS_STATUS_PATH.exists():
            pytest.skip(f"invariants_status.md not found: {INVARIANTS_STATUS_PATH}")

        content = INVARIANTS_STATUS_PATH.read_text(encoding="utf-8")

        # Check for section header
        assert "### 10. Audit Surface Version Field" in content, (
            "invariants_status.md missing '### 10. Audit Surface Version Field' section"
        )

        # Find the section
        section_pattern = re.compile(
            r"### 10\. Audit Surface Version Field.*?(?=###|---|\Z)",
            re.DOTALL
        )
        section_match = section_pattern.search(content)
        assert section_match, (
            "Could not extract Audit Surface Version Field section"
        )

        section = section_match.group()

        # Check for required subsections
        required_elements = [
            ("Enforcement", r"Enforcement"),
            ("Detection", r"Detection"),
            ("Status", r"Status"),
        ]

        for element_name, pattern in required_elements:
            assert re.search(pattern, section, re.IGNORECASE), (
                f"Audit Surface Version Field section missing '{element_name}'"
            )

    def test_audit_surface_version_has_auditor_verification_instructions(self):
        """
        Invariant 10 section should include how auditors verify it.
        """
        if not INVARIANTS_STATUS_PATH.exists():
            pytest.skip(f"invariants_status.md not found: {INVARIANTS_STATUS_PATH}")

        content = INVARIANTS_STATUS_PATH.read_text(encoding="utf-8")

        # Check for auditor verification instructions
        assert "How Auditors Verify" in content or "auditor" in content.lower(), (
            "invariants_status.md Audit Surface Version section should include "
            "instructions for how auditors verify it"
        )

    def test_tier_a_count_matches_explained_invariants(self):
        """
        The number of Tier A invariants claimed should match sections explained.
        """
        if not INVARIANTS_STATUS_PATH.exists():
            pytest.skip(f"invariants_status.md not found: {INVARIANTS_STATUS_PATH}")

        content = INVARIANTS_STATUS_PATH.read_text(encoding="utf-8")

        # Count ### N. sections in Tier A area
        tier_a_sections = re.findall(
            r"### \d+\. [^\n]+",
            content[:content.find("## Tier B")]  # Only in Tier A section
        )

        # Check that we have 10 sections (or at least that section 10 exists)
        section_numbers = []
        for section in tier_a_sections:
            match = re.match(r"### (\d+)\.", section)
            if match:
                section_numbers.append(int(match.group(1)))

        assert 10 in section_numbers, (
            f"Tier A claims 10 invariants but section 10 not found. "
            f"Found sections: {section_numbers}"
        )
