"""
Tests for Terminology Consistency (Epistemic Integrity Guard)

Verifies:
1. "machine-checkable proof" is never used in v0.x docs without Lean/Phase II context
2. VERIFIED definition does not overclaim formal proof semantics
3. MV validator coverage limitations are documented

This prevents terminology drift that could mislead auditors about v0 capabilities.

Run with:
    uv run pytest tests/governance/test_terminology_guard.py -v
"""

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).parent.parent.parent
DOCS_DIR = REPO_ROOT / "docs"

# Directories to exclude from terminology checks (external content we don't control)
EXCLUDED_DIRS = [
    "external_audits",  # External auditor observations
    "PAPERS",           # Academic papers with different audience
    "field_manual",     # FM is source of truth, not controlled by this guard
]

# Pattern to find "machine-checkable proof" (case insensitive)
MACHINE_CHECKABLE_PATTERN = re.compile(r"machine-checkable\s+proof", re.IGNORECASE)

# Patterns that indicate proper scoping (must appear near the phrase)
SCOPING_PATTERNS = [
    re.compile(r"lean", re.IGNORECASE),
    re.compile(r"z3", re.IGNORECASE),
    re.compile(r"phase\s*(ii|2|two)", re.IGNORECASE),
    re.compile(r"FV\s*\(", re.IGNORECASE),  # "FV (" indicates trust class context
    re.compile(r"formally\s+verified", re.IGNORECASE),
    re.compile(r"not\s+implemented", re.IGNORECASE),
    re.compile(r"abstained", re.IGNORECASE),
]

# Window size for context check (characters before and after)
CONTEXT_WINDOW = 500


def get_docs_to_check() -> list[Path]:
    """Get all markdown docs that should be checked for terminology."""
    docs = []
    for md_file in DOCS_DIR.rglob("*.md"):
        # Check if file is in excluded directory
        relative_parts = md_file.relative_to(DOCS_DIR).parts
        if any(excluded in relative_parts for excluded in EXCLUDED_DIRS):
            continue
        docs.append(md_file)
    return docs


def find_unscoped_machine_checkable(content: str, filepath: Path) -> list[dict]:
    """
    Find occurrences of 'machine-checkable proof' without proper scoping.

    Returns list of violations with context.
    """
    violations = []

    for match in MACHINE_CHECKABLE_PATTERN.finditer(content):
        start = match.start()
        end = match.end()

        # Extract context window
        context_start = max(0, start - CONTEXT_WINDOW)
        context_end = min(len(content), end + CONTEXT_WINDOW)
        context = content[context_start:context_end]

        # Check if any scoping pattern appears in context
        has_scoping = any(
            pattern.search(context) for pattern in SCOPING_PATTERNS
        )

        if not has_scoping:
            # Find line number
            line_num = content[:start].count('\n') + 1
            violations.append({
                "file": str(filepath),
                "line": line_num,
                "match": match.group(),
                "context": context[:200] + "..." if len(context) > 200 else context,
            })

    return violations


# ---------------------------------------------------------------------------
# Tests: Machine-Checkable Proof Scoping
# ---------------------------------------------------------------------------

class TestMachineCheckableProofScoping:
    """Verify 'machine-checkable proof' is properly scoped to Lean/Phase II."""

    def test_no_unscoped_machine_checkable_proof(self):
        """
        'machine-checkable proof' must not appear in v0.x docs without
        Lean/Phase II/FV context within ~500 characters.
        """
        all_violations = []

        for doc_path in get_docs_to_check():
            content = doc_path.read_text(encoding="utf-8")
            violations = find_unscoped_machine_checkable(content, doc_path)
            all_violations.extend(violations)

        if all_violations:
            msg_parts = [
                "Found 'machine-checkable proof' without Lean/Phase II scoping:",
                ""
            ]
            for v in all_violations:
                msg_parts.append(f"  {v['file']}:{v['line']}")
                msg_parts.append(f"    Context: {v['context'][:100]}...")
                msg_parts.append("")

            pytest.fail("\n".join(msg_parts))

    def test_docs_directory_exists(self):
        """Docs directory must exist."""
        assert DOCS_DIR.exists(), f"Docs directory not found: {DOCS_DIR}"

    def test_at_least_one_doc_checked(self):
        """At least one doc should be checked (sanity check)."""
        docs = get_docs_to_check()
        assert len(docs) > 0, "No docs found to check"


# ---------------------------------------------------------------------------
# Tests: VERIFIED Definition Accuracy
# ---------------------------------------------------------------------------

class TestVerifiedDefinitionAccuracy:
    """Verify VERIFIED is defined accurately in key docs."""

    def test_how_demo_explains_itself_verified_definition(self):
        """
        HOW_THE_DEMO_EXPLAINS_ITSELF.md must define VERIFIED as
        MV arithmetic validator, not formal proof.
        """
        doc_path = DOCS_DIR / "HOW_THE_DEMO_EXPLAINS_ITSELF.md"
        if not doc_path.exists():
            pytest.skip(f"Doc not found: {doc_path}")

        content = doc_path.read_text(encoding="utf-8")

        # Must mention "MV arithmetic validator" or similar
        assert re.search(r"MV\s+(arithmetic\s+)?validator", content, re.IGNORECASE), (
            "VERIFIED definition must reference MV arithmetic validator"
        )

        # Must NOT say VERIFIED means "machine-checkable proof" without context
        # (The old definition was: "VERIFIED means the system found a machine-checkable proof")
        bad_pattern = re.compile(
            r"VERIFIED.*means.*machine-checkable\s+proof",
            re.IGNORECASE | re.DOTALL
        )
        match = bad_pattern.search(content)
        assert not match, (
            "VERIFIED must not be defined as 'machine-checkable proof' "
            "(that's FV/Lean terminology)"
        )

    def test_verified_mentions_limited_coverage(self):
        """VERIFIED definition should mention limited coverage."""
        doc_path = DOCS_DIR / "HOW_THE_DEMO_EXPLAINS_ITSELF.md"
        if not doc_path.exists():
            pytest.skip(f"Doc not found: {doc_path}")

        content = doc_path.read_text(encoding="utf-8")

        # Should mention coverage limitations
        assert "limited" in content.lower() or "coverage" in content.lower(), (
            "VERIFIED definition should mention limited coverage"
        )


# ---------------------------------------------------------------------------
# Tests: Validator Coverage Documentation
# ---------------------------------------------------------------------------

class TestValidatorCoverageDocumentation:
    """Verify MV validator coverage is properly documented."""

    def test_validator_coverage_section_exists(self):
        """There must be a Validator Coverage section."""
        doc_path = DOCS_DIR / "HOW_THE_DEMO_EXPLAINS_ITSELF.md"
        if not doc_path.exists():
            pytest.skip(f"Doc not found: {doc_path}")

        content = doc_path.read_text(encoding="utf-8")

        assert "Validator Coverage" in content, (
            "HOW_THE_DEMO_EXPLAINS_ITSELF.md must have a 'Validator Coverage' section"
        )

    def test_validator_coverage_mentions_integers(self):
        """Validator coverage should specify integer arithmetic."""
        doc_path = DOCS_DIR / "HOW_THE_DEMO_EXPLAINS_ITSELF.md"
        if not doc_path.exists():
            pytest.skip(f"Doc not found: {doc_path}")

        content = doc_path.read_text(encoding="utf-8")

        # Should mention integers or "a op b = c"
        assert "integer" in content.lower() or "a op b" in content, (
            "Validator coverage should specify integer arithmetic"
        )

    def test_validator_coverage_mentions_not_covered(self):
        """Validator coverage should list what's NOT covered."""
        doc_path = DOCS_DIR / "HOW_THE_DEMO_EXPLAINS_ITSELF.md"
        if not doc_path.exists():
            pytest.skip(f"Doc not found: {doc_path}")

        content = doc_path.read_text(encoding="utf-8")

        # Should mention things that are not covered
        not_covered_indicators = [
            "not covered",
            "float",
            "overflow",
            "division by zero",
        ]
        found = any(ind in content.lower() for ind in not_covered_indicators)
        assert found, (
            "Validator coverage should mention what's NOT covered "
            "(floats, overflow, division by zero)"
        )


# ---------------------------------------------------------------------------
# Tests: Terminology in System Boundary Memo
# ---------------------------------------------------------------------------

class TestSystemBoundaryMemoTerminology:
    """Verify V0_SYSTEM_BOUNDARY_MEMO.md has correct terminology."""

    def test_fv_properly_scoped_to_lean(self):
        """FV definition should mention Lean/Z3/Phase II."""
        doc_path = DOCS_DIR / "V0_SYSTEM_BOUNDARY_MEMO.md"
        if not doc_path.exists():
            pytest.skip(f"Doc not found: {doc_path}")

        content = doc_path.read_text(encoding="utf-8")

        # FV row should mention Lean/Phase II
        fv_pattern = re.compile(r"FV.*machine-checkable.*(?:Lean|Z3|Phase)", re.IGNORECASE)
        assert fv_pattern.search(content), (
            "FV definition with 'machine-checkable' should mention Lean/Z3/Phase II"
        )

    def test_mv_mentions_limited_coverage(self):
        """MV definition should mention limited coverage."""
        doc_path = DOCS_DIR / "V0_SYSTEM_BOUNDARY_MEMO.md"
        if not doc_path.exists():
            pytest.skip(f"Doc not found: {doc_path}")

        content = doc_path.read_text(encoding="utf-8")

        # MV definition should mention limited coverage
        assert "limited coverage" in content.lower() or "limited" in content.lower(), (
            "MV definition should mention limited coverage"
        )

    def test_terminology_note_exists(self):
        """There should be a terminology clarification note."""
        doc_path = DOCS_DIR / "V0_SYSTEM_BOUNDARY_MEMO.md"
        if not doc_path.exists():
            pytest.skip(f"Doc not found: {doc_path}")

        content = doc_path.read_text(encoding="utf-8")

        assert "terminology" in content.lower(), (
            "V0_SYSTEM_BOUNDARY_MEMO.md should have a terminology note"
        )
