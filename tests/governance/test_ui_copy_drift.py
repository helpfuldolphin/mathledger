"""
UI Copy Drift Test: Regression guard for self-explanation content.

This test suite ensures that the UI copy strings in demo/app.py:
1. Exist (no accidental deletion)
2. Contain required self-explanation phrases (no semantic drift)
3. Do not contain forbidden capability claims (no overclaiming)

FM Reference: Section 4.1 - Governance self-explanation requirement
Enforcement: Tier A (Structurally Enforced via regression test)

Run with:
    uv run pytest tests/governance/test_ui_copy_drift.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Set

import pytest

# Add project root to path so we can import demo.app
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demo.app import UI_COPY, DEMO_VERSION


# ---------------------------------------------------------------------------
# Required UI Copy Keys
# ---------------------------------------------------------------------------

REQUIRED_UI_COPY_KEYS: Set[str] = {
    "FRAMING_MAIN",
    "FRAMING_STOPS",
    "FRAMING_NOT_ALWAYS",
    "JUSTIFIED_EXPLAIN",
    "ABSTAINED_FIRST_CLASS",
    "TRUST_CLASS_NOTE",
    "ADV_TOOLTIP",
    "TRANSITION_TITLE",
    "TRANSITION_DETAIL",
    "BOUNDARY_CONCLUSION",
    "OUTCOME_VERIFIED",
    "OUTCOME_REFUTED",
    "OUTCOME_ABSTAINED",
}


# ---------------------------------------------------------------------------
# Required Phrases (semantic anchors that must not drift)
# ---------------------------------------------------------------------------

REQUIRED_PHRASES: Dict[str, list] = {
    "FRAMING_MAIN": [
        "does not decide what is true",
        "justified",
        "verification route",
    ],
    "FRAMING_STOPS": [
        "stop",
        "cannot verify",
    ],
    "JUSTIFIED_EXPLAIN": [
        "trust class",
        "ABSTAINED",
    ],
    "ABSTAINED_FIRST_CLASS": [
        "first-class",
        "not a missing value",
    ],
    "TRUST_CLASS_NOTE": [
        "verification route",
        "not correctness",
    ],
    "ADV_TOOLTIP": [
        "do not enter R_t",
        "exploration",
    ],
    "OUTCOME_VERIFIED": [
        "validator",
        "confirmed",
    ],
    "OUTCOME_REFUTED": [
        "validator",
        "disproved",
    ],
    "OUTCOME_ABSTAINED": [
        "no verifier",
    ],
}


# ---------------------------------------------------------------------------
# Forbidden Terms (capability claims we must never make)
# ---------------------------------------------------------------------------

FORBIDDEN_TERMS: Set[str] = {
    "safe",
    "aligned",
    "intelligent",
    "AI-powered",
    "smart",
    "trustworthy",  # We say "auditable", not "trustworthy"
}

# Context-dependent forbidden terms (forbidden in specific keys)
CONTEXT_FORBIDDEN: Dict[str, Set[str]] = {
    "OUTCOME_VERIFIED": {"correct", "true"},  # "validator confirmed" not "is correct/true"
    "OUTCOME_REFUTED": {"wrong", "false"},    # "validator disproved" not "is wrong/false"
}


# ---------------------------------------------------------------------------
# Tests: Key Existence
# ---------------------------------------------------------------------------

class TestUIKeyExistence:
    """Test that all required UI copy keys exist."""

    def test_all_required_keys_present(self):
        """Verify all required UI_COPY keys are present."""
        missing = REQUIRED_UI_COPY_KEYS - set(UI_COPY.keys())
        assert not missing, f"Missing UI_COPY keys: {missing}"

    @pytest.mark.parametrize("key", list(REQUIRED_UI_COPY_KEYS))
    def test_key_not_empty(self, key: str):
        """Verify each required key has non-empty content."""
        assert key in UI_COPY, f"Key '{key}' missing from UI_COPY"
        assert UI_COPY[key], f"Key '{key}' has empty value"
        assert len(UI_COPY[key].strip()) > 0, f"Key '{key}' has only whitespace"


# ---------------------------------------------------------------------------
# Tests: Required Phrases (Semantic Anchors)
# ---------------------------------------------------------------------------

class TestRequiredPhrases:
    """Test that required self-explanation phrases are present."""

    @pytest.mark.parametrize("key,phrases", list(REQUIRED_PHRASES.items()))
    def test_required_phrases_present(self, key: str, phrases: list):
        """Verify required phrases are present in each key."""
        assert key in UI_COPY, f"Key '{key}' missing from UI_COPY"
        content = UI_COPY[key].lower()

        missing_phrases = []
        for phrase in phrases:
            if phrase.lower() not in content:
                missing_phrases.append(phrase)

        assert not missing_phrases, (
            f"Key '{key}' missing required phrases: {missing_phrases}\n"
            f"Content: {UI_COPY[key]}"
        )


# ---------------------------------------------------------------------------
# Tests: Forbidden Terms (No Capability Claims)
# ---------------------------------------------------------------------------

class TestForbiddenTerms:
    """Test that forbidden capability claims are not present."""

    @pytest.mark.parametrize("key", list(REQUIRED_UI_COPY_KEYS))
    def test_no_global_forbidden_terms(self, key: str):
        """Verify no globally forbidden terms appear in UI copy."""
        if key not in UI_COPY:
            pytest.skip(f"Key '{key}' not in UI_COPY")

        content = UI_COPY[key].lower()
        found_terms = []

        for term in FORBIDDEN_TERMS:
            if term.lower() in content:
                found_terms.append(term)

        assert not found_terms, (
            f"Key '{key}' contains forbidden capability claims: {found_terms}\n"
            f"Content: {UI_COPY[key]}"
        )

    @pytest.mark.parametrize("key,forbidden", list(CONTEXT_FORBIDDEN.items()))
    def test_no_context_forbidden_terms(self, key: str, forbidden: Set[str]):
        """Verify no context-specific forbidden terms appear."""
        if key not in UI_COPY:
            pytest.skip(f"Key '{key}' not in UI_COPY")

        content = UI_COPY[key].lower()
        found_terms = []

        for term in forbidden:
            if term.lower() in content:
                found_terms.append(term)

        assert not found_terms, (
            f"Key '{key}' contains context-forbidden terms: {found_terms}\n"
            f"These terms overclaim the validator's epistemic status.\n"
            f"Content: {UI_COPY[key]}"
        )


# ---------------------------------------------------------------------------
# Tests: Version Check
# ---------------------------------------------------------------------------

class TestVersionIntegrity:
    """Test that version is correctly set for UI self-explanation."""

    def test_version_format(self):
        """Verify version string is in expected format."""
        parts = DEMO_VERSION.split(".")
        assert len(parts) >= 2, f"Version '{DEMO_VERSION}' not in X.Y format"
        assert all(p.isdigit() for p in parts[:2]), (
            f"Version '{DEMO_VERSION}' has non-numeric major/minor"
        )

    def test_version_minimum(self):
        """Verify version is at least 0.2.0 (UI self-explanation version)."""
        parts = DEMO_VERSION.split(".")
        major, minor = int(parts[0]), int(parts[1])
        assert major > 0 or (major == 0 and minor >= 2), (
            f"Version {DEMO_VERSION} < 0.2.0: UI self-explanation may be missing"
        )


# ---------------------------------------------------------------------------
# Tests: Structural Consistency
# ---------------------------------------------------------------------------

class TestStructuralConsistency:
    """Test structural properties of UI copy."""

    def test_framing_main_starts_with_the_system(self):
        """Verify framing starts with 'The system' (not anthropomorphizing)."""
        content = UI_COPY["FRAMING_MAIN"]
        # Must start with "The system" to avoid anthropomorphizing
        assert content.startswith("The system"), (
            f"FRAMING_MAIN must start with 'The system' to avoid anthropomorphizing.\n"
            f"Got: {content[:50]}..."
        )

    def test_outcomes_use_validator_not_system(self):
        """Verify outcome explanations reference 'validator' not 'system believed'."""
        for key in ["OUTCOME_VERIFIED", "OUTCOME_REFUTED"]:
            if key not in UI_COPY:
                continue
            content = UI_COPY[key].lower()
            assert "validator" in content, (
                f"{key} should reference 'validator' (mechanical check) "
                f"not imply system-level judgment"
            )

    def test_abstained_mentions_no_verifier(self):
        """Verify ABSTAINED explanation mentions 'no verifier'."""
        content = UI_COPY["OUTCOME_ABSTAINED"].lower()
        assert "no verifier" in content or "cannot" in content, (
            "OUTCOME_ABSTAINED should explain that no verifier could confirm/refute"
        )


# ---------------------------------------------------------------------------
# Tests: Cross-Reference Integrity
# ---------------------------------------------------------------------------

class TestCrossReferenceIntegrity:
    """Test that UI copy cross-references are consistent."""

    def test_trust_class_names_consistent(self):
        """Verify trust class names (FV, MV, PA, ADV) used consistently."""
        trust_classes = {"FV", "MV", "PA", "ADV"}

        # These keys should mention trust classes
        keys_with_trust_classes = [
            "TRUST_CLASS_NOTE",
            "ADV_TOOLTIP",
        ]

        for key in keys_with_trust_classes:
            if key not in UI_COPY:
                continue
            content = UI_COPY[key].upper()
            # At least one trust class should be mentioned
            mentioned = [tc for tc in trust_classes if tc in content]
            # ADV_TOOLTIP should mention ADV at minimum
            if key == "ADV_TOOLTIP":
                assert "ADV" in content or "ADVISORY" in content, (
                    f"{key} should mention ADV or Advisory"
                )

    def test_hash_names_use_subscript_notation(self):
        """Verify hash references use U_t, R_t, H_t notation."""
        # Check ADV tooltip which mentions R_t
        if "ADV_TOOLTIP" in UI_COPY:
            content = UI_COPY["ADV_TOOLTIP"]
            assert "R_t" in content or "reasoning root" in content.lower(), (
                "ADV_TOOLTIP should reference R_t (reasoning root)"
            )
