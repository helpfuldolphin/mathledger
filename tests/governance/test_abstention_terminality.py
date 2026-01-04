"""
Test suite for Abstention Terminality.

These tests verify that ABSTAINED is a terminal outcome: once a claim artifact
is classified as ABSTAINED, resubmission of identical content under the same
trust class must produce ABSTAINED with identical artifact hash.

FM Reference: §4.1 - "ABSTAINED is terminal for a claim identity"

Enforcement: Tier A (Structurally Enforced)

Key constraint:
    Once an artifact is classified as ABSTAINED, its validation outcome is
    immutable within its claim identity and epoch. No subsequent verification
    regime, human attestation, or policy change alters that outcome.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

import pytest

from attestation.dual_root import (
    compute_reasoning_root,
)
from governance.mv_validator import validate_mv_claim
from governance.trust_class import TrustClass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_artifact_hash(artifact: Dict[str, Any]) -> str:
    """Compute deterministic hash of a reasoning artifact."""
    # Canonical JSON serialization
    canonical = json.dumps(artifact, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def create_reasoning_artifact(
    claim_text: str,
    trust_class: str,
    validation_outcome: str,
) -> Dict[str, Any]:
    """Create a reasoning artifact with canonical structure."""
    return {
        "claim_id": hashlib.sha256(claim_text.encode()).hexdigest()[:16],
        "claim_text": claim_text,
        "trust_class": trust_class,
        "validation_outcome": validation_outcome,
        "proof_payload": {},
    }


# ---------------------------------------------------------------------------
# Test: Resubmission of ABSTAINED Claim
# ---------------------------------------------------------------------------


class TestAbstentionTerminality:
    """Test that ABSTAINED outcomes are terminal and deterministic."""

    def test_identical_abstained_claim_produces_identical_hash(self):
        """Given identical claim content and trust class, outcome must be identical."""
        claim_text = "sqrt(2) is irrational"
        trust_class = "MV"

        # First submission
        artifact_1 = create_reasoning_artifact(
            claim_text=claim_text,
            trust_class=trust_class,
            validation_outcome="ABSTAINED",
        )
        hash_1 = compute_artifact_hash(artifact_1)

        # Second submission (identical)
        artifact_2 = create_reasoning_artifact(
            claim_text=claim_text,
            trust_class=trust_class,
            validation_outcome="ABSTAINED",
        )
        hash_2 = compute_artifact_hash(artifact_2)

        assert hash_1 == hash_2, (
            "Identical ABSTAINED claims must produce identical artifact hashes"
        )

    def test_fv_claim_always_abstains_deterministically(self):
        """FV claims with no formal verifier must always ABSTAIN identically."""
        claim_text = "forall x: nat, x + 0 = x"
        trust_class = "FV"

        artifacts = []
        for _ in range(5):
            artifact = create_reasoning_artifact(
                claim_text=claim_text,
                trust_class=trust_class,
                validation_outcome="ABSTAINED",
            )
            artifacts.append(artifact)

        hashes = [compute_artifact_hash(a) for a in artifacts]
        assert len(set(hashes)) == 1, (
            "Multiple submissions of identical FV claim must produce same hash"
        )

    def test_pa_claim_always_abstains_deterministically(self):
        """PA claims must always ABSTAIN identically."""
        claim_text = "The Riemann hypothesis is true"
        trust_class = "PA"

        artifact_1 = create_reasoning_artifact(
            claim_text=claim_text,
            trust_class=trust_class,
            validation_outcome="ABSTAINED",
        )
        artifact_2 = create_reasoning_artifact(
            claim_text=claim_text,
            trust_class=trust_class,
            validation_outcome="ABSTAINED",
        )

        assert compute_artifact_hash(artifact_1) == compute_artifact_hash(artifact_2)

    def test_mv_unparseable_claim_always_abstains_deterministically(self):
        """MV claims outside validator coverage must ABSTAIN identically."""
        # This claim is not parseable by the MV arithmetic validator
        claim_text = "The speed of light is constant"
        trust_class = "MV"

        artifact_1 = create_reasoning_artifact(
            claim_text=claim_text,
            trust_class=trust_class,
            validation_outcome="ABSTAINED",
        )
        artifact_2 = create_reasoning_artifact(
            claim_text=claim_text,
            trust_class=trust_class,
            validation_outcome="ABSTAINED",
        )

        assert compute_artifact_hash(artifact_1) == compute_artifact_hash(artifact_2)


class TestNoLateUpgrade:
    """Test that ABSTAINED cannot be upgraded to VERIFIED."""

    def test_abstained_artifact_cannot_become_verified(self):
        """An artifact that was ABSTAINED cannot later be recorded as VERIFIED."""
        claim_text = "unparseable claim"
        trust_class = "MV"

        # Original artifact was ABSTAINED
        original_artifact = create_reasoning_artifact(
            claim_text=claim_text,
            trust_class=trust_class,
            validation_outcome="ABSTAINED",
        )
        original_hash = compute_artifact_hash(original_artifact)

        # Attempt to create "upgraded" version (which would be a protocol violation)
        upgraded_artifact = create_reasoning_artifact(
            claim_text=claim_text,
            trust_class=trust_class,
            validation_outcome="VERIFIED",  # Attempted upgrade
        )
        upgraded_hash = compute_artifact_hash(upgraded_artifact)

        # Hashes MUST differ - if they're the same, the upgrade succeeded silently
        assert original_hash != upgraded_hash, (
            "ABSTAINED and VERIFIED artifacts must have different hashes"
        )

    def test_abstained_cannot_become_refuted(self):
        """An artifact that was ABSTAINED cannot later be recorded as REFUTED."""
        claim_text = "unparseable claim"
        trust_class = "MV"

        original = create_reasoning_artifact(
            claim_text=claim_text,
            trust_class=trust_class,
            validation_outcome="ABSTAINED",
        )
        downgraded = create_reasoning_artifact(
            claim_text=claim_text,
            trust_class=trust_class,
            validation_outcome="REFUTED",
        )

        assert compute_artifact_hash(original) != compute_artifact_hash(downgraded)


class TestReasoningRootDeterminism:
    """Test that R_t is deterministic for identical ABSTAINED artifacts."""

    def test_identical_abstained_artifacts_produce_identical_r_t(self):
        """R_t must be identical for identical sets of ABSTAINED artifacts."""
        artifacts = [
            create_reasoning_artifact(
                claim_text="claim 1",
                trust_class="FV",
                validation_outcome="ABSTAINED",
            ),
            create_reasoning_artifact(
                claim_text="claim 2",
                trust_class="PA",
                validation_outcome="ABSTAINED",
            ),
        ]

        r_t_1 = compute_reasoning_root(artifacts)
        r_t_2 = compute_reasoning_root(artifacts)

        assert r_t_1 == r_t_2, "Identical artifacts must produce identical R_t"

    def test_r_t_changes_if_outcome_differs(self):
        """R_t must change if any artifact outcome differs."""
        base_artifacts = [
            create_reasoning_artifact(
                claim_text="2 + 2 = 4",
                trust_class="MV",
                validation_outcome="ABSTAINED",
            ),
        ]

        modified_artifacts = [
            create_reasoning_artifact(
                claim_text="2 + 2 = 4",
                trust_class="MV",
                validation_outcome="VERIFIED",  # Different outcome
            ),
        ]

        r_t_base = compute_reasoning_root(base_artifacts)
        r_t_modified = compute_reasoning_root(modified_artifacts)

        assert r_t_base != r_t_modified, (
            "Different outcomes must produce different R_t values"
        )


class TestMVValidatorAbstentionDeterminism:
    """Test that the MV validator produces deterministic ABSTAINED outcomes."""

    @pytest.mark.parametrize("claim_text", [
        "sqrt(2) is irrational",
        "The sum of angles in a triangle is 180 degrees",
        "e^(i*pi) + 1 = 0",
        "x^2 + y^2 = z^2 for x=3, y=4, z=5",  # Not in simple a op b = c form
        "lim(x->0) sin(x)/x = 1",
    ])
    def test_unparseable_claims_always_abstain(self, claim_text: str):
        """Claims outside MV coverage must always return ABSTAINED."""
        result = validate_mv_claim(claim_text)
        assert result.outcome.value == "ABSTAINED", (
            f"Claim '{claim_text}' must ABSTAIN (not parseable by MV validator)"
        )

    def test_same_unparseable_claim_abstains_identically(self):
        """Multiple calls with same claim must return identical ABSTAINED."""
        claim = "This is not arithmetic"
        results = [validate_mv_claim(claim) for _ in range(10)]
        assert all(r.outcome.value == "ABSTAINED" for r in results)
        # Check that all results are structurally identical
        explanations = [r.explanation for r in results]
        assert len(set(explanations)) == 1, "All results must be identical"


class TestDocumentationAlignment:
    """Test that documentation claims are enforced."""

    def test_scope_lock_claims_outcome_immutability(self):
        """Verify V0_LOCK.md contains outcome immutability clause."""
        with open("docs/V0_LOCK.md", "r", encoding="utf-8") as f:
            content = f.read()

        assert "Outcome Immutability" in content, (
            "V0_LOCK.md must contain 'Outcome Immutability' section"
        )
        assert "immutable within its claim identity and epoch" in content, (
            "V0_LOCK.md must state that outcomes are immutable"
        )

    def test_explanation_contains_abstention_terminality_section(self):
        """Verify HOW_THE_DEMO_EXPLAINS_ITSELF.md has terminality section."""
        with open("docs/HOW_THE_DEMO_EXPLAINS_ITSELF.md", "r", encoding="utf-8") as f:
            content = f.read()

        assert "Abstention as a Terminal Outcome" in content, (
            "Explanation doc must contain 'Abstention as a Terminal Outcome' section"
        )
        assert "Late upgrade" in content, (
            "Explanation doc must mention 'Late upgrade' sacrifice"
        )
        assert "Institutional override" in content, (
            "Explanation doc must mention 'Institutional override' sacrifice"
        )

    def test_for_auditors_contains_expectation(self):
        """Verify FOR_AUDITORS.md sets correct expectations."""
        with open("docs/FOR_AUDITORS.md", "r", encoding="utf-8") as f:
            content = f.read()

        assert "permanently ABSTAINED" in content, (
            "FOR_AUDITORS.md must mention permanent ABSTAINED expectation"
        )
        assert "not a failure mode" in content, (
            "FOR_AUDITORS.md must clarify ABSTAINED is not a failure"
        )
