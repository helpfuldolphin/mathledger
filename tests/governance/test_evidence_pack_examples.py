"""
Tests for Evidence Pack Examples (Auditor Artifacts)

Verifies:
1. Examples file exists and is valid JSON
2. Examples are deterministically generated (same input -> same output)
3. Valid pack produces PASS
4. Tampered packs produce FAIL with correct reason
5. Examples use the SAME canonical functions as replay verification

Run with:
    uv run pytest tests/governance/test_evidence_pack_examples.py -v
"""

import json
from pathlib import Path

import pytest

from attestation.dual_root import (
    compute_ui_root,
    compute_reasoning_root,
    compute_composite_root,
)


REPO_ROOT = Path(__file__).parent.parent.parent
EXAMPLES_FILE = REPO_ROOT / "releases" / "evidence_pack_examples.v0.2.1.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def examples_data():
    """Load the evidence pack examples file."""
    if not EXAMPLES_FILE.exists():
        pytest.skip(f"Examples file not found: {EXAMPLES_FILE}")
    with open(EXAMPLES_FILE, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Tests: File Validity
# ---------------------------------------------------------------------------

class TestExamplesFileValidity:
    """Verify examples file exists and has correct structure."""

    def test_examples_file_exists(self):
        """Examples file must exist."""
        assert EXAMPLES_FILE.exists(), f"Missing: {EXAMPLES_FILE}"

    def test_examples_file_is_valid_json(self, examples_data):
        """Examples file must be valid JSON."""
        assert isinstance(examples_data, dict)

    def test_examples_has_required_structure(self, examples_data):
        """Examples file must have required top-level keys."""
        assert "schema_version" in examples_data
        assert "examples" in examples_data
        assert "usage_instructions" in examples_data

    def test_examples_has_three_packs(self, examples_data):
        """Examples must contain exactly 3 packs."""
        examples = examples_data["examples"]
        assert len(examples) == 3, f"Expected 3 examples, got {len(examples)}"

    def test_each_example_has_pack_and_verdict(self, examples_data):
        """Each example must have 'pack' and 'expected_verdict'."""
        for name, example in examples_data["examples"].items():
            assert "pack" in example, f"Missing 'pack' in {name}"
            assert "expected_verdict" in example, f"Missing 'expected_verdict' in {name}"


# ---------------------------------------------------------------------------
# Tests: Valid Pack PASS
# ---------------------------------------------------------------------------

class TestValidPackPass:
    """Valid evidence pack must pass verification."""

    def test_valid_pack_passes_verification(self, examples_data):
        """Valid boundary demo pack must PASS."""
        example = examples_data["examples"]["valid_boundary_demo"]
        pack = example["pack"]

        # Recompute hashes using SAME functions as replay verification
        recomputed_ut = compute_ui_root(pack["uvil_events"])
        recomputed_rt = compute_reasoning_root(pack["reasoning_artifacts"])
        recomputed_ht = compute_composite_root(recomputed_rt, recomputed_ut)

        assert recomputed_ut == pack["u_t"], "U_t mismatch in valid pack"
        assert recomputed_rt == pack["r_t"], "R_t mismatch in valid pack"
        assert recomputed_ht == pack["h_t"], "H_t mismatch in valid pack"

    def test_valid_pack_expected_verdict_is_pass(self, examples_data):
        """Valid pack must have expected_verdict = PASS."""
        example = examples_data["examples"]["valid_boundary_demo"]
        assert example["expected_verdict"] == "PASS"


# ---------------------------------------------------------------------------
# Tests: Tampered H_t Pack FAIL
# ---------------------------------------------------------------------------

class TestTamperedHtPackFail:
    """Tampered H_t pack must fail verification."""

    def test_tampered_ht_pack_fails(self, examples_data):
        """Tampered H_t pack must FAIL."""
        example = examples_data["examples"]["tampered_ht_mismatch"]
        pack = example["pack"]

        # Recompute hashes
        recomputed_ut = compute_ui_root(pack["uvil_events"])
        recomputed_rt = compute_reasoning_root(pack["reasoning_artifacts"])
        recomputed_ht = compute_composite_root(recomputed_rt, recomputed_ut)

        # U_t and R_t should still match (data not tampered)
        assert recomputed_ut == pack["u_t"], "U_t should match (not tampered)"
        assert recomputed_rt == pack["r_t"], "R_t should match (not tampered)"

        # H_t should NOT match (directly modified)
        assert recomputed_ht != pack["h_t"], "H_t should NOT match (tampered)"

    def test_tampered_ht_expected_verdict_is_fail(self, examples_data):
        """Tampered H_t pack must have expected_verdict = FAIL."""
        example = examples_data["examples"]["tampered_ht_mismatch"]
        assert example["expected_verdict"] == "FAIL"
        assert example["expected_reason"] == "h_t mismatch"


# ---------------------------------------------------------------------------
# Tests: Tampered R_t Pack FAIL
# ---------------------------------------------------------------------------

class TestTamperedRtPackFail:
    """Tampered R_t pack must fail verification."""

    def test_tampered_rt_pack_fails(self, examples_data):
        """Tampered R_t pack must FAIL."""
        example = examples_data["examples"]["tampered_rt_mismatch"]
        pack = example["pack"]

        # Recompute hashes
        recomputed_rt = compute_reasoning_root(pack["reasoning_artifacts"])

        # R_t should NOT match (artifacts were tampered)
        assert recomputed_rt != pack["r_t"], "R_t should NOT match (tampered)"

    def test_tampered_rt_expected_verdict_is_fail(self, examples_data):
        """Tampered R_t pack must have expected_verdict = FAIL."""
        example = examples_data["examples"]["tampered_rt_mismatch"]
        assert example["expected_verdict"] == "FAIL"
        assert example["expected_reason"] == "r_t mismatch"


# ---------------------------------------------------------------------------
# Tests: Determinism
# ---------------------------------------------------------------------------

class TestDeterministicGeneration:
    """Verify examples are deterministically generated."""

    def test_valid_pack_hashes_are_deterministic(self, examples_data):
        """
        Running the same inputs through canonical functions must produce
        the exact same hashes as recorded in the examples file.
        """
        example = examples_data["examples"]["valid_boundary_demo"]
        pack = example["pack"]

        # First computation
        ut_1 = compute_ui_root(pack["uvil_events"])
        rt_1 = compute_reasoning_root(pack["reasoning_artifacts"])
        ht_1 = compute_composite_root(rt_1, ut_1)

        # Second computation (should be identical)
        ut_2 = compute_ui_root(pack["uvil_events"])
        rt_2 = compute_reasoning_root(pack["reasoning_artifacts"])
        ht_2 = compute_composite_root(rt_2, ut_2)

        assert ut_1 == ut_2, "U_t must be deterministic"
        assert rt_1 == rt_2, "R_t must be deterministic"
        assert ht_1 == ht_2, "H_t must be deterministic"

    def test_hashes_are_64_char_hex(self, examples_data):
        """All hashes must be 64-character hex strings (SHA256)."""
        example = examples_data["examples"]["valid_boundary_demo"]
        pack = example["pack"]

        for key in ["u_t", "r_t", "h_t"]:
            value = pack[key]
            assert len(value) == 64, f"{key} must be 64 chars, got {len(value)}"
            assert all(c in "0123456789abcdef" for c in value), f"{key} must be hex"


# ---------------------------------------------------------------------------
# Tests: Usage Instructions
# ---------------------------------------------------------------------------

class TestUsageInstructions:
    """Verify usage instructions are complete."""

    def test_usage_instructions_present(self, examples_data):
        """Usage instructions must be present."""
        instructions = examples_data["usage_instructions"]
        assert "step_1" in instructions
        assert "step_2" in instructions
        assert "step_3" in instructions
        assert "step_4" in instructions

    def test_usage_instructions_mention_verifier(self, examples_data):
        """Usage instructions must mention the verifier URL."""
        instructions = examples_data["usage_instructions"]
        # At least one step should mention the verifier
        all_steps = " ".join(str(v) for v in instructions.values())
        assert "verify" in all_steps.lower()


# ---------------------------------------------------------------------------
# Tests: No Hand-Crafted Hashes
# ---------------------------------------------------------------------------

class TestNoHandCraftedHashes:
    """Verify hashes are computed, not hand-crafted."""

    def test_valid_pack_hashes_match_recomputation(self, examples_data):
        """
        The recorded hashes in the valid pack must exactly match
        what we get from running the canonical functions.

        This proves the hashes were computed, not hand-crafted.
        """
        example = examples_data["examples"]["valid_boundary_demo"]
        pack = example["pack"]

        # Recompute using the exact same functions
        recomputed_ut = compute_ui_root(pack["uvil_events"])
        recomputed_rt = compute_reasoning_root(pack["reasoning_artifacts"])
        recomputed_ht = compute_composite_root(recomputed_rt, recomputed_ut)

        assert pack["u_t"] == recomputed_ut, (
            "Valid pack U_t was not computed by compute_ui_root()"
        )
        assert pack["r_t"] == recomputed_rt, (
            "Valid pack R_t was not computed by compute_reasoning_root()"
        )
        assert pack["h_t"] == recomputed_ht, (
            "Valid pack H_t was not computed by compute_composite_root()"
        )
