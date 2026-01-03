"""
Verifier Vectors Artifact Tests

Tests for the canonical test vector artifact at:
    releases/evidence_pack_verifier_vectors.v0.2.0.json

Validates:
1. Vector file exists
2. Vectors are deterministic (stable across runs)
3. Each vector has required fields (name, expected_result, expected_failure_reason for FAIL)
4. Hash verification passes for valid packs
5. Hash verification fails for invalid packs

Spec: docs/EVIDENCE_PACK_VERIFIER_SPEC.md

Run with:
    uv run pytest tests/governance/test_verifier_vectors_artifact.py -v
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Dict

import pytest

from attestation.dual_root import (
    compute_ui_root,
    compute_reasoning_root,
    compute_composite_root,
)
from substrate.crypto.core import rfc8785_canonicalize


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent.parent
VECTORS_PATH = REPO_ROOT / "releases" / "evidence_pack_verifier_vectors.v0.2.0.json"

# Expected content hash (determinism check)
EXPECTED_CONTENT_HASH = "f0ab1ad3815138ce93865e1abdedd90e86745cb1d3f8739463fa4f4860298d35"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def vectors_data() -> Dict[str, Any]:
    """Load the vector artifact."""
    assert VECTORS_PATH.exists(), f"Vector file not found: {VECTORS_PATH}"
    return json.loads(VECTORS_PATH.read_text())


# ---------------------------------------------------------------------------
# Tests: File Existence
# ---------------------------------------------------------------------------


class TestVectorFileExists:
    """Verify vector artifact exists."""

    def test_file_exists(self):
        """Vector file must exist at expected path."""
        assert VECTORS_PATH.exists(), (
            f"Vector artifact not found at {VECTORS_PATH}.\n"
            f"Run: uv run python scripts/generate_verifier_vectors.py"
        )

    def test_file_is_valid_json(self, vectors_data: Dict[str, Any]):
        """Vector file must be valid JSON."""
        assert isinstance(vectors_data, dict)

    def test_file_has_schema(self, vectors_data: Dict[str, Any]):
        """Vector file must declare schema."""
        assert "$schema" in vectors_data


# ---------------------------------------------------------------------------
# Tests: Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Verify vectors are deterministic across runs."""

    def test_content_hash_matches(self, vectors_data: Dict[str, Any]):
        """
        Content hash must match expected value.

        If this test fails, either:
        1. The vectors were regenerated (update EXPECTED_CONTENT_HASH)
        2. The generation code changed (investigate why)
        """
        recorded_hash = vectors_data["metadata"]["content_hash"]
        assert recorded_hash == EXPECTED_CONTENT_HASH, (
            f"Content hash mismatch!\n"
            f"  Expected: {EXPECTED_CONTENT_HASH}\n"
            f"  Got:      {recorded_hash}\n"
            f"If regeneration was intentional, update EXPECTED_CONTENT_HASH in this test."
        )

    def test_valid_pack_hashes_recomputable(self, vectors_data: Dict[str, Any]):
        """Valid pack hashes must be reproducible using SAME functions."""
        for vector in vectors_data["valid_packs"]:
            pack = vector["pack"]

            recomputed_ut = compute_ui_root(pack["uvil_events"])
            recomputed_rt = compute_reasoning_root(pack["reasoning_artifacts"])
            recomputed_ht = compute_composite_root(recomputed_rt, recomputed_ut)

            assert pack["u_t"] == recomputed_ut, f"U_t mismatch in {vector['name']}"
            assert pack["r_t"] == recomputed_rt, f"R_t mismatch in {vector['name']}"
            assert pack["h_t"] == recomputed_ht, f"H_t mismatch in {vector['name']}"


# ---------------------------------------------------------------------------
# Tests: Required Fields
# ---------------------------------------------------------------------------


class TestRequiredFields:
    """Verify each vector has required metadata fields."""

    def test_metadata_fields(self, vectors_data: Dict[str, Any]):
        """Metadata must have required fields."""
        metadata = vectors_data["metadata"]
        required = ["version", "generated_by", "spec", "content_hash", "description"]
        for field in required:
            assert field in metadata, f"Metadata missing {field}"

    def test_valid_packs_have_required_fields(self, vectors_data: Dict[str, Any]):
        """Each valid pack vector must have required fields."""
        required = ["name", "description", "expected_result", "expected_failure_reason", "pack"]
        for vector in vectors_data["valid_packs"]:
            for field in required:
                assert field in vector, f"Valid pack missing {field}: {vector.get('name', 'unknown')}"
            # expected_result must be PASS
            assert vector["expected_result"] == "PASS"
            # expected_failure_reason must be null for PASS
            assert vector["expected_failure_reason"] is None

    def test_invalid_packs_have_required_fields(self, vectors_data: Dict[str, Any]):
        """Each invalid pack vector must have required fields."""
        required = ["name", "description", "expected_result", "expected_failure_reason", "pack"]
        for vector in vectors_data["invalid_packs"]:
            for field in required:
                assert field in vector, f"Invalid pack missing {field}: {vector.get('name', 'unknown')}"
            # expected_result must be FAIL
            assert vector["expected_result"] == "FAIL"
            # expected_failure_reason must NOT be null for FAIL
            assert vector["expected_failure_reason"] is not None, (
                f"FAIL vector missing expected_failure_reason: {vector['name']}"
            )

    def test_canonicalization_tests_have_required_fields(self, vectors_data: Dict[str, Any]):
        """Each canonicalization test must have required fields."""
        required = ["name", "description", "input", "expected_canonical"]
        for test in vectors_data["canonicalization_tests"]:
            for field in required:
                assert field in test, f"Canon test missing {field}: {test.get('name', 'unknown')}"


# ---------------------------------------------------------------------------
# Tests: Vector Counts
# ---------------------------------------------------------------------------


class TestVectorCounts:
    """Verify minimum vector counts per spec."""

    def test_at_least_2_valid_packs(self, vectors_data: Dict[str, Any]):
        """Must have at least 2 valid packs."""
        assert len(vectors_data["valid_packs"]) >= 2

    def test_at_least_3_invalid_packs(self, vectors_data: Dict[str, Any]):
        """Must have at least 3 invalid packs with different failure reasons."""
        invalid = vectors_data["invalid_packs"]
        assert len(invalid) >= 3

        # Verify different failure reasons
        reasons = {v["expected_failure_reason"] for v in invalid}
        assert len(reasons) >= 3, f"Need 3 distinct failure reasons, got {reasons}"

    def test_at_least_2_canonicalization_tests(self, vectors_data: Dict[str, Any]):
        """Must have at least 2 canonicalization edge tests."""
        assert len(vectors_data["canonicalization_tests"]) >= 2


# ---------------------------------------------------------------------------
# Tests: Invalid Pack Verification
# ---------------------------------------------------------------------------


class TestInvalidPackVerification:
    """Verify invalid packs fail verification as expected."""

    def test_tampered_ht_fails(self, vectors_data: Dict[str, Any]):
        """Pack with tampered h_t must fail H_t verification."""
        vector = next(
            v for v in vectors_data["invalid_packs"]
            if v["expected_failure_reason"] == "h_t_mismatch"
        )
        pack = vector["pack"]

        recomputed_ut = compute_ui_root(pack["uvil_events"])
        recomputed_rt = compute_reasoning_root(pack["reasoning_artifacts"])
        recomputed_ht = compute_composite_root(recomputed_rt, recomputed_ut)

        # U_t and R_t should match (not tampered)
        assert pack["u_t"] == recomputed_ut
        assert pack["r_t"] == recomputed_rt

        # H_t should NOT match (tampered)
        assert pack["h_t"] != recomputed_ht, "Expected H_t mismatch"

    def test_tampered_rt_fails(self, vectors_data: Dict[str, Any]):
        """Pack with tampered reasoning_artifacts must fail R_t verification."""
        vector = next(
            v for v in vectors_data["invalid_packs"]
            if v["expected_failure_reason"] == "r_t_mismatch"
        )
        pack = vector["pack"]

        recomputed_rt = compute_reasoning_root(pack["reasoning_artifacts"])

        # R_t should NOT match (artifacts tampered)
        assert pack["r_t"] != recomputed_rt, "Expected R_t mismatch"

    def test_missing_field_detected(self, vectors_data: Dict[str, Any]):
        """Pack with missing field must be flagged."""
        vector = next(
            v for v in vectors_data["invalid_packs"]
            if v["expected_failure_reason"] == "missing_required_field"
        )
        pack = vector["pack"]

        # Verify the artifact is actually missing validation_outcome
        artifacts = pack["reasoning_artifacts"]
        assert len(artifacts) > 0
        assert "validation_outcome" not in artifacts[0], (
            "Expected validation_outcome to be missing"
        )


# ---------------------------------------------------------------------------
# Tests: Canonicalization
# ---------------------------------------------------------------------------


class TestCanonicalization:
    """Verify canonicalization tests are correct."""

    def test_canonicalization_outputs_match(self, vectors_data: Dict[str, Any]):
        """Canonicalization tests must produce expected output."""
        for test in vectors_data["canonicalization_tests"]:
            input_obj = test["input"]
            expected = test["expected_canonical"]

            # Use SAME function as the verifier
            actual = rfc8785_canonicalize(input_obj)

            assert actual == expected, (
                f"Canonicalization mismatch in {test['name']}:\n"
                f"  Input: {input_obj}\n"
                f"  Expected: {expected}\n"
                f"  Got: {actual}"
            )
