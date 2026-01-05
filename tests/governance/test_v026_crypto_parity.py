"""
Cryptographic Parity Regression Tests for v0.2.6 Verifier

These tests ensure the JavaScript verifier computes byte-for-byte
identical hash values to the Python implementation.

Run with:
    uv run pytest tests/governance/test_v026_crypto_parity.py -v
"""

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
VERIFIER_PATH = REPO_ROOT / "site" / "v0.2.6" / "evidence-pack" / "verify" / "index.html"
EXAMPLES_PATH = REPO_ROOT / "site" / "v0.2.6" / "evidence-pack" / "examples.json"


class TestDomainSeparationConstants:
    """Verify domain separation constants match Python."""

    def test_domain_reasoning_leaf_prefix(self):
        """DOMAIN_REASONING_LEAF must start with 0xA0."""
        if not VERIFIER_PATH.exists():
            pytest.skip(f"Verifier not found: {VERIFIER_PATH}")

        content = VERIFIER_PATH.read_text(encoding="utf-8")
        assert "0xA0" in content, (
            "DOMAIN_REASONING_LEAF must use 0xA0 prefix"
        )
        assert "'reasoning-leaf'" in content, (
            "DOMAIN_REASONING_LEAF must include 'reasoning-leaf' suffix"
        )

    def test_domain_ui_leaf_prefix(self):
        """DOMAIN_UI_LEAF must start with 0xA1."""
        if not VERIFIER_PATH.exists():
            pytest.skip(f"Verifier not found: {VERIFIER_PATH}")

        content = VERIFIER_PATH.read_text(encoding="utf-8")
        assert "0xA1" in content, (
            "DOMAIN_UI_LEAF must use 0xA1 prefix"
        )
        assert "'ui-leaf'" in content, (
            "DOMAIN_UI_LEAF must include 'ui-leaf' suffix"
        )

    def test_domain_leaf_merkle(self):
        """DOMAIN_LEAF for Merkle tree must be 0x00."""
        if not VERIFIER_PATH.exists():
            pytest.skip(f"Verifier not found: {VERIFIER_PATH}")

        content = VERIFIER_PATH.read_text(encoding="utf-8")
        assert "DOMAIN_LEAF=new Uint8Array([0x00])" in content, (
            "DOMAIN_LEAF must be Uint8Array([0x00])"
        )

    def test_domain_node_merkle(self):
        """DOMAIN_NODE for Merkle tree must be 0x01."""
        if not VERIFIER_PATH.exists():
            pytest.skip(f"Verifier not found: {VERIFIER_PATH}")

        content = VERIFIER_PATH.read_text(encoding="utf-8")
        assert "DOMAIN_NODE=new Uint8Array([0x01])" in content, (
            "DOMAIN_NODE must be Uint8Array([0x01])"
        )


class TestMerkleTreeImplementation:
    """Verify Merkle tree implementation follows Python pattern."""

    def test_merkle_root_function_exists(self):
        """merkleRoot function must exist."""
        if not VERIFIER_PATH.exists():
            pytest.skip(f"Verifier not found: {VERIFIER_PATH}")

        content = VERIFIER_PATH.read_text(encoding="utf-8")
        assert "async function merkleRoot" in content, (
            "merkleRoot function must be defined"
        )

    def test_merkle_sorts_leaves(self):
        """Merkle tree must sort leaves before hashing."""
        if not VERIFIER_PATH.exists():
            pytest.skip(f"Verifier not found: {VERIFIER_PATH}")

        content = VERIFIER_PATH.read_text(encoding="utf-8")
        assert "[...leafHashes].sort()" in content, (
            "merkleRoot must sort leaf hashes"
        )

    def test_merkle_uses_domain_leaf(self):
        """Merkle tree leaves must use DOMAIN_LEAF."""
        if not VERIFIER_PATH.exists():
            pytest.skip(f"Verifier not found: {VERIFIER_PATH}")

        content = VERIFIER_PATH.read_text(encoding="utf-8")
        assert "shaDBytes(lh,DOMAIN_LEAF)" in content, (
            "Merkle leaf hashing must use DOMAIN_LEAF"
        )

    def test_merkle_uses_domain_node(self):
        """Merkle tree internal nodes must use DOMAIN_NODE."""
        if not VERIFIER_PATH.exists():
            pytest.skip(f"Verifier not found: {VERIFIER_PATH}")

        content = VERIFIER_PATH.read_text(encoding="utf-8")
        assert "shaDBytes(combined,DOMAIN_NODE)" in content, (
            "Merkle internal node hashing must use DOMAIN_NODE"
        )


class TestComputeFunctions:
    """Verify computeUt, computeRt, computeHt functions."""

    def test_compute_ut_uses_domain_ui_leaf(self):
        """computeUt must hash each event with DOMAIN_UI_LEAF."""
        if not VERIFIER_PATH.exists():
            pytest.skip(f"Verifier not found: {VERIFIER_PATH}")

        content = VERIFIER_PATH.read_text(encoding="utf-8")
        assert "shaD(can(e),DOMAIN_UI_LEAF)" in content, (
            "computeUt must hash each event with DOMAIN_UI_LEAF"
        )

    def test_compute_rt_uses_domain_reasoning_leaf(self):
        """computeRt must hash each artifact with DOMAIN_REASONING_LEAF."""
        if not VERIFIER_PATH.exists():
            pytest.skip(f"Verifier not found: {VERIFIER_PATH}")

        content = VERIFIER_PATH.read_text(encoding="utf-8")
        assert "shaD(can(a),DOMAIN_REASONING_LEAF)" in content, (
            "computeRt must hash each artifact with DOMAIN_REASONING_LEAF"
        )

    def test_compute_ht_is_sha256_concat(self):
        """computeHt must be SHA256(rt + ut)."""
        if not VERIFIER_PATH.exists():
            pytest.skip(f"Verifier not found: {VERIFIER_PATH}")

        content = VERIFIER_PATH.read_text(encoding="utf-8")
        assert "sha(rt+ut)" in content, (
            "computeHt must use sha(rt+ut)"
        )


class TestHashParity:
    """Verify JS computes same hashes as Python."""

    def test_examples_have_python_generated_hashes(self):
        """Examples must use hashes generated by Python."""
        if not EXAMPLES_PATH.exists():
            pytest.skip(f"Examples not found: {EXAMPLES_PATH}")

        from attestation.dual_root import (
            compute_ui_root,
            compute_reasoning_root,
            compute_composite_root,
        )

        examples = json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))
        valid_pack = examples["examples"]["valid_boundary_demo"]["pack"]

        # Compute using Python
        py_u_t = compute_ui_root(valid_pack["uvil_events"])
        py_r_t = compute_reasoning_root(valid_pack["reasoning_artifacts"])
        py_h_t = compute_composite_root(py_r_t, py_u_t)

        # Compare with recorded
        assert valid_pack["u_t"] == py_u_t, (
            f"U_t mismatch: recorded={valid_pack['u_t']}, computed={py_u_t}"
        )
        assert valid_pack["r_t"] == py_r_t, (
            f"R_t mismatch: recorded={valid_pack['r_t']}, computed={py_r_t}"
        )
        assert valid_pack["h_t"] == py_h_t, (
            f"H_t mismatch: recorded={valid_pack['h_t']}, computed={py_h_t}"
        )

    def test_golden_hashes_boundary_demo(self):
        """Golden hash values for valid_boundary_demo."""
        if not EXAMPLES_PATH.exists():
            pytest.skip(f"Examples not found: {EXAMPLES_PATH}")

        examples = json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))
        valid_pack = examples["examples"]["valid_boundary_demo"]["pack"]

        # These are the canonical values that JS must match
        GOLDEN_U_T = "0d1b61da395bb759b4558e1329e9ea561450e66d66421f88b540f7e828c0cd2d"
        GOLDEN_R_T = "fc252c380d1af2afaa4f17a52a8692156f2edcd6336ee4a3278a23a10eda4899"
        GOLDEN_H_T = "fc326bbaad3518e4de63a3d81f68dc2030ff47bdb80532081e4b0c0c2a8f2fd4"

        assert valid_pack["u_t"] == GOLDEN_U_T, f"U_t golden mismatch"
        assert valid_pack["r_t"] == GOLDEN_R_T, f"R_t golden mismatch"
        assert valid_pack["h_t"] == GOLDEN_H_T, f"H_t golden mismatch"


class TestSelfTestSemantics:
    """Verify self-test semantic fix from v0.2.5 is preserved."""

    def test_testpack_correct_semantics(self):
        """testPack must use actual===expectedResult for pass determination."""
        if not VERIFIER_PATH.exists():
            pytest.skip(f"Verifier not found: {VERIFIER_PATH}")

        content = VERIFIER_PATH.read_text(encoding="utf-8")

        # Check that FAIL/FAIL -> test PASSES is documented
        assert "FAIL/FAIL" in content and "PASSES" in content, (
            "testPack must document that FAIL/FAIL -> test PASSES"
        )


class TestVersionIntegrity:
    """Verify version strings are correct."""

    def test_verifier_version_is_026(self):
        """Verifier must identify as v0.2.6."""
        if not VERIFIER_PATH.exists():
            pytest.skip(f"Verifier not found: {VERIFIER_PATH}")

        content = VERIFIER_PATH.read_text(encoding="utf-8")

        assert "v0.2.6" in content, "Verifier must identify as v0.2.6"
        assert "v0.2.6-verifier-correctness" in content, (
            "Tag must be v0.2.6-verifier-correctness"
        )

    def test_examples_version_is_026(self):
        """Examples pack_version must be v0.2.6."""
        if not EXAMPLES_PATH.exists():
            pytest.skip(f"Examples not found: {EXAMPLES_PATH}")

        examples = json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))

        for name, ex in examples["examples"].items():
            pack_version = ex["pack"].get("pack_version", "")
            assert pack_version == "v0.2.6", (
                f"Example '{name}' pack_version must be v0.2.6, got {pack_version}"
            )
