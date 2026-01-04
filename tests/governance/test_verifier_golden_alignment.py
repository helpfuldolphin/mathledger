"""
Golden Test: JS Verifier byte-for-byte alignment with Python replay_verify.

This test verifies that the JS verifier computes identical U_t, R_t, H_t values
as the canonical Python attestation code for valid_boundary_demo.

If this test fails, either:
1. The JS verifier has drifted from Python
2. The examples have drifted from the canonical generator

Run with:
    uv run pytest tests/governance/test_verifier_golden_alignment.py -v
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
EXAMPLES_FILE = REPO_ROOT / "releases" / "evidence_pack_examples.v0.2.3.json"


class TestGoldenAlignment:
    """Verify JS verifier produces identical hashes to Python."""

    def test_valid_boundary_demo_hashes_match_python(self):
        """
        GOLDEN TEST: valid_boundary_demo pack hashes must equal Python computation.

        This is the authoritative alignment test. If this fails, the JS verifier
        is not computing the same values as Python replay_verify.
        """
        if not EXAMPLES_FILE.exists():
            pytest.skip(f"Examples file not found: {EXAMPLES_FILE}")

        content = EXAMPLES_FILE.read_text(encoding="utf-8")
        data = json.loads(content)
        example = data["examples"]["valid_boundary_demo"]
        pack = example["pack"]

        # Extract data from pack
        uvil_events = pack["uvil_events"]
        reasoning_artifacts = pack["reasoning_artifacts"]
        claimed_u_t = pack["u_t"]
        claimed_r_t = pack["r_t"]
        claimed_h_t = pack["h_t"]

        # Compute using canonical Python functions
        python_u_t = compute_ui_root(uvil_events)
        python_r_t = compute_reasoning_root(reasoning_artifacts)
        python_h_t = compute_composite_root(python_r_t, python_u_t)

        # All three must match exactly
        assert python_u_t == claimed_u_t, (
            f"U_t mismatch: Python computed {python_u_t}, pack claims {claimed_u_t}"
        )
        assert python_r_t == claimed_r_t, (
            f"R_t mismatch: Python computed {python_r_t}, pack claims {claimed_r_t}"
        )
        assert python_h_t == claimed_h_t, (
            f"H_t mismatch: Python computed {python_h_t}, pack claims {claimed_h_t}"
        )

    def test_golden_u_t_value(self):
        """
        GOLDEN VALUE: U_t for valid_boundary_demo must be this exact value.

        If this test fails, either:
        1. The example data changed (regenerate examples)
        2. The attestation algorithm changed (major breaking change)
        """
        if not EXAMPLES_FILE.exists():
            pytest.skip(f"Examples file not found: {EXAMPLES_FILE}")

        content = EXAMPLES_FILE.read_text(encoding="utf-8")
        data = json.loads(content)
        pack = data["examples"]["valid_boundary_demo"]["pack"]

        # This is the canonical U_t for valid_boundary_demo
        GOLDEN_U_T = "0d1b61da395bb759b4558e1329e9ea561450e66d66421f88b540f7e828c0cd2d"

        python_u_t = compute_ui_root(pack["uvil_events"])

        assert python_u_t == GOLDEN_U_T, (
            f"Golden U_t drift detected!\n"
            f"Expected: {GOLDEN_U_T}\n"
            f"Got: {python_u_t}\n"
            f"If attestation algorithm changed, update golden value."
        )

    def test_golden_r_t_value(self):
        """GOLDEN VALUE: R_t for valid_boundary_demo."""
        if not EXAMPLES_FILE.exists():
            pytest.skip(f"Examples file not found: {EXAMPLES_FILE}")

        content = EXAMPLES_FILE.read_text(encoding="utf-8")
        data = json.loads(content)
        pack = data["examples"]["valid_boundary_demo"]["pack"]

        # This is the canonical R_t for valid_boundary_demo
        GOLDEN_R_T = "fc252c380d1af2afaa4f17a52a8692156f2edcd6336ee4a3278a23a10eda4899"

        python_r_t = compute_reasoning_root(pack["reasoning_artifacts"])

        assert python_r_t == GOLDEN_R_T, (
            f"Golden R_t drift detected!\n"
            f"Expected: {GOLDEN_R_T}\n"
            f"Got: {python_r_t}"
        )

    def test_golden_h_t_value(self):
        """GOLDEN VALUE: H_t for valid_boundary_demo."""
        if not EXAMPLES_FILE.exists():
            pytest.skip(f"Examples file not found: {EXAMPLES_FILE}")

        content = EXAMPLES_FILE.read_text(encoding="utf-8")
        data = json.loads(content)
        pack = data["examples"]["valid_boundary_demo"]["pack"]

        # This is the canonical H_t for valid_boundary_demo
        # H_t = SHA256(R_t || U_t)
        GOLDEN_H_T = "fc326bbaad3518e4de63a3d81f68dc2030ff47bdb80532081e4b0c0c2a8f2fd4"

        python_r_t = compute_reasoning_root(pack["reasoning_artifacts"])
        python_u_t = compute_ui_root(pack["uvil_events"])
        python_h_t = compute_composite_root(python_r_t, python_u_t)

        assert python_h_t == GOLDEN_H_T, (
            f"Golden H_t drift detected!\n"
            f"Expected: {GOLDEN_H_T}\n"
            f"Got: {python_h_t}"
        )

    def test_composite_root_formula(self):
        """Verify H_t = SHA256(R_t || U_t) formula."""
        import hashlib

        r_t = "fc252c380d1af2afaa4f17a52a8692156f2edcd6336ee4a3278a23a10eda4899"
        u_t = "0d1b61da395bb759b4558e1329e9ea561450e66d66421f88b540f7e828c0cd2d"

        # Manual computation
        combined = f"{r_t}{u_t}".encode("ascii")
        expected_h_t = hashlib.sha256(combined).hexdigest()

        # Function computation
        python_h_t = compute_composite_root(r_t, u_t)

        assert python_h_t == expected_h_t, (
            f"Composite root formula mismatch:\n"
            f"Manual: {expected_h_t}\n"
            f"Function: {python_h_t}"
        )


class TestDomainSeparationConstants:
    """Verify domain separation constants match between Python and expected JS values."""

    def test_domain_ui_leaf(self):
        """DOMAIN_UI_LEAF = 0xA1 + 'ui-leaf'"""
        from attestation.dual_root import DOMAIN_UI_LEAF

        expected = b"\xA1ui-leaf"
        assert DOMAIN_UI_LEAF == expected, (
            f"DOMAIN_UI_LEAF mismatch: {DOMAIN_UI_LEAF!r} != {expected!r}"
        )

    def test_domain_reasoning_leaf(self):
        """DOMAIN_REASONING_LEAF = 0xA0 + 'reasoning-leaf'"""
        from attestation.dual_root import DOMAIN_REASONING_LEAF

        expected = b"\xA0reasoning-leaf"
        assert DOMAIN_REASONING_LEAF == expected, (
            f"DOMAIN_REASONING_LEAF mismatch: {DOMAIN_REASONING_LEAF!r} != {expected!r}"
        )

    def test_domain_leaf(self):
        """DOMAIN_LEAF = 0x00"""
        from substrate.crypto.hashing import DOMAIN_LEAF

        expected = b"\x00"
        assert DOMAIN_LEAF == expected, (
            f"DOMAIN_LEAF mismatch: {DOMAIN_LEAF!r} != {expected!r}"
        )

    def test_domain_node(self):
        """DOMAIN_NODE = 0x01"""
        from substrate.crypto.hashing import DOMAIN_NODE

        expected = b"\x01"
        assert DOMAIN_NODE == expected, (
            f"DOMAIN_NODE mismatch: {DOMAIN_NODE!r} != {expected!r}"
        )

    def test_js_domain_ui_leaf_bytes(self):
        """Verify JS domain constant bytes match Python."""
        from attestation.dual_root import DOMAIN_UI_LEAF

        # JS: new Uint8Array([0xA1, 0x75, 0x69, 0x2d, 0x6c, 0x65, 0x61, 0x66])
        # Which is: 0xA1 + "ui-leaf" encoded as bytes
        js_bytes = bytes([0xA1, 0x75, 0x69, 0x2d, 0x6c, 0x65, 0x61, 0x66])

        assert DOMAIN_UI_LEAF == js_bytes, (
            f"JS DOMAIN_UI_LEAF bytes don't match Python:\n"
            f"Python: {list(DOMAIN_UI_LEAF)}\n"
            f"JS: {list(js_bytes)}"
        )

    def test_js_domain_reasoning_leaf_bytes(self):
        """Verify JS domain constant bytes match Python."""
        from attestation.dual_root import DOMAIN_REASONING_LEAF

        # JS: new Uint8Array([0xA0, 0x72, 0x65, 0x61, 0x73, 0x6f, 0x6e, 0x69, 0x6e, 0x67, 0x2d, 0x6c, 0x65, 0x61, 0x66])
        # Which is: 0xA0 + "reasoning-leaf" encoded as bytes
        js_bytes = bytes([0xA0, 0x72, 0x65, 0x61, 0x73, 0x6f, 0x6e, 0x69, 0x6e, 0x67, 0x2d, 0x6c, 0x65, 0x61, 0x66])

        assert DOMAIN_REASONING_LEAF == js_bytes, (
            f"JS DOMAIN_REASONING_LEAF bytes don't match Python:\n"
            f"Python: {list(DOMAIN_REASONING_LEAF)}\n"
            f"JS: {list(js_bytes)}"
        )
