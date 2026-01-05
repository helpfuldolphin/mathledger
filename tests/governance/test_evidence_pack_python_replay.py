"""
Regression test: All expected_verdict=PASS packs must pass Python replay verification.

This test ensures that every evidence pack in releases/evidence_pack_examples.*.json
that is marked as expected_verdict="PASS" actually passes when verified using
the canonical Python attestation code (attestation/dual_root.py).

If this test fails, either:
1. The examples.json packs have incorrect hashes (regenerate with generate_evidence_pack_examples.py)
2. The attestation code has changed (update examples to match)
3. There's a bug in the test or examples

Run with:
    uv run pytest tests/governance/test_evidence_pack_python_replay.py -v
"""

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).parent.parent.parent
RELEASES_DIR = REPO_ROOT / "releases"

# All example pack files to verify
EXAMPLE_PACK_FILES = list(RELEASES_DIR.glob("evidence_pack_examples.*.json"))


def get_attestation_functions():
    """Import the canonical attestation functions."""
    try:
        from attestation.dual_root import (
            compute_ui_root,
            compute_reasoning_root,
            compute_composite_root,
        )
        return compute_ui_root, compute_reasoning_root, compute_composite_root
    except ImportError:
        pytest.skip("attestation.dual_root not available")


def verify_pack_with_python(pack: dict) -> tuple[bool, str]:
    """
    Verify an evidence pack using the canonical Python attestation code.

    Returns:
        (passed, reason) where:
        - passed: True if all hashes match
        - reason: Empty string if passed, otherwise the mismatch type
    """
    compute_ui_root, compute_reasoning_root, compute_composite_root = get_attestation_functions()

    uvil_events = pack.get("uvil_events", [])
    reasoning_artifacts = pack.get("reasoning_artifacts", [])
    claimed_u_t = pack.get("u_t", "")
    claimed_r_t = pack.get("r_t", "")
    claimed_h_t = pack.get("h_t", "")

    # Check for missing required field in artifacts
    for artifact in reasoning_artifacts:
        if "validation_outcome" not in artifact:
            return False, "missing_required_field"

    # Compute hashes using canonical Python code
    computed_u_t = compute_ui_root(uvil_events)
    computed_r_t = compute_reasoning_root(reasoning_artifacts)
    computed_h_t = compute_composite_root(computed_r_t, computed_u_t)

    # Check for mismatches
    if computed_u_t != claimed_u_t:
        return False, f"u_t_mismatch (got {computed_u_t[:16]}..., expected {claimed_u_t[:16]}...)"
    if computed_r_t != claimed_r_t:
        return False, f"r_t_mismatch (got {computed_r_t[:16]}..., expected {claimed_r_t[:16]}...)"
    if computed_h_t != claimed_h_t:
        return False, f"h_t_mismatch (got {computed_h_t[:16]}..., expected {claimed_h_t[:16]}...)"

    return True, ""


class TestEvidencePackPythonReplay:
    """Verify all expected_verdict=PASS packs pass Python replay verification."""

    @pytest.mark.parametrize("examples_file", EXAMPLE_PACK_FILES)
    def test_all_pass_packs_verify(self, examples_file: Path):
        """
        Every pack with expected_verdict=PASS must pass Python replay verification.

        This is the canonical source of truth - if the Python attestation code
        says a pack is valid, the JS verifier must agree (or be fixed).
        """
        if not examples_file.exists():
            pytest.skip(f"Examples file not found: {examples_file}")

        content = examples_file.read_text(encoding="utf-8")
        data = json.loads(content)
        examples = data.get("examples", {})

        failures = []
        for name, example in examples.items():
            expected_verdict = example.get("expected_verdict", "PASS")
            if expected_verdict != "PASS":
                continue  # Only test PASS packs

            pack = example.get("pack", {})
            passed, reason = verify_pack_with_python(pack)

            if not passed:
                failures.append(f"  {name}: {reason}")

        assert not failures, (
            f"The following expected_verdict=PASS packs FAILED Python replay verification:\n"
            + "\n".join(failures)
            + "\n\nRegenerate with: uv run python scripts/generate_evidence_pack_examples.py"
        )

    def test_v023_examples_pass_packs_verify(self):
        """Specific test for v0.2.3 evidence pack examples."""
        examples_file = RELEASES_DIR / "evidence_pack_examples.v0.2.3.json"
        if not examples_file.exists():
            pytest.skip(f"v0.2.3 examples not found: {examples_file}")

        content = examples_file.read_text(encoding="utf-8")
        data = json.loads(content)
        examples = data.get("examples", {})

        pass_count = 0
        for name, example in examples.items():
            expected_verdict = example.get("expected_verdict", "PASS")
            if expected_verdict != "PASS":
                continue

            pack = example.get("pack", {})
            passed, reason = verify_pack_with_python(pack)

            assert passed, f"{name} should PASS but got: {reason}"
            pass_count += 1

        assert pass_count > 0, "No expected_verdict=PASS packs found in v0.2.3 examples"

    def test_fail_packs_have_expected_reason(self):
        """
        Packs with expected_verdict=FAIL should fail with the expected_reason.

        This verifies that tampered packs are correctly detected.
        """
        examples_file = RELEASES_DIR / "evidence_pack_examples.v0.2.3.json"
        if not examples_file.exists():
            pytest.skip(f"v0.2.3 examples not found: {examples_file}")

        content = examples_file.read_text(encoding="utf-8")
        data = json.loads(content)
        examples = data.get("examples", {})

        fail_count = 0
        for name, example in examples.items():
            expected_verdict = example.get("expected_verdict", "PASS")
            if expected_verdict != "FAIL":
                continue

            expected_reason = example.get("expected_reason", "")
            pack = example.get("pack", {})
            passed, actual_reason = verify_pack_with_python(pack)

            assert not passed, f"{name} should FAIL but passed"
            # Check reason matches (normalize underscores/spaces)
            if expected_reason:
                # Normalize to underscore format for comparison
                normalized_expected = expected_reason.replace(" ", "_")
                normalized_actual = actual_reason.replace(" ", "_")
                assert normalized_expected in normalized_actual, (
                    f"{name}: expected reason '{expected_reason}' not in '{actual_reason}'"
                )
            fail_count += 1

        # There should be at least some FAIL cases
        assert fail_count > 0, "No expected_verdict=FAIL packs found to test"


class TestCanonicalFunctionImport:
    """Verify the canonical attestation functions are available."""

    def test_attestation_functions_available(self):
        """The attestation functions must be importable."""
        from attestation.dual_root import (
            compute_ui_root,
            compute_reasoning_root,
            compute_composite_root,
        )

        assert callable(compute_ui_root)
        assert callable(compute_reasoning_root)
        assert callable(compute_composite_root)

    def test_empty_inputs_produce_consistent_hashes(self):
        """Empty inputs should produce deterministic hashes."""
        from attestation.dual_root import (
            compute_ui_root,
            compute_reasoning_root,
            compute_composite_root,
        )

        u_t_1 = compute_ui_root([])
        u_t_2 = compute_ui_root([])
        assert u_t_1 == u_t_2, "Empty UI root should be deterministic"

        r_t_1 = compute_reasoning_root([])
        r_t_2 = compute_reasoning_root([])
        assert r_t_1 == r_t_2, "Empty reasoning root should be deterministic"

        h_t_1 = compute_composite_root(r_t_1, u_t_1)
        h_t_2 = compute_composite_root(r_t_2, u_t_2)
        assert h_t_1 == h_t_2, "Composite root should be deterministic"
