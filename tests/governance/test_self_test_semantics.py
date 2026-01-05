"""
Regression test for verifier self-test semantics (v0.2.5 fix).

This test ensures the self-test logic is NEVER inverted again:
- When Expected = FAIL and Actual = FAIL, the test should PASS
- When Expected = PASS and Actual = PASS, the test should PASS
- Only mismatches should result in test FAIL

Run with:
    uv run pytest tests/governance/test_self_test_semantics.py -v
"""

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).parent.parent.parent
VERIFIER_PATH = REPO_ROOT / "site" / "v0.2.5" / "evidence-pack" / "verify" / "index.html"


class TestSelfTestSemantics:
    """Regression guard for v0.2.5 self-test semantic fix."""

    def test_self_test_semantics_fail_equals_pass(self):
        """
        CRITICAL REGRESSION TEST: Expected=FAIL, Actual=FAIL -> Test PASSES

        This is the core semantic fix from v0.2.5. If this test fails,
        the verifier UI is broken and will show false negatives.

        The verifier correctly detecting tampering (FAIL) should be
        reported as a test PASS, not a test FAIL.
        """
        if not VERIFIER_PATH.exists():
            pytest.skip(f"Verifier not found: {VERIFIER_PATH}")

        content = VERIFIER_PATH.read_text(encoding="utf-8")

        # The testPack function must return pass: actual === expectedResult
        # This regex looks for the semantic fix pattern
        semantic_fix_pattern = r"pass:\s*actual\s*===\s*expectedResult"

        assert re.search(semantic_fix_pattern, content), (
            "REGRESSION: testPack does not use 'pass: actual === expectedResult'\n"
            "The v0.2.5 semantic fix requires that test pass/fail is determined\n"
            "solely by whether actual verdict matches expected verdict.\n"
            "FAIL/FAIL -> PASS, PASS/PASS -> PASS, mismatch -> FAIL"
        )

    def test_no_reason_in_pass_determination(self):
        """
        The pass/fail determination must NOT depend on reason matching.

        Old buggy code: pass: expectedResult==="FAIL" && expectedReason==="h_t_mismatch"
        Fixed code: pass: actual === expectedResult
        """
        if not VERIFIER_PATH.exists():
            pytest.skip(f"Verifier not found: {VERIFIER_PATH}")

        content = VERIFIER_PATH.read_text(encoding="utf-8")

        # The old buggy pattern required reason matching
        buggy_pattern = r'pass:\s*expectedResult\s*===\s*"FAIL"\s*&&\s*expectedReason'

        assert not re.search(buggy_pattern, content), (
            "REGRESSION: testPack still uses reason matching for pass determination.\n"
            "The pass field should be 'actual === expectedResult' with no reason check.\n"
            "Reason is informational only."
        )

    def test_testpack_returns_pass_field(self):
        """testPack must return an object with 'pass' field."""
        if not VERIFIER_PATH.exists():
            pytest.skip(f"Verifier not found: {VERIFIER_PATH}")

        content = VERIFIER_PATH.read_text(encoding="utf-8")

        # Must have return statements with pass field
        assert "return {actual:" in content or "return{actual:" in content, (
            "testPack must return objects with actual and pass fields"
        )

    def test_banner_shows_pass_count_on_success(self):
        """Banner must show 'SELF-TEST PASSED (N vectors)' on success."""
        if not VERIFIER_PATH.exists():
            pytest.skip(f"Verifier not found: {VERIFIER_PATH}")

        content = VERIFIER_PATH.read_text(encoding="utf-8")

        assert "SELF-TEST PASSED" in content, (
            "Banner must show 'SELF-TEST PASSED' text"
        )
        assert "vectors" in content, (
            "Banner must show vector count"
        )

    def test_banner_shows_failure_count_on_failure(self):
        """Banner must show 'SELF-TEST FAILED (N failures)' on failure."""
        if not VERIFIER_PATH.exists():
            pytest.skip(f"Verifier not found: {VERIFIER_PATH}")

        content = VERIFIER_PATH.read_text(encoding="utf-8")

        assert "SELF-TEST FAILED" in content, (
            "Banner must show 'SELF-TEST FAILED' text for failures"
        )
        assert "failure" in content, (
            "Banner must show failure count"
        )

    def test_table_header_says_test_result(self):
        """Table header should say 'Test Result', not 'Pass/Fail'."""
        if not VERIFIER_PATH.exists():
            pytest.skip(f"Verifier not found: {VERIFIER_PATH}")

        content = VERIFIER_PATH.read_text(encoding="utf-8")

        # The header should clarify this is test result, not verification result
        assert "Test Result" in content, (
            "Table header should say 'Test Result' to clarify semantics"
        )

    def test_semantic_comment_present(self):
        """
        The testPack function must have a comment explaining the semantics.

        This serves as documentation to prevent future regressions.
        """
        if not VERIFIER_PATH.exists():
            pytest.skip(f"Verifier not found: {VERIFIER_PATH}")

        content = VERIFIER_PATH.read_text(encoding="utf-8")

        # Look for the semantic documentation
        assert "FAIL/FAIL" in content and "PASS" in content, (
            "testPack must document that FAIL/FAIL -> test PASSES"
        )


class TestExamplesConsistency:
    """Verify examples.json is consistent with verifier expectations."""

    def test_examples_have_expected_verdict(self):
        """All examples must have expected_verdict field."""
        examples_path = REPO_ROOT / "site" / "v0.2.5" / "evidence-pack" / "examples.json"
        if not examples_path.exists():
            pytest.skip(f"Examples not found: {examples_path}")

        import json
        content = examples_path.read_text(encoding="utf-8")
        data = json.loads(content)

        examples = data.get("examples", {})
        for name, ex in examples.items():
            assert "expected_verdict" in ex, (
                f"Example '{name}' missing expected_verdict field"
            )
            assert ex["expected_verdict"] in ("PASS", "FAIL"), (
                f"Example '{name}' has invalid expected_verdict: {ex['expected_verdict']}"
            )

    def test_tampered_examples_expect_fail(self):
        """Examples with 'tampered' in name must expect FAIL."""
        examples_path = REPO_ROOT / "site" / "v0.2.5" / "evidence-pack" / "examples.json"
        if not examples_path.exists():
            pytest.skip(f"Examples not found: {examples_path}")

        import json
        content = examples_path.read_text(encoding="utf-8")
        data = json.loads(content)

        examples = data.get("examples", {})
        for name, ex in examples.items():
            if "tampered" in name.lower():
                assert ex.get("expected_verdict") == "FAIL", (
                    f"Tampered example '{name}' should expect FAIL, "
                    f"got {ex.get('expected_verdict')}"
                )

    def test_valid_examples_expect_pass(self):
        """Examples with 'valid' in name must expect PASS."""
        examples_path = REPO_ROOT / "site" / "v0.2.5" / "evidence-pack" / "examples.json"
        if not examples_path.exists():
            pytest.skip(f"Examples not found: {examples_path}")

        import json
        content = examples_path.read_text(encoding="utf-8")
        data = json.loads(content)

        examples = data.get("examples", {})
        for name, ex in examples.items():
            if "valid" in name.lower():
                assert ex.get("expected_verdict") == "PASS", (
                    f"Valid example '{name}' should expect PASS, "
                    f"got {ex.get('expected_verdict')}"
                )
