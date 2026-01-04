"""
Regression tests for self-test semantics (Gate 3: Runtime Verifier).

These tests enforce the required truth table:

| Expected | Actual | Test Verdict |
|----------|--------|--------------|
| PASS     | PASS   | PASS         |
| FAIL     | FAIL   | PASS         |  <- THE KEY FIX
| PASS     | FAIL   | FAIL         |
| FAIL     | PASS   | FAIL         |

The bug in v0.2.5: testPack required exact reason matching
(expectedReason==="h_t_mismatch") but examples.json used different
format ("h_t mismatch" with space). This caused FAIL/FAIL cases to
incorrectly show test verdict FAIL.

The fix in v0.2.6: pass = (actual === expected), reason is informational only.
"""

import pytest
import re
from pathlib import Path


class TestSelfTestSemantics:
    """Tests that enforce correct self-test semantics in verifier JS."""

    @pytest.fixture
    def verifier_html(self):
        """Load v0.2.6 verifier HTML."""
        path = Path("site/v0.2.6/evidence-pack/verify/index.html")
        if not path.exists():
            pytest.skip("v0.2.6 verifier not built yet")
        return path.read_text(encoding="utf-8")

    def test_testpack_no_reason_matching(self, verifier_html):
        """testPack must NOT require exact reason matching.

        The old broken logic had patterns like:
        pass:expectedResult==="FAIL"&&expectedReason==="h_t_mismatch"

        The fixed logic should be:
        pass:expectedResult==="FAIL"

        This test fails on the old broken logic.
        """
        # Must NOT contain reason matching
        assert "&&expectedReason===" not in verifier_html, \
            "testPack must not require exact reason matching (bug in v0.2.5)"

    def test_testpack_uses_simple_pass_logic(self, verifier_html):
        """testPack must use pass: (actual === expected) semantics.

        For FAIL cases, the pattern should be:
        pass:expectedResult==="FAIL"

        NOT:
        pass:expectedResult==="FAIL"&&expectedReason==="..."
        """
        # Should have simple pass logic for FAIL cases
        # Count occurrences of the correct pattern
        correct_pattern = 'pass:expectedResult==="FAIL"'
        # The pattern should appear 4 times (for u_t, r_t, h_t mismatch, and missing field)
        count = verifier_html.count(correct_pattern)
        assert count >= 4, \
            f"Expected at least 4 occurrences of '{correct_pattern}', found {count}"

    def test_testpack_has_pass_check(self, verifier_html):
        """testPack must have correct PASS check: pass:expectedResult==="PASS"."""
        assert 'pass:expectedResult==="PASS"' in verifier_html, \
            "testPack must check expectedResult==='PASS' for valid packs"


class TestBannerLogic:
    """Tests that banner shows correct message based on test verdicts."""

    @pytest.fixture
    def verifier_html(self):
        """Load v0.2.6 verifier HTML."""
        path = Path("site/v0.2.6/evidence-pack/verify/index.html")
        if not path.exists():
            pytest.skip("v0.2.6 verifier not built yet")
        return path.read_text(encoding="utf-8")

    def test_banner_shows_passed_when_all_pass(self, verifier_html):
        """Banner must show SELF-TEST PASSED when allPass is true."""
        # The banner logic: status.textContent=allPass?"SELF-TEST PASSED ("+results.length+" vectors)":"SELF-TEST FAILED"
        assert 'SELF-TEST PASSED' in verifier_html, \
            "Banner must contain 'SELF-TEST PASSED' text"
        assert 'SELF-TEST FAILED' in verifier_html, \
            "Banner must contain 'SELF-TEST FAILED' text"

    def test_banner_uses_allpass_correctly(self, verifier_html):
        """Banner must use allPass variable to determine message."""
        # Check the ternary pattern
        assert 'allPass?' in verifier_html or 'allPass ?' in verifier_html, \
            "Banner must use allPass variable to determine message"


class TestExamplesFormat:
    """Tests that examples.json has correct structure."""

    @pytest.fixture
    def examples_json(self):
        """Load v0.2.6 examples.json."""
        import json
        path = Path("site/v0.2.6/evidence-pack/examples.json")
        if not path.exists():
            pytest.skip("v0.2.6 examples.json not built yet")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_has_valid_boundary_demo(self, examples_json):
        """Must have valid_boundary_demo with expected_verdict PASS."""
        examples = examples_json.get("examples", {})
        assert "valid_boundary_demo" in examples, \
            "Must have valid_boundary_demo example"
        assert examples["valid_boundary_demo"]["expected_verdict"] == "PASS", \
            "valid_boundary_demo must expect PASS"

    def test_has_tampered_examples(self, examples_json):
        """Must have tampered examples with expected_verdict FAIL."""
        examples = examples_json.get("examples", {})

        tampered_names = [k for k in examples.keys() if k.startswith("tampered_")]
        assert len(tampered_names) >= 2, \
            "Must have at least 2 tampered examples"

        for name in tampered_names:
            assert examples[name]["expected_verdict"] == "FAIL", \
                f"{name} must expect FAIL"

    def test_three_vectors_total(self, examples_json):
        """Must have exactly 3 test vectors (1 PASS + 2 FAIL)."""
        examples = examples_json.get("examples", {})
        assert len(examples) == 3, \
            f"Must have exactly 3 examples, found {len(examples)}"


class TestTruthTable:
    """Tests that encode the required truth table semantics."""

    def test_truth_table_pass_pass(self):
        """Expected=PASS, Actual=PASS → Test Verdict PASS."""
        expected = "PASS"
        actual = "PASS"
        # Simulating the fixed JS logic: pass = (actual === expected)
        test_verdict = "PASS" if actual == expected else "FAIL"
        assert test_verdict == "PASS"

    def test_truth_table_fail_fail(self):
        """Expected=FAIL, Actual=FAIL → Test Verdict PASS.

        THIS IS THE KEY SEMANTIC: when we expect FAIL and get FAIL,
        the TEST passed because the verifier correctly detected the tampering.
        """
        expected = "FAIL"
        actual = "FAIL"
        # Simulating the fixed JS logic: pass = (actual === expected)
        test_verdict = "PASS" if actual == expected else "FAIL"
        assert test_verdict == "PASS", \
            "FAIL/FAIL must result in test verdict PASS (correctness relative to expectation)"

    def test_truth_table_pass_fail(self):
        """Expected=PASS, Actual=FAIL → Test Verdict FAIL."""
        expected = "PASS"
        actual = "FAIL"
        test_verdict = "PASS" if actual == expected else "FAIL"
        assert test_verdict == "FAIL"

    def test_truth_table_fail_pass(self):
        """Expected=FAIL, Actual=PASS → Test Verdict FAIL."""
        expected = "FAIL"
        actual = "PASS"
        test_verdict = "PASS" if actual == expected else "FAIL"
        assert test_verdict == "FAIL"


class TestV025BrokenLogicDocumentation:
    """Documents the bug that existed in v0.2.5 before the fix.

    NOTE: After rebuild, v0.2.5 site files also get the fix applied
    (build script regenerates all versions). This test is now skipped
    but kept as documentation of what the bug looked like.
    """

    @pytest.fixture
    def v025_verifier(self):
        """Load v0.2.5 verifier HTML if it exists."""
        path = Path("site/v0.2.5/evidence-pack/verify/index.html")
        if not path.exists():
            pytest.skip("v0.2.5 not available")
        return path.read_text(encoding="utf-8")

    @pytest.mark.skip(reason="v0.2.5 regenerated with fix; kept for documentation")
    def test_v025_originally_had_broken_reason_matching(self, v025_verifier):
        """v0.2.5 ORIGINALLY had broken reason matching logic.

        The broken pattern was:
        pass:expectedResult==="FAIL"&&expectedReason==="h_t_mismatch"

        This required exact reason string matching, but examples.json
        used "h_t mismatch" (with space) not "h_t_mismatch" (with underscore).

        After the build script fix, all versions including v0.2.5 are
        regenerated with the correct logic.
        """
        has_broken_pattern = "&&expectedReason===" in v025_verifier
        assert has_broken_pattern, \
            "v0.2.5 should have the broken reason matching pattern"


class TestV026FixedLogic:
    """Tests that verify v0.2.6 fix is correct."""

    @pytest.fixture
    def v026_verifier(self):
        """Load v0.2.6 verifier HTML."""
        path = Path("site/v0.2.6/evidence-pack/verify/index.html")
        if not path.exists():
            pytest.skip("v0.2.6 not built yet")
        return path.read_text(encoding="utf-8")

    def test_v026_no_reason_matching(self, v026_verifier):
        """v0.2.6 must NOT have reason matching logic."""
        assert "&&expectedReason===" not in v026_verifier, \
            "v0.2.6 must not have broken reason matching"

    def test_v026_simple_pass_logic(self, v026_verifier):
        """v0.2.6 must use simple pass = (actual === expected) logic."""
        # Check for the simple pattern without reason matching
        # Pattern: pass:expectedResult==="FAIL", without &&expectedReason
        pattern = r'pass:expectedResult==="FAIL"[^&]'
        matches = re.findall(pattern, v026_verifier)
        assert len(matches) >= 1, \
            "v0.2.6 must use simple pass logic without reason matching"
