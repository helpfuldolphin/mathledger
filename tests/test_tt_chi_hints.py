"""
Tests for Phase II UX & Policy Layer features.

TASK 1: CHI-Based Timeout Hint Generator
TASK 2: Minimal Policy Hook for Hardness (HardnessPolicySignal)
TASK 3: Diagnostics Snapshot Formatter for Incident Bundles

Agent B2 - Truth-Table Oracle & CHI Engineer
"""

import json
import pytest
from typing import List, Tuple


# =============================================================================
# TASK 1: TIMEOUT HINT GENERATOR TESTS
# =============================================================================

class TestTimeoutHintGenerator:
    """
    Tests for suggest_timeout_ms().
    
    Properties to verify:
    - MONOTONICITY: Higher CHI => never smaller suggested timeout
    - DETERMINISM: Same CHI => same suggestion
    - REASONABLE RANGES: "trivial" vs "extreme" should differ significantly
    """

    def test_monotonicity_increasing_chi(self):
        """Higher CHI values should never produce smaller timeout hints."""
        from normalization.tt_chi import suggest_timeout_ms
        
        chi_values = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 50.0, 100.0]
        previous_timeout = 0
        
        for chi in chi_values:
            timeout = suggest_timeout_ms(chi)
            assert timeout >= previous_timeout, \
                f"Monotonicity violated: CHI {chi} gave timeout {timeout}, but previous was {previous_timeout}"
            previous_timeout = timeout

    def test_monotonicity_fine_grained(self):
        """Test monotonicity with fine-grained CHI increments."""
        from normalization.tt_chi import suggest_timeout_ms
        
        # Test across threshold boundaries
        for base_chi in [0.0, 3.0, 8.0, 15.0, 25.0]:
            for delta in [0.0, 0.5, 1.0, 1.5, 2.0]:
                chi1 = base_chi + delta
                chi2 = base_chi + delta + 0.1
                
                t1 = suggest_timeout_ms(chi1)
                t2 = suggest_timeout_ms(chi2)
                
                assert t2 >= t1, f"Monotonicity violated at CHI {chi1} -> {chi2}: {t1} -> {t2}"

    def test_determinism_same_chi(self):
        """Same CHI value should always produce same timeout suggestion."""
        from normalization.tt_chi import suggest_timeout_ms
        
        test_values = [1.5, 5.0, 10.0, 20.0, 35.0]
        
        for chi in test_values:
            results = [suggest_timeout_ms(chi) for _ in range(100)]
            assert all(r == results[0] for r in results), \
                f"Non-deterministic results for CHI {chi}: {set(results)}"

    def test_reasonable_range_trivial(self):
        """Trivial CHI (< 3) should suggest minimal timeout (~100ms)."""
        from normalization.tt_chi import suggest_timeout_ms
        
        for chi in [0.0, 0.5, 1.0, 2.0, 2.9]:
            timeout = suggest_timeout_ms(chi)
            assert timeout == 100, f"Trivial CHI {chi} should suggest 100ms, got {timeout}"

    def test_reasonable_range_extreme(self):
        """Extreme CHI (>= 25) should suggest large timeout (2000ms+)."""
        from normalization.tt_chi import suggest_timeout_ms
        
        for chi in [25.0, 30.0, 50.0, 75.0]:
            timeout = suggest_timeout_ms(chi)
            assert timeout >= 2000, f"Extreme CHI {chi} should suggest >= 2000ms, got {timeout}"

    def test_significant_difference_trivial_vs_extreme(self):
        """Trivial and extreme timeouts should differ significantly."""
        from normalization.tt_chi import suggest_timeout_ms
        
        trivial_timeout = suggest_timeout_ms(1.0)  # trivial
        extreme_timeout = suggest_timeout_ms(50.0)  # extreme
        
        # Extreme should be at least 10x trivial
        assert extreme_timeout >= trivial_timeout * 10, \
            f"Extreme ({extreme_timeout}) should be >> trivial ({trivial_timeout})"

    def test_category_boundaries(self):
        """Test behavior at category boundaries."""
        from normalization.tt_chi import suggest_timeout_ms, classify_hardness
        
        # Just below/above each threshold
        boundaries = [
            (2.9, "trivial"),
            (3.0, "easy"),
            (7.9, "easy"),
            (8.0, "moderate"),
            (14.9, "moderate"),
            (15.0, "hard"),
            (24.9, "hard"),
            (25.0, "extreme"),
        ]
        
        for chi, expected_cat in boundaries:
            actual_cat = classify_hardness(chi)
            assert actual_cat == expected_cat, f"CHI {chi} should be {expected_cat}, got {actual_cat}"
            
            # Timeout should be in reasonable range for category
            timeout = suggest_timeout_ms(chi)
            assert timeout > 0, f"Timeout must be positive for CHI {chi}"

    def test_chi_result_suggested_timeout_property(self):
        """CHIResult.suggested_timeout_ms should use suggest_timeout_ms()."""
        from normalization.tt_chi import CHIResult, suggest_timeout_ms
        
        result = CHIResult(
            chi=12.0,
            atom_count=4,
            assignment_count=16,
            assignments_evaluated=16,
            elapsed_ns=16000,
            efficiency_ratio=1.0,
            throughput_ns_per_assignment=1000.0,
        )
        
        expected = suggest_timeout_ms(12.0)
        assert result.suggested_timeout_ms == expected


# =============================================================================
# TASK 2: POLICY SIGNAL TESTS
# =============================================================================

class TestHardnessPolicySignal:
    """
    Tests for HardnessPolicySignal.
    
    Verify:
    - JSON serialization is stable
    - All fields are present
    - from_chi() and from_chi_result() work correctly
    """

    def test_from_chi_creates_signal(self):
        """from_chi() should create a valid policy signal."""
        from normalization.tt_chi import HardnessPolicySignal, classify_hardness, suggest_timeout_ms
        
        signal = HardnessPolicySignal.from_chi(10.0)
        
        assert signal.chi == 10.0
        assert signal.category == classify_hardness(10.0)
        assert signal.suggested_timeout_ms == suggest_timeout_ms(10.0)
        assert len(signal.description) > 0

    def test_from_chi_result_creates_signal(self):
        """from_chi_result() should create a valid policy signal."""
        from normalization.tt_chi import HardnessPolicySignal, CHIResult
        
        chi_result = CHIResult(
            chi=15.5,
            atom_count=4,
            assignment_count=16,
            assignments_evaluated=16,
            elapsed_ns=16000,
            efficiency_ratio=1.0,
            throughput_ns_per_assignment=1000.0,
        )
        
        signal = HardnessPolicySignal.from_chi_result(chi_result)
        
        assert signal.chi == chi_result.chi
        assert signal.category == chi_result.hardness_category

    def test_to_dict_structure(self):
        """to_dict() should return dict with all required fields."""
        from normalization.tt_chi import HardnessPolicySignal
        
        signal = HardnessPolicySignal.from_chi(8.0)
        d = signal.to_dict()
        
        required_keys = {"chi", "category", "suggested_timeout_ms", "description"}
        assert set(d.keys()) == required_keys, f"Missing keys: {required_keys - set(d.keys())}"

    def test_to_json_is_valid_json(self):
        """to_json() should produce valid JSON string."""
        from normalization.tt_chi import HardnessPolicySignal
        
        signal = HardnessPolicySignal.from_chi(20.0)
        json_str = signal.to_json()
        
        # Should be parseable
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        
        # Round-trip should preserve structure
        assert parsed["category"] == signal.category
        assert parsed["suggested_timeout_ms"] == signal.suggested_timeout_ms

    def test_json_is_stable_ordering(self):
        """JSON output should have stable key ordering (sorted)."""
        from normalization.tt_chi import HardnessPolicySignal
        
        signal = HardnessPolicySignal.from_chi(5.0)
        
        # Multiple serializations should produce identical output
        json1 = signal.to_json()
        json2 = signal.to_json()
        json3 = signal.to_json()
        
        assert json1 == json2 == json3, "JSON output should be deterministic"

    def test_json_compact_mode(self):
        """to_json(indent=None) should produce compact JSON."""
        from normalization.tt_chi import HardnessPolicySignal
        
        signal = HardnessPolicySignal.from_chi(5.0)
        compact = signal.to_json(indent=None)
        
        # Compact JSON should not have newlines
        assert "\n" not in compact

    def test_all_categories_have_descriptions(self):
        """All hardness categories should have descriptions."""
        from normalization.tt_chi import HardnessPolicySignal
        
        test_chi_values = [1.0, 5.0, 10.0, 20.0, 30.0]  # trivial to extreme
        
        for chi in test_chi_values:
            signal = HardnessPolicySignal.from_chi(chi)
            assert len(signal.description) > 0, f"No description for CHI {chi}"
            assert signal.description != "Unknown hardness category."

    def test_chi_precision_in_dict(self):
        """CHI value in dict should be rounded to 4 decimal places."""
        from normalization.tt_chi import HardnessPolicySignal
        
        signal = HardnessPolicySignal.from_chi(10.123456789)
        d = signal.to_dict()
        
        # Should be rounded to 4 places
        assert d["chi"] == 10.1235  # Rounded from 10.123456789


# =============================================================================
# TASK 3: DIAGNOSTICS FORMATTER TESTS
# =============================================================================

class TestDiagnosticsFormatter:
    """
    Tests for format_diagnostics_for_report().
    
    Verify:
    - Stable formatting (no random ordering)
    - All required fields present
    - Semantics unchanged when formatting enabled
    """

    def test_format_produces_stable_output(self):
        """format_diagnostics_for_report() should produce deterministic output."""
        from normalization.tt_chi import format_diagnostics_for_report
        
        diag = {
            "formula": "p -> q",
            "normalized_formula": "(p)->((q))",
            "atom_count": 2,
            "assignment_count": 4,
            "assignments_evaluated": 2,
            "elapsed_ns": 5000,
            "short_circuit_triggered": True,
            "timeout_flag": False,
            "result": False,
        }
        
        # Multiple calls should produce identical output
        output1 = format_diagnostics_for_report(diag)
        output2 = format_diagnostics_for_report(diag)
        output3 = format_diagnostics_for_report(diag)
        
        assert output1 == output2 == output3, "Output should be deterministic"

    def test_format_contains_all_key_fields(self):
        """Output should contain all key diagnostic fields."""
        from normalization.tt_chi import format_diagnostics_for_report
        
        diag = {
            "formula": "p -> q",
            "normalized_formula": "(p)->((q))",
            "atom_count": 2,
            "assignment_count": 4,
            "assignments_evaluated": 2,
            "elapsed_ns": 5000,
            "short_circuit_triggered": True,
            "timeout_flag": False,
            "result": False,
        }
        
        output = format_diagnostics_for_report(diag)
        
        # Check for key information
        assert "p -> q" in output or "p->q" in output, "Formula should be in output"
        assert "2" in output, "Atom count should be in output"
        assert "4" in output, "Assignment count should be in output"
        assert "Short-Circuit" in output or "short" in output.lower()
        assert "Timeout" in output or "timeout" in output.lower()

    def test_format_handles_long_formula(self):
        """Long formulas should be truncated gracefully."""
        from normalization.tt_chi import format_diagnostics_for_report
        
        long_formula = "a -> (b -> (c -> (d -> (e -> (f -> (g -> (h -> (i -> (j -> (k -> (l -> a))))))))))))"
        
        diag = {
            "formula": long_formula,
            "normalized_formula": long_formula,
            "atom_count": 12,
            "assignment_count": 4096,
            "assignments_evaluated": 4096,
            "elapsed_ns": 50000,
            "short_circuit_triggered": False,
            "timeout_flag": False,
            "result": True,
        }
        
        output = format_diagnostics_for_report(diag)
        
        # Should not exceed reasonable line length
        lines = output.split("\n")
        for line in lines:
            assert len(line) < 100, f"Line too long: {line}"

    def test_format_handles_timeout_case(self):
        """Timeout flag should be clearly indicated."""
        from normalization.tt_chi import format_diagnostics_for_report
        
        diag = {
            "formula": "heavy formula",
            "normalized_formula": "heavy formula",
            "atom_count": 15,
            "assignment_count": 32768,
            "assignments_evaluated": 1000,
            "elapsed_ns": 1000000,
            "short_circuit_triggered": False,
            "timeout_flag": True,
            "result": None,
        }
        
        output = format_diagnostics_for_report(diag)
        
        # Timeout should be clearly marked
        assert "Yes" in output or "True" in output, "Timeout flag should show Yes/True"
        assert "None" in output or "timeout" in output.lower(), "Result should indicate timeout"

    def test_format_handles_missing_optional_fields(self):
        """Formatter should handle missing optional fields gracefully."""
        from normalization.tt_chi import format_diagnostics_for_report
        
        # Minimal diagnostics dict
        diag = {
            "atom_count": 2,
            "assignment_count": 4,
            "assignments_evaluated": 4,
            "elapsed_ns": 1000,
        }
        
        # Should not raise exception
        output = format_diagnostics_for_report(diag)
        assert len(output) > 0

    def test_semantics_unchanged_when_formatting(self):
        """Formatting diagnostics should not alter oracle semantics."""
        import os
        os.environ["TT_ORACLE_DIAGNOSTIC"] = "1"
        
        from normalization.taut import truth_table_is_tautology, get_last_diagnostics, clear_diagnostics
        from normalization.tt_chi import format_diagnostics_for_report
        
        clear_diagnostics()
        
        # Run oracle
        result1 = truth_table_is_tautology("p -> p")
        diag = get_last_diagnostics()
        
        # Format diagnostics (should not affect anything)
        if diag:
            formatted = format_diagnostics_for_report(diag)
        
        # Run oracle again
        result2 = truth_table_is_tautology("p -> p")
        
        # Results should be identical
        assert result1 == result2 == True, "Oracle semantics should be unchanged"

    def test_format_efficiency_percentage(self):
        """Efficiency should be shown as percentage."""
        from normalization.tt_chi import format_diagnostics_for_report
        
        diag = {
            "formula": "p -> q",
            "normalized_formula": "(p)->((q))",
            "atom_count": 2,
            "assignment_count": 4,
            "assignments_evaluated": 2,  # 50% efficiency
            "elapsed_ns": 5000,
            "short_circuit_triggered": True,
            "timeout_flag": False,
            "result": False,
        }
        
        output = format_diagnostics_for_report(diag)
        
        # Should show efficiency as percentage
        assert "50" in output, "50% efficiency should be in output"
        assert "%" in output, "Percentage sign should be in output"


class TestDiagnosticsFormatterIntegration:
    """Integration tests for diagnostics formatter with CLI."""

    def test_formatter_with_real_diagnostics(self):
        """Test formatter with real diagnostics from oracle."""
        import os
        os.environ["TT_ORACLE_DIAGNOSTIC"] = "1"
        
        from normalization.taut import truth_table_is_tautology, get_last_diagnostics, clear_diagnostics
        from normalization.tt_chi import format_diagnostics_for_report
        
        clear_diagnostics()
        
        # Run oracle on various formulas
        test_cases = [
            ("p -> p", True),
            ("p -> q", False),
            ("(p /\\ q) -> p", True),
            ("p /\\ ~p", False),
        ]
        
        for formula, expected in test_cases:
            clear_diagnostics()
            result = truth_table_is_tautology(formula)
            diag = get_last_diagnostics()
            
            assert result == expected
            
            if diag:
                formatted = format_diagnostics_for_report(diag)
                # Should be non-empty and well-formed
                assert len(formatted) > 50, "Formatted report should have content"
                assert "===" in formatted, "Report should have header"


# =============================================================================
# CROSS-FEATURE INTEGRATION TESTS
# =============================================================================

class TestUXPolicyLayerIntegration:
    """Integration tests across all three features."""

    def test_full_workflow_trivial_formula(self):
        """Test full workflow for a trivial formula."""
        import os
        import importlib
        os.environ["TT_ORACLE_DIAGNOSTIC"] = "1"
        
        # Reload to pick up env var change
        import normalization.taut as taut_module
        importlib.reload(taut_module)
        
        from normalization.tt_chi import (
            chi_from_diagnostics,
            suggest_timeout_ms,
            HardnessPolicySignal,
            format_diagnostics_for_report,
        )
        
        taut_module.clear_diagnostics()
        
        # 1. Run oracle
        result = taut_module.truth_table_is_tautology("p -> p")
        assert result is True
        
        # 2. Get diagnostics
        diag = taut_module.get_last_diagnostics()
        assert diag is not None
        
        # 3. Compute CHI
        chi_result = chi_from_diagnostics(diag)
        # CHI can vary due to system timing - just verify it's computed
        assert chi_result.chi > 0
        assert chi_result.hardness_category in ["trivial", "easy", "moderate", "hard", "extreme"]
        
        # 4. Get timeout hint (Task 1)
        hint = suggest_timeout_ms(chi_result.chi)
        assert hint >= 100  # At least minimum
        
        # 5. Create policy signal (Task 2)
        signal = HardnessPolicySignal.from_chi_result(chi_result)
        json_str = signal.to_json()
        assert signal.category in json_str
        
        # 6. Format for report (Task 3)
        report = format_diagnostics_for_report(diag)
        assert "p -> p" in report or "p->p" in report

    def test_full_workflow_moderate_formula(self):
        """Test full workflow for a moderate complexity formula."""
        import os
        import importlib
        os.environ["TT_ORACLE_DIAGNOSTIC"] = "1"
        
        # Reload to pick up env var change
        import normalization.taut as taut_module
        importlib.reload(taut_module)
        
        from normalization.tt_chi import (
            chi_from_diagnostics,
            suggest_timeout_ms,
            HardnessPolicySignal,
            format_diagnostics_for_report,
        )
        
        taut_module.clear_diagnostics()
        
        # Formula with 4 atoms
        formula = "((p -> q) -> (q -> r)) -> (p -> r)"
        result = taut_module.truth_table_is_tautology(formula)
        
        diag = taut_module.get_last_diagnostics()
        if diag:
            chi_result = chi_from_diagnostics(diag)
            
            # CHI can vary wildly based on system load/timing
            # Just verify it's computed and positive
            assert chi_result.chi > 0
            assert chi_result.hardness_category in ["trivial", "easy", "moderate", "hard", "extreme"]
            
            # Timeout hint should be at least minimum
            hint = suggest_timeout_ms(chi_result.chi)
            assert hint >= 100
            
            # Policy signal should be serializable
            signal = HardnessPolicySignal.from_chi_result(chi_result)
            parsed = json.loads(signal.to_json())
            assert parsed["category"] == chi_result.hardness_category

    def test_all_features_are_side_effect_free(self):
        """Verify that all features are observation-only."""
        import os
        os.environ["TT_ORACLE_DIAGNOSTIC"] = "1"
        
        from normalization.taut import truth_table_is_tautology, get_last_diagnostics, clear_diagnostics
        from normalization.tt_chi import (
            chi_from_diagnostics,
            suggest_timeout_ms,
            HardnessPolicySignal,
            format_diagnostics_for_report,
            classify_hardness,
        )
        
        # Run baseline
        clear_diagnostics()
        baseline_taut = truth_table_is_tautology("p -> p")
        baseline_non = truth_table_is_tautology("p -> q")
        
        # Use all features
        diag = get_last_diagnostics()
        if diag:
            _ = chi_from_diagnostics(diag)
            _ = suggest_timeout_ms(10.0)
            _ = HardnessPolicySignal.from_chi(10.0)
            _ = format_diagnostics_for_report(diag)
            _ = classify_hardness(10.0)
        
        # Oracle should behave identically
        assert truth_table_is_tautology("p -> p") == baseline_taut
        assert truth_table_is_tautology("p -> q") == baseline_non


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

