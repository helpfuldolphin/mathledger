"""
Tests for Phase III — Risk Envelope & Policy Hooks.

TASK 1: Hardness Risk Envelope Contract
TASK 2: Policy Guardrail Translator (Advisory Only)
TASK 3: Global Health / Governance Hook

Agent B2 - Truth-Table Oracle & CHI Engineer
"""

import json
import pytest
from typing import Dict, Any, List


# =============================================================================
# TASK 1: RISK ENVELOPE CONTRACT TESTS
# =============================================================================

class TestHardnessRiskEnvelope:
    """
    Tests for build_hardness_risk_envelope().
    
    Verify:
    - Contract schema is stable
    - Risk band mapping is correct
    - Output is deterministic and JSON-safe
    """

    def test_envelope_schema_version(self):
        """Envelope must include schema version for compatibility."""
        from normalization.tt_chi import (
            build_hardness_risk_envelope,
            CHIResult,
            RISK_ENVELOPE_SCHEMA_VERSION,
        )
        
        chi_result = CHIResult(
            chi=5.0,
            atom_count=2,
            assignment_count=4,
            assignments_evaluated=4,
            elapsed_ns=5000,
            efficiency_ratio=1.0,
            throughput_ns_per_assignment=1250.0,
        )
        
        envelope = build_hardness_risk_envelope(chi_result)
        
        assert "schema_version" in envelope
        assert envelope["schema_version"] == RISK_ENVELOPE_SCHEMA_VERSION
        assert envelope["schema_version"] == "1.0.0"

    def test_envelope_contains_all_required_fields(self):
        """Envelope must contain all specified fields."""
        from normalization.tt_chi import build_hardness_risk_envelope, CHIResult
        
        chi_result = CHIResult(
            chi=10.0,
            atom_count=3,
            assignment_count=8,
            assignments_evaluated=8,
            elapsed_ns=10000,
            efficiency_ratio=1.0,
            throughput_ns_per_assignment=1250.0,
        )
        
        envelope = build_hardness_risk_envelope(chi_result)
        
        required_fields = {
            "schema_version",
            "chi",
            "category",
            "suggested_timeout_ms",
            "risk_band",
            "notes",
        }
        
        assert set(envelope.keys()) == required_fields

    def test_risk_band_mapping_trivial(self):
        """Trivial category should map to LOW risk band."""
        from normalization.tt_chi import build_hardness_risk_envelope, CHIResult
        
        chi_result = CHIResult(
            chi=2.0,  # trivial
            atom_count=1,
            assignment_count=2,
            assignments_evaluated=2,
            elapsed_ns=2000,
            efficiency_ratio=1.0,
            throughput_ns_per_assignment=1000.0,
        )
        
        envelope = build_hardness_risk_envelope(chi_result)
        
        assert envelope["category"] == "trivial"
        assert envelope["risk_band"] == "LOW"

    def test_risk_band_mapping_easy(self):
        """Easy category should map to LOW risk band."""
        from normalization.tt_chi import build_hardness_risk_envelope, CHIResult
        
        chi_result = CHIResult(
            chi=5.0,  # easy
            atom_count=2,
            assignment_count=4,
            assignments_evaluated=4,
            elapsed_ns=5000,
            efficiency_ratio=1.0,
            throughput_ns_per_assignment=1250.0,
        )
        
        envelope = build_hardness_risk_envelope(chi_result)
        
        assert envelope["category"] == "easy"
        assert envelope["risk_band"] == "LOW"

    def test_risk_band_mapping_moderate(self):
        """Moderate category should map to MEDIUM risk band."""
        from normalization.tt_chi import build_hardness_risk_envelope, CHIResult
        
        chi_result = CHIResult(
            chi=12.0,  # moderate
            atom_count=3,
            assignment_count=8,
            assignments_evaluated=8,
            elapsed_ns=12000,
            efficiency_ratio=1.0,
            throughput_ns_per_assignment=1500.0,
        )
        
        envelope = build_hardness_risk_envelope(chi_result)
        
        assert envelope["category"] == "moderate"
        assert envelope["risk_band"] == "MEDIUM"

    def test_risk_band_mapping_hard(self):
        """Hard category should map to HIGH risk band."""
        from normalization.tt_chi import build_hardness_risk_envelope, CHIResult
        
        chi_result = CHIResult(
            chi=20.0,  # hard
            atom_count=4,
            assignment_count=16,
            assignments_evaluated=16,
            elapsed_ns=20000,
            efficiency_ratio=1.0,
            throughput_ns_per_assignment=1250.0,
        )
        
        envelope = build_hardness_risk_envelope(chi_result)
        
        assert envelope["category"] == "hard"
        assert envelope["risk_band"] == "HIGH"

    def test_risk_band_mapping_extreme(self):
        """Extreme category should map to EXTREME risk band."""
        from normalization.tt_chi import build_hardness_risk_envelope, CHIResult
        
        chi_result = CHIResult(
            chi=30.0,  # extreme
            atom_count=5,
            assignment_count=32,
            assignments_evaluated=32,
            elapsed_ns=30000,
            efficiency_ratio=1.0,
            throughput_ns_per_assignment=937.5,
        )
        
        envelope = build_hardness_risk_envelope(chi_result)
        
        assert envelope["category"] == "extreme"
        assert envelope["risk_band"] == "EXTREME"

    def test_envelope_is_json_safe(self):
        """Envelope must be JSON-serializable."""
        from normalization.tt_chi import build_hardness_risk_envelope, CHIResult
        
        chi_result = CHIResult(
            chi=15.5,
            atom_count=4,
            assignment_count=16,
            assignments_evaluated=16,
            elapsed_ns=15500,
            efficiency_ratio=1.0,
            throughput_ns_per_assignment=968.75,
        )
        
        envelope = build_hardness_risk_envelope(chi_result)
        
        # Should not raise
        json_str = json.dumps(envelope)
        parsed = json.loads(json_str)
        
        assert parsed["chi"] == envelope["chi"]
        assert parsed["risk_band"] == envelope["risk_band"]

    def test_envelope_is_deterministic(self):
        """Same CHIResult should always produce identical envelope."""
        from normalization.tt_chi import build_hardness_risk_envelope, CHIResult
        
        chi_result = CHIResult(
            chi=18.0,
            atom_count=4,
            assignment_count=16,
            assignments_evaluated=16,
            elapsed_ns=18000,
            efficiency_ratio=1.0,
            throughput_ns_per_assignment=1125.0,
        )
        
        envelopes = [build_hardness_risk_envelope(chi_result) for _ in range(10)]
        
        # All should be identical
        for envelope in envelopes:
            assert envelope == envelopes[0]

    def test_envelope_chi_precision(self):
        """CHI value should be rounded to 4 decimal places."""
        from normalization.tt_chi import build_hardness_risk_envelope, CHIResult
        
        chi_result = CHIResult(
            chi=12.123456789,
            atom_count=3,
            assignment_count=8,
            assignments_evaluated=8,
            elapsed_ns=12000,
            efficiency_ratio=1.0,
            throughput_ns_per_assignment=1500.0,
        )
        
        envelope = build_hardness_risk_envelope(chi_result)
        
        assert envelope["chi"] == 12.1235  # Rounded

    def test_envelope_notes_are_neutral(self):
        """Notes should be descriptive only, not value judgments."""
        from normalization.tt_chi import build_hardness_risk_envelope, CHIResult, RISK_BAND_NOTES
        
        # Check all notes don't contain "good", "bad", etc.
        for band, note in RISK_BAND_NOTES.items():
            assert "good" not in note.lower()
            assert "bad" not in note.lower()
            assert "wrong" not in note.lower()
            assert "correct" not in note.lower()


# =============================================================================
# TASK 2: POLICY GUARDRAIL TRANSLATOR TESTS
# =============================================================================

class TestPolicyGuardrailTranslator:
    """
    Tests for derive_timeout_policy_recommendation().
    
    Verify:
    - Correct policy hints for each risk band
    - Human review flag only for EXTREME
    - Advisory nature (no behavioral side effects)
    """

    def test_policy_contains_required_fields(self):
        """Policy recommendation must contain all specified fields."""
        from normalization.tt_chi import (
            build_hardness_risk_envelope,
            derive_timeout_policy_recommendation,
            CHIResult,
        )
        
        chi_result = CHIResult(
            chi=10.0,
            atom_count=3,
            assignment_count=8,
            assignments_evaluated=8,
            elapsed_ns=10000,
            efficiency_ratio=1.0,
            throughput_ns_per_assignment=1250.0,
        )
        
        envelope = build_hardness_risk_envelope(chi_result)
        policy = derive_timeout_policy_recommendation(envelope)
        
        required_fields = {
            "recommended_timeout_ms",
            "requires_human_review",
            "policy_hint",
        }
        
        assert set(policy.keys()) == required_fields

    def test_policy_hint_low_risk(self):
        """LOW risk band should suggest SAFE_FOR_INTERACTIVE."""
        from normalization.tt_chi import (
            build_hardness_risk_envelope,
            derive_timeout_policy_recommendation,
            CHIResult,
        )
        
        chi_result = CHIResult(
            chi=2.0,  # trivial → LOW
            atom_count=1,
            assignment_count=2,
            assignments_evaluated=2,
            elapsed_ns=2000,
            efficiency_ratio=1.0,
            throughput_ns_per_assignment=1000.0,
        )
        
        envelope = build_hardness_risk_envelope(chi_result)
        policy = derive_timeout_policy_recommendation(envelope)
        
        assert policy["policy_hint"] == "SAFE_FOR_INTERACTIVE"
        assert policy["requires_human_review"] is False

    def test_policy_hint_medium_risk(self):
        """MEDIUM risk band should suggest CONSIDER_ASYNC."""
        from normalization.tt_chi import (
            build_hardness_risk_envelope,
            derive_timeout_policy_recommendation,
            CHIResult,
        )
        
        chi_result = CHIResult(
            chi=12.0,  # moderate → MEDIUM
            atom_count=3,
            assignment_count=8,
            assignments_evaluated=8,
            elapsed_ns=12000,
            efficiency_ratio=1.0,
            throughput_ns_per_assignment=1500.0,
        )
        
        envelope = build_hardness_risk_envelope(chi_result)
        policy = derive_timeout_policy_recommendation(envelope)
        
        assert policy["policy_hint"] == "CONSIDER_ASYNC"
        assert policy["requires_human_review"] is False

    def test_policy_hint_high_risk(self):
        """HIGH risk band should suggest USE_TIMEOUT_AND_MONITOR."""
        from normalization.tt_chi import (
            build_hardness_risk_envelope,
            derive_timeout_policy_recommendation,
            CHIResult,
        )
        
        chi_result = CHIResult(
            chi=20.0,  # hard → HIGH
            atom_count=4,
            assignment_count=16,
            assignments_evaluated=16,
            elapsed_ns=20000,
            efficiency_ratio=1.0,
            throughput_ns_per_assignment=1250.0,
        )
        
        envelope = build_hardness_risk_envelope(chi_result)
        policy = derive_timeout_policy_recommendation(envelope)
        
        assert policy["policy_hint"] == "USE_TIMEOUT_AND_MONITOR"
        assert policy["requires_human_review"] is False

    def test_policy_hint_extreme_risk(self):
        """EXTREME risk band should suggest NOT_SUITABLE_FOR_NAIVE_TT."""
        from normalization.tt_chi import (
            build_hardness_risk_envelope,
            derive_timeout_policy_recommendation,
            CHIResult,
        )
        
        chi_result = CHIResult(
            chi=30.0,  # extreme → EXTREME
            atom_count=5,
            assignment_count=32,
            assignments_evaluated=32,
            elapsed_ns=30000,
            efficiency_ratio=1.0,
            throughput_ns_per_assignment=937.5,
        )
        
        envelope = build_hardness_risk_envelope(chi_result)
        policy = derive_timeout_policy_recommendation(envelope)
        
        assert policy["policy_hint"] == "NOT_SUITABLE_FOR_NAIVE_TT"
        assert policy["requires_human_review"] is True  # Only EXTREME requires review

    def test_human_review_only_for_extreme(self):
        """requires_human_review should be True ONLY for EXTREME."""
        from normalization.tt_chi import derive_timeout_policy_recommendation
        
        test_cases = [
            ({"risk_band": "LOW", "suggested_timeout_ms": 100}, False),
            ({"risk_band": "MEDIUM", "suggested_timeout_ms": 500}, False),
            ({"risk_band": "HIGH", "suggested_timeout_ms": 2000}, False),
            ({"risk_band": "EXTREME", "suggested_timeout_ms": 10000}, True),
        ]
        
        for envelope, expected_review in test_cases:
            policy = derive_timeout_policy_recommendation(envelope)
            assert policy["requires_human_review"] == expected_review, \
                f"Risk band {envelope['risk_band']} should have review={expected_review}"

    def test_recommended_timeout_matches_suggested(self):
        """recommended_timeout_ms should equal suggested_timeout_ms from envelope."""
        from normalization.tt_chi import (
            build_hardness_risk_envelope,
            derive_timeout_policy_recommendation,
            CHIResult,
        )
        
        chi_result = CHIResult(
            chi=15.0,
            atom_count=4,
            assignment_count=16,
            assignments_evaluated=16,
            elapsed_ns=15000,
            efficiency_ratio=1.0,
            throughput_ns_per_assignment=937.5,
        )
        
        envelope = build_hardness_risk_envelope(chi_result)
        policy = derive_timeout_policy_recommendation(envelope)
        
        assert policy["recommended_timeout_ms"] == envelope["suggested_timeout_ms"]

    def test_policy_is_advisory_only(self):
        """
        Policy recommendations should not affect oracle behavior.
        
        This test verifies that deriving policies has no side effects.
        """
        import os
        os.environ["TT_ORACLE_DIAGNOSTIC"] = "1"
        
        from normalization.taut import truth_table_is_tautology
        from normalization.tt_chi import (
            build_hardness_risk_envelope,
            derive_timeout_policy_recommendation,
            CHIResult,
        )
        
        # Baseline oracle result
        result1 = truth_table_is_tautology("p -> p")
        
        # Build and derive policy (should have no side effects)
        chi_result = CHIResult(
            chi=30.0,
            atom_count=5,
            assignment_count=32,
            assignments_evaluated=32,
            elapsed_ns=30000,
            efficiency_ratio=1.0,
            throughput_ns_per_assignment=937.5,
        )
        envelope = build_hardness_risk_envelope(chi_result)
        policy = derive_timeout_policy_recommendation(envelope)
        
        # Oracle should behave identically
        result2 = truth_table_is_tautology("p -> p")
        
        assert result1 == result2 == True

    def test_all_policy_hints_are_valid(self):
        """All valid risk bands should produce valid policy hints."""
        from normalization.tt_chi import (
            derive_timeout_policy_recommendation,
            POLICY_HINT_BY_RISK_BAND,
        )
        
        valid_hints = set(POLICY_HINT_BY_RISK_BAND.values())
        
        for risk_band in ["LOW", "MEDIUM", "HIGH", "EXTREME"]:
            envelope = {"risk_band": risk_band, "suggested_timeout_ms": 1000}
            policy = derive_timeout_policy_recommendation(envelope)
            
            assert policy["policy_hint"] in valid_hints


# =============================================================================
# TASK 3: GLOBAL HEALTH / GOVERNANCE HOOK TESTS
# =============================================================================

class TestGlobalHealthSummary:
    """
    Tests for summarize_tt_hardness_for_global_health().
    
    Verify:
    - Correct aggregation of risk envelopes
    - Status determination logic
    - Empty input handling
    """

    def test_empty_input_returns_ok(self):
        """Empty envelope list should return OK status."""
        from normalization.tt_chi import summarize_tt_hardness_for_global_health
        
        health = summarize_tt_hardness_for_global_health([])
        
        assert health["extreme_case_count"] == 0
        assert health["hard_case_count"] == 0
        assert health["fraction_safe_or_easy"] == 1.0
        assert health["total_cases"] == 0
        assert health["status"] == "OK"

    def test_all_low_risk_returns_ok(self):
        """All LOW risk cases should produce OK status."""
        from normalization.tt_chi import summarize_tt_hardness_for_global_health
        
        envelopes = [
            {"risk_band": "LOW"},
            {"risk_band": "LOW"},
            {"risk_band": "LOW"},
            {"risk_band": "LOW"},
            {"risk_band": "LOW"},
        ]
        
        health = summarize_tt_hardness_for_global_health(envelopes)
        
        assert health["extreme_case_count"] == 0
        assert health["hard_case_count"] == 0
        assert health["fraction_safe_or_easy"] == 1.0
        assert health["total_cases"] == 5
        assert health["status"] == "OK"

    def test_single_extreme_triggers_attention(self):
        """Single EXTREME case should trigger ATTENTION status."""
        from normalization.tt_chi import summarize_tt_hardness_for_global_health
        
        envelopes = [
            {"risk_band": "LOW"},
            {"risk_band": "LOW"},
            {"risk_band": "LOW"},
            {"risk_band": "LOW"},
            {"risk_band": "EXTREME"},
        ]
        
        health = summarize_tt_hardness_for_global_health(envelopes)
        
        assert health["extreme_case_count"] == 1
        assert health["hard_case_count"] == 1
        assert health["fraction_safe_or_easy"] == 0.8
        assert health["status"] == "ATTENTION"

    def test_three_extreme_triggers_hot(self):
        """Three or more EXTREME cases should trigger HOT status."""
        from normalization.tt_chi import summarize_tt_hardness_for_global_health
        
        envelopes = [
            {"risk_band": "LOW"},
            {"risk_band": "EXTREME"},
            {"risk_band": "EXTREME"},
            {"risk_band": "EXTREME"},
        ]
        
        health = summarize_tt_hardness_for_global_health(envelopes)
        
        assert health["extreme_case_count"] == 3
        assert health["status"] == "HOT"

    def test_low_safe_fraction_triggers_attention(self):
        """fraction_safe_or_easy < 0.8 should trigger ATTENTION."""
        from normalization.tt_chi import summarize_tt_hardness_for_global_health
        
        # 3 LOW out of 5 = 60% safe
        envelopes = [
            {"risk_band": "LOW"},
            {"risk_band": "LOW"},
            {"risk_band": "LOW"},
            {"risk_band": "MEDIUM"},
            {"risk_band": "HIGH"},
        ]
        
        health = summarize_tt_hardness_for_global_health(envelopes)
        
        assert health["fraction_safe_or_easy"] == 0.6
        assert health["status"] == "ATTENTION"

    def test_very_low_safe_fraction_triggers_hot(self):
        """fraction_safe_or_easy < 0.5 should trigger HOT status."""
        from normalization.tt_chi import summarize_tt_hardness_for_global_health
        
        # 1 LOW out of 5 = 20% safe
        envelopes = [
            {"risk_band": "LOW"},
            {"risk_band": "MEDIUM"},
            {"risk_band": "MEDIUM"},
            {"risk_band": "HIGH"},
            {"risk_band": "HIGH"},
        ]
        
        health = summarize_tt_hardness_for_global_health(envelopes)
        
        assert health["fraction_safe_or_easy"] == 0.2
        assert health["status"] == "HOT"

    def test_hard_case_count_includes_high_and_extreme(self):
        """hard_case_count should include both HIGH and EXTREME."""
        from normalization.tt_chi import summarize_tt_hardness_for_global_health
        
        envelopes = [
            {"risk_band": "LOW"},
            {"risk_band": "MEDIUM"},
            {"risk_band": "HIGH"},
            {"risk_band": "HIGH"},
            {"risk_band": "EXTREME"},
        ]
        
        health = summarize_tt_hardness_for_global_health(envelopes)
        
        assert health["extreme_case_count"] == 1
        assert health["hard_case_count"] == 3  # 2 HIGH + 1 EXTREME

    def test_summary_contains_all_required_fields(self):
        """Summary must contain all specified fields."""
        from normalization.tt_chi import summarize_tt_hardness_for_global_health
        
        envelopes = [{"risk_band": "LOW"}]
        
        health = summarize_tt_hardness_for_global_health(envelopes)
        
        required_fields = {
            "extreme_case_count",
            "hard_case_count",
            "fraction_safe_or_easy",
            "total_cases",
            "status",
        }
        
        assert set(health.keys()) == required_fields

    def test_fraction_precision(self):
        """Fraction should be rounded to 4 decimal places."""
        from normalization.tt_chi import summarize_tt_hardness_for_global_health
        
        # 3 LOW out of 7 = 0.428571...
        envelopes = [
            {"risk_band": "LOW"},
            {"risk_band": "LOW"},
            {"risk_band": "LOW"},
            {"risk_band": "MEDIUM"},
            {"risk_band": "MEDIUM"},
            {"risk_band": "HIGH"},
            {"risk_band": "HIGH"},
        ]
        
        health = summarize_tt_hardness_for_global_health(envelopes)
        
        assert health["fraction_safe_or_easy"] == 0.4286  # Rounded to 4 places

    def test_summary_is_json_safe(self):
        """Summary must be JSON-serializable."""
        from normalization.tt_chi import summarize_tt_hardness_for_global_health
        
        envelopes = [
            {"risk_band": "LOW"},
            {"risk_band": "EXTREME"},
        ]
        
        health = summarize_tt_hardness_for_global_health(envelopes)
        
        # Should not raise
        json_str = json.dumps(health)
        parsed = json.loads(json_str)
        
        assert parsed["status"] == health["status"]


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestPhaseIIIIntegration:
    """Integration tests across all Phase III functions."""

    def test_full_workflow_from_chi_result(self):
        """Test complete workflow from CHIResult to global health."""
        from normalization.tt_chi import (
            build_hardness_risk_envelope,
            derive_timeout_policy_recommendation,
            summarize_tt_hardness_for_global_health,
            CHIResult,
        )
        
        # Create several CHI results
        chi_results = [
            CHIResult(chi=2.0, atom_count=1, assignment_count=2,
                     assignments_evaluated=2, elapsed_ns=2000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1000.0),
            CHIResult(chi=5.0, atom_count=2, assignment_count=4,
                     assignments_evaluated=4, elapsed_ns=5000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1250.0),
            CHIResult(chi=12.0, atom_count=3, assignment_count=8,
                     assignments_evaluated=8, elapsed_ns=12000,
                     efficiency_ratio=1.0, throughput_ns_per_assignment=1500.0),
        ]
        
        # Build envelopes
        envelopes = [build_hardness_risk_envelope(cr) for cr in chi_results]
        
        # Derive policies
        policies = [derive_timeout_policy_recommendation(env) for env in envelopes]
        
        # Summarize for health
        health = summarize_tt_hardness_for_global_health(envelopes)
        
        # Verify chain works correctly
        assert len(envelopes) == 3
        assert len(policies) == 3
        assert health["total_cases"] == 3
        assert health["fraction_safe_or_easy"] == pytest.approx(0.6667, rel=0.01)  # 2/3

    def test_all_functions_are_pure(self):
        """All Phase III functions should be pure (no side effects)."""
        from normalization.tt_chi import (
            build_hardness_risk_envelope,
            derive_timeout_policy_recommendation,
            summarize_tt_hardness_for_global_health,
            CHIResult,
        )
        
        chi_result = CHIResult(
            chi=15.0,
            atom_count=4,
            assignment_count=16,
            assignments_evaluated=16,
            elapsed_ns=15000,
            efficiency_ratio=1.0,
            throughput_ns_per_assignment=937.5,
        )
        
        # Multiple calls should produce identical results
        for _ in range(5):
            env1 = build_hardness_risk_envelope(chi_result)
            env2 = build_hardness_risk_envelope(chi_result)
            assert env1 == env2
            
            pol1 = derive_timeout_policy_recommendation(env1)
            pol2 = derive_timeout_policy_recommendation(env1)
            assert pol1 == pol2
            
            health1 = summarize_tt_hardness_for_global_health([env1])
            health2 = summarize_tt_hardness_for_global_health([env1])
            assert health1 == health2

    def test_workflow_with_real_oracle_diagnostics(self):
        """Test workflow using real oracle diagnostics."""
        import os
        import importlib
        os.environ["TT_ORACLE_DIAGNOSTIC"] = "1"
        
        import normalization.taut as taut_module
        importlib.reload(taut_module)
        
        from normalization.tt_chi import (
            chi_from_diagnostics,
            build_hardness_risk_envelope,
            derive_timeout_policy_recommendation,
            summarize_tt_hardness_for_global_health,
        )
        
        # Run oracle on various formulas
        formulas = [
            "p -> p",
            "p -> q",
            "(p /\\ q) -> p",
        ]
        
        envelopes = []
        for formula in formulas:
            taut_module.clear_diagnostics()
            taut_module.truth_table_is_tautology(formula)
            
            diag = taut_module.get_last_diagnostics()
            if diag:
                chi_result = chi_from_diagnostics(diag)
                envelope = build_hardness_risk_envelope(chi_result)
                envelopes.append(envelope)
        
        # Should have created envelopes
        if envelopes:
            # Derive policies
            policies = [derive_timeout_policy_recommendation(env) for env in envelopes]
            assert all("policy_hint" in p for p in policies)
            
            # Get health summary
            health = summarize_tt_hardness_for_global_health(envelopes)
            assert health["total_cases"] == len(envelopes)
            assert health["status"] in ["OK", "ATTENTION", "HOT"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

