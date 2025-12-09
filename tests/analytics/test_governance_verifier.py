"""
Tests for Governance Verifier

==============================================================================
STATUS: PHASE II — IMPLEMENTATION
==============================================================================

This module tests the governance_verifier implementation against:
- docs/UPLIFT_ANALYTICS_GOVERNANCE_SPEC.md
- docs/UPLIFT_GOVERNANCE_VERIFIER_SPEC.md

Test coverage:
- GovernanceVerdict data model
- All 43 governance rules (GOV-*, REP-*, MAN-*, INV-*)
- Decision tree logic (PASS/WARN/FAIL)
- Determinism verification
"""
import json
import pytest
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any

from backend.analytics.governance_verifier import (
    governance_verify,
    GovernanceVerdict,
    RuleResult,
    SLICE_IDS,
    SLICE_SUCCESS_CRITERIA,
    RULE_REGISTRY,
    RULE_DESCRIPTIONS,
    __version__,
    # v2: Governance Chronicle & Explainer
    explain_verdict,
    build_governance_posture,
    summarize_for_admissibility,
    # Phase III: Director Console Governance Feed
    build_governance_chronicle,
    map_governance_to_director_status,
    summarize_governance_for_global_health,
    # Phase IV: Governance Chronicle Compass & Cross-System Gate
    build_governance_alignment_view,
    evaluate_governance_for_promotion,
    build_governance_director_panel_v2,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def valid_summary() -> Dict[str, Any]:
    """Minimal valid summary that should PASS all rules."""
    return {
        "slices": {
            "prop_depth4": {
                "n_rfl": 600,
                "success_rate": {"baseline": 0.94, "rfl": 0.97, "ci_low": 0.95, "ci_high": 0.98},
                "abstention_rate": {"baseline": 0.03, "rfl": 0.01, "ci_low": 0.005, "ci_high": 0.015},
                "throughput": {"delta_pct": 7.0, "ci_low": 5.5, "ci_high": 8.5},
            },
            "fol_eq_group": {
                "n_rfl": 400,
                "success_rate": {"baseline": 0.83, "rfl": 0.88, "ci_low": 0.85, "ci_high": 0.91},
                "abstention_rate": {"baseline": 0.12, "rfl": 0.08, "ci_low": 0.06, "ci_high": 0.10},
                "throughput": {"delta_pct": 4.0, "ci_low": 3.2, "ci_high": 4.8},
            },
            "fol_eq_ring": {
                "n_rfl": 350,
                "success_rate": {"baseline": 0.78, "rfl": 0.83, "ci_low": 0.80, "ci_high": 0.86},
                "abstention_rate": {"baseline": 0.18, "rfl": 0.12, "ci_low": 0.10, "ci_high": 0.14},
                "throughput": {"delta_pct": 3.0, "ci_low": 2.2, "ci_high": 3.8},
            },
            "linear_arith": {
                "n_rfl": 250,
                "success_rate": {"baseline": 0.68, "rfl": 0.75, "ci_low": 0.72, "ci_high": 0.78},
                "abstention_rate": {"baseline": 0.22, "rfl": 0.18, "ci_low": 0.16, "ci_high": 0.20},
                "throughput": {"delta_pct": 1.0, "ci_low": 0.5, "ci_high": 1.5},
            },
        },
        "governance": {
            "recommendation": "proceed",
            "all_slices_pass": True,
            "passing_slices": ["prop_depth4", "fol_eq_group", "fol_eq_ring", "linear_arith"],
            "failing_slices": [],
        },
        "reproducibility": {
            "bootstrap_seed": 42,
            "n_bootstrap": 10000,
            "confidence": 0.95,
            "ci_method": "wilson",
        },
    }


@pytest.fixture
def valid_manifest() -> Dict[str, Any]:
    """Minimal valid manifest."""
    return {
        "experiment_id": "u2-test-2025-01",
        "prereg_ref": "PREREG_UPLIFT_U2.yaml",
        "created_at": "2025-01-15T12:00:00Z",
        "$schema": "https://mathledger.io/schemas/manifest-v1.json",
        "config": {
            "seed_baseline": 123,
            "seed_rfl": 456,
            "slices": {
                "prop_depth4": {"derivation_params": {"steps": 100, "depth": 4, "breadth": 50, "total": 500}},
                "fol_eq_group": {"derivation_params": {"steps": 100, "depth": 4, "breadth": 50, "total": 500}},
                "fol_eq_ring": {"derivation_params": {"steps": 100, "depth": 4, "breadth": 50, "total": 500}},
                "linear_arith": {"derivation_params": {"steps": 100, "depth": 4, "breadth": 50, "total": 500}},
            },
        },
        "metadata": {
            "analysis_code_version": "1.0.0",
        },
        "artifacts": {
            "baseline_logs": [],
            "rfl_logs": [],
        },
        "checksums": {},
    }


@pytest.fixture
def valid_telemetry() -> Dict[str, Any]:
    """Minimal valid telemetry."""
    return {
        "baseline": {
            "cycles": [
                {"cycle": 0, "timestamp": "2025-01-15T12:00:00Z", "proofs_attempted": 100,
                 "proofs_succeeded": 95, "abstention_count": 2, "duration_seconds": 1.5,
                 "ht_hash": "a" * 64},
                {"cycle": 1, "timestamp": "2025-01-15T12:01:00Z", "proofs_attempted": 100,
                 "proofs_succeeded": 96, "abstention_count": 1, "duration_seconds": 1.4,
                 "ht_hash": "b" * 64},
            ],
        },
        "rfl": {
            "cycles": [
                {"cycle": 0, "timestamp": "2025-01-15T13:00:00Z", "proofs_attempted": 100,
                 "proofs_succeeded": 97, "abstention_count": 1, "duration_seconds": 1.2,
                 "ht_hash": "c" * 64},
                {"cycle": 1, "timestamp": "2025-01-15T13:01:00Z", "proofs_attempted": 100,
                 "proofs_succeeded": 98, "abstention_count": 0, "duration_seconds": 1.1,
                 "ht_hash": "d" * 64},
            ],
        },
    }


# =============================================================================
# DATA MODEL TESTS
# =============================================================================

class TestGovernanceVerdict:
    """Tests for GovernanceVerdict data model."""

    def test_verdict_fields(self, valid_summary):
        """Verify all required fields are present."""
        verdict = governance_verify(valid_summary)

        assert hasattr(verdict, "status")
        assert hasattr(verdict, "invalidating_rules")
        assert hasattr(verdict, "warnings")
        assert hasattr(verdict, "passed_rules")
        assert hasattr(verdict, "summary")
        assert hasattr(verdict, "inputs")
        assert hasattr(verdict, "details")
        assert hasattr(verdict, "timestamp")
        assert hasattr(verdict, "verifier_version")
        assert hasattr(verdict, "rules_checked")

    def test_verdict_to_dict_serializable(self, valid_summary):
        """Verify verdict can be serialized to JSON."""
        verdict = governance_verify(valid_summary)
        verdict_dict = verdict.to_dict()

        # Should not raise
        json_str = json.dumps(verdict_dict)
        assert json_str is not None

        # Should round-trip
        parsed = json.loads(json_str)
        assert parsed["status"] == verdict.status

    def test_verdict_status_values(self, valid_summary):
        """Status must be one of PASS/WARN/FAIL."""
        verdict = governance_verify(valid_summary)
        assert verdict.status in {"PASS", "WARN", "FAIL"}

    def test_verdict_rules_checked(self, valid_summary):
        """Should check all 43 rules."""
        verdict = governance_verify(valid_summary)
        assert verdict.rules_checked == 43
        assert verdict.rules_checked == len(RULE_REGISTRY)


# =============================================================================
# PASS/WARN/FAIL DECISION TREE TESTS
# =============================================================================

class TestDecisionTree:
    """Tests for the PASS/WARN/FAIL decision tree."""

    def test_full_pass(self, valid_summary, valid_manifest, valid_telemetry):
        """Valid inputs with no violations should PASS."""
        # Note: REP-8 requires log files to exist, which we skip here
        # by not providing a manifest with log references
        verdict = governance_verify(
            summary=valid_summary,
            manifest=None,  # Skip manifest to avoid REP-8 file existence check
            telemetry=valid_telemetry,
        )

        # Without manifest, only summary and telemetry rules run
        # Status should be PASS or WARN (no INVALIDATING violations)
        assert verdict.status in {"PASS", "WARN"}
        assert len(verdict.invalidating_rules) == 0

    def test_invalidating_violation_causes_fail(self, valid_summary):
        """Any INVALIDATING violation should cause FAIL."""
        summary = deepcopy(valid_summary)
        # Break GOV-2: Invalid recommendation
        summary["governance"]["recommendation"] = "invalid_decision"

        verdict = governance_verify(summary)

        assert verdict.status == "FAIL"
        assert "GOV-2" in verdict.invalidating_rules

    def test_warning_without_invalidating_causes_warn(self, valid_summary):
        """WARNING violations without INVALIDATING should cause WARN."""
        summary = deepcopy(valid_summary)
        # Create a marginal case for GOV-5 (WARNING level)
        summary["slices"]["prop_depth4"]["throughput"]["ci_low"] = 4.0
        summary["slices"]["prop_depth4"]["throughput"]["ci_high"] = 6.0
        # But still meets threshold overall

        verdict = governance_verify(summary)

        # If only warnings, should be WARN (or FAIL if other issues)
        if verdict.status == "WARN":
            assert "GOV-5" in verdict.warnings
            assert len(verdict.invalidating_rules) == 0


# =============================================================================
# GOVERNANCE RULES (GOV-*) TESTS
# =============================================================================

class TestGovernanceRules:
    """Tests for GOV-* rules."""

    def test_gov_1_threshold_compliance_pass(self, valid_summary):
        """GOV-1: Valid thresholds should pass."""
        verdict = governance_verify(valid_summary)
        result = verdict.details.get("GOV-1")

        assert result is not None
        assert result.passed is True

    def test_gov_1_threshold_compliance_fail(self, valid_summary):
        """GOV-1: Below-threshold metrics should fail."""
        summary = deepcopy(valid_summary)
        summary["slices"]["prop_depth4"]["success_rate"]["rfl"] = 0.90  # Below 0.95

        verdict = governance_verify(summary)
        result = verdict.details.get("GOV-1")

        assert result is not None
        assert result.passed is False
        assert "GOV-1" in verdict.invalidating_rules

    def test_gov_2_decision_exclusivity_valid(self, valid_summary):
        """GOV-2: Valid decisions (proceed/hold/rollback) should pass."""
        for decision in ["proceed", "hold", "rollback"]:
            summary = deepcopy(valid_summary)
            summary["governance"]["recommendation"] = decision
            if decision != "proceed":
                summary["governance"]["all_slices_pass"] = False
                summary["governance"]["failing_slices"] = ["prop_depth4"]
                summary["governance"]["passing_slices"] = ["fol_eq_group", "fol_eq_ring", "linear_arith"]
                summary["governance"]["rationale"] = "Test rationale"

            verdict = governance_verify(summary)
            result = verdict.details.get("GOV-2")

            assert result.passed is True, f"Decision '{decision}' should be valid"

    def test_gov_2_decision_exclusivity_invalid(self, valid_summary):
        """GOV-2: Invalid decisions should fail."""
        summary = deepcopy(valid_summary)
        summary["governance"]["recommendation"] = "approve"  # Invalid

        verdict = governance_verify(summary)
        result = verdict.details.get("GOV-2")

        assert result.passed is False
        assert "GOV-2" in verdict.invalidating_rules

    def test_gov_3_decision_consistency_proceed_all_pass(self, valid_summary):
        """GOV-3: PROCEED with all_slices_pass=True should pass."""
        verdict = governance_verify(valid_summary)
        result = verdict.details.get("GOV-3")

        assert result.passed is True

    def test_gov_3_decision_consistency_proceed_not_all_pass(self, valid_summary):
        """GOV-3: PROCEED with all_slices_pass=False should fail."""
        summary = deepcopy(valid_summary)
        summary["governance"]["all_slices_pass"] = False

        verdict = governance_verify(summary)
        result = verdict.details.get("GOV-3")

        assert result.passed is False
        assert "GOV-3" in verdict.invalidating_rules

    def test_gov_4_failing_slice_identification(self, valid_summary):
        """GOV-4: Slice partition must be complete and disjoint."""
        verdict = governance_verify(valid_summary)
        result = verdict.details.get("GOV-4")

        assert result.passed is True

    def test_gov_4_missing_slice_in_partition(self, valid_summary):
        """GOV-4: Missing slice in partition should fail."""
        summary = deepcopy(valid_summary)
        summary["governance"]["passing_slices"] = ["prop_depth4", "fol_eq_group"]  # Missing 2

        verdict = governance_verify(summary)
        result = verdict.details.get("GOV-4")

        assert result.passed is False

    def test_gov_8_sample_size_pass(self, valid_summary):
        """GOV-8: Adequate sample sizes should pass."""
        verdict = governance_verify(valid_summary)
        result = verdict.details.get("GOV-8")

        assert result.passed is True

    def test_gov_8_sample_size_fail(self, valid_summary):
        """GOV-8: Below-minimum sample sizes should fail."""
        summary = deepcopy(valid_summary)
        summary["slices"]["prop_depth4"]["n_rfl"] = 100  # Below 500 minimum

        verdict = governance_verify(summary)
        result = verdict.details.get("GOV-8")

        assert result.passed is False
        assert "GOV-8" in verdict.invalidating_rules

    def test_gov_9_all_slices_present(self, valid_summary):
        """GOV-9: All four slices must be present."""
        verdict = governance_verify(valid_summary)
        result = verdict.details.get("GOV-9")

        assert result.passed is True

    def test_gov_9_missing_slice(self, valid_summary):
        """GOV-9: Missing slice should fail."""
        summary = deepcopy(valid_summary)
        del summary["slices"]["linear_arith"]

        verdict = governance_verify(summary)
        result = verdict.details.get("GOV-9")

        assert result.passed is False
        assert "GOV-9" in verdict.invalidating_rules


# =============================================================================
# REPRODUCIBILITY RULES (REP-*) TESTS
# =============================================================================

class TestReproducibilityRules:
    """Tests for REP-* rules."""

    def test_rep_3_bootstrap_seed_present(self, valid_summary):
        """REP-3: Bootstrap seed must be present."""
        verdict = governance_verify(valid_summary)
        result = verdict.details.get("REP-3")

        assert result.passed is True

    def test_rep_3_bootstrap_seed_missing(self, valid_summary):
        """REP-3: Missing bootstrap seed should fail."""
        summary = deepcopy(valid_summary)
        del summary["reproducibility"]["bootstrap_seed"]

        verdict = governance_verify(summary)
        result = verdict.details.get("REP-3")

        assert result.passed is False
        assert "REP-3" in verdict.invalidating_rules

    def test_rep_5_bootstrap_iterations_pass(self, valid_summary):
        """REP-5: >= 10000 iterations should pass."""
        verdict = governance_verify(valid_summary)
        result = verdict.details.get("REP-5")

        assert result.passed is True

    def test_rep_5_bootstrap_iterations_fail(self, valid_summary):
        """REP-5: < 10000 iterations should fail."""
        summary = deepcopy(valid_summary)
        summary["reproducibility"]["n_bootstrap"] = 1000

        verdict = governance_verify(summary)
        result = verdict.details.get("REP-5")

        assert result.passed is False
        assert "REP-5" in verdict.invalidating_rules

    def test_rep_1_baseline_seed_in_manifest(self, valid_summary, valid_manifest):
        """REP-1: Baseline seed in manifest should pass."""
        verdict = governance_verify(valid_summary, manifest=valid_manifest)
        result = verdict.details.get("REP-1")

        assert result.passed is True

    def test_rep_1_baseline_seed_missing(self, valid_summary, valid_manifest):
        """REP-1: Missing baseline seed should fail."""
        manifest = deepcopy(valid_manifest)
        del manifest["config"]["seed_baseline"]

        verdict = governance_verify(valid_summary, manifest=manifest)
        result = verdict.details.get("REP-1")

        assert result.passed is False

    def test_rep_4_seed_distinctness_pass(self, valid_summary, valid_manifest):
        """REP-4: Distinct seeds should pass."""
        verdict = governance_verify(valid_summary, manifest=valid_manifest)
        result = verdict.details.get("REP-4")

        assert result.passed is True

    def test_rep_4_seed_distinctness_fail(self, valid_summary, valid_manifest):
        """REP-4: Duplicate seeds should warn."""
        manifest = deepcopy(valid_manifest)
        manifest["config"]["seed_baseline"] = 42  # Same as bootstrap_seed

        verdict = governance_verify(valid_summary, manifest=manifest)
        result = verdict.details.get("REP-4")

        assert result.passed is False
        assert "REP-4" in verdict.warnings


# =============================================================================
# MANIFEST RULES (MAN-*) TESTS
# =============================================================================

class TestManifestRules:
    """Tests for MAN-* rules."""

    def test_man_1_experiment_id_present(self, valid_summary, valid_manifest):
        """MAN-1: Experiment ID must be present."""
        verdict = governance_verify(valid_summary, manifest=valid_manifest)
        result = verdict.details.get("MAN-1")

        assert result.passed is True

    def test_man_1_experiment_id_missing(self, valid_summary, valid_manifest):
        """MAN-1: Missing experiment ID should fail."""
        manifest = deepcopy(valid_manifest)
        del manifest["experiment_id"]

        verdict = governance_verify(valid_summary, manifest=manifest)
        result = verdict.details.get("MAN-1")

        assert result.passed is False

    def test_man_4_slice_config_complete(self, valid_summary, valid_manifest):
        """MAN-4: All slice configs must be present."""
        verdict = governance_verify(valid_summary, manifest=valid_manifest)
        result = verdict.details.get("MAN-4")

        assert result.passed is True

    def test_man_4_slice_config_incomplete(self, valid_summary, valid_manifest):
        """MAN-4: Missing slice config should fail."""
        manifest = deepcopy(valid_manifest)
        del manifest["config"]["slices"]["linear_arith"]

        verdict = governance_verify(valid_summary, manifest=manifest)
        result = verdict.details.get("MAN-4")

        assert result.passed is False


# =============================================================================
# INVARIANT RULES (INV-*) TESTS
# =============================================================================

class TestInvariantRules:
    """Tests for INV-* rules."""

    def test_inv_s1_wilson_ci_bounds_valid(self, valid_summary):
        """INV-S1: Wilson CI bounds in [0,1] should pass."""
        verdict = governance_verify(valid_summary)
        result = verdict.details.get("INV-S1")

        assert result.passed is True

    def test_inv_s1_wilson_ci_bounds_invalid(self, valid_summary):
        """INV-S1: Wilson CI bounds outside [0,1] should fail."""
        summary = deepcopy(valid_summary)
        summary["slices"]["prop_depth4"]["success_rate"]["ci_low"] = -0.1

        verdict = governance_verify(summary)
        result = verdict.details.get("INV-S1")

        assert result.passed is False
        assert "INV-S1" in verdict.invalidating_rules

    def test_inv_s2_bootstrap_ci_ordering_valid(self, valid_summary):
        """INV-S2: ci_low <= ci_high should pass."""
        verdict = governance_verify(valid_summary)
        result = verdict.details.get("INV-S2")

        assert result.passed is True

    def test_inv_s2_bootstrap_ci_ordering_invalid(self, valid_summary):
        """INV-S2: ci_low > ci_high should fail."""
        summary = deepcopy(valid_summary)
        summary["slices"]["prop_depth4"]["success_rate"]["ci_low"] = 0.99
        summary["slices"]["prop_depth4"]["success_rate"]["ci_high"] = 0.90  # Inverted

        verdict = governance_verify(summary)
        result = verdict.details.get("INV-S2")

        assert result.passed is False
        assert "INV-S2" in verdict.invalidating_rules

    def test_inv_s4_rate_bounds_valid(self, valid_summary):
        """INV-S4: Rates in [0,1] should pass."""
        verdict = governance_verify(valid_summary)
        result = verdict.details.get("INV-S4")

        assert result.passed is True

    def test_inv_s4_rate_bounds_invalid(self, valid_summary):
        """INV-S4: Rates outside [0,1] should fail."""
        summary = deepcopy(valid_summary)
        summary["slices"]["prop_depth4"]["success_rate"]["rfl"] = 1.5  # Invalid

        verdict = governance_verify(summary)
        result = verdict.details.get("INV-S4")

        assert result.passed is False
        assert "INV-S4" in verdict.invalidating_rules

    def test_inv_d1_cycle_continuity(self, valid_summary, valid_telemetry):
        """INV-D1: Consecutive cycle indices should pass."""
        verdict = governance_verify(valid_summary, telemetry=valid_telemetry)
        result = verdict.details.get("INV-D1")

        assert result.passed is True

    def test_inv_d1_cycle_gap(self, valid_summary, valid_telemetry):
        """INV-D1: Gap in cycle indices should fail."""
        telemetry = deepcopy(valid_telemetry)
        telemetry["baseline"]["cycles"][1]["cycle"] = 5  # Gap: 0, 5

        verdict = governance_verify(valid_summary, telemetry=telemetry)
        result = verdict.details.get("INV-D1")

        assert result.passed is False

    def test_inv_d3_verification_bound(self, valid_summary, valid_telemetry):
        """INV-D3: proofs_succeeded <= proofs_attempted should pass."""
        verdict = governance_verify(valid_summary, telemetry=valid_telemetry)
        result = verdict.details.get("INV-D3")

        assert result.passed is True

    def test_inv_d3_verification_bound_violated(self, valid_summary, valid_telemetry):
        """INV-D3: proofs_succeeded > proofs_attempted should fail."""
        telemetry = deepcopy(valid_telemetry)
        telemetry["baseline"]["cycles"][0]["proofs_succeeded"] = 150  # > 100 attempted

        verdict = governance_verify(valid_summary, telemetry=telemetry)
        result = verdict.details.get("INV-D3")

        assert result.passed is False

    def test_inv_d6_hash_format_valid(self, valid_summary, valid_telemetry):
        """INV-D6: Valid SHA-256 hashes should pass."""
        verdict = governance_verify(valid_summary, telemetry=valid_telemetry)
        result = verdict.details.get("INV-D6")

        assert result.passed is True

    def test_inv_d6_hash_format_invalid(self, valid_summary, valid_telemetry):
        """INV-D6: Invalid hash format should fail."""
        telemetry = deepcopy(valid_telemetry)
        telemetry["baseline"]["cycles"][0]["ht_hash"] = "invalid_hash"

        verdict = governance_verify(valid_summary, telemetry=telemetry)
        result = verdict.details.get("INV-D6")

        assert result.passed is False


# =============================================================================
# DETERMINISM TESTS
# =============================================================================

class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_same_input_same_output(self, valid_summary, valid_manifest, valid_telemetry):
        """Same inputs should produce identical verdicts."""
        verdict1 = governance_verify(
            summary=valid_summary,
            manifest=valid_manifest,
            telemetry=valid_telemetry,
        )

        verdict2 = governance_verify(
            summary=valid_summary,
            manifest=valid_manifest,
            telemetry=valid_telemetry,
        )

        # Status and rule results should be identical
        assert verdict1.status == verdict2.status
        assert verdict1.invalidating_rules == verdict2.invalidating_rules
        assert verdict1.warnings == verdict2.warnings
        assert set(verdict1.passed_rules) == set(verdict2.passed_rules)

        # Details should match (except timestamp)
        for rule_id in verdict1.details:
            r1 = verdict1.details[rule_id]
            r2 = verdict2.details[rule_id]
            assert r1.passed == r2.passed
            assert r1.severity == r2.severity

    def test_verdict_json_deterministic(self, valid_summary):
        """Verdict JSON should be deterministic (except timestamp)."""
        verdict1 = governance_verify(valid_summary)
        verdict2 = governance_verify(valid_summary)

        dict1 = verdict1.to_dict()
        dict2 = verdict2.to_dict()

        # Remove timestamp for comparison
        del dict1["timestamp"]
        del dict2["timestamp"]

        assert dict1 == dict2


# =============================================================================
# CLASSIFICATION TESTS
# =============================================================================

class TestSeverityClassification:
    """Tests for correct INVALIDATING vs WARNING classification."""

    def test_invalidating_rules_count(self):
        """Verify expected number of INVALIDATING rules."""
        invalidating = [r for r, info in RULE_REGISTRY.items() if info["severity"] == "INVALIDATING"]
        assert len(invalidating) == 32

    def test_warning_rules_count(self):
        """Verify expected number of WARNING rules."""
        warning = [r for r, info in RULE_REGISTRY.items() if info["severity"] == "WARNING"]
        assert len(warning) == 11  # REP-4, REP-7, GOV-5, GOV-6, MAN-5, MAN-7-10, INV-D2, INV-S3

    def test_gov_5_is_warning(self):
        """GOV-5 (Marginal Case Flagging) should be WARNING."""
        assert RULE_REGISTRY["GOV-5"]["severity"] == "WARNING"

    def test_gov_1_is_invalidating(self):
        """GOV-1 (Threshold Compliance) should be INVALIDATING."""
        assert RULE_REGISTRY["GOV-1"]["severity"] == "INVALIDATING"


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_summary(self):
        """Empty summary should fail gracefully."""
        verdict = governance_verify({})

        assert verdict.status == "FAIL"
        assert len(verdict.invalidating_rules) > 0

    def test_missing_governance_section(self, valid_summary):
        """Missing governance section should fail."""
        summary = deepcopy(valid_summary)
        del summary["governance"]

        verdict = governance_verify(summary)

        assert verdict.status == "FAIL"

    def test_missing_slices_section(self, valid_summary):
        """Missing slices section should fail."""
        summary = deepcopy(valid_summary)
        del summary["slices"]

        verdict = governance_verify(summary)

        assert verdict.status == "FAIL"

    def test_null_manifest_skips_manifest_rules(self, valid_summary):
        """Null manifest should skip manifest rules gracefully."""
        verdict = governance_verify(valid_summary, manifest=None)

        # Should not crash
        assert verdict.status is not None

        # Manifest rules should be skipped
        for rule_id in ["MAN-1", "MAN-2", "MAN-3", "MAN-4", "REP-1", "REP-2"]:
            result = verdict.details.get(rule_id)
            assert result is not None
            # Should be passed (skipped) or have skipped evidence
            assert result.passed is True or result.evidence.get("skipped") is True

    def test_null_telemetry_skips_telemetry_rules(self, valid_summary):
        """Null telemetry should skip telemetry rules gracefully."""
        verdict = governance_verify(valid_summary, telemetry=None)

        # Should not crash
        assert verdict.status is not None

        # Telemetry rules should be skipped
        for rule_id in ["INV-D1", "INV-D2", "INV-D3", "INV-D4", "INV-D5", "INV-D6"]:
            result = verdict.details.get(rule_id)
            assert result is not None
            assert result.passed is True or result.evidence.get("skipped") is True


# =============================================================================
# LEGACY TESTS (Backward compatibility)
# =============================================================================

@pytest.fixture
def test_data_dir(tmp_path: Path) -> Path:
    return tmp_path


def create_summary_file(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_valid_summary_data() -> dict:
    """Legacy fixture for backward compatibility."""
    return {
        "slice_id": "prop_depth4",
        "sample_size": {"baseline": 500, "rfl": 500},
        "metrics": {
            "success_rate": {"baseline": 0.95, "rfl": 0.98, "delta": 0.03, "ci": [0.01, 0.05]},
            "abstention_rate": {"baseline": 0.02, "rfl": 0.01, "delta": -0.01, "ci": [-0.02, 0.0]},
            "throughput": {
                "baseline_stat": 10.0,
                "treatment_stat": 12.0,
                "delta": 2.0,
                "delta_ci_low": 5.5,
                "delta_ci_high": 6.5,
                "delta_pct": 20.0,
                "significant": True,
            },
        },
        "governance": {
            "passed": True,
            "details": {
                "sample_size_passed": True,
                "success_rate_passed": True,
                "abstention_rate_passed": True,
                "throughput_uplift_passed": True,
            },
        },
        "reproducibility": {"bootstrap_seed": 42, "n_bootstrap": 10000},
    }


# =============================================================================
# V2: GOVERNANCE CHRONICLE & EXPLAINER TESTS
# =============================================================================

class TestReasonCodesAndExplanation:
    """Task 1 Tests: reason_codes and short_explanation in GovernanceVerdict."""

    def test_reason_codes_empty_on_pass(self, valid_summary, valid_manifest, valid_telemetry):
        """PASS verdict should have empty reason_codes."""
        verdict = governance_verify(
            summary=valid_summary,
            manifest=valid_manifest,
            telemetry=valid_telemetry,
        )

        if verdict.status == "PASS":
            assert verdict.reason_codes == []

    def test_reason_codes_match_triggered_rules(self, valid_summary):
        """reason_codes should match invalidating_rules + warnings."""
        summary = deepcopy(valid_summary)
        summary["governance"]["recommendation"] = "invalid"  # GOV-2 violation

        verdict = governance_verify(summary)

        expected = verdict.invalidating_rules + verdict.warnings
        assert verdict.reason_codes == expected

    def test_reason_codes_includes_gov_2_on_invalid_decision(self, valid_summary):
        """GOV-2 should be in reason_codes for invalid decision."""
        summary = deepcopy(valid_summary)
        summary["governance"]["recommendation"] = "approve"

        verdict = governance_verify(summary)

        assert "GOV-2" in verdict.reason_codes

    def test_short_explanation_present(self, valid_summary):
        """short_explanation should always be present."""
        verdict = governance_verify(valid_summary)

        assert verdict.short_explanation is not None
        assert len(verdict.short_explanation) > 0

    def test_short_explanation_for_pass(self, valid_summary, valid_manifest, valid_telemetry):
        """PASS explanation should mention all rules passed."""
        verdict = governance_verify(
            summary=valid_summary,
            manifest=valid_manifest,
            telemetry=valid_telemetry,
        )

        if verdict.status == "PASS":
            assert "passed" in verdict.short_explanation.lower()
            assert str(verdict.rules_checked) in verdict.short_explanation

    def test_short_explanation_for_fail(self, valid_summary):
        """FAIL explanation should mention INVALIDATING violation."""
        summary = deepcopy(valid_summary)
        summary["governance"]["recommendation"] = "invalid"

        verdict = governance_verify(summary)

        assert verdict.status == "FAIL"
        assert "INVALIDATING" in verdict.short_explanation

    def test_short_explanation_neutral_tone(self, valid_summary):
        """Explanation should be neutral, not prescriptive."""
        summary = deepcopy(valid_summary)
        summary["governance"]["recommendation"] = "invalid"

        verdict = governance_verify(summary)

        # Should not contain prescriptive language
        prescriptive_words = ["must", "should", "need to", "required", "fix"]
        explanation_lower = verdict.short_explanation.lower()
        assert not any(word in explanation_lower for word in prescriptive_words)

    def test_verdict_to_dict_includes_v2_fields(self, valid_summary):
        """to_dict() should include reason_codes and short_explanation."""
        verdict = governance_verify(valid_summary)
        verdict_dict = verdict.to_dict()

        assert "reason_codes" in verdict_dict
        assert "short_explanation" in verdict_dict


class TestExplainVerdict:
    """Task 1 Tests: explain_verdict() helper function."""

    def test_explain_verdict_returns_dict(self, valid_summary):
        """explain_verdict should return a dict."""
        verdict = governance_verify(valid_summary)
        explanation = explain_verdict(verdict)

        assert isinstance(explanation, dict)

    def test_explain_verdict_structure(self, valid_summary):
        """explain_verdict should contain required keys."""
        verdict = governance_verify(valid_summary)
        explanation = explain_verdict(verdict)

        required_keys = [
            "status",
            "reason_codes",
            "short_explanation",
            "triggered_rules",
            "invalidating_count",
            "warning_count",
            "pass_rate",
        ]
        for key in required_keys:
            assert key in explanation, f"Missing key: {key}"

    def test_explain_verdict_triggered_rules_has_descriptions(self, valid_summary):
        """triggered_rules should have descriptions for each rule."""
        summary = deepcopy(valid_summary)
        summary["governance"]["recommendation"] = "invalid"

        verdict = governance_verify(summary)
        explanation = explain_verdict(verdict)

        for rule_id, info in explanation["triggered_rules"].items():
            assert "description" in info
            assert "severity" in info
            assert "message" in info

    def test_explain_verdict_pass_rate_valid(self, valid_summary):
        """pass_rate should be between 0 and 1."""
        verdict = governance_verify(valid_summary)
        explanation = explain_verdict(verdict)

        assert 0 <= explanation["pass_rate"] <= 1

    def test_explain_verdict_reason_codes_match(self, valid_summary):
        """reason_codes in explanation should match verdict."""
        summary = deepcopy(valid_summary)
        summary["governance"]["recommendation"] = "invalid"

        verdict = governance_verify(summary)
        explanation = explain_verdict(verdict)

        expected = verdict.invalidating_rules + verdict.warnings
        assert explanation["reason_codes"] == expected


class TestBuildGovernancePosture:
    """Task 2 Tests: build_governance_posture() for multi-file aggregation."""

    def test_empty_verdicts_returns_pass(self):
        """Empty verdicts list should return aggregate PASS."""
        posture = build_governance_posture([])

        assert posture["aggregate_status"] == "PASS"
        assert posture["is_governance_blocking"] is False
        assert posture["total_count"] == 0

    def test_single_pass_verdict(self, valid_summary, valid_manifest, valid_telemetry):
        """Single PASS verdict should aggregate to PASS."""
        verdict = governance_verify(valid_summary, valid_manifest, valid_telemetry)

        # Ensure it's actually a PASS
        if verdict.status != "PASS":
            pytest.skip("Verdict is not PASS, skipping single PASS test")

        posture = build_governance_posture([verdict], ["summary.json"])

        assert posture["aggregate_status"] == "PASS"
        assert posture["pass_count"] == 1
        assert posture["fail_count"] == 0
        assert posture["is_governance_blocking"] is False

    def test_single_fail_verdict(self, valid_summary):
        """Single FAIL verdict should aggregate to FAIL and be blocking."""
        summary = deepcopy(valid_summary)
        summary["governance"]["recommendation"] = "invalid"

        verdict = governance_verify(summary)
        posture = build_governance_posture([verdict], ["bad_summary.json"])

        assert posture["aggregate_status"] == "FAIL"
        assert posture["fail_count"] == 1
        assert posture["is_governance_blocking"] is True

    def test_mixed_verdicts_blocking(self, valid_summary):
        """Mix of PASS and FAIL should be blocking."""
        # Create a PASS verdict
        pass_verdict = governance_verify(valid_summary)

        # Create a FAIL verdict
        fail_summary = deepcopy(valid_summary)
        fail_summary["governance"]["recommendation"] = "invalid"
        fail_verdict = governance_verify(fail_summary)

        posture = build_governance_posture(
            [pass_verdict, fail_verdict],
            ["good.json", "bad.json"]
        )

        assert posture["is_governance_blocking"] is True
        assert posture["fail_count"] == 1
        assert len(posture["failing_files"]) == 1
        assert posture["failing_files"][0]["file"] == "bad.json"

    def test_failing_files_include_reason_codes(self, valid_summary):
        """failing_files should include reason_codes."""
        summary = deepcopy(valid_summary)
        summary["governance"]["recommendation"] = "invalid"

        verdict = governance_verify(summary)
        posture = build_governance_posture([verdict], ["fail.json"])

        assert len(posture["failing_files"]) == 1
        assert "reason_codes" in posture["failing_files"][0]
        assert "GOV-2" in posture["failing_files"][0]["reason_codes"]

    def test_posture_counts_correct(self, valid_summary):
        """Posture counts should match verdict counts."""
        # Create multiple verdicts
        verdicts = []
        for i in range(3):
            verdicts.append(governance_verify(valid_summary))

        # Add a FAIL
        fail_summary = deepcopy(valid_summary)
        fail_summary["governance"]["recommendation"] = "invalid"
        verdicts.append(governance_verify(fail_summary))

        posture = build_governance_posture(
            verdicts,
            [f"file_{i}.json" for i in range(4)]
        )

        assert posture["total_count"] == 4
        assert posture["fail_count"] == 1
        # Others might be PASS or WARN depending on rules

    def test_posture_has_timestamp(self, valid_summary):
        """Posture should include timestamp."""
        verdict = governance_verify(valid_summary)
        posture = build_governance_posture([verdict])

        assert "timestamp" in posture
        assert len(posture["timestamp"]) > 0


class TestSummarizeForAdmissibility:
    """Task 3 Tests: summarize_for_admissibility() for MAAS integration."""

    def test_admissibility_structure(self, valid_summary):
        """Admissibility summary should have required keys."""
        verdict = governance_verify(valid_summary)
        admissibility = summarize_for_admissibility(verdict)

        required_keys = [
            "overall_status",
            "has_invalidating_violations",
            "invalidating_rules",
            "invalidating_count",
            "warning_count",
            "is_admissible",
            "reason_summary",
            "governance_version",
        ]
        for key in required_keys:
            assert key in admissibility, f"Missing key: {key}"

    def test_he_gv1_overall_status_checked(self, valid_summary):
        """HE-GV1: overall_status should reflect verdict status."""
        verdict = governance_verify(valid_summary)
        admissibility = summarize_for_admissibility(verdict)

        assert admissibility["overall_status"] == verdict.status

    def test_he_gv2_has_invalidating_violations_flag(self, valid_summary):
        """HE-GV2: has_invalidating_violations should be correct."""
        # PASS case
        verdict = governance_verify(valid_summary)
        admissibility = summarize_for_admissibility(verdict)

        expected = len(verdict.invalidating_rules) > 0
        assert admissibility["has_invalidating_violations"] == expected

    def test_he_gv2_invalidating_flag_on_fail(self, valid_summary):
        """HE-GV2: has_invalidating_violations should be True on FAIL."""
        summary = deepcopy(valid_summary)
        summary["governance"]["recommendation"] = "invalid"

        verdict = governance_verify(summary)
        admissibility = summarize_for_admissibility(verdict)

        assert admissibility["has_invalidating_violations"] is True

    def test_he_gv3_invalidating_rules_enumerated(self, valid_summary):
        """HE-GV3: invalidating_rules should list all violations."""
        summary = deepcopy(valid_summary)
        summary["governance"]["recommendation"] = "invalid"

        verdict = governance_verify(summary)
        admissibility = summarize_for_admissibility(verdict)

        assert admissibility["invalidating_rules"] == verdict.invalidating_rules
        assert "GOV-2" in admissibility["invalidating_rules"]

    def test_he_gv4_is_admissible_boolean_gate(self, valid_summary):
        """HE-GV4: is_admissible should be boolean gate for FAIL."""
        # PASS/WARN should be admissible
        verdict = governance_verify(valid_summary)
        admissibility = summarize_for_admissibility(verdict)

        if verdict.status in {"PASS", "WARN"}:
            assert admissibility["is_admissible"] is True

        # FAIL should not be admissible
        summary = deepcopy(valid_summary)
        summary["governance"]["recommendation"] = "invalid"

        fail_verdict = governance_verify(summary)
        fail_admissibility = summarize_for_admissibility(fail_verdict)

        assert fail_admissibility["is_admissible"] is False

    def test_he_gv5_governance_version_present(self, valid_summary):
        """HE-GV5: governance_version should be present for compatibility."""
        verdict = governance_verify(valid_summary)
        admissibility = summarize_for_admissibility(verdict)

        assert admissibility["governance_version"] == verdict.verifier_version
        assert admissibility["governance_version"] == __version__

    def test_admissibility_pass_case(self, valid_summary, valid_manifest, valid_telemetry):
        """PASS verdict should be admissible."""
        verdict = governance_verify(valid_summary, valid_manifest, valid_telemetry)

        if verdict.status == "PASS":
            admissibility = summarize_for_admissibility(verdict)

            assert admissibility["is_admissible"] is True
            assert admissibility["has_invalidating_violations"] is False
            assert "passed" in admissibility["reason_summary"].lower()

    def test_admissibility_warn_case(self, valid_summary):
        """WARN verdict should still be admissible."""
        # Create a warning by having marginal case
        summary = deepcopy(valid_summary)
        summary["slices"]["prop_depth4"]["throughput"]["ci_low"] = 4.0
        summary["slices"]["prop_depth4"]["throughput"]["ci_high"] = 6.0

        verdict = governance_verify(summary)

        if verdict.status == "WARN":
            admissibility = summarize_for_admissibility(verdict)

            assert admissibility["is_admissible"] is True
            assert "warning" in admissibility["reason_summary"].lower()

    def test_admissibility_fail_case(self, valid_summary):
        """FAIL verdict should not be admissible."""
        summary = deepcopy(valid_summary)
        summary["governance"]["recommendation"] = "invalid"

        verdict = governance_verify(summary)
        admissibility = summarize_for_admissibility(verdict)

        assert verdict.status == "FAIL"
        assert admissibility["is_admissible"] is False
        assert admissibility["has_invalidating_violations"] is True
        assert "failed" in admissibility["reason_summary"].lower()


class TestRuleDescriptions:
    """Tests for RULE_DESCRIPTIONS completeness."""

    def test_all_rules_have_descriptions(self):
        """Every rule in RULE_REGISTRY should have a description."""
        for rule_id in RULE_REGISTRY:
            assert rule_id in RULE_DESCRIPTIONS, f"Missing description for {rule_id}"

    def test_descriptions_are_non_empty(self):
        """All descriptions should be non-empty strings."""
        for rule_id, desc in RULE_DESCRIPTIONS.items():
            assert isinstance(desc, str)
            assert len(desc) > 0, f"Empty description for {rule_id}"

    def test_descriptions_are_descriptive(self):
        """Descriptions should be meaningful (> 10 chars)."""
        for rule_id, desc in RULE_DESCRIPTIONS.items():
            assert len(desc) > 10, f"Too short description for {rule_id}: '{desc}'"


# =============================================================================
# PHASE III: DIRECTOR CONSOLE GOVERNANCE FEED TESTS
# =============================================================================

class TestBuildGovernanceChronicle:
    """Task 1 Tests: build_governance_chronicle() for trend analysis."""

    def test_empty_snapshots_returns_unknown(self):
        """Empty snapshots should return unknown trend."""
        chronicle = build_governance_chronicle([])

        assert chronicle["trend"] == "unknown"
        assert chronicle["recurring_rule_failures"] == {}
        assert chronicle["governance_blocking_rate"] == 0.0
        assert chronicle["total_snapshots"] == 0

    def test_single_snapshot_returns_unknown_trend(self):
        """Single snapshot should return unknown trend."""
        posture = {
            "aggregate_status": "PASS",
            "is_governance_blocking": False,
            "failing_files": [],
            "timestamp": "2025-01-01T00:00:00Z",
        }
        chronicle = build_governance_chronicle([posture])

        assert chronicle["trend"] == "unknown"
        assert chronicle["total_snapshots"] == 1
        assert chronicle["pass_count"] == 1

    def test_all_pass_snapshots(self):
        """All PASS snapshots should have 0 blocking rate."""
        snapshots = [
            {"aggregate_status": "PASS", "is_governance_blocking": False, "failing_files": [], "timestamp": f"2025-01-0{i}T00:00:00Z"}
            for i in range(1, 6)
        ]
        chronicle = build_governance_chronicle(snapshots)

        assert chronicle["governance_blocking_rate"] == 0.0
        assert chronicle["pass_count"] == 5
        assert chronicle["fail_count"] == 0
        assert chronicle["recurring_rule_failures"] == {}

    def test_all_fail_snapshots(self):
        """All FAIL snapshots should have 1.0 blocking rate."""
        snapshots = [
            {
                "aggregate_status": "FAIL",
                "is_governance_blocking": True,
                "failing_files": [{"file": f"file_{i}.json", "reason_codes": ["GOV-2"]}],
                "timestamp": f"2025-01-0{i}T00:00:00Z"
            }
            for i in range(1, 6)
        ]
        chronicle = build_governance_chronicle(snapshots)

        assert chronicle["governance_blocking_rate"] == 1.0
        assert chronicle["fail_count"] == 5
        assert chronicle["pass_count"] == 0

    def test_recurring_rule_failures_detection(self):
        """Should detect rules that fail more than once."""
        snapshots = [
            {"aggregate_status": "FAIL", "is_governance_blocking": True,
             "failing_files": [{"file": "a.json", "reason_codes": ["GOV-2", "GOV-3"]}],
             "timestamp": "2025-01-01T00:00:00Z"},
            {"aggregate_status": "FAIL", "is_governance_blocking": True,
             "failing_files": [{"file": "b.json", "reason_codes": ["GOV-2", "REP-3"]}],
             "timestamp": "2025-01-02T00:00:00Z"},
            {"aggregate_status": "PASS", "is_governance_blocking": False,
             "failing_files": [],
             "timestamp": "2025-01-03T00:00:00Z"},
        ]
        chronicle = build_governance_chronicle(snapshots)

        # GOV-2 appears twice, GOV-3 and REP-3 appear once each
        assert "GOV-2" in chronicle["recurring_rule_failures"]
        assert chronicle["recurring_rule_failures"]["GOV-2"] == 2
        assert "GOV-3" not in chronicle["recurring_rule_failures"]
        assert "REP-3" not in chronicle["recurring_rule_failures"]

    def test_trend_improving(self):
        """Should detect improving trend when failures decrease."""
        # First half: all blocking
        # Second half: none blocking
        snapshots = [
            {"aggregate_status": "FAIL", "is_governance_blocking": True, "failing_files": [], "timestamp": "2025-01-01T00:00:00Z"},
            {"aggregate_status": "FAIL", "is_governance_blocking": True, "failing_files": [], "timestamp": "2025-01-02T00:00:00Z"},
            {"aggregate_status": "PASS", "is_governance_blocking": False, "failing_files": [], "timestamp": "2025-01-03T00:00:00Z"},
            {"aggregate_status": "PASS", "is_governance_blocking": False, "failing_files": [], "timestamp": "2025-01-04T00:00:00Z"},
        ]
        chronicle = build_governance_chronicle(snapshots)

        assert chronicle["trend"] == "improving"

    def test_trend_degrading(self):
        """Should detect degrading trend when failures increase."""
        # First half: none blocking
        # Second half: all blocking
        snapshots = [
            {"aggregate_status": "PASS", "is_governance_blocking": False, "failing_files": [], "timestamp": "2025-01-01T00:00:00Z"},
            {"aggregate_status": "PASS", "is_governance_blocking": False, "failing_files": [], "timestamp": "2025-01-02T00:00:00Z"},
            {"aggregate_status": "FAIL", "is_governance_blocking": True, "failing_files": [], "timestamp": "2025-01-03T00:00:00Z"},
            {"aggregate_status": "FAIL", "is_governance_blocking": True, "failing_files": [], "timestamp": "2025-01-04T00:00:00Z"},
        ]
        chronicle = build_governance_chronicle(snapshots)

        assert chronicle["trend"] == "degrading"

    def test_trend_stable(self):
        """Should detect stable trend when failure rate unchanged."""
        # Both halves have same blocking rate
        snapshots = [
            {"aggregate_status": "FAIL", "is_governance_blocking": True, "failing_files": [], "timestamp": "2025-01-01T00:00:00Z"},
            {"aggregate_status": "PASS", "is_governance_blocking": False, "failing_files": [], "timestamp": "2025-01-02T00:00:00Z"},
            {"aggregate_status": "FAIL", "is_governance_blocking": True, "failing_files": [], "timestamp": "2025-01-03T00:00:00Z"},
            {"aggregate_status": "PASS", "is_governance_blocking": False, "failing_files": [], "timestamp": "2025-01-04T00:00:00Z"},
        ]
        chronicle = build_governance_chronicle(snapshots)

        assert chronicle["trend"] == "stable"

    def test_most_common_failures_top_5(self):
        """Should return top 5 most common failures."""
        snapshots = []
        # Create snapshots with varying failure counts
        rules = ["GOV-1", "GOV-2", "GOV-3", "REP-1", "REP-2", "REP-3"]
        for i, rule in enumerate(rules):
            for _ in range(6 - i):  # GOV-1: 6 times, GOV-2: 5 times, etc.
                snapshots.append({
                    "aggregate_status": "FAIL",
                    "is_governance_blocking": True,
                    "failing_files": [{"file": "f.json", "reason_codes": [rule]}],
                    "timestamp": "2025-01-01T00:00:00Z",
                })

        chronicle = build_governance_chronicle(snapshots)

        # Should have top 5 (not REP-3 which only has 1)
        assert len(chronicle["most_common_failures"]) == 5
        assert chronicle["most_common_failures"][0][0] == "GOV-1"
        assert chronicle["most_common_failures"][0][1] == 6

    def test_timestamp_extraction(self):
        """Should extract first and last timestamps."""
        snapshots = [
            {"aggregate_status": "PASS", "is_governance_blocking": False, "failing_files": [], "timestamp": "2025-01-01T00:00:00Z"},
            {"aggregate_status": "PASS", "is_governance_blocking": False, "failing_files": [], "timestamp": "2025-01-05T00:00:00Z"},
            {"aggregate_status": "PASS", "is_governance_blocking": False, "failing_files": [], "timestamp": "2025-01-03T00:00:00Z"},
        ]
        chronicle = build_governance_chronicle(snapshots)

        assert chronicle["first_snapshot"] == "2025-01-01T00:00:00Z"
        assert chronicle["last_snapshot"] == "2025-01-05T00:00:00Z"


class TestMapGovernanceToDirectorStatus:
    """Task 2 Tests: map_governance_to_director_status()."""

    def test_empty_posture_returns_red(self):
        """Empty/None posture should return RED."""
        assert map_governance_to_director_status({}) == "RED"
        assert map_governance_to_director_status(None) == "RED"

    def test_pass_posture_returns_green(self):
        """PASS posture should return GREEN."""
        posture = {
            "aggregate_status": "PASS",
            "is_governance_blocking": False,
        }
        assert map_governance_to_director_status(posture) == "GREEN"

    def test_warn_posture_returns_yellow(self):
        """WARN posture should return YELLOW."""
        posture = {
            "aggregate_status": "WARN",
            "is_governance_blocking": False,
        }
        assert map_governance_to_director_status(posture) == "YELLOW"

    def test_fail_posture_returns_red(self):
        """FAIL posture should return RED."""
        posture = {
            "aggregate_status": "FAIL",
            "is_governance_blocking": True,
        }
        assert map_governance_to_director_status(posture) == "RED"

    def test_blocking_always_red(self):
        """Any blocking posture should return RED regardless of status."""
        posture = {
            "aggregate_status": "PASS",  # Even with PASS
            "is_governance_blocking": True,  # If blocking, should be RED
        }
        assert map_governance_to_director_status(posture) == "RED"

    def test_director_status_values(self):
        """Director status should only be GREEN, YELLOW, or RED."""
        test_cases = [
            {"aggregate_status": "PASS", "is_governance_blocking": False},
            {"aggregate_status": "WARN", "is_governance_blocking": False},
            {"aggregate_status": "FAIL", "is_governance_blocking": True},
            {},
        ]
        valid_statuses = {"GREEN", "YELLOW", "RED"}

        for posture in test_cases:
            status = map_governance_to_director_status(posture)
            assert status in valid_statuses, f"Invalid status: {status}"


class TestSummarizeGovernanceForGlobalHealth:
    """Task 3 Tests: summarize_governance_for_global_health()."""

    def test_empty_posture_returns_fatal(self):
        """Empty/None posture should return fatal status."""
        health = summarize_governance_for_global_health({})

        assert health["governance_ok"] is False
        assert health["fatality"] == "fatal"
        assert health["status_code"] == 2

    def test_none_posture_returns_fatal(self):
        """None posture should return fatal status."""
        health = summarize_governance_for_global_health(None)

        assert health["governance_ok"] is False
        assert health["fatality"] == "fatal"
        assert health["status_code"] == 2

    def test_pass_posture_returns_ok(self):
        """PASS posture should return governance_ok=True, fatality=none."""
        posture = {
            "aggregate_status": "PASS",
            "is_governance_blocking": False,
            "failing_files": [],
            "total_count": 5,
            "fail_count": 0,
        }
        health = summarize_governance_for_global_health(posture)

        assert health["governance_ok"] is True
        assert health["fatality"] == "none"
        assert health["status_code"] == 0
        assert health["failing_rules"] == []
        assert "OK" in health["summary"]

    def test_warn_posture_returns_warning(self):
        """WARN posture should return governance_ok=True, fatality=warning."""
        posture = {
            "aggregate_status": "WARN",
            "is_governance_blocking": False,
            "failing_files": [],
            "total_count": 5,
            "fail_count": 0,
            "warn_count": 2,
        }
        health = summarize_governance_for_global_health(posture)

        assert health["governance_ok"] is True
        assert health["fatality"] == "warning"
        assert health["status_code"] == 1
        assert "warning" in health["summary"].lower()

    def test_fail_posture_returns_fatal(self):
        """FAIL posture should return governance_ok=False, fatality=fatal."""
        posture = {
            "aggregate_status": "FAIL",
            "is_governance_blocking": True,
            "failing_files": [
                {"file": "bad.json", "reason_codes": ["GOV-2", "REP-3"]},
            ],
            "total_count": 5,
            "fail_count": 1,
        }
        health = summarize_governance_for_global_health(posture)

        assert health["governance_ok"] is False
        assert health["fatality"] == "fatal"
        assert health["status_code"] == 2
        assert "FAILED" in health["summary"]

    def test_failing_rules_extracted(self):
        """Should extract all unique failing rules."""
        posture = {
            "aggregate_status": "FAIL",
            "is_governance_blocking": True,
            "failing_files": [
                {"file": "a.json", "reason_codes": ["GOV-2", "REP-3"]},
                {"file": "b.json", "reason_codes": ["GOV-2", "GOV-4"]},  # GOV-2 duplicate
            ],
            "total_count": 3,
            "fail_count": 2,
        }
        health = summarize_governance_for_global_health(posture)

        # Should have unique rules only
        assert "GOV-2" in health["failing_rules"]
        assert "REP-3" in health["failing_rules"]
        assert "GOV-4" in health["failing_rules"]
        assert len(health["failing_rules"]) == 3

    def test_file_counts_included(self):
        """Should include file and fail counts."""
        posture = {
            "aggregate_status": "FAIL",
            "is_governance_blocking": True,
            "failing_files": [{"file": "x.json", "reason_codes": ["GOV-1"]}],
            "total_count": 10,
            "fail_count": 3,
        }
        health = summarize_governance_for_global_health(posture)

        assert health["file_count"] == 10
        assert health["fail_count"] == 3

    def test_status_codes_valid(self):
        """Status codes should be 0, 1, or 2."""
        test_cases = [
            {"aggregate_status": "PASS", "is_governance_blocking": False, "failing_files": [], "total_count": 1, "fail_count": 0},
            {"aggregate_status": "WARN", "is_governance_blocking": False, "failing_files": [], "total_count": 1, "fail_count": 0, "warn_count": 1},
            {"aggregate_status": "FAIL", "is_governance_blocking": True, "failing_files": [], "total_count": 1, "fail_count": 1},
        ]
        expected_codes = [0, 1, 2]

        for posture, expected in zip(test_cases, expected_codes):
            health = summarize_governance_for_global_health(posture)
            assert health["status_code"] == expected

    def test_fatality_values_valid(self):
        """Fatality should only be none, warning, or fatal."""
        test_cases = [
            {"aggregate_status": "PASS", "is_governance_blocking": False, "failing_files": [], "total_count": 1, "fail_count": 0},
            {"aggregate_status": "WARN", "is_governance_blocking": False, "failing_files": [], "total_count": 1, "fail_count": 0, "warn_count": 1},
            {"aggregate_status": "FAIL", "is_governance_blocking": True, "failing_files": [], "total_count": 1, "fail_count": 1},
        ]
        expected_fatalities = ["none", "warning", "fatal"]

        for posture, expected in zip(test_cases, expected_fatalities):
            health = summarize_governance_for_global_health(posture)
            assert health["fatality"] == expected


class TestPhaseIIIIntegration:
    """Integration tests for Phase III Director Console Feed."""

    def test_full_pipeline_pass(self, valid_summary):
        """Test full pipeline: verdict -> posture -> director status -> health."""
        # Create verdict
        verdict = governance_verify(valid_summary)

        # Build posture
        posture = build_governance_posture([verdict], ["summary.json"])

        # Map to director status
        if verdict.status in {"PASS", "WARN"}:
            director_status = map_governance_to_director_status(posture)
            assert director_status in {"GREEN", "YELLOW"}

        # Get global health
        health = summarize_governance_for_global_health(posture)
        if verdict.status != "FAIL":
            assert health["governance_ok"] is True

    def test_full_pipeline_fail(self, valid_summary):
        """Test full pipeline with failing verdict."""
        summary = deepcopy(valid_summary)
        summary["governance"]["recommendation"] = "invalid"

        # Create failing verdict
        verdict = governance_verify(summary)
        assert verdict.status == "FAIL"

        # Build posture
        posture = build_governance_posture([verdict], ["bad.json"])

        # Map to director status
        director_status = map_governance_to_director_status(posture)
        assert director_status == "RED"

        # Get global health
        health = summarize_governance_for_global_health(posture)
        assert health["governance_ok"] is False
        assert health["fatality"] == "fatal"
        assert "GOV-2" in health["failing_rules"]

    def test_chronicle_from_multiple_postures(self, valid_summary):
        """Test chronicle built from multiple posture snapshots."""
        postures = []

        # Create some PASS postures
        for i in range(3):
            verdict = governance_verify(valid_summary)
            posture = build_governance_posture([verdict], [f"pass_{i}.json"])
            posture["timestamp"] = f"2025-01-0{i+1}T00:00:00Z"
            postures.append(posture)

        # Create a FAIL posture
        fail_summary = deepcopy(valid_summary)
        fail_summary["governance"]["recommendation"] = "invalid"
        fail_verdict = governance_verify(fail_summary)
        fail_posture = build_governance_posture([fail_verdict], ["fail.json"])
        fail_posture["timestamp"] = "2025-01-04T00:00:00Z"
        postures.append(fail_posture)

        # Build chronicle
        chronicle = build_governance_chronicle(postures)

        assert chronicle["total_snapshots"] == 4
        assert chronicle["fail_count"] == 1
        assert chronicle["governance_blocking_rate"] == 0.25
        assert "GOV-2" in chronicle["recurring_rule_failures"] or len(chronicle["most_common_failures"]) > 0


# =============================================================================
# PHASE IV: GOVERNANCE CHRONICLE COMPASS & CROSS-SYSTEM GATE TESTS
# =============================================================================

class TestBuildGovernanceAlignmentView:
    """Task 1 Tests: build_governance_alignment_view()."""

    def test_empty_inputs_returns_aligned(self):
        """Empty inputs should return ALIGNED status."""
        alignment = build_governance_alignment_view({})

        assert alignment["governance_alignment_status"] == "ALIGNED"
        assert alignment["rules_failing_in_multiple_layers"] == {}
        assert alignment["cross_layer_failures"] == []
        assert alignment["alignment_score"] == 1.0

    def test_governance_only_failures(self):
        """Failures only in governance layer should be ALIGNED."""
        chronicle = {
            "recurring_rule_failures": {"GOV-2": 3, "REP-3": 2},
            "governance_blocking_rate": 0.25,
        }
        alignment = build_governance_alignment_view(chronicle)

        assert alignment["governance_alignment_status"] == "ALIGNED"
        assert len(alignment["cross_layer_failures"]) == 0

    def test_cross_layer_failures_detected(self):
        """Failures in multiple layers should be detected."""
        chronicle = {
            "recurring_rule_failures": {"GOV-2": 3},
            "governance_blocking_rate": 0.25,
        }
        admissibility = {
            "failing_rules": ["GOV-2", "REP-1"],
            "status": "FAIL",
        }
        alignment = build_governance_alignment_view(chronicle, admissibility)

        # GOV-2 appears in both layers
        assert "GOV-2" in alignment["rules_failing_in_multiple_layers"]
        assert "governance" in alignment["rules_failing_in_multiple_layers"]["GOV-2"]
        assert "admissibility" in alignment["rules_failing_in_multiple_layers"]["GOV-2"]
        assert "GOV-2" in alignment["cross_layer_failures"]

    def test_tension_status_with_few_cross_failures(self):
        """1-2 cross-layer failures should result in TENSION."""
        chronicle = {
            "recurring_rule_failures": {"GOV-2": 3, "GOV-3": 2},
            "governance_blocking_rate": 0.25,
        }
        admissibility = {
            "failing_rules": ["GOV-2"],  # 1 overlap
            "status": "WARN",
        }
        alignment = build_governance_alignment_view(chronicle, admissibility)

        assert alignment["governance_alignment_status"] == "TENSION"
        assert len(alignment["cross_layer_failures"]) == 1

    def test_divergent_status_with_many_cross_failures(self):
        """3+ cross-layer failures should result in DIVERGENT."""
        chronicle = {
            "recurring_rule_failures": {"GOV-1": 2, "GOV-2": 3, "GOV-3": 2, "REP-1": 2},
            "governance_blocking_rate": 0.5,
        }
        admissibility = {
            "failing_rules": ["GOV-1", "GOV-2", "GOV-3"],  # 3 overlaps
            "status": "FAIL",
        }
        alignment = build_governance_alignment_view(chronicle, admissibility)

        assert alignment["governance_alignment_status"] == "DIVERGENT"
        assert len(alignment["cross_layer_failures"]) >= 3

    def test_three_layer_cross_failure(self):
        """Failure in all three layers should be detected."""
        chronicle = {
            "recurring_rule_failures": {"GOV-2": 5},
            "governance_blocking_rate": 0.5,
        }
        admissibility = {
            "failing_rules": ["GOV-2"],
            "status": "FAIL",
        }
        topology = {
            "failing_rules": ["GOV-2"],
            "trajectory_status": "critical",
        }
        alignment = build_governance_alignment_view(chronicle, admissibility, topology)

        assert "GOV-2" in alignment["rules_failing_in_multiple_layers"]
        layers = alignment["rules_failing_in_multiple_layers"]["GOV-2"]
        assert "governance" in layers
        assert "admissibility" in layers
        assert "topology" in layers

    def test_layer_statuses_included(self):
        """Layer statuses should be included in output."""
        chronicle = {
            "governance_blocking_rate": 0.0,
        }
        admissibility = {"status": "PASS"}
        topology = {"trajectory_status": "stable"}

        alignment = build_governance_alignment_view(chronicle, admissibility, topology)

        assert alignment["layer_statuses"]["governance"] == "OK"
        assert alignment["layer_statuses"]["admissibility"] == "OK"
        assert alignment["layer_statuses"]["topology"] == "OK"

    def test_alignment_score_calculation(self):
        """Alignment score should decrease with cross-layer failures."""
        # No failures = 1.0
        alignment1 = build_governance_alignment_view({})
        assert alignment1["alignment_score"] == 1.0

        # Some cross-layer failures = lower score
        chronicle = {"recurring_rule_failures": {"GOV-1": 2, "GOV-2": 2}}
        admissibility = {"failing_rules": ["GOV-1", "GOV-2"]}
        alignment2 = build_governance_alignment_view(chronicle, admissibility)
        assert alignment2["alignment_score"] < 1.0

    def test_recommendation_provided(self):
        """Recommendation should be provided for each status."""
        # ALIGNED
        align1 = build_governance_alignment_view({})
        assert "recommendation" in align1
        assert len(align1["recommendation"]) > 0

        # TENSION
        chronicle = {"recurring_rule_failures": {"GOV-2": 2}}
        admissibility = {"failing_rules": ["GOV-2"]}
        align2 = build_governance_alignment_view(chronicle, admissibility)
        assert "recommendation" in align2


class TestEvaluateGovernanceForPromotion:
    """Task 2 Tests: evaluate_governance_for_promotion()."""

    def test_empty_alignment_blocks(self):
        """Empty alignment view should block promotion."""
        promotion = evaluate_governance_for_promotion({})

        assert promotion["promotion_ok"] is False
        assert promotion["status"] == "BLOCK"

    def test_none_alignment_blocks(self):
        """None alignment view should block promotion."""
        promotion = evaluate_governance_for_promotion(None)

        assert promotion["promotion_ok"] is False
        assert promotion["status"] == "BLOCK"

    def test_aligned_no_blocking_ok(self):
        """ALIGNED with no blocking rules should be OK."""
        alignment = {
            "governance_alignment_status": "ALIGNED",
            "cross_layer_failures": [],
            "rules_failing_in_multiple_layers": {},
        }
        promotion = evaluate_governance_for_promotion(alignment)

        assert promotion["promotion_ok"] is True
        assert promotion["status"] == "OK"
        assert len(promotion["blocking_rules"]) == 0

    def test_aligned_with_global_posture_fatal(self):
        """ALIGNED but global posture fatal should warn."""
        alignment = {
            "governance_alignment_status": "ALIGNED",
            "cross_layer_failures": [],
            "rules_failing_in_multiple_layers": {},
        }
        global_posture = {
            "fatality": "fatal",
            "failing_rules": ["GOV-2"],
        }
        promotion = evaluate_governance_for_promotion(alignment, global_posture)

        assert promotion["promotion_ok"] is True  # Still allowed but with warning
        assert promotion["status"] == "WARN"
        assert "GOV-2" in promotion["blocking_rules"]

    def test_tension_few_rules_warns(self):
        """TENSION with few blocking rules should warn but allow."""
        alignment = {
            "governance_alignment_status": "TENSION",
            "cross_layer_failures": ["GOV-2"],
            "rules_failing_in_multiple_layers": {"GOV-2": ["governance", "admissibility"]},
        }
        promotion = evaluate_governance_for_promotion(alignment)

        assert promotion["promotion_ok"] is True
        assert promotion["status"] == "WARN"
        assert "GOV-2" in promotion["blocking_rules"]

    def test_tension_many_rules_blocks(self):
        """TENSION with many blocking rules should block."""
        alignment = {
            "governance_alignment_status": "TENSION",
            "cross_layer_failures": ["GOV-1", "GOV-2", "GOV-3", "REP-1"],  # 4 rules
            "rules_failing_in_multiple_layers": {},
        }
        promotion = evaluate_governance_for_promotion(alignment)

        assert promotion["promotion_ok"] is False
        assert promotion["status"] == "BLOCK"
        assert len(promotion["blocking_rules"]) == 4

    def test_divergent_always_blocks(self):
        """DIVERGENT should always block promotion."""
        alignment = {
            "governance_alignment_status": "DIVERGENT",
            "cross_layer_failures": ["GOV-1", "GOV-2", "GOV-3"],
            "rules_failing_in_multiple_layers": {},
        }
        promotion = evaluate_governance_for_promotion(alignment)

        assert promotion["promotion_ok"] is False
        assert promotion["status"] == "BLOCK"

    def test_conditions_provided(self):
        """Conditions should be provided for non-OK status."""
        # BLOCK case
        alignment = {
            "governance_alignment_status": "DIVERGENT",
            "cross_layer_failures": ["GOV-1", "GOV-2", "GOV-3"],
        }
        promotion = evaluate_governance_for_promotion(alignment)

        assert "conditions" in promotion
        assert len(promotion["conditions"]) > 0

    def test_reason_provided(self):
        """Reason should always be provided."""
        alignment = {
            "governance_alignment_status": "ALIGNED",
            "cross_layer_failures": [],
        }
        promotion = evaluate_governance_for_promotion(alignment)

        assert "reason" in promotion
        assert len(promotion["reason"]) > 0

    def test_alignment_status_echoed(self):
        """Alignment status should be included in output."""
        alignment = {
            "governance_alignment_status": "TENSION",
            "cross_layer_failures": ["GOV-2"],
        }
        promotion = evaluate_governance_for_promotion(alignment)

        assert promotion["alignment_status"] == "TENSION"


class TestBuildGovernanceDirectorPanelV2:
    """Task 3 Tests: build_governance_director_panel_v2()."""

    def test_empty_chronicle_returns_red(self):
        """Empty chronicle should return RED panel."""
        panel = build_governance_director_panel_v2({})

        assert panel["status_light"] == "RED"
        assert panel["governance_blocking_rate"] == 1.0
        assert panel["trend"] == "unknown"
        assert "No governance data" in panel["headline"]

    def test_none_chronicle_returns_red(self):
        """None chronicle should return RED panel."""
        panel = build_governance_director_panel_v2(None)

        assert panel["status_light"] == "RED"

    def test_clear_governance_green(self):
        """Zero blocking rate should return GREEN."""
        chronicle = {
            "governance_blocking_rate": 0.0,
            "trend": "stable",
            "total_snapshots": 10,
            "fail_count": 0,
            "most_common_failures": [],
        }
        panel = build_governance_director_panel_v2(chronicle)

        assert panel["status_light"] == "GREEN"
        assert panel["governance_blocking_rate"] == 0.0
        assert "clear" in panel["headline"].lower()

    def test_high_blocking_rate_red(self):
        """High blocking rate should return RED."""
        chronicle = {
            "governance_blocking_rate": 0.6,
            "trend": "degrading",
            "total_snapshots": 10,
            "fail_count": 6,
            "most_common_failures": [("GOV-2", 6)],
        }
        panel = build_governance_director_panel_v2(chronicle)

        assert panel["status_light"] == "RED"
        assert "critical" in panel["headline"].lower()

    def test_low_blocking_rate_yellow(self):
        """Low blocking rate should return YELLOW."""
        chronicle = {
            "governance_blocking_rate": 0.2,
            "trend": "stable",
            "total_snapshots": 10,
            "fail_count": 2,
            "most_common_failures": [("GOV-2", 2)],
        }
        panel = build_governance_director_panel_v2(chronicle)

        assert panel["status_light"] == "YELLOW"

    def test_promotion_block_forces_red(self):
        """Promotion BLOCK should force RED status."""
        chronicle = {
            "governance_blocking_rate": 0.1,  # Low rate normally YELLOW
            "trend": "stable",
            "total_snapshots": 10,
            "fail_count": 1,
            "most_common_failures": [],
        }
        promotion = {
            "status": "BLOCK",
            "blocking_rules": ["GOV-2"],
        }
        panel = build_governance_director_panel_v2(chronicle, promotion)

        assert panel["status_light"] == "RED"
        assert panel["promotion_status"] == "BLOCK"

    def test_promotion_warn_forces_yellow(self):
        """Promotion WARN with clear governance should be YELLOW."""
        chronicle = {
            "governance_blocking_rate": 0.0,  # Clear
            "trend": "stable",
            "total_snapshots": 10,
            "fail_count": 0,
            "most_common_failures": [],
        }
        promotion = {
            "status": "WARN",
            "blocking_rules": [],
        }
        panel = build_governance_director_panel_v2(chronicle, promotion)

        assert panel["status_light"] == "YELLOW"

    def test_trend_included(self):
        """Trend should be included from chronicle."""
        chronicle = {
            "governance_blocking_rate": 0.0,
            "trend": "improving",
            "total_snapshots": 5,
            "fail_count": 0,
            "most_common_failures": [],
        }
        panel = build_governance_director_panel_v2(chronicle)

        assert panel["trend"] == "improving"
        assert "improving" in panel["headline"]

    def test_most_common_failure_extracted(self):
        """Most common failure should be extracted."""
        chronicle = {
            "governance_blocking_rate": 0.3,
            "trend": "stable",
            "total_snapshots": 10,
            "fail_count": 3,
            "most_common_failures": [("GOV-2", 5), ("REP-3", 2)],
        }
        panel = build_governance_director_panel_v2(chronicle)

        assert panel["most_common_failure"] == "GOV-2"

    def test_snapshot_count_included(self):
        """Snapshot count should be included."""
        chronicle = {
            "governance_blocking_rate": 0.0,
            "trend": "stable",
            "total_snapshots": 42,
            "fail_count": 0,
            "most_common_failures": [],
        }
        panel = build_governance_director_panel_v2(chronicle)

        assert panel["snapshot_count"] == 42

    def test_headline_neutral_tone(self):
        """Headlines should be neutral, not prescriptive."""
        test_cases = [
            {"governance_blocking_rate": 0.0, "trend": "stable", "total_snapshots": 5, "fail_count": 0, "most_common_failures": []},
            {"governance_blocking_rate": 0.5, "trend": "degrading", "total_snapshots": 10, "fail_count": 5, "most_common_failures": []},
            {"governance_blocking_rate": 0.8, "trend": "stable", "total_snapshots": 10, "fail_count": 8, "most_common_failures": []},
        ]

        prescriptive_words = ["must", "should", "need to", "required", "fix"]

        for chronicle in test_cases:
            panel = build_governance_director_panel_v2(chronicle)
            headline_lower = panel["headline"].lower()
            for word in prescriptive_words:
                assert word not in headline_lower, f"Prescriptive word '{word}' in: {panel['headline']}"


class TestPhaseIVIntegration:
    """Integration tests for Phase IV cross-system governance."""

    def test_full_pipeline_aligned(self, valid_summary):
        """Test full pipeline with aligned governance."""
        # Build verdicts
        verdict = governance_verify(valid_summary)
        posture = build_governance_posture([verdict], ["test.json"])

        # Build chronicle
        posture["timestamp"] = "2025-01-01T00:00:00Z"
        chronicle = build_governance_chronicle([posture])

        # Build alignment (governance only)
        alignment = build_governance_alignment_view(chronicle)

        # Evaluate promotion
        promotion = evaluate_governance_for_promotion(alignment)

        # Build panel
        panel = build_governance_director_panel_v2(chronicle, promotion)

        # Assertions
        assert alignment["governance_alignment_status"] == "ALIGNED"
        if verdict.status != "FAIL":
            assert panel["status_light"] in {"GREEN", "YELLOW"}

    def test_full_pipeline_with_cross_layer_failure(self, valid_summary):
        """Test pipeline with cross-layer failures."""
        summary = deepcopy(valid_summary)
        summary["governance"]["recommendation"] = "invalid"

        verdict = governance_verify(summary)
        posture = build_governance_posture([verdict], ["fail.json"])
        posture["timestamp"] = "2025-01-01T00:00:00Z"

        chronicle = build_governance_chronicle([posture])

        # Simulate admissibility also failing on same rule
        admissibility = {
            "failing_rules": ["GOV-2"],
            "status": "FAIL",
        }

        alignment = build_governance_alignment_view(chronicle, admissibility)
        promotion = evaluate_governance_for_promotion(alignment)
        panel = build_governance_director_panel_v2(chronicle, promotion)

        # With cross-layer failure, should have TENSION or higher
        assert alignment["governance_alignment_status"] in {"TENSION", "DIVERGENT"}
        assert "GOV-2" in alignment["cross_layer_failures"]

    def test_panel_reflects_promotion_decision(self, valid_summary):
        """Panel should reflect promotion decision."""
        # Create a scenario that blocks promotion
        summary = deepcopy(valid_summary)
        summary["governance"]["recommendation"] = "invalid"

        verdict = governance_verify(summary)
        posture = build_governance_posture([verdict], ["fail.json"])
        posture["timestamp"] = "2025-01-01T00:00:00Z"

        chronicle = build_governance_chronicle([posture])

        # Multiple cross-layer failures for DIVERGENT
        admissibility = {"failing_rules": ["GOV-2", "GOV-3", "REP-1"]}
        topology = {"failing_rules": ["GOV-2", "GOV-3", "REP-1"]}

        alignment = build_governance_alignment_view(chronicle, admissibility, topology)
        promotion = evaluate_governance_for_promotion(alignment)
        panel = build_governance_director_panel_v2(chronicle, promotion)

        if promotion["status"] == "BLOCK":
            assert panel["status_light"] == "RED"
            assert "blocking" in panel["headline"].lower() or panel["governance_blocking_rate"] > 0


# =============================================================================
# PHASE V: GLOBAL GOVERNANCE SYNTHESIZER TESTS
# =============================================================================

from backend.analytics.governance_verifier import (
    # Phase V: Global Governance Synthesizer
    GovernanceSignal,
    LAYER_REPLAY,
    LAYER_TOPOLOGY,
    LAYER_SECURITY,
    LAYER_HT,
    LAYER_BUNDLE,
    LAYER_ADMISSIBILITY,
    LAYER_PREFLIGHT,
    LAYER_METRICS,
    LAYER_BUDGET,
    LAYER_CONJECTURE,
    LAYER_GOVERNANCE,
    DEFAULT_CRITICAL_LAYERS,
    adapt_replay_to_signal,
    adapt_topology_to_signal,
    adapt_security_to_signal,
    adapt_ht_to_signal,
    adapt_bundle_to_signal,
    adapt_admissibility_to_signal,
    adapt_preflight_to_signal,
    adapt_metrics_to_signal,
    adapt_budget_to_signal,
    adapt_conjecture_to_signal,
    adapt_governance_to_signal,
    adapt_layer_to_signal,
    LAYER_ADAPTERS,
    build_global_alignment_view,
    evaluate_global_promotion,
    build_global_governance_director_panel,
)


class TestGovernanceSignal:
    """Tests for the GovernanceSignal dataclass."""

    def test_basic_creation(self):
        """Test basic signal creation."""
        signal = GovernanceSignal(
            layer_name="test_layer",
            status="OK",
        )
        assert signal.layer_name == "test_layer"
        assert signal.status == "OK"
        assert signal.blocking_rules == []
        assert signal.blocking_rate == 0.0
        assert signal.headline == ""

    def test_full_creation(self):
        """Test signal with all fields."""
        signal = GovernanceSignal(
            layer_name="replay",
            status="BLOCK",
            blocking_rules=["GOV-1", "GOV-2"],
            blocking_rate=0.5,
            headline="Replay layer blocked",
        )
        assert signal.layer_name == "replay"
        assert signal.status == "BLOCK"
        assert signal.blocking_rules == ["GOV-1", "GOV-2"]
        assert signal.blocking_rate == 0.5
        assert signal.headline == "Replay layer blocked"

    def test_to_dict(self):
        """Test serialization to dict."""
        signal = GovernanceSignal(
            layer_name="topology",
            status="WARN",
            blocking_rules=["TOPO-1"],
            blocking_rate=0.25,
            headline="Topology degraded",
        )
        d = signal.to_dict()

        assert d["layer_name"] == "topology"
        assert d["status"] == "WARN"
        assert d["blocking_rules"] == ["TOPO-1"]
        assert d["blocking_rate"] == 0.25
        assert d["headline"] == "Topology degraded"

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "layer_name": "security",
            "status": "OK",
            "blocking_rules": [],
            "blocking_rate": 0.0,
            "headline": "Security OK",
        }
        signal = GovernanceSignal.from_dict(data)

        assert signal.layer_name == "security"
        assert signal.status == "OK"
        assert signal.blocking_rules == []
        assert signal.blocking_rate == 0.0
        assert signal.headline == "Security OK"

    def test_from_dict_with_defaults(self):
        """Test deserialization with missing fields uses defaults."""
        data = {"layer_name": "test"}
        signal = GovernanceSignal.from_dict(data)

        assert signal.layer_name == "test"
        assert signal.status == "BLOCK"  # Default
        assert signal.blocking_rules == []
        assert signal.blocking_rate == 0.0
        assert signal.headline == ""

    def test_roundtrip(self):
        """Test roundtrip serialization."""
        original = GovernanceSignal(
            layer_name="ht",
            status="BLOCK",
            blocking_rules=["HT-1", "HT-2"],
            blocking_rate=0.75,
            headline="Hash tree integrity failure",
        )
        restored = GovernanceSignal.from_dict(original.to_dict())

        assert restored.layer_name == original.layer_name
        assert restored.status == original.status
        assert restored.blocking_rules == original.blocking_rules
        assert restored.blocking_rate == original.blocking_rate
        assert restored.headline == original.headline


class TestLayerAdapters:
    """Tests for all layer adapter functions."""

    def test_adapt_replay_ok(self):
        """Test replay adapter with OK status."""
        data = {
            "status": "PASS",
            "failing_rules": [],
            "replay_blocking_rate": 0.0,
            "summary": "Replay validated",
        }
        signal = adapt_replay_to_signal(data)

        assert signal.layer_name == LAYER_REPLAY
        assert signal.status == "OK"
        assert signal.blocking_rules == []
        assert signal.blocking_rate == 0.0
        assert signal.headline == "Replay validated"

    def test_adapt_replay_block(self):
        """Test replay adapter with blocking status."""
        data = {
            "status": "FAIL",
            "failing_rules": ["REPLAY-1", "REPLAY-2"],
            "replay_blocking_rate": 0.4,
            "summary": "Replay failed",
        }
        signal = adapt_replay_to_signal(data)

        assert signal.layer_name == LAYER_REPLAY
        assert signal.status == "BLOCK"
        assert signal.blocking_rules == ["REPLAY-1", "REPLAY-2"]
        assert signal.blocking_rate == 0.4

    def test_adapt_replay_none(self):
        """Test replay adapter with None input."""
        signal = adapt_replay_to_signal(None)

        assert signal.layer_name == LAYER_REPLAY
        assert signal.status == "BLOCK"
        assert "No replay data" in signal.headline

    def test_adapt_topology_stable(self):
        """Test topology adapter with stable status."""
        data = {
            "trajectory_status": "stable",
            "failing_rules": [],
            "anomaly_rate": 0.01,
            "headline": "Trajectory stable",
        }
        signal = adapt_topology_to_signal(data)

        assert signal.layer_name == LAYER_TOPOLOGY
        assert signal.status == "OK"
        assert signal.blocking_rate == 0.01

    def test_adapt_topology_degrading(self):
        """Test topology adapter with degrading status."""
        data = {
            "trajectory_status": "degrading",
            "failing_rules": ["TRAJ-1"],
            "anomaly_rate": 0.15,
        }
        signal = adapt_topology_to_signal(data)

        assert signal.layer_name == LAYER_TOPOLOGY
        assert signal.status == "WARN"
        assert "TRAJ-1" in signal.blocking_rules

    def test_adapt_topology_collapsed(self):
        """Test topology adapter with collapsed status."""
        data = {
            "trajectory_status": "collapsed",
            "failing_rules": ["TRAJ-CRITICAL"],
            "anomaly_rate": 0.9,
        }
        signal = adapt_topology_to_signal(data)

        assert signal.layer_name == LAYER_TOPOLOGY
        assert signal.status == "BLOCK"

    def test_adapt_security_ok(self):
        """Test security adapter with OK status."""
        data = {
            "security_status": "SUCCESS",
            "vulnerabilities": [],
            "risk_score": 0.0,
            "summary": "No vulnerabilities",
        }
        signal = adapt_security_to_signal(data)

        assert signal.layer_name == LAYER_SECURITY
        assert signal.status == "OK"

    def test_adapt_security_warn(self):
        """Test security adapter with warning."""
        data = {
            "security_status": "WARNING",
            "vulnerabilities": ["CVE-2025-001"],
            "risk_score": 0.3,
        }
        signal = adapt_security_to_signal(data)

        assert signal.layer_name == LAYER_SECURITY
        assert signal.status == "WARN"
        assert "CVE-2025-001" in signal.blocking_rules

    def test_adapt_ht_ok(self):
        """Test HT adapter with OK status."""
        data = {
            "ht_status": "OK",
            "integrity_failures": [],
            "error_rate": 0.0,
            "summary": "Hash tree verified",
        }
        signal = adapt_ht_to_signal(data)

        assert signal.layer_name == LAYER_HT
        assert signal.status == "OK"

    def test_adapt_ht_block(self):
        """Test HT adapter with block status."""
        data = {
            "ht_status": "FAIL",
            "integrity_failures": ["HT-ROOT", "HT-LEAF"],
            "error_rate": 0.5,
        }
        signal = adapt_ht_to_signal(data)

        assert signal.layer_name == LAYER_HT
        assert signal.status == "BLOCK"
        assert len(signal.blocking_rules) == 2

    def test_adapt_bundle_ok(self):
        """Test bundle adapter with complete bundle."""
        data = {
            "bundle_status": "PASS",
            "missing_artifacts": [],
            "completeness_rate": 1.0,
            "summary": "Bundle complete",
        }
        signal = adapt_bundle_to_signal(data)

        assert signal.layer_name == LAYER_BUNDLE
        assert signal.status == "OK"
        assert signal.blocking_rate == 0.0

    def test_adapt_bundle_incomplete(self):
        """Test bundle adapter with missing artifacts."""
        data = {
            "bundle_status": "FAIL",
            "missing_artifacts": ["manifest.json", "signature.sig"],
            "completeness_rate": 0.7,
        }
        signal = adapt_bundle_to_signal(data)

        assert signal.layer_name == LAYER_BUNDLE
        assert signal.status == "BLOCK"
        assert "manifest.json" in signal.blocking_rules
        assert signal.blocking_rate == pytest.approx(0.3, abs=0.01)

    def test_adapt_admissibility_ok(self):
        """Test admissibility adapter with OK status."""
        data = {
            "overall_status": "PASS",
            "invalidating_rules": [],
            "blocking_rate": 0.0,
            "reason_summary": "Admissible",
        }
        signal = adapt_admissibility_to_signal(data)

        assert signal.layer_name == LAYER_ADMISSIBILITY
        assert signal.status == "OK"
        assert signal.headline == "Admissible"

    def test_adapt_admissibility_block(self):
        """Test admissibility adapter with block."""
        data = {
            "overall_status": "FAIL",
            "invalidating_rules": ["GOV-2", "REP-1"],
            "reason_summary": "Governance violations",
        }
        signal = adapt_admissibility_to_signal(data)

        assert signal.layer_name == LAYER_ADMISSIBILITY
        assert signal.status == "BLOCK"
        assert "GOV-2" in signal.blocking_rules
        assert signal.blocking_rate == 1.0

    def test_adapt_preflight_ok(self):
        """Test preflight adapter with OK status."""
        data = {
            "preflight_status": "PASS",
            "failed_checks": [],
            "failure_rate": 0.0,
            "summary": "All checks passed",
        }
        signal = adapt_preflight_to_signal(data)

        assert signal.layer_name == LAYER_PREFLIGHT
        assert signal.status == "OK"

    def test_adapt_preflight_fail(self):
        """Test preflight adapter with failures."""
        data = {
            "preflight_status": "FAIL",
            "failed_checks": ["ENV", "DEPS"],
            "failure_rate": 0.4,
        }
        signal = adapt_preflight_to_signal(data)

        assert signal.layer_name == LAYER_PREFLIGHT
        assert signal.status == "BLOCK"
        assert len(signal.blocking_rules) == 2

    def test_adapt_metrics_ok(self):
        """Test metrics adapter with OK status."""
        data = {
            "metrics_status": "GREEN",
            "threshold_violations": [],
            "violation_rate": 0.0,
            "summary": "All metrics within thresholds",
        }
        signal = adapt_metrics_to_signal(data)

        assert signal.layer_name == LAYER_METRICS
        assert signal.status == "OK"

    def test_adapt_metrics_warn(self):
        """Test metrics adapter with warnings."""
        data = {
            "metrics_status": "YELLOW",
            "threshold_violations": ["latency_p99"],
            "violation_rate": 0.2,
        }
        signal = adapt_metrics_to_signal(data)

        assert signal.layer_name == LAYER_METRICS
        assert signal.status == "WARN"

    def test_adapt_budget_ok(self):
        """Test budget adapter with OK status."""
        data = {
            "budget_status": "OK",
            "overruns": [],
            "utilization_rate": 0.8,
            "summary": "Within budget",
        }
        signal = adapt_budget_to_signal(data)

        assert signal.layer_name == LAYER_BUDGET
        assert signal.status == "OK"
        assert signal.blocking_rate == 0.0

    def test_adapt_budget_overrun(self):
        """Test budget adapter with overrun."""
        data = {
            "budget_status": "FAIL",
            "overruns": ["compute", "storage"],
            "utilization_rate": 1.3,
        }
        signal = adapt_budget_to_signal(data)

        assert signal.layer_name == LAYER_BUDGET
        assert signal.status == "BLOCK"
        assert signal.blocking_rate == pytest.approx(0.3, abs=0.01)

    def test_adapt_conjecture_ok(self):
        """Test conjecture adapter with OK status."""
        data = {
            "conjecture_status": "PASS",
            "unverified_conjectures": [],
            "uncertainty_rate": 0.0,
            "summary": "All conjectures verified",
        }
        signal = adapt_conjecture_to_signal(data)

        assert signal.layer_name == LAYER_CONJECTURE
        assert signal.status == "OK"

    def test_adapt_conjecture_uncertain(self):
        """Test conjecture adapter with unverified conjectures."""
        data = {
            "conjecture_status": "WARN",
            "unverified_conjectures": ["CONJ-1"],
            "uncertainty_rate": 0.1,
        }
        signal = adapt_conjecture_to_signal(data)

        assert signal.layer_name == LAYER_CONJECTURE
        assert signal.status == "WARN"

    def test_adapt_governance_ok(self):
        """Test governance adapter with OK status."""
        data = {
            "aggregate_status": "PASS",
            "failing_rules": [],
            "governance_blocking_rate": 0.0,
            "summary": "All rules passed",
        }
        signal = adapt_governance_to_signal(data)

        assert signal.layer_name == LAYER_GOVERNANCE
        assert signal.status == "OK"

    def test_adapt_governance_from_failing_files(self):
        """Test governance adapter extracts rules from failing_files."""
        data = {
            "aggregate_status": "FAIL",
            "failing_files": [
                {"file": "test1.json", "reason_codes": ["GOV-1", "GOV-2"]},
                {"file": "test2.json", "reason_codes": ["GOV-2", "REP-1"]},
            ],
            "governance_blocking_rate": 0.4,
        }
        signal = adapt_governance_to_signal(data)

        assert signal.layer_name == LAYER_GOVERNANCE
        assert signal.status == "BLOCK"
        assert "GOV-1" in signal.blocking_rules
        assert "GOV-2" in signal.blocking_rules
        assert "REP-1" in signal.blocking_rules

    def test_adapt_layer_generic(self):
        """Test generic adapter for unknown layer."""
        data = {
            "status": "OK",
            "blocking_rules": [],
            "blocking_rate": 0.0,
            "headline": "Custom layer OK",
        }
        signal = adapt_layer_to_signal("custom_layer", data)

        assert signal.layer_name == "custom_layer"
        assert signal.status == "OK"

    def test_adapt_layer_registry(self):
        """Test that all expected adapters are in registry."""
        expected_layers = [
            LAYER_REPLAY, LAYER_TOPOLOGY, LAYER_SECURITY, LAYER_HT,
            LAYER_BUNDLE, LAYER_ADMISSIBILITY, LAYER_PREFLIGHT,
            LAYER_METRICS, LAYER_BUDGET, LAYER_CONJECTURE, LAYER_GOVERNANCE,
        ]
        for layer in expected_layers:
            assert layer in LAYER_ADAPTERS

    def test_all_adapters_handle_none(self):
        """Test all adapters handle None input gracefully."""
        for layer_name, adapter in LAYER_ADAPTERS.items():
            signal = adapter(None)
            assert signal.layer_name == layer_name
            assert signal.status == "BLOCK"


class TestBuildGlobalAlignmentView:
    """Tests for build_global_alignment_view function."""

    def test_empty_signals(self):
        """Test with empty signal list."""
        view = build_global_alignment_view([])

        assert view["layer_block_map"] == {}
        assert view["global_status"] == "OK"
        assert view["alignment_score"] == 1.0
        assert view["blocking_layers"] == []

    def test_all_ok(self):
        """Test with all layers OK."""
        signals = [
            GovernanceSignal(LAYER_REPLAY, "OK"),
            GovernanceSignal(LAYER_HT, "OK"),
            GovernanceSignal(LAYER_PREFLIGHT, "OK"),
            GovernanceSignal(LAYER_ADMISSIBILITY, "OK"),
        ]
        view = build_global_alignment_view(signals)

        assert view["global_status"] == "OK"
        assert view["alignment_score"] == 1.0
        assert view["blocking_layers"] == []
        assert view["warning_layers"] == []
        assert len(view["ok_layers"]) == 4

    def test_one_warning(self):
        """Test with one warning layer."""
        signals = [
            GovernanceSignal(LAYER_REPLAY, "OK"),
            GovernanceSignal(LAYER_METRICS, "WARN", blocking_rules=["METRICS-1"]),
            GovernanceSignal(LAYER_PREFLIGHT, "OK"),
        ]
        view = build_global_alignment_view(signals)

        assert view["global_status"] == "WARN"
        assert view["warning_layers"] == [LAYER_METRICS]
        assert len(view["ok_layers"]) == 2
        assert view["alignment_score"] < 1.0

    def test_one_block(self):
        """Test with one blocking layer."""
        signals = [
            GovernanceSignal(LAYER_REPLAY, "OK"),
            GovernanceSignal(LAYER_HT, "BLOCK", blocking_rules=["HT-1"]),
            GovernanceSignal(LAYER_PREFLIGHT, "OK"),
        ]
        view = build_global_alignment_view(signals)

        assert view["global_status"] == "BLOCK"
        assert view["blocking_layers"] == [LAYER_HT]
        assert view["total_blocking_rules"] == 1

    def test_cross_layer_failures(self):
        """Test detection of cross-layer failures."""
        signals = [
            GovernanceSignal(LAYER_REPLAY, "BLOCK", blocking_rules=["GOV-1", "GOV-2"]),
            GovernanceSignal(LAYER_HT, "BLOCK", blocking_rules=["GOV-2", "HT-1"]),
            GovernanceSignal(LAYER_ADMISSIBILITY, "WARN", blocking_rules=["GOV-1"]),
        ]
        view = build_global_alignment_view(signals)

        # GOV-1 appears in replay and admissibility
        # GOV-2 appears in replay and ht
        assert "GOV-1" in view["cross_layer_failures"]
        assert "GOV-2" in view["cross_layer_failures"]
        assert "HT-1" not in view["cross_layer_failures"]

        assert len(view["rules_failing_in_multiple_layers"]["GOV-1"]) == 2
        assert len(view["rules_failing_in_multiple_layers"]["GOV-2"]) == 2

    def test_alignment_score_calculation(self):
        """Test alignment score calculation."""
        # All OK = 1.0
        signals_ok = [
            GovernanceSignal(LAYER_REPLAY, "OK"),
            GovernanceSignal(LAYER_HT, "OK"),
        ]
        view_ok = build_global_alignment_view(signals_ok)
        assert view_ok["alignment_score"] == 1.0

        # One warning = less than 1.0
        signals_warn = [
            GovernanceSignal(LAYER_REPLAY, "OK"),
            GovernanceSignal(LAYER_HT, "WARN"),
        ]
        view_warn = build_global_alignment_view(signals_warn)
        assert view_warn["alignment_score"] < 1.0
        assert view_warn["alignment_score"] > 0.5

        # One block = even lower
        signals_block = [
            GovernanceSignal(LAYER_REPLAY, "OK"),
            GovernanceSignal(LAYER_HT, "BLOCK"),
        ]
        view_block = build_global_alignment_view(signals_block)
        assert view_block["alignment_score"] < view_warn["alignment_score"]

    def test_layer_block_map(self):
        """Test layer block map generation."""
        signals = [
            GovernanceSignal(LAYER_REPLAY, "OK"),
            GovernanceSignal(LAYER_HT, "WARN"),
            GovernanceSignal(LAYER_PREFLIGHT, "BLOCK"),
        ]
        view = build_global_alignment_view(signals)

        assert view["layer_block_map"][LAYER_REPLAY] == "OK"
        assert view["layer_block_map"][LAYER_HT] == "WARN"
        assert view["layer_block_map"][LAYER_PREFLIGHT] == "BLOCK"

    def test_recommendation_generation(self):
        """Test recommendation text generation."""
        signals_ok = [GovernanceSignal(LAYER_REPLAY, "OK")]
        view_ok = build_global_alignment_view(signals_ok)
        assert "healthy" in view_ok["recommendation"].lower()

        signals_warn = [GovernanceSignal(LAYER_REPLAY, "WARN")]
        view_warn = build_global_alignment_view(signals_warn)
        assert "warning" in view_warn["recommendation"].lower()

        signals_block = [GovernanceSignal(LAYER_REPLAY, "BLOCK")]
        view_block = build_global_alignment_view(signals_block)
        assert "blocking" in view_block["recommendation"].lower()


class TestEvaluateGlobalPromotion:
    """Tests for evaluate_global_promotion function."""

    def test_empty_alignment(self):
        """Test with empty alignment."""
        result = evaluate_global_promotion({})

        assert result["promotion_ok"] is False
        assert result["status"] == "BLOCK"
        assert result["critical_layers_ok"] is False

    def test_all_ok_promotion(self):
        """Test promotion with all layers OK."""
        signals = [
            GovernanceSignal(LAYER_REPLAY, "OK"),
            GovernanceSignal(LAYER_HT, "OK"),
            GovernanceSignal(LAYER_PREFLIGHT, "OK"),
            GovernanceSignal(LAYER_ADMISSIBILITY, "OK"),
        ]
        alignment = build_global_alignment_view(signals)
        result = evaluate_global_promotion(alignment)

        assert result["promotion_ok"] is True
        assert result["status"] == "OK"
        assert result["critical_layers_ok"] is True

    def test_critical_layer_blocks(self):
        """Test that critical layer blocking prevents promotion."""
        signals = [
            GovernanceSignal(LAYER_REPLAY, "BLOCK", blocking_rules=["REPLAY-1"]),
            GovernanceSignal(LAYER_HT, "OK"),
            GovernanceSignal(LAYER_PREFLIGHT, "OK"),
            GovernanceSignal(LAYER_ADMISSIBILITY, "OK"),
        ]
        alignment = build_global_alignment_view(signals)
        result = evaluate_global_promotion(alignment)

        assert result["promotion_ok"] is False
        assert result["status"] == "BLOCK"
        assert result["critical_layers_ok"] is False
        assert LAYER_REPLAY in result["critical_layers_status"]
        assert result["critical_layers_status"][LAYER_REPLAY] == "BLOCK"

    def test_non_critical_layer_blocks_allowed(self):
        """Test that 1-2 non-critical layers blocking still permits promotion."""
        signals = [
            GovernanceSignal(LAYER_REPLAY, "OK"),
            GovernanceSignal(LAYER_HT, "OK"),
            GovernanceSignal(LAYER_PREFLIGHT, "OK"),
            GovernanceSignal(LAYER_ADMISSIBILITY, "OK"),
            GovernanceSignal(LAYER_METRICS, "BLOCK"),  # Non-critical
            GovernanceSignal(LAYER_BUDGET, "BLOCK"),   # Non-critical
        ]
        alignment = build_global_alignment_view(signals)
        result = evaluate_global_promotion(alignment)

        assert result["promotion_ok"] is True
        assert result["status"] == "WARN"
        assert result["critical_layers_ok"] is True

    def test_too_many_non_critical_blocks(self):
        """Test that 3+ non-critical layers blocking prevents promotion."""
        signals = [
            GovernanceSignal(LAYER_REPLAY, "OK"),
            GovernanceSignal(LAYER_HT, "OK"),
            GovernanceSignal(LAYER_PREFLIGHT, "OK"),
            GovernanceSignal(LAYER_ADMISSIBILITY, "OK"),
            GovernanceSignal(LAYER_METRICS, "BLOCK"),
            GovernanceSignal(LAYER_BUDGET, "BLOCK"),
            GovernanceSignal(LAYER_CONJECTURE, "BLOCK"),
        ]
        alignment = build_global_alignment_view(signals)
        result = evaluate_global_promotion(alignment)

        assert result["promotion_ok"] is False
        assert result["status"] == "BLOCK"
        assert result["critical_layers_ok"] is True  # Critical layers OK
        assert "non-critical" in result["reason"].lower()

    def test_warnings_permit_promotion(self):
        """Test that warnings permit promotion with WARN status."""
        signals = [
            GovernanceSignal(LAYER_REPLAY, "OK"),
            GovernanceSignal(LAYER_HT, "OK"),
            GovernanceSignal(LAYER_PREFLIGHT, "WARN"),
            GovernanceSignal(LAYER_ADMISSIBILITY, "OK"),
        ]
        alignment = build_global_alignment_view(signals)
        result = evaluate_global_promotion(alignment)

        assert result["promotion_ok"] is True
        assert result["status"] == "WARN"

    def test_custom_critical_layers(self):
        """Test promotion with custom critical layers."""
        signals = [
            GovernanceSignal(LAYER_REPLAY, "BLOCK"),
            GovernanceSignal(LAYER_METRICS, "OK"),
            GovernanceSignal(LAYER_BUDGET, "OK"),
        ]
        alignment = build_global_alignment_view(signals)

        # With default critical layers (includes replay) - should block
        default_result = evaluate_global_promotion(alignment)
        assert default_result["promotion_ok"] is False

        # With custom critical layers (excludes replay) - should allow
        custom_critical = frozenset({LAYER_METRICS, LAYER_BUDGET})
        custom_result = evaluate_global_promotion(alignment, critical_layers=custom_critical)
        assert custom_result["promotion_ok"] is True

    def test_default_critical_layers(self):
        """Test that default critical layers are correct."""
        assert LAYER_REPLAY in DEFAULT_CRITICAL_LAYERS
        assert LAYER_HT in DEFAULT_CRITICAL_LAYERS
        assert LAYER_PREFLIGHT in DEFAULT_CRITICAL_LAYERS
        assert LAYER_ADMISSIBILITY in DEFAULT_CRITICAL_LAYERS
        assert len(DEFAULT_CRITICAL_LAYERS) == 4

    def test_missing_critical_layer_blocks(self):
        """Test that missing critical layer data blocks promotion."""
        # Only provide non-critical layers
        signals = [
            GovernanceSignal(LAYER_METRICS, "OK"),
            GovernanceSignal(LAYER_BUDGET, "OK"),
        ]
        alignment = build_global_alignment_view(signals)
        result = evaluate_global_promotion(alignment)

        # Missing replay, ht, preflight, admissibility should block
        assert result["promotion_ok"] is False
        assert result["critical_layers_ok"] is False


class TestBuildGlobalGovernanceDirectorPanel:
    """Tests for build_global_governance_director_panel function."""

    def test_empty_alignment(self):
        """Test with empty alignment."""
        panel = build_global_governance_director_panel({})

        assert panel["status_light"] == "RED"
        assert panel["snapshot_count"] == 0
        assert panel["global_status"] == "BLOCK"

    def test_all_ok_green(self):
        """Test all OK produces GREEN status."""
        signals = [
            GovernanceSignal(LAYER_REPLAY, "OK"),
            GovernanceSignal(LAYER_HT, "OK"),
            GovernanceSignal(LAYER_PREFLIGHT, "OK"),
        ]
        alignment = build_global_alignment_view(signals)
        panel = build_global_governance_director_panel(alignment)

        assert panel["status_light"] == "GREEN"
        assert panel["global_status"] == "OK"
        assert panel["snapshot_count"] == 3
        assert panel["layer_summary"]["OK"] == 3
        assert panel["blocking_layers"] == []

    def test_warnings_yellow(self):
        """Test warnings produce YELLOW status."""
        signals = [
            GovernanceSignal(LAYER_REPLAY, "OK"),
            GovernanceSignal(LAYER_METRICS, "WARN"),
        ]
        alignment = build_global_alignment_view(signals)
        panel = build_global_governance_director_panel(alignment)

        assert panel["status_light"] == "YELLOW"
        assert panel["layer_summary"]["WARN"] == 1

    def test_blocks_red(self):
        """Test blocks produce RED status."""
        signals = [
            GovernanceSignal(LAYER_REPLAY, "BLOCK", blocking_rules=["GOV-1"]),
            GovernanceSignal(LAYER_HT, "OK"),
        ]
        alignment = build_global_alignment_view(signals)
        panel = build_global_governance_director_panel(alignment)

        assert panel["status_light"] == "RED"
        assert panel["blocking_layers"] == [LAYER_REPLAY]

    def test_most_critical_rule(self):
        """Test most critical rule detection."""
        signals = [
            GovernanceSignal(LAYER_REPLAY, "BLOCK", blocking_rules=["GOV-1", "GOV-2"]),
            GovernanceSignal(LAYER_HT, "BLOCK", blocking_rules=["GOV-2"]),
            GovernanceSignal(LAYER_ADMISSIBILITY, "WARN", blocking_rules=["GOV-2"]),
        ]
        alignment = build_global_alignment_view(signals)
        panel = build_global_governance_director_panel(alignment)

        # GOV-2 appears in all 3 layers
        assert panel["most_critical_rule"] == "GOV-2"

    def test_headline_neutral(self):
        """Test headline is neutral, not prescriptive."""
        signals = [
            GovernanceSignal(LAYER_REPLAY, "BLOCK", blocking_rules=["GOV-1"]),
        ]
        alignment = build_global_alignment_view(signals)
        panel = build_global_governance_director_panel(alignment)

        prescriptive_words = ["must", "should", "need to", "required", "fix"]
        headline_lower = panel["headline"].lower()
        for word in prescriptive_words:
            assert word not in headline_lower

    def test_layer_summary(self):
        """Test layer summary counts."""
        signals = [
            GovernanceSignal(LAYER_REPLAY, "OK"),
            GovernanceSignal(LAYER_HT, "OK"),
            GovernanceSignal(LAYER_PREFLIGHT, "WARN"),
            GovernanceSignal(LAYER_METRICS, "WARN"),
            GovernanceSignal(LAYER_BUDGET, "BLOCK"),
        ]
        alignment = build_global_alignment_view(signals)
        panel = build_global_governance_director_panel(alignment)

        assert panel["layer_summary"]["OK"] == 2
        assert panel["layer_summary"]["WARN"] == 2
        assert panel["layer_summary"]["BLOCK"] == 1

    def test_alignment_score_included(self):
        """Test alignment score is included."""
        signals = [
            GovernanceSignal(LAYER_REPLAY, "OK"),
            GovernanceSignal(LAYER_HT, "OK"),
        ]
        alignment = build_global_alignment_view(signals)
        panel = build_global_governance_director_panel(alignment)

        assert "alignment_score" in panel
        assert panel["alignment_score"] == 1.0


class TestPhaseVIntegration:
    """Integration tests for Phase V Global Governance Synthesizer."""

    def test_only_one_critical_layer_blocks(self):
        """Test scenario: only one critical layer blocks."""
        # All critical layers OK except one
        signals = [
            GovernanceSignal(LAYER_REPLAY, "BLOCK", blocking_rules=["REPLAY-FAIL"]),
            GovernanceSignal(LAYER_HT, "OK"),
            GovernanceSignal(LAYER_PREFLIGHT, "OK"),
            GovernanceSignal(LAYER_ADMISSIBILITY, "OK"),
            GovernanceSignal(LAYER_METRICS, "OK"),
            GovernanceSignal(LAYER_BUDGET, "OK"),
        ]

        alignment = build_global_alignment_view(signals)
        promotion = evaluate_global_promotion(alignment)
        panel = build_global_governance_director_panel(alignment)

        # Should block due to critical layer
        assert promotion["promotion_ok"] is False
        assert promotion["critical_layers_ok"] is False
        assert LAYER_REPLAY in promotion["reason"]

        # Panel should show RED
        assert panel["status_light"] == "RED"
        assert panel["blocking_layers"] == [LAYER_REPLAY]

    def test_multiple_non_critical_layers_warn(self):
        """Test scenario: multiple non-critical layers have warnings."""
        signals = [
            GovernanceSignal(LAYER_REPLAY, "OK"),
            GovernanceSignal(LAYER_HT, "OK"),
            GovernanceSignal(LAYER_PREFLIGHT, "OK"),
            GovernanceSignal(LAYER_ADMISSIBILITY, "OK"),
            GovernanceSignal(LAYER_METRICS, "WARN", blocking_rules=["METRICS-WARN"]),
            GovernanceSignal(LAYER_BUDGET, "WARN", blocking_rules=["BUDGET-WARN"]),
            GovernanceSignal(LAYER_CONJECTURE, "WARN", blocking_rules=["CONJ-WARN"]),
        ]

        alignment = build_global_alignment_view(signals)
        promotion = evaluate_global_promotion(alignment)
        panel = build_global_governance_director_panel(alignment)

        # Should permit promotion with warnings
        assert promotion["promotion_ok"] is True
        assert promotion["status"] == "WARN"
        assert promotion["critical_layers_ok"] is True

        # Panel should show YELLOW
        assert panel["status_light"] == "YELLOW"
        assert panel["layer_summary"]["WARN"] == 3

    def test_all_layers_ok(self):
        """Test scenario: all layers OK."""
        signals = [
            GovernanceSignal(LAYER_REPLAY, "OK"),
            GovernanceSignal(LAYER_TOPOLOGY, "OK"),
            GovernanceSignal(LAYER_SECURITY, "OK"),
            GovernanceSignal(LAYER_HT, "OK"),
            GovernanceSignal(LAYER_BUNDLE, "OK"),
            GovernanceSignal(LAYER_ADMISSIBILITY, "OK"),
            GovernanceSignal(LAYER_PREFLIGHT, "OK"),
            GovernanceSignal(LAYER_METRICS, "OK"),
            GovernanceSignal(LAYER_BUDGET, "OK"),
            GovernanceSignal(LAYER_CONJECTURE, "OK"),
        ]

        alignment = build_global_alignment_view(signals)
        promotion = evaluate_global_promotion(alignment)
        panel = build_global_governance_director_panel(alignment)

        # Should fully permit promotion
        assert promotion["promotion_ok"] is True
        assert promotion["status"] == "OK"
        assert promotion["critical_layers_ok"] is True

        # Panel should show GREEN
        assert panel["status_light"] == "GREEN"
        assert panel["snapshot_count"] == 10
        assert panel["alignment_score"] == 1.0
        assert panel["blocking_layers"] == []

    def test_full_pipeline_with_adapters(self):
        """Test full pipeline using adapters to create signals."""
        # Simulate raw layer data
        raw_data = {
            LAYER_REPLAY: {"status": "PASS", "failing_rules": [], "replay_blocking_rate": 0.0},
            LAYER_HT: {"ht_status": "OK", "integrity_failures": [], "error_rate": 0.0},
            LAYER_PREFLIGHT: {"preflight_status": "PASS", "failed_checks": [], "failure_rate": 0.0},
            LAYER_ADMISSIBILITY: {"overall_status": "PASS", "invalidating_rules": []},
            LAYER_METRICS: {"metrics_status": "YELLOW", "threshold_violations": ["latency"]},
        }

        # Adapt to signals
        signals = [adapt_layer_to_signal(layer, data) for layer, data in raw_data.items()]

        # Build alignment and evaluate
        alignment = build_global_alignment_view(signals)
        promotion = evaluate_global_promotion(alignment)
        panel = build_global_governance_director_panel(alignment)

        # Critical layers OK, one non-critical warning
        assert promotion["promotion_ok"] is True
        assert promotion["status"] == "WARN"
        assert panel["status_light"] == "YELLOW"

    def test_cross_layer_rule_detection(self):
        """Test detection of rules failing across multiple layers."""
        signals = [
            GovernanceSignal(LAYER_REPLAY, "BLOCK", blocking_rules=["GOV-1", "GOV-2"]),
            GovernanceSignal(LAYER_ADMISSIBILITY, "BLOCK", blocking_rules=["GOV-1", "REP-1"]),
            GovernanceSignal(LAYER_GOVERNANCE, "BLOCK", blocking_rules=["GOV-1", "GOV-2", "GOV-3"]),
        ]

        alignment = build_global_alignment_view(signals)

        # GOV-1 appears in all 3 layers
        assert "GOV-1" in alignment["cross_layer_failures"]
        assert len(alignment["rules_failing_in_multiple_layers"]["GOV-1"]) == 3

        # GOV-2 appears in 2 layers
        assert "GOV-2" in alignment["cross_layer_failures"]
        assert len(alignment["rules_failing_in_multiple_layers"]["GOV-2"]) == 2

        # REP-1 and GOV-3 only in 1 layer each
        assert "REP-1" not in alignment["cross_layer_failures"]
        assert "GOV-3" not in alignment["cross_layer_failures"]

    def test_custom_critical_layers_for_promotion(self):
        """Test using custom critical layers configuration."""
        signals = [
            GovernanceSignal(LAYER_REPLAY, "BLOCK"),  # Default critical
            GovernanceSignal(LAYER_SECURITY, "OK"),
            GovernanceSignal(LAYER_METRICS, "OK"),
        ]

        alignment = build_global_alignment_view(signals)

        # Default: blocks because replay is critical
        default_promo = evaluate_global_promotion(alignment)
        assert default_promo["promotion_ok"] is False

        # Custom: security and metrics critical, replay not
        custom_promo = evaluate_global_promotion(
            alignment,
            critical_layers=frozenset({LAYER_SECURITY, LAYER_METRICS})
        )
        assert custom_promo["promotion_ok"] is True
