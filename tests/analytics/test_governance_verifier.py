"""
Tests for Governance Verifier

==============================================================================
STATUS: PHASE V — GLOBAL GOVERNANCE SYNTHESIZER + END-TO-END WIRING
==============================================================================

This module tests the governance_verifier implementation against:
- docs/UPLIFT_ANALYTICS_GOVERNANCE_SPEC.md
- docs/UPLIFT_GOVERNANCE_VERIFIER_SPEC.md

Test coverage:
- GovernanceVerdict data model
- All 43 governance rules (GOV-*, REP-*, MAN-*, INV-*)
- Decision tree logic (PASS/WARN/FAIL)
- Determinism verification
- Phase III-V: Director Console, Chronicle, Cross-System Gate
- End-to-End Wiring with TDA, Telemetry, Slice Identity
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
    LAYER_TDA,
    LAYER_TELEMETRY_TDA,
    LAYER_SLICE_IDENTITY,
    DEFAULT_CRITICAL_LAYERS,
    LAYER_ADAPTERS,
    adapt_layer_to_signal,
    adapt_tda_to_signal,
    adapt_telemetry_tda_to_signal,
    adapt_slice_identity_to_signal,
    build_global_alignment_view,
    evaluate_global_promotion,
    build_global_governance_director_panel,
    # End-to-End
    collect_all_layer_signals,
    evaluate_full_cortex_body_promotion,
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
        "experiment_id": "U2-TEST-001",
        "timestamp": "2025-01-15T10:00:00Z",
        "version": "1.0.0",
        "seeds": {
            "baseline": 12345,
            "rfl": 67890,
        },
        "slice_config": {
            "prop_depth4": {"enabled": True},
            "fol_eq_group": {"enabled": True},
            "fol_eq_ring": {"enabled": True},
            "linear_arith": {"enabled": True},
        },
        "hashes": {
            "config": "a" * 64,
            "data": "b" * 64,
        },
    }


# =============================================================================
# BASIC TESTS
# =============================================================================

class TestGovernanceVerdict:
    """Tests for GovernanceVerdict data model."""

    def test_verdict_fields(self, valid_summary):
        """Verdict should have all required fields."""
        verdict = governance_verify(valid_summary)
        assert hasattr(verdict, "status")
        assert hasattr(verdict, "rules_checked")
        assert hasattr(verdict, "passed_rules")
        assert hasattr(verdict, "warnings")
        assert hasattr(verdict, "invalidating_rules")
        assert hasattr(verdict, "details")
        assert hasattr(verdict, "timestamp")
        assert hasattr(verdict, "verifier_version")

    def test_verdict_to_dict_serializable(self, valid_summary):
        """Verdict should be JSON serializable."""
        verdict = governance_verify(valid_summary)
        d = verdict.to_dict()
        json_str = json.dumps(d)
        assert json_str is not None

    def test_verdict_status_values(self, valid_summary):
        """Status should be PASS, WARN, or FAIL."""
        verdict = governance_verify(valid_summary)
        assert verdict.status in {"PASS", "WARN", "FAIL"}


class TestDecisionTree:
    """Tests for decision tree logic."""

    def test_full_pass(self, valid_summary):
        """Valid summary should PASS."""
        verdict = governance_verify(valid_summary)
        assert verdict.status == "PASS"

    def test_invalidating_violation_causes_fail(self, valid_summary):
        """INVALIDATING violation should cause FAIL."""
        summary = deepcopy(valid_summary)
        summary["governance"]["recommendation"] = "invalid_decision"
        verdict = governance_verify(summary)
        assert verdict.status == "FAIL"
        assert "GOV-2" in verdict.invalidating_rules


class TestGovernanceRules:
    """Tests for governance rules."""

    def test_gov_1_threshold_compliance_pass(self, valid_summary):
        """GOV-1: Should pass when thresholds met."""
        verdict = governance_verify(valid_summary)
        assert "GOV-1" in verdict.passed_rules

    def test_gov_1_threshold_compliance_fail(self, valid_summary):
        """GOV-1: Should fail when threshold not met."""
        summary = deepcopy(valid_summary)
        summary["slices"]["prop_depth4"]["success_rate"]["rfl"] = 0.50
        verdict = governance_verify(summary)
        assert "GOV-1" in verdict.invalidating_rules

    def test_gov_2_decision_exclusivity_valid(self, valid_summary):
        """GOV-2: Valid decision should pass."""
        for decision in ["proceed", "proceed-with-monitoring", "hold", "rollback"]:
            summary = deepcopy(valid_summary)
            summary["governance"]["recommendation"] = decision
            verdict = governance_verify(summary)
            assert "GOV-2" in verdict.passed_rules

    def test_gov_2_decision_exclusivity_invalid(self, valid_summary):
        """GOV-2: Invalid decision should fail."""
        summary = deepcopy(valid_summary)
        summary["governance"]["recommendation"] = "invalid"
        verdict = governance_verify(summary)
        assert "GOV-2" in verdict.invalidating_rules


# =============================================================================
# PHASE V: GOVERNANCE SIGNAL TESTS
# =============================================================================

class TestGovernanceSignal:
    """Tests for GovernanceSignal dataclass."""

    def test_basic_creation(self):
        """Test basic signal creation."""
        signal = GovernanceSignal(layer_name="test", status="OK")
        assert signal.layer_name == "test"
        assert signal.status == "OK"
        assert signal.blocking_rules == []
        assert signal.blocking_rate == 0.0

    def test_to_dict(self):
        """Test serialization."""
        signal = GovernanceSignal(
            layer_name="replay",
            status="BLOCK",
            blocking_rules=["GOV-1"],
            blocking_rate=0.5,
            headline="Test headline",
        )
        d = signal.to_dict()
        assert d["layer_name"] == "replay"
        assert d["status"] == "BLOCK"
        assert d["blocking_rules"] == ["GOV-1"]

    def test_from_dict(self):
        """Test deserialization."""
        data = {"layer_name": "ht", "status": "WARN", "blocking_rules": ["HT-1"]}
        signal = GovernanceSignal.from_dict(data)
        assert signal.layer_name == "ht"
        assert signal.status == "WARN"

    def test_roundtrip(self):
        """Test roundtrip serialization."""
        original = GovernanceSignal("tda", "OK", [], 0.0, "TDA healthy")
        restored = GovernanceSignal.from_dict(original.to_dict())
        assert restored.layer_name == original.layer_name
        assert restored.status == original.status


# =============================================================================
# TDA / TELEMETRY / SLICE IDENTITY ADAPTER TESTS
# =============================================================================

class TestTDAAdapter:
    """Tests for TDA layer adapter."""

    def test_adapt_tda_healthy(self):
        """Test TDA adapter with healthy status."""
        data = {
            "governance_signal": "HEALTHY",
            "tda_status": "OK",
            "block_rate": 0.0,
            "hss_trend": "STABLE",
            "notes": ["All HSS values within bounds"],
        }
        signal = adapt_tda_to_signal(data)
        assert signal.layer_name == LAYER_TDA
        assert signal.status == "OK"
        assert signal.blocking_rate == 0.0

    def test_adapt_tda_degraded(self):
        """Test TDA adapter with degraded status."""
        data = {
            "governance_signal": "DEGRADED",
            "block_rate": 0.2,
            "hss_trend": "DEGRADING",
        }
        signal = adapt_tda_to_signal(data)
        assert signal.layer_name == LAYER_TDA
        assert signal.status == "WARN"
        assert signal.blocking_rate == 0.2

    def test_adapt_tda_critical(self):
        """Test TDA adapter with critical status."""
        data = {
            "governance_signal": "CRITICAL",
            "failing_rules": ["TDA-HSS-001"],
            "block_rate": 0.8,
        }
        signal = adapt_tda_to_signal(data)
        assert signal.layer_name == LAYER_TDA
        assert signal.status == "BLOCK"
        assert "TDA-HSS-001" in signal.blocking_rules

    def test_adapt_tda_none(self):
        """Test TDA adapter with None input."""
        signal = adapt_tda_to_signal(None)
        assert signal.layer_name == LAYER_TDA
        assert signal.status == "BLOCK"
        assert "No TDA data" in signal.headline


class TestTelemetryTDAAdapter:
    """Tests for Telemetry TDA layer adapter."""

    def test_adapt_telemetry_tda_ok(self):
        """Test telemetry TDA adapter with OK status."""
        data = {
            "governance_signal": "OK",
            "tda_status": "OK",
            "block_rate": 0.0,
            "mean_hss": 0.85,
            "hss_trend": "IMPROVING",
        }
        signal = adapt_telemetry_tda_to_signal(data)
        assert signal.layer_name == LAYER_TELEMETRY_TDA
        assert signal.status == "OK"
        assert "0.850" in signal.headline

    def test_adapt_telemetry_tda_attention(self):
        """Test telemetry TDA adapter with attention status."""
        data = {
            "tda_status": "ATTENTION",
            "block_rate": 0.15,
            "hss_trend": "DEGRADING",
        }
        signal = adapt_telemetry_tda_to_signal(data)
        assert signal.layer_name == LAYER_TELEMETRY_TDA
        assert signal.status == "WARN"

    def test_adapt_telemetry_tda_none(self):
        """Test telemetry TDA adapter with None."""
        signal = adapt_telemetry_tda_to_signal(None)
        assert signal.status == "BLOCK"


class TestSliceIdentityAdapter:
    """Tests for Slice Identity layer adapter."""

    def test_adapt_slice_identity_aligned(self):
        """Test slice identity adapter with aligned status."""
        data = {
            "signal": "OK",
            "alignment_status": "ALIGNED",
            "blocking_slices": [],
            "average_stability": 0.98,
        }
        signal = adapt_slice_identity_to_signal(data)
        assert signal.layer_name == LAYER_SLICE_IDENTITY
        assert signal.status == "OK"
        assert signal.blocking_rate == pytest.approx(0.02, abs=0.01)
        assert "ALIGNED" in signal.headline

    def test_adapt_slice_identity_partial(self):
        """Test slice identity adapter with partial alignment."""
        data = {
            "signal": "WARN",
            "alignment_status": "PARTIAL",
            "blocking_slices": ["prop_depth4"],
            "average_stability": 0.75,
        }
        signal = adapt_slice_identity_to_signal(data)
        assert signal.layer_name == LAYER_SLICE_IDENTITY
        assert signal.status == "WARN"
        assert "prop_depth4" in signal.blocking_rules

    def test_adapt_slice_identity_broken(self):
        """Test slice identity adapter with broken alignment."""
        data = {
            "signal": "BLOCK",
            "alignment_status": "BROKEN",
            "blocking_slices": ["fol_eq_group", "fol_eq_ring"],
            "average_stability": 0.3,
        }
        signal = adapt_slice_identity_to_signal(data)
        assert signal.layer_name == LAYER_SLICE_IDENTITY
        assert signal.status == "BLOCK"
        assert len(signal.blocking_rules) == 2

    def test_adapt_slice_identity_none(self):
        """Test slice identity adapter with None."""
        signal = adapt_slice_identity_to_signal(None)
        assert signal.status == "BLOCK"


# =============================================================================
# LAYER ADAPTER REGISTRY TESTS
# =============================================================================

class TestLayerAdapterRegistry:
    """Tests for layer adapter registry."""

    def test_all_layers_registered(self):
        """All 14 layers should be in the registry."""
        expected_layers = [
            LAYER_REPLAY, LAYER_TOPOLOGY, LAYER_SECURITY, LAYER_HT,
            LAYER_BUNDLE, LAYER_ADMISSIBILITY, LAYER_PREFLIGHT,
            LAYER_METRICS, LAYER_BUDGET, LAYER_CONJECTURE, LAYER_GOVERNANCE,
            LAYER_TDA, LAYER_TELEMETRY_TDA, LAYER_SLICE_IDENTITY,
        ]
        for layer in expected_layers:
            assert layer in LAYER_ADAPTERS, f"Missing adapter for {layer}"

    def test_all_adapters_handle_none(self):
        """All adapters should handle None gracefully."""
        for layer_name, adapter in LAYER_ADAPTERS.items():
            signal = adapter(None)
            assert signal.layer_name == layer_name
            assert signal.status == "BLOCK"

    def test_adapt_layer_to_signal_fallback(self):
        """Generic adapter should work for unknown layers."""
        data = {"status": "OK", "blocking_rules": [], "headline": "Custom OK"}
        signal = adapt_layer_to_signal("custom_layer", data)
        assert signal.layer_name == "custom_layer"
        assert signal.status == "OK"


# =============================================================================
# DEFAULT_CRITICAL_LAYERS TESTS (First Light Gating)
# =============================================================================

class TestDefaultCriticalLayers:
    """Tests for DEFAULT_CRITICAL_LAYERS configuration."""

    def test_critical_layers_include_tda(self):
        """TDA should be in critical layers for First Light."""
        assert LAYER_TDA in DEFAULT_CRITICAL_LAYERS

    def test_critical_layers_include_replay(self):
        """Replay should be critical."""
        assert LAYER_REPLAY in DEFAULT_CRITICAL_LAYERS

    def test_critical_layers_include_ht(self):
        """HT should be critical."""
        assert LAYER_HT in DEFAULT_CRITICAL_LAYERS

    def test_critical_layers_include_preflight(self):
        """Preflight should be critical."""
        assert LAYER_PREFLIGHT in DEFAULT_CRITICAL_LAYERS

    def test_critical_layers_include_admissibility(self):
        """Admissibility should be critical."""
        assert LAYER_ADMISSIBILITY in DEFAULT_CRITICAL_LAYERS

    def test_critical_layers_count(self):
        """Should have exactly 5 critical layers for First Light."""
        assert len(DEFAULT_CRITICAL_LAYERS) == 5


# =============================================================================
# GLOBAL ALIGNMENT VIEW TESTS
# =============================================================================

class TestBuildGlobalAlignmentViewWithNewLayers:
    """Tests for build_global_alignment_view with new layers."""

    def test_all_layers_ok(self):
        """All layers OK should produce OK status."""
        signals = [
            GovernanceSignal(LAYER_REPLAY, "OK"),
            GovernanceSignal(LAYER_HT, "OK"),
            GovernanceSignal(LAYER_PREFLIGHT, "OK"),
            GovernanceSignal(LAYER_ADMISSIBILITY, "OK"),
            GovernanceSignal(LAYER_TDA, "OK"),
        ]
        view = build_global_alignment_view(signals)
        assert view["global_status"] == "OK"
        assert view["alignment_score"] == 1.0
        assert len(view["blocking_layers"]) == 0

    def test_tda_blocks(self):
        """TDA blocking should affect global status."""
        signals = [
            GovernanceSignal(LAYER_REPLAY, "OK"),
            GovernanceSignal(LAYER_HT, "OK"),
            GovernanceSignal(LAYER_TDA, "BLOCK", ["TDA-001"]),
        ]
        view = build_global_alignment_view(signals)
        assert view["global_status"] == "BLOCK"
        assert LAYER_TDA in view["blocking_layers"]

    def test_cross_layer_rule_detection(self):
        """Rules appearing in multiple layers should be detected."""
        signals = [
            GovernanceSignal(LAYER_REPLAY, "BLOCK", ["GOV-1", "GOV-2"]),
            GovernanceSignal(LAYER_TDA, "BLOCK", ["GOV-1"]),
            GovernanceSignal(LAYER_ADMISSIBILITY, "WARN", ["GOV-1", "ADM-1"]),
        ]
        view = build_global_alignment_view(signals)
        assert "GOV-1" in view["cross_layer_failures"]
        assert len(view["rules_failing_in_multiple_layers"]["GOV-1"]) == 3


# =============================================================================
# EVALUATE_GLOBAL_PROMOTION TESTS
# =============================================================================

class TestEvaluateGlobalPromotionWithTDA:
    """Tests for evaluate_global_promotion with TDA as critical layer."""

    def test_tda_block_prevents_promotion(self):
        """TDA block should prevent promotion (it's critical)."""
        signals = [
            GovernanceSignal(LAYER_REPLAY, "OK"),
            GovernanceSignal(LAYER_HT, "OK"),
            GovernanceSignal(LAYER_PREFLIGHT, "OK"),
            GovernanceSignal(LAYER_ADMISSIBILITY, "OK"),
            GovernanceSignal(LAYER_TDA, "BLOCK", ["TDA-HSS"]),
        ]
        alignment = build_global_alignment_view(signals)
        result = evaluate_global_promotion(alignment)

        assert result["promotion_ok"] is False
        assert result["status"] == "BLOCK"
        assert result["critical_layers_ok"] is False
        assert LAYER_TDA in result["reason"]

    def test_all_critical_ok_allows_promotion(self):
        """All critical layers OK should allow promotion."""
        signals = [
            GovernanceSignal(LAYER_REPLAY, "OK"),
            GovernanceSignal(LAYER_HT, "OK"),
            GovernanceSignal(LAYER_PREFLIGHT, "OK"),
            GovernanceSignal(LAYER_ADMISSIBILITY, "OK"),
            GovernanceSignal(LAYER_TDA, "OK"),
            GovernanceSignal(LAYER_METRICS, "BLOCK"),  # Non-critical
        ]
        alignment = build_global_alignment_view(signals)
        result = evaluate_global_promotion(alignment)

        assert result["promotion_ok"] is True
        assert result["critical_layers_ok"] is True

    def test_custom_critical_layers(self):
        """Custom critical layers should override defaults."""
        signals = [
            GovernanceSignal(LAYER_TDA, "BLOCK"),
            GovernanceSignal(LAYER_SLICE_IDENTITY, "OK"),
        ]
        alignment = build_global_alignment_view(signals)

        # With default: TDA blocks
        default_result = evaluate_global_promotion(alignment)
        assert default_result["promotion_ok"] is False

        # With custom: only slice_identity critical
        custom_result = evaluate_global_promotion(
            alignment,
            critical_layers=frozenset({LAYER_SLICE_IDENTITY})
        )
        assert custom_result["promotion_ok"] is True


# =============================================================================
# END-TO-END WIRING TESTS
# =============================================================================

class TestEndToEndWiring:
    """Tests for end-to-end wiring functions."""

    def test_collect_all_layer_signals(self):
        """collect_all_layer_signals should adapt all provided layers."""
        layer_data = {
            LAYER_REPLAY: {"status": "PASS"},
            LAYER_TDA: {"governance_signal": "HEALTHY"},
            LAYER_HT: {"ht_status": "OK"},
        }
        signals = collect_all_layer_signals(layer_data)

        assert len(signals) == 3
        layer_names = {s.layer_name for s in signals}
        assert LAYER_REPLAY in layer_names
        assert LAYER_TDA in layer_names
        assert LAYER_HT in layer_names

    def test_evaluate_full_cortex_body_promotion_all_ok(self):
        """Full pipeline with all layers OK should permit promotion."""
        layer_data = {
            LAYER_REPLAY: {"status": "PASS"},
            LAYER_HT: {"ht_status": "OK"},
            LAYER_PREFLIGHT: {"preflight_status": "PASS"},
            LAYER_ADMISSIBILITY: {"overall_status": "PASS"},
            LAYER_TDA: {"governance_signal": "HEALTHY"},
        }
        result = evaluate_full_cortex_body_promotion(layer_data)

        assert result["promotion_ok"] is True
        assert result["status"] == "OK"
        assert result["director_panel"]["status_light"] == "GREEN"

    def test_evaluate_full_cortex_body_promotion_tda_block(self):
        """Full pipeline with TDA block should fail promotion."""
        layer_data = {
            LAYER_REPLAY: {"status": "PASS"},
            LAYER_HT: {"ht_status": "OK"},
            LAYER_PREFLIGHT: {"preflight_status": "PASS"},
            LAYER_ADMISSIBILITY: {"overall_status": "PASS"},
            LAYER_TDA: {"governance_signal": "CRITICAL", "failing_rules": ["TDA-001"]},
        }
        result = evaluate_full_cortex_body_promotion(layer_data)

        assert result["promotion_ok"] is False
        assert result["status"] == "BLOCK"
        assert result["director_panel"]["status_light"] == "RED"
        assert LAYER_TDA in result["critical_layers_status"]

    def test_evaluate_full_cortex_body_with_all_14_layers(self):
        """Full pipeline with all 14 layers should work."""
        layer_data = {
            LAYER_REPLAY: {"status": "PASS"},
            LAYER_TOPOLOGY: {"trajectory_status": "stable"},
            LAYER_SECURITY: {"security_status": "SUCCESS"},
            LAYER_HT: {"ht_status": "OK"},
            LAYER_BUNDLE: {"bundle_status": "PASS"},
            LAYER_ADMISSIBILITY: {"overall_status": "PASS"},
            LAYER_PREFLIGHT: {"preflight_status": "PASS"},
            LAYER_METRICS: {"metrics_status": "GREEN"},
            LAYER_BUDGET: {"budget_status": "OK"},
            LAYER_CONJECTURE: {"conjecture_status": "PASS"},
            LAYER_GOVERNANCE: {"aggregate_status": "PASS"},
            LAYER_TDA: {"governance_signal": "HEALTHY"},
            LAYER_TELEMETRY_TDA: {"tda_status": "OK"},
            LAYER_SLICE_IDENTITY: {"signal": "OK", "alignment_status": "ALIGNED"},
        }
        result = evaluate_full_cortex_body_promotion(layer_data)

        assert result["promotion_ok"] is True
        assert result["status"] == "OK"
        assert len(result["signals"]) == 14
        assert result["global_alignment"]["alignment_score"] == 1.0

    def test_evaluate_full_cortex_body_missing_critical(self):
        """Missing critical layer should block promotion."""
        # Only provide non-critical layers
        layer_data = {
            LAYER_METRICS: {"metrics_status": "GREEN"},
            LAYER_BUDGET: {"budget_status": "OK"},
        }
        result = evaluate_full_cortex_body_promotion(layer_data)

        assert result["promotion_ok"] is False
        # Check critical layers status shows missing as BLOCK
        for layer in DEFAULT_CRITICAL_LAYERS:
            assert result["critical_layers_status"].get(layer) == "BLOCK"


# =============================================================================
# DIRECTOR PANEL TESTS
# =============================================================================

class TestDirectorPanelWithNewLayers:
    """Tests for director panel with new layers."""

    def test_director_panel_shows_tda_blocking(self):
        """Director panel should show TDA in blocking layers."""
        signals = [
            GovernanceSignal(LAYER_REPLAY, "OK"),
            GovernanceSignal(LAYER_TDA, "BLOCK", ["TDA-CRITICAL"]),
        ]
        alignment = build_global_alignment_view(signals)
        panel = build_global_governance_director_panel(alignment, signals)

        assert panel["status_light"] == "RED"
        assert LAYER_TDA in panel["blocking_layers"]

    def test_director_panel_layer_summary(self):
        """Director panel should count layers correctly."""
        signals = [
            GovernanceSignal(LAYER_REPLAY, "OK"),
            GovernanceSignal(LAYER_TDA, "WARN"),
            GovernanceSignal(LAYER_SLICE_IDENTITY, "OK"),
            GovernanceSignal(LAYER_HT, "BLOCK"),
        ]
        alignment = build_global_alignment_view(signals)
        panel = build_global_governance_director_panel(alignment, signals)

        assert panel["layer_summary"]["OK"] == 2
        assert panel["layer_summary"]["WARN"] == 1
        assert panel["layer_summary"]["BLOCK"] == 1
        assert panel["snapshot_count"] == 4


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestPhaseVIntegration:
    """Integration tests for Phase V with end-to-end wiring."""

    def test_first_light_gating_scenario(self):
        """Test First Light gating: all critical layers must pass."""
        # Scenario: TDA is critical and blocks
        layer_data = {
            LAYER_REPLAY: {"status": "PASS"},
            LAYER_HT: {"ht_status": "OK"},
            LAYER_PREFLIGHT: {"preflight_status": "PASS"},
            LAYER_ADMISSIBILITY: {"overall_status": "PASS"},
            LAYER_TDA: {"governance_signal": "DEGRADED"},  # WARN
        }
        result = evaluate_full_cortex_body_promotion(layer_data)

        # TDA WARN doesn't block, only BLOCK does
        assert result["promotion_ok"] is True
        assert result["status"] == "WARN"

        # Now TDA blocks
        layer_data[LAYER_TDA] = {"governance_signal": "CRITICAL"}
        result = evaluate_full_cortex_body_promotion(layer_data)
        assert result["promotion_ok"] is False

    def test_cortex_body_integration(self):
        """Test full Cortex+Body system integration."""
        # Cortex layers
        cortex_data = {
            LAYER_REPLAY: {"status": "PASS", "summary": "Replay verified"},
            LAYER_TDA: {"governance_signal": "HEALTHY", "hss_trend": "STABLE"},
            LAYER_ADMISSIBILITY: {"overall_status": "PASS"},
        }

        # Body layers
        body_data = {
            LAYER_HT: {"ht_status": "OK"},
            LAYER_PREFLIGHT: {"preflight_status": "PASS"},
            LAYER_SLICE_IDENTITY: {"signal": "OK", "alignment_status": "ALIGNED"},
        }

        # Combine
        layer_data = {**cortex_data, **body_data}
        result = evaluate_full_cortex_body_promotion(layer_data)

        assert result["promotion_ok"] is True
        assert result["status"] == "OK"
        assert len(result["signals"]) == 6

    def test_cross_layer_failure_detection_integration(self):
        """Test cross-layer failure detection across new layers."""
        layer_data = {
            LAYER_REPLAY: {"status": "FAIL", "failing_rules": ["GOV-1"]},
            LAYER_TDA: {"governance_signal": "CRITICAL", "failing_rules": ["GOV-1"]},
            LAYER_SLICE_IDENTITY: {"signal": "BLOCK", "blocking_slices": ["GOV-1"]},
        }
        result = evaluate_full_cortex_body_promotion(layer_data)

        assert result["promotion_ok"] is False
        # GOV-1 should appear in cross-layer failures
        assert "GOV-1" in result["global_alignment"]["cross_layer_failures"]
