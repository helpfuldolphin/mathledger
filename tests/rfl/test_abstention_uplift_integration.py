"""Integration tests for abstention-uplift coupling.

Tests the integration of Phase V abstention functions with:
- Uplift pipeline (compute_maas_uplift_gate_v3)
- Global console (build_global_health_surface)
- Governance tile builder
"""

import json
from typing import Any, Dict

import pytest

from rfl.verification import (
    build_abstention_storyline,
    build_epistemic_abstention_profile,
    build_epistemic_drift_timeline,
    compose_abstention_with_uplift_decision,
    summarize_abstention_for_global_console,
)
from rfl.verification.governance_tile import build_uplift_governance_tile


class TestUpliftDecisionComposition:
    """Test compose_abstention_with_uplift_decision integration."""

    def test_compose_ok_with_pass(self):
        """Test OK epistemic + PASS uplift = OK final."""
        epistemic_eval = {
            "status": "OK",
            "blocking_slices": [],
            "reasons": [],
        }
        uplift_eval = {
            "uplift_safety_decision": "PASS",
            "decision_rationale": ["All indicators OK"],
            "risk_band": "LOW",
            "blocking_slices": [],
        }

        result = compose_abstention_with_uplift_decision(epistemic_eval, uplift_eval)

        assert result["final_status"] == "OK"
        assert result["uplift_ok"] is True
        assert result["epistemic_upgrade_applied"] is False
        assert result["blocking_slices"] == []

    def test_compose_block_with_pass(self):
        """Test BLOCK epistemic + PASS uplift = BLOCK final (epistemic veto)."""
        epistemic_eval = {
            "status": "BLOCK",
            "blocking_slices": ["slice_a"],
            "reasons": ["High epistemic risk"],
        }
        uplift_eval = {
            "uplift_safety_decision": "PASS",
            "decision_rationale": ["All indicators OK"],
            "risk_band": "LOW",
            "blocking_slices": [],
        }

        result = compose_abstention_with_uplift_decision(epistemic_eval, uplift_eval)

        assert result["final_status"] == "BLOCK"
        assert result["uplift_ok"] is False
        assert result["epistemic_upgrade_applied"] is True
        assert "slice_a" in result["blocking_slices"]
        assert result["advisory"] is not None

    def test_compose_warn_with_pass(self):
        """Test WARN epistemic + PASS uplift = WARN final (epistemic upgrade)."""
        epistemic_eval = {
            "status": "WARN",
            "blocking_slices": ["slice_b"],
            "reasons": ["Moderate epistemic risk"],
        }
        uplift_eval = {
            "uplift_safety_decision": "PASS",
            "decision_rationale": ["All indicators OK"],
            "risk_band": "LOW",
            "blocking_slices": [],
        }

        result = compose_abstention_with_uplift_decision(epistemic_eval, uplift_eval)

        assert result["final_status"] == "WARN"
        assert result["uplift_ok"] is True  # WARN doesn't block
        assert result["epistemic_upgrade_applied"] is True
        assert "slice_b" in result["blocking_slices"]

    def test_compose_ok_with_block(self):
        """Test OK epistemic + BLOCK uplift = BLOCK final."""
        epistemic_eval = {
            "status": "OK",
            "blocking_slices": [],
            "reasons": [],
        }
        uplift_eval = {
            "uplift_safety_decision": "BLOCK",
            "decision_rationale": ["High risk band"],
            "risk_band": "HIGH",
            "blocking_slices": ["slice_c"],
        }

        result = compose_abstention_with_uplift_decision(epistemic_eval, uplift_eval)

        assert result["final_status"] == "BLOCK"
        assert result["uplift_ok"] is False
        assert result["epistemic_upgrade_applied"] is False
        assert "slice_c" in result["blocking_slices"]

    def test_compose_combines_blocking_slices(self):
        """Test that blocking slices from both evaluations are combined."""
        epistemic_eval = {
            "status": "WARN",
            "blocking_slices": ["slice_a", "slice_b"],
            "reasons": ["Epistemic risk"],
        }
        uplift_eval = {
            "uplift_safety_decision": "WARN",
            "decision_rationale": ["Uplift risk"],
            "risk_band": "MEDIUM",
            "blocking_slices": ["slice_b", "slice_c"],
        }

        result = compose_abstention_with_uplift_decision(epistemic_eval, uplift_eval)

        assert set(result["blocking_slices"]) == {"slice_a", "slice_b", "slice_c"}
        assert len(result["blocking_slices"]) == 3

    def test_compose_combines_reasons(self):
        """Test that reasons from both evaluations are combined."""
        epistemic_eval = {
            "status": "WARN",
            "blocking_slices": [],
            "reasons": ["Epistemic risk detected"],
        }
        uplift_eval = {
            "uplift_safety_decision": "WARN",
            "decision_rationale": ["Risk band is MEDIUM"],
            "risk_band": "MEDIUM",
            "blocking_slices": [],
        }

        result = compose_abstention_with_uplift_decision(epistemic_eval, uplift_eval)

        assert len(result["reasons"]) >= 2
        assert any("Epistemic" in r for r in result["reasons"])
        assert any("Uplift gate" in r or "MEDIUM" in r for r in result["reasons"])


class TestGovernanceTile:
    """Test build_uplift_governance_tile."""

    def test_tile_ok_status(self):
        """Test governance tile for OK status."""
        combined_eval = {
            "final_status": "OK",
            "epistemic_upgrade_applied": False,
            "blocking_slices": [],
            "reasons": [],
        }

        tile = build_uplift_governance_tile(combined_eval)

        assert tile["schema_version"] == "1.0.0"
        assert tile["final_status"] == "OK"
        assert tile["epistemic_upgrade_applied"] is False
        assert tile["blocking_slices"] == []
        assert "OK" in tile["headline"]
        assert "No blocking conditions" in tile["headline"]

    def test_tile_block_status(self):
        """Test governance tile for BLOCK status."""
        combined_eval = {
            "final_status": "BLOCK",
            "epistemic_upgrade_applied": True,
            "blocking_slices": ["slice_a", "slice_b"],
            "reasons": ["High risk"],
        }

        tile = build_uplift_governance_tile(combined_eval)

        assert tile["final_status"] == "BLOCK"
        assert tile["epistemic_upgrade_applied"] is True
        assert tile["blocking_slices"] == ["slice_a", "slice_b"]
        assert "BLOCKED" in tile["headline"]
        assert "2 slice(s)" in tile["headline"]
        assert "Epistemic gate applied" in tile["headline"]

    def test_tile_warn_status(self):
        """Test governance tile for WARN status."""
        combined_eval = {
            "final_status": "WARN",
            "epistemic_upgrade_applied": False,
            "blocking_slices": ["slice_c"],
            "reasons": ["Moderate risk"],
        }

        tile = build_uplift_governance_tile(combined_eval)

        assert tile["final_status"] == "WARN"
        assert tile["epistemic_upgrade_applied"] is False
        assert "WARN" in tile["headline"]
        assert "Review recommended" in tile["headline"]

    def test_tile_json_serializable(self):
        """Test that governance tile is JSON serializable."""
        combined_eval = {
            "final_status": "OK",
            "epistemic_upgrade_applied": False,
            "blocking_slices": [],
            "reasons": [],
        }

        tile = build_uplift_governance_tile(combined_eval)

        # Should not raise
        json_str = json.dumps(tile)
        assert json_str is not None
        assert "final_status" in json_str


class TestGlobalConsoleIntegration:
    """Test summarize_abstention_for_global_console."""

    def test_console_summary_green(self):
        """Test console summary for GREEN status (LOW risk, STABLE drift)."""
        profile = {
            "epistemic_risk_band": "LOW",
            "slice_name": "test_slice",
        }
        storyline = {
            "global_epistemic_trend": "STABLE",
            "story": "Stable pattern",
        }
        drift_timeline = {
            "risk_band": "STABLE",
            "drift_index": 0.1,
        }

        summary = summarize_abstention_for_global_console(
            profile, storyline, drift_timeline
        )

        assert summary["schema_version"] == "1.0.0"
        assert summary["abstention_status_light"] == "GREEN"
        assert summary["epistemic_risk"] == "LOW"
        assert summary["drift_band"] == "STABLE"
        assert "LOW" in summary["headline"]
        assert "STABLE" in summary["headline"]

    def test_console_summary_yellow(self):
        """Test console summary for YELLOW status (MEDIUM risk)."""
        profile = {
            "epistemic_risk_band": "MEDIUM",
            "slice_name": "test_slice",
        }
        storyline = {
            "global_epistemic_trend": "STABLE",
            "story": "Moderate risk",
        }
        drift_timeline = {
            "risk_band": "STABLE",
            "drift_index": 0.1,
        }

        summary = summarize_abstention_for_global_console(
            profile, storyline, drift_timeline
        )

        assert summary["abstention_status_light"] == "YELLOW"
        assert summary["epistemic_risk"] == "MEDIUM"

    def test_console_summary_red(self):
        """Test console summary for RED status (HIGH risk)."""
        profile = {
            "epistemic_risk_band": "HIGH",
            "slice_name": "test_slice",
        }
        storyline = {
            "global_epistemic_trend": "DEGRADING",
            "story": "High risk",
        }
        drift_timeline = {
            "risk_band": "VOLATILE",
            "drift_index": 0.8,
        }

        summary = summarize_abstention_for_global_console(
            profile, storyline, drift_timeline
        )

        assert summary["abstention_status_light"] == "RED"
        assert summary["epistemic_risk"] == "HIGH"
        assert summary["drift_band"] == "VOLATILE"

    def test_console_summary_includes_storyline(self):
        """Test that console summary includes storyline snapshot."""
        profile = {
            "epistemic_risk_band": "LOW",
            "slice_name": "test_slice",
        }
        storyline = {
            "global_epistemic_trend": "IMPROVING",
            "story": "Improving pattern",
        }
        drift_timeline = {
            "risk_band": "STABLE",
            "drift_index": 0.1,
        }

        summary = summarize_abstention_for_global_console(
            profile, storyline, drift_timeline
        )

        assert "storyline_snapshot" in summary
        assert summary["storyline_snapshot"]["trend"] == "IMPROVING"
        assert "Improving pattern" in summary["storyline_snapshot"]["story"]

    def test_console_summary_json_serializable(self):
        """Test that console summary is JSON serializable."""
        profile = {
            "epistemic_risk_band": "LOW",
            "slice_name": "test_slice",
        }
        storyline = {
            "global_epistemic_trend": "STABLE",
            "story": "Stable",
        }
        drift_timeline = {
            "risk_band": "STABLE",
            "drift_index": 0.1,
        }

        summary = summarize_abstention_for_global_console(
            profile, storyline, drift_timeline
        )

        # Should not raise
        json_str = json.dumps(summary)
        assert json_str is not None


class TestUpliftPipelineIntegration:
    """Test integration with uplift pipeline (compute_maas_uplift_gate_v3)."""

    def test_uplift_gate_without_epistemic(self):
        """Test uplift gate works without epistemic evaluation (backward compatible)."""
        from scripts.uplift_safety_engine_v6 import compute_maas_uplift_gate_v3

        safety_tensor = {
            "uplift_risk_band": "LOW",
            "tensor_norm": 0.2,
        }
        stability_forecaster = {
            "current_stability": "STABLE",
            "stability_trend": "STABLE",
        }

        result = compute_maas_uplift_gate_v3(
            safety_tensor, stability_forecaster, additional_gates=None
        )

        assert result["uplift_safety_decision"] == "PASS"
        assert result["governance_tile"] is None

    def test_uplift_gate_with_epistemic_ok(self):
        """Test uplift gate with OK epistemic evaluation."""
        from scripts.uplift_safety_engine_v6 import compute_maas_uplift_gate_v3

        safety_tensor = {
            "uplift_risk_band": "LOW",
            "tensor_norm": 0.2,
        }
        stability_forecaster = {
            "current_stability": "STABLE",
            "stability_trend": "STABLE",
        }
        epistemic_eval = {
            "status": "OK",
            "blocking_slices": [],
            "reasons": [],
        }

        result = compute_maas_uplift_gate_v3(
            safety_tensor,
            stability_forecaster,
            additional_gates=None,
            epistemic_eval=epistemic_eval,
        )

        assert result["uplift_safety_decision"] == "PASS"
        assert result["governance_tile"] is not None
        assert result["governance_tile"]["final_status"] == "OK"

    def test_uplift_gate_with_epistemic_block(self):
        """Test uplift gate with BLOCK epistemic evaluation (veto power)."""
        from scripts.uplift_safety_engine_v6 import compute_maas_uplift_gate_v3

        safety_tensor = {
            "uplift_risk_band": "LOW",
            "tensor_norm": 0.2,
        }
        stability_forecaster = {
            "current_stability": "STABLE",
            "stability_trend": "STABLE",
        }
        epistemic_eval = {
            "status": "BLOCK",
            "blocking_slices": ["slice_a"],
            "reasons": ["High epistemic risk"],
        }

        result = compute_maas_uplift_gate_v3(
            safety_tensor,
            stability_forecaster,
            additional_gates=None,
            epistemic_eval=epistemic_eval,
        )

        # Epistemic gate should upgrade PASS to BLOCK
        assert result["uplift_safety_decision"] == "BLOCK"
        assert result["governance_tile"] is not None
        assert result["governance_tile"]["final_status"] == "BLOCK"
        assert result["governance_tile"]["epistemic_upgrade_applied"] is True
        assert "slice_a" in result["governance_tile"]["blocking_slices"]


class TestDeterminism:
    """Test deterministic behavior of integration functions."""

    def test_compose_deterministic(self):
        """Test that compose_abstention_with_uplift_decision is deterministic."""
        epistemic_eval = {
            "status": "WARN",
            "blocking_slices": ["slice_a"],
            "reasons": ["Moderate risk"],
        }
        uplift_eval = {
            "uplift_safety_decision": "PASS",
            "decision_rationale": ["All OK"],
            "risk_band": "LOW",
            "blocking_slices": [],
        }

        result1 = compose_abstention_with_uplift_decision(epistemic_eval, uplift_eval)
        result2 = compose_abstention_with_uplift_decision(epistemic_eval, uplift_eval)

        assert result1 == result2

    def test_governance_tile_deterministic(self):
        """Test that build_uplift_governance_tile is deterministic."""
        combined_eval = {
            "final_status": "WARN",
            "epistemic_upgrade_applied": True,
            "blocking_slices": ["slice_a"],
            "reasons": ["Risk detected"],
        }

        tile1 = build_uplift_governance_tile(combined_eval)
        tile2 = build_uplift_governance_tile(combined_eval)

        assert tile1 == tile2

