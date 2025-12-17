# tests/ci/test_adversarial_governance_tile_serializes.py
"""
CI test for adversarial governance tile serialization.

Verifies:
- Build from synthetic coverage_index + robustness_radar
- JSON-serializable
- Deterministic
- No governance writes or exit codes (SHADOW MODE)
"""

import json
import pytest
from typing import Dict, Any

from backend.health.adversarial_pressure_adapter import (
    build_adversarial_governance_tile,
    extract_adversarial_signal_for_release,
    attach_adversarial_governance_to_evidence,
    summarize_adversarial_for_uplift_council,
    extract_adversarial_summary_for_p3_stability,
    extract_adversarial_calibration_for_p4,
    build_first_light_adversarial_coverage_annex,
    ADVERSARIAL_GOVERNANCE_TILE_SCHEMA_VERSION,
)

# Import Phase V functions (these may not exist if experiments module is unavailable)
try:
    from experiments.metrics_adversarial_harness import (
        build_adversarial_pressure_model,
        build_evolving_adversarial_scenario_plan,
        build_adversarial_failover_plan_v2,
        build_metric_adversarial_coverage_index,
        build_metric_robustness_radar,
        evaluate_adversarial_readiness_for_promotion,
        build_adversarial_failover_plan,
        build_robustness_scorecard,
        HarnessSummary,
        METRIC_KINDS,
    )
    HAS_ADVERSARIAL_HARNESS = True
except ImportError:
    HAS_ADVERSARIAL_HARNESS = False


@pytest.fixture
def synthetic_coverage_index() -> Dict[str, Any]:
    """Synthetic coverage index for testing."""
    return {
        "metrics": {
            "goal_hit": {
                "robustness_tag": "WELL_EXERCISED",
                "scenario_count": 3,
                "regression_count": 0,
                "coverage_status": "OK",
            },
            "density": {
                "robustness_tag": "PARTIALLY_TESTED",
                "scenario_count": 2,
                "regression_count": 0,
                "coverage_status": "SPARSE",
            },
            "chain_length": {
                "robustness_tag": "SPARSELY_TESTED",
                "scenario_count": 1,
                "regression_count": 0,
                "coverage_status": "SPARSE",
            },
            "multi_goal": {
                "robustness_tag": "SPARSELY_TESTED",
                "scenario_count": 0,
                "regression_count": 0,
                "coverage_status": "AT_RISK",
            },
        },
        "global": {
            "metrics_at_risk": ["multi_goal"],
            "coverage_ok": False,
        },
    }


@pytest.fixture
def synthetic_robustness_radar() -> Dict[str, Any]:
    """Synthetic robustness radar for testing."""
    return {
        "metrics": {
            "goal_hit": {
                "scenarios_exercised": ["baseline_sanity", "goal_hit_boundary", "ci_quick"],
            },
            "density": {
                "scenarios_exercised": ["baseline_sanity", "density_boundary"],
            },
            "chain_length": {
                "scenarios_exercised": ["baseline_sanity"],
            },
            "multi_goal": {
                "scenarios_exercised": [],
            },
        },
        "global": {
            "metrics_at_risk": ["multi_goal"],
        },
    }


@pytest.fixture
def synthetic_pressure_model(synthetic_coverage_index, synthetic_robustness_radar) -> Dict[str, Any]:
    """Synthetic pressure model for testing."""
    if not HAS_ADVERSARIAL_HARNESS:
        # Return synthetic model if harness unavailable
        return {
            "metric_pressure_scores": {
                "goal_hit": 0.2,
                "density": 0.4,
                "chain_length": 0.7,
                "multi_goal": 1.0,
            },
            "scenario_pressure_targets": {
                "chain_length": ["baseline_sanity"],
                "multi_goal": [],
            },
            "global_pressure_index": 0.575,
            "pressure_band": "MEDIUM",
            "neutral_notes": ["chain_length: Pressure score 0.700 exceeds priority threshold"],
        }
    
    return build_adversarial_pressure_model(synthetic_coverage_index, synthetic_robustness_radar)


@pytest.fixture
def synthetic_scenario_plan(synthetic_pressure_model, synthetic_coverage_index) -> Dict[str, Any]:
    """Synthetic scenario plan for testing."""
    if not HAS_ADVERSARIAL_HARNESS:
        # Return synthetic plan if harness unavailable
        readiness = {
            "promotion_ok": False,
            "metrics_blocking_promotion": ["multi_goal"],
            "status": "BLOCK",
            "reasons": ["multi_goal: Core metric has no scenario coverage - blocking promotion"],
        }
        failover_plan = {
            "has_failover": False,
            "metrics_without_failover": ["multi_goal"],
            "status": "BLOCK",
            "recommendations": ["multi_goal: Core metric has no scenario coverage - blocking promotion"],
        }
    else:
        readiness = evaluate_adversarial_readiness_for_promotion(synthetic_coverage_index)
        failover_plan = build_adversarial_failover_plan(synthetic_coverage_index, readiness)
    
    if not HAS_ADVERSARIAL_HARNESS:
        return {
            "scenario_backlog": [
                {
                    "name": "multi_goal_critical_failover",
                    "profile": "full",
                    "metric_kinds": ["multi_goal"],
                    "modes": ["fault", "mutation", "replay"],
                    "priority": 1,
                    "rationale": "Core metric multi_goal blocking promotion with no failover coverage",
                }
            ],
            "priority_order": ["multi_goal_critical_failover"],
            "multi_metric_scenarios": [],
            "neutral_rationale": ["Priority 1: multi_goal requires immediate failover scenario"],
        }
    
    return build_evolving_adversarial_scenario_plan(synthetic_pressure_model, failover_plan, readiness)


@pytest.fixture
def synthetic_failover_plan_v2(synthetic_coverage_index, synthetic_robustness_radar) -> Dict[str, Any]:
    """Synthetic failover plan v2 for testing."""
    if not HAS_ADVERSARIAL_HARNESS:
        # Return synthetic plan v2 if harness unavailable
        return {
            "has_failover": False,
            "metrics_without_failover": ["multi_goal"],
            "status": "BLOCK",
            "recommendations": ["multi_goal: Core metric has no scenario coverage - blocking promotion"],
            "v2_metrics": {
                "goal_hit": {
                    "redundancy_depth": 3,
                    "scenario_diversity": 0.6,
                    "failure_case_sensitivity": 0.9,
                },
                "density": {
                    "redundancy_depth": 2,
                    "scenario_diversity": 0.4,
                    "failure_case_sensitivity": 0.3,
                },
                "chain_length": {
                    "redundancy_depth": 1,
                    "scenario_diversity": 0.1,
                    "failure_case_sensitivity": 0.3,
                },
                "multi_goal": {
                    "redundancy_depth": 0,
                    "scenario_diversity": 0.0,
                    "failure_case_sensitivity": 0.5,
                },
            },
            "v2_aggregates": {
                "average_redundancy_depth": 1.5,
                "average_scenario_diversity": 0.275,
                "average_failure_sensitivity": 0.5,
            },
        }
    
    readiness = evaluate_adversarial_readiness_for_promotion(synthetic_coverage_index)
    return build_adversarial_failover_plan_v2(synthetic_coverage_index, readiness, synthetic_robustness_radar)


@pytest.mark.hermetic
class TestAdversarialGovernanceTileSerialization:
    """Tests for adversarial governance tile serialization."""

    def test_tile_builds_from_synthetic_data(
        self,
        synthetic_pressure_model,
        synthetic_scenario_plan,
        synthetic_failover_plan_v2,
    ):
        """Tile builds successfully from synthetic data."""
        tile = build_adversarial_governance_tile(
            pressure_model=synthetic_pressure_model,
            scenario_plan=synthetic_scenario_plan,
            failover_plan_v2=synthetic_failover_plan_v2,
        )
        
        assert isinstance(tile, dict)
        assert "schema_version" in tile
        assert tile["schema_version"] == ADVERSARIAL_GOVERNANCE_TILE_SCHEMA_VERSION

    def test_tile_is_json_serializable(
        self,
        synthetic_pressure_model,
        synthetic_scenario_plan,
        synthetic_failover_plan_v2,
    ):
        """Tile is JSON-serializable."""
        tile = build_adversarial_governance_tile(
            pressure_model=synthetic_pressure_model,
            scenario_plan=synthetic_scenario_plan,
            failover_plan_v2=synthetic_failover_plan_v2,
        )
        
        # Should not raise
        json_str = json.dumps(tile, sort_keys=True)
        assert isinstance(json_str, str)
        
        # Should be able to parse back
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert parsed["schema_version"] == ADVERSARIAL_GOVERNANCE_TILE_SCHEMA_VERSION

    def test_tile_has_required_fields(
        self,
        synthetic_pressure_model,
        synthetic_scenario_plan,
        synthetic_failover_plan_v2,
    ):
        """Tile has all required fields."""
        tile = build_adversarial_governance_tile(
            pressure_model=synthetic_pressure_model,
            scenario_plan=synthetic_scenario_plan,
            failover_plan_v2=synthetic_failover_plan_v2,
        )
        
        required_fields = [
            "schema_version",
            "status_light",
            "global_pressure_index",
            "pressure_band",
            "priority_scenarios",
            "has_failover",
            "metrics_without_failover",
            "headline",
        ]
        
        for field in required_fields:
            assert field in tile, f"Missing required field: {field}"

    def test_tile_status_light_valid(
        self,
        synthetic_pressure_model,
        synthetic_scenario_plan,
        synthetic_failover_plan_v2,
    ):
        """Tile status_light is valid (GREEN/YELLOW/RED)."""
        tile = build_adversarial_governance_tile(
            pressure_model=synthetic_pressure_model,
            scenario_plan=synthetic_scenario_plan,
            failover_plan_v2=synthetic_failover_plan_v2,
        )
        
        assert tile["status_light"] in ("GREEN", "YELLOW", "RED")

    def test_tile_pressure_band_valid(
        self,
        synthetic_pressure_model,
        synthetic_scenario_plan,
        synthetic_failover_plan_v2,
    ):
        """Tile pressure_band is valid (LOW/MEDIUM/HIGH)."""
        tile = build_adversarial_governance_tile(
            pressure_model=synthetic_pressure_model,
            scenario_plan=synthetic_scenario_plan,
            failover_plan_v2=synthetic_failover_plan_v2,
        )
        
        assert tile["pressure_band"] in ("LOW", "MEDIUM", "HIGH")

    def test_tile_is_deterministic(
        self,
        synthetic_pressure_model,
        synthetic_scenario_plan,
        synthetic_failover_plan_v2,
    ):
        """Tile is deterministic (same inputs → same output)."""
        tile1 = build_adversarial_governance_tile(
            pressure_model=synthetic_pressure_model,
            scenario_plan=synthetic_scenario_plan,
            failover_plan_v2=synthetic_failover_plan_v2,
        )
        tile2 = build_adversarial_governance_tile(
            pressure_model=synthetic_pressure_model,
            scenario_plan=synthetic_scenario_plan,
            failover_plan_v2=synthetic_failover_plan_v2,
        )
        
        json1 = json.dumps(tile1, sort_keys=True)
        json2 = json.dumps(tile2, sort_keys=True)
        assert json1 == json2

    def test_tile_priority_scenarios_max_3(
        self,
        synthetic_pressure_model,
        synthetic_scenario_plan,
        synthetic_failover_plan_v2,
    ):
        """Tile priority_scenarios contains at most 3 scenarios."""
        tile = build_adversarial_governance_tile(
            pressure_model=synthetic_pressure_model,
            scenario_plan=synthetic_scenario_plan,
            failover_plan_v2=synthetic_failover_plan_v2,
        )
        
        assert isinstance(tile["priority_scenarios"], list)
        assert len(tile["priority_scenarios"]) <= 3

    def test_tile_red_status_for_block(
        self,
        synthetic_pressure_model,
        synthetic_scenario_plan,
        synthetic_failover_plan_v2,
    ):
        """Tile status_light is RED when status is BLOCK."""
        # Ensure failover plan has BLOCK status
        failover_plan_block = dict(synthetic_failover_plan_v2)
        failover_plan_block["status"] = "BLOCK"
        
        tile = build_adversarial_governance_tile(
            pressure_model=synthetic_pressure_model,
            scenario_plan=synthetic_scenario_plan,
            failover_plan_v2=failover_plan_block,
        )
        
        assert tile["status_light"] == "RED"

    def test_tile_yellow_status_for_warn(
        self,
        synthetic_pressure_model,
        synthetic_scenario_plan,
        synthetic_failover_plan_v2,
    ):
        """Tile status_light is YELLOW when status is WARN."""
        # Ensure failover plan has WARN status
        failover_plan_warn = dict(synthetic_failover_plan_v2)
        failover_plan_warn["status"] = "WARN"
        
        tile = build_adversarial_governance_tile(
            pressure_model=synthetic_pressure_model,
            scenario_plan=synthetic_scenario_plan,
            failover_plan_v2=failover_plan_warn,
        )
        
        assert tile["status_light"] == "YELLOW"

    def test_release_signal_extracts(
        self,
        synthetic_pressure_model,
        synthetic_failover_plan_v2,
    ):
        """Release signal extracts correctly."""
        signal = extract_adversarial_signal_for_release(
            pressure_model=synthetic_pressure_model,
            failover_plan_v2=synthetic_failover_plan_v2,
        )
        
        assert isinstance(signal, dict)
        assert "global_pressure_index" in signal
        assert "pressure_band" in signal
        assert "has_failover" in signal
        assert "metrics_without_failover" in signal

    def test_release_signal_is_json_serializable(
        self,
        synthetic_pressure_model,
        synthetic_failover_plan_v2,
    ):
        """Release signal is JSON-serializable."""
        signal = extract_adversarial_signal_for_release(
            pressure_model=synthetic_pressure_model,
            failover_plan_v2=synthetic_failover_plan_v2,
        )
        
        # Should not raise
        json_str = json.dumps(signal, sort_keys=True)
        assert isinstance(json_str, str)
        
        # Should be able to parse back
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_release_signal_is_deterministic(
        self,
        synthetic_pressure_model,
        synthetic_failover_plan_v2,
    ):
        """Release signal is deterministic."""
        signal1 = extract_adversarial_signal_for_release(
            pressure_model=synthetic_pressure_model,
            failover_plan_v2=synthetic_failover_plan_v2,
        )
        signal2 = extract_adversarial_signal_for_release(
            pressure_model=synthetic_pressure_model,
            failover_plan_v2=synthetic_failover_plan_v2,
        )
        
        json1 = json.dumps(signal1, sort_keys=True)
        json2 = json.dumps(signal2, sort_keys=True)
        assert json1 == json2

    def test_attach_to_evidence_non_mutating(
        self,
        synthetic_pressure_model,
        synthetic_scenario_plan,
        synthetic_failover_plan_v2,
    ):
        """attach_adversarial_governance_to_evidence does not mutate input."""
        evidence = {"timestamp": "2024-01-01", "data": {"test": "value"}}
        tile = build_adversarial_governance_tile(
            pressure_model=synthetic_pressure_model,
            scenario_plan=synthetic_scenario_plan,
            failover_plan_v2=synthetic_failover_plan_v2,
        )
        signal = extract_adversarial_signal_for_release(
            pressure_model=synthetic_pressure_model,
            failover_plan_v2=synthetic_failover_plan_v2,
        )
        
        original_evidence = evidence.copy()
        enriched = attach_adversarial_governance_to_evidence(evidence, tile, signal)
        
        # Original should be unchanged
        assert evidence == original_evidence
        # Enriched should have governance key
        assert "governance" in enriched
        assert "adversarial" in enriched["governance"]

    def test_attach_to_evidence_has_required_fields(
        self,
        synthetic_pressure_model,
        synthetic_scenario_plan,
        synthetic_failover_plan_v2,
    ):
        """Attached evidence has required adversarial fields."""
        evidence = {"timestamp": "2024-01-01"}
        tile = build_adversarial_governance_tile(
            pressure_model=synthetic_pressure_model,
            scenario_plan=synthetic_scenario_plan,
            failover_plan_v2=synthetic_failover_plan_v2,
        )
        signal = extract_adversarial_signal_for_release(
            pressure_model=synthetic_pressure_model,
            failover_plan_v2=synthetic_failover_plan_v2,
        )
        
        enriched = attach_adversarial_governance_to_evidence(evidence, tile, signal)
        adversarial = enriched["governance"]["adversarial"]
        
        assert "global_pressure_index" in adversarial
        assert "pressure_band" in adversarial
        assert "priority_scenarios" in adversarial
        assert "has_failover" in adversarial
        assert "metrics_without_failover" in adversarial

    def test_attach_to_evidence_deterministic(
        self,
        synthetic_pressure_model,
        synthetic_scenario_plan,
        synthetic_failover_plan_v2,
    ):
        """attach_adversarial_governance_to_evidence is deterministic."""
        evidence = {"timestamp": "2024-01-01"}
        tile = build_adversarial_governance_tile(
            pressure_model=synthetic_pressure_model,
            scenario_plan=synthetic_scenario_plan,
            failover_plan_v2=synthetic_failover_plan_v2,
        )
        signal = extract_adversarial_signal_for_release(
            pressure_model=synthetic_pressure_model,
            failover_plan_v2=synthetic_failover_plan_v2,
        )
        
        e1 = attach_adversarial_governance_to_evidence(evidence, tile, signal)
        e2 = attach_adversarial_governance_to_evidence(evidence, tile, signal)
        
        json1 = json.dumps(e1, sort_keys=True)
        json2 = json.dumps(e2, sort_keys=True)
        assert json1 == json2

    def test_uplift_council_summary_has_status(
        self,
        synthetic_pressure_model,
        synthetic_scenario_plan,
        synthetic_failover_plan_v2,
    ):
        """Uplift council summary has status field."""
        tile = build_adversarial_governance_tile(
            pressure_model=synthetic_pressure_model,
            scenario_plan=synthetic_scenario_plan,
            failover_plan_v2=synthetic_failover_plan_v2,
        )
        
        summary = summarize_adversarial_for_uplift_council(tile)
        
        assert "status" in summary
        assert summary["status"] in ("OK", "WARN", "BLOCK")

    def test_uplift_council_summary_blocks_high_pressure(
        self,
        synthetic_pressure_model,
        synthetic_scenario_plan,
        synthetic_failover_plan_v2,
    ):
        """Uplift council summary blocks HIGH pressure."""
        # Create tile with HIGH pressure
        high_pressure_model = dict(synthetic_pressure_model)
        high_pressure_model["pressure_band"] = "HIGH"
        high_pressure_model["global_pressure_index"] = 0.8
        
        tile = build_adversarial_governance_tile(
            pressure_model=high_pressure_model,
            scenario_plan=synthetic_scenario_plan,
            failover_plan_v2=synthetic_failover_plan_v2,
        )
        
        summary = summarize_adversarial_for_uplift_council(tile)
        
        assert summary["status"] == "BLOCK"

    def test_p3_stability_summary_extracts(
        self,
        synthetic_pressure_model,
        synthetic_scenario_plan,
        synthetic_failover_plan_v2,
    ):
        """P3 stability summary extracts correctly."""
        tile = build_adversarial_governance_tile(
            pressure_model=synthetic_pressure_model,
            scenario_plan=synthetic_scenario_plan,
            failover_plan_v2=synthetic_failover_plan_v2,
        )
        
        summary = extract_adversarial_summary_for_p3_stability(tile)
        
        assert "global_pressure_index" in summary
        assert "pressure_band" in summary
        assert "status_light" in summary
        assert "priority_scenarios" in summary
        assert "has_failover" in summary
        assert "metrics_without_failover" in summary

    def test_p4_calibration_extracts(
        self,
        synthetic_pressure_model,
        synthetic_failover_plan_v2,
    ):
        """P4 calibration extracts correctly."""
        tile = {
            "global_pressure_index": 0.5,
            "pressure_band": "MEDIUM",
            "status_light": "YELLOW",
            "priority_scenarios": ["scenario1"],
            "has_failover": True,
            "metrics_without_failover": [],
        }
        signal = extract_adversarial_signal_for_release(
            pressure_model=synthetic_pressure_model,
            failover_plan_v2=synthetic_failover_plan_v2,
        )
        
        calibration = extract_adversarial_calibration_for_p4(tile, signal)
        
        assert "global_pressure_index" in calibration
        assert "pressure_band" in calibration
        assert "has_failover" in calibration
        assert "metrics_without_failover" in calibration
        assert "notes" in calibration
        assert isinstance(calibration["notes"], list)

    def test_coverage_annex_builds(
        self,
        synthetic_pressure_model,
        synthetic_scenario_plan,
        synthetic_failover_plan_v2,
    ):
        """Coverage annex builds from P3 and P4 data."""
        tile = build_adversarial_governance_tile(
            pressure_model=synthetic_pressure_model,
            scenario_plan=synthetic_scenario_plan,
            failover_plan_v2=synthetic_failover_plan_v2,
        )
        
        p3_summary = extract_adversarial_summary_for_p3_stability(tile)
        signal = extract_adversarial_signal_for_release(
            pressure_model=synthetic_pressure_model,
            failover_plan_v2=synthetic_failover_plan_v2,
        )
        p4_calibration = extract_adversarial_calibration_for_p4(tile, signal)
        
        annex = build_first_light_adversarial_coverage_annex(p3_summary, p4_calibration)
        
        assert "schema_version" in annex
        assert "p3_pressure_band" in annex
        assert "p4_pressure_band" in annex
        assert "priority_scenarios" in annex
        assert "has_failover" in annex

    def test_coverage_annex_is_json_serializable(
        self,
        synthetic_pressure_model,
        synthetic_scenario_plan,
        synthetic_failover_plan_v2,
    ):
        """Coverage annex is JSON-serializable."""
        tile = build_adversarial_governance_tile(
            pressure_model=synthetic_pressure_model,
            scenario_plan=synthetic_scenario_plan,
            failover_plan_v2=synthetic_failover_plan_v2,
        )
        
        p3_summary = extract_adversarial_summary_for_p3_stability(tile)
        signal = extract_adversarial_signal_for_release(
            pressure_model=synthetic_pressure_model,
            failover_plan_v2=synthetic_failover_plan_v2,
        )
        p4_calibration = extract_adversarial_calibration_for_p4(tile, signal)
        
        annex = build_first_light_adversarial_coverage_annex(p3_summary, p4_calibration)
        
        # Should not raise
        json_str = json.dumps(annex, sort_keys=True)
        assert isinstance(json_str, str)
        
        # Should be able to parse back
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_coverage_annex_is_deterministic(
        self,
        synthetic_pressure_model,
        synthetic_scenario_plan,
        synthetic_failover_plan_v2,
    ):
        """Coverage annex is deterministic."""
        tile = build_adversarial_governance_tile(
            pressure_model=synthetic_pressure_model,
            scenario_plan=synthetic_scenario_plan,
            failover_plan_v2=synthetic_failover_plan_v2,
        )
        
        p3_summary = extract_adversarial_summary_for_p3_stability(tile)
        signal = extract_adversarial_signal_for_release(
            pressure_model=synthetic_pressure_model,
            failover_plan_v2=synthetic_failover_plan_v2,
        )
        p4_calibration = extract_adversarial_calibration_for_p4(tile, signal)
        
        a1 = build_first_light_adversarial_coverage_annex(p3_summary, p4_calibration)
        a2 = build_first_light_adversarial_coverage_annex(p3_summary, p4_calibration)
        
        json1 = json.dumps(a1, sort_keys=True)
        json2 = json.dumps(a2, sort_keys=True)
        assert json1 == json2

    def test_evidence_attach_with_coverage_annex_non_mutating(
        self,
        synthetic_pressure_model,
        synthetic_scenario_plan,
        synthetic_failover_plan_v2,
    ):
        """Evidence attach with coverage annex does not mutate input."""
        evidence = {"timestamp": "2024-01-01", "data": {"test": "value"}}
        tile = build_adversarial_governance_tile(
            pressure_model=synthetic_pressure_model,
            scenario_plan=synthetic_scenario_plan,
            failover_plan_v2=synthetic_failover_plan_v2,
        )
        signal = extract_adversarial_signal_for_release(
            pressure_model=synthetic_pressure_model,
            failover_plan_v2=synthetic_failover_plan_v2,
        )
        p3_summary = extract_adversarial_summary_for_p3_stability(tile)
        p4_calibration = extract_adversarial_calibration_for_p4(tile, signal)
        
        original_evidence = evidence.copy()
        enriched = attach_adversarial_governance_to_evidence(
            evidence, tile, signal, p3_summary=p3_summary, p4_calibration=p4_calibration
        )
        
        # Original should be unchanged
        assert evidence == original_evidence
        # Enriched should have coverage annex
        assert "first_light_coverage" in enriched["governance"]["adversarial"]

    def test_council_blocks_high_pressure_and_missing_failover(
        self,
        synthetic_pressure_model,
        synthetic_scenario_plan,
    ):
        """Council summary blocks HIGH pressure AND missing failover for core metrics."""
        # Create failover plan with core metrics missing failover
        failover_plan_v2 = {
            "has_failover": False,
            "status": "BLOCK",
            "metrics_without_failover": ["goal_hit", "density"],  # Core metrics
        }
        
        # Create HIGH pressure model
        high_pressure_model = dict(synthetic_pressure_model)
        high_pressure_model["pressure_band"] = "HIGH"
        high_pressure_model["global_pressure_index"] = 0.8
        
        tile = build_adversarial_governance_tile(
            pressure_model=high_pressure_model,
            scenario_plan=synthetic_scenario_plan,
            failover_plan_v2=failover_plan_v2,
        )
        
        summary = summarize_adversarial_for_uplift_council(tile)
        
        # Should BLOCK: HIGH pressure AND core metrics lack failover
        assert summary["status"] == "BLOCK"
        assert len(summary["core_metrics_without_failover"]) > 0

    def test_council_blocks_high_pressure_alone(
        self,
        synthetic_pressure_model,
        synthetic_scenario_plan,
        synthetic_failover_plan_v2,
    ):
        """Council summary blocks HIGH pressure even if failover exists."""
        # Create HIGH pressure model
        high_pressure_model = dict(synthetic_pressure_model)
        high_pressure_model["pressure_band"] = "HIGH"
        high_pressure_model["global_pressure_index"] = 0.8
        
        tile = build_adversarial_governance_tile(
            pressure_model=high_pressure_model,
            scenario_plan=synthetic_scenario_plan,
            failover_plan_v2=synthetic_failover_plan_v2,
        )
        
        summary = summarize_adversarial_for_uplift_council(tile)
        
        # Should BLOCK: HIGH pressure alone
        assert summary["status"] == "BLOCK"

