"""
CI tests for uplift safety governance tile serialization and determinism.

PHASE X — UPLIFT SAFETY GOVERNANCE TILE

Tests that the uplift safety governance tile:
  - Serializes correctly to JSON
  - Is deterministic (same inputs → same outputs)
  - Maps decisions to status lights correctly
  - Uses neutral language in all text fields
  - Handles missing/optional fields gracefully

SHADOW MODE CONTRACT:
  - These tests verify observational behavior only
  - No control flow dependencies
  - No deployment blocking logic
"""

import json
import pytest
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.health.uplift_safety_adapter import (
    build_uplift_safety_governance_tile,
    extract_uplift_safety_signal_for_first_light,
    build_p3_uplift_safety_summary,
    build_p4_uplift_safety_calibration,
    build_first_light_uplift_gate_annex,
    export_gate_annex_per_experiment,
    build_gate_alignment_panel,
    extract_uplift_gate_alignment_signal,
    attach_uplift_safety_to_evidence,
    summarize_uplift_safety_for_council,
    StatusLight,
    UPLIFT_SAFETY_GOVERNANCE_TILE_SCHEMA_VERSION,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES: Synthetic safety tensor, forecaster, and gate decision
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def synthetic_safety_tensor_pass() -> Dict[str, Any]:
    """Synthetic safety tensor with LOW risk (PASS scenario)."""
    return {
        "schema_version": "1.0.0",
        "tensor_norm": 0.5,
        "uplift_risk_band": "LOW",
        "hotspot_axes": [],
        "risk_indicators": {
            "epistemic_uncertainty": 0.2,
            "drift_risk": 0.3,
            "atlas_risk": 0.2,
            "telemetry_risk": 0.1,
        },
        "neutral_notes": ["Tensor norm: 0.500", "Risk band: LOW"],
    }


@pytest.fixture
def synthetic_safety_tensor_warn() -> Dict[str, Any]:
    """Synthetic safety tensor with MEDIUM risk (WARN scenario)."""
    return {
        "schema_version": "1.0.0",
        "tensor_norm": 1.2,
        "uplift_risk_band": "MEDIUM",
        "hotspot_axes": ["epistemic_uncertainty", "drift_risk"],
        "risk_indicators": {
            "epistemic_uncertainty": 0.6,
            "drift_risk": 0.5,
            "atlas_risk": 0.4,
            "telemetry_risk": 0.3,
        },
        "neutral_notes": ["Tensor norm: 1.200", "Risk band: MEDIUM"],
    }


@pytest.fixture
def synthetic_safety_tensor_block() -> Dict[str, Any]:
    """Synthetic safety tensor with HIGH risk (BLOCK scenario)."""
    return {
        "schema_version": "1.0.0",
        "tensor_norm": 2.5,
        "uplift_risk_band": "HIGH",
        "hotspot_axes": ["epistemic_uncertainty", "drift_risk", "atlas_risk"],
        "risk_indicators": {
            "epistemic_uncertainty": 0.9,
            "drift_risk": 0.8,
            "atlas_risk": 0.7,
            "telemetry_risk": 0.6,
        },
        "neutral_notes": ["Tensor norm: 2.500", "Risk band: HIGH"],
    }


@pytest.fixture
def synthetic_stability_forecaster_stable() -> Dict[str, Any]:
    """Synthetic stability forecaster with STABLE status."""
    return {
        "schema_version": "1.0.0",
        "current_stability": "STABLE",
        "stability_trend": "STABLE",
        "instability_prediction": {
            "schema_version": "1.0.0",
            "predicted_instability_cycles": [],
            "predicted_instability_days": [],
            "predicted_instability_versions": [],
            "confidence": 0.0,
            "neutral_notes": ["No instability windows predicted"],
        },
        "neutral_notes": ["Current stability: STABLE", "Stability trend: STABLE"],
    }


@pytest.fixture
def synthetic_stability_forecaster_degrading() -> Dict[str, Any]:
    """Synthetic stability forecaster with DEGRADING status."""
    return {
        "schema_version": "1.0.0",
        "current_stability": "DEGRADING",
        "stability_trend": "DEGRADING",
        "instability_prediction": {
            "schema_version": "1.0.0",
            "predicted_instability_cycles": [],
            "predicted_instability_days": [],
            "predicted_instability_versions": [],
            "confidence": 0.5,
            "neutral_notes": ["Stability trend indicates degradation"],
        },
        "neutral_notes": ["Current stability: DEGRADING", "Stability trend: DEGRADING"],
    }


@pytest.fixture
def synthetic_stability_forecaster_unstable() -> Dict[str, Any]:
    """Synthetic stability forecaster with UNSTABLE status and predicted instability."""
    return {
        "schema_version": "1.0.0",
        "current_stability": "UNSTABLE",
        "stability_trend": "DEGRADING",
        "instability_prediction": {
            "schema_version": "1.0.0",
            "predicted_instability_cycles": [11, 12, 13],
            "predicted_instability_days": [11.0, 12.0, 13.0],
            "predicted_instability_versions": [11, 12, 13],
            "confidence": 0.8,
            "neutral_notes": ["Instability predicted in cycles: [11, 12, 13]"],
        },
        "neutral_notes": ["Current stability: UNSTABLE", "Stability trend: DEGRADING"],
    }


@pytest.fixture
def synthetic_gate_decision_pass() -> Dict[str, Any]:
    """Synthetic gate decision with PASS."""
    return {
        "schema_version": "1.0.0",
        "gate_version": "v3",
        "uplift_safety_decision": "PASS",
        "decision_rationale": ["All safety indicators within acceptable ranges"],
        "risk_band": "LOW",
        "stability_status": "STABLE",
        "stability_trend": "STABLE",
        "neutral_notes": ["Uplift safety decision: PASS"],
    }


@pytest.fixture
def synthetic_gate_decision_warn() -> Dict[str, Any]:
    """Synthetic gate decision with WARN."""
    return {
        "schema_version": "1.0.0",
        "gate_version": "v3",
        "uplift_safety_decision": "WARN",
        "decision_rationale": ["Risk band is MEDIUM", "Stability trend is DEGRADING"],
        "risk_band": "MEDIUM",
        "stability_status": "DEGRADING",
        "stability_trend": "DEGRADING",
        "neutral_notes": ["Uplift safety decision: WARN"],
    }


@pytest.fixture
def synthetic_gate_decision_block() -> Dict[str, Any]:
    """Synthetic gate decision with BLOCK."""
    return {
        "schema_version": "1.0.0",
        "gate_version": "v3",
        "uplift_safety_decision": "BLOCK",
        "decision_rationale": [
            "Risk band is HIGH",
            "Current stability is UNSTABLE",
            "Instability predicted in cycles: [11, 12, 13]",
        ],
        "risk_band": "HIGH",
        "stability_status": "UNSTABLE",
        "stability_trend": "DEGRADING",
        "neutral_notes": ["Uplift safety decision: BLOCK"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS: Governance Tile Serialization and Schema
# ═══════════════════════════════════════════════════════════════════════════════

class TestGovernanceTileSerialization:
    """Tests for governance tile serialization and schema."""

    def test_001_tile_schema_complete(
        self,
        synthetic_safety_tensor_pass,
        synthetic_stability_forecaster_stable,
        synthetic_gate_decision_pass,
    ):
        """build_uplift_safety_governance_tile returns complete schema."""
        tile = build_uplift_safety_governance_tile(
            safety_tensor=synthetic_safety_tensor_pass,
            stability_forecaster=synthetic_stability_forecaster_stable,
            gate_decision=synthetic_gate_decision_pass,
        )

        assert "status_light" in tile
        assert "uplift_safety_decision" in tile
        assert "risk_band" in tile
        assert "tensor_norm" in tile
        assert "current_stability" in tile
        assert "stability_trend" in tile
        assert "instability_prediction" in tile
        assert "decision_rationale" in tile
        assert "headline" in tile
        assert "schema_version" in tile

    def test_002_tile_json_serializable(
        self,
        synthetic_safety_tensor_pass,
        synthetic_stability_forecaster_stable,
        synthetic_gate_decision_pass,
    ):
        """build_uplift_safety_governance_tile output is JSON serializable."""
        tile = build_uplift_safety_governance_tile(
            safety_tensor=synthetic_safety_tensor_pass,
            stability_forecaster=synthetic_stability_forecaster_stable,
            gate_decision=synthetic_gate_decision_pass,
        )

        # Should not raise
        json_str = json.dumps(tile)
        assert isinstance(json_str, str)
        assert len(json_str) > 0

        # Should round-trip
        tile_roundtrip = json.loads(json_str)
        assert tile_roundtrip == tile

    def test_003_tile_schema_version(
        self,
        synthetic_safety_tensor_pass,
        synthetic_stability_forecaster_stable,
        synthetic_gate_decision_pass,
    ):
        """build_uplift_safety_governance_tile uses correct schema version."""
        tile = build_uplift_safety_governance_tile(
            safety_tensor=synthetic_safety_tensor_pass,
            stability_forecaster=synthetic_stability_forecaster_stable,
            gate_decision=synthetic_gate_decision_pass,
        )

        assert tile["schema_version"] == UPLIFT_SAFETY_GOVERNANCE_TILE_SCHEMA_VERSION

    def test_004_tile_deterministic(
        self,
        synthetic_safety_tensor_pass,
        synthetic_stability_forecaster_stable,
        synthetic_gate_decision_pass,
    ):
        """build_uplift_safety_governance_tile is deterministic."""
        tile1 = build_uplift_safety_governance_tile(
            safety_tensor=synthetic_safety_tensor_pass,
            stability_forecaster=synthetic_stability_forecaster_stable,
            gate_decision=synthetic_gate_decision_pass,
        )
        tile2 = build_uplift_safety_governance_tile(
            safety_tensor=synthetic_safety_tensor_pass,
            stability_forecaster=synthetic_stability_forecaster_stable,
            gate_decision=synthetic_gate_decision_pass,
        )

        assert tile1 == tile2

    def test_005_tile_status_light_pass(
        self,
        synthetic_safety_tensor_pass,
        synthetic_stability_forecaster_stable,
        synthetic_gate_decision_pass,
    ):
        """build_uplift_safety_governance_tile maps PASS → GREEN."""
        tile = build_uplift_safety_governance_tile(
            safety_tensor=synthetic_safety_tensor_pass,
            stability_forecaster=synthetic_stability_forecaster_stable,
            gate_decision=synthetic_gate_decision_pass,
        )

        assert tile["status_light"] == StatusLight.GREEN
        assert tile["uplift_safety_decision"] == "PASS"

    def test_006_tile_status_light_warn(
        self,
        synthetic_safety_tensor_warn,
        synthetic_stability_forecaster_degrading,
        synthetic_gate_decision_warn,
    ):
        """build_uplift_safety_governance_tile maps WARN → YELLOW."""
        tile = build_uplift_safety_governance_tile(
            safety_tensor=synthetic_safety_tensor_warn,
            stability_forecaster=synthetic_stability_forecaster_degrading,
            gate_decision=synthetic_gate_decision_warn,
        )

        assert tile["status_light"] == StatusLight.YELLOW
        assert tile["uplift_safety_decision"] == "WARN"

    def test_007_tile_status_light_block(
        self,
        synthetic_safety_tensor_block,
        synthetic_stability_forecaster_unstable,
        synthetic_gate_decision_block,
    ):
        """build_uplift_safety_governance_tile maps BLOCK → RED."""
        tile = build_uplift_safety_governance_tile(
            safety_tensor=synthetic_safety_tensor_block,
            stability_forecaster=synthetic_stability_forecaster_unstable,
            gate_decision=synthetic_gate_decision_block,
        )

        assert tile["status_light"] == StatusLight.RED
        assert tile["uplift_safety_decision"] == "BLOCK"

    def test_008_tile_neutral_language_headline(
        self,
        synthetic_safety_tensor_pass,
        synthetic_stability_forecaster_stable,
        synthetic_gate_decision_pass,
    ):
        """build_uplift_safety_governance_tile uses neutral language in headline."""
        tile = build_uplift_safety_governance_tile(
            safety_tensor=synthetic_safety_tensor_pass,
            stability_forecaster=synthetic_stability_forecaster_stable,
            gate_decision=synthetic_gate_decision_pass,
        )

        headline = tile["headline"]
        assert isinstance(headline, str)
        assert len(headline) > 0

        # Check for neutral language (no evaluative terms)
        evaluative_terms = ["good", "bad", "healthy", "unhealthy", "excellent", "poor"]
        headline_lower = headline.lower()
        for term in evaluative_terms:
            assert term not in headline_lower, f"Headline contains evaluative term: {term}"

    def test_009_tile_neutral_language_rationale(
        self,
        synthetic_safety_tensor_pass,
        synthetic_stability_forecaster_stable,
        synthetic_gate_decision_pass,
    ):
        """build_uplift_safety_governance_tile uses neutral language in rationale."""
        tile = build_uplift_safety_governance_tile(
            safety_tensor=synthetic_safety_tensor_pass,
            stability_forecaster=synthetic_stability_forecaster_stable,
            gate_decision=synthetic_gate_decision_pass,
        )

        rationale = tile["decision_rationale"]
        assert isinstance(rationale, list)

        # Check for neutral language in rationale items
        evaluative_terms = ["good", "bad", "healthy", "unhealthy", "excellent", "poor"]
        for item in rationale:
            assert isinstance(item, str)
            item_lower = item.lower()
            for term in evaluative_terms:
                assert term not in item_lower, f"Rationale contains evaluative term: {term}"

    def test_010_tile_missing_fields_handled(self):
        """build_uplift_safety_governance_tile handles missing fields gracefully."""
        # Minimal inputs with missing fields
        safety_tensor = {"uplift_risk_band": "LOW"}
        stability_forecaster = {"current_stability": "STABLE"}
        gate_decision = {"uplift_safety_decision": "PASS"}

        # Should not raise
        tile = build_uplift_safety_governance_tile(
            safety_tensor=safety_tensor,
            stability_forecaster=stability_forecaster,
            gate_decision=gate_decision,
        )

        assert tile["status_light"] in (StatusLight.GREEN, StatusLight.YELLOW, StatusLight.RED)
        assert tile["uplift_safety_decision"] == "PASS"


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS: First Light Signal Extraction
# ═══════════════════════════════════════════════════════════════════════════════

class TestFirstLightSignalExtraction:
    """Tests for First Light signal extraction."""

    def test_011_first_light_signal_schema(
        self,
        synthetic_safety_tensor_pass,
        synthetic_stability_forecaster_stable,
        synthetic_gate_decision_pass,
    ):
        """extract_uplift_safety_signal_for_first_light returns correct schema."""
        signal = extract_uplift_safety_signal_for_first_light(
            safety_tensor=synthetic_safety_tensor_pass,
            stability_forecaster=synthetic_stability_forecaster_stable,
            gate_decision=synthetic_gate_decision_pass,
        )

        assert "risk_band" in signal
        assert "uplift_safety_decision" in signal
        assert "current_stability" in signal
        assert "stability_trend" in signal

    def test_012_first_light_signal_with_predicted_cycles(
        self,
        synthetic_safety_tensor_block,
        synthetic_stability_forecaster_unstable,
        synthetic_gate_decision_block,
    ):
        """extract_uplift_safety_signal_for_first_light includes predicted cycles if present."""
        signal = extract_uplift_safety_signal_for_first_light(
            safety_tensor=synthetic_safety_tensor_block,
            stability_forecaster=synthetic_stability_forecaster_unstable,
            gate_decision=synthetic_gate_decision_block,
        )

        assert "predicted_instability_cycles" in signal
        assert signal["predicted_instability_cycles"] == [11, 12, 13]

    def test_013_first_light_signal_without_predicted_cycles(
        self,
        synthetic_safety_tensor_pass,
        synthetic_stability_forecaster_stable,
        synthetic_gate_decision_pass,
    ):
        """extract_uplift_safety_signal_for_first_light omits predicted cycles if empty."""
        signal = extract_uplift_safety_signal_for_first_light(
            safety_tensor=synthetic_safety_tensor_pass,
            stability_forecaster=synthetic_stability_forecaster_stable,
            gate_decision=synthetic_gate_decision_pass,
        )

        assert "predicted_instability_cycles" not in signal

    def test_014_first_light_signal_json_serializable(
        self,
        synthetic_safety_tensor_pass,
        synthetic_stability_forecaster_stable,
        synthetic_gate_decision_pass,
    ):
        """extract_uplift_safety_signal_for_first_light output is JSON serializable."""
        signal = extract_uplift_safety_signal_for_first_light(
            safety_tensor=synthetic_safety_tensor_pass,
            stability_forecaster=synthetic_stability_forecaster_stable,
            gate_decision=synthetic_gate_decision_pass,
        )

        # Should not raise
        json_str = json.dumps(signal)
        assert isinstance(json_str, str)

        # Should round-trip
        signal_roundtrip = json.loads(json_str)
        assert signal_roundtrip == signal

    def test_015_first_light_signal_deterministic(
        self,
        synthetic_safety_tensor_pass,
        synthetic_stability_forecaster_stable,
        synthetic_gate_decision_pass,
    ):
        """extract_uplift_safety_signal_for_first_light is deterministic."""
        signal1 = extract_uplift_safety_signal_for_first_light(
            safety_tensor=synthetic_safety_tensor_pass,
            stability_forecaster=synthetic_stability_forecaster_stable,
            gate_decision=synthetic_gate_decision_pass,
        )
        signal2 = extract_uplift_safety_signal_for_first_light(
            safety_tensor=synthetic_safety_tensor_pass,
            stability_forecaster=synthetic_stability_forecaster_stable,
            gate_decision=synthetic_gate_decision_pass,
        )

        assert signal1 == signal2


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS: P3 and P4 Integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestP3P4Integration:
    """Tests for P3 and P4 integration functions."""

    def test_016_p3_summary_schema(
        self,
        synthetic_safety_tensor_pass,
        synthetic_stability_forecaster_stable,
        synthetic_gate_decision_pass,
    ):
        """build_p3_uplift_safety_summary returns correct schema."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        summary = build_p3_uplift_safety_summary(signal)

        assert "risk_band" in summary
        assert "uplift_safety_decision" in summary
        assert "current_stability" in summary
        assert "stability_trend" in summary

    def test_017_p3_summary_with_predicted_cycles(
        self,
        synthetic_safety_tensor_block,
        synthetic_stability_forecaster_unstable,
        synthetic_gate_decision_block,
    ):
        """build_p3_uplift_safety_summary includes predicted cycles if present."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        summary = build_p3_uplift_safety_summary(signal)

        assert "predicted_instability_cycles" in summary
        assert summary["predicted_instability_cycles"] == [11, 12, 13]

    def test_018_p3_summary_json_serializable(
        self,
        synthetic_safety_tensor_pass,
        synthetic_stability_forecaster_stable,
        synthetic_gate_decision_pass,
    ):
        """build_p3_uplift_safety_summary output is JSON serializable."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        summary = build_p3_uplift_safety_summary(signal)

        json_str = json.dumps(summary)
        assert isinstance(json_str, str)

    def test_019_p4_calibration_schema(
        self,
        synthetic_safety_tensor_pass,
        synthetic_stability_forecaster_stable,
        synthetic_gate_decision_pass,
    ):
        """build_p4_uplift_safety_calibration returns correct schema."""
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        calibration = build_p4_uplift_safety_calibration(tile)

        assert "tensor_norm" in calibration
        assert "risk_band" in calibration
        assert "stability_trend" in calibration
        assert "decision" in calibration
        assert "decision_rationale" in calibration

    def test_020_p4_calibration_with_predicted_cycles(
        self,
        synthetic_safety_tensor_block,
        synthetic_stability_forecaster_unstable,
        synthetic_gate_decision_block,
    ):
        """build_p4_uplift_safety_calibration includes predicted cycles if present."""
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        calibration = build_p4_uplift_safety_calibration(tile)

        assert "predicted_instability_cycles" in calibration
        assert calibration["predicted_instability_cycles"] == [11, 12, 13]

    def test_021_p4_calibration_json_serializable(
        self,
        synthetic_safety_tensor_pass,
        synthetic_stability_forecaster_stable,
        synthetic_gate_decision_pass,
    ):
        """build_p4_uplift_safety_calibration output is JSON serializable."""
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        calibration = build_p4_uplift_safety_calibration(tile)

        json_str = json.dumps(calibration)
        assert isinstance(json_str, str)


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS: Evidence and Council Integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvidenceAndCouncilIntegration:
    """Tests for evidence and council integration functions."""

    def test_022_attach_to_evidence_schema(
        self,
        synthetic_safety_tensor_pass,
        synthetic_stability_forecaster_stable,
        synthetic_gate_decision_pass,
    ):
        """attach_uplift_safety_to_evidence attaches data correctly."""
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )

        evidence = {"governance": {}}
        result = attach_uplift_safety_to_evidence(evidence, tile, signal)

        assert "governance" in result
        assert "uplift_safety" in result["governance"]
        assert result["governance"]["uplift_safety"]["tile"] == tile
        assert result["governance"]["uplift_safety"]["signal"] == signal

    def test_023_attach_to_evidence_immutable(
        self,
        synthetic_safety_tensor_pass,
        synthetic_stability_forecaster_stable,
        synthetic_gate_decision_pass,
    ):
        """attach_uplift_safety_to_evidence does not modify input evidence."""
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )

        evidence = {"governance": {}}
        result = attach_uplift_safety_to_evidence(evidence, tile, signal)

        # Original evidence should not have uplift_safety
        assert "uplift_safety" not in evidence.get("governance", {})
        # Result should have it
        assert "uplift_safety" in result["governance"]

    def test_024_council_summary_pass_maps_to_ok(
        self,
        synthetic_safety_tensor_pass,
        synthetic_stability_forecaster_stable,
        synthetic_gate_decision_pass,
    ):
        """summarize_uplift_safety_for_council maps PASS → OK."""
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        summary = summarize_uplift_safety_for_council(tile)

        assert summary["status"] == "OK"
        assert "risk_band" in summary
        assert "rationale" in summary

    def test_025_council_summary_warn_maps_to_warn(
        self,
        synthetic_safety_tensor_warn,
        synthetic_stability_forecaster_degrading,
        synthetic_gate_decision_warn,
    ):
        """summarize_uplift_safety_for_council maps WARN → WARN."""
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_warn,
            synthetic_stability_forecaster_degrading,
            synthetic_gate_decision_warn,
        )
        summary = summarize_uplift_safety_for_council(tile)

        assert summary["status"] == "WARN"

    def test_026_council_summary_block_maps_to_block(
        self,
        synthetic_safety_tensor_block,
        synthetic_stability_forecaster_unstable,
        synthetic_gate_decision_block,
    ):
        """summarize_uplift_safety_for_council maps BLOCK → BLOCK."""
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        summary = summarize_uplift_safety_for_council(tile)

        assert summary["status"] == "BLOCK"
        assert "predicted_instability_horizon" in summary
        assert summary["predicted_instability_horizon"] == [11, 12, 13]

    def test_027_council_summary_without_predicted_cycles(
        self,
        synthetic_safety_tensor_pass,
        synthetic_stability_forecaster_stable,
        synthetic_gate_decision_pass,
    ):
        """summarize_uplift_safety_for_council omits horizon if no predicted cycles."""
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        summary = summarize_uplift_safety_for_council(tile)

        assert "predicted_instability_horizon" not in summary

    def test_028_council_summary_json_serializable(
        self,
        synthetic_safety_tensor_pass,
        synthetic_stability_forecaster_stable,
        synthetic_gate_decision_pass,
    ):
        """summarize_uplift_safety_for_council output is JSON serializable."""
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        summary = summarize_uplift_safety_for_council(tile)

        json_str = json.dumps(summary)
        assert isinstance(json_str, str)


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS: Gate Readiness Annex
# ═══════════════════════════════════════════════════════════════════════════════

class TestGateReadinessAnnex:
    """Tests for Gate Readiness Annex."""

    def test_029_gate_annex_schema(
        self,
        synthetic_safety_tensor_pass,
        synthetic_stability_forecaster_stable,
        synthetic_gate_decision_pass,
    ):
        """build_first_light_uplift_gate_annex returns correct schema."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        annex = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)

        assert "schema_version" in annex
        assert "p3_decision" in annex
        assert "p3_risk_band" in annex
        assert "p4_decision" in annex
        assert "p4_risk_band" in annex
        assert "stability_trend" in annex

    def test_030_gate_annex_json_serializable(
        self,
        synthetic_safety_tensor_pass,
        synthetic_stability_forecaster_stable,
        synthetic_gate_decision_pass,
    ):
        """build_first_light_uplift_gate_annex output is JSON serializable."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        annex = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)

        json_str = json.dumps(annex)
        assert isinstance(json_str, str)

    def test_031_gate_annex_deterministic(
        self,
        synthetic_safety_tensor_pass,
        synthetic_stability_forecaster_stable,
        synthetic_gate_decision_pass,
    ):
        """build_first_light_uplift_gate_annex is deterministic."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        annex1 = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
        annex2 = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)

        assert annex1 == annex2

    def test_032_evidence_with_annex(
        self,
        synthetic_safety_tensor_pass,
        synthetic_stability_forecaster_stable,
        synthetic_gate_decision_pass,
    ):
        """attach_uplift_safety_to_evidence includes annex when P3/P4 provided."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        evidence = {"governance": {}}
        result = attach_uplift_safety_to_evidence(
            evidence, tile, signal, p3_summary, p4_calibration
        )

        assert "first_light_gate_annex" in result["governance"]["uplift_safety"]
        annex = result["governance"]["uplift_safety"]["first_light_gate_annex"]
        assert annex["p3_decision"] == "PASS"
        assert annex["p4_decision"] == "PASS"

    def test_033_evidence_without_annex(
        self,
        synthetic_safety_tensor_pass,
        synthetic_stability_forecaster_stable,
        synthetic_gate_decision_pass,
    ):
        """attach_uplift_safety_to_evidence omits annex when P3/P4 not provided."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )

        evidence = {"governance": {}}
        result = attach_uplift_safety_to_evidence(evidence, tile, signal)

        assert "first_light_gate_annex" not in result["governance"]["uplift_safety"]

    def test_034_council_gate_alignment_ok_pass_pass(
        self,
        synthetic_safety_tensor_pass,
        synthetic_stability_forecaster_stable,
        synthetic_gate_decision_pass,
    ):
        """summarize_uplift_safety_for_council sets gate_alignment_ok=True for PASS/PASS.
        
        Note: gate_alignment_ok is intended as an auditor hint. If False, look harder
        at the discrepancy between P3 vs P4 safety decisions.
        """
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        summary = summarize_uplift_safety_for_council(tile, p3_summary, p4_calibration)

        assert summary["gate_alignment_ok"] is True

    def test_035_council_gate_alignment_ok_warn_warn(
        self,
        synthetic_safety_tensor_warn,
        synthetic_stability_forecaster_degrading,
        synthetic_gate_decision_warn,
    ):
        """summarize_uplift_safety_for_council sets gate_alignment_ok=True for WARN/WARN."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_warn,
            synthetic_stability_forecaster_degrading,
            synthetic_gate_decision_warn,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_warn,
            synthetic_stability_forecaster_degrading,
            synthetic_gate_decision_warn,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        summary = summarize_uplift_safety_for_council(tile, p3_summary, p4_calibration)

        assert summary["gate_alignment_ok"] is True

    def test_036_council_gate_alignment_ok_pass_warn(
        self,
        synthetic_safety_tensor_pass,
        synthetic_stability_forecaster_stable,
        synthetic_gate_decision_pass,
    ):
        """summarize_uplift_safety_for_council sets gate_alignment_ok=True for PASS/WARN."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        # Create P4 with WARN
        p4_calibration = {
            "decision": "WARN",
            "risk_band": "MEDIUM",
            "stability_trend": "DEGRADING",
        }

        summary = summarize_uplift_safety_for_council(tile, p3_summary, p4_calibration)

        assert summary["gate_alignment_ok"] is True

    def test_037_council_gate_alignment_ok_block_any(
        self,
        synthetic_safety_tensor_block,
        synthetic_stability_forecaster_unstable,
        synthetic_gate_decision_block,
    ):
        """summarize_uplift_safety_for_council sets gate_alignment_ok=False if either is BLOCK."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        summary = summarize_uplift_safety_for_council(tile, p3_summary, p4_calibration)

        assert summary["gate_alignment_ok"] is False

    def test_038_council_gate_alignment_ok_without_p3_p4(
        self,
        synthetic_safety_tensor_pass,
        synthetic_stability_forecaster_stable,
        synthetic_gate_decision_pass,
    ):
        """summarize_uplift_safety_for_council omits gate_alignment_ok without P3/P4."""
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )

        summary = summarize_uplift_safety_for_council(tile)

        assert "gate_alignment_ok" not in summary


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS: Gate Alignment Panel
# ═══════════════════════════════════════════════════════════════════════════════

class TestGateAlignmentPanel:
    """Tests for gate alignment panel and per-experiment export."""

    def test_039_export_gate_annex_per_experiment(
        self,
        synthetic_safety_tensor_pass,
        synthetic_stability_forecaster_stable,
        synthetic_gate_decision_pass,
        tmp_path,
    ):
        """export_gate_annex_per_experiment writes annex to file."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)
        annex = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)

        output_path = export_gate_annex_per_experiment(
            annex, "CAL-EXP-1", tmp_path / "calibration"
        )

        assert output_path.exists()
        assert output_path.name == "uplift_gate_annex_CAL-EXP-1.json"

        # Verify file contents
        with open(output_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded["cal_id"] == "CAL-EXP-1"
        assert loaded["p3_decision"] == "PASS"
        assert loaded["p4_decision"] == "PASS"

    def test_040_export_gate_annex_json_safe(
        self,
        synthetic_safety_tensor_pass,
        synthetic_stability_forecaster_stable,
        synthetic_gate_decision_pass,
        tmp_path,
    ):
        """export_gate_annex_per_experiment produces JSON-safe output."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)
        annex = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)

        output_path = export_gate_annex_per_experiment(
            annex, "CAL-EXP-1", tmp_path / "calibration"
        )

        # Should be readable JSON
        with open(output_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        assert isinstance(loaded, dict)

    def test_041_build_gate_alignment_panel_schema(
        self,
        synthetic_safety_tensor_pass,
        synthetic_stability_forecaster_stable,
        synthetic_gate_decision_pass,
    ):
        """build_gate_alignment_panel returns correct schema."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        annex1 = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
        annex1["cal_id"] = "CAL-EXP-1"

        panel = build_gate_alignment_panel([annex1])

        assert "schema_version" in panel
        assert "total_experiments" in panel
        assert "aligned_count" in panel
        assert "misaligned_count" in panel
        assert "experiments_misaligned" in panel
        assert "alignment_rate" in panel

    def test_042_build_gate_alignment_panel_aligned(
        self,
        synthetic_safety_tensor_pass,
        synthetic_stability_forecaster_stable,
        synthetic_gate_decision_pass,
    ):
        """build_gate_alignment_panel correctly identifies aligned experiments."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        annex1 = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
        annex1["cal_id"] = "CAL-EXP-1"

        annex2 = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
        annex2["cal_id"] = "CAL-EXP-2"

        panel = build_gate_alignment_panel([annex1, annex2])

        assert panel["total_experiments"] == 2
        assert panel["aligned_count"] == 2
        assert panel["misaligned_count"] == 0
        assert panel["experiments_misaligned"] == []
        assert panel["alignment_rate"] == 1.0

    def test_043_build_gate_alignment_panel_misaligned(
        self,
        synthetic_safety_tensor_block,
        synthetic_stability_forecaster_unstable,
        synthetic_gate_decision_block,
    ):
        """build_gate_alignment_panel correctly identifies misaligned experiments."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        annex1 = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
        annex1["cal_id"] = "CAL-EXP-1"

        panel = build_gate_alignment_panel([annex1])

        assert panel["total_experiments"] == 1
        assert panel["aligned_count"] == 0
        assert panel["misaligned_count"] == 1
        assert "CAL-EXP-1" in panel["experiments_misaligned"]
        assert panel["alignment_rate"] == 0.0

    def test_044_build_gate_alignment_panel_mixed(
        self,
        synthetic_safety_tensor_pass,
        synthetic_stability_forecaster_stable,
        synthetic_gate_decision_pass,
        synthetic_safety_tensor_block,
        synthetic_stability_forecaster_unstable,
        synthetic_gate_decision_block,
    ):
        """build_gate_alignment_panel handles mixed aligned/misaligned experiments."""
        # Aligned experiment
        signal_pass = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        tile_pass = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        p3_summary_pass = build_p3_uplift_safety_summary(signal_pass)
        p4_calibration_pass = build_p4_uplift_safety_calibration(tile_pass)
        annex1 = build_first_light_uplift_gate_annex(p3_summary_pass, p4_calibration_pass)
        annex1["cal_id"] = "CAL-EXP-1"

        # Misaligned experiment
        signal_block = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        tile_block = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        p3_summary_block = build_p3_uplift_safety_summary(signal_block)
        p4_calibration_block = build_p4_uplift_safety_calibration(tile_block)
        annex2 = build_first_light_uplift_gate_annex(p3_summary_block, p4_calibration_block)
        annex2["cal_id"] = "CAL-EXP-2"

        panel = build_gate_alignment_panel([annex1, annex2])

        assert panel["total_experiments"] == 2
        assert panel["aligned_count"] == 1
        assert panel["misaligned_count"] == 1
        assert "CAL-EXP-2" in panel["experiments_misaligned"]
        assert "CAL-EXP-1" not in panel["experiments_misaligned"]
        assert panel["alignment_rate"] == 0.5

    def test_045_build_gate_alignment_panel_empty(self):
        """build_gate_alignment_panel handles empty annex list."""
        panel = build_gate_alignment_panel([])

        assert panel["total_experiments"] == 0
        assert panel["aligned_count"] == 0
        assert panel["misaligned_count"] == 0
        assert panel["experiments_misaligned"] == []
        assert panel["alignment_rate"] == 0.0

    def test_046_build_gate_alignment_panel_deterministic(
        self,
        synthetic_safety_tensor_pass,
        synthetic_stability_forecaster_stable,
        synthetic_gate_decision_pass,
    ):
        """build_gate_alignment_panel is deterministic."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        annex1 = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
        annex1["cal_id"] = "CAL-EXP-1"

        annex2 = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
        annex2["cal_id"] = "CAL-EXP-2"

        panel1 = build_gate_alignment_panel([annex1, annex2])
        panel2 = build_gate_alignment_panel([annex1, annex2])

        assert panel1 == panel2

    def test_047_build_gate_alignment_panel_json_serializable(
        self,
        synthetic_safety_tensor_pass,
        synthetic_stability_forecaster_stable,
        synthetic_gate_decision_pass,
    ):
        """build_gate_alignment_panel output is JSON serializable."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        annex1 = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
        annex1["cal_id"] = "CAL-EXP-1"

        panel = build_gate_alignment_panel([annex1])

        json_str = json.dumps(panel)
        assert isinstance(json_str, str)

    def test_048_evidence_with_alignment_panel(
        self,
        synthetic_safety_tensor_pass,
        synthetic_stability_forecaster_stable,
        synthetic_gate_decision_pass,
    ):
        """attach_uplift_safety_to_evidence includes alignment panel when provided."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        annex1 = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
        annex1["cal_id"] = "CAL-EXP-1"
        alignment_panel = build_gate_alignment_panel([annex1])

        evidence = {"governance": {}}
        result = attach_uplift_safety_to_evidence(
            evidence, tile, signal, p3_summary, p4_calibration, alignment_panel
        )

        assert "uplift_gate_alignment_panel" in result["governance"]
        assert result["governance"]["uplift_gate_alignment_panel"] == alignment_panel

    def test_049_evidence_without_alignment_panel(
        self,
        synthetic_safety_tensor_pass,
        synthetic_stability_forecaster_stable,
        synthetic_gate_decision_pass,
    ):
        """attach_uplift_safety_to_evidence omits alignment panel when not provided."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        evidence = {"governance": {}}
        result = attach_uplift_safety_to_evidence(
            evidence, tile, signal, p3_summary, p4_calibration
        )

        assert "uplift_gate_alignment_panel" not in result["governance"]


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS: Misalignment Reason Codes and Status Signal
# ═══════════════════════════════════════════════════════════════════════════════

class TestMisalignmentReasonCodes:
    """Tests for misalignment reason codes and status signal extraction."""

    def test_050_panel_includes_misalignment_details(
        self,
        synthetic_safety_tensor_block,
        synthetic_stability_forecaster_unstable,
        synthetic_gate_decision_block,
    ):
        """build_gate_alignment_panel includes misalignment_details."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        annex1 = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
        annex1["cal_id"] = "CAL-EXP-1"

        panel = build_gate_alignment_panel([annex1])

        assert "misalignment_details" in panel
        assert len(panel["misalignment_details"]) == 1
        assert panel["misalignment_details"][0]["cal_id"] == "CAL-EXP-1"
        assert "reason_code" in panel["misalignment_details"][0]
        assert "p3_decision" in panel["misalignment_details"][0]
        assert "p4_decision" in panel["misalignment_details"][0]

    def test_051_reason_code_p3_block(
        self,
        synthetic_safety_tensor_block,
        synthetic_stability_forecaster_unstable,
        synthetic_gate_decision_block,
        synthetic_safety_tensor_pass,
        synthetic_stability_forecaster_stable,
        synthetic_gate_decision_pass,
    ):
        """build_gate_alignment_panel correctly identifies P3_BLOCK reason code."""
        # P3 BLOCK, P4 PASS
        signal_block = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        tile_pass = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        p3_summary_block = build_p3_uplift_safety_summary(signal_block)
        p4_calibration_pass = build_p4_uplift_safety_calibration(tile_pass)

        annex = build_first_light_uplift_gate_annex(p3_summary_block, p4_calibration_pass)
        annex["cal_id"] = "CAL-EXP-1"

        panel = build_gate_alignment_panel([annex])

        assert len(panel["misalignment_details"]) == 1
        assert panel["misalignment_details"][0]["reason_code"] == "P3_BLOCK"
        assert panel["misalignment_details"][0]["p3_decision"] == "BLOCK"
        assert panel["misalignment_details"][0]["p4_decision"] == "PASS"

    def test_052_reason_code_p4_block(
        self,
        synthetic_safety_tensor_pass,
        synthetic_stability_forecaster_stable,
        synthetic_gate_decision_pass,
        synthetic_safety_tensor_block,
        synthetic_stability_forecaster_unstable,
        synthetic_gate_decision_block,
    ):
        """build_gate_alignment_panel correctly identifies P4_BLOCK reason code."""
        # P3 PASS, P4 BLOCK
        signal_pass = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        tile_block = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        p3_summary_pass = build_p3_uplift_safety_summary(signal_pass)
        p4_calibration_block = build_p4_uplift_safety_calibration(tile_block)

        annex = build_first_light_uplift_gate_annex(p3_summary_pass, p4_calibration_block)
        annex["cal_id"] = "CAL-EXP-1"

        panel = build_gate_alignment_panel([annex])

        assert len(panel["misalignment_details"]) == 1
        assert panel["misalignment_details"][0]["reason_code"] == "P4_BLOCK"
        assert panel["misalignment_details"][0]["p3_decision"] == "PASS"
        assert panel["misalignment_details"][0]["p4_decision"] == "BLOCK"

    def test_053_reason_code_both_block(
        self,
        synthetic_safety_tensor_block,
        synthetic_stability_forecaster_unstable,
        synthetic_gate_decision_block,
    ):
        """build_gate_alignment_panel correctly identifies BOTH_BLOCK reason code."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        annex = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
        annex["cal_id"] = "CAL-EXP-1"

        panel = build_gate_alignment_panel([annex])

        assert len(panel["misalignment_details"]) == 1
        assert panel["misalignment_details"][0]["reason_code"] == "BOTH_BLOCK"
        assert panel["misalignment_details"][0]["p3_decision"] == "BLOCK"
        assert panel["misalignment_details"][0]["p4_decision"] == "BLOCK"

    def test_054_reason_codes_mixed(
        self,
        synthetic_safety_tensor_pass,
        synthetic_stability_forecaster_stable,
        synthetic_gate_decision_pass,
        synthetic_safety_tensor_block,
        synthetic_stability_forecaster_unstable,
        synthetic_gate_decision_block,
    ):
        """build_gate_alignment_panel handles mixed reason codes."""
        # P3 BLOCK, P4 PASS
        signal_block = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        tile_pass = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        p3_summary_block = build_p3_uplift_safety_summary(signal_block)
        p4_calibration_pass = build_p4_uplift_safety_calibration(tile_pass)
        annex1 = build_first_light_uplift_gate_annex(p3_summary_block, p4_calibration_pass)
        annex1["cal_id"] = "CAL-EXP-1"

        # P3 PASS, P4 BLOCK
        signal_pass = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        tile_block = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        p3_summary_pass = build_p3_uplift_safety_summary(signal_pass)
        p4_calibration_block = build_p4_uplift_safety_calibration(tile_block)
        annex2 = build_first_light_uplift_gate_annex(p3_summary_pass, p4_calibration_block)
        annex2["cal_id"] = "CAL-EXP-2"

        # Both BLOCK
        annex3 = build_first_light_uplift_gate_annex(p3_summary_block, p4_calibration_block)
        annex3["cal_id"] = "CAL-EXP-3"

        panel = build_gate_alignment_panel([annex1, annex2, annex3])

        assert len(panel["misalignment_details"]) == 3
        reason_codes = [detail["reason_code"] for detail in panel["misalignment_details"]]
        assert "P3_BLOCK" in reason_codes
        assert "P4_BLOCK" in reason_codes
        assert "BOTH_BLOCK" in reason_codes

    def test_055_misalignment_details_deterministic(
        self,
        synthetic_safety_tensor_block,
        synthetic_stability_forecaster_unstable,
        synthetic_gate_decision_block,
    ):
        """build_gate_alignment_panel misalignment_details are deterministic."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        annex1 = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
        annex1["cal_id"] = "CAL-EXP-2"

        annex2 = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
        annex2["cal_id"] = "CAL-EXP-1"

        panel1 = build_gate_alignment_panel([annex1, annex2])
        panel2 = build_gate_alignment_panel([annex1, annex2])

        assert panel1["misalignment_details"] == panel2["misalignment_details"]
        # Should be sorted by cal_id
        assert panel1["misalignment_details"][0]["cal_id"] == "CAL-EXP-1"
        assert panel1["misalignment_details"][1]["cal_id"] == "CAL-EXP-2"

    def test_056_misalignment_details_json_safe(
        self,
        synthetic_safety_tensor_block,
        synthetic_stability_forecaster_unstable,
        synthetic_gate_decision_block,
    ):
        """build_gate_alignment_panel misalignment_details are JSON serializable."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        annex = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
        annex["cal_id"] = "CAL-EXP-1"

        panel = build_gate_alignment_panel([annex])

        json_str = json.dumps(panel["misalignment_details"])
        assert isinstance(json_str, str)

    def test_057_experiments_misaligned_backward_compatible(
        self,
        synthetic_safety_tensor_block,
        synthetic_stability_forecaster_unstable,
        synthetic_gate_decision_block,
    ):
        """build_gate_alignment_panel preserves experiments_misaligned for backward compatibility."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        annex1 = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
        annex1["cal_id"] = "CAL-EXP-1"

        annex2 = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
        annex2["cal_id"] = "CAL-EXP-2"

        panel = build_gate_alignment_panel([annex1, annex2])

        # experiments_misaligned should still be present
        assert "experiments_misaligned" in panel
        assert isinstance(panel["experiments_misaligned"], list)
        assert len(panel["experiments_misaligned"]) == 2
        assert "CAL-EXP-1" in panel["experiments_misaligned"]
        assert "CAL-EXP-2" in panel["experiments_misaligned"]

    def test_058_extract_alignment_signal_schema(
        self,
        synthetic_safety_tensor_block,
        synthetic_stability_forecaster_unstable,
        synthetic_gate_decision_block,
    ):
        """extract_uplift_gate_alignment_signal returns correct schema."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        annex = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
        annex["cal_id"] = "CAL-EXP-1"

        panel = build_gate_alignment_panel([annex])
        status_signal = extract_uplift_gate_alignment_signal(panel)

        assert "alignment_rate" in status_signal
        assert "misaligned_count" in status_signal
        assert "top_misaligned_cal_ids" in status_signal
        assert "reason_code_histogram" in status_signal

    def test_059_extract_alignment_signal_values(
        self,
        synthetic_safety_tensor_block,
        synthetic_stability_forecaster_unstable,
        synthetic_gate_decision_block,
    ):
        """extract_uplift_gate_alignment_signal extracts correct values."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        annex1 = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
        annex1["cal_id"] = "CAL-EXP-1"

        annex2 = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
        annex2["cal_id"] = "CAL-EXP-2"

        panel = build_gate_alignment_panel([annex1, annex2])
        status_signal = extract_uplift_gate_alignment_signal(panel)

        assert status_signal["alignment_rate"] == 0.0
        assert status_signal["misaligned_count"] == 2
        assert len(status_signal["top_misaligned_cal_ids"]) == 2
        assert "CAL-EXP-1" in status_signal["top_misaligned_cal_ids"]
        assert "CAL-EXP-2" in status_signal["top_misaligned_cal_ids"]
        assert "BOTH_BLOCK" in status_signal["reason_code_histogram"]
        assert status_signal["reason_code_histogram"]["BOTH_BLOCK"] == 2

    def test_060_extract_alignment_signal_top_3(
        self,
        synthetic_safety_tensor_block,
        synthetic_stability_forecaster_unstable,
        synthetic_gate_decision_block,
    ):
        """extract_uplift_gate_alignment_signal limits top_misaligned_cal_ids to top 3."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        # Create 5 misaligned experiments
        annexes = []
        for i in range(5):
            annex = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
            annex["cal_id"] = f"CAL-EXP-{i+1}"
            annexes.append(annex)

        panel = build_gate_alignment_panel(annexes)
        status_signal = extract_uplift_gate_alignment_signal(panel)

        assert len(status_signal["top_misaligned_cal_ids"]) == 3
        assert status_signal["top_misaligned_cal_ids"] == ["CAL-EXP-1", "CAL-EXP-2", "CAL-EXP-3"]

    def test_061_extract_alignment_signal_reason_histogram(
        self,
        synthetic_safety_tensor_pass,
        synthetic_stability_forecaster_stable,
        synthetic_gate_decision_pass,
        synthetic_safety_tensor_block,
        synthetic_stability_forecaster_unstable,
        synthetic_gate_decision_block,
    ):
        """extract_uplift_gate_alignment_signal builds correct reason_code_histogram."""
        # P3 BLOCK, P4 PASS
        signal_block = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        tile_pass = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        p3_summary_block = build_p3_uplift_safety_summary(signal_block)
        p4_calibration_pass = build_p4_uplift_safety_calibration(tile_pass)
        annex1 = build_first_light_uplift_gate_annex(p3_summary_block, p4_calibration_pass)
        annex1["cal_id"] = "CAL-EXP-1"

        # P4 BLOCK
        signal_pass = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        tile_block = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        p3_summary_pass = build_p3_uplift_safety_summary(signal_pass)
        p4_calibration_block = build_p4_uplift_safety_calibration(tile_block)
        annex2 = build_first_light_uplift_gate_annex(p3_summary_pass, p4_calibration_block)
        annex2["cal_id"] = "CAL-EXP-2"

        # Both BLOCK
        annex3 = build_first_light_uplift_gate_annex(p3_summary_block, p4_calibration_block)
        annex3["cal_id"] = "CAL-EXP-3"

        panel = build_gate_alignment_panel([annex1, annex2, annex3])
        status_signal = extract_uplift_gate_alignment_signal(panel)

        histogram = status_signal["reason_code_histogram"]
        assert histogram["P3_BLOCK"] == 1
        assert histogram["P4_BLOCK"] == 1
        assert histogram["BOTH_BLOCK"] == 1

    def test_062_extract_alignment_signal_deterministic(
        self,
        synthetic_safety_tensor_block,
        synthetic_stability_forecaster_unstable,
        synthetic_gate_decision_block,
    ):
        """extract_uplift_gate_alignment_signal is deterministic."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        annex = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
        annex["cal_id"] = "CAL-EXP-1"

        panel = build_gate_alignment_panel([annex])
        signal1 = extract_uplift_gate_alignment_signal(panel)
        signal2 = extract_uplift_gate_alignment_signal(panel)

        assert signal1 == signal2

    def test_063_extract_alignment_signal_json_safe(
        self,
        synthetic_safety_tensor_block,
        synthetic_stability_forecaster_unstable,
        synthetic_gate_decision_block,
    ):
        """extract_uplift_gate_alignment_signal output is JSON serializable."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        annex = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
        annex["cal_id"] = "CAL-EXP-1"

        panel = build_gate_alignment_panel([annex])
        status_signal = extract_uplift_gate_alignment_signal(panel)

        json_str = json.dumps(status_signal)
        assert isinstance(json_str, str)

    def test_064_evidence_with_alignment_signal(
        self,
        synthetic_safety_tensor_block,
        synthetic_stability_forecaster_unstable,
        synthetic_gate_decision_block,
    ):
        """attach_uplift_safety_to_evidence includes alignment signal when panel provided."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        annex = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
        annex["cal_id"] = "CAL-EXP-1"
        alignment_panel = build_gate_alignment_panel([annex])

        evidence = {"governance": {}}
        result = attach_uplift_safety_to_evidence(
            evidence, tile, signal, p3_summary, p4_calibration, alignment_panel
        )

        assert "signals" in result
        assert "uplift_gate_alignment" in result["signals"]
        assert "alignment_rate" in result["signals"]["uplift_gate_alignment"]
        assert "misaligned_count" in result["signals"]["uplift_gate_alignment"]
        assert "top_misaligned_cal_ids" in result["signals"]["uplift_gate_alignment"]
        assert "reason_code_histogram" in result["signals"]["uplift_gate_alignment"]

    def test_065_evidence_without_alignment_signal(
        self,
        synthetic_safety_tensor_pass,
        synthetic_stability_forecaster_stable,
        synthetic_gate_decision_pass,
    ):
        """attach_uplift_safety_to_evidence omits alignment signal when panel not provided."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        evidence = {"governance": {}}
        result = attach_uplift_safety_to_evidence(
            evidence, tile, signal, p3_summary, p4_calibration
        )

        # Should not have signals if panel not provided
        if "signals" in result:
            assert "uplift_gate_alignment" not in result["signals"]


# ═══════════════════════════════════════════════════════════════════════════════
# RUN TESTS
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

