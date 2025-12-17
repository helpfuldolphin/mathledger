"""
CI smoke test for readiness governance tile serialization.

Tests that the readiness governance tile:
  - Builds from synthetic tensor + polygraph + eval
  - Serializes via json.dumps
  - Is deterministic across repeated calls
  - Uses neutral language (no "good/bad/success/failure")
"""

import json
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Any

from backend.health.readiness_tensor_adapter import (
    attach_metric_readiness_panel_to_evidence,
    attach_metric_readiness_to_evidence,
    build_first_light_metric_readiness_annex,
    build_metric_readiness_panel,
    build_readiness_governance_tile,
    emit_cal_exp_metric_readiness_annex,
    extract_readiness_signal_for_first_light,
    summarize_readiness_for_uplift_council,
    summarize_readiness_perf_budget_consistency,
    READINESS_ANNEX_SCHEMA_VERSION,
    READINESS_GOVERNANCE_TILE_SCHEMA_VERSION,
    READINESS_PANEL_SCHEMA_VERSION,
    StatusLight,
)


# =============================================================================
# Test Fixtures
# =============================================================================

def create_synthetic_tensor() -> Dict[str, Any]:
    """Create a synthetic readiness tensor for testing."""
    return {
        "slice_vectors": {
            "slice_a": {
                "readiness_score": 0.8,
                "drift_component": 0.9,
                "budget_component": 0.7,
                "metric_consistency_component": 1.0,
                "vector_norm": 1.5
            },
            "slice_b": {
                "readiness_score": 0.5,
                "drift_component": 0.4,
                "budget_component": 0.3,
                "metric_consistency_component": 0.8,
                "vector_norm": 1.0
            }
        },
        "global_norm": 1.25,
        "ranked_slices": ["slice_a", "slice_b"],
        "schema_version": "1.0.0"
    }


def create_synthetic_polygraph() -> Dict[str, Any]:
    """Create a synthetic drift polygraph for testing."""
    return {
        "drift_momentum": -0.05,
        "entangled_pairs": [["slice_a", "slice_b"]],
        "poly_fail_detected": False,
        "neutral_notes": ["slice_a and slice_b: Similar drift patterns"]
    }


def create_synthetic_phase_transition() -> Dict[str, Any]:
    """Create a synthetic phase transition eval for testing."""
    return {
        "transition_safe": True,
        "blocking_conditions": [],
        "transition_band": "MEDIUM",
        "recommendations": ["Transition band: MEDIUM - proceed with caution"]
    }


# =============================================================================
# Task 3: CI Smoke Tests
# =============================================================================

class TestReadinessGovernanceTileSerializes(unittest.TestCase):
    """CI smoke tests for readiness governance tile."""
    
    def test_tile_builds_from_synthetic_inputs(self):
        """Tile builds from synthetic tensor + polygraph + eval."""
        tensor = create_synthetic_tensor()
        polygraph = create_synthetic_polygraph()
        phase_transition = create_synthetic_phase_transition()
        
        tile = build_readiness_governance_tile(
            tensor=tensor,
            polygraph=polygraph,
            phase_transition_eval=phase_transition
        )
        
        self.assertIsInstance(tile, dict)
        self.assertIn("status_light", tile)
        self.assertIn("global_norm", tile)
        self.assertIn("drift_momentum", tile)
        self.assertIn("poly_fail_detected", tile)
        self.assertIn("transition_band", tile)
        self.assertIn("slices_at_risk", tile)
        self.assertIn("headline", tile)
        self.assertIn("schema_version", tile)
    
    def test_tile_json_serializes(self):
        """Tile serializes via json.dumps."""
        tensor = create_synthetic_tensor()
        polygraph = create_synthetic_polygraph()
        phase_transition = create_synthetic_phase_transition()
        
        tile = build_readiness_governance_tile(
            tensor=tensor,
            polygraph=polygraph,
            phase_transition_eval=phase_transition
        )
        
        # Should not raise
        json_str = json.dumps(tile, sort_keys=True)
        self.assertIsInstance(json_str, str)
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        self.assertEqual(parsed["schema_version"], READINESS_GOVERNANCE_TILE_SCHEMA_VERSION)
    
    def test_tile_deterministic(self):
        """Tile output is deterministic across repeated calls."""
        tensor = create_synthetic_tensor()
        polygraph = create_synthetic_polygraph()
        phase_transition = create_synthetic_phase_transition()
        
        tiles = [
            build_readiness_governance_tile(
                tensor=tensor,
                polygraph=polygraph,
                phase_transition_eval=phase_transition
            )
            for _ in range(5)
        ]
        
        json_outputs = [json.dumps(t, sort_keys=True) for t in tiles]
        
        # All outputs should be identical
        self.assertTrue(all(j == json_outputs[0] for j in json_outputs))
    
    def test_tile_neutral_language(self):
        """Tile uses neutral language (no value judgments)."""
        tensor = create_synthetic_tensor()
        polygraph = create_synthetic_polygraph()
        phase_transition = create_synthetic_phase_transition()
        
        tile = build_readiness_governance_tile(
            tensor=tensor,
            polygraph=polygraph,
            phase_transition_eval=phase_transition
        )
        
        headline = tile["headline"].lower()
        
        # Check for absence of value judgment words
        self.assertNotIn("good", headline)
        self.assertNotIn("bad", headline)
        self.assertNotIn("better", headline)
        self.assertNotIn("worse", headline)
        self.assertNotIn("success", headline)
        self.assertNotIn("failure", headline)
        self.assertNotIn("excellent", headline)
        self.assertNotIn("terrible", headline)
    
    def test_tile_status_light_red_when_not_safe(self):
        """status_light is RED when transition_safe == False."""
        tensor = create_synthetic_tensor()
        polygraph = create_synthetic_polygraph()
        phase_transition = {
            "transition_safe": False,
            "blocking_conditions": ["Test condition"],
            "transition_band": "LOW",
            "recommendations": []
        }
        
        tile = build_readiness_governance_tile(
            tensor=tensor,
            polygraph=polygraph,
            phase_transition_eval=phase_transition
        )
        
        self.assertEqual(tile["status_light"], StatusLight.RED)
    
    def test_tile_status_light_yellow_when_drift_momentum_negative(self):
        """status_light is YELLOW when drift_momentum < -0.1."""
        tensor = create_synthetic_tensor()
        polygraph = {
            "drift_momentum": -0.15,  # < -0.1
            "entangled_pairs": [],
            "poly_fail_detected": False,
            "neutral_notes": []
        }
        phase_transition = {
            "transition_safe": True,
            "blocking_conditions": [],
            "transition_band": "MEDIUM",
            "recommendations": []
        }
        
        tile = build_readiness_governance_tile(
            tensor=tensor,
            polygraph=polygraph,
            phase_transition_eval=phase_transition
        )
        
        self.assertEqual(tile["status_light"], StatusLight.YELLOW)
    
    def test_tile_status_light_yellow_when_poly_fail(self):
        """status_light is YELLOW when poly_fail_detected."""
        tensor = create_synthetic_tensor()
        polygraph = {
            "drift_momentum": 0.0,
            "entangled_pairs": [],
            "poly_fail_detected": True,  # Poly-fail
            "neutral_notes": []
        }
        phase_transition = {
            "transition_safe": True,
            "blocking_conditions": [],
            "transition_band": "MEDIUM",
            "recommendations": []
        }
        
        tile = build_readiness_governance_tile(
            tensor=tensor,
            polygraph=polygraph,
            phase_transition_eval=phase_transition
        )
        
        self.assertEqual(tile["status_light"], StatusLight.YELLOW)
    
    def test_tile_status_light_green_when_stable(self):
        """status_light is GREEN when all stable."""
        tensor = create_synthetic_tensor()
        polygraph = {
            "drift_momentum": 0.0,
            "entangled_pairs": [],
            "poly_fail_detected": False,
            "neutral_notes": []
        }
        phase_transition = {
            "transition_safe": True,
            "blocking_conditions": [],
            "transition_band": "MEDIUM",
            "recommendations": []
        }
        
        tile = build_readiness_governance_tile(
            tensor=tensor,
            polygraph=polygraph,
            phase_transition_eval=phase_transition
        )
        
        self.assertEqual(tile["status_light"], StatusLight.GREEN)
    
    def test_tile_slices_at_risk_from_entangled_pairs(self):
        """slices_at_risk includes slices from entangled pairs."""
        tensor = create_synthetic_tensor()
        polygraph = {
            "drift_momentum": 0.0,
            "entangled_pairs": [["slice_a", "slice_b"]],
            "poly_fail_detected": False,
            "neutral_notes": []
        }
        phase_transition = create_synthetic_phase_transition()
        
        tile = build_readiness_governance_tile(
            tensor=tensor,
            polygraph=polygraph,
            phase_transition_eval=phase_transition
        )
        
        self.assertIn("slice_a", tile["slices_at_risk"])
        self.assertIn("slice_b", tile["slices_at_risk"])
    
    def test_tile_slices_at_risk_from_low_readiness(self):
        """slices_at_risk includes slices with low readiness_score."""
        tensor = {
            "slice_vectors": {
                "slice_low": {
                    "readiness_score": 0.3,  # < 0.5
                    "drift_component": 0.5,
                    "budget_component": 0.5,
                    "metric_consistency_component": 1.0,
                    "vector_norm": 1.0
                }
            },
            "global_norm": 1.0,
            "ranked_slices": ["slice_low"],
            "schema_version": "1.0.0"
        }
        polygraph = {
            "drift_momentum": 0.0,
            "entangled_pairs": [],
            "poly_fail_detected": False,
            "neutral_notes": []
        }
        phase_transition = create_synthetic_phase_transition()
        
        tile = build_readiness_governance_tile(
            tensor=tensor,
            polygraph=polygraph,
            phase_transition_eval=phase_transition
        )
        
        self.assertIn("slice_low", tile["slices_at_risk"])
    
    def test_first_light_signal_extracts(self):
        """extract_readiness_signal_for_first_light returns compact signal."""
        tensor = create_synthetic_tensor()
        polygraph = create_synthetic_polygraph()
        phase_transition = create_synthetic_phase_transition()
        
        signal = extract_readiness_signal_for_first_light(
            tensor=tensor,
            polygraph=polygraph,
            phase_transition_eval=phase_transition
        )
        
        self.assertIn("global_norm", signal)
        self.assertIn("drift_momentum", signal)
        self.assertIn("transition_band", signal)
        self.assertIn("poly_fail_detected", signal)
        
        # Should be JSON-serializable
        json_str = json.dumps(signal, sort_keys=True)
        self.assertIsInstance(json_str, str)
    
    def test_first_light_signal_deterministic(self):
        """First-Light signal is deterministic."""
        tensor = create_synthetic_tensor()
        polygraph = create_synthetic_polygraph()
        phase_transition = create_synthetic_phase_transition()
        
        signals = [
            extract_readiness_signal_for_first_light(
                tensor=tensor,
                polygraph=polygraph,
                phase_transition_eval=phase_transition
            )
            for _ in range(5)
        ]
        
        json_outputs = [json.dumps(s, sort_keys=True) for s in signals]
        
        # All outputs should be identical
        self.assertTrue(all(j == json_outputs[0] for j in json_outputs))


def create_synthetic_tile() -> Dict[str, Any]:
    """Create a synthetic readiness tile for testing."""
    return {
        "status_light": StatusLight.GREEN,
        "global_norm": 0.8,
        "drift_momentum": 0.0,
        "poly_fail_detected": False,
        "transition_band": "MEDIUM",
        "slices_at_risk": ["slice_a"],
        "headline": "Test headline",
        "schema_version": "1.0.0"
    }


def create_synthetic_signal() -> Dict[str, Any]:
    """Create a synthetic readiness signal for testing."""
    return {
        "global_norm": 0.8,
        "drift_momentum": 0.0,
        "transition_band": "MEDIUM",
        "poly_fail_detected": False
    }


class TestAttachMetricReadinessToEvidence(unittest.TestCase):
    """Tests for attach_metric_readiness_to_evidence helper."""
    
    def test_evidence_attachment_shape(self):
        """Evidence attachment has correct shape."""
        evidence = {"governance": {}}
        tile = create_synthetic_tile()
        signal = create_synthetic_signal()
        
        result = attach_metric_readiness_to_evidence(evidence, tile, signal)
        
        self.assertIn("governance", result)
        self.assertIn("metric_readiness", result["governance"])
        readiness = result["governance"]["metric_readiness"]
        self.assertIn("global_norm", readiness)
        self.assertIn("transition_band", readiness)
        self.assertIn("transition_safe", readiness)
        self.assertIn("drift_momentum", readiness)
        self.assertIn("poly_fail_detected", readiness)
    
    def test_evidence_attachment_deterministic(self):
        """Evidence attachment is deterministic."""
        evidence = {"governance": {}}
        tile = create_synthetic_tile()
        signal = create_synthetic_signal()
        
        results = [
            attach_metric_readiness_to_evidence(evidence, tile, signal)
            for _ in range(5)
        ]
        
        json_outputs = [json.dumps(r, sort_keys=True) for r in results]
        self.assertTrue(all(j == json_outputs[0] for j in json_outputs))
    
    def test_evidence_attachment_json_round_trip(self):
        """Evidence attachment survives JSON round-trip."""
        evidence = {"governance": {}}
        tile = create_synthetic_tile()
        signal = create_synthetic_signal()
        
        result = attach_metric_readiness_to_evidence(evidence, tile, signal)
        
        # Round-trip through JSON
        json_str = json.dumps(result, sort_keys=True)
        parsed = json.loads(json_str)
        
        self.assertEqual(
            parsed["governance"]["metric_readiness"]["global_norm"],
            result["governance"]["metric_readiness"]["global_norm"]
        )
    
    def test_evidence_attachment_non_mutating(self):
        """Evidence attachment does not mutate input."""
        evidence = {"governance": {}}
        tile = create_synthetic_tile()
        signal = create_synthetic_signal()
        
        original_evidence = dict(evidence)
        result = attach_metric_readiness_to_evidence(evidence, tile, signal)
        
        # Original should be unchanged
        self.assertEqual(evidence, original_evidence)
        # Result should be different (new dict)
        self.assertIsNot(result, evidence)
        # Result should have readiness data
        self.assertIn("metric_readiness", result["governance"])


class TestSummarizeReadinessForUpliftCouncil(unittest.TestCase):
    """Tests for summarize_readiness_for_uplift_council adapter."""
    
    def test_council_summary_has_required_fields(self):
        """Council summary has all required fields."""
        tile = create_synthetic_tile()
        
        summary = summarize_readiness_for_uplift_council(tile)
        
        self.assertIn("status", summary)
        self.assertIn("slices_to_hold", summary)
        self.assertIn("slices_safe_to_progress", summary)
        self.assertIn("global_norm", summary)
        self.assertIn("transition_band", summary)
        self.assertIn("transition_safe", summary)
    
    def test_council_status_block_when_not_safe(self):
        """status is BLOCK when transition_safe == False."""
        tile = {
            "status_light": StatusLight.RED,  # Not safe
            "poly_fail_detected": False,
            "transition_band": "LOW",
            "drift_momentum": 0.0,
            "global_norm": 0.3,
            "slices_at_risk": []
        }
        
        summary = summarize_readiness_for_uplift_council(tile)
        
        self.assertEqual(summary["status"], "BLOCK")
    
    def test_council_status_block_when_poly_fail(self):
        """status is BLOCK when poly_fail_detected."""
        tile = {
            "status_light": StatusLight.GREEN,
            "poly_fail_detected": True,  # Poly-fail
            "transition_band": "MEDIUM",
            "drift_momentum": 0.0,
            "global_norm": 0.8,
            "slices_at_risk": []
        }
        
        summary = summarize_readiness_for_uplift_council(tile)
        
        self.assertEqual(summary["status"], "BLOCK")
    
    def test_council_status_warn_when_medium_band(self):
        """status is WARN when transition_band == MEDIUM."""
        tile = {
            "status_light": StatusLight.GREEN,
            "poly_fail_detected": False,
            "transition_band": "MEDIUM",  # Medium band
            "drift_momentum": 0.0,
            "global_norm": 0.6,
            "slices_at_risk": []
        }
        
        summary = summarize_readiness_for_uplift_council(tile)
        
        self.assertEqual(summary["status"], "WARN")
    
    def test_council_status_warn_when_negative_momentum(self):
        """status is WARN when drift_momentum < -0.1."""
        tile = {
            "status_light": StatusLight.GREEN,
            "poly_fail_detected": False,
            "transition_band": "HIGH",
            "drift_momentum": -0.15,  # < -0.1
            "global_norm": 0.8,
            "slices_at_risk": []
        }
        
        summary = summarize_readiness_for_uplift_council(tile)
        
        self.assertEqual(summary["status"], "WARN")
    
    def test_council_status_ok_when_stable(self):
        """status is OK when all stable."""
        tile = {
            "status_light": StatusLight.GREEN,
            "poly_fail_detected": False,
            "transition_band": "HIGH",
            "drift_momentum": 0.0,
            "global_norm": 0.8,
            "slices_at_risk": []
        }
        
        summary = summarize_readiness_for_uplift_council(tile)
        
        self.assertEqual(summary["status"], "OK")
    
    def test_council_slices_to_hold_from_at_risk(self):
        """slices_to_hold includes slices_at_risk."""
        tile = {
            "status_light": StatusLight.GREEN,
            "poly_fail_detected": False,
            "transition_band": "HIGH",
            "drift_momentum": 0.0,
            "global_norm": 0.8,
            "slices_at_risk": ["slice_a", "slice_b"]
        }
        
        summary = summarize_readiness_for_uplift_council(tile)
        
        self.assertEqual(set(summary["slices_to_hold"]), {"slice_a", "slice_b"})
    
    def test_council_summary_deterministic(self):
        """Council summary is deterministic."""
        tile = create_synthetic_tile()
        
        summaries = [
            summarize_readiness_for_uplift_council(tile)
            for _ in range(5)
        ]
        
        json_outputs = [json.dumps(s, sort_keys=True) for s in summaries]
        self.assertTrue(all(j == json_outputs[0] for j in json_outputs))


class TestBuildFirstLightMetricReadinessAnnex(unittest.TestCase):
    """Tests for build_first_light_metric_readiness_annex function."""
    
    def test_annex_has_required_fields(self):
        """Annex has all required fields."""
        p3_summary = {
            "global_norm": 0.8,
            "transition_band": "MEDIUM",
            "poly_fail_detected": False
        }
        p4_calibration = {
            "global_norm": 0.75,
            "transition_band": "HIGH",
            "poly_fail_detected": False
        }
        
        annex = build_first_light_metric_readiness_annex(p3_summary, p4_calibration)
        
        self.assertIn("schema_version", annex)
        self.assertIn("p3_global_norm", annex)
        self.assertIn("p3_transition_band", annex)
        self.assertIn("p4_global_norm", annex)
        self.assertIn("p4_transition_band", annex)
        self.assertIn("poly_fail_detected", annex)
    
    def test_annex_schema_version(self):
        """Annex has correct schema version."""
        p3_summary = {"global_norm": 0.8, "transition_band": "MEDIUM", "poly_fail_detected": False}
        p4_calibration = {"global_norm": 0.75, "transition_band": "HIGH", "poly_fail_detected": False}
        
        annex = build_first_light_metric_readiness_annex(p3_summary, p4_calibration)
        
        self.assertEqual(annex["schema_version"], READINESS_ANNEX_SCHEMA_VERSION)
    
    def test_annex_poly_fail_or_logic(self):
        """poly_fail_detected is OR of P3 and P4."""
        p3_summary = {"global_norm": 0.8, "transition_band": "MEDIUM", "poly_fail_detected": True}
        p4_calibration = {"global_norm": 0.75, "transition_band": "HIGH", "poly_fail_detected": False}
        
        annex = build_first_light_metric_readiness_annex(p3_summary, p4_calibration)
        
        self.assertTrue(annex["poly_fail_detected"])
    
    def test_annex_json_round_trip(self):
        """Annex survives JSON round-trip."""
        p3_summary = {"global_norm": 0.8, "transition_band": "MEDIUM", "poly_fail_detected": False}
        p4_calibration = {"global_norm": 0.75, "transition_band": "HIGH", "poly_fail_detected": False}
        
        annex = build_first_light_metric_readiness_annex(p3_summary, p4_calibration)
        
        json_str = json.dumps(annex, sort_keys=True)
        parsed = json.loads(json_str)
        
        self.assertEqual(parsed["p3_global_norm"], annex["p3_global_norm"])
        self.assertEqual(parsed["p4_global_norm"], annex["p4_global_norm"])
    
    def test_annex_deterministic(self):
        """Annex output is deterministic."""
        p3_summary = {"global_norm": 0.8, "transition_band": "MEDIUM", "poly_fail_detected": False}
        p4_calibration = {"global_norm": 0.75, "transition_band": "HIGH", "poly_fail_detected": False}
        
        annexes = [
            build_first_light_metric_readiness_annex(p3_summary, p4_calibration)
            for _ in range(5)
        ]
        
        json_outputs = [json.dumps(a, sort_keys=True) for a in annexes]
        self.assertTrue(all(j == json_outputs[0] for j in json_outputs))


class TestAttachMetricReadinessToEvidenceWithAnnex(unittest.TestCase):
    """Tests for attach_metric_readiness_to_evidence with annex."""
    
    def test_evidence_attach_with_annex_non_mutating(self):
        """Evidence attachment with annex does not mutate input."""
        evidence = {"governance": {}}
        tile = create_synthetic_tile()
        signal = create_synthetic_signal()
        p3_summary = {"global_norm": 0.8, "transition_band": "MEDIUM", "poly_fail_detected": False}
        p4_calibration = {"global_norm": 0.75, "transition_band": "HIGH", "poly_fail_detected": False}
        
        original_evidence = dict(evidence)
        result = attach_metric_readiness_to_evidence(
            evidence, tile, signal, p3_summary=p3_summary, p4_calibration=p4_calibration
        )
        
        # Original should be unchanged
        self.assertEqual(evidence, original_evidence)
        # Result should be different (new dict)
        self.assertIsNot(result, evidence)
        # Result should have annex
        self.assertIn("first_light_annex", result["governance"]["metric_readiness"])
    
    def test_evidence_attach_without_annex(self):
        """Evidence attachment without P3/P4 does not include annex."""
        evidence = {"governance": {}}
        tile = create_synthetic_tile()
        signal = create_synthetic_signal()
        
        result = attach_metric_readiness_to_evidence(evidence, tile, signal)
        
        # Should not have annex
        self.assertNotIn("first_light_annex", result["governance"]["metric_readiness"])
    
    def test_evidence_attach_annex_shape(self):
        """Annex in evidence has correct shape."""
        evidence = {"governance": {}}
        tile = create_synthetic_tile()
        signal = create_synthetic_signal()
        p3_summary = {"global_norm": 0.8, "transition_band": "MEDIUM", "poly_fail_detected": False}
        p4_calibration = {"global_norm": 0.75, "transition_band": "HIGH", "poly_fail_detected": False}
        
        result = attach_metric_readiness_to_evidence(
            evidence, tile, signal, p3_summary=p3_summary, p4_calibration=p4_calibration
        )
        
        annex = result["governance"]["metric_readiness"]["first_light_annex"]
        self.assertIn("schema_version", annex)
        self.assertIn("p3_global_norm", annex)
        self.assertIn("p4_global_norm", annex)


class TestSummarizeReadinessPerfBudgetConsistency(unittest.TestCase):
    """Tests for summarize_readiness_perf_budget_consistency cross-check."""
    
    def test_consistency_has_required_fields(self):
        """Consistency summary has all required fields."""
        annex = {
            "p3_global_norm": 0.8,
            "p4_global_norm": 0.75,
            "poly_fail_detected": False
        }
        
        summary = summarize_readiness_perf_budget_consistency(annex, None, None)
        
        self.assertIn("consistency_status", summary)
        self.assertIn("advisory_notes", summary)
        self.assertIn("readiness_status", summary)
        self.assertIn("perf_status", summary)
        self.assertIn("budget_status", summary)
    
    def test_consistency_detects_readiness_ok_perf_block(self):
        """Consistency detects readiness OK while perf BLOCK."""
        annex = {
            "p3_global_norm": 0.8,  # High norm
            "p4_global_norm": 0.75,  # High norm
            "poly_fail_detected": False
        }
        perf_tile = {"status_light": StatusLight.RED}  # BLOCK
        budget_tile = None
        
        summary = summarize_readiness_perf_budget_consistency(annex, perf_tile, budget_tile)
        
        self.assertEqual(summary["readiness_status"], "OK")
        self.assertEqual(summary["perf_status"], "BLOCK")
        self.assertEqual(summary["consistency_status"], "INCONSISTENT")
        self.assertGreater(len(summary["advisory_notes"]), 0)
        self.assertIn("performance", summary["advisory_notes"][0].lower())
    
    def test_consistency_detects_readiness_ok_budget_block(self):
        """Consistency detects readiness OK while budget BLOCK."""
        annex = {
            "p3_global_norm": 0.8,
            "p4_global_norm": 0.75,
            "poly_fail_detected": False
        }
        perf_tile = None
        budget_tile = {"status_light": StatusLight.RED}  # BLOCK
        
        summary = summarize_readiness_perf_budget_consistency(annex, perf_tile, budget_tile)
        
        self.assertEqual(summary["readiness_status"], "OK")
        self.assertEqual(summary["budget_status"], "BLOCK")
        self.assertEqual(summary["consistency_status"], "INCONSISTENT")
        self.assertIn("budget", summary["advisory_notes"][0].lower())
    
    def test_consistency_detects_both_perf_and_budget_block(self):
        """Consistency detects readiness OK while both perf and budget BLOCK."""
        annex = {
            "p3_global_norm": 0.8,
            "p4_global_norm": 0.75,
            "poly_fail_detected": False
        }
        perf_tile = {"status_light": StatusLight.RED}
        budget_tile = {"status_light": StatusLight.RED}
        
        summary = summarize_readiness_perf_budget_consistency(annex, perf_tile, budget_tile)
        
        self.assertEqual(summary["readiness_status"], "OK")
        self.assertEqual(summary["perf_status"], "BLOCK")
        self.assertEqual(summary["budget_status"], "BLOCK")
        self.assertEqual(summary["consistency_status"], "INCONSISTENT")
        # Should have note about both
        notes_str = " ".join(summary["advisory_notes"]).lower()
        self.assertIn("performance", notes_str)
        self.assertIn("budget", notes_str)
    
    def test_consistency_consistent_when_all_ok(self):
        """Consistency is CONSISTENT when all statuses are OK."""
        annex = {
            "p3_global_norm": 0.8,
            "p4_global_norm": 0.75,
            "poly_fail_detected": False
        }
        perf_tile = {"status_light": StatusLight.GREEN}
        budget_tile = {"status_light": StatusLight.GREEN}
        
        summary = summarize_readiness_perf_budget_consistency(annex, perf_tile, budget_tile)
        
        self.assertEqual(summary["readiness_status"], "OK")
        self.assertEqual(summary["perf_status"], "OK")
        self.assertEqual(summary["budget_status"], "OK")
        self.assertEqual(summary["consistency_status"], "CONSISTENT")
        self.assertEqual(len(summary["advisory_notes"]), 0)
    
    def test_consistency_readiness_warn_when_low_norm(self):
        """Readiness status is WARN when norms are low."""
        annex = {
            "p3_global_norm": 0.4,  # Low but not very low
            "p4_global_norm": 0.4,
            "poly_fail_detected": False
        }
        
        summary = summarize_readiness_perf_budget_consistency(annex, None, None)
        
        self.assertEqual(summary["readiness_status"], "WARN")
    
    def test_consistency_readiness_block_when_very_low_norm(self):
        """Readiness status is BLOCK when norms are very low."""
        annex = {
            "p3_global_norm": 0.2,  # Very low
            "p4_global_norm": 0.2,
            "poly_fail_detected": False
        }
        
        summary = summarize_readiness_perf_budget_consistency(annex, None, None)
        
        self.assertEqual(summary["readiness_status"], "BLOCK")
    
    def test_consistency_readiness_block_when_poly_fail(self):
        """Readiness status is BLOCK when poly_fail_detected."""
        annex = {
            "p3_global_norm": 0.8,
            "p4_global_norm": 0.75,
            "poly_fail_detected": True  # Poly-fail
        }
        
        summary = summarize_readiness_perf_budget_consistency(annex, None, None)
        
        self.assertEqual(summary["readiness_status"], "BLOCK")
    
    def test_consistency_deterministic(self):
        """Consistency summary is deterministic."""
        annex = {
            "p3_global_norm": 0.8,
            "p4_global_norm": 0.75,
            "poly_fail_detected": False
        }
        perf_tile = {"status_light": StatusLight.RED}
        budget_tile = {"status_light": StatusLight.RED}
        
        summaries = [
            summarize_readiness_perf_budget_consistency(annex, perf_tile, budget_tile)
            for _ in range(5)
        ]
        
        json_outputs = [json.dumps(s, sort_keys=True) for s in summaries]
        self.assertTrue(all(j == json_outputs[0] for j in json_outputs))
    
    def test_consistency_neutral_language(self):
        """Advisory notes use neutral language."""
        annex = {
            "p3_global_norm": 0.8,
            "p4_global_norm": 0.75,
            "poly_fail_detected": False
        }
        perf_tile = {"status_light": StatusLight.RED}
        
        summary = summarize_readiness_perf_budget_consistency(annex, perf_tile, None)
        
        for note in summary["advisory_notes"]:
            note_lower = note.lower()
            # Check for absence of value judgment words
            self.assertNotIn("good", note_lower)
            self.assertNotIn("bad", note_lower)
            self.assertNotIn("better", note_lower)
            self.assertNotIn("worse", note_lower)
            self.assertNotIn("success", note_lower)
            self.assertNotIn("failure", note_lower)


class TestEmitCalExpMetricReadinessAnnex(unittest.TestCase):
    """Tests for emit_cal_exp_metric_readiness_annex function."""
    
    def test_annex_shape_and_type_validation(self):
        """Annex has correct shape and types."""
        annex = {
            "p3_global_norm": 0.8,
            "p3_transition_band": "MEDIUM",
            "p4_global_norm": 0.75,
            "p4_transition_band": "HIGH",
            "poly_fail_detected": False
        }
        
        result = emit_cal_exp_metric_readiness_annex("CAL-EXP-1", annex)
        
        self.assertIn("schema_version", result)
        self.assertIn("cal_id", result)
        self.assertEqual(result["cal_id"], "CAL-EXP-1")
        self.assertIsInstance(result["p3_global_norm"], float)
        self.assertIsInstance(result["p3_transition_band"], str)
        self.assertIsInstance(result["p4_global_norm"], float)
        self.assertIsInstance(result["p4_transition_band"], str)
        self.assertIsInstance(result["poly_fail_detected"], bool)
    
    def test_annex_json_round_trip(self):
        """Annex survives JSON round-trip."""
        annex = {
            "p3_global_norm": 0.8,
            "p3_transition_band": "MEDIUM",
            "p4_global_norm": 0.75,
            "p4_transition_band": "HIGH",
            "poly_fail_detected": False
        }
        
        result = emit_cal_exp_metric_readiness_annex("CAL-EXP-1", annex)
        
        json_str = json.dumps(result, sort_keys=True)
        parsed = json.loads(json_str)
        
        self.assertEqual(parsed["cal_id"], result["cal_id"])
        self.assertEqual(parsed["p3_global_norm"], result["p3_global_norm"])
    
    def test_annex_deterministic_with_same_cal_id(self):
        """Annex output is deterministic with same cal_id + annex."""
        annex = {
            "p3_global_norm": 0.8,
            "p3_transition_band": "MEDIUM",
            "p4_global_norm": 0.75,
            "p4_transition_band": "HIGH",
            "poly_fail_detected": False
        }
        
        results = [
            emit_cal_exp_metric_readiness_annex("CAL-EXP-1", annex)
            for _ in range(5)
        ]
        
        json_outputs = [json.dumps(r, sort_keys=True) for r in results]
        self.assertTrue(all(j == json_outputs[0] for j in json_outputs))
    
    def test_annex_file_persistence(self):
        """Annex is persisted to file when output_dir provided."""
        annex = {
            "p3_global_norm": 0.8,
            "p3_transition_band": "MEDIUM",
            "p4_global_norm": 0.75,
            "p4_transition_band": "HIGH",
            "poly_fail_detected": False
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "calibration"
            result = emit_cal_exp_metric_readiness_annex("CAL-EXP-1", annex, output_dir)
            
            # Check file was created
            file_path = output_dir / "metric_readiness_annex_CAL-EXP-1.json"
            self.assertTrue(file_path.exists())
            
            # Check file contents
            with open(file_path, "r") as f:
                file_data = json.load(f)
            
            self.assertEqual(file_data["cal_id"], "CAL-EXP-1")
            self.assertEqual(file_data["p3_global_norm"], 0.8)
    
    def test_annex_non_mutating(self):
        """Annex emission does not mutate input."""
        annex = {
            "p3_global_norm": 0.8,
            "p3_transition_band": "MEDIUM",
            "p4_global_norm": 0.75,
            "p4_transition_band": "HIGH",
            "poly_fail_detected": False
        }
        original_annex = dict(annex)
        
        result = emit_cal_exp_metric_readiness_annex("CAL-EXP-1", annex)
        
        # Original should be unchanged
        self.assertEqual(annex, original_annex)
        # Result should be different (new dict with cal_id)
        self.assertIsNot(result, annex)
        self.assertIn("cal_id", result)


class TestBuildMetricReadinessPanel(unittest.TestCase):
    """Tests for build_metric_readiness_panel function."""
    
    def test_panel_has_required_fields(self):
        """Panel has all required fields."""
        annexes = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_global_norm": 0.8,
                "p4_global_norm": 0.75,
                "poly_fail_detected": False
            },
            {
                "cal_id": "CAL-EXP-2",
                "p3_global_norm": 0.6,
                "p4_global_norm": 0.5,
                "poly_fail_detected": False
            }
        ]
        
        panel = build_metric_readiness_panel(annexes)
        
        self.assertIn("schema_version", panel)
        self.assertIn("num_experiments", panel)
        self.assertIn("num_ok", panel)
        self.assertIn("num_warn", panel)
        self.assertIn("num_block", panel)
        self.assertIn("num_poly_fail", panel)
        self.assertIn("global_norm_range", panel)
    
    def test_panel_aggregation_3_experiments(self):
        """Panel correctly aggregates 3-experiment synthetic set."""
        annexes = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_global_norm": 0.8,  # OK
                "p4_global_norm": 0.75,  # OK
                "poly_fail_detected": False
            },
            {
                "cal_id": "CAL-EXP-2",
                "p3_global_norm": 0.4,  # WARN
                "p4_global_norm": 0.4,  # WARN
                "poly_fail_detected": False
            },
            {
                "cal_id": "CAL-EXP-3",
                "p3_global_norm": 0.2,  # BLOCK
                "p4_global_norm": 0.2,  # BLOCK
                "poly_fail_detected": True  # BLOCK
            }
        ]
        
        panel = build_metric_readiness_panel(annexes)
        
        self.assertEqual(panel["num_experiments"], 3)
        self.assertEqual(panel["num_ok"], 1)
        self.assertEqual(panel["num_warn"], 1)
        self.assertEqual(panel["num_block"], 1)
        self.assertEqual(panel["num_poly_fail"], 1)
    
    def test_panel_norm_range_calculation(self):
        """Panel correctly calculates global_norm_range."""
        annexes = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_global_norm": 0.8,
                "p4_global_norm": 0.75,
                "poly_fail_detected": False
            },
            {
                "cal_id": "CAL-EXP-2",
                "p3_global_norm": 0.4,
                "p4_global_norm": 0.3,
                "poly_fail_detected": False
            }
        ]
        
        panel = build_metric_readiness_panel(annexes)
        
        range_data = panel["global_norm_range"]
        self.assertEqual(range_data["p3_min"], 0.4)
        self.assertEqual(range_data["p3_max"], 0.8)
        self.assertEqual(range_data["p4_min"], 0.3)
        self.assertEqual(range_data["p4_max"], 0.75)
    
    def test_panel_empty_list(self):
        """Panel handles empty annex list."""
        panel = build_metric_readiness_panel([])
        
        self.assertEqual(panel["num_experiments"], 0)
        self.assertEqual(panel["num_ok"], 0)
        self.assertEqual(panel["num_warn"], 0)
        self.assertEqual(panel["num_block"], 0)
        self.assertEqual(panel["num_poly_fail"], 0)
    
    def test_panel_deterministic(self):
        """Panel output is deterministic."""
        annexes = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_global_norm": 0.8,
                "p4_global_norm": 0.75,
                "poly_fail_detected": False
            }
        ]
        
        panels = [
            build_metric_readiness_panel(annexes)
            for _ in range(5)
        ]
        
        json_outputs = [json.dumps(p, sort_keys=True) for p in panels]
        self.assertTrue(all(j == json_outputs[0] for j in json_outputs))
    
    def test_panel_schema_version(self):
        """Panel has correct schema version."""
        annexes = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_global_norm": 0.8,
                "p4_global_norm": 0.75,
                "poly_fail_detected": False
            }
        ]
        
        panel = build_metric_readiness_panel(annexes)
        
        self.assertEqual(panel["schema_version"], READINESS_PANEL_SCHEMA_VERSION)
    
    def test_panel_has_top_driver_cal_ids(self):
        """Panel includes top_driver_cal_ids field."""
        annexes = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_global_norm": 0.8,
                "p4_global_norm": 0.75,
                "poly_fail_detected": False
            }
        ]
        
        panel = build_metric_readiness_panel(annexes)
        
        self.assertIn("top_driver_cal_ids", panel)
        self.assertIsInstance(panel["top_driver_cal_ids"], list)
    
    def test_panel_top_driver_ordering_block_warn_ok(self):
        """Top drivers are ordered by severity: BLOCK > WARN > OK."""
        annexes = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_global_norm": 0.8,  # OK
                "p4_global_norm": 0.75,  # OK
                "poly_fail_detected": False
            },
            {
                "cal_id": "CAL-EXP-2",
                "p3_global_norm": 0.4,  # WARN
                "p4_global_norm": 0.4,  # WARN
                "poly_fail_detected": False
            },
            {
                "cal_id": "CAL-EXP-3",
                "p3_global_norm": 0.2,  # BLOCK
                "p4_global_norm": 0.2,  # BLOCK
                "poly_fail_detected": False
            }
        ]
        
        panel = build_metric_readiness_panel(annexes)
        
        top_drivers = panel["top_driver_cal_ids"]
        self.assertEqual(len(top_drivers), 3)
        # First should be BLOCK (CAL-EXP-3)
        self.assertEqual(top_drivers[0], "CAL-EXP-3")
        # Second should be WARN (CAL-EXP-2)
        self.assertEqual(top_drivers[1], "CAL-EXP-2")
        # Third should be OK (CAL-EXP-1)
        self.assertEqual(top_drivers[2], "CAL-EXP-1")
    
    def test_panel_top_driver_poly_fail_tie_break(self):
        """Top drivers tie-break: poly_fail_detected=True first."""
        annexes = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_global_norm": 0.4,  # WARN
                "p4_global_norm": 0.4,  # WARN
                "poly_fail_detected": False
            },
            {
                "cal_id": "CAL-EXP-2",
                "p3_global_norm": 0.4,  # WARN (same norm)
                "p4_global_norm": 0.4,  # WARN (same norm)
                "poly_fail_detected": True  # Poly-fail breaks tie
            }
        ]
        
        panel = build_metric_readiness_panel(annexes)
        
        top_drivers = panel["top_driver_cal_ids"]
        # CAL-EXP-2 should come first due to poly_fail
        self.assertEqual(top_drivers[0], "CAL-EXP-2")
        self.assertEqual(top_drivers[1], "CAL-EXP-1")
    
    def test_panel_top_driver_lowest_p4_norm_tie_break(self):
        """Top drivers tie-break: lowest p4_global_norm when same status and no poly-fail."""
        annexes = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_global_norm": 0.4,  # WARN
                "p4_global_norm": 0.5,  # Higher norm
                "poly_fail_detected": False
            },
            {
                "cal_id": "CAL-EXP-2",
                "p3_global_norm": 0.4,  # WARN (same status)
                "p4_global_norm": 0.3,  # Lower norm (worse)
                "poly_fail_detected": False
            }
        ]
        
        panel = build_metric_readiness_panel(annexes)
        
        top_drivers = panel["top_driver_cal_ids"]
        # CAL-EXP-2 should come first due to lower p4_norm
        self.assertEqual(top_drivers[0], "CAL-EXP-2")
        self.assertEqual(top_drivers[1], "CAL-EXP-1")
    
    def test_panel_top_driver_max_3(self):
        """Top drivers limited to maximum 3."""
        annexes = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_global_norm": 0.8,
                "p4_global_norm": 0.75,
                "poly_fail_detected": False
            },
            {
                "cal_id": "CAL-EXP-2",
                "p3_global_norm": 0.6,
                "p4_global_norm": 0.5,
                "poly_fail_detected": False
            },
            {
                "cal_id": "CAL-EXP-3",
                "p3_global_norm": 0.4,
                "p4_global_norm": 0.3,
                "poly_fail_detected": False
            },
            {
                "cal_id": "CAL-EXP-4",
                "p3_global_norm": 0.2,
                "p4_global_norm": 0.1,
                "poly_fail_detected": False
            }
        ]
        
        panel = build_metric_readiness_panel(annexes)
        
        top_drivers = panel["top_driver_cal_ids"]
        self.assertLessEqual(len(top_drivers), 3)
    
    def test_panel_top_driver_deterministic(self):
        """Top driver ordering is deterministic."""
        annexes = [
            {
                "cal_id": "CAL-EXP-1",
                "p3_global_norm": 0.8,
                "p4_global_norm": 0.75,
                "poly_fail_detected": False
            },
            {
                "cal_id": "CAL-EXP-2",
                "p3_global_norm": 0.4,
                "p4_global_norm": 0.3,
                "poly_fail_detected": False
            }
        ]
        
        panels = [
            build_metric_readiness_panel(annexes)
            for _ in range(5)
        ]
        
        # All panels should have same top_driver_cal_ids
        first_top_drivers = panels[0]["top_driver_cal_ids"]
        for panel in panels[1:]:
            self.assertEqual(panel["top_driver_cal_ids"], first_top_drivers)
    
    def test_panel_top_driver_empty_list(self):
        """Top drivers is empty list when no annexes."""
        panel = build_metric_readiness_panel([])
        
        self.assertEqual(panel["top_driver_cal_ids"], [])


class TestAttachMetricReadinessPanelToEvidence(unittest.TestCase):
    """Tests for attach_metric_readiness_panel_to_evidence function."""
    
    def test_evidence_attach_panel_shape(self):
        """Panel attachment has correct shape."""
        evidence = {"governance": {}}
        panel = {
            "schema_version": "1.0.0",
            "num_experiments": 3,
            "num_ok": 1,
            "num_warn": 1,
            "num_block": 1,
            "num_poly_fail": 0,
            "global_norm_range": {}
        }
        
        result = attach_metric_readiness_panel_to_evidence(evidence, panel)
        
        self.assertIn("governance", result)
        self.assertIn("metric_readiness_panel", result["governance"])
        self.assertEqual(result["governance"]["metric_readiness_panel"]["num_experiments"], 3)
    
    def test_evidence_attach_panel_json_safe(self):
        """Panel attachment is JSON-safe."""
        evidence = {"governance": {}}
        panel = {
            "schema_version": "1.0.0",
            "num_experiments": 2,
            "num_ok": 1,
            "num_warn": 1,
            "num_block": 0,
            "num_poly_fail": 0,
            "global_norm_range": {}
        }
        
        result = attach_metric_readiness_panel_to_evidence(evidence, panel)
        
        # Should serialize without error
        json_str = json.dumps(result, sort_keys=True)
        parsed = json.loads(json_str)
        self.assertIn("metric_readiness_panel", parsed["governance"])
    
    def test_evidence_attach_panel_non_mutating(self):
        """Panel attachment does not mutate input."""
        evidence = {"governance": {}}
        panel = {
            "schema_version": "1.0.0",
            "num_experiments": 1,
            "num_ok": 1,
            "num_warn": 0,
            "num_block": 0,
            "num_poly_fail": 0,
            "global_norm_range": {}
        }
        
        original_evidence = dict(evidence)
        result = attach_metric_readiness_panel_to_evidence(evidence, panel)
        
        # Original should be unchanged
        self.assertEqual(evidence, original_evidence)
        # Result should be different (new dict)
        self.assertIsNot(result, evidence)
        # Result should have panel
        self.assertIn("metric_readiness_panel", result["governance"])


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)

