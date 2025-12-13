"""
Tests for Convergence Pressure Evidence Pack Integration.

Tests cover:
- attach_convergence_pressure_to_evidence() helper
- summarize_convergence_for_uplift_council() function
- Shape validation, determinism, non-mutation, JSON round-trip
"""

import json
import unittest
from typing import Dict, Any

from backend.health.convergence_pressure_adapter import (
    attach_convergence_pressure_to_evidence,
    summarize_convergence_for_uplift_council,
    build_convergence_pressure_tile,
)


class TestAttachConvergencePressureToEvidence(unittest.TestCase):
    """Tests for attach_convergence_pressure_to_evidence()."""

    def _make_mock_tile(self) -> Dict[str, Any]:
        """Create mock convergence pressure tile."""
        return {
            "schema_version": "1.0.0",
            "tile_type": "convergence_pressure",
            "status_light": "YELLOW",
            "global_pressure_norm": 1.5,
            "transition_likelihood_band": "MEDIUM",
            "slices_at_risk": ["slice_a", "slice_b"],
            "pressure_drivers": [
                "Global pressure norm elevated (1.50)",
                "Phase boundary forecast confidence moderate (50.0%)",
            ],
            "headline": "Convergence status: pressure norm: 1.50, transition likelihood: medium, 2 slice(s) at elevated risk",
        }

    def test_evidence_shape(self) -> None:
        """Test that evidence pack has correct shape after attachment."""
        evidence = {"timestamp": "2024-01-01", "data": {"test": "value"}}
        tile = self._make_mock_tile()
        early_warning = {"time_to_inflection_estimate": 5}

        enriched = attach_convergence_pressure_to_evidence(evidence, tile, early_warning)

        # Verify governance section exists
        self.assertIn("governance", enriched)
        self.assertIn("convergence_pressure", enriched["governance"])

        # Verify convergence_pressure structure
        cp = enriched["governance"]["convergence_pressure"]
        self.assertIn("global_pressure_norm", cp)
        self.assertIn("transition_likelihood_band", cp)
        self.assertIn("slices_at_risk", cp)
        self.assertIn("pressure_drivers", cp)
        self.assertIn("horizon_estimate", cp)

        # Verify values
        self.assertEqual(cp["global_pressure_norm"], 1.5)
        self.assertEqual(cp["transition_likelihood_band"], "MEDIUM")
        self.assertEqual(cp["slices_at_risk"], ["slice_a", "slice_b"])
        self.assertEqual(len(cp["pressure_drivers"]), 2)
        self.assertEqual(cp["horizon_estimate"], 5)

    def test_evidence_determinism(self) -> None:
        """Test that evidence attachment is deterministic."""
        evidence = {"timestamp": "2024-01-01", "data": {"test": "value"}}
        tile = self._make_mock_tile()
        early_warning = {"time_to_inflection_estimate": 5}

        enriched1 = attach_convergence_pressure_to_evidence(evidence, tile, early_warning)
        enriched2 = attach_convergence_pressure_to_evidence(evidence, tile, early_warning)

        # Serialize and compare
        json1 = json.dumps(enriched1, sort_keys=True)
        json2 = json.dumps(enriched2, sort_keys=True)

        self.assertEqual(json1, json2, "Evidence attachment should be deterministic")

    def test_evidence_non_mutation(self) -> None:
        """Test that input evidence is not modified."""
        evidence = {"timestamp": "2024-01-01", "data": {"test": "value"}}
        tile = self._make_mock_tile()
        early_warning = {"time_to_inflection_estimate": 5}

        # Create copies to verify inputs are not modified
        evidence_copy = json.loads(json.dumps(evidence))
        tile_copy = json.loads(json.dumps(tile))
        early_warning_copy = json.loads(json.dumps(early_warning))

        enriched = attach_convergence_pressure_to_evidence(evidence, tile, early_warning)

        # Verify inputs unchanged
        self.assertEqual(
            json.dumps(evidence, sort_keys=True),
            json.dumps(evidence_copy, sort_keys=True),
            "evidence should not be modified",
        )
        self.assertEqual(
            json.dumps(tile, sort_keys=True),
            json.dumps(tile_copy, sort_keys=True),
            "tile should not be modified",
        )
        self.assertEqual(
            json.dumps(early_warning, sort_keys=True),
            json.dumps(early_warning_copy, sort_keys=True),
            "early_warning should not be modified",
        )

        # Verify enriched is a new dict (not a reference)
        self.assertIsNot(enriched, evidence)
        self.assertIsNot(enriched["governance"], evidence.get("governance"))

    def test_evidence_json_round_trip(self) -> None:
        """Test that enriched evidence can be JSON serialized and deserialized."""
        evidence = {"timestamp": "2024-01-01", "data": {"test": "value"}}
        tile = self._make_mock_tile()
        early_warning = {"time_to_inflection_estimate": 5}

        enriched = attach_convergence_pressure_to_evidence(evidence, tile, early_warning)

        # Serialize
        json_str = json.dumps(enriched, sort_keys=True)
        self.assertIsInstance(json_str, str)
        self.assertGreater(len(json_str), 0)

        # Deserialize
        parsed = json.loads(json_str)
        self.assertIsInstance(parsed, dict)
        self.assertIn("governance", parsed)
        self.assertIn("convergence_pressure", parsed["governance"])

        # Verify structure preserved
        cp = parsed["governance"]["convergence_pressure"]
        self.assertEqual(cp["global_pressure_norm"], 1.5)
        self.assertEqual(cp["transition_likelihood_band"], "MEDIUM")

    def test_evidence_slices_sorted(self) -> None:
        """Test that slices_at_risk are sorted for determinism."""
        evidence = {"timestamp": "2024-01-01"}
        tile = self._make_mock_tile()
        # Use unsorted slices
        tile["slices_at_risk"] = ["slice_b", "slice_a", "slice_c"]
        early_warning = {}

        enriched = attach_convergence_pressure_to_evidence(evidence, tile, early_warning)

        slices = enriched["governance"]["convergence_pressure"]["slices_at_risk"]
        self.assertEqual(slices, ["slice_a", "slice_b", "slice_c"])

    def test_evidence_empty_governance(self) -> None:
        """Test that function works when evidence has no governance section."""
        evidence = {"timestamp": "2024-01-01"}
        tile = self._make_mock_tile()
        early_warning = {}

        enriched = attach_convergence_pressure_to_evidence(evidence, tile, early_warning)

        self.assertIn("governance", enriched)
        self.assertIn("convergence_pressure", enriched["governance"])

    def test_evidence_existing_governance(self) -> None:
        """Test that function preserves existing governance data."""
        evidence = {
            "timestamp": "2024-01-01",
            "governance": {"other_tile": {"status": "OK"}},
        }
        tile = self._make_mock_tile()
        early_warning = {}

        enriched = attach_convergence_pressure_to_evidence(evidence, tile, early_warning)

        # Verify existing governance preserved
        self.assertIn("other_tile", enriched["governance"])
        self.assertEqual(enriched["governance"]["other_tile"]["status"], "OK")

        # Verify convergence_pressure added
        self.assertIn("convergence_pressure", enriched["governance"])


class TestSummarizeConvergenceForUpliftCouncil(unittest.TestCase):
    """Tests for summarize_convergence_for_uplift_council()."""

    def test_council_summary_structure(self) -> None:
        """Test that council summary has correct structure."""
        tile = {
            "global_pressure_norm": 1.5,
            "transition_likelihood_band": "MEDIUM",
            "slices_at_risk": ["slice_a", "slice_b"],
        }

        summary = summarize_convergence_for_uplift_council(tile)

        self.assertIn("status", summary)
        self.assertIn("slices_at_risk", summary)
        self.assertIn("band", summary)

    def test_council_summary_block_status(self) -> None:
        """Test that BLOCK status is returned for HIGH band."""
        tile = {
            "global_pressure_norm": 1.5,
            "transition_likelihood_band": "HIGH",
            "slices_at_risk": ["slice_a"],
        }

        summary = summarize_convergence_for_uplift_council(tile)

        self.assertEqual(summary["status"], "BLOCK")
        self.assertEqual(summary["band"], "HIGH")

    def test_council_summary_block_status_high_pressure(self) -> None:
        """Test that BLOCK status is returned for pressure > 2.0."""
        tile = {
            "global_pressure_norm": 2.5,
            "transition_likelihood_band": "MEDIUM",
            "slices_at_risk": ["slice_a"],
        }

        summary = summarize_convergence_for_uplift_council(tile)

        self.assertEqual(summary["status"], "BLOCK")

    def test_council_summary_warn_status(self) -> None:
        """Test that WARN status is returned for MEDIUM band."""
        tile = {
            "global_pressure_norm": 1.5,
            "transition_likelihood_band": "MEDIUM",
            "slices_at_risk": ["slice_a"],
        }

        summary = summarize_convergence_for_uplift_council(tile)

        self.assertEqual(summary["status"], "WARN")
        self.assertEqual(summary["band"], "MEDIUM")

    def test_council_summary_ok_status(self) -> None:
        """Test that OK status is returned for LOW band."""
        tile = {
            "global_pressure_norm": 0.5,
            "transition_likelihood_band": "LOW",
            "slices_at_risk": [],
        }

        summary = summarize_convergence_for_uplift_council(tile)

        self.assertEqual(summary["status"], "OK")
        self.assertEqual(summary["band"], "LOW")

    def test_council_summary_slices_sorted(self) -> None:
        """Test that slices_at_risk are sorted for determinism."""
        tile = {
            "global_pressure_norm": 1.5,
            "transition_likelihood_band": "MEDIUM",
            "slices_at_risk": ["slice_b", "slice_a", "slice_c"],
        }

        summary = summarize_convergence_for_uplift_council(tile)

        self.assertEqual(summary["slices_at_risk"], ["slice_a", "slice_b", "slice_c"])

    def test_council_summary_determinism(self) -> None:
        """Test that council summary is deterministic."""
        tile = {
            "global_pressure_norm": 1.5,
            "transition_likelihood_band": "MEDIUM",
            "slices_at_risk": ["slice_a", "slice_b"],
        }

        summary1 = summarize_convergence_for_uplift_council(tile)
        summary2 = summarize_convergence_for_uplift_council(tile)

        self.assertEqual(
            json.dumps(summary1, sort_keys=True),
            json.dumps(summary2, sort_keys=True),
        )

    def test_council_summary_json_serializable(self) -> None:
        """Test that council summary is JSON-serializable."""
        tile = {
            "global_pressure_norm": 1.5,
            "transition_likelihood_band": "MEDIUM",
            "slices_at_risk": ["slice_a"],
        }

        summary = summarize_convergence_for_uplift_council(tile)

        json_str = json.dumps(summary, sort_keys=True)
        parsed = json.loads(json_str)

        self.assertEqual(parsed["status"], "WARN")
        self.assertEqual(parsed["band"], "MEDIUM")


class TestConvergencePressureIntegration(unittest.TestCase):
    """Integration tests for convergence pressure evidence and council."""

    def test_end_to_end_evidence_and_council(self) -> None:
        """Test end-to-end flow: tile → evidence → council."""
        # Create mock inputs
        pressure_tensor = {
            "global_pressure_norm": 1.5,
            "pressure_ranked_slices": ["slice_a", "slice_b"],
            "slice_pressure_vectors": {},
        }
        early_warning = {
            "transition_likelihood_band": "MEDIUM",
            "root_drivers": ["Driver 1", "Driver 2"],
            "first_slices_at_risk": ["slice_a", "slice_b"],
        }
        director_tile = {"status_light": "YELLOW"}

        # Build tile
        tile = build_convergence_pressure_tile(
            pressure_tensor=pressure_tensor,
            early_warning=early_warning,
            director_tile=director_tile,
        )

        # Attach to evidence
        evidence = {"timestamp": "2024-01-01"}
        enriched = attach_convergence_pressure_to_evidence(evidence, tile, early_warning)

        # Summarize for council
        council_summary = summarize_convergence_for_uplift_council(tile)

        # Verify evidence structure
        self.assertIn("governance", enriched)
        self.assertIn("convergence_pressure", enriched["governance"])

        # Verify council summary
        self.assertEqual(council_summary["status"], "WARN")
        self.assertEqual(council_summary["band"], "MEDIUM")
        self.assertGreater(len(council_summary["slices_at_risk"]), 0)


if __name__ == "__main__":
    unittest.main()

