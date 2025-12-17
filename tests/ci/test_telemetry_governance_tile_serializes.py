"""
PHASE X — Telemetry Governance Tile CI Serialization Tests

Tests for telemetry governance tile serialization, determinism, JSON compatibility,
neutrality, and SHADOW-mode constraints.

Author: metrics-engineer-4 (Agent D4)
"""

from __future__ import annotations

import json
import unittest
from typing import Any, Dict

import pytest

from backend.health.telemetry_fusion_adapter import (
    build_telemetry_governance_tile,
    extract_telemetry_signal_for_first_light,
)


# -----------------------------------------------------------------------------
# Test: Telemetry Governance Tile Serialization
# -----------------------------------------------------------------------------

class TestTelemetryGovernanceTileSerializes(unittest.TestCase):
    """Tests for telemetry governance tile serialization and determinism."""

    def test_tile_is_json_serializable(self):
        """Telemetry governance tile should be JSON serializable."""
        fusion_tile = {
            "fusion_band": "LOW",
            "fusion_risk_score": 0.2,
            "incoherence_vectors": [],
            "neutral_notes": [],
        }
        uplift_gate = {
            "uplift_gate_status": "OK",
            "drivers": [],
            "recommended_hold_slices": [],
            "headline": "Telemetry indicates uplift analysis can proceed.",
        }
        director_tile_v2 = {
            "status_light": "GREEN",
            "fusion_band": "LOW",
            "uplift_gate_status": "OK",
            "telemetry_ok": True,
            "headline": "Telemetry systems are fully operational and aligned.",
        }
        
        tile = build_telemetry_governance_tile(
            fusion_tile=fusion_tile,
            uplift_gate=uplift_gate,
            director_tile_v2=director_tile_v2,
        )
        
        # Should not raise exception
        json_str = json.dumps(tile)
        self.assertIsInstance(json_str, str)
        
        # Should be able to deserialize
        deserialized = json.loads(json_str)
        self.assertEqual(deserialized["schema_version"], "1.0.0")
        self.assertEqual(deserialized["status_light"], "GREEN")

    def test_tile_is_deterministic(self):
        """Telemetry governance tile should be deterministic for fixed inputs."""
        fusion_tile = {
            "fusion_band": "MEDIUM",
            "fusion_risk_score": 0.5,
            "incoherence_vectors": ["vector1", "vector2"],
            "neutral_notes": ["Note 1"],
        }
        uplift_gate = {
            "uplift_gate_status": "ATTENTION",
            "drivers": ["Driver 1"],
            "recommended_hold_slices": [],
            "headline": "Uplift analysis ATTENTION.",
        }
        director_tile_v2 = {
            "status_light": "YELLOW",
            "fusion_band": "MEDIUM",
            "uplift_gate_status": "ATTENTION",
            "telemetry_ok": True,
            "headline": "Telemetry systems show warnings.",
        }
        
        tile1 = build_telemetry_governance_tile(
            fusion_tile=fusion_tile,
            uplift_gate=uplift_gate,
            director_tile_v2=director_tile_v2,
        )
        tile2 = build_telemetry_governance_tile(
            fusion_tile=fusion_tile,
            uplift_gate=uplift_gate,
            director_tile_v2=director_tile_v2,
        )
        
        # Should produce identical results
        self.assertEqual(tile1, tile2)
        self.assertEqual(json.dumps(tile1, sort_keys=True), json.dumps(tile2, sort_keys=True))

    def test_tile_contains_required_fields(self):
        """Telemetry governance tile should contain all required fields."""
        fusion_tile = {
            "fusion_band": "HIGH",
            "fusion_risk_score": 0.8,
            "incoherence_vectors": ["vector1"],
            "neutral_notes": [],
        }
        uplift_gate = {
            "uplift_gate_status": "BLOCK",
            "drivers": ["High fusion risk"],
            "recommended_hold_slices": ["slice1"],
            "headline": "Uplift analysis BLOCKED.",
        }
        director_tile_v2 = {
            "status_light": "RED",
            "fusion_band": "HIGH",
            "uplift_gate_status": "BLOCK",
            "telemetry_ok": False,
            "headline": "Critical telemetry issues detected.",
        }
        
        tile = build_telemetry_governance_tile(
            fusion_tile=fusion_tile,
            uplift_gate=uplift_gate,
            director_tile_v2=director_tile_v2,
        )
        
        # Check required fields
        self.assertIn("schema_version", tile)
        self.assertIn("status_light", tile)
        self.assertIn("fusion_band", tile)
        self.assertIn("uplift_gate_status", tile)
        self.assertIn("telemetry_ok", tile)
        self.assertIn("incoherence_vectors", tile)
        self.assertIn("headline", tile)
        
        # Check schema version
        self.assertEqual(tile["schema_version"], "1.0.0")

    def test_tile_is_neutral(self):
        """Telemetry governance tile should use neutral language."""
        fusion_tile = {
            "fusion_band": "MEDIUM",
            "fusion_risk_score": 0.5,
            "incoherence_vectors": ["telemetry:moderate_drift_band"],
            "neutral_notes": ["Neutral note about drift"],
        }
        uplift_gate = {
            "uplift_gate_status": "ATTENTION",
            "drivers": ["Moderate fusion risk detected"],
            "recommended_hold_slices": [],
            "headline": "Uplift analysis may proceed with attention.",
        }
        director_tile_v2 = {
            "status_light": "YELLOW",
            "fusion_band": "MEDIUM",
            "uplift_gate_status": "ATTENTION",
            "telemetry_ok": True,
            "headline": "Telemetry systems show warnings; proceed with caution.",
        }
        
        tile = build_telemetry_governance_tile(
            fusion_tile=fusion_tile,
            uplift_gate=uplift_gate,
            director_tile_v2=director_tile_v2,
        )
        
        # Headline should be descriptive, not judgmental
        headline = tile["headline"]
        self.assertIsInstance(headline, str)
        # Should not contain overly emotional language
        self.assertNotIn("CRITICAL FAILURE", headline.upper())
        self.assertNotIn("DISASTER", headline.upper())

    def test_tile_with_telemetry_health(self):
        """Telemetry governance tile should use telemetry_health when provided."""
        fusion_tile = {
            "fusion_band": "LOW",
            "fusion_risk_score": 0.1,
            "incoherence_vectors": [],
            "neutral_notes": [],
        }
        uplift_gate = {
            "uplift_gate_status": "OK",
            "drivers": [],
            "recommended_hold_slices": [],
            "headline": "OK",
        }
        director_tile_v2 = {
            "status_light": "GREEN",
            "fusion_band": "LOW",
            "uplift_gate_status": "OK",
            "telemetry_ok": True,
            "headline": "OK",
        }
        telemetry_health = {
            "telemetry_ok": False,  # Override director_tile_v2
            "status": "WARN",
        }
        
        tile = build_telemetry_governance_tile(
            fusion_tile=fusion_tile,
            uplift_gate=uplift_gate,
            director_tile_v2=director_tile_v2,
            telemetry_health=telemetry_health,
        )
        
        # Should use telemetry_health value
        self.assertFalse(tile["telemetry_ok"])

    def test_tile_without_telemetry_health(self):
        """Telemetry governance tile should use director_tile_v2 when telemetry_health not provided."""
        fusion_tile = {
            "fusion_band": "LOW",
            "fusion_risk_score": 0.1,
            "incoherence_vectors": [],
            "neutral_notes": [],
        }
        uplift_gate = {
            "uplift_gate_status": "OK",
            "drivers": [],
            "recommended_hold_slices": [],
            "headline": "OK",
        }
        director_tile_v2 = {
            "status_light": "GREEN",
            "fusion_band": "LOW",
            "uplift_gate_status": "OK",
            "telemetry_ok": True,
            "headline": "OK",
        }
        
        tile = build_telemetry_governance_tile(
            fusion_tile=fusion_tile,
            uplift_gate=uplift_gate,
            director_tile_v2=director_tile_v2,
        )
        
        # Should use director_tile_v2 value
        self.assertTrue(tile["telemetry_ok"])


# -----------------------------------------------------------------------------
# Test: First Light Signal Extraction
# -----------------------------------------------------------------------------

class TestExtractTelemetrySignalForFirstLight(unittest.TestCase):
    """Tests for First Light signal extraction."""

    def test_extract_signal_is_json_serializable(self):
        """First Light signal should be JSON serializable."""
        fusion_tile = {
            "fusion_band": "LOW",
            "incoherence_vectors": [],
        }
        uplift_gate = {
            "uplift_gate_status": "OK",
        }
        
        signal = extract_telemetry_signal_for_first_light(
            fusion_tile=fusion_tile,
            uplift_gate=uplift_gate,
        )
        
        # Should not raise exception
        json_str = json.dumps(signal)
        self.assertIsInstance(json_str, str)
        
        # Should be able to deserialize
        deserialized = json.loads(json_str)
        self.assertIn("fusion_band", deserialized)
        self.assertIn("uplift_gate_status", deserialized)
        self.assertIn("num_incoherence_vectors", deserialized)

    def test_extract_signal_contains_required_fields(self):
        """First Light signal should contain all required fields."""
        fusion_tile = {
            "fusion_band": "HIGH",
            "incoherence_vectors": ["vector1", "vector2", "vector3"],
        }
        uplift_gate = {
            "uplift_gate_status": "BLOCK",
        }
        
        signal = extract_telemetry_signal_for_first_light(
            fusion_tile=fusion_tile,
            uplift_gate=uplift_gate,
        )
        
        self.assertEqual(signal["fusion_band"], "HIGH")
        self.assertEqual(signal["uplift_gate_status"], "BLOCK")
        self.assertEqual(signal["num_incoherence_vectors"], 3)

    def test_extract_signal_is_deterministic(self):
        """First Light signal should be deterministic for fixed inputs."""
        fusion_tile = {
            "fusion_band": "MEDIUM",
            "incoherence_vectors": ["vector1"],
        }
        uplift_gate = {
            "uplift_gate_status": "ATTENTION",
        }
        
        signal1 = extract_telemetry_signal_for_first_light(
            fusion_tile=fusion_tile,
            uplift_gate=uplift_gate,
        )
        signal2 = extract_telemetry_signal_for_first_light(
            fusion_tile=fusion_tile,
            uplift_gate=uplift_gate,
        )
        
        # Should produce identical results
        self.assertEqual(signal1, signal2)

    def test_extract_signal_handles_missing_incoherence_vectors(self):
        """First Light signal should handle missing incoherence_vectors gracefully."""
        fusion_tile = {
            "fusion_band": "LOW",
            # Missing incoherence_vectors
        }
        uplift_gate = {
            "uplift_gate_status": "OK",
        }
        
        signal = extract_telemetry_signal_for_first_light(
            fusion_tile=fusion_tile,
            uplift_gate=uplift_gate,
        )
        
        self.assertEqual(signal["num_incoherence_vectors"], 0)

    def test_extract_signal_handles_non_list_incoherence_vectors(self):
        """First Light signal should handle non-list incoherence_vectors gracefully."""
        fusion_tile = {
            "fusion_band": "LOW",
            "incoherence_vectors": "not_a_list",  # Wrong type
        }
        uplift_gate = {
            "uplift_gate_status": "OK",
        }
        
        signal = extract_telemetry_signal_for_first_light(
            fusion_tile=fusion_tile,
            uplift_gate=uplift_gate,
        )
        
        # Should default to 0
        self.assertEqual(signal["num_incoherence_vectors"], 0)


# -----------------------------------------------------------------------------
# Test: SHADOW Mode Constraints
# -----------------------------------------------------------------------------

class TestShadowModeConstraints(unittest.TestCase):
    """Tests for SHADOW mode constraints (read-only, no side effects)."""

    def test_governance_tile_is_read_only(self):
        """build_telemetry_governance_tile should not modify input dictionaries."""
        fusion_tile = {
            "fusion_band": "LOW",
            "fusion_risk_score": 0.2,
            "incoherence_vectors": [],
            "neutral_notes": [],
        }
        uplift_gate = {
            "uplift_gate_status": "OK",
            "drivers": [],
            "recommended_hold_slices": [],
            "headline": "OK",
        }
        director_tile_v2 = {
            "status_light": "GREEN",
            "fusion_band": "LOW",
            "uplift_gate_status": "OK",
            "telemetry_ok": True,
            "headline": "OK",
        }
        
        # Create copies to compare
        fusion_tile_copy = fusion_tile.copy()
        uplift_gate_copy = uplift_gate.copy()
        director_tile_v2_copy = director_tile_v2.copy()
        
        # Build tile
        tile = build_telemetry_governance_tile(
            fusion_tile=fusion_tile,
            uplift_gate=uplift_gate,
            director_tile_v2=director_tile_v2,
        )
        
        # Input dictionaries should be unchanged
        self.assertEqual(fusion_tile, fusion_tile_copy)
        self.assertEqual(uplift_gate, uplift_gate_copy)
        self.assertEqual(director_tile_v2, director_tile_v2_copy)
        
        # Tile should be a new dictionary
        self.assertIsNot(tile, fusion_tile)
        self.assertIsNot(tile, uplift_gate)
        self.assertIsNot(tile, director_tile_v2)

    def test_extract_signal_is_read_only(self):
        """extract_telemetry_signal_for_first_light should not modify input dictionaries."""
        fusion_tile = {
            "fusion_band": "LOW",
            "incoherence_vectors": ["vector1"],
        }
        uplift_gate = {
            "uplift_gate_status": "OK",
        }
        
        # Create copies to compare
        fusion_tile_copy = fusion_tile.copy()
        uplift_gate_copy = uplift_gate.copy()
        
        # Extract signal
        signal = extract_telemetry_signal_for_first_light(
            fusion_tile=fusion_tile,
            uplift_gate=uplift_gate,
        )
        
        # Input dictionaries should be unchanged
        self.assertEqual(fusion_tile, fusion_tile_copy)
        self.assertEqual(uplift_gate, uplift_gate_copy)
        
        # Signal should be a new dictionary
        self.assertIsNot(signal, fusion_tile)
        self.assertIsNot(signal, uplift_gate)


if __name__ == "__main__":
    unittest.main()

