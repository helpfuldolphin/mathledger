"""
CI tests for topology pressure tile serialization.

Tests verify that topology pressure governance tiles:
- Have required fields
- Are deterministic
- Use neutral language
- Serialize to JSON correctly
"""

import json
import unittest
from typing import Dict, Any

from backend.health.topology_pressure_adapter import (
    TOPOLOGY_PRESSURE_TILE_SCHEMA_VERSION,
    build_topology_pressure_governance_tile,
)


class TestTopologyPressureTileSerialization(unittest.TestCase):
    """Tests for topology pressure tile serialization."""

    def _make_pressure_field(
        self,
        pressure_band: str = "LOW",
        slice_pressure: float = 0.3,
    ) -> Dict[str, Any]:
        """Helper to create mock pressure field."""
        return {
            "pressure_band": pressure_band,
            "slice_pressure": slice_pressure,
            "pressure_components": {
                "depth": 0.3,
                "branching": 0.3,
                "risk": 0.3,
            },
            "neutral_notes": [],
        }

    def _make_promotion_gate(
        self,
        promotion_status: str = "OK",
    ) -> Dict[str, Any]:
        """Helper to create mock promotion gate."""
        return {
            "promotion_status": promotion_status,
            "explanations": [],
            "slices_at_risk": [],
        }

    def _make_console_tile(
        self,
        status_light: str = "GREEN",
        headline: str = "Topology status: stable",
    ) -> Dict[str, Any]:
        """Helper to create mock console tile."""
        return {
            "status_light": status_light,
            "promotion_gate": "OK",
            "pressure_hotspots": [],
            "headline": headline,
        }

    def test_01_required_fields_present(self):
        """Test tile has all required fields."""
        pressure_field = self._make_pressure_field()
        promotion_gate = self._make_promotion_gate()
        console_tile = self._make_console_tile()

        tile = build_topology_pressure_governance_tile(
            pressure_field, promotion_gate, console_tile
        )

        required_fields = [
            "schema_version",
            "status_light",
            "pressure_band",
            "promotion_gate_status",
            "pressure_hotspots",
            "headline",
        ]
        for field in required_fields:
            self.assertIn(field, tile, f"Missing required field: {field}")

    def test_02_schema_version_correct(self):
        """Test schema version is correct."""
        pressure_field = self._make_pressure_field()
        promotion_gate = self._make_promotion_gate()
        console_tile = self._make_console_tile()

        tile = build_topology_pressure_governance_tile(
            pressure_field, promotion_gate, console_tile
        )

        self.assertEqual(
            tile["schema_version"],
            TOPOLOGY_PRESSURE_TILE_SCHEMA_VERSION,
            "Schema version mismatch",
        )

    def test_03_status_light_valid(self):
        """Test status_light is valid."""
        pressure_field = self._make_pressure_field()
        promotion_gate = self._make_promotion_gate()

        for status_light in ["GREEN", "YELLOW", "RED"]:
            console_tile = self._make_console_tile(status_light=status_light)
            tile = build_topology_pressure_governance_tile(
                pressure_field, promotion_gate, console_tile
            )
            self.assertIn(
                tile["status_light"],
                ["GREEN", "YELLOW", "RED"],
                f"Invalid status_light: {tile['status_light']}",
            )

    def test_04_pressure_band_valid(self):
        """Test pressure_band is valid."""
        promotion_gate = self._make_promotion_gate()
        console_tile = self._make_console_tile()

        for pressure_band in ["LOW", "MEDIUM", "HIGH"]:
            pressure_field = self._make_pressure_field(pressure_band=pressure_band)
            tile = build_topology_pressure_governance_tile(
                pressure_field, promotion_gate, console_tile
            )
            self.assertIn(
                tile["pressure_band"],
                ["LOW", "MEDIUM", "HIGH"],
                f"Invalid pressure_band: {tile['pressure_band']}",
            )

    def test_05_promotion_gate_status_valid(self):
        """Test promotion_gate_status is valid."""
        pressure_field = self._make_pressure_field()
        console_tile = self._make_console_tile()

        for promotion_status in ["OK", "ATTENTION", "BLOCK"]:
            promotion_gate = self._make_promotion_gate(
                promotion_status=promotion_status
            )
            tile = build_topology_pressure_governance_tile(
                pressure_field, promotion_gate, console_tile
            )
            self.assertIn(
                tile["promotion_gate_status"],
                ["OK", "ATTENTION", "BLOCK"],
                f"Invalid promotion_gate_status: {tile['promotion_gate_status']}",
            )

    def test_06_deterministic_output(self):
        """Test tile output is deterministic."""
        pressure_field = self._make_pressure_field()
        promotion_gate = self._make_promotion_gate()
        console_tile = self._make_console_tile()

        tile1 = build_topology_pressure_governance_tile(
            pressure_field, promotion_gate, console_tile
        )
        tile2 = build_topology_pressure_governance_tile(
            pressure_field, promotion_gate, console_tile
        )

        # Compare as JSON strings for exact match
        json1 = json.dumps(tile1, sort_keys=True)
        json2 = json.dumps(tile2, sort_keys=True)

        self.assertEqual(
            json1, json2, "Tile output is not deterministic"
        )

    def test_07_json_serializable(self):
        """Test tile is JSON serializable."""
        pressure_field = self._make_pressure_field()
        promotion_gate = self._make_promotion_gate()
        console_tile = self._make_console_tile()

        tile = build_topology_pressure_governance_tile(
            pressure_field, promotion_gate, console_tile
        )

        # Should not raise
        json_str = json.dumps(tile, sort_keys=True)
        self.assertIsInstance(json_str, str)

        # Should deserialize correctly
        deserialized = json.loads(json_str)
        self.assertEqual(deserialized, tile)

    def test_08_neutral_language(self):
        """Test headline uses neutral language."""
        pressure_field = self._make_pressure_field()
        promotion_gate = self._make_promotion_gate()
        console_tile = self._make_console_tile(
            headline="Topology pressure status: stable depth trend"
        )

        tile = build_topology_pressure_governance_tile(
            pressure_field, promotion_gate, console_tile
        )

        headline = tile["headline"].lower()
        forbidden_words = [
            "better",
            "worse",
            "improve",
            "degrade",
            "success",
            "failure",
            "good",
            "bad",
        ]

        for word in forbidden_words:
            self.assertNotIn(
                word,
                headline,
                f"Found forbidden word in headline: {word}",
            )

    def test_09_pressure_hotspots_list(self):
        """Test pressure_hotspots is a list."""
        pressure_field = self._make_pressure_field()
        promotion_gate = self._make_promotion_gate()
        console_tile = self._make_console_tile()

        tile = build_topology_pressure_governance_tile(
            pressure_field, promotion_gate, console_tile
        )

        self.assertIsInstance(
            tile["pressure_hotspots"],
            list,
            "pressure_hotspots must be a list",
        )

    def test_10_headline_present(self):
        """Test headline is present and non-empty."""
        pressure_field = self._make_pressure_field()
        promotion_gate = self._make_promotion_gate()
        console_tile = self._make_console_tile()

        tile = build_topology_pressure_governance_tile(
            pressure_field, promotion_gate, console_tile
        )

        self.assertIn("headline", tile)
        self.assertIsInstance(tile["headline"], str)
        self.assertGreater(len(tile["headline"]), 0)


if __name__ == "__main__":
    unittest.main()

