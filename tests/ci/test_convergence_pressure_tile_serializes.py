"""
Phase X: CI Smoke Test for Convergence Pressure Tile Serialization

This test verifies that the convergence pressure tile can be produced and serialized
without error. It does NOT test governance logic, pressure computation semantics, or
control flow behavior.

SHADOW MODE CONTRACT:
- This test only verifies serialization and structural stability
- No governance decisions are tested or modified
- No pressure tensor computation or real convergence analysis is run
- The test is purely for observability validation

Test requirements (per Phase X spec):
1. Create mock pressure_tensor, early_warning, and director_tile
2. Call build_convergence_pressure_tile()
3. Assert: isinstance(tile, dict)
4. Assert: json.dumps(tile) does not raise
5. Assert: Tile uses neutral language
6. Assert: Tile is deterministic
7. Assert: No governance modification
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest


class TestConvergencePressureTileSerializes:
    """
    CI smoke tests for convergence pressure tile serialization.

    SHADOW MODE: These tests verify serialization only.
    No governance logic is tested.
    """

    def _make_mock_pressure_tensor(self) -> Dict[str, Any]:
        """Create mock pressure tensor for testing."""
        return {
            "schema_version": "1.0.0",
            "global_pressure_norm": 1.5,
            "pressure_ranked_slices": ["slice_a", "slice_b"],
            "slice_pressure_vectors": {
                "slice_a": {"alignment": 0.5, "drift": 0.6, "metric": 0.4},
                "slice_b": {"alignment": 0.3, "drift": 0.4, "metric": 0.3},
            },
        }

    def _make_mock_early_warning(self) -> Dict[str, Any]:
        """Create mock early-warning radar for testing."""
        return {
            "transition_likelihood_band": "MEDIUM",
            "root_drivers": [
                "Global pressure norm elevated (1.50)",
                "Phase boundary forecast confidence moderate (50.0%)",
            ],
            "first_slices_at_risk": ["slice_a", "slice_b"],
            "time_to_inflection_estimate": 5,
        }

    def _make_mock_director_tile(self) -> Dict[str, Any]:
        """Create mock director tile for testing."""
        return {
            "status_light": "YELLOW",
            "transition_band": "MEDIUM",
            "global_pressure_norm": 1.5,
            "headline": "Convergence status: pressure norm: 1.50, transition likelihood: medium, 2 slice(s) at elevated risk",
            "pressure_drivers": [
                "Global pressure norm elevated (1.50)",
                "Phase boundary forecast confidence moderate (50.0%)",
            ],
        }

    def test_convergence_pressure_tile_serializes_without_error(self) -> None:
        """
        Verify convergence pressure tile can be produced and serialized.

        This is the primary CI gate test per Phase X spec.
        """
        from backend.health.convergence_pressure_adapter import (
            build_convergence_pressure_tile,
        )

        # 1. Create mock inputs
        pressure_tensor = self._make_mock_pressure_tensor()
        early_warning = self._make_mock_early_warning()
        director_tile = self._make_mock_director_tile()

        # 2. Call build_convergence_pressure_tile()
        tile = build_convergence_pressure_tile(
            pressure_tensor=pressure_tensor,
            early_warning=early_warning,
            director_tile=director_tile,
        )

        # 3. Assert: isinstance(tile, dict)
        assert tile is not None, "Tile should not be None"
        assert isinstance(tile, dict), f"Tile should be dict, got {type(tile)}"

        # 4. Assert: json.dumps(tile) does not raise
        json_str = json.dumps(tile, sort_keys=True)
        assert json_str is not None
        assert len(json_str) > 0

        # Verify round-trip
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_convergence_pressure_tile_has_required_fields(self) -> None:
        """Verify tile contains required fields per schema."""
        from backend.health.convergence_pressure_adapter import (
            build_convergence_pressure_tile,
            CONVERGENCE_PRESSURE_TILE_SCHEMA_VERSION,
        )

        pressure_tensor = self._make_mock_pressure_tensor()
        early_warning = self._make_mock_early_warning()
        director_tile = self._make_mock_director_tile()

        tile = build_convergence_pressure_tile(
            pressure_tensor=pressure_tensor,
            early_warning=early_warning,
            director_tile=director_tile,
        )

        # Required fields per Phase X spec
        required_fields = [
            "schema_version",
            "tile_type",
            "status_light",
            "global_pressure_norm",
            "transition_likelihood_band",
            "slices_at_risk",
            "pressure_drivers",
            "headline",
        ]

        for field in required_fields:
            assert field in tile, f"Missing required field: {field}"

        # Verify schema version
        assert tile["schema_version"] == CONVERGENCE_PRESSURE_TILE_SCHEMA_VERSION
        assert tile["tile_type"] == "convergence_pressure"

    def test_convergence_pressure_tile_uses_neutral_language(self) -> None:
        """Verify tile headline uses neutral language only."""
        from backend.health.convergence_pressure_adapter import (
            build_convergence_pressure_tile,
        )

        pressure_tensor = self._make_mock_pressure_tensor()
        early_warning = self._make_mock_early_warning()
        director_tile = self._make_mock_director_tile()

        tile = build_convergence_pressure_tile(
            pressure_tensor=pressure_tensor,
            early_warning=early_warning,
            director_tile=director_tile,
        )

        headline = tile.get("headline", "")
        headline_lower = headline.lower()

        # Forbidden words for neutral language
        forbidden_words = [
            "good",
            "bad",
            "better",
            "worse",
            "improve",
            "improvement",
            "should",
            "must",
            "need",
            "required",
            "fail",
            "success",
            "correct",
            "incorrect",
            "right",
            "wrong",
            "fix",
            "broken",
        ]

        for word in forbidden_words:
            assert word not in headline_lower, f"Headline contains forbidden word: {word}"

        # Verify drivers are also neutral
        for driver in tile.get("pressure_drivers", []):
            driver_lower = driver.lower()
            for word in forbidden_words:
                assert word not in driver_lower, f"Driver contains forbidden word: {word}"

    def test_convergence_pressure_tile_is_deterministic(self) -> None:
        """Verify tile is deterministic across multiple calls."""
        from backend.health.convergence_pressure_adapter import (
            build_convergence_pressure_tile,
        )

        pressure_tensor = self._make_mock_pressure_tensor()
        early_warning = self._make_mock_early_warning()
        director_tile = self._make_mock_director_tile()

        # Call twice with same inputs
        tile1 = build_convergence_pressure_tile(
            pressure_tensor=pressure_tensor,
            early_warning=early_warning,
            director_tile=director_tile,
        )
        tile2 = build_convergence_pressure_tile(
            pressure_tensor=pressure_tensor,
            early_warning=early_warning,
            director_tile=director_tile,
        )

        # Serialize both and compare
        json1 = json.dumps(tile1, sort_keys=True)
        json2 = json.dumps(tile2, sort_keys=True)

        assert json1 == json2, "Tile should be deterministic"

    def test_convergence_pressure_tile_status_light_logic(self) -> None:
        """Verify status light computation follows director tile rules."""
        from backend.health.convergence_pressure_adapter import (
            build_convergence_pressure_tile,
        )

        # Test GREEN: pressure < 1.0 AND transition_band == 'LOW'
        pressure_tensor_green = {
            "global_pressure_norm": 0.8,
            "pressure_ranked_slices": [],
            "slice_pressure_vectors": {},
        }
        early_warning_green = {
            "transition_likelihood_band": "LOW",
            "root_drivers": [],
            "first_slices_at_risk": [],
        }
        director_tile_green = {"status_light": "GREEN"}

        tile_green = build_convergence_pressure_tile(
            pressure_tensor=pressure_tensor_green,
            early_warning=early_warning_green,
            director_tile=director_tile_green,
        )
        assert tile_green["status_light"] == "GREEN"

        # Test RED: pressure > 2.0 OR transition_band == 'HIGH'
        pressure_tensor_red = {
            "global_pressure_norm": 2.5,
            "pressure_ranked_slices": [],
            "slice_pressure_vectors": {},
        }
        early_warning_red = {
            "transition_likelihood_band": "HIGH",
            "root_drivers": [],
            "first_slices_at_risk": [],
        }
        director_tile_red = {"status_light": "RED"}

        tile_red = build_convergence_pressure_tile(
            pressure_tensor=pressure_tensor_red,
            early_warning=early_warning_red,
            director_tile=director_tile_red,
        )
        assert tile_red["status_light"] == "RED"

    def test_convergence_pressure_tile_no_governance_modification(self) -> None:
        """
        Verify tile does not modify governance state.

        This test ensures the tile is purely observational.
        """
        from backend.health.convergence_pressure_adapter import (
            build_convergence_pressure_tile,
        )

        pressure_tensor = self._make_mock_pressure_tensor()
        early_warning = self._make_mock_early_warning()
        director_tile = self._make_mock_director_tile()

        # Create copies to verify inputs are not modified
        pressure_tensor_copy = json.loads(json.dumps(pressure_tensor))
        early_warning_copy = json.loads(json.dumps(early_warning))
        director_tile_copy = json.loads(json.dumps(director_tile))

        # Build tile
        tile = build_convergence_pressure_tile(
            pressure_tensor=pressure_tensor,
            early_warning=early_warning,
            director_tile=director_tile,
        )

        # Verify inputs unchanged
        assert json.dumps(pressure_tensor, sort_keys=True) == json.dumps(
            pressure_tensor_copy, sort_keys=True
        ), "pressure_tensor should not be modified"
        assert json.dumps(early_warning, sort_keys=True) == json.dumps(
            early_warning_copy, sort_keys=True
        ), "early_warning should not be modified"
        assert json.dumps(director_tile, sort_keys=True) == json.dumps(
            director_tile_copy, sort_keys=True
        ), "director_tile should not be modified"

        # Verify tile is a new dict (not a reference)
        assert tile is not pressure_tensor
        assert tile is not early_warning
        assert tile is not director_tile


class TestGlobalHealthSurfaceConvergencePressureIntegration:
    """
    Tests for convergence pressure tile integration with GlobalHealthSurface.

    SHADOW MODE: These tests verify the tile attachment mechanism only.
    """

    def test_build_global_health_surface_without_convergence_pressure(self) -> None:
        """Verify build works without convergence pressure inputs."""
        from backend.health.global_surface import build_global_health_surface

        payload = build_global_health_surface()

        assert isinstance(payload, dict)
        assert "schema_version" in payload
        assert "dynamics" in payload
        assert "convergence_pressure" not in payload  # Should not be present

    def test_build_global_health_surface_with_convergence_pressure(self) -> None:
        """Verify convergence pressure tile attached when all inputs provided."""
        from backend.health.global_surface import build_global_health_surface
        from backend.health.convergence_pressure_adapter import (
            build_convergence_pressure_tile,
        )

        # Create mock inputs
        pressure_tensor = {
            "global_pressure_norm": 1.5,
            "pressure_ranked_slices": [],
            "slice_pressure_vectors": {},
        }
        early_warning = {
            "transition_likelihood_band": "MEDIUM",
            "root_drivers": [],
            "first_slices_at_risk": [],
        }
        director_tile = {"status_light": "YELLOW"}

        payload = build_global_health_surface(
            convergence_pressure_tensor=pressure_tensor,
            convergence_early_warning=early_warning,
            convergence_director_tile=director_tile,
        )

        assert isinstance(payload, dict)
        assert "convergence_pressure" in payload, "Convergence pressure tile should be present"

        convergence_tile = payload["convergence_pressure"]
        assert isinstance(convergence_tile, dict)
        assert "headline" in convergence_tile

        # Verify serializable
        import json

        json_str = json.dumps(payload)
        assert len(json_str) > 0

    def test_convergence_pressure_tile_does_not_affect_dynamics(self) -> None:
        """Verify convergence pressure tile presence doesn't change dynamics tile."""
        from backend.health.global_surface import build_global_health_surface

        # Build without convergence pressure
        payload_without = build_global_health_surface()

        # Build with convergence pressure
        pressure_tensor = {
            "global_pressure_norm": 1.5,
            "pressure_ranked_slices": [],
            "slice_pressure_vectors": {},
        }
        early_warning = {
            "transition_likelihood_band": "MEDIUM",
            "root_drivers": [],
            "first_slices_at_risk": [],
        }
        director_tile = {"status_light": "YELLOW"}

        payload_with = build_global_health_surface(
            convergence_pressure_tensor=pressure_tensor,
            convergence_early_warning=early_warning,
            convergence_director_tile=director_tile,
        )

        # Dynamics should be identical
        assert payload_without["dynamics"] == payload_with["dynamics"]

