"""
Phase X: CI Smoke Test for Evidence Quality Governance Tile Serialization

This test verifies that the evidence quality governance tile can be produced and serialized
without error. It does NOT test governance logic, evidence computation semantics, or
control flow behavior.

SHADOW MODE CONTRACT:
- This test only verifies serialization and structural stability
- No governance decisions are tested or modified
- No evidence quality computation or real phase-portrait analysis is run
- The test is purely for observability validation

Test requirements (per Phase X spec):
1. Create mock phase_portrait, forecast, and director_panel_v2
2. Call build_evidence_governance_tile()
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


class TestEvidenceGovernanceTileSerializes:
    """
    CI smoke tests for evidence quality governance tile serialization.

    SHADOW MODE: These tests verify serialization only.
    No governance logic is tested.
    """

    def _make_mock_phase_portrait(self) -> Dict[str, Any]:
        """Create mock phase portrait for testing."""
        return {
            "phase_points": [[0, 2], [1, 3], [2, 2]],
            "trajectory_class": "IMPROVING",
            "neutral_notes": ["trajectory shows consistent quality increases"],
        }

    def _make_mock_forecast(self) -> Dict[str, Any]:
        """Create mock forecast for testing."""
        return {
            "predicted_band": "HIGH",
            "confidence": 0.75,
            "cycles_until_risk": 5,
            "neutral_explanation": ["evidence quality is improving with high recent quality"],
        }

    def _make_mock_director_panel_v2(self) -> Dict[str, Any]:
        """Create mock director panel v2 for testing."""
        return {
            "status_light": "GREEN",
            "quality_tier": "TIER_3",
            "trajectory_class": "IMPROVING",
            "regression_status": "OK",
            "analysis_count": 3,
            "evidence_ok": True,
            "headline": "Evidence pack has TIER_3 quality with improving trajectory.",
            "flags": [],
        }

    def test_evidence_governance_tile_serializes_without_error(self) -> None:
        """
        Verify evidence quality governance tile can be produced and serialized.

        This is the primary CI gate test per Phase X spec.
        """
        from backend.health.evidence_quality_adapter import (
            build_evidence_governance_tile,
        )

        # 1. Create mock inputs
        phase_portrait = self._make_mock_phase_portrait()
        forecast = self._make_mock_forecast()
        director_panel_v2 = self._make_mock_director_panel_v2()

        # 2. Call build_evidence_governance_tile()
        tile = build_evidence_governance_tile(
            phase_portrait=phase_portrait,
            forecast=forecast,
            director_panel_v2=director_panel_v2,
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

    def test_evidence_governance_tile_has_required_fields(self) -> None:
        """Verify tile contains required fields per schema."""
        from backend.health.evidence_quality_adapter import (
            build_evidence_governance_tile,
            EVIDENCE_QUALITY_TILE_SCHEMA_VERSION,
        )

        phase_portrait = self._make_mock_phase_portrait()
        forecast = self._make_mock_forecast()
        director_panel_v2 = self._make_mock_director_panel_v2()

        tile = build_evidence_governance_tile(
            phase_portrait=phase_portrait,
            forecast=forecast,
            director_panel_v2=director_panel_v2,
        )

        # Required fields per Phase X spec
        required_fields = [
            "schema_version",
            "status_light",
            "trajectory_class",
            "predicted_band",
            "cycles_until_risk",
            "regression_status",
            "flags",
            "headline",
        ]

        for field in required_fields:
            assert field in tile, f"Missing required field: {field}"

        # Verify schema version
        assert tile["schema_version"] == EVIDENCE_QUALITY_TILE_SCHEMA_VERSION

    def test_evidence_governance_tile_uses_neutral_language(self) -> None:
        """Verify tile headline uses neutral language only."""
        from backend.health.evidence_quality_adapter import (
            build_evidence_governance_tile,
        )
        from backend.metrics.u2_analysis import GOVERNANCE_SUMMARY_FORBIDDEN_WORDS

        phase_portrait = self._make_mock_phase_portrait()
        forecast = self._make_mock_forecast()
        director_panel_v2 = self._make_mock_director_panel_v2()

        tile = build_evidence_governance_tile(
            phase_portrait=phase_portrait,
            forecast=forecast,
            director_panel_v2=director_panel_v2,
        )

        headline = tile.get("headline", "")
        headline_lower = headline.lower()
        tile_str = json.dumps(tile).lower()

        # Check forbidden words
        for word in GOVERNANCE_SUMMARY_FORBIDDEN_WORDS:
            assert word.lower() not in headline_lower, f"Headline contains forbidden word: {word}"
            assert word.lower() not in tile_str, f"Tile contains forbidden word: {word}"

    def test_evidence_governance_tile_is_deterministic(self) -> None:
        """Verify tile is deterministic across multiple calls."""
        from backend.health.evidence_quality_adapter import (
            build_evidence_governance_tile,
        )

        phase_portrait = self._make_mock_phase_portrait()
        forecast = self._make_mock_forecast()
        director_panel_v2 = self._make_mock_director_panel_v2()

        # Call twice with same inputs
        tile1 = build_evidence_governance_tile(
            phase_portrait=phase_portrait,
            forecast=forecast,
            director_panel_v2=director_panel_v2,
        )
        tile2 = build_evidence_governance_tile(
            phase_portrait=phase_portrait,
            forecast=forecast,
            director_panel_v2=director_panel_v2,
        )

        # Serialize both and compare
        json1 = json.dumps(tile1, sort_keys=True)
        json2 = json.dumps(tile2, sort_keys=True)

        assert json1 == json2, "Tile should be deterministic"

    def test_evidence_governance_tile_handles_missing_inputs(self) -> None:
        """Verify tile handles None inputs gracefully."""
        from backend.health.evidence_quality_adapter import (
            build_evidence_governance_tile,
        )

        # Test with all None
        tile = build_evidence_governance_tile()
        assert isinstance(tile, dict)
        assert tile["status_light"] == "UNKNOWN"
        assert tile["trajectory_class"] == "UNKNOWN"
        assert tile["predicted_band"] == "UNKNOWN"

        # Test with partial inputs
        phase_portrait = self._make_mock_phase_portrait()
        tile = build_evidence_governance_tile(phase_portrait=phase_portrait)
        assert tile["trajectory_class"] == "IMPROVING"
        assert tile["status_light"] == "UNKNOWN"

    def test_evidence_governance_tile_no_governance_modification(self) -> None:
        """
        Verify tile does not modify governance state.

        This test ensures the tile is purely observational.
        """
        from backend.health.evidence_quality_adapter import (
            build_evidence_governance_tile,
        )

        phase_portrait = self._make_mock_phase_portrait()
        forecast = self._make_mock_forecast()
        director_panel_v2 = self._make_mock_director_panel_v2()

        # Create copies to verify inputs are not modified
        phase_portrait_copy = json.loads(json.dumps(phase_portrait))
        forecast_copy = json.loads(json.dumps(forecast))
        director_panel_v2_copy = json.loads(json.dumps(director_panel_v2))

        # Build tile
        tile = build_evidence_governance_tile(
            phase_portrait=phase_portrait,
            forecast=forecast,
            director_panel_v2=director_panel_v2,
        )

        # Verify inputs unchanged
        assert json.dumps(phase_portrait, sort_keys=True) == json.dumps(
            phase_portrait_copy, sort_keys=True
        ), "phase_portrait should not be modified"
        assert json.dumps(forecast, sort_keys=True) == json.dumps(
            forecast_copy, sort_keys=True
        ), "forecast should not be modified"
        assert json.dumps(director_panel_v2, sort_keys=True) == json.dumps(
            director_panel_v2_copy, sort_keys=True
        ), "director_panel_v2 should not be modified"

        # Verify tile is a new dict (not a reference)
        assert tile is not phase_portrait
        assert tile is not forecast
        assert tile is not director_panel_v2


class TestGlobalHealthSurfaceEvidenceQualityIntegration:
    """
    Tests for evidence quality tile integration with GlobalHealthSurface.

    SHADOW MODE: These tests verify the tile attachment mechanism only.
    """

    def test_build_global_health_surface_without_evidence_quality(self) -> None:
        """Verify build works without evidence quality inputs."""
        from backend.health.global_surface import build_global_health_surface

        payload = build_global_health_surface()

        assert isinstance(payload, dict)
        assert "schema_version" in payload
        assert "dynamics" in payload
        assert "evidence_quality" not in payload  # Should not be present

    def test_build_global_health_surface_with_evidence_quality(self) -> None:
        """Verify evidence quality tile attached when inputs provided."""
        from backend.health.global_surface import build_global_health_surface

        # Create mock inputs
        phase_portrait = {
            "phase_points": [[0, 2], [1, 3]],
            "trajectory_class": "IMPROVING",
            "neutral_notes": [],
        }
        forecast = {
            "predicted_band": "HIGH",
            "confidence": 0.75,
            "cycles_until_risk": 5,
            "neutral_explanation": [],
        }
        director_panel_v2 = {
            "status_light": "GREEN",
            "quality_tier": "TIER_3",
            "trajectory_class": "IMPROVING",
            "regression_status": "OK",
            "analysis_count": 3,
            "evidence_ok": True,
            "headline": "Evidence pack has TIER_3 quality.",
            "flags": [],
        }

        payload = build_global_health_surface(
            evidence_phase_portrait=phase_portrait,
            evidence_forecast=forecast,
            evidence_director_panel_v2=director_panel_v2,
        )

        assert isinstance(payload, dict)
        assert "evidence_quality" in payload, "Evidence quality tile should be present"

        evidence_tile = payload["evidence_quality"]
        assert isinstance(evidence_tile, dict)
        assert "headline" in evidence_tile

        # Verify serializable
        json_str = json.dumps(payload)
        assert len(json_str) > 0

    def test_evidence_quality_tile_does_not_affect_dynamics(self) -> None:
        """Verify evidence quality tile presence doesn't change dynamics tile."""
        from backend.health.global_surface import build_global_health_surface

        # Build without evidence quality
        payload_without = build_global_health_surface()

        # Build with evidence quality
        phase_portrait = {
            "phase_points": [[0, 2]],
            "trajectory_class": "STABLE",
            "neutral_notes": [],
        }
        forecast = {
            "predicted_band": "MEDIUM",
            "confidence": 0.6,
            "cycles_until_risk": 3,
            "neutral_explanation": [],
        }
        director_panel_v2 = {
            "status_light": "YELLOW",
            "quality_tier": "TIER_2",
            "trajectory_class": "STABLE",
            "regression_status": "OK",
            "analysis_count": 1,
            "evidence_ok": True,
            "headline": "Evidence pack has TIER_2 quality.",
            "flags": [],
        }

        payload_with = build_global_health_surface(
            evidence_phase_portrait=phase_portrait,
            evidence_forecast=forecast,
            evidence_director_panel_v2=director_panel_v2,
        )

        # Dynamics should be identical
        assert payload_without["dynamics"] == payload_with["dynamics"]

