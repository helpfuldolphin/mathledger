"""
Phase X: CI Smoke Test for Semantic Drift Governance Tile Serialization

This test verifies that the semantic drift governance tile can be produced and serialized
without error. It does NOT test governance logic, drift computation, or
promotion decisions.

SHADOW MODE CONTRACT:
- This test only verifies serialization and structural stability
- No governance decisions are tested or modified
- No drift computation or real data processing is performed
- The test is purely for observability validation

Test requirements (per Phase X spec):
1. Create mock drift_tensor, counterfactual, drift_director_panel
2. Call build_semantic_drift_governance_tile()
3. Assert: isinstance(tile, dict)
4. Assert: json.dumps(tile) does not raise
5. Assert: tile has required fields (status_light, tensor_norm, semantic_hotspots, projected_instability_count, headline)
6. Assert: extract_drift_advisory_for_uplift() works
7. Assert: determinism and neutrality
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest


class TestSemanticDriftGovernanceTileSerializes:
    """
    CI smoke tests for semantic drift governance tile serialization.

    SHADOW MODE: These tests verify serialization only.
    No governance logic is tested.
    """

    def test_semantic_drift_governance_tile_serializes_without_error(self) -> None:
        """
        Verify semantic drift governance tile can be produced and serialized.

        This is the primary CI gate test per Phase X spec.
        """
        from backend.health.semantic_drift_adapter import (
            build_semantic_drift_governance_tile,
            SEMANTIC_DRIFT_GOVERNANCE_TILE_SCHEMA_VERSION,
        )

        # 1. Create mock inputs
        drift_tensor = {
            "drift_components": {
                "slice_a": {
                    "semantic": 0.7,
                    "causal": 0.6,
                    "metric_correlated": 0.5,
                },
                "slice_b": {
                    "semantic": 0.3,
                    "causal": 0.2,
                    "metric_correlated": 0.1,
                },
            },
            "semantic_hotspots": ["slice_a"],
            "tensor_norm": 1.224,
        }

        counterfactual = {
            "projected_unstable_slices": ["slice_a"],
            "stability_timeline": {
                "slice_a": {
                    "current_stability": 0.3,
                    "projected_stability": [0.25, 0.2, 0.15],
                    "becomes_unstable_at": 2,
                },
                "slice_b": {
                    "current_stability": 0.8,
                    "projected_stability": [0.75, 0.7, 0.65],
                    "becomes_unstable_at": None,
                },
            },
            "neutral_notes": [
                "1 of 2 slice(s) projected to become unstable within 3 time step(s) if current drift rate continues.",
            ],
        }

        drift_director_panel = {
            "status_light": "YELLOW",
            "semantic_hotspots": ["slice_a"],
            "projected_instability_count": 1,
            "gating_recommendation": "WARN",
            "recommendation_reasons": [
                "1 semantic hotspot(s) identified: slice_a",
                "1 slice(s) projected to become unstable.",
            ],
            "headline": "Semantic drift analysis indicates potential instability; gating recommendation: WARN.",
        }

        # 2. Call build_semantic_drift_governance_tile()
        tile = build_semantic_drift_governance_tile(
            drift_tensor=drift_tensor,
            counterfactual=counterfactual,
            drift_director_panel=drift_director_panel,
        )

        # 3. Assert: isinstance(tile, dict)
        assert tile is not None, "Tile should not be None"
        assert isinstance(tile, dict), f"Tile should be dict, got {type(tile)}"

        # 4. Assert: json.dumps(tile) does not raise
        json_str = json.dumps(tile)
        assert json_str is not None
        assert len(json_str) > 0

        # Verify round-trip
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_semantic_drift_governance_tile_has_required_fields(self) -> None:
        """Verify tile contains required fields per schema."""
        from backend.health.semantic_drift_adapter import (
            build_semantic_drift_governance_tile,
            SEMANTIC_DRIFT_GOVERNANCE_TILE_SCHEMA_VERSION,
        )

        drift_tensor = {
            "drift_components": {},
            "semantic_hotspots": [],
            "tensor_norm": 0.0,
        }

        counterfactual = {
            "projected_unstable_slices": [],
            "stability_timeline": {},
            "neutral_notes": [],
        }

        drift_director_panel = {
            "status_light": "GREEN",
            "semantic_hotspots": [],
            "projected_instability_count": 0,
            "gating_recommendation": "OK",
            "recommendation_reasons": ["No significant semantic drift indicators detected."],
            "headline": "Semantic drift analysis indicates system stability; gating recommendation: OK.",
        }

        tile = build_semantic_drift_governance_tile(
            drift_tensor=drift_tensor,
            counterfactual=counterfactual,
            drift_director_panel=drift_director_panel,
        )

        # Required fields per Phase X spec
        required_fields = [
            "schema_version",
            "status_light",
            "tensor_norm",
            "semantic_hotspots",
            "projected_instability_count",
            "gating_recommendation",
            "recommendation_reasons",
            "headline",
        ]

        for field in required_fields:
            assert field in tile, f"Missing required field: {field}"

        # Verify schema version
        assert tile["schema_version"] == SEMANTIC_DRIFT_GOVERNANCE_TILE_SCHEMA_VERSION

        # Verify status_light is valid
        assert tile["status_light"] in ("GREEN", "YELLOW", "RED")

        # Verify gating_recommendation is valid
        assert tile["gating_recommendation"] in ("OK", "WARN", "BLOCK")

        # Verify semantic_hotspots is a list
        assert isinstance(tile["semantic_hotspots"], list)

        # Verify projected_instability_count is an int
        assert isinstance(tile["projected_instability_count"], int)

        # Verify tensor_norm is a float
        assert isinstance(tile["tensor_norm"], float)

    def test_extract_drift_advisory_for_uplift(self) -> None:
        """Verify extract_drift_advisory_for_uplift() works correctly."""
        from backend.health.semantic_drift_adapter import (
            extract_drift_advisory_for_uplift,
        )

        drift_director_panel = {
            "status_light": "YELLOW",
            "semantic_hotspots": ["slice_a", "slice_b"],
            "projected_instability_count": 1,
            "gating_recommendation": "WARN",
            "recommendation_reasons": [
                "2 semantic hotspot(s) identified: slice_a, slice_b",
            ],
            "headline": "Semantic drift analysis indicates potential instability.",
        }

        advisory = extract_drift_advisory_for_uplift(drift_director_panel=drift_director_panel)

        assert isinstance(advisory, dict)
        assert "status_light" in advisory
        assert "gating_recommendation" in advisory
        assert "semantic_hotspots" in advisory

        assert advisory["status_light"] in ("GREEN", "YELLOW", "RED")
        assert advisory["gating_recommendation"] in ("OK", "WARN", "BLOCK")
        assert isinstance(advisory["semantic_hotspots"], list)
        assert "slice_a" in advisory["semantic_hotspots"]
        assert "slice_b" in advisory["semantic_hotspots"]

        # Verify hotspots are sorted
        assert advisory["semantic_hotspots"] == sorted(advisory["semantic_hotspots"])

        # Verify serializable
        json_str = json.dumps(advisory)
        assert len(json_str) > 0

    def test_semantic_drift_governance_tile_deterministic(self) -> None:
        """Verify tile output is deterministic."""
        from backend.health.semantic_drift_adapter import build_semantic_drift_governance_tile

        drift_tensor = {
            "drift_components": {
                "slice_a": {
                    "semantic": 0.5,
                    "causal": 0.4,
                    "metric_correlated": 0.3,
                },
            },
            "semantic_hotspots": ["slice_a"],
            "tensor_norm": 0.707,
        }

        counterfactual = {
            "projected_unstable_slices": [],
            "stability_timeline": {
                "slice_a": {
                    "current_stability": 0.5,
                    "projected_stability": [0.45, 0.4, 0.35],
                    "becomes_unstable_at": None,
                },
            },
            "neutral_notes": [],
        }

        drift_director_panel = {
            "status_light": "GREEN",
            "semantic_hotspots": ["slice_a"],
            "projected_instability_count": 0,
            "gating_recommendation": "OK",
            "recommendation_reasons": [],
            "headline": "Semantic drift analysis indicates system stability.",
        }

        tile1 = build_semantic_drift_governance_tile(
            drift_tensor=drift_tensor,
            counterfactual=counterfactual,
            drift_director_panel=drift_director_panel,
        )

        tile2 = build_semantic_drift_governance_tile(
            drift_tensor=drift_tensor,
            counterfactual=counterfactual,
            drift_director_panel=drift_director_panel,
        )

        assert tile1 == tile2, "Tile output should be deterministic"

        # Verify JSON serialization is also deterministic
        json1 = json.dumps(tile1, sort_keys=True)
        json2 = json.dumps(tile2, sort_keys=True)
        assert json1 == json2

    def test_semantic_drift_governance_tile_neutral_language(self) -> None:
        """Verify tile uses neutral, descriptive language only."""
        from backend.health.semantic_drift_adapter import build_semantic_drift_governance_tile

        drift_tensor = {
            "drift_components": {
                "slice_a": {
                    "semantic": 0.9,
                    "causal": 0.8,
                    "metric_correlated": 0.7,
                },
            },
            "semantic_hotspots": ["slice_a"],
            "tensor_norm": 1.414,
        }

        counterfactual = {
            "projected_unstable_slices": ["slice_a"],
            "stability_timeline": {
                "slice_a": {
                    "current_stability": 0.1,
                    "projected_stability": [0.05, 0.0, -0.05],
                    "becomes_unstable_at": 1,
                },
            },
            "neutral_notes": [
                "1 of 1 slice(s) projected to become unstable within 3 time step(s).",
            ],
        }

        drift_director_panel = {
            "status_light": "RED",
            "semantic_hotspots": ["slice_a"],
            "projected_instability_count": 1,
            "gating_recommendation": "BLOCK",
            "recommendation_reasons": [
                "1 semantic hotspot(s) identified: slice_a",
                "1 slice(s) projected to become unstable.",
                "Tensor norm (1.41) indicates elevated drift across system.",
            ],
            "headline": "Semantic drift analysis indicates system instability risk; gating recommendation: BLOCK.",
        }

        tile = build_semantic_drift_governance_tile(
            drift_tensor=drift_tensor,
            counterfactual=counterfactual,
            drift_director_panel=drift_director_panel,
        )

        # Check headline for evaluative language (should be descriptive only)
        headline = tile["headline"].lower()
        forbidden_words = ["good", "bad", "success", "failure", "better", "worse", "improve", "degrade"]
        for word in forbidden_words:
            assert word not in headline, f"Headline should not contain evaluative word: {word}"

        # Headline should be descriptive
        assert len(tile["headline"]) > 0
        assert tile["headline"].endswith(".")

        # Check recommendation_reasons for evaluative language
        reasons_text = " ".join(tile["recommendation_reasons"]).lower()
        for word in forbidden_words:
            assert word not in reasons_text, f"Recommendation reasons should not contain evaluative word: {word}"


class TestGlobalHealthSurfaceSemanticDriftIntegration:
    """
    Tests for semantic drift governance tile integration with GlobalHealthSurface.

    SHADOW MODE: These tests verify the tile attachment mechanism only.
    """

    def test_build_global_health_surface_with_semantic_drift(self) -> None:
        """Verify semantic drift governance tile attached when inputs provided."""
        from backend.health.global_surface import build_global_health_surface

        drift_tensor = {
            "drift_components": {},
            "semantic_hotspots": [],
            "tensor_norm": 0.0,
        }

        counterfactual = {
            "projected_unstable_slices": [],
            "stability_timeline": {},
            "neutral_notes": [],
        }

        drift_director_panel = {
            "status_light": "GREEN",
            "semantic_hotspots": [],
            "projected_instability_count": 0,
            "gating_recommendation": "OK",
            "recommendation_reasons": [],
            "headline": "Semantic drift analysis indicates system stability.",
        }

        payload = build_global_health_surface(
            semantic_drift_tensor=drift_tensor,
            semantic_drift_counterfactual=counterfactual,
            semantic_drift_director_panel=drift_director_panel,
        )

        assert isinstance(payload, dict)
        assert "schema_version" in payload
        assert "dynamics" in payload
        assert "semantic_drift" in payload, "Semantic drift governance tile should be present when inputs provided"

        semantic_drift_tile = payload["semantic_drift"]
        assert isinstance(semantic_drift_tile, dict)
        assert "status_light" in semantic_drift_tile
        assert "tensor_norm" in semantic_drift_tile
        assert "semantic_hotspots" in semantic_drift_tile
        assert "projected_instability_count" in semantic_drift_tile
        assert "headline" in semantic_drift_tile

        # Verify serializable
        json_str = json.dumps(payload)
        assert len(json_str) > 0

    def test_build_global_health_surface_without_semantic_drift(self) -> None:
        """Verify build works without semantic drift inputs."""
        from backend.health.global_surface import build_global_health_surface

        payload = build_global_health_surface()

        assert isinstance(payload, dict)
        assert "schema_version" in payload
        assert "dynamics" in payload
        assert "semantic_drift" not in payload, "Semantic drift governance tile should not be present when inputs missing"

    def test_semantic_drift_governance_tile_does_not_affect_dynamics(self) -> None:
        """Verify semantic drift governance tile presence doesn't change dynamics tile."""
        from backend.health.global_surface import build_global_health_surface

        drift_tensor = {
            "drift_components": {},
            "semantic_hotspots": [],
            "tensor_norm": 0.0,
        }

        counterfactual = {
            "projected_unstable_slices": [],
            "stability_timeline": {},
            "neutral_notes": [],
        }

        drift_director_panel = {
            "status_light": "GREEN",
            "semantic_hotspots": [],
            "projected_instability_count": 0,
            "gating_recommendation": "OK",
            "recommendation_reasons": [],
            "headline": "Semantic drift analysis indicates system stability.",
        }

        # Build without semantic drift governance
        payload_without = build_global_health_surface()

        # Build with semantic drift governance
        payload_with = build_global_health_surface(
            semantic_drift_tensor=drift_tensor,
            semantic_drift_counterfactual=counterfactual,
            semantic_drift_director_panel=drift_director_panel,
        )

        # Dynamics should be identical
        assert payload_without["dynamics"] == payload_with["dynamics"]

