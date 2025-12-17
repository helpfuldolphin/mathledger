"""
Phase X: CI Smoke Test for Drift Governance Tile Serialization

This test verifies that the drift governance tile can be produced and serialized
without error. It does NOT test governance logic, drift computation, or
promotion decisions.

SHADOW MODE CONTRACT:
- This test only verifies serialization and structural stability
- No governance decisions are tested or modified
- No drift computation or real data processing is performed
- The test is purely for observability validation

Test requirements (per Phase X spec):
1. Create mock drift_tensor, poly_cause_view, director_tile_v2
2. Call build_drift_governance_tile()
3. Assert: isinstance(tile, dict)
4. Assert: json.dumps(tile) does not raise
5. Assert: tile has required fields
6. Assert: extract_drift_signal_for_shadow() works
7. Assert: determinism and neutrality
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest


class TestDriftGovernanceTileSerializes:
    """
    CI smoke tests for drift governance tile serialization.

    SHADOW MODE: These tests verify serialization only.
    No governance logic is tested.
    """

    def test_drift_governance_tile_serializes_without_error(self) -> None:
        """
        Verify drift governance tile can be produced and serialized.

        This is the primary CI gate test per Phase X spec.
        """
        from backend.health.drift_tensor_adapter import (
            build_drift_governance_tile,
            DRIFT_GOVERNANCE_TILE_SCHEMA_VERSION,
        )

        # 1. Create mock inputs
        drift_tensor = {
            "tensor": {
                "slice_a": {"drift": 0.5, "budget": 0.3, "metric": 0.0, "semantic": 0.0},
                "slice_b": {"drift": 0.2, "budget": 0.0, "metric": 0.4, "semantic": 0.0},
            },
            "global_tensor_norm": 0.707,
            "ranked_slices": ["slice_a", "slice_b"],
            "schema_version": "1.0.0",
        }

        poly_cause_view = {
            "poly_cause_detected": True,
            "cause_vectors": [
                {
                    "slice": "slice_a",
                    "axes": ["drift", "budget"],
                    "drift_scores": {"drift": 0.5, "budget": 0.3},
                }
            ],
            "risk_band": "MEDIUM",
            "notes": ["Slice slice_a: multiple axes show drift (budget, drift)"],
        }

        director_tile_v2 = {
            "status_light": "YELLOW",
            "tensor_norm": 0.707,
            "poly_cause_status": "DETECTED",
            "risk_band": "MEDIUM",
            "uplift_envelope_impact": {
                "status": "ATTENTION",
                "uplift_safe": True,
                "blocking_axes": [],
            },
            "headline": "Tensor norm: 0.707. Poly-cause patterns detected. Risk band: MEDIUM.",
        }

        # 2. Call build_drift_governance_tile()
        tile = build_drift_governance_tile(
            drift_tensor=drift_tensor,
            poly_cause_view=poly_cause_view,
            director_tile_v2=director_tile_v2,
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

    def test_drift_governance_tile_has_required_fields(self) -> None:
        """Verify tile contains required fields per schema."""
        from backend.health.drift_tensor_adapter import (
            build_drift_governance_tile,
            DRIFT_GOVERNANCE_TILE_SCHEMA_VERSION,
        )

        drift_tensor = {
            "tensor": {},
            "global_tensor_norm": 0.0,
            "ranked_slices": [],
            "schema_version": "1.0.0",
        }

        poly_cause_view = {
            "poly_cause_detected": False,
            "cause_vectors": [],
            "risk_band": "LOW",
            "notes": [],
        }

        director_tile_v2 = {
            "status_light": "GREEN",
            "tensor_norm": 0.0,
            "poly_cause_status": "NONE",
            "risk_band": "LOW",
            "uplift_envelope_impact": {
                "status": "OK",
                "uplift_safe": True,
                "blocking_axes": [],
            },
            "headline": "No drift issues detected.",
        }

        tile = build_drift_governance_tile(
            drift_tensor=drift_tensor,
            poly_cause_view=poly_cause_view,
            director_tile_v2=director_tile_v2,
        )

        # Required fields per Phase X spec
        required_fields = [
            "status_light",
            "global_tensor_norm",
            "risk_band",
            "slices_with_poly_cause_drift",
            "headline",
            "schema_version",
        ]

        for field in required_fields:
            assert field in tile, f"Missing required field: {field}"

        # Verify schema version
        assert tile["schema_version"] == DRIFT_GOVERNANCE_TILE_SCHEMA_VERSION

        # Verify status_light is valid
        assert tile["status_light"] in ("GREEN", "YELLOW", "RED")

        # Verify risk_band is valid
        assert tile["risk_band"] in ("LOW", "MEDIUM", "HIGH")

        # Verify slices_with_poly_cause_drift is a list
        assert isinstance(tile["slices_with_poly_cause_drift"], list)

    def test_extract_drift_signal_for_shadow(self) -> None:
        """Verify extract_drift_signal_for_shadow() works correctly."""
        from backend.health.drift_tensor_adapter import (
            extract_drift_signal_for_shadow,
        )

        drift_tensor = {
            "tensor": {
                "slice_a": {"drift": 0.5, "budget": 0.3, "metric": 0.0, "semantic": 0.0},
            },
            "global_tensor_norm": 0.583,
            "ranked_slices": ["slice_a"],
            "schema_version": "1.0.0",
        }

        poly_cause = {
            "poly_cause_detected": True,
            "cause_vectors": [
                {
                    "slice": "slice_a",
                    "axes": ["drift", "budget"],
                    "drift_scores": {"drift": 0.5, "budget": 0.3},
                }
            ],
            "risk_band": "MEDIUM",
            "notes": ["Slice slice_a: multiple axes show drift (budget, drift)"],
        }

        signal = extract_drift_signal_for_shadow(tensor=drift_tensor, poly_cause=poly_cause)

        assert isinstance(signal, dict)
        assert "global_tensor_norm" in signal
        assert "risk_band" in signal
        assert "slices_with_poly_cause_drift" in signal

        assert signal["risk_band"] in ("LOW", "MEDIUM", "HIGH")
        assert isinstance(signal["slices_with_poly_cause_drift"], list)
        assert "slice_a" in signal["slices_with_poly_cause_drift"]

        # Verify serializable
        json_str = json.dumps(signal)
        assert len(json_str) > 0

    def test_drift_governance_tile_deterministic(self) -> None:
        """Verify tile output is deterministic."""
        from backend.health.drift_tensor_adapter import build_drift_governance_tile

        drift_tensor = {
            "tensor": {
                "slice_a": {"drift": 0.5, "budget": 0.3, "metric": 0.0, "semantic": 0.0},
            },
            "global_tensor_norm": 0.583,
            "ranked_slices": ["slice_a"],
            "schema_version": "1.0.0",
        }

        poly_cause_view = {
            "poly_cause_detected": True,
            "cause_vectors": [
                {
                    "slice": "slice_a",
                    "axes": ["drift", "budget"],
                    "drift_scores": {"drift": 0.5, "budget": 0.3},
                }
            ],
            "risk_band": "MEDIUM",
            "notes": ["Slice slice_a: multiple axes show drift (budget, drift)"],
        }

        director_tile_v2 = {
            "status_light": "YELLOW",
            "tensor_norm": 0.583,
            "poly_cause_status": "DETECTED",
            "risk_band": "MEDIUM",
            "uplift_envelope_impact": {
                "status": "ATTENTION",
                "uplift_safe": True,
                "blocking_axes": [],
            },
            "headline": "Tensor norm: 0.583. Poly-cause patterns detected.",
        }

        tile1 = build_drift_governance_tile(
            drift_tensor=drift_tensor,
            poly_cause_view=poly_cause_view,
            director_tile_v2=director_tile_v2,
        )

        tile2 = build_drift_governance_tile(
            drift_tensor=drift_tensor,
            poly_cause_view=poly_cause_view,
            director_tile_v2=director_tile_v2,
        )

        assert tile1 == tile2, "Tile output should be deterministic"

        # Verify JSON serialization is also deterministic
        json1 = json.dumps(tile1, sort_keys=True)
        json2 = json.dumps(tile2, sort_keys=True)
        assert json1 == json2

    def test_drift_governance_tile_neutral_language(self) -> None:
        """Verify tile uses neutral, descriptive language only."""
        from backend.health.drift_tensor_adapter import build_drift_governance_tile

        drift_tensor = {
            "tensor": {
                "slice_a": {"drift": 0.8, "budget": 0.7, "metric": 0.6, "semantic": 0.0},
            },
            "global_tensor_norm": 1.2,
            "ranked_slices": ["slice_a"],
            "schema_version": "1.0.0",
        }

        poly_cause_view = {
            "poly_cause_detected": True,
            "cause_vectors": [
                {
                    "slice": "slice_a",
                    "axes": ["drift", "budget", "metric"],
                    "drift_scores": {"drift": 0.8, "budget": 0.7, "metric": 0.6},
                }
            ],
            "risk_band": "HIGH",
            "notes": ["Slice slice_a: multiple axes show drift (budget, drift, metric)"],
        }

        director_tile_v2 = {
            "status_light": "RED",
            "tensor_norm": 1.2,
            "poly_cause_status": "DETECTED",
            "risk_band": "HIGH",
            "uplift_envelope_impact": {
                "status": "BLOCK",
                "uplift_safe": False,
                "blocking_axes": ["drift", "budget"],
            },
            "headline": "Tensor norm: 1.200. Poly-cause patterns detected. Risk band: HIGH.",
        }

        tile = build_drift_governance_tile(
            drift_tensor=drift_tensor,
            poly_cause_view=poly_cause_view,
            director_tile_v2=director_tile_v2,
        )

        # Check headline for evaluative language (should be descriptive only)
        headline = tile["headline"].lower()
        forbidden_words = ["good", "bad", "success", "failure", "better", "worse", "improve", "degrade"]
        for word in forbidden_words:
            assert word not in headline, f"Headline should not contain evaluative word: {word}"

        # Headline should be descriptive
        assert len(tile["headline"]) > 0
        assert tile["headline"].endswith(".")


class TestGlobalHealthSurfaceDriftGovernanceIntegration:
    """
    Tests for drift governance tile integration with GlobalHealthSurface.

    SHADOW MODE: These tests verify the tile attachment mechanism only.
    """

    def test_build_global_health_surface_with_drift_governance(self) -> None:
        """Verify drift governance tile attached when inputs provided."""
        from backend.health.global_surface import build_global_health_surface

        drift_tensor = {
            "tensor": {},
            "global_tensor_norm": 0.0,
            "ranked_slices": [],
            "schema_version": "1.0.0",
        }

        poly_cause_view = {
            "poly_cause_detected": False,
            "cause_vectors": [],
            "risk_band": "LOW",
            "notes": [],
        }

        director_tile_v2 = {
            "status_light": "GREEN",
            "tensor_norm": 0.0,
            "poly_cause_status": "NONE",
            "risk_band": "LOW",
            "uplift_envelope_impact": {
                "status": "OK",
                "uplift_safe": True,
                "blocking_axes": [],
            },
            "headline": "No drift issues detected.",
        }

        payload = build_global_health_surface(
            drift_tensor=drift_tensor,
            poly_cause_view=poly_cause_view,
            director_tile_v2=director_tile_v2,
        )

        assert isinstance(payload, dict)
        assert "schema_version" in payload
        assert "dynamics" in payload
        assert "drift_governance" in payload, "Drift governance tile should be present when inputs provided"

        drift_tile = payload["drift_governance"]
        assert isinstance(drift_tile, dict)
        assert "status_light" in drift_tile
        assert "global_tensor_norm" in drift_tile

        # Verify serializable
        json_str = json.dumps(payload)
        assert len(json_str) > 0

    def test_build_global_health_surface_without_drift_governance(self) -> None:
        """Verify build works without drift governance inputs."""
        from backend.health.global_surface import build_global_health_surface

        payload = build_global_health_surface()

        assert isinstance(payload, dict)
        assert "schema_version" in payload
        assert "dynamics" in payload
        assert "drift_governance" not in payload, "Drift governance tile should not be present when inputs missing"

    def test_drift_governance_tile_does_not_affect_dynamics(self) -> None:
        """Verify drift governance tile presence doesn't change dynamics tile."""
        from backend.health.global_surface import build_global_health_surface

        drift_tensor = {
            "tensor": {},
            "global_tensor_norm": 0.0,
            "ranked_slices": [],
            "schema_version": "1.0.0",
        }

        poly_cause_view = {
            "poly_cause_detected": False,
            "cause_vectors": [],
            "risk_band": "LOW",
            "notes": [],
        }

        director_tile_v2 = {
            "status_light": "GREEN",
            "tensor_norm": 0.0,
            "poly_cause_status": "NONE",
            "risk_band": "LOW",
            "uplift_envelope_impact": {
                "status": "OK",
                "uplift_safe": True,
                "blocking_axes": [],
            },
            "headline": "No drift issues detected.",
        }

        # Build without drift governance
        payload_without = build_global_health_surface()

        # Build with drift governance
        payload_with = build_global_health_surface(
            drift_tensor=drift_tensor,
            poly_cause_view=poly_cause_view,
            director_tile_v2=director_tile_v2,
        )

        # Dynamics should be identical
        assert payload_without["dynamics"] == payload_with["dynamics"]

