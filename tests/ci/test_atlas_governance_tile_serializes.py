"""
Phase X: CI Smoke Test for Atlas Governance Tile Serialization

This test verifies that the atlas governance tile can be produced and serialized
without error. It does NOT test governance logic, lattice computation semantics, or
control flow behavior.

SHADOW MODE CONTRACT:
- This test only verifies serialization and structural stability
- No governance decisions are tested or modified
- No lattice computation or real atlas analysis is run
- The test is purely for observability validation

Test requirements (per Phase X spec):
1. Create mock lattice, phase_gate, and director_tile_v2
2. Call build_atlas_governance_tile()
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


class TestAtlasGovernanceTileSerializes:
    """
    CI smoke tests for atlas governance tile serialization.

    SHADOW MODE: These tests verify serialization only.
    No governance logic is tested.
    """

    def _make_mock_lattice(self) -> Dict[str, Any]:
        """Create mock lattice for testing."""
        return {
            "convergence_band": "COHERENT",
            "global_lattice_norm": 0.85,
            "lattice_vectors": {
                "slice_a": 0.9,
                "slice_b": 0.8,
            },
            "neutral_notes": [
                "Lattice computed for 2 slices.",
                "Global lattice norm: 0.850.",
                "Convergence band: COHERENT.",
            ],
        }

    def _make_mock_phase_gate(self) -> Dict[str, Any]:
        """Create mock phase transition gate for testing."""
        return {
            "transition_status": "OK",
            "drivers": ["All checks passed"],
            "slices_ready": ["slice_a", "slice_b"],
            "slices_needing_alignment": [],
            "headline": "Phase transition appears safe: 2 slices ready with coherent lattice convergence (norm: 0.850).",
        }

    def _make_mock_director_tile_v2(self) -> Dict[str, Any]:
        """Create mock director tile v2 for testing."""
        return {
            "status_light": "GREEN",
            "lattice_coherence": "COHERENT",
            "structural_status": "OK",
            "transition_recommendation": "OK: Phase transition appears safe",
            "atlas_ok": True,
            "headline": "Atlas director tile: 2 slices, 2 ready, lattice coherent (norm: 0.850), transition safe.",
        }

    def test_atlas_governance_tile_serializes_without_error(self) -> None:
        """
        Verify atlas governance tile can be produced and serialized.

        This is the primary CI gate test per Phase X spec.
        """
        from backend.health.atlas_governance_adapter import (
            build_atlas_governance_tile,
        )

        # 1. Create mock inputs
        lattice = self._make_mock_lattice()
        phase_gate = self._make_mock_phase_gate()
        director_tile_v2 = self._make_mock_director_tile_v2()

        # 2. Call build_atlas_governance_tile()
        tile = build_atlas_governance_tile(
            lattice=lattice,
            phase_gate=phase_gate,
            director_tile_v2=director_tile_v2,
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

    def test_atlas_governance_tile_has_required_fields(self) -> None:
        """Verify tile contains required fields per schema."""
        from backend.health.atlas_governance_adapter import (
            ATLAS_GOVERNANCE_TILE_SCHEMA_VERSION,
            build_atlas_governance_tile,
        )

        lattice = self._make_mock_lattice()
        phase_gate = self._make_mock_phase_gate()
        director_tile_v2 = self._make_mock_director_tile_v2()

        tile = build_atlas_governance_tile(
            lattice=lattice,
            phase_gate=phase_gate,
            director_tile_v2=director_tile_v2,
        )

        # Required fields per Phase X spec
        required_fields = [
            "schema_version",
            "tile_type",
            "status_light",
            "lattice_coherence_band",
            "global_lattice_norm",
            "transition_status",
            "slices_ready",
            "slices_needing_alignment",
            "headline",
        ]

        for field in required_fields:
            assert field in tile, f"Missing required field: {field}"

        # Verify schema version
        assert tile["schema_version"] == ATLAS_GOVERNANCE_TILE_SCHEMA_VERSION
        assert tile["tile_type"] == "atlas_governance"

    def test_atlas_governance_tile_uses_neutral_language(self) -> None:
        """Verify tile headline uses neutral language only."""
        from backend.health.atlas_governance_adapter import (
            build_atlas_governance_tile,
        )

        lattice = self._make_mock_lattice()
        phase_gate = self._make_mock_phase_gate()
        director_tile_v2 = self._make_mock_director_tile_v2()

        tile = build_atlas_governance_tile(
            lattice=lattice,
            phase_gate=phase_gate,
            director_tile_v2=director_tile_v2,
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

    def test_atlas_governance_tile_is_deterministic(self) -> None:
        """Verify tile is deterministic across multiple calls."""
        from backend.health.atlas_governance_adapter import (
            build_atlas_governance_tile,
        )

        lattice = self._make_mock_lattice()
        phase_gate = self._make_mock_phase_gate()
        director_tile_v2 = self._make_mock_director_tile_v2()

        # Call twice with same inputs
        tile1 = build_atlas_governance_tile(
            lattice=lattice,
            phase_gate=phase_gate,
            director_tile_v2=director_tile_v2,
        )
        tile2 = build_atlas_governance_tile(
            lattice=lattice,
            phase_gate=phase_gate,
            director_tile_v2=director_tile_v2,
        )

        # Serialize both and compare
        json1 = json.dumps(tile1, sort_keys=True)
        json2 = json.dumps(tile2, sort_keys=True)

        assert json1 == json2, "Tile should be deterministic"

    def test_atlas_governance_tile_status_light_propagated(self) -> None:
        """Verify status light is propagated from director_tile_v2."""
        from backend.health.atlas_governance_adapter import (
            build_atlas_governance_tile,
        )

        lattice = self._make_mock_lattice()
        phase_gate = self._make_mock_phase_gate()

        # Test GREEN
        director_tile_v2_green = {
            "status_light": "GREEN",
            "lattice_coherence": "COHERENT",
            "structural_status": "OK",
            "transition_recommendation": "OK: Phase transition appears safe",
            "atlas_ok": True,
            "headline": "Atlas director tile: 2 slices, 2 ready, lattice coherent (norm: 0.850), transition safe.",
        }

        tile_green = build_atlas_governance_tile(
            lattice=lattice,
            phase_gate=phase_gate,
            director_tile_v2=director_tile_v2_green,
        )
        assert tile_green["status_light"] == "GREEN"

        # Test RED
        director_tile_v2_red = {
            "status_light": "RED",
            "lattice_coherence": "MISALIGNED",
            "structural_status": "VOLATILE",
            "transition_recommendation": "BLOCK: Do not proceed with phase transition",
            "atlas_ok": False,
            "headline": "Atlas director tile: 2 slices, lattice norm 0.300 (MISALIGNED), transition BLOCK.",
        }

        lattice_red = {
            "convergence_band": "MISALIGNED",
            "global_lattice_norm": 0.3,
            "lattice_vectors": {},
            "neutral_notes": [],
        }
        phase_gate_red = {
            "transition_status": "BLOCK",
            "drivers": ["Lattice convergence band is misaligned"],
            "slices_ready": [],
            "slices_needing_alignment": ["slice_a", "slice_b"],
            "headline": "Phase transition blocked: lattice convergence is misaligned (norm: 0.300).",
        }

        tile_red = build_atlas_governance_tile(
            lattice=lattice_red,
            phase_gate=phase_gate_red,
            director_tile_v2=director_tile_v2_red,
        )
        assert tile_red["status_light"] == "RED"

    def test_atlas_governance_tile_no_governance_modification(self) -> None:
        """
        Verify tile does not modify governance state.

        This test ensures the tile is purely observational.
        """
        from backend.health.atlas_governance_adapter import (
            build_atlas_governance_tile,
        )

        lattice = self._make_mock_lattice()
        phase_gate = self._make_mock_phase_gate()
        director_tile_v2 = self._make_mock_director_tile_v2()

        # Create copies to verify inputs are not modified
        lattice_copy = json.loads(json.dumps(lattice))
        phase_gate_copy = json.loads(json.dumps(phase_gate))
        director_tile_v2_copy = json.loads(json.dumps(director_tile_v2))

        # Build tile
        tile = build_atlas_governance_tile(
            lattice=lattice,
            phase_gate=phase_gate,
            director_tile_v2=director_tile_v2,
        )

        # Verify inputs unchanged
        assert json.dumps(lattice, sort_keys=True) == json.dumps(
            lattice_copy, sort_keys=True
        ), "lattice should not be modified"
        assert json.dumps(phase_gate, sort_keys=True) == json.dumps(
            phase_gate_copy, sort_keys=True
        ), "phase_gate should not be modified"
        assert json.dumps(director_tile_v2, sort_keys=True) == json.dumps(
            director_tile_v2_copy, sort_keys=True
        ), "director_tile_v2 should not be modified"

        # Verify tile is a new dict (not a reference)
        assert tile is not lattice
        assert tile is not phase_gate
        assert tile is not director_tile_v2

    def test_extract_atlas_signal_for_first_light(self) -> None:
        """Verify First Light signal extraction works."""
        from backend.health.atlas_governance_adapter import (
            extract_atlas_signal_for_first_light,
        )

        lattice = self._make_mock_lattice()
        phase_gate = self._make_mock_phase_gate()

        signal = extract_atlas_signal_for_first_light(
            lattice=lattice,
            phase_gate=phase_gate,
        )

        assert isinstance(signal, dict)
        assert "global_lattice_norm" in signal
        assert "lattice_convergence_band" in signal
        assert "transition_status" in signal

        assert signal["global_lattice_norm"] == 0.85
        assert signal["lattice_convergence_band"] == "COHERENT"
        assert signal["transition_status"] == "OK"

        # Verify serializable
        json_str = json.dumps(signal, sort_keys=True)
        assert len(json_str) > 0


class TestGlobalHealthSurfaceAtlasGovernanceIntegration:
    """
    Tests for atlas governance tile integration with GlobalHealthSurface.

    SHADOW MODE: These tests verify the tile attachment mechanism only.
    """

    def test_build_global_health_surface_without_atlas_governance(self) -> None:
        """Verify build works without atlas governance inputs."""
        from backend.health.global_surface import build_global_health_surface

        payload = build_global_health_surface()

        assert isinstance(payload, dict)
        assert "schema_version" in payload
        assert "dynamics" in payload
        assert "atlas_governance" not in payload  # Should not be present

    def test_build_global_health_surface_with_atlas_governance(self) -> None:
        """Verify atlas governance tile attached when all inputs provided."""
        from backend.health.global_surface import build_global_health_surface

        # Create mock inputs
        lattice = {
            "convergence_band": "COHERENT",
            "global_lattice_norm": 0.85,
            "lattice_vectors": {},
            "neutral_notes": [],
        }
        phase_gate = {
            "transition_status": "OK",
            "drivers": [],
            "slices_ready": ["slice_a"],
            "slices_needing_alignment": [],
            "headline": "Phase transition appears safe.",
        }
        director_tile_v2 = {
            "status_light": "GREEN",
            "lattice_coherence": "COHERENT",
            "structural_status": "OK",
            "transition_recommendation": "OK: Phase transition appears safe",
            "atlas_ok": True,
            "headline": "Atlas director tile: 2 slices, 2 ready, lattice coherent.",
        }

        payload = build_global_health_surface(
            atlas_lattice=lattice,
            atlas_phase_gate=phase_gate,
            atlas_director_tile_v2=director_tile_v2,
        )

        assert isinstance(payload, dict)
        assert "atlas_governance" in payload, "Atlas governance tile should be present"

        atlas_tile = payload["atlas_governance"]
        assert isinstance(atlas_tile, dict)
        assert "headline" in atlas_tile

        # Verify serializable
        json_str = json.dumps(payload)
        assert len(json_str) > 0

    def test_atlas_governance_tile_does_not_affect_dynamics(self) -> None:
        """Verify atlas governance tile presence doesn't change dynamics tile."""
        from backend.health.global_surface import build_global_health_surface

        # Build without atlas governance
        payload_without = build_global_health_surface()

        # Build with atlas governance
        lattice = {
            "convergence_band": "COHERENT",
            "global_lattice_norm": 0.85,
            "lattice_vectors": {},
            "neutral_notes": [],
        }
        phase_gate = {
            "transition_status": "OK",
            "drivers": [],
            "slices_ready": [],
            "slices_needing_alignment": [],
            "headline": "Phase transition appears safe.",
        }
        director_tile_v2 = {
            "status_light": "GREEN",
            "lattice_coherence": "COHERENT",
            "structural_status": "OK",
            "transition_recommendation": "OK: Phase transition appears safe",
            "atlas_ok": True,
            "headline": "Atlas director tile: 2 slices, 2 ready, lattice coherent.",
        }

        payload_with = build_global_health_surface(
            atlas_lattice=lattice,
            atlas_phase_gate=phase_gate,
            atlas_director_tile_v2=director_tile_v2,
        )

        # Dynamics should be identical
        assert payload_without["dynamics"] == payload_with["dynamics"]

