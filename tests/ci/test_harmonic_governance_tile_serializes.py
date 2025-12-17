"""
CI test for harmonic governance tile serialization and determinism.

STATUS: PHASE X — HARMONIC GOVERNANCE TILE

Tests that the harmonic governance tile:
- Serializes to JSON correctly
- Is deterministic (same inputs produce same outputs)
- Uses neutral language in headlines
- Has all required keys
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from backend.health.harmonic_alignment_adapter import (
    build_harmonic_governance_tile,
    extract_harmonic_signal_for_curriculum,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def synthetic_harmonic_map() -> dict[str, Any]:
    """Create synthetic harmonic map for testing."""
    return {
        "harmonic_scores": {
            "slice_a": 1.0,
            "slice_b": 0.6,
            "slice_c": 0.3,
        },
        "convergence_band": "PARTIAL",
        "misaligned_concepts": ["slice_b"],
        "neutral_notes": ["Average harmonic score: 0.633 across 3 slice(s)"],
    }


@pytest.fixture
def synthetic_evolution_forecaster() -> dict[str, Any]:
    """Create synthetic evolution forecaster for testing."""
    return {
        "forecast_status": "EVOLVING",
        "forecasted_adjustments": [
            {
                "slice": "slice_b",
                "adjustment_kind": "curriculum_sync",
                "priority": "HIGH",
                "neutral_reason": "Slice 'slice_b' has low harmonic score (0.600), curriculum alignment needed",
            },
            {
                "slice": "slice_c",
                "adjustment_kind": "curriculum_sync",
                "priority": "HIGH",
                "neutral_reason": "Slice 'slice_c' has low harmonic score (0.300), curriculum alignment needed",
            },
            {
                "slice": "slice_a",
                "adjustment_kind": "atlas_sync",
                "priority": "LOW",
                "neutral_reason": "Slice 'slice_a' has good alignment (1.000), atlas coupling could be improved",
            },
        ],
        "neutral_notes": ["Alignment is partial: 1 concept(s) need adjustment"],
    }


@pytest.fixture
def synthetic_harmonic_director_panel() -> dict[str, Any]:
    """Create synthetic harmonic director panel for testing."""
    return {
        "status_light": "YELLOW",
        "convergence_band": "PARTIAL",
        "forecast_status": "EVOLVING",
        "misaligned_count": 1,
        "headline": "Semantic-curriculum alignment is partial: 1 concept(s) need attention",
        "integrated_risks": ["Harmonic map shows partial alignment", "Evolution forecast indicates ongoing adjustments needed"],
    }


# =============================================================================
# TEST GROUP 1: TILE SERIALIZATION (Tests 1-4)
# =============================================================================


class TestHarmonicGovernanceTileSerialization:
    """Tests for harmonic governance tile JSON serialization."""

    def test_01_tile_serializes_to_json(
        self,
        synthetic_harmonic_map: dict[str, Any],
        synthetic_evolution_forecaster: dict[str, Any],
        synthetic_harmonic_director_panel: dict[str, Any],
    ) -> None:
        """Test that tile can be serialized to JSON."""
        tile = build_harmonic_governance_tile(
            harmonic_map=synthetic_harmonic_map,
            evolution_forecaster=synthetic_evolution_forecaster,
            harmonic_director_panel=synthetic_harmonic_director_panel,
        )

        # Should not raise
        json_str = json.dumps(tile)
        assert json_str

        # Should be able to deserialize
        deserialized = json.loads(json_str)
        assert deserialized == tile

    def test_02_tile_has_required_keys(
        self,
        synthetic_harmonic_map: dict[str, Any],
        synthetic_evolution_forecaster: dict[str, Any],
        synthetic_harmonic_director_panel: dict[str, Any],
    ) -> None:
        """Test that tile has all required keys."""
        tile = build_harmonic_governance_tile(
            harmonic_map=synthetic_harmonic_map,
            evolution_forecaster=synthetic_evolution_forecaster,
            harmonic_director_panel=synthetic_harmonic_director_panel,
        )

        required_keys = {
            "schema_version",
            "tile_type",
            "status_light",
            "harmonic_band",
            "global_harmonic_score",
            "misaligned_concepts",
            "evolution_status",
            "prioritized_adjustments",
            "headline",
        }
        assert required_keys.issubset(set(tile.keys()))

    def test_03_tile_is_deterministic(
        self,
        synthetic_harmonic_map: dict[str, Any],
        synthetic_evolution_forecaster: dict[str, Any],
        synthetic_harmonic_director_panel: dict[str, Any],
    ) -> None:
        """Test that tile is deterministic (same inputs produce same outputs)."""
        tiles = [
            build_harmonic_governance_tile(
                harmonic_map=synthetic_harmonic_map,
                evolution_forecaster=synthetic_evolution_forecaster,
                harmonic_director_panel=synthetic_harmonic_director_panel,
            )
            for _ in range(5)
        ]

        # All tiles should be identical
        for i in range(1, len(tiles)):
            assert tiles[0] == tiles[i]

    def test_04_tile_headline_is_neutral(
        self,
        synthetic_harmonic_map: dict[str, Any],
        synthetic_evolution_forecaster: dict[str, Any],
        synthetic_harmonic_director_panel: dict[str, Any],
    ) -> None:
        """Test that headline uses neutral language."""
        tile = build_harmonic_governance_tile(
            harmonic_map=synthetic_harmonic_map,
            evolution_forecaster=synthetic_evolution_forecaster,
            harmonic_director_panel=synthetic_harmonic_director_panel,
        )

        headline = tile["headline"].lower()
        forbidden_words = {
            "good", "bad", "better", "worse", "improve", "improvement",
            "should", "must", "need", "required", "fail", "success",
            "correct", "incorrect", "right", "wrong", "fix", "broken",
        }

        for word in forbidden_words:
            assert word not in headline, f"Forbidden word '{word}' found in headline: {tile['headline']}"


# =============================================================================
# TEST GROUP 2: TILE CONTENT (Tests 5-8)
# =============================================================================


class TestHarmonicGovernanceTileContent:
    """Tests for harmonic governance tile content correctness."""

    def test_05_tile_extracts_status_light(
        self,
        synthetic_harmonic_map: dict[str, Any],
        synthetic_evolution_forecaster: dict[str, Any],
        synthetic_harmonic_director_panel: dict[str, Any],
    ) -> None:
        """Test that status_light is extracted from director panel."""
        tile = build_harmonic_governance_tile(
            harmonic_map=synthetic_harmonic_map,
            evolution_forecaster=synthetic_evolution_forecaster,
            harmonic_director_panel=synthetic_harmonic_director_panel,
        )

        assert tile["status_light"] == "YELLOW"
        assert tile["status_light"] == synthetic_harmonic_director_panel["status_light"]

    def test_06_tile_calculates_global_harmonic_score(
        self,
        synthetic_harmonic_map: dict[str, Any],
        synthetic_evolution_forecaster: dict[str, Any],
        synthetic_harmonic_director_panel: dict[str, Any],
    ) -> None:
        """Test that global_harmonic_score is calculated correctly (mean of slice scores)."""
        tile = build_harmonic_governance_tile(
            harmonic_map=synthetic_harmonic_map,
            evolution_forecaster=synthetic_evolution_forecaster,
            harmonic_director_panel=synthetic_harmonic_director_panel,
        )

        # Expected: (1.0 + 0.6 + 0.3) / 3 = 0.633333...
        expected_score = (1.0 + 0.6 + 0.3) / 3
        assert abs(tile["global_harmonic_score"] - expected_score) < 0.0001

    def test_07_tile_extracts_top_3_adjustments(
        self,
        synthetic_harmonic_map: dict[str, Any],
        synthetic_evolution_forecaster: dict[str, Any],
        synthetic_harmonic_director_panel: dict[str, Any],
    ) -> None:
        """Test that prioritized_adjustments contains top 3 slice names (HIGH priority first)."""
        tile = build_harmonic_governance_tile(
            harmonic_map=synthetic_harmonic_map,
            evolution_forecaster=synthetic_evolution_forecaster,
            harmonic_director_panel=synthetic_harmonic_director_panel,
        )

        adjustments = tile["prioritized_adjustments"]
        assert len(adjustments) <= 3
        # Should be slice names only (not full adjustment dicts)
        assert all(isinstance(adj, str) for adj in adjustments)
        # Should include HIGH priority slices first
        assert "slice_b" in adjustments or "slice_c" in adjustments

    def test_08_tile_extracts_misaligned_concepts(
        self,
        synthetic_harmonic_map: dict[str, Any],
        synthetic_evolution_forecaster: dict[str, Any],
        synthetic_harmonic_director_panel: dict[str, Any],
    ) -> None:
        """Test that misaligned_concepts are extracted and sorted."""
        tile = build_harmonic_governance_tile(
            harmonic_map=synthetic_harmonic_map,
            evolution_forecaster=synthetic_evolution_forecaster,
            harmonic_director_panel=synthetic_harmonic_director_panel,
        )

        assert tile["misaligned_concepts"] == ["slice_b"]
        assert tile["misaligned_concepts"] == sorted(tile["misaligned_concepts"])


# =============================================================================
# TEST GROUP 3: CURRICULUM SIGNAL EXTRACTOR (Tests 9-10)
# =============================================================================


class TestHarmonicSignalExtractor:
    """Tests for extract_harmonic_signal_for_curriculum helper."""

    def test_09_signal_extractor_has_required_keys(
        self,
        synthetic_harmonic_map: dict[str, Any],
        synthetic_evolution_forecaster: dict[str, Any],
    ) -> None:
        """Test that extracted signal has all required keys."""
        signal = extract_harmonic_signal_for_curriculum(
            harmonic_map=synthetic_harmonic_map,
            evolution_forecaster=synthetic_evolution_forecaster,
        )

        required_keys = {
            "harmonic_band",
            "num_misaligned_concepts",
            "evolution_status",
        }
        assert required_keys.issubset(set(signal.keys()))

    def test_10_signal_extractor_values_correct(
        self,
        synthetic_harmonic_map: dict[str, Any],
        synthetic_evolution_forecaster: dict[str, Any],
    ) -> None:
        """Test that extracted signal values are correct."""
        signal = extract_harmonic_signal_for_curriculum(
            harmonic_map=synthetic_harmonic_map,
            evolution_forecaster=synthetic_evolution_forecaster,
        )

        assert signal["harmonic_band"] == "PARTIAL"
        assert signal["num_misaligned_concepts"] == 1
        assert signal["evolution_status"] == "EVOLVING"

