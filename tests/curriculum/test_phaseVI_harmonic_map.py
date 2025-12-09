"""
Phase VI — Semantic-Curriculum Harmonic Map Tests

Agent: doc-ops-2 (E2) — Curriculum–Semantic Mapper & Evolution Planner
Tests for harmonic map, evolution forecaster, and director panel functionality.

Run with: uv run pytest tests/curriculum/test_phaseVI_harmonic_map.py -v
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from scripts.schema_ontology_builder import (
    build_curriculum_evolution_forecaster,
    build_harmonic_director_panel,
    build_semantic_curriculum_harmonic_map,
)


# =============================================================================
# TEST GROUP 1: HARMONIC MAP (Tests 1-8)
# =============================================================================


class TestSemanticCurriculumHarmonicMap:
    """Tests for build_semantic_curriculum_harmonic_map functionality."""

    @pytest.fixture
    def sample_semantic_alignment(self) -> dict[str, Any]:
        """Create sample semantic alignment data."""
        return {
            "slice_alignment": {
                "slice_a": True,
                "slice_b": True,
                "slice_c": False,
            }
        }

    @pytest.fixture
    def sample_curriculum_alignment(self) -> dict[str, Any]:
        """Create sample curriculum alignment data."""
        return {
            "slice_alignment": {
                "slice_a": True,
                "slice_b": False,  # Misaligned: semantic_ok=True but curriculum_ok=False
                "slice_c": False,
            }
        }

    @pytest.fixture
    def sample_atlas_coupling(self) -> dict[str, Any]:
        """Create sample atlas coupling view."""
        return {
            "slices_with_atlas_support": ["slice_a", "slice_b"],
            "slices_without_atlas_support": ["slice_c"],
            "coupling_status": "LOOSE",
        }

    def test_01_harmonic_map_has_required_keys(
        self,
        sample_semantic_alignment: dict[str, Any],
        sample_curriculum_alignment: dict[str, Any],
        sample_atlas_coupling: dict[str, Any],
    ) -> None:
        """Test that harmonic map has all required keys."""
        result = build_semantic_curriculum_harmonic_map(
            sample_semantic_alignment, sample_curriculum_alignment, sample_atlas_coupling
        )

        required_keys = {
            "harmonic_scores",
            "convergence_band",
            "misaligned_concepts",
            "neutral_notes",
        }
        assert required_keys.issubset(set(result.keys()))

    def test_02_harmonic_map_calculates_scores(
        self,
        sample_semantic_alignment: dict[str, Any],
        sample_curriculum_alignment: dict[str, Any],
        sample_atlas_coupling: dict[str, Any],
    ) -> None:
        """Test that harmonic scores are calculated correctly."""
        result = build_semantic_curriculum_harmonic_map(
            sample_semantic_alignment, sample_curriculum_alignment, sample_atlas_coupling
        )

        scores = result["harmonic_scores"]
        assert "slice_a" in scores
        assert "slice_b" in scores
        assert "slice_c" in scores

        # slice_a: semantic=True, curriculum=True, atlas=True → 1.0
        assert scores["slice_a"] == 1.0

        # slice_b: semantic=True, curriculum=False, atlas=True → 0.4*1 + 0.4*0 + 0.2*1 = 0.6
        assert abs(scores["slice_b"] - 0.6) < 0.01

    def test_03_harmonic_map_detects_misaligned_concepts(
        self,
        sample_semantic_alignment: dict[str, Any],
        sample_curriculum_alignment: dict[str, Any],
        sample_atlas_coupling: dict[str, Any],
    ) -> None:
        """Test that misaligned concepts are detected (semantic_ok=True, curriculum_ok=False)."""
        result = build_semantic_curriculum_harmonic_map(
            sample_semantic_alignment, sample_curriculum_alignment, sample_atlas_coupling
        )

        misaligned = result["misaligned_concepts"]
        assert "slice_b" in misaligned  # semantic_ok=True but curriculum_ok=False
        assert "slice_c" not in misaligned  # both False, so not misaligned

    def test_04_harmonic_map_convergence_band_coherent(self) -> None:
        """Test that convergence band is COHERENT when scores are high."""
        semantic = {"slice_alignment": {"slice1": True, "slice2": True}}
        curriculum = {"slice_alignment": {"slice1": True, "slice2": True}}
        atlas = {"slices_with_atlas_support": ["slice1", "slice2"]}

        result = build_semantic_curriculum_harmonic_map(semantic, curriculum, atlas)

        assert result["convergence_band"] == "COHERENT"

    def test_05_harmonic_map_convergence_band_partial(self) -> None:
        """Test that convergence band is PARTIAL when scores are moderate."""
        semantic = {"slice_alignment": {"slice1": True, "slice2": False}}
        curriculum = {"slice_alignment": {"slice1": True, "slice2": False}}
        atlas = {"slices_with_atlas_support": ["slice1"]}

        result = build_semantic_curriculum_harmonic_map(semantic, curriculum, atlas)

        assert result["convergence_band"] == "PARTIAL"

    def test_06_harmonic_map_convergence_band_mismatched(self) -> None:
        """Test that convergence band is MISMATCHED when scores are low."""
        semantic = {"slice_alignment": {"slice1": False, "slice2": False}}
        curriculum = {"slice_alignment": {"slice1": False, "slice2": False}}
        atlas = {"slices_with_atlas_support": []}

        result = build_semantic_curriculum_harmonic_map(semantic, curriculum, atlas)

        assert result["convergence_band"] == "MISMATCHED"

    def test_07_harmonic_map_deterministic(
        self,
        sample_semantic_alignment: dict[str, Any],
        sample_curriculum_alignment: dict[str, Any],
        sample_atlas_coupling: dict[str, Any],
    ) -> None:
        """Test that harmonic map is deterministic."""
        results = [
            build_semantic_curriculum_harmonic_map(
                sample_semantic_alignment, sample_curriculum_alignment, sample_atlas_coupling
            )
            for _ in range(5)
        ]

        for i in range(1, len(results)):
            assert results[0] == results[i]

    def test_08_harmonic_map_neutral_notes(
        self,
        sample_semantic_alignment: dict[str, Any],
        sample_curriculum_alignment: dict[str, Any],
        sample_atlas_coupling: dict[str, Any],
    ) -> None:
        """Test that notes use neutral language."""
        result = build_semantic_curriculum_harmonic_map(
            sample_semantic_alignment, sample_curriculum_alignment, sample_atlas_coupling
        )

        for note in result["neutral_notes"]:
            lower = note.lower()
            assert "good" not in lower
            assert "bad" not in lower
            assert "healthy" not in lower
            assert "unhealthy" not in lower


# =============================================================================
# TEST GROUP 2: EVOLUTION FORECASTER (Tests 9-14)
# =============================================================================


class TestCurriculumEvolutionForecaster:
    """Tests for build_curriculum_evolution_forecaster functionality."""

    @pytest.fixture
    def sample_harmonic_map_coherent(self) -> dict[str, Any]:
        """Create sample coherent harmonic map."""
        return {
            "harmonic_scores": {
                "slice_a": 1.0,
                "slice_b": 0.95,
            },
            "convergence_band": "COHERENT",
            "misaligned_concepts": [],
            "neutral_notes": [],
        }

    @pytest.fixture
    def sample_harmonic_map_partial(self) -> dict[str, Any]:
        """Create sample partial harmonic map with misalignments."""
        return {
            "harmonic_scores": {
                "slice_a": 0.6,  # Moderate score
                "slice_b": 0.3,  # Low score
            },
            "convergence_band": "PARTIAL",
            "misaligned_concepts": ["slice_b"],
            "neutral_notes": [],
        }

    def test_09_forecaster_has_required_keys(self, sample_harmonic_map_partial: dict[str, Any]) -> None:
        """Test that forecaster has all required keys."""
        result = build_curriculum_evolution_forecaster(sample_harmonic_map_partial)

        required_keys = {
            "forecasted_adjustments",
            "forecast_status",
            "neutral_notes",
        }
        assert required_keys.issubset(set(result.keys()))

    def test_10_forecaster_generates_adjustments_for_misaligned(
        self, sample_harmonic_map_partial: dict[str, Any]
    ) -> None:
        """Test that forecaster generates adjustments for misaligned concepts."""
        result = build_curriculum_evolution_forecaster(sample_harmonic_map_partial)

        adjustments = result["forecasted_adjustments"]
        assert len(adjustments) > 0

        # Should have adjustment for slice_b (misaligned)
        slice_b_adjustments = [a for a in adjustments if a["slice"] == "slice_b"]
        assert len(slice_b_adjustments) == 1
        assert slice_b_adjustments[0]["priority"] == "HIGH"  # Low score = HIGH priority

    def test_11_forecaster_status_stable_for_coherent(self, sample_harmonic_map_coherent: dict[str, Any]) -> None:
        """Test that forecast status is STABLE for coherent maps."""
        result = build_curriculum_evolution_forecaster(sample_harmonic_map_coherent)

        assert result["forecast_status"] == "STABLE"

    def test_12_forecaster_status_evolving_for_partial(self, sample_harmonic_map_partial: dict[str, Any]) -> None:
        """Test that forecast status is EVOLVING for partial maps."""
        result = build_curriculum_evolution_forecaster(sample_harmonic_map_partial)

        assert result["forecast_status"] == "EVOLVING"

    def test_13_forecaster_adjustments_sorted_by_priority(self, sample_harmonic_map_partial: dict[str, Any]) -> None:
        """Test that adjustments are sorted by priority (HIGH first)."""
        result = build_curriculum_evolution_forecaster(sample_harmonic_map_partial)

        adjustments = result["forecasted_adjustments"]
        if len(adjustments) > 1:
            priorities = [a["priority"] for a in adjustments]
            priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
            assert all(
                priority_order.get(priorities[i], 99) <= priority_order.get(priorities[i + 1], 99)
                for i in range(len(priorities) - 1)
            )

    def test_14_forecaster_neutral_reasons(self, sample_harmonic_map_partial: dict[str, Any]) -> None:
        """Test that adjustment reasons use neutral language."""
        result = build_curriculum_evolution_forecaster(sample_harmonic_map_partial)

        for adjustment in result["forecasted_adjustments"]:
            reason = adjustment["neutral_reason"].lower()
            assert "wrong" not in reason
            assert "bad" not in reason
            assert "error" not in reason


# =============================================================================
# TEST GROUP 3: HARMONIC DIRECTOR PANEL (Tests 15-20)
# =============================================================================


class TestHarmonicDirectorPanel:
    """Tests for build_harmonic_director_panel functionality."""

    @pytest.fixture
    def sample_harmonic_map_coherent(self) -> dict[str, Any]:
        """Create sample coherent harmonic map."""
        return {
            "convergence_band": "COHERENT",
            "misaligned_concepts": [],
            "harmonic_scores": {"slice_a": 1.0},
        }

    @pytest.fixture
    def sample_harmonic_map_mismatched(self) -> dict[str, Any]:
        """Create sample mismatched harmonic map."""
        return {
            "convergence_band": "MISMATCHED",
            "misaligned_concepts": ["slice_a", "slice_b"],
            "harmonic_scores": {"slice_a": 0.3, "slice_b": 0.2},
        }

    @pytest.fixture
    def sample_evolution_forecast_stable(self) -> dict[str, Any]:
        """Create sample stable evolution forecast."""
        return {
            "forecast_status": "STABLE",
            "forecasted_adjustments": [],
        }

    @pytest.fixture
    def sample_evolution_forecast_diverging(self) -> dict[str, Any]:
        """Create sample diverging evolution forecast."""
        return {
            "forecast_status": "DIVERGING",
            "forecasted_adjustments": [{"slice": "slice_a", "priority": "HIGH"}],
        }

    @pytest.fixture
    def sample_d6_lattice_status(self) -> dict[str, Any]:
        """Create sample D6 lattice status."""
        return {
            "status_light": "YELLOW",
            "status": "ATTENTION",
        }

    @pytest.fixture
    def sample_c2_drift_risk(self) -> dict[str, Any]:
        """Create sample C2 drift risk."""
        return {
            "risk_level": "MEDIUM",
            "status": "WARN",
        }

    def test_15_director_panel_has_required_keys(
        self, sample_harmonic_map_coherent: dict[str, Any]
    ) -> None:
        """Test that director panel has all required keys."""
        result = build_harmonic_director_panel(sample_harmonic_map_coherent)

        required_keys = {
            "status_light",
            "convergence_band",
            "forecast_status",
            "misaligned_count",
            "headline",
            "integrated_risks",
        }
        assert required_keys.issubset(set(result.keys()))

    def test_16_director_panel_green_for_coherent(
        self, sample_harmonic_map_coherent: dict[str, Any], sample_evolution_forecast_stable: dict[str, Any]
    ) -> None:
        """Test that status light is GREEN for coherent alignment."""
        result = build_harmonic_director_panel(
            sample_harmonic_map_coherent, evolution_forecast=sample_evolution_forecast_stable
        )

        assert result["status_light"] == "GREEN"
        assert result["convergence_band"] == "COHERENT"

    def test_17_director_panel_red_for_mismatched(
        self, sample_harmonic_map_mismatched: dict[str, Any], sample_evolution_forecast_diverging: dict[str, Any]
    ) -> None:
        """Test that status light is RED for mismatched alignment."""
        result = build_harmonic_director_panel(
            sample_harmonic_map_mismatched, evolution_forecast=sample_evolution_forecast_diverging
        )

        assert result["status_light"] == "RED"
        assert result["misaligned_count"] == 2

    def test_18_director_panel_integrates_d6_lattice(
        self,
        sample_harmonic_map_coherent: dict[str, Any],
        sample_d6_lattice_status: dict[str, Any],
    ) -> None:
        """Test that director panel integrates D6 lattice status."""
        result = build_harmonic_director_panel(
            sample_harmonic_map_coherent, d6_lattice_status=sample_d6_lattice_status
        )

        # D6 status YELLOW should elevate GREEN to YELLOW
        assert result["status_light"] == "YELLOW"
        assert any("D6" in risk for risk in result["integrated_risks"])

    def test_19_director_panel_integrates_c2_drift(
        self,
        sample_harmonic_map_coherent: dict[str, Any],
        sample_c2_drift_risk: dict[str, Any],
    ) -> None:
        """Test that director panel integrates C2 drift risk."""
        result = build_harmonic_director_panel(sample_harmonic_map_coherent, c2_drift_risk=sample_c2_drift_risk)

        # C2 risk MEDIUM should elevate GREEN to YELLOW
        assert result["status_light"] == "YELLOW"
        assert any("C2" in risk for risk in result["integrated_risks"])

    def test_20_director_panel_headline_neutral(
        self, sample_harmonic_map_mismatched: dict[str, Any]
    ) -> None:
        """Test that headline uses neutral language."""
        result = build_harmonic_director_panel(sample_harmonic_map_mismatched)

        headline = result["headline"].lower()
        assert "good" not in headline
        assert "bad" not in headline
        assert "healthy" not in headline
        assert "unhealthy" not in headline

