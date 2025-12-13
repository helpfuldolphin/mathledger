"""
Tests for harmonic alignment P3/P4 integration, evidence attachment, and council adapter.

STATUS: PHASE X — HARMONIC GOVERNANCE TILE

Tests that harmonic alignment integrates correctly with:
- P3 First-Light summary.json
- P4 calibration reports
- Evidence packs
- Uplift council summaries
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests.factories.first_light_factories import make_summary_payload

from backend.health.harmonic_alignment_adapter import (
    attach_harmonic_alignment_to_evidence,
    build_harmonic_alignment_calibration_for_p4,
    build_harmonic_alignment_summary_for_p3,
    extract_harmonic_signal_for_curriculum,
    summarize_harmonic_for_uplift_council,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_harmonic_tile() -> dict[str, Any]:
    """Sample harmonic governance tile."""
    return {
        "schema_version": "1.0.0",
        "tile_type": "semantic_curriculum_harmonic",
        "status_light": "YELLOW",
        "harmonic_band": "PARTIAL",
        "global_harmonic_score": 0.633,
        "misaligned_concepts": ["slice_b"],
        "evolution_status": "EVOLVING",
        "prioritized_adjustments": ["slice_b", "slice_c", "slice_a"],
        "headline": "Semantic-curriculum alignment is partial: 1 concept(s) need attention",
    }


@pytest.fixture
def sample_harmonic_signal() -> dict[str, Any]:
    """Sample harmonic signal."""
    return {
        "harmonic_band": "PARTIAL",
        "num_misaligned_concepts": 1,
        "evolution_status": "EVOLVING",
    }


@pytest.fixture
def sample_p3_summary() -> dict[str, Any]:
    """Sample P3 First-Light summary structure."""
    return make_summary_payload(total_cycles=1000, seed=2025)


@pytest.fixture
def sample_p4_calibration_report() -> dict[str, Any]:
    """Sample P4 calibration report structure."""
    return {
        "schema_version": "1.0.0",
        "run_id": "p4_20251209_120000_abc123",
        "timing": {
            "start_time": "2025-12-09T12:00:00.000000+00:00",
            "end_time": "2025-12-09T12:16:40.000000+00:00",
            "cycles_observed": 1000,
        },
        "divergence_statistics": {
            "total_divergences": 15,
            "divergences_by_type": {"SUCCESS": 5, "BLOCK": 10},
        },
        "accuracy_metrics": {
            "success_accuracy": 0.95,
            "blocked_accuracy": 0.92,
        },
    }


@pytest.fixture
def sample_evidence() -> dict[str, Any]:
    """Sample evidence pack structure."""
    return {
        "timestamp": "2025-12-09T12:00:00.000000+00:00",
        "run_id": "test_run_123",
        "data": {
            "metrics": {"success_rate": 0.85},
        },
    }


# =============================================================================
# TEST GROUP 1: P3 FIRST-LIGHT INTEGRATION
# =============================================================================


class TestBuildHarmonicAlignmentSummaryForP3:
    """Tests for P3 First-Light summary integration."""

    def test_01_summary_has_required_keys(self, sample_harmonic_tile: dict[str, Any]) -> None:
        """Test that summary has all required keys."""
        summary = build_harmonic_alignment_summary_for_p3(sample_harmonic_tile)

        required_keys = {
            "global_harmonic_score",
            "harmonic_band",
            "misaligned_concepts",
            "priority_adjustments",
            "status_light",
        }
        assert required_keys.issubset(set(summary.keys()))

    def test_02_summary_extracts_correct_values(self, sample_harmonic_tile: dict[str, Any]) -> None:
        """Test that summary extracts correct values from tile."""
        summary = build_harmonic_alignment_summary_for_p3(sample_harmonic_tile)

        assert summary["global_harmonic_score"] == 0.633
        assert summary["harmonic_band"] == "PARTIAL"
        assert summary["misaligned_concepts"] == ["slice_b"]
        assert summary["priority_adjustments"] == ["slice_b", "slice_c", "slice_a"]
        assert summary["status_light"] == "YELLOW"

    def test_03_summary_serializes_to_json(self, sample_harmonic_tile: dict[str, Any]) -> None:
        """Test that summary can be serialized to JSON."""
        summary = build_harmonic_alignment_summary_for_p3(sample_harmonic_tile)

        json_str = json.dumps(summary)
        assert json_str

        deserialized = json.loads(json_str)
        assert deserialized == summary

    def test_04_summary_is_deterministic(self, sample_harmonic_tile: dict[str, Any]) -> None:
        """Test that summary is deterministic."""
        summaries = [
            build_harmonic_alignment_summary_for_p3(sample_harmonic_tile) for _ in range(5)
        ]

        for i in range(1, len(summaries)):
            assert summaries[0] == summaries[i]


# =============================================================================
# TEST GROUP 2: P4 CALIBRATION INTEGRATION
# =============================================================================


class TestBuildHarmonicAlignmentCalibrationForP4:
    """Tests for P4 calibration report integration."""

    def test_05_calibration_has_required_keys(
        self, sample_harmonic_tile: dict[str, Any], sample_harmonic_signal: dict[str, Any]
    ) -> None:
        """Test that calibration has all required keys."""
        calibration = build_harmonic_alignment_calibration_for_p4(
            sample_harmonic_tile, sample_harmonic_signal
        )

        required_keys = {
            "harmonic_band",
            "misaligned_concepts",
            "evolution_status",
            "priority_adjustments",
            "global_harmonic_score",
        }
        assert required_keys.issubset(set(calibration.keys()))

    def test_06_calibration_extracts_correct_values(
        self, sample_harmonic_tile: dict[str, Any], sample_harmonic_signal: dict[str, Any]
    ) -> None:
        """Test that calibration extracts correct values."""
        calibration = build_harmonic_alignment_calibration_for_p4(
            sample_harmonic_tile, sample_harmonic_signal
        )

        assert calibration["harmonic_band"] == "PARTIAL"
        assert calibration["misaligned_concepts"] == ["slice_b"]
        assert calibration["evolution_status"] == "EVOLVING"
        assert calibration["priority_adjustments"] == ["slice_b", "slice_c", "slice_a"]
        assert calibration["global_harmonic_score"] == 0.633

    def test_07_calibration_serializes_to_json(
        self, sample_harmonic_tile: dict[str, Any], sample_harmonic_signal: dict[str, Any]
    ) -> None:
        """Test that calibration can be serialized to JSON."""
        calibration = build_harmonic_alignment_calibration_for_p4(
            sample_harmonic_tile, sample_harmonic_signal
        )

        json_str = json.dumps(calibration)
        assert json_str

        deserialized = json.loads(json_str)
        assert deserialized == calibration


# =============================================================================
# TEST GROUP 3: EVIDENCE ATTACHMENT
# =============================================================================


class TestAttachHarmonicAlignmentToEvidence:
    """Tests for evidence pack integration."""

    def test_08_attaches_to_evidence(
        self, sample_evidence: dict[str, Any], sample_harmonic_tile: dict[str, Any], sample_harmonic_signal: dict[str, Any]
    ) -> None:
        """Test that harmonic alignment is attached to evidence."""
        enriched = attach_harmonic_alignment_to_evidence(
            sample_evidence, sample_harmonic_tile, sample_harmonic_signal
        )

        assert "governance" in enriched
        assert "harmonic_alignment" in enriched["governance"]

        harmonic = enriched["governance"]["harmonic_alignment"]
        assert harmonic["harmonic_band"] == "PARTIAL"
        assert harmonic["score"] == 0.633
        assert harmonic["misaligned_concepts"] == ["slice_b"]

    def test_09_evidence_non_mutating(
        self, sample_evidence: dict[str, Any], sample_harmonic_tile: dict[str, Any], sample_harmonic_signal: dict[str, Any]
    ) -> None:
        """Test that evidence attachment does not mutate input."""
        original = dict(sample_evidence)

        attach_harmonic_alignment_to_evidence(
            sample_evidence, sample_harmonic_tile, sample_harmonic_signal
        )

        assert sample_evidence == original
        assert "governance" not in sample_evidence

    def test_10_evidence_has_minimal_fields(
        self, sample_evidence: dict[str, Any], sample_harmonic_tile: dict[str, Any], sample_harmonic_signal: dict[str, Any]
    ) -> None:
        """Test that evidence has minimal required fields."""
        enriched = attach_harmonic_alignment_to_evidence(
            sample_evidence, sample_harmonic_tile, sample_harmonic_signal
        )

        harmonic = enriched["governance"]["harmonic_alignment"]
        required_fields = {"harmonic_band", "score", "misaligned_concepts"}
        assert required_fields.issubset(set(harmonic.keys()))

    def test_11_evidence_serializes_to_json(
        self, sample_evidence: dict[str, Any], sample_harmonic_tile: dict[str, Any], sample_harmonic_signal: dict[str, Any]
    ) -> None:
        """Test that enriched evidence can be serialized to JSON."""
        enriched = attach_harmonic_alignment_to_evidence(
            sample_evidence, sample_harmonic_tile, sample_harmonic_signal
        )

        json_str = json.dumps(enriched)
        assert json_str

        deserialized = json.loads(json_str)
        assert "governance" in deserialized
        assert "harmonic_alignment" in deserialized["governance"]


# =============================================================================
# TEST GROUP 4: UPLIFT COUNCIL ADAPTER
# =============================================================================


class TestSummarizeHarmonicForUpliftCouncil:
    """Tests for uplift council adapter."""

    def test_12_council_summary_has_required_keys(self, sample_harmonic_tile: dict[str, Any]) -> None:
        """Test that council summary has all required keys."""
        summary = summarize_harmonic_for_uplift_council(sample_harmonic_tile)

        required_keys = {
            "status",
            "misaligned_concepts",
            "priority_adjustments",
        }
        assert required_keys.issubset(set(summary.keys()))

    def test_13_council_maps_mismatched_to_block(self) -> None:
        """Test that MISMATCHED band maps to BLOCK status."""
        tile = {
            "harmonic_band": "MISMATCHED",
            "misaligned_concepts": ["slice_a", "slice_b"],
            "prioritized_adjustments": ["slice_a"],
        }

        summary = summarize_harmonic_for_uplift_council(tile)

        assert summary["status"] == "BLOCK"

    def test_14_council_maps_partial_to_warn(self) -> None:
        """Test that PARTIAL band maps to WARN status."""
        tile = {
            "harmonic_band": "PARTIAL",
            "misaligned_concepts": ["slice_b"],
            "prioritized_adjustments": ["slice_b"],
        }

        summary = summarize_harmonic_for_uplift_council(tile)

        assert summary["status"] == "WARN"

    def test_15_council_maps_coherent_to_ok(self) -> None:
        """Test that COHERENT band maps to OK status."""
        tile = {
            "harmonic_band": "COHERENT",
            "misaligned_concepts": [],
            "prioritized_adjustments": [],
        }

        summary = summarize_harmonic_for_uplift_council(tile)

        assert summary["status"] == "OK"

    def test_16_council_extracts_misaligned_concepts(self, sample_harmonic_tile: dict[str, Any]) -> None:
        """Test that council summary extracts misaligned concepts."""
        summary = summarize_harmonic_for_uplift_council(sample_harmonic_tile)

        assert summary["misaligned_concepts"] == ["slice_b"]
        assert summary["misaligned_concepts"] == sorted(summary["misaligned_concepts"])

    def test_17_council_extracts_priority_adjustments(self, sample_harmonic_tile: dict[str, Any]) -> None:
        """Test that council summary extracts priority adjustments."""
        summary = summarize_harmonic_for_uplift_council(sample_harmonic_tile)

        assert summary["priority_adjustments"] == ["slice_b", "slice_c", "slice_a"]

    def test_18_council_serializes_to_json(self, sample_harmonic_tile: dict[str, Any]) -> None:
        """Test that council summary can be serialized to JSON."""
        summary = summarize_harmonic_for_uplift_council(sample_harmonic_tile)

        json_str = json.dumps(summary)
        assert json_str

        deserialized = json.loads(json_str)
        assert deserialized == summary
