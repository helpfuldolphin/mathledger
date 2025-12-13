"""
Tests for harmonic curriculum annex integration.

STATUS: PHASE X — HARMONIC CURRICULUM ANNEX

IMPORTANT: This annex is intended as a curriculum health diagnostic for reviewers,
not a control signal. The annex is evidence-only and does not gate or block any
operations. It provides observational data for curriculum alignment assessment.

Tests that:
- Curriculum annex is built correctly from P3/P4 data
- Evidence attachment includes curriculum annex when P3/P4 provided
- Council summary includes counts
- All outputs are JSON-safe, deterministic, and non-mutating
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from backend.health.harmonic_alignment_adapter import (
    attach_harmonic_alignment_to_evidence,
    summarize_harmonic_for_uplift_council,
)
from backend.health.harmonic_alignment_p3p4_integration import (
    build_curriculum_harmonic_annex,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_p3_summary() -> dict[str, Any]:
    """Sample P3 summary with harmonic_alignment_summary."""
    return {
        "harmonic_band": "PARTIAL",
        "misaligned_concepts": ["slice_b", "slice_c"],
        "priority_adjustments": ["slice_b", "slice_c", "slice_a", "slice_d", "slice_e", "slice_f"],
        "status_light": "YELLOW",
        "global_harmonic_score": 0.633,
    }


@pytest.fixture
def sample_p4_calibration() -> dict[str, Any]:
    """Sample P4 calibration with harmonic_alignment."""
    return {
        "harmonic_band": "PARTIAL",
        "misaligned_concepts": ["slice_b", "slice_d"],  # Overlap with P3
        "evolution_status": "EVOLVING",
        "priority_adjustments": ["slice_b"],
        "global_harmonic_score": 0.633,
    }


@pytest.fixture
def sample_tile() -> dict[str, Any]:
    """Sample harmonic governance tile."""
    return {
        "harmonic_band": "PARTIAL",
        "global_harmonic_score": 0.633,
        "misaligned_concepts": ["slice_b"],
        "prioritized_adjustments": ["slice_b", "slice_c", "slice_a"],
    }


@pytest.fixture
def sample_signal() -> dict[str, Any]:
    """Sample harmonic signal."""
    return {
        "harmonic_band": "PARTIAL",
        "num_misaligned_concepts": 1,
        "evolution_status": "EVOLVING",
    }


@pytest.fixture
def sample_evidence() -> dict[str, Any]:
    """Sample evidence pack."""
    return {
        "timestamp": "2025-12-09T12:00:00.000000+00:00",
        "run_id": "test_run_123",
        "data": {"metrics": {"success_rate": 0.85}},
    }


# =============================================================================
# TEST GROUP 1: CURRICULUM ANNEX
# =============================================================================


class TestBuildCurriculumHarmonicAnnex:
    """Tests for build_curriculum_harmonic_annex."""

    def test_01_annex_has_required_keys(
        self, sample_p3_summary: dict[str, Any], sample_p4_calibration: dict[str, Any]
    ) -> None:
        """Test that annex has all required keys."""
        annex = build_curriculum_harmonic_annex(sample_p3_summary, sample_p4_calibration)

        required_keys = {
            "schema_version",
            "harmonic_band",
            "evolution_status",
            "misaligned_concepts",
            "priority_adjustments",
        }
        assert required_keys.issubset(set(annex.keys()))

    def test_02_annex_combines_p3_p4_data(
        self, sample_p3_summary: dict[str, Any], sample_p4_calibration: dict[str, Any]
    ) -> None:
        """Test that annex combines data from P3 and P4."""
        annex = build_curriculum_harmonic_annex(sample_p3_summary, sample_p4_calibration)

        assert annex["harmonic_band"] == "PARTIAL"  # From P3
        assert annex["evolution_status"] == "EVOLVING"  # From P4
        # Should deduplicate misaligned concepts
        assert set(annex["misaligned_concepts"]) == {"slice_b", "slice_c", "slice_d"}
        assert annex["misaligned_concepts"] == sorted(annex["misaligned_concepts"])

    def test_03_annex_limits_misaligned_concepts(
        self, sample_p3_summary: dict[str, Any], sample_p4_calibration: dict[str, Any]
    ) -> None:
        """Test that annex limits misaligned concepts to 10."""
        # Create P3/P4 with more than 10 concepts
        p3_many = {
            **sample_p3_summary,
            "misaligned_concepts": [f"slice_{i}" for i in range(15)],
        }
        p4_many = {
            **sample_p4_calibration,
            "misaligned_concepts": [f"slice_{i}" for i in range(10, 20)],
        }

        annex = build_curriculum_harmonic_annex(p3_many, p4_many)

        assert len(annex["misaligned_concepts"]) <= 10

    def test_04_annex_limits_priority_adjustments(
        self, sample_p3_summary: dict[str, Any], sample_p4_calibration: dict[str, Any]
    ) -> None:
        """Test that annex limits priority adjustments to 5."""
        annex = build_curriculum_harmonic_annex(sample_p3_summary, sample_p4_calibration)

        # P3 has 6 adjustments, should be limited to 5 (preserves order from P3)
        assert len(annex["priority_adjustments"]) <= 5
        assert annex["priority_adjustments"] == ["slice_b", "slice_c", "slice_a", "slice_d", "slice_e"]

    def test_05_annex_is_deterministic(
        self, sample_p3_summary: dict[str, Any], sample_p4_calibration: dict[str, Any]
    ) -> None:
        """Test that annex is deterministic."""
        annexes = [
            build_curriculum_harmonic_annex(sample_p3_summary, sample_p4_calibration)
            for _ in range(5)
        ]

        for i in range(1, len(annexes)):
            assert annexes[0] == annexes[i]

    def test_06_annex_serializes_to_json(
        self, sample_p3_summary: dict[str, Any], sample_p4_calibration: dict[str, Any]
    ) -> None:
        """Test that annex can be serialized to JSON."""
        annex = build_curriculum_harmonic_annex(sample_p3_summary, sample_p4_calibration)

        json_str = json.dumps(annex)
        assert json_str

        deserialized = json.loads(json_str)
        assert deserialized == annex


# =============================================================================
# TEST GROUP 2: EVIDENCE ATTACHMENT WITH ANNEX
# =============================================================================


class TestAttachHarmonicAlignmentToEvidenceWithAnnex:
    """Tests for evidence attachment with curriculum annex."""

    def test_07_evidence_includes_annex_when_p3_p4_provided(
        self,
        sample_evidence: dict[str, Any],
        sample_tile: dict[str, Any],
        sample_signal: dict[str, Any],
        sample_p3_summary: dict[str, Any],
        sample_p4_calibration: dict[str, Any],
    ) -> None:
        """Test that evidence includes curriculum annex when P3/P4 provided."""
        enriched = attach_harmonic_alignment_to_evidence(
            sample_evidence,
            sample_tile,
            sample_signal,
            p3_summary=sample_p3_summary,
            p4_calibration=sample_p4_calibration,
        )

        assert "governance" in enriched
        assert "harmonic_alignment" in enriched["governance"]
        assert "curriculum_annex" in enriched["governance"]["harmonic_alignment"]

        annex = enriched["governance"]["harmonic_alignment"]["curriculum_annex"]
        assert annex["harmonic_band"] == "PARTIAL"
        assert annex["evolution_status"] == "EVOLVING"

    def test_08_evidence_no_annex_when_p3_p4_missing(
        self,
        sample_evidence: dict[str, Any],
        sample_tile: dict[str, Any],
        sample_signal: dict[str, Any],
    ) -> None:
        """Test that evidence does not include annex when P3/P4 not provided."""
        enriched = attach_harmonic_alignment_to_evidence(
            sample_evidence, sample_tile, sample_signal
        )

        assert "governance" in enriched
        assert "harmonic_alignment" in enriched["governance"]
        assert "curriculum_annex" not in enriched["governance"]["harmonic_alignment"]

    def test_09_evidence_non_mutating_with_annex(
        self,
        sample_evidence: dict[str, Any],
        sample_tile: dict[str, Any],
        sample_signal: dict[str, Any],
        sample_p3_summary: dict[str, Any],
        sample_p4_calibration: dict[str, Any],
    ) -> None:
        """Test that evidence attachment does not mutate input."""
        original = dict(sample_evidence)

        attach_harmonic_alignment_to_evidence(
            sample_evidence,
            sample_tile,
            sample_signal,
            p3_summary=sample_p3_summary,
            p4_calibration=sample_p4_calibration,
        )

        assert sample_evidence == original

    def test_10_evidence_handles_nested_p3_p4(
        self,
        sample_evidence: dict[str, Any],
        sample_tile: dict[str, Any],
        sample_signal: dict[str, Any],
        sample_p3_summary: dict[str, Any],
        sample_p4_calibration: dict[str, Any],
    ) -> None:
        """Test that evidence handles nested P3/P4 structures."""
        # P3 with nested harmonic_alignment_summary
        p3_nested = {"harmonic_alignment_summary": sample_p3_summary}
        # P4 with nested harmonic_alignment
        p4_nested = {"harmonic_alignment": sample_p4_calibration}

        enriched = attach_harmonic_alignment_to_evidence(
            sample_evidence, sample_tile, sample_signal, p3_summary=p3_nested, p4_calibration=p4_nested
        )

        assert "curriculum_annex" in enriched["governance"]["harmonic_alignment"]


# =============================================================================
# TEST GROUP 3: COUNCIL SUMMARY WITH COUNTS
# =============================================================================


class TestSummarizeHarmonicForUpliftCouncilWithCounts:
    """Tests for council summary with counts."""

    def test_11_council_summary_includes_counts(self, sample_tile: dict[str, Any]) -> None:
        """Test that council summary includes num_misaligned_concepts and num_priority_adjustments."""
        summary = summarize_harmonic_for_uplift_council(sample_tile)

        assert "num_misaligned_concepts" in summary
        assert "num_priority_adjustments" in summary
        assert summary["num_misaligned_concepts"] == 1
        assert summary["num_priority_adjustments"] == 3

    def test_12_council_counts_match_lists(self, sample_tile: dict[str, Any]) -> None:
        """Test that counts match the length of the lists."""
        summary = summarize_harmonic_for_uplift_council(sample_tile)

        assert summary["num_misaligned_concepts"] == len(summary["misaligned_concepts"])
        assert summary["num_priority_adjustments"] == len(summary["priority_adjustments"])

    def test_13_council_summary_remains_neutral(self, sample_tile: dict[str, Any]) -> None:
        """Test that council summary uses neutral language."""
        summary = summarize_harmonic_for_uplift_council(sample_tile)

        # Check that summary doesn't contain judgmental words
        summary_str = json.dumps(summary).lower()
        forbidden_words = {
            "good", "bad", "better", "worse", "improve", "improvement",
            "should", "must", "need", "required", "fail", "success",
        }

        for word in forbidden_words:
            assert word not in summary_str, f"Forbidden word '{word}' found in summary"

    def test_14_council_summary_serializes_to_json(self, sample_tile: dict[str, Any]) -> None:
        """Test that council summary with counts can be serialized to JSON."""
        summary = summarize_harmonic_for_uplift_council(sample_tile)

        json_str = json.dumps(summary)
        assert json_str

        deserialized = json.loads(json_str)
        assert deserialized == summary
        assert "num_misaligned_concepts" in deserialized
        assert "num_priority_adjustments" in deserialized

