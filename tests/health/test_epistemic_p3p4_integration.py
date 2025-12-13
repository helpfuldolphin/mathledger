"""Tests for epistemic alignment P3/P4 integration.

PHASE X — EPISTEMIC GOVERNANCE INTEGRATION

Tests for:
- P3 stability report integration
- P4 calibration report integration
- Evidence pack attachment
- Uplift council summarization
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.health.epistemic_p3p4_integration import (
    attach_epistemic_alignment_to_evidence,
    attach_epistemic_alignment_to_p3_stability_report,
    attach_epistemic_alignment_to_p4_calibration_report,
    attach_epistemic_panel_to_evidence,
    build_epistemic_consistency_panel,
    build_first_light_epistemic_annex,
    emit_cal_exp_epistemic_annex,
    persist_cal_exp_epistemic_annex,
    summarize_epistemic_behavior_consistency,
    summarize_epistemic_for_uplift_council,
)


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def sample_epistemic_tile() -> Dict[str, Any]:
    """Create sample epistemic alignment governance tile."""
    return {
        "schema_version": "1.0.0",
        "status_light": "YELLOW",
        "alignment_band": "MEDIUM",
        "forecast_band": "MEDIUM",
        "tensor_norm": 0.65,
        "misalignment_hotspots": ["slice_hard", "slice_medium"],
        "headline": "Epistemic alignment shows mixed signals across domains.",
        "flags": [
            "Alignment tensor norm below threshold: 0.65",
            "Misalignment hotspots detected: 2 slice(s)",
        ],
    }


@pytest.fixture
def sample_forecast() -> Dict[str, Any]:
    """Create sample misalignment forecast."""
    return {
        "schema_version": "1.0.0",
        "forecast_id": "test_forecast_123",
        "predicted_band": "MEDIUM",
        "confidence": 0.75,
        "time_to_drift_event": 10,
        "neutral_explanation": ["Current alignment tensor norm: 0.65"],
    }


@pytest.fixture
def sample_p3_stability_report() -> Dict[str, Any]:
    """Create sample P3 stability report."""
    return {
        "schema_version": "1.0.0",
        "run_id": "test_run_123",
        "timing": {
            "start_time": "2025-01-01T00:00:00Z",
            "end_time": "2025-01-01T01:00:00Z",
            "cycles_completed": 100,
        },
        "criteria": [],
        "summary": {},
    }


@pytest.fixture
def sample_p4_calibration_report() -> Dict[str, Any]:
    """Create sample P4 calibration report."""
    return {
        "schema_version": "1.0.0",
        "run_id": "test_run_123",
        "timing": {
            "start_time": "2025-01-01T00:00:00Z",
            "end_time": "2025-01-01T01:00:00Z",
            "cycles_observed": 100,
        },
        "divergence_statistics": {},
        "accuracy_metrics": {},
        "calibration_assessment": {},
    }


@pytest.fixture
def sample_evidence() -> Dict[str, Any]:
    """Create sample evidence dictionary."""
    return {
        "schema_version": "1.0.0",
        "experiment_id": "test_exp_123",
        "artifacts": [],
    }


@pytest.fixture
def sample_compact_signal() -> Dict[str, Any]:
    """Create sample compact epistemic signal."""
    return {
        "schema_version": "1.0.0",
        "tensor_norm": 0.65,
        "predicted_band": "MEDIUM",
        "misalignment_hotspots": ["slice_hard"],
        "confidence": 0.75,
    }


# ==============================================================================
# P3 STABILITY REPORT TESTS
# ==============================================================================


class TestP3StabilityReportIntegration:
    """Tests for P3 stability report integration."""

    def test_attaches_epistemic_alignment_summary(
        self,
        sample_p3_stability_report: Dict[str, Any],
        sample_epistemic_tile: Dict[str, Any],
    ) -> None:
        """Test that epistemic_alignment_summary is attached to P3 report."""
        updated = attach_epistemic_alignment_to_p3_stability_report(
            sample_p3_stability_report, sample_epistemic_tile
        )

        assert "epistemic_alignment_summary" in updated
        summary = updated["epistemic_alignment_summary"]
        assert summary["tensor_norm"] == 0.65
        assert summary["alignment_band"] == "MEDIUM"
        assert summary["forecast_band"] == "MEDIUM"
        assert summary["misalignment_hotspots"] == ["slice_hard", "slice_medium"]
        assert summary["status_light"] == "YELLOW"

    def test_is_non_mutating(
        self,
        sample_p3_stability_report: Dict[str, Any],
        sample_epistemic_tile: Dict[str, Any],
    ) -> None:
        """Test that function is non-mutating (returns new dict)."""
        original_id = id(sample_p3_stability_report)
        updated = attach_epistemic_alignment_to_p3_stability_report(
            sample_p3_stability_report, sample_epistemic_tile
        )

        assert id(updated) != original_id
        assert "epistemic_alignment_summary" not in sample_p3_stability_report

    def test_serializes_to_json(
        self,
        sample_p3_stability_report: Dict[str, Any],
        sample_epistemic_tile: Dict[str, Any],
    ) -> None:
        """Test that updated report serializes to JSON."""
        updated = attach_epistemic_alignment_to_p3_stability_report(
            sample_p3_stability_report, sample_epistemic_tile
        )

        # Should not raise
        json_str = json.dumps(updated)
        assert isinstance(json_str, str)

        # Should be able to parse back
        parsed = json.loads(json_str)
        assert "epistemic_alignment_summary" in parsed


# ==============================================================================
# P4 CALIBRATION REPORT TESTS
# ==============================================================================


class TestP4CalibrationReportIntegration:
    """Tests for P4 calibration report integration."""

    def test_attaches_epistemic_alignment(
        self,
        sample_p4_calibration_report: Dict[str, Any],
        sample_epistemic_tile: Dict[str, Any],
        sample_forecast: Dict[str, Any],
    ) -> None:
        """Test that epistemic_alignment is attached to P4 report."""
        updated = attach_epistemic_alignment_to_p4_calibration_report(
            sample_p4_calibration_report, sample_epistemic_tile, sample_forecast
        )

        assert "epistemic_alignment" in updated
        alignment = updated["epistemic_alignment"]
        assert alignment["tensor_norm"] == 0.65
        assert alignment["alignment_band"] == "MEDIUM"
        assert alignment["forecast_band"] == "MEDIUM"
        assert alignment["misalignment_hotspots"] == ["slice_hard", "slice_medium"]
        assert alignment["confidence"] == 0.75

    def test_works_without_forecast(
        self,
        sample_p4_calibration_report: Dict[str, Any],
        sample_epistemic_tile: Dict[str, Any],
    ) -> None:
        """Test that function works without forecast (uses default confidence)."""
        updated = attach_epistemic_alignment_to_p4_calibration_report(
            sample_p4_calibration_report, sample_epistemic_tile
        )

        assert "epistemic_alignment" in updated
        alignment = updated["epistemic_alignment"]
        assert alignment["confidence"] == 0.5  # Default

    def test_is_non_mutating(
        self,
        sample_p4_calibration_report: Dict[str, Any],
        sample_epistemic_tile: Dict[str, Any],
    ) -> None:
        """Test that function is non-mutating (returns new dict)."""
        original_id = id(sample_p4_calibration_report)
        updated = attach_epistemic_alignment_to_p4_calibration_report(
            sample_p4_calibration_report, sample_epistemic_tile
        )

        assert id(updated) != original_id
        assert "epistemic_alignment" not in sample_p4_calibration_report

    def test_serializes_to_json(
        self,
        sample_p4_calibration_report: Dict[str, Any],
        sample_epistemic_tile: Dict[str, Any],
    ) -> None:
        """Test that updated report serializes to JSON."""
        updated = attach_epistemic_alignment_to_p4_calibration_report(
            sample_p4_calibration_report, sample_epistemic_tile
        )

        # Should not raise
        json_str = json.dumps(updated)
        assert isinstance(json_str, str)

        # Should be able to parse back
        parsed = json.loads(json_str)
        assert "epistemic_alignment" in parsed


# ==============================================================================
# EVIDENCE PACK TESTS
# ==============================================================================


class TestEvidencePackIntegration:
    """Tests for evidence pack integration."""

    def test_attaches_under_governance(
        self,
        sample_evidence: Dict[str, Any],
        sample_epistemic_tile: Dict[str, Any],
        sample_compact_signal: Dict[str, Any],
    ) -> None:
        """Test that epistemic alignment is attached under governance."""
        updated = attach_epistemic_alignment_to_evidence(
            sample_evidence, sample_epistemic_tile, sample_compact_signal
        )

        assert "governance" in updated
        assert "epistemic_alignment" in updated["governance"]
        alignment = updated["governance"]["epistemic_alignment"]
        assert alignment["tensor_norm"] == 0.65
        assert alignment["alignment_band"] == "MEDIUM"
        assert alignment["forecast_band"] == "MEDIUM"
        assert alignment["hotspots"] == ["slice_hard"]

    def test_works_without_compact_signal(
        self,
        sample_evidence: Dict[str, Any],
        sample_epistemic_tile: Dict[str, Any],
    ) -> None:
        """Test that function works without compact signal."""
        updated = attach_epistemic_alignment_to_evidence(
            sample_evidence, sample_epistemic_tile
        )

        assert "governance" in updated
        assert "epistemic_alignment" in updated["governance"]
        alignment = updated["governance"]["epistemic_alignment"]
        assert alignment["tensor_norm"] == 0.65
        assert alignment["hotspots"] == ["slice_hard", "slice_medium"]

    def test_creates_governance_if_missing(
        self,
        sample_evidence: Dict[str, Any],
        sample_epistemic_tile: Dict[str, Any],
    ) -> None:
        """Test that governance structure is created if missing."""
        # Remove governance if present
        if "governance" in sample_evidence:
            del sample_evidence["governance"]

        updated = attach_epistemic_alignment_to_evidence(
            sample_evidence, sample_epistemic_tile
        )

        assert "governance" in updated
        assert "epistemic_alignment" in updated["governance"]

    def test_is_non_mutating(
        self,
        sample_evidence: Dict[str, Any],
        sample_epistemic_tile: Dict[str, Any],
    ) -> None:
        """Test that function is non-mutating (returns new dict)."""
        original_id = id(sample_evidence)
        updated = attach_epistemic_alignment_to_evidence(
            sample_evidence, sample_epistemic_tile
        )

        assert id(updated) != original_id
        if "governance" in sample_evidence:
            assert "epistemic_alignment" not in sample_evidence.get("governance", {})

    def test_serializes_to_json(
        self,
        sample_evidence: Dict[str, Any],
        sample_epistemic_tile: Dict[str, Any],
    ) -> None:
        """Test that updated evidence serializes to JSON."""
        updated = attach_epistemic_alignment_to_evidence(
            sample_evidence, sample_epistemic_tile
        )

        # Should not raise
        json_str = json.dumps(updated)
        assert isinstance(json_str, str)

        # Should be able to parse back
        parsed = json.loads(json_str)
        assert "governance" in parsed
        assert "epistemic_alignment" in parsed["governance"]

    def test_shape_validation(
        self,
        sample_evidence: Dict[str, Any],
        sample_epistemic_tile: Dict[str, Any],
    ) -> None:
        """Test that output has correct shape."""
        updated = attach_epistemic_alignment_to_evidence(
            sample_evidence, sample_epistemic_tile
        )

        alignment = updated["governance"]["epistemic_alignment"]
        required_keys = {"tensor_norm", "alignment_band", "forecast_band", "hotspots"}
        assert required_keys.issubset(set(alignment.keys()))

        # Type checks
        assert isinstance(alignment["tensor_norm"], (int, float))
        assert isinstance(alignment["alignment_band"], str)
        assert isinstance(alignment["forecast_band"], str)
        assert isinstance(alignment["hotspots"], list)


# ==============================================================================
# UPLIFT COUNCIL TESTS
# ==============================================================================


class TestUpliftCouncilIntegration:
    """Tests for uplift council summarization."""

    def test_blocks_on_low_alignment(
        self,
        sample_epistemic_tile: Dict[str, Any],
    ) -> None:
        """Test that BLOCK status is returned for LOW alignment_band."""
        low_tile = dict(sample_epistemic_tile)
        low_tile["alignment_band"] = "LOW"
        low_tile["forecast_band"] = "MEDIUM"

        summary = summarize_epistemic_for_uplift_council(low_tile)

        assert summary["status"] == "BLOCK"
        assert summary["alignment_band"] == "LOW"

    def test_blocks_on_high_forecast(
        self,
        sample_epistemic_tile: Dict[str, Any],
    ) -> None:
        """Test that BLOCK status is returned for HIGH forecast_band."""
        high_forecast_tile = dict(sample_epistemic_tile)
        high_forecast_tile["alignment_band"] = "MEDIUM"
        high_forecast_tile["forecast_band"] = "HIGH"

        summary = summarize_epistemic_for_uplift_council(high_forecast_tile)

        assert summary["status"] == "BLOCK"
        assert summary["forecast_band"] == "HIGH"

    def test_warns_on_medium_alignment(
        self,
        sample_epistemic_tile: Dict[str, Any],
    ) -> None:
        """Test that WARN status is returned for MEDIUM alignment_band."""
        summary = summarize_epistemic_for_uplift_council(sample_epistemic_tile)

        assert summary["status"] == "WARN"
        assert summary["alignment_band"] == "MEDIUM"

    def test_warns_on_medium_forecast(
        self,
        sample_epistemic_tile: Dict[str, Any],
    ) -> None:
        """Test that WARN status is returned for MEDIUM forecast_band."""
        medium_forecast_tile = dict(sample_epistemic_tile)
        medium_forecast_tile["alignment_band"] = "HIGH"
        medium_forecast_tile["forecast_band"] = "MEDIUM"

        summary = summarize_epistemic_for_uplift_council(medium_forecast_tile)

        assert summary["status"] == "WARN"
        assert summary["forecast_band"] == "MEDIUM"

    def test_ok_on_high_alignment_low_forecast(
        self,
        sample_epistemic_tile: Dict[str, Any],
    ) -> None:
        """Test that OK status is returned for HIGH alignment and LOW forecast."""
        ok_tile = dict(sample_epistemic_tile)
        ok_tile["alignment_band"] = "HIGH"
        ok_tile["forecast_band"] = "LOW"

        summary = summarize_epistemic_for_uplift_council(ok_tile)

        assert summary["status"] == "OK"
        assert summary["alignment_band"] == "HIGH"
        assert summary["forecast_band"] == "LOW"

    def test_priority_hotspots_limited(
        self,
        sample_epistemic_tile: Dict[str, Any],
    ) -> None:
        """Test that priority_hotspots is limited to top 5."""
        many_hotspots_tile = dict(sample_epistemic_tile)
        many_hotspots_tile["misalignment_hotspots"] = [
            "slice1",
            "slice2",
            "slice3",
            "slice4",
            "slice5",
            "slice6",
            "slice7",
        ]

        summary = summarize_epistemic_for_uplift_council(many_hotspots_tile)

        assert len(summary["priority_hotspots"]) == 5
        assert summary["priority_hotspots"] == [
            "slice1",
            "slice2",
            "slice3",
            "slice4",
            "slice5",
        ]

    def test_has_required_fields(
        self,
        sample_epistemic_tile: Dict[str, Any],
    ) -> None:
        """Test that summary has all required fields."""
        summary = summarize_epistemic_for_uplift_council(sample_epistemic_tile)

        required_keys = {
            "status",
            "alignment_band",
            "forecast_band",
            "priority_hotspots",
        }
        assert required_keys.issubset(set(summary.keys()))

        # Status must be one of the valid values
        assert summary["status"] in {"OK", "WARN", "BLOCK"}

    def test_is_deterministic(
        self,
        sample_epistemic_tile: Dict[str, Any],
    ) -> None:
        """Test that output is deterministic."""
        summary1 = summarize_epistemic_for_uplift_council(sample_epistemic_tile)
        summary2 = summarize_epistemic_for_uplift_council(sample_epistemic_tile)

        assert summary1 == summary2


# ==============================================================================
# FIRST-LIGHT ANNEX TESTS
# ==============================================================================


class TestFirstLightEpistemicAnnex:
    """Tests for First-Light epistemic annex."""

    def test_builds_annex_from_p3_p4(
        self,
        sample_p3_stability_report: Dict[str, Any],
        sample_p4_calibration_report: Dict[str, Any],
    ) -> None:
        """Test that annex builds from P3 and P4 reports."""
        # Add epistemic alignment data to reports
        p3_with_epistemic = attach_epistemic_alignment_to_p3_stability_report(
            sample_p3_stability_report,
            {
                "tensor_norm": 0.65,
                "alignment_band": "MEDIUM",
                "misalignment_hotspots": ["slice_hard"],
            },
        )
        p4_with_epistemic = attach_epistemic_alignment_to_p4_calibration_report(
            sample_p4_calibration_report,
            {
                "tensor_norm": 0.70,
                "alignment_band": "HIGH",
                "misalignment_hotspots": ["slice_medium"],
            },
        )

        annex = build_first_light_epistemic_annex(p3_with_epistemic, p4_with_epistemic)

        assert annex["schema_version"] == "1.0.0"
        assert annex["p3_tensor_norm"] == 0.65
        assert annex["p3_alignment_band"] == "MEDIUM"
        assert annex["p4_tensor_norm"] == 0.70
        assert annex["p4_alignment_band"] == "HIGH"
        assert "hotspot_union" in annex

    def test_hotspot_union_sorted_and_limited(
        self,
        sample_p3_stability_report: Dict[str, Any],
        sample_p4_calibration_report: Dict[str, Any],
    ) -> None:
        """Test that hotspot_union is sorted and limited to top 5."""
        # Add epistemic alignment data with many hotspots
        p3_with_epistemic = attach_epistemic_alignment_to_p3_stability_report(
            sample_p3_stability_report,
            {
                "tensor_norm": 0.65,
                "alignment_band": "MEDIUM",
                "misalignment_hotspots": ["slice_c", "slice_a", "slice_b"],
            },
        )
        p4_with_epistemic = attach_epistemic_alignment_to_p4_calibration_report(
            sample_p4_calibration_report,
            {
                "tensor_norm": 0.70,
                "alignment_band": "HIGH",
                "misalignment_hotspots": ["slice_d", "slice_e", "slice_f", "slice_g"],
            },
        )

        annex = build_first_light_epistemic_annex(p3_with_epistemic, p4_with_epistemic)

        # Should be sorted
        hotspots = annex["hotspot_union"]
        assert hotspots == sorted(hotspots)
        # Should be limited to top 5
        assert len(hotspots) <= 5

    def test_annex_is_json_safe(
        self,
        sample_p3_stability_report: Dict[str, Any],
        sample_p4_calibration_report: Dict[str, Any],
    ) -> None:
        """Test that annex is JSON-safe."""
        p3_with_epistemic = attach_epistemic_alignment_to_p3_stability_report(
            sample_p3_stability_report,
            {
                "tensor_norm": 0.65,
                "alignment_band": "MEDIUM",
                "misalignment_hotspots": ["slice_hard"],
            },
        )
        p4_with_epistemic = attach_epistemic_alignment_to_p4_calibration_report(
            sample_p4_calibration_report,
            {
                "tensor_norm": 0.70,
                "alignment_band": "HIGH",
                "misalignment_hotspots": ["slice_medium"],
            },
        )

        annex = build_first_light_epistemic_annex(p3_with_epistemic, p4_with_epistemic)

        # Should not raise
        json_str = json.dumps(annex)
        assert isinstance(json_str, str)

        # Should be able to parse back
        parsed = json.loads(json_str)
        assert parsed["schema_version"] == "1.0.0"

    def test_annex_is_deterministic(
        self,
        sample_p3_stability_report: Dict[str, Any],
        sample_p4_calibration_report: Dict[str, Any],
    ) -> None:
        """Test that annex is deterministic."""
        p3_with_epistemic = attach_epistemic_alignment_to_p3_stability_report(
            sample_p3_stability_report,
            {
                "tensor_norm": 0.65,
                "alignment_band": "MEDIUM",
                "misalignment_hotspots": ["slice_hard"],
            },
        )
        p4_with_epistemic = attach_epistemic_alignment_to_p4_calibration_report(
            sample_p4_calibration_report,
            {
                "tensor_norm": 0.70,
                "alignment_band": "HIGH",
                "misalignment_hotspots": ["slice_medium"],
            },
        )

        annex1 = build_first_light_epistemic_annex(p3_with_epistemic, p4_with_epistemic)
        annex2 = build_first_light_epistemic_annex(p3_with_epistemic, p4_with_epistemic)

        # Compare JSON serializations for exact match
        json1 = json.dumps(annex1, sort_keys=True)
        json2 = json.dumps(annex2, sort_keys=True)

        assert json1 == json2
        assert annex1 == annex2


# ==============================================================================
# EVIDENCE ATTACH WITH ANNEX TESTS
# ==============================================================================


class TestEvidenceAttachWithAnnex:
    """Tests for evidence attachment with First-Light annex."""

    def test_attaches_annex_when_p3_p4_provided(
        self,
        sample_evidence: Dict[str, Any],
        sample_epistemic_tile: Dict[str, Any],
        sample_p3_stability_report: Dict[str, Any],
        sample_p4_calibration_report: Dict[str, Any],
    ) -> None:
        """Test that annex is attached when both P3 and P4 reports are provided."""
        # Add epistemic alignment to P3 and P4
        p3_with_epistemic = attach_epistemic_alignment_to_p3_stability_report(
            sample_p3_stability_report, sample_epistemic_tile
        )
        p4_with_epistemic = attach_epistemic_alignment_to_p4_calibration_report(
            sample_p4_calibration_report, sample_epistemic_tile
        )

        updated = attach_epistemic_alignment_to_evidence(
            sample_evidence,
            sample_epistemic_tile,
            p3_report=p3_with_epistemic,
            p4_report=p4_with_epistemic,
        )

        assert "governance" in updated
        assert "epistemic_alignment" in updated["governance"]
        assert "first_light_annex" in updated["governance"]["epistemic_alignment"]

    def test_remains_non_mutating_with_annex(
        self,
        sample_evidence: Dict[str, Any],
        sample_epistemic_tile: Dict[str, Any],
        sample_p3_stability_report: Dict[str, Any],
        sample_p4_calibration_report: Dict[str, Any],
    ) -> None:
        """Test that function remains non-mutating when annex is included."""
        p3_with_epistemic = attach_epistemic_alignment_to_p3_stability_report(
            sample_p3_stability_report, sample_epistemic_tile
        )
        p4_with_epistemic = attach_epistemic_alignment_to_p4_calibration_report(
            sample_p4_calibration_report, sample_epistemic_tile
        )

        original_id = id(sample_evidence)
        updated = attach_epistemic_alignment_to_evidence(
            sample_evidence,
            sample_epistemic_tile,
            p3_report=p3_with_epistemic,
            p4_report=p4_with_epistemic,
        )

        assert id(updated) != original_id
        if "governance" in sample_evidence:
            assert "first_light_annex" not in sample_evidence.get("governance", {}).get(
                "epistemic_alignment", {}
            )


# ==============================================================================
# CONSISTENCY SUMMARY TESTS
# ==============================================================================


class TestEpistemicBehaviorConsistency:
    """
    Tests for epistemic behavior consistency summary.

    These scenarios model what an external safety auditor should look for when
    reconciling epistemic signals with evidence-quality trajectory. The consistency
    summary answers: "Did the epistemic picture and the observed behavior stay in sync?"

    Key scenarios:
    - INCONSISTENT: Epistemic alignment degrades while evidence quality improves/remains stable
      → Suggests epistemic model may be overly conservative or detecting issues not reflected
        in actual behavior
    - CONSISTENT: Epistemic alignment and evidence quality move together
      → Suggests epistemic model is accurately tracking system behavior
    - UNKNOWN: No evidence quality data available
      → Cannot assess consistency without trajectory data

    All tests use neutral, descriptive language in advisory notes (no evaluative terms).
    """

    def test_flags_inconsistency_low_epistemic_ok_evidence(
        self,
    ) -> None:
        """Test that inconsistency is flagged when epistemic LOW while evidence quality OK."""
        annex = {
            "schema_version": "1.0.0",
            "p3_tensor_norm": 0.7,
            "p3_alignment_band": "HIGH",
            "p4_tensor_norm": 0.3,
            "p4_alignment_band": "LOW",  # Degraded
            "hotspot_union": ["slice_hard"],
        }
        evidence_quality = {
            "trajectory_class": "IMPROVING",  # Evidence quality OK
        }

        summary = summarize_epistemic_behavior_consistency(annex, evidence_quality)

        assert summary["consistency_status"] == "INCONSISTENT"
        assert len(summary["advisory_notes"]) > 0
        # Check that note mentions the inconsistency
        notes_text = " ".join(summary["advisory_notes"]).lower()
        assert "degraded" in notes_text
        assert "improving" in notes_text or "stable" in notes_text

    def test_flags_inconsistency_medium_epistemic_stable_evidence(
        self,
    ) -> None:
        """Test that inconsistency is flagged when epistemic degrades to MEDIUM while evidence STABLE."""
        annex = {
            "schema_version": "1.0.0",
            "p3_tensor_norm": 0.8,
            "p3_alignment_band": "HIGH",
            "p4_tensor_norm": 0.5,
            "p4_alignment_band": "MEDIUM",  # Degraded
            "hotspot_union": [],
        }
        evidence_quality = {
            "trajectory_class": "STABLE",  # Evidence quality OK
        }

        summary = summarize_epistemic_behavior_consistency(annex, evidence_quality)

        assert summary["consistency_status"] == "INCONSISTENT"
        notes_text = " ".join(summary["advisory_notes"]).lower()
        assert "degraded" in notes_text
        assert "stable" in notes_text

    def test_consistent_when_both_degrade(
        self,
    ) -> None:
        """Test that status is CONSISTENT when both epistemic and evidence degrade."""
        annex = {
            "schema_version": "1.0.0",
            "p3_tensor_norm": 0.7,
            "p3_alignment_band": "HIGH",
            "p4_tensor_norm": 0.3,
            "p4_alignment_band": "LOW",  # Degraded
            "hotspot_union": [],
        }
        evidence_quality = {
            "trajectory_class": "DEGRADING",  # Evidence also degrading
        }

        summary = summarize_epistemic_behavior_consistency(annex, evidence_quality)

        assert summary["consistency_status"] == "CONSISTENT"
        notes_text = " ".join(summary["advisory_notes"]).lower()
        assert "degraded" in notes_text

    def test_consistent_when_both_improve(
        self,
    ) -> None:
        """Test that status is CONSISTENT when both improve."""
        annex = {
            "schema_version": "1.0.0",
            "p3_tensor_norm": 0.3,
            "p3_alignment_band": "LOW",
            "p4_tensor_norm": 0.7,
            "p4_alignment_band": "HIGH",  # Improved
            "hotspot_union": [],
        }
        evidence_quality = {
            "trajectory_class": "IMPROVING",
        }

        summary = summarize_epistemic_behavior_consistency(annex, evidence_quality)

        assert summary["consistency_status"] == "CONSISTENT"

    def test_unknown_when_no_evidence_quality(
        self,
    ) -> None:
        """Test that status is UNKNOWN when evidence quality is not provided."""
        annex = {
            "schema_version": "1.0.0",
            "p3_tensor_norm": 0.7,
            "p3_alignment_band": "HIGH",
            "p4_tensor_norm": 0.3,
            "p4_alignment_band": "LOW",
            "hotspot_union": [],
        }

        summary = summarize_epistemic_behavior_consistency(annex, None)

        assert summary["consistency_status"] == "UNKNOWN"
        notes_text = " ".join(summary["advisory_notes"]).lower()
        assert "not available" in notes_text

    def test_advisory_notes_neutral(
        self,
    ) -> None:
        """Test that advisory notes use neutral, descriptive language."""
        annex = {
            "schema_version": "1.0.0",
            "p3_tensor_norm": 0.7,
            "p3_alignment_band": "HIGH",
            "p4_tensor_norm": 0.3,
            "p4_alignment_band": "LOW",
            "hotspot_union": [],
        }
        evidence_quality = {
            "trajectory_class": "IMPROVING",
        }

        summary = summarize_epistemic_behavior_consistency(annex, evidence_quality)

        notes_text = " ".join(summary["advisory_notes"]).lower()
        # Check for neutral language (no evaluative terms)
        forbidden_terms = ["good", "bad", "wrong", "error", "mistake", "fix", "broken"]
        for term in forbidden_terms:
            assert term not in notes_text, f"Evaluative term '{term}' found in notes"

    def test_has_required_fields(
        self,
    ) -> None:
        """Test that summary has all required fields."""
        annex = {
            "schema_version": "1.0.0",
            "p3_tensor_norm": 0.7,
            "p3_alignment_band": "HIGH",
            "p4_tensor_norm": 0.5,
            "p4_alignment_band": "MEDIUM",
            "hotspot_union": [],
        }

        summary = summarize_epistemic_behavior_consistency(annex)

        required_keys = {"consistency_status", "advisory_notes"}
        assert required_keys.issubset(set(summary.keys()))
        assert summary["consistency_status"] in {
            "CONSISTENT",
            "INCONSISTENT",
            "UNKNOWN",
        }
        assert isinstance(summary["advisory_notes"], list)


# ==============================================================================
# CAL-EXP ANNEX EMISSION TESTS
# ==============================================================================


class TestCalExpEpistemicAnnex:
    """Tests for calibration experiment epistemic annex emission."""

    def test_emits_annex_with_cal_id(
        self,
    ) -> None:
        """Test that annex is emitted with cal_id and schema_version."""
        annex = {
            "schema_version": "1.0.0",
            "p3_tensor_norm": 0.65,
            "p3_alignment_band": "MEDIUM",
            "p4_tensor_norm": 0.70,
            "p4_alignment_band": "HIGH",
            "hotspot_union": ["slice_hard"],
        }

        cal_annex = emit_cal_exp_epistemic_annex("CAL-EXP-1", annex)

        assert cal_annex["schema_version"] == "1.0.0"
        assert cal_annex["cal_id"] == "CAL-EXP-1"
        assert cal_annex["p3_tensor_norm"] == 0.65
        assert cal_annex["p3_alignment_band"] == "MEDIUM"
        assert cal_annex["p4_tensor_norm"] == 0.70
        assert cal_annex["p4_alignment_band"] == "HIGH"
        assert cal_annex["hotspot_union"] == ["slice_hard"]

    def test_preserves_all_annex_fields(
        self,
    ) -> None:
        """Test that all original annex fields are preserved."""
        annex = {
            "schema_version": "1.0.0",
            "p3_tensor_norm": 0.65,
            "p3_alignment_band": "MEDIUM",
            "p4_tensor_norm": 0.70,
            "p4_alignment_band": "HIGH",
            "hotspot_union": ["slice_hard", "slice_medium"],
        }

        cal_annex = emit_cal_exp_epistemic_annex("CAL-EXP-2", annex)

        # All original fields should be present
        assert "p3_tensor_norm" in cal_annex
        assert "p3_alignment_band" in cal_annex
        assert "p4_tensor_norm" in cal_annex
        assert "p4_alignment_band" in cal_annex
        assert "hotspot_union" in cal_annex

    def test_is_deterministic(
        self,
    ) -> None:
        """Test that output is deterministic."""
        annex = {
            "schema_version": "1.0.0",
            "p3_tensor_norm": 0.65,
            "p3_alignment_band": "MEDIUM",
            "p4_tensor_norm": 0.70,
            "p4_alignment_band": "HIGH",
            "hotspot_union": ["slice_hard"],
        }

        cal_annex1 = emit_cal_exp_epistemic_annex("CAL-EXP-1", annex)
        cal_annex2 = emit_cal_exp_epistemic_annex("CAL-EXP-1", annex)

        assert cal_annex1 == cal_annex2

    def test_persists_to_file(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that annex is persisted to file correctly."""
        annex = {
            "schema_version": "1.0.0",
            "cal_id": "CAL-EXP-1",
            "p3_tensor_norm": 0.65,
            "p3_alignment_band": "MEDIUM",
            "p4_tensor_norm": 0.70,
            "p4_alignment_band": "HIGH",
            "hotspot_union": ["slice_hard"],
        }

        output_path = persist_cal_exp_epistemic_annex(annex, tmp_path)

        assert output_path.exists()
        assert output_path.name == "epistemic_annex_CAL-EXP-1.json"

        # Verify file contents
        with open(output_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)

        assert loaded["cal_id"] == "CAL-EXP-1"
        assert loaded["p3_tensor_norm"] == 0.65

    def test_json_round_trip(
        self,
    ) -> None:
        """Test that annex survives JSON round-trip."""
        annex = {
            "schema_version": "1.0.0",
            "p3_tensor_norm": 0.65,
            "p3_alignment_band": "MEDIUM",
            "p4_tensor_norm": 0.70,
            "p4_alignment_band": "HIGH",
            "hotspot_union": ["slice_hard"],
        }

        cal_annex = emit_cal_exp_epistemic_annex("CAL-EXP-1", annex)

        # Serialize and deserialize
        json_str = json.dumps(cal_annex)
        loaded = json.loads(json_str)

        assert loaded["cal_id"] == "CAL-EXP-1"
        assert loaded["p3_tensor_norm"] == 0.65
        assert loaded["p4_alignment_band"] == "HIGH"


# ==============================================================================
# EPISTEMIC CONSISTENCY PANEL TESTS
# ==============================================================================


class TestEpistemicConsistencyPanel:
    """Tests for epistemic consistency panel aggregation."""

    def test_aggregates_consistency_counts(
        self,
    ) -> None:
        """Test that panel correctly aggregates consistency counts."""
        annexes = [
            emit_cal_exp_epistemic_annex("CAL-EXP-1", {
                "schema_version": "1.0.0",
                "p3_tensor_norm": 0.7,
                "p3_alignment_band": "HIGH",
                "p4_tensor_norm": 0.3,
                "p4_alignment_band": "LOW",
                "hotspot_union": [],
            }),
            emit_cal_exp_epistemic_annex("CAL-EXP-2", {
                "schema_version": "1.0.0",
                "p3_tensor_norm": 0.5,
                "p3_alignment_band": "MEDIUM",
                "p4_tensor_norm": 0.6,
                "p4_alignment_band": "MEDIUM",
                "hotspot_union": [],
            }),
            emit_cal_exp_epistemic_annex("CAL-EXP-3", {
                "schema_version": "1.0.0",
                "p3_tensor_norm": 0.8,
                "p3_alignment_band": "HIGH",
                "p4_tensor_norm": 0.9,
                "p4_alignment_band": "HIGH",
                "hotspot_union": [],
            }),
        ]

        consistency_blocks = [
            summarize_epistemic_behavior_consistency(
                {
                    "schema_version": "1.0.0",
                    "p3_tensor_norm": 0.7,
                    "p3_alignment_band": "HIGH",
                    "p4_tensor_norm": 0.3,
                    "p4_alignment_band": "LOW",
                    "hotspot_union": [],
                },
                {"trajectory_class": "IMPROVING"},  # INCONSISTENT
            ),
            summarize_epistemic_behavior_consistency(
                {
                    "schema_version": "1.0.0",
                    "p3_tensor_norm": 0.5,
                    "p3_alignment_band": "MEDIUM",
                    "p4_tensor_norm": 0.6,
                    "p4_alignment_band": "MEDIUM",
                    "hotspot_union": [],
                },
                {"trajectory_class": "DEGRADING"},  # CONSISTENT
            ),
            summarize_epistemic_behavior_consistency(
                {
                    "schema_version": "1.0.0",
                    "p3_tensor_norm": 0.8,
                    "p3_alignment_band": "HIGH",
                    "p4_tensor_norm": 0.9,
                    "p4_alignment_band": "HIGH",
                    "hotspot_union": [],
                },
                None,  # UNKNOWN
            ),
        ]

        evidence_quality_list = [
            {"trajectory_class": "IMPROVING"},
            {"trajectory_class": "DEGRADING"},
            None,
        ]

        panel = build_epistemic_consistency_panel(annexes, consistency_blocks, evidence_quality_list)

        assert panel["num_experiments"] == 3
        assert panel["num_consistent"] == 1
        assert panel["num_inconsistent"] == 1
        assert panel["num_unknown"] == 1

    def test_lists_inconsistent_experiments(
        self,
    ) -> None:
        """Test that panel lists inconsistent experiments with reasons and reason codes."""
        annexes = [
            emit_cal_exp_epistemic_annex("CAL-EXP-1", {
                "schema_version": "1.0.0",
                "p3_tensor_norm": 0.7,
                "p3_alignment_band": "HIGH",
                "p4_tensor_norm": 0.3,
                "p4_alignment_band": "LOW",
                "hotspot_union": [],
            }),
        ]

        consistency_blocks = [
            summarize_epistemic_behavior_consistency(
                {
                    "schema_version": "1.0.0",
                    "p3_tensor_norm": 0.7,
                    "p3_alignment_band": "HIGH",
                    "p4_tensor_norm": 0.3,
                    "p4_alignment_band": "LOW",
                    "hotspot_union": [],
                },
                {"trajectory_class": "IMPROVING"},
            ),
        ]

        evidence_quality_list = [{"trajectory_class": "IMPROVING"}]

        panel = build_epistemic_consistency_panel(annexes, consistency_blocks, evidence_quality_list)

        assert len(panel["experiments_inconsistent"]) == 1
        assert panel["experiments_inconsistent"][0]["cal_id"] == "CAL-EXP-1"
        assert "reason" in panel["experiments_inconsistent"][0]
        assert "reason_code" in panel["experiments_inconsistent"][0]
        assert panel["experiments_inconsistent"][0]["reason_code"] == "EPI_DEGRADED_EVID_IMPROVING"

    def test_sorts_inconsistent_experiments(
        self,
    ) -> None:
        """Test that inconsistent experiments are sorted by cal_id."""
        annexes = [
            emit_cal_exp_epistemic_annex("CAL-EXP-3", {
                "schema_version": "1.0.0",
                "p3_tensor_norm": 0.7,
                "p3_alignment_band": "HIGH",
                "p4_tensor_norm": 0.3,
                "p4_alignment_band": "LOW",
                "hotspot_union": [],
            }),
            emit_cal_exp_epistemic_annex("CAL-EXP-1", {
                "schema_version": "1.0.0",
                "p3_tensor_norm": 0.7,
                "p3_alignment_band": "HIGH",
                "p4_tensor_norm": 0.3,
                "p4_alignment_band": "LOW",
                "hotspot_union": [],
            }),
        ]

        consistency_blocks = [
            summarize_epistemic_behavior_consistency(
                {
                    "schema_version": "1.0.0",
                    "p3_tensor_norm": 0.7,
                    "p3_alignment_band": "HIGH",
                    "p4_tensor_norm": 0.3,
                    "p4_alignment_band": "LOW",
                    "hotspot_union": [],
                },
                {"trajectory_class": "IMPROVING"},
            ),
            summarize_epistemic_behavior_consistency(
                {
                    "schema_version": "1.0.0",
                    "p3_tensor_norm": 0.7,
                    "p3_alignment_band": "HIGH",
                    "p4_tensor_norm": 0.3,
                    "p4_alignment_band": "LOW",
                    "hotspot_union": [],
                },
                {"trajectory_class": "IMPROVING"},
            ),
        ]

        evidence_quality_list = [
            {"trajectory_class": "IMPROVING"},
            {"trajectory_class": "IMPROVING"},
        ]

        panel = build_epistemic_consistency_panel(annexes, consistency_blocks, evidence_quality_list)

        # Should be sorted by cal_id
        cal_ids = [exp["cal_id"] for exp in panel["experiments_inconsistent"]]
        assert cal_ids == sorted(cal_ids)
        assert cal_ids[0] == "CAL-EXP-1"
        assert cal_ids[1] == "CAL-EXP-3"

    def test_is_non_mutating(
        self,
    ) -> None:
        """Test that function is non-mutating."""
        annexes = [
            emit_cal_exp_epistemic_annex("CAL-EXP-1", {
                "schema_version": "1.0.0",
                "p3_tensor_norm": 0.7,
                "p3_alignment_band": "HIGH",
                "p4_tensor_norm": 0.3,
                "p4_alignment_band": "LOW",
                "hotspot_union": [],
            }),
        ]

        consistency_blocks = [
            summarize_epistemic_behavior_consistency(
                {
                    "schema_version": "1.0.0",
                    "p3_tensor_norm": 0.7,
                    "p3_alignment_band": "HIGH",
                    "p4_tensor_norm": 0.3,
                    "p4_alignment_band": "LOW",
                    "hotspot_union": [],
                },
                {"trajectory_class": "IMPROVING"},
            ),
        ]

        evidence_quality_list = [{"trajectory_class": "IMPROVING"}]

        original_annex_id = id(annexes[0])
        panel = build_epistemic_consistency_panel(annexes, consistency_blocks, evidence_quality_list)

        # Original annex should not be modified
        assert id(annexes[0]) == original_annex_id
        assert panel["num_experiments"] == 1

    def test_serializes_to_json(
        self,
    ) -> None:
        """Test that panel serializes to JSON."""
        annexes = [
            emit_cal_exp_epistemic_annex("CAL-EXP-1", {
                "schema_version": "1.0.0",
                "p3_tensor_norm": 0.7,
                "p3_alignment_band": "HIGH",
                "p4_tensor_norm": 0.3,
                "p4_alignment_band": "LOW",
                "hotspot_union": [],
            }),
        ]

        consistency_blocks = [
            summarize_epistemic_behavior_consistency(
                {
                    "schema_version": "1.0.0",
                    "p3_tensor_norm": 0.7,
                    "p3_alignment_band": "HIGH",
                    "p4_tensor_norm": 0.3,
                    "p4_alignment_band": "LOW",
                    "hotspot_union": [],
                },
                {"trajectory_class": "IMPROVING"},
            ),
        ]

        evidence_quality_list = [{"trajectory_class": "IMPROVING"}]

        panel = build_epistemic_consistency_panel(annexes, consistency_blocks, evidence_quality_list)

        # Should not raise
        json_str = json.dumps(panel)
        assert isinstance(json_str, str)

        # Should be able to parse back
        parsed = json.loads(json_str)
        assert parsed["num_experiments"] == 1

    def test_reason_code_assignment_deterministic(
        self,
    ) -> None:
        """Test that reason codes are assigned deterministically."""
        # Test EPI_DEGRADED_EVID_IMPROVING
        annex1 = emit_cal_exp_epistemic_annex("CAL-EXP-1", {
            "schema_version": "1.0.0",
            "p3_tensor_norm": 0.7,
            "p3_alignment_band": "HIGH",
            "p4_tensor_norm": 0.3,
            "p4_alignment_band": "LOW",
            "hotspot_union": [],
        })
        consistency1 = summarize_epistemic_behavior_consistency(
            {
                "schema_version": "1.0.0",
                "p3_tensor_norm": 0.7,
                "p3_alignment_band": "HIGH",
                "p4_tensor_norm": 0.3,
                "p4_alignment_band": "LOW",
                "hotspot_union": [],
            },
            {"trajectory_class": "IMPROVING"},
        )
        panel1 = build_epistemic_consistency_panel(
            [annex1], [consistency1], [{"trajectory_class": "IMPROVING"}]
        )

        # Run again with same inputs
        panel2 = build_epistemic_consistency_panel(
            [annex1], [consistency1], [{"trajectory_class": "IMPROVING"}]
        )

        assert panel1["experiments_inconsistent"][0]["reason_code"] == panel2["experiments_inconsistent"][0]["reason_code"]
        assert panel1["experiments_inconsistent"][0]["reason_code"] == "EPI_DEGRADED_EVID_IMPROVING"

    def test_reason_code_variants(
        self,
    ) -> None:
        """Test that different reason codes are assigned correctly."""
        test_cases = [
            # (p3_band, p4_band, evidence_quality, expected_code)
            ("HIGH", "LOW", {"trajectory_class": "IMPROVING"}, "EPI_DEGRADED_EVID_IMPROVING"),
            ("HIGH", "MEDIUM", {"trajectory_class": "STABLE"}, "EPI_DEGRADED_EVID_STABLE"),
            ("LOW", "HIGH", {"trajectory_class": "DEGRADING"}, "EPI_IMPROVED_EVID_DEGRADING"),
            ("HIGH", "LOW", None, "EPI_DEGRADED_EVID_UNKNOWN"),
            ("HIGH", "MEDIUM", {"trajectory_class": "UNKNOWN"}, "EPI_UNKNOWN_EVID_PRESENT"),
        ]

        for p3_band, p4_band, evidence_quality, expected_code in test_cases:
            annex = emit_cal_exp_epistemic_annex("CAL-EXP-TEST", {
                "schema_version": "1.0.0",
                "p3_tensor_norm": 0.7,
                "p3_alignment_band": p3_band,
                "p4_tensor_norm": 0.3,
                "p4_alignment_band": p4_band,
                "hotspot_union": [],
            })
            consistency = summarize_epistemic_behavior_consistency(
                {
                    "schema_version": "1.0.0",
                    "p3_tensor_norm": 0.7,
                    "p3_alignment_band": p3_band,
                    "p4_tensor_norm": 0.3,
                    "p4_alignment_band": p4_band,
                    "hotspot_union": [],
                },
                evidence_quality,
            )

            # Only test if inconsistent
            if consistency.get("consistency_status") == "INCONSISTENT":
                panel = build_epistemic_consistency_panel(
                    [annex], [consistency], [evidence_quality]
                )
                if panel["experiments_inconsistent"]:
                    actual_code = panel["experiments_inconsistent"][0]["reason_code"]
                    assert actual_code == expected_code, (
                        f"Expected {expected_code} for p3={p3_band}, p4={p4_band}, "
                        f"evidence={evidence_quality}, got {actual_code}"
                    )


# ==============================================================================
# EVIDENCE PANEL ATTACHMENT TESTS
# ==============================================================================


class TestEpistemicPanelEvidenceAttachment:
    """Tests for attaching epistemic panel to evidence."""

    def test_attaches_panel_under_governance(
        self,
        sample_evidence: Dict[str, Any],
    ) -> None:
        """Test that panel is attached under governance."""
        panel = {
            "schema_version": "1.0.0",
            "num_experiments": 3,
            "num_consistent": 2,
            "num_inconsistent": 1,
            "num_unknown": 0,
            "experiments_inconsistent": [],
        }

        updated = attach_epistemic_panel_to_evidence(sample_evidence, panel)

        assert "governance" in updated
        assert "epistemic_panel" in updated["governance"]
        assert updated["governance"]["epistemic_panel"]["num_experiments"] == 3

    def test_is_non_mutating(
        self,
        sample_evidence: Dict[str, Any],
    ) -> None:
        """Test that function is non-mutating."""
        panel = {
            "schema_version": "1.0.0",
            "num_experiments": 1,
            "num_consistent": 0,
            "num_inconsistent": 1,
            "num_unknown": 0,
            "experiments_inconsistent": [],
        }

        original_id = id(sample_evidence)
        updated = attach_epistemic_panel_to_evidence(sample_evidence, panel)

        assert id(updated) != original_id
        if "governance" in sample_evidence:
            assert "epistemic_panel" not in sample_evidence.get("governance", {})

    def test_serializes_to_json(
        self,
        sample_evidence: Dict[str, Any],
    ) -> None:
        """Test that updated evidence serializes to JSON."""
        panel = {
            "schema_version": "1.0.0",
            "num_experiments": 1,
            "num_consistent": 0,
            "num_inconsistent": 1,
            "num_unknown": 0,
            "experiments_inconsistent": [],
        }

        updated = attach_epistemic_panel_to_evidence(sample_evidence, panel)

        # Should not raise
        json_str = json.dumps(updated)
        assert isinstance(json_str, str)

        # Should be able to parse back
        parsed = json.loads(json_str)
        assert "governance" in parsed
        assert "epistemic_panel" in parsed["governance"]

