"""Tests for atlas governance P3/P4 integration and evidence attachment.

STATUS: PHASE X — ATLAS GOVERNANCE INTEGRATION

Tests verify:
- Atlas governance attachment to P3 stability reports
- Atlas governance attachment to P4 calibration reports
- Atlas governance attachment to evidence packs
- Uplift council summary generation
- Determinism and JSON safety
"""

import json
from typing import Any, Dict

import pytest

from backend.health.atlas_governance_adapter import (
    attach_atlas_governance_to_evidence,
    attach_atlas_governance_to_p3_stability_report,
    attach_atlas_governance_to_p4_calibration_report,
    build_first_light_structural_cohesion_annex,
    extract_atlas_signal_for_first_light,
    summarize_atlas_for_uplift_council,
)


class TestAttachAtlasGovernanceToP3StabilityReport:
    """Tests for attach_atlas_governance_to_p3_stability_report."""

    def test_attaches_atlas_summary(self, sample_atlas_governance_tile, sample_p3_stability_report):
        """Verify atlas governance summary is attached."""
        updated = attach_atlas_governance_to_p3_stability_report(
            sample_p3_stability_report, sample_atlas_governance_tile
        )
        
        assert "atlas_governance_summary" in updated
        summary = updated["atlas_governance_summary"]
        assert summary["lattice_coherence_band"] == "COHERENT"
        assert summary["global_lattice_norm"] == 0.85
        assert summary["transition_status"] == "OK"
        assert "slices_ready" in summary
        assert "slices_needing_alignment" in summary

    def test_non_mutating(self, sample_atlas_governance_tile, sample_p3_stability_report):
        """Verify function does not mutate input."""
        original = dict(sample_p3_stability_report)
        
        attach_atlas_governance_to_p3_stability_report(
            sample_p3_stability_report, sample_atlas_governance_tile
        )
        
        assert sample_p3_stability_report == original
        assert "atlas_governance_summary" not in sample_p3_stability_report

    def test_deterministic(self, sample_atlas_governance_tile, sample_p3_stability_report):
        """Verify function is deterministic."""
        result1 = attach_atlas_governance_to_p3_stability_report(
            sample_p3_stability_report, sample_atlas_governance_tile
        )
        result2 = attach_atlas_governance_to_p3_stability_report(
            sample_p3_stability_report, sample_atlas_governance_tile
        )
        
        assert json.dumps(result1, sort_keys=True) == json.dumps(result2, sort_keys=True)

    def test_json_safe(self, sample_atlas_governance_tile, sample_p3_stability_report):
        """Verify output is JSON serializable."""
        updated = attach_atlas_governance_to_p3_stability_report(
            sample_p3_stability_report, sample_atlas_governance_tile
        )
        
        # Should not raise
        json_str = json.dumps(updated)
        assert len(json_str) > 0
        
        # Should round-trip
        parsed = json.loads(json_str)
        assert "atlas_governance_summary" in parsed


class TestAttachAtlasGovernanceToP4CalibrationReport:
    """Tests for attach_atlas_governance_to_p4_calibration_report."""

    def test_attaches_atlas_calibration(self, sample_atlas_governance_tile, sample_p4_calibration_report):
        """Verify atlas governance calibration is attached."""
        updated = attach_atlas_governance_to_p4_calibration_report(
            sample_p4_calibration_report, sample_atlas_governance_tile
        )
        
        assert "atlas_governance_calibration" in updated
        calibration = updated["atlas_governance_calibration"]
        assert calibration["lattice_coherence_band"] == "COHERENT"
        assert calibration["global_lattice_norm"] == 0.85
        assert calibration["transition_status"] == "OK"
        assert "slices_ready" in calibration
        assert "slices_needing_alignment" in calibration
        assert "calibration_notes" in calibration

    def test_uses_atlas_signal_if_provided(self, sample_atlas_governance_tile, sample_p4_calibration_report):
        """Verify atlas signal values are preferred when provided."""
        atlas_signal = {
            "global_lattice_norm": 0.92,
            "lattice_convergence_band": "COHERENT",
            "transition_status": "OK",
        }
        
        updated = attach_atlas_governance_to_p4_calibration_report(
            sample_p4_calibration_report, sample_atlas_governance_tile, atlas_signal
        )
        
        calibration = updated["atlas_governance_calibration"]
        assert calibration["global_lattice_norm"] == 0.92  # From signal

    def test_non_mutating(self, sample_atlas_governance_tile, sample_p4_calibration_report):
        """Verify function does not mutate input."""
        original = dict(sample_p4_calibration_report)
        
        attach_atlas_governance_to_p4_calibration_report(
            sample_p4_calibration_report, sample_atlas_governance_tile
        )
        
        assert sample_p4_calibration_report == original
        assert "atlas_governance_calibration" not in sample_p4_calibration_report

    def test_deterministic(self, sample_atlas_governance_tile, sample_p4_calibration_report):
        """Verify function is deterministic."""
        result1 = attach_atlas_governance_to_p4_calibration_report(
            sample_p4_calibration_report, sample_atlas_governance_tile
        )
        result2 = attach_atlas_governance_to_p4_calibration_report(
            sample_p4_calibration_report, sample_atlas_governance_tile
        )
        
        assert json.dumps(result1, sort_keys=True) == json.dumps(result2, sort_keys=True)

    def test_json_safe(self, sample_atlas_governance_tile, sample_p4_calibration_report):
        """Verify output is JSON serializable."""
        updated = attach_atlas_governance_to_p4_calibration_report(
            sample_p4_calibration_report, sample_atlas_governance_tile
        )
        
        # Should not raise
        json_str = json.dumps(updated)
        assert len(json_str) > 0


class TestAttachAtlasGovernanceToEvidence:
    """Tests for attach_atlas_governance_to_evidence."""

    def test_attaches_to_governance_section(self, sample_atlas_governance_tile, sample_evidence):
        """Verify atlas governance is attached under governance.atlas."""
        updated = attach_atlas_governance_to_evidence(
            sample_evidence, sample_atlas_governance_tile
        )
        
        assert "governance" in updated
        assert "atlas" in updated["governance"]
        atlas_data = updated["governance"]["atlas"]
        assert atlas_data["lattice_coherence_band"] == "COHERENT"
        assert atlas_data["global_lattice_norm"] == 0.85
        assert atlas_data["transition_status"] == "OK"

    def test_creates_governance_section_if_missing(self, sample_atlas_governance_tile):
        """Verify governance section is created if missing."""
        evidence = {"timestamp": "2024-01-01", "data": {}}
        
        updated = attach_atlas_governance_to_evidence(evidence, sample_atlas_governance_tile)
        
        assert "governance" in updated
        assert "atlas" in updated["governance"]

    def test_uses_atlas_signal_if_provided(self, sample_atlas_governance_tile, sample_evidence):
        """Verify atlas signal values are preferred when provided."""
        atlas_signal = {
            "global_lattice_norm": 0.92,
            "lattice_convergence_band": "COHERENT",
            "transition_status": "OK",
        }
        
        updated = attach_atlas_governance_to_evidence(
            sample_evidence, sample_atlas_governance_tile, atlas_signal
        )
        
        atlas_data = updated["governance"]["atlas"]
        assert atlas_data["global_lattice_norm"] == 0.92  # From signal
        assert atlas_data["lattice_convergence_band"] == "COHERENT"  # From signal

    def test_non_mutating(self, sample_atlas_governance_tile, sample_evidence):
        """Verify function does not mutate input."""
        original = dict(sample_evidence)
        
        attach_atlas_governance_to_evidence(sample_evidence, sample_atlas_governance_tile)
        
        assert sample_evidence == original
        assert "governance" not in sample_evidence

    def test_deterministic(self, sample_atlas_governance_tile, sample_evidence):
        """Verify function is deterministic."""
        result1 = attach_atlas_governance_to_evidence(
            sample_evidence, sample_atlas_governance_tile
        )
        result2 = attach_atlas_governance_to_evidence(
            sample_evidence, sample_atlas_governance_tile
        )
        
        assert json.dumps(result1, sort_keys=True) == json.dumps(result2, sort_keys=True)

    def test_json_safe(self, sample_atlas_governance_tile, sample_evidence):
        """Verify output is JSON serializable."""
        updated = attach_atlas_governance_to_evidence(
            sample_evidence, sample_atlas_governance_tile
        )
        
        # Should not raise
        json_str = json.dumps(updated)
        assert len(json_str) > 0


class TestSummarizeAtlasForUpliftCouncil:
    """Tests for summarize_atlas_for_uplift_council."""

    def test_maps_misaligned_to_block(self):
        """Verify MISALIGNED maps to BLOCK."""
        tile = {
            "lattice_coherence_band": "MISALIGNED",
            "transition_status": "ATTENTION",
            "slices_needing_alignment": ["slice_a"],
        }
        
        summary = summarize_atlas_for_uplift_council(tile)
        
        assert summary["status"] == "BLOCK"
        assert summary["lattice_coherence_band"] == "MISALIGNED"

    def test_maps_block_transition_to_block(self):
        """Verify transition_status=BLOCK maps to BLOCK."""
        tile = {
            "lattice_coherence_band": "PARTIAL",
            "transition_status": "BLOCK",
            "slices_needing_alignment": ["slice_a"],
        }
        
        summary = summarize_atlas_for_uplift_council(tile)
        
        assert summary["status"] == "BLOCK"

    def test_maps_partial_to_warn(self):
        """Verify PARTIAL maps to WARN."""
        tile = {
            "lattice_coherence_band": "PARTIAL",
            "transition_status": "ATTENTION",
            "slices_needing_alignment": ["slice_a"],
        }
        
        summary = summarize_atlas_for_uplift_council(tile)
        
        assert summary["status"] == "WARN"

    def test_maps_attention_transition_to_warn(self):
        """Verify transition_status=ATTENTION maps to WARN."""
        tile = {
            "lattice_coherence_band": "COHERENT",
            "transition_status": "ATTENTION",
            "slices_needing_alignment": [],
        }
        
        summary = summarize_atlas_for_uplift_council(tile)
        
        assert summary["status"] == "WARN"

    def test_maps_coherent_ok_to_ok(self):
        """Verify COHERENT + OK maps to OK."""
        tile = {
            "lattice_coherence_band": "COHERENT",
            "transition_status": "OK",
            "slices_needing_alignment": [],
        }
        
        summary = summarize_atlas_for_uplift_council(tile)
        
        assert summary["status"] == "OK"

    def test_includes_slices_needing_alignment(self):
        """Verify slices_needing_alignment is included."""
        tile = {
            "lattice_coherence_band": "PARTIAL",
            "transition_status": "ATTENTION",
            "slices_needing_alignment": ["slice_a", "slice_b"],
        }
        
        summary = summarize_atlas_for_uplift_council(tile)
        
        assert summary["slices_needing_alignment"] == ["slice_a", "slice_b"]
        assert any("slice_a" in note for note in summary["advisory_notes"])

    def test_has_required_keys(self):
        """Verify summary has all required keys."""
        tile = {
            "lattice_coherence_band": "COHERENT",
            "transition_status": "OK",
            "slices_needing_alignment": [],
        }
        
        summary = summarize_atlas_for_uplift_council(tile)
        
        assert "status" in summary
        assert "lattice_coherence_band" in summary
        assert "transition_status" in summary
        assert "slices_needing_alignment" in summary
        assert "advisory_notes" in summary

    def test_deterministic(self):
        """Verify function is deterministic."""
        tile = {
            "lattice_coherence_band": "COHERENT",
            "transition_status": "OK",
            "slices_needing_alignment": [],
        }
        
        result1 = summarize_atlas_for_uplift_council(tile)
        result2 = summarize_atlas_for_uplift_council(tile)
        
        assert json.dumps(result1, sort_keys=True) == json.dumps(result2, sort_keys=True)

    def test_json_safe(self):
        """Verify output is JSON serializable."""
        tile = {
            "lattice_coherence_band": "COHERENT",
            "transition_status": "OK",
            "slices_needing_alignment": [],
        }
        
        summary = summarize_atlas_for_uplift_council(tile)
        
        # Should not raise
        json_str = json.dumps(summary)
        assert len(json_str) > 0


class TestBuildFirstLightStructuralCohesionAnnex:
    """Tests for build_first_light_structural_cohesion_annex."""

    def test_has_required_keys(self):
        """Verify annex has all required keys."""
        atlas_tile = {
            "lattice_coherence_band": "COHERENT",
            "transition_status": "OK",
        }
        
        annex = build_first_light_structural_cohesion_annex(atlas_tile)
        
        assert "schema_version" in annex
        assert "lattice_band" in annex
        assert "transition_status" in annex
        assert "lean_shadow_status" in annex
        assert "coherence_band" in annex

    def test_extracts_atlas_fields(self):
        """Verify atlas fields are extracted correctly."""
        atlas_tile = {
            "lattice_coherence_band": "COHERENT",
            "transition_status": "OK",
        }
        
        annex = build_first_light_structural_cohesion_annex(atlas_tile)
        
        assert annex["lattice_band"] == "COHERENT"
        assert annex["transition_status"] == "OK"

    def test_includes_lean_shadow_status_when_provided(self):
        """Verify lean shadow status is included when structure tile provided."""
        atlas_tile = {
            "lattice_coherence_band": "COHERENT",
            "transition_status": "OK",
        }
        structure_tile = {
            "status": "WARN",
        }
        
        annex = build_first_light_structural_cohesion_annex(
            atlas_tile, structure_tile=structure_tile
        )
        
        assert annex["lean_shadow_status"] == "WARN"

    def test_includes_coherence_band_when_provided(self):
        """Verify coherence band is included when coherence tile provided."""
        atlas_tile = {
            "lattice_coherence_band": "COHERENT",
            "transition_status": "OK",
        }
        coherence_tile = {
            "coherence_band": "PARTIAL",
        }
        
        annex = build_first_light_structural_cohesion_annex(
            atlas_tile, coherence_tile=coherence_tile
        )
        
        assert annex["coherence_band"] == "PARTIAL"

    def test_handles_missing_structure_tile(self):
        """Verify annex works when structure tile is None."""
        atlas_tile = {
            "lattice_coherence_band": "COHERENT",
            "transition_status": "OK",
        }
        
        annex = build_first_light_structural_cohesion_annex(atlas_tile)
        
        assert annex["lean_shadow_status"] is None
        assert annex["lattice_band"] == "COHERENT"

    def test_handles_missing_coherence_tile(self):
        """Verify annex works when coherence tile is None."""
        atlas_tile = {
            "lattice_coherence_band": "COHERENT",
            "transition_status": "OK",
        }
        
        annex = build_first_light_structural_cohesion_annex(atlas_tile)
        
        assert annex["coherence_band"] is None
        assert annex["lattice_band"] == "COHERENT"

    def test_handles_all_tiles_provided(self):
        """Verify annex works with all tiles provided."""
        atlas_tile = {
            "lattice_coherence_band": "COHERENT",
            "transition_status": "OK",
        }
        structure_tile = {
            "status": "OK",
        }
        coherence_tile = {
            "coherence_band": "COHERENT",
        }
        
        annex = build_first_light_structural_cohesion_annex(
            atlas_tile, structure_tile=structure_tile, coherence_tile=coherence_tile
        )
        
        assert annex["lattice_band"] == "COHERENT"
        assert annex["transition_status"] == "OK"
        assert annex["lean_shadow_status"] == "OK"
        assert annex["coherence_band"] == "COHERENT"

    def test_deterministic(self):
        """Verify annex is deterministic."""
        atlas_tile = {
            "lattice_coherence_band": "COHERENT",
            "transition_status": "OK",
        }
        structure_tile = {
            "status": "OK",
        }
        coherence_tile = {
            "coherence_band": "COHERENT",
        }
        
        result1 = build_first_light_structural_cohesion_annex(
            atlas_tile, structure_tile=structure_tile, coherence_tile=coherence_tile
        )
        result2 = build_first_light_structural_cohesion_annex(
            atlas_tile, structure_tile=structure_tile, coherence_tile=coherence_tile
        )
        
        assert json.dumps(result1, sort_keys=True) == json.dumps(result2, sort_keys=True)

    def test_json_safe(self):
        """Verify annex is JSON serializable."""
        atlas_tile = {
            "lattice_coherence_band": "COHERENT",
            "transition_status": "OK",
        }
        structure_tile = {
            "status": "OK",
        }
        coherence_tile = {
            "coherence_band": "COHERENT",
        }
        
        annex = build_first_light_structural_cohesion_annex(
            atlas_tile, structure_tile=structure_tile, coherence_tile=coherence_tile
        )
        
        # Should not raise
        json_str = json.dumps(annex)
        assert len(json_str) > 0
        
        # Should round-trip
        parsed = json.loads(json_str)
        assert parsed["lattice_band"] == "COHERENT"

    def test_non_mutating(self):
        """Verify function does not mutate inputs."""
        atlas_tile = {
            "lattice_coherence_band": "COHERENT",
            "transition_status": "OK",
        }
        structure_tile = {
            "status": "OK",
        }
        coherence_tile = {
            "coherence_band": "COHERENT",
        }
        
        atlas_copy = dict(atlas_tile)
        structure_copy = dict(structure_tile)
        coherence_copy = dict(coherence_tile)
        
        build_first_light_structural_cohesion_annex(
            atlas_tile, structure_tile=structure_tile, coherence_tile=coherence_tile
        )
        
        assert atlas_tile == atlas_copy
        assert structure_tile == structure_copy
        assert coherence_tile == coherence_copy


class TestAttachAtlasGovernanceToEvidenceWithAnnex:
    """Tests for attach_atlas_governance_to_evidence with structural cohesion annex."""

    def test_attaches_annex_when_structure_tile_provided(self, sample_atlas_governance_tile, sample_evidence):
        """Verify annex is attached when structure tile provided."""
        structure_tile = {
            "status": "OK",
        }
        
        updated = attach_atlas_governance_to_evidence(
            sample_evidence, sample_atlas_governance_tile, structure_tile=structure_tile
        )
        
        assert "first_light_structural_annex" in updated["governance"]["atlas"]
        annex = updated["governance"]["atlas"]["first_light_structural_annex"]
        assert annex["lean_shadow_status"] == "OK"

    def test_attaches_annex_when_coherence_tile_provided(self, sample_atlas_governance_tile, sample_evidence):
        """Verify annex is attached when coherence tile provided."""
        coherence_tile = {
            "coherence_band": "COHERENT",
        }
        
        updated = attach_atlas_governance_to_evidence(
            sample_evidence, sample_atlas_governance_tile, coherence_tile=coherence_tile
        )
        
        assert "first_light_structural_annex" in updated["governance"]["atlas"]
        annex = updated["governance"]["atlas"]["first_light_structural_annex"]
        assert annex["coherence_band"] == "COHERENT"

    def test_attaches_annex_when_both_provided(self, sample_atlas_governance_tile, sample_evidence):
        """Verify annex is attached when both structure and coherence tiles provided."""
        structure_tile = {
            "status": "WARN",
        }
        coherence_tile = {
            "coherence_band": "PARTIAL",
        }
        
        updated = attach_atlas_governance_to_evidence(
            sample_evidence,
            sample_atlas_governance_tile,
            structure_tile=structure_tile,
            coherence_tile=coherence_tile,
        )
        
        assert "first_light_structural_annex" in updated["governance"]["atlas"]
        annex = updated["governance"]["atlas"]["first_light_structural_annex"]
        assert annex["lean_shadow_status"] == "WARN"
        assert annex["coherence_band"] == "PARTIAL"

    def test_no_annex_when_neither_provided(self, sample_atlas_governance_tile, sample_evidence):
        """Verify annex is not attached when neither structure nor coherence tiles provided."""
        updated = attach_atlas_governance_to_evidence(
            sample_evidence, sample_atlas_governance_tile
        )
        
        assert "first_light_structural_annex" not in updated["governance"]["atlas"]

    def test_annex_json_safe(self, sample_atlas_governance_tile, sample_evidence):
        """Verify annex in evidence is JSON serializable."""
        structure_tile = {
            "status": "OK",
        }
        coherence_tile = {
            "coherence_band": "COHERENT",
        }
        
        updated = attach_atlas_governance_to_evidence(
            sample_evidence,
            sample_atlas_governance_tile,
            structure_tile=structure_tile,
            coherence_tile=coherence_tile,
        )
        
        # Should not raise
        json_str = json.dumps(updated)
        assert len(json_str) > 0
        
        # Should round-trip
        parsed = json.loads(json_str)
        assert "first_light_structural_annex" in parsed["governance"]["atlas"]


@pytest.fixture
def sample_atlas_governance_tile():
    """Fixture for sample atlas governance tile."""
    return {
        "schema_version": "1.0.0",
        "tile_type": "atlas_governance",
        "status_light": "GREEN",
        "lattice_coherence_band": "COHERENT",
        "global_lattice_norm": 0.85,
        "transition_status": "OK",
        "slices_ready": ["slice_a", "slice_b"],
        "slices_needing_alignment": [],
        "headline": "Atlas governance: 2 slices ready, 0 need alignment, lattice norm 0.850 (COHERENT), transition OK",
    }


@pytest.fixture
def sample_p3_stability_report():
    """Fixture for sample P3 stability report."""
    return {
        "schema_version": "1.0.0",
        "run_id": "test_run_001",
        "slice_name": "test_slice",
        "cycles_completed": 1000,
        "stability_metrics": {
            "mean_rsi": 0.95,
            "min_rsi": 0.90,
            "max_rsi": 1.0,
        },
    }


@pytest.fixture
def sample_p4_calibration_report():
    """Fixture for sample P4 calibration report."""
    return {
        "schema_version": "1.0.0",
        "run_id": "test_p4_001",
        "timing": {
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-01-01T01:00:00Z",
            "cycles_observed": 1000,
        },
        "divergence_statistics": {
            "total_divergences": 5,
        },
    }


@pytest.fixture
def sample_evidence():
    """Fixture for sample evidence pack."""
    return {
        "timestamp": "2024-01-01T00:00:00Z",
        "run_id": "test_run_001",
        "data": {"some": "data"},
    }

