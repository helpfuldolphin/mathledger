"""
Integration tests for Phase V: Taxonomy P3/P4 & Evidence Integration.

Tests cover:
- Global health binding (taxonomy tile attachment)
- Evidence pack attachment
- P3 taxonomy_summary integration
- P4 taxonomy_calibration integration
- Tile shape validation and determinism
"""

import json
import pytest
from typing import Dict, Any

from scripts.taxonomy_governance import (
    build_taxonomy_integrity_radar,
    build_global_console_tile,
    build_taxonomy_drift_timeline,
    attach_taxonomy_to_evidence,
    build_p3_taxonomy_summary,
    build_p4_taxonomy_calibration,
    build_first_light_curriculum_coherence_summary,
    build_first_light_curriculum_coherence_tile,
    build_curriculum_coherence_timeseries,
    summarize_coherence_vs_curriculum_governance,
    build_cal_exp_curriculum_coherence_snapshot,
    persist_curriculum_coherence_snapshot,
    build_curriculum_coherence_panel,
)
from backend.health.taxonomy_adapter import build_taxonomy_tile_for_global_health


class TestGlobalHealthBinding:
    """Test taxonomy tile attachment to global health."""

    def test_taxonomy_tile_attachment_with_radar(self):
        """Test taxonomy tile can be built from radar."""
        radar = {
            "integrity_status": "OK",
            "alignment_score": 1.0,
            "curriculum_impacted": [],
            "docs_impacted": [],
        }
        risk_analysis = {"risk_level": "LOW", "breaking_changes": []}
        tile = build_global_console_tile(radar, risk_analysis)
        
        result = build_taxonomy_tile_for_global_health(
            radar=radar,
            tile=tile,
        )
        
        assert result is not None
        assert result["status_light"] == "GREEN"
        assert result["alignment_score"] == 1.0
        assert result["critical_breaks_count"] == 0
        assert "headline" in result
        assert "docs_impacted" in result

    def test_taxonomy_tile_attachment_with_components(self):
        """Test taxonomy tile can be built from component data."""
        metrics_impact = {"affected_metric_kinds": [], "status": "OK"}
        docs_alignment = {"missing_doc_updates": [], "alignment_status": "ALIGNED"}
        curriculum_alignment = {"slices_with_outdated_types": [], "alignment_status": "ALIGNED"}
        risk_analysis = {"risk_level": "LOW", "breaking_changes": []}
        
        result = build_taxonomy_tile_for_global_health(
            metrics_impact=metrics_impact,
            docs_alignment=docs_alignment,
            curriculum_alignment=curriculum_alignment,
            risk_analysis=risk_analysis,
        )
        
        assert result is not None
        assert result["status_light"] == "GREEN"
        assert result["alignment_score"] == 1.0

    def test_taxonomy_tile_graceful_degradation(self):
        """Test taxonomy tile gracefully degrades with insufficient data."""
        result = build_taxonomy_tile_for_global_health()
        
        assert result is None

    def test_taxonomy_tile_shape_validation(self):
        """Test taxonomy tile has required shape."""
        radar = {
            "integrity_status": "WARN",
            "alignment_score": 0.75,
            "curriculum_impacted": [],
            "docs_impacted": ["doc1.md"],
        }
        risk_analysis = {"risk_level": "MEDIUM", "breaking_changes": []}
        tile = build_global_console_tile(radar, risk_analysis)
        
        result = build_taxonomy_tile_for_global_health(radar=radar, tile=tile)
        
        # Validate required fields
        assert "status_light" in result
        assert "alignment_score" in result
        assert "critical_breaks_count" in result
        assert "headline" in result
        assert "docs_impacted" in result
        
        # Validate types
        assert result["status_light"] in ["GREEN", "YELLOW", "RED"]
        assert isinstance(result["alignment_score"], float)
        assert 0.0 <= result["alignment_score"] <= 1.0
        assert isinstance(result["critical_breaks_count"], int)
        assert isinstance(result["headline"], str)
        assert isinstance(result["docs_impacted"], list)

    def test_taxonomy_tile_determinism(self):
        """Test taxonomy tile output is deterministic."""
        radar = {
            "integrity_status": "OK",
            "alignment_score": 1.0,
            "curriculum_impacted": [],
            "docs_impacted": [],
        }
        risk_analysis = {"risk_level": "LOW", "breaking_changes": []}
        tile = build_global_console_tile(radar, risk_analysis)
        
        result1 = build_taxonomy_tile_for_global_health(radar=radar, tile=tile)
        result2 = build_taxonomy_tile_for_global_health(radar=radar, tile=tile)
        
        assert result1 == result2


class TestEvidencePackIntegration:
    """Test taxonomy attachment to evidence packs."""

    def test_attach_taxonomy_to_evidence(self):
        """Test taxonomy data can be attached to evidence pack."""
        evidence = {
            "timestamp": "2024-01-01T00:00:00Z",
            "experiment_id": "test_001",
        }
        radar = {
            "integrity_status": "OK",
            "alignment_score": 1.0,
            "curriculum_impacted": [],
            "docs_impacted": [],
        }
        risk_analysis = {"risk_level": "LOW", "breaking_changes": []}
        tile = build_global_console_tile(radar, risk_analysis)
        
        result = attach_taxonomy_to_evidence(evidence, radar, tile)
        
        assert "governance" in result
        assert "taxonomy" in result["governance"]
        assert result["governance"]["taxonomy"]["radar"] == radar
        assert result["governance"]["taxonomy"]["tile"] == tile
        assert result["governance"]["taxonomy"]["schema_version"] == "1.0.0"

    def test_attach_taxonomy_with_drift_timeline(self):
        """Test taxonomy attachment includes drift timeline when provided."""
        evidence = {"timestamp": "2024-01-01T00:00:00Z"}
        radar = {
            "integrity_status": "OK",
            "alignment_score": 1.0,
            "curriculum_impacted": [],
            "docs_impacted": [],
        }
        risk_analysis = {"risk_level": "LOW", "breaking_changes": []}
        tile = build_global_console_tile(radar, risk_analysis)
        drift_timeline = build_taxonomy_drift_timeline([])
        
        result = attach_taxonomy_to_evidence(evidence, radar, tile, drift_timeline)
        
        assert "drift_timeline" in result["governance"]["taxonomy"]
        assert result["governance"]["taxonomy"]["drift_timeline"] == drift_timeline

    def test_attach_taxonomy_key_breakpoints(self):
        """Test taxonomy attachment includes key breakpoints when curriculum affected."""
        evidence = {"timestamp": "2024-01-01T00:00:00Z"}
        radar = {
            "integrity_status": "BLOCK",
            "alignment_score": 0.3,
            "curriculum_impacted": ["slice1", "slice2"],
            "docs_impacted": [],
        }
        risk_analysis = {"risk_level": "HIGH", "breaking_changes": ["removed_type"]}
        tile = build_global_console_tile(radar, risk_analysis)
        
        result = attach_taxonomy_to_evidence(evidence, radar, tile)
        
        assert "key_breakpoints" in result["governance"]["taxonomy"]
        assert result["governance"]["taxonomy"]["key_breakpoints"]["curriculum_slices_affected"] == ["slice1", "slice2"]
        assert result["governance"]["taxonomy"]["key_breakpoints"]["critical_breaks_count"] == 2

    def test_attach_taxonomy_no_mutation(self):
        """Test attach_taxonomy_to_evidence does not mutate input."""
        evidence = {"timestamp": "2024-01-01T00:00:00Z", "data": {"key": "value"}}
        radar = {
            "integrity_status": "OK",
            "alignment_score": 1.0,
            "curriculum_impacted": [],
            "docs_impacted": [],
        }
        risk_analysis = {"risk_level": "LOW", "breaking_changes": []}
        tile = build_global_console_tile(radar, risk_analysis)
        
        original_evidence = dict(evidence)
        result = attach_taxonomy_to_evidence(evidence, radar, tile)
        
        # Original should be unchanged
        assert evidence == original_evidence
        # Result should be different (has governance section)
        assert result != evidence
        assert "governance" in result

    def test_attach_taxonomy_json_serializable(self):
        """Test attached taxonomy data is JSON serializable."""
        evidence = {"timestamp": "2024-01-01T00:00:00Z"}
        radar = {
            "integrity_status": "OK",
            "alignment_score": 1.0,
            "curriculum_impacted": [],
            "docs_impacted": [],
        }
        risk_analysis = {"risk_level": "LOW", "breaking_changes": []}
        tile = build_global_console_tile(radar, risk_analysis)
        
        result = attach_taxonomy_to_evidence(evidence, radar, tile)
        
        # Should not raise
        json_str = json.dumps(result)
        result_roundtrip = json.loads(json_str)
        assert result_roundtrip == result


class TestP3Integration:
    """Test P3 taxonomy_summary integration."""

    def test_p3_taxonomy_summary_basic(self):
        """Test P3 taxonomy summary has required fields."""
        radar = {
            "integrity_status": "OK",
            "alignment_score": 1.0,
            "docs_impacted": [],
        }
        
        summary = build_p3_taxonomy_summary(radar)
        
        assert "alignment_score" in summary
        assert "docs_impacted" in summary
        assert "integrity_status" in summary
        assert summary["alignment_score"] == 1.0
        assert summary["docs_impacted"] == []
        assert summary["integrity_status"] == "OK"

    def test_p3_taxonomy_summary_with_impacts(self):
        """Test P3 taxonomy summary includes impacted docs."""
        radar = {
            "integrity_status": "WARN",
            "alignment_score": 0.75,
            "docs_impacted": ["doc1.md", "doc2.md"],
        }
        
        summary = build_p3_taxonomy_summary(radar)
        
        assert summary["alignment_score"] == 0.75
        assert len(summary["docs_impacted"]) == 2
        assert "doc1.md" in summary["docs_impacted"]

    def test_p3_taxonomy_summary_json_serializable(self):
        """Test P3 taxonomy summary is JSON serializable."""
        radar = {
            "integrity_status": "OK",
            "alignment_score": 1.0,
            "docs_impacted": [],
        }
        
        summary = build_p3_taxonomy_summary(radar)
        
        json_str = json.dumps(summary)
        summary_roundtrip = json.loads(json_str)
        assert summary_roundtrip == summary


class TestP4Integration:
    """Test P4 taxonomy_calibration integration."""

    def test_p4_taxonomy_calibration_basic(self):
        """Test P4 taxonomy calibration has required fields."""
        drift_timeline = {
            "drift_band": "STABLE",
            "change_intensity": 0.0,
            "first_break_index": None,
        }
        
        calibration = build_p4_taxonomy_calibration(drift_timeline)
        
        assert "drift_band" in calibration
        assert "projected_horizon" in calibration
        assert "critical_break_index" in calibration
        assert calibration["drift_band"] == "STABLE"
        assert calibration["projected_horizon"] == 0.0
        assert calibration["critical_break_index"] is None

    def test_p4_taxonomy_calibration_with_drift(self):
        """Test P4 taxonomy calibration with drift."""
        drift_timeline = {
            "drift_band": "MEDIUM_DRIFT",
            "change_intensity": 0.4,
            "first_break_index": 2,
        }
        
        calibration = build_p4_taxonomy_calibration(drift_timeline)
        
        assert calibration["drift_band"] == "MEDIUM_DRIFT"
        assert calibration["projected_horizon"] > 0.0
        assert calibration["projected_horizon"] <= 1.0
        assert calibration["critical_break_index"] == 2

    def test_p4_taxonomy_calibration_projected_horizon_capped(self):
        """Test P4 projected horizon is capped at 1.0."""
        drift_timeline = {
            "drift_band": "HIGH_DRIFT",
            "change_intensity": 0.9,
            "first_break_index": 0,
        }
        
        calibration = build_p4_taxonomy_calibration(drift_timeline)
        
        assert calibration["projected_horizon"] <= 1.0

    def test_p4_taxonomy_calibration_json_serializable(self):
        """Test P4 taxonomy calibration is JSON serializable."""
        drift_timeline = {
            "drift_band": "LOW_DRIFT",
            "change_intensity": 0.1,
            "first_break_index": None,
        }
        
        calibration = build_p4_taxonomy_calibration(drift_timeline)
        
        json_str = json.dumps(calibration)
        calibration_roundtrip = json.loads(json_str)
        assert calibration_roundtrip == calibration


class TestFirstLightCurriculumCoherence:
    """Test First Light curriculum coherence summary."""

    def test_curriculum_coherence_summary_basic(self):
        """Test curriculum coherence summary combines P3 + P4 data."""
        p3_summary = {
            "alignment_score": 1.0,
            "integrity_status": "OK",
            "docs_impacted": [],
        }
        p4_calibration = {
            "drift_band": "STABLE",
            "projected_horizon": 0.0,
            "critical_break_index": None,
        }
        
        summary = build_first_light_curriculum_coherence_summary(p3_summary, p4_calibration)
        
        assert summary["schema_version"] == "1.0.0"
        assert summary["alignment_score"] == 1.0
        assert summary["integrity_status"] == "OK"
        assert summary["drift_band"] == "STABLE"
        assert summary["projected_horizon"] == 0.0

    def test_curriculum_coherence_summary_with_drift(self):
        """Test curriculum coherence summary with drift."""
        p3_summary = {
            "alignment_score": 0.75,
            "integrity_status": "WARN",
            "docs_impacted": ["doc1.md"],
        }
        p4_calibration = {
            "drift_band": "MEDIUM_DRIFT",
            "projected_horizon": 0.6,
            "critical_break_index": 2,
        }
        
        summary = build_first_light_curriculum_coherence_summary(p3_summary, p4_calibration)
        
        assert summary["alignment_score"] == 0.75
        assert summary["integrity_status"] == "WARN"
        assert summary["drift_band"] == "MEDIUM_DRIFT"
        assert summary["projected_horizon"] == 0.6

    def test_curriculum_coherence_summary_json_serializable(self):
        """Test curriculum coherence summary is JSON serializable."""
        p3_summary = {
            "alignment_score": 1.0,
            "integrity_status": "OK",
            "docs_impacted": [],
        }
        p4_calibration = {
            "drift_band": "STABLE",
            "projected_horizon": 0.0,
            "critical_break_index": None,
        }
        
        summary = build_first_light_curriculum_coherence_summary(p3_summary, p4_calibration)
        
        json_str = json.dumps(summary)
        summary_roundtrip = json.loads(json_str)
        assert summary_roundtrip == summary

    def test_curriculum_coherence_summary_determinism(self):
        """Test curriculum coherence summary is deterministic."""
        p3_summary = {
            "alignment_score": 0.8,
            "integrity_status": "WARN",
            "docs_impacted": ["doc1.md"],
        }
        p4_calibration = {
            "drift_band": "LOW_DRIFT",
            "projected_horizon": 0.15,
            "critical_break_index": None,
        }
        
        summary1 = build_first_light_curriculum_coherence_summary(p3_summary, p4_calibration)
        summary2 = build_first_light_curriculum_coherence_summary(p3_summary, p4_calibration)
        
        assert summary1 == summary2

    def test_attach_taxonomy_with_coherence_summary(self):
        """Test taxonomy attachment includes curriculum coherence summary."""
        evidence = {"timestamp": "2024-01-01T00:00:00Z"}
        radar = {
            "integrity_status": "OK",
            "alignment_score": 1.0,
            "curriculum_impacted": [],
            "docs_impacted": [],
        }
        risk_analysis = {"risk_level": "LOW", "breaking_changes": []}
        tile = build_global_console_tile(radar, risk_analysis)
        
        p3_summary = build_p3_taxonomy_summary(radar)
        drift_timeline = build_taxonomy_drift_timeline([])
        p4_calibration = build_p4_taxonomy_calibration(drift_timeline)
        coherence_summary = build_first_light_curriculum_coherence_summary(p3_summary, p4_calibration)
        
        result = attach_taxonomy_to_evidence(
            evidence, radar, tile, curriculum_coherence_summary=coherence_summary
        )
        
        assert "curriculum_coherence_summary" in result["governance"]["taxonomy"]
        assert result["governance"]["taxonomy"]["curriculum_coherence_summary"] == coherence_summary

    def test_attach_taxonomy_with_coherence_summary_json_serializable(self):
        """Test evidence with coherence summary is JSON serializable."""
        evidence = {"timestamp": "2024-01-01T00:00:00Z", "experiment_id": "test_001"}
        radar = {
            "integrity_status": "OK",
            "alignment_score": 1.0,
            "curriculum_impacted": [],
            "docs_impacted": [],
        }
        risk_analysis = {"risk_level": "LOW", "breaking_changes": []}
        tile = build_global_console_tile(radar, risk_analysis)
        
        p3_summary = build_p3_taxonomy_summary(radar)
        drift_timeline = build_taxonomy_drift_timeline([])
        p4_calibration = build_p4_taxonomy_calibration(drift_timeline)
        coherence_summary = build_first_light_curriculum_coherence_summary(p3_summary, p4_calibration)
        
        result = attach_taxonomy_to_evidence(
            evidence, radar, tile, curriculum_coherence_summary=coherence_summary
        )
        
        json_str = json.dumps(result)
        result_roundtrip = json.loads(json_str)
        assert result_roundtrip == result


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    def test_full_flow_global_health_to_evidence(self):
        """Test full flow: global health → evidence pack."""
        # Build taxonomy data
        metrics_impact = {"affected_metric_kinds": [], "status": "OK"}
        docs_alignment = {"missing_doc_updates": [], "alignment_status": "ALIGNED"}
        curriculum_alignment = {"slices_with_outdated_types": [], "alignment_status": "ALIGNED"}
        risk_analysis = {"risk_level": "LOW", "breaking_changes": []}
        
        radar = build_taxonomy_integrity_radar(metrics_impact, docs_alignment, curriculum_alignment)
        tile = build_global_console_tile(radar, risk_analysis)
        
        # Attach to global health
        global_health_tile = build_taxonomy_tile_for_global_health(radar=radar, tile=tile)
        assert global_health_tile is not None
        
        # Attach to evidence
        evidence = {"timestamp": "2024-01-01T00:00:00Z"}
        evidence_with_taxonomy = attach_taxonomy_to_evidence(evidence, radar, tile)
        
        assert "governance" in evidence_with_taxonomy
        assert "taxonomy" in evidence_with_taxonomy["governance"]

    def test_full_flow_p3_p4_reports(self):
        """Test full flow: P3 and P4 report integration."""
        # Build taxonomy data
        metrics_impact = {"affected_metric_kinds": [], "status": "OK"}
        docs_alignment = {"missing_doc_updates": ["doc1.md"], "alignment_status": "PARTIAL"}
        curriculum_alignment = {"slices_with_outdated_types": [], "alignment_status": "ALIGNED"}
        
        radar = build_taxonomy_integrity_radar(metrics_impact, docs_alignment, curriculum_alignment)
        
        # P3 summary
        p3_summary = build_p3_taxonomy_summary(radar)
        assert p3_summary["alignment_score"] == radar["alignment_score"]
        assert p3_summary["docs_impacted"] == radar["docs_impacted"]
        
        # P4 calibration
        historical_impacts = [
            {"breaking_changes": [], "non_breaking_changes": ["added_1"]},
        ]
        drift_timeline = build_taxonomy_drift_timeline(historical_impacts)
        p4_calibration = build_p4_taxonomy_calibration(drift_timeline)
        
        assert p4_calibration["drift_band"] == drift_timeline["drift_band"]
        assert p4_calibration["critical_break_index"] == drift_timeline["first_break_index"]

    def test_full_flow_p3_p4_to_coherence_to_evidence(self):
        """Test full flow: P3 + P4 → coherence summary → evidence."""
        # Build P3 data
        metrics_impact = {"affected_metric_kinds": [], "status": "OK"}
        docs_alignment = {"missing_doc_updates": [], "alignment_status": "ALIGNED"}
        curriculum_alignment = {"slices_with_outdated_types": [], "alignment_status": "ALIGNED"}
        radar = build_taxonomy_integrity_radar(metrics_impact, docs_alignment, curriculum_alignment)
        p3_summary = build_p3_taxonomy_summary(radar)
        
        # Build P4 data
        historical_impacts = [
            {"breaking_changes": [], "non_breaking_changes": ["added_1"]},
        ]
        drift_timeline = build_taxonomy_drift_timeline(historical_impacts)
        p4_calibration = build_p4_taxonomy_calibration(drift_timeline)
        
        # Build coherence summary
        coherence_summary = build_first_light_curriculum_coherence_summary(p3_summary, p4_calibration)
        assert coherence_summary["alignment_score"] == p3_summary["alignment_score"]
        assert coherence_summary["drift_band"] == p4_calibration["drift_band"]
        
        # Attach to evidence
        risk_analysis = {"risk_level": "LOW", "breaking_changes": []}
        tile = build_global_console_tile(radar, risk_analysis)
        evidence = {"timestamp": "2024-01-01T00:00:00Z", "experiment_id": "first_light_001"}
        evidence_with_coherence = attach_taxonomy_to_evidence(
            evidence, radar, tile, curriculum_coherence_summary=coherence_summary
        )
        
        assert "curriculum_coherence_summary" in evidence_with_coherence["governance"]["taxonomy"]
        assert evidence_with_coherence["governance"]["taxonomy"]["curriculum_coherence_summary"]["alignment_score"] == 1.0
        # Drift band depends on change intensity calculation
        drift_band = evidence_with_coherence["governance"]["taxonomy"]["curriculum_coherence_summary"]["drift_band"]
        assert drift_band in ["STABLE", "LOW_DRIFT", "MEDIUM_DRIFT", "HIGH_DRIFT"]


# Scenario: First Light Curriculum Coherence Tile
#
# The tests in this class correspond to a realistic First Light scenario where
# P3 (synthetic) shows strong taxonomy alignment (alignment_score: 0.95,
# integrity_status: "OK"), but P4 (real-runner shadow) shows high drift
# (drift_band: "HIGH_DRIFT", projected_horizon: 0.75). This coherence tile
# is what an external reviewer will use to spot curriculum instability across
# phases—specifically, the synthetic-real gap where synthetic experiments
# show clean alignment but real-runner observations reveal taxonomy evolution.
# The tile provides a single-page summary combining P3 alignment signals with
# P4 drift signals, serving as the curriculum alignment witness for First Light
# evidence packs.
#
class TestFirstLightCurriculumCoherenceTile:
    """Test First Light curriculum coherence tile (Phase X)."""

    def test_curriculum_coherence_tile_basic(self):
        """Test curriculum coherence tile has required fields."""
        p3_summary = {
            "alignment_score": 1.0,
            "integrity_status": "OK",
            "docs_impacted": ["doc1.md", "doc2.md"],
        }
        p4_calibration = {
            "drift_band": "STABLE",
            "projected_horizon": 0.0,
        }
        
        tile = build_first_light_curriculum_coherence_tile(p3_summary, p4_calibration)
        
        assert tile["schema_version"] == "1.0.0"
        assert tile["alignment_score"] == 1.0
        assert tile["integrity_status"] == "OK"
        assert tile["drift_band"] == "STABLE"
        assert tile["projected_horizon"] == 0.0
        assert tile["docs_impacted"] == ["doc1.md", "doc2.md"]  # Should be sorted

    def test_curriculum_coherence_tile_docs_sorted(self):
        """Test that docs_impacted are sorted for determinism."""
        p3_summary = {
            "alignment_score": 0.8,
            "integrity_status": "WARN",
            "docs_impacted": ["doc2.md", "doc1.md", "doc3.md"],  # Unsorted
        }
        p4_calibration = {
            "drift_band": "LOW_DRIFT",
            "projected_horizon": 0.15,
        }
        
        tile = build_first_light_curriculum_coherence_tile(p3_summary, p4_calibration)
        
        assert tile["docs_impacted"] == ["doc1.md", "doc2.md", "doc3.md"]

    def test_curriculum_coherence_tile_determinism(self):
        """Test curriculum coherence tile is deterministic."""
        p3_summary = {
            "alignment_score": 0.75,
            "integrity_status": "WARN",
            "docs_impacted": ["doc1.md"],
        }
        p4_calibration = {
            "drift_band": "MEDIUM_DRIFT",
            "projected_horizon": 0.6,
        }
        
        tile1 = build_first_light_curriculum_coherence_tile(p3_summary, p4_calibration)
        tile2 = build_first_light_curriculum_coherence_tile(p3_summary, p4_calibration)
        
        assert json.dumps(tile1, sort_keys=True) == json.dumps(tile2, sort_keys=True)

    def test_curriculum_coherence_tile_json_serializable(self):
        """Test curriculum coherence tile is JSON serializable."""
        p3_summary = {
            "alignment_score": 1.0,
            "integrity_status": "OK",
            "docs_impacted": [],
        }
        p4_calibration = {
            "drift_band": "STABLE",
            "projected_horizon": 0.0,
        }
        
        tile = build_first_light_curriculum_coherence_tile(p3_summary, p4_calibration)
        
        json_str = json.dumps(tile, sort_keys=True)
        tile_roundtrip = json.loads(json_str)
        assert tile_roundtrip == tile

    def test_attach_taxonomy_auto_builds_coherence_tile(self):
        """Test attach_taxonomy_to_evidence automatically builds coherence tile from P3+P4."""
        evidence = {"timestamp": "2024-01-01T00:00:00Z"}
        radar = {
            "integrity_status": "OK",
            "alignment_score": 1.0,
            "curriculum_impacted": [],
            "docs_impacted": ["doc1.md"],
        }
        risk_analysis = {"risk_level": "LOW", "breaking_changes": []}
        tile = build_global_console_tile(radar, risk_analysis)
        
        p3_summary = build_p3_taxonomy_summary(radar)
        drift_timeline = build_taxonomy_drift_timeline([])
        p4_calibration = build_p4_taxonomy_calibration(drift_timeline)
        
        result = attach_taxonomy_to_evidence(
            evidence,
            radar,
            tile,
            p3_taxonomy_summary=p3_summary,
            p4_taxonomy_calibration=p4_calibration,
        )
        
        assert "first_light_curriculum_coherence" in result["governance"]["taxonomy"]
        coherence_tile = result["governance"]["taxonomy"]["first_light_curriculum_coherence"]
        assert coherence_tile["schema_version"] == "1.0.0"
        assert coherence_tile["alignment_score"] == 1.0
        assert coherence_tile["integrity_status"] == "OK"
        assert coherence_tile["drift_band"] == "STABLE"
        assert coherence_tile["projected_horizon"] == 0.0
        assert coherence_tile["docs_impacted"] == ["doc1.md"]

    def test_attach_taxonomy_coherence_tile_no_mutation(self):
        """Test attach_taxonomy_to_evidence does not mutate input when building coherence tile."""
        evidence = {"timestamp": "2024-01-01T00:00:00Z", "data": {"key": "value"}}
        radar = {
            "integrity_status": "WARN",
            "alignment_score": 0.75,
            "curriculum_impacted": [],
            "docs_impacted": ["doc2.md", "doc1.md"],  # Unsorted
        }
        risk_analysis = {"risk_level": "MEDIUM", "breaking_changes": []}
        tile = build_global_console_tile(radar, risk_analysis)
        
        p3_summary = build_p3_taxonomy_summary(radar)
        drift_timeline = build_taxonomy_drift_timeline([])
        p4_calibration = build_p4_taxonomy_calibration(drift_timeline)
        
        # Create copies to verify inputs are not modified
        evidence_copy = json.loads(json.dumps(evidence))
        p3_summary_copy = json.loads(json.dumps(p3_summary))
        p4_calibration_copy = json.loads(json.dumps(p4_calibration))
        
        result = attach_taxonomy_to_evidence(
            evidence,
            radar,
            tile,
            p3_taxonomy_summary=p3_summary,
            p4_taxonomy_calibration=p4_calibration,
        )
        
        # Verify inputs unchanged
        assert json.dumps(evidence, sort_keys=True) == json.dumps(evidence_copy, sort_keys=True)
        assert json.dumps(p3_summary, sort_keys=True) == json.dumps(p3_summary_copy, sort_keys=True)
        assert json.dumps(p4_calibration, sort_keys=True) == json.dumps(p4_calibration_copy, sort_keys=True)
        
        # Verify result is a new dict
        assert result is not evidence

    def test_attach_taxonomy_coherence_tile_json_round_trip(self):
        """Test evidence with auto-built coherence tile can be JSON serialized and deserialized."""
        evidence = {"timestamp": "2024-01-01T00:00:00Z", "experiment_id": "test_001"}
        radar = {
            "integrity_status": "OK",
            "alignment_score": 1.0,
            "curriculum_impacted": [],
            "docs_impacted": ["doc1.md"],
        }
        risk_analysis = {"risk_level": "LOW", "breaking_changes": []}
        tile = build_global_console_tile(radar, risk_analysis)
        
        p3_summary = build_p3_taxonomy_summary(radar)
        drift_timeline = build_taxonomy_drift_timeline([])
        p4_calibration = build_p4_taxonomy_calibration(drift_timeline)
        
        result = attach_taxonomy_to_evidence(
            evidence,
            radar,
            tile,
            p3_taxonomy_summary=p3_summary,
            p4_taxonomy_calibration=p4_calibration,
        )
        
        # Serialize
        json_str = json.dumps(result, sort_keys=True)
        assert len(json_str) > 0
        
        # Deserialize
        parsed = json.loads(json_str)
        assert "governance" in parsed
        assert "taxonomy" in parsed["governance"]
        assert "first_light_curriculum_coherence" in parsed["governance"]["taxonomy"]
        
        # Verify structure preserved
        coherence_tile = parsed["governance"]["taxonomy"]["first_light_curriculum_coherence"]
        assert coherence_tile["alignment_score"] == 1.0
        assert coherence_tile["docs_impacted"] == ["doc1.md"]

    def test_attach_taxonomy_coherence_tile_only_when_both_provided(self):
        """Test coherence tile is only built when both P3 and P4 summaries are provided."""
        evidence = {"timestamp": "2024-01-01T00:00:00Z"}
        radar = {
            "integrity_status": "OK",
            "alignment_score": 1.0,
            "curriculum_impacted": [],
            "docs_impacted": [],
        }
        risk_analysis = {"risk_level": "LOW", "breaking_changes": []}
        tile = build_global_console_tile(radar, risk_analysis)
        
        p3_summary = build_p3_taxonomy_summary(radar)
        drift_timeline = build_taxonomy_drift_timeline([])
        p4_calibration = build_p4_taxonomy_calibration(drift_timeline)
        
        # Only P3 provided
        result_p3_only = attach_taxonomy_to_evidence(
            evidence,
            radar,
            tile,
            p3_taxonomy_summary=p3_summary,
        )
        assert "first_light_curriculum_coherence" not in result_p3_only["governance"]["taxonomy"]
        
        # Only P4 provided
        result_p4_only = attach_taxonomy_to_evidence(
            evidence,
            radar,
            tile,
            p4_taxonomy_calibration=p4_calibration,
        )
        assert "first_light_curriculum_coherence" not in result_p4_only["governance"]["taxonomy"]
        
        # Both provided
        result_both = attach_taxonomy_to_evidence(
            evidence,
            radar,
            tile,
            p3_taxonomy_summary=p3_summary,
            p4_taxonomy_calibration=p4_calibration,
        )
        assert "first_light_curriculum_coherence" in result_both["governance"]["taxonomy"]


class TestCalibrationCurriculumCoherencePanel:
    """Test P5 calibration curriculum coherence panel (Phase X)."""

    def test_cal_exp_snapshot_basic(self):
        """Test calibration experiment snapshot has required fields."""
        tile = {
            "schema_version": "1.0.0",
            "alignment_score": 0.95,
            "integrity_status": "OK",
            "drift_band": "MEDIUM_DRIFT",
            "projected_horizon": 0.45,
            "docs_impacted": ["doc1.md"],
        }
        
        snapshot = build_cal_exp_curriculum_coherence_snapshot("CAL-EXP-1", tile)
        
        assert snapshot["schema_version"] == "1.0.0"
        assert snapshot["cal_id"] == "CAL-EXP-1"
        assert snapshot["alignment_score"] == 0.95
        assert snapshot["integrity_status"] == "OK"
        assert snapshot["drift_band"] == "MEDIUM_DRIFT"
        assert snapshot["projected_horizon"] == 0.45

    def test_cal_exp_snapshot_determinism(self):
        """Test calibration experiment snapshot is deterministic."""
        tile = {
            "alignment_score": 0.753,
            "integrity_status": "WARN",
            "drift_band": "HIGH_DRIFT",
            "projected_horizon": 0.789,
        }
        
        snapshot1 = build_cal_exp_curriculum_coherence_snapshot("CAL-EXP-2", tile)
        snapshot2 = build_cal_exp_curriculum_coherence_snapshot("CAL-EXP-2", tile)
        
        assert json.dumps(snapshot1, sort_keys=True) == json.dumps(snapshot2, sort_keys=True)
        # Verify rounding
        assert snapshot1["alignment_score"] == 0.753
        assert snapshot1["projected_horizon"] == 0.789

    def test_cal_exp_snapshot_json_serializable(self):
        """Test calibration experiment snapshot is JSON serializable."""
        tile = {
            "alignment_score": 1.0,
            "integrity_status": "OK",
            "drift_band": "STABLE",
            "projected_horizon": 0.0,
        }
        
        snapshot = build_cal_exp_curriculum_coherence_snapshot("CAL-EXP-3", tile)
        
        json_str = json.dumps(snapshot, sort_keys=True)
        snapshot_roundtrip = json.loads(json_str)
        assert snapshot_roundtrip == snapshot

    def test_persist_snapshot(self, tmp_path):
        """Test snapshot persistence to disk."""
        snapshot = {
            "schema_version": "1.0.0",
            "cal_id": "CAL-EXP-1",
            "alignment_score": 0.95,
            "integrity_status": "OK",
            "drift_band": "MEDIUM_DRIFT",
            "projected_horizon": 0.45,
        }
        
        output_dir = tmp_path / "calibration"
        snapshot_path = persist_curriculum_coherence_snapshot(snapshot, output_dir)
        
        assert snapshot_path.exists()
        assert snapshot_path.name == "curriculum_coherence_CAL-EXP-1.json"
        
        # Verify contents
        with open(snapshot_path, "r") as f:
            loaded = json.load(f)
        assert loaded == snapshot

    def test_coherence_panel_basic(self):
        """Test curriculum coherence panel aggregates snapshots correctly."""
        snapshots = [
            {
                "cal_id": "CAL-EXP-1",
                "alignment_score": 1.0,
                "integrity_status": "OK",
                "drift_band": "STABLE",
            },
            {
                "cal_id": "CAL-EXP-2",
                "alignment_score": 0.8,
                "integrity_status": "WARN",
                "drift_band": "MEDIUM_DRIFT",
            },
            {
                "cal_id": "CAL-EXP-3",
                "alignment_score": 0.6,
                "integrity_status": "BLOCK",
                "drift_band": "HIGH_DRIFT",
            },
        ]
        
        panel = build_curriculum_coherence_panel(snapshots)
        
        assert panel["schema_version"] == "1.0.0"
        assert panel["num_experiments"] == 3
        assert panel["num_ok"] == 1
        assert panel["num_warn"] == 1
        assert panel["num_block"] == 1
        assert panel["num_high_drift"] == 1
        assert panel["median_alignment_score"] == 0.8  # Median of [0.6, 0.8, 1.0]

    def test_coherence_panel_empty(self):
        """Test curriculum coherence panel with empty snapshots."""
        panel = build_curriculum_coherence_panel([])
        
        assert panel["num_experiments"] == 0
        assert panel["num_ok"] == 0
        assert panel["num_warn"] == 0
        assert panel["num_block"] == 0
        assert panel["num_high_drift"] == 0
        assert panel["median_alignment_score"] == 0.0

    def test_coherence_panel_median_even_count(self):
        """Test panel median calculation with even number of snapshots."""
        snapshots = [
            {"alignment_score": 0.5, "integrity_status": "OK", "drift_band": "STABLE"},
            {"alignment_score": 0.7, "integrity_status": "OK", "drift_band": "STABLE"},
            {"alignment_score": 0.8, "integrity_status": "WARN", "drift_band": "LOW_DRIFT"},
            {"alignment_score": 1.0, "integrity_status": "OK", "drift_band": "STABLE"},
        ]
        
        panel = build_curriculum_coherence_panel(snapshots)
        
        # Median of [0.5, 0.7, 0.8, 1.0] = (0.7 + 0.8) / 2 = 0.75
        assert panel["median_alignment_score"] == 0.75

    def test_coherence_panel_json_serializable(self):
        """Test curriculum coherence panel is JSON serializable."""
        snapshots = [
            {
                "cal_id": "CAL-EXP-1",
                "alignment_score": 0.95,
                "integrity_status": "OK",
                "drift_band": "STABLE",
            },
        ]
        
        panel = build_curriculum_coherence_panel(snapshots)
        
        json_str = json.dumps(panel, sort_keys=True)
        panel_roundtrip = json.loads(json_str)
        assert panel_roundtrip == panel

    def test_coherence_panel_no_mutation(self):
        """Test panel building does not mutate input snapshots."""
        snapshots = [
            {
                "cal_id": "CAL-EXP-1",
                "alignment_score": 0.95,
                "integrity_status": "OK",
                "drift_band": "STABLE",
            },
            {
                "cal_id": "CAL-EXP-2",
                "alignment_score": 0.8,
                "integrity_status": "WARN",
                "drift_band": "MEDIUM_DRIFT",
            },
        ]
        
        snapshots_copy = json.loads(json.dumps(snapshots))
        panel = build_curriculum_coherence_panel(snapshots)
        
        # Verify snapshots unchanged
        assert json.dumps(snapshots, sort_keys=True) == json.dumps(snapshots_copy, sort_keys=True)
        assert panel["num_experiments"] == 2

    def test_attach_panel_to_evidence(self):
        """Test curriculum coherence panel can be attached to evidence."""
        evidence = {"timestamp": "2024-01-01T00:00:00Z"}
        radar = {
            "integrity_status": "OK",
            "alignment_score": 1.0,
            "curriculum_impacted": [],
            "docs_impacted": [],
        }
        risk_analysis = {"risk_level": "LOW", "breaking_changes": []}
        tile = build_global_console_tile(radar, risk_analysis)
        
        panel = {
            "schema_version": "1.0.0",
            "num_experiments": 3,
            "num_ok": 2,
            "num_warn": 1,
            "num_block": 0,
            "num_high_drift": 1,
            "median_alignment_score": 0.9,
        }
        
        result = attach_taxonomy_to_evidence(
            evidence,
            radar,
            tile,
            curriculum_coherence_panel=panel,
        )
        
        assert "governance" in result
        assert "curriculum_coherence_panel" in result["governance"]
        assert result["governance"]["curriculum_coherence_panel"] == panel
        assert result["governance"]["curriculum_coherence_panel"]["num_experiments"] == 3


class TestCurriculumCoherenceTimeseries:
    """Test curriculum coherence time-series extraction (Phase X)."""

    def test_timeseries_empty_list(self):
        """Test timeseries with empty summaries list."""
        ts = build_curriculum_coherence_timeseries([])
        
        assert ts["schema_version"] == "1.0.0"
        assert ts["points"] == []

    def test_timeseries_basic(self):
        """Test timeseries extraction from summaries."""
        summaries = [
            {
                "alignment_score": 1.0,
                "drift_band": "STABLE",
            },
            {
                "alignment_score": 0.95,
                "drift_band": "LOW_DRIFT",
            },
            {
                "alignment_score": 0.9,
                "drift_band": "MEDIUM_DRIFT",
            },
        ]
        
        ts = build_curriculum_coherence_timeseries(summaries)
        
        assert ts["schema_version"] == "1.0.0"
        assert len(ts["points"]) == 3
        assert ts["points"][0]["cycle_or_run_idx"] == 0
        assert ts["points"][0]["alignment_score"] == 1.0
        assert ts["points"][0]["drift_band"] == "STABLE"
        assert ts["points"][1]["cycle_or_run_idx"] == 1
        assert ts["points"][2]["cycle_or_run_idx"] == 2

    def test_timeseries_with_explicit_indices(self):
        """Test timeseries with explicit cycle_or_run_idx values."""
        summaries = [
            {
                "cycle_or_run_idx": 10,
                "alignment_score": 1.0,
                "drift_band": "STABLE",
            },
            {
                "cycle_or_run_idx": 20,
                "alignment_score": 0.95,
                "drift_band": "LOW_DRIFT",
            },
            {
                "cycle_or_run_idx": 30,
                "alignment_score": 0.9,
                "drift_band": "MEDIUM_DRIFT",
            },
        ]
        
        ts = build_curriculum_coherence_timeseries(summaries)
        
        assert ts["points"][0]["cycle_or_run_idx"] == 10
        assert ts["points"][1]["cycle_or_run_idx"] == 20
        assert ts["points"][2]["cycle_or_run_idx"] == 30

    def test_timeseries_monotone_indexing(self):
        """Test that timeseries enforces monotone increasing indices."""
        summaries = [
            {
                "cycle_or_run_idx": 10,
                "alignment_score": 1.0,
                "drift_band": "STABLE",
            },
            {
                "cycle_or_run_idx": 5,  # Out of order - should be corrected
                "alignment_score": 0.95,
                "drift_band": "LOW_DRIFT",
            },
            {
                "cycle_or_run_idx": 30,
                "alignment_score": 0.9,
                "drift_band": "MEDIUM_DRIFT",
            },
        ]
        
        ts = build_curriculum_coherence_timeseries(summaries)
        
        # Second point should be corrected to 11 (next_idx after 10)
        assert ts["points"][0]["cycle_or_run_idx"] == 10
        assert ts["points"][1]["cycle_or_run_idx"] == 11  # Corrected
        assert ts["points"][2]["cycle_or_run_idx"] == 30
        
        # Verify monotone property
        for i in range(1, len(ts["points"])):
            assert ts["points"][i]["cycle_or_run_idx"] >= ts["points"][i-1]["cycle_or_run_idx"]

    def test_timeseries_json_serializable(self):
        """Test that timeseries is JSON serializable."""
        summaries = [
            {
                "alignment_score": 1.0,
                "drift_band": "STABLE",
            },
            {
                "alignment_score": 0.95,
                "drift_band": "LOW_DRIFT",
            },
        ]
        
        ts = build_curriculum_coherence_timeseries(summaries)
        
        json_str = json.dumps(ts, sort_keys=True)
        ts_roundtrip = json.loads(json_str)
        assert ts_roundtrip == ts

    def test_timeseries_determinism(self):
        """Test that timeseries output is deterministic."""
        summaries = [
            {
                "alignment_score": 0.9,
                "drift_band": "MEDIUM_DRIFT",
            },
            {
                "alignment_score": 0.85,
                "drift_band": "HIGH_DRIFT",
            },
        ]
        
        ts1 = build_curriculum_coherence_timeseries(summaries)
        ts2 = build_curriculum_coherence_timeseries(summaries)
        
        assert json.dumps(ts1, sort_keys=True) == json.dumps(ts2, sort_keys=True)

    def test_timeseries_mixed_indices(self):
        """Test timeseries with mix of explicit and implicit indices."""
        summaries = [
            {
                "cycle_or_run_idx": 100,
                "alignment_score": 1.0,
                "drift_band": "STABLE",
            },
            {
                # No explicit index - should be 101
                "alignment_score": 0.95,
                "drift_band": "LOW_DRIFT",
            },
            {
                "cycle_or_run_idx": 200,
                "alignment_score": 0.9,
                "drift_band": "MEDIUM_DRIFT",
            },
        ]
        
        ts = build_curriculum_coherence_timeseries(summaries)
        
        assert ts["points"][0]["cycle_or_run_idx"] == 100
        assert ts["points"][1]["cycle_or_run_idx"] == 101  # Assigned sequentially
        assert ts["points"][2]["cycle_or_run_idx"] == 200


class TestCoherenceVsGovernanceCrossCheck:
    """Test coherence vs curriculum governance cross-check (Phase X)."""

    # CONSISTENT status tests
    
    def test_crosscheck_consistent_stable_high_alignment(self):
        """Test CONSISTENT status: STABLE drift with high alignment."""
        coherence_ts = {
            "schema_version": "1.0.0",
            "points": [
                {"cycle_or_run_idx": 0, "alignment_score": 1.0, "drift_band": "STABLE"},
                {"cycle_or_run_idx": 100, "alignment_score": 0.95, "drift_band": "STABLE"},
            ],
        }
        curriculum_signals = []
        
        result = summarize_coherence_vs_curriculum_governance(coherence_ts, curriculum_signals)
        
        assert result["status"] == "CONSISTENT"
        assert result["episodes"] == []
        assert len(result["advisory_notes"]) > 0
        assert "consistency criteria" in result["advisory_notes"][0].lower()

    def test_crosscheck_consistent_medium_drift_high_alignment(self):
        """Test CONSISTENT status: MEDIUM_DRIFT with high alignment."""
        coherence_ts = {
            "schema_version": "1.0.0",
            "points": [
                {"cycle_or_run_idx": 0, "alignment_score": 0.85, "drift_band": "MEDIUM_DRIFT"},
                {"cycle_or_run_idx": 100, "alignment_score": 0.9, "drift_band": "LOW_DRIFT"},
            ],
        }
        curriculum_signals = []
        
        result = summarize_coherence_vs_curriculum_governance(coherence_ts, curriculum_signals)
        
        assert result["status"] == "CONSISTENT"
        assert result["episodes"] == []

    def test_crosscheck_consistent_mixed_but_all_consistent(self):
        """Test CONSISTENT status: Mixed drift bands but all meet consistency criteria."""
        coherence_ts = {
            "schema_version": "1.0.0",
            "points": [
                {"cycle_or_run_idx": 0, "alignment_score": 0.9, "drift_band": "STABLE"},
                {"cycle_or_run_idx": 50, "alignment_score": 0.85, "drift_band": "LOW_DRIFT"},
                {"cycle_or_run_idx": 100, "alignment_score": 0.8, "drift_band": "MEDIUM_DRIFT"},
            ],
        }
        curriculum_signals = []
        
        result = summarize_coherence_vs_curriculum_governance(coherence_ts, curriculum_signals)
        
        assert result["status"] == "CONSISTENT"
        assert result["episodes"] == []

    # TENSION status tests
    
    def test_crosscheck_tension_high_drift_high_alignment(self):
        """Test TENSION status: HIGH_DRIFT with high alignment (XOR condition)."""
        coherence_ts = {
            "schema_version": "1.0.0",
            "points": [
                {"cycle_or_run_idx": 0, "alignment_score": 0.9, "drift_band": "HIGH_DRIFT"},
                {"cycle_or_run_idx": 100, "alignment_score": 0.95, "drift_band": "HIGH_DRIFT"},
            ],
        }
        curriculum_signals = []
        
        result = summarize_coherence_vs_curriculum_governance(coherence_ts, curriculum_signals)
        
        assert result["status"] == "TENSION"
        assert len(result["episodes"]) > 0
        assert len(result["advisory_notes"]) > 0
        assert any("high drift" in note.lower() for note in result["advisory_notes"])

    def test_crosscheck_tension_low_alignment_stable_drift(self):
        """Test TENSION status: Low alignment with STABLE drift (XOR condition)."""
        coherence_ts = {
            "schema_version": "1.0.0",
            "points": [
                {"cycle_or_run_idx": 0, "alignment_score": 0.5, "drift_band": "STABLE"},
                {"cycle_or_run_idx": 100, "alignment_score": 0.4, "drift_band": "LOW_DRIFT"},
            ],
        }
        curriculum_signals = []
        
        result = summarize_coherence_vs_curriculum_governance(coherence_ts, curriculum_signals)
        
        assert result["status"] == "TENSION"
        assert len(result["episodes"]) > 0
        assert len(result["advisory_notes"]) > 0
        assert any("alignment score below" in note.lower() for note in result["advisory_notes"])

    def test_crosscheck_tension_mixed_conditions(self):
        """Test TENSION status: Mixed conditions creating tension."""
        coherence_ts = {
            "schema_version": "1.0.0",
            "points": [
                {"cycle_or_run_idx": 0, "alignment_score": 0.9, "drift_band": "HIGH_DRIFT"},  # TENSION
                {"cycle_or_run_idx": 50, "alignment_score": 0.5, "drift_band": "STABLE"},  # TENSION
                {"cycle_or_run_idx": 100, "alignment_score": 0.85, "drift_band": "MEDIUM_DRIFT"},  # CONSISTENT
            ],
        }
        curriculum_signals = []
        
        result = summarize_coherence_vs_curriculum_governance(coherence_ts, curriculum_signals)
        
        assert result["status"] == "TENSION"
        assert len(result["episodes"]) == 1  # Single contiguous tension episode (0-50)
        assert result["episodes"][0]["start_idx"] == 0
        assert result["episodes"][0]["end_idx"] == 50

    # CONFLICT status tests
    
    def test_crosscheck_conflict_high_drift_low_alignment(self):
        """Test CONFLICT status: HIGH_DRIFT with low alignment."""
        coherence_ts = {
            "schema_version": "1.0.0",
            "points": [
                {"cycle_or_run_idx": 0, "alignment_score": 0.5, "drift_band": "HIGH_DRIFT"},
                {"cycle_or_run_idx": 100, "alignment_score": 0.4, "drift_band": "HIGH_DRIFT"},
            ],
        }
        curriculum_signals = []
        
        result = summarize_coherence_vs_curriculum_governance(coherence_ts, curriculum_signals)
        
        assert result["status"] == "CONFLICT"
        assert len(result["episodes"]) > 0
        assert len(result["advisory_notes"]) > 0
        assert any("high drift" in note.lower() and "below 0.6" in note.lower() for note in result["advisory_notes"])

    def test_crosscheck_conflict_worst_case(self):
        """Test CONFLICT status: Worst case scenario."""
        coherence_ts = {
            "schema_version": "1.0.0",
            "points": [
                {"cycle_or_run_idx": 0, "alignment_score": 0.3, "drift_band": "HIGH_DRIFT"},
                {"cycle_or_run_idx": 50, "alignment_score": 0.2, "drift_band": "HIGH_DRIFT"},
                {"cycle_or_run_idx": 100, "alignment_score": 0.1, "drift_band": "HIGH_DRIFT"},
            ],
        }
        curriculum_signals = []
        
        result = summarize_coherence_vs_curriculum_governance(coherence_ts, curriculum_signals)
        
        assert result["status"] == "CONFLICT"
        assert len(result["episodes"]) == 1  # Single conflict episode
        assert result["episodes"][0]["start_idx"] == 0
        assert result["episodes"][0]["end_idx"] == 100

    def test_crosscheck_conflict_overrides_tension(self):
        """Test CONFLICT status overrides TENSION when both conditions present."""
        coherence_ts = {
            "schema_version": "1.0.0",
            "points": [
                {"cycle_or_run_idx": 0, "alignment_score": 0.9, "drift_band": "HIGH_DRIFT"},  # TENSION
                {"cycle_or_run_idx": 50, "alignment_score": 0.5, "drift_band": "HIGH_DRIFT"},  # CONFLICT
                {"cycle_or_run_idx": 100, "alignment_score": 0.4, "drift_band": "HIGH_DRIFT"},  # CONFLICT
            ],
        }
        curriculum_signals = []
        
        result = summarize_coherence_vs_curriculum_governance(coherence_ts, curriculum_signals)
        
        assert result["status"] == "CONFLICT"  # Worst status wins
        assert len(result["episodes"]) >= 1

    # Additional tests
    
    def test_crosscheck_empty_points(self):
        """Test cross-check with empty points returns CONSISTENT with note."""
        coherence_ts = {
            "schema_version": "1.0.0",
            "points": [],
        }
        curriculum_signals = []
        
        result = summarize_coherence_vs_curriculum_governance(coherence_ts, curriculum_signals)
        
        assert result["status"] == "CONSISTENT"
        assert len(result["advisory_notes"]) > 0
        assert "no time-series points" in result["advisory_notes"][0].lower()

    def test_crosscheck_episodes_contiguous(self):
        """Test that episodes correctly identify contiguous non-CONSISTENT windows."""
        coherence_ts = {
            "schema_version": "1.0.0",
            "points": [
                {"cycle_or_run_idx": 0, "alignment_score": 0.9, "drift_band": "STABLE"},  # CONSISTENT
                {"cycle_or_run_idx": 50, "alignment_score": 0.5, "drift_band": "STABLE"},  # TENSION
                {"cycle_or_run_idx": 100, "alignment_score": 0.4, "drift_band": "STABLE"},  # TENSION
                {"cycle_or_run_idx": 150, "alignment_score": 0.9, "drift_band": "STABLE"},  # CONSISTENT
                {"cycle_or_run_idx": 200, "alignment_score": 0.3, "drift_band": "HIGH_DRIFT"},  # CONFLICT
            ],
        }
        curriculum_signals = []
        
        result = summarize_coherence_vs_curriculum_governance(coherence_ts, curriculum_signals)
        
        assert result["status"] == "CONFLICT"
        assert len(result["episodes"]) == 2  # Two separate episodes
        assert result["episodes"][0]["start_idx"] == 50
        assert result["episodes"][0]["end_idx"] == 100
        assert result["episodes"][1]["start_idx"] == 200
        assert result["episodes"][1]["end_idx"] == 200

    def test_crosscheck_determinism(self):
        """Test that cross-check output is deterministic."""
        coherence_ts = {
            "schema_version": "1.0.0",
            "points": [
                {"cycle_or_run_idx": 0, "alignment_score": 0.5, "drift_band": "HIGH_DRIFT"},
                {"cycle_or_run_idx": 100, "alignment_score": 0.4, "drift_band": "HIGH_DRIFT"},
            ],
        }
        curriculum_signals = []
        
        result1 = summarize_coherence_vs_curriculum_governance(coherence_ts, curriculum_signals)
        result2 = summarize_coherence_vs_curriculum_governance(coherence_ts, curriculum_signals)
        
        assert json.dumps(result1, sort_keys=True) == json.dumps(result2, sort_keys=True)

    def test_crosscheck_json_serializable(self):
        """Test cross-check result is JSON serializable."""
        coherence_ts = {
            "schema_version": "1.0.0",
            "points": [
                {"cycle_or_run_idx": 0, "alignment_score": 0.5, "drift_band": "HIGH_DRIFT"},
            ],
        }
        curriculum_signals = []
        
        result = summarize_coherence_vs_curriculum_governance(coherence_ts, curriculum_signals)
        
        json_str = json.dumps(result, sort_keys=True)
        result_roundtrip = json.loads(json_str)
        assert result_roundtrip == result

    def test_crosscheck_advisory_notes_limit(self):
        """Test that advisory notes are limited to 3."""
        coherence_ts = {
            "schema_version": "1.0.0",
            "points": [
                {"cycle_or_run_idx": 0, "alignment_score": 0.5, "drift_band": "HIGH_DRIFT"},
            ],
        }
        curriculum_signals = []
        
        result = summarize_coherence_vs_curriculum_governance(coherence_ts, curriculum_signals)
        
        assert len(result["advisory_notes"]) <= 3

    def test_crosscheck_evidence_attachment(self):
        """Test that cross-check can be attached to evidence pack."""
        coherence_ts = {
            "schema_version": "1.0.0",
            "points": [
                {"cycle_or_run_idx": 0, "alignment_score": 0.5, "drift_band": "HIGH_DRIFT"},
            ],
        }
        curriculum_signals = []
        
        crosscheck = summarize_coherence_vs_curriculum_governance(coherence_ts, curriculum_signals)
        
        evidence = {"timestamp": "2024-01-01T00:00:00Z"}
        radar = {"integrity_status": "OK"}
        tile = {"status_light": "GREEN"}
        
        result = attach_taxonomy_to_evidence(
            evidence,
            radar,
            tile,
            coherence_crosscheck=crosscheck,
        )
        
        assert "governance" in result
        assert "curriculum_coherence_crosscheck" in result["governance"]
        assert result["governance"]["curriculum_coherence_crosscheck"]["status"] == "CONFLICT"
        assert result["governance"]["curriculum_coherence_crosscheck"]["episodes"] == crosscheck["episodes"]
        assert result["governance"]["curriculum_coherence_crosscheck"]["advisory_notes"] == crosscheck["advisory_notes"]

    # Parameterization tests
    
    def test_crosscheck_parameterization_defaults_identical(self):
        """Test that default parameters produce identical outputs to hardcoded values."""
        coherence_ts = {
            "schema_version": "1.0.0",
            "points": [
                {"cycle_or_run_idx": 0, "alignment_score": 0.5, "drift_band": "HIGH_DRIFT"},
            ],
        }
        curriculum_signals = []
        
        result_defaults = summarize_coherence_vs_curriculum_governance(coherence_ts, curriculum_signals)
        result_explicit = summarize_coherence_vs_curriculum_governance(
            coherence_ts,
            curriculum_signals,
            alignment_ok_threshold=0.8,
            alignment_conflict_threshold=0.6,
            high_drift_value="HIGH_DRIFT",
        )
        
        assert json.dumps(result_defaults, sort_keys=True) == json.dumps(result_explicit, sort_keys=True)

    def test_crosscheck_parameterization_custom_thresholds(self):
        """Test that custom thresholds affect status determination."""
        coherence_ts = {
            "schema_version": "1.0.0",
            "points": [
                {"cycle_or_run_idx": 0, "alignment_score": 0.75, "drift_band": "STABLE"},
            ],
        }
        curriculum_signals = []
        
        # With default threshold (0.8), this should be TENSION (0.75 < 0.8)
        result_default = summarize_coherence_vs_curriculum_governance(coherence_ts, curriculum_signals)
        assert result_default["status"] == "TENSION"
        
        # With custom threshold (0.7), this should be CONSISTENT (0.75 >= 0.7)
        result_custom = summarize_coherence_vs_curriculum_governance(
            coherence_ts,
            curriculum_signals,
            alignment_ok_threshold=0.7,
        )
        assert result_custom["status"] == "CONSISTENT"

    def test_crosscheck_parameterization_custom_high_drift(self):
        """Test that custom high_drift_value affects status determination."""
        coherence_ts = {
            "schema_version": "1.0.0",
            "points": [
                {"cycle_or_run_idx": 0, "alignment_score": 0.5, "drift_band": "CUSTOM_HIGH"},
            ],
        }
        curriculum_signals = []
        
        # With default high_drift_value ("HIGH_DRIFT"), this should be TENSION
        result_default = summarize_coherence_vs_curriculum_governance(coherence_ts, curriculum_signals)
        assert result_default["status"] == "TENSION"
        
        # With custom high_drift_value ("CUSTOM_HIGH"), this should be CONFLICT
        result_custom = summarize_coherence_vs_curriculum_governance(
            coherence_ts,
            curriculum_signals,
            high_drift_value="CUSTOM_HIGH",
        )
        assert result_custom["status"] == "CONFLICT"

    # Episode metadata tests
    
    def test_crosscheck_episode_metadata_point_count(self):
        """Test that episodes include point_count metadata."""
        coherence_ts = {
            "schema_version": "1.0.0",
            "points": [
                {"cycle_or_run_idx": 0, "alignment_score": 0.5, "drift_band": "STABLE"},  # TENSION
                {"cycle_or_run_idx": 50, "alignment_score": 0.4, "drift_band": "STABLE"},  # TENSION
                {"cycle_or_run_idx": 100, "alignment_score": 0.9, "drift_band": "STABLE"},  # CONSISTENT
            ],
        }
        curriculum_signals = []
        
        result = summarize_coherence_vs_curriculum_governance(coherence_ts, curriculum_signals)
        
        assert len(result["episodes"]) == 1
        assert result["episodes"][0]["point_count"] == 2

    def test_crosscheck_episode_metadata_max_drift_band(self):
        """Test that episodes include max_drift_band_seen metadata."""
        coherence_ts = {
            "schema_version": "1.0.0",
            "points": [
                {"cycle_or_run_idx": 0, "alignment_score": 0.5, "drift_band": "LOW_DRIFT"},  # TENSION
                {"cycle_or_run_idx": 50, "alignment_score": 0.4, "drift_band": "MEDIUM_DRIFT"},  # TENSION
                {"cycle_or_run_idx": 100, "alignment_score": 0.9, "drift_band": "STABLE"},  # CONSISTENT
            ],
        }
        curriculum_signals = []
        
        result = summarize_coherence_vs_curriculum_governance(coherence_ts, curriculum_signals)
        
        assert len(result["episodes"]) == 1
        assert result["episodes"][0]["max_drift_band_seen"] == "MEDIUM_DRIFT"

    def test_crosscheck_episode_metadata_min_alignment_score(self):
        """Test that episodes include min_alignment_score_seen metadata."""
        coherence_ts = {
            "schema_version": "1.0.0",
            "points": [
                {"cycle_or_run_idx": 0, "alignment_score": 0.5, "drift_band": "STABLE"},  # TENSION
                {"cycle_or_run_idx": 50, "alignment_score": 0.3, "drift_band": "STABLE"},  # TENSION
                {"cycle_or_run_idx": 100, "alignment_score": 0.9, "drift_band": "STABLE"},  # CONSISTENT
            ],
        }
        curriculum_signals = []
        
        result = summarize_coherence_vs_curriculum_governance(coherence_ts, curriculum_signals)
        
        assert len(result["episodes"]) == 1
        assert result["episodes"][0]["min_alignment_score_seen"] == 0.3

    def test_crosscheck_episode_metadata_all_fields(self):
        """Test that episodes include all required metadata fields."""
        coherence_ts = {
            "schema_version": "1.0.0",
            "points": [
                {"cycle_or_run_idx": 0, "alignment_score": 0.5, "drift_band": "HIGH_DRIFT"},  # CONFLICT
            ],
        }
        curriculum_signals = []
        
        result = summarize_coherence_vs_curriculum_governance(coherence_ts, curriculum_signals)
        
        assert len(result["episodes"]) == 1
        episode = result["episodes"][0]
        assert "point_count" in episode
        assert "max_drift_band_seen" in episode
        assert "min_alignment_score_seen" in episode
        assert episode["point_count"] == 1
        assert episode["max_drift_band_seen"] == "HIGH_DRIFT"
        assert episode["min_alignment_score_seen"] == 0.5

    # Summary block tests
    
    def test_crosscheck_summary_block_basic(self):
        """Test that summary block includes required fields."""
        coherence_ts = {
            "schema_version": "1.0.0",
            "points": [
                {"cycle_or_run_idx": 0, "alignment_score": 0.5, "drift_band": "HIGH_DRIFT"},
            ],
        }
        curriculum_signals = []
        
        result = summarize_coherence_vs_curriculum_governance(coherence_ts, curriculum_signals)
        
        assert "summary" in result
        assert result["summary"]["num_points"] == 1
        assert result["summary"]["num_episodes"] == 1
        assert result["summary"]["worst_status"] == "CONFLICT"
        assert result["summary"]["worst_episode"] is not None

    def test_crosscheck_summary_block_no_episodes(self):
        """Test summary block when there are no episodes."""
        coherence_ts = {
            "schema_version": "1.0.0",
            "points": [
                {"cycle_or_run_idx": 0, "alignment_score": 0.9, "drift_band": "STABLE"},
            ],
        }
        curriculum_signals = []
        
        result = summarize_coherence_vs_curriculum_governance(coherence_ts, curriculum_signals)
        
        assert result["summary"]["num_points"] == 1
        assert result["summary"]["num_episodes"] == 0
        assert result["summary"]["worst_status"] == "CONSISTENT"
        assert result["summary"]["worst_episode"] is None

    def test_crosscheck_summary_block_worst_episode_selection(self):
        """Test that worst_episode is selected by severity then length."""
        coherence_ts = {
            "schema_version": "1.0.0",
            "points": [
                {"cycle_or_run_idx": 0, "alignment_score": 0.5, "drift_band": "HIGH_DRIFT"},  # CONFLICT
                {"cycle_or_run_idx": 50, "alignment_score": 0.4, "drift_band": "HIGH_DRIFT"},  # CONFLICT
                {"cycle_or_run_idx": 100, "alignment_score": 0.9, "drift_band": "STABLE"},  # CONSISTENT
                {"cycle_or_run_idx": 150, "alignment_score": 0.7, "drift_band": "STABLE"},  # TENSION
            ],
        }
        curriculum_signals = []
        
        result = summarize_coherence_vs_curriculum_governance(coherence_ts, curriculum_signals)
        
        assert result["summary"]["worst_episode"] is not None
        assert result["summary"]["worst_episode"]["status"] == "CONFLICT"
        assert result["summary"]["worst_episode"]["start_idx"] == 0
        assert result["summary"]["worst_episode"]["end_idx"] == 50

    def test_crosscheck_summary_block_json_serializable(self):
        """Test that summary block is JSON serializable."""
        coherence_ts = {
            "schema_version": "1.0.0",
            "points": [
                {"cycle_or_run_idx": 0, "alignment_score": 0.5, "drift_band": "HIGH_DRIFT"},
            ],
        }
        curriculum_signals = []
        
        result = summarize_coherence_vs_curriculum_governance(coherence_ts, curriculum_signals)
        
        json_str = json.dumps(result["summary"], sort_keys=True)
        summary_roundtrip = json.loads(json_str)
        assert summary_roundtrip == result["summary"]

    # Episode severity score tests
    
    def test_crosscheck_episode_severity_score_present(self):
        """Test that episodes include episode_severity_score."""
        coherence_ts = {
            "schema_version": "1.0.0",
            "points": [
                {"cycle_or_run_idx": 0, "alignment_score": 0.5, "drift_band": "HIGH_DRIFT"},
            ],
        }
        curriculum_signals = []
        
        result = summarize_coherence_vs_curriculum_governance(coherence_ts, curriculum_signals)
        
        assert len(result["episodes"]) == 1
        assert "episode_severity_score" in result["episodes"][0]
        assert isinstance(result["episodes"][0]["episode_severity_score"], (int, float))

    def test_crosscheck_episode_severity_score_monotonicity(self):
        """Test that severity score increases monotonically with worse status."""
        coherence_ts = {
            "schema_version": "1.0.0",
            "points": [
                {"cycle_or_run_idx": 0, "alignment_score": 0.9, "drift_band": "STABLE"},  # CONSISTENT (should be 0)
                {"cycle_or_run_idx": 50, "alignment_score": 0.7, "drift_band": "STABLE"},  # TENSION
                {"cycle_or_run_idx": 100, "alignment_score": 0.5, "drift_band": "HIGH_DRIFT"},  # CONFLICT
            ],
        }
        curriculum_signals = []
        
        result = summarize_coherence_vs_curriculum_governance(coherence_ts, curriculum_signals)
        
        # Only TENSION and CONFLICT episodes (CONSISTENT points don't create episodes)
        assert len(result["episodes"]) == 2
        
        tension_episode = next(e for e in result["episodes"] if e["status"] == "TENSION")
        conflict_episode = next(e for e in result["episodes"] if e["status"] == "CONFLICT")
        
        # CONFLICT should have higher severity score than TENSION
        assert conflict_episode["episode_severity_score"] > tension_episode["episode_severity_score"]
        
        # Base scores: TENSION=50, CONFLICT=100
        assert tension_episode["episode_severity_score"] >= 50.0
        assert conflict_episode["episode_severity_score"] >= 100.0

    def test_crosscheck_episode_severity_score_drift_bump(self):
        """Test that HIGH_DRIFT adds +20 to severity score."""
        coherence_ts = {
            "schema_version": "1.0.0",
            "points": [
                {"cycle_or_run_idx": 0, "alignment_score": 0.7, "drift_band": "STABLE"},  # TENSION, no drift bump
                {"cycle_or_run_idx": 50, "alignment_score": 0.9, "drift_band": "STABLE"},  # CONSISTENT (breaks episode)
                {"cycle_or_run_idx": 100, "alignment_score": 0.7, "drift_band": "HIGH_DRIFT"},  # TENSION, with drift bump
            ],
        }
        curriculum_signals = []
        
        result = summarize_coherence_vs_curriculum_governance(coherence_ts, curriculum_signals)
        
        assert len(result["episodes"]) == 2
        
        stable_episode = next(e for e in result["episodes"] if e["max_drift_band_seen"] == "STABLE")
        high_drift_episode = next(e for e in result["episodes"] if e["max_drift_band_seen"] == "HIGH_DRIFT")
        
        # HIGH_DRIFT episode should have +20 more than STABLE episode (same status, same alignment)
        assert high_drift_episode["episode_severity_score"] == stable_episode["episode_severity_score"] + 20.0

    def test_crosscheck_episode_severity_score_alignment_bump(self):
        """Test that lower alignment scores increase severity score."""
        coherence_ts = {
            "schema_version": "1.0.0",
            "points": [
                {"cycle_or_run_idx": 0, "alignment_score": 0.75, "drift_band": "STABLE"},  # TENSION, closer to threshold
                {"cycle_or_run_idx": 50, "alignment_score": 0.9, "drift_band": "STABLE"},  # CONSISTENT (breaks episode)
                {"cycle_or_run_idx": 100, "alignment_score": 0.5, "drift_band": "STABLE"},  # TENSION, further from threshold
            ],
        }
        curriculum_signals = []
        
        result = summarize_coherence_vs_curriculum_governance(coherence_ts, curriculum_signals)
        
        assert len(result["episodes"]) == 2
        
        high_alignment_episode = next(e for e in result["episodes"] if e["min_alignment_score_seen"] == 0.75)
        low_alignment_episode = next(e for e in result["episodes"] if e["min_alignment_score_seen"] == 0.5)
        
        # Lower alignment should have higher severity score (same status, same drift)
        assert low_alignment_episode["episode_severity_score"] > high_alignment_episode["episode_severity_score"]

    def test_crosscheck_worst_episode_selection_by_severity_score(self):
        """Test that worst_episode is selected by highest severity score."""
        coherence_ts = {
            "schema_version": "1.0.0",
            "points": [
                {"cycle_or_run_idx": 0, "alignment_score": 0.7, "drift_band": "STABLE"},  # TENSION
                {"cycle_or_run_idx": 50, "alignment_score": 0.4, "drift_band": "HIGH_DRIFT"},  # CONFLICT (higher score)
            ],
        }
        curriculum_signals = []
        
        result = summarize_coherence_vs_curriculum_governance(coherence_ts, curriculum_signals)
        
        assert result["summary"]["worst_episode"] is not None
        assert result["summary"]["worst_episode"]["status"] == "CONFLICT"
        
        # Verify it has the highest severity score
        max_severity = max(e["episode_severity_score"] for e in result["episodes"])
        assert result["summary"]["worst_episode"]["episode_severity_score"] == max_severity

    def test_crosscheck_worst_episode_tie_break_by_duration(self):
        """Test that worst_episode tie-breaks by longer duration."""
        coherence_ts = {
            "schema_version": "1.0.0",
            "points": [
                {"cycle_or_run_idx": 0, "alignment_score": 0.5, "drift_band": "HIGH_DRIFT"},  # CONFLICT, 1 point
                {"cycle_or_run_idx": 50, "alignment_score": 0.4, "drift_band": "HIGH_DRIFT"},  # CONFLICT, 1 point
                {"cycle_or_run_idx": 100, "alignment_score": 0.45, "drift_band": "HIGH_DRIFT"},  # CONFLICT, extends first
            ],
        }
        curriculum_signals = []
        
        result = summarize_coherence_vs_curriculum_governance(coherence_ts, curriculum_signals)
        
        # Should have 2 episodes: [0-100] (3 points) and [50] (1 point)
        # Both CONFLICT with HIGH_DRIFT, similar alignment scores
        # Longer episode should be selected
        assert result["summary"]["worst_episode"] is not None
        worst_duration = result["summary"]["worst_episode"]["end_idx"] - result["summary"]["worst_episode"]["start_idx"]
        
        # Verify it's the longest episode
        max_duration = max(e["end_idx"] - e["start_idx"] for e in result["episodes"])
        assert worst_duration == max_duration

    def test_crosscheck_worst_episode_tie_break_by_start_idx(self):
        """Test that worst_episode tie-breaks by smaller start_idx when duration is equal."""
        coherence_ts = {
            "schema_version": "1.0.0",
            "points": [
                {"cycle_or_run_idx": 0, "alignment_score": 0.5, "drift_band": "HIGH_DRIFT"},  # CONFLICT, earlier
                {"cycle_or_run_idx": 50, "alignment_score": 0.9, "drift_band": "STABLE"},  # CONSISTENT (breaks episode)
                {"cycle_or_run_idx": 100, "alignment_score": 0.5, "drift_band": "HIGH_DRIFT"},  # CONFLICT, later
            ],
        }
        curriculum_signals = []
        
        result = summarize_coherence_vs_curriculum_governance(coherence_ts, curriculum_signals)
        
        # Both episodes have same status, drift, alignment, and duration (1 point each)
        # Should select the one with smaller start_idx (earlier in timeline)
        assert result["summary"]["worst_episode"] is not None
        assert result["summary"]["worst_episode"]["start_idx"] == 0  # Earlier episode

    def test_crosscheck_episode_severity_score_deterministic(self):
        """Test that episode severity scores are deterministic."""
        coherence_ts = {
            "schema_version": "1.0.0",
            "points": [
                {"cycle_or_run_idx": 0, "alignment_score": 0.5, "drift_band": "HIGH_DRIFT"},
            ],
        }
        curriculum_signals = []
        
        result1 = summarize_coherence_vs_curriculum_governance(coherence_ts, curriculum_signals)
        result2 = summarize_coherence_vs_curriculum_governance(coherence_ts, curriculum_signals)
        
        assert result1["episodes"][0]["episode_severity_score"] == result2["episodes"][0]["episode_severity_score"]

    def test_crosscheck_severity_score_monotonicity_extreme_bumps(self):
        """Test that severity score maintains monotonicity even with extreme drift/alignment bumps."""
        coherence_ts = {
            "schema_version": "1.0.0",
            "points": [
                {"cycle_or_run_idx": 0, "alignment_score": 0.9, "drift_band": "STABLE"},  # CONSISTENT (score: 0)
                {"cycle_or_run_idx": 50, "alignment_score": 0.0, "drift_band": "STABLE"},  # TENSION, extreme alignment gap
                {"cycle_or_run_idx": 100, "alignment_score": 0.0, "drift_band": "HIGH_DRIFT"},  # CONFLICT, extreme bumps
            ],
        }
        curriculum_signals = []
        
        result = summarize_coherence_vs_curriculum_governance(coherence_ts, curriculum_signals)
        
        # Only TENSION and CONFLICT episodes (CONSISTENT doesn't create episodes)
        assert len(result["episodes"]) == 2
        
        tension_episode = next(e for e in result["episodes"] if e["status"] == "TENSION")
        conflict_episode = next(e for e in result["episodes"] if e["status"] == "CONFLICT")
        
        # Even with extreme alignment bump, CONFLICT > TENSION (monotonicity preserved)
        # TENSION: 50 (base) + 8.0 (alignment: (0.8-0.0)*10 capped at 30, but 0.8*10=8.0) = 58.0
        # CONFLICT: 100 (base) + 20 (HIGH_DRIFT) + 8.0 (alignment) = 128.0
        assert conflict_episode["episode_severity_score"] > tension_episode["episode_severity_score"]
        
        # Verify scores maintain monotonicity
        assert tension_episode["episode_severity_score"] >= 50.0  # At least base TENSION score
        assert conflict_episode["episode_severity_score"] >= 100.0  # At least base CONFLICT score
        assert conflict_episode["episode_severity_score"] > tension_episode["episode_severity_score"] + 50.0  # CONFLICT base > TENSION base

    def test_crosscheck_severity_score_basis_present(self):
        """Test that severity_score_basis is present in crosscheck output."""
        coherence_ts = {
            "schema_version": "1.0.0",
            "points": [
                {"cycle_or_run_idx": 0, "alignment_score": 0.5, "drift_band": "HIGH_DRIFT"},
            ],
        }
        curriculum_signals = []
        
        result = summarize_coherence_vs_curriculum_governance(coherence_ts, curriculum_signals)
        
        assert "severity_score_basis" in result
        assert "status_weights" in result["severity_score_basis"]
        assert "drift_bump" in result["severity_score_basis"]
        assert "alignment_bump_formula" in result["severity_score_basis"]
        assert result["severity_score_basis"]["status_weights"]["CONSISTENT"] == 0.0
        assert result["severity_score_basis"]["status_weights"]["TENSION"] == 50.0
        assert result["severity_score_basis"]["status_weights"]["CONFLICT"] == 100.0
        assert result["severity_score_basis"]["drift_bump"] == 20.0

    def test_crosscheck_worst_episode_selected_by_field(self):
        """Test that worst_episode includes selected_by audit field."""
        coherence_ts = {
            "schema_version": "1.0.0",
            "points": [
                {"cycle_or_run_idx": 0, "alignment_score": 0.5, "drift_band": "HIGH_DRIFT"},  # CONFLICT
            ],
        }
        curriculum_signals = []
        
        result = summarize_coherence_vs_curriculum_governance(coherence_ts, curriculum_signals)
        
        assert result["summary"]["worst_episode"] is not None
        assert "selected_by" in result["summary"]["worst_episode"]
        assert result["summary"]["worst_episode"]["selected_by"] == ["severity_score"]

    def test_crosscheck_worst_episode_selected_by_tie_breakers(self):
        """Test that worst_episode selected_by includes tie-breakers when used."""
        coherence_ts = {
            "schema_version": "1.0.0",
            "points": [
                {"cycle_or_run_idx": 0, "alignment_score": 0.5, "drift_band": "HIGH_DRIFT"},  # CONFLICT
                {"cycle_or_run_idx": 50, "alignment_score": 0.9, "drift_band": "STABLE"},  # CONSISTENT
                {"cycle_or_run_idx": 100, "alignment_score": 0.5, "drift_band": "HIGH_DRIFT"},  # CONFLICT, same score
                {"cycle_or_run_idx": 150, "alignment_score": 0.5, "drift_band": "HIGH_DRIFT"},  # CONFLICT, extends previous
            ],
        }
        curriculum_signals = []
        
        result = summarize_coherence_vs_curriculum_governance(coherence_ts, curriculum_signals)
        
        assert result["summary"]["worst_episode"] is not None
        assert "selected_by" in result["summary"]["worst_episode"]
        # Should use duration tie-breaker (episode 100-150 is longer than 0)
        assert "duration" in result["summary"]["worst_episode"]["selected_by"]

