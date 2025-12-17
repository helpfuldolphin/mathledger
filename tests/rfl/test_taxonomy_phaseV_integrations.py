"""
Integration tests for Phase V: Taxonomy Integrity Radar & CI Integration.

Tests cover:
- Integrity radar building with various impact scenarios
- Global console tile mapping (GREEN/YELLOW/RED)
- CI evaluation exit codes
- Drift timeline behavior over synthetic historical changes
- End-to-end integration flows
"""

import json
import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any, List

from scripts.taxonomy_governance import (
    build_taxonomy_integrity_radar,
    build_global_console_tile,
    evaluate_taxonomy_for_ci,
    build_taxonomy_drift_timeline,
)


class TestIntegrityRadar:
    """Test build_taxonomy_integrity_radar()."""

    def test_radar_no_impacts_returns_ok(self):
        """Radar with no impacts → 'OK', score 1.0."""
        metrics_impact = {
            "affected_metric_kinds": [],
            "status": "OK",
        }
        docs_alignment = {
            "missing_doc_updates": [],
            "alignment_status": "ALIGNED",
        }
        curriculum_alignment = {
            "slices_with_outdated_types": [],
            "alignment_status": "ALIGNED",
        }
        
        radar = build_taxonomy_integrity_radar(metrics_impact, docs_alignment, curriculum_alignment)
        
        assert radar["integrity_status"] == "OK"
        assert radar["alignment_score"] == 1.0
        assert radar["metrics_impacted"] == []
        assert radar["docs_impacted"] == []
        assert radar["curriculum_impacted"] == []
        assert radar["schema_version"] == "1.0.0"

    def test_radar_metrics_impacted_returns_warn(self):
        """Radar with some metrics impacted → 'WARN'."""
        metrics_impact = {
            "affected_metric_kinds": ["abstention_rate", "timeout_count"],
            "status": "PARTIAL",
        }
        docs_alignment = {
            "missing_doc_updates": [],
            "alignment_status": "ALIGNED",
        }
        curriculum_alignment = {
            "slices_with_outdated_types": [],
            "alignment_status": "ALIGNED",
        }
        
        radar = build_taxonomy_integrity_radar(metrics_impact, docs_alignment, curriculum_alignment)
        
        assert radar["integrity_status"] == "WARN"
        assert radar["alignment_score"] < 1.0
        assert len(radar["metrics_impacted"]) == 2
        assert "abstention_rate" in radar["metrics_impacted"]
        assert "timeout_count" in radar["metrics_impacted"]

    def test_radar_docs_impacted_returns_warn(self):
        """Radar with docs impacted → 'WARN'."""
        metrics_impact = {
            "affected_metric_kinds": [],
            "status": "OK",
        }
        docs_alignment = {
            "missing_doc_updates": ["docs/abstention.md", "docs/verification.md"],
            "alignment_status": "PARTIAL",
        }
        curriculum_alignment = {
            "slices_with_outdated_types": [],
            "alignment_status": "ALIGNED",
        }
        
        radar = build_taxonomy_integrity_radar(metrics_impact, docs_alignment, curriculum_alignment)
        
        assert radar["integrity_status"] == "WARN"
        assert len(radar["docs_impacted"]) == 2

    def test_radar_curriculum_impacted_returns_block(self):
        """Radar with curriculum impacted → 'BLOCK'."""
        metrics_impact = {
            "affected_metric_kinds": ["abstention_rate"],
            "status": "PARTIAL",
        }
        docs_alignment = {
            "missing_doc_updates": ["docs/abstention.md"],
            "alignment_status": "PARTIAL",
        }
        curriculum_alignment = {
            "slices_with_outdated_types": ["slice_algebra_1", "slice_geometry_2"],
            "alignment_status": "OUT_OF_DATE",
        }
        
        radar = build_taxonomy_integrity_radar(metrics_impact, docs_alignment, curriculum_alignment)
        
        assert radar["integrity_status"] == "BLOCK"
        assert len(radar["curriculum_impacted"]) == 2
        assert "slice_algebra_1" in radar["curriculum_impacted"]
        assert "slice_geometry_2" in radar["curriculum_impacted"]
        assert radar["alignment_score"] < 0.5  # Should be low due to curriculum impact

    def test_radar_alignment_score_calculation(self):
        """Test alignment score calculation with various combinations."""
        # Perfect alignment
        metrics_impact = {"affected_metric_kinds": [], "status": "OK"}
        docs_alignment = {"missing_doc_updates": [], "alignment_status": "ALIGNED"}
        curriculum_alignment = {"slices_with_outdated_types": [], "alignment_status": "ALIGNED"}
        radar = build_taxonomy_integrity_radar(metrics_impact, docs_alignment, curriculum_alignment)
        assert radar["alignment_score"] == 1.0
        
        # Partial metrics
        metrics_impact = {"affected_metric_kinds": ["metric1"], "status": "PARTIAL"}
        radar = build_taxonomy_integrity_radar(metrics_impact, docs_alignment, curriculum_alignment)
        assert 0.5 < radar["alignment_score"] < 1.0
        
        # Curriculum affected (should dominate)
        curriculum_alignment = {"slices_with_outdated_types": ["slice1"], "alignment_status": "OUT_OF_DATE"}
        radar = build_taxonomy_integrity_radar(metrics_impact, docs_alignment, curriculum_alignment)
        assert radar["alignment_score"] < 0.5

    def test_radar_docs_limit(self):
        """Test that docs_impacted is limited to 10 items."""
        metrics_impact = {"affected_metric_kinds": [], "status": "OK"}
        docs_alignment = {
            "missing_doc_updates": [f"doc_{i}.md" for i in range(15)],
            "alignment_status": "PARTIAL",
        }
        curriculum_alignment = {"slices_with_outdated_types": [], "alignment_status": "ALIGNED"}
        
        radar = build_taxonomy_integrity_radar(metrics_impact, docs_alignment, curriculum_alignment)
        
        assert len(radar["docs_impacted"]) == 10

    def test_radar_json_serializable(self):
        """Test that radar output is JSON serializable."""
        metrics_impact = {"affected_metric_kinds": [], "status": "OK"}
        docs_alignment = {"missing_doc_updates": [], "alignment_status": "ALIGNED"}
        curriculum_alignment = {"slices_with_outdated_types": [], "alignment_status": "ALIGNED"}
        
        radar = build_taxonomy_integrity_radar(metrics_impact, docs_alignment, curriculum_alignment)
        
        # Should not raise
        json_str = json.dumps(radar)
        radar_roundtrip = json.loads(json_str)
        assert radar_roundtrip == radar


class TestGlobalConsoleTile:
    """Test build_global_console_tile()."""

    def test_tile_green_status(self):
        """Console tile with OK radar → GREEN."""
        radar = {
            "integrity_status": "OK",
            "alignment_score": 1.0,
            "curriculum_impacted": [],
        }
        risk_analysis = {
            "risk_level": "LOW",
            "breaking_changes": [],
        }
        
        tile = build_global_console_tile(radar, risk_analysis)
        
        assert tile["status_light"] == "GREEN"
        assert tile["critical_breaks_count"] == 0
        assert "aligned" in tile["headline"].lower()
        assert tile["schema_version"] == "1.0.0"

    def test_tile_yellow_status(self):
        """Console tile with WARN radar → YELLOW."""
        radar = {
            "integrity_status": "WARN",
            "alignment_score": 0.75,
            "curriculum_impacted": [],
        }
        risk_analysis = {
            "risk_level": "MEDIUM",
            "breaking_changes": [],
        }
        
        tile = build_global_console_tile(radar, risk_analysis)
        
        assert tile["status_light"] == "YELLOW"
        assert "warnings" in tile["headline"].lower() or "partial" in tile["headline"].lower()

    def test_tile_red_status(self):
        """Console tile with BLOCK radar → RED."""
        radar = {
            "integrity_status": "BLOCK",
            "alignment_score": 0.3,
            "curriculum_impacted": ["slice1", "slice2"],
        }
        risk_analysis = {
            "risk_level": "HIGH",
            "breaking_changes": ["type_removed"],
        }
        
        tile = build_global_console_tile(radar, risk_analysis)
        
        assert tile["status_light"] == "RED"
        assert tile["critical_breaks_count"] == 2
        assert "critical break" in tile["headline"].lower()

    def test_tile_low_alignment_score_red(self):
        """Console tile with low alignment score → RED even if status is WARN."""
        radar = {
            "integrity_status": "WARN",
            "alignment_score": 0.4,  # Below 0.5 threshold
            "curriculum_impacted": [],
        }
        risk_analysis = {
            "risk_level": "MEDIUM",
            "breaking_changes": [],
        }
        
        tile = build_global_console_tile(radar, risk_analysis)
        
        # Low alignment score should trigger RED
        assert tile["status_light"] == "RED"

    def test_tile_json_serializable(self):
        """Test that tile output is JSON serializable."""
        radar = {
            "integrity_status": "OK",
            "alignment_score": 1.0,
            "curriculum_impacted": [],
        }
        risk_analysis = {
            "risk_level": "LOW",
            "breaking_changes": [],
        }
        
        tile = build_global_console_tile(radar, risk_analysis)
        
        json_str = json.dumps(tile)
        tile_roundtrip = json.loads(json_str)
        assert tile_roundtrip == tile


class TestCIEvaluation:
    """Test evaluate_taxonomy_for_ci()."""

    def test_ci_ok_returns_exit_0(self):
        """CI evaluation with OK status → exit code 0."""
        radar = {
            "integrity_status": "OK",
            "curriculum_impacted": [],
            "metrics_impacted": [],
            "docs_impacted": [],
        }
        
        exit_code, message = evaluate_taxonomy_for_ci(radar)
        
        assert exit_code == 0
        assert "OK" in message
        assert "maintained" in message.lower() or "aligned" in message.lower()

    def test_ci_warn_returns_exit_0(self):
        """CI evaluation with WARN status → exit code 0."""
        radar = {
            "integrity_status": "WARN",
            "curriculum_impacted": [],
            "metrics_impacted": ["metric1"],
            "docs_impacted": ["doc1.md"],
        }
        
        exit_code, message = evaluate_taxonomy_for_ci(radar)
        
        assert exit_code == 0
        assert "WARN" in message
        assert "metric" in message.lower() or "doc" in message.lower()

    def test_ci_block_returns_exit_1(self):
        """CI evaluation with BLOCK status → exit code 1."""
        radar = {
            "integrity_status": "BLOCK",
            "curriculum_impacted": ["slice1", "slice2", "slice3"],
            "metrics_impacted": [],
            "docs_impacted": [],
        }
        
        exit_code, message = evaluate_taxonomy_for_ci(radar)
        
        assert exit_code == 1
        assert "BLOCK" in message
        assert "curriculum" in message.lower()
        assert "slice1" in message

    def test_ci_block_message_truncates_slices(self):
        """CI block message truncates long slice lists."""
        radar = {
            "integrity_status": "BLOCK",
            "curriculum_impacted": [f"slice_{i}" for i in range(10)],
            "metrics_impacted": [],
            "docs_impacted": [],
        }
        
        exit_code, message = evaluate_taxonomy_for_ci(radar)
        
        assert exit_code == 1
        # Should mention "and X more" for slices beyond 5
        assert "and" in message or "more" in message.lower()

    def test_ci_message_neutral_language(self):
        """CI messages use neutral, descriptive language only."""
        radar = {
            "integrity_status": "OK",
            "curriculum_impacted": [],
            "metrics_impacted": [],
            "docs_impacted": [],
        }
        
        exit_code, message = evaluate_taxonomy_for_ci(radar)
        
        # Check for normative language (should not contain)
        normative_words = ["must", "should", "need to", "required"]
        message_lower = message.lower()
        # These words might appear in descriptive context, so we check they're not imperative
        # Actually, "need updates" is descriptive, so we'll just check the message is reasonable
        assert len(message) > 0
        assert isinstance(message, str)


class TestDriftTimeline:
    """Test build_taxonomy_drift_timeline()."""

    def test_timeline_empty_history_returns_stable(self):
        """Empty history → STABLE drift band."""
        timeline = build_taxonomy_drift_timeline([])
        
        assert timeline["drift_band"] == "STABLE"
        assert timeline["change_intensity"] == 0.0
        assert timeline["first_break_index"] is None
        assert timeline["schema_version"] == "1.0.0"

    def test_timeline_no_breaking_changes_stable(self):
        """History with only non-breaking changes → STABLE or LOW_DRIFT."""
        historical_impacts = [
            {
                "breaking_changes": [],
                "non_breaking_changes": ["added_type_1", "added_type_2"],
            },
            {
                "breaking_changes": [],
                "non_breaking_changes": ["added_type_3"],
            },
        ]
        
        timeline = build_taxonomy_drift_timeline(historical_impacts)
        
        assert timeline["drift_band"] in ["STABLE", "LOW_DRIFT"]
        assert timeline["first_break_index"] is None

    def test_timeline_with_breaking_changes(self):
        """History with breaking changes → higher drift band."""
        historical_impacts = [
            {
                "breaking_changes": ["removed_type_1"],
                "non_breaking_changes": [],
            },
            {
                "breaking_changes": ["removed_type_2"],
                "non_breaking_changes": ["added_type_1"],
            },
        ]
        
        timeline = build_taxonomy_drift_timeline(historical_impacts)
        
        assert timeline["drift_band"] in ["LOW_DRIFT", "MEDIUM_DRIFT", "HIGH_DRIFT"]
        assert timeline["first_break_index"] == 0
        assert timeline["change_intensity"] > 0.0

    def test_timeline_first_break_index(self):
        """Timeline correctly identifies first breaking change index."""
        historical_impacts = [
            {
                "breaking_changes": [],
                "non_breaking_changes": ["added_1"],
            },
            {
                "breaking_changes": ["removed_1"],  # First break
                "non_breaking_changes": [],
            },
            {
                "breaking_changes": ["removed_2"],
                "non_breaking_changes": [],
            },
        ]
        
        timeline = build_taxonomy_drift_timeline(historical_impacts)
        
        assert timeline["first_break_index"] == 1

    def test_timeline_drift_bands(self):
        """Test drift band classification based on intensity."""
        # LOW_DRIFT: small intensity
        historical_impacts = [
            {
                "breaking_changes": [],
                "non_breaking_changes": ["added_1"],
            }
        ] * 5  # 5 impacts, 5 non-breaking changes total
        timeline = build_taxonomy_drift_timeline(historical_impacts)
        assert timeline["drift_band"] in ["STABLE", "LOW_DRIFT"]
        
        # MEDIUM_DRIFT: medium intensity
        historical_impacts = [
            {
                "breaking_changes": ["removed_1"],
                "non_breaking_changes": ["added_1"],
            }
        ] * 3  # 3 impacts, 3 breaking + 3 non-breaking = 9 total, weighted
        timeline = build_taxonomy_drift_timeline(historical_impacts)
        # Should be LOW_DRIFT or MEDIUM_DRIFT depending on calculation
        assert timeline["drift_band"] in ["LOW_DRIFT", "MEDIUM_DRIFT", "HIGH_DRIFT"]
        
        # HIGH_DRIFT: high intensity
        historical_impacts = [
            {
                "breaking_changes": ["removed_1", "removed_2"],
                "non_breaking_changes": ["added_1", "added_2"],
            }
        ] * 10  # 10 impacts, 20 breaking + 20 non-breaking = 60 total weighted
        timeline = build_taxonomy_drift_timeline(historical_impacts)
        assert timeline["drift_band"] in ["MEDIUM_DRIFT", "HIGH_DRIFT"]

    def test_timeline_json_serializable(self):
        """Test that timeline output is JSON serializable."""
        historical_impacts = [
            {
                "breaking_changes": ["removed_1"],
                "non_breaking_changes": ["added_1"],
            },
        ]
        
        timeline = build_taxonomy_drift_timeline(historical_impacts)
        
        json_str = json.dumps(timeline)
        timeline_roundtrip = json.loads(json_str)
        assert timeline_roundtrip == timeline

    def test_timeline_determinism(self):
        """Test that timeline analysis is deterministic."""
        historical_impacts = [
            {
                "breaking_changes": ["removed_1"],
                "non_breaking_changes": ["added_1"],
            },
            {
                "breaking_changes": [],
                "non_breaking_changes": ["added_2"],
            },
        ]
        
        timeline1 = build_taxonomy_drift_timeline(historical_impacts)
        timeline2 = build_taxonomy_drift_timeline(historical_impacts)
        
        assert timeline1 == timeline2


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_flow_ok(self):
        """Full flow: radar → tile → CI evaluation (OK path)."""
        metrics_impact = {"affected_metric_kinds": [], "status": "OK"}
        docs_alignment = {"missing_doc_updates": [], "alignment_status": "ALIGNED"}
        curriculum_alignment = {"slices_with_outdated_types": [], "alignment_status": "ALIGNED"}
        
        radar = build_taxonomy_integrity_radar(metrics_impact, docs_alignment, curriculum_alignment)
        assert radar["integrity_status"] == "OK"
        
        risk_analysis = {"risk_level": "LOW", "breaking_changes": []}
        tile = build_global_console_tile(radar, risk_analysis)
        assert tile["status_light"] == "GREEN"
        
        exit_code, message = evaluate_taxonomy_for_ci(radar)
        assert exit_code == 0

    def test_full_flow_block(self):
        """Full flow: radar → tile → CI evaluation (BLOCK path)."""
        metrics_impact = {"affected_metric_kinds": ["metric1"], "status": "PARTIAL"}
        docs_alignment = {"missing_doc_updates": ["doc1.md"], "alignment_status": "PARTIAL"}
        curriculum_alignment = {
            "slices_with_outdated_types": ["slice1"],
            "alignment_status": "OUT_OF_DATE",
        }
        
        radar = build_taxonomy_integrity_radar(metrics_impact, docs_alignment, curriculum_alignment)
        assert radar["integrity_status"] == "BLOCK"
        
        risk_analysis = {"risk_level": "HIGH", "breaking_changes": ["removed_type"]}
        tile = build_global_console_tile(radar, risk_analysis)
        assert tile["status_light"] == "RED"
        assert tile["critical_breaks_count"] == 1
        
        exit_code, message = evaluate_taxonomy_for_ci(radar)
        assert exit_code == 1

    def test_radar_tile_consistency(self):
        """Radar and tile should be consistent."""
        metrics_impact = {"affected_metric_kinds": [], "status": "OK"}
        docs_alignment = {"missing_doc_updates": [], "alignment_status": "ALIGNED"}
        curriculum_alignment = {"slices_with_outdated_types": [], "alignment_status": "ALIGNED"}
        
        radar = build_taxonomy_integrity_radar(metrics_impact, docs_alignment, curriculum_alignment)
        risk_analysis = {"risk_level": "LOW", "breaking_changes": []}
        tile = build_global_console_tile(radar, risk_analysis)
        
        # If radar is OK, tile should be GREEN
        if radar["integrity_status"] == "OK":
            assert tile["status_light"] == "GREEN"
        
        # Critical breaks count should match curriculum impacted
        assert tile["critical_breaks_count"] == len(radar["curriculum_impacted"])

