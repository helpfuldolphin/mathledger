"""
test_envelope_v4_p3p4_integration.py — Tests for Global Health Envelope v4 P3/P4/Evidence/Council Integration

PHASE VI — GLOBAL HEALTH ENVELOPE v4 INTEGRATION

Tests for envelope v4 integration into P3 stability reports, P4 calibration reports,
evidence packs, and uplift council summaries.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

from backend.health.envelope_v4_p3p4_integration import (
    attach_envelope_v4_to_p3_stability_report,
    attach_envelope_v4_to_p4_calibration_report,
    attach_envelope_v4_to_evidence,
    attach_release_attitude_strip_to_evidence,
    build_cal_exp_release_attitude_annex,
    build_first_light_release_attitude_annex,
    build_release_attitude_strip,
    extract_release_attitude_strip_signal,
    export_cal_exp_release_attitude_annex,
    summarize_envelope_v4_for_uplift_council,
)


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def sample_envelope_v4() -> Dict[str, Any]:
    """Sample Global Health Envelope v4."""
    return {
        "schema_version": "4.0.0",
        "global_band": "GREEN",
        "envelope_components": {
            "metric_health": {"band": "GREEN", "present": True},
            "drift_envelope": {"band": "GREEN", "present": True},
            "semantic_envelope": {"band": "GREEN", "present": True},
        },
        "cross_signal_hotspots": [],
        "headline": "Global health envelope: all components within acceptable ranges",
    }


@pytest.fixture
def sample_coherence_analysis() -> Dict[str, Any]:
    """Sample coherence analysis."""
    return {
        "schema_version": "1.0.0",
        "coherence_status": "COHERENT",
        "mismatches": [],
        "notes": ["All component signals aligned"],
    }


@pytest.fixture
def sample_director_mega_panel() -> Dict[str, Any]:
    """Sample director mega-panel."""
    return {
        "schema_version": "1.0.0",
        "release_ready": True,
        "mega_status_light": "🟢",
        "component_summary": {
            "E1_governance": {"status": "OK", "band": "GREEN"},
            "E5_narrative": {"status": "OK", "band": "GREEN"},
        },
        "global_envelope": {
            "global_band": "GREEN",
            "cross_signal_hotspots": [],
        },
        "coherence_analysis": {
            "coherence_status": "COHERENT",
            "mismatches": [],
        },
        "executive_headline": "System release-ready: all component signals within acceptable ranges",
    }


@pytest.fixture
def red_envelope_v4() -> Dict[str, Any]:
    """Critical envelope v4 for testing."""
    return {
        "schema_version": "4.0.0",
        "global_band": "RED",
        "envelope_components": {
            "metric_health": {"band": "RED", "present": True},
            "drift_envelope": {"band": "RED", "present": True},
        },
        "cross_signal_hotspots": [
            {"component": "metric_health", "issue": "Critical health issue"},
        ],
        "headline": "Global health envelope: 2 component(s) in critical state",
    }


@pytest.fixture
def mismatched_coherence_analysis() -> Dict[str, Any]:
    """Coherence analysis with mismatches."""
    return {
        "schema_version": "1.0.0",
        "coherence_status": "MISMATCHED",
        "mismatches": [
            {
                "components": ["metric_health", "semantic_envelope"],
                "issue": "Metrics indicate stable state while semantic consistency degraded",
                "severity": "HIGH",
            }
        ],
        "notes": ["1 high-severity coherence mismatch(es) detected"],
    }


@pytest.fixture
def sample_release_attitude_annex() -> Dict[str, Any]:
    """Sample release attitude annex."""
    return {
        "schema_version": "1.0.0",
        "global_band": "GREEN",
        "system_alignment": "ALIGNED",
        "release_ready": True,
        "status_light": "🟢",
    }


# ==============================================================================
# P3 STABILITY REPORT INTEGRATION TESTS
# ==============================================================================


class TestP3StabilityReportIntegration:
    """Tests for attaching envelope v4 to P3 stability reports."""
    
    def test_attach_envelope_v4_to_p3_has_required_fields(
        self,
        sample_envelope_v4,
        sample_coherence_analysis,
    ):
        """Test 1: Attached envelope v4 has required fields."""
        stability_report = {"schema_version": "1.0.0", "run_id": "test"}
        updated = attach_envelope_v4_to_p3_stability_report(
            stability_report,
            sample_envelope_v4,
            sample_coherence_analysis,
        )
        
        assert "global_health_envelope_v4" in updated
        envelope_summary = updated["global_health_envelope_v4"]
        assert "coherence_score" in envelope_summary
        assert "system_alignment" in envelope_summary
        assert "pressure_status" in envelope_summary
        assert "status_light" in envelope_summary
    
    def test_p3_envelope_v4_coherence_score_from_analysis(
        self,
        sample_envelope_v4,
        sample_coherence_analysis,
    ):
        """Test 2: Coherence score extracted from coherence analysis."""
        stability_report = {"schema_version": "1.0.0"}
        updated = attach_envelope_v4_to_p3_stability_report(
            stability_report,
            sample_envelope_v4,
            sample_coherence_analysis,
        )
        
        envelope_summary = updated["global_health_envelope_v4"]
        assert envelope_summary["coherence_score"] == 1.0  # COHERENT
    
    def test_p3_envelope_v4_status_light_green(
        self,
        sample_envelope_v4,
    ):
        """Test 3: Status light maps correctly to GREEN."""
        stability_report = {"schema_version": "1.0.0"}
        updated = attach_envelope_v4_to_p3_stability_report(
            stability_report,
            sample_envelope_v4,
        )
        
        envelope_summary = updated["global_health_envelope_v4"]
        assert envelope_summary["status_light"] == "GREEN"
    
    def test_p3_envelope_v4_status_light_red(
        self,
        red_envelope_v4,
    ):
        """Test 4: Status light maps correctly to RED."""
        stability_report = {"schema_version": "1.0.0"}
        updated = attach_envelope_v4_to_p3_stability_report(
            stability_report,
            red_envelope_v4,
        )
        
        envelope_summary = updated["global_health_envelope_v4"]
        assert envelope_summary["status_light"] == "RED"
    
    def test_p3_envelope_v4_pressure_status_elevated(
        self,
        red_envelope_v4,
    ):
        """Test 5: Pressure status reflects hotspot count."""
        stability_report = {"schema_version": "1.0.0"}
        updated = attach_envelope_v4_to_p3_stability_report(
            stability_report,
            red_envelope_v4,
        )
        
        envelope_summary = updated["global_health_envelope_v4"]
        assert envelope_summary["pressure_status"] in ["ELEVATED", "HIGH"]


# ==============================================================================
# P4 CALIBRATION REPORT INTEGRATION TESTS
# ==============================================================================


class TestP4CalibrationReportIntegration:
    """Tests for attaching envelope v4 to P4 calibration reports."""
    
    def test_attach_envelope_v4_to_p4_has_required_fields(
        self,
        sample_envelope_v4,
        sample_coherence_analysis,
        sample_director_mega_panel,
    ):
        """Test 6: Attached envelope v4 has required fields."""
        calibration_report = {"schema_version": "1.0.0", "run_id": "test"}
        updated = attach_envelope_v4_to_p4_calibration_report(
            calibration_report,
            sample_envelope_v4,
            sample_coherence_analysis,
            sample_director_mega_panel,
        )
        
        assert "envelope_v4" in updated
        envelope_calibration = updated["envelope_v4"]
        assert "envelope_v4" in envelope_calibration
        assert "coherence_analysis" in envelope_calibration
        assert "director_mega_panel" in envelope_calibration
    
    def test_p4_envelope_v4_non_mutating(
        self,
        sample_envelope_v4,
        sample_coherence_analysis,
        sample_director_mega_panel,
    ):
        """Test 7: P4 attachment is non-mutating."""
        calibration_report = {"schema_version": "1.0.0", "run_id": "test"}
        original_id = id(calibration_report)
        updated = attach_envelope_v4_to_p4_calibration_report(
            calibration_report,
            sample_envelope_v4,
            sample_coherence_analysis,
            sample_director_mega_panel,
        )
        
        assert id(updated) != original_id
        assert "envelope_v4" not in calibration_report


# ==============================================================================
# EVIDENCE PACK INTEGRATION TESTS
# ==============================================================================


class TestEvidencePackIntegration:
    """Tests for attaching envelope v4 to evidence packs."""
    
    def test_attach_envelope_v4_to_evidence_has_required_fields(
        self,
        sample_envelope_v4,
        sample_coherence_analysis,
        sample_director_mega_panel,
    ):
        """Test 8: Attached envelope v4 has required fields."""
        evidence = {"schema_version": "1.0.0"}
        updated = attach_envelope_v4_to_evidence(
            evidence,
            sample_envelope_v4,
            sample_coherence_analysis,
            sample_director_mega_panel,
        )
        
        assert "governance" in updated
        assert "envelope_v4" in updated["governance"]
        envelope_evidence = updated["governance"]["envelope_v4"]
        assert "envelope_v4" in envelope_evidence
        assert "coherence_analysis" in envelope_evidence
        assert "director_mega_panel" in envelope_evidence
    
    def test_evidence_envelope_v4_preserves_existing_governance(
        self,
        sample_envelope_v4,
        sample_coherence_analysis,
        sample_director_mega_panel,
    ):
        """Test 9: Evidence attachment preserves existing governance."""
        evidence = {
            "schema_version": "1.0.0",
            "governance": {"existing_field": "value"},
        }
        updated = attach_envelope_v4_to_evidence(
            evidence,
            sample_envelope_v4,
            sample_coherence_analysis,
            sample_director_mega_panel,
        )
        
        assert updated["governance"]["existing_field"] == "value"
        assert "envelope_v4" in updated["governance"]
    
    def test_evidence_envelope_v4_non_mutating(
        self,
        sample_envelope_v4,
        sample_coherence_analysis,
        sample_director_mega_panel,
    ):
        """Test 10: Evidence attachment is non-mutating."""
        evidence = {"schema_version": "1.0.0"}
        original_id = id(evidence)
        updated = attach_envelope_v4_to_evidence(
            evidence,
            sample_envelope_v4,
            sample_coherence_analysis,
            sample_director_mega_panel,
        )
        
        assert id(updated) != original_id
        assert "governance" not in evidence


# ==============================================================================
# UPLIFT COUNCIL INTEGRATION TESTS
# ==============================================================================


class TestUpliftCouncilIntegration:
    """Tests for summarizing envelope v4 for uplift council."""
    
    def test_summarize_envelope_v4_council_has_required_fields(
        self,
        sample_director_mega_panel,
    ):
        """Test 11: Council summary has required fields."""
        summary = summarize_envelope_v4_for_uplift_council(sample_director_mega_panel)
        
        assert "council_status" in summary
        assert "status_light" in summary
        assert "top_drivers" in summary
        assert "coherence_discrepancies" in summary
        assert "cross_pillar_faults" in summary
        assert "rationale" in summary
    
    def test_council_status_green_maps_to_ok(
        self,
        sample_director_mega_panel,
    ):
        """Test 12: GREEN status light maps to OK."""
        summary = summarize_envelope_v4_for_uplift_council(sample_director_mega_panel)
        
        assert summary["council_status"] == "OK"
        assert summary["status_light"] == "🟢"
    
    def test_council_status_red_maps_to_block(self):
        """Test 13: RED status light maps to BLOCK."""
        red_mega_panel = {
            "mega_status_light": "🔴",
            "global_envelope": {
                "global_band": "RED",
                "cross_signal_hotspots": [{"component": "test", "issue": "test"}],
            },
            "coherence_analysis": {"mismatches": []},
            "component_summary": {
                "E1_governance": {"status": "CRITICAL", "band": "RED"},
            },
            "release_ready": False,
        }
        summary = summarize_envelope_v4_for_uplift_council(red_mega_panel)
        
        assert summary["council_status"] == "BLOCK"
        assert summary["status_light"] == "🔴"
    
    def test_council_status_yellow_maps_to_warn(self):
        """Test 14: YELLOW status light maps to WARN."""
        yellow_mega_panel = {
            "mega_status_light": "🟡",
            "global_envelope": {
                "global_band": "YELLOW",
                "cross_signal_hotspots": [],
            },
            "coherence_analysis": {"mismatches": []},
            "component_summary": {
                "E1_governance": {"status": "WARN", "band": "YELLOW"},
            },
            "release_ready": True,
        }
        summary = summarize_envelope_v4_for_uplift_council(yellow_mega_panel)
        
        assert summary["council_status"] == "WARN"
        assert summary["status_light"] == "🟡"
    
    def test_council_extracts_top_drivers(
        self,
        sample_director_mega_panel,
    ):
        """Test 15: Council summary extracts top drivers."""
        summary = summarize_envelope_v4_for_uplift_council(sample_director_mega_panel)
        
        assert isinstance(summary["top_drivers"], list)
    
    def test_council_extracts_coherence_discrepancies(self):
        """Test 16: Council summary extracts coherence discrepancies."""
        mega_panel_with_mismatches = {
            "mega_status_light": "🟡",
            "global_envelope": {
                "global_band": "YELLOW",
                "cross_signal_hotspots": [],
            },
            "coherence_analysis": {
                "mismatches": [
                    {
                        "components": ["metric_health", "semantic_envelope"],
                        "issue": "Test mismatch",
                        "severity": "HIGH",
                    }
                ],
            },
            "component_summary": {},
            "release_ready": True,
        }
        summary = summarize_envelope_v4_for_uplift_council(mega_panel_with_mismatches)
        
        assert len(summary["coherence_discrepancies"]) == 1
        assert summary["coherence_discrepancies"][0]["severity"] == "HIGH"
    
    def test_council_extracts_cross_pillar_faults(self):
        """Test 17: Council summary extracts cross-pillar faults."""
        mega_panel_with_hotspots = {
            "mega_status_light": "🟡",
            "global_envelope": {
                "global_band": "YELLOW",
                "cross_signal_hotspots": [
                    {"component": "metric_health", "issue": "Test hotspot"},
                ],
            },
            "coherence_analysis": {"mismatches": []},
            "component_summary": {},
            "release_ready": True,
        }
        summary = summarize_envelope_v4_for_uplift_council(mega_panel_with_hotspots)
        
        assert len(summary["cross_pillar_faults"]) == 1
        assert summary["cross_pillar_faults"][0]["component"] == "metric_health"
    
    def test_council_rationale_neutral_language(
        self,
        sample_director_mega_panel,
    ):
        """Test 18: Council rationale uses neutral language."""
        summary = summarize_envelope_v4_for_uplift_council(sample_director_mega_panel)
        
        rationale = summary["rationale"].lower()
        forbidden = ["bad", "good", "broken", "fixed", "healthy", "unhealthy"]
        for word in forbidden:
            assert word not in rationale
    
    def test_council_rationale_block_includes_critical_count(self):
        """Test 19: BLOCK rationale mentions critical components."""
        red_mega_panel = {
            "mega_status_light": "🔴",
            "global_envelope": {
                "global_band": "RED",
                "cross_signal_hotspots": [],
            },
            "coherence_analysis": {"mismatches": []},
            "component_summary": {
                "E1_governance": {"status": "CRITICAL", "band": "RED"},
                "E5_narrative": {"status": "CRITICAL", "band": "RED"},
            },
            "release_ready": False,
        }
        summary = summarize_envelope_v4_for_uplift_council(red_mega_panel)
        
        assert summary["council_status"] == "BLOCK"
        assert "critical" in summary["rationale"].lower() or "component" in summary["rationale"].lower()
    
    def test_council_release_ready_preserved(
        self,
        sample_director_mega_panel,
    ):
        """Test 20: Council summary preserves release_ready field."""
        summary = summarize_envelope_v4_for_uplift_council(sample_director_mega_panel)
        
        assert "release_ready" in summary
        assert summary["release_ready"] is True


# ==============================================================================
# RELEASE ATTITUDE ANNEX TESTS
# ==============================================================================
#
# IMPORTANT: The Release Attitude Annex is a SUMMARY BADGE, not an enforcement gate.
# It provides observational information about system health posture across all
# governance pillars (E1-E5). Decisions regarding release remain with human
# governance processes and are not automated based on this annex alone.
#
# The annex answers: "What attitude is the whole system in?" regarding release
# readiness. It synthesizes global_band, system_alignment, release_ready, and
# status_light into a concise cross-pillar summary for auditors and governance.
#
# ==============================================================================


class TestReleaseAttitudeAnnex:
    """Tests for First-Light Release Attitude Annex."""
    
    def test_release_attitude_annex_has_required_fields(
        self,
        sample_envelope_v4,
        sample_director_mega_panel,
    ):
        """Test 21: Release attitude annex has required fields."""
        annex = build_first_light_release_attitude_annex(
            sample_envelope_v4,
            sample_director_mega_panel,
        )
        
        assert "schema_version" in annex
        assert "global_band" in annex
        assert "system_alignment" in annex
        assert "release_ready" in annex
        assert "status_light" in annex
        assert annex["schema_version"] == "1.0.0"
    
    def test_release_attitude_annex_global_band_extracted(
        self,
        sample_envelope_v4,
        sample_director_mega_panel,
    ):
        """Test 22: Global band extracted from envelope v4."""
        annex = build_first_light_release_attitude_annex(
            sample_envelope_v4,
            sample_director_mega_panel,
        )
        
        assert annex["global_band"] == "GREEN"
    
    def test_release_attitude_annex_system_alignment_aligned(
        self,
        sample_envelope_v4,
        sample_director_mega_panel,
    ):
        """Test 23: System alignment calculated correctly for aligned system."""
        annex = build_first_light_release_attitude_annex(
            sample_envelope_v4,
            sample_director_mega_panel,
        )
        
        assert annex["system_alignment"] == "ALIGNED"
    
    def test_release_attitude_annex_system_alignment_partial(self):
        """Test 24: System alignment calculated correctly for partial alignment."""
        partial_envelope = {
            "schema_version": "4.0.0",
            "global_band": "YELLOW",
            "envelope_components": {
                "metric_health": {"band": "GREEN", "present": True},
                "drift_envelope": {"band": "YELLOW", "present": True},
                "semantic_envelope": {"band": "YELLOW", "present": True},
            },
        }
        mega_panel = {
            "release_ready": False,
            "mega_status_light": "🟡",
        }
        
        annex = build_first_light_release_attitude_annex(partial_envelope, mega_panel)
        
        # 1/3 aligned = 0.33 < 0.5, should be MISALIGNED
        # Actually, 1/3 = 0.33 < 0.5, so MISALIGNED
        assert annex["system_alignment"] in ["PARTIAL", "MISALIGNED"]
    
    def test_release_attitude_annex_release_ready_extracted(
        self,
        sample_envelope_v4,
        sample_director_mega_panel,
    ):
        """Test 25: Release ready extracted from director mega-panel."""
        annex = build_first_light_release_attitude_annex(
            sample_envelope_v4,
            sample_director_mega_panel,
        )
        
        assert annex["release_ready"] is True
    
    def test_release_attitude_annex_status_light_extracted(
        self,
        sample_envelope_v4,
        sample_director_mega_panel,
    ):
        """Test 26: Status light extracted from director mega-panel."""
        annex = build_first_light_release_attitude_annex(
            sample_envelope_v4,
            sample_director_mega_panel,
        )
        
        assert annex["status_light"] == "🟢"
    
    def test_release_attitude_annex_non_mutating(
        self,
        sample_envelope_v4,
        sample_director_mega_panel,
    ):
        """Test 27: Release attitude annex is non-mutating."""
        original_envelope = dict(sample_envelope_v4)
        original_panel = dict(sample_director_mega_panel)
        
        build_first_light_release_attitude_annex(
            sample_envelope_v4,
            sample_director_mega_panel,
        )
        
        assert sample_envelope_v4 == original_envelope
        assert sample_director_mega_panel == original_panel
    
    def test_release_attitude_annex_json_serializable(
        self,
        sample_envelope_v4,
        sample_director_mega_panel,
    ):
        """Test 28: Release attitude annex is JSON serializable."""
        import json
        
        annex = build_first_light_release_attitude_annex(
            sample_envelope_v4,
            sample_director_mega_panel,
        )
        
        json_str = json.dumps(annex)
        parsed = json.loads(json_str)
        
        assert parsed == annex
    
    def test_release_attitude_annex_deterministic(
        self,
        sample_envelope_v4,
        sample_director_mega_panel,
    ):
        """Test 29: Release attitude annex is deterministic."""
        annex1 = build_first_light_release_attitude_annex(
            sample_envelope_v4,
            sample_director_mega_panel,
        )
        annex2 = build_first_light_release_attitude_annex(
            sample_envelope_v4,
            sample_director_mega_panel,
        )
        
        assert annex1 == annex2
    
    def test_evidence_includes_release_attitude_annex(
        self,
        sample_envelope_v4,
        sample_coherence_analysis,
        sample_director_mega_panel,
    ):
        """Test 30: Evidence attachment includes release attitude annex."""
        evidence = {"schema_version": "1.0.0"}
        updated = attach_envelope_v4_to_evidence(
            evidence,
            sample_envelope_v4,
            sample_coherence_analysis,
            sample_director_mega_panel,
        )
        
        assert "governance" in updated
        assert "envelope_v4" in updated["governance"]
        assert "first_light_release_attitude" in updated["governance"]["envelope_v4"]
        
        annex = updated["governance"]["envelope_v4"]["first_light_release_attitude"]
        assert annex["schema_version"] == "1.0.0"
        assert "global_band" in annex
        assert "system_alignment" in annex
        assert "release_ready" in annex
        assert "status_light" in annex


# ==============================================================================
# RELEASE ATTITUDE STRIP TESTS
# ==============================================================================


class TestCalExpReleaseAttitudeAnnex:
    """Tests for build_cal_exp_release_attitude_annex()."""
    
    def test_builds_annex_with_cal_id(self, sample_release_attitude_annex):
        """Test that annex includes cal_id and all required fields."""
        result = build_cal_exp_release_attitude_annex("CAL-EXP-1", sample_release_attitude_annex)
        
        assert result["schema_version"] == "1.0.0"
        assert result["cal_id"] == "CAL-EXP-1"
        assert result["global_band"] == "GREEN"
        assert result["system_alignment"] == "ALIGNED"
        assert result["release_ready"] is True
        assert result["status_light"] == "🟢"
    
    def test_preserves_annex_fields(self):
        """Test that all annex fields are preserved in cal_exp annex."""
        annex = {
            "global_band": "YELLOW",
            "system_alignment": "PARTIAL",
            "release_ready": False,
            "status_light": "🟡",
        }
        result = build_cal_exp_release_attitude_annex("CAL-EXP-2", annex)
        
        assert result["cal_id"] == "CAL-EXP-2"
        assert result["global_band"] == "YELLOW"
        assert result["system_alignment"] == "PARTIAL"
        assert result["release_ready"] is False
        assert result["status_light"] == "🟡"
    
    def test_handles_missing_fields(self):
        """Test that missing fields default appropriately."""
        annex = {}
        result = build_cal_exp_release_attitude_annex("CAL-EXP-3", annex)
        
        assert result["cal_id"] == "CAL-EXP-3"
        assert result["global_band"] == "GREEN"  # Default
        assert result["system_alignment"] == "ALIGNED"  # Default
        assert result["release_ready"] is True  # Default
        assert result["status_light"] == "🟢"  # Default
    
    def test_non_mutating(self, sample_release_attitude_annex):
        """Test that input annex is not modified."""
        original = dict(sample_release_attitude_annex)
        _ = build_cal_exp_release_attitude_annex("CAL-EXP-1", sample_release_attitude_annex)
        
        assert sample_release_attitude_annex == original


class TestExportCalExpReleaseAttitudeAnnex:
    """Tests for export_cal_exp_release_attitude_annex()."""
    
    def test_exports_to_correct_path(self, sample_release_attitude_annex):
        """Test that annex is exported to correct file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            path = export_cal_exp_release_attitude_annex("CAL-EXP-1", sample_release_attitude_annex, output_dir)
            
            assert path == output_dir / "calibration" / "release_attitude_annex_CAL-EXP-1.json"
            assert path.exists()
    
    def test_creates_calibration_directory(self, sample_release_attitude_annex):
        """Test that calibration directory is created if missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            calibration_dir = output_dir / "calibration"
            assert not calibration_dir.exists()
            
            _ = export_cal_exp_release_attitude_annex("CAL-EXP-1", sample_release_attitude_annex, output_dir)
            
            assert calibration_dir.exists()
            assert calibration_dir.is_dir()
    
    def test_exported_file_is_valid_json(self, sample_release_attitude_annex):
        """Test that exported file contains valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            path = export_cal_exp_release_attitude_annex("CAL-EXP-1", sample_release_attitude_annex, output_dir)
            
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert data["schema_version"] == "1.0.0"
            assert data["cal_id"] == "CAL-EXP-1"
            assert "global_band" in data
            assert "system_alignment" in data
            assert "release_ready" in data
            assert "status_light" in data
    
    def test_exported_file_contains_cal_id(self, sample_release_attitude_annex):
        """Test that exported file includes cal_id in structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            path = export_cal_exp_release_attitude_annex("CAL-EXP-2", sample_release_attitude_annex, output_dir)
            
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert data["cal_id"] == "CAL-EXP-2"


class TestReleaseAttitudeStrip:
    """Tests for build_release_attitude_strip()."""
    
    def test_builds_strip_from_annexes(self):
        """Test that strip is built from list of annexes."""
        annexes = [
            {
                "cal_id": "CAL-EXP-1",
                "global_band": "GREEN",
                "system_alignment": "ALIGNED",
                "release_ready": True,
                "status_light": "🟢",
            },
            {
                "cal_id": "CAL-EXP-2",
                "global_band": "YELLOW",
                "system_alignment": "PARTIAL",
                "release_ready": False,
                "status_light": "🟡",
            },
        ]
        result = build_release_attitude_strip(annexes)
        
        assert result["schema_version"] == "1.0.0"
        assert len(result["experiments"]) == 2
        assert result["summary"]["total_count"] == 2
        assert result["summary"]["release_ready_count"] == 1
        assert result["summary"]["release_ready_ratio"] == 0.5
    
    def test_experiments_sorted_by_cal_id(self):
        """Test that experiments are sorted by cal_id for determinism."""
        annexes = [
            {"cal_id": "CAL-EXP-3", "global_band": "GREEN", "system_alignment": "ALIGNED", "release_ready": True, "status_light": "🟢"},
            {"cal_id": "CAL-EXP-1", "global_band": "GREEN", "system_alignment": "ALIGNED", "release_ready": True, "status_light": "🟢"},
            {"cal_id": "CAL-EXP-2", "global_band": "GREEN", "system_alignment": "ALIGNED", "release_ready": True, "status_light": "🟢"},
        ]
        result = build_release_attitude_strip(annexes)
        
        cal_ids = [exp["cal_id"] for exp in result["experiments"]]
        assert cal_ids == ["CAL-EXP-1", "CAL-EXP-2", "CAL-EXP-3"]
    
    def test_summary_counts_release_ready(self):
        """Test that summary correctly counts release_ready experiments."""
        annexes = [
            {"cal_id": "CAL-EXP-1", "global_band": "GREEN", "system_alignment": "ALIGNED", "release_ready": True, "status_light": "🟢"},
            {"cal_id": "CAL-EXP-2", "global_band": "GREEN", "system_alignment": "ALIGNED", "release_ready": True, "status_light": "🟢"},
            {"cal_id": "CAL-EXP-3", "global_band": "RED", "system_alignment": "MISALIGNED", "release_ready": False, "status_light": "🔴"},
        ]
        result = build_release_attitude_strip(annexes)
        
        assert result["summary"]["total_count"] == 3
        assert result["summary"]["release_ready_count"] == 2
        assert result["summary"]["release_ready_ratio"] == pytest.approx(0.667, abs=0.001)
    
    def test_handles_empty_list(self):
        """Test that empty annex list produces valid strip."""
        result = build_release_attitude_strip([])
        
        assert result["schema_version"] == "1.0.0"
        assert result["experiments"] == []
        assert result["summary"]["total_count"] == 0
        assert result["summary"]["release_ready_count"] == 0
        assert result["summary"]["release_ready_ratio"] == 0.0
    
    def test_handles_missing_fields(self):
        """Test that missing fields default appropriately."""
        annexes = [
            {"cal_id": "CAL-EXP-1"},  # Missing fields
        ]
        result = build_release_attitude_strip(annexes)
        
        assert len(result["experiments"]) == 1
        exp = result["experiments"][0]
        assert exp["cal_id"] == "CAL-EXP-1"
        assert exp["global_band"] == "GREEN"  # Default
        assert exp["system_alignment"] == "ALIGNED"  # Default
        assert exp["release_ready"] is True  # Default
        assert exp["status_light"] == "🟢"  # Default
    
    def test_non_mutating(self):
        """Test that input annexes are not modified."""
        annexes = [
            {"cal_id": "CAL-EXP-1", "global_band": "GREEN", "system_alignment": "ALIGNED", "release_ready": True, "status_light": "🟢"},
        ]
        original = [dict(a) for a in annexes]
        _ = build_release_attitude_strip(annexes)
        
        assert annexes == original
    
    def test_json_serializable(self):
        """Test that strip is JSON serializable."""
        annexes = [
            {"cal_id": "CAL-EXP-1", "global_band": "GREEN", "system_alignment": "ALIGNED", "release_ready": True, "status_light": "🟢"},
        ]
        result = build_release_attitude_strip(annexes)
        
        # Should not raise
        json_str = json.dumps(result, ensure_ascii=False)
        assert isinstance(json_str, str)
        assert len(json_str) > 0
    
    def test_deterministic_output(self):
        """Test that identical inputs produce identical outputs."""
        annexes = [
            {"cal_id": "CAL-EXP-2", "global_band": "YELLOW", "system_alignment": "PARTIAL", "release_ready": False, "status_light": "🟡"},
            {"cal_id": "CAL-EXP-1", "global_band": "GREEN", "system_alignment": "ALIGNED", "release_ready": True, "status_light": "🟢"},
        ]
        
        result1 = build_release_attitude_strip(annexes)
        result2 = build_release_attitude_strip(annexes)
        
        assert result1 == result2
        # Also verify ordering is consistent
        assert [exp["cal_id"] for exp in result1["experiments"]] == ["CAL-EXP-1", "CAL-EXP-2"]
    
    def test_trend_improving(self):
        """Test that trend is IMPROVING when status improves over sequence."""
        annexes = [
            {"cal_id": "CAL-EXP-1", "global_band": "RED", "system_alignment": "MISALIGNED", "release_ready": False, "status_light": "🔴"},
            {"cal_id": "CAL-EXP-2", "global_band": "YELLOW", "system_alignment": "PARTIAL", "release_ready": False, "status_light": "🟡"},
            {"cal_id": "CAL-EXP-3", "global_band": "GREEN", "system_alignment": "ALIGNED", "release_ready": True, "status_light": "🟢"},
        ]
        result = build_release_attitude_strip(annexes)
        
        assert result["summary"]["trend"] == "IMPROVING"
    
    def test_trend_degrading(self):
        """Test that trend is DEGRADING when status degrades over sequence."""
        annexes = [
            {"cal_id": "CAL-EXP-1", "global_band": "GREEN", "system_alignment": "ALIGNED", "release_ready": True, "status_light": "🟢"},
            {"cal_id": "CAL-EXP-2", "global_band": "YELLOW", "system_alignment": "PARTIAL", "release_ready": False, "status_light": "🟡"},
            {"cal_id": "CAL-EXP-3", "global_band": "RED", "system_alignment": "MISALIGNED", "release_ready": False, "status_light": "🔴"},
        ]
        result = build_release_attitude_strip(annexes)
        
        assert result["summary"]["trend"] == "DEGRADING"
    
    def test_trend_stable(self):
        """Test that trend is STABLE when status remains constant."""
        annexes = [
            {"cal_id": "CAL-EXP-1", "global_band": "GREEN", "system_alignment": "ALIGNED", "release_ready": True, "status_light": "🟢"},
            {"cal_id": "CAL-EXP-2", "global_band": "GREEN", "system_alignment": "ALIGNED", "release_ready": True, "status_light": "🟢"},
        ]
        result = build_release_attitude_strip(annexes)
        
        assert result["summary"]["trend"] == "STABLE"
    
    def test_trend_stable_with_single_experiment(self):
        """Test that trend is STABLE with only one experiment."""
        annexes = [
            {"cal_id": "CAL-EXP-1", "global_band": "GREEN", "system_alignment": "ALIGNED", "release_ready": True, "status_light": "🟢"},
        ]
        result = build_release_attitude_strip(annexes)
        
        assert result["summary"]["trend"] == "STABLE"
    
    def test_trend_stable_with_empty_list(self):
        """Test that trend is STABLE with empty list."""
        result = build_release_attitude_strip([])
        
        assert result["summary"]["trend"] == "STABLE"
    
    def test_trend_handles_unknown_status_light(self):
        """Test that unknown status lights are handled gracefully."""
        annexes = [
            {"cal_id": "CAL-EXP-1", "status_light": "UNKNOWN"},
            {"cal_id": "CAL-EXP-2", "status_light": "🟢"},
        ]
        result = build_release_attitude_strip(annexes)
        
        # Should not raise, trend should be computed
        assert "trend" in result["summary"]
        assert result["summary"]["trend"] in ["IMPROVING", "STABLE", "DEGRADING"]


class TestAttachReleaseAttitudeStripToEvidence:
    """Tests for attach_release_attitude_strip_to_evidence()."""
    
    def test_attaches_strip_to_evidence(self):
        """Test that strip is attached to evidence under correct path."""
        evidence = {"metadata": {"run_id": "test-001"}}
        strip = {
            "schema_version": "1.0.0",
            "experiments": [],
            "summary": {"total_count": 0, "release_ready_count": 0, "release_ready_ratio": 0.0},
        }
        
        result = attach_release_attitude_strip_to_evidence(evidence, strip)
        
        assert result["governance"]["release_attitude_strip"] == strip
        assert result["metadata"]["run_id"] == "test-001"  # Original preserved
    
    def test_creates_governance_section_if_missing(self):
        """Test that governance section is created if not present."""
        evidence = {}
        strip = {
            "schema_version": "1.0.0",
            "experiments": [],
            "summary": {"total_count": 0, "release_ready_count": 0, "release_ready_ratio": 0.0},
        }
        
        result = attach_release_attitude_strip_to_evidence(evidence, strip)
        
        assert "governance" in result
        assert "release_attitude_strip" in result["governance"]
    
    def test_preserves_existing_governance(self):
        """Test that existing governance entries are preserved."""
        evidence = {
            "governance": {
                "envelope_v4": {"global_band": "GREEN"},
            },
        }
        strip = {
            "schema_version": "1.0.0",
            "experiments": [],
            "summary": {"total_count": 0, "release_ready_count": 0, "release_ready_ratio": 0.0},
        }
        
        result = attach_release_attitude_strip_to_evidence(evidence, strip)
        
        assert result["governance"]["envelope_v4"]["global_band"] == "GREEN"
        assert "release_attitude_strip" in result["governance"]
    
    def test_non_mutating(self):
        """Test that input evidence is not modified."""
        evidence = {"metadata": {"run_id": "test-001"}}
        strip = {
            "schema_version": "1.0.0",
            "experiments": [],
            "summary": {"total_count": 0, "release_ready_count": 0, "release_ready_ratio": 0.0},
        }
        original_evidence = dict(evidence)
        
        _ = attach_release_attitude_strip_to_evidence(evidence, strip)
        
        assert evidence == original_evidence


class TestExtractReleaseAttitudeStripSignal:
    """Tests for extract_release_attitude_strip_signal()."""
    
    def test_extracts_signal_fields(self):
        """Test that signal includes all required fields."""
        strip = {
            "schema_version": "1.0.0",
            "experiments": [
                {"cal_id": "CAL-EXP-1", "status_light": "🟢"},
                {"cal_id": "CAL-EXP-2", "status_light": "🟡"},
            ],
            "summary": {
                "total_count": 2,
                "release_ready_count": 1,
                "release_ready_ratio": 0.5,
                "trend": "DEGRADING",
            },
        }
        signal = extract_release_attitude_strip_signal(strip)
        
        assert "total_count" in signal
        assert "release_ready_ratio" in signal
        assert "trend" in signal
        assert "first_status_light" in signal
        assert "last_status_light" in signal
        assert "advisory_warning" in signal
        
        assert signal["total_count"] == 2
        assert signal["release_ready_ratio"] == 0.5
        assert signal["trend"] == "DEGRADING"
        assert signal["first_status_light"] == "🟢"
        assert signal["last_status_light"] == "🟡"
    
    def test_advisory_warning_on_degrading(self):
        """Test that advisory warning is set when trend is DEGRADING."""
        strip = {
            "schema_version": "1.0.0",
            "experiments": [
                {"cal_id": "CAL-EXP-1", "status_light": "🟢"},
                {"cal_id": "CAL-EXP-2", "status_light": "🔴"},
            ],
            "summary": {
                "total_count": 2,
                "release_ready_count": 1,
                "release_ready_ratio": 0.5,
                "trend": "DEGRADING",
            },
        }
        signal = extract_release_attitude_strip_signal(strip)
        
        assert signal["advisory_warning"] is not None
        assert "DEGRADING" in signal["advisory_warning"]
    
    def test_no_advisory_warning_on_improving(self):
        """Test that advisory warning is None when trend is IMPROVING."""
        strip = {
            "schema_version": "1.0.0",
            "experiments": [
                {"cal_id": "CAL-EXP-1", "status_light": "🔴"},
                {"cal_id": "CAL-EXP-2", "status_light": "🟢"},
            ],
            "summary": {
                "total_count": 2,
                "release_ready_count": 1,
                "release_ready_ratio": 0.5,
                "trend": "IMPROVING",
            },
        }
        signal = extract_release_attitude_strip_signal(strip)
        
        assert signal["advisory_warning"] is None
    
    def test_no_advisory_warning_on_stable(self):
        """Test that advisory warning is None when trend is STABLE."""
        strip = {
            "schema_version": "1.0.0",
            "experiments": [
                {"cal_id": "CAL-EXP-1", "status_light": "🟢"},
                {"cal_id": "CAL-EXP-2", "status_light": "🟢"},
            ],
            "summary": {
                "total_count": 2,
                "release_ready_count": 2,
                "release_ready_ratio": 1.0,
                "trend": "STABLE",
            },
        }
        signal = extract_release_attitude_strip_signal(strip)
        
        assert signal["advisory_warning"] is None
    
    def test_handles_empty_experiments(self):
        """Test that signal handles empty experiments list."""
        strip = {
            "schema_version": "1.0.0",
            "experiments": [],
            "summary": {
                "total_count": 0,
                "release_ready_count": 0,
                "release_ready_ratio": 0.0,
                "trend": "STABLE",
            },
        }
        signal = extract_release_attitude_strip_signal(strip)
        
        assert signal["total_count"] == 0
        assert signal["release_ready_ratio"] == 0.0
        assert signal["trend"] == "STABLE"
        assert signal["first_status_light"] == "🟢"  # Default
        assert signal["last_status_light"] == "🟢"  # Default
    
    def test_json_serializable(self):
        """Test that signal is JSON serializable."""
        strip = {
            "schema_version": "1.0.0",
            "experiments": [
                {"cal_id": "CAL-EXP-1", "status_light": "🟢"},
            ],
            "summary": {
                "total_count": 1,
                "release_ready_count": 1,
                "release_ready_ratio": 1.0,
                "trend": "STABLE",
            },
        }
        signal = extract_release_attitude_strip_signal(strip)
        
        # Should not raise
        json_str = json.dumps(signal, ensure_ascii=False)
        assert isinstance(json_str, str)
        assert len(json_str) > 0
    
    def test_signal_attached_to_evidence(self):
        """Test that signal is attached to evidence when using attach function."""
        evidence = {}
        strip = {
            "schema_version": "1.0.0",
            "experiments": [
                {"cal_id": "CAL-EXP-1", "status_light": "🟢"},
            ],
            "summary": {
                "total_count": 1,
                "release_ready_count": 1,
                "release_ready_ratio": 1.0,
                "trend": "STABLE",
            },
        }
        updated = attach_release_attitude_strip_to_evidence(evidence, strip)
        
        assert "signals" in updated
        assert "release_attitude_strip" in updated["signals"]
        signal = updated["signals"]["release_attitude_strip"]
        assert signal["total_count"] == 1
        assert signal["trend"] == "STABLE"

