"""Tests for coherence P3/P4 integration.

Tests verify:
- P3 stability report integration
- P4 calibration report integration
- Evidence pack integration
- Council normalization
- JSON serialization
- Determinism
- Non-mutation
- No prescriptive language
"""

import json
import pytest
from typing import Any, Dict

from backend.health.coherence_p3p4_integration import (
    attach_coherence_to_p3_stability_report,
    attach_coherence_to_p4_calibration_report,
    attach_coherence_to_evidence,
    summarize_coherence_for_uplift_council,
)
from backend.health.coherence_adapter import extract_coherence_drift_signal


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_coherence_tile() -> Dict[str, Any]:
    """Sample coherence governance tile."""
    return {
        "schema_version": "1.0.0",
        "status_light": "YELLOW",
        "coherence_band": "PARTIAL",
        "global_coherence_index": 0.583,
        "slices_at_risk": ["slice3"],
        "drivers": [
            "2 slices with coherence below 0.45 threshold.",
            "1 slice with MAJOR drift severity.",
        ],
        "horizon_estimate": 5,
        "headline": "Coherence status: YELLOW (PARTIAL).",
    }


@pytest.fixture
def sample_drift_signal() -> Dict[str, Any]:
    """Sample coherence drift signal."""
    return {
        "coherence_band": "PARTIAL",
        "low_slices": ["slice3"],
        "global_index": 0.583,
    }


@pytest.fixture
def sample_p3_stability_report() -> Dict[str, Any]:
    """Sample P3 stability report."""
    return {
        "schema_version": "1.0.0",
        "run_id": "p3_run_001",
        "cycles_completed": 1000,
        "u2_success_rate_final": 0.85,
        "mean_rsi": 0.75,
    }


@pytest.fixture
def sample_p4_calibration_report() -> Dict[str, Any]:
    """Sample P4 calibration report."""
    return {
        "schema_version": "1.0.0",
        "run_id": "p4_run_001",
        "cycles_observed": 1000,
        "divergence_statistics": {
            "mean_divergence": 0.05,
        },
    }


@pytest.fixture
def sample_evidence() -> Dict[str, Any]:
    """Sample evidence dictionary."""
    return {
        "evidence_id": "ev_001",
        "timestamp": "2025-01-01T00:00:00Z",
        "governance": {},
    }


# =============================================================================
# TASK 1: P3 STABILITY REPORT INTEGRATION
# =============================================================================

class TestAttachCoherenceToP3StabilityReport:
    """Tests for P3 stability report integration."""
    
    def test_attaches_coherence_summary(
        self, sample_p3_stability_report, sample_coherence_tile
    ):
        """Should attach coherence_summary to stability report."""
        updated = attach_coherence_to_p3_stability_report(
            sample_p3_stability_report, sample_coherence_tile
        )
        
        assert "coherence_summary" in updated
        summary = updated["coherence_summary"]
        
        assert summary["coherence_band"] == "PARTIAL"
        assert summary["global_coherence_index"] == 0.583
        assert summary["slices_at_risk"] == ["slice3"]
        assert len(summary["root_incoherence_causes"]) > 0
    
    def test_non_mutating(self, sample_p3_stability_report, sample_coherence_tile):
        """Should not modify input dictionaries."""
        original = dict(sample_p3_stability_report)
        
        attach_coherence_to_p3_stability_report(
            sample_p3_stability_report, sample_coherence_tile
        )
        
        assert sample_p3_stability_report == original
        assert "coherence_summary" not in sample_p3_stability_report
    
    def test_json_serializable(
        self, sample_p3_stability_report, sample_coherence_tile
    ):
        """Result should be JSON serializable."""
        updated = attach_coherence_to_p3_stability_report(
            sample_p3_stability_report, sample_coherence_tile
        )
        
        json_str = json.dumps(updated)
        parsed = json.loads(json_str)
        
        assert parsed["coherence_summary"] == updated["coherence_summary"]
    
    def test_deterministic(self, sample_p3_stability_report, sample_coherence_tile):
        """Result should be deterministic."""
        result1 = attach_coherence_to_p3_stability_report(
            sample_p3_stability_report, sample_coherence_tile
        )
        result2 = attach_coherence_to_p3_stability_report(
            sample_p3_stability_report, sample_coherence_tile
        )
        
        assert result1 == result2


# =============================================================================
# TASK 2: P4 CALIBRATION REPORT INTEGRATION
# =============================================================================

class TestAttachCoherenceToP4CalibrationReport:
    """Tests for P4 calibration report integration."""
    
    def test_attaches_coherence_calibration(
        self, sample_p4_calibration_report, sample_drift_signal
    ):
        """Should attach coherence_calibration to calibration report."""
        updated = attach_coherence_to_p4_calibration_report(
            sample_p4_calibration_report, sample_drift_signal
        )
        
        assert "coherence_calibration" in updated
        calibration = updated["coherence_calibration"]
        
        assert calibration["global_coherence_index"] == 0.583
        assert calibration["coherence_band"] == "PARTIAL"
        assert calibration["low_slices"] == ["slice3"]
        assert len(calibration["structural_notes"]) > 0
    
    def test_non_mutating(self, sample_p4_calibration_report, sample_drift_signal):
        """Should not modify input dictionaries."""
        original = dict(sample_p4_calibration_report)
        
        attach_coherence_to_p4_calibration_report(
            sample_p4_calibration_report, sample_drift_signal
        )
        
        assert sample_p4_calibration_report == original
        assert "coherence_calibration" not in sample_p4_calibration_report
    
    def test_json_serializable(
        self, sample_p4_calibration_report, sample_drift_signal
    ):
        """Result should be JSON serializable."""
        updated = attach_coherence_to_p4_calibration_report(
            sample_p4_calibration_report, sample_drift_signal
        )
        
        json_str = json.dumps(updated)
        parsed = json.loads(json_str)
        
        assert parsed["coherence_calibration"] == updated["coherence_calibration"]
    
    def test_deterministic(self, sample_p4_calibration_report, sample_drift_signal):
        """Result should be deterministic."""
        result1 = attach_coherence_to_p4_calibration_report(
            sample_p4_calibration_report, sample_drift_signal
        )
        result2 = attach_coherence_to_p4_calibration_report(
            sample_p4_calibration_report, sample_drift_signal
        )
        
        assert result1 == result2


# =============================================================================
# TASK 3: EVIDENCE HELPER
# =============================================================================

class TestAttachCoherenceToEvidence:
    """Tests for evidence integration."""
    
    def test_attaches_coherence_to_governance(
        self, sample_evidence, sample_coherence_tile, sample_drift_signal
    ):
        """Should attach coherence under evidence['governance']['coherence']."""
        updated = attach_coherence_to_evidence(
            sample_evidence, sample_coherence_tile, sample_drift_signal
        )
        
        assert "governance" in updated
        assert "coherence" in updated["governance"]
        
        coherence = updated["governance"]["coherence"]
        assert coherence["band"] == "PARTIAL"
        assert coherence["global_index"] == 0.583
        assert coherence["slices_at_risk"] == ["slice3"]
        assert "drift_annotations" in coherence
    
    def test_attaches_first_light_summary(
        self, sample_evidence, sample_coherence_tile, sample_drift_signal
    ):
        """Should attach first_light_summary to coherence evidence."""
        updated = attach_coherence_to_evidence(
            sample_evidence, sample_coherence_tile, sample_drift_signal
        )
        
        coherence = updated["governance"]["coherence"]
        assert "first_light_summary" in coherence
        
        summary = coherence["first_light_summary"]
        assert summary["coherence_band"] == "PARTIAL"
        assert summary["global_index"] == 0.583
        assert summary["slices_at_risk"] == ["slice3"]
    
    def test_first_light_summary_compact(self, sample_evidence, sample_coherence_tile):
        """First Light summary should be compact (only essential fields)."""
        updated = attach_coherence_to_evidence(
            sample_evidence, sample_coherence_tile
        )
        
        summary = updated["governance"]["coherence"]["first_light_summary"]
        
        # Should only have these three fields
        assert set(summary.keys()) == {"coherence_band", "global_index", "slices_at_risk"}
    
    def test_first_light_summary_present_without_drift_signal(
        self, sample_evidence, sample_coherence_tile
    ):
        """First Light summary should be present even without drift_signal."""
        updated = attach_coherence_to_evidence(
            sample_evidence, sample_coherence_tile
        )
        
        assert "first_light_summary" in updated["governance"]["coherence"]
        summary = updated["governance"]["coherence"]["first_light_summary"]
        assert summary["coherence_band"] == "PARTIAL"
        assert summary["global_index"] == 0.583
    
    def test_first_light_summary_json_safe(
        self, sample_evidence, sample_coherence_tile
    ):
        """First Light summary should be JSON serializable."""
        updated = attach_coherence_to_evidence(
            sample_evidence, sample_coherence_tile
        )
        
        summary = updated["governance"]["coherence"]["first_light_summary"]
        
        json_str = json.dumps(summary)
        parsed = json.loads(json_str)
        
        assert parsed == summary
    
    def test_first_light_summary_deterministic(
        self, sample_evidence, sample_coherence_tile
    ):
        """First Light summary should be deterministic."""
        updated1 = attach_coherence_to_evidence(
            sample_evidence, sample_coherence_tile
        )
        updated2 = attach_coherence_to_evidence(
            sample_evidence, sample_coherence_tile
        )
        
        summary1 = updated1["governance"]["coherence"]["first_light_summary"]
        summary2 = updated2["governance"]["coherence"]["first_light_summary"]
        
        assert summary1 == summary2
    
    def test_creates_governance_if_missing(
        self, sample_coherence_tile, sample_drift_signal
    ):
        """Should create governance structure if missing."""
        evidence = {"evidence_id": "ev_001"}
        
        updated = attach_coherence_to_evidence(
            evidence, sample_coherence_tile, sample_drift_signal
        )
        
        assert "governance" in updated
        assert "coherence" in updated["governance"]
    
    def test_works_without_drift_signal(
        self, sample_evidence, sample_coherence_tile
    ):
        """Should work without drift_signal."""
        updated = attach_coherence_to_evidence(
            sample_evidence, sample_coherence_tile
        )
        
        coherence = updated["governance"]["coherence"]
        assert "band" in coherence
        assert "drift_annotations" not in coherence
    
    def test_non_mutating(
        self, sample_evidence, sample_coherence_tile, sample_drift_signal
    ):
        """Should not modify input dictionaries."""
        original = dict(sample_evidence)
        
        attach_coherence_to_evidence(
            sample_evidence, sample_coherence_tile, sample_drift_signal
        )
        
        assert sample_evidence == original
    
    def test_json_serializable(
        self, sample_evidence, sample_coherence_tile, sample_drift_signal
    ):
        """Result should be JSON serializable."""
        updated = attach_coherence_to_evidence(
            sample_evidence, sample_coherence_tile, sample_drift_signal
        )
        
        json_str = json.dumps(updated)
        parsed = json.loads(json_str)
        
        assert parsed["governance"]["coherence"] == updated["governance"]["coherence"]
    
    def test_deterministic(
        self, sample_evidence, sample_coherence_tile, sample_drift_signal
    ):
        """Result should be deterministic."""
        result1 = attach_coherence_to_evidence(
            sample_evidence, sample_coherence_tile, sample_drift_signal
        )
        result2 = attach_coherence_to_evidence(
            sample_evidence, sample_coherence_tile, sample_drift_signal
        )
        
        assert result1 == result2


# =============================================================================
# TASK 4: COUNCIL NORMALIZATION
# =============================================================================

class TestSummarizeCoherenceForUpliftCouncil:
    """Tests for council normalization."""
    
    def test_misaligned_maps_to_block(self, sample_coherence_tile):
        """MISALIGNED should map to BLOCK."""
        tile = {**sample_coherence_tile, "coherence_band": "MISALIGNED"}
        
        summary = summarize_coherence_for_uplift_council(tile)
        
        assert summary["status"] == "BLOCK"
        assert summary["coherence_band"] == "MISALIGNED"
    
    def test_partial_maps_to_warn(self, sample_coherence_tile):
        """PARTIAL should map to WARN."""
        summary = summarize_coherence_for_uplift_council(sample_coherence_tile)
        
        assert summary["status"] == "WARN"
        assert summary["coherence_band"] == "PARTIAL"
    
    def test_coherent_maps_to_ok(self, sample_coherence_tile):
        """COHERENT should map to OK."""
        tile = {**sample_coherence_tile, "coherence_band": "COHERENT"}
        
        summary = summarize_coherence_for_uplift_council(tile)
        
        assert summary["status"] == "OK"
        assert summary["coherence_band"] == "COHERENT"
    
    def test_council_mapping_all_bands(self):
        """Verify council mapping for all coherence bands."""
        # MISALIGNED → BLOCK
        tile_misaligned = {
            "coherence_band": "MISALIGNED",
            "slices_at_risk": ["slice1"],
            "headline": "Test",
        }
        summary = summarize_coherence_for_uplift_council(tile_misaligned)
        assert summary["status"] == "BLOCK"
        
        # PARTIAL → WARN
        tile_partial = {
            "coherence_band": "PARTIAL",
            "slices_at_risk": ["slice1"],
            "headline": "Test",
        }
        summary = summarize_coherence_for_uplift_council(tile_partial)
        assert summary["status"] == "WARN"
        
        # COHERENT → OK
        tile_coherent = {
            "coherence_band": "COHERENT",
            "slices_at_risk": [],
            "headline": "Test",
        }
        summary = summarize_coherence_for_uplift_council(tile_coherent)
        assert summary["status"] == "OK"
    
    def test_includes_slices_at_risk(self, sample_coherence_tile):
        """Should include slices_at_risk."""
        summary = summarize_coherence_for_uplift_council(sample_coherence_tile)
        
        assert "slices_at_risk" in summary
        assert summary["slices_at_risk"] == ["slice3"]
    
    def test_includes_headline(self, sample_coherence_tile):
        """Should include headline from tile."""
        summary = summarize_coherence_for_uplift_council(sample_coherence_tile)
        
        assert "headline" in summary
        assert summary["headline"] == sample_coherence_tile.get("headline", "")
    
    def test_headline_no_forbidden_words(self, sample_coherence_tile):
        """Headline should not contain forbidden words."""
        FORBIDDEN = {
            "fix", "change", "modify", "improve", "correct",
            "better", "worse", "should", "must", "need to",
        }
        
        summary = summarize_coherence_for_uplift_council(sample_coherence_tile)
        
        headline_lower = summary["headline"].lower()
        for word in FORBIDDEN:
            assert word not in headline_lower, f"Found forbidden word '{word}' in headline '{summary['headline']}'"
    
    def test_json_serializable(self, sample_coherence_tile):
        """Result should be JSON serializable."""
        summary = summarize_coherence_for_uplift_council(sample_coherence_tile)
        
        json_str = json.dumps(summary)
        parsed = json.loads(json_str)
        
        assert parsed == summary
    
    def test_deterministic(self, sample_coherence_tile):
        """Result should be deterministic."""
        result1 = summarize_coherence_for_uplift_council(sample_coherence_tile)
        result2 = summarize_coherence_for_uplift_council(sample_coherence_tile)
        
        assert result1 == result2


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests."""
    
    def test_full_workflow(
        self,
        sample_p3_stability_report,
        sample_p4_calibration_report,
        sample_evidence,
        sample_coherence_tile,
        sample_drift_signal,
    ):
        """Test full workflow from coherence tile to reports and evidence."""
        # P3 integration
        p3_updated = attach_coherence_to_p3_stability_report(
            sample_p3_stability_report, sample_coherence_tile
        )
        assert "coherence_summary" in p3_updated
        
        # P4 integration
        p4_updated = attach_coherence_to_p4_calibration_report(
            sample_p4_calibration_report, sample_drift_signal
        )
        assert "coherence_calibration" in p4_updated
        
        # Evidence integration
        evidence_updated = attach_coherence_to_evidence(
            sample_evidence, sample_coherence_tile, sample_drift_signal
        )
        assert "governance" in evidence_updated
        assert "coherence" in evidence_updated["governance"]
        
        # Council summary
        council_summary = summarize_coherence_for_uplift_council(sample_coherence_tile)
        assert council_summary["status"] in ("OK", "WARN", "BLOCK")

