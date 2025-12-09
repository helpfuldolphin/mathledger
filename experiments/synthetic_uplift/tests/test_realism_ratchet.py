#!/usr/bin/env python3
"""
==============================================================================
PHASE II — SYNTHETIC TEST DATA ONLY
==============================================================================

Test Suite for Synthetic Realism Ratchet & Scenario Calibration Console
-------------------------------------------------------------------------

Tests for:
    - Realism Ratchet (Task 1)
    - Scenario Calibration Console (Task 2)
    - Stability classes
    - Calibration status transitions
    - Deterministic ratchet output

NOT derived from real derivations; NOT part of Evidence Pack.

==============================================================================
"""

import pytest
import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.synthetic_uplift.noise_models import SAFETY_LABEL
from experiments.synthetic_uplift.realism_ratchet import (
    SCHEMA_VERSION,
    StabilityClass,
    CalibrationStatus,
    build_synthetic_realism_ratchet,
    build_scenario_calibration_console,
    build_complete_calibration_view,
    format_calibration_console,
)


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def mock_consistency_view():
    """Mock consistency view."""
    return {
        "consistency_status": "ALIGNED",
        "scenarios_consistent_with_real": ["synthetic_consistent"],
        "scenarios_more_aggressive_than_real": ["synthetic_aggressive"],
        "scenarios_under_exploring": ["synthetic_under"],
    }


@pytest.fixture
def mock_realism_summary():
    """Mock realism summary."""
    return {
        "realism_ok": True,
        "status": "OK",
        "envelope_breach_rate": 0.1,
        "scenarios_needing_review": [],
    }


@pytest.fixture
def mock_scenario_policy():
    """Mock scenario policy."""
    return {
        "status": "OK",
        "scenarios_needing_tuning": [],
        "scenarios_recommended_for_rfl_experiments": ["synthetic_consistent"],
    }


# ==============================================================================
# TASK 1: REALISM RATCHET TESTS
# ==============================================================================

class TestRealismRatchet:
    """Tests for build_synthetic_realism_ratchet."""
    
    def test_ratchet_has_required_fields(
        self,
        mock_consistency_view,
        mock_realism_summary,
    ):
        """Ratchet should have all required fields."""
        ratchet = build_synthetic_realism_ratchet(
            mock_consistency_view,
            mock_realism_summary,
        )
        
        assert "realism_pressure" in ratchet
        assert "scenario_retention_score" in ratchet
        assert "stability_class" in ratchet
        assert "global_realism_pressure" in ratchet
    
    def test_ratchet_schema_version(
        self,
        mock_consistency_view,
        mock_realism_summary,
    ):
        """Ratchet should have correct schema version."""
        ratchet = build_synthetic_realism_ratchet(
            mock_consistency_view,
            mock_realism_summary,
        )
        
        assert ratchet["schema_version"] == SCHEMA_VERSION
    
    def test_realism_pressure_in_range(
        self,
        mock_consistency_view,
        mock_realism_summary,
    ):
        """Realism pressure should be in [0, 1]."""
        ratchet = build_synthetic_realism_ratchet(
            mock_consistency_view,
            mock_realism_summary,
        )
        
        pressure = ratchet["realism_pressure"]
        assert 0.0 <= pressure <= 1.0
    
    def test_realism_pressure_increases_with_breach_rate(self):
        """Pressure should increase with higher breach rates."""
        view1 = {
            "consistency_status": "ALIGNED",
            "scenarios_consistent_with_real": ["s1"],
            "scenarios_more_aggressive_than_real": [],
            "scenarios_under_exploring": [],
        }
        
        summary1 = {
            "envelope_breach_rate": 0.1,
            "scenarios_needing_review": [],
        }
        
        summary2 = {
            "envelope_breach_rate": 0.5,
            "scenarios_needing_review": [],
        }
        
        ratchet1 = build_synthetic_realism_ratchet(view1, summary1)
        ratchet2 = build_synthetic_realism_ratchet(view1, summary2)
        
        assert ratchet2["realism_pressure"] > ratchet1["realism_pressure"]
    
    def test_realism_pressure_increases_with_misalignment(self):
        """Pressure should increase with consistency misalignment."""
        summary = {
            "envelope_breach_rate": 0.1,
            "scenarios_needing_review": [],
        }
        
        view_aligned = {
            "consistency_status": "ALIGNED",
            "scenarios_consistent_with_real": ["s1"],
            "scenarios_more_aggressive_than_real": [],
            "scenarios_under_exploring": [],
        }
        
        view_misaligned = {
            "consistency_status": "MISALIGNED",
            "scenarios_consistent_with_real": [],
            "scenarios_more_aggressive_than_real": ["s1"],
            "scenarios_under_exploring": [],
        }
        
        ratchet1 = build_synthetic_realism_ratchet(view_aligned, summary)
        ratchet2 = build_synthetic_realism_ratchet(view_misaligned, summary)
        
        assert ratchet2["realism_pressure"] > ratchet1["realism_pressure"]
    
    def test_retention_scores_in_range(
        self,
        mock_consistency_view,
        mock_realism_summary,
    ):
        """Retention scores should be in [0, 1]."""
        ratchet = build_synthetic_realism_ratchet(
            mock_consistency_view,
            mock_realism_summary,
        )
        
        scores = ratchet["scenario_retention_score"]
        for score in scores.values():
            assert 0.0 <= score <= 1.0
    
    def test_retention_score_lower_for_aggressive(
        self,
        mock_consistency_view,
        mock_realism_summary,
    ):
        """Aggressive scenarios should have lower retention scores."""
        ratchet = build_synthetic_realism_ratchet(
            mock_consistency_view,
            mock_realism_summary,
        )
        
        scores = ratchet["scenario_retention_score"]
        
        if "synthetic_aggressive" in scores and "synthetic_consistent" in scores:
            assert scores["synthetic_aggressive"] < scores["synthetic_consistent"]
    
    def test_stability_class_stable_for_consistent(
        self,
        mock_consistency_view,
        mock_realism_summary,
    ):
        """Consistent scenarios should be STABLE."""
        ratchet = build_synthetic_realism_ratchet(
            mock_consistency_view,
            mock_realism_summary,
        )
        
        stability = ratchet["stability_class"]
        
        if "synthetic_consistent" in stability:
            assert stability["synthetic_consistent"] == StabilityClass.STABLE.value
    
    def test_stability_class_soft_drift_for_under_exploring(
        self,
        mock_consistency_view,
        mock_realism_summary,
    ):
        """Under-exploring scenarios should be SOFT_DRIFT."""
        ratchet = build_synthetic_realism_ratchet(
            mock_consistency_view,
            mock_realism_summary,
        )
        
        stability = ratchet["stability_class"]
        
        if "synthetic_under" in stability:
            assert stability["synthetic_under"] == StabilityClass.SOFT_DRIFT.value
    
    def test_stability_class_sharp_drift_for_aggressive_with_review(self):
        """Aggressive scenarios needing review should be SHARP_DRIFT."""
        view = {
            "consistency_status": "PARTIAL",
            "scenarios_consistent_with_real": [],
            "scenarios_more_aggressive_than_real": ["synthetic_aggressive"],
            "scenarios_under_exploring": [],
        }
        
        summary = {
            "envelope_breach_rate": 0.2,
            "scenarios_needing_review": ["synthetic_aggressive"],
        }
        
        ratchet = build_synthetic_realism_ratchet(view, summary)
        
        stability = ratchet["stability_class"]
        
        if "synthetic_aggressive" in stability:
            assert stability["synthetic_aggressive"] == StabilityClass.SHARP_DRIFT.value
    
    def test_ratchet_deterministic(
        self,
        mock_consistency_view,
        mock_realism_summary,
    ):
        """Ratchet output should be deterministic for same inputs."""
        ratchet1 = build_synthetic_realism_ratchet(
            mock_consistency_view,
            mock_realism_summary,
        )
        
        ratchet2 = build_synthetic_realism_ratchet(
            mock_consistency_view,
            mock_realism_summary,
        )
        
        # Compare key fields (excluding timestamp)
        assert ratchet1["realism_pressure"] == ratchet2["realism_pressure"]
        assert ratchet1["scenario_retention_score"] == ratchet2["scenario_retention_score"]
        assert ratchet1["stability_class"] == ratchet2["stability_class"]
    
    def test_ratchet_includes_safety_label(
        self,
        mock_consistency_view,
        mock_realism_summary,
    ):
        """Ratchet should include safety label."""
        ratchet = build_synthetic_realism_ratchet(
            mock_consistency_view,
            mock_realism_summary,
        )
        
        assert ratchet["label"] == SAFETY_LABEL


# ==============================================================================
# TASK 2: CALIBRATION CONSOLE TESTS
# ==============================================================================

class TestCalibrationConsole:
    """Tests for build_scenario_calibration_console."""
    
    def test_console_has_required_fields(
        self,
        mock_consistency_view,
        mock_realism_summary,
        mock_scenario_policy,
    ):
        """Console should have all required fields."""
        ratchet = build_synthetic_realism_ratchet(
            mock_consistency_view,
            mock_realism_summary,
        )
        
        console = build_scenario_calibration_console(ratchet, mock_scenario_policy)
        
        assert "calibration_status" in console
        assert "slices_to_recalibrate" in console
        assert "advisory_notes" in console
    
    def test_calibration_status_ok(self):
        """Should return OK when conditions are good."""
        ratchet = {
            "realism_pressure": 0.1,
            "scenario_retention_score": {"s1": 0.9},
            "stability_class": {"s1": StabilityClass.STABLE.value},
        }
        
        policy = {
            "status": "OK",
            "scenarios_needing_tuning": [],
        }
        
        console = build_scenario_calibration_console(ratchet, policy)
        
        assert console["calibration_status"] == CalibrationStatus.OK.value
    
    def test_calibration_status_attention(self):
        """Should return ATTENTION for moderate issues."""
        ratchet = {
            "realism_pressure": 0.4,
            "scenario_retention_score": {"s1": 0.6},
            "stability_class": {"s1": StabilityClass.SOFT_DRIFT.value},
        }
        
        policy = {
            "status": "ATTENTION",
            "scenarios_needing_tuning": ["s1"],
        }
        
        console = build_scenario_calibration_console(ratchet, policy)
        
        assert console["calibration_status"] == CalibrationStatus.ATTENTION.value
    
    def test_calibration_status_block_high_pressure(self):
        """Should return BLOCK for high realism pressure."""
        ratchet = {
            "realism_pressure": 0.8,
            "scenario_retention_score": {"s1": 0.3},
            "stability_class": {"s1": StabilityClass.SHARP_DRIFT.value},
        }
        
        policy = {
            "status": "OK",
            "scenarios_needing_tuning": [],
        }
        
        console = build_scenario_calibration_console(ratchet, policy)
        
        assert console["calibration_status"] == CalibrationStatus.BLOCK.value
    
    def test_calibration_status_block_policy_block(self):
        """Should return BLOCK when policy status is BLOCK."""
        ratchet = {
            "realism_pressure": 0.2,
            "scenario_retention_score": {"s1": 0.7},
            "stability_class": {"s1": StabilityClass.STABLE.value},
        }
        
        policy = {
            "status": "BLOCK",
            "scenarios_needing_tuning": ["s1"],
        }
        
        console = build_scenario_calibration_console(ratchet, policy)
        
        assert console["calibration_status"] == CalibrationStatus.BLOCK.value
    
    def test_calibration_status_block_sharp_drift(self):
        """Should return BLOCK when sharp drift is present."""
        ratchet = {
            "realism_pressure": 0.2,
            "scenario_retention_score": {"s1": 0.4},
            "stability_class": {"s1": StabilityClass.SHARP_DRIFT.value},
        }
        
        policy = {
            "status": "OK",
            "scenarios_needing_tuning": [],
        }
        
        console = build_scenario_calibration_console(ratchet, policy)
        
        assert console["calibration_status"] == CalibrationStatus.BLOCK.value
    
    def test_slices_to_recalibrate_includes_low_retention(self):
        """Should include scenarios with low retention scores."""
        ratchet = {
            "realism_pressure": 0.3,
            "scenario_retention_score": {
                "s1": 0.4,  # Low retention
                "s2": 0.8,  # High retention
            },
            "stability_class": {
                "s1": StabilityClass.STABLE.value,
                "s2": StabilityClass.STABLE.value,
            },
        }
        
        policy = {
            "status": "OK",
            "scenarios_needing_tuning": [],
        }
        
        console = build_scenario_calibration_console(ratchet, policy)
        
        assert "s1" in console["slices_to_recalibrate"]
        assert "s2" not in console["slices_to_recalibrate"]
    
    def test_slices_to_recalibrate_includes_sharp_drift(self):
        """Should include scenarios with SHARP_DRIFT stability."""
        ratchet = {
            "realism_pressure": 0.3,
            "scenario_retention_score": {
                "s1": 0.6,  # Not low enough alone
            },
            "stability_class": {
                "s1": StabilityClass.SHARP_DRIFT.value,
            },
        }
        
        policy = {
            "status": "OK",
            "scenarios_needing_tuning": [],
        }
        
        console = build_scenario_calibration_console(ratchet, policy)
        
        assert "s1" in console["slices_to_recalibrate"]
    
    def test_slices_to_recalibrate_includes_policy_tuning(self):
        """Should include scenarios from policy needing tuning."""
        ratchet = {
            "realism_pressure": 0.2,
            "scenario_retention_score": {
                "s1": 0.7,
            },
            "stability_class": {
                "s1": StabilityClass.STABLE.value,
            },
        }
        
        policy = {
            "status": "OK",
            "scenarios_needing_tuning": ["s1"],
        }
        
        console = build_scenario_calibration_console(ratchet, policy)
        
        assert "s1" in console["slices_to_recalibrate"]
    
    def test_advisory_notes_present(self):
        """Console should include advisory notes."""
        ratchet = {
            "realism_pressure": 0.1,
            "scenario_retention_score": {},
            "stability_class": {},
        }
        
        policy = {
            "status": "OK",
            "scenarios_needing_tuning": [],
        }
        
        console = build_scenario_calibration_console(ratchet, policy)
        
        assert "advisory_notes" in console
        assert len(console["advisory_notes"]) > 0
        # Should not contain uplift claims
        assert "uplift" not in console["advisory_notes"].lower()
    
    def test_stability_summary_present(self):
        """Console should include stability summary."""
        ratchet = {
            "realism_pressure": 0.2,
            "scenario_retention_score": {
                "s1": 0.8,
                "s2": 0.6,
                "s3": 0.4,
            },
            "stability_class": {
                "s1": StabilityClass.STABLE.value,
                "s2": StabilityClass.SOFT_DRIFT.value,
                "s3": StabilityClass.SHARP_DRIFT.value,
            },
        }
        
        policy = {
            "status": "OK",
            "scenarios_needing_tuning": [],
        }
        
        console = build_scenario_calibration_console(ratchet, policy)
        
        summary = console.get("stability_summary", {})
        assert summary.get("stable_count", 0) == 1
        assert summary.get("soft_drift_count", 0) == 1
        assert summary.get("sharp_drift_count", 0) == 1
    
    def test_console_includes_safety_label(
        self,
        mock_consistency_view,
        mock_realism_summary,
        mock_scenario_policy,
    ):
        """Console should include safety label."""
        ratchet = build_synthetic_realism_ratchet(
            mock_consistency_view,
            mock_realism_summary,
        )
        
        console = build_scenario_calibration_console(ratchet, mock_scenario_policy)
        
        assert console["label"] == SAFETY_LABEL


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================

class TestIntegration:
    """Integration tests for complete calibration view."""
    
    def test_build_complete_calibration_view(
        self,
        mock_consistency_view,
        mock_realism_summary,
        mock_scenario_policy,
    ):
        """Should build complete calibration view."""
        view = build_complete_calibration_view(
            consistency_view=mock_consistency_view,
            realism_summary=mock_realism_summary,
            scenario_policy=mock_scenario_policy,
        )
        
        assert "ratchet" in view
        assert "calibration_console" in view
    
    def test_format_calibration_console(self):
        """Should format console as readable text."""
        console = {
            "calibration_status": "OK",
            "realism_pressure": 0.1,
            "slices_to_recalibrate": [],
            "advisory_notes": "Test notes.",
            "stability_summary": {
                "stable_count": 1,
                "soft_drift_count": 0,
                "sharp_drift_count": 0,
            },
        }
        
        formatted = format_calibration_console(console)
        
        assert SAFETY_LABEL in formatted
        assert "SCENARIO CALIBRATION CONSOLE" in formatted
        assert "OK" in formatted
    
    def test_complete_view_includes_safety_label(
        self,
        mock_consistency_view,
        mock_realism_summary,
        mock_scenario_policy,
    ):
        """Complete view should include safety label."""
        view = build_complete_calibration_view(
            consistency_view=mock_consistency_view,
            realism_summary=mock_realism_summary,
            scenario_policy=mock_scenario_policy,
        )
        
        assert view["label"] == SAFETY_LABEL


# ==============================================================================
# STABILITY CLASS TESTS
# ==============================================================================

class TestStabilityClasses:
    """Tests for stability class determination."""
    
    def test_stable_classification(self):
        """Should classify stable scenarios correctly."""
        view = {
            "consistency_status": "ALIGNED",
            "scenarios_consistent_with_real": ["s1"],
            "scenarios_more_aggressive_than_real": [],
            "scenarios_under_exploring": [],
        }
        
        summary = {
            "envelope_breach_rate": 0.05,
            "scenarios_needing_review": [],
        }
        
        ratchet = build_synthetic_realism_ratchet(view, summary)
        
        if "s1" in ratchet["stability_class"]:
            assert ratchet["stability_class"]["s1"] == StabilityClass.STABLE.value
    
    def test_soft_drift_classification(self):
        """Should classify soft drift scenarios correctly."""
        view = {
            "consistency_status": "PARTIAL",
            "scenarios_consistent_with_real": [],
            "scenarios_more_aggressive_than_real": [],
            "scenarios_under_exploring": ["s1"],
        }
        
        summary = {
            "envelope_breach_rate": 0.1,
            "scenarios_needing_review": [],
        }
        
        ratchet = build_synthetic_realism_ratchet(view, summary)
        
        if "s1" in ratchet["stability_class"]:
            assert ratchet["stability_class"]["s1"] == StabilityClass.SOFT_DRIFT.value
    
    def test_sharp_drift_classification(self):
        """Should classify sharp drift scenarios correctly."""
        view = {
            "consistency_status": "PARTIAL",
            "scenarios_consistent_with_real": [],
            "scenarios_more_aggressive_than_real": ["s1"],
            "scenarios_under_exploring": [],
        }
        
        summary = {
            "envelope_breach_rate": 0.2,
            "scenarios_needing_review": ["s1"],
        }
        
        ratchet = build_synthetic_realism_ratchet(view, summary)
        
        if "s1" in ratchet["stability_class"]:
            assert ratchet["stability_class"]["s1"] == StabilityClass.SHARP_DRIFT.value


# ==============================================================================
# CALIBRATION STATUS TRANSITION TESTS
# ==============================================================================

class TestCalibrationStatusTransitions:
    """Tests for calibration status transitions."""
    
    def test_ok_to_attention_transition(self):
        """Should transition from OK to ATTENTION with moderate pressure."""
        # OK: Low pressure
        ratchet_ok = {
            "realism_pressure": 0.2,
            "scenario_retention_score": {},
            "stability_class": {},
        }
        
        policy_ok = {"status": "OK", "scenarios_needing_tuning": []}
        console_ok = build_scenario_calibration_console(ratchet_ok, policy_ok)
        
        # ATTENTION: Moderate pressure
        ratchet_attention = {
            "realism_pressure": 0.4,
            "scenario_retention_score": {},
            "stability_class": {},
        }
        
        policy_attention = {"status": "ATTENTION", "scenarios_needing_tuning": ["s1"]}
        console_attention = build_scenario_calibration_console(ratchet_attention, policy_attention)
        
        assert console_ok["calibration_status"] == CalibrationStatus.OK.value
        assert console_attention["calibration_status"] == CalibrationStatus.ATTENTION.value
    
    def test_attention_to_block_transition(self):
        """Should transition from ATTENTION to BLOCK with high pressure."""
        # ATTENTION: Moderate pressure
        ratchet_attention = {
            "realism_pressure": 0.4,
            "scenario_retention_score": {},
            "stability_class": {},
        }
        
        policy_attention = {"status": "ATTENTION", "scenarios_needing_tuning": []}
        console_attention = build_scenario_calibration_console(ratchet_attention, policy_attention)
        
        # BLOCK: High pressure
        ratchet_block = {
            "realism_pressure": 0.8,
            "scenario_retention_score": {},
            "stability_class": {},
        }
        
        policy_block = {"status": "OK", "scenarios_needing_tuning": []}
        console_block = build_scenario_calibration_console(ratchet_block, policy_block)
        
        assert console_attention["calibration_status"] == CalibrationStatus.ATTENTION.value
        assert console_block["calibration_status"] == CalibrationStatus.BLOCK.value
    
    def test_block_to_ok_transition(self):
        """Should transition from BLOCK to OK when conditions improve."""
        # BLOCK: High pressure
        ratchet_block = {
            "realism_pressure": 0.8,
            "scenario_retention_score": {},
            "stability_class": {},
        }
        
        policy_block = {"status": "BLOCK", "scenarios_needing_tuning": []}
        console_block = build_scenario_calibration_console(ratchet_block, policy_block)
        
        # OK: Low pressure, no issues
        ratchet_ok = {
            "realism_pressure": 0.1,
            "scenario_retention_score": {},
            "stability_class": {},
        }
        
        policy_ok = {"status": "OK", "scenarios_needing_tuning": []}
        console_ok = build_scenario_calibration_console(ratchet_ok, policy_ok)
        
        assert console_block["calibration_status"] == CalibrationStatus.BLOCK.value
        assert console_ok["calibration_status"] == CalibrationStatus.OK.value


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

