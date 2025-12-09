"""
Tests for Phase V: Cross-Tile Perf Governance & Evidence Adapter
================================================================

PERF ONLY — NO BEHAVIOR CHANGE

These tests verify the joint governance view and release bundle adapter
that integrate performance with budget and metric conformance data.

Marked with @pytest.mark.perf to exclude from default CI runs.
"""

import pytest

# Mark all tests in this module
pytestmark = pytest.mark.perf


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def perf_trend_low_risk():
    """Perf trend with LOW risk (all passing)."""
    return {
        "schema_version": "1.0",
        "runs": [
            {"run_id": "run_1", "status": "OK", "worst_component": None, "worst_delta_pct": -5.0},
            {"run_id": "run_2", "status": "OK", "worst_component": None, "worst_delta_pct": -3.0},
            {"run_id": "run_3", "status": "OK", "worst_component": None, "worst_delta_pct": -7.0},
        ],
        "components_with_repeated_breaches": [],
        "release_risk_level": "LOW",
        "total_runs": 3,
        "fail_count": 0,
        "warn_count": 0,
        "pass_count": 3,
    }


@pytest.fixture
def perf_trend_high_risk():
    """Perf trend with HIGH risk (multiple failures)."""
    return {
        "schema_version": "1.0",
        "runs": [
            {"run_id": "run_1", "status": "BLOCK", "worst_component": "scoring", "worst_delta_pct": 30.0, "slice_name": "slice_medium"},
            {"run_id": "run_2", "status": "BLOCK", "worst_component": "scoring", "worst_delta_pct": 28.0, "slice_name": "slice_medium"},
            {"run_id": "run_3", "status": "BLOCK", "worst_component": "derivation", "worst_delta_pct": 25.0, "slice_name": "slice_large"},
        ],
        "components_with_repeated_breaches": ["scoring"],
        "release_risk_level": "HIGH",
        "total_runs": 3,
        "fail_count": 3,
        "warn_count": 0,
        "pass_count": 0,
    }


@pytest.fixture
def perf_trend_medium_risk():
    """Perf trend with MEDIUM risk (some warnings)."""
    return {
        "schema_version": "1.0",
        "runs": [
            {"run_id": "run_1", "status": "OK", "worst_component": None, "worst_delta_pct": -5.0},
            {"run_id": "run_2", "status": "WARN", "worst_component": "derivation", "worst_delta_pct": 8.0, "slice_name": "slice_medium"},
            {"run_id": "run_3", "status": "OK", "worst_component": None, "worst_delta_pct": -3.0},
        ],
        "components_with_repeated_breaches": [],
        "release_risk_level": "MEDIUM",
        "total_runs": 3,
        "fail_count": 0,
        "warn_count": 1,
        "pass_count": 2,
    }


@pytest.fixture
def budget_trend_low_risk():
    """Budget trend with LOW risk."""
    return {
        "risk_level": "LOW",
        "slices": {
            "slice_medium": {"risk_level": "LOW"},
            "slice_large": {"risk_level": "LOW"},
        },
        "uplift_slices": ["slice_medium", "slice_large"],
    }


@pytest.fixture
def budget_trend_high_risk():
    """Budget trend with HIGH risk."""
    return {
        "risk_level": "HIGH",
        "slices": {
            "slice_medium": {"risk_level": "HIGH"},
            "slice_large": {"risk_level": "MEDIUM"},
        },
        "uplift_slices": ["slice_medium", "slice_large"],
    }


@pytest.fixture
def metric_conformance_ok():
    """Metric conformance with all slices OK."""
    return {
        "slices": ["slice_medium", "slice_large"],
        "critical_slices": ["slice_medium"],
        "conformance_status": "OK",
    }


@pytest.fixture
def metric_conformance_critical():
    """Metric conformance with critical slices."""
    return {
        "slices": ["slice_medium", "slice_large", "slice_small"],
        "critical_slices": ["slice_medium", "slice_small"],
        "conformance_status": "OK",
    }


# ---------------------------------------------------------------------------
# Task 1: Perf × Budget × Metric Joint View Tests
# ---------------------------------------------------------------------------

class TestPerfJointGovernanceView:
    """Tests for build_perf_joint_governance_view() function."""
    
    def test_joint_view_low_risk_all_ok(
        self,
        perf_trend_low_risk,
        budget_trend_low_risk,
        metric_conformance_ok,
    ):
        """Test joint view with all LOW risk."""
        from experiments.verify_perf_equivalence import build_perf_joint_governance_view
        
        view = build_perf_joint_governance_view(
            perf_trend_low_risk,
            budget_trend_low_risk,
            metric_conformance_ok,
        )
        
        assert view["perf_risk"] == "LOW"
        assert len(view["slices_with_perf_regressions"]) == 0
        assert len(view["slices_where_perf_blocks_uplift"]) == 0
        assert "low performance risk" in view["summary_note"].lower() or "acceptable bounds" in view["summary_note"].lower()
    
    def test_joint_view_high_perf_risk(
        self,
        perf_trend_high_risk,
        budget_trend_low_risk,
        metric_conformance_ok,
    ):
        """Test joint view with HIGH perf risk."""
        from experiments.verify_perf_equivalence import build_perf_joint_governance_view
        
        view = build_perf_joint_governance_view(
            perf_trend_high_risk,
            budget_trend_low_risk,
            metric_conformance_ok,
        )
        
        assert view["perf_risk"] == "HIGH"
        assert len(view["slices_with_perf_regressions"]) > 0
        # Should not block uplift if budget risk is LOW
        assert len(view["slices_where_perf_blocks_uplift"]) == 0
    
    def test_joint_view_blocks_uplift_high_budget_risk(
        self,
        perf_trend_high_risk,
        budget_trend_high_risk,
        metric_conformance_ok,
    ):
        """Test joint view blocks uplift when perf + budget both HIGH."""
        from experiments.verify_perf_equivalence import build_perf_joint_governance_view
        
        view = build_perf_joint_governance_view(
            perf_trend_high_risk,
            budget_trend_high_risk,
            metric_conformance_ok,
        )
        
        assert view["perf_risk"] == "HIGH"
        assert len(view["slices_with_perf_regressions"]) > 0
        # Should block uplift on slices with both perf regression and high budget risk
        assert len(view["slices_where_perf_blocks_uplift"]) > 0
        assert "uplift blocked" in view["summary_note"].lower() or "blocks uplift" in view["summary_note"].lower()
    
    def test_joint_view_medium_risk(
        self,
        perf_trend_medium_risk,
        budget_trend_low_risk,
        metric_conformance_ok,
    ):
        """Test joint view with MEDIUM perf risk."""
        from experiments.verify_perf_equivalence import build_perf_joint_governance_view
        
        view = build_perf_joint_governance_view(
            perf_trend_medium_risk,
            budget_trend_low_risk,
            metric_conformance_ok,
        )
        
        assert view["perf_risk"] == "MEDIUM"
        assert "medium" in view["summary_note"].lower()
    
    def test_joint_view_critical_slices(
        self,
        perf_trend_high_risk,
        budget_trend_low_risk,
        metric_conformance_critical,
    ):
        """Test joint view with critical slices."""
        from experiments.verify_perf_equivalence import build_perf_joint_governance_view
        
        view = build_perf_joint_governance_view(
            perf_trend_high_risk,
            budget_trend_low_risk,
            metric_conformance_critical,
        )
        
        assert view["perf_risk"] == "HIGH"
        assert "slice_medium" in view["critical_slices"]
        assert len(view["slices_with_perf_regressions"]) > 0


# ---------------------------------------------------------------------------
# Task 2: Release Bundle Adapter Tests
# ---------------------------------------------------------------------------

class TestSummarizePerfForGlobalRelease:
    """Tests for summarize_perf_for_global_release() function."""
    
    def test_release_ok_perf_improvement_everywhere(
        self,
        perf_trend_low_risk,
        budget_trend_low_risk,
        metric_conformance_ok,
    ):
        """Test: Perf improvement everywhere → OK."""
        from experiments.verify_perf_equivalence import (
            build_perf_joint_governance_view,
            summarize_perf_for_global_release,
        )
        
        joint_view = build_perf_joint_governance_view(
            perf_trend_low_risk,
            budget_trend_low_risk,
            metric_conformance_ok,
        )
        
        release_summary = summarize_perf_for_global_release(joint_view)
        
        assert release_summary["release_ok"] is True
        assert release_summary["status"] == "OK"
        assert len(release_summary["blocking_components"]) == 0
        assert "ready" in release_summary["headline"].lower() or "acceptable" in release_summary["headline"].lower()
    
    def test_release_warn_perf_regressions_critical_slices_stable_budget(
        self,
        perf_trend_medium_risk,
        budget_trend_low_risk,
        metric_conformance_critical,
    ):
        """Test: Perf regressions on critical slices with consistent budget OK → WARN (not BLOCK)."""
        from experiments.verify_perf_equivalence import (
            build_perf_joint_governance_view,
            summarize_perf_for_global_release,
        )
        
        # Create perf trend with regressions on critical slice
        perf_trend = {
            "schema_version": "1.0",
            "runs": [
                {"run_id": "run_1", "status": "WARN", "worst_component": "scoring", "worst_delta_pct": 10.0, "slice_name": "slice_medium"},
                {"run_id": "run_2", "status": "WARN", "worst_component": "derivation", "worst_delta_pct": 8.0, "slice_name": "slice_medium"},
            ],
            "components_with_repeated_breaches": [],
            "release_risk_level": "MEDIUM",
            "total_runs": 2,
            "fail_count": 0,
            "warn_count": 2,
            "pass_count": 0,
        }
        
        joint_view = build_perf_joint_governance_view(
            perf_trend,
            budget_trend_low_risk,
            metric_conformance_critical,
        )
        
        release_summary = summarize_perf_for_global_release(joint_view)
        
        # Should WARN, not BLOCK, because budget is stable (LOW risk)
        assert release_summary["release_ok"] is True  # WARN doesn't block
        assert release_summary["status"] == "WARN"
        assert len(release_summary["blocking_components"]) == 0
        assert "warning" in release_summary["headline"].lower()
        assert "stable budget" in release_summary["headline"].lower() or "consistent budget" in release_summary["headline"].lower()
    
    def test_release_block_perf_regressions_high_budget_uplift(
        self,
        perf_trend_high_risk,
        budget_trend_high_risk,
        metric_conformance_ok,
    ):
        """Test: Perf regressions + budget HIGH risk on uplift slices → BLOCK."""
        from experiments.verify_perf_equivalence import (
            build_perf_joint_governance_view,
            summarize_perf_for_global_release,
        )
        
        joint_view = build_perf_joint_governance_view(
            perf_trend_high_risk,
            budget_trend_high_risk,
            metric_conformance_ok,
        )
        
        release_summary = summarize_perf_for_global_release(joint_view)
        
        # Should BLOCK because perf regressions + high budget risk on uplift slices
        assert release_summary["release_ok"] is False
        assert release_summary["status"] == "BLOCK"
        assert len(release_summary["blocking_components"]) > 0
        assert "blocked" in release_summary["headline"].lower()
        assert "uplift" in release_summary["headline"].lower()
    
    def test_release_block_high_perf_risk_critical_slices(
        self,
        perf_trend_high_risk,
        budget_trend_low_risk,
        metric_conformance_critical,
    ):
        """Test: BLOCK if HIGH perf risk on critical slices."""
        from experiments.verify_perf_equivalence import (
            build_perf_joint_governance_view,
            summarize_perf_for_global_release,
        )
        
        joint_view = build_perf_joint_governance_view(
            perf_trend_high_risk,
            budget_trend_low_risk,
            metric_conformance_critical,
        )
        
        release_summary = summarize_perf_for_global_release(joint_view)
        
        # Should BLOCK because HIGH perf risk affects critical slices
        assert release_summary["release_ok"] is False
        assert release_summary["status"] == "BLOCK"
        assert "critical" in release_summary["headline"].lower()
    
    def test_release_block_high_perf_risk_general(
        self,
        perf_trend_high_risk,
        budget_trend_low_risk,
        metric_conformance_ok,
    ):
        """Test: BLOCK if HIGH perf risk even without critical slices."""
        from experiments.verify_perf_equivalence import (
            build_perf_joint_governance_view,
            summarize_perf_for_global_release,
        )
        
        joint_view = build_perf_joint_governance_view(
            perf_trend_high_risk,
            budget_trend_low_risk,
            metric_conformance_ok,
        )
        
        release_summary = summarize_perf_for_global_release(joint_view)
        
        # Should BLOCK because perf risk is HIGH
        assert release_summary["release_ok"] is False
        assert release_summary["status"] == "BLOCK"
        assert "high performance risk" in release_summary["headline"].lower()
    
    def test_release_warn_medium_perf_risk(
        self,
        perf_trend_medium_risk,
        budget_trend_low_risk,
        metric_conformance_ok,
    ):
        """Test: WARN if MEDIUM perf risk."""
        from experiments.verify_perf_equivalence import (
            build_perf_joint_governance_view,
            summarize_perf_for_global_release,
        )
        
        joint_view = build_perf_joint_governance_view(
            perf_trend_medium_risk,
            budget_trend_low_risk,
            metric_conformance_ok,
        )
        
        release_summary = summarize_perf_for_global_release(joint_view)
        
        # Should WARN for MEDIUM risk
        assert release_summary["release_ok"] is True
        assert release_summary["status"] == "WARN"
        assert "medium" in release_summary["headline"].lower()
    
    def test_release_summary_structure(self):
        """Test that release summary has all required fields."""
        from experiments.verify_perf_equivalence import (
            build_perf_joint_governance_view,
            summarize_perf_for_global_release,
        )
        
        perf_trend = {
            "schema_version": "1.0",
            "runs": [],
            "components_with_repeated_breaches": [],
            "release_risk_level": "LOW",
        }
        
        budget_trend = {
            "risk_level": "LOW",
            "slices": {},
            "uplift_slices": [],
        }
        
        metric_conformance = {
            "slices": [],
            "critical_slices": [],
        }
        
        joint_view = build_perf_joint_governance_view(
            perf_trend,
            budget_trend,
            metric_conformance,
        )
        
        release_summary = summarize_perf_for_global_release(joint_view)
        
        # Verify all required fields
        assert "release_ok" in release_summary
        assert "status" in release_summary
        assert "blocking_components" in release_summary
        assert "headline" in release_summary
        assert "perf_risk" in release_summary
        assert "budget_risk" in release_summary
        
        # Verify types
        assert isinstance(release_summary["release_ok"], bool)
        assert release_summary["status"] in ("OK", "WARN", "BLOCK")
        assert isinstance(release_summary["blocking_components"], list)
        assert isinstance(release_summary["headline"], str)

