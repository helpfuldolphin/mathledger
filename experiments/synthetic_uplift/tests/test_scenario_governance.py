#!/usr/bin/env python3
"""
==============================================================================
PHASE II — SYNTHETIC TEST DATA ONLY
==============================================================================

Test Suite for Scenario Governance & Envelope Analytics
---------------------------------------------------------

Tests for:
    - Scenario Realism Snapshot (Task 1)
    - Realism Envelope Timeline (Task 2)
    - Global Health Hook (Task 3)

NOT derived from real derivations; NOT part of Evidence Pack.

==============================================================================
"""

import pytest
import sys
from datetime import datetime, timezone
from pathlib import Path

project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.synthetic_uplift.noise_models import SAFETY_LABEL
from experiments.synthetic_uplift.scenario_governance import (
    SCHEMA_VERSION,
    CRITICAL_SCENARIOS,
    HealthStatus,
    build_scenario_realism_snapshot,
    build_realism_envelope_timeline,
    summarize_synthetic_realism_for_global_health,
    build_governance_report,
    load_and_build_governance_report,
    format_governance_report,
    _extract_drift_mode,
    _build_rare_event_profile,
)


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def sample_config():
    """Sample scenario configuration."""
    return {
        "parameters": {
            "seed": 42,
            "num_cycles": 500,
            "probabilities": {
                "baseline": {"class_a": 0.7, "class_b": 0.6},
                "rfl": {"class_a": 0.75, "class_b": 0.65},
            },
            "drift": {"mode": "cyclical", "amplitude": 0.15, "period": 100},
            "correlation": {"rho": 0.3},
            "rare_events": [
                {"type": "catastrophic_collapse", "trigger_probability": 0.02, "magnitude": -0.5},
                {"type": "recovery_spike", "trigger_probability": 0.01, "magnitude": 0.3},
            ],
        }
    }


@pytest.fixture
def sample_envelope_pass():
    """Sample passing envelope result."""
    return {
        "passed": True,
        "violations": [],
    }


@pytest.fixture
def sample_envelope_fail():
    """Sample failing envelope result."""
    return {
        "passed": False,
        "violations": [
            {"parameter": "drift.amplitude", "bound_type": "max", "actual_value": 0.4, "bound_value": 0.25},
            {"parameter": "correlation.rho", "bound_type": "max", "actual_value": 0.95, "bound_value": 0.9},
        ],
    }


# ==============================================================================
# TASK 1: SCENARIO REALISM SNAPSHOT TESTS
# ==============================================================================

class TestScenarioRealismSnapshot:
    """Tests for build_scenario_realism_snapshot."""
    
    def test_snapshot_has_required_fields(self, sample_config, sample_envelope_pass):
        """Snapshot should have all required fields."""
        snapshot = build_scenario_realism_snapshot(
            "synthetic_test",
            sample_config,
            sample_envelope_pass,
        )
        
        assert "schema_version" in snapshot
        assert "scenario_name" in snapshot
        assert "drift_mode" in snapshot
        assert "envelope_pass" in snapshot
        assert "violated_checks" in snapshot
        assert "rare_event_profile" in snapshot
        assert "timestamp" in snapshot
    
    def test_snapshot_schema_version(self, sample_config, sample_envelope_pass):
        """Snapshot should have correct schema version."""
        snapshot = build_scenario_realism_snapshot(
            "synthetic_test",
            sample_config,
            sample_envelope_pass,
        )
        
        assert snapshot["schema_version"] == SCHEMA_VERSION
    
    def test_snapshot_scenario_name(self, sample_config, sample_envelope_pass):
        """Snapshot should include scenario name."""
        snapshot = build_scenario_realism_snapshot(
            "synthetic_my_scenario",
            sample_config,
            sample_envelope_pass,
        )
        
        assert snapshot["scenario_name"] == "synthetic_my_scenario"
    
    def test_snapshot_rejects_non_synthetic_name(self, sample_config, sample_envelope_pass):
        """Snapshot should reject non-synthetic scenario names."""
        with pytest.raises(ValueError, match="synthetic_"):
            build_scenario_realism_snapshot(
                "invalid_name",
                sample_config,
                sample_envelope_pass,
            )
    
    def test_snapshot_drift_mode_sinusoidal(self, sample_config, sample_envelope_pass):
        """Should detect sinusoidal (cyclical) drift mode."""
        snapshot = build_scenario_realism_snapshot(
            "synthetic_test",
            sample_config,
            sample_envelope_pass,
        )
        
        assert snapshot["drift_mode"] == "sinusoidal"
    
    def test_snapshot_drift_mode_linear(self, sample_envelope_pass):
        """Should detect linear (monotonic) drift mode."""
        config = {
            "parameters": {
                "drift": {"mode": "monotonic", "slope": 0.001},
            }
        }
        
        snapshot = build_scenario_realism_snapshot(
            "synthetic_test",
            config,
            sample_envelope_pass,
        )
        
        assert snapshot["drift_mode"] == "linear"
    
    def test_snapshot_drift_mode_step(self, sample_envelope_pass):
        """Should detect step (shock) drift mode."""
        config = {
            "parameters": {
                "drift": {"mode": "shock", "shock_cycle": 250},
            }
        }
        
        snapshot = build_scenario_realism_snapshot(
            "synthetic_test",
            config,
            sample_envelope_pass,
        )
        
        assert snapshot["drift_mode"] == "step"
    
    def test_snapshot_drift_mode_none(self, sample_envelope_pass):
        """Should detect no drift."""
        config = {
            "parameters": {
                "drift": {"mode": "none"},
            }
        }
        
        snapshot = build_scenario_realism_snapshot(
            "synthetic_test",
            config,
            sample_envelope_pass,
        )
        
        assert snapshot["drift_mode"] == "none"
    
    def test_snapshot_envelope_pass_true(self, sample_config, sample_envelope_pass):
        """Should correctly record envelope pass."""
        snapshot = build_scenario_realism_snapshot(
            "synthetic_test",
            sample_config,
            sample_envelope_pass,
        )
        
        assert snapshot["envelope_pass"] is True
        assert snapshot["violated_checks"] == []
    
    def test_snapshot_envelope_pass_false(self, sample_config, sample_envelope_fail):
        """Should correctly record envelope failure."""
        snapshot = build_scenario_realism_snapshot(
            "synthetic_test",
            sample_config,
            sample_envelope_fail,
        )
        
        assert snapshot["envelope_pass"] is False
        assert len(snapshot["violated_checks"]) == 2
    
    def test_snapshot_violated_checks_format(self, sample_config, sample_envelope_fail):
        """Violated checks should be in check_id format."""
        snapshot = build_scenario_realism_snapshot(
            "synthetic_test",
            sample_config,
            sample_envelope_fail,
        )
        
        checks = snapshot["violated_checks"]
        assert "drift.amplitude:max" in checks
        assert "correlation.rho:max" in checks
    
    def test_snapshot_rare_event_profile(self, sample_config, sample_envelope_pass):
        """Should build rare event profile."""
        snapshot = build_scenario_realism_snapshot(
            "synthetic_test",
            sample_config,
            sample_envelope_pass,
        )
        
        profile = snapshot["rare_event_profile"]
        assert profile["count"] == 2
        assert profile["has_catastrophic"] is True
        assert profile["has_recovery"] is True
    
    def test_snapshot_has_timestamp(self, sample_config, sample_envelope_pass):
        """Snapshot should have ISO timestamp."""
        snapshot = build_scenario_realism_snapshot(
            "synthetic_test",
            sample_config,
            sample_envelope_pass,
        )
        
        # Should be valid ISO format
        timestamp = snapshot["timestamp"]
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    
    def test_snapshot_has_config_hash(self, sample_config, sample_envelope_pass):
        """Snapshot should have config hash."""
        snapshot = build_scenario_realism_snapshot(
            "synthetic_test",
            sample_config,
            sample_envelope_pass,
        )
        
        assert "config_hash" in snapshot
        assert len(snapshot["config_hash"]) == 16  # SHA256 truncated to 16 chars
    
    def test_snapshot_includes_safety_label(self, sample_config, sample_envelope_pass):
        """Snapshot should include safety label."""
        snapshot = build_scenario_realism_snapshot(
            "synthetic_test",
            sample_config,
            sample_envelope_pass,
        )
        
        assert snapshot["label"] == SAFETY_LABEL


# ==============================================================================
# TASK 2: REALISM ENVELOPE TIMELINE TESTS
# ==============================================================================

class TestRealismEnvelopeTimeline:
    """Tests for build_realism_envelope_timeline."""
    
    def test_empty_snapshots(self):
        """Empty snapshots should produce empty timeline."""
        timeline = build_realism_envelope_timeline([])
        
        assert timeline["per_scenario"] == {}
        assert timeline["global"]["scenario_count"] == 0
        assert timeline["global"]["total_snapshots"] == 0
    
    def test_single_passing_snapshot(self):
        """Single passing snapshot should be tracked."""
        snapshots = [{
            "scenario_name": "synthetic_test",
            "envelope_pass": True,
            "violated_checks": [],
            "timestamp": "2025-01-01T00:00:00Z",
        }]
        
        timeline = build_realism_envelope_timeline(snapshots)
        
        assert "synthetic_test" in timeline["per_scenario"]
        record = timeline["per_scenario"]["synthetic_test"]
        assert record["times_passed"] == 1
        assert record["times_failed"] == 0
        assert record["pass_rate"] == 1.0
    
    def test_single_failing_snapshot(self):
        """Single failing snapshot should be tracked."""
        snapshots = [{
            "scenario_name": "synthetic_test",
            "envelope_pass": False,
            "violated_checks": ["drift.amplitude:max"],
            "timestamp": "2025-01-01T00:00:00Z",
        }]
        
        timeline = build_realism_envelope_timeline(snapshots)
        
        record = timeline["per_scenario"]["synthetic_test"]
        assert record["times_passed"] == 0
        assert record["times_failed"] == 1
        assert record["pass_rate"] == 0.0
    
    def test_multiple_snapshots_same_scenario(self):
        """Multiple snapshots for same scenario should be aggregated."""
        snapshots = [
            {"scenario_name": "synthetic_test", "envelope_pass": True, "violated_checks": [], "timestamp": "T1"},
            {"scenario_name": "synthetic_test", "envelope_pass": False, "violated_checks": ["check1"], "timestamp": "T2"},
            {"scenario_name": "synthetic_test", "envelope_pass": True, "violated_checks": [], "timestamp": "T3"},
        ]
        
        timeline = build_realism_envelope_timeline(snapshots)
        
        record = timeline["per_scenario"]["synthetic_test"]
        assert record["times_passed"] == 2
        assert record["times_failed"] == 1
        assert record["pass_rate"] == pytest.approx(2/3)
    
    def test_multiple_scenarios(self):
        """Multiple scenarios should be tracked separately."""
        snapshots = [
            {"scenario_name": "synthetic_a", "envelope_pass": True, "violated_checks": [], "timestamp": "T1"},
            {"scenario_name": "synthetic_b", "envelope_pass": False, "violated_checks": ["check1"], "timestamp": "T2"},
        ]
        
        timeline = build_realism_envelope_timeline(snapshots)
        
        assert len(timeline["per_scenario"]) == 2
        assert timeline["per_scenario"]["synthetic_a"]["times_passed"] == 1
        assert timeline["per_scenario"]["synthetic_b"]["times_failed"] == 1
    
    def test_global_scenario_count(self):
        """Global stats should count unique scenarios."""
        snapshots = [
            {"scenario_name": "synthetic_a", "envelope_pass": True, "violated_checks": [], "timestamp": "T1"},
            {"scenario_name": "synthetic_a", "envelope_pass": True, "violated_checks": [], "timestamp": "T2"},
            {"scenario_name": "synthetic_b", "envelope_pass": True, "violated_checks": [], "timestamp": "T3"},
        ]
        
        timeline = build_realism_envelope_timeline(snapshots)
        
        assert timeline["global"]["scenario_count"] == 2
        assert timeline["global"]["total_snapshots"] == 3
    
    def test_global_envelope_breach_rate(self):
        """Should compute correct global breach rate."""
        snapshots = [
            {"scenario_name": "synthetic_a", "envelope_pass": True, "violated_checks": [], "timestamp": "T1"},
            {"scenario_name": "synthetic_a", "envelope_pass": False, "violated_checks": ["x"], "timestamp": "T2"},
            {"scenario_name": "synthetic_b", "envelope_pass": False, "violated_checks": ["y"], "timestamp": "T3"},
            {"scenario_name": "synthetic_b", "envelope_pass": True, "violated_checks": [], "timestamp": "T4"},
        ]
        
        timeline = build_realism_envelope_timeline(snapshots)
        
        # 2 failures out of 4 snapshots = 50%
        assert timeline["global"]["envelope_breach_rate"] == 0.5
    
    def test_scenarios_with_repeated_breaches(self):
        """Should identify scenarios with multiple breaches."""
        snapshots = [
            {"scenario_name": "synthetic_a", "envelope_pass": False, "violated_checks": ["x"], "timestamp": "T1"},
            {"scenario_name": "synthetic_a", "envelope_pass": False, "violated_checks": ["x"], "timestamp": "T2"},
            {"scenario_name": "synthetic_b", "envelope_pass": False, "violated_checks": ["y"], "timestamp": "T3"},
            {"scenario_name": "synthetic_b", "envelope_pass": True, "violated_checks": [], "timestamp": "T4"},
        ]
        
        timeline = build_realism_envelope_timeline(snapshots)
        
        # synthetic_a has 2 failures, synthetic_b has only 1
        assert timeline["global"]["scenarios_with_repeated_breaches"] == ["synthetic_a"]
    
    def test_violated_checks_history(self):
        """Should track violated checks over time."""
        snapshots = [
            {"scenario_name": "synthetic_test", "envelope_pass": False, "violated_checks": ["check1"], "timestamp": "T1"},
            {"scenario_name": "synthetic_test", "envelope_pass": False, "violated_checks": ["check2"], "timestamp": "T2"},
        ]
        
        timeline = build_realism_envelope_timeline(snapshots)
        
        history = timeline["per_scenario"]["synthetic_test"]["violated_checks_history"]
        assert len(history) == 2
        assert history[0]["checks"] == ["check1"]
        assert history[1]["checks"] == ["check2"]
    
    def test_timeline_includes_safety_label(self):
        """Timeline should include safety label."""
        timeline = build_realism_envelope_timeline([])
        
        assert timeline["label"] == SAFETY_LABEL


# ==============================================================================
# TASK 3: GLOBAL HEALTH HOOK TESTS
# ==============================================================================

class TestGlobalHealthHook:
    """Tests for summarize_synthetic_realism_for_global_health."""
    
    def test_empty_timeline_ok(self):
        """Empty timeline should result in OK status."""
        timeline = build_realism_envelope_timeline([])
        summary = summarize_synthetic_realism_for_global_health(timeline)
        
        assert summary["status"] == "OK"
        assert summary["realism_ok"] is True
    
    def test_all_passing_ok(self):
        """All passing scenarios should result in OK status."""
        snapshots = [
            {"scenario_name": "synthetic_null_uplift", "envelope_pass": True, "violated_checks": [], "timestamp": "T1"},
            {"scenario_name": "synthetic_positive_uplift", "envelope_pass": True, "violated_checks": [], "timestamp": "T2"},
        ]
        timeline = build_realism_envelope_timeline(snapshots)
        summary = summarize_synthetic_realism_for_global_health(timeline)
        
        assert summary["status"] == "OK"
        assert summary["realism_ok"] is True
        assert summary["envelope_breach_rate"] == 0.0
        assert summary["scenarios_needing_review"] == []
    
    def test_single_failure_warn(self):
        """Single failure should result in WARN status."""
        snapshots = [
            {"scenario_name": "synthetic_test", "envelope_pass": False, "violated_checks": ["x"], "timestamp": "T1"},
        ]
        timeline = build_realism_envelope_timeline(snapshots)
        summary = summarize_synthetic_realism_for_global_health(timeline)
        
        assert summary["status"] == "WARN"
        assert summary["realism_ok"] is False
        assert "synthetic_test" in summary["scenarios_needing_review"]
    
    def test_non_critical_repeated_breach_warn(self):
        """Non-critical scenario with repeated breaches should WARN."""
        snapshots = [
            {"scenario_name": "synthetic_other", "envelope_pass": False, "violated_checks": ["x"], "timestamp": "T1"},
            {"scenario_name": "synthetic_other", "envelope_pass": False, "violated_checks": ["x"], "timestamp": "T2"},
        ]
        timeline = build_realism_envelope_timeline(snapshots)
        summary = summarize_synthetic_realism_for_global_health(timeline)
        
        assert summary["status"] == "WARN"
    
    def test_critical_repeated_breach_block(self):
        """Critical scenario with repeated breaches should BLOCK."""
        snapshots = [
            {"scenario_name": "synthetic_null_uplift", "envelope_pass": False, "violated_checks": ["x"], "timestamp": "T1"},
            {"scenario_name": "synthetic_null_uplift", "envelope_pass": False, "violated_checks": ["x"], "timestamp": "T2"},
        ]
        timeline = build_realism_envelope_timeline(snapshots)
        summary = summarize_synthetic_realism_for_global_health(timeline)
        
        assert summary["status"] == "BLOCK"
    
    def test_critical_single_breach_warn_not_block(self):
        """Critical scenario with single breach should WARN (not BLOCK)."""
        snapshots = [
            {"scenario_name": "synthetic_null_uplift", "envelope_pass": False, "violated_checks": ["x"], "timestamp": "T1"},
        ]
        timeline = build_realism_envelope_timeline(snapshots)
        summary = summarize_synthetic_realism_for_global_health(timeline)
        
        # Single breach is WARN, not BLOCK
        assert summary["status"] == "WARN"
    
    def test_scenarios_needing_review_includes_failures(self):
        """Scenarios needing review should include all failures."""
        snapshots = [
            {"scenario_name": "synthetic_a", "envelope_pass": False, "violated_checks": ["x"], "timestamp": "T1"},
            {"scenario_name": "synthetic_b", "envelope_pass": True, "violated_checks": [], "timestamp": "T2"},
            {"scenario_name": "synthetic_c", "envelope_pass": False, "violated_checks": ["y"], "timestamp": "T3"},
        ]
        timeline = build_realism_envelope_timeline(snapshots)
        summary = summarize_synthetic_realism_for_global_health(timeline)
        
        needing_review = summary["scenarios_needing_review"]
        assert "synthetic_a" in needing_review
        assert "synthetic_c" in needing_review
        assert "synthetic_b" not in needing_review
    
    def test_scenarios_needing_review_details(self):
        """Should include detailed info for scenarios needing review."""
        snapshots = [
            {"scenario_name": "synthetic_null_uplift", "envelope_pass": False, "violated_checks": ["x"], "timestamp": "T1"},
        ]
        timeline = build_realism_envelope_timeline(snapshots)
        summary = summarize_synthetic_realism_for_global_health(timeline)
        
        details = summary["scenarios_needing_review_details"]
        assert len(details) == 1
        assert details[0]["scenario"] == "synthetic_null_uplift"
        assert details[0]["is_critical"] is True
        assert details[0]["times_failed"] == 1
    
    def test_critical_scenarios_sorted_first(self):
        """Critical scenarios should appear first in review list."""
        snapshots = [
            {"scenario_name": "synthetic_other", "envelope_pass": False, "violated_checks": ["x"], "timestamp": "T1"},
            {"scenario_name": "synthetic_null_uplift", "envelope_pass": False, "violated_checks": ["y"], "timestamp": "T2"},
        ]
        timeline = build_realism_envelope_timeline(snapshots)
        summary = summarize_synthetic_realism_for_global_health(timeline)
        
        details = summary["scenarios_needing_review_details"]
        # Critical should come first
        assert details[0]["scenario"] == "synthetic_null_uplift"
        assert details[0]["is_critical"] is True
    
    def test_envelope_breach_rate_calculation(self):
        """Should correctly calculate breach rate."""
        snapshots = [
            {"scenario_name": "synthetic_a", "envelope_pass": True, "violated_checks": [], "timestamp": "T1"},
            {"scenario_name": "synthetic_a", "envelope_pass": False, "violated_checks": ["x"], "timestamp": "T2"},
        ]
        timeline = build_realism_envelope_timeline(snapshots)
        summary = summarize_synthetic_realism_for_global_health(timeline)
        
        assert summary["envelope_breach_rate"] == 0.5
    
    def test_critical_scenarios_monitored_list(self):
        """Should list which scenarios are considered critical."""
        timeline = build_realism_envelope_timeline([])
        summary = summarize_synthetic_realism_for_global_health(timeline)
        
        assert "critical_scenarios_monitored" in summary
        assert "synthetic_null_uplift" in summary["critical_scenarios_monitored"]
    
    def test_summary_includes_safety_label(self):
        """Summary should include safety label."""
        timeline = build_realism_envelope_timeline([])
        summary = summarize_synthetic_realism_for_global_health(timeline)
        
        assert summary["label"] == SAFETY_LABEL


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================

class TestGovernanceIntegration:
    """Integration tests for governance report building."""
    
    def test_build_governance_report(self):
        """Should build complete governance report."""
        configs = {
            "synthetic_test1": {
                "parameters": {
                    "drift": {"mode": "none"},
                    "rare_events": [],
                }
            },
            "synthetic_test2": {
                "parameters": {
                    "drift": {"mode": "cyclical", "amplitude": 0.1, "period": 100},
                    "rare_events": [],
                }
            },
        }
        
        envelope_results = {
            "synthetic_test1": {"passed": True, "violations": []},
            "synthetic_test2": {"passed": True, "violations": []},
        }
        
        report = build_governance_report(configs, envelope_results)
        
        assert "snapshots" in report
        assert "timeline" in report
        assert "health_summary" in report
        assert len(report["snapshots"]) == 2
    
    def test_load_and_build_governance_report(self):
        """Should load registry and build report."""
        report = load_and_build_governance_report()
        
        assert report["label"] == SAFETY_LABEL
        assert "snapshots" in report
        assert len(report["snapshots"]) > 0
    
    def test_format_governance_report(self):
        """Should format report as human-readable text."""
        report = load_and_build_governance_report()
        formatted = format_governance_report(report)
        
        assert SAFETY_LABEL in formatted
        assert "SCENARIO GOVERNANCE REPORT" in formatted
        assert "HEALTH STATUS" in formatted


# ==============================================================================
# HELPER FUNCTION TESTS
# ==============================================================================

class TestHelperFunctions:
    """Tests for helper functions."""
    
    def test_extract_drift_mode_cyclical(self):
        """Should map cyclical to sinusoidal."""
        config = {"parameters": {"drift": {"mode": "cyclical"}}}
        assert _extract_drift_mode(config) == "sinusoidal"
    
    def test_extract_drift_mode_monotonic(self):
        """Should map monotonic to linear."""
        config = {"parameters": {"drift": {"mode": "monotonic"}}}
        assert _extract_drift_mode(config) == "linear"
    
    def test_extract_drift_mode_shock(self):
        """Should map shock to step."""
        config = {"parameters": {"drift": {"mode": "shock"}}}
        assert _extract_drift_mode(config) == "step"
    
    def test_extract_drift_mode_none(self):
        """Should preserve none."""
        config = {"parameters": {"drift": {"mode": "none"}}}
        assert _extract_drift_mode(config) == "none"
    
    def test_build_rare_event_profile_empty(self):
        """Empty rare events should produce zero counts."""
        config = {"parameters": {"rare_events": []}}
        profile = _build_rare_event_profile(config)
        
        assert profile["count"] == 0
        assert profile["types"] == []
        assert profile["has_catastrophic"] is False
    
    def test_build_rare_event_profile_with_events(self):
        """Should detect rare event types."""
        config = {
            "parameters": {
                "rare_events": [
                    {"type": "catastrophic_collapse", "trigger_probability": 0.02},
                    {"type": "outlier_burst", "trigger_probability": 0.01},
                ]
            }
        }
        profile = _build_rare_event_profile(config)
        
        assert profile["count"] == 2
        assert profile["has_catastrophic"] is True
        assert profile["has_burst"] is True
        assert profile["total_trigger_probability"] == pytest.approx(0.03)


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

