"""
Tests for Mock Oracle Governance & Adversarial Coverage Radar.

Verifies:
1. build_metric_scenario_coverage_view() produces correct metric-centric view
2. summarize_mock_oracle_drift_for_governance() generates governance summaries
3. build_mock_oracle_director_panel() creates director-level panels
4. Integration of all three functions

SAFEGUARD: These tests verify governance infrastructure — no production impact.
"""

from __future__ import annotations

from typing import Dict, List

import pytest

from backend.verification.mock_config import (
    DRIFT_STATUS_BROKEN,
    DRIFT_STATUS_DRIFTED,
    DRIFT_STATUS_IN_CONTRACT,
    GOVERNANCE_STATUS_ATTENTION,
    GOVERNANCE_STATUS_BLOCK,
    GOVERNANCE_STATUS_OK,
    MIN_SCENARIOS_FOR_WELL_EXERCISED,
    STATUS_LIGHT_GREEN,
    STATUS_LIGHT_RED,
    STATUS_LIGHT_YELLOW,
    VERIFICATION_METRICS,
    build_metric_scenario_coverage_view,
    build_mock_oracle_director_panel,
    detect_scenario_drift,
    export_mock_oracle_contract,
    summarize_mock_oracle_drift_for_governance,
    summarize_scenario_results,
)
from backend.verification.mock_oracle import MockVerifiableOracle
from backend.verification.mock_config import MockOracleConfig


class TestBuildMetricScenarioCoverageView:
    """Tests for build_metric_scenario_coverage_view() function."""

    def _make_scenario_summary(
        self,
        scenario_name: str,
        status: str,
        drift_signals: List[Dict] = None,
        empirical: Dict = None,
    ) -> Dict:
        """Helper to create a scenario summary dict."""
        if empirical is None:
            empirical = {
                "verified": 60.0,
                "failed": 15.0,
                "abstain": 10.0,
                "timeout": 8.0,
                "error": 4.0,
                "crash": 3.0,
                "negative_control": 0.0,
            }
        
        if drift_signals is None:
            drift_signals = []
        
        return {
            "scenario_name": scenario_name,
            "empirical_distribution": empirical,
            "drift_report": {
                "status": status,
                "scenario_name": scenario_name,
                "drift_signals": drift_signals,
            },
        }

    def test_coverage_view_has_metrics(self):
        """Coverage view includes all verification metrics."""
        summaries = [
            self._make_scenario_summary("test1", DRIFT_STATUS_IN_CONTRACT),
        ]
        
        view = build_metric_scenario_coverage_view(summaries)
        
        assert "metrics" in view
        for metric in VERIFICATION_METRICS:
            assert metric in view["metrics"]

    def test_coverage_view_tracks_scenarios_tested(self):
        """Coverage view tracks which scenarios test each metric."""
        summaries = [
            self._make_scenario_summary("scenario_a", DRIFT_STATUS_IN_CONTRACT),
            self._make_scenario_summary("scenario_b", DRIFT_STATUS_IN_CONTRACT),
        ]
        
        view = build_metric_scenario_coverage_view(summaries)
        
        # All metrics should be tested by both scenarios
        for metric in VERIFICATION_METRICS:
            scenarios = view["metrics"][metric]["scenarios_tested"]
            assert "scenario_a" in scenarios
            assert "scenario_b" in scenarios

    def test_coverage_view_tracks_drift_free_count(self):
        """Coverage view counts drift-free scenarios per metric."""
        summaries = [
            self._make_scenario_summary("good1", DRIFT_STATUS_IN_CONTRACT),
            self._make_scenario_summary("good2", DRIFT_STATUS_IN_CONTRACT),
            self._make_scenario_summary("drifted", DRIFT_STATUS_DRIFTED),
        ]
        
        view = build_metric_scenario_coverage_view(summaries)
        
        # Each metric should have 2 drift-free scenarios
        for metric in VERIFICATION_METRICS:
            assert view["metrics"][metric]["drift_free_count"] == 2

    def test_coverage_view_tracks_drifted_scenarios(self):
        """Coverage view tracks drifted scenarios per metric."""
        summaries = [
            self._make_scenario_summary(
                "drifted_scenario",
                DRIFT_STATUS_DRIFTED,
                drift_signals=[{"bucket": "verified", "severity": "WARNING"}],
            ),
        ]
        
        view = build_metric_scenario_coverage_view(summaries)
        
        assert "drifted_scenario" in view["metrics"]["verified"]["drifted_scenarios"]
        assert len(view["metrics"]["verified"]["broken_scenarios"]) == 0

    def test_coverage_view_tracks_broken_scenarios(self):
        """Coverage view tracks broken scenarios per metric."""
        summaries = [
            self._make_scenario_summary(
                "broken_scenario",
                DRIFT_STATUS_BROKEN,
                drift_signals=[{"bucket": "failed", "severity": "BROKEN"}],
            ),
        ]
        
        view = build_metric_scenario_coverage_view(summaries)
        
        assert "broken_scenario" in view["metrics"]["failed"]["broken_scenarios"]
        assert len(view["metrics"]["failed"]["drifted_scenarios"]) == 0

    def test_coverage_view_identifies_well_exercised_metrics(self):
        """Coverage view identifies well-exercised metrics."""
        # Create enough scenarios to make metrics well-exercised
        summaries = []
        for i in range(MIN_SCENARIOS_FOR_WELL_EXERCISED + 1):
            summaries.append(
                self._make_scenario_summary(f"scenario_{i}", DRIFT_STATUS_IN_CONTRACT)
            )
        
        view = build_metric_scenario_coverage_view(summaries)
        
        assert len(view["metrics_well_exercised"]) > 0
        for metric in view["metrics_well_exercised"]:
            assert len(view["metrics"][metric]["scenarios_tested"]) >= MIN_SCENARIOS_FOR_WELL_EXERCISED

    def test_coverage_view_identifies_under_tested_metrics(self):
        """Coverage view identifies under-tested metrics."""
        # Create fewer scenarios than required
        summaries = [
            self._make_scenario_summary("scenario_1", DRIFT_STATUS_IN_CONTRACT),
        ]
        
        view = build_metric_scenario_coverage_view(summaries)
        
        assert len(view["metrics_under_tested"]) > 0
        for metric in view["metrics_under_tested"]:
            assert len(view["metrics"][metric]["scenarios_tested"]) < MIN_SCENARIOS_FOR_WELL_EXERCISED

    def test_coverage_view_has_total_scenarios(self):
        """Coverage view includes total scenario count."""
        summaries = [
            self._make_scenario_summary("s1", DRIFT_STATUS_IN_CONTRACT),
            self._make_scenario_summary("s2", DRIFT_STATUS_IN_CONTRACT),
            self._make_scenario_summary("s3", DRIFT_STATUS_IN_CONTRACT),
        ]
        
        view = build_metric_scenario_coverage_view(summaries)
        
        assert view["total_scenarios"] == 3

    def test_coverage_view_deterministic(self):
        """Coverage view is deterministic (sorted lists)."""
        summaries = [
            self._make_scenario_summary("z_scenario", DRIFT_STATUS_IN_CONTRACT),
            self._make_scenario_summary("a_scenario", DRIFT_STATUS_IN_CONTRACT),
        ]
        
        view = build_metric_scenario_coverage_view(summaries)
        
        # Scenario lists should be sorted
        for metric in VERIFICATION_METRICS:
            scenarios = view["metrics"][metric]["scenarios_tested"]
            assert scenarios == sorted(scenarios)


class TestSummarizeMockOracleDriftForGovernance:
    """Tests for summarize_mock_oracle_drift_for_governance() function."""

    def _make_coverage_view(
        self,
        broken_scenarios: List[str] = None,
        drifted_scenarios: List[str] = None,
    ) -> Dict:
        """Helper to create a coverage view."""
        if broken_scenarios is None:
            broken_scenarios = []
        if drifted_scenarios is None:
            drifted_scenarios = []
        
        metrics = {}
        for metric in VERIFICATION_METRICS:
            metrics[metric] = {
                "scenarios_tested": ["test_scenario"],
                "drift_free_count": 1 if not broken_scenarios and not drifted_scenarios else 0,
                "drifted_scenarios": drifted_scenarios if "test_scenario" in drifted_scenarios else [],
                "broken_scenarios": broken_scenarios if "test_scenario" in broken_scenarios else [],
            }
        
        return {
            "metrics": metrics,
            "metrics_well_exercised": [],
            "metrics_under_tested": VERIFICATION_METRICS,
            "total_scenarios": 1,
        }

    def test_governance_summary_has_broken_flag(self):
        """Governance summary includes has_broken_scenarios flag."""
        view = self._make_coverage_view(broken_scenarios=["broken_scenario"])
        # Ensure broken scenario is actually in a metric's broken_scenarios list
        view["metrics"]["verified"]["broken_scenarios"] = ["broken_scenario"]
        
        summary = summarize_mock_oracle_drift_for_governance(view)
        
        assert "has_broken_scenarios" in summary
        assert summary["has_broken_scenarios"] is True

    def test_governance_summary_no_broken(self):
        """Governance summary correctly identifies no broken scenarios."""
        view = self._make_coverage_view()
        
        summary = summarize_mock_oracle_drift_for_governance(view)
        
        assert summary["has_broken_scenarios"] is False

    def test_governance_summary_tracks_impacted_metrics(self):
        """Governance summary tracks metrics impacted by drift."""
        view = self._make_coverage_view(
            broken_scenarios=["broken_scenario"],
            drifted_scenarios=["drifted_scenario"],
        )
        
        # Add broken to verified, drifted to failed
        view["metrics"]["verified"]["broken_scenarios"] = ["broken_scenario"]
        view["metrics"]["failed"]["drifted_scenarios"] = ["drifted_scenario"]
        
        summary = summarize_mock_oracle_drift_for_governance(view)
        
        assert "metrics_impacted_by_drift" in summary
        assert "verified" in summary["metrics_impacted_by_drift"]
        assert "failed" in summary["metrics_impacted_by_drift"]

    def test_governance_summary_status_ok(self):
        """Governance status is OK when no drift."""
        view = self._make_coverage_view()
        
        summary = summarize_mock_oracle_drift_for_governance(view)
        
        assert summary["status"] == GOVERNANCE_STATUS_OK

    def test_governance_summary_status_attention(self):
        """Governance status is ATTENTION when drifted but not broken."""
        view = self._make_coverage_view(drifted_scenarios=["drifted_scenario"])
        view["metrics"]["verified"]["drifted_scenarios"] = ["drifted_scenario"]
        
        summary = summarize_mock_oracle_drift_for_governance(view)
        
        assert summary["status"] == GOVERNANCE_STATUS_ATTENTION

    def test_governance_summary_status_block(self):
        """Governance status is BLOCK when broken scenarios exist."""
        view = self._make_coverage_view(broken_scenarios=["broken_scenario"])
        view["metrics"]["verified"]["broken_scenarios"] = ["broken_scenario"]
        
        summary = summarize_mock_oracle_drift_for_governance(view)
        
        assert summary["status"] == GOVERNANCE_STATUS_BLOCK

    def test_governance_summary_counts(self):
        """Governance summary includes scenario counts."""
        view = self._make_coverage_view(
            broken_scenarios=["b1", "b2"],
            drifted_scenarios=["d1"],
        )
        view["metrics"]["verified"]["broken_scenarios"] = ["b1", "b2"]
        view["metrics"]["failed"]["drifted_scenarios"] = ["d1"]
        
        summary = summarize_mock_oracle_drift_for_governance(view)
        
        assert summary["broken_scenario_count"] == 2
        assert summary["drifted_scenario_count"] == 1
        assert summary["total_scenarios"] == 1


class TestBuildMockOracleDirectorPanel:
    """Tests for build_mock_oracle_director_panel() function."""

    def _make_governance_summary(
        self,
        status: str = GOVERNANCE_STATUS_OK,
        broken_count: int = 0,
        drifted_count: int = 0,
        impacted_metrics: List[str] = None,
    ) -> Dict:
        """Helper to create a governance summary."""
        if impacted_metrics is None:
            impacted_metrics = []
        
        return {
            "has_broken_scenarios": broken_count > 0,
            "metrics_impacted_by_drift": impacted_metrics,
            "status": status,
            "broken_scenario_count": broken_count,
            "drifted_scenario_count": drifted_count,
            "total_scenarios": 5,
        }

    def _make_coverage_view(self) -> Dict:
        """Helper to create a coverage view."""
        return {
            "metrics": {},
            "metrics_well_exercised": ["verified", "failed"],
            "metrics_under_tested": ["abstain"],
            "total_scenarios": 5,
        }

    def test_director_panel_has_status_light(self):
        """Director panel includes status_light."""
        coverage = self._make_coverage_view()
        governance = self._make_governance_summary()
        
        panel = build_mock_oracle_director_panel(coverage, governance)
        
        assert "status_light" in panel
        assert panel["status_light"] in [STATUS_LIGHT_GREEN, STATUS_LIGHT_YELLOW, STATUS_LIGHT_RED]

    def test_director_panel_status_light_green(self):
        """Status light is GREEN when status is OK."""
        coverage = self._make_coverage_view()
        governance = self._make_governance_summary(status=GOVERNANCE_STATUS_OK)
        
        panel = build_mock_oracle_director_panel(coverage, governance)
        
        assert panel["status_light"] == STATUS_LIGHT_GREEN

    def test_director_panel_status_light_yellow(self):
        """Status light is YELLOW when status is ATTENTION."""
        coverage = self._make_coverage_view()
        governance = self._make_governance_summary(
            status=GOVERNANCE_STATUS_ATTENTION,
            drifted_count=1,
        )
        
        panel = build_mock_oracle_director_panel(coverage, governance)
        
        assert panel["status_light"] == STATUS_LIGHT_YELLOW

    def test_director_panel_status_light_red(self):
        """Status light is RED when status is BLOCK."""
        coverage = self._make_coverage_view()
        governance = self._make_governance_summary(
            status=GOVERNANCE_STATUS_BLOCK,
            broken_count=1,
        )
        
        panel = build_mock_oracle_director_panel(coverage, governance)
        
        assert panel["status_light"] == STATUS_LIGHT_RED

    def test_director_panel_has_scenario_count(self):
        """Director panel includes scenario_count."""
        coverage = self._make_coverage_view()
        governance = self._make_governance_summary()
        
        panel = build_mock_oracle_director_panel(coverage, governance)
        
        assert "scenario_count" in panel
        assert panel["scenario_count"] == 5

    def test_director_panel_has_broken_count(self):
        """Director panel includes broken_scenario_count."""
        coverage = self._make_coverage_view()
        governance = self._make_governance_summary(broken_count=2)
        
        panel = build_mock_oracle_director_panel(coverage, governance)
        
        assert "broken_scenario_count" in panel
        assert panel["broken_scenario_count"] == 2

    def test_director_panel_has_impacted_metrics(self):
        """Director panel includes impacted_metrics."""
        coverage = self._make_coverage_view()
        governance = self._make_governance_summary(impacted_metrics=["verified", "failed"])
        
        panel = build_mock_oracle_director_panel(coverage, governance)
        
        assert "impacted_metrics" in panel
        assert panel["impacted_metrics"] == ["verified", "failed"]

    def test_director_panel_has_headline(self):
        """Director panel includes headline."""
        coverage = self._make_coverage_view()
        governance = self._make_governance_summary()
        
        panel = build_mock_oracle_director_panel(coverage, governance)
        
        assert "headline" in panel
        assert len(panel["headline"]) > 0

    def test_director_panel_headline_broken(self):
        """Headline mentions broken scenarios when present."""
        coverage = self._make_coverage_view()
        governance = self._make_governance_summary(
            status=GOVERNANCE_STATUS_BLOCK,
            broken_count=1,
            impacted_metrics=["verified"],
        )
        
        panel = build_mock_oracle_director_panel(coverage, governance)
        
        assert "critical drift" in panel["headline"].lower()
        assert "1" in panel["headline"]

    def test_director_panel_headline_drifted(self):
        """Headline mentions drifted scenarios when present."""
        coverage = self._make_coverage_view()
        governance = self._make_governance_summary(
            status=GOVERNANCE_STATUS_ATTENTION,
            drifted_count=2,
            impacted_metrics=["failed"],
        )
        
        panel = build_mock_oracle_director_panel(coverage, governance)
        
        assert "minor drift" in panel["headline"].lower()
        assert "2" in panel["headline"]

    def test_director_panel_headline_under_tested(self):
        """Headline mentions under-tested metrics when present."""
        coverage = self._make_coverage_view()
        governance = self._make_governance_summary()
        
        panel = build_mock_oracle_director_panel(coverage, governance)
        
        assert "under-tested" in panel["headline"].lower() or "coverage" in panel["headline"].lower()

    def test_director_panel_has_metric_counts(self):
        """Director panel includes metric exercise counts."""
        coverage = self._make_coverage_view()
        governance = self._make_governance_summary()
        
        panel = build_mock_oracle_director_panel(coverage, governance)
        
        assert "metrics_well_exercised_count" in panel
        assert "metrics_under_tested_count" in panel
        assert panel["metrics_well_exercised_count"] == 2
        assert panel["metrics_under_tested_count"] == 1


class TestIntegrationGovernancePipeline:
    """Integration tests for the full governance pipeline."""

    def test_full_pipeline_in_contract(self):
        """Full pipeline works when all scenarios are IN_CONTRACT."""
        # Use NC scenarios for guaranteed IN_CONTRACT status
        oracle = MockVerifiableOracle(MockOracleConfig(negative_control=True, seed=42))
        contract = export_mock_oracle_contract()
        
        # Create summaries for multiple scenarios
        summaries = []
        for i in range(3):
            outcomes = [oracle.verify(f"test_{i}_{j}") for j in range(100)]
            summary = summarize_scenario_results("nc_baseline", outcomes)
            # Use wider thresholds to account for sampling variance
            drift_report = detect_scenario_drift(
                contract, summary, warning_threshold=6.0, broken_threshold=15.0
            )
            summary["drift_report"] = drift_report
            summaries.append(summary)
        
        # Build coverage view
        coverage_view = build_metric_scenario_coverage_view(summaries)
        
        # Generate governance summary
        governance_summary = summarize_mock_oracle_drift_for_governance(coverage_view)
        
        # Build director panel
        panel = build_mock_oracle_director_panel(coverage_view, governance_summary)
        
        # Verify pipeline
        assert governance_summary["status"] == GOVERNANCE_STATUS_OK
        assert panel["status_light"] == STATUS_LIGHT_GREEN
        assert panel["scenario_count"] == 3

    def test_full_pipeline_with_drift(self):
        """Full pipeline correctly identifies drift."""
        # Create a summary with artificial drift
        summary = {
            "scenario_name": "test_scenario",
            "empirical_distribution": {
                "verified": 75.0,  # +15% drift (exceeds broken threshold)
                "failed": 15.0,
                "abstain": 10.0,
                "timeout": 8.0,
                "error": 4.0,
                "crash": 3.0,
                "negative_control": 0.0,
            },
            "expected_distribution": {
                "verified": 60.0,
                "failed": 15.0,
                "abstain": 10.0,
                "timeout": 8.0,
                "error": 4.0,
                "crash": 3.0,
                "negative_control": 0.0,
            },
            "deltas": {
                "verified": 15.0,
                "failed": 0.0,
                "abstain": 0.0,
                "timeout": 0.0,
                "error": 0.0,
                "crash": 0.0,
                "negative_control": 0.0,
            },
            "max_delta": 15.0,
            "sample_count": 100,
        }
        
        contract = export_mock_oracle_contract()
        drift_report = detect_scenario_drift(contract, summary, warning_threshold=3.0, broken_threshold=10.0)
        summary["drift_report"] = drift_report
        
        # Build coverage view
        coverage_view = build_metric_scenario_coverage_view([summary])
        
        # Generate governance summary
        governance_summary = summarize_mock_oracle_drift_for_governance(coverage_view)
        
        # Build director panel
        panel = build_mock_oracle_director_panel(coverage_view, governance_summary)
        
        # Verify drift detected
        assert drift_report["status"] == DRIFT_STATUS_BROKEN
        assert governance_summary["status"] == GOVERNANCE_STATUS_BLOCK
        assert panel["status_light"] == STATUS_LIGHT_RED
        assert "verified" in governance_summary["metrics_impacted_by_drift"]

    def test_pipeline_handles_empty_summaries(self):
        """Pipeline handles empty scenario summaries gracefully."""
        coverage_view = build_metric_scenario_coverage_view([])
        
        assert coverage_view["total_scenarios"] == 0
        assert len(coverage_view["metrics_well_exercised"]) == 0
        
        governance_summary = summarize_mock_oracle_drift_for_governance(coverage_view)
        
        assert governance_summary["status"] == GOVERNANCE_STATUS_OK
        assert governance_summary["has_broken_scenarios"] is False


class TestGovernanceConstants:
    """Tests for governance status constants."""

    def test_governance_status_values(self):
        """Governance status constants have expected values."""
        assert GOVERNANCE_STATUS_OK == "OK"
        assert GOVERNANCE_STATUS_ATTENTION == "ATTENTION"
        assert GOVERNANCE_STATUS_BLOCK == "BLOCK"

    def test_status_light_values(self):
        """Status light constants have expected values."""
        assert STATUS_LIGHT_GREEN == "GREEN"
        assert STATUS_LIGHT_YELLOW == "YELLOW"
        assert STATUS_LIGHT_RED == "RED"

    def test_verification_metrics_defined(self):
        """VERIFICATION_METRICS includes all expected buckets."""
        expected = ["verified", "failed", "abstain", "timeout", "error", "crash", "negative_control"]
        assert set(VERIFICATION_METRICS) == set(expected)

    def test_min_scenarios_threshold(self):
        """MIN_SCENARIOS_FOR_WELL_EXERCISED is reasonable."""
        assert MIN_SCENARIOS_FOR_WELL_EXERCISED >= 2
        assert MIN_SCENARIOS_FOR_WELL_EXERCISED <= 10

