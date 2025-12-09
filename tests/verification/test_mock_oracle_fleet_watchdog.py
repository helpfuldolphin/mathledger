"""
Tests for Mock Oracle Scenario Fleet Console & Regression Watchdog.

Verifies:
1. build_scenario_fleet_summary() produces correct fleet summaries
2. detect_mock_oracle_regression() correctly identifies regressions
3. Integration with coverage views

SAFEGUARD: These tests verify watchdog infrastructure — no production impact.
"""

from __future__ import annotations

from typing import Dict, List

import pytest

from backend.verification.mock_config import (
    CRITICAL_METRICS,
    FLEET_STATUS_ATTENTION,
    FLEET_STATUS_GAP,
    FLEET_STATUS_OK,
    MIN_SCENARIOS_FOR_SPECIFIED,
    MIN_SCENARIOS_FOR_WELL_EXERCISED,
    VERIFICATION_METRICS,
    build_metric_scenario_coverage_view,
    build_scenario_fleet_summary,
    detect_mock_oracle_regression,
)


class TestBuildScenarioFleetSummary:
    """Tests for build_scenario_fleet_summary() function."""

    def _make_coverage_view(
        self,
        total_scenarios: int = 5,
        well_exercised: List[str] = None,
        under_tested: List[str] = None,
        broken_scenarios: Dict[str, List[str]] = None,
    ) -> Dict:
        """Helper to create a coverage view."""
        if well_exercised is None:
            well_exercised = ["verified", "failed"]
        if under_tested is None:
            under_tested = ["abstain"]
        if broken_scenarios is None:
            broken_scenarios = {}
        
        metrics = {}
        for metric in VERIFICATION_METRICS:
            scenarios_tested = (
                MIN_SCENARIOS_FOR_WELL_EXERCISED + 1
                if metric in well_exercised
                else 1
            )
            metrics[metric] = {
                "scenarios_tested": [f"scenario_{i}" for i in range(scenarios_tested)],
                "drift_free_count": scenarios_tested - len(broken_scenarios.get(metric, [])),
                "drifted_scenarios": [],
                "broken_scenarios": broken_scenarios.get(metric, []),
            }
        
        return {
            "metrics": metrics,
            "metrics_well_exercised": well_exercised,
            "metrics_under_tested": under_tested,
            "total_scenarios": total_scenarios,
        }

    def test_fleet_summary_has_total_scenarios(self):
        """Fleet summary includes total_scenarios."""
        view = self._make_coverage_view(total_scenarios=10)
        
        summary = build_scenario_fleet_summary(view)
        
        assert "total_scenarios" in summary
        assert summary["total_scenarios"] == 10

    def test_fleet_summary_has_metrics_well_exercised(self):
        """Fleet summary includes metrics_well_exercised."""
        view = self._make_coverage_view(well_exercised=["verified", "failed", "abstain"])
        
        summary = build_scenario_fleet_summary(view)
        
        assert "metrics_well_exercised" in summary
        assert "verified" in summary["metrics_well_exercised"]
        assert "failed" in summary["metrics_well_exercised"]

    def test_fleet_summary_has_metrics_underspecified(self):
        """Fleet summary includes metrics_underspecified."""
        view = self._make_coverage_view()
        
        summary = build_scenario_fleet_summary(view)
        
        assert "metrics_underspecified" in summary
        assert isinstance(summary["metrics_underspecified"], list)

    def test_fleet_summary_has_fleet_status(self):
        """Fleet summary includes fleet_status."""
        view = self._make_coverage_view()
        
        summary = build_scenario_fleet_summary(view)
        
        assert "fleet_status" in summary
        assert summary["fleet_status"] in [FLEET_STATUS_OK, FLEET_STATUS_ATTENTION, FLEET_STATUS_GAP]

    def test_fleet_summary_has_summary_text(self):
        """Fleet summary includes summary_text."""
        view = self._make_coverage_view()
        
        summary = build_scenario_fleet_summary(view)
        
        assert "summary_text" in summary
        assert len(summary["summary_text"]) > 0

    def test_fleet_status_ok_when_all_well_exercised(self):
        """Fleet status is OK when all metrics are well-exercised."""
        well_exercised = list(VERIFICATION_METRICS)
        view = self._make_coverage_view(
            well_exercised=well_exercised,
            under_tested=[],
        )
        
        summary = build_scenario_fleet_summary(view)
        
        assert summary["fleet_status"] == FLEET_STATUS_OK

    def test_fleet_status_attention_when_some_underspecified(self):
        """Fleet status is ATTENTION when some non-critical metrics are underspecified."""
        # Use non-critical metrics to avoid GAP status
        view = self._make_coverage_view(
            well_exercised=["verified", "failed", "abstain"],  # All critical metrics well-exercised
            under_tested=["timeout", "error"],  # Non-critical metrics under-tested
        )
        
        summary = build_scenario_fleet_summary(view)
        
        assert summary["fleet_status"] == FLEET_STATUS_ATTENTION

    def test_fleet_status_gap_when_critical_underspecified(self):
        """Fleet status is GAP when critical metrics are underspecified."""
        view = self._make_coverage_view(
            well_exercised=[],
            under_tested=list(CRITICAL_METRICS),
        )
        
        summary = build_scenario_fleet_summary(view)
        
        assert summary["fleet_status"] == FLEET_STATUS_GAP

    def test_fleet_status_gap_when_critical_has_many_broken(self):
        """Fleet status is GAP when critical metrics have many broken scenarios."""
        # Create a view where verified has many broken scenarios
        broken_scenarios = {
            "verified": ["broken_1", "broken_2", "broken_3"],
        }
        metrics = {}
        for metric in VERIFICATION_METRICS:
            scenarios_tested = ["scenario_1", "scenario_2", "scenario_3", "scenario_4"]
            metrics[metric] = {
                "scenarios_tested": scenarios_tested,
                "drift_free_count": 1,
                "drifted_scenarios": [],
                "broken_scenarios": broken_scenarios.get(metric, []),
            }
        
        view = {
            "metrics": metrics,
            "metrics_well_exercised": VERIFICATION_METRICS,
            "metrics_under_tested": [],
            "total_scenarios": 4,
        }
        
        summary = build_scenario_fleet_summary(view)
        
        assert summary["fleet_status"] == FLEET_STATUS_GAP

    def test_underspecified_includes_low_scenario_count(self):
        """Metrics with < MIN_SCENARIOS_FOR_SPECIFIED are underspecified."""
        metrics = {}
        for metric in VERIFICATION_METRICS:
            # Give each metric only 1 scenario (below threshold)
            metrics[metric] = {
                "scenarios_tested": ["scenario_1"],
                "drift_free_count": 1,
                "drifted_scenarios": [],
                "broken_scenarios": [],
            }
        
        view = {
            "metrics": metrics,
            "metrics_well_exercised": [],
            "metrics_under_tested": VERIFICATION_METRICS,
            "total_scenarios": 1,
        }
        
        summary = build_scenario_fleet_summary(view)
        
        # All metrics should be underspecified
        assert len(summary["metrics_underspecified"]) == len(VERIFICATION_METRICS)

    def test_summary_text_varies_by_status(self):
        """Summary text differs based on fleet status."""
        # OK status
        view_ok = self._make_coverage_view(
            well_exercised=VERIFICATION_METRICS,
            under_tested=[],
        )
        summary_ok = build_scenario_fleet_summary(view_ok)
        
        # ATTENTION status (non-critical metrics underspecified)
        view_attention = self._make_coverage_view(
            well_exercised=["verified", "failed", "abstain"],  # All critical well-exercised
            under_tested=["timeout", "error"],  # Non-critical underspecified
        )
        summary_attention = build_scenario_fleet_summary(view_attention)
        
        # GAP status
        view_gap = self._make_coverage_view(
            well_exercised=[],
            under_tested=CRITICAL_METRICS,
        )
        summary_gap = build_scenario_fleet_summary(view_gap)
        
        # All should have different text
        assert "adequate" in summary_ok["summary_text"].lower()
        assert "monitoring" in summary_attention["summary_text"].lower() or "underspecified" in summary_attention["summary_text"].lower()
        assert "gap" in summary_gap["summary_text"].lower() or "required" in summary_gap["summary_text"].lower()


class TestDetectMockOracleRegression:
    """Tests for detect_mock_oracle_regression() function."""

    def _make_coverage_view(
        self,
        broken_scenarios: Dict[str, List[str]] = None,
        drifted_scenarios: Dict[str, List[str]] = None,
        scenarios_per_metric: int = 5,
    ) -> Dict:
        """Helper to create a coverage view."""
        if broken_scenarios is None:
            broken_scenarios = {}
        if drifted_scenarios is None:
            drifted_scenarios = {}
        
        metrics = {}
        for metric in VERIFICATION_METRICS:
            scenarios_tested = [f"scenario_{i}" for i in range(scenarios_per_metric)]
            metrics[metric] = {
                "scenarios_tested": scenarios_tested,
                "drift_free_count": scenarios_per_metric - len(broken_scenarios.get(metric, [])) - len(drifted_scenarios.get(metric, [])),
                "drifted_scenarios": drifted_scenarios.get(metric, []),
                "broken_scenarios": broken_scenarios.get(metric, []),
            }
        
        return {
            "metrics": metrics,
            "metrics_well_exercised": VERIFICATION_METRICS,
            "metrics_under_tested": [],
            "total_scenarios": scenarios_per_metric,
        }

    def test_regression_detection_has_regression_detected(self):
        """Regression detection includes regression_detected flag."""
        previous = self._make_coverage_view()
        current = self._make_coverage_view()
        
        result = detect_mock_oracle_regression(previous, current)
        
        assert "regression_detected" in result
        assert isinstance(result["regression_detected"], bool)

    def test_regression_detection_has_affected_metrics(self):
        """Regression detection includes affected_metrics list."""
        previous = self._make_coverage_view()
        current = self._make_coverage_view()
        
        result = detect_mock_oracle_regression(previous, current)
        
        assert "affected_metrics" in result
        assert isinstance(result["affected_metrics"], list)

    def test_regression_detection_has_notes(self):
        """Regression detection includes notes list."""
        previous = self._make_coverage_view()
        current = self._make_coverage_view()
        
        result = detect_mock_oracle_regression(previous, current)
        
        assert "notes" in result
        assert isinstance(result["notes"], list)

    def test_no_regression_when_unchanged(self):
        """No regression detected when views are identical."""
        view = self._make_coverage_view()
        
        result = detect_mock_oracle_regression(view, view)
        
        assert result["regression_detected"] is False
        assert len(result["affected_metrics"]) == 0
        assert len(result["notes"]) == 0

    def test_regression_detected_when_broken_increases(self):
        """Regression detected when broken scenarios increase."""
        previous = self._make_coverage_view(
            broken_scenarios={"verified": ["broken_1"]},
        )
        current = self._make_coverage_view(
            broken_scenarios={"verified": ["broken_1", "broken_2", "broken_3"]},
        )
        
        result = detect_mock_oracle_regression(previous, current)
        
        assert result["regression_detected"] is True
        assert "verified" in result["affected_metrics"]
        assert len(result["notes"]) > 0
        assert any("broken" in note.lower() for note in result["notes"])

    def test_regression_detected_when_drifted_increases_significantly(self):
        """Regression detected when drifted scenarios increase significantly."""
        previous = self._make_coverage_view(
            drifted_scenarios={"failed": ["drifted_1"]},
        )
        current = self._make_coverage_view(
            drifted_scenarios={"failed": ["drifted_1", "drifted_2", "drifted_3", "drifted_4"]},
        )
        
        result = detect_mock_oracle_regression(previous, current)
        
        assert result["regression_detected"] is True
        assert "failed" in result["affected_metrics"]
        assert any("drifted" in note.lower() for note in result["notes"])

    def test_no_regression_when_drifted_increases_slightly(self):
        """No regression when drifted scenarios increase slightly."""
        previous = self._make_coverage_view(
            drifted_scenarios={"failed": ["drifted_1"]},
        )
        current = self._make_coverage_view(
            drifted_scenarios={"failed": ["drifted_1", "drifted_2"]},
        )
        
        result = detect_mock_oracle_regression(previous, current)
        
        # Small increase (1) should not trigger regression
        assert result["regression_detected"] is False

    def test_regression_detected_when_metric_becomes_under_tested(self):
        """Regression detected when metric becomes under-tested."""
        previous = self._make_coverage_view(scenarios_per_metric=MIN_SCENARIOS_FOR_WELL_EXERCISED + 1)
        current = self._make_coverage_view(scenarios_per_metric=MIN_SCENARIOS_FOR_WELL_EXERCISED - 1)
        
        result = detect_mock_oracle_regression(previous, current)
        
        assert result["regression_detected"] is True
        assert len(result["affected_metrics"]) > 0
        assert any("under-tested" in note.lower() for note in result["notes"])

    def test_regression_notes_are_descriptive(self):
        """Regression notes provide clear descriptions."""
        previous = self._make_coverage_view(
            broken_scenarios={"verified": []},
        )
        current = self._make_coverage_view(
            broken_scenarios={"verified": ["broken_1", "broken_2"]},
        )
        
        result = detect_mock_oracle_regression(previous, current)
        
        assert len(result["notes"]) > 0
        note = result["notes"][0]
        assert "verified" in note.lower()
        assert "broken" in note.lower()
        assert "0" in note or "2" in note  # Should mention counts

    def test_regression_tracks_multiple_affected_metrics(self):
        """Regression detection tracks multiple affected metrics."""
        previous = self._make_coverage_view(
            broken_scenarios={"verified": [], "failed": []},
        )
        current = self._make_coverage_view(
            broken_scenarios={"verified": ["broken_1"], "failed": ["broken_1", "broken_2"]},
        )
        
        result = detect_mock_oracle_regression(previous, current)
        
        assert result["regression_detected"] is True
        assert "verified" in result["affected_metrics"]
        assert "failed" in result["affected_metrics"]
        assert len(result["affected_metrics"]) == 2

    def test_regression_affected_metrics_sorted(self):
        """Affected metrics list is sorted."""
        previous = self._make_coverage_view(
            broken_scenarios={"z_metric": [], "a_metric": []},
        )
        current = self._make_coverage_view(
            broken_scenarios={"z_metric": ["broken_1"], "a_metric": ["broken_1"]},
        )
        
        # Use actual metric names
        previous = self._make_coverage_view(
            broken_scenarios={"verified": [], "abstain": []},
        )
        current = self._make_coverage_view(
            broken_scenarios={"verified": ["broken_1"], "abstain": ["broken_1"]},
        )
        
        result = detect_mock_oracle_regression(previous, current)
        
        assert result["affected_metrics"] == sorted(result["affected_metrics"])


class TestIntegrationFleetWatchdog:
    """Integration tests for fleet summary and regression detection."""

    def test_fleet_summary_with_real_coverage_view(self):
        """Fleet summary works with real coverage view structure."""
        # Create a realistic coverage view
        metrics = {}
        for metric in VERIFICATION_METRICS:
            scenarios_tested = (
                [f"scenario_{i}" for i in range(5)]
                if metric in ["verified", "failed"]
                else [f"scenario_{i}" for i in range(1)]
            )
            metrics[metric] = {
                "scenarios_tested": scenarios_tested,
                "drift_free_count": len(scenarios_tested),
                "drifted_scenarios": [],
                "broken_scenarios": [],
            }
        
        view = {
            "metrics": metrics,
            "metrics_well_exercised": ["verified", "failed"],
            "metrics_under_tested": [m for m in VERIFICATION_METRICS if m not in ["verified", "failed"]],
            "total_scenarios": 5,
        }
        
        summary = build_scenario_fleet_summary(view)
        
        assert summary["total_scenarios"] == 5
        assert summary["fleet_status"] in [FLEET_STATUS_OK, FLEET_STATUS_ATTENTION, FLEET_STATUS_GAP]

    def test_regression_detection_with_real_views(self):
        """Regression detection works with real coverage views."""
        # Previous: all good
        previous = {
            "metrics": {
                metric: {
                    "scenarios_tested": ["scenario_1", "scenario_2"],
                    "drift_free_count": 2,
                    "drifted_scenarios": [],
                    "broken_scenarios": [],
                }
                for metric in VERIFICATION_METRICS
            },
            "metrics_well_exercised": VERIFICATION_METRICS,
            "metrics_under_tested": [],
            "total_scenarios": 2,
        }
        
        # Current: some broken
        current = {
            "metrics": {
                metric: {
                    "scenarios_tested": ["scenario_1", "scenario_2"],
                    "drift_free_count": 1 if metric == "verified" else 2,
                    "drifted_scenarios": [],
                    "broken_scenarios": ["scenario_1"] if metric == "verified" else [],
                }
                for metric in VERIFICATION_METRICS
            },
            "metrics_well_exercised": VERIFICATION_METRICS,
            "metrics_under_tested": [],
            "total_scenarios": 2,
        }
        
        result = detect_mock_oracle_regression(previous, current)
        
        assert result["regression_detected"] is True
        assert "verified" in result["affected_metrics"]

    def test_fleet_summary_and_regression_integration(self):
        """Fleet summary and regression detection work together."""
        # Create two views
        view1 = {
            "metrics": {
                metric: {
                    "scenarios_tested": ["scenario_1"],
                    "drift_free_count": 1,
                    "drifted_scenarios": [],
                    "broken_scenarios": [],
                }
                for metric in VERIFICATION_METRICS
            },
            "metrics_well_exercised": [],
            "metrics_under_tested": VERIFICATION_METRICS,
            "total_scenarios": 1,
        }
        
        view2 = {
            "metrics": {
                metric: {
                    "scenarios_tested": ["scenario_1"],
                    "drift_free_count": 0,
                    "drifted_scenarios": [],
                    "broken_scenarios": ["scenario_1"],
                }
                for metric in VERIFICATION_METRICS
            },
            "metrics_well_exercised": [],
            "metrics_under_tested": VERIFICATION_METRICS,
            "total_scenarios": 1,
        }
        
        # Get fleet summaries
        summary1 = build_scenario_fleet_summary(view1)
        summary2 = build_scenario_fleet_summary(view2)
        
        # Detect regression
        regression = detect_mock_oracle_regression(view1, view2)
        
        # Verify integration
        assert summary1["total_scenarios"] == summary2["total_scenarios"]
        assert regression["regression_detected"] is True


class TestFleetConstants:
    """Tests for fleet status constants."""

    def test_fleet_status_values(self):
        """Fleet status constants have expected values."""
        assert FLEET_STATUS_OK == "OK"
        assert FLEET_STATUS_ATTENTION == "ATTENTION"
        assert FLEET_STATUS_GAP == "GAP"

    def test_critical_metrics_defined(self):
        """CRITICAL_METRICS includes expected metrics."""
        assert "verified" in CRITICAL_METRICS
        assert "failed" in CRITICAL_METRICS
        assert "abstain" in CRITICAL_METRICS

    def test_min_scenarios_for_specified(self):
        """MIN_SCENARIOS_FOR_SPECIFIED is reasonable."""
        assert MIN_SCENARIOS_FOR_SPECIFIED >= 1
        assert MIN_SCENARIOS_FOR_SPECIFIED <= 5

