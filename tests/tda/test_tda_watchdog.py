"""
TDA Watchdog Tests — Phase VI

Operation CORTEX: Phase VI Auto-Watchdog & Global Health Coupler
=================================================================

Tests for:
1. Synthetic log files → correct status + exit codes
2. Configuration loading
3. Alert evaluation
4. Error handling when logs are missing/corrupt
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Ensure imports work
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.tda_watchdog import (
    # Constants
    EXIT_OK,
    EXIT_ATTENTION,
    EXIT_ALERT,
    WATCHDOG_REPORT_SCHEMA_VERSION,
    # Data structures
    Alert,
    WatchdogReport,
    WatchdogConfig,
    # Functions
    load_config,
    load_snapshot,
    load_snapshots_from_glob,
    aggregate_snapshots,
    evaluate_alerts,
    determine_status,
    generate_watchdog_report,
)
from backend.health.tda_adapter import (
    TDA_STATUS_OK,
    TDA_STATUS_ATTENTION,
    TDA_STATUS_ALERT,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def healthy_snapshots() -> List[Dict[str, Any]]:
    """List of healthy governance snapshots."""
    return [
        {
            "cycle_count": 50,
            "block_rate": 0.0,
            "warn_rate": 0.0,
            "mean_hss": 0.8,
            "hss_trend": "stable",
            "golden_alignment": "ALIGNED",
            "exception_windows_active": 0,
            "governance_signal": "HEALTHY",
        },
        {
            "cycle_count": 50,
            "block_rate": 0.02,
            "warn_rate": 0.02,
            "mean_hss": 0.75,
            "hss_trend": "stable",
            "golden_alignment": "ALIGNED",
            "exception_windows_active": 0,
            "governance_signal": "HEALTHY",
        },
    ]


@pytest.fixture
def attention_snapshots() -> List[Dict[str, Any]]:
    """List of attention-level governance snapshots."""
    return [
        {
            "cycle_count": 50,
            "block_rate": 0.08,
            "warn_rate": 0.05,
            "mean_hss": 0.55,
            "hss_trend": "degrading",
            "golden_alignment": "DRIFTING",
            "exception_windows_active": 1,
            "governance_signal": "DEGRADED",
        },
    ]


@pytest.fixture
def alert_snapshots() -> List[Dict[str, Any]]:
    """List of alert-level governance snapshots."""
    return [
        {
            "cycle_count": 100,
            "block_rate": 0.25,
            "warn_rate": 0.10,
            "mean_hss": 0.35,
            "hss_trend": "degrading",
            "golden_alignment": "BROKEN",
            "exception_windows_active": 2,
            "governance_signal": "CRITICAL",
        },
    ]


@pytest.fixture
def default_config() -> WatchdogConfig:
    """Default watchdog configuration."""
    return WatchdogConfig()


# ============================================================================
# Test: WatchdogConfig
# ============================================================================

class TestWatchdogConfig:
    """Tests for watchdog configuration loading."""

    def test_default_config_has_sensible_values(self):
        """Default config has reasonable default values."""
        config = WatchdogConfig()

        assert config.block_rate_max_ok == 0.05
        assert config.block_rate_max_attention == 0.15
        assert config.mean_hss_min_ok == 0.6
        assert config.mean_hss_min_attention == 0.4
        assert config.min_runs_for_strong_signal == 10

    def test_load_config_from_yaml(self, tmp_path: Path):
        """Loads configuration from YAML file."""
        config_yaml = """
schema_version: "1.0.0"
block_rate:
  max_ok: 0.10
  max_attention: 0.20
mean_hss:
  min_ok: 0.7
  min_attention: 0.5
signal_strength:
  min_runs_for_strong_signal: 20
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_yaml)

        config = load_config(config_path)

        assert config.block_rate_max_ok == 0.10
        assert config.block_rate_max_attention == 0.20
        assert config.mean_hss_min_ok == 0.7
        assert config.min_runs_for_strong_signal == 20

    def test_load_config_missing_file_returns_defaults(self, tmp_path: Path):
        """Missing config file returns default configuration."""
        config = load_config(tmp_path / "nonexistent.yaml")

        assert config.block_rate_max_ok == 0.05  # Default value


# ============================================================================
# Test: Snapshot Loading
# ============================================================================

class TestSnapshotLoading:
    """Tests for snapshot loading functions."""

    def test_load_snapshot_direct(self, tmp_path: Path):
        """Loads snapshot directly from JSON file."""
        snapshot = {"block_rate": 0.1, "mean_hss": 0.7}
        path = tmp_path / "snapshot.json"
        path.write_text(json.dumps(snapshot))

        loaded = load_snapshot(path)

        assert loaded == snapshot

    def test_load_snapshot_wrapped(self, tmp_path: Path):
        """Loads snapshot from wrapped format."""
        data = {
            "governance_snapshot": {"block_rate": 0.1, "mean_hss": 0.7}
        }
        path = tmp_path / "snapshot.json"
        path.write_text(json.dumps(data))

        loaded = load_snapshot(path)

        assert loaded["block_rate"] == 0.1

    def test_load_snapshot_invalid_json(self, tmp_path: Path):
        """Returns None for invalid JSON."""
        path = tmp_path / "invalid.json"
        path.write_text("not valid json")

        loaded = load_snapshot(path)

        assert loaded is None

    def test_load_snapshots_from_glob(self, tmp_path: Path):
        """Loads multiple snapshots from glob pattern."""
        for i in range(3):
            path = tmp_path / f"snapshot_{i}.json"
            path.write_text(json.dumps({"block_rate": i * 0.1}))

        snapshots = load_snapshots_from_glob(str(tmp_path / "*.json"))

        assert len(snapshots) == 3


# ============================================================================
# Test: Snapshot Aggregation
# ============================================================================

class TestSnapshotAggregation:
    """Tests for snapshot aggregation."""

    def test_aggregate_empty_returns_defaults(self):
        """Aggregating empty list returns default values."""
        aggregated = aggregate_snapshots([])

        assert aggregated["cycle_count"] == 0
        assert aggregated["block_rate"] == 0.0

    def test_aggregate_single_returns_same(self, healthy_snapshots: List[Dict[str, Any]]):
        """Aggregating single snapshot returns same values."""
        single = [healthy_snapshots[0]]
        aggregated = aggregate_snapshots(single)

        assert aggregated == healthy_snapshots[0]

    def test_aggregate_multiple_sums_cycles(self, healthy_snapshots: List[Dict[str, Any]]):
        """Aggregating multiple snapshots sums cycle counts."""
        aggregated = aggregate_snapshots(healthy_snapshots)

        expected_cycles = sum(s["cycle_count"] for s in healthy_snapshots)
        assert aggregated["cycle_count"] == expected_cycles

    def test_aggregate_uses_latest_trend(self, healthy_snapshots: List[Dict[str, Any]]):
        """Aggregation uses most recent snapshot's trend."""
        snapshots = [
            {"cycle_count": 50, "block_rate": 0.0, "hss_trend": "improving"},
            {"cycle_count": 50, "block_rate": 0.0, "hss_trend": "degrading"},
        ]
        aggregated = aggregate_snapshots(snapshots)

        assert aggregated["hss_trend"] == "degrading"


# ============================================================================
# Test: Alert Evaluation
# ============================================================================

class TestAlertEvaluation:
    """Tests for alert evaluation logic."""

    def test_no_alerts_for_healthy_snapshot(
        self,
        healthy_snapshots: List[Dict[str, Any]],
        default_config: WatchdogConfig,
    ):
        """Healthy snapshot generates no alerts (except possibly weak signal)."""
        aggregated = aggregate_snapshots(healthy_snapshots)
        alerts = evaluate_alerts(aggregated, default_config)

        # Only weak signal alert if cycles < min_runs
        non_weak_alerts = [a for a in alerts if "WEAK" not in a.code]
        assert len(non_weak_alerts) == 0

    def test_high_block_rate_generates_alert(self, default_config: WatchdogConfig):
        """High block rate generates ALERT severity alert."""
        snapshot = {
            "block_rate": 0.20,
            "cycle_count": 100,
        }

        alerts = evaluate_alerts(snapshot, default_config)

        high_alerts = [a for a in alerts if a.code == default_config.block_rate_code_high]
        assert len(high_alerts) == 1
        assert high_alerts[0].severity == "ALERT"

    def test_elevated_block_rate_generates_attention(self, default_config: WatchdogConfig):
        """Elevated block rate generates ATTENTION severity alert."""
        snapshot = {
            "block_rate": 0.10,
            "cycle_count": 100,
        }

        alerts = evaluate_alerts(snapshot, default_config)

        elevated_alerts = [a for a in alerts if a.code == default_config.block_rate_code_elevated]
        assert len(elevated_alerts) == 1
        assert elevated_alerts[0].severity == "ATTENTION"

    def test_low_hss_generates_alert(self, default_config: WatchdogConfig):
        """Low mean HSS generates ATTENTION or ALERT."""
        snapshot = {
            "block_rate": 0.0,
            "mean_hss": 0.35,
            "cycle_count": 100,
        }

        alerts = evaluate_alerts(snapshot, default_config)

        hss_alerts = [a for a in alerts if "HSS" in a.code]
        assert len(hss_alerts) >= 1

    def test_degrading_trend_generates_attention(self, default_config: WatchdogConfig):
        """Degrading HSS trend generates ATTENTION alert."""
        snapshot = {
            "block_rate": 0.0,
            "hss_trend": "DEGRADING",
            "cycle_count": 100,
        }

        alerts = evaluate_alerts(snapshot, default_config)

        trend_alerts = [a for a in alerts if "TREND" in a.code]
        assert len(trend_alerts) == 1
        assert trend_alerts[0].severity == "ATTENTION"

    def test_broken_alignment_generates_alert(self, default_config: WatchdogConfig):
        """BROKEN golden alignment generates ALERT."""
        snapshot = {
            "block_rate": 0.0,
            "golden_alignment": "BROKEN",
            "cycle_count": 100,
        }

        alerts = evaluate_alerts(snapshot, default_config)

        golden_alerts = [a for a in alerts if "GOLDEN" in a.code and "BROKEN" in a.code]
        assert len(golden_alerts) == 1
        assert golden_alerts[0].severity == "ALERT"

    def test_combined_rule_block_with_degrading(self, default_config: WatchdogConfig):
        """Combined rule: high block rate + degrading trend triggers ALERT."""
        snapshot = {
            "block_rate": 0.25,
            "hss_trend": "DEGRADING",
            "cycle_count": 100,
        }

        alerts = evaluate_alerts(snapshot, default_config)

        combined_alerts = [a for a in alerts if "COMBINED" in a.code]
        assert len(combined_alerts) >= 1

    def test_weak_signal_alert_for_low_cycles(self, default_config: WatchdogConfig):
        """Low cycle count generates weak signal alert."""
        snapshot = {
            "block_rate": 0.0,
            "cycle_count": 5,  # Below min_runs_for_strong_signal
        }

        alerts = evaluate_alerts(snapshot, default_config)

        weak_alerts = [a for a in alerts if "WEAK" in a.code]
        assert len(weak_alerts) == 1


# ============================================================================
# Test: Status Determination
# ============================================================================

class TestStatusDetermination:
    """Tests for overall status determination."""

    def test_no_alerts_returns_ok(self):
        """No alerts returns OK status."""
        alerts = []
        status = determine_status(alerts)
        assert status == TDA_STATUS_OK

    def test_attention_alerts_return_attention(self):
        """Only ATTENTION alerts returns ATTENTION status."""
        alerts = [
            Alert(code="TEST", severity="ATTENTION", message="test"),
        ]
        status = determine_status(alerts)
        assert status == TDA_STATUS_ATTENTION

    def test_alert_severity_returns_alert(self):
        """Any ALERT severity returns ALERT status."""
        alerts = [
            Alert(code="TEST1", severity="ATTENTION", message="test"),
            Alert(code="TEST2", severity="ALERT", message="test"),
        ]
        status = determine_status(alerts)
        assert status == TDA_STATUS_ALERT


# ============================================================================
# Test: Report Generation
# ============================================================================

class TestReportGeneration:
    """Tests for watchdog report generation."""

    def test_report_has_schema_version(
        self,
        healthy_snapshots: List[Dict[str, Any]],
        default_config: WatchdogConfig,
    ):
        """Report includes schema version."""
        report = generate_watchdog_report(healthy_snapshots, default_config)
        assert report.schema_version == WATCHDOG_REPORT_SCHEMA_VERSION

    def test_report_includes_metrics(
        self,
        healthy_snapshots: List[Dict[str, Any]],
        default_config: WatchdogConfig,
    ):
        """Report includes metrics when configured."""
        default_config.include_metrics = True
        report = generate_watchdog_report(healthy_snapshots, default_config)
        assert report.metrics is not None
        assert "cycle_count" in report.metrics

    def test_report_to_json(
        self,
        healthy_snapshots: List[Dict[str, Any]],
        default_config: WatchdogConfig,
    ):
        """Report serializes to valid JSON."""
        report = generate_watchdog_report(healthy_snapshots, default_config)
        json_str = report.to_json()
        parsed = json.loads(json_str)

        assert parsed["schema_version"] == WATCHDOG_REPORT_SCHEMA_VERSION

    def test_healthy_snapshots_produce_ok_or_attention(
        self,
        healthy_snapshots: List[Dict[str, Any]],
        default_config: WatchdogConfig,
    ):
        """Healthy snapshots produce OK or ATTENTION status."""
        report = generate_watchdog_report(healthy_snapshots, default_config)
        # May be ATTENTION due to weak signal or non-zero block rate
        assert report.tda_status in (TDA_STATUS_OK, TDA_STATUS_ATTENTION)

    def test_alert_snapshots_produce_alert(
        self,
        alert_snapshots: List[Dict[str, Any]],
        default_config: WatchdogConfig,
    ):
        """Alert-level snapshots produce ALERT status."""
        report = generate_watchdog_report(alert_snapshots, default_config)
        assert report.tda_status == TDA_STATUS_ALERT

    def test_signal_strength_strong_for_many_cycles(
        self,
        healthy_snapshots: List[Dict[str, Any]],
        default_config: WatchdogConfig,
    ):
        """Signal strength is 'strong' for sufficient cycles."""
        # Ensure enough cycles
        for s in healthy_snapshots:
            s["cycle_count"] = 100

        report = generate_watchdog_report(healthy_snapshots, default_config)
        assert report.signal_strength == "strong"

    def test_signal_strength_weak_for_few_cycles(self, default_config: WatchdogConfig):
        """Signal strength is 'weak' for insufficient cycles."""
        snapshots = [{"cycle_count": 3, "block_rate": 0.0}]
        report = generate_watchdog_report(snapshots, default_config)
        assert report.signal_strength == "weak"


# ============================================================================
# Test: Exit Codes
# ============================================================================

class TestExitCodes:
    """Tests for exit code mapping."""

    def test_exit_ok_value(self):
        """EXIT_OK is 0."""
        assert EXIT_OK == 0

    def test_exit_attention_value(self):
        """EXIT_ATTENTION is 1."""
        assert EXIT_ATTENTION == 1

    def test_exit_alert_value(self):
        """EXIT_ALERT is 2."""
        assert EXIT_ALERT == 2


# ============================================================================
# Test: Error Handling
# ============================================================================

class TestErrorHandling:
    """Tests for error handling with missing/corrupt data."""

    def test_handles_empty_snapshot_list(self, default_config: WatchdogConfig):
        """Handles empty snapshot list gracefully."""
        # generate_watchdog_report with empty list should work
        report = generate_watchdog_report([], default_config)
        assert report.recent_runs == 0

    def test_handles_corrupt_snapshot_files(self, tmp_path: Path):
        """Skips corrupt snapshot files without crashing."""
        # Create one valid and one invalid file
        valid_path = tmp_path / "valid.json"
        valid_path.write_text(json.dumps({"block_rate": 0.1}))

        invalid_path = tmp_path / "invalid.json"
        invalid_path.write_text("not json")

        snapshots = load_snapshots_from_glob(str(tmp_path / "*.json"))

        # Should have loaded the valid one
        assert len(snapshots) == 1

    def test_handles_missing_fields_in_snapshot(self, default_config: WatchdogConfig):
        """Handles snapshots with missing fields."""
        snapshots = [{}]  # Empty snapshot
        report = generate_watchdog_report(snapshots, default_config)

        # Should not crash
        assert report is not None


# ============================================================================
# Test: Deterministic Behavior
# ============================================================================

class TestDeterministicBehavior:
    """Tests for deterministic behavior."""

    def test_same_snapshots_produce_same_report(
        self,
        healthy_snapshots: List[Dict[str, Any]],
        default_config: WatchdogConfig,
    ):
        """Same snapshots produce consistent report (except timestamp)."""
        report1 = generate_watchdog_report(healthy_snapshots, default_config)
        report2 = generate_watchdog_report(healthy_snapshots, default_config)

        # Compare everything except generated_at
        assert report1.tda_status == report2.tda_status
        assert report1.block_rate == report2.block_rate
        assert report1.mean_hss == report2.mean_hss
        assert len(report1.alerts) == len(report2.alerts)

    def test_alert_order_is_deterministic(
        self,
        alert_snapshots: List[Dict[str, Any]],
        default_config: WatchdogConfig,
    ):
        """Alerts are generated in deterministic order."""
        reports = [
            generate_watchdog_report(alert_snapshots, default_config)
            for _ in range(5)
        ]

        alert_codes = [tuple(a.code for a in r.alerts) for r in reports]

        # All should be identical
        assert len(set(alert_codes)) == 1
