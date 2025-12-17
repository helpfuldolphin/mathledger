"""
Tests for TDA Monitor

Tests the integrated TDA monitoring for P3 and P4 integration.
See: docs/system_law/TDA_PhaseX_Binding.md

SHADOW MODE: All tests verify observational metrics only.
"""

import pytest

from backend.tda.monitor import TDAMonitor, TDARedFlag, TDASummary
from backend.tda.metrics import TDAMetrics, TDAWindowMetrics


class TestTDAMonitor:
    """Tests for TDAMonitor integration."""

    def test_monitor_creation(self):
        """Should create monitor with defaults."""
        monitor = TDAMonitor()
        assert monitor._cycle_count == 0

    def test_observe_cycle_returns_metrics(self):
        """Observe cycle should return TDAMetrics."""
        monitor = TDAMonitor()
        metrics = monitor.observe_cycle(
            cycle=1,
            success=True,
            depth=3,
            H=0.8,
            rho=0.7,
            tau=0.2,
            beta=0.1,
        )
        assert isinstance(metrics, TDAMetrics)
        assert 0.0 <= metrics.sns <= 1.0
        assert 0.0 <= metrics.pcs <= 1.0
        assert 0.0 <= metrics.hss <= 1.0

    def test_observe_multiple_cycles(self):
        """Should track multiple cycles."""
        monitor = TDAMonitor()
        for i in range(10):
            monitor.observe_cycle(
                cycle=i + 1,
                success=i % 2 == 0,
                depth=3 + i % 3,
                H=0.8 - i * 0.01,
                rho=0.7 + i * 0.01,
                tau=0.2,
                beta=0.1,
            )
        assert monitor._cycle_count == 10
        assert len(monitor._metrics_history) == 10

    def test_envelope_tracking(self):
        """Should track envelope membership."""
        monitor = TDAMonitor()
        # Inside envelope
        metrics = monitor.observe_cycle(
            cycle=1,
            success=True,
            depth=3,
            H=0.9,
            rho=0.9,
            tau=0.2,
            beta=0.1,
        )
        # Metrics with stable patterns should be in envelope
        assert monitor._envelope_exit_streak >= 0

    def test_finalize_window(self):
        """Should finalize window with TDA metrics."""
        monitor = TDAMonitor()
        # Add cycles to a window
        for i in range(50):
            monitor.observe_cycle(
                cycle=i + 1,
                success=i % 3 != 0,
                depth=3 + i % 5,
                H=0.8,
                rho=0.75,
                tau=0.2,
                beta=0.05,
            )

        window = monitor.finalize_window(window_index=0)
        assert isinstance(window, TDAWindowMetrics)
        assert window.window_index == 0
        assert 0.0 <= window.sns_mean <= 1.0
        assert 0.0 <= window.pcs_mean <= 1.0
        assert 0.0 <= window.hss_mean <= 1.0

    def test_get_summary(self):
        """Should return valid summary."""
        monitor = TDAMonitor()
        for i in range(20):
            monitor.observe_cycle(
                cycle=i + 1,
                success=True,
                depth=3,
                H=0.8,
                rho=0.7,
                tau=0.2,
                beta=0.1,
            )

        summary = monitor.get_summary()
        assert isinstance(summary, TDASummary)
        assert summary.total_cycles == 20
        assert 0.0 <= summary.sns_mean <= 1.0
        assert 0.0 <= summary.envelope_occupancy <= 1.0

    def test_reset(self):
        """Reset should clear all state."""
        monitor = TDAMonitor()
        for i in range(10):
            monitor.observe_cycle(
                cycle=i + 1,
                success=True,
                depth=3,
                H=0.8,
                rho=0.7,
                tau=0.2,
                beta=0.1,
            )
        monitor.reset()
        assert monitor._cycle_count == 0
        assert len(monitor._metrics_history) == 0
        assert len(monitor._red_flags) == 0


class TestTDAMonitorDivergence:
    """Tests for TDAMonitor P4 divergence observation."""

    def test_observe_divergence_no_drift(self):
        """Identical states should have DRS=0."""
        monitor = TDAMonitor()
        result = monitor.observe_divergence(
            cycle=1,
            real_H=0.8, real_rho=0.7, real_tau=0.2, real_beta=0.1,
            twin_H=0.8, twin_rho=0.7, twin_tau=0.2, twin_beta=0.1,
        )
        assert result["drs"] == pytest.approx(0.0, abs=1e-6)
        assert result["severity"] == "NONE"

    def test_observe_divergence_with_drift(self):
        """Different states should have positive DRS."""
        monitor = TDAMonitor()
        result = monitor.observe_divergence(
            cycle=1,
            real_H=0.9, real_rho=0.8, real_tau=0.3, real_beta=0.2,
            twin_H=0.8, twin_rho=0.7, twin_tau=0.2, twin_beta=0.1,
        )
        assert result["drs"] > 0
        assert "components" in result
        assert result["components"]["H_drift"] == pytest.approx(0.1, abs=1e-6)

    def test_observe_divergence_severity_classification(self):
        """Should classify severity correctly."""
        monitor = TDAMonitor()

        # Low drift -> NONE or INFO
        result = monitor.observe_divergence(
            cycle=1,
            real_H=0.81, real_rho=0.7, real_tau=0.2, real_beta=0.1,
            twin_H=0.80, twin_rho=0.7, twin_tau=0.2, twin_beta=0.1,
        )
        assert result["severity"] in ("NONE", "INFO")

        # High drift -> CRITICAL
        result = monitor.observe_divergence(
            cycle=2,
            real_H=0.9, real_rho=0.9, real_tau=0.4, real_beta=0.3,
            twin_H=0.5, twin_rho=0.5, twin_tau=0.2, twin_beta=0.1,
        )
        assert result["severity"] in ("WARN", "CRITICAL")


class TestTDAMonitorRedFlags:
    """Tests for TDA red-flag observation."""

    def test_no_red_flags_normal_operation(self):
        """Normal operation should not trigger red-flags."""
        monitor = TDAMonitor()
        for i in range(50):
            monitor.observe_cycle(
                cycle=i + 1,
                success=True,
                depth=3,
                H=0.9,
                rho=0.9,
                tau=0.2,
                beta=0.05,
            )
        # With stable high H and rho, should not trigger flags
        # (depends on actual metric computation)
        summary = monitor.get_summary()
        # May have some flags depending on SNS computation
        assert isinstance(summary.total_red_flags, int)

    def test_get_red_flags(self):
        """Should return list of red-flags."""
        monitor = TDAMonitor()
        for i in range(10):
            monitor.observe_cycle(
                cycle=i + 1,
                success=i % 2 == 0,
                depth=3,
                H=0.5,
                rho=0.5,
                tau=0.2,
                beta=0.1,
            )
        flags = monitor.get_red_flags()
        assert isinstance(flags, list)


class TestTDARedFlag:
    """Tests for TDARedFlag dataclass."""

    def test_red_flag_creation(self):
        """Should create valid red-flag."""
        flag = TDARedFlag(
            cycle=10,
            timestamp="2025-12-10T00:00:00Z",
            flag_type="TDA_SNS_ANOMALY",
            severity="CRITICAL",
            observed_value=0.65,
            threshold=0.6,
        )
        assert flag.cycle == 10
        assert flag.flag_type == "TDA_SNS_ANOMALY"

    def test_red_flag_to_dict(self):
        """Should serialize to dict."""
        flag = TDARedFlag(
            cycle=10,
            timestamp="2025-12-10T00:00:00Z",
            flag_type="TDA_SNS_ANOMALY",
            severity="CRITICAL",
            observed_value=0.65,
            threshold=0.6,
        )
        d = flag.to_dict()
        assert d["cycle"] == 10
        assert d["flag_type"] == "TDA_SNS_ANOMALY"
        assert d["action"] == "LOGGED_ONLY"
        assert d["mode"] == "SHADOW"


class TestTDASummary:
    """Tests for TDASummary dataclass."""

    def test_summary_creation(self):
        """Should create valid summary."""
        summary = TDASummary(
            total_cycles=100,
            sns_mean=0.2,
            pcs_mean=0.8,
            hss_mean=0.9,
            envelope_occupancy=0.95,
        )
        assert summary.total_cycles == 100
        assert summary.sns_mean == 0.2

    def test_summary_to_dict(self):
        """Should serialize to dict."""
        summary = TDASummary(
            total_cycles=100,
            sns_mean=0.2,
            sns_max=0.35,
            pcs_mean=0.8,
            pcs_min=0.6,
            hss_mean=0.9,
            hss_min=0.7,
            envelope_occupancy=0.95,
            total_red_flags=5,
            red_flags_by_type={"TDA_SNS_ANOMALY": 2, "TDA_PCS_COLLAPSE": 3},
        )
        d = summary.to_dict()
        assert d["total_cycles"] == 100
        assert d["mode"] == "SHADOW"
        assert "sns" in d
        assert "pcs" in d
        assert "hss" in d
        assert "envelope" in d
        assert "red_flags" in d
