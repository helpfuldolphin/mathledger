"""
Tests for TDA Integration with Phase X P3/P4

Tests MetricsWindow and DivergenceSnapshot TDA integration.
See: docs/system_law/TDA_PhaseX_Binding.md

SHADOW MODE: All tests verify observational metrics only.
"""

import pytest

from backend.tda.monitor import TDAMonitor
from backend.topology.first_light.metrics_window import MetricsWindow, MetricsAccumulator
from backend.topology.first_light.data_structures_p4 import (
    RealCycleObservation,
    TwinCycleObservation,
    DivergenceSnapshot,
)


class TestMetricsWindowTDAIntegration:
    """Tests for MetricsWindow TDA integration."""

    def test_finalize_with_tda(self):
        """finalize_with_tda should include TDA metrics."""
        window = MetricsWindow(window_size=10)
        tda_monitor = TDAMonitor()

        # Add cycles to both window and TDA monitor
        for i in range(10):
            window.add(
                success=True,
                abstained=False,
                in_omega=True,
                hard_ok=True,
                rsi=0.8,
                blocked=False,
                H=0.85,
            )
            tda_monitor.observe_cycle(
                cycle=i + 1,
                success=True,
                depth=3,
                H=0.85,
                rho=0.8,
                tau=0.2,
                beta=0.05,
            )

        # Finalize with TDA
        result = window.finalize_with_tda(tda_monitor)

        assert "tda_metrics" in result
        tda = result["tda_metrics"]
        assert "sns" in tda
        assert "pcs" in tda
        assert "hss" in tda
        assert "envelope" in tda
        assert tda["mode"] == "SHADOW"

    def test_finalize_without_tda(self):
        """Standard finalize should still work."""
        window = MetricsWindow(window_size=10)
        for i in range(10):
            window.add(
                success=True,
                abstained=False,
                in_omega=True,
                hard_ok=True,
                rsi=0.8,
                blocked=False,
            )

        result = window.finalize()
        assert "tda_metrics" not in result  # Not included in standard finalize
        assert "success_metrics" in result
        assert "mode" in result


class TestDivergenceSnapshotTDAContext:
    """Tests for DivergenceSnapshot TDA context."""

    def test_from_observations_includes_drs(self):
        """DRS should be computed from observations."""
        real = RealCycleObservation(
            cycle=1,
            H=0.9, rho=0.8, tau=0.3, beta=0.2,
            success=True, in_omega=True, hard_ok=True,
        )
        twin = TwinCycleObservation(
            real_cycle=1,
            twin_H=0.8, twin_rho=0.7, twin_tau=0.2, twin_beta=0.1,
            predicted_success=True, predicted_in_omega=True, predicted_hard_ok=True,
        )

        snapshot = DivergenceSnapshot.from_observations(real, twin)

        # Check DRS fields
        assert snapshot.drs > 0
        assert snapshot.drs_severity in ("NONE", "INFO", "WARN", "CRITICAL")
        assert snapshot.drs_H_drift == pytest.approx(0.1, abs=1e-6)
        assert snapshot.drs_rho_drift == pytest.approx(0.1, abs=1e-6)
        assert snapshot.drs_tau_drift == pytest.approx(0.1, abs=1e-6)
        assert snapshot.drs_beta_drift == pytest.approx(0.1, abs=1e-6)

    def test_from_observations_no_drift(self):
        """Identical states should have DRS=0."""
        real = RealCycleObservation(
            cycle=1,
            H=0.8, rho=0.7, tau=0.2, beta=0.1,
            success=True, in_omega=True, hard_ok=True,
        )
        twin = TwinCycleObservation(
            real_cycle=1,
            twin_H=0.8, twin_rho=0.7, twin_tau=0.2, twin_beta=0.1,
            predicted_success=True, predicted_in_omega=True, predicted_hard_ok=True,
        )

        snapshot = DivergenceSnapshot.from_observations(real, twin)
        assert snapshot.drs == pytest.approx(0.0, abs=1e-6)
        assert snapshot.drs_severity == "NONE"

    def test_to_dict_includes_tda_context(self):
        """to_dict should include TDA context."""
        real = RealCycleObservation(
            cycle=1,
            H=0.9, rho=0.8, tau=0.3, beta=0.2,
            success=True, in_omega=True, hard_ok=True,
        )
        twin = TwinCycleObservation(
            real_cycle=1,
            twin_H=0.8, twin_rho=0.7, twin_tau=0.2, twin_beta=0.1,
            predicted_success=False, predicted_in_omega=True, predicted_hard_ok=True,
        )

        snapshot = DivergenceSnapshot.from_observations(real, twin)
        d = snapshot.to_dict()

        assert "tda_context" in d
        tda = d["tda_context"]
        assert "drs" in tda
        assert "drs_severity" in tda
        assert "drs_components" in tda
        assert tda["drs_components"]["H_drift"] == pytest.approx(0.1, abs=1e-6)

    def test_drs_severity_classification(self):
        """DRS severity should follow TDA spec thresholds."""
        # Low drift -> NONE
        real = RealCycleObservation(
            cycle=1,
            H=0.81, rho=0.7, tau=0.2, beta=0.1,
            success=True, in_omega=True, hard_ok=True,
        )
        twin = TwinCycleObservation(
            real_cycle=1,
            twin_H=0.80, twin_rho=0.7, twin_tau=0.2, twin_beta=0.1,
            predicted_success=True, predicted_in_omega=True, predicted_hard_ok=True,
        )
        snapshot = DivergenceSnapshot.from_observations(real, twin)
        # DRS = 0.01, should be NONE
        assert snapshot.drs_severity == "NONE"

        # High drift -> CRITICAL
        real2 = RealCycleObservation(
            cycle=2,
            H=0.9, rho=0.9, tau=0.5, beta=0.4,
            success=True, in_omega=True, hard_ok=True,
        )
        twin2 = TwinCycleObservation(
            real_cycle=2,
            twin_H=0.5, twin_rho=0.5, twin_tau=0.2, twin_beta=0.1,
            predicted_success=True, predicted_in_omega=True, predicted_hard_ok=True,
        )
        snapshot2 = DivergenceSnapshot.from_observations(real2, twin2)
        # Large drift -> CRITICAL
        assert snapshot2.drs_severity in ("WARN", "CRITICAL")


class TestTDASchemaCompliance:
    """Tests for TDA JSON schema compliance."""

    def test_tda_window_metrics_schema(self):
        """TDAWindowMetrics.to_dict() should match schema."""
        from backend.tda.metrics import TDAWindowMetrics

        window = TDAWindowMetrics(
            window_index=1,
            window_start_cycle=50,
            window_end_cycle=99,
            sns_mean=0.2,
            sns_max=0.35,
            sns_min=0.1,
            sns_std=0.05,
            pcs_mean=0.85,
            pcs_max=0.92,
            pcs_min=0.71,
            pcs_std=0.04,
            hss_mean=0.88,
            hss_max=0.95,
            hss_min=0.74,
            hss_std=0.03,
            envelope_occupancy_rate=0.94,
            envelope_exit_count=3,
            max_envelope_exit_streak=2,
        )
        d = window.to_dict()

        # Check required fields per tda_metrics_p3.schema.json
        assert d["schema_version"] == "1.0.0"
        assert d["window_index"] == 1
        assert d["mode"] == "SHADOW"
        assert "sns" in d
        assert "mean" in d["sns"]
        assert "max" in d["sns"]
        assert "min" in d["sns"]
        assert "std" in d["sns"]
        assert "pcs" in d
        assert "hss" in d
        assert "envelope" in d
        assert "occupancy_rate" in d["envelope"]

    def test_divergence_tda_context_schema(self):
        """DivergenceSnapshot TDA context should match schema."""
        real = RealCycleObservation(
            cycle=1,
            H=0.9, rho=0.8, tau=0.3, beta=0.2,
            success=True, in_omega=True, hard_ok=True,
        )
        twin = TwinCycleObservation(
            real_cycle=1,
            twin_H=0.8, twin_rho=0.7, twin_tau=0.2, twin_beta=0.1,
            predicted_success=True, predicted_in_omega=True, predicted_hard_ok=True,
        )

        snapshot = DivergenceSnapshot.from_observations(real, twin)
        d = snapshot.to_dict()
        tda = d["tda_context"]

        # Check fields per tda_metrics_p4.schema.json
        assert "drs" in tda
        assert "drs_severity" in tda
        assert tda["drs_severity"] in ("NONE", "INFO", "WARN", "CRITICAL")
        assert "drs_components" in tda
        assert "H_drift" in tda["drs_components"]
        assert "rho_drift" in tda["drs_components"]
        assert "tau_drift" in tda["drs_components"]
        assert "beta_drift" in tda["drs_components"]
        assert "in_tda_envelope" in tda


class TestTDAConsoleTile:
    """Tests for TDA console tile rendering."""

    def test_render_from_metrics(self):
        """Should render tile from TDAMetrics."""
        from backend.tda.console_tile import render_tda_tile
        from backend.tda.metrics import TDAMetrics

        metrics = TDAMetrics(sns=0.2, pcs=0.85, drs=0.05, hss=0.9)
        output = render_tda_tile(metrics=metrics, use_color=False)

        assert "TDA Mind Scanner" in output
        assert "SNS" in output
        assert "PCS" in output
        assert "DRS" in output
        assert "HSS" in output
        assert "SHADOW" in output

    def test_render_from_summary(self):
        """Should render tile from TDASummary."""
        from backend.tda.console_tile import render_tda_tile
        from backend.tda.monitor import TDASummary

        summary = TDASummary(
            total_cycles=100,
            sns_mean=0.2,
            pcs_mean=0.85,
            drs_mean=0.05,
            hss_mean=0.9,
            envelope_occupancy=0.95,
            total_red_flags=2,
        )
        output = render_tda_tile(summary=summary, use_color=False)

        assert "TDA Mind Scanner" in output
        assert "Cycles:" in output or "100" in output

    def test_render_tda_summary(self):
        """Should render detailed summary."""
        from backend.tda.console_tile import render_tda_summary
        from backend.tda.monitor import TDASummary

        summary = TDASummary(
            total_cycles=100,
            sns_mean=0.2,
            sns_max=0.35,
            sns_anomaly_count=0,
            pcs_mean=0.85,
            pcs_min=0.65,
            pcs_collapse_count=0,
            hss_mean=0.9,
            hss_min=0.75,
            hss_degradation_count=0,
            envelope_occupancy=0.95,
            total_red_flags=0,
            red_flags_by_type={},
        )
        output = render_tda_summary(summary, use_color=False)

        assert "TDA Mind Scanner Summary" in output
        assert "SHADOW" in output
        assert "Structural Novelty" in output
        assert "Proof Coherence" in output
        assert "Drift Rate" in output
        assert "Homological Stability" in output
