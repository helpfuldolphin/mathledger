"""
Tests for TDA Metric Computations

Tests SNS, PCS, DRS, HSS computation and window aggregation.
See: docs/system_law/TDA_PhaseX_Binding.md

SHADOW MODE: All tests verify observational metrics only.
"""

import math
import pytest
from typing import List, Tuple

from backend.tda.metrics import (
    SNSComputer,
    PCSComputer,
    DRSComputer,
    HSSComputer,
    TDAMetrics,
    TDAWindowMetrics,
    compute_sns,
    compute_pcs,
    compute_drs,
    compute_hss,
    compute_window_statistics,
    classify_drs_severity,
)


class TestSNSComputer:
    """Tests for Structural Novelty Score computation."""

    def test_sns_initial_value(self):
        """First pattern should have moderate novelty."""
        computer = SNSComputer()
        sns = computer.compute(success=True, depth=3)
        assert 0.0 <= sns <= 1.0
        assert sns == pytest.approx(0.5, abs=0.1)  # Moderate initial

    def test_sns_repeated_pattern_decreases_novelty(self):
        """Repeating the same pattern should decrease SNS."""
        computer = SNSComputer()
        # Establish history
        for _ in range(10):
            computer.compute(success=True, depth=3)

        # Same pattern again
        sns = computer.compute(success=True, depth=3)
        assert sns < 0.5  # Less novel due to repetition

    def test_sns_novel_pattern_increases(self):
        """New pattern should have higher SNS."""
        computer = SNSComputer()
        # Establish history with one pattern
        for _ in range(20):
            computer.compute(success=True, depth=3)

        # Completely new pattern
        sns = computer.compute(success=False, depth=10)
        assert sns > 0.3  # More novel

    def test_sns_range(self):
        """SNS should always be in [0, 1]."""
        computer = SNSComputer()
        for i in range(100):
            success = i % 3 == 0
            depth = (i % 5) + 1
            sns = computer.compute(success=success, depth=depth)
            assert 0.0 <= sns <= 1.0

    def test_sns_reset(self):
        """Reset should clear history."""
        computer = SNSComputer()
        for _ in range(10):
            computer.compute(success=True, depth=3)
        computer.reset()
        sns = computer.compute(success=True, depth=3)
        assert sns == pytest.approx(0.5, abs=0.1)  # Back to initial


class TestSNSStandaloneFunction:
    """Tests for standalone compute_sns function."""

    def test_compute_sns_no_history(self):
        """Empty history should return moderate SNS."""
        sns = compute_sns(success=True, depth=3, historical_patterns=[])
        assert sns == 0.5

    def test_compute_sns_with_matching_history(self):
        """Matching history should decrease SNS."""
        history = [(True, 3), (True, 3), (True, 3), (True, 3)]
        sns = compute_sns(success=True, depth=3, historical_patterns=history)
        assert sns < 0.5

    def test_compute_sns_with_different_history(self):
        """Different history should increase SNS."""
        history = [(False, 5), (False, 5), (False, 5), (False, 5)]
        sns = compute_sns(success=True, depth=3, historical_patterns=history)
        assert sns > 0.3


class TestPCSComputer:
    """Tests for Proof Coherence Score computation."""

    def test_pcs_initial_value(self):
        """Initial PCS should be high (coherent)."""
        computer = PCSComputer()
        pcs = computer.compute(success=True, rho=0.8, H=0.9)
        assert pcs == 1.0  # Not enough data yet

    def test_pcs_stable_pattern_stays_high(self):
        """Stable success patterns should maintain high PCS."""
        computer = PCSComputer()
        for _ in range(10):
            pcs = computer.compute(success=True, rho=0.8, H=0.8)
        assert pcs >= 0.7  # High coherence

    def test_pcs_oscillating_pattern_decreases(self):
        """Rapid oscillation should decrease PCS."""
        computer = PCSComputer()
        for i in range(20):
            success = i % 2 == 0  # Alternating
            pcs = computer.compute(success=success, rho=0.5, H=0.5)

        assert pcs < 0.7  # Lower due to oscillation

    def test_pcs_range(self):
        """PCS should always be in [0, 1]."""
        computer = PCSComputer()
        for i in range(100):
            success = i % 2 == 0
            rho = 0.2 + (i % 8) * 0.1
            H = 0.3 + (i % 7) * 0.1
            pcs = computer.compute(success=success, rho=rho, H=H)
            assert 0.0 <= pcs <= 1.0


class TestPCSStandaloneFunction:
    """Tests for standalone compute_pcs function."""

    def test_compute_pcs_stable_trajectory(self):
        """Stable trajectory should yield high PCS."""
        success_traj = [True] * 10
        rho_traj = [0.8] * 10
        H_traj = [0.9] * 10
        pcs = compute_pcs(success_traj, rho_traj, H_traj)
        assert pcs >= 0.8

    def test_compute_pcs_oscillating_trajectory(self):
        """Oscillating trajectory should yield lower PCS."""
        success_traj = [True, False] * 10
        rho_traj = [0.5] * 20
        H_traj = [0.5] * 20
        pcs = compute_pcs(success_traj, rho_traj, H_traj)
        assert pcs < 0.7

    def test_compute_pcs_short_trajectory(self):
        """Short trajectory should return 1.0."""
        pcs = compute_pcs([True, True], [0.8, 0.8], [0.9, 0.9])
        assert pcs == 1.0


class TestDRSComputer:
    """Tests for Drift Rate Score computation."""

    def test_drs_identical_states(self):
        """Identical states should yield DRS = 0."""
        computer = DRSComputer()
        drs = computer.compute(
            real_H=0.8, real_rho=0.7, real_tau=0.2, real_beta=0.1,
            twin_H=0.8, twin_rho=0.7, twin_tau=0.2, twin_beta=0.1,
        )
        assert drs == pytest.approx(0.0, abs=1e-10)

    def test_drs_single_dimension_drift(self):
        """Single dimension drift should be sqrt of that delta^2."""
        computer = DRSComputer()
        drs = computer.compute(
            real_H=0.9, real_rho=0.7, real_tau=0.2, real_beta=0.1,
            twin_H=0.8, twin_rho=0.7, twin_tau=0.2, twin_beta=0.1,
        )
        # Only H differs by 0.1
        expected = 0.1  # sqrt(0.01 + 0 + 0 + 0)
        assert drs == pytest.approx(expected, abs=1e-6)

    def test_drs_multiple_dimension_drift(self):
        """Multiple dimension drift should be L2 norm."""
        computer = DRSComputer()
        drs = computer.compute(
            real_H=0.9, real_rho=0.8, real_tau=0.3, real_beta=0.2,
            twin_H=0.8, twin_rho=0.7, twin_tau=0.2, twin_beta=0.1,
        )
        # Each delta is 0.1, so L2 = sqrt(4 * 0.01) = 0.2
        expected = 0.2
        assert drs == pytest.approx(expected, abs=1e-6)

    def test_drs_components(self):
        """Test component breakdown."""
        computer = DRSComputer()
        result = computer.compute_components(
            real_H=0.9, real_rho=0.8, real_tau=0.3, real_beta=0.2,
            twin_H=0.8, twin_rho=0.7, twin_tau=0.2, twin_beta=0.1,
        )
        assert result["H_drift"] == pytest.approx(0.1, abs=1e-6)
        assert result["rho_drift"] == pytest.approx(0.1, abs=1e-6)
        assert result["tau_drift"] == pytest.approx(0.1, abs=1e-6)
        assert result["beta_drift"] == pytest.approx(0.1, abs=1e-6)
        assert result["drs"] == pytest.approx(0.2, abs=1e-6)


class TestDRSStandaloneFunction:
    """Tests for standalone compute_drs function."""

    def test_compute_drs_identical(self):
        """Identical states should return 0."""
        drs = compute_drs((0.8, 0.7, 0.2, 0.1), (0.8, 0.7, 0.2, 0.1))
        assert drs == pytest.approx(0.0, abs=1e-10)

    def test_compute_drs_with_drift(self):
        """Should compute L2 norm."""
        drs = compute_drs((0.9, 0.8, 0.3, 0.2), (0.8, 0.7, 0.2, 0.1))
        expected = math.sqrt(4 * 0.01)  # 0.2
        assert drs == pytest.approx(expected, abs=1e-6)


class TestDRSSeverityClassification:
    """Tests for DRS severity classification."""

    def test_classify_drs_none(self):
        """DRS <= 0.05 should be NONE."""
        assert classify_drs_severity(0.0) == "NONE"
        assert classify_drs_severity(0.03) == "NONE"
        assert classify_drs_severity(0.05) == "NONE"

    def test_classify_drs_info(self):
        """DRS 0.05-0.10 should be INFO."""
        assert classify_drs_severity(0.06) == "INFO"
        assert classify_drs_severity(0.08) == "INFO"
        assert classify_drs_severity(0.10) == "INFO"

    def test_classify_drs_warn(self):
        """DRS 0.10-0.20 should be WARN."""
        assert classify_drs_severity(0.11) == "WARN"
        assert classify_drs_severity(0.15) == "WARN"
        assert classify_drs_severity(0.20) == "WARN"

    def test_classify_drs_critical(self):
        """DRS > 0.20 should be CRITICAL."""
        assert classify_drs_severity(0.21) == "CRITICAL"
        assert classify_drs_severity(0.5) == "CRITICAL"
        assert classify_drs_severity(1.0) == "CRITICAL"


class TestHSSComputer:
    """Tests for Homological Stability Score computation."""

    def test_hss_initial_value(self):
        """Initial HSS should be 1.0 (baseline)."""
        computer = HSSComputer()
        hss = computer.compute(b0=1, b1=0, b2=0)
        assert hss == 1.0

    def test_hss_growth_stays_high(self):
        """Growth in features should maintain high HSS."""
        computer = HSSComputer()
        computer.compute(b0=1, b1=0, b2=0)  # Baseline
        hss = computer.compute(b0=5, b1=1, b2=0)  # Growth
        assert hss >= 0.7  # Still stable

    def test_hss_loss_decreases(self):
        """Loss of features should decrease HSS."""
        computer = HSSComputer()
        computer.compute(b0=10, b1=2, b2=0)  # Baseline with many features
        hss = computer.compute(b0=5, b1=0, b2=0)  # Significant loss
        assert hss < 0.7  # Degraded

    def test_hss_range(self):
        """HSS should always be in [0, 1]."""
        computer = HSSComputer()
        computer.compute(b0=5, b1=1, b2=0)
        for i in range(20):
            hss = computer.compute(b0=max(1, 5 - i//2), b1=i % 3, b2=0)
            assert 0.0 <= hss <= 1.0

    def test_hss_from_dag_size(self):
        """Simplified computation from DAG stats."""
        computer = HSSComputer()
        hss = computer.compute_from_dag_size(dag_size=100, proof_count=10)
        assert 0.0 <= hss <= 1.0


class TestHSSStandaloneFunction:
    """Tests for standalone compute_hss function."""

    def test_compute_hss_same_betti(self):
        """Same Betti numbers should return high HSS."""
        hss = compute_hss((5, 1, 0), (5, 1, 0))
        assert hss == 1.0

    def test_compute_hss_growth(self):
        """Growth should maintain stability."""
        hss = compute_hss((10, 2, 0), (5, 1, 0))  # Current > baseline
        assert hss >= 0.7

    def test_compute_hss_loss(self):
        """Loss should decrease HSS."""
        hss = compute_hss((2, 0, 0), (10, 2, 0))  # Current < baseline
        assert hss < 0.8


class TestTDAMetrics:
    """Tests for TDAMetrics dataclass."""

    def test_metrics_creation(self):
        """Should create valid metrics."""
        metrics = TDAMetrics(sns=0.2, pcs=0.8, drs=0.05, hss=0.9)
        assert metrics.sns == 0.2
        assert metrics.pcs == 0.8
        assert metrics.drs == 0.05
        assert metrics.hss == 0.9

    def test_metrics_to_dict(self):
        """Should serialize to dict."""
        metrics = TDAMetrics(sns=0.2, pcs=0.8, drs=0.05, hss=0.9)
        d = metrics.to_dict()
        assert d["sns"] == pytest.approx(0.2, abs=1e-6)
        assert d["pcs"] == pytest.approx(0.8, abs=1e-6)
        assert d["drs"] == pytest.approx(0.05, abs=1e-6)
        assert d["hss"] == pytest.approx(0.9, abs=1e-6)

    def test_envelope_check_inside(self):
        """Should be inside envelope when thresholds met."""
        metrics = TDAMetrics(sns=0.3, pcs=0.7, hss=0.8)
        assert metrics.check_envelope() is True

    def test_envelope_check_outside_sns(self):
        """SNS too high should be outside envelope."""
        metrics = TDAMetrics(sns=0.5, pcs=0.7, hss=0.8)
        assert metrics.check_envelope() is False

    def test_envelope_check_outside_pcs(self):
        """PCS too low should be outside envelope."""
        metrics = TDAMetrics(sns=0.3, pcs=0.5, hss=0.8)
        assert metrics.check_envelope() is False

    def test_envelope_check_outside_hss(self):
        """HSS too low should be outside envelope."""
        metrics = TDAMetrics(sns=0.3, pcs=0.7, hss=0.5)
        assert metrics.check_envelope() is False


class TestTDAWindowMetrics:
    """Tests for TDAWindowMetrics dataclass."""

    def test_window_metrics_creation(self):
        """Should create valid window metrics."""
        window = TDAWindowMetrics(
            window_index=0,
            sns_mean=0.2,
            pcs_mean=0.8,
            hss_mean=0.9,
            envelope_occupancy_rate=0.95,
        )
        assert window.window_index == 0
        assert window.sns_mean == 0.2

    def test_window_metrics_to_dict(self):
        """Should serialize per schema."""
        window = TDAWindowMetrics(
            window_index=1,
            window_start_cycle=50,
            window_end_cycle=99,
            sns_mean=0.2,
            sns_max=0.35,
            sns_min=0.1,
            pcs_mean=0.85,
            hss_mean=0.9,
            envelope_occupancy_rate=0.94,
        )
        d = window.to_dict()
        assert d["schema_version"] == "1.0.0"
        assert d["window_index"] == 1
        assert d["mode"] == "SHADOW"
        assert "sns" in d
        assert "pcs" in d
        assert "hss" in d
        assert "envelope" in d


class TestComputeWindowStatistics:
    """Tests for window statistics helper."""

    def test_empty_list(self):
        """Empty list should return zeros."""
        stats = compute_window_statistics([])
        assert stats["mean"] == 0.0
        assert stats["std"] == 0.0

    def test_single_value(self):
        """Single value should have zero std."""
        stats = compute_window_statistics([0.5])
        assert stats["mean"] == 0.5
        assert stats["min"] == 0.5
        assert stats["max"] == 0.5
        assert stats["std"] == 0.0

    def test_multiple_values(self):
        """Multiple values should compute correctly."""
        stats = compute_window_statistics([0.2, 0.4, 0.6, 0.8])
        assert stats["mean"] == pytest.approx(0.5, abs=1e-6)
        assert stats["min"] == 0.2
        assert stats["max"] == 0.8
        assert stats["std"] > 0
