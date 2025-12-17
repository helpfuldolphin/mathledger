"""
Phase X P3: TDA Monitor Tests

Tests for the TDAMonitor class that computes TDA metrics at window boundaries.

SHADOW MODE CONTRACT:
- All tests verify observational behavior only
- No tests involve governance modification
- Tests ensure deterministic output for reproducibility
"""

import pytest
from backend.ht.tda_monitor import TDAMetricsSnapshot, TDAMonitor


class TestTDAMetricsSnapshot:
    """Tests for TDAMetricsSnapshot dataclass."""

    def test_snapshot_creation(self) -> None:
        """Test basic snapshot creation with valid values."""
        snapshot = TDAMetricsSnapshot(
            window_index=0,
            SNS=0.8,
            PCS=0.75,
            DRS=0.2,
            HSS=0.7,
        )

        assert snapshot.window_index == 0
        assert snapshot.SNS == 0.8
        assert snapshot.PCS == 0.75
        assert snapshot.DRS == 0.2
        assert snapshot.HSS == 0.7

    def test_snapshot_to_dict(self) -> None:
        """Test TDAMetricsSnapshot.to_dict() produces valid dictionary."""
        snapshot = TDAMetricsSnapshot(
            window_index=5,
            SNS=0.123456789,
            PCS=0.987654321,
            DRS=0.111111111,
            HSS=0.555555555,
        )

        result = snapshot.to_dict()

        assert result["window_index"] == 5
        assert result["SNS"] == 0.123457  # Rounded to 6 decimals
        assert result["PCS"] == 0.987654
        assert result["DRS"] == 0.111111
        assert result["HSS"] == 0.555556

    def test_snapshot_to_dict_structure(self) -> None:
        """Test to_dict() returns expected keys."""
        snapshot = TDAMetricsSnapshot(
            window_index=0,
            SNS=0.5,
            PCS=0.5,
            DRS=0.5,
            HSS=0.5,
        )

        result = snapshot.to_dict()

        assert set(result.keys()) == {"window_index", "SNS", "PCS", "DRS", "HSS"}

    def test_snapshot_is_healthy_above_threshold(self) -> None:
        """Test is_healthy returns True when HSS >= threshold."""
        snapshot = TDAMetricsSnapshot(
            window_index=0, SNS=0.8, PCS=0.8, DRS=0.2, HSS=0.7
        )

        assert snapshot.is_healthy(threshold=0.5) is True
        assert snapshot.is_healthy(threshold=0.7) is True
        assert snapshot.is_healthy(threshold=0.71) is False

    def test_snapshot_is_healthy_default_threshold(self) -> None:
        """Test is_healthy with default threshold of 0.5."""
        healthy = TDAMetricsSnapshot(
            window_index=0, SNS=0.8, PCS=0.8, DRS=0.2, HSS=0.6
        )
        unhealthy = TDAMetricsSnapshot(
            window_index=1, SNS=0.3, PCS=0.3, DRS=0.8, HSS=0.4
        )

        assert healthy.is_healthy() is True
        assert unhealthy.is_healthy() is False


class TestTDAMonitorInit:
    """Tests for TDAMonitor initialization."""

    def test_default_initialization(self) -> None:
        """Test monitor initializes with default weights."""
        monitor = TDAMonitor()

        # Weights should sum to 1.0
        assert monitor._sns_weight == TDAMonitor.DEFAULT_SNS_WEIGHT
        assert monitor._pcs_weight == TDAMonitor.DEFAULT_PCS_WEIGHT
        assert monitor._drs_weight == TDAMonitor.DEFAULT_DRS_WEIGHT
        assert monitor._success_weight == TDAMonitor.DEFAULT_SUCCESS_WEIGHT

    def test_custom_weights(self) -> None:
        """Test monitor with custom weights."""
        monitor = TDAMonitor(
            sns_weight=0.4,
            pcs_weight=0.3,
            drs_weight=0.2,
            success_weight=0.1,
        )

        assert monitor._sns_weight == 0.4
        assert monitor._pcs_weight == 0.3
        assert monitor._drs_weight == 0.2
        assert monitor._success_weight == 0.1

    def test_invalid_weights_raises_error(self) -> None:
        """Test that weights not summing to 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="weights must sum to 1.0"):
            TDAMonitor(
                sns_weight=0.5,
                pcs_weight=0.5,
                drs_weight=0.5,
                success_weight=0.5,
            )

    def test_weights_with_small_tolerance(self) -> None:
        """Test that small rounding errors in weights are acceptable."""
        # This should not raise (0.3 + 0.3 + 0.2 + 0.2 = 1.0 with float precision)
        monitor = TDAMonitor(
            sns_weight=0.30000001,
            pcs_weight=0.29999999,
            drs_weight=0.2,
            success_weight=0.2,
        )
        assert monitor._sns_weight == 0.30000001


class TestTDAMonitorCompute:
    """Tests for TDAMonitor.compute() method."""

    def test_compute_returns_valid_snapshot(self) -> None:
        """Test that compute returns a valid TDAMetricsSnapshot."""
        monitor = TDAMonitor()

        H_series = [0.5, 0.6, 0.65, 0.7, 0.75]
        rho_series = [0.7, 0.72, 0.75, 0.78, 0.8]
        success_series = [True, True, False, True, True]

        snapshot = monitor.compute(H_series, rho_series, success_series, window_index=0)

        assert isinstance(snapshot, TDAMetricsSnapshot)
        assert snapshot.window_index == 0
        assert 0.0 <= snapshot.SNS <= 1.0
        assert 0.0 <= snapshot.PCS <= 1.0
        assert 0.0 <= snapshot.DRS <= 1.0
        assert 0.0 <= snapshot.HSS <= 1.0

    def test_compute_handles_empty_series(self) -> None:
        """Test compute returns neutral values for empty series."""
        monitor = TDAMonitor()

        snapshot = monitor.compute([], [], [], window_index=0)

        assert snapshot.window_index == 0
        assert snapshot.SNS == 0.5
        assert snapshot.PCS == 0.5
        assert snapshot.DRS == 0.5
        assert snapshot.HSS == 0.5

    def test_compute_handles_single_element_series(self) -> None:
        """Test compute handles single-element series gracefully."""
        monitor = TDAMonitor()

        snapshot = monitor.compute([0.5], [0.5], [True], window_index=0)

        # Should return neutral values for insufficient data
        assert snapshot.SNS == 0.5
        assert snapshot.PCS == 0.5
        assert snapshot.DRS == 0.5

    def test_compute_deterministic(self) -> None:
        """Test compute returns identical results for identical inputs."""
        monitor1 = TDAMonitor()
        monitor2 = TDAMonitor()

        H_series = [0.5, 0.6, 0.65, 0.7, 0.75]
        rho_series = [0.7, 0.72, 0.75, 0.78, 0.8]
        success_series = [True, True, False, True, True]

        snapshot1 = monitor1.compute(H_series, rho_series, success_series, window_index=0)
        snapshot2 = monitor2.compute(H_series, rho_series, success_series, window_index=0)

        assert snapshot1.SNS == snapshot2.SNS
        assert snapshot1.PCS == snapshot2.PCS
        assert snapshot1.DRS == snapshot2.DRS
        assert snapshot1.HSS == snapshot2.HSS

    def test_compute_stable_trajectory_high_sns(self) -> None:
        """Test that stable (low variance) trajectories produce high SNS."""
        monitor = TDAMonitor()

        # Very stable trajectory - almost constant values
        H_series = [0.7, 0.71, 0.69, 0.7, 0.7]
        rho_series = [0.8, 0.81, 0.79, 0.8, 0.8]
        success_series = [True] * 5

        snapshot = monitor.compute(H_series, rho_series, success_series, window_index=0)

        # High stability should produce high SNS
        assert snapshot.SNS > 0.7

    def test_compute_unstable_trajectory_low_sns(self) -> None:
        """Test that unstable (high variance) trajectories produce lower SNS."""
        monitor = TDAMonitor()

        # Unstable trajectory - large swings
        H_series = [0.1, 0.9, 0.2, 0.8, 0.3]
        rho_series = [0.2, 0.9, 0.1, 0.8, 0.2]
        success_series = [True, False, True, False, True]

        snapshot = monitor.compute(H_series, rho_series, success_series, window_index=0)

        # High variance should produce lower SNS
        assert snapshot.SNS < 0.7

    def test_compute_smooth_trajectory_high_pcs(self) -> None:
        """Test that smooth trajectories produce high PCS."""
        monitor = TDAMonitor()

        # Smooth trajectory - small incremental changes
        H_series = [0.5, 0.52, 0.54, 0.56, 0.58]
        rho_series = [0.6, 0.62, 0.64, 0.66, 0.68]
        success_series = [True] * 5

        snapshot = monitor.compute(H_series, rho_series, success_series, window_index=0)

        # Smooth paths should produce high PCS
        assert snapshot.PCS > 0.7

    def test_compute_improving_trajectory_low_drs(self) -> None:
        """Test that improving trajectories produce low DRS."""
        monitor = TDAMonitor()

        # Improving trajectory - H and rho increasing
        H_series = [0.3, 0.4, 0.5, 0.6, 0.7]
        rho_series = [0.4, 0.5, 0.6, 0.7, 0.8]
        success_series = [True] * 5

        snapshot = monitor.compute(H_series, rho_series, success_series, window_index=0)

        # Improving trajectory (positive trend) should have low DRS
        assert snapshot.DRS < 0.3

    def test_compute_declining_trajectory_high_drs(self) -> None:
        """Test that declining trajectories produce high DRS."""
        monitor = TDAMonitor()

        # Declining trajectory - H and rho decreasing
        H_series = [0.8, 0.7, 0.6, 0.5, 0.4]
        rho_series = [0.9, 0.8, 0.7, 0.6, 0.5]
        success_series = [True] * 5

        snapshot = monitor.compute(H_series, rho_series, success_series, window_index=0)

        # Declining trajectory (negative trend) should have higher DRS
        assert snapshot.DRS > 0.2

    def test_compute_high_success_rate_affects_hss(self) -> None:
        """Test that success rate contributes to HSS."""
        monitor = TDAMonitor()

        H_series = [0.6, 0.65, 0.7, 0.65, 0.6]
        rho_series = [0.7, 0.72, 0.75, 0.72, 0.7]

        # High success rate
        high_success = [True, True, True, True, True]
        snapshot_high = monitor.compute(H_series, rho_series, high_success, window_index=0)

        monitor.reset()

        # Low success rate
        low_success = [False, False, False, False, False]
        snapshot_low = monitor.compute(H_series, rho_series, low_success, window_index=0)

        # Higher success rate should result in higher HSS
        assert snapshot_high.HSS > snapshot_low.HSS


class TestTDAMonitorState:
    """Tests for TDAMonitor state management."""

    def test_get_computed_snapshots_empty(self) -> None:
        """Test get_computed_snapshots returns empty list initially."""
        monitor = TDAMonitor()

        assert monitor.get_computed_snapshots() == []

    def test_get_computed_snapshots_after_compute(self) -> None:
        """Test get_computed_snapshots returns all computed snapshots."""
        monitor = TDAMonitor()

        H_series = [0.5, 0.6, 0.65]
        rho_series = [0.7, 0.72, 0.75]
        success_series = [True, True, False]

        monitor.compute(H_series, rho_series, success_series, window_index=0)
        monitor.compute(H_series, rho_series, success_series, window_index=1)
        monitor.compute(H_series, rho_series, success_series, window_index=2)

        snapshots = monitor.get_computed_snapshots()

        assert len(snapshots) == 3
        assert snapshots[0].window_index == 0
        assert snapshots[1].window_index == 1
        assert snapshots[2].window_index == 2

    def test_get_latest_snapshot_empty(self) -> None:
        """Test get_latest_snapshot returns None when no snapshots computed."""
        monitor = TDAMonitor()

        assert monitor.get_latest_snapshot() is None

    def test_get_latest_snapshot(self) -> None:
        """Test get_latest_snapshot returns most recent snapshot."""
        monitor = TDAMonitor()

        H_series = [0.5, 0.6]
        rho_series = [0.7, 0.72]
        success_series = [True, False]

        monitor.compute(H_series, rho_series, success_series, window_index=0)
        monitor.compute(H_series, rho_series, success_series, window_index=5)

        latest = monitor.get_latest_snapshot()

        assert latest is not None
        assert latest.window_index == 5

    def test_get_trajectory(self) -> None:
        """Test get_trajectory returns all metric trajectories."""
        monitor = TDAMonitor()

        H_series = [0.5, 0.6, 0.7]
        rho_series = [0.7, 0.75, 0.8]
        success_series = [True, True, True]

        for i in range(3):
            monitor.compute(H_series, rho_series, success_series, window_index=i)

        trajectory = monitor.get_trajectory()

        assert "SNS" in trajectory
        assert "PCS" in trajectory
        assert "DRS" in trajectory
        assert "HSS" in trajectory
        assert len(trajectory["SNS"]) == 3
        assert len(trajectory["HSS"]) == 3

    def test_reset_clears_state(self) -> None:
        """Test reset clears all computed snapshots."""
        monitor = TDAMonitor()

        H_series = [0.5, 0.6]
        rho_series = [0.7, 0.72]
        success_series = [True, True]

        monitor.compute(H_series, rho_series, success_series, window_index=0)
        monitor.compute(H_series, rho_series, success_series, window_index=1)

        assert len(monitor.get_computed_snapshots()) == 2

        monitor.reset()

        assert len(monitor.get_computed_snapshots()) == 0
        assert monitor.get_latest_snapshot() is None

    def test_returned_snapshots_list_is_copy(self) -> None:
        """Test that get_computed_snapshots returns a copy, not the internal list."""
        monitor = TDAMonitor()

        H_series = [0.5, 0.6]
        rho_series = [0.7, 0.72]
        success_series = [True, True]

        monitor.compute(H_series, rho_series, success_series, window_index=0)

        snapshots = monitor.get_computed_snapshots()
        snapshots.clear()

        # Internal list should be unaffected
        assert len(monitor.get_computed_snapshots()) == 1


class TestTDAMonitorIntegration:
    """Integration tests for TDAMonitor with realistic data."""

    def test_multiple_windows_realistic_scenario(self) -> None:
        """Test computing TDA metrics across multiple windows with realistic data."""
        monitor = TDAMonitor()

        # Simulate 5 windows of improving performance
        for window_idx in range(5):
            base_h = 0.5 + window_idx * 0.05
            base_rho = 0.6 + window_idx * 0.04

            H_series = [base_h + i * 0.01 for i in range(10)]
            rho_series = [base_rho + i * 0.01 for i in range(10)]
            success_rate = 0.6 + window_idx * 0.08
            success_series = [i < int(success_rate * 10) for i in range(10)]

            monitor.compute(H_series, rho_series, success_series, window_index=window_idx)

        # Check trajectory shows improvement
        trajectory = monitor.get_trajectory()

        # HSS should generally increase over windows
        assert trajectory["HSS"][4] > trajectory["HSS"][0]

        # All metrics should be valid
        for metric in ["SNS", "PCS", "DRS", "HSS"]:
            assert all(0.0 <= v <= 1.0 for v in trajectory[metric])

    def test_100_cycle_smoke_test(self) -> None:
        """Smoke test: 100 cycles distributed across multiple windows."""
        monitor = TDAMonitor()
        window_size = 10

        for window_idx in range(10):  # 10 windows x 10 cycles = 100 cycles
            # Generate synthetic data
            H_series = [0.5 + 0.3 * ((i + window_idx * 10) / 100) for i in range(window_size)]
            rho_series = [0.6 + 0.2 * ((i + window_idx * 10) / 100) for i in range(window_size)]
            success_series = [True if (i + window_idx) % 3 != 0 else False for i in range(window_size)]

            snapshot = monitor.compute(H_series, rho_series, success_series, window_index=window_idx)

            # Each snapshot should be valid
            assert snapshot.window_index == window_idx
            assert 0.0 <= snapshot.HSS <= 1.0

        # Should have 10 snapshots
        assert len(monitor.get_computed_snapshots()) == 10
