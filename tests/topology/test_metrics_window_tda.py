"""
Phase X P3: MetricsWindow TDA Integration Tests

Tests for TDA trajectory tracking in MetricsWindow and MetricsAccumulator.

SHADOW MODE CONTRACT:
- All tests verify observation-only behavior
- TDA inputs are collected for offline analysis
"""

import pytest
from backend.topology.first_light.metrics_window import MetricsWindow, MetricsAccumulator


class TestMetricsWindowTDAInputs:
    """Tests for TDA trajectory tracking in MetricsWindow."""

    def test_window_stores_h_trajectory(self) -> None:
        """Test that H values are tracked in H_trajectory."""
        window = MetricsWindow(window_size=5)

        for i in range(5):
            window.add(
                success=True,
                abstained=False,
                in_omega=True,
                hard_ok=True,
                rsi=0.8,
                blocked=False,
                H=0.5 + i * 0.1,
            )

        assert len(window.H_trajectory) == 5
        assert window.H_trajectory[0] == 0.5
        assert window.H_trajectory[4] == 0.9

    def test_window_stores_rho_trajectory(self) -> None:
        """Test that rho (rsi) values are tracked in rho_trajectory."""
        window = MetricsWindow(window_size=5)

        for i in range(5):
            window.add(
                success=True,
                abstained=False,
                in_omega=True,
                hard_ok=True,
                rsi=0.6 + i * 0.05,
                blocked=False,
                H=0.7,
            )

        assert len(window.rho_trajectory) == 5
        assert window.rho_trajectory[0] == 0.6
        assert abs(window.rho_trajectory[4] - 0.8) < 0.001

    def test_window_stores_success_trajectory(self) -> None:
        """Test that success values are tracked in success_trajectory."""
        window = MetricsWindow(window_size=5)

        successes = [True, True, False, True, False]
        for i, success in enumerate(successes):
            window.add(
                success=success,
                abstained=False,
                in_omega=True,
                hard_ok=True,
                rsi=0.8,
                blocked=False,
                H=0.7,
            )

        assert window.success_trajectory == successes

    def test_window_without_h_still_tracks_rho_and_success(self) -> None:
        """Test backwards compatibility: H is optional."""
        window = MetricsWindow(window_size=3)

        for i in range(3):
            window.add(
                success=i % 2 == 0,
                abstained=False,
                in_omega=True,
                hard_ok=True,
                rsi=0.7 + i * 0.1,
                blocked=False,
                # No H parameter
            )

        # H_trajectory should be empty (no H provided)
        assert len(window.H_trajectory) == 0

        # But rho and success should be tracked
        assert len(window.rho_trajectory) == 3
        assert len(window.success_trajectory) == 3

    def test_finalize_includes_tda_inputs(self) -> None:
        """Test that finalize() returns tda_inputs dict."""
        window = MetricsWindow(window_size=3)

        for i in range(3):
            window.add(
                success=True,
                abstained=False,
                in_omega=True,
                hard_ok=True,
                rsi=0.8,
                blocked=False,
                H=0.7,
            )

        result = window.finalize()

        assert "tda_inputs" in result
        assert "H_trajectory" in result["tda_inputs"]
        assert "rho_trajectory" in result["tda_inputs"]
        assert "success_trajectory" in result["tda_inputs"]

    def test_tda_inputs_are_copies(self) -> None:
        """Test that tda_inputs returns copies, not references."""
        window = MetricsWindow(window_size=3)

        for i in range(3):
            window.add(True, False, True, True, 0.8, False, H=0.7)

        result = window.finalize()

        # Modify the returned lists
        result["tda_inputs"]["H_trajectory"].clear()

        # Window's internal data should be unaffected
        assert len(window.H_trajectory) == 3

    def test_reset_clears_tda_trajectories(self) -> None:
        """Test that reset() clears TDA trajectory data."""
        window = MetricsWindow(window_size=5)

        for i in range(3):
            window.add(True, False, True, True, 0.8, False, H=0.7)

        assert len(window.H_trajectory) == 3
        assert len(window.rho_trajectory) == 3
        assert len(window.success_trajectory) == 3

        window.reset(window_index=1, start_cycle=10)

        assert len(window.H_trajectory) == 0
        assert len(window.rho_trajectory) == 0
        assert len(window.success_trajectory) == 0


class TestMetricsAccumulatorTDAInputs:
    """Tests for TDA trajectory tracking in MetricsAccumulator."""

    def test_accumulator_passes_h_to_window(self) -> None:
        """Test that accumulator passes H to underlying window."""
        acc = MetricsAccumulator(window_size=5)

        for i in range(5):
            acc.add(
                success=True,
                abstained=False,
                in_omega=True,
                hard_ok=True,
                rsi=0.8,
                blocked=False,
                H=0.5 + i * 0.1,
            )

        # Window should have completed and been replaced
        windows = acc.get_completed_windows()
        assert len(windows) == 1

        # Check tda_inputs in completed window
        assert "tda_inputs" in windows[0]
        assert len(windows[0]["tda_inputs"]["H_trajectory"]) == 5

    def test_get_all_windows_includes_partial(self) -> None:
        """Test that get_all_windows() includes partial current window."""
        acc = MetricsAccumulator(window_size=10)

        # Add 15 cycles (1 full window + 5 partial)
        for i in range(15):
            acc.add(True, False, True, True, 0.8, False, H=0.7)

        # get_completed_windows should have 1
        completed = acc.get_completed_windows()
        assert len(completed) == 1

        # get_all_windows should have 2 (including partial)
        all_windows = acc.get_all_windows()
        assert len(all_windows) == 2

        # Partial window should have 5 entries
        assert all_windows[1]["total_count"] == 5

    def test_get_all_windows_empty_current(self) -> None:
        """Test get_all_windows when current window is empty."""
        acc = MetricsAccumulator(window_size=5)

        # Add exactly 1 full window
        for i in range(5):
            acc.add(True, False, True, True, 0.8, False, H=0.7)

        all_windows = acc.get_all_windows()
        # Should have 1 window (current is empty)
        assert len(all_windows) == 1

    def test_get_all_windows_only_partial(self) -> None:
        """Test get_all_windows with only partial window."""
        acc = MetricsAccumulator(window_size=10)

        # Add 3 cycles (no completed windows)
        for i in range(3):
            acc.add(True, False, True, True, 0.8, False, H=0.7)

        all_windows = acc.get_all_windows()
        assert len(all_windows) == 1
        assert all_windows[0]["total_count"] == 3

    def test_tda_inputs_across_multiple_windows(self) -> None:
        """Test TDA inputs are tracked correctly across multiple windows."""
        acc = MetricsAccumulator(window_size=5)

        # Add 15 cycles (3 windows)
        for i in range(15):
            window_idx = i // 5
            acc.add(
                success=True,
                abstained=False,
                in_omega=True,
                hard_ok=True,
                rsi=0.6 + window_idx * 0.1,  # Different rho per window
                blocked=False,
                H=0.5 + window_idx * 0.1,  # Different H per window
            )

        windows = acc.get_completed_windows()
        assert len(windows) == 3

        # Window 0 should have H around 0.5
        assert all(abs(h - 0.5) < 0.001 for h in windows[0]["tda_inputs"]["H_trajectory"])

        # Window 1 should have H around 0.6
        assert all(abs(h - 0.6) < 0.001 for h in windows[1]["tda_inputs"]["H_trajectory"])

        # Window 2 should have H around 0.7
        assert all(abs(h - 0.7) < 0.001 for h in windows[2]["tda_inputs"]["H_trajectory"])


class TestMetricsWindowTDAWithTDAMonitor:
    """Integration tests: MetricsWindow + TDAMonitor."""

    def test_window_tda_inputs_compatible_with_tda_monitor(self) -> None:
        """Test that window tda_inputs can be fed to TDAMonitor."""
        from backend.ht.tda_monitor import TDAMonitor

        window = MetricsWindow(window_size=10)

        for i in range(10):
            window.add(
                success=i % 3 != 0,
                abstained=False,
                in_omega=True,
                hard_ok=True,
                rsi=0.7 + i * 0.02,
                blocked=False,
                H=0.5 + i * 0.03,
            )

        result = window.finalize()
        tda_inputs = result["tda_inputs"]

        # Create TDAMonitor and compute
        monitor = TDAMonitor()
        snapshot = monitor.compute(
            H_series=tda_inputs["H_trajectory"],
            rho_series=tda_inputs["rho_trajectory"],
            success_series=tda_inputs["success_trajectory"],
            window_index=result["window_index"],
        )

        # Should produce valid TDA metrics
        assert 0.0 <= snapshot.SNS <= 1.0
        assert 0.0 <= snapshot.PCS <= 1.0
        assert 0.0 <= snapshot.DRS <= 1.0
        assert 0.0 <= snapshot.HSS <= 1.0

    def test_accumulator_windows_compatible_with_tda_monitor(self) -> None:
        """Test that accumulator windows can be processed by TDAMonitor."""
        from backend.ht.tda_monitor import TDAMonitor

        acc = MetricsAccumulator(window_size=10)

        # Add 30 cycles (3 windows)
        for i in range(30):
            acc.add(
                success=i % 4 != 0,
                abstained=False,
                in_omega=True,
                hard_ok=True,
                rsi=0.6 + (i % 10) * 0.03,
                blocked=False,
                H=0.5 + (i % 10) * 0.04,
            )

        monitor = TDAMonitor()

        for window in acc.get_completed_windows():
            tda_inputs = window["tda_inputs"]
            snapshot = monitor.compute(
                H_series=tda_inputs["H_trajectory"],
                rho_series=tda_inputs["rho_trajectory"],
                success_series=tda_inputs["success_trajectory"],
                window_index=window["window_index"],
            )
            assert snapshot.window_index == window["window_index"]

        # Should have 3 snapshots
        assert len(monitor.get_computed_snapshots()) == 3
