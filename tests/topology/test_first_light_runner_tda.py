"""
Phase X P3: First-Light Runner TDA Integration Tests

Tests for TDA metrics computation in FirstLightShadowRunner.

SHADOW MODE CONTRACT:
- All tests verify observation-only behavior
- TDA metrics are computed for offline analysis
"""

import pytest
from backend.topology.first_light import FirstLightConfig, FirstLightShadowRunner


class TestFirstLightRunnerTDAMetrics:
    """Tests for TDA metrics in FirstLightShadowRunner."""

    def test_runner_computes_tda_metrics(self) -> None:
        """Test that runner computes TDA metrics in finalize()."""
        config = FirstLightConfig(
            total_cycles=100,
            success_window=20,
            shadow_mode=True,
        )
        runner = FirstLightShadowRunner(config, seed=42)
        result = runner.run()

        # Should have TDA metrics
        assert len(result.tda_metrics) > 0

    def test_runner_tda_metrics_have_correct_structure(self) -> None:
        """Test that TDA metrics have expected fields."""
        config = FirstLightConfig(
            total_cycles=100,
            success_window=20,
            shadow_mode=True,
        )
        runner = FirstLightShadowRunner(config, seed=42)
        result = runner.run()

        for tda in result.tda_metrics:
            assert "window_index" in tda
            assert "SNS" in tda
            assert "PCS" in tda
            assert "DRS" in tda
            assert "HSS" in tda

    def test_runner_tda_metrics_valid_ranges(self) -> None:
        """Test that TDA metrics are in valid ranges [0, 1]."""
        config = FirstLightConfig(
            total_cycles=100,
            success_window=20,
            shadow_mode=True,
        )
        runner = FirstLightShadowRunner(config, seed=42)
        result = runner.run()

        for tda in result.tda_metrics:
            assert 0.0 <= tda["SNS"] <= 1.0, f"SNS out of range: {tda['SNS']}"
            assert 0.0 <= tda["PCS"] <= 1.0, f"PCS out of range: {tda['PCS']}"
            assert 0.0 <= tda["DRS"] <= 1.0, f"DRS out of range: {tda['DRS']}"
            assert 0.0 <= tda["HSS"] <= 1.0, f"HSS out of range: {tda['HSS']}"

    def test_runner_tda_metrics_window_count_matches(self) -> None:
        """Test that TDA metrics count matches window count."""
        config = FirstLightConfig(
            total_cycles=100,
            success_window=20,  # Should produce 5 windows
            shadow_mode=True,
        )
        runner = FirstLightShadowRunner(config, seed=42)
        result = runner.run()

        # 100 cycles / 20 window size = 5 complete windows
        assert len(result.tda_metrics) == 5

    def test_runner_result_to_dict_includes_tda(self) -> None:
        """Test that result.to_dict() includes TDA metrics."""
        config = FirstLightConfig(
            total_cycles=50,
            success_window=10,
            shadow_mode=True,
        )
        runner = FirstLightShadowRunner(config, seed=42)
        result = runner.run()

        result_dict = result.to_dict()

        assert "tda_metrics" in result_dict
        assert len(result_dict["tda_metrics"]) == 5  # 50/10 = 5 windows

    def test_runner_tda_deterministic_with_seed(self) -> None:
        """Test that TDA metrics are deterministic with same seed."""
        config = FirstLightConfig(
            total_cycles=100,
            success_window=20,
            shadow_mode=True,
        )

        runner1 = FirstLightShadowRunner(config, seed=12345)
        result1 = runner1.run()

        runner2 = FirstLightShadowRunner(config, seed=12345)
        result2 = runner2.run()

        assert len(result1.tda_metrics) == len(result2.tda_metrics)
        for tda1, tda2 in zip(result1.tda_metrics, result2.tda_metrics):
            assert tda1["SNS"] == tda2["SNS"]
            assert tda1["PCS"] == tda2["PCS"]
            assert tda1["DRS"] == tda2["DRS"]
            assert tda1["HSS"] == tda2["HSS"]

    def test_runner_reset_clears_tda(self) -> None:
        """Test that runner.reset() clears TDA state."""
        config = FirstLightConfig(
            total_cycles=50,
            success_window=10,
            shadow_mode=True,
        )
        runner = FirstLightShadowRunner(config, seed=42)

        # First run
        result1 = runner.run()
        assert len(result1.tda_metrics) > 0

        # Reset and run again
        runner.reset()
        result2 = runner.run()

        # Should have fresh TDA metrics
        assert len(result2.tda_metrics) > 0

    def test_runner_partial_window_tda(self) -> None:
        """Test TDA is computed for partial windows."""
        config = FirstLightConfig(
            total_cycles=35,  # 3 full windows of 10 + 5 partial
            success_window=10,
            shadow_mode=True,
        )
        runner = FirstLightShadowRunner(config, seed=42)
        result = runner.run()

        # Should have 4 TDA snapshots (3 full + 1 partial)
        assert len(result.tda_metrics) == 4


class TestFirstLightRunner100CycleSmoke:
    """Smoke test for 100-cycle run with TDA."""

    def test_100_cycle_run_with_tda(self) -> None:
        """Smoke test: 100-cycle run produces valid TDA metrics."""
        config = FirstLightConfig(
            total_cycles=100,
            success_window=10,
            shadow_mode=True,
        )
        runner = FirstLightShadowRunner(config, seed=42)
        result = runner.run()

        # Basic result validation
        assert result.cycles_completed == 100
        assert len(result.tda_metrics) == 10  # 100/10 = 10 windows

        # TDA metrics should show reasonable values
        hss_values = [tda["HSS"] for tda in result.tda_metrics]
        mean_hss = sum(hss_values) / len(hss_values)
        assert 0.3 <= mean_hss <= 0.9, f"Mean HSS {mean_hss} seems unreasonable"

    def test_1000_cycle_run_with_tda(self) -> None:
        """Larger smoke test: 1000 cycles (typical First-Light run)."""
        config = FirstLightConfig(
            total_cycles=1000,
            success_window=50,
            shadow_mode=True,
        )
        runner = FirstLightShadowRunner(config, seed=42)
        result = runner.run()

        # Should complete successfully
        assert result.cycles_completed == 1000
        assert len(result.tda_metrics) == 20  # 1000/50 = 20 windows

        # Check TDA trajectory shows reasonable pattern
        hss_trajectory = [tda["HSS"] for tda in result.tda_metrics]
        # With synthetic generator, HSS should generally be stable or improving
        assert min(hss_trajectory) > 0.2
        assert max(hss_trajectory) < 1.0
