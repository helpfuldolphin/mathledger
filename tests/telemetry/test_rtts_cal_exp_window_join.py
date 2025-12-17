"""
RTTS × CAL-EXP Window Join Tests

Phase X P5.2: Tests for RTTS validation join with CAL-EXP windows.

Tests:
1. Deterministic ordering of annotated windows
2. Missing RTTS handled gracefully
3. Mock flags propagate correctly
4. Continuity violation rate propagates correctly
5. Window metrics preserved from CAL-EXP

SHADOW MODE CONTRACT:
- All tests verify OBSERVATIONAL outputs only
- No gating or enforcement is tested
- mode="SHADOW" and action="LOGGED_ONLY" are always verified
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pytest


# =============================================================================
# Mock MetricsWindow for testing (simplified)
# =============================================================================

@dataclass
class MockMetricsWindow:
    """Mock MetricsWindow for testing without full dependency."""

    window_index: int = 0
    start_cycle: int = 0
    end_cycle: int = 50
    success_rate: Optional[float] = 0.85
    omega_occupancy: Optional[float] = 0.92
    mean_rsi: Optional[float] = 0.75
    block_rate: Optional[float] = 0.05


# =============================================================================
# Mock RTTSValidationBlock for testing
# =============================================================================

@dataclass
class MockMockDetection:
    """Mock MockDetectionResult for testing."""

    mock_001_var_H_low: bool = False
    mock_002_var_rho_low: bool = False
    mock_003_cor_low: bool = False
    mock_004_cor_high: bool = False
    mock_005_acf_low: bool = False
    mock_006_acf_high: bool = False
    mock_007_kurtosis_low: bool = False
    mock_008_kurtosis_high: bool = False
    mock_009_jump_H: bool = False
    mock_010_discrete_rho: bool = False


@dataclass
class MockContinuityStats:
    """Mock ContinuityStats for testing."""

    violation_rate: float = 0.0


@dataclass
class MockRTTSValidationBlock:
    """Mock RTTSValidationBlock for testing."""

    overall_status: str = "OK"
    warning_count: int = 0
    mock_detection: Optional[MockMockDetection] = None
    continuity: Optional[MockContinuityStats] = None


# =============================================================================
# Tests for annotate_window_with_rtts
# =============================================================================

class TestAnnotateWindowWithRTTS:
    """Tests for annotate_window_with_rtts function."""

    def test_annotate_preserves_window_metrics(self):
        """Test that CAL-EXP window metrics are preserved."""
        from backend.telemetry.rtts_cal_exp_window_join import annotate_window_with_rtts

        window = MockMetricsWindow(
            window_index=3,
            start_cycle=150,
            end_cycle=200,
            success_rate=0.88,
            omega_occupancy=0.94,
            mean_rsi=0.78,
            block_rate=0.03,
        )

        annotated = annotate_window_with_rtts(window, None)

        assert annotated.window_index == 3
        assert annotated.start_cycle == 150
        assert annotated.end_cycle == 200
        assert annotated.success_rate == 0.88
        assert annotated.omega_occupancy == 0.94
        assert annotated.mean_rsi == 0.78
        assert annotated.block_rate == 0.03

    def test_annotate_without_rtts_sets_unavailable(self):
        """Test that missing RTTS sets rtts_available=False."""
        from backend.telemetry.rtts_cal_exp_window_join import annotate_window_with_rtts

        window = MockMetricsWindow()
        annotated = annotate_window_with_rtts(window, None)

        assert annotated.rtts_available is False
        assert annotated.mock_flags_count == 0
        assert annotated.mock_flags == []
        assert annotated.continuity_violation_rate == 0.0
        assert annotated.rtts_overall_status == "UNKNOWN"

    def test_annotate_with_rtts_sets_available(self):
        """Test that present RTTS sets rtts_available=True."""
        from backend.telemetry.rtts_cal_exp_window_join import annotate_window_with_rtts

        window = MockMetricsWindow()
        rtts = MockRTTSValidationBlock(
            overall_status="OK",
            warning_count=0,
        )

        annotated = annotate_window_with_rtts(window, rtts)

        assert annotated.rtts_available is True
        assert annotated.rtts_overall_status == "OK"

    def test_annotate_extracts_mock_flags(self):
        """Test that MOCK flags are extracted correctly."""
        from backend.telemetry.rtts_cal_exp_window_join import annotate_window_with_rtts

        window = MockMetricsWindow()
        mock_detection = MockMockDetection(
            mock_001_var_H_low=True,
            mock_003_cor_low=True,
            mock_009_jump_H=True,
        )
        rtts = MockRTTSValidationBlock(
            overall_status="WARN",
            warning_count=3,
            mock_detection=mock_detection,
        )

        annotated = annotate_window_with_rtts(window, rtts)

        assert annotated.mock_flags_count == 3
        assert "MOCK-001" in annotated.mock_flags
        assert "MOCK-003" in annotated.mock_flags
        assert "MOCK-009" in annotated.mock_flags
        assert "MOCK-002" not in annotated.mock_flags

    def test_annotate_extracts_continuity_violation_rate(self):
        """Test that continuity violation rate is extracted."""
        from backend.telemetry.rtts_cal_exp_window_join import annotate_window_with_rtts

        window = MockMetricsWindow()
        continuity = MockContinuityStats(violation_rate=0.15)
        rtts = MockRTTSValidationBlock(
            overall_status="ATTENTION",
            continuity=continuity,
        )

        annotated = annotate_window_with_rtts(window, rtts)

        assert annotated.continuity_violation_rate == 0.15

    def test_annotate_shadow_mode_markers(self):
        """Test that SHADOW MODE markers are set."""
        from backend.telemetry.rtts_cal_exp_window_join import annotate_window_with_rtts

        window = MockMetricsWindow()
        annotated = annotate_window_with_rtts(window, None)

        assert annotated.mode == "SHADOW"
        assert annotated.action == "LOGGED_ONLY"


# =============================================================================
# Tests for join_rtts_to_cal_exp_windows
# =============================================================================

class TestJoinRTTSToCalExpWindows:
    """Tests for join_rtts_to_cal_exp_windows function."""

    def test_join_preserves_window_order(self):
        """Test that window order is preserved (deterministic)."""
        from backend.telemetry.rtts_cal_exp_window_join import join_rtts_to_cal_exp_windows

        windows = [
            MockMetricsWindow(window_index=0, start_cycle=0),
            MockMetricsWindow(window_index=1, start_cycle=50),
            MockMetricsWindow(window_index=2, start_cycle=100),
            MockMetricsWindow(window_index=3, start_cycle=150),
        ]

        annotated = join_rtts_to_cal_exp_windows(windows, None)

        assert len(annotated) == 4
        assert annotated[0].window_index == 0
        assert annotated[1].window_index == 1
        assert annotated[2].window_index == 2
        assert annotated[3].window_index == 3

    def test_join_without_rtts_all_unavailable(self):
        """Test that all windows get rtts_available=False when RTTS missing."""
        from backend.telemetry.rtts_cal_exp_window_join import join_rtts_to_cal_exp_windows

        windows = [
            MockMetricsWindow(window_index=i) for i in range(5)
        ]

        annotated = join_rtts_to_cal_exp_windows(windows, None)

        for w in annotated:
            assert w.rtts_available is False

    def test_join_with_rtts_all_get_annotations(self):
        """Test that all windows get RTTS annotations when available."""
        from backend.telemetry.rtts_cal_exp_window_join import join_rtts_to_cal_exp_windows

        windows = [
            MockMetricsWindow(window_index=i) for i in range(3)
        ]
        mock_detection = MockMockDetection(mock_001_var_H_low=True)
        rtts = MockRTTSValidationBlock(
            overall_status="ATTENTION",
            warning_count=1,
            mock_detection=mock_detection,
        )

        annotated = join_rtts_to_cal_exp_windows(windows, rtts)

        for w in annotated:
            assert w.rtts_available is True
            assert w.rtts_overall_status == "ATTENTION"
            assert w.mock_flags_count == 1
            assert "MOCK-001" in w.mock_flags

    def test_join_empty_windows_returns_empty(self):
        """Test that empty windows list returns empty result."""
        from backend.telemetry.rtts_cal_exp_window_join import join_rtts_to_cal_exp_windows

        annotated = join_rtts_to_cal_exp_windows([], None)

        assert annotated == []

    def test_join_determinism_same_input_same_output(self):
        """Test that same input produces same output (deterministic)."""
        from backend.telemetry.rtts_cal_exp_window_join import join_rtts_to_cal_exp_windows

        windows = [
            MockMetricsWindow(window_index=i, success_rate=0.8 + i * 0.01)
            for i in range(5)
        ]
        mock_detection = MockMockDetection(
            mock_001_var_H_low=True,
            mock_003_cor_low=True,
        )
        rtts = MockRTTSValidationBlock(
            overall_status="WARN",
            warning_count=2,
            mock_detection=mock_detection,
        )

        # Run twice
        result1 = join_rtts_to_cal_exp_windows(windows, rtts)
        result2 = join_rtts_to_cal_exp_windows(windows, rtts)

        # Compare
        assert len(result1) == len(result2)
        for i in range(len(result1)):
            assert result1[i].window_index == result2[i].window_index
            assert result1[i].mock_flags == result2[i].mock_flags
            assert result1[i].mock_flags_count == result2[i].mock_flags_count


# =============================================================================
# Tests for join_rtts_dict_to_cal_exp_windows
# =============================================================================

class TestJoinRTTSDictToCalExpWindows:
    """Tests for join_rtts_dict_to_cal_exp_windows function."""

    def test_join_dict_extracts_mock_flags(self):
        """Test that MOCK flags are extracted from dict format."""
        from backend.telemetry.rtts_cal_exp_window_join import join_rtts_dict_to_cal_exp_windows

        windows = [MockMetricsWindow(window_index=0)]
        rtts_dict = {
            "overall_status": "WARN",
            "warning_count": 2,
            "mock_detection_flags": ["MOCK-001", "MOCK-005"],
            "continuity": {"violation_rate": 0.1},
        }

        annotated = join_rtts_dict_to_cal_exp_windows(windows, rtts_dict)

        assert len(annotated) == 1
        assert annotated[0].rtts_available is True
        assert annotated[0].mock_flags_count == 2
        assert annotated[0].mock_flags == ["MOCK-001", "MOCK-005"]
        assert annotated[0].continuity_violation_rate == 0.1

    def test_join_dict_without_rtts(self):
        """Test that missing RTTS dict is handled."""
        from backend.telemetry.rtts_cal_exp_window_join import join_rtts_dict_to_cal_exp_windows

        windows = [MockMetricsWindow(window_index=0)]

        annotated = join_rtts_dict_to_cal_exp_windows(windows, None)

        assert annotated[0].rtts_available is False


# =============================================================================
# Tests for RTTSAnnotatedWindow.to_dict
# =============================================================================

class TestRTTSAnnotatedWindowToDict:
    """Tests for RTTSAnnotatedWindow.to_dict serialization."""

    def test_to_dict_includes_all_fields(self):
        """Test that to_dict includes all required fields."""
        from backend.telemetry.rtts_cal_exp_window_join import RTTSAnnotatedWindow

        annotated = RTTSAnnotatedWindow(
            window_index=2,
            start_cycle=100,
            end_cycle=150,
            success_rate=0.9,
            omega_occupancy=0.95,
            mean_rsi=0.8,
            block_rate=0.02,
            mock_flags_count=2,
            mock_flags=["MOCK-001", "MOCK-003"],
            continuity_violation_rate=0.05,
            rtts_overall_status="ATTENTION",
            rtts_warning_count=2,
            rtts_available=True,
        )

        d = annotated.to_dict()

        assert d["schema_version"] == "1.0.0"
        assert d["window_index"] == 2
        assert d["start_cycle"] == 100
        assert d["end_cycle"] == 150
        assert d["cal_exp_metrics"]["success_rate"] == 0.9
        assert d["cal_exp_metrics"]["omega_occupancy"] == 0.95
        assert d["rtts_annotations"]["mock_flags_count"] == 2
        assert d["rtts_annotations"]["mock_flags"] == ["MOCK-001", "MOCK-003"]
        assert d["rtts_annotations"]["continuity_violation_rate"] == 0.05
        assert d["rtts_annotations"]["rtts_available"] is True
        assert d["mode"] == "SHADOW"
        assert d["action"] == "LOGGED_ONLY"

    def test_to_dict_handles_none_metrics(self):
        """Test that None metrics are serialized correctly."""
        from backend.telemetry.rtts_cal_exp_window_join import RTTSAnnotatedWindow

        annotated = RTTSAnnotatedWindow(
            window_index=0,
            success_rate=None,
            omega_occupancy=None,
        )

        d = annotated.to_dict()

        assert d["cal_exp_metrics"]["success_rate"] is None
        assert d["cal_exp_metrics"]["omega_occupancy"] is None


# =============================================================================
# Run all tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
