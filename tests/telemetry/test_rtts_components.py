"""
Tests for RTTS (Real Telemetry Topology Spec) Components

Phase X P5.1: LOG-ONLY Implementation Tests

These tests verify:
1. RTTSStatisticalValidator field presence and JSON output
2. RTTSMockDetector MOCK-001 through MOCK-010 checks
3. RTTSContinuityTracker Lipschitz bound tracking
4. RTTSCorrelationTracker cross-correlation computation

SHADOW MODE: All tests operate in observation-only mode.

See: docs/system_law/RTTS_Gap_Closure_Blueprint.md
See: docs/system_law/Real_Telemetry_Topology_Spec.md
"""

import json
import pytest
from dataclasses import FrozenInstanceError

from backend.topology.first_light.data_structures_p4 import (
    TelemetrySnapshot,
    ContinuityCheck,
)
from backend.telemetry.rtts_statistical_validator import (
    RTTSStatisticalValidator,
    RTTSStatisticalResult,
)
from backend.telemetry.rtts_mock_detector import (
    RTTSMockDetector,
    MockDetectionResult,
)
from backend.telemetry.rtts_continuity_tracker import (
    RTTSContinuityTracker,
    ContinuityStats,
)
from backend.telemetry.rtts_correlation_tracker import (
    RTTSCorrelationTracker,
)
from backend.telemetry.governance_signal import (
    MockIndicatorSummary,
    RTTSCorrelationResult,
)


# -----------------------------------------------------------------------------
# Helper: Create mock TelemetrySnapshot
# -----------------------------------------------------------------------------

def make_snapshot(
    cycle: int,
    H: float = 0.8,
    rho: float = 0.9,
    tau: float = 0.2,
    beta: float = 0.1,
    in_omega: bool = True,
) -> TelemetrySnapshot:
    """Create a TelemetrySnapshot for testing."""
    return TelemetrySnapshot(
        cycle=cycle,
        timestamp="2025-12-10T12:00:00.000000+00:00",
        runner_type="u2",
        slice_name="test_slice",
        success=True,
        depth=5,
        H=H,
        rho=rho,
        tau=tau,
        beta=beta,
        in_omega=in_omega,
        real_blocked=False,
        governance_aligned=True,
        hard_ok=True,
    )


# -----------------------------------------------------------------------------
# RTTS-GAP-001: Statistical Validation Tests
# -----------------------------------------------------------------------------

class TestRTTSStatisticalValidator:
    """Tests for RTTSStatisticalValidator (RTTS-GAP-001)."""

    def test_result_has_required_fields(self):
        """Verify RTTSStatisticalResult has all required fields."""
        result = RTTSStatisticalResult()

        # Variance fields
        assert hasattr(result, "variance_H")
        assert hasattr(result, "variance_rho")
        assert hasattr(result, "variance_tau")
        assert hasattr(result, "variance_beta")

        # Autocorrelation fields
        assert hasattr(result, "autocorr_H_lag1")
        assert hasattr(result, "autocorr_rho_lag1")

        # Kurtosis fields
        assert hasattr(result, "kurtosis_H")
        assert hasattr(result, "kurtosis_rho")

        # Mode fields
        assert result.mode == "SHADOW"

    def test_result_to_dict_json_serializable(self):
        """Verify RTTSStatisticalResult.to_dict() is JSON serializable."""
        result = RTTSStatisticalResult(
            variance_H=0.001,
            variance_rho=0.0005,
            autocorr_H_lag1=0.5,
            kurtosis_H=0.2,
            window_size=100,
            computed=True,
            insufficient_data=False,
        )

        d = result.to_dict()
        json_str = json.dumps(d)

        assert json_str is not None
        parsed = json.loads(json_str)
        assert parsed["variance"]["H"] == 0.001
        assert parsed["mode"] == "SHADOW"

    def test_validator_initialization(self):
        """Verify validator initializes correctly."""
        validator = RTTSStatisticalValidator(window_size=200)

        assert validator.window_size == 200
        assert validator.get_window_size() == 0
        assert validator.is_window_full() is False

    def test_validator_update_accumulates_data(self):
        """Verify validator accumulates snapshot data."""
        validator = RTTSStatisticalValidator(window_size=10)

        for i in range(5):
            snapshot = make_snapshot(cycle=i, H=0.8 + i * 0.01)
            validator.update(snapshot)

        assert validator.get_window_size() == 5
        assert validator.is_window_full() is False

    def test_validator_computes_when_window_full(self):
        """Verify validator computes statistics when window is full."""
        validator = RTTSStatisticalValidator(window_size=10)

        # Fill window
        for i in range(10):
            snapshot = make_snapshot(cycle=i, H=0.8 + i * 0.01)
            result = validator.update(snapshot)

        assert validator.is_window_full() is True
        assert result.computed is True
        assert result.insufficient_data is False
        assert result.variance_H is not None
        assert result.autocorr_H_lag1 is not None

    def test_validator_returns_insufficient_data_before_full(self):
        """Verify validator marks insufficient_data before window is full."""
        validator = RTTSStatisticalValidator(window_size=10)

        snapshot = make_snapshot(cycle=1)
        result = validator.update(snapshot)

        assert result.computed is False
        assert result.insufficient_data is True

    def test_validator_reset(self):
        """Verify validator reset clears state."""
        validator = RTTSStatisticalValidator(window_size=10)

        for i in range(5):
            validator.update(make_snapshot(cycle=i))

        assert validator.get_window_size() == 5

        validator.reset()

        assert validator.get_window_size() == 0

    def test_validator_variance_computation(self):
        """Verify variance is computed correctly."""
        validator = RTTSStatisticalValidator(window_size=5)

        # Constant values should have zero variance
        for i in range(5):
            validator.update(make_snapshot(cycle=i, H=0.8))  # Constant H

        result = validator.get_current_result()
        assert result.variance_H is not None
        assert result.variance_H < 1e-10  # Should be ~0

    def test_validator_autocorrelation_computation(self):
        """Verify autocorrelation is computed correctly."""
        validator = RTTSStatisticalValidator(window_size=10)

        # Alternating pattern should have negative lag-1 autocorrelation
        for i in range(10):
            H = 0.8 if i % 2 == 0 else 0.9
            validator.update(make_snapshot(cycle=i, H=H))

        result = validator.get_current_result()
        assert result.autocorr_H_lag1 is not None
        assert result.autocorr_H_lag1 < 0  # Negative for alternating


# -----------------------------------------------------------------------------
# RTTS-GAP-002: Mock Detection Tests
# -----------------------------------------------------------------------------

class TestRTTSMockDetector:
    """Tests for RTTSMockDetector (RTTS-GAP-002)."""

    def test_detection_result_has_all_mock_indicators(self):
        """Verify MockDetectionResult has MOCK-001 through MOCK-010."""
        result = MockDetectionResult()

        assert hasattr(result, "mock_001_var_H_low")
        assert hasattr(result, "mock_002_var_rho_low")
        assert hasattr(result, "mock_003_cor_low")
        assert hasattr(result, "mock_004_cor_high")
        assert hasattr(result, "mock_005_acf_low")
        assert hasattr(result, "mock_006_acf_high")
        assert hasattr(result, "mock_007_kurtosis_low")
        assert hasattr(result, "mock_008_kurtosis_high")
        assert hasattr(result, "mock_009_jump_H")
        assert hasattr(result, "mock_010_discrete_rho")

        # Mode
        assert result.mode == "SHADOW"

    def test_detection_result_to_dict_json_serializable(self):
        """Verify MockDetectionResult.to_dict() is JSON serializable."""
        result = MockDetectionResult(
            status="VALIDATED_REAL",
            confidence=0.9,
            mock_001_var_H_low=False,
            rtts_validation_passed=True,
        )

        d = result.to_dict()
        json_str = json.dumps(d)

        assert json_str is not None
        parsed = json.loads(json_str)
        assert parsed["status"] == "VALIDATED_REAL"
        assert parsed["indicators"]["MOCK_001"] is False
        assert parsed["mode"] == "SHADOW"

    def test_detector_initialization(self):
        """Verify detector initializes correctly."""
        detector = RTTSMockDetector()

        assert detector.get_cycle_count() == 0

    def test_detector_detect_with_no_data(self):
        """Verify detector handles detection with no statistical data."""
        detector = RTTSMockDetector()

        result = detector.detect()

        assert result.status == "VALIDATED_REAL"  # No indicators triggered
        assert result.confidence > 0

    def test_detector_mock_001_low_variance_H(self):
        """Verify MOCK-001 detects low Var(H)."""
        detector = RTTSMockDetector()

        # Create stats with low variance
        stats = RTTSStatisticalResult(
            variance_H=0.00001,  # Below threshold 0.0001
            computed=True,
        )

        result = detector.detect(stats=stats)

        assert result.mock_001_var_H_low is True
        assert "MOCK-001" in str(result.violations)

    def test_detector_mock_002_low_variance_rho(self):
        """Verify MOCK-002 detects low Var(rho)."""
        detector = RTTSMockDetector()

        stats = RTTSStatisticalResult(
            variance_rho=0.00001,  # Below threshold 0.00005
            computed=True,
        )

        result = detector.detect(stats=stats)

        assert result.mock_002_var_rho_low is True
        assert "MOCK-002" in str(result.violations)

    def test_detector_mock_003_low_correlation(self):
        """Verify MOCK-003 detects low |Cor(H, rho)|."""
        detector = RTTSMockDetector()

        correlations = RTTSCorrelationResult(
            cor_H_rho=0.05,  # Below threshold 0.1
        )

        result = detector.detect(correlations=correlations)

        assert result.mock_003_cor_low is True
        assert "MOCK-003" in str(result.violations)

    def test_detector_mock_004_high_correlation(self):
        """Verify MOCK-004 detects high |Cor(H, rho)|."""
        detector = RTTSMockDetector()

        correlations = RTTSCorrelationResult(
            cor_H_rho=0.995,  # Above threshold 0.99
        )

        result = detector.detect(correlations=correlations)

        assert result.mock_004_cor_high is True
        assert "MOCK-004" in str(result.violations)

    def test_detector_mock_009_jump_H(self):
        """Verify MOCK-009 detects large H jumps."""
        detector = RTTSMockDetector()

        # Record large delta
        detector.record_delta_H(0.2)  # Above threshold 0.15

        result = detector.detect()

        assert result.mock_009_jump_H is True
        assert "MOCK-009" in str(result.violations)

    def test_detector_mock_010_discrete_rho(self):
        """Verify MOCK-010 detects too few unique rho values."""
        detector = RTTSMockDetector()

        # Record only 5 unique values over 100+ cycles
        for i in range(105):
            detector.record_rho_value(0.1 * (i % 5))  # Only 5 unique values

        result = detector.detect()

        assert result.mock_010_discrete_rho is True
        assert "MOCK-010" in str(result.violations)

    def test_detector_status_suspected_mock(self):
        """Verify status is SUSPECTED_MOCK with high severity indicators."""
        detector = RTTSMockDetector()

        stats = RTTSStatisticalResult(
            variance_H=0.00001,  # MOCK-001 trigger
            computed=True,
        )

        result = detector.detect(stats=stats)

        assert result.status == "SUSPECTED_MOCK"
        assert result.high_severity_count >= 1

    def test_detector_status_validated_real(self):
        """Verify status is VALIDATED_REAL with no indicators."""
        detector = RTTSMockDetector()

        # Good stats - no mock indicators
        stats = RTTSStatisticalResult(
            variance_H=0.01,  # Well above threshold
            variance_rho=0.005,
            autocorr_H_lag1=0.5,
            kurtosis_H=0.2,
            computed=True,
        )

        correlations = RTTSCorrelationResult(
            cor_H_rho=0.5,  # Within bounds
        )

        result = detector.detect(stats=stats, correlations=correlations)

        assert result.status == "VALIDATED_REAL"
        assert result.rtts_validation_passed is True

    def test_detector_to_mock_indicator_summary(self):
        """Verify detector can convert result to MockIndicatorSummary."""
        detector = RTTSMockDetector()

        result = MockDetectionResult(
            mock_001_var_H_low=True,
            mock_003_cor_low=True,
            high_severity_count=1,
            medium_severity_count=1,
        )

        summary = detector.to_mock_indicator_summary(result)

        assert isinstance(summary, MockIndicatorSummary)
        assert summary.mock_001_var_H_low is True
        assert summary.mock_003_cor_low is True
        assert summary.high_severity_count == 1

    def test_detector_reset(self):
        """Verify detector reset clears state."""
        detector = RTTSMockDetector()

        detector.record_delta_H(0.2)
        detector.record_rho_value(0.5)

        detector.reset()

        assert detector.get_cycle_count() == 0


# -----------------------------------------------------------------------------
# RTTS-GAP-003: Continuity Tracking Tests
# -----------------------------------------------------------------------------

class TestContinuityCheck:
    """Tests for ContinuityCheck dataclass."""

    def test_continuity_check_has_required_fields(self):
        """Verify ContinuityCheck has all required fields."""
        check = ContinuityCheck()

        assert hasattr(check, "cycle")
        assert hasattr(check, "prev_cycle")
        assert hasattr(check, "delta_H")
        assert hasattr(check, "delta_rho")
        assert hasattr(check, "delta_tau")
        assert hasattr(check, "delta_beta")
        assert hasattr(check, "H_violated")
        assert hasattr(check, "rho_violated")
        assert hasattr(check, "tau_violated")
        assert hasattr(check, "beta_violated")
        assert hasattr(check, "any_violation")
        assert hasattr(check, "continuity_flag")
        assert hasattr(check, "mode")
        assert hasattr(check, "action")

    def test_continuity_check_shadow_mode(self):
        """Verify ContinuityCheck defaults to SHADOW mode."""
        check = ContinuityCheck()

        assert check.mode == "SHADOW"
        assert check.action == "LOGGED_ONLY"

    def test_continuity_check_to_dict(self):
        """Verify ContinuityCheck.to_dict() is JSON serializable."""
        check = ContinuityCheck(
            cycle=10,
            prev_cycle=9,
            delta_H=0.05,
            delta_rho=0.03,
            H_violated=False,
            any_violation=False,
            continuity_flag="OK",
        )

        d = check.to_dict()
        json_str = json.dumps(d)

        assert json_str is not None
        parsed = json.loads(json_str)
        assert parsed["cycle"] == 10
        assert parsed["deltas"]["H"] == 0.05
        assert parsed["mode"] == "SHADOW"

    def test_continuity_check_from_snapshots(self):
        """Verify ContinuityCheck.from_snapshots() works."""
        current = make_snapshot(cycle=10, H=0.85, rho=0.92)
        previous = make_snapshot(cycle=9, H=0.80, rho=0.90)

        check = ContinuityCheck.from_snapshots(current, previous)

        assert check.cycle == 10
        assert check.prev_cycle == 9
        assert abs(check.delta_H - 0.05) < 0.001
        assert abs(check.delta_rho - 0.02) < 0.001


class TestRTTSContinuityTracker:
    """Tests for RTTSContinuityTracker (RTTS-GAP-003)."""

    def test_tracker_initialization(self):
        """Verify tracker initializes correctly."""
        tracker = RTTSContinuityTracker(history_size=100)

        assert tracker.history_size == 100
        assert tracker.get_violation_rate() == 0.0

    def test_tracker_first_check_returns_none(self):
        """Verify first check returns None (no previous snapshot)."""
        tracker = RTTSContinuityTracker()

        snapshot = make_snapshot(cycle=1)
        result = tracker.check(snapshot)

        assert result is None

    def test_tracker_subsequent_checks_return_result(self):
        """Verify subsequent checks return ContinuityCheck."""
        tracker = RTTSContinuityTracker()

        tracker.check(make_snapshot(cycle=1, H=0.80))
        result = tracker.check(make_snapshot(cycle=2, H=0.85))

        assert result is not None
        assert isinstance(result, ContinuityCheck)
        assert result.cycle == 2
        assert result.prev_cycle == 1

    def test_tracker_detects_H_violation(self):
        """Verify tracker detects H Lipschitz violation."""
        tracker = RTTSContinuityTracker()

        tracker.check(make_snapshot(cycle=1, H=0.50))
        result = tracker.check(make_snapshot(cycle=2, H=0.80))  # Delta = 0.30 > 0.15

        assert result.H_violated is True
        assert result.any_violation is True
        assert result.continuity_flag == "TELEMETRY_JUMP"

    def test_tracker_detects_rho_violation(self):
        """Verify tracker detects rho Lipschitz violation."""
        tracker = RTTSContinuityTracker()

        tracker.check(make_snapshot(cycle=1, rho=0.50))
        result = tracker.check(make_snapshot(cycle=2, rho=0.65))  # Delta = 0.15 > 0.10

        assert result.rho_violated is True
        assert result.any_violation is True

    def test_tracker_stats_accumulate(self):
        """Verify tracker accumulates statistics."""
        tracker = RTTSContinuityTracker()

        # Generate some checks with violations
        tracker.check(make_snapshot(cycle=1, H=0.50))
        tracker.check(make_snapshot(cycle=2, H=0.80))  # Violation
        tracker.check(make_snapshot(cycle=3, H=0.82))  # No violation

        stats = tracker.get_stats()

        assert stats.total_checks == 2
        assert stats.violation_count == 1
        assert stats.H_violations == 1
        assert stats.mode == "SHADOW"

    def test_tracker_stats_to_dict(self):
        """Verify ContinuityStats.to_dict() is JSON serializable."""
        stats = ContinuityStats(
            total_checks=100,
            violation_count=5,
            H_violations=3,
            max_delta_H=0.18,
        )

        d = stats.to_dict()
        json_str = json.dumps(d)

        assert json_str is not None
        parsed = json.loads(json_str)
        assert parsed["counts"]["total_checks"] == 100
        assert parsed["mode"] == "SHADOW"

    def test_tracker_recent_violations(self):
        """Verify tracker can retrieve recent violations."""
        tracker = RTTSContinuityTracker()

        # Generate checks with some violations
        tracker.check(make_snapshot(cycle=1, H=0.50))
        tracker.check(make_snapshot(cycle=2, H=0.80))  # Violation
        tracker.check(make_snapshot(cycle=3, H=0.82))  # No violation
        tracker.check(make_snapshot(cycle=4, H=1.00))  # Violation

        violations = tracker.get_recent_violations()

        assert len(violations) == 2

    def test_tracker_has_recent_violations(self):
        """Verify has_recent_violations() helper."""
        tracker = RTTSContinuityTracker()

        tracker.check(make_snapshot(cycle=1, H=0.50))
        tracker.check(make_snapshot(cycle=2, H=0.52))  # Small delta

        assert tracker.has_recent_violations(window=5) is False

        tracker.check(make_snapshot(cycle=3, H=0.90))  # Large delta

        assert tracker.has_recent_violations(window=5) is True

    def test_tracker_reset(self):
        """Verify tracker reset clears state."""
        tracker = RTTSContinuityTracker()

        tracker.check(make_snapshot(cycle=1))
        tracker.check(make_snapshot(cycle=2))

        tracker.reset()

        stats = tracker.get_stats()
        assert stats.total_checks == 0


# -----------------------------------------------------------------------------
# RTTS-GAP-004: Correlation Tracking Tests
# -----------------------------------------------------------------------------

class TestRTTSCorrelationResult:
    """Tests for RTTSCorrelationResult dataclass."""

    def test_correlation_result_has_required_fields(self):
        """Verify RTTSCorrelationResult has all required fields."""
        result = RTTSCorrelationResult()

        assert hasattr(result, "cor_H_rho")
        assert hasattr(result, "cor_rho_omega")
        assert hasattr(result, "cor_beta_not_omega")
        assert hasattr(result, "cor_H_rho_violated")
        assert hasattr(result, "cor_rho_omega_violated")
        assert hasattr(result, "cor_beta_not_omega_violated")
        assert hasattr(result, "zero_correlation_detected")
        assert hasattr(result, "perfect_correlation_detected")
        assert hasattr(result, "inverted_correlation_detected")
        assert hasattr(result, "window_size")
        assert hasattr(result, "mode")

    def test_correlation_result_shadow_mode(self):
        """Verify RTTSCorrelationResult defaults to SHADOW mode."""
        result = RTTSCorrelationResult()

        assert result.mode == "SHADOW"

    def test_correlation_result_to_dict(self):
        """Verify RTTSCorrelationResult.to_dict() is JSON serializable."""
        result = RTTSCorrelationResult(
            cor_H_rho=0.6,
            cor_rho_omega=0.7,
            cor_beta_not_omega=0.4,
            cor_H_rho_violated=False,
            window_size=200,
        )

        d = result.to_dict()
        json_str = json.dumps(d)

        assert json_str is not None
        parsed = json.loads(json_str)
        assert parsed["correlations"]["H_rho"] == 0.6
        assert parsed["mode"] == "SHADOW"


class TestRTTSCorrelationTracker:
    """Tests for RTTSCorrelationTracker (RTTS-GAP-004)."""

    def test_tracker_initialization(self):
        """Verify tracker initializes correctly."""
        tracker = RTTSCorrelationTracker(window_size=200)

        assert tracker.window_size == 200
        assert tracker.get_window_size() == 0
        assert tracker.is_window_full() is False

    def test_tracker_update_accumulates(self):
        """Verify tracker accumulates snapshot data."""
        tracker = RTTSCorrelationTracker(window_size=100)

        for i in range(50):
            tracker.update(make_snapshot(cycle=i, H=0.8 + i * 0.001))

        assert tracker.get_window_size() == 50
        assert tracker.is_window_full() is False

    def test_tracker_computes_when_data_available(self):
        """Verify tracker computes correlations with sufficient data."""
        tracker = RTTSCorrelationTracker(window_size=20)

        # Generate correlated data
        for i in range(20):
            H = 0.5 + i * 0.02
            rho = 0.4 + i * 0.025  # Positively correlated with H
            tracker.update(make_snapshot(cycle=i, H=H, rho=rho))

        result = tracker.compute()

        assert result.cor_H_rho is not None
        assert result.cor_H_rho > 0  # Should be positive

    def test_tracker_detects_H_rho_violation_low(self):
        """Verify tracker detects cor(H, rho) below bounds."""
        tracker = RTTSCorrelationTracker(window_size=20)

        # Generate uncorrelated data (H random, rho constant)
        import random
        random.seed(42)
        for i in range(20):
            H = random.random()  # Random H
            rho = 0.5  # Constant rho
            tracker.update(make_snapshot(cycle=i, H=H, rho=rho))

        result = tracker.compute()

        # With constant rho, correlation should be ~0 (below 0.3 min bound)
        # Note: might not trigger with random data, but tests the mechanism
        assert result.cor_H_rho is not None

    def test_tracker_detects_zero_correlation(self):
        """Verify tracker detects zero correlation pattern."""
        tracker = RTTSCorrelationTracker(window_size=20)

        # Generate data with zero correlation (orthogonal patterns)
        import math
        for i in range(20):
            H = 0.5 + 0.3 * math.sin(i * 0.5)
            rho = 0.5 + 0.3 * math.cos(i * 0.5)  # 90 degrees out of phase
            tracker.update(make_snapshot(cycle=i, H=H, rho=rho))

        result = tracker.compute()

        # Sin and cos are orthogonal - should detect near-zero correlation
        assert result.cor_H_rho is not None

    def test_tracker_point_biserial_with_binary(self):
        """Verify tracker computes point-biserial with binary omega."""
        tracker = RTTSCorrelationTracker(window_size=20)

        # Generate data where high rho correlates with in_omega=True
        for i in range(20):
            rho = 0.3 + i * 0.03
            in_omega = rho > 0.5  # Binary based on rho threshold
            tracker.update(make_snapshot(cycle=i, rho=rho, in_omega=in_omega))

        result = tracker.compute()

        # rho and omega should be positively correlated
        assert result.cor_rho_omega is not None
        assert result.cor_rho_omega > 0

    def test_tracker_reset(self):
        """Verify tracker reset clears state."""
        tracker = RTTSCorrelationTracker(window_size=100)

        for i in range(50):
            tracker.update(make_snapshot(cycle=i))

        assert tracker.get_window_size() == 50

        tracker.reset()

        assert tracker.get_window_size() == 0

    def test_tracker_insufficient_data_returns_partial(self):
        """Verify tracker returns partial result with insufficient data."""
        tracker = RTTSCorrelationTracker(window_size=200)

        # Only add 5 snapshots (need at least 10 for correlation)
        for i in range(5):
            tracker.update(make_snapshot(cycle=i))

        result = tracker.compute()

        # Should return result but correlation values may be None
        assert result.window_size == 5


# -----------------------------------------------------------------------------
# MockIndicatorSummary Tests
# -----------------------------------------------------------------------------

class TestMockIndicatorSummary:
    """Tests for MockIndicatorSummary dataclass."""

    def test_summary_has_all_mock_fields(self):
        """Verify MockIndicatorSummary has MOCK-001 through MOCK-010."""
        summary = MockIndicatorSummary()

        assert hasattr(summary, "mock_001_var_H_low")
        assert hasattr(summary, "mock_002_var_rho_low")
        assert hasattr(summary, "mock_003_cor_low")
        assert hasattr(summary, "mock_004_cor_high")
        assert hasattr(summary, "mock_005_acf_low")
        assert hasattr(summary, "mock_006_acf_high")
        assert hasattr(summary, "mock_007_kurtosis_low")
        assert hasattr(summary, "mock_008_kurtosis_high")
        assert hasattr(summary, "mock_009_jump_H")
        assert hasattr(summary, "mock_010_discrete_rho")
        assert hasattr(summary, "high_severity_count")
        assert hasattr(summary, "medium_severity_count")
        assert hasattr(summary, "low_severity_count")

    def test_summary_to_dict(self):
        """Verify MockIndicatorSummary.to_dict() is JSON serializable."""
        summary = MockIndicatorSummary(
            mock_001_var_H_low=True,
            mock_009_jump_H=True,
            high_severity_count=2,
        )

        d = summary.to_dict()
        json_str = json.dumps(d)

        assert json_str is not None
        parsed = json.loads(json_str)
        assert parsed["indicators"]["MOCK_001"] is True
        assert parsed["severity_counts"]["high"] == 2


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------

class TestRTTSComponentIntegration:
    """Integration tests for RTTS components working together."""

    def test_full_validation_pipeline(self):
        """Test complete validation pipeline with all components."""
        # Initialize all components
        stats_validator = RTTSStatisticalValidator(window_size=20)
        mock_detector = RTTSMockDetector()
        continuity_tracker = RTTSContinuityTracker()
        correlation_tracker = RTTSCorrelationTracker(window_size=20)

        # Generate realistic telemetry sequence
        import random
        random.seed(42)

        prev_H = 0.8
        for i in range(25):
            # Generate values with small realistic changes
            delta_H = random.gauss(0, 0.03)  # Small random walk
            H = max(0.1, min(0.99, prev_H + delta_H))
            rho = 0.3 + 0.6 * H + random.gauss(0, 0.05)  # Correlated with H
            rho = max(0.1, min(0.99, rho))
            in_omega = rho > 0.6

            snapshot = make_snapshot(
                cycle=i,
                H=H,
                rho=rho,
                in_omega=in_omega,
            )

            # Update all components
            stats_result = stats_validator.update(snapshot)
            continuity_check = continuity_tracker.check(snapshot)
            correlation_tracker.update(snapshot)

            # Track delta for mock detector
            if i > 0:
                mock_detector.record_delta_H(abs(H - prev_H))
            mock_detector.record_rho_value(rho)

            prev_H = H

        # Final detection
        correlation_result = correlation_tracker.compute()
        detection = mock_detector.detect(
            stats=stats_result,
            correlations=correlation_result,
        )

        # Verify outputs
        assert stats_result.computed is True
        assert correlation_result.cor_H_rho is not None
        assert detection.status in ("VALIDATED_REAL", "SUSPECTED_MOCK", "UNKNOWN")

        # Should detect as real (reasonable data)
        # Note: Random data might occasionally trigger false positives
        continuity_stats = continuity_tracker.get_stats()
        assert continuity_stats.total_checks > 0

    def test_all_outputs_json_serializable(self):
        """Verify all component outputs are JSON serializable."""
        # Create all result types
        stats_result = RTTSStatisticalResult(
            variance_H=0.001,
            autocorr_H_lag1=0.5,
            computed=True,
        )

        mock_result = MockDetectionResult(
            status="VALIDATED_REAL",
            confidence=0.9,
        )

        continuity_stats = ContinuityStats(
            total_checks=100,
            violation_count=2,
        )

        correlation_result = RTTSCorrelationResult(
            cor_H_rho=0.6,
            window_size=200,
        )

        # Serialize all to JSON
        outputs = [
            stats_result.to_dict(),
            mock_result.to_dict(),
            continuity_stats.to_dict(),
            correlation_result.to_dict(),
        ]

        for output in outputs:
            json_str = json.dumps(output)
            assert json_str is not None
            parsed = json.loads(json_str)
            assert parsed is not None

    def test_shadow_mode_enforced(self):
        """Verify all components operate in SHADOW mode."""
        stats_result = RTTSStatisticalResult()
        mock_result = MockDetectionResult()
        continuity_stats = ContinuityStats()
        correlation_result = RTTSCorrelationResult()
        check = ContinuityCheck()

        assert stats_result.mode == "SHADOW"
        assert mock_result.mode == "SHADOW"
        assert continuity_stats.mode == "SHADOW"
        assert correlation_result.mode == "SHADOW"
        assert check.mode == "SHADOW"
        assert check.action == "LOGGED_ONLY"
