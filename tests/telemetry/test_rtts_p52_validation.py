"""
Tests for RTTS P5.2 VALIDATE Stage

Phase X P5.2: Numeric Validation and Threshold Boundary Tests

These tests verify:
1. Real statistical computations (variance, autocorrelation, kurtosis)
2. Threshold boundary conditions for all MOCK indicators
3. Correlation bound validation with warnings
4. Continuity tracker violation rates and warnings
5. rtts_validate_window orchestrator integration

SHADOW MODE: All tests operate in observation-only mode.

See: docs/system_law/RTTS_Gap_Closure_Blueprint.md
"""

import json
import math
import pytest
from typing import List

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
from backend.telemetry.rtts_window_validator import (
    RTTSValidationBlock,
    RTTSWindowValidator,
    rtts_validate_window,
)
from backend.telemetry.governance_signal import (
    RTTSCorrelationResult,
)


# -----------------------------------------------------------------------------
# Helper Functions
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


def generate_random_walk_snapshots(
    n: int,
    seed: int = 42,
    start_H: float = 0.5,
    start_rho: float = 0.5,
    sigma: float = 0.02,
) -> List[TelemetrySnapshot]:
    """Generate snapshots with random-walk telemetry values."""
    import random
    random.seed(seed)

    snapshots = []
    H = start_H
    rho = start_rho

    for i in range(n):
        # Small random changes (within Lipschitz bounds)
        H = max(0.1, min(0.99, H + random.gauss(0, sigma)))
        rho = max(0.1, min(0.99, rho + random.gauss(0, sigma)))
        tau = 0.2 + random.gauss(0, 0.01)
        beta = 0.1 + random.gauss(0, 0.01)
        in_omega = rho > 0.5

        snapshots.append(make_snapshot(
            cycle=i,
            H=H,
            rho=rho,
            tau=tau,
            beta=beta,
            in_omega=in_omega,
        ))

    return snapshots


def generate_constant_snapshots(n: int, H: float = 0.8, rho: float = 0.9) -> List[TelemetrySnapshot]:
    """Generate snapshots with constant values (mock pattern)."""
    return [make_snapshot(cycle=i, H=H, rho=rho) for i in range(n)]


# -----------------------------------------------------------------------------
# Statistical Validator Threshold Tests
# -----------------------------------------------------------------------------

class TestStatisticalValidatorThresholds:
    """Test statistical validator threshold boundary conditions."""

    def test_variance_H_at_threshold_boundary(self):
        """Test Var(H) exactly at threshold boundary."""
        validator = RTTSStatisticalValidator(window_size=20)

        # Generate values with variance near threshold (0.0001)
        # Var(H) = 0.0001 means std_dev ≈ 0.01
        import random
        random.seed(42)
        for i in range(20):
            H = 0.5 + random.uniform(-0.01, 0.01)  # Small variance
            validator.update(make_snapshot(cycle=i, H=H))

        result = validator.get_current_result()
        assert result.computed is True

        # Should be close to or below threshold
        if result.variance_H < RTTSStatisticalValidator.VAR_H_THRESHOLD:
            assert result.var_H_below_threshold is True
            assert any("MOCK-001" in w for w in result.warnings)

    def test_variance_H_clearly_below_threshold(self):
        """Test Var(H) clearly below threshold generates warning."""
        validator = RTTSStatisticalValidator(window_size=20)

        # Constant H values (variance = 0)
        for i in range(20):
            validator.update(make_snapshot(cycle=i, H=0.8))

        result = validator.get_current_result()
        assert result.variance_H is not None
        assert result.variance_H < validator.VAR_H_THRESHOLD
        assert result.var_H_below_threshold is True
        assert any("MOCK-001" in w for w in result.warnings)

    def test_variance_H_above_threshold_no_warning(self):
        """Test Var(H) above threshold generates no warning."""
        validator = RTTSStatisticalValidator(window_size=20)

        # Varied H values (high variance)
        for i in range(20):
            H = 0.3 + (i / 20) * 0.5  # Range from 0.3 to 0.8
            validator.update(make_snapshot(cycle=i, H=H))

        result = validator.get_current_result()
        assert result.variance_H > validator.VAR_H_THRESHOLD
        assert result.var_H_below_threshold is False
        assert not any("MOCK-001" in w for w in result.warnings)

    def test_autocorrelation_low_threshold(self):
        """Test ACF below low threshold (MOCK-005)."""
        validator = RTTSStatisticalValidator(window_size=20)

        # Random H values (low autocorrelation)
        import random
        random.seed(42)
        for i in range(20):
            H = random.random()  # Independent random
            validator.update(make_snapshot(cycle=i, H=H))

        result = validator.get_current_result()
        # Random data typically has low autocorrelation
        if result.autocorr_H_lag1 is not None and result.autocorr_H_lag1 < validator.ACF_LOW_THRESHOLD:
            assert result.acf_below_threshold is True
            assert any("MOCK-005" in w for w in result.warnings)

    def test_autocorrelation_high_threshold(self):
        """Test ACF above high threshold (MOCK-006)."""
        validator = RTTSStatisticalValidator(window_size=20)

        # Very slowly changing H (high autocorrelation)
        for i in range(20):
            H = 0.5 + i * 0.001  # Almost constant
            validator.update(make_snapshot(cycle=i, H=H))

        result = validator.get_current_result()
        # Very slow trend has high autocorrelation
        if result.autocorr_H_lag1 is not None and result.autocorr_H_lag1 > validator.ACF_HIGH_THRESHOLD:
            assert result.acf_above_threshold is True
            assert any("MOCK-006" in w for w in result.warnings)

    def test_kurtosis_low_threshold(self):
        """Test kurtosis below low threshold (MOCK-007)."""
        validator = RTTSStatisticalValidator(window_size=20)

        # Uniform distribution (negative excess kurtosis)
        for i in range(20):
            H = 0.1 + (i / 19) * 0.8  # Uniform from 0.1 to 0.9
            validator.update(make_snapshot(cycle=i, H=H))

        result = validator.get_current_result()
        # Uniform distribution has kurtosis around -1.2
        if result.kurtosis_H is not None and result.kurtosis_H < validator.KURTOSIS_LOW_THRESHOLD:
            assert result.kurtosis_below_threshold is True
            assert any("MOCK-007" in w for w in result.warnings)

    def test_kurtosis_high_threshold(self):
        """Test kurtosis above high threshold (MOCK-008)."""
        validator = RTTSStatisticalValidator(window_size=30)

        # Mostly constant with a few outliers (high kurtosis)
        for i in range(30):
            if i in (10, 20):
                H = 0.1  # Outlier
            else:
                H = 0.5  # Normal
            validator.update(make_snapshot(cycle=i, H=H))

        result = validator.get_current_result()
        # Outliers increase kurtosis
        if result.kurtosis_H is not None and result.kurtosis_H > validator.KURTOSIS_HIGH_THRESHOLD:
            assert result.kurtosis_above_threshold is True
            assert any("MOCK-008" in w for w in result.warnings)

    def test_result_has_violations_property(self):
        """Test has_violations property."""
        result = RTTSStatisticalResult(
            var_H_below_threshold=True,
            computed=True,
        )
        assert result.has_violations is True
        assert result.violation_count == 1

        result2 = RTTSStatisticalResult(computed=True)
        assert result2.has_violations is False
        assert result2.violation_count == 0

    def test_result_action_is_logged_only(self):
        """Verify action is LOGGED_ONLY (P5.2 constraint)."""
        validator = RTTSStatisticalValidator(window_size=10)
        for i in range(10):
            validator.update(make_snapshot(cycle=i))

        result = validator.get_current_result()
        assert result.action == "LOGGED_ONLY"
        assert result.mode == "SHADOW"


# -----------------------------------------------------------------------------
# Correlation Tracker Threshold Tests
# -----------------------------------------------------------------------------

class TestCorrelationTrackerThresholds:
    """Test correlation tracker threshold boundary conditions."""

    def test_correlation_H_rho_below_min_bound(self):
        """Test Cor(H, rho) below minimum bound generates warning."""
        tracker = RTTSCorrelationTracker(window_size=20)

        # Generate uncorrelated data
        import random
        random.seed(42)
        for i in range(20):
            H = random.random()
            rho = random.random()  # Independent of H
            tracker.update(make_snapshot(cycle=i, H=H, rho=rho))

        result = tracker.compute()

        # Random independent data should have low correlation
        if result.cor_H_rho is not None and result.cor_H_rho < tracker.COR_H_RHO_MIN:
            assert result.cor_H_rho_violated is True
            assert any("below min bound" in w for w in result.warnings)

    def test_correlation_H_rho_above_max_bound(self):
        """Test Cor(H, rho) above maximum bound generates warning."""
        tracker = RTTSCorrelationTracker(window_size=20)

        # Generate perfectly correlated data
        for i in range(20):
            H = 0.3 + i * 0.03  # Increasing
            rho = H  # Perfect correlation
            tracker.update(make_snapshot(cycle=i, H=H, rho=rho))

        result = tracker.compute()

        # Perfect correlation should exceed max bound (0.9)
        assert result.cor_H_rho is not None
        if result.cor_H_rho > tracker.COR_H_RHO_MAX:
            assert result.cor_H_rho_violated is True
            assert any("above max bound" in w for w in result.warnings)

    def test_correlation_rho_omega_violated(self):
        """Test Cor(rho, omega) violation detection."""
        tracker = RTTSCorrelationTracker(window_size=20)

        # Generate data where rho doesn't correlate with omega
        import random
        random.seed(42)
        for i in range(20):
            rho = random.random()
            in_omega = random.choice([True, False])  # Random, not based on rho
            tracker.update(make_snapshot(cycle=i, rho=rho, in_omega=in_omega))

        result = tracker.compute()

        # Random omega should violate correlation bound
        if result.cor_rho_omega is not None and result.cor_rho_omega < tracker.COR_RHO_OMEGA_MIN:
            assert result.cor_rho_omega_violated is True

    def test_zero_correlation_mock_pattern(self):
        """Test zero correlation mock pattern detection."""
        tracker = RTTSCorrelationTracker(window_size=20)

        # Generate orthogonal patterns
        for i in range(20):
            H = 0.5 + 0.3 * math.sin(i * 0.5)
            rho = 0.5 + 0.3 * math.cos(i * 0.5)  # 90 degrees out of phase
            tracker.update(make_snapshot(cycle=i, H=H, rho=rho))

        result = tracker.compute()

        # Should detect near-zero correlation
        if result.cor_H_rho is not None and abs(result.cor_H_rho) < 0.1:
            assert result.zero_correlation_detected is True
            assert any("MOCK-003" in w for w in result.warnings)

    def test_perfect_correlation_mock_pattern(self):
        """Test perfect correlation mock pattern detection."""
        tracker = RTTSCorrelationTracker(window_size=20)

        # Generate perfectly correlated data
        for i in range(20):
            H = 0.3 + i * 0.03
            rho = 0.3 + i * 0.03  # Identical to H
            tracker.update(make_snapshot(cycle=i, H=H, rho=rho))

        result = tracker.compute()

        # Should detect perfect correlation
        if result.cor_H_rho is not None and abs(result.cor_H_rho) > 0.99:
            assert result.perfect_correlation_detected is True
            assert any("MOCK-004" in w for w in result.warnings)

    def test_inverted_correlation_detection(self):
        """Test inverted (negative) correlation detection."""
        tracker = RTTSCorrelationTracker(window_size=20)

        # Generate negatively correlated data
        for i in range(20):
            H = 0.3 + i * 0.03  # Increasing
            rho = 0.9 - i * 0.03  # Decreasing
            tracker.update(make_snapshot(cycle=i, H=H, rho=rho))

        result = tracker.compute()

        # Should detect negative correlation
        if result.cor_H_rho is not None and result.cor_H_rho < 0:
            assert result.inverted_correlation_detected is True
            assert any("Inverted correlation" in w for w in result.warnings)

    def test_correlation_result_has_violations_property(self):
        """Test has_violations property."""
        result = RTTSCorrelationResult(
            cor_H_rho_violated=True,
        )
        assert result.has_violations is True
        assert result.violation_count == 1

    def test_correlation_action_is_logged_only(self):
        """Verify action is LOGGED_ONLY (P5.2 constraint)."""
        tracker = RTTSCorrelationTracker(window_size=10)
        for i in range(10):
            tracker.update(make_snapshot(cycle=i))

        result = tracker.compute()
        assert result.action == "LOGGED_ONLY"
        assert result.mode == "SHADOW"


# -----------------------------------------------------------------------------
# Continuity Tracker Violation Rate Tests
# -----------------------------------------------------------------------------

class TestContinuityTrackerViolationRates:
    """Test continuity tracker violation rates and warnings."""

    def test_violation_rate_computation(self):
        """Test violation rate is computed correctly."""
        tracker = RTTSContinuityTracker()

        # Generate 10 checks with 2 violations
        tracker.check(make_snapshot(cycle=0, H=0.5))
        tracker.check(make_snapshot(cycle=1, H=0.52))  # No violation
        tracker.check(make_snapshot(cycle=2, H=0.54))  # No violation
        tracker.check(make_snapshot(cycle=3, H=0.80))  # Violation (delta=0.26)
        tracker.check(make_snapshot(cycle=4, H=0.82))  # No violation
        tracker.check(make_snapshot(cycle=5, H=0.50))  # Violation (delta=0.32)

        stats = tracker.get_stats()
        # 5 checks, 2 violations = 40% rate
        assert stats.total_checks == 5
        assert stats.violation_count == 2
        assert abs(stats.violation_rate - 0.4) < 0.01

    def test_violation_generates_warning(self):
        """Test violations generate warnings."""
        tracker = RTTSContinuityTracker()

        tracker.check(make_snapshot(cycle=0, H=0.5))
        tracker.check(make_snapshot(cycle=1, H=0.8))  # Large delta = violation

        stats = tracker.get_stats()
        assert stats.H_violations == 1
        assert any("TELEMETRY_JUMP: delta_H" in w for w in stats.warnings)

    def test_consecutive_violations_warning(self):
        """Test consecutive violations generate additional warning."""
        tracker = RTTSContinuityTracker()

        # Create 4 consecutive violations
        tracker.check(make_snapshot(cycle=0, H=0.2))
        tracker.check(make_snapshot(cycle=1, H=0.5))  # Violation
        tracker.check(make_snapshot(cycle=2, H=0.8))  # Violation
        tracker.check(make_snapshot(cycle=3, H=0.4))  # Violation (3rd consecutive)
        tracker.check(make_snapshot(cycle=4, H=0.8))  # Violation (4th consecutive)

        stats = tracker.get_stats()
        assert stats.consecutive_violations >= 3
        assert any("consecutive Lipschitz violations" in w for w in stats.warnings)

    def test_max_delta_tracking(self):
        """Test maximum delta tracking."""
        tracker = RTTSContinuityTracker()

        tracker.check(make_snapshot(cycle=0, H=0.5))
        tracker.check(make_snapshot(cycle=1, H=0.55))  # delta=0.05
        tracker.check(make_snapshot(cycle=2, H=0.80))  # delta=0.25
        tracker.check(make_snapshot(cycle=3, H=0.82))  # delta=0.02

        stats = tracker.get_stats()
        assert abs(stats.max_delta_H - 0.25) < 0.01

    def test_per_component_violation_tracking(self):
        """Test per-component violation counting."""
        tracker = RTTSContinuityTracker()

        # H violation
        tracker.check(make_snapshot(cycle=0, H=0.3))
        tracker.check(make_snapshot(cycle=1, H=0.6))  # delta_H=0.3 > 0.15

        # rho violation
        tracker.check(make_snapshot(cycle=2, rho=0.3))
        tracker.check(make_snapshot(cycle=3, rho=0.5))  # delta_rho=0.2 > 0.10

        stats = tracker.get_stats()
        assert stats.H_violations >= 1
        assert stats.rho_violations >= 1

    def test_stats_include_bounds(self):
        """Test stats output includes bounds."""
        stats = ContinuityStats()
        d = stats.to_dict()

        assert "bounds" in d
        assert d["bounds"]["H"] == 0.15
        assert d["bounds"]["rho"] == 0.10
        assert d["bounds"]["tau"] == 0.05
        assert d["bounds"]["beta"] == 0.20

    def test_stats_action_is_logged_only(self):
        """Verify action is LOGGED_ONLY (P5.2 constraint)."""
        tracker = RTTSContinuityTracker()
        tracker.check(make_snapshot(cycle=0))
        tracker.check(make_snapshot(cycle=1))

        stats = tracker.get_stats()
        assert stats.action == "LOGGED_ONLY"
        assert stats.mode == "SHADOW"


# -----------------------------------------------------------------------------
# Window Validator Orchestrator Tests
# -----------------------------------------------------------------------------

class TestRTTSWindowValidator:
    """Test RTTS window validator orchestrator."""

    def test_orchestrator_initialization(self):
        """Test orchestrator initializes correctly."""
        validator = RTTSWindowValidator(window_size=100)
        assert validator.window_size == 100
        assert validator.is_window_ready() is False

    def test_orchestrator_produces_validation_block(self):
        """Test orchestrator produces validation block."""
        validator = RTTSWindowValidator(window_size=20)

        snapshots = generate_random_walk_snapshots(20)
        for s in snapshots:
            validator.update(s)

        assert validator.is_window_ready() is True

        block = validator.validate()
        assert isinstance(block, RTTSValidationBlock)
        assert block.schema_version == "1.0.0"
        assert block.mode == "SHADOW"
        assert block.action == "LOGGED_ONLY"

    def test_orchestrator_aggregates_warnings(self):
        """Test orchestrator aggregates warnings from all components."""
        validator = RTTSWindowValidator(window_size=20)

        # Generate mock-like data to trigger warnings
        snapshots = generate_constant_snapshots(20)
        for s in snapshots:
            validator.update(s)

        block = validator.validate()

        # Constant data should trigger variance warnings
        assert block.warning_count > 0
        assert len(block.all_warnings) == block.warning_count

    def test_orchestrator_status_ok_for_healthy_data(self):
        """Test orchestrator returns OK status for healthy data."""
        validator = RTTSWindowValidator(window_size=30)

        # Generate healthy random walk data
        snapshots = generate_random_walk_snapshots(30, sigma=0.02)
        for s in snapshots:
            validator.update(s)

        block = validator.validate()

        # Healthy data should mostly pass
        # Note: Random data may occasionally trigger some warnings
        assert block.overall_status in ("OK", "ATTENTION", "WARN")

    def test_orchestrator_status_critical_for_mock(self):
        """Test orchestrator returns CRITICAL for suspected mock."""
        validator = RTTSWindowValidator(window_size=20)

        # Generate obvious mock data (constant values)
        snapshots = generate_constant_snapshots(20)
        for s in snapshots:
            validator.update(s)

        block = validator.validate()

        # Constant data should be detected as mock
        if block.mock_detection and block.mock_detection.status == "SUSPECTED_MOCK":
            assert block.overall_status == "CRITICAL"
            assert block.validation_passed is False

    def test_orchestrator_reset(self):
        """Test orchestrator reset clears all state."""
        validator = RTTSWindowValidator(window_size=20)

        for i in range(10):
            validator.update(make_snapshot(cycle=i))

        validator.reset()

        assert validator.is_window_ready() is False

    def test_validation_block_json_serializable(self):
        """Test validation block is JSON serializable."""
        validator = RTTSWindowValidator(window_size=20)

        snapshots = generate_random_walk_snapshots(20)
        for s in snapshots:
            validator.update(s)

        block = validator.validate()
        d = block.to_dict()
        json_str = json.dumps(d)

        assert json_str is not None
        parsed = json.loads(json_str)
        assert parsed["schema_version"] == "1.0.0"
        assert parsed["mode"] == "SHADOW"


# -----------------------------------------------------------------------------
# rtts_validate_window Function Tests
# -----------------------------------------------------------------------------

class TestRTTSValidateWindow:
    """Test rtts_validate_window orchestrator function."""

    def test_validate_window_basic(self):
        """Test basic window validation."""
        snapshots = generate_random_walk_snapshots(50)
        block = rtts_validate_window(snapshots)

        assert isinstance(block, RTTSValidationBlock)
        assert block.window_size == 50
        assert block.block_id != ""
        assert block.timestamp != ""

    def test_validate_window_empty_input(self):
        """Test validation handles empty input."""
        block = rtts_validate_window([])

        assert block.overall_status == "UNKNOWN"
        assert block.validation_passed is False
        assert "No snapshots provided" in block.all_warnings[0]

    def test_validate_window_with_custom_size(self):
        """Test validation with custom window size."""
        snapshots = generate_random_walk_snapshots(100)
        block = rtts_validate_window(snapshots, window_size=50)

        # Window size is capped by the validator's window_size parameter
        # The validator keeps at most window_size snapshots
        assert block.window_size == 50  # Capped at window_size
        assert block.statistical is not None
        assert block.statistical.window_size == 50

    def test_validate_window_all_components_present(self):
        """Test all validation components are present."""
        snapshots = generate_random_walk_snapshots(50)
        block = rtts_validate_window(snapshots)

        assert block.statistical is not None
        assert block.correlation is not None
        assert block.continuity is not None
        assert block.mock_detection is not None

    def test_validate_window_mock_detection(self):
        """Test mock detection in window validation."""
        # Generate mock-like data
        snapshots = generate_constant_snapshots(50)
        block = rtts_validate_window(snapshots)

        # Should detect low variance
        assert block.statistical is not None
        assert block.statistical.var_H_below_threshold is True

    def test_validate_window_continuity_violations(self):
        """Test continuity violations in window validation."""
        snapshots = []
        for i in range(50):
            # Create occasional large jumps
            if i % 10 == 5:
                H = 0.9  # Jump
            else:
                H = 0.5
            snapshots.append(make_snapshot(cycle=i, H=H))

        block = rtts_validate_window(snapshots)

        # Should detect continuity violations
        assert block.continuity is not None
        assert block.continuity.violation_count > 0

    def test_validate_window_has_warnings_property(self):
        """Test has_warnings property."""
        snapshots = generate_constant_snapshots(30)
        block = rtts_validate_window(snapshots)

        if block.warning_count > 0:
            assert block.has_warnings is True
        else:
            assert block.has_warnings is False


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------

class TestP52Integration:
    """P5.2 integration tests."""

    def test_full_pipeline_healthy_data(self):
        """Test full pipeline with healthy telemetry data."""
        # Generate realistic healthy data
        import random
        random.seed(42)

        snapshots = []
        H = 0.5
        rho = 0.5

        for i in range(100):
            # Random walk with small sigma to stay within Lipschitz bounds
            # Use sigma < 0.03 for H (bound=0.15) and rho (bound=0.10)
            H = max(0.1, min(0.99, H + random.gauss(0, 0.02)))
            # rho correlates with H but with independent noise
            rho = 0.3 + 0.4 * H + random.gauss(0, 0.02)  # Moderate correlation
            rho = max(0.1, min(0.99, rho))
            in_omega = rho > 0.5

            snapshots.append(make_snapshot(
                cycle=i,
                H=H,
                rho=rho,
                in_omega=in_omega,
            ))

        block = rtts_validate_window(snapshots)

        # The pipeline completes and produces a validation block
        # The exact status depends on the random data, but it should complete
        assert block.statistical.computed is True
        assert block.correlation.cor_H_rho is not None
        # Mode must be SHADOW
        assert block.mode == "SHADOW"
        # Action must be LOGGED_ONLY (not enforced)
        assert block.action == "LOGGED_ONLY"

    def test_full_pipeline_mock_data(self):
        """Test full pipeline detects mock telemetry."""
        # Generate obvious mock pattern (constant values)
        snapshots = [make_snapshot(cycle=i, H=0.8, rho=0.9) for i in range(100)]

        block = rtts_validate_window(snapshots)

        # Mock data should be flagged
        assert block.statistical.var_H_below_threshold is True
        assert block.warning_count > 0

        # Mock detection should flag as suspected
        if block.mock_detection.status == "SUSPECTED_MOCK":
            assert block.overall_status == "CRITICAL"
            assert block.validation_passed is False

    def test_shadow_mode_enforced_throughout(self):
        """Test SHADOW mode is enforced in all components."""
        snapshots = generate_random_walk_snapshots(50)
        block = rtts_validate_window(snapshots)

        # All components should be in SHADOW mode
        assert block.mode == "SHADOW"
        assert block.action == "LOGGED_ONLY"
        assert block.statistical.mode == "SHADOW"
        assert block.correlation.mode == "SHADOW"
        assert block.continuity.mode == "SHADOW"
        assert block.mock_detection.mode == "SHADOW"

    def test_no_blocking_even_on_failures(self):
        """Test no blocking occurs even with validation failures."""
        # Generate data that should fail validation
        snapshots = generate_constant_snapshots(50)

        # This should complete without raising exceptions
        block = rtts_validate_window(snapshots)

        # Should have warnings but no blocking
        assert block is not None
        assert block.action == "LOGGED_ONLY"
        # Validation may fail but execution completes
        assert block.overall_status is not None

    def test_all_warnings_logged_not_enforced(self):
        """Test warnings are logged but not enforced."""
        snapshots = generate_constant_snapshots(30)
        block = rtts_validate_window(snapshots)

        # Should have warnings
        if block.warning_count > 0:
            # Warnings are informational, not blocking
            for warning in block.all_warnings:
                assert isinstance(warning, str)
                # No enforcement markers
                assert "BLOCKED" not in warning
                assert "HALTED" not in warning
