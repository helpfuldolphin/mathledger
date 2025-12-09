"""
Tests for mock oracle failure modes.

Verifies that timeout, abstention, error, and crash behaviors work
correctly under various configurations.

ABSOLUTE SAFEGUARD: These tests exercise the mock oracle only — never production.
"""

from __future__ import annotations

import os

import pytest

os.environ["MATHLEDGER_ALLOW_MOCK_ORACLE"] = "1"

from backend.verification import (
    MockOracleConfig,
    MockVerifiableOracle,
    MockOracleCrashError,
    MockOracleExpectations,
)


@pytest.mark.unit
class TestTimeoutBehavior:
    """Tests for timeout simulation."""
    
    def test_timeout_result_flags(self, default_oracle: MockVerifiableOracle):
        """Timeout bucket sets correct flags."""
        formula = MockOracleExpectations.get_timeout_formula("default")
        result = default_oracle.verify(formula)
        
        assert result.timed_out is True
        assert result.verified is False
        assert result.abstained is False
        assert result.crashed is False
        assert result.reason == "mock-timeout"
        assert result.bucket == "timeout"
    
    def test_timeout_latency_uses_config(self):
        """Timeout latency uses configured timeout_ms."""
        config = MockOracleConfig(
            slice_profile="default",
            timeout_ms=200,
            latency_jitter_pct=0.0,  # No jitter for exact test
        )
        oracle = MockVerifiableOracle(config)
        
        formula = MockOracleExpectations.get_timeout_formula("default")
        result = oracle.verify(formula)
        
        if result.bucket == "timeout":  # Guard in case formula bucket changes
            assert result.latency_ms == 200
    
    def test_timeout_with_jitter(self):
        """Timeout latency respects jitter configuration."""
        config = MockOracleConfig(
            slice_profile="default",
            timeout_ms=100,
            latency_jitter_pct=0.20,  # ±20%
        )
        oracle = MockVerifiableOracle(config)
        
        formula = MockOracleExpectations.get_timeout_formula("default")
        result = oracle.verify(formula)
        
        if result.bucket == "timeout":
            assert 80 <= result.latency_ms <= 120


@pytest.mark.unit
class TestAbstentionBehavior:
    """Tests for abstention simulation."""
    
    def test_abstain_result_flags(self, default_oracle: MockVerifiableOracle):
        """Abstain bucket sets correct flags."""
        formula = MockOracleExpectations.get_abstain_formula("default")
        result = default_oracle.verify(formula)
        
        assert result.abstained is True
        assert result.verified is False
        assert result.timed_out is False
        assert result.crashed is False
        assert result.reason == "mock-abstain"
        assert result.bucket == "abstain"
    
    def test_abstain_is_not_failure(self, default_oracle: MockVerifiableOracle):
        """Abstention is distinct from failure."""
        abstain_formula = MockOracleExpectations.get_abstain_formula("default")
        failed_formula = MockOracleExpectations.get_failed_formula("default")
        
        abstain_result = default_oracle.verify(abstain_formula)
        failed_result = default_oracle.verify(failed_formula)
        
        assert abstain_result.abstained is True
        assert abstain_result.verified is False
        
        assert failed_result.abstained is False
        assert failed_result.verified is False


@pytest.mark.unit
class TestErrorBehavior:
    """Tests for error simulation."""
    
    def test_error_result_flags(self, default_oracle: MockVerifiableOracle):
        """Error bucket sets correct flags."""
        formula = MockOracleExpectations.get_error_formula("default")
        result = default_oracle.verify(formula)
        
        assert result.verified is False
        assert result.abstained is False
        assert result.timed_out is False
        assert result.crashed is False
        assert result.reason == "mock-error"
        assert result.bucket == "error"
    
    def test_error_has_low_latency(self, default_oracle: MockVerifiableOracle):
        """Error bucket has low base latency (fast failure)."""
        from backend.verification.mock_config import BUCKET_BASE_LATENCY
        
        formula = MockOracleExpectations.get_error_formula("default")
        result = default_oracle.verify(formula)
        
        if result.bucket == "error":
            assert result.latency_ms <= 10  # Should be fast


@pytest.mark.unit
class TestCrashBehavior:
    """Tests for crash simulation."""
    
    def test_crash_disabled_returns_result(self, default_oracle: MockVerifiableOracle):
        """With crashes disabled, crash bucket returns result instead of raising."""
        formula = MockOracleExpectations.get_crash_formula("default")
        
        # default_oracle has enable_crashes=False
        result = default_oracle.verify(formula)
        
        assert result.crashed is True
        assert result.verified is False
        assert result.reason == "mock-crash-disabled"
    
    def test_crash_enabled_raises(self, crash_enabled_oracle: MockVerifiableOracle):
        """With crashes enabled, crash bucket raises MockOracleCrashError."""
        formula = MockOracleExpectations.get_crash_formula("default")
        
        with pytest.raises(MockOracleCrashError) as exc_info:
            crash_enabled_oracle.verify(formula)
        
        assert "mock-crash" in str(exc_info.value)
        assert exc_info.value.formula == formula
    
    def test_crash_exception_contains_hash(self, crash_enabled_oracle: MockVerifiableOracle):
        """Crash exception contains the formula hash."""
        formula = MockOracleExpectations.get_crash_formula("default")
        
        try:
            crash_enabled_oracle.verify(formula)
            pytest.fail("Expected MockOracleCrashError")
        except MockOracleCrashError as e:
            assert e.hash_int > 0
            assert e.formula == formula
    
    def test_batch_with_crash_stops_early(self, crash_enabled_oracle: MockVerifiableOracle):
        """Batch verification stops on first crash when crashes enabled."""
        formulas = [
            MockOracleExpectations.get_verified_formula("default"),
            MockOracleExpectations.get_crash_formula("default"),
            MockOracleExpectations.get_verified_formula("default", index=1),
        ]
        
        with pytest.raises(MockOracleCrashError):
            crash_enabled_oracle.verify_batch(formulas)


@pytest.mark.unit
class TestFailedBehavior:
    """Tests for explicit failure simulation."""
    
    def test_failed_result_flags(self, default_oracle: MockVerifiableOracle):
        """Failed bucket sets all flags to False (except implicit failure)."""
        formula = MockOracleExpectations.get_failed_formula("default")
        result = default_oracle.verify(formula)
        
        assert result.verified is False
        assert result.abstained is False
        assert result.timed_out is False
        assert result.crashed is False
        assert result.reason == "mock-failed"
        assert result.bucket == "failed"


@pytest.mark.unit
class TestVerifiedBehavior:
    """Tests for successful verification simulation."""
    
    def test_verified_result_flags(self, default_oracle: MockVerifiableOracle):
        """Verified bucket sets verified=True."""
        formula = MockOracleExpectations.get_verified_formula("default")
        result = default_oracle.verify(formula)
        
        assert result.verified is True
        assert result.abstained is False
        assert result.timed_out is False
        assert result.crashed is False
        assert result.reason == "mock-verified"
        assert result.bucket == "verified"


@pytest.mark.unit
class TestConfigValidation:
    """Tests for configuration validation."""
    
    def test_invalid_profile_rejected(self):
        """Invalid slice_profile raises ValueError."""
        with pytest.raises(ValueError, match="Invalid slice_profile"):
            MockOracleConfig(slice_profile="nonexistent")
    
    def test_negative_timeout_rejected(self):
        """Negative timeout_ms raises ValueError."""
        with pytest.raises(ValueError, match="timeout_ms must be non-negative"):
            MockOracleConfig(timeout_ms=-10)
    
    def test_invalid_jitter_rejected(self):
        """Jitter outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="latency_jitter_pct must be"):
            MockOracleConfig(latency_jitter_pct=1.5)
        
        with pytest.raises(ValueError, match="latency_jitter_pct must be"):
            MockOracleConfig(latency_jitter_pct=-0.1)


@pytest.mark.unit
class TestResultValidation:
    """Tests for result dataclass validation."""
    
    def test_multiple_true_flags_rejected(self):
        """Result with multiple True flags raises ValueError."""
        from backend.verification.mock_config import MockVerificationResult
        
        with pytest.raises(ValueError, match="At most one"):
            MockVerificationResult(
                verified=True,
                abstained=True,  # Can't be both!
                timed_out=False,
                crashed=False,
                reason="invalid",
                latency_ms=10,
            )
    
    def test_all_false_is_valid(self):
        """Result with all flags False is valid (explicit failure)."""
        from backend.verification.mock_config import MockVerificationResult
        
        result = MockVerificationResult(
            verified=False,
            abstained=False,
            timed_out=False,
            crashed=False,
            reason="mock-failed",
            latency_ms=10,
        )
        assert result.reason == "mock-failed"

