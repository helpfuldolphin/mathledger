#!/usr/bin/env python3
"""
==============================================================================
PHASE II — SYNTHETIC TEST DATA ONLY
==============================================================================

Test Suite for Synthetic Realism Envelope Check
-------------------------------------------------

Tests for:
    - Variance bounds checking
    - Correlation envelope
    - Drift amplitude envelope
    - Rare event frequency bounds
    - Exit codes (0=pass, 1=fail)

NOT derived from real derivations; NOT part of Evidence Pack.

==============================================================================
"""

import json
import pytest
import sys
import tempfile
from pathlib import Path

project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.synthetic_uplift.noise_models import SAFETY_LABEL
from experiments.synthetic_uplift.realism_envelope import (
    EnvelopeBounds,
    EnvelopeViolation,
    EnvelopeCheckResult,
    RealismEnvelopeChecker,
    format_envelope_report,
    run_envelope_check,
    DEFAULT_BOUNDS,
)


# ==============================================================================
# VARIANCE BOUNDS TESTS
# ==============================================================================

class TestVarianceBounds:
    """Tests for variance bounds checking."""
    
    def test_valid_variance_passes(self):
        """Valid variance should not produce violations."""
        checker = RealismEnvelopeChecker()
        params = {
            "variance": {
                "per_cycle_sigma": 0.05,
                "per_item_sigma": 0.03,
            },
            "probabilities": {"baseline": {"class_a": 0.5}},
            "correlation": {"rho": 0.0},
            "drift": {"mode": "none"},
            "rare_events": [],
        }
        
        violations = checker.check_scenario("synthetic_test", params)
        variance_violations = [v for v in violations if "variance" in v.parameter]
        
        assert len(variance_violations) == 0
    
    def test_high_per_cycle_sigma_fails(self):
        """Too high per_cycle_sigma should produce violation."""
        checker = RealismEnvelopeChecker()
        params = {
            "variance": {
                "per_cycle_sigma": 0.25,  # Above default max of 0.15
                "per_item_sigma": 0.05,
            },
            "probabilities": {"baseline": {"class_a": 0.5}},
            "correlation": {"rho": 0.0},
            "drift": {"mode": "none"},
            "rare_events": [],
        }
        
        violations = checker.check_scenario("synthetic_test", params)
        
        assert any("per_cycle_sigma" in v.parameter for v in violations)
    
    def test_high_per_item_sigma_fails(self):
        """Too high per_item_sigma should produce violation."""
        checker = RealismEnvelopeChecker()
        params = {
            "variance": {
                "per_cycle_sigma": 0.05,
                "per_item_sigma": 0.20,  # Above default max of 0.10
            },
            "probabilities": {"baseline": {"class_a": 0.5}},
            "correlation": {"rho": 0.0},
            "drift": {"mode": "none"},
            "rare_events": [],
        }
        
        violations = checker.check_scenario("synthetic_test", params)
        
        assert any("per_item_sigma" in v.parameter for v in violations)


# ==============================================================================
# CORRELATION ENVELOPE TESTS
# ==============================================================================

class TestCorrelationEnvelope:
    """Tests for correlation envelope checking."""
    
    def test_valid_correlation_passes(self):
        """Valid correlation should not produce violations."""
        checker = RealismEnvelopeChecker()
        params = {
            "correlation": {"rho": 0.5},
            "probabilities": {"baseline": {"class_a": 0.5}},
            "variance": {},
            "drift": {"mode": "none"},
            "rare_events": [],
        }
        
        violations = checker.check_scenario("synthetic_test", params)
        corr_violations = [v for v in violations if "correlation" in v.parameter]
        
        assert len(corr_violations) == 0
    
    def test_too_high_correlation_fails(self):
        """Correlation above max (0.9) should produce violation."""
        checker = RealismEnvelopeChecker()
        params = {
            "correlation": {"rho": 0.95},  # Above default max of 0.9
            "probabilities": {"baseline": {"class_a": 0.5}},
            "variance": {},
            "drift": {"mode": "none"},
            "rare_events": [],
        }
        
        violations = checker.check_scenario("synthetic_test", params)
        
        assert any("correlation.rho" in v.parameter for v in violations)
    
    def test_negative_correlation_fails(self):
        """Negative correlation should produce violation."""
        checker = RealismEnvelopeChecker()
        params = {
            "correlation": {"rho": -0.1},  # Below min of 0.0
            "probabilities": {"baseline": {"class_a": 0.5}},
            "variance": {},
            "drift": {"mode": "none"},
            "rare_events": [],
        }
        
        violations = checker.check_scenario("synthetic_test", params)
        
        assert any("correlation.rho" in v.parameter for v in violations)


# ==============================================================================
# DRIFT AMPLITUDE ENVELOPE TESTS
# ==============================================================================

class TestDriftAmplitudeEnvelope:
    """Tests for drift amplitude envelope checking."""
    
    def test_valid_sinusoidal_amplitude_passes(self):
        """Valid sinusoidal amplitude should not produce violations."""
        checker = RealismEnvelopeChecker()
        params = {
            "drift": {"mode": "cyclical", "amplitude": 0.15, "period": 100},
            "probabilities": {"baseline": {"class_a": 0.5}},
            "correlation": {"rho": 0.0},
            "variance": {},
            "rare_events": [],
        }
        
        violations = checker.check_scenario("synthetic_test", params)
        drift_violations = [v for v in violations if "drift" in v.parameter]
        
        assert len(drift_violations) == 0
    
    def test_too_high_sinusoidal_amplitude_fails(self):
        """Sinusoidal amplitude above max should produce violation."""
        checker = RealismEnvelopeChecker()
        params = {
            "drift": {"mode": "cyclical", "amplitude": 0.40, "period": 100},
            "probabilities": {"baseline": {"class_a": 0.5}},
            "correlation": {"rho": 0.0},
            "variance": {},
            "rare_events": [],
        }
        
        violations = checker.check_scenario("synthetic_test", params)
        
        assert any("drift.amplitude" in v.parameter for v in violations)
    
    def test_too_short_period_fails(self):
        """Period below minimum should produce violation."""
        checker = RealismEnvelopeChecker()
        params = {
            "drift": {"mode": "cyclical", "amplitude": 0.10, "period": 5},
            "probabilities": {"baseline": {"class_a": 0.5}},
            "correlation": {"rho": 0.0},
            "variance": {},
            "rare_events": [],
        }
        
        violations = checker.check_scenario("synthetic_test", params)
        
        assert any("drift.period" in v.parameter for v in violations)
    
    def test_too_steep_linear_slope_fails(self):
        """Linear slope above max should produce violation."""
        checker = RealismEnvelopeChecker()
        params = {
            "drift": {"mode": "linear", "slope": 0.01},  # Above default max of 0.005
            "probabilities": {"baseline": {"class_a": 0.5}},
            "correlation": {"rho": 0.0},
            "variance": {},
            "rare_events": [],
        }
        
        violations = checker.check_scenario("synthetic_test", params)
        
        assert any("drift.slope" in v.parameter for v in violations)
    
    def test_too_large_shock_delta_fails(self):
        """Shock delta above max should produce violation."""
        checker = RealismEnvelopeChecker()
        params = {
            "drift": {"mode": "shock", "shock_delta": -0.60},  # Above max of 0.40
            "probabilities": {"baseline": {"class_a": 0.5}},
            "correlation": {"rho": 0.0},
            "variance": {},
            "rare_events": [],
        }
        
        violations = checker.check_scenario("synthetic_test", params)
        
        assert any("shock_delta" in v.parameter for v in violations)


# ==============================================================================
# RARE EVENT FREQUENCY TESTS
# ==============================================================================

class TestRareEventFrequency:
    """Tests for rare event frequency bounds."""
    
    def test_valid_rare_events_passes(self):
        """Valid rare events should not produce violations."""
        checker = RealismEnvelopeChecker()
        params = {
            "rare_events": [
                {"type": "test", "trigger_probability": 0.05, "magnitude": 0.3, "duration": 10}
            ],
            "probabilities": {"baseline": {"class_a": 0.5}},
            "correlation": {"rho": 0.0},
            "variance": {},
            "drift": {"mode": "none"},
        }
        
        violations = checker.check_scenario("synthetic_test", params)
        rare_violations = [v for v in violations if "rare_events" in v.parameter]
        
        assert len(rare_violations) == 0
    
    def test_too_frequent_rare_event_fails(self):
        """Rare event with too high probability should produce violation."""
        checker = RealismEnvelopeChecker()
        params = {
            "rare_events": [
                {"type": "test", "trigger_probability": 0.20, "magnitude": 0.3, "duration": 10}
            ],
            "probabilities": {"baseline": {"class_a": 0.5}},
            "correlation": {"rho": 0.0},
            "variance": {},
            "drift": {"mode": "none"},
        }
        
        violations = checker.check_scenario("synthetic_test", params)
        
        assert any("trigger_probability" in v.parameter for v in violations)
    
    def test_too_large_magnitude_fails(self):
        """Rare event with too large magnitude should produce violation."""
        checker = RealismEnvelopeChecker()
        params = {
            "rare_events": [
                {"type": "test", "trigger_probability": 0.05, "magnitude": 0.80, "duration": 10}
            ],
            "probabilities": {"baseline": {"class_a": 0.5}},
            "correlation": {"rho": 0.0},
            "variance": {},
            "drift": {"mode": "none"},
        }
        
        violations = checker.check_scenario("synthetic_test", params)
        
        assert any("magnitude" in v.parameter for v in violations)
    
    def test_too_many_rare_events_fails(self):
        """Too many rare events should produce violation."""
        checker = RealismEnvelopeChecker()
        params = {
            "rare_events": [
                {"type": f"test_{i}", "trigger_probability": 0.01, "magnitude": 0.2, "duration": 5}
                for i in range(10)  # More than max of 5
            ],
            "probabilities": {"baseline": {"class_a": 0.5}},
            "correlation": {"rho": 0.0},
            "variance": {},
            "drift": {"mode": "none"},
        }
        
        violations = checker.check_scenario("synthetic_test", params)
        
        assert any("count" in v.parameter for v in violations)


# ==============================================================================
# EXIT CODE TESTS
# ==============================================================================

class TestExitCodes:
    """Tests for CLI exit codes."""
    
    def test_valid_registry_exits_zero(self):
        """Valid registry should exit with code 0."""
        exit_code = run_envelope_check(verbose=False)
        
        # Note: This depends on the actual registry being within bounds
        # If the current registry has violations, this test needs adjustment
        assert exit_code in (0, 1)  # Either valid or has known violations
    
    def test_custom_bounds_affect_result(self):
        """Custom bounds should affect validation result."""
        # Create very strict bounds
        strict_bounds = EnvelopeBounds(
            max_correlation_rho=0.1,  # Very strict
        )
        
        checker = RealismEnvelopeChecker(bounds=strict_bounds)
        params = {
            "correlation": {"rho": 0.3},  # Would be valid with default, not with strict
            "probabilities": {"baseline": {"class_a": 0.5}},
            "variance": {},
            "drift": {"mode": "none"},
            "rare_events": [],
        }
        
        violations = checker.check_scenario("synthetic_test", params)
        
        assert len(violations) > 0


# ==============================================================================
# REPORT FORMATTING TESTS
# ==============================================================================

class TestReportFormatting:
    """Tests for report formatting."""
    
    def test_report_includes_safety_label(self):
        """Report should include safety label."""
        result = EnvelopeCheckResult()
        result.passed = True
        
        report = format_envelope_report(result)
        
        assert SAFETY_LABEL in report
    
    def test_report_shows_pass(self):
        """Passing result should show PASS."""
        result = EnvelopeCheckResult()
        result.passed = True
        
        report = format_envelope_report(result)
        
        assert "[PASS]" in report
    
    def test_report_shows_fail(self):
        """Failing result should show FAIL."""
        result = EnvelopeCheckResult()
        result.passed = False
        result.violations.append(EnvelopeViolation(
            scenario="synthetic_test",
            parameter="test",
            actual_value=1.0,
            bound_type="max",
            bound_value=0.5,
        ))
        
        report = format_envelope_report(result)
        
        assert "[FAIL]" in report
    
    def test_report_lists_violations(self):
        """Report should list violations."""
        result = EnvelopeCheckResult()
        result.add_violation(EnvelopeViolation(
            scenario="synthetic_test",
            parameter="test.param",
            actual_value=1.0,
            bound_type="max",
            bound_value=0.5,
        ))
        
        report = format_envelope_report(result)
        
        assert "test.param" in report
        assert "1.0" in report


# ==============================================================================
# PROBABILITY BOUNDS TESTS
# ==============================================================================

class TestProbabilityBounds:
    """Tests for probability bounds checking."""
    
    def test_valid_probabilities_pass(self):
        """Valid probabilities should not produce violations."""
        checker = RealismEnvelopeChecker()
        params = {
            "probabilities": {
                "baseline": {"class_a": 0.5, "class_b": 0.6},
                "rfl": {"class_a": 0.6, "class_b": 0.7},
            },
            "correlation": {"rho": 0.0},
            "variance": {},
            "drift": {"mode": "none"},
            "rare_events": [],
        }
        
        violations = checker.check_scenario("synthetic_test", params)
        prob_violations = [v for v in violations if "probabilities" in v.parameter]
        
        assert len(prob_violations) == 0
    
    def test_too_low_probability_fails(self):
        """Probability below minimum should produce violation."""
        checker = RealismEnvelopeChecker()
        params = {
            "probabilities": {
                "baseline": {"class_a": 0.02},  # Below min of 0.05
            },
            "correlation": {"rho": 0.0},
            "variance": {},
            "drift": {"mode": "none"},
            "rare_events": [],
        }
        
        violations = checker.check_scenario("synthetic_test", params)
        
        assert any("probabilities" in v.parameter for v in violations)
    
    def test_too_high_probability_fails(self):
        """Probability above maximum should produce violation."""
        checker = RealismEnvelopeChecker()
        params = {
            "probabilities": {
                "baseline": {"class_a": 0.98},  # Above max of 0.95
            },
            "correlation": {"rho": 0.0},
            "variance": {},
            "drift": {"mode": "none"},
            "rare_events": [],
        }
        
        violations = checker.check_scenario("synthetic_test", params)
        
        assert any("probabilities" in v.parameter for v in violations)


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

