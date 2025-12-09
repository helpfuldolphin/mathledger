"""
Test Suite for Telemetry Runtime

Tests for run_lean_with_monitoring and related functions.

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
"""

import unittest
import time
from backend.verification.telemetry import (
    run_lean_with_monitoring,
    LeanVerificationTelemetry,
)
from backend.verification.error_codes import VerifierErrorCode, VerifierTier


class TestTelemetryRuntime(unittest.TestCase):
    """Test suite for telemetry runtime."""
    
    def test_basic_execution(self):
        """Test basic execution without noise."""
        telemetry = run_lean_with_monitoring(
            module_name="Test.Module",
            tier=VerifierTier.BALANCED,
            timeout_s=5.0,
            context="test_basic",
            master_seed=12345,
            noise_config=None,
        )
        
        # Should complete
        self.assertIsInstance(telemetry, LeanVerificationTelemetry)
        self.assertEqual(telemetry.module_name, "Test.Module")
        self.assertGreater(telemetry.duration_ms, 0)
    
    def test_timeout_injection(self):
        """Test timeout noise injection."""
        noise_config = {
            "balanced": {
                "timeout_rate": 1.0,  # Always timeout
                "spurious_fail_rate": 0.0,
                "spurious_pass_rate": 0.0,
            }
        }
        
        telemetry = run_lean_with_monitoring(
            module_name="Test.Module",
            tier=VerifierTier.BALANCED,
            timeout_s=5.0,
            context="test_timeout",
            master_seed=12345,
            noise_config=noise_config,
        )
        
        # Should timeout
        self.assertEqual(telemetry.outcome, VerifierErrorCode.VERIFIER_TIMEOUT)
        self.assertFalse(telemetry.success)
        self.assertTrue(telemetry.noise_injected)
        self.assertEqual(telemetry.noise_type, "timeout")
    
    def test_deterministic_noise(self):
        """Test that identical seeds produce identical noise."""
        noise_config = {
            "balanced": {
                "timeout_rate": 0.5,
                "spurious_fail_rate": 0.3,
                "spurious_pass_rate": 0.1,
            }
        }
        
        # Run twice with same seed
        telemetry1 = run_lean_with_monitoring(
            module_name="Test.Module",
            tier=VerifierTier.BALANCED,
            timeout_s=5.0,
            context="test_deterministic",
            master_seed=12345,
            noise_config=noise_config,
        )
        
        telemetry2 = run_lean_with_monitoring(
            module_name="Test.Module",
            tier=VerifierTier.BALANCED,
            timeout_s=5.0,
            context="test_deterministic",
            master_seed=12345,
            noise_config=noise_config,
        )
        
        # Outcomes should be identical
        self.assertEqual(telemetry1.outcome, telemetry2.outcome)
        self.assertEqual(telemetry1.noise_injected, telemetry2.noise_injected)
        self.assertEqual(telemetry1.noise_type, telemetry2.noise_type)


if __name__ == "__main__":
    unittest.main()
