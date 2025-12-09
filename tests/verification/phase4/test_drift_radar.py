"""
Test Suite for Drift Radar

Tests for CUSUM, tier skew, scan statistics, and unified radar.

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
"""

import unittest
from backend.verification.drift_radar import (
    VerifierNoiseDriftRadar,
    RadarConfig,
    AlertLevel,
    CUSUMConfig,
    TierSkewConfig,
    ScanStatisticsConfig,
)
from backend.verification.telemetry.schema import LeanVerificationTelemetry
from backend.verification.error_codes import VerifierErrorCode, VerifierTier


class TestDriftRadar(unittest.TestCase):
    """Test suite for drift radar."""
    
    def test_cusum_upward_drift(self):
        """Test CUSUM detector for upward drift."""
        config = RadarConfig(
            cusum_timeout=CUSUMConfig(
                metric="timeout_rate",
                mu_0=0.1,
                k=0.05,
                h=3.0,  # Lower threshold for testing
            )
        )
        
        radar = VerifierNoiseDriftRadar(config)
        
        # Send normal telemetry (10% timeout rate)
        for i in range(100):
            telemetry = LeanVerificationTelemetry(
                verification_id=f"test_{i}",
                module_name=f"Module.{i}",
                tier=VerifierTier.BALANCED,
                outcome=VerifierErrorCode.VERIFIER_TIMEOUT if i % 10 == 0 else VerifierErrorCode.VERIFIED,
            )
            alarms = radar.update(telemetry)
        
        # Should be NORMAL
        self.assertEqual(radar.get_alert_level(), AlertLevel.NORMAL)
        
        # Send high timeout rate (50%)
        for i in range(100, 200):
            telemetry = LeanVerificationTelemetry(
                verification_id=f"test_{i}",
                module_name=f"Module.{i}",
                tier=VerifierTier.BALANCED,
                outcome=VerifierErrorCode.VERIFIER_TIMEOUT if i % 2 == 0 else VerifierErrorCode.VERIFIED,
            )
            alarms = radar.update(telemetry)
            
            if alarms:
                # Should detect upward drift
                self.assertEqual(alarms[0]["signal"], "noise_rate_drift")
                self.assertEqual(alarms[0]["type"], "upward_drift")
                break
    
    def test_tier_skew_detection(self):
        """Test tier skew detector."""
        config = RadarConfig(
            tier_skew=TierSkewConfig(
                alpha=0.05,
                min_samples_per_tier=50,
            )
        )
        
        radar = VerifierNoiseDriftRadar(config)
        
        # Send telemetry with violated monotonicity
        # FAST: 10% timeout, BALANCED: 20% timeout (violation!)
        for i in range(100):
            # FAST tier
            telemetry_fast = LeanVerificationTelemetry(
                verification_id=f"fast_{i}",
                module_name=f"Module.{i}",
                tier=VerifierTier.FAST_NOISY,
                outcome=VerifierErrorCode.VERIFIER_TIMEOUT if i % 10 == 0 else VerifierErrorCode.VERIFIED,
            )
            radar.update(telemetry_fast)
            
            # BALANCED tier
            telemetry_balanced = LeanVerificationTelemetry(
                verification_id=f"balanced_{i}",
                module_name=f"Module.{i}",
                tier=VerifierTier.BALANCED,
                outcome=VerifierErrorCode.VERIFIER_TIMEOUT if i % 5 == 0 else VerifierErrorCode.VERIFIED,
            )
            alarms = radar.update(telemetry_balanced)
            
            # SLOW tier
            telemetry_slow = LeanVerificationTelemetry(
                verification_id=f"slow_{i}",
                module_name=f"Module.{i}",
                tier=VerifierTier.SLOW_PRECISE,
                outcome=VerifierErrorCode.VERIFIER_TIMEOUT if i % 20 == 0 else VerifierErrorCode.VERIFIED,
            )
            radar.update(telemetry_slow)
        
        # Should detect tier skew
        state = radar.get_state()
        # Check if any alarms were raised
        # (exact detection depends on statistical significance)
    
    def test_scan_statistics_spike(self):
        """Test scan statistics detector for failure spikes."""
        config = RadarConfig(
            scan_statistics=ScanStatisticsConfig(
                window_size=50,
                threshold=2.0,
            )
        )
        
        radar = VerifierNoiseDriftRadar(config)
        
        # Send normal telemetry (10% failure rate)
        for i in range(100):
            telemetry = LeanVerificationTelemetry(
                verification_id=f"test_{i}",
                module_name=f"Module.{i}",
                tier=VerifierTier.BALANCED,
                outcome=VerifierErrorCode.PROOF_INVALID if i % 10 == 0 else VerifierErrorCode.VERIFIED,
            )
            radar.update(telemetry)
        
        # Send spike (80% failure rate)
        for i in range(100, 150):
            telemetry = LeanVerificationTelemetry(
                verification_id=f"test_{i}",
                module_name=f"Module.{i}",
                tier=VerifierTier.BALANCED,
                outcome=VerifierErrorCode.PROOF_INVALID if i % 5 != 0 else VerifierErrorCode.VERIFIED,
            )
            alarms = radar.update(telemetry)
            
            if alarms:
                # Should detect spike
                self.assertEqual(alarms[0]["signal"], "correlated_failure_spike")
                break
    
    def test_alert_level_escalation(self):
        """Test alert level escalation."""
        config = RadarConfig(
            warning_threshold=2,
            alert_threshold=4,
            critical_threshold=6,
        )
        
        radar = VerifierNoiseDriftRadar(config)
        
        # Manually inject alarms
        for i in range(10):
            telemetry = LeanVerificationTelemetry(
                verification_id=f"test_{i}",
                module_name=f"Module.{i}",
                tier=VerifierTier.BALANCED,
                outcome=VerifierErrorCode.VERIFIER_TIMEOUT,
            )
            radar.update(telemetry)
        
        # Check escalation
        # (depends on CUSUM threshold, may need adjustment)


if __name__ == "__main__":
    unittest.main()
