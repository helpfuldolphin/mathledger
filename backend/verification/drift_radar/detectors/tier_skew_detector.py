"""
Tier Skew Detector

Detects violations of tier monotonicity invariant using two-proportion z-test.

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
Status: Production Ready
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from collections import defaultdict
import math
from .base import DriftDetector, DetectorConfig
from backend.verification.telemetry.schema import LeanVerificationTelemetry
from backend.verification.error_codes import VerifierErrorCode, VerifierTier


@dataclass
class TierSkewConfig(DetectorConfig):
    """Configuration for tier skew detector."""
    
    alpha: float = 0.05  # Significance level
    min_samples_per_tier: int = 100  # Minimum samples before testing


class TierSkewDetector(DriftDetector):
    """Tier skew detector using two-proportion z-test.
    
    Invariant: timeout_rate(FAST) >= timeout_rate(BALANCED) >= timeout_rate(SLOW)
    
    Test:
        H_0: p1 >= p2
        H_1: p1 < p2
        
        z = (p1 - p2) / sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
    """
    
    def __init__(self, config: TierSkewConfig):
        """Initialize tier skew detector."""
        super().__init__(config)
        self.config: TierSkewConfig = config
        
        # Tier statistics
        self.tier_counts: Dict[str, int] = defaultdict(int)
        self.tier_timeouts: Dict[str, int] = defaultdict(int)
    
    def update(self, telemetry: LeanVerificationTelemetry) -> Optional[Dict[str, Any]]:
        """Update tier statistics."""
        
        if not self.config.enabled:
            return None
        
        # Update counts
        tier_name = telemetry.tier.value
        self.tier_counts[tier_name] += 1
        
        if telemetry.outcome == VerifierErrorCode.VERIFIER_TIMEOUT:
            self.tier_timeouts[tier_name] += 1
        
        # Check for skew if enough samples
        if all(self.tier_counts[tier] >= self.config.min_samples_per_tier 
               for tier in ["fast_noisy", "balanced", "slow_precise"]):
            return self._check_tier_monotonicity()
        
        return None
    
    def _check_tier_monotonicity(self) -> Optional[Dict[str, Any]]:
        """Check tier monotonicity invariant."""
        
        # Compute timeout rates
        rates = {}
        for tier in ["fast_noisy", "balanced", "slow_precise"]:
            n = self.tier_counts[tier]
            k = self.tier_timeouts[tier]
            rates[tier] = k / n if n > 0 else 0.0
        
        # Check pairs: FAST >= BALANCED, BALANCED >= SLOW
        pairs = [
            ("fast_noisy", "balanced"),
            ("balanced", "slow_precise"),
        ]
        
        for tier1, tier2 in pairs:
            p1 = rates[tier1]
            p2 = rates[tier2]
            n1 = self.tier_counts[tier1]
            n2 = self.tier_counts[tier2]
            
            # Two-proportion z-test
            if p1 < p2:
                # Potential violation
                z, p_value = self._two_proportion_z_test(
                    self.tier_timeouts[tier1], n1,
                    self.tier_timeouts[tier2], n2,
                )
                
                if p_value < self.config.alpha:
                    # Significant violation
                    alarm = {
                        "signal": "tier_skew",
                        "detector": "tier_skew",
                        "tier1": tier1,
                        "tier2": tier2,
                        "rate1": p1,
                        "rate2": p2,
                        "z": z,
                        "p_value": p_value,
                        "alpha": self.config.alpha,
                    }
                    self.alarm_count += 1
                    return alarm
        
        return None
    
    def _two_proportion_z_test(
        self,
        k1: int, n1: int,
        k2: int, n2: int,
    ) -> tuple:
        """Compute two-proportion z-test.
        
        Args:
            k1: Successes in sample 1
            n1: Size of sample 1
            k2: Successes in sample 2
            n2: Size of sample 2
        
        Returns:
            Tuple of (z_statistic, p_value)
        """
        
        p1 = k1 / n1 if n1 > 0 else 0.0
        p2 = k2 / n2 if n2 > 0 else 0.0
        
        # Pooled proportion
        p_pooled = (k1 + k2) / (n1 + n2) if (n1 + n2) > 0 else 0.0
        
        # Standard error
        se = math.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
        
        if se == 0:
            return 0.0, 1.0
        
        # Z-statistic
        z = (p1 - p2) / se
        
        # P-value (one-tailed, left)
        from scipy import stats
        p_value = stats.norm.cdf(z)
        
        return z, p_value
    
    def reset(self) -> None:
        """Reset tier statistics."""
        self.tier_counts.clear()
        self.tier_timeouts.clear()
    
    def get_state(self) -> Dict[str, Any]:
        """Get current tier statistics."""
        rates = {}
        for tier in self.tier_counts:
            n = self.tier_counts[tier]
            k = self.tier_timeouts[tier]
            rates[tier] = k / n if n > 0 else 0.0
        
        return {
            "tier_counts": dict(self.tier_counts),
            "tier_timeouts": dict(self.tier_timeouts),
            "timeout_rates": rates,
            "alarm_count": self.alarm_count,
        }
