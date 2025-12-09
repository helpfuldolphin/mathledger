"""
CUSUM Detector for Noise Rate Drift

Detects small shifts in noise rate using cumulative sum control chart.

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
Status: Production Ready
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from .base import DriftDetector, DetectorConfig
from backend.verification.telemetry.schema import LeanVerificationTelemetry
from backend.verification.error_codes import VerifierErrorCode


@dataclass
class CUSUMConfig(DetectorConfig):
    """Configuration for CUSUM detector."""
    
    mu_0: float = 0.1      # Target noise rate
    k: float = 0.05        # Slack parameter (typically 0.5 * sigma)
    h: float = 5.0         # Threshold for alarm
    metric: str = "timeout_rate"  # Metric to monitor


class CUSUMDetector(DriftDetector):
    """CUSUM control chart for noise rate drift detection.
    
    Model:
        S_t^+ = max(0, S_{t-1}^+ + x_t - (μ_0 + k))  (upward drift)
        S_t^- = max(0, S_{t-1}^- - x_t + (μ_0 - k))  (downward drift)
    
    Alarm if S_t^+ > h or S_t^- > h
    """
    
    def __init__(self, config: CUSUMConfig):
        """Initialize CUSUM detector."""
        super().__init__(config)
        self.config: CUSUMConfig = config
        
        # CUSUM statistics
        self.S_plus = 0.0
        self.S_minus = 0.0
        
        # Sample count
        self.n = 0
    
    def update(self, telemetry: LeanVerificationTelemetry) -> Optional[Dict[str, Any]]:
        """Update CUSUM with new observation."""
        
        if not self.config.enabled:
            return None
        
        # Extract observation based on metric
        if self.config.metric == "timeout_rate":
            x = 1.0 if telemetry.outcome == VerifierErrorCode.VERIFIER_TIMEOUT else 0.0
        elif self.config.metric == "spurious_fail_rate":
            x = 1.0 if (telemetry.outcome == VerifierErrorCode.PROOF_INVALID and 
                       telemetry.ground_truth == "VERIFIED") else 0.0
        elif self.config.metric == "spurious_pass_rate":
            x = 1.0 if (telemetry.outcome == VerifierErrorCode.VERIFIED and 
                       telemetry.ground_truth == "INVALID") else 0.0
        else:
            return None
        
        # Update CUSUM statistics
        self.S_plus = max(0.0, self.S_plus + x - (self.config.mu_0 + self.config.k))
        self.S_minus = max(0.0, self.S_minus - x + (self.config.mu_0 - self.config.k))
        self.n += 1
        
        # Check for alarm
        alarm = None
        
        if self.S_plus > self.config.h:
            alarm = {
                "signal": "noise_rate_drift",
                "detector": "cusum",
                "type": "upward_drift",
                "metric": self.config.metric,
                "S_plus": self.S_plus,
                "threshold": self.config.h,
                "n": self.n,
            }
            self.alarm_count += 1
            # Reset after alarm
            self.S_plus = 0.0
        
        elif self.S_minus > self.config.h:
            alarm = {
                "signal": "noise_rate_drift",
                "detector": "cusum",
                "type": "downward_drift",
                "metric": self.config.metric,
                "S_minus": self.S_minus,
                "threshold": self.config.h,
                "n": self.n,
            }
            self.alarm_count += 1
            # Reset after alarm
            self.S_minus = 0.0
        
        return alarm
    
    def reset(self) -> None:
        """Reset CUSUM statistics."""
        self.S_plus = 0.0
        self.S_minus = 0.0
        self.n = 0
    
    def get_state(self) -> Dict[str, Any]:
        """Get current CUSUM state."""
        return {
            "S_plus": self.S_plus,
            "S_minus": self.S_minus,
            "n": self.n,
            "alarm_count": self.alarm_count,
        }
