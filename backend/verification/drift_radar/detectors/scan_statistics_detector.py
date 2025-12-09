"""
Scan Statistics Detector for Correlated Failure Spikes

Detects localized bursts of failures using scan statistics.

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
Status: Production Ready
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from collections import deque
import math
from .base import DriftDetector, DetectorConfig
from backend.verification.telemetry.schema import LeanVerificationTelemetry
from backend.verification.error_codes import VerifierErrorCode


@dataclass
class ScanStatisticsConfig(DetectorConfig):
    """Configuration for scan statistics detector."""
    
    window_size: int = 100  # Sliding window size
    threshold: float = 3.0  # Threshold for alarm (in standard deviations)


class ScanStatisticsDetector(DriftDetector):
    """Scan statistics detector for correlated failure spikes.
    
    Model:
        S(t, w) = (C(t, w) - E(t, w)) / sqrt(E(t, w))
    
    where:
        C(t, w) = count of failures in window [t-w, t]
        E(t, w) = expected count under baseline rate
    
    Alarm if S(t, w) > threshold
    """
    
    def __init__(self, config: ScanStatisticsConfig):
        """Initialize scan statistics detector."""
        super().__init__(config)
        self.config: ScanStatisticsConfig = config
        
        # Sliding window of failure indicators
        self.window: deque = deque(maxlen=config.window_size)
        
        # Baseline failure rate (estimated from first window)
        self.baseline_rate: Optional[float] = None
    
    def update(self, telemetry: LeanVerificationTelemetry) -> Optional[Dict[str, Any]]:
        """Update scan statistics with new observation."""
        
        if not self.config.enabled:
            return None
        
        # Extract failure indicator
        is_failure = (telemetry.outcome in [
            VerifierErrorCode.PROOF_INVALID,
            VerifierErrorCode.VERIFIER_TIMEOUT,
            VerifierErrorCode.VERIFIER_INTERNAL_ERROR,
        ])
        
        # Add to window
        self.window.append(1 if is_failure else 0)
        
        # Estimate baseline rate from first full window
        if self.baseline_rate is None and len(self.window) == self.config.window_size:
            self.baseline_rate = sum(self.window) / len(self.window)
            # Avoid zero baseline
            if self.baseline_rate == 0:
                self.baseline_rate = 0.01
        
        # Check for spike if baseline established
        if self.baseline_rate is not None and len(self.window) == self.config.window_size:
            return self._check_spike()
        
        return None
    
    def _check_spike(self) -> Optional[Dict[str, Any]]:
        """Check for failure spike in current window."""
        
        # Count failures in window
        C = sum(self.window)
        
        # Expected count under baseline
        E = self.baseline_rate * len(self.window)
        
        # Avoid division by zero
        if E == 0:
            E = 0.01
        
        # Scan statistic
        S = (C - E) / math.sqrt(E)
        
        # Check threshold
        if S > self.config.threshold:
            alarm = {
                "signal": "correlated_failure_spike",
                "detector": "scan_statistics",
                "S": S,
                "threshold": self.config.threshold,
                "C": C,
                "E": E,
                "window_size": len(self.window),
                "baseline_rate": self.baseline_rate,
            }
            self.alarm_count += 1
            return alarm
        
        return None
    
    def reset(self) -> None:
        """Reset scan statistics."""
        self.window.clear()
        self.baseline_rate = None
    
    def get_state(self) -> Dict[str, Any]:
        """Get current scan statistics state."""
        C = sum(self.window) if self.window else 0
        E = self.baseline_rate * len(self.window) if self.baseline_rate else 0
        S = (C - E) / math.sqrt(E) if E > 0 else 0.0
        
        return {
            "window_size": len(self.window),
            "C": C,
            "E": E,
            "S": S,
            "baseline_rate": self.baseline_rate,
            "alarm_count": self.alarm_count,
        }
