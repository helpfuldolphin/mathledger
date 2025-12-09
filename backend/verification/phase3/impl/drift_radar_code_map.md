# Task C5: Drift Radar Code Map

**Author**: Manus-C (Telemetry Architect)  
**Date**: 2025-12-06  
**Status**: Code Map Ready  
**Target**: `backend/verification/drift_radar/`

---

## 1. Overview

The **Drift Radar** is a real-time monitoring system that detects six critical signals of verifier drift and degradation. This document provides the complete module tree, detector class specifications, alarm schema, state machine, and dashboard JSON schema.

---

## 2. Module Tree

```
backend/verification/drift_radar/
├── __init__.py                      # Package initialization
├── detectors/
│   ├── __init__.py                  # Detector package
│   ├── base.py                      # Base detector interface
│   ├── cusum_detector.py            # Signal 1: Noise rate drift (CUSUM)
│   ├── tier_skew_detector.py        # Signal 2: Tier skew (z-test)
│   ├── scan_detector.py             # Signal 3: Correlated failure spikes
│   ├── changepoint_detector.py      # Signal 4: Lean version drift
│   ├── nondeterminism_detector.py   # Signal 5: Tactic engine nondeterminism
│   └── resource_detector.py         # Signal 6: Resource exhaustion
├── radar.py                         # Unified drift radar
├── alarm_schema.py                  # Alarm dataclass and schema
├── state_machine.py                 # Multi-signal alerting state machine
├── dashboard_schema.py              # Dashboard JSON schema
└── config.py                        # Radar configuration
```

---

## 3. Base Detector Interface

### 3.1 File: `backend/verification/drift_radar/detectors/base.py`

```python
"""
Base Detector Interface

All drift detectors inherit from this base class.

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
from backend.verification.telemetry_schema import LeanVerificationTelemetry


@dataclass
class DetectorConfig:
    """Base configuration for detectors."""
    
    enabled: bool = True
    name: str = "base_detector"


class DriftDetector(ABC):
    """Base class for all drift detectors."""
    
    def __init__(self, config: DetectorConfig):
        """Initialize detector.
        
        Args:
            config: Detector configuration
        """
        self.config = config
        self.alarm_count = 0
    
    @abstractmethod
    def update(self, telemetry: LeanVerificationTelemetry) -> Optional[Dict[str, Any]]:
        """Update detector with new telemetry.
        
        Args:
            telemetry: Lean verification telemetry
        
        Returns:
            Alarm dict if drift detected, None otherwise
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset detector state."""
        pass
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get current detector state for logging.
        
        Returns:
            Dict with detector state
        """
        pass
```

---

## 4. Signal 1: Noise Rate Drift (CUSUM)

### 4.1 File: `backend/verification/drift_radar/detectors/cusum_detector.py`

```python
"""
CUSUM Detector for Noise Rate Drift

Detects small shifts in noise rate using cumulative sum control chart.

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from .base import DriftDetector, DetectorConfig
from backend.verification.telemetry_schema import LeanVerificationTelemetry
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
```

---

## 5. Signal 2: Tier Skew (Z-Test)

### 5.1 File: `backend/verification/drift_radar/detectors/tier_skew_detector.py`

```python
"""
Tier Skew Detector

Detects violation of tier monotonicity invariant using two-proportion z-test.

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from collections import defaultdict
import numpy as np
from scipy.stats import norm
from .base import DriftDetector, DetectorConfig
from backend.verification.telemetry_schema import LeanVerificationTelemetry
from backend.verification.error_codes import VerifierErrorCode, VerifierTier


@dataclass
class TierSkewConfig(DetectorConfig):
    """Configuration for tier skew detector."""
    
    window_size: int = 100  # Window size for rate estimation
    significance_level: float = 0.05  # Significance level for z-test


class TierSkewDetector(DriftDetector):
    """Tier skew detector using two-proportion z-test.
    
    Tests invariant: θ_FAST ≥ θ_BALANCED ≥ θ_SLOW
    """
    
    def __init__(self, config: TierSkewConfig):
        """Initialize tier skew detector."""
        super().__init__(config)
        self.config: TierSkewConfig = config
        
        # Counts per tier
        self.tier_counts = defaultdict(lambda: {"total": 0, "timeout": 0})
        
        # Tier order
        self.tier_order = [
            VerifierTier.FAST_NOISY,
            VerifierTier.BALANCED,
            VerifierTier.SLOW_PRECISE,
        ]
    
    def update(self, telemetry: LeanVerificationTelemetry) -> Optional[Dict[str, Any]]:
        """Update tier counts and check for skew."""
        
        if not self.config.enabled:
            return None
        
        # Update counts
        tier = telemetry.tier
        self.tier_counts[tier]["total"] += 1
        if telemetry.outcome == VerifierErrorCode.VERIFIER_TIMEOUT:
            self.tier_counts[tier]["timeout"] += 1
        
        # Check for skew every window_size samples
        total_samples = sum(counts["total"] for counts in self.tier_counts.values())
        if total_samples % self.config.window_size != 0:
            return None
        
        # Test tier monotonicity
        for i in range(len(self.tier_order) - 1):
            tier1 = self.tier_order[i]
            tier2 = self.tier_order[i + 1]
            
            counts1 = self.tier_counts[tier1]
            counts2 = self.tier_counts[tier2]
            
            if counts1["total"] < 10 or counts2["total"] < 10:
                continue  # Not enough samples
            
            # Compute rates
            p1 = counts1["timeout"] / counts1["total"]
            p2 = counts2["timeout"] / counts2["total"]
            
            # Two-proportion z-test
            n1 = counts1["total"]
            n2 = counts2["total"]
            
            p_pooled = (counts1["timeout"] + counts2["timeout"]) / (n1 + n2)
            se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
            
            if se == 0:
                continue
            
            z = (p1 - p2) / se
            p_value = 1 - norm.cdf(z)  # One-tailed test
            
            # Check for alarm
            if z < 0:  # Reversed monotonicity
                alarm = {
                    "signal": "tier_skew",
                    "detector": "tier_skew",
                    "type": "reversed_monotonicity",
                    "tier1": tier1.value,
                    "tier2": tier2.value,
                    "rate1": p1,
                    "rate2": p2,
                    "z_statistic": z,
                    "p_value": p_value,
                }
                self.alarm_count += 1
                return alarm
            
            elif p_value > self.config.significance_level:  # Not differentiated
                alarm = {
                    "signal": "tier_skew",
                    "detector": "tier_skew",
                    "type": "not_differentiated",
                    "tier1": tier1.value,
                    "tier2": tier2.value,
                    "rate1": p1,
                    "rate2": p2,
                    "z_statistic": z,
                    "p_value": p_value,
                }
                self.alarm_count += 1
                return alarm
        
        return None
    
    def reset(self) -> None:
        """Reset tier counts."""
        self.tier_counts.clear()
    
    def get_state(self) -> Dict[str, Any]:
        """Get current tier counts."""
        return {
            "tier_counts": dict(self.tier_counts),
            "alarm_count": self.alarm_count,
        }
```

---

## 6. Signal 3: Correlated Failure Spikes (Scan Statistics)

### 6.1 File: `backend/verification/drift_radar/detectors/scan_detector.py`

```python
"""
Scan Statistics Detector for Correlated Failure Spikes

Detects localized clusters of failures using scan statistics.

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from collections import deque
import numpy as np
from .base import DriftDetector, DetectorConfig
from backend.verification.telemetry_schema import LeanVerificationTelemetry


@dataclass
class ScanConfig(DetectorConfig):
    """Configuration for scan statistics detector."""
    
    window_size: int = 50  # Window size for scan
    threshold: float = 3.0  # Threshold for alarm (3-sigma rule)
    mu_0: float = 0.1       # Expected failure rate


class ScanStatisticsDetector(DriftDetector):
    """Scan statistics detector for correlated failure spikes.
    
    Model:
        S(t, w) = (C(t, w) - E(t, w)) / sqrt(E(t, w))
        where C(t, w) = observed failures in window
              E(t, w) = expected failures = w * μ_0
    
    Alarm if S(t, w) > threshold
    """
    
    def __init__(self, config: ScanConfig):
        """Initialize scan statistics detector."""
        super().__init__(config)
        self.config: ScanConfig = config
        
        # Sliding window of outcomes
        self.window = deque(maxlen=config.window_size)
    
    def update(self, telemetry: LeanVerificationTelemetry) -> Optional[Dict[str, Any]]:
        """Update window and check for spike."""
        
        if not self.config.enabled:
            return None
        
        # Add outcome to window (1 = failure, 0 = success)
        failure = 0 if telemetry.success else 1
        self.window.append(failure)
        
        # Wait until window is full
        if len(self.window) < self.config.window_size:
            return None
        
        # Compute scan statistic
        C = sum(self.window)  # Observed failures
        E = self.config.window_size * self.config.mu_0  # Expected failures
        
        if E == 0:
            return None
        
        S = (C - E) / np.sqrt(E)
        
        # Check for alarm
        if S > self.config.threshold:
            alarm = {
                "signal": "correlated_failure_spike",
                "detector": "scan_statistics",
                "type": "failure_spike",
                "observed_failures": C,
                "expected_failures": E,
                "scan_statistic": S,
                "threshold": self.config.threshold,
                "window_size": self.config.window_size,
            }
            self.alarm_count += 1
            # Clear window after alarm
            self.window.clear()
            return alarm
        
        return None
    
    def reset(self) -> None:
        """Reset window."""
        self.window.clear()
    
    def get_state(self) -> Dict[str, Any]:
        """Get current window state."""
        return {
            "window_size": len(self.window),
            "current_failures": sum(self.window),
            "alarm_count": self.alarm_count,
        }
```

---

## 7. Signal 4: Lean Version Drift (Bayesian Change-Point)

### 7.1 File: `backend/verification/drift_radar/detectors/changepoint_detector.py`

```python
"""
Bayesian Change-Point Detector for Lean Version Drift

Detects change-points in noise rate using Bayesian inference.

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import numpy as np
from .base import DriftDetector, DetectorConfig
from backend.verification.telemetry_schema import LeanVerificationTelemetry
from backend.verification.error_codes import VerifierErrorCode


@dataclass
class ChangePointConfig(DetectorConfig):
    """Configuration for change-point detector."""
    
    min_segment_length: int = 20  # Minimum segment length
    posterior_threshold: float = 0.9  # Posterior probability threshold


class ChangePointDetector(DriftDetector):
    """Bayesian change-point detector for Lean version drift.
    
    Model:
        x_t ~ Bernoulli(θ_1) for t < τ
        x_t ~ Bernoulli(θ_2) for t ≥ τ
    
    Find τ that maximizes P(τ | x_{1:T})
    """
    
    def __init__(self, config: ChangePointConfig):
        """Initialize change-point detector."""
        super().__init__(config)
        self.config: ChangePointConfig = config
        
        # Observation history
        self.observations: List[int] = []
    
    def update(self, telemetry: LeanVerificationTelemetry) -> Optional[Dict[str, Any]]:
        """Update observations and detect change-point."""
        
        if not self.config.enabled:
            return None
        
        # Add observation (1 = timeout, 0 = no timeout)
        x = 1 if telemetry.outcome == VerifierErrorCode.VERIFIER_TIMEOUT else 0
        self.observations.append(x)
        
        # Need at least 2 * min_segment_length observations
        if len(self.observations) < 2 * self.config.min_segment_length:
            return None
        
        # Find MAP change-point
        tau_map, posterior_prob, theta_1, theta_2 = self._find_changepoint()
        
        # Check for alarm
        if posterior_prob > self.config.posterior_threshold and abs(theta_2 - theta_1) > 0.1:
            alarm = {
                "signal": "lean_version_drift",
                "detector": "changepoint",
                "type": "changepoint_detected",
                "changepoint_index": tau_map,
                "posterior_probability": posterior_prob,
                "theta_before": theta_1,
                "theta_after": theta_2,
                "total_observations": len(self.observations),
            }
            self.alarm_count += 1
            # Clear observations after alarm
            self.observations.clear()
            return alarm
        
        return None
    
    def _find_changepoint(self) -> tuple:
        """Find MAP change-point using Bayesian inference."""
        
        T = len(self.observations)
        min_len = self.config.min_segment_length
        
        best_tau = min_len
        best_log_posterior = -np.inf
        best_theta_1 = 0.0
        best_theta_2 = 0.0
        
        # Iterate over possible change-points
        for tau in range(min_len, T - min_len):
            # Split observations
            obs_before = self.observations[:tau]
            obs_after = self.observations[tau:]
            
            # MLE for each segment
            theta_1 = sum(obs_before) / len(obs_before) if obs_before else 0.0
            theta_2 = sum(obs_after) / len(obs_after) if obs_after else 0.0
            
            # Compute log-likelihood
            log_lik_1 = sum(
                x * np.log(theta_1 + 1e-10) + (1 - x) * np.log(1 - theta_1 + 1e-10)
                for x in obs_before
            )
            log_lik_2 = sum(
                x * np.log(theta_2 + 1e-10) + (1 - x) * np.log(1 - theta_2 + 1e-10)
                for x in obs_after
            )
            
            # Uniform prior on tau
            log_prior = -np.log(T - 2 * min_len)
            
            # Log-posterior
            log_posterior = log_lik_1 + log_lik_2 + log_prior
            
            if log_posterior > best_log_posterior:
                best_log_posterior = log_posterior
                best_tau = tau
                best_theta_1 = theta_1
                best_theta_2 = theta_2
        
        # Compute posterior probability (approximate)
        posterior_prob = 1.0 / (1.0 + np.exp(-best_log_posterior))
        
        return best_tau, posterior_prob, best_theta_1, best_theta_2
    
    def reset(self) -> None:
        """Reset observations."""
        self.observations.clear()
    
    def get_state(self) -> Dict[str, Any]:
        """Get current observation count."""
        return {
            "observation_count": len(self.observations),
            "alarm_count": self.alarm_count,
        }
```

---

## 8. Signal 5: Tactic Engine Nondeterminism

### 8.1 File: `backend/verification/drift_radar/detectors/nondeterminism_detector.py`

```python
"""
Nondeterminism Detector for Tactic Engine

Detects nondeterminism by repeated verification with identical configuration.

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Tuple
from collections import defaultdict
from .base import DriftDetector, DetectorConfig
from backend.verification.telemetry_schema import LeanVerificationTelemetry


@dataclass
class NondeterminismConfig(DetectorConfig):
    """Configuration for nondeterminism detector."""
    
    repeat_count: int = 10  # Number of repeated verifications
    sample_rate: float = 0.01  # Fraction of items to test


class NondeterminismDetector(DriftDetector):
    """Nondeterminism detector using repeated verification.
    
    Runs verification N times with identical configuration and checks for
    unique outcomes.
    """
    
    def __init__(
        self,
        config: NondeterminismConfig,
        execute_fn: Optional[Callable[[str, int], Tuple[bool, Any]]] = None,
    ):
        """Initialize nondeterminism detector.
        
        Args:
            config: Detector configuration
            execute_fn: Verification execution function
        """
        super().__init__(config)
        self.config: NondeterminismConfig = config
        self.execute_fn = execute_fn
        
        # Item outcomes
        self.item_outcomes = defaultdict(list)
    
    def update(self, telemetry: LeanVerificationTelemetry) -> Optional[Dict[str, Any]]:
        """Update with telemetry (nondeterminism detection is manual)."""
        
        # This detector doesn't use telemetry updates
        # Instead, it's triggered manually via detect_nondeterminism()
        return None
    
    def detect_nondeterminism(
        self,
        item: str,
        cycle: int,
    ) -> Optional[Dict[str, Any]]:
        """Detect nondeterminism for a specific item.
        
        Args:
            item: Item to test
            cycle: Cycle number
        
        Returns:
            Alarm dict if nondeterminism detected, None otherwise
        """
        
        if not self.config.enabled or self.execute_fn is None:
            return None
        
        # Run verification N times
        outcomes = []
        for i in range(self.config.repeat_count):
            success, result = self.execute_fn(item, cycle)
            outcome = result.get("outcome", "UNKNOWN")
            outcomes.append(outcome)
        
        # Count unique outcomes
        unique_outcomes = set(outcomes)
        
        # Check for alarm
        if len(unique_outcomes) > 1:
            alarm = {
                "signal": "tactic_engine_nondeterminism",
                "detector": "nondeterminism",
                "type": "nondeterministic_outcome",
                "item": item,
                "repeat_count": self.config.repeat_count,
                "unique_outcomes": len(unique_outcomes),
                "outcome_distribution": {
                    outcome: outcomes.count(outcome) / len(outcomes)
                    for outcome in unique_outcomes
                },
            }
            self.alarm_count += 1
            return alarm
        
        return None
    
    def reset(self) -> None:
        """Reset item outcomes."""
        self.item_outcomes.clear()
    
    def get_state(self) -> Dict[str, Any]:
        """Get current item outcome counts."""
        return {
            "tested_items": len(self.item_outcomes),
            "alarm_count": self.alarm_count,
        }
```

---

## 9. Signal 6: Resource Exhaustion

### 9.1 File: `backend/verification/drift_radar/detectors/resource_detector.py`

```python
"""
Resource Exhaustion Detector

Detects upward trends in resource usage using linear regression.

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import numpy as np
from scipy.stats import linregress
from .base import DriftDetector, DetectorConfig
from backend.verification.telemetry_schema import LeanVerificationTelemetry


@dataclass
class ResourceExhaustionConfig(DetectorConfig):
    """Configuration for resource exhaustion detector."""
    
    window_size: int = 100  # Window size for trend analysis
    significance_level: float = 0.05  # Significance level for trend test
    metric: str = "memory_peak_mb"  # Metric to monitor


class ResourceExhaustionDetector(DriftDetector):
    """Resource exhaustion detector using linear regression.
    
    Model: y_t = β_0 + β_1 * t + ε_t
    Test: H0: β_1 = 0 vs H1: β_1 > 0
    """
    
    def __init__(self, config: ResourceExhaustionConfig):
        """Initialize resource exhaustion detector."""
        super().__init__(config)
        self.config: ResourceExhaustionConfig = config
        
        # Resource usage history
        self.resource_history: List[float] = []
        self.time_indices: List[int] = []
    
    def update(self, telemetry: LeanVerificationTelemetry) -> Optional[Dict[str, Any]]:
        """Update resource history and check for trend."""
        
        if not self.config.enabled:
            return None
        
        # Extract resource metric
        if self.config.metric == "memory_peak_mb":
            value = telemetry.memory_peak_mb
        elif self.config.metric == "duration_ms":
            value = telemetry.duration_ms
        elif self.config.metric == "cpu_time_ms":
            value = telemetry.cpu_time_ms
        else:
            return None
        
        if value is None:
            return None
        
        # Add to history
        self.resource_history.append(value)
        self.time_indices.append(len(self.resource_history))
        
        # Keep only last window_size samples
        if len(self.resource_history) > self.config.window_size:
            self.resource_history.pop(0)
            self.time_indices.pop(0)
        
        # Wait until window is full
        if len(self.resource_history) < self.config.window_size:
            return None
        
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = linregress(
            self.time_indices,
            self.resource_history,
        )
        
        # One-tailed test for upward trend
        p_value_one_tailed = p_value / 2 if slope > 0 else 1.0
        
        # Check for alarm
        if slope > 0 and p_value_one_tailed < self.config.significance_level:
            alarm = {
                "signal": "resource_exhaustion",
                "detector": "resource_exhaustion",
                "type": "upward_trend",
                "metric": self.config.metric,
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_value ** 2,
                "p_value": p_value_one_tailed,
                "window_size": len(self.resource_history),
            }
            self.alarm_count += 1
            # Clear history after alarm
            self.resource_history.clear()
            self.time_indices.clear()
            return alarm
        
        return None
    
    def reset(self) -> None:
        """Reset resource history."""
        self.resource_history.clear()
        self.time_indices.clear()
    
    def get_state(self) -> Dict[str, Any]:
        """Get current resource history."""
        return {
            "history_length": len(self.resource_history),
            "current_mean": np.mean(self.resource_history) if self.resource_history else 0.0,
            "alarm_count": self.alarm_count,
        }
```

---

## 10. Unified Drift Radar

### 10.1 File: `backend/verification/drift_radar/radar.py`

```python
"""
Unified Drift Radar

Combines all six drift detectors into a unified monitoring system.

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from backend.verification.telemetry_schema import LeanVerificationTelemetry
from .detectors.cusum_detector import CUSUMDetector, CUSUMConfig
from .detectors.tier_skew_detector import TierSkewDetector, TierSkewConfig
from .detectors.scan_detector import ScanStatisticsDetector, ScanConfig
from .detectors.changepoint_detector import ChangePointDetector, ChangePointConfig
from .detectors.nondeterminism_detector import NondeterminismDetector, NondeterminismConfig
from .detectors.resource_detector import ResourceExhaustionDetector, ResourceExhaustionConfig
from .alarm_schema import Alarm, AlarmSeverity
from .state_machine import AlertingStateMachine


@dataclass
class RadarConfig:
    """Configuration for unified drift radar."""
    
    cusum_timeout: CUSUMConfig = field(default_factory=lambda: CUSUMConfig(metric="timeout_rate"))
    cusum_spurious_fail: CUSUMConfig = field(default_factory=lambda: CUSUMConfig(metric="spurious_fail_rate"))
    tier_skew: TierSkewConfig = field(default_factory=TierSkewConfig)
    scan: ScanConfig = field(default_factory=ScanConfig)
    changepoint: ChangePointConfig = field(default_factory=ChangePointConfig)
    nondeterminism: NondeterminismConfig = field(default_factory=NondeterminismConfig)
    resource_memory: ResourceExhaustionConfig = field(default_factory=lambda: ResourceExhaustionConfig(metric="memory_peak_mb"))
    resource_duration: ResourceExhaustionConfig = field(default_factory=lambda: ResourceExhaustionConfig(metric="duration_ms"))


class VerifierNoiseDriftRadar:
    """Unified drift radar combining all six signals.
    
    Usage:
        config = RadarConfig(...)
        radar = VerifierNoiseDriftRadar(config)
        
        # For each telemetry record
        alarms = radar.update(telemetry)
        
        # Get status report
        report = radar.get_status_report()
    """
    
    def __init__(self, config: RadarConfig):
        """Initialize drift radar.
        
        Args:
            config: Radar configuration
        """
        self.config = config
        
        # Initialize detectors
        self.cusum_timeout = CUSUMDetector(config.cusum_timeout)
        self.cusum_spurious_fail = CUSUMDetector(config.cusum_spurious_fail)
        self.tier_skew = TierSkewDetector(config.tier_skew)
        self.scan = ScanStatisticsDetector(config.scan)
        self.changepoint = ChangePointDetector(config.changepoint)
        self.nondeterminism = NondeterminismDetector(config.nondeterminism)
        self.resource_memory = ResourceExhaustionDetector(config.resource_memory)
        self.resource_duration = ResourceExhaustionDetector(config.resource_duration)
        
        # Alarm history
        self.alarms: List[Alarm] = []
        
        # Alerting state machine
        self.state_machine = AlertingStateMachine()
    
    def update(self, telemetry: LeanVerificationTelemetry) -> List[Alarm]:
        """Update all detectors with new telemetry.
        
        Args:
            telemetry: Lean verification telemetry
        
        Returns:
            List of alarms triggered by this update
        """
        
        new_alarms = []
        
        # Update all detectors
        detectors = [
            self.cusum_timeout,
            self.cusum_spurious_fail,
            self.tier_skew,
            self.scan,
            self.changepoint,
            self.resource_memory,
            self.resource_duration,
        ]
        
        for detector in detectors:
            alarm_dict = detector.update(telemetry)
            if alarm_dict:
                # Convert to Alarm object
                alarm = Alarm(
                    signal=alarm_dict["signal"],
                    detector=alarm_dict["detector"],
                    severity=self._determine_severity(alarm_dict),
                    timestamp=telemetry.timestamp,
                    details=alarm_dict,
                )
                new_alarms.append(alarm)
                self.alarms.append(alarm)
                
                # Update state machine
                self.state_machine.process_alarm(alarm)
        
        return new_alarms
    
    def _determine_severity(self, alarm_dict: Dict[str, Any]) -> AlarmSeverity:
        """Determine alarm severity based on alarm details."""
        
        signal = alarm_dict["signal"]
        
        # Critical alarms
        if signal in ["tier_skew", "tactic_engine_nondeterminism"]:
            return AlarmSeverity.CRITICAL
        
        # High alarms
        if signal in ["lean_version_drift", "resource_exhaustion"]:
            return AlarmSeverity.HIGH
        
        # Medium alarms
        if signal in ["correlated_failure_spike"]:
            return AlarmSeverity.MEDIUM
        
        # Low alarms
        return AlarmSeverity.LOW
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report.
        
        Returns:
            Dict with radar status, alarm counts, and detector states
        """
        
        return {
            "total_alarms": len(self.alarms),
            "alarms_by_signal": self._count_alarms_by_signal(),
            "alarms_by_severity": self._count_alarms_by_severity(),
            "recent_alarms": [alarm.to_dict() for alarm in self.alarms[-10:]],
            "detector_states": {
                "cusum_timeout": self.cusum_timeout.get_state(),
                "cusum_spurious_fail": self.cusum_spurious_fail.get_state(),
                "tier_skew": self.tier_skew.get_state(),
                "scan": self.scan.get_state(),
                "changepoint": self.changepoint.get_state(),
                "nondeterminism": self.nondeterminism.get_state(),
                "resource_memory": self.resource_memory.get_state(),
                "resource_duration": self.resource_duration.get_state(),
            },
            "state_machine": self.state_machine.get_state(),
        }
    
    def _count_alarms_by_signal(self) -> Dict[str, int]:
        """Count alarms by signal type."""
        from collections import Counter
        return dict(Counter(alarm.signal for alarm in self.alarms))
    
    def _count_alarms_by_severity(self) -> Dict[str, int]:
        """Count alarms by severity."""
        from collections import Counter
        return dict(Counter(alarm.severity.value for alarm in self.alarms))
    
    def reset(self) -> None:
        """Reset all detectors and alarm history."""
        self.cusum_timeout.reset()
        self.cusum_spurious_fail.reset()
        self.tier_skew.reset()
        self.scan.reset()
        self.changepoint.reset()
        self.nondeterminism.reset()
        self.resource_memory.reset()
        self.resource_duration.reset()
        self.alarms.clear()
        self.state_machine.reset()
```

---

## 11. Alarm Schema

### 11.1 File: `backend/verification/drift_radar/alarm_schema.py`

```python
"""
Alarm Schema

Dataclass and schema for drift radar alarms.

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
"""

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, Any


class AlarmSeverity(Enum):
    """Alarm severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Alarm:
    """Drift radar alarm."""
    
    signal: str  # Signal type (e.g., "noise_rate_drift")
    detector: str  # Detector name (e.g., "cusum")
    severity: AlarmSeverity
    timestamp: float
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "signal": self.signal,
            "detector": self.detector,
            "severity": self.severity.value,
            "timestamp": self.timestamp,
            "details": self.details,
        }
```

---

## 12. State Machine for Multi-Signal Alerting

### 12.1 File: `backend/verification/drift_radar/state_machine.py`

```python
"""
Alerting State Machine

State machine for multi-signal alerting with escalation.

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List
from .alarm_schema import Alarm, AlarmSeverity


class AlertState(Enum):
    """Alert states."""
    NORMAL = "normal"
    WARNING = "warning"
    ALERT = "alert"
    CRITICAL = "critical"


@dataclass
class AlertingStateMachine:
    """State machine for multi-signal alerting.
    
    State transitions:
        NORMAL → WARNING: 1+ LOW/MEDIUM alarms
        WARNING → ALERT: 1+ HIGH alarms or 3+ MEDIUM alarms
        ALERT → CRITICAL: 1+ CRITICAL alarms or 5+ HIGH alarms
        Any state → NORMAL: Manual reset
    """
    
    def __init__(self):
        """Initialize state machine."""
        self.state = AlertState.NORMAL
        self.alarm_counts = {
            AlarmSeverity.LOW: 0,
            AlarmSeverity.MEDIUM: 0,
            AlarmSeverity.HIGH: 0,
            AlarmSeverity.CRITICAL: 0,
        }
    
    def process_alarm(self, alarm: Alarm) -> AlertState:
        """Process alarm and update state.
        
        Args:
            alarm: Alarm to process
        
        Returns:
            New alert state
        """
        
        # Update alarm counts
        self.alarm_counts[alarm.severity] += 1
        
        # State transitions
        if self.alarm_counts[AlarmSeverity.CRITICAL] >= 1:
            self.state = AlertState.CRITICAL
        elif self.alarm_counts[AlarmSeverity.HIGH] >= 5:
            self.state = AlertState.CRITICAL
        elif self.alarm_counts[AlarmSeverity.HIGH] >= 1:
            self.state = AlertState.ALERT
        elif self.alarm_counts[AlarmSeverity.MEDIUM] >= 3:
            self.state = AlertState.ALERT
        elif self.alarm_counts[AlarmSeverity.MEDIUM] >= 1 or self.alarm_counts[AlarmSeverity.LOW] >= 1:
            if self.state == AlertState.NORMAL:
                self.state = AlertState.WARNING
        
        return self.state
    
    def reset(self) -> None:
        """Reset state machine to NORMAL."""
        self.state = AlertState.NORMAL
        for severity in self.alarm_counts:
            self.alarm_counts[severity] = 0
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state."""
        return {
            "state": self.state.value,
            "alarm_counts": {
                severity.value: count
                for severity, count in self.alarm_counts.items()
            },
        }
```

---

## 13. Dashboard JSON Schema

### 13.1 File: `backend/verification/drift_radar/dashboard_schema.py`

```python
"""
Dashboard JSON Schema

JSON schema for Grafana dashboard configuration.

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
"""

from typing import Dict, Any


def generate_dashboard_schema() -> Dict[str, Any]:
    """Generate Grafana dashboard JSON schema.
    
    Returns:
        Dashboard JSON schema
    """
    
    return {
        "dashboard": {
            "title": "Verifier Noise Drift Radar",
            "tags": ["phase3", "drift", "noise"],
            "timezone": "browser",
            "panels": [
                # Panel 1: Noise Rate Drift (CUSUM)
                {
                    "id": 1,
                    "title": "Noise Rate Drift (CUSUM)",
                    "type": "timeseries",
                    "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8},
                    "targets": [
                        {
                            "expr": "drift_radar_cusum_s_plus",
                            "legendFormat": "S+ (upward drift)",
                        },
                        {
                            "expr": "drift_radar_cusum_s_minus",
                            "legendFormat": "S- (downward drift)",
                        },
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "thresholds": {
                                "mode": "absolute",
                                "steps": [
                                    {"value": 0, "color": "green"},
                                    {"value": 3, "color": "yellow"},
                                    {"value": 5, "color": "red"},
                                ],
                            },
                        },
                    },
                },
                
                # Panel 2: Tier Skew
                {
                    "id": 2,
                    "title": "Tier Skew (Z-Statistics)",
                    "type": "heatmap",
                    "gridPos": {"x": 12, "y": 0, "w": 12, "h": 8},
                    "targets": [
                        {
                            "expr": "drift_radar_tier_skew_z_statistic",
                            "legendFormat": "{{tier1}} vs {{tier2}}",
                        },
                    ],
                },
                
                # Panel 3: Correlated Failure Spikes
                {
                    "id": 3,
                    "title": "Correlated Failure Spikes (Scan Statistics)",
                    "type": "timeseries",
                    "gridPos": {"x": 0, "y": 8, "w": 12, "h": 8},
                    "targets": [
                        {
                            "expr": "drift_radar_scan_statistic",
                            "legendFormat": "Scan Statistic",
                        },
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "thresholds": {
                                "mode": "absolute",
                                "steps": [
                                    {"value": 0, "color": "green"},
                                    {"value": 2, "color": "yellow"},
                                    {"value": 3, "color": "red"},
                                ],
                            },
                        },
                    },
                },
                
                # Panel 4: Lean Version Drift (Change-Point)
                {
                    "id": 4,
                    "title": "Lean Version Drift (Change-Point Detection)",
                    "type": "timeseries",
                    "gridPos": {"x": 12, "y": 8, "w": 12, "h": 8},
                    "targets": [
                        {
                            "expr": "drift_radar_changepoint_posterior",
                            "legendFormat": "Posterior Probability",
                        },
                    ],
                    "annotations": {
                        "list": [
                            {
                                "name": "Change-Points",
                                "datasource": "Prometheus",
                                "expr": "drift_radar_changepoint_detected",
                            },
                        ],
                    },
                },
                
                # Panel 5: Tactic Engine Nondeterminism
                {
                    "id": 5,
                    "title": "Tactic Engine Nondeterminism",
                    "type": "table",
                    "gridPos": {"x": 0, "y": 16, "w": 12, "h": 8},
                    "targets": [
                        {
                            "expr": "drift_radar_nondeterminism_items",
                            "format": "table",
                        },
                    ],
                },
                
                # Panel 6: Resource Exhaustion Patterns
                {
                    "id": 6,
                    "title": "Resource Exhaustion Patterns (Trend Slopes)",
                    "type": "timeseries",
                    "gridPos": {"x": 12, "y": 16, "w": 12, "h": 8},
                    "targets": [
                        {
                            "expr": "drift_radar_resource_memory_slope",
                            "legendFormat": "Memory Trend Slope",
                        },
                        {
                            "expr": "drift_radar_resource_duration_slope",
                            "legendFormat": "Duration Trend Slope",
                        },
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "thresholds": {
                                "mode": "absolute",
                                "steps": [
                                    {"value": 0, "color": "green"},
                                    {"value": 0.1, "color": "yellow"},
                                    {"value": 0.5, "color": "red"},
                                ],
                            },
                        },
                    },
                },
            ],
        },
    }
```

---

## 14. Deployment Checklist

- [ ] Implement all six detector classes
- [ ] Implement unified drift radar
- [ ] Implement alarm schema
- [ ] Implement alerting state machine
- [ ] Generate dashboard JSON schema
- [ ] Write unit tests for each detector
- [ ] Write integration tests for radar
- [ ] Test with real telemetry data
- [ ] Deploy dashboard to Grafana
- [ ] Configure alerting (PagerDuty, Slack)
- [ ] Document radar usage and configuration

---

**Manus-C — Telemetry Architect**  
*"Every packet accounted for, every signal explained."*

**Status**: Code Map Complete  
**Next**: Deliver All Implementation Plans
