# Phase III: Verifier Noise Drift Radar

**Author**: Manus-C (Telemetry Architect)  
**Date**: 2025-12-06  
**Mission**: Detect verifier noise drift, tier skew, correlated failures, and Lean version drift

---

## 1. Overview

The **Verifier Noise Drift Radar** is a real-time monitoring system that detects anomalous changes in verifier behavior. It tracks six critical signals:

**Signal 1: Noise Rate Drift** — Gradual or sudden changes in timeout/failure rates  
**Signal 2: Tier Skew** — Divergence in noise rates across verifier tiers  
**Signal 3: Correlated Failure Spikes** — Sudden bursts of correlated failures  
**Signal 4: Lean Version Drift** — Changes in verifier behavior after Lean upgrades  
**Signal 5: Tactic Engine Nondeterminism** — Inconsistent outcomes for identical proofs  
**Signal 6: Resource Exhaustion Patterns** — Memory leaks or CPU throttling

The radar operates in **two modes**:

**Mode 1: Online Monitoring** — Real-time detection during U2 experiments  
**Mode 2: Offline Analysis** — Post-hoc analysis of telemetry logs

---

## 2. Signal 1: Noise Rate Drift Detection

### 2.1 Problem Statement

Noise rates (timeout, spurious failure, spurious pass) should remain stable over time. **Drift** occurs when rates change gradually or suddenly due to:
- Lean version updates
- Hardware degradation
- Workload shifts
- Configuration errors

### 2.2 Detection Algorithm: CUSUM (Cumulative Sum Control Chart)

**CUSUM** detects small shifts in mean by accumulating deviations from target:

```
Let x_t be the empirical noise rate at time t (0 or 1 for binary outcomes)
Let μ_0 be the target noise rate (configured rate)

CUSUM statistics:
S_t^+ = max(0, S_{t-1}^+ + x_t - (μ_0 + k))  (upward drift)
S_t^- = max(0, S_{t-1}^- - x_t + (μ_0 - k))  (downward drift)

where k is the slack parameter (typically k = 0.5 * σ)

Alarm if S_t^+ > h or S_t^- > h, where h is the threshold
```

**Parameters**:
- μ_0: Target noise rate (e.g., 0.05 for 5% timeout rate)
- k: Slack parameter (controls sensitivity)
- h: Threshold (controls false alarm rate)

**Calibration**:
- Set k = 0.5 * σ where σ = sqrt(μ_0 * (1 - μ_0)) for binary outcomes
- Set h to achieve desired average run length (ARL) under no drift

### 2.3 Implementation

```python
@dataclass
class CUSUMDetector:
    """CUSUM detector for noise rate drift."""
    
    mu_0: float  # Target noise rate
    k: float  # Slack parameter
    h: float  # Threshold
    
    def __post_init__(self):
        self.S_plus = 0.0  # Upward CUSUM
        self.S_minus = 0.0  # Downward CUSUM
        self.t = 0  # Time step
    
    def update(self, x: float) -> Optional[str]:
        """Update CUSUM with new observation.
        
        Args:
            x: Binary outcome (0 or 1)
        
        Returns:
            "upward_drift" if upward drift detected,
            "downward_drift" if downward drift detected,
            None otherwise
        """
        self.t += 1
        
        # Update CUSUM statistics
        self.S_plus = max(0, self.S_plus + x - (self.mu_0 + self.k))
        self.S_minus = max(0, self.S_minus - x + (self.mu_0 - self.k))
        
        # Check for alarm
        if self.S_plus > self.h:
            return "upward_drift"
        elif self.S_minus > self.h:
            return "downward_drift"
        else:
            return None
    
    def reset(self) -> None:
        """Reset CUSUM statistics."""
        self.S_plus = 0.0
        self.S_minus = 0.0
        self.t = 0
```

### 2.4 Calibration

Use Monte Carlo simulation to calibrate h for desired ARL:

```python
def calibrate_cusum_threshold(
    mu_0: float,
    k: float,
    target_arl: int = 1000,
    n_simulations: int = 10000,
) -> float:
    """Calibrate CUSUM threshold for target ARL.
    
    Args:
        mu_0: Target noise rate
        k: Slack parameter
        target_arl: Target average run length (cycles until false alarm)
        n_simulations: Number of Monte Carlo simulations
    
    Returns:
        Calibrated threshold h
    """
    
    # Binary search for h
    h_low, h_high = 0.0, 100.0
    
    while h_high - h_low > 0.01:
        h = (h_low + h_high) / 2
        
        # Simulate ARL
        run_lengths = []
        for _ in range(n_simulations):
            detector = CUSUMDetector(mu_0, k, h)
            t = 0
            while t < 10000:  # Max run length
                x = np.random.binomial(1, mu_0)
                alarm = detector.update(x)
                t += 1
                if alarm:
                    break
            run_lengths.append(t)
        
        arl = np.mean(run_lengths)
        
        if arl < target_arl:
            h_low = h
        else:
            h_high = h
    
    return (h_low + h_high) / 2
```

---

## 3. Signal 2: Tier Skew Detection

### 3.1 Problem Statement

Verifier tiers should have **monotonically decreasing** noise rates:

```
θ_timeout(FAST) ≥ θ_timeout(BALANCED) ≥ θ_timeout(SLOW)
```

**Tier skew** occurs when this invariant is violated, indicating misconfiguration or tier-specific bugs.

### 3.2 Detection Algorithm: Two-Proportion Z-Test

For each pair of tiers, test whether noise rates are significantly different:

```
H0: θ_tier1 = θ_tier2
H1: θ_tier1 > θ_tier2  (one-tailed test)

Z = (p_tier1 - p_tier2) / SE

where:
SE = sqrt(p_pooled * (1 - p_pooled) * (1/n_tier1 + 1/n_tier2))
p_pooled = (n_tier1 * p_tier1 + n_tier2 * p_tier2) / (n_tier1 + n_tier2)

Reject H0 if Z > z_α (e.g., z_0.05 = 1.645)
```

**Tier skew alarm** if:
- Z < 0 (tier1 has lower rate than tier2, violating monotonicity)
- p-value > 0.05 (rates are not significantly different, tiers not differentiated)

### 3.3 Implementation

```python
def detect_tier_skew(
    tier1_outcomes: List[bool],
    tier2_outcomes: List[bool],
    significance_level: float = 0.05,
) -> Dict[str, Any]:
    """Detect tier skew using two-proportion z-test.
    
    Args:
        tier1_outcomes: Binary outcomes for tier 1 (True = noise injected)
        tier2_outcomes: Binary outcomes for tier 2
        significance_level: Significance level for hypothesis test
    
    Returns:
        Detection result with z-statistic, p-value, and skew flag
    """
    
    n1 = len(tier1_outcomes)
    n2 = len(tier2_outcomes)
    p1 = sum(tier1_outcomes) / n1
    p2 = sum(tier2_outcomes) / n2
    
    # Pooled proportion
    p_pooled = (n1 * p1 + n2 * p2) / (n1 + n2)
    
    # Standard error
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
    
    # Z-statistic
    z = (p1 - p2) / se if se > 0 else 0
    
    # P-value (one-tailed test)
    from scipy.stats import norm
    p_value = 1 - norm.cdf(z)
    
    # Skew detection
    is_skew = (z < 0) or (p_value > significance_level)
    
    return {
        "n1": n1,
        "n2": n2,
        "p1": p1,
        "p2": p2,
        "z": z,
        "p_value": p_value,
        "is_skew": is_skew,
        "skew_type": "reversed" if z < 0 else "not_differentiated" if p_value > significance_level else "none",
    }
```

---

## 4. Signal 3: Correlated Failure Spike Detection

### 4.1 Problem Statement

Failures should be **independent** across items. **Correlated failure spikes** occur when multiple items fail simultaneously due to:
- Shared tactic bugs
- Module dependency failures
- Verifier crashes

### 4.2 Detection Algorithm: Scan Statistics

Use **scan statistics** to detect localized clusters of failures:

```
For each time window [t - w, t]:
  Count failures: C(t, w) = Σ_{s=t-w}^{t} x_s
  
  Compute expected failures: E(t, w) = w * μ_0
  
  Compute scan statistic: S(t, w) = (C(t, w) - E(t, w)) / sqrt(E(t, w))
  
  Alarm if S(t, w) > threshold
```

**Parameters**:
- w: Window size (e.g., 100 cycles)
- μ_0: Expected failure rate
- threshold: Alarm threshold (e.g., 3.0 for 3-sigma rule)

### 4.3 Implementation

```python
@dataclass
class ScanStatisticsDetector:
    """Scan statistics detector for correlated failure spikes."""
    
    window_size: int  # Window size in cycles
    mu_0: float  # Expected failure rate
    threshold: float  # Alarm threshold
    
    def __post_init__(self):
        self.outcomes = []  # Sliding window of outcomes
    
    def update(self, x: bool) -> Optional[Dict[str, Any]]:
        """Update detector with new outcome.
        
        Args:
            x: Binary outcome (True = failure)
        
        Returns:
            Alarm dict if spike detected, None otherwise
        """
        self.outcomes.append(x)
        
        # Keep only last window_size outcomes
        if len(self.outcomes) > self.window_size:
            self.outcomes.pop(0)
        
        # Compute scan statistic
        if len(self.outcomes) == self.window_size:
            C = sum(self.outcomes)
            E = self.window_size * self.mu_0
            S = (C - E) / np.sqrt(E) if E > 0 else 0
            
            if S > self.threshold:
                return {
                    "alarm": "correlated_failure_spike",
                    "window_size": self.window_size,
                    "observed_failures": C,
                    "expected_failures": E,
                    "scan_statistic": S,
                    "threshold": self.threshold,
                }
        
        return None
```

---

## 5. Signal 4: Lean Version Drift Detection

### 5.1 Problem Statement

Lean version updates can introduce **behavioral changes**:
- Performance regressions (increased timeouts)
- Bug fixes (decreased spurious failures)
- New bugs (increased spurious failures)

**Version drift** is detected by comparing noise rates before and after version update.

### 5.2 Detection Algorithm: Change-Point Detection

Use **Bayesian change-point detection** to identify the time of version update:

```
Model:
x_t ~ Bernoulli(θ_1)  for t < τ  (before change)
x_t ~ Bernoulli(θ_2)  for t ≥ τ  (after change)

Posterior:
P(τ | x_{1:T}) ∝ P(x_{1:τ} | θ_1) * P(x_{τ+1:T} | θ_2) * P(τ)

where P(τ) is uniform prior over [1, T]
```

**Detection**: Find τ that maximizes posterior probability.

### 5.3 Implementation

```python
def detect_changepoint(
    outcomes: List[bool],
) -> Dict[str, Any]:
    """Detect change-point in noise rate using Bayesian method.
    
    Args:
        outcomes: Binary outcomes (True = noise injected)
    
    Returns:
        Change-point location and posterior probability
    """
    
    T = len(outcomes)
    log_posteriors = []
    
    for tau in range(1, T):
        # Split data at tau
        before = outcomes[:tau]
        after = outcomes[tau:]
        
        # Estimate rates
        theta_1 = sum(before) / len(before) if len(before) > 0 else 0.5
        theta_2 = sum(after) / len(after) if len(after) > 0 else 0.5
        
        # Compute log-likelihood
        log_lik_before = sum(np.log(theta_1) if x else np.log(1 - theta_1) for x in before)
        log_lik_after = sum(np.log(theta_2) if x else np.log(1 - theta_2) for x in after)
        
        # Uniform prior
        log_prior = -np.log(T)
        
        # Log posterior
        log_posterior = log_lik_before + log_lik_after + log_prior
        log_posteriors.append(log_posterior)
    
    # Find MAP estimate
    tau_map = np.argmax(log_posteriors) + 1
    
    # Compute posterior probability
    posteriors = np.exp(log_posteriors - np.max(log_posteriors))
    posteriors /= np.sum(posteriors)
    
    return {
        "changepoint": tau_map,
        "posterior_prob": posteriors[tau_map - 1],
        "theta_before": sum(outcomes[:tau_map]) / tau_map,
        "theta_after": sum(outcomes[tau_map:]) / (T - tau_map),
    }
```

---

## 6. Signal 5: Tactic Engine Nondeterminism Detection

### 6.1 Problem Statement

Lean's tactic engine should be **deterministic**: identical proofs should produce identical outcomes. **Nondeterminism** occurs when:
- Tactics use unseeded randomness
- Tactics depend on system state (time, memory addresses)
- Tactics have race conditions

**Detection**: Run identical proofs multiple times and check for inconsistencies.

### 6.2 Detection Algorithm: Repeated Verification

```
For each proof:
  Run verification N times with identical configuration
  Count unique outcomes
  
  If unique outcomes > 1:
    Alarm: nondeterminism detected
```

### 6.3 Implementation

```python
def detect_nondeterminism(
    item: str,
    verifier: Callable,
    n_repetitions: int = 10,
) -> Dict[str, Any]:
    """Detect nondeterminism by repeated verification.
    
    Args:
        item: Item to verify
        verifier: Verifier function
        n_repetitions: Number of repetitions
    
    Returns:
        Detection result with outcome distribution
    """
    
    outcomes = []
    for _ in range(n_repetitions):
        outcome = verifier(item)
        outcomes.append(outcome.error_code.value)
    
    # Count unique outcomes
    unique_outcomes = set(outcomes)
    outcome_counts = {o: outcomes.count(o) for o in unique_outcomes}
    
    # Nondeterminism if more than one unique outcome
    is_nondeterministic = len(unique_outcomes) > 1
    
    return {
        "item": item,
        "n_repetitions": n_repetitions,
        "unique_outcomes": len(unique_outcomes),
        "outcome_distribution": outcome_counts,
        "is_nondeterministic": is_nondeterministic,
    }
```

---

## 7. Signal 6: Resource Exhaustion Pattern Detection

### 7.1 Problem Statement

Resource exhaustion (memory leaks, CPU throttling) causes **gradual degradation**:
- Memory usage increases over time
- Verification duration increases over time
- Timeout rate increases over time

**Detection**: Monitor resource metrics and detect upward trends.

### 7.2 Detection Algorithm: Linear Regression with Trend Test

```
Model: y_t = β_0 + β_1 * t + ε_t

where:
- y_t is resource metric at time t (memory, duration, etc.)
- β_1 is trend coefficient (slope)
- ε_t is noise

Hypothesis test:
H0: β_1 = 0  (no trend)
H1: β_1 > 0  (upward trend)

Test statistic: t = β_1 / SE(β_1)

Reject H0 if t > t_α (e.g., t_0.05 = 1.645)
```

### 7.3 Implementation

```python
def detect_resource_exhaustion(
    timestamps: List[float],
    resource_values: List[float],
    significance_level: float = 0.05,
) -> Dict[str, Any]:
    """Detect resource exhaustion using linear regression.
    
    Args:
        timestamps: Time points
        resource_values: Resource metric values (memory, duration, etc.)
        significance_level: Significance level for hypothesis test
    
    Returns:
        Detection result with trend coefficient and p-value
    """
    
    from scipy import stats
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(timestamps, resource_values)
    
    # One-tailed test for upward trend
    t_stat = slope / std_err if std_err > 0 else 0
    p_value_one_tailed = 1 - stats.t.cdf(t_stat, df=len(timestamps) - 2)
    
    # Exhaustion alarm if upward trend is significant
    is_exhaustion = (slope > 0) and (p_value_one_tailed < significance_level)
    
    return {
        "n_samples": len(timestamps),
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_value ** 2,
        "p_value": p_value_one_tailed,
        "is_exhaustion": is_exhaustion,
    }
```

---

## 8. Unified Drift Radar

### 8.1 Architecture

Combine all detectors into a unified radar:

```python
@dataclass
class VerifierNoiseDriftRadar:
    """Unified drift radar combining all detectors."""
    
    # Detectors
    timeout_drift_detector: CUSUMDetector
    spurious_fail_drift_detector: CUSUMDetector
    tier_skew_detector: Callable
    correlated_spike_detector: ScanStatisticsDetector
    changepoint_detector: Callable
    nondeterminism_detector: Callable
    resource_exhaustion_detector: Callable
    
    # State
    alarms: List[Dict[str, Any]]
    
    def __post_init__(self):
        self.alarms = []
    
    def update(self, telemetry: LeanVerificationTelemetry) -> List[Dict[str, Any]]:
        """Update radar with new telemetry.
        
        Args:
            telemetry: Verification telemetry
        
        Returns:
            List of alarms triggered
        """
        new_alarms = []
        
        # Signal 1: Noise rate drift
        if telemetry.outcome == VerifierErrorCode.VERIFIER_TIMEOUT:
            alarm = self.timeout_drift_detector.update(1.0)
            if alarm:
                new_alarms.append({
                    "signal": "timeout_drift",
                    "type": alarm,
                    "timestamp": telemetry.timestamp,
                })
        
        # Signal 3: Correlated failure spike
        if not telemetry.success:
            alarm = self.correlated_spike_detector.update(True)
            if alarm:
                new_alarms.append({
                    "signal": "correlated_failure_spike",
                    **alarm,
                    "timestamp": telemetry.timestamp,
                })
        
        # Store alarms
        self.alarms.extend(new_alarms)
        
        return new_alarms
    
    def get_status(self) -> Dict[str, Any]:
        """Get radar status.
        
        Returns:
            Status dict with alarm counts and recent alarms
        """
        return {
            "total_alarms": len(self.alarms),
            "alarms_by_signal": {
                signal: len([a for a in self.alarms if a["signal"] == signal])
                for signal in ["timeout_drift", "tier_skew", "correlated_failure_spike",
                              "version_drift", "nondeterminism", "resource_exhaustion"]
            },
            "recent_alarms": self.alarms[-10:],  # Last 10 alarms
        }
```

### 8.2 Dashboard Schema

```yaml
verifier_drift_radar_dashboard:
  title: "Verifier Noise Drift Radar"
  
  panels:
    - id: "noise_rate_drift"
      title: "Noise Rate Drift (CUSUM)"
      type: "timeseries"
      metrics:
        - cusum_timeout_upward
        - cusum_timeout_downward
        - cusum_spurious_fail_upward
        - cusum_spurious_fail_downward
      thresholds:
        - value: 5.0
          color: "yellow"
          label: "Warning"
        - value: 10.0
          color: "red"
          label: "Critical"
    
    - id: "tier_skew"
      title: "Tier Skew (Z-Test)"
      type: "heatmap"
      metrics:
        - tier_skew_fast_vs_balanced
        - tier_skew_balanced_vs_slow
      thresholds:
        - value: -1.645
          color: "red"
          label: "Reversed"
        - value: 1.645
          color: "yellow"
          label: "Not Differentiated"
    
    - id: "correlated_failures"
      title: "Correlated Failure Spikes (Scan Statistics)"
      type: "timeseries"
      metrics:
        - scan_statistic_100
        - scan_statistic_500
      thresholds:
        - value: 3.0
          color: "yellow"
          label: "Warning"
        - value: 5.0
          color: "red"
          label: "Critical"
    
    - id: "version_drift"
      title: "Lean Version Drift (Change-Point)"
      type: "annotation"
      metrics:
        - changepoint_location
        - changepoint_posterior_prob
    
    - id: "nondeterminism"
      title: "Tactic Engine Nondeterminism"
      type: "table"
      metrics:
        - nondeterministic_items
        - outcome_distribution
    
    - id: "resource_exhaustion"
      title: "Resource Exhaustion Patterns"
      type: "timeseries"
      metrics:
        - memory_trend_slope
        - duration_trend_slope
      thresholds:
        - value: 0.0
          color: "green"
          label: "Stable"
        - value: 0.01
          color: "yellow"
          label: "Gradual Increase"
        - value: 0.1
          color: "red"
          label: "Rapid Increase"
```

---

## 9. Implementation Checklist

- [ ] Implement `CUSUMDetector` for noise rate drift
- [ ] Implement `detect_tier_skew` for tier skew detection
- [ ] Implement `ScanStatisticsDetector` for correlated failure spikes
- [ ] Implement `detect_changepoint` for Lean version drift
- [ ] Implement `detect_nondeterminism` for tactic engine nondeterminism
- [ ] Implement `detect_resource_exhaustion` for resource exhaustion patterns
- [ ] Implement `VerifierNoiseDriftRadar` unified radar
- [ ] Create dashboard configuration YAML
- [ ] Integrate radar with U2 runner
- [ ] Create alerting system for critical alarms
- [ ] Implement radar telemetry export (Prometheus, InfluxDB)
- [ ] Create radar visualization dashboard (Grafana)

---

## 10. Summary

The Verifier Noise Drift Radar provides real-time monitoring and detection of six critical signals:

**1. Noise Rate Drift**: CUSUM detects gradual or sudden changes in noise rates  
**2. Tier Skew**: Two-proportion z-test detects tier misconfiguration  
**3. Correlated Failure Spikes**: Scan statistics detect localized failure clusters  
**4. Lean Version Drift**: Bayesian change-point detection identifies version updates  
**5. Tactic Engine Nondeterminism**: Repeated verification detects inconsistencies  
**6. Resource Exhaustion**: Linear regression detects upward trends in resource usage

The unified radar combines all detectors and provides a comprehensive dashboard for monitoring verifier health.

---

**Manus-C — Telemetry Architect**  
*"Every packet accounted for, every signal explained."*
