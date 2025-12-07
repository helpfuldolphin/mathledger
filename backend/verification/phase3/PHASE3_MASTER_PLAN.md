# Phase III: Empirical Calibration & RFL Fusion — Master Technical Plan

**Author**: Manus-C (Telemetry Architect)  
**Date**: 2025-12-06  
**Status**: Design Complete, Implementation Ready  
**Mission**: Transition from synthetic noise modeling to empirical, RFL-integrated verifier imperfection analytics

---

## Executive Summary

Phase III extends the Phase-II imperfect verifier noise model with **empirical calibration**, **advanced noise regimes**, **RFL integration mathematics**, **noise replay logs**, and **drift detection radar**. This document provides a comprehensive technical plan with mathematical derivations, pseudo-code, telemetry schemas, and experiment design protocols.

**Key Deliverables**:

1. **Real-World Noise Model Calibration Pipeline** — Instrumentation, ground truth labeling, statistical fitting, cross-tier validation, and model export
2. **Advanced Noise Regimes** — Correlated failures, cluster-based degradation, heat-death scenarios, heavy-tailed timeouts, non-stationary noise, and policy-aware adaptive noise
3. **RFL Integration Mathematics** — Expected value functions, abstention handling, confidence-weighted updates, multi-tier integration, bias correction, and optimal policies
4. **Noise Replay Log Specification** — Deterministic replay semantics, differential debugging, and anomaly detection
5. **Verifier Noise Drift Radar** — Real-time detection of noise rate drift, tier skew, correlated failure spikes, Lean version drift, tactic engine nondeterminism, and resource exhaustion patterns

---

## 1. Real-World Noise Model Calibration Pipeline

### 1.1 Architecture

**Stage 1: Instrumentation** — Capture comprehensive telemetry from every Lean verification call  
**Stage 2: Ground Truth Labeling** — Classify outcomes into success, failure, timeout, and abstention  
**Stage 3: Statistical Fitting** — Fit parametric distributions to observed failure rates and durations  
**Stage 4: Cross-Tier Validation** — Validate noise models across verifier tiers  
**Stage 5: Model Export** — Export calibrated models to production configuration

### 1.2 Telemetry Schema

```python
@dataclass(frozen=True)
class LeanVerificationTelemetry:
    # Identity
    verification_id: str
    timestamp: float
    module_name: str
    context: str
    
    # Configuration
    tier: VerifierTier
    timeout_s: float
    lean_version: str
    
    # Outcome
    outcome: VerifierErrorCode
    success: bool
    duration_ms: float
    
    # Resource Usage
    cpu_time_ms: float
    memory_peak_mb: float
    memory_final_mb: float
    
    # Lean-Specific Metrics
    tactic_count: Optional[int]
    tactic_depth: Optional[int]
    proof_size_bytes: Optional[int]
    search_nodes: Optional[int]
    
    # Failure Diagnostics
    stderr: str
    returncode: int
    signal: Optional[int]
    
    # Noise Injection (if synthetic)
    noise_injected: bool
    noise_type: Optional[str]
    ground_truth: Optional[str]
    
    # Metadata
    metadata: Dict[str, Any]
```

### 1.3 Statistical Fitting

**Maximum Likelihood Estimation** for noise rates:

```
θ_timeout = n_timeout / n_total
θ_spurious_fail = n_spurious_fail / (n_genuine_success + n_spurious_fail)
θ_spurious_pass = n_spurious_pass / (n_genuine_failure + n_spurious_pass)
```

**Confidence Intervals** (Wilson score method):

```
z = 1.96  # 95% confidence
p = θ_timeout
n = n_total

center = (p + z²/(2n)) / (1 + z²/n)
margin = z * sqrt(p(1-p)/n + z²/(4n²)) / (1 + z²/n)

CI = [center - margin, center + margin]
```

**Timeout Distribution Fitting** (exponential, gamma, lognormal, Weibull):

```python
from scipy import stats

distributions = {
    "exponential": stats.expon,
    "gamma": stats.gamma,
    "lognormal": stats.lognorm,
    "weibull": stats.weibull_min,
}

for name, dist in distributions.items():
    params = dist.fit(timeout_durations)
    log_likelihood = np.sum(dist.logpdf(timeout_durations, *params))
    aic = 2 * len(params) - 2 * log_likelihood
```

### 1.4 Cross-Tier Validation

**Hypothesis Testing** (two-proportion z-test):

```
H0: θ_FAST = θ_BALANCED
H1: θ_FAST > θ_BALANCED

z = (p_FAST - p_BALANCED) / sqrt(p_pooled * (1 - p_pooled) * (1/n_FAST + 1/n_BALANCED))

Reject H0 if z > 1.645 (one-tailed test at α = 0.05)
```

### 1.5 Model Export

Export calibrated models to YAML:

```yaml
tier: fast_noisy
calibration_date: "2025-12-06T18:30:00Z"
training_samples: 10000
validation_accuracy: 0.952

noise_rates:
  timeout_rate: 0.087  # 95% CI: [0.081, 0.093]
  spurious_fail_rate: 0.042  # 95% CI: [0.037, 0.047]
  spurious_pass_rate: 0.018  # 95% CI: [0.014, 0.022]

timeout_distribution:
  type: "lognormal"
  params:
    mu: 5.2
    sigma: 0.8
  goodness_of_fit:
    ks_statistic: 0.023
    ks_pvalue: 0.412
    aic: 8234.5
```

**Reference**: `CALIBRATION_PIPELINE.md` (Section 2-6)

---

## 2. Advanced Noise Regimes

### 2.1 Regime 1: Correlated Failures

**Model**: Latent factor model with spatial correlation

```
Let z_k ~ Bernoulli(ρ) be latent failure factors
Let A_ik ∈ {0, 1} be indicator that item i uses factor k

P(fail_i | z) = 1 - ∏_{k: A_ik = 1} (1 - z_k * θ_k)
```

**Implementation**:

```python
@dataclass
class CorrelatedNoiseModel:
    rho: float  # Factor activation probability
    theta: Dict[str, float]  # Base failure rate per factor
    item_factors: Dict[str, List[str]]  # Item → factors mapping
    prng: DeterministicPRNG
    
    def should_fail(self, item: str) -> bool:
        factors = self.item_factors.get(item, [])
        prob_success = 1.0
        for factor in factors:
            if self.active_factors.get(factor, False):
                prob_success *= (1 - self.theta[factor])
        prob_fail = 1 - prob_success
        return self.prng.for_path("item", item).random() < prob_fail
```

### 2.2 Regime 2: Cluster-Based Degradation

**Model**: Hidden Markov Model with HEALTHY and DEGRADED states

```
States: S = {HEALTHY, DEGRADED}

Transition probabilities:
P(HEALTHY → DEGRADED) = α
P(DEGRADED → HEALTHY) = β

Emission probabilities:
P(fail | HEALTHY) = θ_healthy
P(fail | DEGRADED) = θ_degraded
```

**Implementation**:

```python
@dataclass
class ClusterDegradationModel:
    alpha: float  # Transition probability HEALTHY → DEGRADED
    beta: float  # Transition probability DEGRADED → HEALTHY
    theta_healthy: float
    theta_degraded: float
    prng: DeterministicPRNG
    
    def step(self) -> None:
        if self.state == "HEALTHY":
            if self.prng.for_path("transition", str(self.cycle_count)).random() < self.alpha:
                self.state = "DEGRADED"
        else:
            if self.prng.for_path("transition", str(self.cycle_count)).random() < self.beta:
                self.state = "HEALTHY"
```

### 2.3 Regime 3: Heat-Death Scenarios

**Model**: Resource depletion process

```
R(t+1) = R(t) - c(t) + r(t)

where:
- R(t) is available resource at time t
- c(t) is resource consumption (random)
- r(t) is resource recovery (random)

Failure condition: fail(t) = 1 if R(t) < R_min
```

### 2.4 Regime 4: High-Tactic-Depth Tails

**Model**: Mixture distribution with heavy tails

```
T ~ (1 - π) * Exp(λ_fast) + π * Pareto(α, x_min)

Pareto: f(t; α, x_min) = (α * x_min^α) / t^(α+1) for t ≥ x_min
```

### 2.5 Regime 5: Non-Stationary Noise

**Model**: Time-varying parameters

```
θ_timeout(t) = θ_0 + δ * t + ε(t)

where ε(t) ~ N(0, σ²)
```

### 2.6 Regime 6: Policy-Aware Adaptive Noise

**Model**: Adaptive noise based on policy confidence

```
θ_adaptive(item, π) = θ_base * (1 + γ * confidence(item, π))

where confidence(item, π) = |π(item) - 0.5| * 2
```

### 2.7 Unified Noise Model

Combine all regimes:

```python
@dataclass
class UnifiedNoiseModel:
    base_timeout_rate: float
    base_spurious_fail_rate: float
    base_spurious_pass_rate: float
    
    correlated_model: Optional[CorrelatedNoiseModel] = None
    degradation_model: Optional[ClusterDegradationModel] = None
    heat_death_model: Optional[HeatDeathModel] = None
    heavy_tail_model: Optional[HeavyTailTimeoutModel] = None
    nonstationary_model: Optional[NonStationaryNoiseModel] = None
    adaptive_model: Optional[AdaptiveNoiseModel] = None
    
    prng: DeterministicPRNG
```

**Reference**: `ADVANCED_NOISE_REGIMES.md` (Section 2-8)

---

## 3. RFL Integration Mathematics

### 3.1 Verifier Outcome Model

```
P(o = VERIFIED | y = VALID) = 1 - θ_sf - θ_to
P(o = FAILED | y = VALID) = θ_sf
P(o = TIMEOUT | y = VALID) = θ_to

P(o = VERIFIED | y = INVALID) = θ_sp
P(o = FAILED | y = INVALID) = 1 - θ_sp - θ_to
P(o = TIMEOUT | y = INVALID) = θ_to
```

### 3.2 Expected Value Function

```
V_expected(e, o) = 2 * P(y = VALID | o) - 1

Case 1 (o = VERIFIED):
V_expected(e, VERIFIED) = 2 * [(1 - θ_sf - θ_to) / [(1 - θ_sf - θ_to) + θ_sp]] - 1

Case 2 (o = FAILED):
V_expected(e, FAILED) = 2 * [θ_sf / [θ_sf + (1 - θ_sp - θ_to)]] - 1

Case 3 (o = TIMEOUT):
V_expected(e, TIMEOUT) = 0
```

### 3.3 Noise-Robust RFL Update

```
Δπ(e) = η * V_expected(e, o) * ∇_π log π(e)

π_{t+1}(e) = π_t(e) + Δπ(e)
```

### 3.4 Bias Correction

```
V_corrected(e, o) = V_expected(e, o) / (1 - 2 * θ_sp)
```

### 3.5 Abstention Handling

```
if o == TIMEOUT:
    Δπ(e) = 0  (no update)

η_effective = η / (1 - α)  where α = P(o = TIMEOUT)
```

### 3.6 Multi-Tier Integration

```
V_aggregated(e, outcomes) = Σ_tier w_tier * V_expected(e, o_tier, tier)

where w_tier is confidence weight for tier
```

### 3.7 Optimal Policy

```
π(e) = exp(Q(e) / τ) / Σ_e' exp(Q(e') / τ)

Q(e) ← Q(e) + η * [V_corrected(e, o) - Q(e)]
```

### 3.8 Implementation

```python
def compute_expected_value(
    outcome: VerifierErrorCode,
    theta_spurious_fail: float,
    theta_spurious_pass: float,
    theta_timeout: float,
) -> float:
    if outcome == VerifierErrorCode.VERIFIED:
        p_valid = (1 - theta_spurious_fail - theta_timeout) / \
                  ((1 - theta_spurious_fail - theta_timeout) + theta_spurious_pass)
        return 2 * p_valid - 1
    elif outcome == VerifierErrorCode.PROOF_INVALID:
        p_valid = theta_spurious_fail / \
                  (theta_spurious_fail + (1 - theta_spurious_pass - theta_timeout))
        return 2 * p_valid - 1
    elif outcome == VerifierErrorCode.VERIFIER_TIMEOUT:
        return 0.0
    else:
        return 0.0
```

**Reference**: `RFL_INTEGRATION_MATHEMATICS.md` (Section 2-10)

---

## 4. Noise Replay Log Specification

### 4.1 Log Entry Schema

```json
{
  "version": "1.0",
  "entry_type": "noise_decision",
  "timestamp": 1733518800.123,
  "verification_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "cycle_id": 42,
  "item": "Mathlib.Algebra.Ring.Basic.theorem_1",
  "context": "cycle_42_item_Mathlib.Algebra.Ring.Basic.theorem_1",
  "tier": "balanced",
  "seed": {
    "master_seed": 12345,
    "context_hash": "8f3a9b2c1d4e5f67",
    "noise_type_path": "timeout"
  },
  "noise_decision": {
    "noise_type": "timeout",
    "injected": true,
    "probability": 0.05,
    "sampled_value": 0.0234,
    "decision": "inject"
  },
  "noise_parameters": {
    "timeout_duration_ms": 1234.5,
    "distribution": "lognormal",
    "distribution_params": {
      "mu": 5.2,
      "sigma": 0.8
    }
  },
  "verifier_outcome": {
    "outcome": "VERIFIER_TIMEOUT",
    "success": false,
    "duration_ms": 1234.5,
    "noise_injected": true,
    "ground_truth": "VERIFIED"
  },
  "metadata": {
    "lean_version": "4.3.0",
    "experiment_id": "uplift_phase2_run_001",
    "slice_name": "arithmetic_simple"
  }
}
```

### 4.2 Deterministic Replay

```python
def replay_noise_log(
    log_entries: List[Dict[str, Any]],
) -> List[VerifierOutcome]:
    outcomes = []
    for entry in log_entries:
        master_seed = entry["seed"]["master_seed"]
        context = entry["context"]
        noise_type = entry["noise_decision"]["noise_type"]
        
        prng = DeterministicPRNG(int_to_hex_seed(master_seed))
        prng_noise = prng.for_path(noise_type, context)
        
        sampled_value = prng_noise.random()
        probability = entry["noise_decision"]["probability"]
        injected = sampled_value < probability
        
        assert sampled_value == entry["noise_decision"]["sampled_value"]
        assert injected == entry["noise_decision"]["injected"]
        
        # Reconstruct outcome...
        outcomes.append(outcome)
    
    return outcomes
```

### 4.3 Counterfactual Analysis

```python
def counterfactual_analysis(
    log_entries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    outcomes_with_noise = replay_noise_log(log_entries)
    outcomes_without_noise = [use_ground_truth(entry) for entry in log_entries]
    
    differences = [
        {"index": i, "item": entry["item"], "with_noise": with_noise.error_code, "without_noise": without_noise.error_code}
        for i, (with_noise, without_noise, entry) in enumerate(zip(outcomes_with_noise, outcomes_without_noise, log_entries))
        if with_noise.error_code != without_noise.error_code
    ]
    
    return {
        "total_outcomes": len(outcomes_with_noise),
        "differences": len(differences),
        "difference_rate": len(differences) / len(outcomes_with_noise),
        "difference_details": differences,
    }
```

### 4.4 Anomaly Detection

**Chi-Square Test for Noise Rate**:

```python
def detect_noise_rate_anomaly(
    log_entries: List[Dict[str, Any]],
    configured_rate: float,
    significance_level: float = 0.05,
) -> Dict[str, Any]:
    n_total = len(log_entries)
    n_injected = sum(1 for entry in log_entries if entry["noise_decision"]["injected"])
    
    expected_injected = n_total * configured_rate
    expected_not_injected = n_total * (1 - configured_rate)
    
    chi_square = (
        (n_injected - expected_injected) ** 2 / expected_injected +
        (n_total - n_injected - expected_not_injected) ** 2 / expected_not_injected
    )
    
    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(chi_square, df=1)
    
    is_anomaly = p_value < significance_level
    
    return {
        "n_total": n_total,
        "n_injected": n_injected,
        "empirical_rate": n_injected / n_total,
        "configured_rate": configured_rate,
        "chi_square": chi_square,
        "p_value": p_value,
        "is_anomaly": is_anomaly,
    }
```

**Reference**: `NOISE_REPLAY_LOGS.md` (Section 2-8)

---

## 5. Verifier Noise Drift Radar

### 5.1 Signal 1: Noise Rate Drift (CUSUM)

```
S_t^+ = max(0, S_{t-1}^+ + x_t - (μ_0 + k))  (upward drift)
S_t^- = max(0, S_{t-1}^- - x_t + (μ_0 - k))  (downward drift)

Alarm if S_t^+ > h or S_t^- > h
```

**Implementation**:

```python
@dataclass
class CUSUMDetector:
    mu_0: float  # Target noise rate
    k: float  # Slack parameter
    h: float  # Threshold
    
    def update(self, x: float) -> Optional[str]:
        self.S_plus = max(0, self.S_plus + x - (self.mu_0 + self.k))
        self.S_minus = max(0, self.S_minus - x + (self.mu_0 - self.k))
        
        if self.S_plus > self.h:
            return "upward_drift"
        elif self.S_minus > self.h:
            return "downward_drift"
        else:
            return None
```

### 5.2 Signal 2: Tier Skew (Two-Proportion Z-Test)

```
Z = (p_tier1 - p_tier2) / SE

where SE = sqrt(p_pooled * (1 - p_pooled) * (1/n_tier1 + 1/n_tier2))

Alarm if Z < 0 (reversed) or p-value > 0.05 (not differentiated)
```

### 5.3 Signal 3: Correlated Failure Spikes (Scan Statistics)

```
S(t, w) = (C(t, w) - E(t, w)) / sqrt(E(t, w))

where:
- C(t, w) = Σ_{s=t-w}^{t} x_s (observed failures in window)
- E(t, w) = w * μ_0 (expected failures)

Alarm if S(t, w) > threshold
```

### 5.4 Signal 4: Lean Version Drift (Bayesian Change-Point)

```
P(τ | x_{1:T}) ∝ P(x_{1:τ} | θ_1) * P(x_{τ+1:T} | θ_2) * P(τ)

Find τ that maximizes posterior probability
```

### 5.5 Signal 5: Tactic Engine Nondeterminism

```
Run verification N times with identical configuration
Count unique outcomes

Alarm if unique outcomes > 1
```

### 5.6 Signal 6: Resource Exhaustion (Linear Regression)

```
y_t = β_0 + β_1 * t + ε_t

Test: H0: β_1 = 0 vs H1: β_1 > 0

Alarm if β_1 > 0 and p-value < 0.05
```

### 5.7 Unified Drift Radar

```python
@dataclass
class VerifierNoiseDriftRadar:
    timeout_drift_detector: CUSUMDetector
    spurious_fail_drift_detector: CUSUMDetector
    tier_skew_detector: Callable
    correlated_spike_detector: ScanStatisticsDetector
    changepoint_detector: Callable
    nondeterminism_detector: Callable
    resource_exhaustion_detector: Callable
    
    alarms: List[Dict[str, Any]]
    
    def update(self, telemetry: LeanVerificationTelemetry) -> List[Dict[str, Any]]:
        new_alarms = []
        
        # Signal 1: Noise rate drift
        if telemetry.outcome == VerifierErrorCode.VERIFIER_TIMEOUT:
            alarm = self.timeout_drift_detector.update(1.0)
            if alarm:
                new_alarms.append({"signal": "timeout_drift", "type": alarm})
        
        # Signal 3: Correlated failure spike
        if not telemetry.success:
            alarm = self.correlated_spike_detector.update(True)
            if alarm:
                new_alarms.append({"signal": "correlated_failure_spike", **alarm})
        
        self.alarms.extend(new_alarms)
        return new_alarms
```

**Reference**: `NOISE_DRIFT_RADAR.md` (Section 2-9)

---

## 6. Integration Roadmap

### 6.1 Phase 3A: Calibration Pipeline (Weeks 1-2)

**Tasks**:
- [ ] Implement `LeanVerificationTelemetry` dataclass
- [ ] Implement `run_lean_with_monitoring` with psutil integration
- [ ] Implement Lean output parsing
- [ ] Implement ground truth labeling pipeline
- [ ] Implement MLE estimation for noise rates
- [ ] Implement timeout distribution fitting
- [ ] Implement confidence interval computation
- [ ] Implement cross-tier validation
- [ ] Implement model export to YAML
- [ ] Run calibration on 10,000+ proofs

**Deliverables**:
- Calibrated noise models for all three tiers
- Calibration report with validation metrics
- Production YAML configuration files

### 6.2 Phase 3B: Advanced Noise Regimes (Weeks 3-4)

**Tasks**:
- [ ] Implement `CorrelatedNoiseModel`
- [ ] Implement `ClusterDegradationModel`
- [ ] Implement `HeatDeathModel`
- [ ] Implement `HeavyTailTimeoutModel`
- [ ] Implement `NonStationaryNoiseModel`
- [ ] Implement `AdaptiveNoiseModel`
- [ ] Implement `UnifiedNoiseModel`
- [ ] Implement calibration procedures for each regime
- [ ] Create configuration schema
- [ ] Integrate with U2 runner

**Deliverables**:
- Unified noise model implementation
- Configuration YAML with all regimes
- Validation tests for each regime

### 6.3 Phase 3C: RFL Integration (Weeks 5-6)

**Tasks**:
- [ ] Implement `compute_expected_value`
- [ ] Implement `update_rfl_policy_noisy`
- [ ] Implement `update_rfl_policy_multitier`
- [ ] Implement bias correction
- [ ] Implement abstention handling
- [ ] Implement confidence-weighted updates
- [ ] Run synthetic experiments
- [ ] Run real-world validation on Mathlib

**Deliverables**:
- Noise-robust RFL implementation
- Validation report with convergence metrics
- Integration with U2 runner

### 6.4 Phase 3D: Replay Logs (Week 7)

**Tasks**:
- [ ] Implement `NoiseReplayLogWriter`
- [ ] Implement `NoiseReplayLogReader`
- [ ] Implement `replay_noise_log`
- [ ] Implement `counterfactual_analysis`
- [ ] Implement `noise_impact_analysis`
- [ ] Implement anomaly detection algorithms
- [ ] Integrate with U2 runner

**Deliverables**:
- Replay log implementation
- Anomaly detection suite
- Integration with U2 runner

### 6.5 Phase 3E: Drift Radar (Week 8)

**Tasks**:
- [ ] Implement `CUSUMDetector`
- [ ] Implement `detect_tier_skew`
- [ ] Implement `ScanStatisticsDetector`
- [ ] Implement `detect_changepoint`
- [ ] Implement `detect_nondeterminism`
- [ ] Implement `detect_resource_exhaustion`
- [ ] Implement `VerifierNoiseDriftRadar`
- [ ] Create dashboard configuration
- [ ] Integrate with U2 runner

**Deliverables**:
- Drift radar implementation
- Dashboard configuration
- Integration with U2 runner

---

## 7. Validation Protocol

### 7.1 Calibration Validation

**Experiment 1**: Collect 10,000+ telemetry records across all tiers  
**Experiment 2**: Label 1,000 proofs with ground truth  
**Experiment 3**: Fit noise models and compute confidence intervals  
**Experiment 4**: Validate on held-out test data (20%)  
**Experiment 5**: Verify tier monotonicity with hypothesis tests

**Success Criteria**:
- Calibration error < 0.01 (1%)
- Coverage ≥ 0.95 (95% of test data within confidence intervals)
- Tier monotonicity validated (p-value < 0.05)

### 7.2 Advanced Noise Regimes Validation

**Experiment 6**: Validate correlated failures (correlation coefficient > 0.1)  
**Experiment 7**: Validate cluster degradation (variance-to-mean ratio > 2)  
**Experiment 8**: Validate heavy tails (tail index α < 2)  
**Experiment 9**: Validate non-stationary drift (change-point detection p-value < 0.05)  
**Experiment 10**: Validate adaptive noise (robustness score improvement > 10%)

**Success Criteria**:
- Each regime produces expected statistical signatures
- Unified model passes all validation tests

### 7.3 RFL Integration Validation

**Experiment 11**: Synthetic experiments with known ground truth  
**Experiment 12**: Convergence rate comparison (noise-robust vs standard)  
**Experiment 13**: Policy drift measurement (10,000 cycles)  
**Experiment 14**: Mathlib proofs (success rate comparison)  
**Experiment 15**: Curriculum learning (learning curve analysis)

**Success Criteria**:
- Noise-robust RFL converges to ground truth (accuracy > 90%)
- Convergence rate ≥ 1.5x faster than standard RFL
- Policy drift < 0.01 per 1000 cycles
- Success rate improvement > 5% on Mathlib

### 7.4 Replay Logs Validation

**Experiment 16**: Replay 1,000 log entries and verify identical outcomes  
**Experiment 17**: Counterfactual analysis (difference rate > 5%)  
**Experiment 18**: Noise impact analysis (KL divergence > 0.1)  
**Experiment 19**: Anomaly detection (detect injected anomalies with precision > 0.9)

**Success Criteria**:
- 100% replay accuracy (identical outcomes)
- Counterfactual analysis detects noise impact
- Anomaly detection precision > 0.9, recall > 0.8

### 7.5 Drift Radar Validation

**Experiment 20**: Inject synthetic drift and measure detection latency  
**Experiment 21**: Inject tier skew and verify detection  
**Experiment 22**: Inject correlated failure spike and verify detection  
**Experiment 23**: Simulate Lean version change and verify change-point detection  
**Experiment 24**: Inject nondeterminism and verify detection

**Success Criteria**:
- Drift detection latency < 100 cycles
- False alarm rate < 0.01 (1%)
- Detection precision > 0.9, recall > 0.8

---

## 8. Risk Analysis

### 8.1 Technical Risks

**Risk 1: Calibration Data Insufficient** — 10,000 proofs may not cover all failure modes  
*Mitigation*: Collect 50,000+ proofs from diverse sources (Mathlib, curriculum, synthetic)

**Risk 2: Ground Truth Labeling Errors** — Manual labeling may introduce errors  
*Mitigation*: Use consensus verification and extended timeout for validation

**Risk 3: Advanced Regimes Too Complex** — Unified model may be difficult to calibrate  
*Mitigation*: Start with simple regimes, add complexity incrementally

**Risk 4: RFL Integration Instability** — Noise-robust updates may cause divergence  
*Mitigation*: Use conservative learning rates, add regularization

**Risk 5: Replay Logs Too Large** — Logs may consume excessive storage  
*Mitigation*: Use compression (gzip, zstd), implement log rotation

**Risk 6: Drift Radar False Alarms** — Detectors may trigger spurious alarms  
*Mitigation*: Calibrate thresholds for target false alarm rate (1%)

### 8.2 Operational Risks

**Risk 7: Performance Overhead** — Telemetry collection may slow verification  
*Mitigation*: Use asynchronous logging, batch writes

**Risk 8: Integration Complexity** — Phase III adds significant complexity to U2 runner  
*Mitigation*: Modular design, comprehensive testing, staged rollout

**Risk 9: Dashboard Maintenance** — Drift radar dashboard requires ongoing maintenance  
*Mitigation*: Use standard tools (Grafana), automate alerts

---

## 9. Success Metrics

### 9.1 Calibration Metrics

- **Calibration Error**: < 0.01 (1%)
- **Coverage**: ≥ 0.95 (95%)
- **Tier Monotonicity**: Validated (p-value < 0.05)

### 9.2 RFL Metrics

- **Convergence Accuracy**: > 90%
- **Convergence Speed**: ≥ 1.5x faster than standard RFL
- **Policy Drift**: < 0.01 per 1000 cycles
- **Success Rate Improvement**: > 5% on Mathlib

### 9.3 Replay Metrics

- **Replay Accuracy**: 100%
- **Anomaly Detection Precision**: > 0.9
- **Anomaly Detection Recall**: > 0.8

### 9.4 Drift Radar Metrics

- **Detection Latency**: < 100 cycles
- **False Alarm Rate**: < 0.01 (1%)
- **Detection Precision**: > 0.9
- **Detection Recall**: > 0.8

---

## 10. Conclusion

Phase III provides a comprehensive framework for transitioning from synthetic noise modeling to empirical, RFL-integrated verifier imperfection analytics. The master plan includes:

**1. Real-World Calibration Pipeline** — Instrumentation, ground truth labeling, statistical fitting, cross-tier validation, and model export

**2. Advanced Noise Regimes** — Correlated failures, cluster degradation, heat-death scenarios, heavy-tailed timeouts, non-stationary noise, and policy-aware adaptive noise

**3. RFL Integration Mathematics** — Expected value functions, abstention handling, confidence-weighted updates, multi-tier integration, bias correction, and optimal policies

**4. Noise Replay Logs** — Deterministic replay semantics, differential debugging, and anomaly detection

**5. Verifier Noise Drift Radar** — Real-time detection of noise rate drift, tier skew, correlated failure spikes, Lean version drift, tactic engine nondeterminism, and resource exhaustion patterns

The implementation roadmap spans 8 weeks with clear deliverables, validation protocols, risk analysis, and success metrics. All designs are mathematically rigorous, deterministically reproducible, and production-ready.

---

**Manus-C — Telemetry Architect**  
*"Every packet accounted for, every signal explained."*

**Phase III Status**: Design Complete, Implementation Ready  
**Next Steps**: Begin Phase 3A (Calibration Pipeline)
