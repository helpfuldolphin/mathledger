# Phase III: Advanced Noise Regimes

**Author**: Manus-C (Telemetry Architect)  
**Date**: 2025-12-06  
**Mission**: Extend beyond independent Bernoulli noise to correlated, adaptive, and non-stationary noise models

---

## 1. Overview

Phase-II implemented **independent Bernoulli noise**: each verifier call has independent probability of timeout, spurious failure, or spurious pass. This is a reasonable first-order approximation, but real-world verifier behavior exhibits **correlation**, **degradation**, and **non-stationarity**.

Phase III introduces **advanced noise regimes** that model these phenomena:

**Regime 1: Correlated Failures** — Verifier failures are correlated across items (spatial correlation)  
**Regime 2: Cluster-Based Degradation** — Verifier degrades in clusters (temporal correlation)  
**Regime 3: Heat-Death Scenarios** — Resource exhaustion patterns (memory leaks, CPU throttling)  
**Regime 4: High-Tactic-Depth Tails** — Heavy-tailed timeout distributions for complex proofs  
**Regime 5: Non-Stationary Noise** — Noise rates drift over time  
**Regime 6: Policy-Aware Adaptive Noise** — Noise adapts to RFL policy confidence

---

## 2. Regime 1: Correlated Failures

### 2.1 Motivation

Real-world verifier failures are not independent. If the verifier fails on proof A due to a bug in tactic X, it will likely fail on all proofs that use tactic X. This creates **spatial correlation** across items.

### 2.2 Mathematical Model

Model correlated failures using a **latent factor model**:

```
Let z_k ~ Bernoulli(ρ) be latent failure factors (k = 1, ..., K)
Let A_ik ∈ {0, 1} be indicator that item i uses factor k

Then the failure probability for item i is:
P(fail_i | z) = 1 - ∏_{k: A_ik = 1} (1 - z_k * θ_k)

where θ_k is the base failure rate for factor k
```

**Interpretation**: Each factor k represents a potential failure mode (e.g., bug in tactic, memory leak in module). If factor k is active (z_k = 1), all items using factor k have elevated failure probability.

**Parameters**:
- ρ: Probability that a factor is active (controls correlation strength)
- θ_k: Base failure rate for factor k
- A_ik: Item-factor incidence matrix (which items use which factors)

### 2.3 Factor Identification

Factors can be identified from proof structure:

**Factor 1: Tactic Usage** — Items using the same tactic (e.g., `simp`, `ring`, `omega`) share a factor

**Factor 2: Module Dependency** — Items importing the same module share a factor

**Factor 3: Proof Technique** — Items using the same proof technique (induction, case analysis, rewriting) share a factor

**Factor 4: Lemma Usage** — Items using the same lemma share a factor

### 2.4 Implementation

```python
@dataclass
class CorrelatedNoiseModel:
    """Correlated noise model with latent factors."""
    
    rho: float  # Factor activation probability
    theta: Dict[str, float]  # Base failure rate per factor
    item_factors: Dict[str, List[str]]  # Item → factors mapping
    prng: DeterministicPRNG
    
    def __post_init__(self):
        # Sample factor activations (deterministic given seed)
        self.active_factors = {
            factor: self.prng.for_path("factor", factor).random() < self.rho
            for factor in self.theta.keys()
        }
    
    def should_fail(self, item: str) -> bool:
        """Determine if item should fail due to correlated factors."""
        factors = self.item_factors.get(item, [])
        
        # Compute failure probability
        prob_success = 1.0
        for factor in factors:
            if self.active_factors.get(factor, False):
                prob_success *= (1 - self.theta[factor])
        
        prob_fail = 1 - prob_success
        
        # Sample failure decision
        return self.prng.for_path("item", item).random() < prob_fail
```

### 2.5 Calibration

Estimate ρ and θ_k from empirical data:

**Step 1**: Cluster failures by shared factors (e.g., all failures using tactic X)

**Step 2**: Estimate ρ as the fraction of factors with elevated failure rates

**Step 3**: Estimate θ_k as the empirical failure rate within each cluster

**Step 4**: Validate by checking if correlated model fits data better than independent model (use likelihood ratio test)

---

## 3. Regime 2: Cluster-Based Degradation

### 3.1 Motivation

Verifiers degrade over time due to memory leaks, CPU throttling, or resource exhaustion. Failures occur in **temporal clusters**: a period of normal operation followed by a burst of failures, then recovery.

### 3.2 Mathematical Model

Model temporal clustering using a **hidden Markov model (HMM)**:

```
States: S = {HEALTHY, DEGRADED}

Transition probabilities:
P(HEALTHY → DEGRADED) = α
P(DEGRADED → HEALTHY) = β

Emission probabilities:
P(fail | HEALTHY) = θ_healthy
P(fail | DEGRADED) = θ_degraded

where θ_degraded >> θ_healthy
```

**Interpretation**: The verifier alternates between HEALTHY and DEGRADED states. In HEALTHY state, failures are rare. In DEGRADED state, failures are common.

**Parameters**:
- α: Probability of transitioning to DEGRADED state (controls cluster frequency)
- β: Probability of recovering to HEALTHY state (controls cluster duration)
- θ_healthy: Failure rate in HEALTHY state
- θ_degraded: Failure rate in DEGRADED state

### 3.3 Implementation

```python
@dataclass
class ClusterDegradationModel:
    """Cluster-based degradation model with HMM."""
    
    alpha: float  # Transition probability HEALTHY → DEGRADED
    beta: float  # Transition probability DEGRADED → HEALTHY
    theta_healthy: float  # Failure rate in HEALTHY state
    theta_degraded: float  # Failure rate in DEGRADED state
    prng: DeterministicPRNG
    
    def __post_init__(self):
        # Initialize state
        self.state = "HEALTHY"
        self.cycle_count = 0
    
    def step(self) -> None:
        """Advance HMM by one time step."""
        self.cycle_count += 1
        
        # Sample state transition
        if self.state == "HEALTHY":
            if self.prng.for_path("transition", str(self.cycle_count)).random() < self.alpha:
                self.state = "DEGRADED"
        else:  # DEGRADED
            if self.prng.for_path("transition", str(self.cycle_count)).random() < self.beta:
                self.state = "HEALTHY"
    
    def should_fail(self, item: str) -> bool:
        """Determine if item should fail given current state."""
        theta = self.theta_degraded if self.state == "DEGRADED" else self.theta_healthy
        return self.prng.for_path("item", item, str(self.cycle_count)).random() < theta
```

### 3.4 Calibration

Estimate α, β, θ_healthy, θ_degraded using **Baum-Welch algorithm** (EM for HMMs):

**E-step**: Compute posterior probabilities of states given observed failures

**M-step**: Update parameters to maximize likelihood

**Convergence**: Iterate until log-likelihood converges

Alternatively, use **Viterbi algorithm** to infer most likely state sequence, then estimate parameters from state assignments.

---

## 4. Regime 3: Heat-Death Scenarios

### 4.1 Motivation

Verifiers can enter **heat-death** scenarios where resource exhaustion leads to cascading failures. Examples:

- **Memory leak**: Memory usage grows unboundedly, eventually causing OOM
- **CPU throttling**: CPU temperature exceeds threshold, causing throttling and timeouts
- **Disk exhaustion**: Temporary files fill disk, causing I/O errors

### 4.2 Mathematical Model

Model heat-death using a **resource depletion process**:

```
Let R(t) be available resource at time t (e.g., free memory in MB)

Resource dynamics:
R(t+1) = R(t) - c(t) + r(t)

where:
- c(t) is resource consumption at time t (random variable)
- r(t) is resource recovery at time t (e.g., garbage collection)

Failure condition:
fail(t) = 1 if R(t) < R_min else 0

where R_min is minimum resource threshold
```

**Interpretation**: Each verification consumes resources (memory, CPU, disk). If resources drop below threshold, verification fails. Resources recover through garbage collection or cooling.

**Parameters**:
- R_0: Initial resource level
- R_min: Minimum resource threshold
- c_mean, c_std: Mean and std dev of resource consumption
- r_mean, r_std: Mean and std dev of resource recovery

### 4.3 Implementation

```python
@dataclass
class HeatDeathModel:
    """Heat-death model with resource depletion."""
    
    R_0: float  # Initial resource level (e.g., 16000 MB)
    R_min: float  # Minimum resource threshold (e.g., 1000 MB)
    c_mean: float  # Mean resource consumption per verification
    c_std: float  # Std dev of resource consumption
    r_mean: float  # Mean resource recovery per cycle
    r_std: float  # Std dev of resource recovery
    prng: DeterministicPRNG
    
    def __post_init__(self):
        self.R = self.R_0  # Current resource level
        self.cycle_count = 0
    
    def step(self, item: str) -> bool:
        """Execute one verification and update resource level."""
        self.cycle_count += 1
        
        # Sample resource consumption
        c = self.prng.for_path("consumption", str(self.cycle_count)).gauss(self.c_mean, self.c_std)
        c = max(0, c)  # Consumption cannot be negative
        
        # Sample resource recovery
        r = self.prng.for_path("recovery", str(self.cycle_count)).gauss(self.r_mean, self.r_std)
        r = max(0, r)  # Recovery cannot be negative
        
        # Update resource level
        self.R = self.R - c + r
        self.R = max(0, self.R)  # Resource cannot be negative
        
        # Check failure condition
        if self.R < self.R_min:
            return True  # Fail due to resource exhaustion
        else:
            return False  # Success
```

### 4.4 Calibration

Estimate parameters from telemetry data:

**Step 1**: Extract memory usage traces from telemetry (memory_peak_mb, memory_final_mb)

**Step 2**: Compute resource consumption: c(t) = memory_peak(t) - memory_final(t-1)

**Step 3**: Compute resource recovery: r(t) = memory_final(t-1) - memory_final(t) (if positive)

**Step 4**: Fit Gaussian distributions to c(t) and r(t) using MLE

**Step 5**: Estimate R_min as the 5th percentile of memory_final_mb for failed verifications

---

## 5. Regime 4: High-Tactic-Depth Tails

### 5.1 Motivation

Timeout distributions are **heavy-tailed**: most proofs complete quickly, but a small fraction take exponentially longer. This is due to high-tactic-depth proofs that explore large search spaces.

### 5.2 Mathematical Model

Model heavy tails using a **mixture distribution**:

```
Timeout duration T ~ (1 - π) * Exp(λ_fast) + π * Pareto(α, x_min)

where:
- Exp(λ_fast) is exponential distribution for fast proofs
- Pareto(α, x_min) is Pareto distribution for slow proofs
- π is mixing probability (fraction of slow proofs)
```

**Pareto distribution**: f(t; α, x_min) = (α * x_min^α) / t^(α+1) for t ≥ x_min

**Heavy tail property**: P(T > t) ~ t^(-α) as t → ∞ (power-law decay)

**Parameters**:
- λ_fast: Rate parameter for fast proofs
- α: Shape parameter for slow proofs (α < 2 implies infinite variance)
- x_min: Minimum timeout for slow proofs
- π: Mixing probability

### 5.3 Implementation

```python
@dataclass
class HeavyTailTimeoutModel:
    """Heavy-tailed timeout model with mixture distribution."""
    
    lambda_fast: float  # Rate for exponential component
    alpha: float  # Shape for Pareto component
    x_min: float  # Minimum for Pareto component
    pi: float  # Mixing probability
    prng: DeterministicPRNG
    
    def sample_timeout_duration(self, item: str) -> float:
        """Sample timeout duration from mixture distribution."""
        # Decide which component to sample from
        u = self.prng.for_path("mixture", item).random()
        
        if u < self.pi:
            # Sample from Pareto (slow proofs)
            v = self.prng.for_path("pareto", item).random()
            return self.x_min * (1 - v) ** (-1 / self.alpha)
        else:
            # Sample from Exponential (fast proofs)
            v = self.prng.for_path("exponential", item).random()
            return -np.log(1 - v) / self.lambda_fast
```

### 5.4 Calibration

Estimate parameters using **EM algorithm for mixture models**:

**E-step**: Compute posterior probability that each timeout came from Pareto component

**M-step**: Update λ_fast, α, x_min, π to maximize likelihood

**Convergence**: Iterate until log-likelihood converges

Alternatively, use **Hill estimator** to estimate α from tail data, then fit exponential to non-tail data.

---

## 6. Regime 5: Non-Stationary Noise

### 6.1 Motivation

Noise rates drift over time due to:
- **Lean version updates**: New Lean versions introduce bugs or performance regressions
- **Tactic engine changes**: Tactic implementations change, affecting failure rates
- **Hardware degradation**: CPU/memory performance degrades over time
- **Workload shifts**: Proof corpus changes, affecting failure patterns

### 6.2 Mathematical Model

Model non-stationary noise using **time-varying parameters**:

```
θ_timeout(t) = θ_0 + δ * t + ε(t)

where:
- θ_0 is initial timeout rate
- δ is drift rate (linear trend)
- ε(t) ~ N(0, σ²) is random noise (Gaussian)
```

For more complex drift, use **Gaussian process**:

```
θ_timeout(t) ~ GP(μ(t), k(t, t'))

where:
- μ(t) is mean function (e.g., linear or polynomial)
- k(t, t') is covariance function (e.g., squared exponential)
```

### 6.3 Implementation

```python
@dataclass
class NonStationaryNoiseModel:
    """Non-stationary noise model with time-varying parameters."""
    
    theta_0: float  # Initial noise rate
    delta: float  # Drift rate
    sigma: float  # Noise std dev
    prng: DeterministicPRNG
    
    def get_noise_rate(self, cycle: int) -> float:
        """Get noise rate at given cycle."""
        # Linear drift
        theta_drift = self.theta_0 + self.delta * cycle
        
        # Add Gaussian noise
        epsilon = self.prng.for_path("drift", str(cycle)).gauss(0, self.sigma)
        theta = theta_drift + epsilon
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, theta))
    
    def should_fail(self, item: str, cycle: int) -> bool:
        """Determine if item should fail at given cycle."""
        theta = self.get_noise_rate(cycle)
        return self.prng.for_path("item", item, str(cycle)).random() < theta
```

### 6.4 Calibration

Estimate drift parameters using **online learning**:

**Step 1**: Partition telemetry data into time windows (e.g., daily, weekly)

**Step 2**: Estimate noise rate within each window using MLE

**Step 3**: Fit linear regression: θ(t) = θ_0 + δ * t

**Step 4**: Compute residuals: ε(t) = θ_empirical(t) - θ_fitted(t)

**Step 5**: Estimate σ as std dev of residuals

For Gaussian process, use **GP regression** libraries (e.g., scikit-learn, GPy).

---

## 7. Regime 6: Policy-Aware Adaptive Noise

### 7.1 Motivation

RFL policies learn to avoid items that frequently fail. If noise is static, the policy will eventually learn to avoid all noisy items, making the policy overly conservative. **Adaptive noise** adjusts to policy confidence: inject more noise on items where policy is confident, less noise on items where policy is uncertain.

### 7.2 Mathematical Model

Model adaptive noise as a function of policy confidence:

```
θ_adaptive(item, π) = θ_base * (1 + γ * confidence(item, π))

where:
- θ_base is base noise rate
- confidence(item, π) is policy confidence on item (e.g., |π(item) - 0.5|)
- γ is adaptation strength
```

**Interpretation**: If policy is confident (π(item) ≈ 1 or π(item) ≈ 0), inject more noise to challenge the policy. If policy is uncertain (π(item) ≈ 0.5), inject less noise to allow learning.

### 7.3 Implementation

```python
@dataclass
class AdaptiveNoiseModel:
    """Adaptive noise model that responds to RFL policy confidence."""
    
    theta_base: float  # Base noise rate
    gamma: float  # Adaptation strength
    prng: DeterministicPRNG
    
    def get_noise_rate(self, item: str, policy_prob: float) -> float:
        """Get noise rate for item given policy probability."""
        # Compute policy confidence (distance from 0.5)
        confidence = abs(policy_prob - 0.5) * 2  # Normalize to [0, 1]
        
        # Adaptive noise rate
        theta = self.theta_base * (1 + self.gamma * confidence)
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, theta))
    
    def should_fail(self, item: str, policy_prob: float) -> bool:
        """Determine if item should fail given policy probability."""
        theta = self.get_noise_rate(item, policy_prob)
        return self.prng.for_path("item", item).random() < theta
```

### 7.4 Calibration

Estimate γ by maximizing **policy robustness**:

**Step 1**: Train RFL policy with different values of γ (e.g., γ ∈ {0, 0.5, 1.0, 2.0})

**Step 2**: Evaluate policy robustness: fraction of items where policy is correct under noise

**Step 3**: Select γ that maximizes robustness

Alternatively, use **adversarial training**: train policy and noise model jointly, with noise model trying to maximize policy error and policy trying to minimize error.

---

## 8. Unified Noise Model

### 8.1 Architecture

Combine all noise regimes into a unified model:

```python
@dataclass
class UnifiedNoiseModel:
    """Unified noise model combining all regimes."""
    
    # Base noise rates
    base_timeout_rate: float
    base_spurious_fail_rate: float
    base_spurious_pass_rate: float
    
    # Advanced regimes (optional)
    correlated_model: Optional[CorrelatedNoiseModel] = None
    degradation_model: Optional[ClusterDegradationModel] = None
    heat_death_model: Optional[HeatDeathModel] = None
    heavy_tail_model: Optional[HeavyTailTimeoutModel] = None
    nonstationary_model: Optional[NonStationaryNoiseModel] = None
    adaptive_model: Optional[AdaptiveNoiseModel] = None
    
    prng: DeterministicPRNG
    
    def should_timeout(self, item: str, cycle: int, policy_prob: float) -> bool:
        """Determine if item should timeout."""
        # Start with base rate
        theta = self.base_timeout_rate
        
        # Apply non-stationary drift
        if self.nonstationary_model:
            theta = self.nonstationary_model.get_noise_rate(cycle)
        
        # Apply adaptive adjustment
        if self.adaptive_model:
            theta = self.adaptive_model.get_noise_rate(item, policy_prob)
        
        # Apply correlated failures
        if self.correlated_model and self.correlated_model.should_fail(item):
            return True
        
        # Apply cluster degradation
        if self.degradation_model:
            self.degradation_model.step()
            if self.degradation_model.should_fail(item):
                return True
        
        # Apply heat death
        if self.heat_death_model and self.heat_death_model.step(item):
            return True
        
        # Apply base timeout decision
        return self.prng.for_path("timeout", item, str(cycle)).random() < theta
    
    def sample_timeout_duration(self, item: str) -> float:
        """Sample timeout duration."""
        if self.heavy_tail_model:
            return self.heavy_tail_model.sample_timeout_duration(item)
        else:
            # Fallback to exponential
            return self.prng.for_path("duration", item).expovariate(1.0 / 1000.0)
```

### 8.2 Configuration

Enable/disable regimes via configuration:

```yaml
unified_noise_model:
  base_timeout_rate: 0.05
  base_spurious_fail_rate: 0.02
  base_spurious_pass_rate: 0.01
  
  regimes:
    correlated_failures:
      enabled: true
      rho: 0.1  # 10% of factors are active
      factors:
        tactic_simp: 0.05
        tactic_ring: 0.03
        module_algebra: 0.02
    
    cluster_degradation:
      enabled: true
      alpha: 0.01  # 1% chance of entering degraded state per cycle
      beta: 0.1  # 10% chance of recovering per cycle
      theta_healthy: 0.02
      theta_degraded: 0.2
    
    heat_death:
      enabled: false  # Disabled for now
    
    heavy_tail_timeouts:
      enabled: true
      lambda_fast: 0.001  # Mean 1000ms for fast proofs
      alpha: 1.5  # Heavy tail with infinite variance
      x_min: 5000  # Slow proofs start at 5000ms
      pi: 0.05  # 5% of proofs are slow
    
    nonstationary_drift:
      enabled: true
      theta_0: 0.05
      delta: 0.0001  # 0.01% increase per cycle
      sigma: 0.005
    
    adaptive_noise:
      enabled: true
      gamma: 0.5  # Moderate adaptation strength
```

---

## 9. Validation and Testing

### 9.1 Validation Metrics

**Metric 1: Correlation Coefficient** — Measure correlation between failures on items sharing factors

**Metric 2: Cluster Coefficient** — Measure temporal clustering of failures (ratio of variance to mean)

**Metric 3: Tail Index** — Estimate tail index α from timeout durations using Hill estimator

**Metric 4: Drift Detection** — Use change-point detection to identify drift in noise rates

**Metric 5: Robustness Score** — Measure RFL policy robustness under adaptive noise

### 9.2 Testing Protocol

**Test 1: Correlation Test** — Verify that correlated model produces higher failure correlation than independent model

**Test 2: Cluster Test** — Verify that degradation model produces temporal clusters (high variance-to-mean ratio)

**Test 3: Tail Test** — Verify that heavy-tail model produces power-law tails (log-log plot is linear)

**Test 4: Drift Test** — Verify that non-stationary model produces detectable drift

**Test 5: Adaptation Test** — Verify that adaptive model increases noise on confident items

---

## 10. Implementation Checklist

- [ ] Implement `CorrelatedNoiseModel` with latent factors
- [ ] Implement `ClusterDegradationModel` with HMM
- [ ] Implement `HeatDeathModel` with resource depletion
- [ ] Implement `HeavyTailTimeoutModel` with mixture distribution
- [ ] Implement `NonStationaryNoiseModel` with time-varying parameters
- [ ] Implement `AdaptiveNoiseModel` with policy-aware adjustment
- [ ] Implement `UnifiedNoiseModel` combining all regimes
- [ ] Implement calibration procedures for each regime
- [ ] Implement validation metrics and testing protocol
- [ ] Create configuration schema for unified model
- [ ] Integrate unified model with U2 runner
- [ ] Run experiments to validate each regime

---

**Manus-C — Telemetry Architect**  
*"Every packet accounted for, every signal explained."*
