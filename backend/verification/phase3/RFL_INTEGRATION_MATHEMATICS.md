# Phase III: RFL Integration Mathematics with Verifier Epistemic Uncertainty

**Author**: Manus-C (Telemetry Architect)  
**Date**: 2025-12-06  
**Mission**: Formalize RFL update equations under verifier imperfection and epistemic uncertainty

---

## 1. Overview

**Reinforcement from Lean Feedback (RFL)** is a policy learning algorithm that updates a policy π based on verifier feedback. In Phase-I and Phase-II, RFL assumed **perfect verifier feedback**: if the verifier says VERIFIED, the proof is valid; if it says FAILED, the proof is invalid.

Phase-III introduces **imperfect verifier feedback** with epistemic uncertainty:
- Timeouts provide no information (abstention)
- Spurious failures are false negatives (verifier says FAILED but proof is valid)
- Spurious passes are false positives (verifier says VERIFIED but proof is invalid)

This document formalizes the RFL update equation under verifier imperfection and derives optimal policies that are robust to noise.

---

## 2. RFL Update Equation (Phase-I Baseline)

### 2.1 Standard RFL Update

In Phase-I, RFL updates the policy π using the following equation:

```
Δπ(e) = η * V(e, o) * ∇_π log π(e)

where:
- π(e) is the probability of selecting proof attempt e
- V(e, o) is the value function (reward) for attempt e with outcome o
- η is the learning rate
- ∇_π log π(e) is the policy gradient
```

**Value Function**:
```
V(e, VERIFIED) = +1  (positive reward)
V(e, FAILED) = -1  (negative reward)
```

**Policy Update**:
```
π_{t+1}(e) = π_t(e) + Δπ(e)
```

**Normalization**:
```
π_{t+1}(e) ← π_{t+1}(e) / Σ_e' π_{t+1}(e')
```

### 2.2 Limitations

This formulation assumes **perfect verifier feedback**:
- V(e, VERIFIED) = +1 assumes verifier is always correct when it says VERIFIED
- V(e, FAILED) = -1 assumes verifier is always correct when it says FAILED

Under imperfect verifier feedback, these assumptions break down:
- Timeouts provide no information about proof validity
- Spurious failures penalize valid proofs
- Spurious passes reward invalid proofs

---

## 3. RFL with Epistemic Uncertainty

### 3.1 Verifier Outcome Model

Model verifier outcomes as noisy observations of ground truth:

```
Let y ∈ {VALID, INVALID} be ground truth proof validity
Let o ∈ {VERIFIED, FAILED, TIMEOUT} be verifier outcome

Verifier model:
P(o = VERIFIED | y = VALID) = 1 - θ_spurious_fail - θ_timeout
P(o = FAILED | y = VALID) = θ_spurious_fail
P(o = TIMEOUT | y = VALID) = θ_timeout

P(o = VERIFIED | y = INVALID) = θ_spurious_pass
P(o = FAILED | y = INVALID) = 1 - θ_spurious_pass - θ_timeout
P(o = TIMEOUT | y = INVALID) = θ_timeout
```

**Interpretation**:
- θ_spurious_fail: Probability of spurious failure (false negative)
- θ_spurious_pass: Probability of spurious pass (false positive)
- θ_timeout: Probability of timeout (abstention)

### 3.2 Posterior Ground Truth Probability

Given verifier outcome o, compute posterior probability of ground truth validity:

```
P(y = VALID | o) = P(o | y = VALID) * P(y = VALID) / P(o)

where:
P(o) = P(o | y = VALID) * P(y = VALID) + P(o | y = INVALID) * P(y = INVALID)
```

**Assume uniform prior**: P(y = VALID) = P(y = INVALID) = 0.5

**Case 1: o = VERIFIED**
```
P(y = VALID | o = VERIFIED) = (1 - θ_sf - θ_to) / [(1 - θ_sf - θ_to) + θ_sp]

where θ_sf = θ_spurious_fail, θ_to = θ_timeout, θ_sp = θ_spurious_pass
```

**Case 2: o = FAILED**
```
P(y = VALID | o = FAILED) = θ_sf / [θ_sf + (1 - θ_sp - θ_to)]
```

**Case 3: o = TIMEOUT**
```
P(y = VALID | o = TIMEOUT) = 0.5  (no information)
```

### 3.3 Expected Value Function

Define expected value function as expected reward under posterior:

```
V_expected(e, o) = P(y = VALID | o) * V(e, VALID) + P(y = INVALID | o) * V(e, INVALID)

where:
V(e, VALID) = +1  (reward for valid proof)
V(e, INVALID) = -1  (penalty for invalid proof)
```

**Simplification**:
```
V_expected(e, o) = 2 * P(y = VALID | o) - 1
```

**Case 1: o = VERIFIED**
```
V_expected(e, VERIFIED) = 2 * [(1 - θ_sf - θ_to) / [(1 - θ_sf - θ_to) + θ_sp]] - 1
```

**Case 2: o = FAILED**
```
V_expected(e, FAILED) = 2 * [θ_sf / [θ_sf + (1 - θ_sp - θ_to)]] - 1
```

**Case 3: o = TIMEOUT**
```
V_expected(e, TIMEOUT) = 0  (no information, no update)
```

### 3.4 Noise-Robust RFL Update

Update policy using expected value function:

```
Δπ(e) = η * V_expected(e, o) * ∇_π log π(e)

π_{t+1}(e) = π_t(e) + Δπ(e)
```

**Key Insight**: Expected value function down-weights noisy outcomes:
- If θ_spurious_fail is high, V_expected(e, FAILED) is closer to 0 (less penalty)
- If θ_spurious_pass is high, V_expected(e, VERIFIED) is closer to 0 (less reward)
- If θ_timeout is high, more outcomes result in no update (abstention)

---

## 4. Abstention Handling

### 4.1 Abstention Policy

When verifier times out, RFL should **abstain** from updating the policy:

```
if o == TIMEOUT:
    Δπ(e) = 0  (no update)
```

**Rationale**: Timeouts provide no information about proof validity. Updating the policy based on timeouts would inject noise into the learning process.

### 4.2 Abstention Rate

Define abstention rate as fraction of outcomes that result in no update:

```
α = P(o = TIMEOUT) = θ_timeout
```

**Impact on Learning**:
- High abstention rate (α → 1) slows learning (fewer updates)
- Low abstention rate (α → 0) speeds learning (more updates)

**Trade-off**: Abstention reduces noise but slows convergence.

### 4.3 Effective Learning Rate

Adjust learning rate to account for abstention:

```
η_effective = η / (1 - α)
```

**Rationale**: If α fraction of outcomes result in abstention, effective learning rate is reduced by factor (1 - α). Compensate by increasing η.

---

## 5. Confidence-Weighted Updates

### 5.1 Motivation

Not all verifier outcomes are equally informative. VERIFIED outcomes with low spurious pass rate are highly informative. VERIFIED outcomes with high spurious pass rate are less informative.

**Confidence-weighted updates** scale the policy update by the confidence in the verifier outcome.

### 5.2 Confidence Function

Define confidence as the absolute value of expected value:

```
C(e, o) = |V_expected(e, o)|
```

**Interpretation**:
- C(e, o) = 1: High confidence (verifier outcome is highly informative)
- C(e, o) = 0: Low confidence (verifier outcome is uninformative)

**Examples**:
- If θ_spurious_fail = 0, then C(e, FAILED) = 1 (high confidence in failure)
- If θ_spurious_fail = 0.5, then C(e, FAILED) ≈ 0 (low confidence in failure)

### 5.3 Confidence-Weighted Update

Scale policy update by confidence:

```
Δπ(e) = η * C(e, o) * sign(V_expected(e, o)) * ∇_π log π(e)

where sign(x) = +1 if x > 0, -1 if x < 0, 0 if x = 0
```

**Rationale**: High-confidence outcomes receive large updates. Low-confidence outcomes receive small updates.

---

## 6. Multi-Tier Verifier Integration

### 6.1 Tier-Specific Noise Rates

Each verifier tier has different noise rates:

```
Tier 1 (FAST_NOISY): θ_sf = 0.05, θ_sp = 0.02, θ_to = 0.10
Tier 2 (BALANCED): θ_sf = 0.02, θ_sp = 0.01, θ_to = 0.05
Tier 3 (SLOW_PRECISE): θ_sf = 0.005, θ_sp = 0.001, θ_to = 0.01
```

### 6.2 Tier-Weighted Value Function

Compute expected value function for each tier:

```
V_expected(e, o, tier) = 2 * P(y = VALID | o, tier) - 1

where P(y = VALID | o, tier) uses tier-specific noise rates
```

### 6.3 Escalation-Aware Updates

When escalation occurs, use the final tier's outcome:

```
if escalation occurred:
    tier_final = final tier reached (e.g., SLOW_PRECISE)
    o_final = final outcome
    Δπ(e) = η * V_expected(e, o_final, tier_final) * ∇_π log π(e)
```

**Rationale**: Final tier has lowest noise rates, so its outcome is most informative.

### 6.4 Multi-Outcome Aggregation

Alternatively, aggregate outcomes from all tiers:

```
V_aggregated(e, outcomes) = Σ_tier w_tier * V_expected(e, o_tier, tier)

where:
- outcomes = [(o_1, tier_1), (o_2, tier_2), ...]
- w_tier = confidence weight for tier (e.g., w_SLOW = 1.0, w_BALANCED = 0.5, w_FAST = 0.2)
```

**Rationale**: Use all available information, weighted by tier confidence.

---

## 7. RFL-Stable Policies Under Noise

### 7.1 Definition

A policy π is **RFL-stable** under noise if:

```
E[Δπ(e) | π, θ] = 0  for all e

where expectation is over verifier noise
```

**Interpretation**: On average, policy updates cancel out, so policy does not drift.

### 7.2 Stability Condition

For RFL-stability, the expected value function must be unbiased:

```
E[V_expected(e, o) | y] = V(e, y)

where expectation is over verifier outcomes o given ground truth y
```

**Verification**:
```
E[V_expected(e, o) | y = VALID]
  = P(o = VERIFIED | y = VALID) * V_expected(e, VERIFIED)
    + P(o = FAILED | y = VALID) * V_expected(e, FAILED)
    + P(o = TIMEOUT | y = VALID) * V_expected(e, TIMEOUT)

  = (1 - θ_sf - θ_to) * V_expected(e, VERIFIED)
    + θ_sf * V_expected(e, FAILED)
    + θ_to * 0

  = (1 - θ_sf - θ_to) * [2 * P(VALID | VERIFIED) - 1]
    + θ_sf * [2 * P(VALID | FAILED) - 1]
```

After algebraic simplification (assuming uniform prior):

```
E[V_expected(e, o) | y = VALID] = 1 - 2 * θ_sp  (approximately)
```

**Bias**: If θ_sp > 0, expected value is biased downward. Policy will under-estimate value of valid proofs.

### 7.3 Bias Correction

Correct bias by adjusting value function:

```
V_corrected(e, o) = V_expected(e, o) / (1 - 2 * θ_sp)
```

**Verification**:
```
E[V_corrected(e, o) | y = VALID] = E[V_expected(e, o) | y = VALID] / (1 - 2 * θ_sp)
                                  ≈ (1 - 2 * θ_sp) / (1 - 2 * θ_sp)
                                  = 1
```

**RFL-stable update**:
```
Δπ(e) = η * V_corrected(e, o) * ∇_π log π(e)
```

---

## 8. Optimal Policy Under Verifier Imperfection

### 8.1 Problem Formulation

Find policy π that maximizes expected reward under verifier imperfection:

```
max_π E[R(π, θ)]

where:
R(π, θ) = Σ_e π(e) * E[V(e, y) | e, θ]

and expectation is over ground truth y and verifier outcomes o
```

### 8.2 Optimal Policy (Softmax)

Assume softmax policy:

```
π(e) = exp(Q(e) / τ) / Σ_e' exp(Q(e') / τ)

where:
- Q(e) is the Q-value (expected reward) for proof attempt e
- τ is the temperature parameter (controls exploration)
```

**Q-Value Update**:
```
Q(e) ← Q(e) + η * [V_corrected(e, o) - Q(e)]
```

**Convergence**: As t → ∞, Q(e) → E[V(e, y) | e, θ], and π converges to optimal policy.

### 8.3 Exploration-Exploitation Trade-Off

**High temperature (τ → ∞)**: Uniform exploration (π(e) ≈ 1/|E| for all e)  
**Low temperature (τ → 0)**: Greedy exploitation (π(e) ≈ 1 for e = argmax Q(e))

**Optimal temperature**: Balance exploration and exploitation to maximize cumulative reward.

**Adaptive temperature**: Decrease τ over time (simulated annealing):

```
τ_t = τ_0 / (1 + t / t_decay)
```

---

## 9. Implementation

### 9.1 Noise-Robust RFL Update Function

```python
def compute_expected_value(
    outcome: VerifierErrorCode,
    theta_spurious_fail: float,
    theta_spurious_pass: float,
    theta_timeout: float,
) -> float:
    """Compute expected value function under verifier imperfection.
    
    Args:
        outcome: Verifier outcome
        theta_spurious_fail: Spurious failure rate
        theta_spurious_pass: Spurious pass rate
        theta_timeout: Timeout rate
    
    Returns:
        Expected value in [-1, +1]
    """
    
    if outcome == VerifierErrorCode.VERIFIED:
        # P(VALID | VERIFIED)
        p_valid = (1 - theta_spurious_fail - theta_timeout) / \
                  ((1 - theta_spurious_fail - theta_timeout) + theta_spurious_pass)
        return 2 * p_valid - 1
    
    elif outcome == VerifierErrorCode.PROOF_INVALID:
        # P(VALID | FAILED)
        p_valid = theta_spurious_fail / \
                  (theta_spurious_fail + (1 - theta_spurious_pass - theta_timeout))
        return 2 * p_valid - 1
    
    elif outcome == VerifierErrorCode.VERIFIER_TIMEOUT:
        # No information
        return 0.0
    
    else:
        # Abstention or other error
        return 0.0


def update_rfl_policy_noisy(
    policy: Dict[str, float],
    item: str,
    outcome: VerifierErrorCode,
    theta_spurious_fail: float,
    theta_spurious_pass: float,
    theta_timeout: float,
    learning_rate: float,
    abstention_rate: float,
) -> Dict[str, float]:
    """Update RFL policy under verifier imperfection.
    
    Args:
        policy: Current policy (item → probability)
        item: Item that was verified
        outcome: Verifier outcome
        theta_spurious_fail: Spurious failure rate
        theta_spurious_pass: Spurious pass rate
        theta_timeout: Timeout rate
        learning_rate: Base learning rate
        abstention_rate: Fraction of outcomes that result in abstention
    
    Returns:
        Updated policy
    """
    
    # Compute expected value
    v_expected = compute_expected_value(
        outcome,
        theta_spurious_fail,
        theta_spurious_pass,
        theta_timeout,
    )
    
    # Bias correction
    v_corrected = v_expected / (1 - 2 * theta_spurious_pass) if theta_spurious_pass < 0.5 else v_expected
    
    # Adjust learning rate for abstention
    eta_effective = learning_rate / (1 - abstention_rate) if abstention_rate < 1.0 else learning_rate
    
    # Compute policy gradient (simplified: assume log-linear policy)
    grad_log_pi = 1.0 / policy[item] if policy[item] > 0 else 0.0
    
    # Policy update
    delta_pi = eta_effective * v_corrected * grad_log_pi
    
    # Update policy
    policy_new = policy.copy()
    policy_new[item] = max(0.0, policy[item] + delta_pi)
    
    # Normalize
    total = sum(policy_new.values())
    if total > 0:
        policy_new = {k: v / total for k, v in policy_new.items()}
    
    return policy_new
```

### 9.2 Multi-Tier Integration

```python
def update_rfl_policy_multitier(
    policy: Dict[str, float],
    item: str,
    outcomes: List[Tuple[VerifierErrorCode, VerifierTier]],
    tier_noise_rates: Dict[VerifierTier, Dict[str, float]],
    tier_weights: Dict[VerifierTier, float],
    learning_rate: float,
) -> Dict[str, float]:
    """Update RFL policy using multi-tier verifier outcomes.
    
    Args:
        policy: Current policy
        item: Item that was verified
        outcomes: List of (outcome, tier) pairs from escalation
        tier_noise_rates: Noise rates for each tier
        tier_weights: Confidence weights for each tier
        learning_rate: Base learning rate
    
    Returns:
        Updated policy
    """
    
    # Compute weighted expected value
    v_aggregated = 0.0
    total_weight = 0.0
    
    for outcome, tier in outcomes:
        noise_rates = tier_noise_rates[tier]
        v_expected = compute_expected_value(
            outcome,
            noise_rates["spurious_fail"],
            noise_rates["spurious_pass"],
            noise_rates["timeout"],
        )
        
        weight = tier_weights[tier]
        v_aggregated += weight * v_expected
        total_weight += weight
    
    if total_weight > 0:
        v_aggregated /= total_weight
    
    # Bias correction (use final tier's spurious pass rate)
    final_tier = outcomes[-1][1]
    theta_sp = tier_noise_rates[final_tier]["spurious_pass"]
    v_corrected = v_aggregated / (1 - 2 * theta_sp) if theta_sp < 0.5 else v_aggregated
    
    # Policy update (same as single-tier)
    grad_log_pi = 1.0 / policy[item] if policy[item] > 0 else 0.0
    delta_pi = learning_rate * v_corrected * grad_log_pi
    
    policy_new = policy.copy()
    policy_new[item] = max(0.0, policy[item] + delta_pi)
    
    # Normalize
    total = sum(policy_new.values())
    if total > 0:
        policy_new = {k: v / total for k, v in policy_new.items()}
    
    return policy_new
```

---

## 10. Validation

### 10.1 Synthetic Experiments

**Experiment 1: Bias Verification**
- Generate synthetic ground truth labels (50% valid, 50% invalid)
- Apply verifier noise model with known θ_sf, θ_sp, θ_to
- Run RFL with noise-robust updates
- Verify that policy converges to ground truth distribution

**Experiment 2: Convergence Rate**
- Compare convergence rate of noise-robust RFL vs standard RFL
- Measure number of cycles to reach 90% accuracy
- Verify that noise-robust RFL converges faster under high noise

**Experiment 3: Stability**
- Run RFL for 10,000 cycles with fixed ground truth
- Measure policy drift (change in policy over time)
- Verify that noise-robust RFL has lower drift than standard RFL

### 10.2 Real-World Validation

**Experiment 4: Mathlib Proofs**
- Run RFL on Mathlib proof corpus with real Lean verifier
- Compare noise-robust RFL vs standard RFL on success rate
- Verify that noise-robust RFL achieves higher success rate

**Experiment 5: Curriculum Learning**
- Run RFL on curriculum with increasing difficulty
- Measure learning curve (success rate vs cycle)
- Verify that noise-robust RFL learns faster on noisy curriculum

---

## 11. Summary

This document formalizes the RFL update equation under verifier imperfection and derives optimal policies that are robust to noise. Key contributions:

**1. Expected Value Function**: Compute expected value under posterior ground truth probability, down-weighting noisy outcomes.

**2. Abstention Handling**: Abstain from updates on timeouts, adjusting effective learning rate.

**3. Confidence-Weighted Updates**: Scale updates by confidence in verifier outcome.

**4. Multi-Tier Integration**: Aggregate outcomes from multiple tiers, weighted by confidence.

**5. Bias Correction**: Correct bias due to spurious passes, ensuring RFL-stability.

**6. Optimal Policy**: Derive softmax policy with adaptive temperature for exploration-exploitation trade-off.

---

**Manus-C — Telemetry Architect**  
*"Every packet accounted for, every signal explained."*
