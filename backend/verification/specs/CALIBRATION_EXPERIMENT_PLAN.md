# Calibration Experiment Plan

**Author**: Manus-C (Telemetry Architect)  
**Date**: 2025-12-06  
**Version**: 1.0  
**Status**: Test-Ready

---

## Overview

This document specifies the **complete procedure** for collecting 10,000 noise calibration samples across three verifier tiers (FAST, BALANCED, SLOW). It defines expected distributions, acceptance envelopes, and validation criteria.

---

## 1. Experiment Objectives

### 1.1 Primary Objectives

1. **Estimate Noise Rates**: Measure empirical timeout, spurious fail, and spurious pass rates for each tier
2. **Fit Timeout Distributions**: Determine best-fit distribution (exponential, gamma, lognormal, Weibull) for timeout durations
3. **Validate Tier Monotonicity**: Confirm `timeout_rate(FAST) ≥ timeout_rate(BALANCED) ≥ timeout_rate(SLOW)`
4. **Establish Baseline**: Create calibrated noise models for production use

### 1.2 Secondary Objectives

1. **Resource Profiling**: Measure CPU, memory, and I/O usage across tiers
2. **Tactic Analysis**: Identify common tactics and their correlation with timeouts
3. **Error Taxonomy**: Validate error code coverage and mapping accuracy
4. **Performance Benchmarking**: Measure verification throughput and latency

---

## 2. Experiment Design

### 2.1 Sample Size Justification

**Target**: 10,000 samples per tier (30,000 total)

**Statistical Power**:
- For rate estimation with 95% confidence and ±1% margin of error:
  - Required n ≈ 9,604 (for p = 0.5, worst case)
  - Our n = 10,000 provides adequate power

- For distribution fitting with AIC:
  - Minimum n ≈ 1,000 for reliable parameter estimation
  - Our n = 10,000 provides high confidence

### 2.2 Tier Configuration

| Tier | Timeout (s) | Expected Timeout Rate | Expected Fail Rate | Expected Pass Rate |
|------|-------------|----------------------|-------------------|-------------------|
| **FAST_NOISY** | 30 | 15-25% | 8-12% | 3-5% |
| **BALANCED** | 60 | 8-12% | 4-6% | 1-3% |
| **SLOW_PRECISE** | 120 | 3-7% | 1-3% | 0.5-1.5% |

**Tier Monotonicity Invariant**: `timeout_rate(FAST) > timeout_rate(BALANCED) > timeout_rate(SLOW)`

### 2.3 Module Selection

**Strategy**: Stratified random sampling from MathLib

**Strata**:
1. **Simple modules** (30%): Basic definitions, simple lemmas (e.g., `Data.Nat.Basic`)
2. **Medium modules** (50%): Standard theorems, moderate complexity (e.g., `Algebra.Group.Defs`)
3. **Complex modules** (20%): Advanced theorems, high complexity (e.g., `Analysis.Calculus.Deriv`)

**Sampling Procedure**:
1. Enumerate all MathLib modules
2. Classify by complexity (using heuristics: file size, import depth, theorem count)
3. Sample 10,000 modules per tier using stratified random sampling
4. Ensure no module is sampled more than once per tier

### 2.4 Execution Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Workers** | 16-32 | Maximize throughput on multi-core systems |
| **Master Seed** | 42 | Fixed seed for reproducibility |
| **Noise Injection** | Disabled | Calibration measures real verifier behavior |
| **Retry Logic** | Disabled | Measure raw timeout rate without retries |
| **Lean Version** | 4.3.0 (or latest stable) | Fixed version for consistency |

---

## 3. Expected Distributions

### 3.1 Timeout Rate Distribution

**Model**: Binomial distribution with tier-specific rate θ_timeout

**Expected Parameters**:
- **FAST**: θ_timeout ∈ [0.15, 0.25]
- **BALANCED**: θ_timeout ∈ [0.08, 0.12]
- **SLOW**: θ_timeout ∈ [0.03, 0.07]

**Validation**: Wilson score confidence interval should overlap with expected range

### 3.2 Timeout Duration Distribution

**Candidate Models**:
1. **Exponential**: `f(x) = λ exp(-λx)` — Memoryless, constant hazard rate
2. **Gamma**: `f(x) = (λ^k / Γ(k)) x^(k-1) exp(-λx)` — Erlang-type, shape parameter k
3. **Lognormal**: `f(x) = (1 / (x σ √(2π))) exp(-(ln x - μ)² / (2σ²))` — Heavy right tail
4. **Weibull**: `f(x) = (k/λ) (x/λ)^(k-1) exp(-(x/λ)^k)` — Flexible shape

**Expected Best Fit**: Lognormal or Weibull (heavy right tail due to pathological proofs)

**Model Selection**: Akaike Information Criterion (AIC)
```
AIC = 2k - 2 ln(L)
```
where k = number of parameters, L = likelihood

**Acceptance Criterion**: Best model has AIC at least 10 lower than next-best model

### 3.3 Resource Usage Distribution

**CPU Time**:
- **Expected**: Lognormal distribution (most proofs fast, some very slow)
- **Mean**: 80-90% of wall-clock duration (high CPU utilization)
- **Tail**: Heavy tail for complex proofs

**Memory Usage**:
- **Expected**: Gamma or lognormal distribution
- **Peak**: 100-500 MB for typical proofs, up to 2-4 GB for complex proofs
- **Leak**: `memory_final / memory_peak` should be > 0.9 (minimal leakage)

**Tactic Count**:
- **Expected**: Poisson or negative binomial distribution
- **Mean**: 10-30 tactics per proof
- **Correlation**: Positive correlation with duration and memory

---

## 4. Acceptance Envelopes

### 4.1 Timeout Rate Envelope

**Definition**: Acceptable range for empirical timeout rate

| Tier | Lower Bound | Upper Bound | Tolerance |
|------|-------------|-------------|-----------|
| **FAST** | 0.15 | 0.25 | ±2% |
| **BALANCED** | 0.08 | 0.12 | ±1.5% |
| **SLOW** | 0.03 | 0.07 | ±1% |

**Validation**:
```python
def validate_timeout_rate(tier: str, empirical_rate: float, ci: Tuple[float, float]) -> bool:
    lower, upper = ACCEPTANCE_ENVELOPE[tier]
    ci_lower, ci_upper = ci
    
    # Check if confidence interval overlaps with acceptance envelope
    return (ci_lower <= upper) and (ci_upper >= lower)
```

**Failure Action**: If validation fails, investigate:
1. Lean version mismatch
2. Module selection bias
3. System resource constraints
4. Timeout configuration error

### 4.2 Tier Monotonicity Envelope

**Invariant**: `timeout_rate(FAST) > timeout_rate(BALANCED) > timeout_rate(SLOW)`

**Validation**: Two-proportion z-test with α = 0.05

```python
def validate_tier_monotonicity(
    rate_fast: float, n_fast: int,
    rate_balanced: float, n_balanced: int,
    rate_slow: float, n_slow: int,
) -> bool:
    # Test FAST > BALANCED
    z_fb, p_fb = two_proportion_z_test(rate_fast, n_fast, rate_balanced, n_balanced)
    assert p_fb < 0.05 and rate_fast > rate_balanced, "FAST <= BALANCED"
    
    # Test BALANCED > SLOW
    z_bs, p_bs = two_proportion_z_test(rate_balanced, n_balanced, rate_slow, n_slow)
    assert p_bs < 0.05 and rate_balanced > rate_slow, "BALANCED <= SLOW"
    
    return True
```

**Failure Action**: If monotonicity fails, investigate:
1. Timeout configuration (verify FAST < BALANCED < SLOW)
2. Module selection (ensure same modules across tiers)
3. System load (ensure consistent resource availability)

### 4.3 Distribution Goodness-of-Fit Envelope

**Criterion**: Best-fit distribution should have:
1. **AIC Margin**: AIC(best) < AIC(next_best) - 10
2. **Kolmogorov-Smirnov Test**: p-value > 0.05 (fail to reject H0: data follows distribution)
3. **Chi-Square Test**: p-value > 0.05 (for binned data)

**Validation**:
```python
def validate_distribution_fit(
    data: np.ndarray,
    best_dist: str,
    best_params: Tuple,
    best_aic: float,
    next_best_aic: float,
) -> bool:
    # Check AIC margin
    assert best_aic < next_best_aic - 10, f"AIC margin too small: {next_best_aic - best_aic}"
    
    # Kolmogorov-Smirnov test
    dist = getattr(scipy.stats, best_dist)
    ks_stat, ks_p = scipy.stats.kstest(data, lambda x: dist.cdf(x, *best_params))
    assert ks_p > 0.05, f"KS test failed: p={ks_p}"
    
    return True
```

**Failure Action**: If goodness-of-fit fails:
1. Try additional distributions (Pareto, mixture models)
2. Check for outliers or data quality issues
3. Consider non-parametric models

### 4.4 Resource Usage Envelope

**CPU Utilization**:
- **Expected**: 70-95% (CPU time / wall-clock time)
- **Acceptance**: 60-100%
- **Failure**: < 60% suggests I/O bottleneck or system contention

**Memory Efficiency**:
- **Expected**: `memory_final / memory_peak` > 0.85
- **Acceptance**: > 0.75
- **Failure**: < 0.75 suggests memory leaks

**Tactic Count**:
- **Expected**: Mean 10-30, median 15-25
- **Acceptance**: Mean 5-50, median 10-40
- **Failure**: Outside range suggests tactic parser error

---

## 5. Execution Procedure

### 5.1 Pre-Experiment Checklist

- [ ] **Environment Setup**
  - [ ] Lean 4.3.0 (or latest stable) installed and accessible
  - [ ] MathLib cloned and built
  - [ ] Telemetry runtime tested and validated
  - [ ] Calibration CLI tested on 100 samples
  - [ ] System resources available (16-32 cores, 64+ GB RAM)

- [ ] **Module Selection**
  - [ ] MathLib modules enumerated (expected: 5,000-10,000 modules)
  - [ ] Complexity classification complete
  - [ ] Stratified sample drawn (10,000 modules per tier)
  - [ ] Module list saved to `calibration_modules.txt`

- [ ] **Configuration**
  - [ ] Tier timeouts configured (FAST=30s, BALANCED=60s, SLOW=120s)
  - [ ] Master seed set (42)
  - [ ] Noise injection disabled
  - [ ] Worker count set (16-32)
  - [ ] Output directory created (`calibration_output/`)

### 5.2 Execution Commands

#### Step 1: FAST Tier Calibration

```bash
python -m backend.verification.calibration.calibrate_noise \
    --tiers FAST \
    --n 10000 \
    --workers 32 \
    --modules calibration_modules.txt \
    --seed 42 \
    --export calibration_output/calibration_fast.yaml \
    > calibration_output/calibration_fast.log 2>&1
```

**Expected Duration**: 8-12 hours (30s timeout × 10,000 samples / 32 workers)

#### Step 2: BALANCED Tier Calibration

```bash
python -m backend.verification.calibration.calibrate_noise \
    --tiers BALANCED \
    --n 10000 \
    --workers 32 \
    --modules calibration_modules.txt \
    --seed 42 \
    --export calibration_output/calibration_balanced.yaml \
    > calibration_output/calibration_balanced.log 2>&1
```

**Expected Duration**: 16-24 hours (60s timeout × 10,000 samples / 32 workers)

#### Step 3: SLOW Tier Calibration

```bash
python -m backend.verification.calibration.calibrate_noise \
    --tiers SLOW \
    --n 10000 \
    --workers 32 \
    --modules calibration_modules.txt \
    --seed 42 \
    --export calibration_output/calibration_slow.yaml \
    > calibration_output/calibration_slow.log 2>&1
```

**Expected Duration**: 32-48 hours (120s timeout × 10,000 samples / 32 workers)

#### Step 4: Combined Calibration (All Tiers)

```bash
python -m backend.verification.calibration.calibrate_noise \
    --tiers FAST BALANCED SLOW \
    --n 10000 \
    --workers 32 \
    --modules calibration_modules.txt \
    --seed 42 \
    --export calibration_output/calibration_all_tiers.yaml \
    > calibration_output/calibration_all_tiers.log 2>&1
```

**Expected Duration**: 56-84 hours (total across all tiers)

**Recommendation**: Run tiers sequentially to validate FAST before investing time in SLOW.

### 5.3 Monitoring During Execution

**Real-Time Metrics** (check every 1-2 hours):
1. **Progress**: Number of samples completed (from log file)
2. **Throughput**: Samples per second (should be 0.5-2.0 for FAST, 0.2-1.0 for BALANCED, 0.1-0.5 for SLOW)
3. **Error Rate**: Fraction of internal errors (should be < 1%)
4. **System Load**: CPU utilization (should be 80-95%), memory usage (should be < 80% of total)

**Monitoring Commands**:
```bash
# Progress
tail -f calibration_output/calibration_fast.log | grep "Completed"

# Throughput
grep "Throughput" calibration_output/calibration_fast.log | tail -1

# Error rate
grep "VERIFIER_INTERNAL_ERROR" calibration_raw_fast.jsonl | wc -l
```

**Intervention Criteria**:
- **Throughput < 50% of expected**: Investigate system bottleneck (CPU, I/O, network)
- **Error rate > 5%**: Stop experiment, investigate Lean configuration
- **Memory usage > 90%**: Reduce worker count or increase system memory

---

## 6. Post-Experiment Validation

### 6.1 Data Quality Checks

**Check 1: Sample Count**
```bash
wc -l calibration_raw_fast.jsonl
# Expected: 10000
```

**Check 2: Schema Validation**
```python
python -m backend.verification.validation.validate_schema \
    calibration_raw_fast.jsonl
# Expected: 0 errors
```

**Check 3: Completeness**
```python
python -m backend.verification.validation.check_completeness \
    calibration_raw_fast.jsonl
# Expected: All required fields present
```

### 6.2 Statistical Validation

**Validation Script**: `backend/verification/validation/validate_calibration.py`

```python
def validate_calibration(calibration_yaml: Path) -> bool:
    with open(calibration_yaml) as f:
        data = yaml.safe_load(f)
    
    for tier, model in data["noise_models"].items():
        # Validate timeout rate
        rate = model["timeout_rate"]
        ci = model["timeout_rate_ci"]
        assert validate_timeout_rate(tier, rate, ci), f"{tier} timeout rate out of envelope"
        
        # Validate distribution fit
        dist = model["timeout_distribution"]
        assert dist["aic"] < 1000, f"{tier} AIC too high"
        
        # Validate sample count
        assert model["n_samples"] == 10000, f"{tier} sample count mismatch"
    
    # Validate tier monotonicity
    fast_rate = data["noise_models"]["fast"]["timeout_rate"]
    balanced_rate = data["noise_models"]["balanced"]["timeout_rate"]
    slow_rate = data["noise_models"]["slow"]["timeout_rate"]
    
    assert fast_rate > balanced_rate > slow_rate, "Tier monotonicity violated"
    
    return True
```

**Execution**:
```bash
python -m backend.verification.validation.validate_calibration \
    calibration_output/calibration_all_tiers.yaml
```

### 6.3 Acceptance Criteria

**Pass Criteria** (all must be true):
1. ✅ Sample count = 10,000 per tier
2. ✅ Schema validation passes (0 errors)
3. ✅ Timeout rates within acceptance envelope
4. ✅ Tier monotonicity validated (p < 0.05)
5. ✅ Distribution fit validated (AIC margin > 10, KS p > 0.05)
6. ✅ Resource usage within envelope (CPU 60-100%, memory efficiency > 0.75)
7. ✅ Error rate < 5%

**Conditional Pass** (requires investigation but may proceed):
- Timeout rate at edge of envelope (within 0.5% of boundary)
- AIC margin 5-10 (acceptable but not ideal)
- Resource efficiency at lower bound (60% CPU, 0.75 memory efficiency)

**Fail Criteria** (requires re-run):
- Sample count < 9,500 per tier
- Schema validation errors > 1%
- Timeout rate outside envelope
- Tier monotonicity violated
- Distribution fit fails (AIC margin < 5 or KS p < 0.01)
- Error rate > 10%

---

## 7. Output Artifacts

### 7.1 Raw Telemetry

**Files**:
- `calibration_raw_fast.jsonl` (10,000 lines)
- `calibration_raw_balanced.jsonl` (10,000 lines)
- `calibration_raw_slow.jsonl` (10,000 lines)

**Format**: JSONL (see Telemetry to Evidence Interface Spec)

**Storage**: Persistent storage for future analysis and replay

### 7.2 Calibrated Noise Models

**File**: `calibration_all_tiers.yaml`

**Format**:
```yaml
metadata:
  timestamp: 1733529600.0
  n_samples_per_tier: 10000
  seed: 42
  tiers: [FAST, BALANCED, SLOW]
  lean_version: "4.3.0"

noise_models:
  fast:
    timeout_rate: 0.20
    timeout_rate_ci: [0.19, 0.21]
    spurious_fail_rate: 0.0  # Not measured in calibration
    spurious_pass_rate: 0.0  # Not measured in calibration
    timeout_distribution:
      distribution: lognormal
      parameters: [3.5, 0.8]
      aic: 45678.9
      mean: 15000.0
      std: 12000.0
      median: 12000.0
    n_samples: 10000
    outcome_counts:
      verified: 7500
      verifier_timeout: 2000
      proof_invalid: 400
      memory_limit_exceeded: 100

  balanced:
    timeout_rate: 0.10
    timeout_rate_ci: [0.09, 0.11]
    spurious_fail_rate: 0.0
    spurious_pass_rate: 0.0
    timeout_distribution:
      distribution: weibull
      parameters: [1.2, 45000.0]
      aic: 56789.0
      mean: 40000.0
      std: 35000.0
      median: 35000.0
    n_samples: 10000
    outcome_counts:
      verified: 8700
      verifier_timeout: 1000
      proof_invalid: 250
      memory_limit_exceeded: 50

  slow:
    timeout_rate: 0.05
    timeout_rate_ci: [0.04, 0.06]
    spurious_fail_rate: 0.0
    spurious_pass_rate: 0.0
    timeout_distribution:
      distribution: gamma
      parameters: [2.0, 0.00002]
      aic: 67890.1
      mean: 100000.0
      std: 70000.0
      median: 85000.0
    n_samples: 10000
    outcome_counts:
      verified: 9300
      verifier_timeout: 500
      proof_invalid: 180
      memory_limit_exceeded: 20
```

### 7.3 Validation Report

**File**: `calibration_validation_report.md`

**Contents**:
1. Executive summary (pass/fail, key findings)
2. Timeout rate analysis (per-tier rates, confidence intervals, envelope validation)
3. Tier monotonicity analysis (z-test results, p-values)
4. Distribution fitting analysis (best-fit distributions, AIC comparison, goodness-of-fit tests)
5. Resource usage analysis (CPU, memory, I/O statistics)
6. Tactic analysis (most common tactics, correlation with timeouts)
7. Error taxonomy coverage (error code distribution)
8. Recommendations (production deployment, parameter tuning, future calibration)

### 7.4 Visualization Artifacts

**Files**:
- `timeout_rate_comparison.png` — Bar chart of timeout rates by tier with confidence intervals
- `timeout_duration_histograms.png` — Histograms of timeout durations with fitted distributions
- `resource_usage_boxplots.png` — Box plots of CPU, memory by tier
- `tactic_frequency_heatmap.png` — Heatmap of tactic usage by outcome
- `outcome_distribution_pie.png` — Pie charts of outcome distribution by tier

---

## 8. Troubleshooting

### 8.1 Common Issues

**Issue 1: Low Throughput**

**Symptoms**: Samples/second < 50% of expected

**Diagnosis**:
```bash
# Check CPU utilization
top -b -n 1 | grep "Cpu(s)"
# Expected: > 80%

# Check I/O wait
iostat -x 1 10
# Expected: %iowait < 20%

# Check network latency (if Lean is remote)
ping <lean_server>
# Expected: < 10ms
```

**Solutions**:
- Increase worker count if CPU < 80%
- Reduce worker count if I/O wait > 20%
- Check Lean installation (should be local, not remote)
- Check system load from other processes

**Issue 2: High Error Rate**

**Symptoms**: VERIFIER_INTERNAL_ERROR > 5%

**Diagnosis**:
```bash
# Extract error messages
grep "VERIFIER_INTERNAL_ERROR" calibration_raw_fast.jsonl | \
    jq -r '.stderr' | sort | uniq -c | sort -rn | head -10
```

**Solutions**:
- Check Lean version compatibility
- Check MathLib build status
- Check system resource limits (ulimit -a)
- Reduce worker count to avoid resource contention

**Issue 3: Tier Monotonicity Violation**

**Symptoms**: timeout_rate(FAST) ≤ timeout_rate(BALANCED) or timeout_rate(BALANCED) ≤ timeout_rate(SLOW)

**Diagnosis**:
```bash
# Check tier configuration
grep "timeout_s" calibration_output/*.yaml
# Expected: FAST < BALANCED < SLOW
```

**Solutions**:
- Verify timeout configuration in code
- Ensure same modules used across tiers
- Check for system load variation during experiments
- Re-run with fixed configuration

---

## 9. Post-Calibration Actions

### 9.1 Production Deployment

1. **Copy Calibrated Models**:
   ```bash
   cp calibration_output/calibration_all_tiers.yaml config/noise_production.yaml
   ```

2. **Update U2 Configuration**:
   ```python
   u2_config = U2Config(
       noise_config_path=Path("config/noise_production.yaml"),
       enable_noise_injection=True,
   )
   ```

3. **Run Validation Experiment**:
   ```bash
   python experiments/run_uplift_u2.py \
       --slice validation_slice \
       --mode rfl \
       --cycles 100 \
       --seed 12345
   ```

4. **Monitor Drift Radar**:
   - Deploy drift radar with calibrated baseline rates
   - Set CUSUM μ_0 to calibrated timeout rates
   - Monitor for 1 week and adjust thresholds

### 9.2 Continuous Calibration

**Frequency**: Quarterly or after major Lean version updates

**Procedure**:
1. Run calibration with 1,000 samples per tier (10% of original)
2. Compare to previous calibration
3. If drift > 20%, run full 10,000-sample calibration
4. Update production noise models

---

## 10. Summary

This calibration experiment plan provides a **complete, test-ready procedure** for collecting 10,000 noise samples per tier:

✅ **Experiment Design**: 10,000 samples per tier, stratified module sampling, 16-32 workers  
✅ **Expected Distributions**: Timeout rates, timeout durations, resource usage, tactic counts  
✅ **Acceptance Envelopes**: Timeout rate, tier monotonicity, distribution fit, resource usage  
✅ **Execution Procedure**: Pre-experiment checklist, execution commands, monitoring  
✅ **Validation**: Data quality checks, statistical validation, acceptance criteria  
✅ **Output Artifacts**: Raw telemetry, calibrated models, validation report, visualizations  
✅ **Troubleshooting**: Common issues and solutions  
✅ **Post-Calibration**: Production deployment, continuous calibration

**Status**: Test-ready. Ready for execution.

---

**Manus-C — Telemetry Architect**  
*"Every packet accounted for, every signal explained."*
