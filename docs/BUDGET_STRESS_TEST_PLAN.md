# Budget Stress Test Plan

> **STATUS: PHASE II — NOT RUN IN PHASE I**
>
> This document defines synthetic runtime scenarios and boundary cases for
> stress-testing the verifier budget system. Tests validate correct behavior
> under extreme conditions.

---

## 1. Overview

### 1.1 Purpose

The Budget Stress Test Plan validates that the verifier budget system:

1. **Enforces limits correctly** under all conditions
2. **Produces correct outcomes** at boundary values
3. **Maintains determinism** even under resource pressure
4. **Fails gracefully** when limits are exceeded

### 1.2 Test Categories

| Category | Description | Risk Level |
|----------|-------------|------------|
| Boundary Tests | Values at exact limits | Medium |
| Overflow Tests | Values beyond limits | High |
| Zero-Value Tests | Degenerate configurations | High |
| Contention Tests | Resource competition | Medium |
| Determinism Tests | Reproducibility validation | Critical |

---

## 2. Synthetic Runtime Scenarios

### 2.1 Scenario: Timeout Cascade

**Description:** All candidates time out, exhausting cycle budget via timeouts.

**Configuration:**
```yaml
test_timeout_cascade:
  taut_timeout_s: 0.10
  cycle_budget_s: 1.0
  max_candidates_per_cycle: 20

  # Synthetic oracle: always times out
  oracle_behavior: "always_timeout"
```

**Expected Behavior:**
- First 10 candidates time out (10 × 0.10s = 1.0s)
- Remaining 10 candidates are skipped (budget exhausted)
- All outcomes are `A_timeout` or `S_budget`
- No `V` or `R` outcomes

**Assertions:**
```python
assert all(o in ("A_timeout", "S_budget") for o in outcomes)
assert sum(1 for o in outcomes if o == "A_timeout") == 10
assert sum(1 for o in outcomes if o == "S_budget") == 10
assert attestation.budget_exhausted == True
```

### 2.2 Scenario: Rapid Verification

**Description:** All candidates verify instantly, testing throughput limits.

**Configuration:**
```yaml
test_rapid_verification:
  taut_timeout_s: 0.10
  cycle_budget_s: 5.0
  max_candidates_per_cycle: 1000

  # Synthetic oracle: instant verification
  oracle_behavior: "instant_verify"
  oracle_verify_prob: 0.5
```

**Expected Behavior:**
- All 1000 candidates processed (instant verification << budget)
- ~500 verified, ~500 refuted (probabilistic oracle)
- Cycle completes well under budget
- No timeouts or skips

**Assertions:**
```python
assert len(outcomes) == 1000
assert sum(1 for o in outcomes if o == "V") > 400
assert sum(1 for o in outcomes if o == "R") > 400
assert sum(1 for o in outcomes if o in ("A_timeout", "S_budget")) == 0
assert attestation.budget_exhausted == False
```

### 2.3 Scenario: Mixed Latency

**Description:** Candidates have variable verification times.

**Configuration:**
```yaml
test_mixed_latency:
  taut_timeout_s: 0.10
  cycle_budget_s: 2.0
  max_candidates_per_cycle: 50

  # Synthetic oracle: variable latency
  oracle_behavior: "variable_latency"
  oracle_latency_distribution:
    type: "exponential"
    mean: 0.04  # Most finish quickly
```

**Expected Behavior:**
- Most candidates verify quickly
- Some candidates timeout (tail of distribution)
- Cycle may or may not exhaust budget
- Mix of outcomes

**Assertions:**
```python
assert len(outcomes) <= 50
assert sum(1 for o in outcomes if o == "A_timeout") > 0  # Some timeouts
assert sum(1 for o in outcomes if o in ("V", "R")) > 0   # Some verifications
# Deterministic given seed
assert outcomes == replay_outcomes(same_seed)
```

### 2.4 Scenario: Budget Boundary

**Description:** Cycle budget exhausts exactly at candidate boundary.

**Configuration:**
```yaml
test_budget_boundary:
  taut_timeout_s: 0.10
  cycle_budget_s: 1.0
  max_candidates_per_cycle: 10

  # Synthetic oracle: exactly 0.10s per verification
  oracle_behavior: "fixed_latency"
  oracle_latency_s: 0.10
```

**Expected Behavior:**
- 10 candidates × 0.10s = 1.0s exactly
- All candidates processed (no skip)
- Budget exhausted at cycle end (not mid-cycle)

**Assertions:**
```python
assert len(outcomes) == 10
assert sum(1 for o in outcomes if o == "S_budget") == 0
assert attestation.budget_exhausted == False  # Completed exactly at budget
```

### 2.5 Scenario: Single Slow Candidate

**Description:** One candidate consumes most of the budget.

**Configuration:**
```yaml
test_single_slow:
  taut_timeout_s: 5.0   # High timeout
  cycle_budget_s: 6.0
  max_candidates_per_cycle: 10

  # Synthetic oracle: first candidate is slow
  oracle_behavior: "first_slow"
  oracle_first_latency_s: 5.5
  oracle_rest_latency_s: 0.01
```

**Expected Behavior:**
- First candidate takes 5.5s (within timeout)
- Remaining budget: 0.5s for 9 candidates
- Maybe 1-2 more candidates before budget exhaustion
- Rest are skipped

**Assertions:**
```python
assert outcomes[0] in ("V", "R")  # First completes
assert sum(1 for o in outcomes if o == "S_budget") >= 7
```

---

## 3. Boundary Case Tests

### 3.1 Zero Budget Tests

**Test: Zero Cycle Budget**
```yaml
test_zero_cycle_budget:
  taut_timeout_s: 0.10
  cycle_budget_s: 0.0  # Zero budget
  max_candidates_per_cycle: 10
```

**Expected:** `ValueError` on config load — cycle_budget_s must be positive.

```python
with pytest.raises(ValueError, match="cycle_budget_s must be positive"):
    load_budget_for_slice("test_zero_cycle_budget")
```

**Test: Zero Timeout**
```yaml
test_zero_timeout:
  taut_timeout_s: 0.0  # Zero timeout
  cycle_budget_s: 5.0
  max_candidates_per_cycle: 10
```

**Expected:** `ValueError` on config load — taut_timeout_s must be positive.

**Test: Zero Candidates**
```yaml
test_zero_candidates:
  taut_timeout_s: 0.10
  cycle_budget_s: 5.0
  max_candidates_per_cycle: 0  # Zero candidates
```

**Expected:** `ValueError` on config load — max_candidates_per_cycle must be positive.

### 3.2 Extreme Value Tests

**Test: Tiny Timeout**
```yaml
test_tiny_timeout:
  taut_timeout_s: 0.001  # 1 millisecond
  cycle_budget_s: 5.0
  max_candidates_per_cycle: 100
```

**Expected Behavior:**
- Most candidates timeout (verification takes > 1ms)
- High abstention rate
- Valid outcomes (A_timeout)

**Test: Huge Budget**
```yaml
test_huge_budget:
  taut_timeout_s: 0.10
  cycle_budget_s: 3600.0  # 1 hour
  max_candidates_per_cycle: 100
```

**Expected Behavior:**
- All candidates processed (budget never exhausted)
- Normal verification outcomes
- Config loads successfully (within allowed range)

**Test: Many Candidates**
```yaml
test_many_candidates:
  taut_timeout_s: 0.001
  cycle_budget_s: 60.0
  max_candidates_per_cycle: 10000  # 10K candidates
```

**Expected Behavior:**
- Processes up to 10000 candidates
- Budget may exhaust before completing all
- Memory usage remains bounded

### 3.3 Type Boundary Tests

**Test: Float Precision**
```yaml
test_float_precision:
  taut_timeout_s: 0.000001  # Microsecond precision
  cycle_budget_s: 0.001
  max_candidates_per_cycle: 10
```

**Expected:** Config loads; timer precision may limit actual enforcement.

**Test: Integer Coercion**
```yaml
test_integer_coercion:
  taut_timeout_s: 1  # Integer, should coerce to float
  cycle_budget_s: 5  # Integer, should coerce to float
  max_candidates_per_cycle: 10.0  # Float, should coerce to int
```

**Expected:** All values coerced to correct types.

---

## 4. Contention and Resource Tests

### 4.1 CPU Contention

**Test: High CPU Load**

Simulate CPU contention during verification:

```python
def test_cpu_contention():
    """Verify budget enforcement under CPU load."""
    import multiprocessing
    import time

    # Start CPU-bound workers
    def cpu_burn():
        while True:
            _ = sum(range(10000))

    workers = [multiprocessing.Process(target=cpu_burn) for _ in range(4)]
    for w in workers:
        w.start()

    try:
        # Run verification under load
        budget = load_budget_for_slice("slice_uplift_goal")
        start = time.monotonic()
        result = run_cycle_with_budget(budget, candidates)
        elapsed = time.monotonic() - start

        # Budget should still be enforced
        assert elapsed <= budget.cycle_budget_s * 1.5  # Allow 50% margin
        assert all_outcomes_valid(result.outcomes)
    finally:
        for w in workers:
            w.terminate()
```

### 4.2 Memory Pressure

**Test: Low Memory**

Verify graceful handling under memory pressure:

```python
def test_memory_pressure():
    """Verify behavior under memory pressure."""
    # Allocate memory to create pressure
    memory_hog = bytearray(500 * 1024 * 1024)  # 500MB

    try:
        budget = load_budget_for_slice("slice_uplift_sparse")
        result = run_cycle_with_budget(budget, large_candidate_set)

        # Should complete without OOM
        assert result is not None
        assert len(result.outcomes) > 0
    finally:
        del memory_hog
```

### 4.3 Clock Skew

**Test: Monotonic Clock**

Verify wall-clock jumps don't break budget enforcement:

```python
def test_clock_independence():
    """Budget uses monotonic clock, not wall clock."""
    import time

    # Mock wall clock jump (would break non-monotonic timing)
    # Note: Can't actually change system clock in test
    # Instead, verify monotonic clock is used in implementation

    # Check budget_loader uses time.monotonic() or equivalent
    source = inspect.getsource(run_cycle_with_budget)
    assert "monotonic" in source or "perf_counter" in source
```

---

## 5. Determinism Tests

### 5.1 Seed Reproducibility

**Test: Same Seed Same Results**

```python
@pytest.mark.parametrize("seed", [0, 42, 12345, 2**31 - 1])
def test_seed_reproducibility(seed):
    """Same seed produces identical results."""
    budget = load_budget_for_slice("slice_uplift_goal")

    result1 = run_cycle_with_budget(budget, candidates, seed=seed)
    result2 = run_cycle_with_budget(budget, candidates, seed=seed)

    assert result1.outcomes == result2.outcomes
    assert result1.attestation_hash == result2.attestation_hash
```

### 5.2 Budget Parameter Sensitivity

**Test: Budget Changes Affect Results**

```python
def test_budget_sensitivity():
    """Different budgets can produce different results."""
    tight_budget = VerifierBudget(
        cycle_budget_s=0.5,
        taut_timeout_s=0.10,
        max_candidates_per_cycle=100,
    )

    loose_budget = VerifierBudget(
        cycle_budget_s=60.0,
        taut_timeout_s=0.10,
        max_candidates_per_cycle=100,
    )

    result_tight = run_cycle_with_budget(tight_budget, slow_candidates, seed=42)
    result_loose = run_cycle_with_budget(loose_budget, slow_candidates, seed=42)

    # Tight budget may skip more candidates
    tight_skips = sum(1 for o in result_tight.outcomes if o == "S_budget")
    loose_skips = sum(1 for o in result_loose.outcomes if o == "S_budget")

    assert tight_skips >= loose_skips
```

### 5.3 Cross-Platform Reproducibility

**Test: Platform Independence**

```python
def test_platform_hash_consistency():
    """Attestation hashes are platform-independent."""
    # Pre-computed expected hash (from reference run)
    expected_hash = "a1b2c3..."

    budget = load_budget_for_slice("slice_uplift_goal")
    result = run_cycle_with_budget(budget, fixed_candidates, seed=42)

    # Hash should match across platforms
    assert result.attestation_hash == expected_hash
```

---

## 6. Error Injection Tests

### 6.1 Oracle Crash

**Test: Verifier Crash Handling**

```python
def test_oracle_crash():
    """Verifier crash produces abstention, not propagated error."""
    def crashing_oracle(candidate):
        if "crash_trigger" in candidate:
            raise RuntimeError("Simulated crash")
        return verify_normally(candidate)

    budget = load_budget_for_slice("slice_uplift_goal")
    candidates = ["normal", "crash_trigger", "normal"]

    result = run_cycle_with_oracle(budget, candidates, oracle=crashing_oracle)

    # Crash should produce abstention
    assert result.outcomes[1] == "A_crash"
    # Other candidates should proceed
    assert result.outcomes[0] in ("V", "R")
    assert result.outcomes[2] in ("V", "R")
```

### 6.2 Config Corruption

**Test: Malformed Config Handling**

```python
def test_malformed_config():
    """Malformed config raises clear error."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        f.write(b"not: valid: yaml: {{{}}")

    with pytest.raises(ValueError, match="Malformed YAML"):
        load_budget_for_slice("any_slice", path=f.name)
```

### 6.3 Missing Config

**Test: Missing Config File**

```python
def test_missing_config():
    """Missing config raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="not found"):
        load_budget_for_slice("any_slice", path="/nonexistent/path.yaml")
```

---

## 7. Test Matrix

### 7.1 Parameter Coverage Matrix

| Test | `taut_timeout_s` | `cycle_budget_s` | `max_candidates` |
|------|------------------|------------------|------------------|
| Zero budget | ✓ | 0.0 | ✓ |
| Zero timeout | 0.0 | ✓ | ✓ |
| Zero candidates | ✓ | ✓ | 0 |
| Tiny timeout | 0.001 | ✓ | ✓ |
| Huge budget | ✓ | 3600 | ✓ |
| Many candidates | ✓ | ✓ | 10000 |
| Boundary exact | 0.10 | 1.0 | 10 |

### 7.2 Scenario Coverage Matrix

| Scenario | Timeout | Budget Exhaust | Skip | Determinism |
|----------|---------|----------------|------|-------------|
| Timeout cascade | ✓ | ✓ | ✓ | ✓ |
| Rapid verification | ✗ | ✗ | ✗ | ✓ |
| Mixed latency | ✓ | ? | ? | ✓ |
| Budget boundary | ✗ | ✓ | ✗ | ✓ |
| Single slow | ✗ | ✓ | ✓ | ✓ |

---

## 8. Execution Plan

### 8.1 Pre-Commit Tests

Run on every commit:
- Zero value tests
- Type boundary tests
- Config error tests
- Basic determinism tests

```bash
pytest tests/test_budget_loader.py -m "not slow"
```

### 8.2 Nightly Tests

Run nightly:
- All synthetic scenarios
- Contention tests
- Cross-platform hash verification

```bash
pytest tests/test_budget_stress.py -m "slow or stress"
```

### 8.3 Release Tests

Run before release:
- Full test matrix
- Extended determinism tests
- Performance benchmarks

```bash
pytest tests/test_budget_*.py --benchmark-enable
```

---

## 9. Test File Locations

| Test File | Description |
|-----------|-------------|
| `tests/test_budget_loader.py` | Config loading and validation |
| `tests/test_budget_stress.py` | Stress scenarios (to be created) |
| `tests/test_budget_determinism.py` | Reproducibility tests (to be created) |
| `tests/fixtures/budget_*.yaml` | Test config fixtures |

---

*End of Budget Stress Test Plan.*
