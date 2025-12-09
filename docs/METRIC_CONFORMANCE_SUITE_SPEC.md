# Metric Conformance Suite Specification

**Version**: 1.0.0
**Status**: NORMATIVE
**Phase**: II — NOT RUN IN PHASE I
**Parent Document**: `docs/METRIC_CORRECTNESS_CONTRACT.md`

---

## 1. Purpose

This specification defines the **Conformance Suite** — the set of tests that MUST exist and MUST pass before any metric implementation can be considered correct. The suite proves that metrics obey the axioms and invariants defined in `METRIC_CORRECTNESS_CONTRACT.md`.

This document does NOT implement tests; it specifies what tests MUST exist.

---

## 2. Conformance Test Taxonomy

### 2.1 Test Naming Convention

```
{METRIC}-T{N}-{INVARIANT}-{PROPERTY}

Examples:
  GOAL-T1-GOAL1-HIT_BOUND
  SPARSE-T3-MON1-ADDITIVE
  CHAIN-T5-CHAIN4-CYCLE_SAFE
```

### 2.2 Test Classification

| Class | Description | Example |
|-------|-------------|---------|
| **BOUND** | Value is bounded by input constraints | `value <= len(targets)` |
| **THRESHOLD** | Success respects threshold semantics | `success <-> value >= T` |
| **DETERMINISM** | Repeated calls yield identical results | 100-iteration replay |
| **ORDER** | Input ordering does not affect output | Permutation test |
| **EMPTY** | Empty input produces defined result | Zero-input baseline |
| **IMPOSSIBLE** | Cannot claim success when impossible | Disjoint sets test |
| **MONOTONIC** | Adding inputs does not decrease value | Subset chain test |
| **BOUNDARY** | Threshold boundary behavior correct | T-1, T, T+1 tests |
| **TYPE** | Return type matches contract | `Tuple[bool, float]` |
| **CYCLE** | Graph cycles do not cause failure | Cyclic graph test |

---

## 3. GOAL_HIT Conformance Tests

### 3.1 Required Tests

| Test ID | Input Class | Expected Property | Expected Outcome | Invariant |
|---------|-------------|-------------------|------------------|-----------|
| GOAL-T1-GOAL1-HIT_BOUND | `targets={t1..tn}`, `verified=[...]` | `value <= len(targets)` | Value never exceeds target count | GOAL-1 |
| GOAL-T2-GOAL2-VERIFIED_BOUND | `verified=[v1..vm]`, `targets={...}` | `value <= len(verified)` | Value never exceeds verified count | GOAL-2 |
| GOAL-T3-GOAL3-EMPTY_TARGETS | `targets={}`, `verified=[...]` | `value == 0.0` | Empty targets yields zero hits | GOAL-3 |
| GOAL-T4-GOAL4-DISJOINT | `verified ∩ targets = {}` | `value == 0.0` | Disjoint sets yield zero hits | GOAL-4 |
| GOAL-T5-GOAL5-THRESHOLD_EQ | `value == min_total_verified` | `success == True` | Exact threshold yields success | GOAL-5 |
| GOAL-T6-GOAL5-THRESHOLD_LT | `value < min_total_verified` | `success == False` | Below threshold yields failure | GOAL-5 |
| GOAL-T7-GOAL5-THRESHOLD_GT | `value > min_total_verified` | `success == True` | Above threshold yields success | GOAL-5 |
| GOAL-T8-GOAL6-DETERMINISM | Same inputs, 100 iterations | All results identical | Deterministic output | GOAL-6, D |
| GOAL-T9-GOAL7-ORDER | Permuted `verified` list | Results identical across permutations | Order-independent | GOAL-7, O |
| GOAL-T10-NC1-EMPTY_VERIFIED | `verified=[]` | `value == 0.0` | Empty verified yields zero | NC-1 |
| GOAL-T11-NC2-IMPOSSIBILITY | More targets than verified | `value <= len(verified)` | Cannot exceed verified count | NC-2 |
| GOAL-T12-NC3-BOUNDARY | T-1, T, T+1 threshold values | Correct success/fail at boundaries | Boundary behavior | NC-3 |
| GOAL-T13-MON1-ADDITIVE | `V1 ⊆ V2` | `value(V1) <= value(V2)` | Adding verified increases value | MON-1 |
| GOAL-T14-TYPE-RETURN | Any valid input | `isinstance(result, tuple)` | Returns `Tuple[bool, float]` | Type |
| GOAL-T15-P-PURITY | Check no side effects | No global state mutation | Pure function | P |

### 3.2 Test Input Templates

```yaml
GOAL-T1-GOAL1-HIT_BOUND:
  inputs:
    verified_statements: [{"hash": "h0"}, ..., {"hash": "h9"}]
    target_hashes: {"h0", "h1", "h2"}
    min_total_verified: 0
  assertion: result[1] <= 3  # len(targets)

GOAL-T8-GOAL6-DETERMINISM:
  inputs:
    verified_statements: [{"hash": f"h{i}"} for i in range(50)]
    target_hashes: {f"h{i}" for i in range(10, 20)}
    min_total_verified: 5
  iterations: 100
  assertion: all(r == results[0] for r in results)

GOAL-T9-GOAL7-ORDER:
  inputs_variants:
    - [{"hash": "a"}, {"hash": "b"}, {"hash": "c"}]
    - [{"hash": "c"}, {"hash": "b"}, {"hash": "a"}]
    - [{"hash": "b"}, {"hash": "a"}, {"hash": "c"}]
  assertion: all(compute(v) == compute(variants[0]) for v in variants)
```

---

## 4. SPARSE_SUCCESS Conformance Tests

### 4.1 Required Tests

| Test ID | Input Class | Expected Property | Expected Outcome | Invariant |
|---------|-------------|-------------------|------------------|-----------|
| SPARSE-T1-SPARSE1-VALUE_EQ | `verified_count=N` | `value == float(N)` | Value equals input count | SPARSE-1 |
| SPARSE-T2-SPARSE2-THRESHOLD_EQ | `verified == min_verified` | `success == True` | Exact threshold succeeds | SPARSE-2 |
| SPARSE-T3-SPARSE2-THRESHOLD_LT | `verified < min_verified` | `success == False` | Below threshold fails | SPARSE-2 |
| SPARSE-T4-SPARSE2-THRESHOLD_GT | `verified > min_verified` | `success == True` | Above threshold succeeds | SPARSE-2 |
| SPARSE-T5-SPARSE3-ATTEMPTED_IGNORED | Varying `attempted_count` | Result unchanged | Attempted is ignored | SPARSE-3 |
| SPARSE-T6-SPARSE4-ZERO_VERIFIED | `verified_count=0` | `value == 0.0` | Zero yields zero | SPARSE-4 |
| SPARSE-T7-SPARSE5-NON_NEGATIVE | Any valid `verified_count` | `value >= 0.0` | Value never negative | SPARSE-5 |
| SPARSE-T8-SPARSE6-DETERMINISM | Same inputs, 100 iterations | All results identical | Deterministic output | SPARSE-6, D |
| SPARSE-T9-NC1-ZERO_INPUT | `verified_count=0`, `min_verified=1` | `success == False` | Zero input baseline | NC-1 |
| SPARSE-T10-NC3-BOUNDARY | T-1, T, T+1 verified counts | Correct success/fail | Boundary behavior | NC-3 |
| SPARSE-T11-MON1-ADDITIVE | `V1 < V2` | `value(V1) < value(V2)` | Increasing count increases value | MON-1 |
| SPARSE-T12-MON2-THRESHOLD | `T1 > T2` | `P[success|T2] >= P[success|T1]` | Lower threshold easier | MON-2 |
| SPARSE-T13-TYPE-RETURN | Any valid input | `isinstance(result, tuple)` | Returns `Tuple[bool, float]` | Type |
| SPARSE-T14-P-PURITY | Check no side effects | No global state mutation | Pure function | P |

### 4.2 Test Input Templates

```yaml
SPARSE-T1-SPARSE1-VALUE_EQ:
  inputs_range: [0, 1, 5, 10, 100, 1000]
  for_each: verified_count
  assertion: result[1] == float(verified_count)

SPARSE-T5-SPARSE3-ATTEMPTED_IGNORED:
  fixed:
    verified_count: 7
    min_verified: 5
  varying:
    attempted_count: [0, 1, 10, 100, 1000, 999999]
  assertion: all(compute(v, a, m) == compute(v, 0, m) for a in varying)
```

---

## 5. CHAIN_SUCCESS Conformance Tests

### 5.1 Required Tests

| Test ID | Input Class | Expected Property | Expected Outcome | Invariant |
|---------|-------------|-------------------|------------------|-----------|
| CHAIN-T1-CHAIN1-VERIFIED_BOUND | `verified=[v1..vn]` | `value <= len(verified)` | Value bounded by verified count | CHAIN-1 |
| CHAIN-T2-CHAIN2-UNVERIFIED_TARGET | `target ∉ verified` | `value == 0.0` | Unverified target yields zero | CHAIN-2 |
| CHAIN-T3-CHAIN3-ISOLATED_TARGET | `target ∈ verified`, `deps={}` | `value == 1.0` | Isolated target yields one | CHAIN-3 |
| CHAIN-T4-CHAIN4-CYCLE_SAFE | Cyclic dependency graph | No RecursionError | Cycle does not crash | CHAIN-4 |
| CHAIN-T5-CHAIN5-THRESHOLD_EQ | `value == min_chain_length` | `success == True` | Exact threshold succeeds | CHAIN-5 |
| CHAIN-T6-CHAIN5-THRESHOLD_LT | `value < min_chain_length` | `success == False` | Below threshold fails | CHAIN-5 |
| CHAIN-T7-CHAIN5-THRESHOLD_GT | `value > min_chain_length` | `success == True` | Above threshold succeeds | CHAIN-5 |
| CHAIN-T8-CHAIN6-LONGEST_PATH | Diamond graph | `value == max_path_length` | Longest path selected | CHAIN-6 |
| CHAIN-T9-CHAIN7-DETERMINISM | Same inputs, 100 iterations | All results identical | Deterministic output | CHAIN-7, D |
| CHAIN-T10-CHAIN8-ORDER | Permuted `verified` list | Results identical across permutations | Order-independent | CHAIN-8, O |
| CHAIN-T11-NC1-EMPTY_VERIFIED | `verified=[]` | `value == 0.0` | Empty verified yields zero | NC-1 |
| CHAIN-T12-NC2-IMPOSSIBILITY | More chain needed than nodes | `success == False` | Cannot exceed node count | NC-2 |
| CHAIN-T13-NC3-BOUNDARY | min_length at T-1, T, T+1 | Correct success/fail | Boundary behavior | NC-3 |
| CHAIN-T14-TYPE-RETURN | Any valid input | `isinstance(result, tuple)` | Returns `Tuple[bool, float]` | Type |
| CHAIN-T15-P-PURITY | Check no side effects | No global state mutation | Pure function | P |
| CHAIN-T16-CHAIN4-DEEP_CYCLE | Self-loop graph | Completes without error | Degenerate cycle safe | CHAIN-4 |
| CHAIN-T17-CHAIN6-BROKEN_CHAIN | Gap in dependency chain | Chain terminates at gap | Correct chain measurement | CHAIN-6 |

### 5.2 Test Input Templates

```yaml
CHAIN-T4-CHAIN4-CYCLE_SAFE:
  inputs:
    verified_statements: [{"hash": "h1"}, {"hash": "h2"}, {"hash": "h3"}]
    dependency_graph:
      h1: ["h3"]
      h2: ["h1"]
      h3: ["h2"]
    chain_target_hash: "h1"
    min_chain_length: 1
  assertion: does_not_raise(RecursionError)

CHAIN-T8-CHAIN6-LONGEST_PATH:
  inputs:
    verified_statements: [{"hash": "h1"}, ..., {"hash": "h4"}]
    dependency_graph:
      h4: ["h2", "h3"]  # Diamond: h4 depends on both h2 and h3
      h2: ["h1"]
      h3: ["h1"]
    chain_target_hash: "h4"
    min_chain_length: 1
  assertion: result[1] == 3.0  # Longest path is h4->h2->h1 or h4->h3->h1
```

---

## 6. MULTI_GOAL_SUCCESS Conformance Tests

### 6.1 Required Tests

| Test ID | Input Class | Expected Property | Expected Outcome | Invariant |
|---------|-------------|-------------------|------------------|-----------|
| MULTI-T1-MULTI1-REQUIRED_BOUND | `required={r1..rn}` | `value <= len(required)` | Value bounded by required count | MULTI-1 |
| MULTI-T2-MULTI2-ALL_MET | `verified ⊇ required` | `success == True` | All goals met yields success | MULTI-2 |
| MULTI-T3-MULTI2-PARTIAL | `verified ∩ required ≠ required` | `success == False` | Partial goals yield failure | MULTI-2 |
| MULTI-T4-MULTI2-NONE_MET | `verified ∩ required = {}` | `success == False`, `value == 0.0` | No goals met | MULTI-2 |
| MULTI-T5-MULTI3-EMPTY_REQUIRED | `required={}` | `success == True`, `value == 0.0` | Empty required is vacuous truth | MULTI-3 |
| MULTI-T6-MULTI4-SUBSET_MONO | `V1 ⊆ V2` | `value(V1) <= value(V2)` | Larger verified, larger value | MULTI-4 |
| MULTI-T7-MULTI5-COUNTING | `verified`, `required` | `value == len(verified ∩ required)` | Correct intersection count | MULTI-5 |
| MULTI-T8-MULTI6-DETERMINISM | Same inputs, 100 iterations | All results identical | Deterministic output | MULTI-6, D |
| MULTI-T9-NC1-EMPTY_VERIFIED | `verified={}`, `required={r1}` | `success == False` | Empty verified baseline | NC-1 |
| MULTI-T10-NC2-IMPOSSIBILITY | `len(verified) < len(required)` | `value <= len(verified)` | Cannot exceed verified | NC-2 |
| MULTI-T11-MON1-ADDITIVE | `V1 ⊆ V2` | `value(V1) <= value(V2)` | Adding verified increases value | MON-1 |
| MULTI-T12-MON3-SUBSET_REQUIRED | `R1 ⊆ R2`, `success(V, R2)` | `success(V, R1)` | Larger required harder | MON-3 |
| MULTI-T13-TYPE-RETURN | Any valid input | `isinstance(result, tuple)` | Returns `Tuple[bool, float]` | Type |
| MULTI-T14-P-PURITY | Check no side effects | No global state mutation | Pure function | P |
| MULTI-T15-O-ORDER | Set operations (inherently unordered) | Consistent results | Order-independent | O |

### 6.2 Test Input Templates

```yaml
MULTI-T5-MULTI3-EMPTY_REQUIRED:
  inputs:
    verified_hashes: {"h1", "h2", "h3"}
    required_goal_hashes: set()
  assertion: result == (True, 0.0)

MULTI-T12-MON3-SUBSET_REQUIRED:
  inputs:
    verified: {"g1", "g2", "g3", "g4", "g5"}
    R1: {"g1", "g2"}
    R2: {"g1", "g2", "g3", "g4", "g5"}
  assertion: if success(V, R2) then success(V, R1)
```

---

## 7. Conformance Coverage Matrix

### 7.1 Invariant → Test Mapping

This matrix ensures **full coverage**: every invariant has at least one test.

| Invariant | Required Tests | Coverage Status |
|-----------|----------------|-----------------|
| **GOAL-1** | GOAL-T1 | REQUIRED |
| **GOAL-2** | GOAL-T2 | REQUIRED |
| **GOAL-3** | GOAL-T3 | REQUIRED |
| **GOAL-4** | GOAL-T4 | REQUIRED |
| **GOAL-5** | GOAL-T5, GOAL-T6, GOAL-T7 | REQUIRED |
| **GOAL-6** | GOAL-T8 | REQUIRED |
| **GOAL-7** | GOAL-T9 | REQUIRED |
| **SPARSE-1** | SPARSE-T1 | REQUIRED |
| **SPARSE-2** | SPARSE-T2, SPARSE-T3, SPARSE-T4 | REQUIRED |
| **SPARSE-3** | SPARSE-T5 | REQUIRED |
| **SPARSE-4** | SPARSE-T6 | REQUIRED |
| **SPARSE-5** | SPARSE-T7 | REQUIRED |
| **SPARSE-6** | SPARSE-T8 | REQUIRED |
| **CHAIN-1** | CHAIN-T1 | REQUIRED |
| **CHAIN-2** | CHAIN-T2 | REQUIRED |
| **CHAIN-3** | CHAIN-T3 | REQUIRED |
| **CHAIN-4** | CHAIN-T4, CHAIN-T16 | REQUIRED |
| **CHAIN-5** | CHAIN-T5, CHAIN-T6, CHAIN-T7 | REQUIRED |
| **CHAIN-6** | CHAIN-T8, CHAIN-T17 | REQUIRED |
| **CHAIN-7** | CHAIN-T9 | REQUIRED |
| **CHAIN-8** | CHAIN-T10 | REQUIRED |
| **MULTI-1** | MULTI-T1 | REQUIRED |
| **MULTI-2** | MULTI-T2, MULTI-T3, MULTI-T4 | REQUIRED |
| **MULTI-3** | MULTI-T5 | REQUIRED |
| **MULTI-4** | MULTI-T6 | REQUIRED |
| **MULTI-5** | MULTI-T7 | REQUIRED |
| **MULTI-6** | MULTI-T8 | REQUIRED |
| **NC-1** | GOAL-T10, SPARSE-T9, CHAIN-T11, MULTI-T9 | REQUIRED |
| **NC-2** | GOAL-T11, CHAIN-T12, MULTI-T10 | REQUIRED |
| **NC-3** | GOAL-T12, SPARSE-T10, CHAIN-T13 | REQUIRED |
| **MON-1** | GOAL-T13, SPARSE-T11, MULTI-T11 | REQUIRED |
| **MON-2** | SPARSE-T12 | REQUIRED |
| **MON-3** | MULTI-T12 | REQUIRED |
| **Axiom D** | GOAL-T8, SPARSE-T8, CHAIN-T9, MULTI-T8 | REQUIRED |
| **Axiom P** | GOAL-T15, SPARSE-T14, CHAIN-T15, MULTI-T14 | REQUIRED |
| **Axiom I** | (Implicit in isolation of test classes) | REQUIRED |
| **Axiom O** | GOAL-T9, CHAIN-T10, MULTI-T15 | REQUIRED |
| **Type Contract** | GOAL-T14, SPARSE-T13, CHAIN-T14, MULTI-T13 | REQUIRED |

### 7.2 Coverage Statistics

| Metric Family | Invariants | Required Tests | Coverage |
|---------------|------------|----------------|----------|
| GOAL_HIT | 7 | 15 | 100% |
| SPARSE_SUCCESS | 6 | 14 | 100% |
| CHAIN_SUCCESS | 8 | 17 | 100% |
| MULTI_GOAL_SUCCESS | 6 | 15 | 100% |
| Cross-Cutting (NC, MON, Axioms) | 9 | 16 | 100% |
| **Total** | **36** | **61** | **100%** |

### 7.3 Coverage Validation Query

```python
def validate_coverage(invariants: Set[str], tests: Dict[str, Set[str]]) -> bool:
    """
    Validate that every invariant has at least one test.

    Args:
        invariants: Set of all invariant IDs (e.g., {"GOAL-1", "SPARSE-1", ...})
        tests: Dict mapping test ID to set of invariants it covers

    Returns:
        True if coverage is complete, False otherwise.
    """
    covered = set()
    for test_id, covers in tests.items():
        covered.update(covers)

    uncovered = invariants - covered
    if uncovered:
        raise CoverageError(f"Uncovered invariants: {uncovered}")
    return True
```

---

## 8. CI Conformance Policy

### 8.1 Conformance Levels

| Level | Name | Description | Required For |
|-------|------|-------------|--------------|
| **L0** | MINIMAL | Type + determinism tests only | Any build |
| **L1** | STANDARD | All invariant tests | PATCH release |
| **L2** | FULL | Standard + monotonicity + boundary | MINOR release |
| **L3** | EXHAUSTIVE | Full + stress tests + golden baseline | MAJOR release |

### 8.2 Test Requirements by Drift Class

| Drift Class | Conformance Level | Additional Requirements |
|-------------|-------------------|------------------------|
| D0 (Cosmetic) | L0 | None |
| D1 (Additive) | L1 | New param tests |
| D2 (Behavioral-Compatible) | L2 | Golden baseline comparison |
| D3 (Behavioral-Breaking) | L3 | Migration tests, governance sign-off |
| D4 (Schema-Breaking) | L3 | Full regression, type migration tests |
| D5 (Semantic-Breaking) | ABORT | Not shippable; requires new experiment |

### 8.3 Shipping Gate Pseudocode

```python
def can_ship_metric(
    drift_class: DriftClass,
    test_results: TestResults,
    golden_baseline: Optional[GoldenBaseline],
    governance_approval: bool = False,
) -> ShipDecision:
    """
    Determine if a metric change can be shipped.

    Returns:
        ShipDecision with (allowed: bool, reason: str, required_actions: List[str])
    """

    # D5 is never shippable
    if drift_class == DriftClass.D5:
        return ShipDecision(
            allowed=False,
            reason="D5 (Semantic-Breaking) changes are not shippable",
            required_actions=["Abort experiment", "File new preregistration"]
        )

    # Determine required conformance level
    required_level = {
        DriftClass.D0: ConformanceLevel.L0,
        DriftClass.D1: ConformanceLevel.L1,
        DriftClass.D2: ConformanceLevel.L2,
        DriftClass.D3: ConformanceLevel.L3,
        DriftClass.D4: ConformanceLevel.L3,
    }[drift_class]

    # Check test results against required level
    if not test_results.meets_level(required_level):
        missing = test_results.missing_for_level(required_level)
        return ShipDecision(
            allowed=False,
            reason=f"Missing conformance tests for level {required_level}",
            required_actions=[f"Implement and pass: {missing}"]
        )

    # Check golden baseline for D2+
    if drift_class >= DriftClass.D2:
        if golden_baseline is None:
            return ShipDecision(
                allowed=False,
                reason="Golden baseline required for D2+ changes",
                required_actions=["Create or update golden baseline"]
            )
        if not test_results.matches_golden_baseline(golden_baseline):
            return ShipDecision(
                allowed=False,
                reason="Golden baseline mismatch",
                required_actions=["Classify as D3+ or fix implementation"]
            )

    # Check governance for D3+
    if drift_class >= DriftClass.D3 and not governance_approval:
        return ShipDecision(
            allowed=False,
            reason="Governance approval required for D3+ changes",
            required_actions=["Obtain governance board approval"]
        )

    # All checks passed
    return ShipDecision(
        allowed=True,
        reason=f"All {required_level} conformance tests passed",
        required_actions=[]
    )
```

### 8.4 CI Workflow Integration

```yaml
# .github/workflows/metric-conformance.yml (specification, not implementation)

name: Metric Conformance Gate

on:
  pull_request:
    paths:
      - 'experiments/slice_success_metrics.py'
      - 'experiments/u2_pipeline.py'

jobs:
  classify-drift:
    # Step 1: Classify the drift class of the change
    outputs:
      drift_class: ${{ steps.classify.outputs.class }}
    steps:
      - id: classify
        run: python scripts/classify_metric_drift.py

  conformance-l0:
    # Minimal: Always required
    needs: classify-drift
    steps:
      - run: pytest tests/test_metric_invariants.py -m "type or determinism"

  conformance-l1:
    # Standard: Required for D1+
    needs: [classify-drift, conformance-l0]
    if: needs.classify-drift.outputs.drift_class >= 'D1'
    steps:
      - run: pytest tests/test_metric_invariants.py

  conformance-l2:
    # Full: Required for D2+
    needs: [classify-drift, conformance-l1]
    if: needs.classify-drift.outputs.drift_class >= 'D2'
    steps:
      - run: pytest tests/test_metric_invariants.py tests/test_metric_monotonicity.py
      - run: python scripts/verify_golden_baseline.py

  conformance-l3:
    # Exhaustive: Required for D3+
    needs: [classify-drift, conformance-l2]
    if: needs.classify-drift.outputs.drift_class >= 'D3'
    steps:
      - run: pytest tests/ -m "metric" --stress
      - run: python scripts/full_regression_check.py

  governance-gate:
    # Human approval gate for D3+
    needs: [classify-drift, conformance-l3]
    if: needs.classify-drift.outputs.drift_class >= 'D3'
    environment: governance-approval
    steps:
      - run: echo "Governance approval obtained"

  ship-decision:
    needs: [classify-drift, conformance-l0, conformance-l1, conformance-l2, conformance-l3, governance-gate]
    if: always()
    steps:
      - run: python scripts/make_ship_decision.py
```

### 8.5 Conformance Level Test Sets

#### L0 — MINIMAL (8 tests)

```
GOAL-T8-GOAL6-DETERMINISM
GOAL-T14-TYPE-RETURN
SPARSE-T8-SPARSE6-DETERMINISM
SPARSE-T13-TYPE-RETURN
CHAIN-T9-CHAIN7-DETERMINISM
CHAIN-T14-TYPE-RETURN
MULTI-T8-MULTI6-DETERMINISM
MULTI-T13-TYPE-RETURN
```

#### L1 — STANDARD (45 tests)

All tests from Section 3-6 excluding stress and boundary exhaustive tests.

#### L2 — FULL (55 tests)

L1 + all monotonicity tests + all NC-3 boundary tests + golden baseline.

#### L3 — EXHAUSTIVE (61+ tests)

L2 + stress tests (1000+ iterations) + edge case exhaustive + migration compatibility.

---

## 9. Golden Baseline Specification

### 9.1 Baseline Structure

```json
{
  "version": "1.0.0-phaseII.u2",
  "generated_at": "2025-XX-XXTXX:XX:XXZ",
  "hash": "sha256:...",
  "test_cases": [
    {
      "test_id": "GOAL-T1-GOAL1-HIT_BOUND",
      "inputs": {
        "verified_statements": [{"hash": "h0"}, {"hash": "h1"}],
        "target_hashes": ["h0", "h1", "h2"],
        "min_total_verified": 1
      },
      "expected_output": [true, 2.0]
    },
    // ... more test cases
  ]
}
```

### 9.2 Baseline Validation

```python
def validate_against_baseline(
    implementation: Callable,
    baseline: GoldenBaseline,
) -> BaselineValidationResult:
    """
    Validate implementation against golden baseline.

    Returns:
        BaselineValidationResult with match status and any mismatches.
    """
    mismatches = []

    for case in baseline.test_cases:
        actual = implementation(**case.inputs)
        expected = tuple(case.expected_output)

        if actual != expected:
            mismatches.append(Mismatch(
                test_id=case.test_id,
                expected=expected,
                actual=actual,
            ))

    return BaselineValidationResult(
        matches=(len(mismatches) == 0),
        mismatch_count=len(mismatches),
        mismatches=mismatches,
    )
```

---

## 10. Conformance Failure Response Protocol

### 10.1 Failure Classification

| Failure Type | Severity | Response |
|--------------|----------|----------|
| Type test failure | BLOCKER | Cannot ship |
| Determinism failure | BLOCKER | Cannot ship |
| Invariant failure | BLOCKER | Cannot ship |
| Boundary test failure | HIGH | Classify drift, may ship with D3 approval |
| Golden baseline mismatch | MEDIUM | Must classify drift class |
| Stress test failure | LOW | Investigate, may ship with note |

### 10.2 Failure Response Pseudocode

```python
def handle_conformance_failure(
    failure: ConformanceFailure,
    current_version: Version,
) -> FailureResponse:
    """
    Handle a conformance test failure.
    """

    if failure.severity == Severity.BLOCKER:
        return FailureResponse(
            action=Action.BLOCK_SHIP,
            message=f"BLOCKER: {failure.test_id} failed. Cannot ship.",
            required_fix="Fix implementation or revert change",
        )

    if failure.severity == Severity.HIGH:
        drift_class = classify_drift_from_failure(failure)
        return FailureResponse(
            action=Action.ESCALATE,
            message=f"HIGH: {failure.test_id} failed. Drift class: {drift_class}",
            required_fix=f"Obtain {drift_class} approval or fix implementation",
        )

    if failure.severity == Severity.MEDIUM:
        return FailureResponse(
            action=Action.REVIEW,
            message=f"MEDIUM: {failure.test_id} failed. Review required.",
            required_fix="Update golden baseline or fix implementation",
        )

    return FailureResponse(
        action=Action.NOTE,
        message=f"LOW: {failure.test_id} failed. Noted for investigation.",
        required_fix="Investigate in next cycle",
    )
```

---

## 11. Appendix: Test Implementation Checklist

Before claiming conformance, verify:

- [ ] All tests from Section 3 (GOAL_HIT) implemented
- [ ] All tests from Section 4 (SPARSE_SUCCESS) implemented
- [ ] All tests from Section 5 (CHAIN_SUCCESS) implemented
- [ ] All tests from Section 6 (MULTI_GOAL_SUCCESS) implemented
- [ ] Coverage matrix (Section 7) shows 100% coverage
- [ ] Golden baseline created and committed
- [ ] CI workflow configured per Section 8.4
- [ ] Failure response protocol documented

---

## 12. Appendix: Quick Reference

```
┌────────────────────────────────────────────────────────────────┐
│              CONFORMANCE SUITE QUICK REFERENCE                  │
├────────────────────────────────────────────────────────────────┤
│ LEVELS:    L0=Minimal  L1=Standard  L2=Full  L3=Exhaustive     │
├────────────────────────────────────────────────────────────────┤
│ BY DRIFT:  D0=L0  D1=L1  D2=L2  D3/D4=L3  D5=ABORT             │
├────────────────────────────────────────────────────────────────┤
│ TESTS:     61 total across 4 metric families                   │
│            GOAL=15  SPARSE=14  CHAIN=17  MULTI=15               │
├────────────────────────────────────────────────────────────────┤
│ COVERAGE:  36 invariants, 100% required                        │
├────────────────────────────────────────────────────────────────┤
│ GATE:      can_ship_metric(drift, tests, baseline, approval)   │
└────────────────────────────────────────────────────────────────┘
```
