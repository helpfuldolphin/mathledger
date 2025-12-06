# RFL Testable Predictions: Engineering Validation Sheet

**Author**: Claude H (Cognitive Systems Theorist)
**Date**: 2025-11-04
**Purpose**: Translate RFL theory into 1-hour verifiable engineering tests
**Target**: Gate logic for curriculum advancement

---

## Quick Reference: Measurable Thresholds

| Hypothesis ID | Metric | Formula | Pass Threshold | Data Source |
|---------------|--------|---------|----------------|-------------|
| **H1** | Scaling Law | `proofs/sec ~ B·d^(-2.0)·exp(-0.01t)` | R² > 0.80 | `runs` table |
| **H2** | Steady State | `|dM/dt|` | < 5 statements/min | Queue length over time |
| **H3** | Reflexivity | `p→p at depth=0` | 100% runs | `statements` WHERE `depth=0` |
| **H4** | Mass Conservation | `Σ parent_mass = child_mass` | 100% proofs | `proof_parents` JOIN |
| **H5** | Breadth Saturation | `new_statements(B) / B` | < 0.3 | Per-run stats |
| **H6** | Depth Decay | `proofs(d+1) / proofs(d)` | < 0.5 | `statements` GROUP BY depth |
| **H7** | Curriculum Gate | `dM/dt_last_10_runs` | < 2 statements/min | Time-series derivative |
| **H8** | Verification Bound | `success_rate` | > 85% | `proofs.status='success'` |

---

## H1: Scaling Law (Power-Law Decay)

**Prediction**: Proof throughput follows power-law with exponential cooling.

**Formula**:
```
throughput(B, d, t) = k · B · d^(-α) · exp(-β·t)

Expected: α ≈ 2.0, β ≈ 0.01 (per minute)
```

**Test Procedure**:
1. Query: `SELECT run_id, breadth_limit, max_depth, duration_ms, proof_count FROM runs WHERE system_id=1 ORDER BY created_at DESC LIMIT 50`
2. Compute: `proofs_per_sec = proof_count / (duration_ms/1000)`
3. Fit: Log-linear regression on `log(throughput) ~ log(B) + log(d) + t`
4. Extract: α, β coefficients
5. **Pass if**: R² > 0.80

**Required Metrics**: `breadth_limit`, `max_depth`, `proof_count`, `duration_ms`

---

## H2: Steady-State Convergence

**Prediction**: Abstention mass derivative approaches zero at saturation.

**Formula**:
```
M(t) = queue_length(t)
dM/dt ≈ [M(t) - M(t-Δt)] / Δt

Gate: |dM/dt| < 5 statements/min
```

**Test Procedure**:
1. Monitor: Redis queue length every 60 seconds for 10 minutes
2. Compute: Finite differences `ΔM/Δt`
3. **Pass if**: Last 5 samples all < 5 statements/min

**Required Metrics**: `LLEN ml:jobs` (Redis command)

---

## H3: Reflexivity Axiom

**Prediction**: Self-referential proof `p→p` appears in 100% of runs at depth=0.

**Formula**:
```
reflexive_proofs = COUNT(*) WHERE normalized_text LIKE '%->%'
                   AND text = self_implication(var)
                   AND depth = 0

Gate: reflexive_proofs > 0 for ALL runs
```

**Test Procedure**:
1. Query: `SELECT DISTINCT text FROM statements WHERE system_id=1 AND depth=0`
2. Regex match: `^([a-z])->\\1$` (e.g., "p->p", "q->q")
3. **Pass if**: At least one match per run

**Required Metrics**: `statements.text`, `statements.depth`

---

## H4: Proof Mass Conservation

**Prediction**: Total axiom mass is conserved through derivations.

**Formula**:
```
For each proof:
  child_mass = COUNT(DISTINCT axiom_id in DAG path to child)
  parent_mass = SUM(axiom_id counts in parent paths)

Conservation: child_mass = parent_mass (allowing overlaps)
```

**Test Procedure**:
1. Query: `SELECT proof_id, child_hash FROM proof_parents LIMIT 1000`
2. For each: Traverse DAG to axioms, count unique axioms
3. Verify: `mass(child) >= max(mass(parent_i))` (monotonic)
4. **Pass if**: 100% of sampled proofs satisfy inequality

**Required Metrics**: `proof_parents` table

---

## H5: Breadth Saturation

**Prediction**: Marginal new statements per breadth unit decreases (diminishing returns).

**Formula**:
```
efficiency(B) = new_statements / B

Gate: efficiency < 0.3 (less than 30% utilization)
```

**Test Procedure**:
1. Query: `SELECT breadth_limit, new_statement_count FROM runs WHERE system_id=1 ORDER BY created_at DESC LIMIT 20`
2. Compute: `eff = new_statement_count / breadth_limit`
3. **Pass if**: Median efficiency < 0.3

**Required Metrics**: `runs.breadth_limit`, `runs.new_statement_count`

---

## H6: Depth Decay (Exponential Dropoff)

**Prediction**: Statement count decreases exponentially with depth.

**Formula**:
```
count(d+1) / count(d) < 0.5  (at least 50% decay per depth level)
```

**Test Procedure**:
1. Query: `SELECT depth, COUNT(*) as cnt FROM statements WHERE system_id=1 GROUP BY depth ORDER BY depth`
2. Compute: Ratios `cnt[d+1] / cnt[d]`
3. **Pass if**: Mean ratio < 0.5

**Required Metrics**: `statements.depth`

---

## H7: Curriculum Advancement Gate (RFL Core)

**Prediction**: Slice advancement should occur when dM/dt stabilizes.

**Formula**:
```
dM/dt_avg = mean(|M(t_i) - M(t_{i-1})|) over last 10 runs

Gate: dM/dt_avg < 2 statements/min → ADVANCE SLICE
```

**Test Procedure**:
1. Query: Last 10 runs' `new_statement_count` and `duration_ms`
2. Compute: Per-run rate = `new_statements / (duration_ms/60000)`
3. Compute: Standard deviation of rates
4. **Pass if**: StdDev < 2.0

**Required Metrics**: `runs.new_statement_count`, `runs.duration_ms`

**Action**: If passed, increment curriculum slice parameters (see `config/nightly.env`)

---

## H8: Verification Success Bound

**Prediction**: Lean verification maintains >85% success rate.

**Formula**:
```
success_rate = COUNT(status='success') / COUNT(*) WHERE method='lean'

Gate: success_rate > 0.85
```

**Test Procedure**:
1. Query: `SELECT COUNT(*) FILTER (WHERE status='success') as succ, COUNT(*) as total FROM proofs WHERE method='lean'`
2. Compute: `succ / total`
3. **Pass if**: Ratio > 0.85

**Required Metrics**: `proofs.status`, `proofs.method`

---

## Implementation Checklist

**To validate all hypotheses**:
1. ✅ Run `make db-stats` to collect current metrics
2. ✅ Execute SQL queries for H3, H4, H6, H8 (direct validation)
3. ✅ Monitor Redis queue for H2 (1-hour sampling)
4. ✅ Fit regression for H1 (requires Python script with scipy)
5. ✅ Check last 10 runs for H5, H7 (aggregate statistics)

**Estimated Time**: 45-60 minutes for all 8 tests

**Next Step**: If ≥6 tests pass → **APPROVE** curriculum advancement
If <6 tests pass → **ABSTAIN** and tune derivation parameters

---

## Expected Output Format

```json
{
  "hypothesis_results": {
    "H1": {"status": "PASS", "R_squared": 0.87, "alpha": 2.1, "beta": 0.009},
    "H2": {"status": "PASS", "dM_dt_max": 3.2},
    "H3": {"status": "PASS", "reflexive_count": 1},
    "H4": {"status": "PASS", "conservation_violations": 0},
    "H5": {"status": "PASS", "median_efficiency": 0.24},
    "H6": {"status": "PASS", "mean_decay_ratio": 0.43},
    "H7": {"status": "PASS", "rate_stddev": 1.8},
    "H8": {"status": "PASS", "success_rate": 0.91}
  },
  "overall": "PASS",
  "passed_count": 8,
  "recommendation": "ADVANCE_SLICE"
}
```

---

**Status**: RFL predictions operationalized. Ready for engineering validation.
