# Curriculum Gate Equations

> **STATUS: PHASE II SPECIFICATION — NOT EXERCISED IN EVIDENCE PACK v1**
>
> This document describes the formal curriculum gating framework. However:
> - **Phase I (Evidence Pack v1)** uses only the existing PL slices in `config/curriculum.yaml`
> - The First Organism closed-loop test does NOT exercise curriculum ratcheting
> - Wide Slice mode (Section 4) is **not activated** in any Phase I runs
> - The 1000-cycle Dyno Chart experiments use fixed slice parameters, not dynamic gating
>
> **What IS proven in Phase I:**
> - Gate evaluation logic exists and passes unit tests (`tests/frontier/test_curriculum_gates.py`)
> - The four gate predicates are implemented in `backend/frontier/curriculum.py`
>
> **What is NOT proven in Phase I:**
> - Actual curriculum advancement via `activate_next_slice()`
> - Wide Slice entropy production claims
> - CI unbiasedness under real (non-synthetic) derivation data

This document formalizes the four curriculum gates used in MathLedger's Reflexive Formal Learning (RFL) system. Each gate enforces deterministic progression conditions that must be satisfied before the curriculum ratchet advances to the next slice.

## 1. Gate Definitions

### 1.1 Coverage Gate (G₁)

The coverage gate ensures the prover demonstrates sufficient success rate with statistical confidence.

**Formal Definition:**

```
G₁(m, θ) = 1  ⟺  (CI_lower(m) ≥ θ.ci_lower_min) ∧
                  (n(m) ≥ θ.sample_min) ∧
                  (θ.require_attestation → H(m) ≠ ∅)
```

Where:
- `CI_lower(m)` = Lower bound of coverage confidence interval from metrics `m`
- `n(m)` = Sample size used in coverage estimation
- `H(m)` = Attestation (Merkle) hash from provenance
- `θ` = Gate threshold specification (CoverageGateSpec)

**Inequality Form:**
```
CI_lower ≥ θ_cov    where θ_cov ∈ (0, 1]
n ≥ n_min           where n_min ∈ ℤ⁺
```

### 1.2 Abstention Gate (G₂)

The abstention gate bounds both the rate and absolute mass of prover abstentions (timeouts, failures, skips).

**Formal Definition:**

```
G₂(m, θ) = 1  ⟺  (r_abs(m) ≤ θ.max_rate_pct) ∧
                  (M_abs(m) ≤ θ.max_mass)
```

Where:
- `r_abs(m)` = Abstention rate as percentage: `(abstentions / attempts) × 100`
- `M_abs(m)` = Abstention mass: `(r_abs / 100) × attempt_mass`
- `θ` = Gate threshold specification (AbstentionGateSpec)

**Inequality Form:**
```
r_abs ≤ r_max       where r_max ∈ [0, 100]
M_abs ≤ M_max       where M_max ∈ ℤ⁺
```

**Derived Constraint:**
```
M_abs = (r_abs / 100) × A
```
where `A` = total attempt mass.

### 1.3 Velocity Gate (G₃)

The velocity gate ensures sustained proof throughput with bounded variance.

**Formal Definition:**

```
G₃(m, θ) = 1  ⟺  (v(m) ≥ θ.min_pph) ∧
                  (CV(m) ≤ θ.stability_cv_max)
```

Where:
- `v(m)` = Proof velocity in proofs per hour (pph)
- `CV(m)` = Coefficient of variation of velocity over window
- `θ.window_minutes` = Measurement window (informational, not gating)

**Inequality Form:**
```
v ≥ v_min           where v_min > 0
CV ≤ CV_max         where CV_max ∈ [0, 1]
```

**Stability Interpretation:**
The CV constraint ensures velocity is not just high on average but stable:
```
CV = σ_v / μ_v ≤ CV_max
```

### 1.4 Caps Gate (G₄)

The caps gate enforces minimum exploration effort and bounds queue congestion.

**Formal Definition:**

```
G₄(m, θ, Π) = 1  ⟺  (A(m) ≥ θ.min_attempt_mass) ∧
                     (A(m) ≤ Π.total_max) ∧
                     (T(m) ≥ θ.min_runtime_minutes) ∧
                     (B(m) ≤ θ.backlog_max)
```

Where:
- `A(m)` = Attempt mass (total derivation attempts in slice)
- `T(m)` = Wallclock runtime in minutes
- `B(m)` = Queue backlog fraction ∈ [0, 1]
- `Π` = Slice parameters (includes `total_max` cap)
- `θ` = Gate threshold specification (CapsGateSpec)

**Inequality Form:**
```
A_min ≤ A ≤ A_cap   where A_min, A_cap ∈ ℤ⁺
T ≥ T_min           where T_min > 0
B ≤ B_max           where B_max ∈ [0, 1]
```

---

## 2. Composite Advancement Predicate

The ratchet advances if and only if all four gates pass:

```
ADVANCE(m, S) = G₁(m, S.gates.coverage) ∧
                G₂(m, S.gates.abstention) ∧
                G₃(m, S.gates.velocity) ∧
                G₄(m, S.gates.caps, S.params)
```

Where `S` is the active `CurriculumSlice`.

**Short-Circuit Evaluation:**
In implementation, gates are evaluated in order `G₁ → G₂ → G₃ → G₄`. The first failing gate halts evaluation and reports the failure reason.

---

## 3. Monotonicity Conditions

Curriculum slices must satisfy monotonicity invariants on designated axes.

### 3.1 Axis Monotonicity

For a sequence of slices `S₀, S₁, ..., Sₙ` and monotonic axis `α`:

```
∀i ∈ [0, n-1]: S_{i+1}.params[α] ≥ Sᵢ.params[α]
```

**MathLedger Default Monotonic Axes (PL):**
- `atoms`: Number of propositional variables
- `depth_max`: Maximum formula depth

### 3.2 Formal Monotonicity Predicate

```
MONO(S₀..ₙ, Axes) = ∀α ∈ Axes, ∀i ∈ [0,n-1]:
                     Sᵢ₊₁.params[α] ≥ Sᵢ.params[α]
```

### 3.3 Implications

Monotonicity ensures:
1. **No Regression**: Once mastered, simpler slices are never revisited
2. **Curriculum Ordering**: Slices form a partial order by complexity
3. **Ratchet Integrity**: Progress is irreversible within a system

---

## 4. Wide Slice Mode Constraints

> **PHASE II — NOT EXERCISED IN EVIDENCE PACK v1**
>
> Wide Slice mode is a **planned feature** for future RFL uplift experiments.
> - `slice_medium` exists in `config/curriculum.yaml` but is **not the active slice**
> - No 1000-cycle runs have been executed against wide slice parameters
> - Entropy production claims below are theoretical, not empirically validated
>
> Phase I Dyno Chart uses `slice_easy_fo` parameters, not wide slice.
>
> **CRITICAL CLARIFICATION:**
> The `fo_rfl.jsonl` file (330 cycles) is **NOT** a Wide Slice run.
> It uses `slice_easy_fo` parameters (atoms=3, depth_max=3).
> **No Wide Slice experiments exist. This evidence does not correspond to Wide Slice.**

"Wide Slice" mode (e.g., `slice_medium`) is designed for RFL uplift experiments. It produces statistically interesting abstention rates without trivial saturation.

### 4.1 Wide Slice Parameter Constraints (Planned)

For a slice to qualify as "wide":

```
WIDE(S) ⟺  (S.params.atoms ≥ 5) ∧
            (S.params.depth_max ≥ 7) ∧
            (S.gates.abstention.max_rate_pct ≥ 10) ∧
            (S.gates.abstention.max_rate_pct ≤ 25) ∧
            (S.gates.coverage.ci_lower_min ≤ 0.90)
```

### 4.2 Entropy Production (Theoretical)

Wide slices target non-trivial abstention (5-20% observed) to provide:
- Sufficient signal-to-noise ratio for RFL measurement
- Variance in proof difficulty for policy learning
- Exploration headroom before saturation

**Entropy Bound:**
```
H(abstention) ≈ -p·log₂(p) - (1-p)·log₂(1-p)
```
Where `p = r_abs / 100`. Maximum entropy at `p = 0.5`, but practical range `p ∈ [0.05, 0.20]` provides measurable signal.

### 4.3 Wide Slice Configuration (slice_medium) — NOT EXERCISED

> **No Wide Slice experiments exist. This evidence does not correspond to Wide Slice.**
> The configuration below is defined in `config/curriculum.yaml` but has never been activated.
> `fo_rfl.jsonl` (330 cycles) uses `slice_easy_fo`, NOT `slice_medium`.

Planned wide slice parameters:
```yaml
params:
  atoms: 5
  depth_max: 7
  breadth_max: 1500
  total_max: 8000
gates:
  coverage:
    ci_lower_min: 0.85
  abstention:
    max_rate_pct: 15.0
    max_mass: 800
```

---

## 5. Coverage CI Unbiasedness Under Synthetic Data

> **PHASE II — THEORETICAL ANALYSIS ONLY**
>
> This proof sketch applies to **synthetic i.i.d. data** assumptions.
> Phase I experiments do not validate CI calibration against ground truth.
> Real derivation data may exhibit temporal correlation or selection bias
> that would violate the i.i.d. assumption.

### 5.1 Proof Sketch

**Claim:** The coverage confidence interval estimator `CI_lower` is an unbiased estimator of the true coverage rate under i.i.d. synthetic sampling.

**Setup:**
- Let `X₁, X₂, ..., Xₙ` be i.i.d. Bernoulli(p) trials where `p` = true proof success rate
- Let `p̂ = (1/n) Σᵢ Xᵢ` be the sample proportion
- The coverage CI uses Wilson score interval or similar

**Proof:**

1. **Sample Mean Unbiasedness:**
   ```
   E[p̂] = E[(1/n) Σᵢ Xᵢ] = (1/n) Σᵢ E[Xᵢ] = (1/n) · n · p = p
   ```

2. **CI Lower Bound Property:**
   For a (1-α) confidence interval `[L, U]`:
   ```
   P(L ≤ p ≤ U) ≥ 1 - α
   ```
   The lower bound `L` is a conservative estimator satisfying:
   ```
   P(L ≤ p) ≥ 1 - α/2  (for two-sided CI)
   ```

3. **Synthetic Data Assumption:**
   Under synthetic data generation with true parameter `p*`:
   - Samples are drawn i.i.d. from Bernoulli(p*)
   - No selection bias or temporal correlation
   - The CI coverage property holds exactly

4. **Unbiasedness of Gating Decision:**
   The gate passes when `CI_lower ≥ θ_cov`. Under null hypothesis `p = θ_cov`:
   ```
   P(Gate passes | p = θ_cov) = P(CI_lower ≥ θ_cov | p = θ_cov) ≈ α/2
   ```
   This is the designed Type I error rate.

5. **Consistency:**
   As `n → ∞`:
   ```
   CI_lower →ᵖ p  (convergence in probability)
   ```
   The gate decision becomes deterministic at the true coverage rate.

**Conclusion:** Under synthetic i.i.d. sampling, `CI_lower` provides unbiased coverage of the true success rate, and the gating decision has calibrated Type I/II error rates. ∎

---

## 6. First Organism Gate Requirements

> **CLARIFICATION: Phase I vs Test Fixtures**
>
> The values below describe the **test fixture** in `make_first_organism_slice()` and
> `build_first_organism_metrics()` — these are used by **unit tests** in
> `tests/frontier/test_curriculum_gates.py`.
>
> The actual Phase I First Organism closed-loop test (`fo_baseline/`, `fo_rfl/`)
> uses the slice parameters from `config/curriculum.yaml` (specifically `slice_easy_fo`),
> **not** the `make_first_organism_slice()` fixture values.
>
> The unit test fixture is designed to demonstrate gate evaluation logic,
> not to match production FO parameters.

The First Organism test validates that the curriculum gating system functions correctly before production deployment.

### 6.1 Gate Pass/Fail Table (Unit Test Fixture)

| Gate       | Expected Result | Rationale |
|------------|-----------------|-----------|
| Coverage   | **FAIL**        | CI_lower (0.90) < threshold (0.915) by design |
| Abstention | PASS            | Rate (13.5%) < max (18%), Mass (432) < max (640) |
| Velocity   | PASS            | PPH (190) > min (160), CV (0.06) < max (0.10) |
| Caps       | PASS            | Mass (3200) > min (2400), Runtime (28m) > min (20m), Backlog (0.31) < max (0.36) |

### 6.2 First Organism Slice Thresholds

```python
# From make_first_organism_slice()
coverage:
  ci_lower_min: 0.915
  sample_min: 16
  require_attestation: True

abstention:
  max_rate_pct: 18.0
  max_mass: 640

velocity:
  min_pph: 160.0
  stability_cv_max: 0.10
  window_minutes: 45

caps:
  min_attempt_mass: 2400
  min_runtime_minutes: 20.0
  backlog_max: 0.36
```

### 6.3 First Organism Metrics (Default)

```python
# From build_first_organism_metrics()
coverage_ci: 0.90         # Deliberately below 0.915
sample_size: 22           # Above 16 ✓
abstention_rate: 13.5     # Below 18.0 ✓
attempt_mass: 3200        # Above 2400 ✓
proof_velocity_pph: 190.0 # Above 160.0 ✓
velocity_cv: 0.06         # Below 0.10 ✓
runtime_minutes: 28.0     # Above 20.0 ✓
backlog_fraction: 0.31    # Below 0.36 ✓
```

### 6.4 Interpretation

The First Organism test is configured so that:
1. **The run is allowed** — all operational gates (abstention, velocity, caps) pass
2. **The ratchet holds** — the coverage gate fails, preventing premature advancement
3. **The audit trail is complete** — attestation hash is captured for provenance

This validates the separation between "run permission" (operational health) and "advancement permission" (curriculum progression).

---

## 7. Gate Evaluation Order

Gates are evaluated in a fixed order with short-circuit semantics:

```
G₁ (Coverage) → G₂ (Abstention) → G₃ (Velocity) → G₄ (Caps)
```

The first failing gate:
1. Sets `advance = False`
2. Populates `reason` with failure message
3. Terminates further evaluation

All gate statuses are recorded in the audit log regardless of pass/fail.

---

## 8. Notation Summary

| Symbol | Meaning |
|--------|---------|
| `Gᵢ` | Gate i predicate (returns 0 or 1) |
| `m` | Metrics payload |
| `θ` | Gate threshold specification |
| `S` | Curriculum slice |
| `Π` | Slice parameters |
| `CI_lower` | Lower bound of coverage confidence interval |
| `r_abs` | Abstention rate (%) |
| `M_abs` | Abstention mass |
| `v` | Proof velocity (proofs per hour) |
| `CV` | Coefficient of variation |
| `A` | Attempt mass |
| `T` | Runtime (minutes) |
| `B` | Backlog fraction |
| `H` | Attestation hash |

---

## 9. Phase I vs Phase II Summary

| Component | Phase I Status | Evidence |
|-----------|---------------|----------|
| Gate predicates (G₁-G₄) | Implemented | `backend/frontier/curriculum.py` |
| Gate unit tests | Passing | `tests/frontier/test_curriculum_gates.py` |
| Curriculum YAML config | Exists | `config/curriculum.yaml` |
| `should_ratchet()` function | Implemented | Lines 1144-1182 of curriculum.py |
| `activate_next_slice()` | Implemented but **not exercised** | No production ratchet events |
| Wide Slice mode | **Not activated** | `active: atoms5-depth6` in config |
| 1000-cycle wide slice runs | **Not executed** | No logs/manifests exist |
| `fo_rfl.jsonl` (330 cycles) | Uses `slice_easy_fo` | **NOT Wide Slice** |
| CI calibration validation | **Not performed** | Theoretical sketch only |

**For Evidence Pack v1 / Paper:**
- Safe to cite: Gate definitions (Section 1), Monotonicity (Section 3), Evaluation order (Section 7)
- Must label as Phase II: Wide Slice (Section 4), CI Unbiasedness (Section 5)
- Must clarify: First Organism section describes unit test fixtures, not production FO parameters

---

## 10. Phase II Uplift-Capable Slices (Design Only)

> **STATUS: PHASE IIb DESIGN — DEFERRED UNTIL U1 COMPLETES**
>
> This section proposes **Lean-enabled** curriculum slices where RFL uplift measurement is conceptually possible.
> These slices are designed to produce **non-degenerate abstention** (neither ~0% nor ~100%)
> so that policy-guided selection has room to improve coverage.
>
> **SEQUENCING:**
> 1. **U1 (Phase II)**: Run canonical experiment on `slice_medium` with truth-table verification FIRST
> 2. **Analyze U1**: Determine if INVALID/NULL/POSITIVE
> 3. **Phase IIb**: Only then implement Lean-enabled slices below
>
> **Prerequisites for Phase IIb execution:**
> - U1 experiment completed and analyzed
> - Lean-enabled verifier mode functional
> - `run_fo_cycles.py` configured for target slice parameters
> - Baseline + RFL comparison runs with sufficient cycle count (≥500)
>
> **None of these slices have been exercised. No data exists.**
> **Do NOT implement these until U1 results are known.**

### 10.1 Design Rationale

Phase I (`slice_easy_fo`) produces degenerate results:
- **atoms=3, depth_max=3**: Formula space is trivially small
- **Lean-disabled**: All verification is mocked → 100% abstention or 100% pass depending on mock
- **No uplift signal**: RFL cannot demonstrate improvement over baseline

For Phase II uplift measurement, we need slices where:
1. **Some proofs succeed**: The verifier can actually prove non-trivial tautologies
2. **Some proofs fail/timeout**: Abstention mass exists for RFL to reduce
3. **Policy has leverage**: Formula selection affects success rate

### 10.2 Proposed Uplift Slices

#### 10.2.1 `slice_pl_uplift_a` — Conservative Entry Point

**Target Abstention Band:** 20-40%

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `atoms` | 4 | Moderate formula space; tractable for Lean |
| `depth_min` | 2 | Skip trivial depth-1 formulas |
| `depth_max` | 5 | Include moderately complex formulas |
| `breadth_max` | 800 | Controlled exploration per step |
| `total_max` | 4000 | ~500 cycles × 8 formulas/cycle |
| `lean_timeout_ms` | 5000 | 5s per proof; allows complex proofs to complete |

**Why some proofs succeed:**
- Depth 2-4 propositional tautologies are well within Lean's `tautCheck` capability
- 4 atoms produces ~65,536 truth table rows — tractable for brute-force verification
- Most valid tautologies at this depth complete in <1s

**Why nontrivial abstention exists:**
- Depth 5 formulas with nested implications can timeout at 5s
- Some derivation paths produce non-tautologies (MP on ill-formed premises)
- Lean startup overhead (~500ms) consumes timeout budget on marginal cases

**RFL Uplift Opportunity:**
- Policy learns to prefer depth 2-4 over depth 5 when timeout risk is high
- Policy learns to avoid derivation paths that historically produce non-tautologies
- Expected uplift: 5-15% abstention reduction (from ~30% to ~20%)

**Gate Tuning (Phase II):**

```yaml
gates:
  coverage:
    ci_lower_min: 0.70      # Relaxed for exploration phase
    sample_min: 50          # Require statistical significance
    require_attestation: true
  abstention:
    max_rate_pct: 45.0      # Allow up to 45% while measuring
    max_mass: 1800          # 45% of 4000
  velocity:
    min_pph: 100            # Lower than production; Lean is slow
    stability_cv_max: 0.20  # Allow variance during exploration
    window_minutes: 90
  caps:
    min_attempt_mass: 2000  # Require ≥500 cycles
    min_runtime_minutes: 30
    backlog_max: 0.50
```

---

#### 10.2.2 `slice_pl_uplift_b` — Moderate Challenge

**Target Abstention Band:** 35-55%

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `atoms` | 5 | Larger formula space; more selection pressure |
| `depth_min` | 3 | Skip easy formulas entirely |
| `depth_max` | 7 | Include challenging derivations |
| `breadth_max` | 1200 | More formulas per step |
| `total_max` | 8000 | ~1000 cycles × 8 formulas/cycle |
| `lean_timeout_ms` | 3000 | 3s timeout; creates harder selection problem |

**Why some proofs succeed:**
- Depth 3-5 tautologies still complete within 3s
- 5 atoms (2^32 truth table) is at the edge but still tractable
- Lean's `tautCheck` handles most cases with efficient BDD-style evaluation

**Why nontrivial abstention exists:**
- 3s timeout is aggressive; depth 6-7 formulas frequently timeout
- Larger atom count means more cases where tautology check is slow
- Higher breadth increases chance of selecting marginal formulas

**RFL Uplift Opportunity:**
- Stronger selection pressure than `uplift_a`
- Policy must learn both depth AND atom-count preferences
- Expected uplift: 10-20% abstention reduction (from ~45% to ~30%)

**Gate Tuning (Phase II):**

```yaml
gates:
  coverage:
    ci_lower_min: 0.55      # Very relaxed for high-abstention slice
    sample_min: 80          # More samples needed for variance
    require_attestation: true
  abstention:
    max_rate_pct: 60.0      # Allow up to 60% during burn-in
    max_mass: 4800          # 60% of 8000
  velocity:
    min_pph: 60             # Expect slow throughput
    stability_cv_max: 0.25  # High variance acceptable
    window_minutes: 120
  caps:
    min_attempt_mass: 4000  # Require ≥500 cycles
    min_runtime_minutes: 60
    backlog_max: 0.55
```

---

#### 10.2.3 `slice_pl_uplift_c` — Stress Test

**Target Abstention Band:** 50-70%

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `atoms` | 6 | Large formula space; strong selection pressure |
| `depth_min` | 4 | Only moderately hard formulas |
| `depth_max` | 9 | Include very challenging derivations |
| `breadth_max` | 1500 | High exploration rate |
| `total_max` | 12000 | ~1500 cycles × 8 formulas/cycle |
| `lean_timeout_ms` | 2000 | 2s timeout; maximizes selection pressure |

**Why some proofs succeed:**
- Depth 4-6 formulas can still complete in 2s if well-structured
- Not all 6-atom formulas hit worst-case truth table enumeration
- Policy-guided selection can find the "easy" subset

**Why nontrivial abstention exists:**
- 2s timeout is very aggressive
- Depth 7-9 almost always times out
- 6 atoms approaches practical limits of naive tautology checking

**RFL Uplift Opportunity:**
- Maximum selection pressure — random policy performs poorly
- Policy must learn complex depth × atoms × structure preferences
- Expected uplift: 15-25% abstention reduction (from ~60% to ~40%)
- **Caution:** If baseline abstention exceeds 70%, uplift measurement becomes noisy

**Gate Tuning (Phase II):**

```yaml
gates:
  coverage:
    ci_lower_min: 0.40      # Very low for stress test
    sample_min: 100         # High sample count for noisy regime
    require_attestation: true
  abstention:
    max_rate_pct: 75.0      # Allow very high abstention
    max_mass: 9000          # 75% of 12000
  velocity:
    min_pph: 40             # Very slow expected
    stability_cv_max: 0.30  # High variance
    window_minutes: 180
  caps:
    min_attempt_mass: 6000  # Require ≥750 cycles
    min_runtime_minutes: 90
    backlog_max: 0.60
```

---

### 10.3 Slice Selection Guide

| Slice | Target Abstention | Lean Timeout | Recommended Use |
|-------|-------------------|--------------|-----------------|
| `slice_pl_uplift_a` | 20-40% | 5s | Initial Phase II validation; confirm RFL machinery works |
| `slice_pl_uplift_b` | 35-55% | 3s | Primary uplift measurement; balanced signal-to-noise |
| `slice_pl_uplift_c` | 50-70% | 2s | Stress test; maximum selection pressure |

**Recommended Phase II Sequence:**
1. Run `slice_pl_uplift_a` baseline + RFL (500 cycles each)
2. If uplift detected, proceed to `slice_pl_uplift_b` (1000 cycles each)
3. If strong uplift confirmed, attempt `slice_pl_uplift_c` for stress validation

### 10.4 Implementation Notes for Code Agents

To implement these slices in `run_fo_cycles.py`:

```python
# Example: Configure for slice_pl_uplift_a
SLICE_CONFIG = {
    "name": "slice_pl_uplift_a",
    "atoms": 4,
    "depth_min": 2,
    "depth_max": 5,
    "breadth_max": 800,
    "total_max": 4000,
    "lean_timeout_ms": 5000,
    "lean_enabled": True,  # CRITICAL: Must be True for Phase II
}
```

**Critical Requirements:**
- `lean_enabled` must be `True` (Phase I used `False`)
- Ensure `backend/lean_proj/` is built and functional
- Verify Lean timeout is respected by worker process
- Metrics collection must capture per-formula success/fail/timeout

---

### 10.5 Relationship to U1 Preregistration

> **CROSS-REFERENCE:** See `experiments/prereg/PREREG_UPLIFT_U1.md` and `RFL_UPLIFT_THEORY.md` Section 10.

**U1 is the canonical first uplift experiment.** It uses `slice_medium` from `config/curriculum.yaml`, which is **different** from the Lean-enabled uplift slices designed in this section.

| Experiment | Slice | Verifier | Effect Threshold | Status | Phase |
|------------|-------|----------|------------------|--------|-------|
| **U1** | `slice_medium` (atoms=5, depth=7) | Truth-table | **≥ 10pp** | **CANONICAL — RUN FIRST** | Phase II |
| U1b | `slice_pl_uplift_a` (atoms=4, depth=5) | Lean-enabled | TBD | Deferred | Phase IIb |
| U1c | `slice_pl_uplift_b` (atoms=5, depth=7) | Lean-enabled | TBD | Deferred | Phase IIb |
| U1d | `slice_pl_uplift_c` (atoms=6, depth=9) | Lean-enabled | TBD | Deferred | Phase IIb |

**Sequencing Decision:**
1. U1 (truth-table) runs FIRST — simpler, no Lean complexity
2. Analyze U1 outcome (INVALID/NULL/POSITIVE)
3. Only design U1b/c/d (Lean-enabled) after U1 results are known

**Current Evidence Status:**
- Phase I runs (`fo_rfl.jsonl`, etc.): 100% abstention, lean-disabled — **plumbing tests only**
- U1 experiment: **CANONICAL** — Preregistered but **not executed** — no `results/uplift_u1/` directory exists
- Phase IIb uplift slices: Design specifications only — **deferred until U1 completes**

**No uplift evidence exists.** All theoretical claims in `RFL_UPLIFT_THEORY.md` remain unvalidated.

---

*Generated for MathLedger RFL Curriculum Gating System*
*CLAUDE G — Curriculum Gatekeeper*
*Document revised under SOBER TRUTH / REVIEWER-2 directive*
