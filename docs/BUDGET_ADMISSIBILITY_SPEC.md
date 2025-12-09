# Budget Admissibility Specification

> **STATUS: PHASE II — NOT RUN IN PHASE I**
>
> This document defines the conditions under which budget-induced abstentions
> are scientifically admissible versus when they invalidate experimental results.
> It establishes the Budget-Induced Abstention Law and Admissibility Regions.

---

## 1. Introduction

### 1.1 The Admissibility Problem

Budget constraints ensure bounded computation but introduce a classification problem:
when does budget exhaustion represent **acceptable resource governance** versus
**experimental contamination**?

A run with 0.1% budget-induced skips is likely valid.
A run with 90% budget-induced skips is likely invalid.
The boundary between these extremes requires formal specification.

### 1.2 Scope

This specification defines:
1. **Admissibility conditions** — When budget exhaustion is acceptable
2. **Rejection conditions** — When budget exhaustion invalidates results
3. **Classification regions** — Partition of the (exhaustion_rate, Δp) space
4. **Reporting requirements** — How budget metrics integrate into artifacts

### 1.3 Non-Scope

This specification does **NOT** define:
- Decision rules (e.g., "if X then rollback")
- Automated remediation procedures
- Code implementation details

Classification only; operational response is separate.

---

## 2. Budget-Induced Abstention Law

### 2.1 Definitions

Let a single experimental run `R` consist of `C` cycles. For each cycle `c`:

| Symbol | Definition |
|--------|------------|
| `N_c` | Total candidates in cycle `c` |
| `V_c` | Candidates verified in cycle `c` |
| `R_c` | Candidates refuted in cycle `c` |
| `A_c` | Candidates abstained (timeout/complexity) in cycle `c` |
| `S_c` | Candidates skipped (budget exhaustion) in cycle `c` |

**Aggregate metrics:**

```
N_total = Σ_c N_c                    # Total candidates across all cycles
S_total = Σ_c S_c                    # Total budget-induced skips
B_rate = S_total / N_total           # Budget exhaustion rate
```

### 2.2 The Fundamental Principle

**Budget-Induced Abstention Law:**

> Budget exhaustion is admissible if and only if:
> 1. The exhaustion rate is bounded below a threshold
> 2. The exhaustion is symmetric between compared conditions
> 3. The exhaustion does not correlate with verification outcome

Formally:

```
ADMISSIBLE(R_baseline, R_rfl) ⟺
    B_rate(R_baseline) ≤ τ_max  ∧
    B_rate(R_rfl) ≤ τ_max  ∧
    |B_rate(R_baseline) - B_rate(R_rfl)| ≤ δ_sym  ∧
    ¬CORRELATED(S, outcome)
```

Where:
- `τ_max` — Maximum admissible exhaustion rate
- `δ_sym` — Maximum admissible asymmetry between conditions

### 2.3 Admissibility Conditions (Budget Exhaustion Allowed)

Budget exhaustion is **ADMISSIBLE** when ALL of the following hold:

#### Condition A1: Bounded Rate

```
B_rate ≤ τ_max

Where τ_max is defined per severity level:
  • τ_max = 0.05 (5%)   — SAFE: No concern
  • τ_max = 0.15 (15%)  — CAUTION: Requires justification
  • τ_max = 0.30 (30%)  — WARNING: Significant concern
```

**Rationale:** Low exhaustion rates mean most candidates were observed,
preserving statistical validity.

#### Condition A2: Symmetry Across Conditions

```
|B_rate(baseline) - B_rate(rfl)| ≤ δ_sym

Where δ_sym = 0.05 (5 percentage points)
```

**Rationale:** If baseline and RFL experience similar exhaustion rates,
the budget constraint affects both equally and does not bias comparison.

#### Condition A3: Exhaustion Independence

```
P(skip | would_verify) ≈ P(skip | would_refute)
```

**Rationale:** If budget exhaustion is independent of what the verification
outcome *would have been*, then skipped candidates are Missing Completely
At Random (MCAR), preserving unbiased estimates.

**Operationalization:** This cannot be directly measured (counterfactual),
but can be approximated by checking:
- Skips are distributed uniformly across cycle positions
- Skip rate does not correlate with candidate complexity proxies
- Skip rate does not trend over the experiment duration

#### Condition A4: Cycle Coverage

```
∀c: (V_c + R_c + A_c) / N_c ≥ κ_min

Where κ_min = 0.50 (at least 50% of candidates observed per cycle)
```

**Rationale:** Ensures each cycle contributes meaningful signal, not just skips.

### 2.4 Rejection Conditions (Budget Exhaustion Invalidates Run)

Budget exhaustion is **grounds for rejection** when ANY of the following hold:

#### Condition R1: Excessive Rate

```
B_rate > τ_reject

Where τ_reject = 0.50 (50%)
```

**Rationale:** If more than half of candidates are skipped, the run does not
provide sufficient evidence for any conclusion.

#### Condition R2: Severe Asymmetry

```
|B_rate(baseline) - B_rate(rfl)| > δ_reject

Where δ_reject = 0.20 (20 percentage points)
```

**Rationale:** Large asymmetry suggests the policy or ordering affects
budget exhaustion itself, confounding the comparison.

#### Condition R3: Systematic Bias Pattern

```
TREND(S_c) is monotonic AND significant
```

**Rationale:** If skip rate systematically increases or decreases over cycles,
suggests non-stationarity in verification difficulty or budget adequacy.

#### Condition R4: Complete Cycle Failures

```
∃c: S_c = N_c (all candidates skipped)

AND fraction of such cycles > 0.10
```

**Rationale:** Cycles with no observations contribute zero information.
More than 10% complete-skip cycles indicates systematic budget inadequacy.

---

## 3. Admissibility Regions

### 3.1 Two-Dimensional Classification

The admissibility space is partitioned by two axes:
- **X-axis:** Budget exhaustion rate (`B_rate`)
- **Y-axis:** Effect size magnitude (`|Δp|`)

```
                           |Δp| (Effect Size Magnitude)
                                    │
                    0.30 ┼──────────┬──────────┬──────────┐
                         │          │          │          │
                         │  SAFE    │ SAFE     │SUSPICIOUS│
                         │  HIGH Δp │ HIGH Δp  │ HIGH Δp  │
                         │          │          │          │
                    0.15 ┼──────────┼──────────┼──────────┤
                         │          │          │          │
                         │  SAFE    │SUSPICIOUS│ INVALID  │
                         │  MED Δp  │ MED Δp   │ MED Δp   │
                         │          │          │          │
                    0.05 ┼──────────┼──────────┼──────────┤
                         │          │          │          │
                         │  SAFE    │SUSPICIOUS│ INVALID  │
                         │  LOW Δp  │ LOW Δp   │ LOW Δp   │
                         │          │          │          │
                    0.00 ┼──────────┴──────────┴──────────┘
                         0.00      0.15       0.30       0.50
                                                              B_rate
                                   Budget Exhaustion Rate ──────────▶
```

### 3.2 Region Definitions

#### SAFE Region

```
SAFE ⟺ B_rate ≤ 0.15 ∧ symmetry_ok
```

| Condition | Interpretation |
|-----------|----------------|
| B_rate ≤ 5% | Budget constraint is not binding |
| B_rate 5-15% | Budget constraint binds occasionally |
| Any |Δp| | Effect size interpretation unaffected |

**Classification:** Run is scientifically valid. Budget metrics are informational only.

#### SUSPICIOUS Region

```
SUSPICIOUS ⟺ (0.15 < B_rate ≤ 0.30) ∨ (B_rate ≤ 0.15 ∧ asymmetry_warning)
```

| Condition | Interpretation |
|-----------|----------------|
| B_rate 15-30% | Budget constraint binds frequently |
| Low |Δp| with moderate B_rate | Effect may be masked by skips |
| Asymmetry 5-20% | Differential budget effects possible |

**Classification:** Run requires additional scrutiny. Results may be valid but
confidence is reduced. Sensitivity analysis recommended.

#### INVALID Region

```
INVALID ⟺ B_rate > 0.30 ∨ asymmetry > 0.20 ∨ systematic_bias
```

| Condition | Interpretation |
|-----------|----------------|
| B_rate > 30% | Majority of evidence is missing |
| B_rate > 50% | Run provides no reliable evidence |
| Asymmetry > 20% | Comparison is fundamentally confounded |
| Systematic bias | Non-random missingness pattern |

**Classification:** Run is scientifically invalid for the intended comparison.
Results cannot be interpreted as evidence for or against uplift.

### 3.3 Partition Table

| B_rate | |Δp| < 0.05 | 0.05 ≤ |Δp| < 0.15 | |Δp| ≥ 0.15 |
|--------|-------------|---------------------|--------------|
| ≤ 5% | SAFE | SAFE | SAFE |
| 5-15% | SAFE | SAFE | SAFE |
| 15-20% | SUSPICIOUS | SUSPICIOUS | SAFE |
| 20-30% | SUSPICIOUS | SUSPICIOUS | SUSPICIOUS |
| 30-50% | INVALID | INVALID | SUSPICIOUS |
| > 50% | INVALID | INVALID | INVALID |

**Key insight:** Large effect sizes are more robust to budget exhaustion.
A clear 20% uplift survives 25% missing data; a marginal 2% uplift does not.

### 3.4 Asymmetry Modifier

The base classification is modified by asymmetry:

| Asymmetry | Modifier |
|-----------|----------|
| ≤ 2% | No change |
| 2-5% | No change (within tolerance) |
| 5-10% | Downgrade SAFE → SUSPICIOUS if B_rate > 10% |
| 10-20% | Downgrade SAFE → SUSPICIOUS always |
| > 20% | Downgrade to INVALID |

### 3.5 Summary Classification Matrix

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     ADMISSIBILITY CLASSIFICATION MATRIX                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   B_rate        Asymmetry       Effect Size      Classification             │
│   ─────────     ──────────      ───────────      ──────────────             │
│   ≤ 15%         ≤ 5%            Any              SAFE                       │
│   ≤ 15%         5-10%           Any              SAFE (with note)           │
│   ≤ 15%         10-20%          Any              SUSPICIOUS                 │
│   ≤ 15%         > 20%           Any              INVALID                    │
│                                                                             │
│   15-30%        ≤ 5%            ≥ 15%            SAFE                       │
│   15-30%        ≤ 5%            < 15%            SUSPICIOUS                 │
│   15-30%        5-20%           Any              SUSPICIOUS                 │
│   15-30%        > 20%           Any              INVALID                    │
│                                                                             │
│   30-50%        Any             ≥ 15%            SUSPICIOUS                 │
│   30-50%        Any             < 15%            INVALID                    │
│                                                                             │
│   > 50%         Any             Any              INVALID                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Formal Metrics

### 4.1 Primary Metrics

| Metric | Formula | Threshold |
|--------|---------|-----------|
| `budget_exhaustion_rate` | `S_total / N_total` | τ_max = 0.15 |
| `budget_asymmetry` | `|B_rate(base) - B_rate(rfl)|` | δ_sym = 0.05 |
| `min_cycle_coverage` | `min_c((V_c + R_c + A_c) / N_c)` | κ_min = 0.50 |
| `complete_skip_cycles` | `count(c: S_c = N_c) / C` | < 0.10 |

### 4.2 Derived Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| `effective_sample_size` | `N_total - S_total` | Actual observations |
| `information_loss` | `S_total / N_total` | Fraction of evidence missing |
| `coverage_stability` | `std(coverage_c) / mean(coverage_c)` | CV of per-cycle coverage |
| `skip_trend_slope` | `linear_regression(c, S_c/N_c).slope` | Trend in exhaustion |

### 4.3 Classification Formula

```python
def classify_admissibility(
    b_rate_base: float,
    b_rate_rfl: float,
    delta_p: float,
    min_coverage: float,
    complete_skip_frac: float,
    skip_trend_significant: bool,
) -> str:
    """
    Classify run admissibility based on budget metrics.

    Returns: "SAFE" | "SUSPICIOUS" | "INVALID"
    """
    b_rate = max(b_rate_base, b_rate_rfl)
    asymmetry = abs(b_rate_base - b_rate_rfl)
    effect_magnitude = abs(delta_p)

    # Hard rejection conditions
    if b_rate > 0.50:
        return "INVALID"
    if asymmetry > 0.20:
        return "INVALID"
    if complete_skip_frac > 0.10:
        return "INVALID"
    if skip_trend_significant:
        return "INVALID"  # Systematic bias

    # Invalid region (30-50% exhaustion with small effect)
    if b_rate > 0.30 and effect_magnitude < 0.15:
        return "INVALID"

    # Suspicious region
    if b_rate > 0.30:  # 30-50% with large effect
        return "SUSPICIOUS"
    if b_rate > 0.15 and effect_magnitude < 0.15:
        return "SUSPICIOUS"
    if asymmetry > 0.10:
        return "SUSPICIOUS"
    if min_coverage < 0.50:
        return "SUSPICIOUS"

    # Safe region
    if asymmetry > 0.05:
        return "SAFE"  # with note about asymmetry

    return "SAFE"
```

---

## 5. Integration Hooks

### 5.1 Manifest Integration

The experiment manifest MUST include budget admissibility metrics:

```json
{
  "manifest_version": "2.0",
  "phase": "II",

  "budget": {
    "configured": {
      "cycle_budget_s": 5.0,
      "taut_timeout_s": 0.10,
      "max_candidates_per_cycle": 100
    },

    "observed": {
      "total_candidates": 50000,
      "total_skipped": 2500,
      "budget_exhaustion_rate": 0.05,
      "exhausted_cycles": 12,
      "complete_skip_cycles": 0
    },

    "admissibility": {
      "classification": "SAFE",
      "exhaustion_rate": 0.05,
      "exhaustion_threshold": 0.15,
      "within_threshold": true
    }
  }
}
```

**Required fields in `manifest.json`:**

| Field Path | Type | Description |
|------------|------|-------------|
| `budget.configured.*` | object | Input budget parameters |
| `budget.observed.total_candidates` | int | Total candidates attempted |
| `budget.observed.total_skipped` | int | Candidates skipped due to budget |
| `budget.observed.budget_exhaustion_rate` | float | S_total / N_total |
| `budget.observed.exhausted_cycles` | int | Cycles where budget was hit |
| `budget.observed.complete_skip_cycles` | int | Cycles with 100% skip |
| `budget.admissibility.classification` | str | SAFE/SUSPICIOUS/INVALID |

### 5.2 Summary Integration

The statistical summary MUST include comparative budget analysis:

```json
{
  "summary_version": "2.0",

  "budget_comparison": {
    "baseline": {
      "exhaustion_rate": 0.048,
      "exhausted_cycles": 6,
      "mean_cycle_coverage": 0.95
    },
    "rfl": {
      "exhaustion_rate": 0.052,
      "exhausted_cycles": 7,
      "mean_cycle_coverage": 0.94
    },
    "asymmetry": {
      "exhaustion_rate_diff": 0.004,
      "within_symmetric_bound": true,
      "symmetric_bound": 0.05
    },
    "admissibility": {
      "joint_classification": "SAFE",
      "baseline_classification": "SAFE",
      "rfl_classification": "SAFE",
      "comparison_valid": true
    }
  },

  "effect_size": {
    "delta_p": 0.12,
    "ci_95": [0.08, 0.16],
    "robust_to_budget": true,
    "budget_sensitivity_note": null
  }
}
```

**Required fields in `summary.json`:**

| Field Path | Type | Description |
|------------|------|-------------|
| `budget_comparison.baseline.*` | object | Baseline budget metrics |
| `budget_comparison.rfl.*` | object | RFL budget metrics |
| `budget_comparison.asymmetry.*` | object | Symmetry analysis |
| `budget_comparison.admissibility.*` | object | Classification results |
| `effect_size.robust_to_budget` | bool | Whether effect survives budget concerns |

### 5.3 Evidence Dossier Integration

The evidence dossier MUST include budget admissibility attestation:

```yaml
# evidence/budget_attestation.yaml

attestation:
  type: "budget_admissibility"
  schema_version: "1.0"

evidence:
  classification: "SAFE"
  classification_basis:
    - "budget_exhaustion_rate = 0.05 ≤ τ_max = 0.15"
    - "asymmetry = 0.004 ≤ δ_sym = 0.05"
    - "min_cycle_coverage = 0.89 ≥ κ_min = 0.50"
    - "complete_skip_cycles = 0%"
    - "no_systematic_trend = true"

  metrics:
    exhaustion_rate_baseline: 0.048
    exhaustion_rate_rfl: 0.052
    exhaustion_asymmetry: 0.004
    effect_magnitude: 0.12
    min_cycle_coverage: 0.89

  thresholds_applied:
    τ_max: 0.15
    δ_sym: 0.05
    κ_min: 0.50
    τ_reject: 0.50
    δ_reject: 0.20

  conclusion: |
    Budget exhaustion is within admissible bounds for both conditions.
    Asymmetry is negligible. Comparison is scientifically valid.

  caveats: []
```

**Required sections in evidence dossier:**

| Section | Purpose |
|---------|---------|
| `attestation.type` | Identifies this as budget attestation |
| `evidence.classification` | Final admissibility classification |
| `evidence.classification_basis` | Human-readable justification |
| `evidence.metrics` | Raw metrics used in classification |
| `evidence.thresholds_applied` | Threshold values used |
| `evidence.conclusion` | Summary statement |
| `evidence.caveats` | Any concerns or notes |

### 5.4 Telemetry Integration

Per-cycle telemetry MUST include budget fields:

```json
{
  "cycle": 42,
  "budget_status": {
    "exhausted": false,
    "elapsed_s": 3.2,
    "budget_s": 5.0,
    "utilization": 0.64,
    "candidates_attempted": 38,
    "candidates_skipped": 0,
    "coverage": 1.0
  }
}
```

For exhausted cycles:

```json
{
  "cycle": 99,
  "budget_status": {
    "exhausted": true,
    "elapsed_s": 5.0,
    "budget_s": 5.0,
    "utilization": 1.0,
    "candidates_attempted": 28,
    "candidates_skipped": 12,
    "coverage": 0.70,
    "exhaustion_reason": "cycle_budget"
  }
}
```

---

## 6. Sensitivity Analysis Requirements

### 6.1 When Required

Sensitivity analysis is REQUIRED when classification is SUSPICIOUS:

```
SUSPICIOUS → MUST perform sensitivity analysis
SAFE → sensitivity analysis optional
INVALID → sensitivity analysis moot (run rejected)
```

### 6.2 Sensitivity Analysis Components

For SUSPICIOUS runs, the summary MUST include:

```json
{
  "sensitivity_analysis": {
    "performed": true,
    "reason": "budget_exhaustion_rate = 0.22 > 0.15",

    "scenarios": {
      "all_skips_verify": {
        "adjusted_delta_p": 0.08,
        "adjusted_ci_95": [0.03, 0.13],
        "conclusion": "effect_persists"
      },
      "all_skips_refute": {
        "adjusted_delta_p": 0.16,
        "adjusted_ci_95": [0.11, 0.21],
        "conclusion": "effect_persists"
      },
      "skips_match_observed": {
        "adjusted_delta_p": 0.12,
        "adjusted_ci_95": [0.07, 0.17],
        "conclusion": "effect_persists"
      }
    },

    "robust_conclusion": true,
    "note": "Effect persists under all imputation scenarios"
  }
}
```

### 6.3 Imputation Scenarios

| Scenario | Description | Purpose |
|----------|-------------|---------|
| `all_skips_verify` | Assume all skipped candidates would verify | Upper bound on effect |
| `all_skips_refute` | Assume all skipped candidates would refute | Lower bound on effect |
| `skips_match_observed` | Impute at observed verification rate | Point estimate |
| `skips_favor_baseline` | Skipped in RFL verify, skipped in baseline refute | Adversarial to RFL |
| `skips_favor_rfl` | Skipped in baseline verify, skipped in RFL refute | Adversarial to baseline |

---

## 7. Reporting Templates

### 7.1 SAFE Classification Report

```
BUDGET ADMISSIBILITY: SAFE
══════════════════════════════════════════════════════════════

Budget Exhaustion Analysis
─────────────────────────────────────────────────────────────
  Baseline exhaustion rate:  4.8%  (threshold: 15%)  ✓
  RFL exhaustion rate:       5.2%  (threshold: 15%)  ✓
  Asymmetry:                 0.4%  (threshold: 5%)   ✓
  Min cycle coverage:        89%   (threshold: 50%)  ✓
  Complete-skip cycles:      0%    (threshold: 10%)  ✓

Classification: SAFE
─────────────────────────────────────────────────────────────
  All admissibility conditions satisfied.
  Budget constraints do not affect scientific validity.
  No sensitivity analysis required.

Conclusion: Results are scientifically valid for comparison.
```

### 7.2 SUSPICIOUS Classification Report

```
BUDGET ADMISSIBILITY: SUSPICIOUS
══════════════════════════════════════════════════════════════

Budget Exhaustion Analysis
─────────────────────────────────────────────────────────────
  Baseline exhaustion rate:  18.2%  (threshold: 15%)  ⚠
  RFL exhaustion rate:       22.4%  (threshold: 15%)  ⚠
  Asymmetry:                  4.2%  (threshold: 5%)   ✓
  Min cycle coverage:         72%   (threshold: 50%)  ✓
  Complete-skip cycles:       2%    (threshold: 10%)  ✓

Classification: SUSPICIOUS
─────────────────────────────────────────────────────────────
  Exhaustion rate exceeds threshold for both conditions.
  Effect magnitude (12%) is below high-confidence threshold (15%).
  Sensitivity analysis REQUIRED.

Sensitivity Analysis Results
─────────────────────────────────────────────────────────────
  All-verify imputation:     Δp = 8%   [3%, 13%]  Effect persists
  All-refute imputation:     Δp = 16%  [11%, 21%] Effect persists
  Observed-rate imputation:  Δp = 12%  [7%, 17%]  Effect persists

Conclusion: Results are valid with reduced confidence.
            Effect appears robust to budget exhaustion.
```

### 7.3 INVALID Classification Report

```
BUDGET ADMISSIBILITY: INVALID
══════════════════════════════════════════════════════════════

Budget Exhaustion Analysis
─────────────────────────────────────────────────────────────
  Baseline exhaustion rate:  32.1%  (threshold: 15%)  ✗
  RFL exhaustion rate:       58.7%  (threshold: 15%)  ✗
  Asymmetry:                 26.6%  (threshold: 5%)   ✗
  Min cycle coverage:         41%   (threshold: 50%)  ✗
  Complete-skip cycles:      14%    (threshold: 10%)  ✗

Classification: INVALID
─────────────────────────────────────────────────────────────
  REJECTION CONDITION R1: Excessive rate (58.7% > 50%)
  REJECTION CONDITION R2: Severe asymmetry (26.6% > 20%)
  REJECTION CONDITION R4: Complete-skip cycles (14% > 10%)

Conclusion: Results are NOT scientifically valid.
            Run must be repeated with adjusted budget parameters.

Recommended Actions
─────────────────────────────────────────────────────────────
  1. Increase cycle_budget_s (current: 5.0s → suggested: 15.0s)
  2. Reduce max_candidates_per_cycle (current: 100 → suggested: 40)
  3. Investigate cause of asymmetric exhaustion
```

---

## 8. References

- `docs/VERIFIER_BUDGET_THEORY.md` — Theoretical foundation
- `docs/BUDGET_STRESS_TEST_PLAN.md` — Testing scenarios
- `docs/BUDGET_EXTENSION_SCHEME.md` — Version evolution
- `backend/lean_control_sandbox_plan.md` §12-14 — Budget implementation

---

## Appendix A: Threshold Summary

| Threshold | Symbol | Value | Purpose |
|-----------|--------|-------|---------|
| Max exhaustion (safe) | τ_max | 0.15 | Upper bound for SAFE |
| Max exhaustion (suspicious) | — | 0.30 | Upper bound for SUSPICIOUS |
| Max exhaustion (reject) | τ_reject | 0.50 | Hard rejection threshold |
| Max asymmetry (safe) | δ_sym | 0.05 | Symmetric bound |
| Max asymmetry (reject) | δ_reject | 0.20 | Hard asymmetry rejection |
| Min cycle coverage | κ_min | 0.50 | Per-cycle observation floor |
| Max complete-skip cycles | — | 0.10 | Complete failure tolerance |

## Appendix B: Classification Decision Tree

```
                              START
                                │
                                ▼
                    ┌───────────────────────┐
                    │ B_rate > 50%?         │
                    └───────────┬───────────┘
                         yes    │    no
                          │     │     │
                          ▼     │     ▼
                      INVALID   │   ┌───────────────────────┐
                                │   │ Asymmetry > 20%?      │
                                │   └───────────┬───────────┘
                                │        yes    │    no
                                │         │     │     │
                                │         ▼     │     ▼
                                │     INVALID   │   ┌───────────────────────┐
                                │               │   │ Complete-skip > 10%?  │
                                │               │   └───────────┬───────────┘
                                │               │        yes    │    no
                                │               │         │     │     │
                                │               │         ▼     │     ▼
                                │               │     INVALID   │   ┌───────────────────────┐
                                │               │               │   │ B_rate > 30%?         │
                                │               │               │   └───────────┬───────────┘
                                │               │               │        yes    │    no
                                │               │               │         │     │     │
                                │               │               │         ▼     │     ▼
                                │               │               │  ┌─────────┐  │   ┌───────────────────────┐
                                │               │               │  │|Δp|≥15%?│  │   │ B_rate > 15%?         │
                                │               │               │  └────┬────┘  │   └───────────┬───────────┘
                                │               │               │  yes  │  no   │        yes    │    no
                                │               │               │   │   │   │   │         │     │     │
                                │               │               │   ▼   │   ▼   │         ▼     │     ▼
                                │               │               │ SUSP  │ INV   │  ┌─────────┐  │   SAFE
                                │               │               │       │       │  │|Δp|≥15%?│  │
                                │               │               │       │       │  └────┬────┘  │
                                │               │               │       │       │  yes  │  no   │
                                │               │               │       │       │   │   │   │   │
                                │               │               │       │       │   ▼   │   ▼   │
                                │               │               │       │       │ SAFE  │ SUSP  │
                                │               │               │       │       │       │       │
                                └───────────────┴───────────────┴───────┴───────┴───────┴───────┘
```

---

*End of Budget Admissibility Specification.*
