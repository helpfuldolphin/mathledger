# CONJECTURE_ENGINE_CONTRACT.md

> **STATUS: PHASE II — NOT RUN IN PHASE I**
>
> This document specifies the formal contract for the Conjecture Engine: the component
> that evaluates experimental evidence against theoretical conjectures.

---

## 1. Purpose and Scope

### 1.1 Engine Role

The **Conjecture Engine** is a deterministic, interpretive component that:

1. **Receives** structured experimental data (JSONL logs, summary, telemetry)
2. **Evaluates** each conjecture against predefined binding rules
3. **Produces** a structured report with evidential status for each conjecture

### 1.2 Fundamental Constraints

> **CRITICAL: The Conjecture Engine operates under strict interpretive boundaries.**

| Constraint | Description |
|------------|-------------|
| **No Invention** | Engine MUST NOT invent new conjectures |
| **No Reinterpretation** | Engine MUST NOT redefine conjecture meanings |
| **No Threshold Modification** | Engine MUST use thresholds from RFL_UPLIFT_THEORY.md |
| **Deterministic** | Same input → same output, always |
| **Traceable** | Every conclusion must cite specific data fields |

---

## 2. Input Contract

### 2.1 Required Inputs

The Conjecture Engine requires exactly three input sources:

```
┌─────────────────────────────────────────────────────────────────┐
│                     CONJECTURE ENGINE INPUTS                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. JSONL Log Files                                             │
│     ├── baseline_log.jsonl (required)                           │
│     └── rfl_log.jsonl (required)                                │
│                                                                 │
│  2. Summary Files                                               │
│     ├── baseline_summary.json (required)                        │
│     └── rfl_summary.json (required)                             │
│                                                                 │
│  3. Telemetry Aggregates                                        │
│     └── telemetry_aggregate.json (required)                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 JSONL Log Schema (Per-Cycle Records)

Each line in `{baseline,rfl}_log.jsonl` MUST contain:

```json
{
  "cycle": "<int, required>",
  "timestamp_utc": "<ISO8601, required>",
  "slice_name": "<string, required>",
  "mode": "<'baseline'|'rfl', required>",

  "candidates": {
    "total": "<int, required>",
    "hashes": "<list[string], optional>"
  },

  "verified": {
    "count": "<int, required>",
    "hashes": "<list[string], optional>",
    "depths": "<list[int], required for Slice C>"
  },

  "abstained": {
    "count": "<int, required>"
  },

  "goals": {
    "target_hashes": "<list[string], required for Slice A/C/D>",
    "hit_count": "<int, required for Slice A/D>",
    "hit_hashes": "<list[string], optional>"
  },

  "policy": {
    "theta": "<list[float], required for RFL mode>",
    "theta_delta": "<list[float], required for RFL mode>",
    "gradient_norm": "<float, required for RFL mode>"
  },

  "metrics": {
    "abstention_rate": "<float, required>",
    "verification_density": "<float, required>",
    "goal_hit": "<bool, required for Slice A>",
    "joint_success": "<bool, required for Slice D>",
    "partial_coverage": "<float, required for Slice D>",
    "max_depth": "<int, optional>"
  },

  "H_t": "<string[64], required>"
}
```

**Validation Rules:**
- Missing required fields → Engine returns `INPUT_INVALID` error
- Type mismatches → Engine returns `INPUT_INVALID` error
- Empty log files → Engine returns `INPUT_INVALID` error

### 2.3 Summary File Schema

Each `{baseline,rfl}_summary.json` MUST contain:

```json
{
  "experiment_id": "<string, required>",
  "slice_name": "<string, required>",
  "mode": "<'baseline'|'rfl', required>",
  "total_cycles": "<int, required>",

  "metrics": {
    "goal_hit_rate": "<float, required for Slice A>",
    "mean_density": "<float, required for Slice B>",
    "mean_depth": "<float, required for Slice C>",
    "joint_success_rate": "<float, required for Slice D>",
    "mean_partial_coverage": "<float, required for Slice D>",
    "mean_abstention_rate": "<float, required>",
    "primary_metric": "<float, required>"
  },

  "time_series": {
    "abstention_rates": "<list[float], required>",
    "success_rates": "<list[float], required>",
    "densities": "<list[float], required for Slice B>",
    "depths": "<list[float], required for Slice C>"
  },

  "policy_final": {
    "theta": "<list[float], required for RFL>",
    "theta_norm": "<float, required for RFL>"
  }
}
```

### 2.4 Telemetry Aggregate Schema

The `telemetry_aggregate.json` MUST contain:

```json
{
  "experiment_id": "<string, required>",
  "generated_at": "<ISO8601, required>",

  "comparison": {
    "delta": "<float, required>",
    "ci_95_lower": "<float, required>",
    "ci_95_upper": "<float, required>",
    "ci_excludes_zero": "<bool, required>",
    "baseline_metric": "<float, required>",
    "rfl_metric": "<float, required>"
  },

  "diagnostics": {
    "policy_stability_index": "<float, required>",
    "oscillation_index": "<float, required>",
    "metric_stationary": "<bool, required>",
    "abstention_trend_tau": "<float, required>",
    "abstention_trend_p": "<float, required>",
    "gradient_norm_trend_tau": "<float, required>"
  },

  "patterns": {
    "detected_pattern": "<string, e.g., 'A.1', 'B.2'>",
    "pattern_confidence": "<float, 0-1>"
  },

  "validity": {
    "baseline_abstention_in_range": "<bool, required>",
    "sufficient_cycles": "<bool, required>",
    "determinism_verified": "<bool, required>"
  }
}
```

### 2.5 Input Validation Contract

The Engine MUST validate inputs before processing:

```python
class InputValidationResult:
    valid: bool
    errors: List[str]
    warnings: List[str]

def validate_inputs(
    baseline_log: Path,
    rfl_log: Path,
    baseline_summary: Path,
    rfl_summary: Path,
    telemetry: Path
) -> InputValidationResult:
    """
    Validate all inputs conform to schema.

    Returns:
        InputValidationResult with valid=True if all checks pass.

    Errors (block processing):
        - Missing required files
        - Missing required fields
        - Type mismatches
        - Empty data

    Warnings (allow processing):
        - Optional fields missing
        - Unexpected extra fields
    """
```

---

## 3. Output Contract

### 3.1 Output Schema: `conjecture_report.json`

The Engine produces exactly one output file:

```json
{
  "report_version": "1.0",
  "generated_at": "<ISO8601>",
  "experiment_id": "<string>",
  "slice_name": "<string>",

  "input_validation": {
    "status": "<VALID|INVALID>",
    "errors": ["<string>", ...],
    "warnings": ["<string>", ...]
  },

  "conjectures": {
    "conjecture_3_1": { ... },
    "conjecture_4_1": { ... },
    "conjecture_6_1": { ... },
    "conjecture_13_2": { ... },
    "conjecture_15_1": { ... },
    "conjecture_15_4": { ... },
    "lemma_2_1": { ... },
    "proposition_2_2": { ... }
  },

  "summary": {
    "total_evaluated": "<int>",
    "supports_count": "<int>",
    "consistent_count": "<int>",
    "contradicts_count": "<int>",
    "inconclusive_count": "<int>",
    "overall_assessment": "<string>"
  },

  "provenance": {
    "input_files": {
      "baseline_log": "<path>",
      "rfl_log": "<path>",
      "baseline_summary": "<path>",
      "rfl_summary": "<path>",
      "telemetry": "<path>"
    },
    "input_hashes": {
      "baseline_log_sha256": "<64-char hex>",
      "rfl_log_sha256": "<64-char hex>",
      "baseline_summary_sha256": "<64-char hex>",
      "rfl_summary_sha256": "<64-char hex>",
      "telemetry_sha256": "<64-char hex>"
    },
    "engine_version": "<string>",
    "theory_document_version": "<string>"
  }
}
```

### 3.2 Per-Conjecture Report Schema

Each conjecture entry in `conjectures` MUST contain:

```json
{
  "conjecture_id": "<string, e.g., 'conjecture_3_1'>",
  "name": "<string, e.g., 'Supermartingale Property'>",
  "theory_reference": "<string, e.g., 'RFL_UPLIFT_THEORY.md §3.1'>",

  "applicable": "<bool>",
  "applicability_reason": "<string, why applicable/not for this slice>",

  "observations": {
    "primary": {
      "metric": "<string, metric name>",
      "value": "<float|bool|string>",
      "source": "<string, field path in input>"
    },
    "secondary": [
      {
        "metric": "<string>",
        "value": "<float|bool|string>",
        "source": "<string>"
      }
    ]
  },

  "evaluation": {
    "rule_applied": "<string, rule ID from binding rules>",
    "threshold": "<float|null>",
    "threshold_source": "<string, theory reference>",
    "comparison": "<string, e.g., 'value > threshold'>",
    "result": "<bool>"
  },

  "evidence_status": "<SUPPORTS|CONSISTENT|CONTRADICTS|INCONCLUSIVE>",
  "evidence_rationale": "<string, explanation of status determination>",

  "diagnostics_used": [
    {
      "diagnostic_id": "<string>",
      "value": "<float|bool>",
      "interpretation": "<string>"
    }
  ],

  "caveats": ["<string>", ...]
}
```

### 3.3 Evidence Status Definitions

The Engine assigns exactly one of four statuses:

| Status | Code | Criteria |
|--------|------|----------|
| **SUPPORTS** | `S` | Observation matches prediction with p < 0.05 |
| **CONSISTENT** | `C` | Observation matches prediction but p ≥ 0.05 |
| **CONTRADICTS** | `X` | Observation contradicts prediction with p < 0.05 |
| **INCONCLUSIVE** | `I` | Cannot evaluate (invalid input, inapplicable, insufficient data) |

---

## 4. Binding Rules

### 4.1 Rule Structure

Each binding rule has the form:

```
RULE <ID>:
  CONJECTURE: <conjecture reference>
  APPLICABLE_WHEN: <slice condition>
  OBSERVE: <metric/field to read>
  THRESHOLD: <value from theory>
  THRESHOLD_SOURCE: <theory reference>
  EVALUATE:
    IF <condition> THEN <status>
    ELSE IF <condition> THEN <status>
    ...
```

### 4.2 Conjecture 3.1: Supermartingale Property

```
RULE R3.1:
  CONJECTURE: Conjecture 3.1 (Supermartingale Property)
  THEORY_REF: RFL_UPLIFT_THEORY.md §3.1, Theorem 3.1
  APPLICABLE_WHEN: All slices (A, B, C, D)

  OBSERVE:
    - PRIMARY: telemetry.diagnostics.abstention_trend_tau
    - SECONDARY: telemetry.diagnostics.abstention_trend_p
    - TERTIARY: summary.time_series.abstention_rates

  THRESHOLD: p < 0.05 for Mann-Kendall trend test
  THRESHOLD_SOURCE: Table 19.1 (RFL_UPLIFT_THEORY.md §19.4)

  EVALUATE:
    IF abstention_trend_tau < 0 AND abstention_trend_p < 0.05:
      THEN SUPPORTS
      RATIONALE: "Abstention rate shows statistically significant decreasing trend"

    ELSE IF abstention_trend_tau < 0 AND abstention_trend_p >= 0.05:
      THEN CONSISTENT
      RATIONALE: "Abstention rate decreasing but not statistically significant"

    ELSE IF abstention_trend_tau > 0 AND abstention_trend_p < 0.05:
      THEN CONTRADICTS
      RATIONALE: "Abstention rate shows statistically significant INCREASING trend"

    ELSE IF abstention_trend_tau ≈ 0 (|tau| < 0.05):
      THEN INCONCLUSIVE
      RATIONALE: "Abstention rate flat; insufficient signal for trend determination"
```

### 4.3 Conjecture 4.1: Logistic Decay

```
RULE R4.1:
  CONJECTURE: Conjecture 4.1 (Logistic Decay)
  THEORY_REF: RFL_UPLIFT_THEORY.md §4.2
  APPLICABLE_WHEN: Slice A (primary), Slice B (secondary)

  OBSERVE:
    - PRIMARY: Logistic fit R² from abstention time series
    - SECONDARY: Alternative model R² (linear, exponential)
    - TERTIARY: summary.time_series.abstention_rates

  THRESHOLD: R² > 0.8 for logistic fit
  THRESHOLD_SOURCE: Table 19.1 (RFL_UPLIFT_THEORY.md §19.4)

  EVALUATE:
    IF logistic_r2 > 0.80:
      THEN SUPPORTS
      RATIONALE: "Abstention curve fits logistic decay model (R² > 0.8)"

    ELSE IF logistic_r2 > 0.60 AND logistic_r2 > alternative_r2:
      THEN CONSISTENT
      RATIONALE: "Abstention curve partially fits logistic model"

    ELSE IF alternative_r2 > logistic_r2 + 0.10:
      THEN CONTRADICTS
      RATIONALE: "Alternative model (linear/step) fits better than logistic"

    ELSE:
      THEN INCONCLUSIVE
      RATIONALE: "No model fits well; data may be noisy or insufficient"

  NOTE: Engine must compute R² fits internally or receive pre-computed values
```

### 4.4 Conjecture 6.1: Almost Sure Convergence

```
RULE R6.1:
  CONJECTURE: Conjecture 6.1 (Almost Sure Convergence)
  THEORY_REF: RFL_UPLIFT_THEORY.md §6.1
  APPLICABLE_WHEN: All slices, after T_max cycles

  OBSERVE:
    - PRIMARY: telemetry.diagnostics.metric_stationary
    - SECONDARY: Final abstention rate from summary
    - TERTIARY: telemetry.diagnostics.policy_stability_index

  THRESHOLD: Stationarity at α → 0 (ADF p < 0.05)
  THRESHOLD_SOURCE: Definition 14.2 (RFL_UPLIFT_THEORY.md §14.2)

  EVALUATE:
    IF metric_stationary == true AND final_abstention_rate < 0.10:
      THEN SUPPORTS
      RATIONALE: "Abstention converged to near-zero level"

    ELSE IF metric_stationary == true AND final_abstention_rate >= 0.10:
      THEN CONSISTENT
      RATIONALE: "Metric stationary but not at zero; may be local optimum"

    ELSE IF metric_stationary == false AND policy_stability_index > 0.05:
      THEN CONTRADICTS
      RATIONALE: "Neither metric nor policy converged after T_max cycles"

    ELSE:
      THEN INCONCLUSIVE
      RATIONALE: "Convergence status unclear; may need more cycles"
```

### 4.5 Theorem 13.2: Multi-Goal Convergence

```
RULE R13.2:
  CONJECTURE: Theorem 13.2 (Multi-Goal RFL Convergence)
  THEORY_REF: RFL_UPLIFT_THEORY.md §13.4
  APPLICABLE_WHEN: Slice C (primary), Slice D (primary)

  OBSERVE:
    - PRIMARY: telemetry.diagnostics.policy_stability_index (Ψ)
    - SECONDARY: Primary metric trajectory
    - TERTIARY: summary.policy_final.theta_norm

  THRESHOLD: Ψ < 0.01 for convergence
  THRESHOLD_SOURCE: Convergence Rule 14.1 (RFL_UPLIFT_THEORY.md §14.1)

  EVALUATE:
    IF policy_stability_index < 0.01 AND primary_metric_improving:
      THEN SUPPORTS
      RATIONALE: "Policy converged (Ψ < 0.01) with improving metric"

    ELSE IF policy_stability_index < 0.05 AND primary_metric_stable:
      THEN CONSISTENT
      RATIONALE: "Policy nearly converged; may be at local optimum"

    ELSE IF policy_stability_index > 0.10:
      THEN CONTRADICTS
      RATIONALE: "Policy failed to converge (Ψ > 0.10)"

    ELSE IF theta_norm → ∞:
      THEN CONTRADICTS
      RATIONALE: "Policy parameters diverged"

    ELSE:
      THEN INCONCLUSIVE
      RATIONALE: "Convergence status unclear"
```

### 4.6 Theorem 15.1: Local Stability

```
RULE R15.1:
  CONJECTURE: Theorem 15.1 (Local Stability Criterion)
  THEORY_REF: RFL_UPLIFT_THEORY.md §15.2
  APPLICABLE_WHEN: All slices with RFL mode

  OBSERVE:
    - PRIMARY: Policy parameter trajectory (theta over time)
    - SECONDARY: summary.policy_final.theta_norm
    - TERTIARY: telemetry.diagnostics.oscillation_index

  THRESHOLD: θ bounded (no divergence)
  THRESHOLD_SOURCE: Theorem 15.1 (RFL_UPLIFT_THEORY.md §15.2)

  EVALUATE:
    IF theta_norm bounded AND oscillation_index < 0.20:
      THEN SUPPORTS
      RATIONALE: "Policy stable: bounded parameters, low oscillation"

    ELSE IF theta_norm bounded AND oscillation_index >= 0.20:
      THEN CONSISTENT
      RATIONALE: "Policy bounded but oscillating; may need momentum"

    ELSE IF theta_norm → ∞ OR theta contains NaN/Inf:
      THEN CONTRADICTS
      RATIONALE: "Policy diverged (unbounded growth)"

    ELSE:
      THEN INCONCLUSIVE
      RATIONALE: "Stability status unclear"
```

### 4.7 Conjecture 15.4: Basin Structure

```
RULE R15.4:
  CONJECTURE: Conjecture 15.4 (Basin Structure for U2 Slices)
  THEORY_REF: RFL_UPLIFT_THEORY.md §15.5
  APPLICABLE_WHEN: All slices (slice-specific predictions)

  OBSERVE:
    - PRIMARY: telemetry.patterns.detected_pattern
    - SECONDARY: Number of distinct convergence points (if multiple runs)
    - TERTIARY: Step-function behavior in depth (Slice C)

  PREDICTIONS:
    - Slice A: Single large basin (expect pattern A.1 or A.2)
    - Slice B: Multiple small basins (expect pattern B.1 or B.4)
    - Slice C: Nested basins (expect pattern C.1 with step jumps)
    - Slice D: Fragmented basins (expect pattern D.3 or D.4)

  EVALUATE:
    IF detected_pattern matches prediction for slice:
      THEN SUPPORTS
      RATIONALE: "Basin structure matches predicted pattern for {slice}"

    ELSE IF detected_pattern partially matches:
      THEN CONSISTENT
      RATIONALE: "Basin structure partially consistent with prediction"

    ELSE IF detected_pattern clearly contradicts:
      THEN CONTRADICTS
      RATIONALE: "Basin structure contradicts prediction (expected {X}, got {Y})"

    ELSE:
      THEN INCONCLUSIVE
      RATIONALE: "Insufficient data to assess basin structure"
```

### 4.8 Lemma 2.1: Variance Amplification

```
RULE R2.1:
  CONJECTURE: Lemma 2.1 (Variance Under Wide Slice)
  THEORY_REF: RFL_UPLIFT_THEORY.md §2.2
  APPLICABLE_WHEN: Slice B (primary), All slices (secondary)

  OBSERVE:
    - PRIMARY: Variance of verification density over time
    - SECONDARY: Early variance vs late variance comparison

  THRESHOLD: Var(δ_early) > Var(δ_late) for learning effect
  THRESHOLD_SOURCE: Lemma 2.1 proof sketch

  EVALUATE:
    IF early_density_variance > late_density_variance AND variance_reduction > 0.10:
      THEN SUPPORTS
      RATIONALE: "Variance reduced as policy specialized (learning effect)"

    ELSE IF early_density_variance ≈ late_density_variance:
      THEN CONSISTENT
      RATIONALE: "Variance stable; policy may not be learning density"

    ELSE IF early_density_variance < late_density_variance:
      THEN CONTRADICTS
      RATIONALE: "Variance increased; policy drifting away from optima"

    ELSE:
      THEN INCONCLUSIVE
      RATIONALE: "Insufficient density variation data"
```

### 4.9 Proposition 2.2: Learning Signal Correspondence

```
RULE R2.2:
  CONJECTURE: Proposition 2.2 (Entropy-Signal Correspondence)
  THEORY_REF: RFL_UPLIFT_THEORY.md §2.3
  APPLICABLE_WHEN: All slices

  OBSERVE:
    - PRIMARY: telemetry.comparison.delta (uplift)
    - SECONDARY: telemetry.comparison.ci_excludes_zero
    - TERTIARY: Variance of derivability (from density variance)

  THRESHOLD: Δ > 0 with CI excluding zero
  THRESHOLD_SOURCE: Definition 17.1 (RFL_UPLIFT_THEORY.md §17.4)

  EVALUATE:
    IF delta > 0 AND ci_excludes_zero:
      THEN SUPPORTS
      RATIONALE: "Positive uplift detected; higher variance yielded learning signal"

    ELSE IF delta > 0 AND NOT ci_excludes_zero:
      THEN CONSISTENT
      RATIONALE: "Positive trend but not statistically significant"

    ELSE IF delta <= 0 AND ci_excludes_zero:
      THEN CONTRADICTS
      RATIONALE: "Negative uplift; learning signal did not translate to improvement"

    ELSE:
      THEN INCONCLUSIVE
      RATIONALE: "Uplift near zero; signal-variance relationship unclear"
```

---

## 5. Binding Rule Summary Table

| Rule ID | Conjecture | Primary Metric | Threshold | Slice Applicability |
|---------|------------|----------------|-----------|---------------------|
| R3.1 | Conj 3.1 (Supermartingale) | `abstention_trend_tau` | p < 0.05 | A, B, C, D |
| R4.1 | Conj 4.1 (Logistic Decay) | `logistic_r2` | R² > 0.80 | A (primary), B |
| R6.1 | Conj 6.1 (A.S. Convergence) | `metric_stationary`, `final_α` | ADF p < 0.05, α → 0 | A, B, C, D |
| R13.2 | Thm 13.2 (Multi-Goal) | `policy_stability_index` | Ψ < 0.01 | C (primary), D |
| R15.1 | Thm 15.1 (Local Stability) | `theta_norm`, `oscillation_index` | bounded, O < 0.20 | A, B, C, D |
| R15.4 | Conj 15.4 (Basin Structure) | `detected_pattern` | matches prediction | A, B, C, D |
| R2.1 | Lemma 2.1 (Variance) | `density_variance` | early > late | B (primary) |
| R2.2 | Prop 2.2 (Learning Signal) | `delta`, `ci_excludes_zero` | Δ > 0, CI excl. 0 | A, B, C, D |

---

## 6. Engine Constraints

### 6.1 Prohibited Operations

The Conjecture Engine MUST NOT:

| Prohibition | Rationale |
|-------------|-----------|
| Invent new conjectures | Theory is fixed; only evaluate existing |
| Modify thresholds | Thresholds are preregistered in theory |
| Reinterpret conjecture meaning | Meanings are defined in RFL_UPLIFT_THEORY.md |
| Perform causal inference | Engine is correlational, not causal |
| Extrapolate beyond data | Only evaluate what data supports |
| Override CONTRADICTS with CONSISTENT | Evidence is what it is |
| Combine evidence across experiments | Each experiment evaluated independently |

### 6.2 Required Operations

The Conjecture Engine MUST:

| Requirement | Rationale |
|-------------|-----------|
| Validate all inputs | Garbage in → error, not garbage out |
| Cite sources for all values | Traceability |
| Apply rules deterministically | Reproducibility |
| Document all caveats | Epistemic honesty |
| Preserve provenance | Auditability |
| Hash input files | Reproducibility verification |

### 6.3 Error Handling

| Error Condition | Engine Response |
|-----------------|-----------------|
| Missing required input file | Return `INPUT_INVALID`, halt |
| Missing required field | Return `INPUT_INVALID`, halt |
| Type mismatch | Return `INPUT_INVALID`, halt |
| Division by zero | Set metric to `null`, status to `INCONCLUSIVE` |
| NaN/Inf in policy params | Set status to `CONTRADICTS` for stability conjectures |
| Empty time series | Set status to `INCONCLUSIVE` |

---

## 7. Engine Interface

### 7.1 Function Signature

```python
def evaluate_conjectures(
    baseline_log: Path,
    rfl_log: Path,
    baseline_summary: Path,
    rfl_summary: Path,
    telemetry_aggregate: Path,
    output_path: Path,
    *,
    engine_version: str = "1.0",
    theory_version: str = "RFL_UPLIFT_THEORY.md v2025-12-06"
) -> ConjectureReport:
    """
    Evaluate experimental data against theoretical conjectures.

    Args:
        baseline_log: Path to baseline JSONL log
        rfl_log: Path to RFL JSONL log
        baseline_summary: Path to baseline summary JSON
        rfl_summary: Path to RFL summary JSON
        telemetry_aggregate: Path to telemetry aggregate JSON
        output_path: Path to write conjecture_report.json
        engine_version: Version of this engine
        theory_version: Version of theory document used

    Returns:
        ConjectureReport object (also written to output_path)

    Raises:
        InputValidationError: If inputs fail validation
        EvaluationError: If evaluation fails unexpectedly
    """
```

### 7.2 Example Usage

```python
from conjecture_engine import evaluate_conjectures

report = evaluate_conjectures(
    baseline_log=Path("results/uplift_u2_goal_baseline.jsonl"),
    rfl_log=Path("results/uplift_u2_goal_rfl.jsonl"),
    baseline_summary=Path("results/uplift_u2_goal_baseline_summary.json"),
    rfl_summary=Path("results/uplift_u2_goal_rfl_summary.json"),
    telemetry_aggregate=Path("results/uplift_u2_goal_telemetry.json"),
    output_path=Path("results/uplift_u2_goal_conjecture_report.json")
)

# Check summary
print(f"Conjectures evaluated: {report.summary.total_evaluated}")
print(f"SUPPORTS: {report.summary.supports_count}")
print(f"CONTRADICTS: {report.summary.contradicts_count}")
```

---

## 8. Relationship to Other Documents

```
┌─────────────────────────────────────────────────────────────────┐
│                    DOCUMENT HIERARCHY                           │
└─────────────────────────────────────────────────────────────────┘

    RFL_UPLIFT_THEORY.md
    ├── Defines conjectures, thresholds, metrics
    │
    ├───▶ UPLIFT_THEORY_CONSISTENCY_MATRIX.md
    │     ├── Maps theory → implementation
    │     ├── Defines telemetry schema
    │     └── Defines manifest schema
    │
    └───▶ CONJECTURE_ENGINE_CONTRACT.md (this document)
          ├── Defines input contract (what engine receives)
          ├── Defines output contract (conjecture_report.json)
          ├── Defines binding rules (how to evaluate)
          └── Defines constraints (what engine cannot do)

    PREREG_UPLIFT_U2.yaml
    ├── Preregisters experiment parameters
    └── References thresholds from theory
```

---

## 9. Revision History

| Date | Revision | Author |
|------|----------|--------|
| 2025-12-06 | Initial creation | CLAUDE M |
| 2025-12-06 | Added binding rules R3.1–R2.2 | CLAUDE M |
| 2025-12-06 | Added constraints and error handling | CLAUDE M |

---

**PHASE II — NOT RUN IN PHASE I**

**THIS CONTRACT IS A DESIGN SPECIFICATION PENDING IMPLEMENTATION**
