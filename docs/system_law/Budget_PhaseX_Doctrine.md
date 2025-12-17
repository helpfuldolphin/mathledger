# Budget Phase X Doctrine

**Status**: Design Specification
**Phase**: X (SHADOW MODE ONLY)
**Version**: 1.2.0
**Date**: 2025-12-11

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Budget Drift and P3 Synthetic Stability](#2-budget-drift-and-p3-synthetic-stability)
3. [Budget Instability and P4 Divergence Interpretation](#3-budget-instability-and-p4-divergence-interpretation)
4. [GovernanceSignal Semantic Binding](#4-governancesignal-semantic-binding)
5. [Schema Surface](#5-schema-surface)
6. [Integration Points](#6-integration-points)
7. [P5 Runtime Expectations](#7-p5-runtime-expectations)
8. [TODO Anchors](#8-todo-anchors)
9. [Version History](#9-version-history)

---

## 1. Executive Summary

This document establishes the formal binding between budget governance and Phase X constraints. Budget admissibility is not merely a resource management concern—it is a structural constraint that affects both P3 synthetic stability measurements and P4 real-vs-twin divergence interpretation.

### Key Principles

| Principle | Description |
|-----------|-------------|
| **Budget as Constraint Surface** | Budget exhaustion affects the interpretation of stability metrics |
| **Drift Propagation** | Budget drift trajectories modulate Δp noise injection thresholds |
| **Divergence Context** | Budget instability provides essential context for P4 divergence severity |
| **GovernanceSignal Binding** | Budget layer emits GovernanceSignal objects for cross-layer aggregation |

### SHADOW MODE CONTRACT

All budget governance in Phase X operates in observation mode only:
- Budget constraints are **logged**, not **enforced** in active governance paths
- Drift trajectories are **computed**, not **acted upon**
- GovernanceSignals are **emitted**, not **consumed** for blocking decisions

---

## 2. Budget Drift and P3 Synthetic Stability

### 2.1 Problem Statement

P3 synthetic experiments (`FirstLightShadowRunner`) compute Δp metrics that represent learning curves. However, these metrics are sensitive to underlying resource availability. When budget drift occurs—whether through exhaustion, throttling, or allocation changes—the Δp signal becomes confounded with resource constraints rather than pure learning dynamics.

### 2.2 Budget Drift Definition

Budget drift is the deviation of actual budget utilization from expected utilization over a measurement window:

```
Budget Drift (BD) = Σ(actual_budget_spent[i] - expected_budget[i]) / window_size
```

Where:
- `actual_budget_spent[i]`: Cycles consumed in measurement window i
- `expected_budget[i]`: Nominal allocation for window i
- `window_size`: Number of windows in the measurement period

### 2.3 Impact on P3 Synthetic Stability

Budget drift affects P3 stability measurements through three mechanisms:

#### 2.3.1 Noise Floor Amplification

When budget drift is negative (under-utilization), the noise floor in Δp measurements increases because fewer cycles contribute to the signal:

```
Effective_Noise_Floor = Base_Noise / sqrt(1 - |BD|)
```

**Implication**: Negative Δp in budget-constrained runs may be noise, not signal.

#### 2.3.2 Success Rate Compression

Positive budget drift (over-utilization) can artificially compress success rates by including marginal attempts that would normally be excluded:

```
Compressed_Success_Rate = Raw_Success_Rate * (1 + BD * compression_factor)
```

Where `compression_factor` depends on the slice difficulty profile.

#### 2.3.3 RSI Stability Index Correlation

The Running Stability Index (ρ) in P3 is correlated with budget stability:

| Budget Drift Range | RSI Impact | Interpretation |
|--------------------|------------|----------------|
| BD ∈ [-0.05, 0.05] | Neutral | RSI reflects true stability |
| BD ∈ [-0.15, -0.05] | Depressed | RSI may under-report stability |
| BD ∈ [0.05, 0.15] | Inflated | RSI may over-report stability |
| |BD| > 0.15 | Confounded | RSI requires drift correction |

### 2.4 P3 Δp Noise Injection Binding

**TODO Anchor: P3-BUDGET-NOISE-001**

The P3 noise injection model should incorporate budget drift as a modulating factor:

```python
# TODO: P3-BUDGET-NOISE-001
# Integrate budget drift trajectory into noise injection model
#
# When budget_drift > threshold:
#   - Increase noise_floor in Δp computation
#   - Add uncertainty band to RSI readings
#   - Flag cycles in drift period with "budget_confounded" marker
#
# Reference: Budget_PhaseX_Doctrine.md Section 2.4
# Depends on: budget_drift_trajectory.schema.json
```

### 2.5 Drift Trajectory Schema Binding

Budget drift trajectory is captured in the `budget_drift_trajectory.schema.json` schema:

```
DriftTrajectory {
  window_index: int,
  expected_budget: float,
  actual_spent: float,
  drift_value: float,
  drift_classification: "STABLE" | "DRIFTING" | "VOLATILE",
  p3_noise_adjustment: float
}
```

---

## 3. Budget Instability and P4 Divergence Interpretation

### 3.1 Problem Statement

P4 (`FirstLightShadowRunnerP4`) compares real runner telemetry against shadow twin predictions. When the real runner operates under budget instability—exhaustion, throttling, or recovery—the divergence between real and twin may be caused by budget effects rather than model inadequacy.

### 3.2 Budget Instability Definition

Budget instability occurs when budget invariant snapshots show:
- `INV-BUD-1` violations (post-exhaustion processing)
- `INV-BUD-2` violations (hard cap exceeded)
- `INV-BUD-3` violations (negative remaining budget)
- Health score below 70.0
- Stability index below 0.7

### 3.3 Impact on P4 Divergence Interpretation

#### 3.3.1 Divergence Severity Adjustment

When budget instability is detected, P4 divergence severity should be adjusted:

| Budget Stability | Divergence Severity Multiplier |
|------------------|-------------------------------|
| STABLE (health ≥ 80, stability ≥ 0.95) | 1.0 (no adjustment) |
| DRIFTING (health ∈ [70, 80), stability ∈ [0.7, 0.95)) | 0.7 |
| VOLATILE (health < 70 or stability < 0.7) | 0.4 |

**Interpretation**: A SEVERE divergence during VOLATILE budget conditions should be logged as MODERATE for analysis purposes.

#### 3.3.2 Budget Confounding in Calibration Windows

**P5 Drift-Modulation Analysis:**

Calibration windows can be annotated with budget-aware modulation fields to distinguish resource-driven drift from model inadequacy:

**Modulation Fields:**
- `budget_confounded: bool` — True if budget constraints are confounding divergence measurements
- `effective_lr_adjustment: float` — Learning rate adjustment needed due to budget constraints (range [0.0, 1.0])
- `drift_classification: "NONE" | "TRANSIENT" | "PERSISTENT"` — Classification of resource-driven drift
- `budget_health_during_window: "SAFE" | "TIGHT" | "STARVED" | "UNKNOWN"` — Budget health during window

**Drift Classification:**
- **NONE**: Budget health is SAFE, no confounding expected
- **TRANSIENT**: Budget STARVED in single window (not frequently starved) → transient resource constraint
- **PERSISTENT**: Budget frequently starved (>50% of runs) → persistent resource constraint requiring calibration adjustment

**Effective LR Adjustment Formula:**
```
effective_lr_adjustment = max(0.0, 1.0 - (budget_exhausted_pct / 100.0))
```

When budget is exhausted, the effective learning rate should be reduced because the system is operating with reduced effective sample size. A budget exhaustion of 10% implies a 10% reduction in effective learning rate.

**Worked Example: Budget Confounding Falsely Inflating Divergence**

Consider a calibration window where:
- Real runner operates under budget exhaustion (STARVED, 15% exhausted)
- Twin model assumes infinite budget (no budget awareness)
- Observed divergence: `delta_p = 0.12` (WARN severity)

**Without Budget Modulation:**
- Divergence interpreted as model inadequacy
- Calibration would attempt to adjust twin parameters
- False signal: model appears to need recalibration

**With Budget Modulation:**
- `budget_confounded = True`
- `effective_lr_adjustment = 0.85` (15% exhaustion → 15% LR reduction)
- `drift_classification = "TRANSIENT"` (single window, not frequently starved)
- Adjusted divergence interpretation: `delta_p_adjusted = 0.12 * 0.7 = 0.084` (INFO severity)

**Interpretation:**
The observed divergence is primarily due to budget constraints, not model inadequacy. The twin model is operating correctly; the real runner is resource-constrained. Calibration should NOT adjust twin parameters based on this window.

**Persistent vs Transient Distinction:**
- **Transient** (single window STARVED): Divergence likely environmental noise, ignore for calibration
- **Persistent** (frequently starved): Divergence reflects structural resource constraint, may require twin model to incorporate budget awareness in future phases

#### 3.3.3 Calibration Exclusion Recommendations

**Cross-Signal Check Logic:**

Calibration exclusion is recommended ONLY when ALL of the following conditions are met:
1. **Budget is confounded** (`budget_confounded = True`)
2. **PRNG is NOT volatile** (`drift_status != "VOLATILE"` or missing)
3. **Topology is stable** (`pressure_band != "HIGH"` or missing)

This ensures we only exclude windows when budget is the primary confounding factor, not when multiple signals indicate instability.

**Exclusion Recommendation Fields:**
- `calibration_exclusion_recommended: bool` — True if window should be excluded from calibration
- `exclusion_reason: "BUDGET_CONFOUNDED_TRANSIENT" | "BUDGET_CONFOUNDED_PERSISTENT" | None` — Reason for exclusion
- `cross_signal_checks: {budget_confounded: bool, prng_not_volatile: bool, topology_stable: bool}` — Individual check results

**Interpretation:**
- **`BUDGET_CONFOUNDED_TRANSIENT`**: Single window budget starvation, other signals stable → exclude from calibration
- **`BUDGET_CONFOUNDED_PERSISTENT`**: Frequently starved budget, other signals stable → exclude from calibration
- **No exclusion**: If PRNG is volatile OR topology is high pressure, do NOT exclude (multiple confounding factors)

**Advisory Only:**
- These recommendations are **advisory only** — no automatic filtering
- Calibration systems should use these recommendations to inform manual review
- Windows with `calibration_exclusion_recommended = True` should be flagged for human review before exclusion

**Exclusion Trace (Auditability):**

Each exclusion recommendation includes an `exclusion_trace` field that provides a complete audit trail of the decision logic:

```json
{
  "exclusion_trace": {
    "missing_signal_policy": "DEFAULT_TRUE_MISSING",
    "checks": {
      "budget_confounded": {
        "value": true,
        "source": "budget_modulation",
        "raw_value": "true"
      },
      "prng_not_volatile": {
        "value": true,
        "source": "prng_signal",
        "raw_value": "STABLE"
      },
      "topology_stable": {
        "value": true,
        "source": "DEFAULT_TRUE_MISSING",
        "raw_value": "UNKNOWN"
      }
    },
    "decision": true,
    "reason": "BUDGET_CONFOUNDED_TRANSIENT",
    "thresholds": {
      "prng_volatile_threshold": "VOLATILE",
      "topology_high_pressure_threshold": "HIGH"
    }
  }
}
```

**Trace Fields:**
- `missing_signal_policy`: Policy for handling missing signals (currently `"DEFAULT_TRUE_MISSING"`)
- `checks`: Individual check results with `value` (boolean), `source` (data source), and `raw_value` (original signal value)
- `decision`: Final boolean decision (matches `calibration_exclusion_recommended`)
- `reason`: Exclusion reason if decision is True, null otherwise
- `thresholds`: Constants used in decision logic (e.g., "VOLATILE" for PRNG, "HIGH" for topology)

**Missing Signal Policy: `DEFAULT_TRUE_MISSING`**

**Current Policy:** When PRNG or topology signals are missing, the system defaults to `true` (safe assumption), which makes exclusion easier when budget is confounded.

**Rationale for Default-True:**
- **Conservative approach**: Missing signals are assumed safe (not volatile/not high pressure)
- **Budget-first logic**: If budget is confounded and other signals are unknown, we err on the side of excluding the window (budget is the known confounding factor)
- **Reduces false negatives**: Better to exclude a window that might be OK than to include a window that is definitely budget-confounded
- **Auditability**: The trace explicitly shows `source: "DEFAULT_TRUE_MISSING"` so reviewers know when defaults were used

**When to Override to Default-False (Future):**
In future phases, if missing signals should be treated as blocking (more conservative), the policy could be changed to `"DEFAULT_FALSE_MISSING"`:
- **Use case**: When signal availability is required for calibration decisions
- **Behavior**: Missing signals would default to `false`, preventing exclusion unless all signals are explicitly provided
- **Trade-off**: More conservative (fewer exclusions) but requires complete signal coverage

**Missing Signal Handling:**
When PRNG or topology signals are missing, the trace explicitly shows:
- `source: "DEFAULT_TRUE_MISSING"` — Indicates signal was not provided
- `raw_value: "UNKNOWN"` — Indicates missing signal
- `value: true` — Default behavior per `missing_signal_policy`

**Trace Determinism:**
- All trace fields use sorted keys for deterministic JSON serialization
- Same inputs always produce identical trace structure
- Dictionary ordering is consistent across runs

This ensures reviewers can reconstruct the exact logic that led to each exclusion recommendation, including when and why defaults were applied.

#### 3.3.2 Divergence Root Cause Attribution

P4 divergence root cause analysis must include budget state:

```
Divergence Root Cause Vector:
1. Model prediction error (twin inaccuracy)
2. Real runner anomaly (non-budget related)
3. Budget-induced divergence (resource constraint)
4. Combined (multiple factors)
```

When budget instability is present, attributing divergence to "Budget-induced" before "Model prediction error" is the conservative interpretation.

#### 3.3.3 Twin Prediction Context

The shadow twin in P4 operates without budget awareness. This asymmetry means:

- Twin predictions assume infinite budget
- Real runner operates under finite budget
- Divergence includes budget constraint effects

**TODO Anchor: P4-BUDGET-TDA-001**

```python
# TODO: P4-BUDGET-TDA-001
# Modulate TDA divergence interpretation based on budget stability
#
# When budget_stability_class == "VOLATILE":
#   - Adjust divergence severity by 0.4 multiplier
#   - Add "budget_instability" to root_cause_vector
#   - Flag divergence as "budget_confounded"
#
# When budget_stability_class == "DRIFTING":
#   - Adjust divergence severity by 0.7 multiplier
#   - Consider budget drift as contributing factor
#
# Reference: Budget_PhaseX_Doctrine.md Section 3.3
# Depends on: budget_governance_signal.schema.json
```

### 3.4 Budget-Modulated TDA Binding

The TDA (Topological Data Analysis) interpretation layer should consume budget signals:

```
TDA Context {
  divergence_raw: float,
  budget_stability_class: "STABLE" | "DRIFTING" | "VOLATILE",
  divergence_adjusted: float,  // divergence_raw * severity_multiplier
  budget_confounded: bool,
  attribution: "MODEL" | "RUNNER" | "BUDGET" | "COMBINED"
}
```

---

## 4. GovernanceSignal Semantic Binding

### 4.1 Budget Layer GovernanceSignal

The budget layer emits `GovernanceSignal` objects for cross-layer aggregation. The semantic contract is:

```python
GovernanceSignal(
    layer="budget",
    status="ok" | "warn" | "block",
    severity="info" | "warning" | "critical",
    message="descriptive message",
    reasons=["reason1", "reason2", ...],
    metadata={
        "stability_class": "STABLE" | "DRIFTING" | "VOLATILE",
        "health_score": float,
        "stability_index": float,
        "drift_trajectory": [...],
        "inv_bud_failures": [...],
    }
)
```

### 4.2 Signal Status Mapping

| Budget Condition | Signal Status | Signal Severity |
|------------------|---------------|-----------------|
| All invariants OK, health ≥ 80, stability ≥ 0.95 | ok | info |
| Minor drift, health ∈ [70, 80) | warn | warning |
| INV-BUD-1/2/3 failure | block | critical |
| Health < 70 or stability < 0.7 | block | critical |

### 4.3 Cross-Layer Consumption

Other layers consume budget GovernanceSignals for context:

| Consumer Layer | Consumption Pattern |
|----------------|---------------------|
| P3 First-Light | Noise floor adjustment, RSI correction |
| P4 Divergence | Severity multiplier, root cause attribution |
| TDA | Divergence interpretation context |
| Director Console | Status light, headline generation |

### 4.4 Signal Emission Pattern

```python
def emit_budget_governance_signal(
    timeline: Dict[str, Any],
    health: Dict[str, Any],
) -> GovernanceSignal:
    """
    Emit GovernanceSignal for budget layer.

    SHADOW MODE: Signal is emitted but not consumed for blocking.
    """
    view = build_budget_invariants_governance_view(timeline, health)

    if view["combined_status"] == "BLOCK":
        return GovernanceSignal(
            layer=LAYER_BUDGET,
            status="block",
            severity="critical",
            message=f"Budget invariant violation: {view['inv_bud_failures']}",
            reasons=view["inv_bud_failures"],
            metadata={
                "stability_class": "VOLATILE",
                "health_score": view["health_score"],
                "stability_index": view["stability_index"],
            }
        )
    elif view["combined_status"] == "WARN":
        return GovernanceSignal(
            layer=LAYER_BUDGET,
            status="warn",
            severity="warning",
            message="Budget stability below threshold",
            reasons=["stability_drift"],
            metadata={
                "stability_class": "DRIFTING",
                "health_score": view["health_score"],
                "stability_index": view["stability_index"],
            }
        )
    else:
        return GovernanceSignal(
            layer=LAYER_BUDGET,
            status="ok",
            severity="info",
            message="Budget invariants nominal",
            reasons=[],
            metadata={
                "stability_class": "STABLE",
                "health_score": view["health_score"],
                "stability_index": view["stability_index"],
            }
        )
```

---

## 5. Schema Surface

### 5.1 Schema Catalog

| Schema | Purpose | Location |
|--------|---------|----------|
| `budget_drift_trajectory.schema.json` | Per-window drift measurements | `docs/system_law/schemas/budget/` |
| `budget_governance_signal.schema.json` | GovernanceSignal for budget layer | `docs/system_law/schemas/budget/` |
| `budget_director_panel.schema.json` | Director Console tile structure | `docs/system_law/schemas/budget/` |

### 5.2 Schema Interoperability

All budget schemas are designed for interoperability with:
- P3 First-Light schemas (`first_light_*.schema.json`)
- P4 schemas (`p4_divergence_log.schema.json`, `p4_twin_trajectory.schema.json`)
- Global health schemas
- GovernanceSignal structures in `governance_verifier.py`

### 5.3 Schema Version Contract

All schemas use semantic versioning:
- Major: Breaking changes to required fields
- Minor: New optional fields
- Patch: Documentation updates

Current version: **1.0.0**

---

## 6. Integration Points

### 6.1 File Locations

| Component | File | Purpose |
|-----------|------|---------|
| Budget Invariants | `derivation/budget_invariants.py` | Core snapshot/timeline/storyline logic |
| GovernanceSignal | `backend/analytics/governance_verifier.py` | Signal emission/consumption |
| P3 Integration | `backend/topology/first_light/` | First-Light shadow runner |
| P4 Integration | `backend/topology/first_light/runner_p4.py` | P4 real coupling |

### 6.2 Data Flow

```
Budget Invariant Snapshot
         │
         ▼
Budget Invariant Timeline
         │
         ▼
Budget Governance View ──────────────────┐
         │                               │
         ▼                               ▼
GovernanceSignal emission         Director Panel
         │
    ┌────┴────────────────┐
    │                     │
    ▼                     ▼
P3 Noise Injection   P4 Divergence Interpretation
```

### 6.3 Dependency Graph

```
budget_invariants.py
         │
         ├── governance_verifier.py (GovernanceSignal)
         │         │
         │         ├── P3 First-Light (noise modulation)
         │         └── P4 Runner (divergence context)
         │
         └── Director Console (status tile)
```

---

## 7. P5 Runtime Expectations

### 7.1 Transition from Synthetic to Real Telemetry

P3 and P4 operate on synthetic or controlled telemetry. P5 introduces **real runtime telemetry** from production-like environments. This transition fundamentally changes how budget metrics should be interpreted.

#### 7.1.1 Expected Variability Increase

When moving from synthetic (P3/P4) to real telemetry (P5), expect:

| Metric | Synthetic (P3/P4) | Real (P5) | Interpretation |
|--------|-------------------|-----------|----------------|
| Drift variance | Low (controlled) | High (environmental) | Wider normal bands needed |
| Health score stability | Steady | Fluctuating | Rolling averages required |
| Stability index | Predictable | Noisy | Longer measurement windows |
| Invariant violations | Rare, systematic | Occasional, transient | Distinguish transient from structural |

**Key insight**: A drift value of 0.08 in P3 is meaningful signal. The same value in P5 may be environmental noise.

#### 7.1.2 False "Budget Drift" Under Heavy Load

Real runtime introduces load-dependent budget behavior that can trigger false drift classifications:

| Condition | Symptom | Risk | Mitigation |
|-----------|---------|------|------------|
| **Burst load** | Sudden spike in `actual_spent` | False DRIFTING classification | Use windowed smoothing |
| **GC pauses** | Intermittent budget exhaustion | False INV-BUD-1 violations | Exclude GC windows from invariant checks |
| **Resource contention** | Oscillating health score | False VOLATILE classification | Require sustained instability (≥3 windows) |
| **Cold start** | Initial over-allocation | False positive drift | Exclude warmup period from baseline |

#### 7.1.3 Recommended P5 Calibration

For P5 real telemetry, the following adjustments are recommended (spec only, not implemented):

```
P5 Calibration Parameters (Advisory):
- Drift classification thresholds: multiply by 1.5× (wider bands)
- Stability window size: increase from 50 to 100 cycles
- Health score smoothing: apply 5-window rolling average
- Invariant violation persistence: require 2+ consecutive violations before flagging
- Warmup exclusion: first 20 cycles excluded from baseline computation
```

#### 7.1.4 Distinguishing Signal from Noise

In P5, the primary challenge is distinguishing:
1. **Structural budget drift** (actual resource constraint) from **transient fluctuation** (environmental noise)
2. **Model inadequacy divergence** from **load-induced divergence**

Recommended heuristics:
- Drift persisting across ≥3 consecutive windows → likely structural
- Drift appearing in single window during load spike → likely transient
- Divergence correlated with load metrics → likely load-induced
- Divergence uncorrelated with load → likely model inadequacy

---

### 7.2 Enablement Note: From Advisory to Influencing

**Current Stance**: All budget multipliers operate in **SHADOW MODE**—advisory only, never influencing severity classifications or blocking decisions.

#### 7.2.1 Preconditions for Enablement

Codex MAY transition budget multipliers from advisory to influencing severity **only after** the following conditions are satisfied:

| Precondition | Verification | Rationale |
|--------------|--------------|-----------|
| **P5 Baseline Established** | ≥1000 cycles of real telemetry collected with budget metrics | Ensures sufficient data to calibrate thresholds |
| **False Positive Rate ≤ 5%** | Budget drift classifications validated against manual review | Prevents spurious severity reductions |
| **False Negative Rate ≤ 2%** | Actual budget-induced divergences correctly attributed | Ensures real budget issues are caught |
| **Calibration Coefficients Validated** | Noise multiplier and severity multiplier values validated against P5 data | Prevents over/under-correction |
| **Governance Board Approval** | Explicit authorization from governance stakeholders | Maintains human oversight |
| **Rollback Mechanism Tested** | Ability to revert to advisory-only mode within 1 cycle | Ensures safety escape hatch |

#### 7.2.2 Enablement Sequence

If all preconditions are met, enablement proceeds in stages:

```
Stage 1: Shadow-with-Logging (current)
  - Multipliers computed and logged
  - No influence on severity
  - Duration: Until P5 baseline established

Stage 2: Shadow-with-Comparison
  - Compute both advisory and would-be-influenced severity
  - Log delta for analysis
  - No actual influence
  - Duration: Until false positive/negative rates validated

Stage 3: Soft Influence
  - Budget multipliers influence severity classification
  - All adjustments logged with "budget_adjusted" flag
  - Governance alerts on multiplier application
  - Duration: Until governance board approval

Stage 4: Full Enablement
  - Budget multipliers fully integrated into severity pipeline
  - Standard logging (no special flags)
  - Rollback mechanism remains available
```

#### 7.2.3 Explicit Non-Enablement Conditions

Budget multipliers MUST NOT be enabled if:
- P5 false positive rate exceeds 5%
- Any invariant violation rate exceeds 10% of cycles
- Governance board has not explicitly approved
- Rollback mechanism is not operational
- Budget telemetry source is not cryptographically attested

#### 7.2.4 Future Phase Placeholder

This enablement path is documented for future phases. **No code implementing enablement should be written** until:
1. P5 baseline is established
2. Preconditions are verified
3. Governance board authorizes implementation

The current implementation MUST remain advisory-only (SHADOW MODE).

---

### 7.3 P5 Budget Calibration Experiment

This section defines a concrete experiment to measure and calibrate budget false-positive rates under real telemetry conditions.

#### 7.3.1 Experiment Design: Three-Phase Approach

The calibration experiment proceeds in three phases, each with distinct load characteristics:

```
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 1: BASELINE        PHASE 2: CONTROLLED      PHASE 3: STRESS │
│  (Synthetic)              (Real, Normal Load)      (Real, Heavy)   │
│                                                                     │
│  ┌─────────────┐          ┌─────────────┐          ┌─────────────┐ │
│  │ Known       │          │ Production  │          │ 2× nominal  │ │
│  │ budget      │   ───►   │ baseline    │   ───►   │ load +      │ │
│  │ behavior    │          │ load        │          │ burst       │ │
│  └─────────────┘          └─────────────┘          └─────────────┘ │
│                                                                     │
│  Duration: 500 cycles     Duration: 1000 cycles    Duration: 500   │
│  Goal: Establish ground   Goal: Measure FP/FN      Goal: Stress    │
│        truth                    under normal             test      │
└─────────────────────────────────────────────────────────────────────┘
```

##### Phase 1: Baseline (Synthetic)

**Purpose**: Establish ground truth for budget classification accuracy.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Duration | 500 cycles | Sufficient for statistical significance |
| Load profile | Synthetic, deterministic | Known budget behavior |
| Injected faults | 10 known budget exhaustions | Ground truth for detection |
| Expected drift | Controlled: 5 STABLE, 3 DRIFTING, 2 VOLATILE windows | Verify classification accuracy |

**Success Criteria**:
- Classification accuracy ≥ 99% (known behavior)
- All 10 injected faults detected
- Zero spurious VOLATILE classifications

##### Phase 2: Controlled Real Load

**Purpose**: Measure false positive/negative rates under production-like conditions.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Duration | 1000 cycles | P5 baseline requirement |
| Load profile | Production baseline (1× nominal) | Representative real behavior |
| Manual review | Sample 100 cycles for ground truth | Validate classifications |
| Expected variance | ±20% from synthetic baseline | Account for environmental noise |

**Success Criteria**:
- False Positive Rate ≤ 5%
- False Negative Rate ≤ 2%
- No sustained false VOLATILE classifications (≥3 consecutive)

##### Phase 3: Stress Load

**Purpose**: Validate calibration under extreme conditions and identify failure modes.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Duration | 500 cycles | Focused stress testing |
| Load profile | 2× nominal + burst injection | Expose edge cases |
| Burst injection | 50ms bursts every 20 cycles | Simulate real-world spikes |
| GC simulation | Forced GC every 100 cycles | Test GC exclusion logic |

**Success Criteria**:
- FP rate ≤ 10% (relaxed for stress)
- FN rate ≤ 5% (relaxed for stress)
- Recovery to STABLE within 5 cycles after burst
- No false INV-BUD-1 violations during GC windows

---

#### 7.3.2 Measurement Metrics

##### FP/FN Measurement Method

**False Positive (FP)**: Budget classified as DRIFTING/VOLATILE when actually STABLE.

```
FP Detection Method:
1. Log all drift classifications with timestamp
2. Correlate with actual budget utilization metrics
3. Manual review sample: If actual_spent within 5% of expected AND health ≥ 80
   AND stability ≥ 0.95, classification should be STABLE
4. Any non-STABLE classification in this condition = FP

FP_Rate = (FP_count / total_non_stable_classifications) × 100%
```

**False Negative (FN)**: Budget classified as STABLE when actually DRIFTING/VOLATILE.

```
FN Detection Method:
1. Log all STABLE classifications
2. Identify cycles with known budget stress:
   - actual_spent > 1.15 × expected (15% over-utilization)
   - health < 70 OR stability < 0.7
   - Any INV-BUD-* violation occurred
3. STABLE classification during these conditions = FN

FN_Rate = (FN_count / total_actual_stress_cycles) × 100%
```

##### Logging Fields Required

Each cycle MUST log the following fields for calibration analysis:

```json
{
  "calibration_log": {
    "cycle": 1234,
    "timestamp": "2025-12-11T10:30:00Z",
    "phase": "PHASE_2_CONTROLLED",

    "budget_metrics": {
      "expected_budget": 100.0,
      "actual_spent": 103.5,
      "drift_value": 0.035,
      "health_score": 82.3,
      "stability_index": 0.91
    },

    "classification": {
      "drift_class": "STABLE",
      "stability_class": "DRIFTING",
      "noise_multiplier": 1.0,
      "severity_multiplier": 0.7,
      "admissibility_hint": "WARN"
    },

    "ground_truth": {
      "manual_label": null,
      "injected_fault": false,
      "known_stress": false
    },

    "environment": {
      "load_factor": 1.0,
      "gc_occurred": false,
      "burst_active": false,
      "warmup_period": false
    },

    "derived": {
      "fp_candidate": false,
      "fn_candidate": false,
      "review_required": false
    }
  }
}
```

##### Per-Phase Logging Requirements

| Phase | Log Frequency | Manual Review Sample | Ground Truth Source |
|-------|---------------|---------------------|---------------------|
| Phase 1 | Every cycle | 100% (all cycles) | Injected faults |
| Phase 2 | Every cycle | 10% random sample | Manual labeling |
| Phase 3 | Every cycle | 20% + all anomalies | Manual + injection |

---

#### 7.3.3 Enablement Gate Checklist

The following checklist transforms Section 7.2 preconditions into actionable engineering steps:

```
╔══════════════════════════════════════════════════════════════════════╗
║           BUDGET P5 ENABLEMENT GATE CHECKLIST                        ║
║           Engineer: ________________  Date: ________________         ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  PHASE 1 COMPLETION                                          □ Pass  ║
║  ├─ □ 500 synthetic cycles completed                                 ║
║  ├─ □ Classification accuracy ≥ 99%                                  ║
║  ├─ □ All injected faults detected (10/10)                          ║
║  └─ □ Zero spurious VOLATILE classifications                         ║
║                                                                      ║
║  PHASE 2 COMPLETION                                          □ Pass  ║
║  ├─ □ 1000 real-load cycles completed                               ║
║  ├─ □ Manual review of 100-cycle sample completed                   ║
║  ├─ □ FP Rate ≤ 5% (Measured: ____%)                                ║
║  ├─ □ FN Rate ≤ 2% (Measured: ____%)                                ║
║  └─ □ No sustained false VOLATILE (max consecutive: ____)           ║
║                                                                      ║
║  PHASE 3 COMPLETION                                          □ Pass  ║
║  ├─ □ 500 stress cycles completed                                    ║
║  ├─ □ FP Rate ≤ 10% under stress (Measured: ____%)                  ║
║  ├─ □ FN Rate ≤ 5% under stress (Measured: ____%)                   ║
║  ├─ □ Recovery to STABLE ≤ 5 cycles (Measured: ____ cycles)         ║
║  └─ □ No false INV-BUD-1 during GC windows                          ║
║                                                                      ║
║  CALIBRATION VALIDATION                                      □ Pass  ║
║  ├─ □ Noise multiplier values validated against P5 data             ║
║  │     STABLE→1.0: Actual effect ____% (target: 0%)                 ║
║  │     DRIFTING→1.3: Actual effect ____% (target: 30%)              ║
║  │     DIVERGING→1.6: Actual effect ____% (target: 60%)             ║
║  │     CRITICAL→2.0: Actual effect ____% (target: 100%)             ║
║  ├─ □ Severity multiplier values validated                          ║
║  │     STABLE→1.0: Severity unchanged ____% (target: 100%)          ║
║  │     DRIFTING→0.7: Severity reduced ____% (target: ~30%)          ║
║  │     VOLATILE→0.4: Severity reduced ____% (target: ~60%)          ║
║  └─ □ Coefficient adjustment recommendations documented             ║
║                                                                      ║
║  INFRASTRUCTURE                                              □ Pass  ║
║  ├─ □ Rollback mechanism tested (time to rollback: ____ sec)        ║
║  ├─ □ Rollback completes within 1 cycle                             ║
║  ├─ □ Budget telemetry source cryptographically attested            ║
║  ├─ □ Calibration logs archived with checksums                      ║
║  └─ □ Monitoring dashboards configured                              ║
║                                                                      ║
║  GOVERNANCE                                                  □ Pass  ║
║  ├─ □ Calibration report submitted to governance board              ║
║  ├─ □ FP/FN rates reviewed and accepted                             ║
║  ├─ □ Coefficient adjustments approved (if any)                     ║
║  ├─ □ Enablement authorization signed                               ║
║  └─ □ Rollback authority designated                                 ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  FINAL GATE                                                          ║
║  ├─ □ All Phase boxes checked                                        ║
║  ├─ □ All measured values within tolerance                          ║
║  ├─ □ Governance approval received                                   ║
║  └─ □ Enablement authorized: YES / NO                               ║
║                                                                      ║
║  Approver: ________________  Signature: ________________            ║
║  Date: ________________                                              ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

#### 7.3.4 Smoke-Test Readiness Checklist

Before initiating the P5 Budget Calibration Experiment, verify the following:

```
╔══════════════════════════════════════════════════════════════════════╗
║     SMOKE-TEST READINESS: P5 BUDGET CALIBRATION EXPERIMENT          ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  CODE READINESS                                                      ║
║  ├─ □ budget_binding.py deployed and importable                     ║
║  ├─ □ build_budget_risk_summary_for_p3() callable                   ║
║  ├─ □ build_budget_context_for_p4() callable                        ║
║  ├─ □ attach_budget_risk_to_evidence() callable                     ║
║  ├─ □ All 64 budget_binding tests passing                           ║
║  └─ □ No regressions in existing test suite                         ║
║                                                                      ║
║  LOGGING INFRASTRUCTURE                                              ║
║  ├─ □ Calibration log schema implemented                            ║
║  ├─ □ Log rotation configured (prevent disk exhaustion)             ║
║  ├─ □ Log ingestion pipeline verified                               ║
║  ├─ □ Query interface for FP/FN analysis available                  ║
║  └─ □ Sample log entry validated against schema                     ║
║                                                                      ║
║  SYNTHETIC BASELINE (Phase 1 Prep)                                   ║
║  ├─ □ Synthetic budget generator configured                         ║
║  ├─ □ Fault injection mechanism tested                              ║
║  ├─ □ 10 fault scenarios defined and documented                     ║
║  ├─ □ Ground truth labels pre-generated                             ║
║  └─ □ 500-cycle synthetic run completes without error               ║
║                                                                      ║
║  REAL LOAD ENVIRONMENT (Phase 2/3 Prep)                              ║
║  ├─ □ Production-like environment available                         ║
║  ├─ □ Load generator configured for 1× and 2× nominal               ║
║  ├─ □ Burst injection mechanism tested                              ║
║  ├─ □ GC simulation trigger available                               ║
║  ├─ □ Environment isolated from production                          ║
║  └─ □ Resource limits match production specs                        ║
║                                                                      ║
║  MEASUREMENT TOOLING                                                 ║
║  ├─ □ FP detection query implemented                                ║
║  ├─ □ FN detection query implemented                                ║
║  ├─ □ Manual review interface available                             ║
║  ├─ □ Sample labeling workflow documented                           ║
║  └─ □ Statistical analysis scripts ready                            ║
║                                                                      ║
║  ROLLBACK MECHANISM                                                  ║
║  ├─ □ Rollback procedure documented                                 ║
║  ├─ □ Rollback tested in isolation                                  ║
║  ├─ □ Rollback time measured: _____ seconds                         ║
║  ├─ □ Rollback authority identified: _________________              ║
║  └─ □ Emergency contact list current                                ║
║                                                                      ║
║  DOCUMENTATION                                                       ║
║  ├─ □ Experiment runbook written                                    ║
║  ├─ □ Phase transition criteria documented                          ║
║  ├─ □ Abort criteria documented                                     ║
║  ├─ □ Success criteria documented                                   ║
║  └─ □ Post-experiment analysis plan ready                           ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  SMOKE TEST EXECUTION                                                ║
║  ├─ □ Run 10 synthetic cycles → logs generated correctly            ║
║  ├─ □ Run 10 real-load cycles → metrics captured                    ║
║  ├─ □ Inject 1 fault → detected and logged                          ║
║  ├─ □ Trigger 1 burst → classification logged                       ║
║  ├─ □ Execute rollback → completes successfully                     ║
║  └─ □ Generate sample FP/FN report → numbers plausible              ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  SMOKE TEST RESULT: □ PASS  □ FAIL                                  ║
║                                                                      ║
║  If PASS: Proceed to Phase 1 of calibration experiment              ║
║  If FAIL: Document blockers and remediate before proceeding         ║
║                                                                      ║
║  Verified by: ________________  Date: ________________               ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

#### 7.3.5 Abort Criteria

The calibration experiment MUST be aborted if any of the following occur:

| Condition | Action | Recovery |
|-----------|--------|----------|
| FP Rate > 20% in Phase 2 | Abort, return to calibration | Review thresholds, widen bands |
| FN Rate > 10% in Phase 2 | Abort, return to calibration | Review detection logic |
| System instability during stress | Abort Phase 3 | Reduce load, investigate |
| Data loss or corruption | Abort all phases | Restore from checkpoint, restart |
| Resource exhaustion | Pause experiment | Scale resources, resume |

---

## 8. TODO Anchors

### 8.1 P3 Budget-Noise Integration

**Anchor**: `P3-BUDGET-NOISE-001`
**Location**: `backend/topology/first_light/delta_p_computer.py` (future)
**Priority**: Phase X P3 Execution Authorization

```python
# TODO: P3-BUDGET-NOISE-001
# Budget risk feed into P3 Δp noise injection
#
# Implementation:
# 1. Import budget_drift_trajectory from budget_invariants
# 2. Compute drift-adjusted noise floor
# 3. Apply noise floor to Δp slope calculation
# 4. Add uncertainty bands to RSI readings
# 5. Emit "budget_confounded" markers in cycle logs
#
# Reference: docs/system_law/Budget_PhaseX_Doctrine.md Section 2.4
# Schema: budget_drift_trajectory.schema.json
```

### 8.2 P4 Budget-TDA Interpretation

**Anchor**: `P4-BUDGET-TDA-001`
**Location**: `backend/topology/first_light/divergence_analyzer.py` (future)
**Priority**: Phase X P4 Execution Authorization

```python
# TODO: P4-BUDGET-TDA-001
# Budget-modulated TDA interpretation
#
# Implementation:
# 1. Import GovernanceSignal from governance_verifier
# 2. Consume budget layer signal for stability_class
# 3. Apply severity multiplier to divergence classification
# 4. Update root_cause_vector with budget attribution
# 5. Flag divergence records with budget_confounded marker
#
# Reference: docs/system_law/Budget_PhaseX_Doctrine.md Section 3.3
# Schema: budget_governance_signal.schema.json
```

### 8.3 Anchor Summary Table

| Anchor | Phase | Status | Depends On |
|--------|-------|--------|------------|
| P3-BUDGET-NOISE-001 | X P3 | NOT AUTHORIZED | P3 execution auth |
| P4-BUDGET-TDA-001 | X P4 | NOT AUTHORIZED | P4 execution auth |

---

## 9. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-10 | Initial Budget Phase X Doctrine specification |
| 1.1.0 | 2025-12-11 | Added P5 Runtime Expectations section (7.1-7.2) |
| 1.2.0 | 2025-12-11 | Added P5 Budget Calibration Experiment (7.3), Enablement Gate Checklist, Smoke-Test Readiness |

---

*Document Status: APPROVED FOR SCHEMA CREATION*

This specification is approved for:
- Schema file creation in `docs/system_law/schemas/budget/`
- TODO anchor documentation
- Cross-reference from P3/P4 specifications

Implementation of budget-aware P3/P4 logic requires separate authorization.
