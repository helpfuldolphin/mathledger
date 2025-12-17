# Chronicle Governance Phase X Specification

**Document Version:** 1.0.0  
**Status:** Reference  
**Phase:** X (SHADOW MODE)  
**Date:** 2025-12-11

---

## 1. Purpose

This document specifies the Chronicle Governance system for Phase X, which provides recurrence projection, invariant checking, and drift analysis for curriculum change management. The system operates in SHADOW MODE: all analysis is purely observational and does not influence control flow.

---

## 2. First Light Recurrence Annex

### 2.1 Overview

The First Light Chronicle Recurrence Annex provides a concise answer to a critical question for reviewers:

> **"How likely is it that this class of failure will recur if you rerun First Light?"**

The annex is attached to evidence packs under `evidence["governance"]["chronicle"]["first_light_annex"]` when recurrence projection and invariant check data are available.

### 2.2 Annex Structure

The annex contains three key fields:

```json
{
  "schema_version": "1.0.0",
  "recurrence_likelihood": 0.65,
  "band": "MEDIUM",
  "invariants_ok": true
}
```

### 2.3 Interpreting Recurrence Likelihood and Band

**`recurrence_likelihood`** (float, range [0.0, 1.0]):
- A continuous probability estimate of how likely similar anomalies are to recur
- Computed from: `causality_strength × drift_event_density × churn_ratio`
- Higher values indicate stronger patterns suggesting recurrence
- **Interpretation:**
  - `0.0 - 0.4`: Low recurrence likelihood — anomalies appear isolated
  - `0.4 - 0.7`: Moderate recurrence likelihood — patterns suggest possible recurrence
  - `0.7 - 1.0`: High recurrence likelihood — strong patterns indicate likely recurrence

**`band`** (string, enum: "LOW" | "MEDIUM" | "HIGH"):
- A discrete classification derived from `recurrence_likelihood`
- **Mapping:**
  - `LOW`: `recurrence_likelihood < 0.4`
  - `MEDIUM`: `0.4 ≤ recurrence_likelihood < 0.7`
  - `HIGH`: `recurrence_likelihood ≥ 0.7`
- **Use for reviewers:**
  - `LOW`: Treat anomalies as likely one-off events; focus on immediate remediation
  - `MEDIUM`: Monitor for recurrence; consider preventive measures
  - `HIGH`: Expect recurrence; prioritize systemic fixes over point solutions

### 2.4 The Role of Invariants as a Safety Latch

**`invariants_ok`** (boolean):
- Indicates whether phase-transition drift invariants are satisfied
- **Interpretation:**
  - `true`: Invariants intact — even if recurrence is likely, the system's structural integrity constraints are satisfied
  - `false`: Invariants violated — structural integrity concerns exist independent of recurrence likelihood

**Safety Latch Logic:**
- **"Yes, it's recurring / No, the invariants are still intact"**: When `recurrence_likelihood` is high but `invariants_ok` is `true`, the system indicates that while patterns suggest recurrence, the underlying structural constraints remain valid. This suggests the recurrence is within expected bounds.
- **"Yes, it's recurring / Yes, invariants are violated"**: When both `recurrence_likelihood` is high and `invariants_ok` is `false`, this indicates a more serious condition: not only is recurrence likely, but structural integrity has been compromised.

### 2.5 Reviewer Guidance

When evaluating First Light evidence packs:

1. **Check for annex presence**: If `first_light_annex` is present, recurrence analysis is available
2. **Interpret band first**: Use the discrete `band` for quick classification
3. **Consider likelihood magnitude**: Use `recurrence_likelihood` for nuanced assessment
4. **Apply safety latch**: Use `invariants_ok` to distinguish between "recurring but bounded" vs. "recurring and unconstrained"
5. **Weight relative to other signals**: The annex is one input among many; combine with divergence logs, calibration reports, and other governance signals

**Decision Framework:**
- `band="LOW"` + `invariants_ok=true`: Treat as isolated incident; standard remediation sufficient
- `band="MEDIUM"` + `invariants_ok=true`: Monitor closely; consider preventive measures
- `band="HIGH"` + `invariants_ok=true`: Expect recurrence; prioritize systemic fixes
- `band="*"` + `invariants_ok=false`: Structural integrity concern; investigate invariant violations regardless of recurrence likelihood

---

## 3. Chronicle Risk Register

### 3.1 Overview

The Chronicle Risk Register provides a CAL-EXP-level aggregation of recurrence patterns across multiple calibration experiments. It answers:

> **"Across all calibration experiments, which ones show the highest recurrence risk?"**

The register aggregates per-experiment recurrence snapshots to provide:
- Band distribution statistics (LOW/MEDIUM/HIGH counts)
- High-risk calibration identification (HIGH recurrence + invariants violated)
- Risk summary for reviewers

### 3.2 Per-Experiment Recurrence Snapshots

Each calibration experiment can generate a recurrence snapshot via `build_cal_exp_recurrence_snapshot()`:

**Snapshot Structure:**
```json
{
  "schema_version": "1.0.0",
  "cal_id": "cal_001",
  "recurrence_likelihood": 0.65,
  "band": "MEDIUM",
  "invariants_ok": true,
  "timestamp": "2025-12-11T10:00:00Z"
}
```

**File Emission:**
- Snapshots are optionally emitted to `calibration/chronicle_recurrence_<cal_id>.json`
- Provides per-experiment audit trail for recurrence analysis

### 3.3 Risk Register Structure

The risk register aggregates snapshots across experiments:

```json
{
  "schema_version": "1.0.0",
  "total_calibrations": 5,
  "band_counts": {
    "LOW": 2,
    "MEDIUM": 2,
    "HIGH": 1
  },
  "high_risk_calibrations": ["cal_003"],
  "high_risk_details": [
    {
      "cal_id": "cal_003",
      "recurrence_likelihood": 0.85,
      "invariants_ok": false,
      "evidence_path_hint": "calibration/chronicle_recurrence_cal_003.json"
    }
  ],
  "risk_summary": "Analyzed 5 calibration experiment(s). 1 high-risk calibration(s) identified (HIGH recurrence + invariants violated)."
}
```

### 3.4 Reading the Risk Register

**For Reviewers:**

1. **Check `total_calibrations`**: Understand the sample size
2. **Review `band_counts`**: Assess distribution of recurrence risk
   - High proportion of HIGH band → systemic recurrence patterns
   - Balanced distribution → mixed risk profile
   - Mostly LOW → isolated incidents
3. **Examine `high_risk_calibrations`**: Focus on calibrations with:
   - `band="HIGH"` AND `invariants_ok=false`
   - These represent the highest priority for investigation
4. **Review `high_risk_details`**: For each high-risk calibration:
   - `cal_id`: Calibration experiment identifier
   - `recurrence_likelihood`: Continuous probability score
   - `invariants_ok`: Safety latch status
   - `evidence_path_hint`: Path to per-experiment recurrence snapshot JSON
   - Use `evidence_path_hint` to navigate to detailed evidence
5. **Read `risk_summary`**: Quick narrative summary of overall risk posture

**Decision Framework:**

- **No high-risk calibrations**: Standard review process sufficient
- **1-2 high-risk calibrations**: Investigate specific calibrations; may indicate isolated issues
- **3+ high-risk calibrations**: Systemic pattern; consider broader investigation
- **High proportion of HIGH band**: Even if invariants intact, recurrence patterns suggest monitoring

**Integration with Other Signals:**

The risk register should be read in conjunction with:
- Individual First Light recurrence annexes (per-experiment detail)
- Calibration reports (twin model accuracy)
- Divergence logs (twin-vs-real alignment)

The register provides the **cross-experiment view**; individual annexes provide **per-experiment detail**.

### 3.5 Evidence Pack Integration

The risk register is attached to evidence packs under:
```
evidence["governance"]["chronicle_risk_register"]
```

**Advisory Nature:**
- The register is **advisory only** (SHADOW MODE)
- It does not gate or block operations
- It provides reviewers with cross-experiment risk context
- Use for prioritization and investigation planning, not enforcement

---

## 4. GGFL Integration (SIG-CHR)

The chronicle risk register provides a low-weight advisory signal to the Global Governance Fusion Layer (GGFL) for cross-subsystem alignment views. The signal is normalized via `chronicle_risk_for_alignment_view()` into the GGFL unified format.

**Signal Type:** `SIG-CHR` (Chronicle Risk)

**Status Determination:**
- `"ok"`: No high-risk calibrations and no invariants violated
- `"warn"`: High-risk calibrations present (`high_risk_count > 0`) OR invariants violated (`has_any_invariants_violated = true`)

**Weight Hint:** `"LOW"` (Chronicle risk is a low-weight advisory signal)

**Conflict:** Always `false` (chronicle risk never triggers conflict directly)

**Drivers:** Deterministic list (up to 3 items):
- `high_risk_count: <count>` (if high-risk calibrations present)
- `HIGH recurrence + invariants violated` (if invariants violated)
- `top_risk_cal_ids: <cal_id1>, <cal_id2>, ...` (top 3 high-risk cal_ids, sorted)

**Extraction Source:** Tracks data provenance:
- `"MANIFEST"`: Register extracted from `manifest["governance"]["chronicle_risk_register"]` (preferred)
- `"EVIDENCE_JSON"`: Register extracted from `evidence["governance"]["chronicle_risk_register"]` (fallback)
- `"MISSING"`: Register not found in either source

**Phase X Status:** SIG-CHR is **advisory-only** during Phase X. It does not gate or block operations, and provides observational context for reviewers to correlate calibration summaries in alignment views. The signal helps reviewers understand recurrence risk patterns across calibration experiments but does not influence governance decisions.

---

## 5. SHADOW MODE Contract

All chronicle governance functions operate under strict SHADOW MODE constraints:

- **Observational only**: No control flow depends on chronicle governance outputs
- **No enforcement**: Chronicle governance does not gate or block operations
- **Logging only**: All outputs are for analysis and review, not runtime decisions
- **Descriptive**: The recurrence annex describes patterns; it does not prescribe actions

---

## 6. Integration Points

### 4.1 Evidence Packs

The recurrence annex is automatically attached to evidence packs when:
- `recurrence_projection` is provided to `attach_chronicle_governance_to_evidence()`
- `invariant_check` is provided to `attach_chronicle_governance_to_evidence()`

Location: `evidence["governance"]["chronicle"]["first_light_annex"]`

### 4.2 Calibration Reports

Chronicle governance data is attached to P4 calibration reports via `attach_chronicle_governance_to_calibration_report()`.

### 4.3 Divergence Logs

Chronicle drift markers can be attached to P4 divergence log entries via the `chronicle_drift` field in `DivergenceLogEntry`.

---

## 7. Related Documents

- `Phase_X_P4_Spec.md`: P4 shadow runner specification
- `Evidence_Pack_Spec_PhaseX.md`: Evidence pack structure
- `Phase_X_Divergence_Metric.md`: Divergence metric definition

---

## 8. Implementation

**Primary Module:** `backend/health/chronicle_governance_adapter.py`

**Key Functions:**
- `build_first_light_chronicle_annex()`: Constructs the recurrence annex
- `attach_chronicle_governance_to_evidence()`: Attaches annex to evidence packs
- `extract_chronicle_drift_signal()`: Extracts drift signal for divergence logs
- `build_cal_exp_recurrence_snapshot()`: Creates per-experiment recurrence snapshot
- `build_chronicle_risk_register()`: Aggregates snapshots into risk register
- `attach_chronicle_risk_register_to_evidence()`: Attaches register to evidence packs
- `chronicle_risk_for_alignment_view()`: Converts chronicle risk signal to GGFL alignment view format (SIG-CHR)

**Tests:** `tests/ci/test_chronicle_governance_tile_serializes.py`

