# Epistemic Alignment Phase X Specification

**STATUS:** PHASE X — EPISTEMIC GOVERNANCE INTEGRATION

**Version:** 1.0.0

**Last Updated:** 2025-01-XX

---

## Overview

The Epistemic Alignment layer provides cross-domain stability analysis by combining semantic, curriculum, metric, and drift signals into a unified alignment tensor. This specification defines the First Light epistemic annex and behavior consistency block for external reviewers and safety auditors.

**SHADOW MODE CONTRACT:**
- All epistemic alignment functions are read-only and side-effect free
- The epistemic alignment tile is purely observational
- It does NOT influence any other tiles or system health classification
- No control flow depends on the tile contents
- Advisory only; council still decides; no new gate semantics

---

## First Light Epistemic Annex & Consistency Block

The First Light epistemic annex combines P3 (synthetic) and P4 (shadow) epistemic alignment signals into a single, human-readable record. This annex serves as a **cross-domain stability badge** in evidence packs, providing external reviewers with a clear view of epistemic alignment across both experimental phases.

### Annex Schema

The annex is stored under `evidence["governance"]["epistemic_alignment"]["first_light_annex"]` when both P3 and P4 reports are available.

**Schema Version:** `1.0.0`

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | string | Schema version identifier ("1.0.0") |
| `p3_tensor_norm` | float | P3 alignment tensor norm [0, 1] (higher = healthier) |
| `p3_alignment_band` | string | P3 alignment band: `"LOW"` \| `"MEDIUM"` \| `"HIGH"` |
| `p4_tensor_norm` | float | P4 alignment tensor norm [0, 1] (higher = healthier) |
| `p4_alignment_band` | string | P4 alignment band: `"LOW"` \| `"MEDIUM"` \| `"HIGH"` |
| `hotspot_union` | array[string] | Union of misalignment hotspots from P3 and P4 (sorted, limited to top 5) |

### Field Interpretations

#### `p3_tensor_norm` and `p4_tensor_norm`
- **Range:** [0, 1]
- **Higher values:** Indicate healthier epistemic alignment across semantic, curriculum, metric, and drift domains
- **Computation:** L2 norm of system axes (semantic, curriculum, metrics, drift) normalized by maximum possible norm
- **Interpretation:**
  - `≥ 0.7`: HIGH alignment band
  - `≥ 0.4`: MEDIUM alignment band
  - `< 0.4`: LOW alignment band

#### `p3_alignment_band` and `p4_alignment_band`
- **`"HIGH"`**: Tensor norm ≥ 0.7; strong cross-domain alignment
- **`"MEDIUM"`**: Tensor norm ≥ 0.4 but < 0.7; moderate alignment with some misalignment
- **`"LOW"`**: Tensor norm < 0.4; significant misalignment across domains

#### `hotspot_union`
- Sorted, deduplicated union of misalignment hotspots from both P3 and P4
- Limited to top 5 for readability
- Hotspots are slices with: low semantic alignment AND low metric readiness AND high drift simultaneously
- Empty list when no hotspots detected

---

## Behavior Consistency Summary

The behavior consistency summary cross-checks epistemic alignment signals against evidence quality trajectory to identify potential inconsistencies between the epistemic model and observed behavior.

### Consistency Status Modes

The consistency summary provides three status modes:

#### `CONSISTENT`
- **Definition:** Epistemic alignment trajectory is consistent with evidence quality trajectory
- **Examples:**
  - Both epistemic alignment and evidence quality degrade together
  - Both epistemic alignment and evidence quality improve together
  - Epistemic alignment improves while evidence quality is stable (model catching up)

#### `INCONSISTENT`
- **Definition:** Epistemic alignment trajectory disagrees with evidence quality trajectory
- **Example A:** Epistemic alignment degraded from HIGH→LOW while evidence quality trajectory is IMPROVING
  - **Interpretation:** The epistemic model indicates misalignment, but observed evidence quality is improving. This suggests the epistemic model may be overly conservative or detecting issues not reflected in actual behavior.
- **Example B:** Epistemic alignment degraded from HIGH→MEDIUM while evidence quality trajectory is STABLE
  - **Interpretation:** The epistemic model shows degradation, but evidence quality remains stable. This may indicate the model is detecting early warning signals not yet visible in evidence quality metrics.

#### `UNKNOWN`
- **Definition:** Evidence quality data is not available for consistency checking
- **Interpretation:** Cannot determine consistency without evidence quality trajectory data

### Consistency Summary Schema

**Schema Version:** `1.0.0`

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `consistency_status` | string | Consistency status: `"CONSISTENT"` \| `"INCONSISTENT"` \| `"UNKNOWN"` |
| `advisory_notes` | array[string] | Neutral, descriptive notes explaining the consistency assessment |

### Advisory Notes

Advisory notes use neutral, descriptive language only. They describe:
- The direction of epistemic alignment change (degraded/improved/stable)
- The evidence quality trajectory (IMPROVING/STABLE/DEGRADING)
- Whether the trajectories are aligned or misaligned

**No evaluative language** (no "good/bad", "wrong/right", "error/mistake") is used.

---

## SHADOW MODE Contract

**CRITICAL:** All epistemic alignment functions operate in SHADOW MODE:

- **Observational only:** Functions are read-only and side-effect free
- **No gating logic:** Epistemic alignment signals do NOT block or gate any pipeline execution
- **Advisory only:** Signals are provided for human review and decision-making
- **No control flow dependencies:** No automated systems depend on epistemic alignment values for control decisions
- **External reviewer lens:** The annex and consistency block are designed for external safety auditors to assess epistemic alignment independently of system behavior

---

## Integration Points

### P3 Stability Report
- Epistemic alignment summary attached via `attach_epistemic_alignment_to_p3_stability_report()`
- Stored under `stability_report["epistemic_alignment_summary"]`

### P4 Calibration Report
- Epistemic alignment calibration attached via `attach_epistemic_alignment_to_p4_calibration_report()`
- Stored under `calibration_report["epistemic_alignment"]`

### Evidence Pack
- Epistemic alignment attached via `attach_epistemic_alignment_to_evidence()`
- First Light annex automatically included when both P3 and P4 reports are provided
- Stored under `evidence["governance"]["epistemic_alignment"]`

### Uplift Council
- Epistemic summary provided via `summarize_epistemic_for_uplift_council()`
- Classification: BLOCK/WARN/OK based on alignment and forecast bands
- Purely advisory; council makes final decision

---

## External Auditor Workflow

External safety auditors should:

1. **Review First Light Annex:**
   - Check P3 vs P4 tensor norms and alignment bands
   - Identify misalignment hotspots
   - Assess whether alignment improved or degraded from P3 to P4

2. **Review Consistency Summary:**
   - Check `consistency_status` for INCONSISTENT flags
   - Read `advisory_notes` for detailed explanation
   - Investigate cases where epistemic model disagrees with evidence quality

3. **Cross-Reference with Evidence Quality:**
   - Compare epistemic alignment trajectory with evidence quality trajectory
   - Flag cases where epistemic signals suggest problems but evidence quality looks fine (or vice versa)
   - Use consistency summary as a "Did the epistemic picture and the observed behavior stay in sync?" lens

---

## Epistemic Panel (Calibration View)

The **Epistemic Panel** provides a calibration-ready cross-experiment view of epistemic alignment consistency across multiple calibration experiments (CAL-EXP-1, CAL-EXP-2, CAL-EXP-3).

**SHADOW MODE CONTRACT:**
- This is a calibration cross-check, not a direct gate
- Panel is observational and does not influence calibration decisions
- Advisory only; council still decides; no new gate semantics

### Per-Experiment Annex Emission

Each calibration experiment emits a per-experiment epistemic annex via `emit_cal_exp_epistemic_annex()`:

- **File:** `calibration/epistemic_annex_<cal_id>.json`
- **Schema:** Same as First-Light annex, with added `cal_id` field
- **Purpose:** Per-experiment snapshot for aggregation across experiments

### Consistency Panel Aggregation

The consistency panel aggregates across multiple experiments via `build_epistemic_consistency_panel()`:

**Panel Schema:**

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | string | Schema version identifier ("1.0.0") |
| `num_experiments` | integer | Total number of calibration experiments analyzed |
| `num_consistent` | integer | Number of experiments with CONSISTENT status |
| `num_inconsistent` | integer | Number of experiments with INCONSISTENT status |
| `num_unknown` | integer | Number of experiments with UNKNOWN status |
| `experiments_inconsistent` | array[object] | List of inconsistent experiments with cal_id and brief neutral reason |

**Evidence Binding Location:**

The panel attaches to the main evidence structure at:

```
evidence["governance"]["epistemic_panel"]
```

### Auditor Interpretation

**High inconsistency counts** indicate that the epistemic model frequently disagrees with observed evidence quality:

- **Example:** `num_inconsistent: 2` out of `num_experiments: 3`
  - **Interpretation:** In 2 out of 3 calibration experiments, the epistemic alignment model indicated degradation while evidence quality was improving or stable. This suggests the epistemic model may be:
    - Overly conservative (detecting issues not reflected in actual behavior)
    - Detecting early warning signals not yet visible in evidence quality metrics
    - Misaligned with the evidence quality assessment methodology

- **Example:** `num_consistent: 3` out of `num_experiments: 3`
  - **Interpretation:** All calibration experiments show consistent alignment between epistemic signals and evidence quality. The epistemic model appears to be tracking system behavior accurately.

**For external auditors:** The panel provides a cross-experiment consistency check. High inconsistency rates warrant investigation into why the epistemic model and evidence quality assessments disagree, but do not automatically invalidate either signal.

### Status Signal Extraction and Provenance

The epistemic panel signal in `first_light_status.json` includes extraction provenance:

**Extraction Source (Coercion Law):**
- **Valid enum:** `"MANIFEST"`, `"EVIDENCE_JSON"`, `"MISSING"`
- **Coercion rule:** Unknown/invalid values → `"MISSING"` + `extraction_source_advisory` field added
- **Meanings:**
  - `"MANIFEST"`: Panel extracted from `manifest.json` (preferred source of truth)
  - `"EVIDENCE_JSON"`: Panel extracted from `evidence.json` fallback (manifest missing)
  - `"MISSING"`: Panel not found in either source (signal absent)
- **Fallback warning:** Exactly one neutral warning when `extraction_source == "EVIDENCE_JSON"`: "Epistemic panel loaded from evidence.json fallback (manifest missing). Advisory only."

**Panel Schema Version:**
- `panel_schema_version`: Passthrough from panel `schema_version` if present, else `"UNKNOWN"`
- `panel_schema_version_present`: Boolean indicating whether `schema_version` was present in panel
- Used to track panel schema evolution independently of status signal schema

**Top Reason Code:**
- `top_reason_code`: Canonically selected from `reason_code_histogram`
- Selection rule: Highest count (descending), then reason_code (ascending) for tie-breaking
- Derived deterministically from histogram (not freeform)
- Explicitly included in status signal for machine-actionable reconciliation

**Top Inconsistent Cal IDs:**
- `top_inconsistent_cal_ids_top3`: Sorted ascending, truncated to top 3
- Deterministic ordering ensures consistent reconciliation across runs

**Status↔GGFL Consistency Block:**
- `consistency`: Advisory-only cross-check between status signal and GGFL adapter output
- **Fields:**
  - `status_consistent`: Boolean indicating whether signal implicit status matches GGFL status
  - `driver_consistent`: Boolean indicating whether `top_reason_code` appears in GGFL drivers
  - `notes`: List of neutral advisory notes describing any detected inconsistencies
- **Purpose:** Detect reconciliation drift between status signal and GGFL representation
- **Location:** `signals["epistemic_panel"]["consistency"]`
- **Advisory only:** Consistency block does not gate or block operations; purely observational

**Advisory-Only / No Gating:**
- All epistemic panel signals are **advisory only**
- No gating logic or CI exit code changes
- Missing panel is non-error (gracefully handled)
- **Warning caps:** ≤1 fallback warning (EVIDENCE_JSON), ≤1 inconsistency warning (num_inconsistent > 0)

### GGFL Adapter (SIG-EPI)

The epistemic panel adapter (`epistemic_panel_for_alignment_view()`) provides a fixed-shape GGFL signal:

**Signal Type:** `"SIG-EPI"`

**Status:** `"ok"` | `"warn"` (warn if `num_inconsistent > 0`)

**Conflict:** `false` (invariant - epistemic panel never triggers conflict directly)

**Drivers:** Up to 3 reason-code drivers (deterministic ordering):
1. `DRIVER_EPI_INCONSISTENT_PRESENT` (if any inconsistencies)
2. `DRIVER_TOP_REASON_<REASON_CODE>` (top reason code from histogram)
3. `DRIVER_TOP_CAL_IDS_PRESENT` (if top cal_ids present)

**Shadow Mode Invariants:**
- `advisory_only: true`
- `no_enforcement: true`
- `conflict_invariant: true`

**Weight Hint:** `"LOW"` (ensures epistemic panel doesn't overpower fusion semantics)

**Determinism:** Identical input → byte-identical GGFL output (for reconciliation without interpretation drift)

---

## References

- **Implementation:** `backend/health/epistemic_p3p4_integration.py`
- **Tests:** `tests/health/test_epistemic_p3p4_integration.py`
- **Global Health Integration:** `backend/health/global_surface.py`
- **Evidence Adapter:** `backend/health/epistemic_evidence_adapter.py`

---

*Generated by E1 — Epistemic Alignment Governance Engineer*

