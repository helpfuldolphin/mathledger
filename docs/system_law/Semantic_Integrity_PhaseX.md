# Semantic Integrity Phase X Specification

**STATUS:** PHASE X — SEMANTIC–CURRICULUM OVERSIGHT LAYER

**Version:** 1.0.0

**Last Updated:** 2025-12-09

---

## Overview

The Semantic Integrity Grid provides a governance layer for ensuring semantic coherence across curriculum, metrics, documentation, and preregistration artifacts. This specification defines the First Light semantic footprint, a compact safety badge for external reviewers.

**SHADOW MODE CONTRACT:**
- All semantic integrity functions are read-only and side-effect free
- The semantic integrity tile is purely observational
- It does NOT influence any other tiles or system health classification
- No control flow depends on the tile contents
- Advisory only; council still decides; no new gate semantics

---

## First Light Semantic Footprint

The First Light semantic footprint combines P3 (synthetic) and P4 (shadow) semantic integrity signals into a single, human-readable record. This footprint serves as a **cognitive/semantic safety badge** in evidence packs, providing external reviewers with a clear view of semantic integrity across both experimental phases.

### Footprint Schema

The footprint is stored under `evidence["governance"]["semantic_integrity"]["first_light_footprint"]` when both P3 and P4 data are available.

**Schema Version:** `1.0.0`

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | string | Schema version identifier ("1.0.0") |
| `invariants_ok` | bool | `True` if all semantic invariants are satisfied, `False` otherwise |
| `broken_invariant_count` | int | Number of broken semantic invariants detected |
| `p3_uplift_semantic_status` | string | P3 uplift semantic status: `"OK"` \| `"WARN"` \| `"BLOCK"` |
| `p4_severity` | string | P4 semantic drift severity: `"OK"` \| `"ATTENTION"` \| `"BROKEN"` |
| `terms_involved` | array[string] | List of terms involved in semantic issues (limited to top 5) |

### Field Interpretations

#### `invariants_ok`
- **`True`**: All semantic invariants are satisfied:
  - Every curriculum term appears in ≥2 of {taxonomy, docs, graph}
  - No taxonomy term has been unused for >N versions
  - No graph node is isolated (degree 0)
- **`False`**: At least one semantic invariant is violated

#### `broken_invariant_count`
- Count of semantic invariants that are currently violated
- `0` when `invariants_ok` is `True`
- May be >0 even if `invariants_ok` is `True` if violations are non-critical (ATTENTION severity)

#### `p3_uplift_semantic_status`
- **`"OK"`**: All semantic checks passed; uplift appears safe from semantic perspective
- **`"WARN"`**: Semantic attention required; review recommended
- **`"BLOCK"`**: Semantic uplift blocked due to critical issues (risk CRITICAL, contract BREACH, or invariants BROKEN)

#### `p4_severity`
- **`"OK"`**: No semantic drift detected in shadow run
- **`"ATTENTION"`**: Semantic drift detected with non-critical severity
- **`"BROKEN"`**: Critical semantic drift detected (BROKEN-level invariant violations)

#### `terms_involved`
- List of terms involved in semantic issues (e.g., curriculum terms with insufficient appearances, isolated graph nodes, unused taxonomy terms)
- Limited to top 5 for readability
- Empty list when no issues detected

---

## 2×2 Safety Grid

The semantic footprint can be interpreted using a 2×2 grid mapping P3 vs P4 states:

### P3 Uplift Semantic Status × P4 Severity

| P3 Status | P4 Severity | Interpretation | Narrative |
|-----------|-------------|----------------|-----------|
| **OK** | **OK** | **Fully Aligned** | Semantic integrity is maintained across both synthetic (P3) and shadow (P4) runs. No semantic drift detected. Uplift appears safe from semantic perspective. |
| **OK** | **ATTENTION** | **Shadow Drift** | P3 synthetic run shows no semantic issues, but P4 shadow run detected non-critical drift. Review recommended to understand divergence between synthetic and real behavior. |
| **OK** | **BROKEN** | **Shadow Critical** | P3 synthetic run shows no semantic issues, but P4 shadow run detected critical semantic drift. Investigate why synthetic run did not surface these issues. |
| **WARN** | **OK** | **Synthetic Warning** | P3 synthetic run shows semantic warnings (e.g., partial misalignment), but P4 shadow run shows no drift. May indicate synthetic run is more conservative or detects issues earlier. |
| **WARN** | **ATTENTION** | **Partial Misalignment** | Both P3 and P4 show semantic issues, but neither is critical. Review recommended to address misalignments before they escalate. |
| **WARN** | **BROKEN** | **Escalating Issues** | P3 shows warnings while P4 shows critical drift. Issues are escalating from synthetic to real runs. Immediate attention required. |
| **BLOCK** | **OK** | **Synthetic Block** | P3 synthetic run blocked uplift due to semantic issues (e.g., contract breach, critical risk), but P4 shadow run shows no drift. May indicate synthetic run is overly conservative or detects issues not present in real runs. |
| **BLOCK** | **ATTENTION** | **Block with Drift** | P3 blocked uplift and P4 shows drift. Semantic issues are present in both phases. Uplift should not proceed until issues are resolved. |
| **BLOCK** | **BROKEN** | **Critical Failure** | Both P3 and P4 show critical semantic issues. Uplift is blocked and semantic integrity is compromised. Immediate remediation required. |

### Simplified 2×2 View (OK vs Not-OK)

For quick assessment, the grid can be simplified to OK vs Not-OK:

| P3 | P4 | Interpretation |
|----|----|----------------|
| **OK** | **OK** | **Green Zone**: Semantic integrity maintained |
| **OK** | **Not-OK** | **Yellow Zone**: Shadow drift detected; investigate |
| **Not-OK** | **OK** | **Yellow Zone**: Synthetic issues; may be conservative |
| **Not-OK** | **Not-OK** | **Red Zone**: Semantic issues in both phases; block uplift |

---

## Usage in Evidence Packs

The semantic footprint is automatically included in evidence packs when both P3 and P4 data are available:

```json
{
  "governance": {
    "semantic_integrity": {
      "invariants_ok": true,
      "broken_invariant_count": 0,
      "severity": "OK",
      "terms_involved": [],
      "critical_signals": [],
      "first_light_footprint": {
        "schema_version": "1.0.0",
        "invariants_ok": true,
        "broken_invariant_count": 0,
        "p3_uplift_semantic_status": "OK",
        "p4_severity": "OK",
        "terms_involved": []
      }
    }
  }
}
```

### External Reviewer Guidance

For external reviewers examining evidence packs:

1. **Check `invariants_ok`**: If `False`, semantic invariants are violated
2. **Check `broken_invariant_count`**: Higher counts indicate more issues
3. **Check P3/P4 alignment**: Use the 2×2 grid to understand the relationship between synthetic and shadow runs
4. **Check `terms_involved`**: Review which terms are causing issues
5. **Interpret status**: Use the narrative interpretations in the 2×2 grid to understand the semantic integrity posture

**Important:** The footprint is a **safety badge** for external reviewers, not a direct gate. The council still makes final decisions based on all available evidence.

---

## Semantic Safety Panel

The Semantic Safety Panel aggregates semantic footprints across multiple calibration experiments (CAL-EXP-1, CAL-EXP-2, CAL-EXP-3) into a single 2×2 grid summary. This panel provides a cross-experiment view of semantic integrity, enabling reviewers to assess consistency and identify patterns across calibration runs.

### Panel Schema

The panel is stored under `evidence["governance"]["semantic_safety_panel"]` when multiple calibration experiment footprints are aggregated.

**Schema Version:** `1.0.0`

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | string | Schema version identifier ("1.0.0") |
| `total_experiments` | int | Total number of calibration experiments included |
| `grid_counts` | object | Counts of experiments in each grid bucket |
| `grid_counts.ok_ok` | int | Count of experiments with P3 OK × P4 OK |
| `grid_counts.ok_not_ok` | int | Count of experiments with P3 OK × P4 Not-OK |
| `grid_counts.not_ok_ok` | int | Count of experiments with P3 Not-OK × P4 OK |
| `grid_counts.not_ok_not_ok` | int | Count of experiments with P3 Not-OK × P4 Not-OK |
| `experiments` | array[object] | Per-experiment details with cal_id, statuses, and grid_bucket |

### Per-Experiment Footprint

Each calibration experiment emits a semantic footprint record that is persisted to disk:

**File Location:** `calibration/semantic_footprint_<cal_id>.json`

**Schema:**

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | string | Schema version identifier ("1.0.0") |
| `cal_id` | string | Calibration experiment identifier (e.g., "CAL-EXP-1") |
| `p3_status` | string | P3 uplift semantic status: `"OK"` \| `"WARN"` \| `"BLOCK"` |
| `p4_status` | string | P4 semantic drift severity: `"OK"` \| `"ATTENTION"` \| `"BROKEN"` |
| `broken_invariant_count` | int | Number of broken semantic invariants |

### Panel Interpretation

The panel provides a quick view of semantic integrity distribution across experiments:

#### Grid Bucket Counts

- **`ok_ok`**: Experiments where both P3 and P4 show no semantic issues. These are the "green zone" experiments.
- **`ok_not_ok`**: Experiments where P3 is OK but P4 shows issues. These indicate shadow drift that wasn't detected in synthetic runs.
- **`not_ok_ok`**: Experiments where P3 shows issues but P4 is OK. These may indicate synthetic runs are more conservative.
- **`not_ok_not_ok`**: Experiments where both P3 and P4 show issues. These are the "red zone" experiments requiring immediate attention.

#### Reading the Panel

1. **Check `total_experiments`**: Verify all expected experiments are included (typically 3 for CAL-EXP-1/2/3).
2. **Review `grid_counts`**: 
   - High `ok_ok` count indicates consistent semantic integrity across experiments.
   - High `not_ok_not_ok` count indicates systemic semantic issues requiring remediation.
   - Mixed buckets may indicate experiment-specific issues or configuration differences.
3. **Examine `experiments` array**: Review per-experiment details to identify specific experiments with issues.
4. **Cross-reference with 2×2 grid**: Use the grid interpretations to understand the semantic posture of each experiment.

### Example Panel

```json
{
  "governance": {
    "semantic_safety_panel": {
      "schema_version": "1.0.0",
      "total_experiments": 3,
      "grid_counts": {
        "ok_ok": 2,
        "ok_not_ok": 0,
        "not_ok_ok": 0,
        "not_ok_not_ok": 1
      },
      "experiments": [
        {
          "cal_id": "CAL-EXP-1",
          "p3_status": "OK",
          "p4_status": "OK",
          "broken_invariant_count": 0,
          "grid_bucket": "OK×OK"
        },
        {
          "cal_id": "CAL-EXP-2",
          "p3_status": "OK",
          "p4_status": "OK",
          "broken_invariant_count": 0,
          "grid_bucket": "OK×OK"
        },
        {
          "cal_id": "CAL-EXP-3",
          "p3_status": "BLOCK",
          "p4_status": "BROKEN",
          "broken_invariant_count": 5,
          "grid_bucket": "Not-OK×Not-OK"
        }
      ]
    }
  }
}
```

### External Reviewer Guidance for Panel

For external reviewers examining the semantic safety panel across CAL-EXP-1/2/3:

1. **Consistency Check**: If all experiments are in `ok_ok`, semantic integrity is consistent across calibration runs.
2. **Pattern Detection**: If experiments cluster in specific buckets, investigate common factors (configuration, data, timing).
3. **Escalation Signals**: If `not_ok_not_ok` count increases across experiments, semantic issues may be escalating.
4. **Shadow Drift**: If `ok_not_ok` count is high, investigate why P4 shadow runs detect issues that P3 synthetic runs miss.
5. **Conservative Synthetic**: If `not_ok_ok` count is high, P3 synthetic runs may be overly conservative.

**Important:** The panel is **advisory only** and provides observability across experiments. The council still makes final decisions based on all available evidence, including individual experiment details and other governance signals.

---

## Integration Points

### P3 Stability Report
- `semantic_integrity_summary` field contains P3-specific semantic integrity metrics
- Used to build the footprint's `p3_uplift_semantic_status` field

### P4 Calibration Report
- `semantic_integrity_calibration` field contains P4-specific semantic drift signals
- Used to build the footprint's `p4_severity` and `terms_involved` fields

### Evidence Pack
- Footprint automatically included when both P3 and P4 data available
- Stored under `evidence["governance"]["semantic_integrity"]["first_light_footprint"]`

---

## References

- **Implementation**: `backend/health/semantic_integrity_adapter.py`
- **Tests**: `tests/health/test_semantic_integrity_adapter.py`
- **Global Health Integration**: `backend/health/global_surface.py`
- **Phase V Invariants**: See Phase V semantic invariant checker specification

---

## Change Log

- **2025-12-09**: Initial specification for First Light semantic footprint

