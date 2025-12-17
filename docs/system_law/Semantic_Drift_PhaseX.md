# Semantic Drift Phase X Specification

**STATUS:** PHASE X — SEMANTIC DRIFT GOVERNANCE INTEGRATION

**Version:** 1.0.0

**Last Updated:** 2025-12-11

---

## Overview

The Semantic Drift Governance system provides a governance layer for tracking and analyzing semantic drift across the repository structure. This specification defines the First Light semantic drift failure shelf, a compact triage list for reviewers identifying the worst semantic drift episodes.

**SHADOW MODE CONTRACT:**
- All semantic drift functions are read-only and side-effect free
- The semantic drift tile is purely observational
- It does NOT influence any other tiles or system health classification
- No control flow depends on the tile contents
- Advisory only; council still decides; no new gate semantics

---

## First Light Semantic Drift Failure Shelf

The First Light semantic drift failure shelf combines P3 (synthetic) and P4 (shadow) semantic drift signals into a compact triage list. This shelf serves as a **reviewer workflow guide** in evidence packs, providing external reviewers with a prioritized list of the worst semantic drift episodes.

### Shelf Schema

The shelf is stored under `evidence["governance"]["semantic_drift"]["first_light_failure_shelf"]` when both P3 and P4 data are available.

**Schema Version:** `1.0.0`

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | string | Schema version identifier ("1.0.0") |
| `p3_tensor_norm` | float | P3 semantic drift tensor norm (magnitude of drift) |
| `p4_tensor_norm` | float | P4 semantic drift tensor norm (magnitude of drift) |
| `semantic_hotspots` | array[string] | Top 5 problematic loci for semantic drift (by severity) |
| `regression_status` | string | P4 regression status: `"REGRESSED"` \| `"ATTENTION"` \| `"STABLE"` |

### Field Interpretations

#### `semantic_hotspots`

The `semantic_hotspots` field contains the **top 5 problematic loci** for semantic drift, ranked by severity. These are the slice identifiers where semantic drift is most pronounced, combining:

- **Semantic drift score**: Measures unexpected file types, naming pattern violations, and structural entropy
- **Causal drift score**: Measures drift in causal relationships and dependencies
- **Metric-correlated drift score**: Measures drift correlated with metric changes

The shelf limits to top 5 to provide a focused triage list: **"If you only inspect 5 places, look here."**

#### `p3_tensor_norm` and `p4_tensor_norm`

Tensor norms represent the overall magnitude of semantic drift across all slices:

- **Higher values** indicate more widespread or severe drift
- **P3 tensor norm** reflects drift in synthetic (P3) runs
- **P4 tensor norm** reflects drift in shadow (P4) runs
- **Comparison** between P3 and P4 norms can reveal divergence between synthetic and real behavior

#### `regression_status`

The regression status indicates the overall drift trajectory:

- **`"REGRESSED"`**: Semantic drift has reached blocking levels (gating recommendation: BLOCK)
- **`"ATTENTION"`**: Semantic drift requires attention but is not blocking (gating recommendation: WARN)
- **`"STABLE"`**: Semantic drift is within acceptable bounds (gating recommendation: OK)

---

## Reviewer Workflow

The failure shelf is designed as a **triage list for auditors**, not a gate. Reviewers should use the following workflow:

### Step 1: Inspect Failure Shelf Slices

1. **Examine `semantic_hotspots`**: Review the top 5 slice identifiers listed
2. **Check tensor norms**: Compare P3 vs P4 norms to understand drift magnitude
3. **Assess regression status**: Determine if drift is regressing, stable, or requiring attention

### Step 2: Correlate with Curriculum Slices and TDA Anomalies

1. **Cross-reference with curriculum**: Map semantic hotspots to curriculum slice definitions
2. **Check TDA metrics**: Correlate semantic drift hotspots with TDA (Topological Data Analysis) anomalies
3. **Review structural health**: Examine directory entropy and structural drift reports for the identified slices

### Step 3: Decide Whether Drift is Tolerable or Requires Adjustment

1. **Evaluate severity**: Assess whether the drift level is acceptable for the current phase
2. **Consider context**: Factor in curriculum requirements, metric stability, and structural health
3. **Make recommendation**: Decide whether drift is tolerable or requires curriculum/model adjustment

**Important:** The failure shelf is a **triage tool** for reviewers, not a direct gate. The council still makes final decisions based on all available evidence.

---

## Usage in Evidence Packs

The failure shelf is automatically included in evidence packs when both P3 and P4 data are available:

```json
{
  "governance": {
    "semantic_drift": {
      "status_light": "YELLOW",
      "tensor_norm": 1.5,
      "semantic_hotspots": ["slice_a", "slice_b"],
      "projected_instability_count": 2,
      "gating_recommendation": "WARN",
      "headline": "Semantic drift analysis indicates potential instability.",
      "first_light_failure_shelf": {
        "schema_version": "1.0.0",
        "p3_tensor_norm": 1.5,
        "p4_tensor_norm": 2.0,
        "semantic_hotspots": ["slice_a", "slice_b", "slice_c", "slice_d", "slice_e"],
        "regression_status": "ATTENTION"
      }
    }
  }
}
```

---

## Semantic Drift Triage Index

The Semantic Drift Triage Index aggregates individual failure shelves from multiple calibration experiments (CAL-EXP-1, CAL-EXP-2, CAL-EXP-3) into a global ranked list. This index helps auditors identify the most persistent or severe semantic drift hotspots across all calibration experiments.

### Index Schema

The index is stored under `evidence["governance"]["semantic_failure_triage_index"]` when multiple CAL-EXP shelves are aggregated.

**Schema Version:** `1.0.0`

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | string | Schema version identifier ("1.0.0") |
| `items` | array[object] | Ranked list of semantic drift items (max 10 by default) |
| `total_shelves` | int | Total number of shelves aggregated |
| `neutral_notes` | array[string] | Neutral descriptive notes about the index |

**Item Structure:**

Each item in the `items` array contains:

| Field | Type | Description |
|-------|------|-------------|
| `cal_id` | string | Calibration experiment identifier (e.g., "CAL-EXP-1") |
| `semantic_hotspots` | array[string] | Top 5 problematic loci for semantic drift |
| `p3_tensor_norm` | float | P3 semantic drift tensor norm |
| `p4_tensor_norm` | float | P4 semantic drift tensor norm |
| `regression_status` | string | P4 regression status: `"REGRESSED"` \| `"ATTENTION"` \| `"STABLE"` |
| `combined_tensor_norm` | float | Sum of P3 and P4 tensor norms |

### Ranking Logic

Items are ranked by severity using the following criteria (in order):

1. **Regression Status**: REGRESSED > ATTENTION > STABLE
2. **Combined Tensor Norm**: Higher values indicate more severe drift
3. **Number of Hotspots**: More hotspots indicate broader drift
4. **Cal ID**: Alphabetical ordering for determinism when other criteria are equal

The index is truncated to the top N items (default: 10) to provide a focused triage list.

### Auditor Usage

Auditors should use the triage index to prioritize their semantic drift investigations:

1. **Start with top-ranked items**: Items at the top of the list have the highest severity scores
2. **Focus on REGRESSED status**: Experiments with REGRESSED status require immediate attention
3. **Cross-reference hotspots**: Compare semantic hotspots across experiments to identify persistent patterns
4. **Review tensor norms**: Higher combined tensor norms indicate more widespread drift
5. **Select top N investigations**: Use the ranked list to pick the top N drift investigations based on available review capacity

**Important:** The triage index is a **prioritization tool** for auditors, not a direct gate. The council still makes final decisions based on all available evidence.

---

## Integration Points

### P3 Stability Report
- `semantic_drift_summary` field contains P3-specific semantic drift metrics
- Used to build the shelf's `p3_tensor_norm` and `semantic_hotspots` fields

### P4 Calibration Report
- `semantic_drift` field contains P4-specific semantic drift calibration
- Used to build the shelf's `p4_tensor_norm` and `regression_status` fields

### Evidence Pack
- Shelf automatically included when both P3 and P4 data available
- Stored under `evidence["governance"]["semantic_drift"]["first_light_failure_shelf"]`

### CAL-EXP Shelf Emission
- Each CAL-EXP run writes its shelf to `calibration/semantic_failure_shelf_<cal_id>.json`
- Shelves are emitted via `emit_cal_exp_semantic_failure_shelf()`

### Global Triage Index
- Triage index aggregates multiple CAL-EXP shelves via `build_semantic_failure_triage_index()`
- Attached to evidence pack under `evidence["governance"]["semantic_failure_triage_index"]`

---

## References

- **Implementation**: `backend/health/semantic_drift_adapter.py`
- **Tests**: `tests/ci/test_semantic_drift_integrations.py`
- **Global Health Integration**: `backend/health/global_surface.py`
- **Related**: `docs/system_law/Semantic_Integrity_PhaseX.md` (semantic integrity grid)

---

## Change Log

- **2025-12-11**: Initial specification for First Light semantic drift failure shelf
- **2025-12-11**: Added semantic drift triage index specification for cross-experiment aggregation

