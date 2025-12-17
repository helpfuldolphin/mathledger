# Realism Constraint Solver — Phase X Specification

**Author**: C3 — Realism Constraint Engineer  
**Date**: 2025-12-11  
**Status**: Phase X — SHADOW MODE ONLY  
**Version**: 1.0.0

---

## Overview

The Realism Constraint Solver provides adaptive constraint solving to self-correct synthetic universes toward realism envelopes. This document specifies the **First Light Realism Annex**, a compact tile included in the First Light evidence pack for auditor interpretation.

**CRITICAL**: The annex is **NOT a gate** or escalation mechanism. All escalation decisions remain in upstream governance systems (ratchet, calibration console, etc.). This annex is purely observational and advisory.

---

## First Light Realism Annex Structure

The annex is located at:
```
evidence["governance"]["realism_constraints"]["first_light_annex"]
```

### Schema

```json
{
  "schema_version": "1.0.0",
  "status_light": "GREEN" | "YELLOW" | "RED",
  "global_pressure": 0.0-1.0,
  "outliers": ["scenario_name_1", "scenario_name_2", ...],
  "overall_confidence": 0.0-1.0
}
```

### Field Descriptions

- **`schema_version`**: Version identifier for the annex schema (currently "1.0.0").
- **`status_light`**: Overall realism status indicator:
  - `GREEN`: All conditions OK (calibration status OK, low pressure, acceptable confidence, few outliers)
  - `YELLOW`: Moderate issues (ATTENTION status, moderate pressure, or many outliers)
  - `RED`: Critical conditions (BLOCK status, high pressure > 0.7, or low confidence < 0.5)
- **`global_pressure`**: Realism pressure value [0.0, 1.0], indicating divergence from envelope bounds. Higher values indicate greater divergence.
- **`outliers`**: List of scenario names identified as outliers (low average similarity to other scenarios).
- **`overall_confidence`**: Confidence score [0.0, 1.0] for parameter adjustments. Higher values indicate higher confidence in adjustment recommendations.

---

## Interpretation Patterns

### Pattern 1: GREEN + Low Pressure + High Confidence

**Example:**
```json
{
  "status_light": "GREEN",
  "global_pressure": 0.15,
  "outliers": [],
  "overall_confidence": 0.85
}
```

**Interpretation**: "Twin realism looks consistent with scenario assumptions."

**Auditor Guidance**: 
- Synthetic scenarios are operating within expected realism envelopes.
- No immediate action required.
- Continue monitoring for drift.

---

### Pattern 2: RED + High Pressure + High Confidence

**Example:**
```json
{
  "status_light": "RED",
  "global_pressure": 0.85,
  "outliers": ["scenario_aggressive", "scenario_extreme"],
  "overall_confidence": 0.80
}
```

**Interpretation**: "Model realism likely out of spec; treat as high-priority investigation."

**Auditor Guidance**:
- Significant divergence from realism envelopes detected.
- High confidence indicates the divergence is well-characterized.
- Investigate:
  - Which scenarios are violating bounds
  - Whether violations are intentional (stress-testing) or unintentional
  - Whether scenario parameters need adjustment
- Review `top_adjustments` in `realism_constraints` for recommended fixes.

---

### Pattern 3: RED + High Pressure + Low Confidence

**Example:**
```json
{
  "status_light": "RED",
  "global_pressure": 0.75,
  "outliers": ["scenario_unstable"],
  "overall_confidence": 0.35
}
```

**Interpretation**: "Realism constraints indicate issues, but confidence in adjustments is low; investigate both the realism violations and the adjustment uncertainty."

**Auditor Guidance**:
- Realism violations detected, but adjustment recommendations have low confidence.
- Possible causes:
  - Unstable scenarios (SHARP_DRIFT) reducing confidence
  - Complex violations requiring multiple parameter adjustments
  - Rare event or variance issues with uncertain fixes
- Investigate:
  - Root cause of low confidence (check `stability_class` in ratchet)
  - Whether violations are systematic or isolated
  - Whether manual parameter tuning is needed vs. automated adjustments

---

### Pattern 4: YELLOW + Moderate Pressure + Moderate Confidence

**Example:**
```json
{
  "status_light": "YELLOW",
  "global_pressure": 0.45,
  "outliers": ["scenario_edge_case"],
  "overall_confidence": 0.65
}
```

**Interpretation**: "Realism constraints show moderate divergence; monitor and review."

**Auditor Guidance**:
- Some divergence from envelopes, but not critical.
- Monitor for trends (increasing pressure over time).
- Review outliers to determine if they are intentional edge cases.
- Consider whether adjustments are needed or if current state is acceptable.

---

## Status Light Determination Logic

The `status_light` is determined by the following rules (in order of precedence):

1. **RED**: If any of:
   - Calibration status is "BLOCK"
   - `global_pressure > 0.7`
   - `overall_confidence < 0.5`

2. **YELLOW**: If any of:
   - Calibration status is "ATTENTION"
   - `global_pressure > 0.3`
   - `len(outliers) > 3`

3. **GREEN**: Otherwise

---

## Relationship to Other Governance Systems

The First Light Realism Annex is derived from:
- **Director Tile**: Provides `status_light`, `global_pressure`, `overall_confidence`
- **Coupling Map**: Provides `outliers`

The annex does **NOT**:
- Modify ratchet outputs
- Influence calibration console decisions
- Gate uplift experiments
- Trigger automatic escalations

All escalation decisions remain in:
- `build_scenario_calibration_console()` (calibration_status)
- `build_synthetic_realism_ratchet()` (stability_class, retention_score)
- Upstream governance systems

---

## Auditor Workflow

1. **Check `status_light`**: Quick indicator of overall realism health
2. **Review `global_pressure`**: Quantifies divergence magnitude
3. **Examine `outliers`**: Identifies scenarios that may need attention
4. **Assess `overall_confidence`**: Indicates reliability of adjustment recommendations
5. **Cross-reference**: Review full `realism_constraints` section for:
   - `director_tile.headline`: Human-readable summary
   - `top_adjustments`: Recommended parameter changes
   - `coupling_summary`: Clustering and outlier details

---

## Implementation Notes

- **Deterministic**: Same inputs always produce same annex output
- **JSON-Safe**: All values are JSON serializable
- **SHADOW-MODE**: Read-only, advisory only, no side effects
- **Non-Mutating**: Does not modify input analysis

---

## Realism Cards for Calibration Experiments

For Phase 5 (P5), calibration experiments use **Realism Cards** to show per-experiment how "realistic" the Twin looks relative to the Real adapter. These cards are compact dashboard tiles for per-experiment realism assessment.

### Card Structure

Cards are located at:
```
evidence["governance"]["realism_cards"]["cards"][<index>]
```

Each card has the following schema:
```json
{
  "schema_version": "1.0.0",
  "cal_id": "cal_exp_001",
  "status_light": "GREEN" | "YELLOW" | "RED",
  "global_pressure": 0.0-1.0,
  "overall_confidence": 0.0-1.0
}
```

### Field Descriptions

- **`schema_version`**: Version identifier (currently "1.0.0")
- **`cal_id`**: Calibration experiment identifier (e.g., "cal_exp_001")
- **`status_light`**: Realism status indicator (same logic as First Light annex)
- **`global_pressure`**: Realism pressure value [0.0, 1.0] for this experiment
- **`overall_confidence`**: Confidence score [0.0, 1.0] for adjustments in this experiment

### Card Roll-up Structure

Multiple cards are stored under:
```
evidence["governance"]["realism_cards"]
```

Structure:
```json
{
  "label": "PHASE II — SYNTHETIC TEST DATA ONLY",
  "schema_version": "1.0.0",
  "generated_at": "ISO timestamp",
  "cards": [card1, card2, ...],
  "card_count": 2
}
```

### Reading Calibration Experiment Cards

**For Reviewers:**

1. **Per-Experiment Assessment**: Each card shows realism status for a single calibration experiment
   - Compare `status_light` across experiments to identify patterns
   - Track `global_pressure` trends across experiments
   - Note experiments with low `overall_confidence` (may need manual review)

2. **Dashboard View**: Cards are designed for dashboard display
   - GREEN cards: Twin realism consistent with Real adapter
   - YELLOW cards: Moderate divergence, monitor
   - RED cards: Significant divergence, investigate

3. **Cross-Experiment Analysis**: Review multiple cards together
   - Are RED cards clustered in specific experiment types?
   - Is `global_pressure` increasing across experiment sequence?
   - Are confidence scores consistent or variable?

4. **Interpretation Patterns** (same as First Light annex):
   - **GREEN + low pressure (< 0.3) + high confidence (> 0.7)**: "Twin realism looks consistent with scenario assumptions for this experiment."
   - **RED + high pressure (> 0.7) + high confidence (> 0.7)**: "Model realism likely out of spec for this experiment; treat as high-priority investigation."
   - **RED + high pressure + low confidence (< 0.5)**: "Realism constraints indicate issues, but confidence in adjustments is low; investigate both violations and uncertainty."
   - **YELLOW + moderate pressure (0.3-0.7) + moderate confidence (0.5-0.7)**: "Realism constraints show moderate divergence; monitor and review."

### Relationship to First Light Annex

- **First Light Annex**: Global realism snapshot across all scenarios
- **Calibration Cards**: Per-experiment realism assessment
- Cards are derived from the same underlying analysis but provide experiment-specific views
- Both are purely observational; no gating semantics

### Realism-Divergence Consistency Check

For each calibration experiment card, a consistency check can be performed against divergence windows to identify when realism pressure aligns with high divergence, or when there is a conflict.

**Function**: `summarize_realism_vs_divergence(realism_card, cal_exp_windows)`

**Output Structure**:
```json
{
  "schema_version": "1.0.0",
  "consistency_status": "CONSISTENT" | "TENSION" | "CONFLICT",
  "advisory_notes": ["note1", "note2", ...]
}
```

**Consistency Status Rules**:

1. **CONSISTENT**: 
   - `status_light` GREEN and `final_divergence_rate` < 0.5 (low divergence)
   - OR `status_light` RED and `final_divergence_rate` >= 0.7 (high divergence)
   - Interpretation: Realism assessment aligns with observed divergence patterns

2. **CONFLICT**: 
   - `status_light` GREEN but persistently high divergence (>= 3 windows with `divergence_rate` >= 0.9)
   - Interpretation: Realism status suggests good alignment, but divergence windows show persistent high divergence. This may indicate a mismatch between realism constraints and observed behavior.

3. **TENSION**: 
   - All other cases
   - Interpretation: Moderate inconsistency between realism assessment and divergence patterns. Monitor and review.

**Advisory Notes**:
- Maximum 3 notes per consistency check
- Neutral, descriptive language only
- Provides context for the consistency status

**Integration with Evidence**:
- When `attach_realism_cards_to_evidence()` is called with a `cal_exp_windows_map` parameter, divergence consistency is automatically computed and attached to each card under `divergence_consistency`
- The windows map should be: `{cal_id: [window1, window2, ...]}`
- Each window should have a `divergence_rate` field (float [0.0, 1.0])

**Example Usage**:
```python
# Build card
card = build_cal_exp_realism_card("cal_exp_001", annex)

# Define windows
windows = [
    {"window_index": 0, "divergence_rate": 0.1},
    {"window_index": 1, "divergence_rate": 0.2},
    {"window_index": 2, "divergence_rate": 0.3},
]

# Check consistency
consistency = summarize_realism_vs_divergence(card, windows)

# Attach to evidence (automatic if windows map provided)
evidence = attach_realism_cards_to_evidence(
    evidence, 
    [card], 
    cal_exp_windows_map={"cal_exp_001": windows}
)
```

### Implementation

- **Builder**: `build_cal_exp_realism_card(cal_id, annex)`
- **Roll-up**: `attach_realism_cards_to_evidence(evidence, cards, cal_exp_windows_map=None)`
- **Consistency Check**: `summarize_realism_vs_divergence(realism_card, cal_exp_windows)`
- **Non-mutating**: All functions create new dicts, do not modify inputs
- **Deterministic**: Same inputs always produce same outputs
- **JSON-Safe**: All values are JSON serializable

---

## References

- Implementation: `experiments/synthetic_uplift/constraint_solver.py`
- Tests: `tests/experiments/test_realism_constraint_solver_rehydrated.py`
- Related: `docs/system_law/First_Light_Auditor_Guide.md`

