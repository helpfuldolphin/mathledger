# Harmonic Alignment Phase X Binding

**STATUS:** PHASE X — HARMONIC GOVERNANCE TILE

**Agent:** E2 — Harmonic Alignment Governance Engineer

**SHADOW MODE:** All harmonic alignment signals are purely observational. No control flow depends on harmonic values.

---

## Overview

The Harmonic Alignment system provides governance tiles that summarize semantic-curriculum harmonic alignment across P3 First-Light and P4 calibration runs. It answers the question: "Is the curriculum harmonically aligned with how we're actually using it?"

The system integrates:
- **Harmonic Map**: Combines semantic alignment, curriculum alignment, and atlas coupling views
- **Evolution Forecaster**: Predicts curriculum evolution needs
- **Director Panel**: Provides unified status for dashboards
- **Curriculum Annex**: Curriculum-centric diagnostic from P3/P4 data

---

## Curriculum Harmonic Annex (First Light)

The **Curriculum Harmonic Annex** is a curriculum-centric summary that combines harmonic alignment data from P3 First-Light summary and P4 calibration report. It provides reviewers with a diagnostic view of curriculum health and alignment stability.

### Schema

The annex is attached to evidence packs under `evidence["governance"]["harmonic_alignment"]["curriculum_annex"]` and has the following structure:

```json
{
  "schema_version": "1.0.0",
  "harmonic_band": "COHERENT" | "PARTIAL" | "MISMATCHED",
  "evolution_status": "STABLE" | "EVOLVING" | "DIVERGING",
  "misaligned_concepts": ["slice_name1", "slice_name2", ...],
  "priority_adjustments": ["slice_name1", "slice_name2", ...]
}
```

### Field Definitions

- **`harmonic_band`**: Overall harmonic alignment status
  - `COHERENT`: High alignment (≥0.8 average harmonic score)
  - `PARTIAL`: Moderate alignment (≥0.5 average harmonic score)
  - `MISMATCHED`: Low alignment (<0.5 average harmonic score)

- **`evolution_status`**: Curriculum evolution forecast
  - `STABLE`: No adjustments needed, coherent alignment
  - `EVOLVING`: Some adjustments needed, partial alignment
  - `DIVERGING`: Significant adjustments needed, mismatched alignment

- **`misaligned_concepts`**: List of slice names where semantic alignment is present but curriculum alignment is missing (max 10, sorted, deduplicated from P3 and P4)

- **`priority_adjustments`**: List of slice names requiring priority attention based on harmonic scores (max 5, from P3 summary, preserves order)

### Example Interpretations

#### Example 1: Stable Curriculum

```json
{
  "harmonic_band": "COHERENT",
  "evolution_status": "STABLE",
  "misaligned_concepts": ["slice_rare_edge_case"],
  "priority_adjustments": []
}
```

**Interpretation:** "Curriculum shape stable across P3/P4."

The curriculum shows coherent harmonic alignment with minimal misalignment. The single misaligned concept is likely an edge case that doesn't require immediate attention. Evolution status indicates stability, suggesting the curriculum structure is well-aligned with semantic and atlas views.

#### Example 2: Diverging Curriculum

```json
{
  "harmonic_band": "MISMATCHED",
  "evolution_status": "DIVERGING",
  "misaligned_concepts": ["slice_a", "slice_b", "slice_c", "slice_d", "slice_e"],
  "priority_adjustments": ["slice_a", "slice_b", "slice_c"]
}
```

**Interpretation:** "Curriculum semantics and harmonic structure diverging; review slice design."

The curriculum shows significant misalignment across multiple slices. The high number of misaligned concepts combined with a diverging evolution status indicates that the curriculum structure is not aligned with how it's being used semantically. Reviewers should examine the slice design for these priority adjustments to understand why semantic alignment exists but curriculum alignment is missing.

### Evidence-Only, Not a Gate

**IMPORTANT:** The Curriculum Harmonic Annex is **evidence-only** and **not a gate**.

- The annex is purely observational and does not influence any control flow
- It does not block or gate any operations
- It is provided for reviewer diagnostics and curriculum health assessment
- No system behavior depends on the annex contents
- It is attached to evidence packs for analysis and review purposes only

The annex serves as a diagnostic tool to help reviewers understand curriculum alignment health, but it does not enforce any policies or block any operations. All harmonic alignment signals operate in **SHADOW MODE**.

---

## Integration Points

### P3 First-Light Summary

The annex extracts `harmonic_band` and `priority_adjustments` from the P3 First-Light summary's `harmonic_alignment_summary` field.

### P4 Calibration Report

The annex extracts `evolution_status` and `misaligned_concepts` from the P4 calibration report's `harmonic_alignment` field.

### Evidence Packs

The annex is attached to evidence packs via `attach_harmonic_alignment_to_evidence()` when both P3 summary and P4 calibration data are provided.

### Uplift Council

The council summary includes counts (`num_misaligned_concepts`, `num_priority_adjustments`) for decision-making, but the annex itself is advisory only.

---

## Curriculum Harmonic Grid (CAL-EXP)

The **Curriculum Harmonic Grid** aggregates curriculum harmonic annexes across multiple calibration experiments (CAL-EXP-1/2/3) to provide a cross-experiment alignment witness over time.

### Grid Schema

The grid is attached to evidence packs under `evidence["governance"]["harmonic_curriculum_panel"]` and has the following structure:

```json
{
  "schema_version": "1.0.0",
  "num_experiments": 3,
  "harmonic_band_counts": {
    "COHERENT": 1,
    "PARTIAL": 1,
    "MISMATCHED": 1
  },
  "top_misaligned_concepts": [
    {
      "concept": "slice_b",
      "frequency": 2,
      "experiments": ["CAL-EXP-2", "CAL-EXP-3"]
    }
  ],
  "top_driver_concepts": ["slice_b", "slice_a", "slice_d"],
  "top_driver_cal_ids": {
    "slice_b": ["CAL-EXP-2", "CAL-EXP-3"],
    "slice_a": ["CAL-EXP-1"],
    "slice_d": ["CAL-EXP-3"]
  }
}
```

### Field Definitions

- **`num_experiments`**: Total number of calibration experiments analyzed

- **`harmonic_band_counts`**: Frequency counts of harmonic bands across experiments
  - `COHERENT`: Number of experiments with coherent alignment
  - `PARTIAL`: Number of experiments with partial alignment
  - `MISMATCHED`: Number of experiments with mismatched alignment

- **`top_misaligned_concepts`**: List of concepts that appear frequently across experiments (sorted by frequency descending, limited to top 10)
  - Each entry contains:
    - `concept`: Slice name
    - `frequency`: Number of experiments where this concept appears as misaligned
    - `experiments`: List of experiment IDs where this concept appears (sorted)

- **`top_driver_concepts`**: List of top 5 misaligned concepts ranked by frequency (descending), then concept name (ascending)
  - These are the primary drivers of harmonic misalignment across experiments
  - Sorted deterministically for consistent reporting

- **`top_driver_cal_ids`**: Mapping from concept name to list of experiment IDs where that concept appears
  - Each list is sorted for determinism
  - Provides quick lookup of which experiments contribute to each top driver

### Per-Experiment Snapshots

Each calibration experiment emits a snapshot via `emit_cal_exp_curriculum_harmonic_annex()` which is persisted to:
- `calibration/curriculum_harmonic_annex_<cal_id>.json`

These snapshots capture the harmonic alignment state for each individual experiment and are aggregated into the grid for cross-experiment analysis.

### Using the Grid for Auditors

**IMPORTANT:** The Curriculum Harmonic Grid is **evidence-only** and **not a gate**.

Auditors should use the grid to:

1. **Alignment Witness Over Time**: Track how harmonic alignment evolves across multiple calibration experiments. A pattern of increasing `MISMATCHED` counts or growing `top_misaligned_concepts` frequencies may indicate curriculum drift.

2. **Cross-Experiment Patterns**: Identify concepts that consistently appear as misaligned across experiments. These may indicate systemic curriculum design issues rather than experiment-specific anomalies.

3. **Stability Assessment**: Compare `harmonic_band_counts` to assess curriculum stability. A high proportion of `COHERENT` experiments suggests stable curriculum alignment, while increasing `MISMATCHED` counts suggest divergence.

**The grid does NOT:**
- Gate or block any operations
- Influence control flow
- Enforce any policies
- Replace human judgment

The grid serves as an **alignment witness** providing observational data for curriculum health assessment across calibration experiments, but it does not enforce decisions or block operations.

### Mock-vs-Real Delta Tracking

When both mock and real calibration experiment grids are available, a **delta** can be computed to compare alignment patterns between mock and real runs. The delta is attached under `evidence["governance"]["harmonic_curriculum_panel"]["delta"]`.

#### Delta Schema

```json
{
  "schema_version": "1.0.0",
  "top_driver_overlap": ["slice_a", "slice_b"],
  "top_driver_only_mock": ["slice_c"],
  "top_driver_only_real": ["slice_d"],
  "frequency_shifts": [
    {
      "concept": "slice_a",
      "mock_frequency": 2,
      "real_frequency": 3,
      "delta": 1
    }
  ]
}
```

#### Delta Field Definitions

- **`top_driver_overlap`**: Concepts that appear in both mock and real top driver lists (sorted)

- **`top_driver_only_mock`**: Concepts that appear only in mock top drivers (sorted)

- **`top_driver_only_real`**: Concepts that appear only in real top drivers (sorted)

- **`frequency_shifts`**: List of concepts with frequency changes between mock and real
  - Each entry contains:
    - `concept`: Slice name
    - `mock_frequency`: Frequency in mock experiments
    - `real_frequency`: Frequency in real experiments
    - `delta`: Difference (real - mock)
  - Sorted by absolute delta (descending), then concept name (ascending)

#### Using the Delta for Auditors

The delta provides observational data for comparing mock and real calibration behavior:

1. **Overlap Analysis**: Concepts appearing in both mock and real top drivers indicate consistent misalignment patterns that may require curriculum design review.

2. **Mock-Only Concepts**: Concepts appearing only in mock may indicate synthetic telemetry artifacts that don't reflect real usage patterns.

3. **Real-Only Concepts**: Concepts appearing only in real may indicate real-world usage patterns not captured in mock telemetry.

4. **Frequency Shifts**: Increasing frequency from mock to real suggests concepts that become more problematic in real environments, while decreasing frequency suggests concepts that are better handled in real runs.

**IMPORTANT:** The delta is **evidence-only** and **not a gate**. It provides observational data for calibration analysis but does not influence any control flow or block operations.

---

## Related Documentation

- `backend/health/harmonic_alignment_adapter.py`: Core adapter implementation
- `backend/health/harmonic_alignment_p3p4_integration.py`: P3/P4 integration, annex builder, and grid builder
- `tests/health/test_harmonic_curriculum_annex.py`: Annex test suite
- `tests/health/test_curriculum_harmonic_grid.py`: Grid test suite

---

**Last Updated:** Phase X — Harmonic Curriculum Annex & CAL-EXP Grid Implementation

