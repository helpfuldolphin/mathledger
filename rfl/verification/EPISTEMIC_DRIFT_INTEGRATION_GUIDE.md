# Epistemic Drift Integration Guide

This guide shows how to integrate epistemic drift analysis into P3/P4 reports and evidence packs.

## Overview

Epistemic drift functions are **observational only** and **never block** pipeline execution. They provide cognitive safety signals for the Phase X Evidence Spine.

## P3 Stability Report Integration

Add `epistemic_summary` block to `first_light_stability_report.json` after metrics windows finalize:

```python
from rfl.verification import build_epistemic_summary_for_p3, build_epistemic_abstention_profile
from rfl.verification.abstention_semantics import build_epistemic_drift_timeline

# After metrics windows finalize, collect epistemic profiles
profiles = []
for window in metrics_windows:
    # Build epistemic profile from window data
    snapshot = {
        "slice_name": config.slice_name,
        "by_category": {
            "timeout_related": window.get("timeout_pct", 0.0),
            "crash_related": window.get("crash_pct", 0.0),
            "invalid_related": window.get("invalid_pct", 0.0),
        }
    }
    profile = build_epistemic_abstention_profile(snapshot)
    profiles.append(profile)

# Build epistemic summary
epistemic_summary = build_epistemic_summary_for_p3(profiles)

# Add to stability report
stability_report["epistemic_summary"] = epistemic_summary
```

**Integration Point:** In `FirstLightShadowRunner.finalize()`, after `_metrics_accumulator.finalize_partial()` and before writing the report.

## P4 Calibration Report Integration

Add `epistemic_calibration` section to `p4_calibration_report.json`:

```python
from rfl.verification import build_epistemic_calibration_for_p4

# Collect epistemic profiles from P4 run
profiles = []
for cycle in p4_cycles:
    snapshot = extract_abstention_snapshot(cycle)
    profile = build_epistemic_abstention_profile(snapshot)
    profiles.append(profile)

# Build epistemic calibration
epistemic_calibration = build_epistemic_calibration_for_p4(profiles)

# Add to calibration report
calibration_report["epistemic_calibration"] = epistemic_calibration
```

**Integration Point:** In `FirstLightShadowRunnerP4.finalize()`, after divergence statistics are computed.

## Evidence Pack Integration

Attach epistemic drift to evidence packs:

```python
from rfl.verification import (
    attach_epistemic_drift_to_evidence,
    build_epistemic_drift_timeline,
    build_abstention_storyline,
)

# Build drift timeline and storyline from profiles
drift_timeline = build_epistemic_drift_timeline(profiles)
storyline = build_abstention_storyline(profiles)

# Attach to evidence pack
evidence_pack = {
    "run_id": run_id,
    "artifacts": {...},
    # governance section will be created by attach function
}
attach_epistemic_drift_to_evidence(evidence_pack, drift_timeline, storyline)

# evidence_pack["governance"]["epistemic_drift"] now contains:
# - drift_band
# - drift_index
# - key_transitions (top 5 by magnitude)
# - storyline_episodes
# - summary_text
```

**Integration Point:** When building evidence pack manifest, after all P3/P4 artifacts are collected.

## Schema Versions

- `EPISTEMIC_SUMMARY_SCHEMA_VERSION = "1.0.0"` (P3)
- `EPISTEMIC_CALIBRATION_SCHEMA_VERSION = "1.0.0"` (P4)
- `EPISTEMIC_EVIDENCE_SCHEMA_VERSION = "1.0.0"` (Evidence)

## Shadow Mode Contract

All functions:
- ✅ Are read-only and side-effect free (except `attach_epistemic_drift_to_evidence` which mutates evidence dict deterministically)
- ✅ Never block or modify pipeline behavior
- ✅ Are deterministic (same inputs = same outputs)
- ✅ Produce JSON-serializable outputs
- ✅ Gracefully handle empty inputs (return defaults)

## Error Handling

Functions never raise exceptions for invalid inputs. They return safe defaults:
- Empty profiles → LOW risk, STABLE drift
- Missing fields → Default values
- Invalid data → Neutral defaults

This ensures the pipeline never fails due to epistemic drift analysis.

## First Light Epistemic Footprint

The First Light epistemic footprint provides a compact cognitive safety snapshot combining P3 (synthetic) and P4 (real shadow) epistemic drift signals.

### Footprint Structure

The footprint is built using `build_first_light_epistemic_footprint()` and contains:

```json
{
  "schema_version": "1.0.0",
  "p3_drift_band": "STABLE" | "DRIFTING" | "VOLATILE",
  "p4_drift_band": "STABLE" | "DRIFTING" | "VOLATILE",
  "p3_mean_risk": "LOW" | "MEDIUM" | "HIGH",
  "p4_mean_risk": "LOW" | "MEDIUM" | "HIGH"
}
```

### Semantics: P3 vs P4 Drift Bands

**P3 Drift Band (`p3_drift_band`):**
- Measures epistemic risk volatility in **synthetic First-Light experiments**
- Based on abstention patterns across synthetic state trajectories
- Indicates how stable epistemic risk is under controlled synthetic conditions
- **STABLE**: Consistent risk band across synthetic windows
- **DRIFTING**: Moderate variation in risk bands
- **VOLATILE**: High variation, suggesting synthetic stress reveals instability

**P4 Drift Band (`p4_drift_band`):**
- Measures epistemic risk volatility in **real shadow runs**
- Based on abstention patterns from actual runner behavior (shadow mode)
- Indicates how stable epistemic risk is in real-world conditions
- **STABLE**: Consistent risk band across real cycles
- **DRIFTING**: Moderate variation in risk bands
- **VOLATILE**: High variation, suggesting real-world conditions reveal instability

### Semantics: P3 vs P4 Mean Risk

**P3 Mean Risk (`p3_mean_risk`):**
- Average epistemic risk across **synthetic experiment windows**
- Reflects how epistemic uncertainty manifests under synthetic stress
- **LOW**: Synthetic conditions show low epistemic uncertainty
- **MEDIUM**: Moderate epistemic uncertainty under synthetic stress
- **HIGH**: High epistemic uncertainty, synthetic stress reveals gaps

**P4 Mean Risk (`p4_mean_risk`):**
- Average epistemic risk across **real shadow cycles**
- Reflects how epistemic uncertainty manifests in actual deployment conditions
- **LOW**: Real-world conditions show low epistemic uncertainty
- **MEDIUM**: Moderate epistemic uncertainty in real conditions
- **HIGH**: High epistemic uncertainty, real conditions reveal gaps

### Interpreting P3 and P4 Together

When reviewing the footprint, consider the relationship between P3 (synthetic) and P4 (real):

**Conservative Pattern (P3 > P4):**
- P3 shows higher risk/drift than P4
- **Interpretation**: Synthetic stress is more aggressive than real-world conditions
- **Implication**: System is conservatively tested; real-world may be safer than synthetic suggests
- **Example**: `P3: DRIFTING + HIGH; P4: STABLE + LOW` → Synthetic stress is higher than real behavior; treat as conservative, not dangerous

**Convergent Pattern (P3 ≈ P4):**
- P3 and P4 show similar risk/drift levels
- **Interpretation**: Synthetic conditions align with real-world behavior
- **Implication**: Synthetic testing accurately reflects real-world epistemic risk
- **Example**: `P3: STABLE + LOW; P4: STABLE + LOW` → Consistent low risk in both synthetic and real conditions

**Divergent Pattern (P4 > P3):**
- P4 shows higher risk/drift than P3
- **Interpretation**: Real-world conditions reveal epistemic uncertainty not captured in synthetic tests
- **Implication**: Synthetic testing may be insufficient; real-world conditions expose additional risk
- **Example**: `P3: STABLE + LOW; P4: VOLATILE + HIGH` → Real conditions show higher epistemic risk than synthetic; requires investigation

**High Risk in Both:**
- Both P3 and P4 show high risk
- **Interpretation**: Epistemic uncertainty is present in both synthetic and real conditions
- **Implication**: Fundamental epistemic gaps exist regardless of test environment
- **Example**: `P3: VOLATILE + HIGH; P4: DRIFTING + HIGH` → Consistent high risk across environments; significant epistemic concerns

### Cognitive Safety Indicator (Not Automatic Block)

**Important**: The First Light epistemic footprint is a **cognitive safety indicator**, not an automatic blocking mechanism.

- **Purpose**: Provides human reviewers with a compact summary of epistemic risk patterns
- **Usage**: Supports evidence-based decision making and risk assessment
- **Not Used For**: Automatic pipeline blocking, hard gates, or enforcement
- **Review Process**: External reviewers should consider the footprint alongside other evidence (TDA metrics, red flags, divergence statistics)

The footprint enables reviewers to quickly assess:
1. Whether synthetic testing (P3) is conservative or aggressive relative to real conditions (P4)
2. Whether epistemic risk is consistent across test environments
3. Whether real-world conditions reveal additional epistemic uncertainty

This information supports informed governance decisions but does not replace human judgment or other safety signals.

## Epistemic Calibration Panel

The epistemic calibration panel provides a multi-experiment triage view for CAL-EXP-* calibration experiments, aggregating footprints across multiple experiments to identify patterns.

### Panel Structure

The panel is built using `build_epistemic_calibration_panel()` and contains:

```json
{
  "schema_version": "1.0.0",
  "num_experiments": 3,
  "num_conservative": 1,  # P3 > P4
  "num_divergent": 1,     # P4 > P3
  "num_high_risk_both": 1,  # Both HIGH risk
  "experiments": [...]  # Original footprints
}
```

### Per-Experiment Footprint Emission

For each calibration experiment (CAL-EXP-1, CAL-EXP-2, CAL-EXP-3), emit a footprint:

**Manual Emission (when P3/P4 data already extracted):**
```python
from rfl.verification import emit_cal_exp_epistemic_footprint

# For each calibration experiment
p3_summary = build_epistemic_summary_for_p3(p3_profiles)
p4_calibration = build_epistemic_calibration_for_p4(p4_profiles)

footprint = emit_cal_exp_epistemic_footprint(
    cal_id="CAL-EXP-1",
    p3_summary=p3_summary,
    p4_calibration=p4_calibration
)

# Persist as calibration/epistemic_footprint_CAL-EXP-1.json
import json
with open(f"calibration/epistemic_footprint_{footprint['cal_id']}.json", "w") as f:
    json.dump(footprint, f, indent=2)
```

**Auto-Emission (from CAL-EXP report):**
```python
from rfl.verification import emit_epistemic_footprint_from_cal_exp_report

# Load CAL-EXP report
with open("cal_exp1_report.json") as f:
    cal_exp_report = json.load(f)

# Auto-emit footprint (extracts P3/P4 data automatically)
footprint = emit_epistemic_footprint_from_cal_exp_report("CAL-EXP-1", cal_exp_report)

# Footprint includes extraction_path indicating how data was extracted:
# - "DIRECT": report["epistemic_summary"] / report["epistemic_calibration"]
# - "NESTED": report["p3"]["epistemic_summary"] / report["p4"]["epistemic_calibration"]
# - "FALLBACK": report["epistemic_alignment_summary"] / report["epistemic_alignment"]
# - "DEFAULTS": Safe defaults (STABLE, LOW) with confidence="LOW" and advisory_note="DEFAULTS_USED"

# Persist footprint
with open(f"calibration/epistemic_footprint_{footprint['cal_id']}.json", "w") as f:
    json.dump(footprint, f, indent=2)
```

**Extraction Precedence (enforced in order):**
1. **DIRECT**: `report["epistemic_summary"]` and `report["epistemic_calibration"]` at top level
2. **NESTED**: `report["p3"]["epistemic_summary"]` and `report["p4"]["epistemic_calibration"]`
3. **FALLBACK**: `report["epistemic_alignment_summary"]` and `report["epistemic_alignment"]` (derived)
4. **DEFAULTS**: Safe defaults (`STABLE`, `LOW`) with `confidence="LOW"` and `advisory_note="DEFAULTS_USED"`

**Default Semantics:**
When `extraction_path="DEFAULTS"`, the footprint includes:
- `p3_drift_band="STABLE"`, `p4_drift_band="STABLE"`
- `p3_mean_risk="LOW"`, `p4_mean_risk="LOW"`
- `confidence="LOW"`
- `advisory_note="DEFAULTS_USED"`

This ensures that default values are never mistaken for actual measurements.

**Extraction Audit:**
Every footprint includes an `extraction_audit` field that logs which extraction paths were checked and what fields were found. This provides transparency for auditors to understand why a particular extraction path was chosen.

The audit log is a list of 4 entries in deterministic order:
1. `{"path": "DIRECT", "found": bool, "fields": List[str]}`
2. `{"path": "NESTED", "found": bool, "fields": List[str]}`
3. `{"path": "FALLBACK", "found": bool, "fields": List[str]}`
4. `{"path": "DEFAULTS", "found": bool, "fields": List[str]}`

Each entry indicates:
- `path`: The extraction path name
- `found`: Whether complete data was found for this path (both P3 and P4)
- `fields`: List of field names that were present (e.g., `["epistemic_summary", "epistemic_calibration"]`)

**Strict Extraction Mode:**
The `emit_epistemic_footprint_from_cal_exp_report()` function accepts an optional `strict_extraction` parameter (default: `False`). When enabled:

- If multiple complete sources are simultaneously present (e.g., both DIRECT and NESTED have complete data), the function:
  - Uses the highest precedence path (DIRECT in this example)
  - Adds `advisory_note="MULTIPLE_SOURCES_PRESENT"` (or combines with `DEFAULTS_USED` if defaults were also used)
  - This is **advisory only** - no exceptions are raised, no blocking occurs

This allows auditors to detect ambiguous cases where multiple extraction sources are available, ensuring transparency without enforcement semantics.

**Example with Strict Mode:**
```python
footprint = emit_epistemic_footprint_from_cal_exp_report(
    "CAL-EXP-1",
    cal_exp_report,
    strict_extraction=True  # Enable multiple source detection
)

# If both DIRECT and NESTED are present:
# - extraction_path = "DIRECT" (precedence)
# - advisory_note = "MULTIPLE_SOURCES_PRESENT"
# - extraction_audit shows both DIRECT and NESTED as found=True
```

**Integration Point:** After each CAL-EXP-* experiment completes, emit and persist the footprint.

### Panel Aggregation

Aggregate multiple experiment footprints into a panel:

```python
from rfl.verification import (
    build_epistemic_calibration_panel,
    attach_epistemic_calibration_panel_to_evidence,
)

# Collect footprints from all experiments
footprints = []
for cal_id in ["CAL-EXP-1", "CAL-EXP-2", "CAL-EXP-3"]:
    with open(f"calibration/epistemic_footprint_{cal_id}.json") as f:
        footprints.append(json.load(f))

# Build panel
panel = build_epistemic_calibration_panel(footprints)

# Panel includes:
# - num_experiments
# - num_conservative (P3 > P4)
# - num_divergent (P4 > P3)
# - num_high_risk_both (both HIGH)
# - dominant_pattern: "CONSERVATIVE" | "CONVERGENT" | "DIVERGENT" | "HIGH_RISK_BOTH" | "MIXED"
# - dominant_pattern_confidence: float in [0.0, 1.0] (margin between top and runner-up)
# - experiments (original footprints)

# Attach to evidence pack with optional advisory notes
advisory_notes = [
    "P3 and P4 show stable alignment across experiments",
    "No significant epistemic drift detected",
]
attach_epistemic_calibration_panel_to_evidence(evidence_pack, panel, advisory_notes)

# evidence_pack["governance"]["epistemic_calibration_panel"] now contains all panel fields
# plus advisory_notes (neutral observation strings)
```

**Dominant Pattern Confidence:**
The `dominant_pattern_confidence` field indicates how clear the dominant pattern is:
- **High confidence (> 0.5)**: Clear majority pattern (e.g., 3/3 conservative)
- **Low confidence (< 0.5)**: Mixed or close patterns (e.g., 1 conservative, 1 divergent, 1 convergent)
- **Confidence = 0.0**: Empty panel or perfectly balanced patterns

Confidence is calculated as the margin between the top pattern percentage and the runner-up percentage, clamped to [0.0, 1.0].

**Integration Point:** When building evidence pack manifest, after all CAL-EXP-* footprints are collected.

### Panel Interpretation

**Conservative Pattern (`num_conservative`):**
- Count of experiments where P3 shows higher risk/drift than P4
- Indicates synthetic stress is more aggressive than real conditions
- **Triage Action**: Review synthetic test parameters; may be conservatively tested

**Divergent Pattern (`num_divergent`):**
- Count of experiments where P4 shows higher risk/drift than P3
- Indicates real conditions reveal epistemic uncertainty not captured in synthetic tests
- **Triage Action**: Investigate real-world conditions; synthetic testing may need enhancement

**High Risk in Both (`num_high_risk_both`):**
- Count of experiments where both P3 and P4 show HIGH risk
- Indicates fundamental epistemic gaps regardless of test environment
- **Triage Action**: Priority review; significant epistemic concerns across environments

**Usage for CAL-EXP Triage:**
- Use panel counts to prioritize which experiments need deeper investigation
- `num_high_risk_both > 0` → Priority review
- `num_divergent > num_conservative` → Review synthetic test adequacy
- `num_conservative > num_divergent` → Synthetic tests may be overly conservative

### Schema Versions

- `CAL_EXP_FOOTPRINT_SCHEMA_VERSION = "1.0.0"` (Per-experiment footprint)
- `CALIBRATION_PANEL_SCHEMA_VERSION = "1.0.0"` (Multi-experiment panel)

