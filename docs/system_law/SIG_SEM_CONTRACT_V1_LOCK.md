# SIG-SEM CONTRACT v1 LOCK

**STATUS**: REAL-READY — Frozen enums, deterministic reason codes, manifest-first extraction

## Overview

This document locks the Semantic Safety Panel (SIG-SEM) contract v1, including:
- Frozen enums for `extraction_source` and `status`
- Deterministic driver normalization with reason codes
- Manifest-first extraction precedence
- GGFL adapter with reason-code drivers
- Warning hygiene (one line, top 3 drivers only)

## Frozen Enums

### extraction_source

```python
extraction_source ∈ {"MANIFEST", "EVIDENCE_JSON_GOVERNANCE", "EVIDENCE_JSON_SIGNALS", "MISSING"}
```

**Extraction Precedence**:
1. `MANIFEST` — Panel found in `manifest.json["governance"]["semantic_safety_panel"]` (preferred)
2. `EVIDENCE_JSON_GOVERNANCE` — Panel found in `evidence.json["governance"]["semantic_safety_panel"]`
3. `EVIDENCE_JSON_SIGNALS` — Signal already extracted in `evidence.json["signals"]["semantic_safety_panel"]`
4. `MISSING` — No panel or signal found

### status

```python
status ∈ {"ok", "warn"}
```

**Status Logic**:
- `"ok"` — `not_ok_not_ok == 0` (all experiments in OK×OK, OK×Not-OK, or Not-OK×OK buckets)
- `"warn"` — `not_ok_not_ok > 0` (at least one experiment in Not-OK×Not-OK bucket)

## Deterministic Driver Normalization

### Reason Code Format

```python
drivers_reason_codes: List[str]  # Format: "SEM-DRV-001:<cal_id>", "SEM-DRV-002:<cal_id>", "SEM-DRV-003:<cal_id>"
```

**Reason Code Constants**:
- `SEM_DRV_001` — First driver (top cal_id)
- `SEM_DRV_002` — Second driver (second cal_id)
- `SEM_DRV_003` — Third driver (third cal_id)

**Mapping Rules**:
1. Drivers are sorted alphabetically (human-readable `cal_id`s)
2. Reason codes map deterministically: first driver → `SEM-DRV-001`, second → `SEM-DRV-002`, third → `SEM-DRV-003`
3. Maximum 3 drivers and 3 reason codes (even if more drivers exist)
4. Format: `"{REASON_CODE}:{cal_id}"` (e.g., `"SEM-DRV-001:CAL-EXP-1"`)

**Example**:
```python
drivers = ["CAL-EXP-1", "CAL-EXP-2", "CAL-EXP-3"]  # Sorted alphabetically
drivers_reason_codes = [
    "SEM-DRV-001:CAL-EXP-1",
    "SEM-DRV-002:CAL-EXP-2",
    "SEM-DRV-003:CAL-EXP-3",
]
```

## GGFL Adapter Contract

### Function: `semantic_safety_panel_for_alignment_view()`

**Input**: Panel or signal dict (from `build_semantic_safety_panel()` or `first_light_status.json`)

**Output**:
```python
{
    "signal_type": "SIG-SEM",  # Constant
    "status": "ok" | "warn",  # Frozen enum
    "conflict": False,  # Always false (semantic safety panel never triggers conflict)
    "weight_hint": "LOW",  # Advisory only
    "drivers": List[str],  # Human-readable cal_ids (up to 3, sorted)
    "drivers_reason_codes": List[str],  # Deterministic reason codes (up to 3)
    "summary": str,  # One neutral sentence
}
```

**SHADOW MODE CONTRACT**:
- Purely observational (no gating)
- Deterministic output for identical inputs
- Never claims "good/bad", only descriptive
- Low weight hint (advisory only)

## Warning Hygiene

### Warning Generation

**Trigger**: `not_ok_not_ok > 0`

**Format**: One line, neutral wording, includes top 3 drivers only

**Example**:
```
Semantic safety panel: 2 experiment(s) show semantic issues in both P3 and P4 phases (top drivers: CAL-EXP-1, CAL-EXP-2, CAL-EXP-3)
```

**Constraints**:
- Maximum 3 drivers in warning (even if more exist)
- One line only (no newlines)
- Neutral language (no evaluative terms: "bad", "wrong", "error", etc.)

## Manifest-First Extraction

### Extraction Logic

```python
# 1. Try manifest.json first (preferred)
if manifest:
    semantic_safety_panel = manifest.get("governance", {}).get("semantic_safety_panel")
    if semantic_safety_panel:
        extraction_source = "MANIFEST"

# 2. Fallback to evidence.json governance
if not semantic_safety_panel and evidence_data:
    semantic_safety_panel = evidence_data.get("governance", {}).get("semantic_safety_panel")
    if semantic_safety_panel:
        extraction_source = "EVIDENCE_JSON_GOVERNANCE"

# 3. Fallback to evidence.json signals (already extracted)
if not semantic_safety_panel and evidence_data:
    semantic_safety_signal = evidence_data.get("signals", {}).get("semantic_safety_panel")
    if semantic_safety_signal:
        extraction_source = "EVIDENCE_JSON_SIGNALS"

# 4. Missing
if not semantic_safety_panel:
    extraction_source = "MISSING"
```

## Test Coverage

### Required Tests

1. **GGFL Adapter Tests**:
   - `test_ggfl_drivers_reason_codes_deterministic_ordering` — Verify reason codes map correctly
   - `test_ggfl_drivers_reason_codes_limited_to_3` — Verify max 3 drivers/codes
   - `test_ggfl_drivers_reason_codes_empty_when_no_drivers` — Verify empty list when no drivers

2. **Status Generator Tests**:
   - `test_semantic_safety_panel_warning_generated_when_not_ok_not_ok` — Verify warning includes only top 3 drivers
   - `test_semantic_safety_panel_warning_includes_top_drivers` — Verify top 3 limit

3. **Extraction Tests**:
   - `test_semantic_safety_panel_extracted_from_governance` — Verify extraction_source set correctly
   - `test_semantic_safety_panel_extracted_from_signals_fallback` — Verify fallback works

## Smoke-Test Readiness Checklist

### Pre-Commit Verification

- [ ] All tests pass: `uv run pytest tests/health/test_semantic_integrity_adapter.py::TestSemanticSafetyPanelForAlignmentView -v`
- [ ] All tests pass: `uv run pytest tests/scripts/test_generate_first_light_status_semantic_safety_panel.py -v`
- [ ] Linter clean: `uv run ruff check backend/health/semantic_integrity_adapter.py scripts/generate_first_light_status.py`

### Expected Status Snippet

```json
{
  "signals": {
    "semantic_safety_panel": {
      "ok_ok": 2,
      "ok_not_ok": 0,
      "not_ok_ok": 0,
      "not_ok_not_ok": 1,
      "top_drivers": ["CAL-EXP-3"],
      "extraction_source": "MANIFEST"
    }
  },
  "warnings": [
    "Semantic safety panel: 1 experiment(s) show semantic issues in both P3 and P4 phases (top drivers: CAL-EXP-3)"
  ]
}
```

### Expected GGFL Snippet

```json
{
  "signal_type": "SIG-SEM",
  "status": "warn",
  "conflict": false,
  "weight_hint": "LOW",
  "drivers": ["CAL-EXP-3"],
  "drivers_reason_codes": ["SEM-DRV-001:CAL-EXP-3"],
  "summary": "Semantic safety panel: 1 of 3 experiments show semantic issues in both P3 and P4 phases."
}
```

### Determinism Verification

- [ ] Run status generator twice with same inputs → identical `drivers_reason_codes` ordering
- [ ] Run GGFL adapter twice with same inputs → identical `drivers_reason_codes` ordering
- [ ] Verify reason codes are stable across runs (no drift in prose)

## Implementation Files

### Core Implementation

- `backend/health/semantic_integrity_adapter.py`:
  - `SEM_DRV_001`, `SEM_DRV_002`, `SEM_DRV_003` constants
  - `semantic_safety_panel_for_alignment_view()` with reason codes

- `scripts/generate_first_light_status.py`:
  - Manifest-first extraction with `extraction_source` tracking
  - Warning generation with top 3 drivers limit

### Tests

- `tests/health/test_semantic_integrity_adapter.py`:
  - `TestSemanticSafetyPanelForAlignmentView` with reason code tests

- `tests/scripts/test_generate_first_light_status_semantic_safety_panel.py`:
  - Extraction tests with `extraction_source` verification
  - Warning tests with top 3 drivers limit

## Version History

- **v1.0.0** (LOCKED) — Initial contract with frozen enums, reason codes, manifest-first extraction

