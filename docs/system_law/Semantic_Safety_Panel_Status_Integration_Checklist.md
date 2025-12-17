# Semantic Safety Panel Status Integration — Smoke-Test Readiness Checklist

**STATUS:** PHASE X — P5 SEMANTIC SAFETY PANEL STATUS SURFACE INTEGRATION

**Last Updated:** 2025-12-09

---

## Overview

This checklist verifies that the semantic safety panel signal is correctly integrated into the First Light status surface (`first_light_status.json`). The integration follows the SHADOW MODE CONTRACT: all signals are advisory only, observational, and do not gate status generation.

---

## Implementation Checklist

### ✅ Evidence Pack Manifest Contract

- [x] **Canonical Location**: Panel stored under `evidence["governance"]["semantic_safety_panel"]`
- [x] **Fallback Location**: Signal can be read from `evidence["signals"]["semantic_safety_panel"]` if already extracted
- [x] **Fallback Logic**: Status generator checks governance first, then signals section
- [x] **Missing Panel Handling**: Missing panel does not cause errors (advisory only)

### ✅ Status Generator Extraction

- [x] **Import Added**: `extract_semantic_safety_panel_signal` imported from `backend.health.semantic_integrity_adapter`
- [x] **Signal Extraction**: Panel extracted from governance using `extract_semantic_safety_panel_signal()` when panel found
- [x] **Signal Structure**: Extracted signal contains:
  - `ok_ok`: int
  - `ok_not_ok`: int
  - `not_ok_ok`: int
  - `not_ok_not_ok`: int
  - `top_drivers`: List[str] (up to 3 cal_ids)
- [x] **Signal Attachment**: Signal attached under `status["signals"]["semantic_safety_panel"]`
- [x] **Exception Handling**: Extraction failures are non-fatal (wrapped in try/except, advisory only)

### ✅ Advisory Warning Generation

- [x] **Warning Condition**: Warning generated when `not_ok_not_ok > 0`
- [x] **Warning Content**: Warning includes:
  - Count of experiments with `not_ok_not_ok` status
  - Top drivers list (comma-separated)
- [x] **Neutral Phrasing**: Warning uses neutral language (no evaluative terms like "bad", "wrong", "error", "critical", "urgent")
- [x] **No Gating**: Warning does not block status generation or modify decisions

### ✅ Test Coverage

- [x] **Status Extraction from Governance**: Test verifies panel extracted from `governance["semantic_safety_panel"]`
- [x] **Status Extraction from Signals Fallback**: Test verifies signal extracted from `signals["semantic_safety_panel"]` when panel not in governance
- [x] **Missing Panel Handling**: Test verifies missing panel does not cause errors
- [x] **Warning Generation**: Test verifies warning generated when `not_ok_not_ok > 0`
- [x] **No Warning When All OK**: Test verifies no warning when all experiments are `ok_ok`
- [x] **Determinism**: Test verifies extraction is deterministic (same input produces same output)
- [x] **Top Drivers in Warning**: Test verifies top drivers are included in warning text
- [x] **Neutral Language**: Test verifies warning uses neutral phrasing (no evaluative terms)

---

## Smoke Test Procedure

### 1. Verify Evidence Pack Structure

```bash
# Check that evidence pack contains semantic safety panel
cat evidence_pack/evidence.json | jq '.governance.semantic_safety_panel'
```

**Expected**: Panel structure with `schema_version`, `total_experiments`, `grid_counts`, `top_drivers`, `experiments`.

### 2. Generate First Light Status

```bash
python scripts/generate_first_light_status.py \
    --p3-dir results/first_light/golden_run/p3 \
    --p4-dir results/first_light/golden_run/p4 \
    --evidence-pack-dir results/first_light/evidence_pack_first_light
```

**Expected**: Status generation completes without errors.

### 3. Verify Signal Extraction

```bash
# Check that signal is present in status
cat evidence_pack/first_light_status.json | jq '.signals.semantic_safety_panel'
```

**Expected**: Signal structure with `ok_ok`, `ok_not_ok`, `not_ok_ok`, `not_ok_not_ok`, `top_drivers`.

### 4. Verify Warning Generation (if applicable)

```bash
# Check warnings for semantic safety panel
cat evidence_pack/first_light_status.json | jq '.warnings[] | select(contains("Semantic safety panel"))'
```

**Expected**: 
- Warning present if `not_ok_not_ok > 0`
- Warning includes count and top drivers
- Warning uses neutral language

### 5. Run Test Suite

```bash
uv run pytest tests/scripts/test_generate_first_light_status_semantic_safety_panel.py -v
```

**Expected**: All 8 tests pass.

---

## Verification Points

### Signal Structure Validation

```json
{
  "signals": {
    "semantic_safety_panel": {
      "ok_ok": 2,
      "ok_not_ok": 0,
      "not_ok_ok": 0,
      "not_ok_not_ok": 1,
      "top_drivers": ["CAL-EXP-3"]
    }
  }
}
```

### Warning Format Validation

```
Semantic safety panel: 1 calibration experiment(s) show semantic issues in both P3 and P4 phases. Top drivers: CAL-EXP-3
```

**Checklist**:
- [ ] Count is accurate
- [ ] Top drivers are listed (comma-separated)
- [ ] Language is neutral (no evaluative terms)
- [ ] Warning does not block status generation

---

## Edge Cases Verified

- [x] **Missing Panel**: Status generation succeeds when panel not present
- [x] **Empty Panel**: Handles empty or malformed panel gracefully
- [x] **Signal Already Extracted**: Uses signal from `signals` section if panel not in governance
- [x] **Extraction Failure**: Non-fatal exception handling (advisory only)
- [x] **All OK**: No warning when all experiments are `ok_ok`
- [x] **Multiple Not-OK**: Warning includes all top drivers (up to 3)

---

## Integration Points

### Evidence Pack Builder

The evidence pack builder should attach the semantic safety panel using:

```python
from backend.health.semantic_integrity_adapter import (
    attach_semantic_safety_panel_to_evidence,
)

evidence = attach_semantic_safety_panel_to_evidence(evidence, panel)
```

This ensures:
- Panel stored under `evidence["governance"]["semantic_safety_panel"]`
- Signal stored under `evidence["signals"]["semantic_safety_panel"]`

### Status Generator

The status generator extracts the signal using:

```python
from backend.health.semantic_integrity_adapter import (
    extract_semantic_safety_panel_signal,
)

semantic_safety_signal = extract_semantic_safety_panel_signal(panel)
```

---

## SHADOW MODE CONTRACT Compliance

- [x] **Read-Only**: Status extraction is read-only (no mutations)
- [x] **Observational**: Signal is purely observational
- [x] **No Gating**: Signal does not gate status generation
- [x] **Advisory Only**: Warnings are advisory only (no blocking)
- [x] **Non-Fatal**: Missing panel or extraction failures are non-fatal
- [x] **Neutral Language**: All warnings use neutral phrasing

---

## References

- **Implementation**: `scripts/generate_first_light_status.py` (lines 1029-1070)
- **Extraction Function**: `backend/health/semantic_integrity_adapter.py::extract_semantic_safety_panel_signal()`
- **Tests**: `tests/scripts/test_generate_first_light_status_semantic_safety_panel.py`
- **Panel Builder**: `backend/health/semantic_integrity_adapter.py::build_semantic_safety_panel()`
- **Documentation**: `docs/system_law/Semantic_Integrity_PhaseX.md`

---

## Status

✅ **READY FOR SMOKE TEST**

All implementation tasks completed, all tests passing, SHADOW MODE CONTRACT verified.

