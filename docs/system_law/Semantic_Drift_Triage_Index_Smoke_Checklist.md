# Semantic Drift Triage Index — Smoke Test Checklist

## Overview

This checklist verifies that the Semantic Drift Triage Index status hook is correctly integrated and functioning. All tests should pass cleanly.

## Pre-Flight Checks

### 1. Import Error Resolution
- [x] **Fixed**: Lazy import guard added for `harmonic_alignment_p3p4_integration` functions
- [x] **Verified**: `generate_first_light_status.py` imports without errors
- [x] **Location**: `scripts/generate_first_light_status.py` lines 2154-2162

### 2. Core Functionality Tests

Run the following test suites:

```bash
# Semantic drift integration tests (42 tests)
uv run pytest tests/ci/test_semantic_drift_integrations.py -q

# Status integration tests (10 tests)
uv run pytest tests/scripts/test_generate_first_light_status_semantic_failure_triage_index.py -q
```

**Expected**: All tests pass (52 total)

## Test Coverage Verification

### Status Integration Tests (10 tests)

1. ✅ **test_triage_index_signal_extracted_from_manifest**
   - Verifies manifest-first extraction
   - Checks `extraction_source == "manifest"`
   - Validates signal structure

2. ✅ **test_triage_index_signal_fallback_to_evidence_json**
   - Verifies fallback to evidence.json
   - Checks `extraction_source == "evidence.json"`

3. ✅ **test_triage_index_signal_missing_index_safe**
   - Verifies graceful handling when index is missing
   - No errors, no signal added

4. ✅ **test_triage_index_signal_advisory_warning_on_regressed**
   - Verifies warning generation when REGRESSED detected
   - Checks warning includes top cal_id

5. ✅ **test_triage_index_signal_no_warning_when_stable**
   - Verifies no warning when all items are STABLE

6. ✅ **test_triage_index_signal_deterministic**
   - Verifies deterministic output across runs

7. ✅ **test_triage_index_signal_top5_truncates_to_5**
   - Verifies top5 truncation logic

8. ✅ **test_triage_index_signal_warning_cap**
   - Verifies exactly one warning (not per item)
   - Checks warning includes top 3 cal_ids

9. ✅ **test_ggfl_adapter_stub_sig_sdrift**
   - Verifies GGFL adapter stub produces correct signal
   - Checks weight="LOW", conflict=False, status="WARN" when REGRESSED

10. ✅ **test_ggfl_adapter_stub_status_ok_when_stable**
    - Verifies GGFL adapter returns OK status when stable

## Signal Schema Verification

### Status Signal Structure

When triage index is present, verify `first_light_status.json` contains:

```json
{
  "signals": {
    "semantic_failure_triage_index": {
      "schema_version": "1.0.0",
      "mode": "SHADOW",
      "extraction_source": "manifest" | "evidence.json",
      "total_items": <int>,
      "top5": [
        {
          "cal_id": "<str>",
          "regression_status": "REGRESSED" | "ATTENTION" | "STABLE",
          "combined_tensor_norm": <float>,
          "hotspots_count": <int>
        }
      ],
      "advisory_warning": "<str>" // Optional, present if REGRESSED detected
    }
  }
}
```

### Warning Hygiene Verification

- [x] **Exactly one warning** emitted when REGRESSED detected (not per item)
- [x] **Top cal_ids included** (≤3) in warning text
- [x] **Format**: `"{base_warning} Top cal_ids: {cal_ids_str}"`

Example warning:
```
Semantic drift triage index contains 3 experiment(s) with REGRESSED status. Review recommended. Top cal_ids: CAL-EXP-1, CAL-EXP-2, CAL-EXP-3
```

## GGFL Adapter Stub (SIG-SDRIFT)

### Signal Structure

```json
{
  "weight": "LOW",
  "conflict": false,
  "status": "OK" | "WARN",
  "advisory_note": "<str>", // Optional, present if REGRESSED in top5
  "total_items": <int>,
  "regressed_count": <int>
}
```

### Behavior Verification

- [x] **Weight**: Always "LOW"
- [x] **Conflict**: Always `false`
- [x] **Status**: "WARN" if any REGRESSED in top5, else "OK"
- [x] **Advisory note**: Present if REGRESSED detected, includes top 3 cal_ids

## Integration Points

### Manifest-First Extraction

1. **Primary**: `manifest["governance"]["semantic_failure_triage_index"]`
2. **Fallback**: `evidence.json["governance"]["semantic_failure_triage_index"]`
3. **Source tracking**: `extraction_source` field indicates source

### Error Handling

- [x] **Non-fatal**: Import failures are caught and skipped
- [x] **Missing index**: Safe (no signal added, no errors)
- [x] **Invalid structure**: Caught in try/except, skipped gracefully

## Smoke Test Commands

### Quick Verification

```bash
# Run all semantic drift tests
uv run pytest tests/ci/test_semantic_drift_integrations.py -q

# Run status integration tests
uv run pytest tests/scripts/test_generate_first_light_status_semantic_failure_triage_index.py -q

# Run both (full smoke)
uv run pytest tests/ci/test_semantic_drift_integrations.py tests/scripts/test_generate_first_light_status_semantic_failure_triage_index.py -q
```

### Expected Output

```
tests/ci/test_semantic_drift_integrations.py ................ [ 42 passed ]
tests/scripts/test_generate_first_light_status_semantic_failure_triage_index.py .......... [ 10 passed ]
============================= 52 passed in X.XXs ==============================
```

## Manual Verification Steps

1. **Generate status with triage index in manifest**:
   ```bash
   python scripts/generate_first_light_status.py \
     --p3-dir <p3_dir> \
     --p4-dir <p4_dir> \
     --evidence-pack-dir <evidence_pack_dir>
   ```

2. **Verify signal appears**:
   - Check `first_light_status.json`
   - Verify `signals["semantic_failure_triage_index"]` exists
   - Verify `extraction_source` is "manifest"

3. **Verify warning (if REGRESSED)**:
   - Check `warnings` array
   - Verify exactly one warning mentioning REGRESSED
   - Verify warning includes top cal_ids (≤3)

4. **Test fallback**:
   - Remove triage index from manifest
   - Add to `evidence.json` instead
   - Regenerate status
   - Verify `extraction_source` is "evidence.json"

## Known Issues

- None

## Status

✅ **All smoke tests passing**
✅ **Import error resolved**
✅ **Warning hygiene verified**
✅ **GGFL adapter stub implemented**
✅ **Extraction source tracking added**

## Last Verified

- Date: 2025-12-11
- Tests: 52/52 passing
- Status: READY FOR PRODUCTION

