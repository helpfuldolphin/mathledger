---

## 13. Smoke-Test Readiness Checklist: Phase-Lag Reconciliation

Before using `explain_phase_lag_vs_divergence()` in production reconciliation workflows, verify the following prerequisites:

### 13.1 Data Prerequisites

| # | Item | Verification Command | Expected |
|---|------|---------------------|----------|
| 1 | `compute_phase_lag_index()` available | `from backend.health.semantic_tda_timeline import compute_phase_lag_index` | No ImportError |
| 2 | `extract_correlation_for_pattern_classifier()` available | `from backend.health.semantic_tda_adapter import extract_correlation_for_pattern_classifier` | No ImportError |
| 3 | `decompose_divergence_components()` available | `from experiments.u2.runtime import decompose_divergence_components` | No ImportError |
| 4 | Phase lag data structure | `phase_lag.get("phase_lag_index")` returns float | Non-None result |
| 5 | Divergence decomposition structure | `divergence_decomp.get("state_divergence_rate")` returns float | Non-None result |

### 13.2 Function Output Validation

| # | Field | Type | Required | Verification |
|---|-------|------|----------|-------------|
| 1 | `schema_version` | str | Yes | Must be "1.0.0" |
| 2 | `phase_lag_index` | float or None | Yes | Must be in [0, 1] or None |
| 3 | `state_divergence_rate` | float or None | Yes | Must be in [0, 1] or None |
| 4 | `outcome_divergence_rate_success` | float or None | Yes | Must be in [0, 1] or None |
| 5 | `interpretation` | str | Yes | Must be one of: STATE_LAG_DOMINANT, OUTCOME_NOISE_DOMINANT, MIXED, INSUFFICIENT_DATA |
| 6 | `notes` | List[str] | Yes | Must be non-empty list |
| 7 | `thresholds` | Dict[str, float] | Yes | Must contain: high_phase_lag (0.4), high_state_divergence (0.3), high_outcome_divergence (0.2) |
| 8 | `basis` | Dict[str, str] | Yes | Must contain: phase_lag ("semantic_tda_timeline"), divergence_decomp ("runtime_profile_calibration.decompose_divergence_components") |

### 13.3 Interpretation Logic Validation

| # | Test Case | Phase Lag | State Div | Outcome Div | Expected Interpretation |
|---|-----------|-----------|-----------|-------------|------------------------|
| 1 | State lag dominant | ≥ 0.4 | ≥ 0.3 | < 0.2 | STATE_LAG_DOMINANT |
| 2 | Outcome noise dominant | ≥ 0.4 | < 0.3 | ≥ 0.2 | OUTCOME_NOISE_DOMINANT |
| 3 | Mixed (both high) | ≥ 0.4 | ≥ 0.3 | ≥ 0.2 | MIXED |
| 4 | Mixed (low lag, high state) | < 0.4 | ≥ 0.3 | < 0.2 | MIXED |
| 5 | Insufficient data | None | None | None | INSUFFICIENT_DATA |
| 6 | Low lag, low divergence | < 0.4 | < 0.3 | < 0.2 | OUTCOME_NOISE_DOMINANT |

### 13.4 Integration Points

| # | Integration | File | Function/Method |
|---|-------------|------|-----------------|
| 1 | Phase lag extraction | `backend/health/semantic_tda_adapter.py` | `extract_correlation_for_pattern_classifier()` |
| 2 | Divergence decomposition | `experiments/u2/runtime/calibration_correlation.py` | `decompose_divergence_components()` |
| 3 | Reconciliation function | `backend/health/semantic_tda_timeline.py` | `explain_phase_lag_vs_divergence()` |
| 4 | Metric law reference | `docs/system_law/calibration/METRIC_DEFINITIONS.md` | Section 8 |

### 13.5 Test File Structure

```
tests/
├── test_phase_lag_reconciliation.py
│   ├── TestPhaseLagVsDivergence
│   │   ├── test_state_lag_dominant_interpretation
│   │   ├── test_outcome_noise_dominant_interpretation
│   │   ├── test_mixed_interpretation
│   │   ├── test_insufficient_data_interpretation
│   │   ├── test_low_phase_lag_with_high_state_divergence
│   │   ├── test_low_phase_lag_with_low_divergence
│   │   ├── test_outcome_divergence_rate_success_key_variant
│   │   ├── test_result_is_json_serializable
│   │   ├── test_result_is_deterministic
│   │   ├── test_high_phase_lag_with_low_divergence_rates
│   │   └── test_integration_with_extract_correlation_for_pattern_classifier
```

### 13.6 SHADOW MODE Verification

| # | Check | Expected |
|---|-------|----------|
| 1 | Function output includes `mode: "SHADOW"` or equivalent note | Yes (via notes) |
| 2 | No enforcement actions triggered by interpretation | Yes |
| 3 | No governance state modified by function | Yes |
| 4 | All outputs logged to telemetry | Yes (advisory-only) |
| 5 | Verifier note present in documentation | Yes (Section 8.10) |

### 13.7 Determinism and JSON Safety

| # | Check | Expected |
|---|-------|----------|
| 1 | Same inputs produce same output | Yes (deterministic) |
| 2 | Output is JSON serializable | Yes |
| 3 | Thresholds are explicit in output | Yes (in `thresholds` field) |
| 4 | Basis fields indicate data sources | Yes (in `basis` field) |
| 5 | Notes are neutral and descriptive | Yes |

### 13.8 Sign-Off Checklist

- [ ] All data prerequisites verified (13.1)
- [ ] Function output structure validated (13.2)
- [ ] Interpretation logic test cases pass (13.3)
- [ ] Integration points identified (13.4)
- [ ] Test file structure complete (13.5)
- [ ] SHADOW MODE contract verified (13.6)
- [ ] Determinism and JSON safety verified (13.7)

**Status:** READY FOR RECONCILIATION when all items checked.

---

## 14. TDA Windowed Patterns: GGFL Adapter + Disagreement Hook

### 14.1 Signal Format (SIG-TDAW)

The GGFL adapter `tda_windowed_patterns_for_alignment_view()` produces:

```json
{
  "signal_type": "SIG-TDAW",
  "status": "ok | warn",
  "conflict": false,
  "weight_hint": "LOW",
  "extraction_source": "MANIFEST | EVIDENCE_JSON | MISSING",
  "drivers": [
    "DRIVER_DOMINANT_PATTERN:<pattern>",
    "DRIVER_STREAK:<pattern>(<length>)",
    "DRIVER_WINDOWED_DETECTED_PATTERN | DRIVER_SINGLE_SHOT_DETECTED_PATTERN"
  ],
  "summary": "<one sentence>"
}
```

### 14.2 Reason Codes

| Reason Code | Meaning |
|-------------|---------|
| `DRIVER_DOMINANT_PATTERN:<X>` | Dominant windowed pattern is X (not NONE) |
| `DRIVER_STREAK:<X>(<N>)` | Pattern X detected in N consecutive windows |
| `DRIVER_WINDOWED_DETECTED_PATTERN` | Single-shot=NONE but windowed detected pattern |
| `DRIVER_SINGLE_SHOT_DETECTED_PATTERN` | Single-shot detected pattern but windowed=NONE |

### 14.3 Extraction Source

| Value | Meaning |
|-------|---------|
| `MANIFEST` | From manifest.signals.tda_windowed_patterns or manifest.governance.tda.windowed_patterns |
| `EVIDENCE_JSON` | Fallback from evidence.json signals or governance |
| `MISSING` | No signal found |

### 14.4 Warning Hygiene

- Windowed patterns warning: max 1 line
- Disagreement warning: max 1 line
- Format: `TDA pattern disagreement: DRIVER_<code> (single-shot=X, windowed_dominant=Y)`

### 14.5 Smoke Checklist

**Test Command:**
```bash
uv run python -m pytest tests/health/test_tda_windowed_patterns_adapter.py -v
```

**Expected:** 49 tests passing

**Key Verifications:**
- [ ] `extraction_source` present in GGFL output
- [ ] `extraction_source` normalized to MANIFEST|EVIDENCE_JSON|MISSING
- [ ] Drivers use `DRIVER_` prefix format
- [ ] Disagreement uses `reason_code` (not `disagreement_type`)
- [ ] No `advisory_note` prose in disagreement result
- [ ] Warnings capped to 1 per type
- [ ] All outputs deterministic

### 14.6 Example: Manifest → Status Pair

**Input Manifest:**
```json
{
  "signals": {
    "tda_windowed_patterns": {
      "schema_version": "1.0.0",
      "mode": "SHADOW",
      "status": {
        "dominant_pattern": "DRIFT",
        "max_streak": {"pattern": "DRIFT", "length": 3},
        "high_confidence_count": 2,
        "coverage": {"total_windows": 10, "windows_with_patterns": 4}
      },
      "top_events": [
        {"window_index": 0, "pattern": "DRIFT", "confidence": 0.85}
      ]
    }
  },
  "governance": {
    "tda": {
      "patterns": {"pattern": "NONE"}
    }
  }
}
```

**Output Status Signal:**
```json
{
  "tda_windowed_patterns": {
    "schema_version": "1.0.0",
    "mode": "SHADOW",
    "dominant_pattern": "DRIFT",
    "max_streak": {"pattern": "DRIFT", "length": 3},
    "high_confidence_count": 2,
    "top_events_count": 1,
    "coverage": {"total_windows": 10, "windows_with_patterns": 4},
    "extraction_source": "MANIFEST"
  },
  "tda_pattern_disagreement": {
    "schema_version": "1.0.0",
    "mode": "SHADOW",
    "disagreement_detected": true,
    "reason_code": "DRIVER_WINDOWED_DETECTED_PATTERN",
    "single_shot_pattern": "NONE",
    "windowed_dominant_pattern": "DRIFT"
  }
}
```

**GGFL Alignment View:**
```json
{
  "signal_type": "SIG-TDAW",
  "status": "warn",
  "conflict": false,
  "weight_hint": "LOW",
  "extraction_source": "MANIFEST",
  "drivers": [
    "DRIVER_DOMINANT_PATTERN:DRIFT",
    "DRIVER_STREAK:DRIFT(3)",
    "DRIVER_WINDOWED_DETECTED_PATTERN"
  ],
  "summary": "TDA windowed analysis detected: dominant pattern DRIFT, disagreement (DRIVER_WINDOWED_DETECTED_PATTERN)."
}
```

**Warnings Generated:**
```
TDA windowed patterns: dominant=DRIFT, streak=DRIFT(3), high_conf=2
TDA pattern disagreement: DRIVER_WINDOWED_DETECTED_PATTERN (single-shot=NONE, windowed_dominant=DRIFT)
```

---

*Document Version: 1.4.2*
*Last Updated: 2025-12-12*
*Status: Specification (Design Freeze)*