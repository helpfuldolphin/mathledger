# Realism↔Divergence Consistency — Smoke-Test Readiness Checklist

**Component**: `summarize_realism_vs_divergence()` + Window Normalization  
**Status**: SHADOW-ONLY, Advisory  
**Last Updated**: 2025-12-12

---

## ✅ Implementation Checklist

### Core Functionality
- [x] **Parameterized Thresholds**: All thresholds are configurable via optional parameters
  - `low_divergence_threshold` (default: 0.5)
  - `high_divergence_threshold` (default: 0.7)
  - `persistent_high_divergence_threshold` (default: 0.9)
  - `persistent_window_count` (default: 3)

- [x] **Window Normalization**: `_extract_divergence_rate()` supports multiple key paths
  - Direct key: `divergence_rate` → source_path: "DIRECT"
  - Nested under metrics: `metrics.divergence_rate` → source_path: "METRICS_NESTED"
  - Nested under summary: `summary.divergence_rate` → source_path: "SUMMARY_NESTED"
  - Returns `(None, "MISSING")` if not found (graceful degradation)
  - Returns tuple: `(value: Optional[float], source_path: str)`

- [x] **Explainability Fields**: Output includes transparency fields
  - `windows_analyzed`: Number of windows successfully analyzed
  - `high_divergence_window_count`: Number of windows with high divergence
  - `divergence_rate_sources`: Dict counting windows by source_path (DIRECT, METRICS_NESTED, SUMMARY_NESTED, MISSING)
  - `windows_dropped_missing_rate`: Count of windows with MISSING divergence_rate

- [x] **Strict Mode**: Optional `strict_window_extraction` parameter
  - If `True` and any window is MISSING, sets status to "INCONCLUSIVE"
  - Still advisory-only (no exceptions, no gating)
  - Provides audit trail for schema compliance
  - Output includes `strict_mode_contract` field with:
    - `enabled`: bool
    - `missing_windows_count`: int
    - `status_when_missing`: "INCONCLUSIVE"

- [x] **Source Path Constants**: Explicit constants and coercion
  - `SOURCE_PATH_DIRECT`, `SOURCE_PATH_METRICS_NESTED`, `SOURCE_PATH_SUMMARY_NESTED`, `SOURCE_PATH_MISSING`
  - `_coerce_source_path()` ensures unknown paths never emitted (falls back to MISSING)
  - Defensive programming: warns on unknown paths but always returns valid value

- [x] **Consistency Status Logic**: Deterministic classification
  - INCONCLUSIVE: (if strict_window_extraction=True and any window is MISSING)
  - CONSISTENT: (GREEN + low divergence) OR (RED + high divergence)
  - CONFLICT: GREEN but persistently high divergence
  - TENSION: All other cases

---

## ✅ Test Coverage

### Parameter Override Tests
- [x] `test_summarize_realism_vs_divergence_parameter_overrides_change_classification`
  - Verifies that custom thresholds change classification

- [x] `test_summarize_realism_vs_divergence_persistent_window_count_override`
  - Verifies that `persistent_window_count` affects CONFLICT detection

- [x] `test_summarize_realism_vs_divergence_persistent_threshold_override`
  - Verifies that `persistent_high_divergence_threshold` affects high divergence count

### Window Normalization Tests
- [x] `test_summarize_realism_vs_divergence_window_normalization_direct_key`
  - Direct `divergence_rate` key extraction

- [x] `test_summarize_realism_vs_divergence_window_normalization_metrics_nested`
  - Nested `metrics.divergence_rate` extraction

- [x] `test_summarize_realism_vs_divergence_window_normalization_summary_nested`
  - Nested `summary.divergence_rate` extraction

- [x] `test_summarize_realism_vs_divergence_window_normalization_mixed_schemas`
  - Mixed window schemas (direct + nested) handled correctly

- [x] `test_summarize_realism_vs_divergence_window_normalization_missing_keys`
  - Graceful handling when some windows lack divergence_rate

### Source Path Tracking Tests
- [x] `test_summarize_realism_vs_divergence_source_path_tracking_direct`
  - Verifies DIRECT source path counting

- [x] `test_summarize_realism_vs_divergence_source_path_tracking_metrics_nested`
  - Verifies METRICS_NESTED source path counting

- [x] `test_summarize_realism_vs_divergence_source_path_tracking_summary_nested`
  - Verifies SUMMARY_NESTED source path counting

- [x] `test_summarize_realism_vs_divergence_source_path_tracking_mixed`
  - Verifies mixed schema source path tracking

- [x] `test_summarize_realism_vs_divergence_source_path_tracking_missing_only`
  - Verifies MISSING source path counting

- [x] `test_summarize_realism_vs_divergence_source_path_deterministic`
  - Verifies determinism of source path tracking

- [x] `test_summarize_realism_vs_divergence_source_path_json_safe`
  - Verifies JSON serializability of source path data

### Strict Mode Tests
- [x] `test_summarize_realism_vs_divergence_strict_mode_no_missing`
  - Verifies strict mode does not affect result when no missing windows

- [x] `test_summarize_realism_vs_divergence_strict_mode_with_missing`
  - Verifies strict mode sets INCONCLUSIVE when any window is MISSING

- [x] `test_summarize_realism_vs_divergence_strict_mode_all_missing`
  - Verifies strict mode sets INCONCLUSIVE when all windows are MISSING

- [x] `test_summarize_realism_vs_divergence_strict_mode_contract_enabled`
  - Verifies strict_mode_contract is present and correct when strict mode enabled

- [x] `test_summarize_realism_vs_divergence_strict_mode_contract_disabled`
  - Verifies strict_mode_contract is present and correct when strict mode disabled

- [x] `test_summarize_realism_vs_divergence_strict_mode_audit_counts_deterministic`
  - Verifies strict mode produces INCONCLUSIVE while still reporting audit counts deterministically

- [x] `test_coerce_source_path_valid_paths`
  - Verifies _coerce_source_path accepts valid paths

- [x] `test_coerce_source_path_invalid_path_fallback`
  - Verifies _coerce_source_path falls back to MISSING for unknown paths

### Explainability Tests
- [x] `test_summarize_realism_vs_divergence_explainability_fields`
  - Verifies `windows_analyzed` and `high_divergence_window_count` are present and correct

### Determinism Tests
- [x] `test_summarize_realism_vs_divergence_deterministic_with_parameters`
  - Verifies determinism with custom parameters

- [x] `test_summarize_realism_vs_divergence_deterministic`
  - Verifies determinism with default parameters

### JSON Safety Tests
- [x] `test_summarize_realism_vs_divergence_json_safe`
  - Verifies JSON serializability

---

## ✅ Integration Checklist

### Evidence Attachment
- [x] `attach_realism_cards_to_evidence()` supports optional `cal_exp_windows_map`
- [x] Divergence consistency automatically computed when windows map provided
- [x] Cards enhanced with `divergence_consistency` field
- [x] Backward compatible: works without windows map

### Edge Cases
- [x] Empty windows list → TENSION status, `windows_analyzed=0`
- [x] Windows without divergence_rate → TENSION status, `windows_analyzed=0`
- [x] Mixed schemas → All valid windows analyzed
- [x] Missing keys → Graceful degradation, only valid windows counted

---

## ✅ Code Quality

### Documentation
- [x] Function docstring includes all parameter descriptions
- [x] Window normalization helper documented
- [x] Consistency status rules explained
- [x] Examples provided in docstring

### Type Safety
- [x] Type hints for all parameters
- [x] Return type annotation
- [x] Optional types used correctly

### Error Handling
- [x] Graceful handling of missing keys
- [x] Type checking for divergence_rate values
- [x] Safe defaults for edge cases

---

## ✅ Smoke Test Commands

### Basic Functionality
```bash
# Test with default parameters
uv run pytest tests/experiments/test_realism_constraint_solver_rehydrated.py::test_summarize_realism_vs_divergence_consistent_green_low_divergence -v

# Test with custom parameters
uv run pytest tests/experiments/test_realism_constraint_solver_rehydrated.py::test_summarize_realism_vs_divergence_parameter_overrides_change_classification -v
```

### Window Normalization
```bash
# Test all normalization paths
uv run pytest tests/experiments/test_realism_constraint_solver_rehydrated.py -k "window_normalization" -v

# Test mixed schemas
uv run pytest tests/experiments/test_realism_constraint_solver_rehydrated.py::test_summarize_realism_vs_divergence_window_normalization_mixed_schemas -v
```

### Explainability
```bash
# Test explainability fields
uv run pytest tests/experiments/test_realism_constraint_solver_rehydrated.py::test_summarize_realism_vs_divergence_explainability_fields -v

# Test source path tracking
uv run pytest tests/experiments/test_realism_constraint_solver_rehydrated.py -k "source_path" -v
```

### Strict Mode
```bash
# Test strict mode behavior
uv run pytest tests/experiments/test_realism_constraint_solver_rehydrated.py -k "strict_mode" -v
```

### Integration
```bash
# Test evidence attachment with divergence consistency
uv run pytest tests/experiments/test_realism_constraint_solver_rehydrated.py::test_attach_realism_cards_to_evidence_with_divergence_consistency -v
```

### Full Suite
```bash
# Run all consistency-related tests
uv run pytest tests/experiments/test_realism_constraint_solver_rehydrated.py -k "divergence" -v
```

---

## ✅ Pre-Deployment Verification

### Determinism
- [x] Same inputs produce same outputs (with and without parameters)
- [x] No time-dependent behavior
- [x] No random number generation

### JSON Safety
- [x] All outputs JSON serializable
- [x] Round-trip serialization works
- [x] No non-serializable types (datetime, etc.)

### Shadow-Mode Compliance
- [x] No side effects
- [x] Non-mutating (creates new dicts)
- [x] No gating semantics
- [x] Purely advisory

### Performance
- [x] O(n) complexity for n windows
- [x] No unnecessary iterations
- [x] Efficient key lookup

---

## 📋 Known Limitations

1. **Window Schema Assumptions**: Assumes windows are dictionaries. Non-dict windows will be skipped.
2. **Threshold Values**: Default thresholds (0.5, 0.7, 0.9) are descriptive only, not RTTS acceptance criteria.
3. **Nested Key Depth**: Only supports one level of nesting (`metrics.divergence_rate`, not `metrics.data.divergence_rate`).
4. **Strict Mode**: When `strict_window_extraction=True` and any window is MISSING, status is set to INCONCLUSIVE. This is advisory-only and does not raise exceptions or gate operations.
5. **Source Path Audit**: Source path tracking provides audit trail but does not validate schema compliance. It only reports which paths were used.

---

## 🚀 Deployment Readiness

**Status**: ✅ READY FOR SMOKE TEST

All implementation tasks complete, test coverage comprehensive, edge cases handled, documentation complete.

**Next Steps**:
1. Run full test suite: `uv run pytest tests/experiments/test_realism_constraint_solver_rehydrated.py -v`
2. Verify integration with calibration experiment harness
3. Validate with real calibration experiment windows
4. Monitor for any schema variations in production windows

---

## 📝 Change Log

- **2025-12-12**: Initial implementation
  - Parameterized thresholds
  - Window normalization helper
  - Explainability fields
  - Comprehensive test coverage

- **2025-12-12**: Window schema audit trail
  - Extended `_extract_divergence_rate()` to return `(value, source_path)` tuple
  - Added `divergence_rate_sources` field tracking counts by source path
  - Added `windows_dropped_missing_rate` field
  - Added `strict_window_extraction` parameter with INCONCLUSIVE status
  - Added comprehensive source path tracking tests
  - Added strict mode tests

- **2025-12-12**: Window Schema Audit v1 Freeze (REALITY LOCK)
  - Added explicit source_path constants: `SOURCE_PATH_DIRECT`, `SOURCE_PATH_METRICS_NESTED`, `SOURCE_PATH_SUMMARY_NESTED`, `SOURCE_PATH_MISSING`
  - Added `_coerce_source_path()` function with assert + fallback to MISSING for unknown paths
  - Added `strict_mode_contract` field to output with `enabled`, `missing_windows_count`, `status_when_missing`
  - Ensured unknown path labels never emitted (coercion enforces valid values)
  - Added tests for strict_mode_contract and audit count determinism
  - Updated all return statements to include strict_mode_contract

