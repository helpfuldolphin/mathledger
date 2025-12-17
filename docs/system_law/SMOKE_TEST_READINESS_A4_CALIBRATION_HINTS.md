# Smoke-Test Readiness Checklist: A4 Calibration Readiness Hints v1.0.0

**Component:** Performance Calibration Summary — Readiness Hints with Schema Versioning  
**Version:** 1.0.0  
**Date:** 2025-12-11  
**Status:** ✅ READY FOR DEPLOYMENT

---

## Schema Versioning Validation

- [x] **Hint Schema Version Field**
  - [x] `hint_schema_version: "1.0.0"` present in all hint dictionaries
  - [x] Schema version is constant across all hint types (monotonic, plateau, oscillatory)
  - [x] Schema version enables version-aware parsing for future migrations

- [x] **Backward Compatibility**
  - [x] `_extract_hint_string()` handles old string format
  - [x] `_extract_hint_string()` handles new dict format
  - [x] `extract_calibration_readiness_status()` supports mixed formats in same run
  - [x] Test coverage for backward compatibility scenarios

---

## Transparency Fields Validation

- [x] **Hint Field**
  - [x] All hints include `hint` field with valid values:
    - [x] `"READY_FOR_EXTENDED_RUN"`
    - [x] `"NEEDS_PARAMETER_TUNING"`
    - [x] `"UNSTABLE_CALIBRATION"`

- [x] **Basis Field**
  - [x] All hints include `basis` field
  - [x] `basis` value is `"delta_p_trajectory_shape"` (only allowed value in v1.0.0)
  - [x] Documentation enumerates allowed basis values
  - [x] Future basis values documented as extensible

- [x] **Scope Note Field**
  - [x] All hints include `scope_note` field
  - [x] `scope_note` value is constant: `"ADVISORY_ONLY_NO_GATE"` in v1.0.0
  - [x] Documentation explicitly states scope_note is constant in v1.0.0
  - [x] Future scope values documented as extensible

---

## Implementation Validation

- [x] **Function: `map_uplift_shape_to_readiness_hint()`**
  - [x] Returns dict with `hint_schema_version`, `hint`, `basis`, `scope_note`
  - [x] Handles all uplift shapes (monotonic, plateau, oscillatory, unknown)
  - [x] Works with and without trajectory data
  - [x] All return paths include schema version

- [x] **Function: `build_perf_calibration_summary()`**
  - [x] Stores expanded hint dictionaries (not strings)
  - [x] Each experiment hint includes all four fields
  - [x] Handles CAL-EXP-1 and CAL-EXP-2 data
  - [x] Handles empty data gracefully

- [x] **Function: `extract_calibration_readiness_status()`**
  - [x] Supports old string format (backward compatibility)
  - [x] Supports new dict format (v1.0.0)
  - [x] Supports mixed formats in same summary
  - [x] Priority ordering works correctly (UNSTABLE > TUNING > READY)

- [x] **Helper Function: `_extract_hint_string()`**
  - [x] Handles string format (old)
  - [x] Handles dict format (new)
  - [x] Returns "UNKNOWN" for invalid formats
  - [x] Exported in `__all__` for testing

---

## Test Coverage Validation

- [x] **Uplift Shape Classification Tests** (6 tests)
  - [x] Monotonic increasing
  - [x] Monotonic decreasing
  - [x] Plateau
  - [x] Oscillatory
  - [x] Insufficient data
  - [x] Two points

- [x] **Readiness Hint Mapping Tests** (6 tests)
  - [x] Monotonic decreasing → READY
  - [x] Monotonic increasing → READY
  - [x] Plateau → NEEDS_TUNING
  - [x] Oscillatory → UNSTABLE
  - [x] Unknown shape → NEEDS_TUNING
  - [x] Without trajectory → works correctly
  - [x] All tests validate `hint_schema_version` field

- [x] **Calibration Summary Tests** (4 tests)
  - [x] CAL-EXP-1 only
  - [x] Both CAL-EXP-1 and CAL-EXP-2
  - [x] Empty data
  - [x] Missing mean_delta_p
  - [x] All tests validate expanded hint structure

- [x] **Evidence Integration Tests** (7 tests)
  - [x] Attachment to evidence pack
  - [x] Read-only contract
  - [x] JSON serialization safety
  - [x] Non-blocking behavior
  - [x] Status extraction (comprehensive)
  - [x] Backward compatibility (old format)
  - [x] **Mixed format ingestion (old string + new dict in same run)** ✅ NEW

**Total Test Count:** 23 tests, all passing ✅

---

## Documentation Validation

- [x] **Evidence Pack Spec Section 11.7.5**
  - [x] Transparency Fields section updated
  - [x] `hint_schema_version` field documented
  - [x] `basis` field enumerates allowed values:
    - [x] `"delta_p_trajectory_shape"` (only value in v1.0.0)
    - [x] Future extensibility noted
  - [x] `scope_note` field explicitly states constant value in v1.0.0
  - [x] Future extensibility noted for scope_note

- [x] **Schema Examples**
  - [x] All JSON examples include `hint_schema_version: "1.0.0"`
  - [x] All examples show complete hint structure
  - [x] Interpretation text references schema version

- [x] **Field Descriptions Table**
  - [x] Updated to reflect dict structure (not string)
  - [x] All four fields documented

---

## Integration Validation

- [x] **Backward Compatibility**
  - [x] Old string format still works
  - [x] New dict format works
  - [x] Mixed formats in same run work
  - [x] Status extraction handles all formats

- [x] **JSON Serialization**
  - [x] All hint structures are JSON-serializable
  - [x] Evidence pack with hints is JSON-serializable
  - [x] Round-trip serialization works

- [x] **Evidence Pack Integration**
  - [x] Hints attach correctly to evidence pack
  - [x] Read-only contract maintained
  - [x] Additive behavior (preserves existing governance data)

---

## Code Quality Validation

- [x] **Type Hints**
  - [x] Function signatures include type hints
  - [x] Return types documented

- [x] **Docstrings**
  - [x] All functions have docstrings
  - [x] Return value structures documented
  - [x] Backward compatibility noted

- [x] **Exports**
  - [x] `_extract_hint_string` exported in `__all__` for testing
  - [x] All public functions exported

- [x] **Linting**
  - [x] No linter errors
  - [x] Code follows project style guidelines

---

## Migration Readiness

- [x] **Version Detection**
  - [x] Schema version field enables version detection
  - [x] Old format (string) can be detected and handled
  - [x] New format (dict) can be detected and parsed

- [x] **Future Extensibility**
  - [x] Schema version field supports future versions
  - [x] Basis field documented as extensible
  - [x] Scope note field documented as extensible (constant in v1.0.0)

- [x] **Migration Path**
  - [x] Backward compatibility maintained
  - [x] Mixed format support enables gradual migration
  - [x] Status extraction handles both formats transparently

---

## Smoke Test Execution

### Pre-Deployment Checklist

- [x] All 23 tests pass
- [x] No linter errors
- [x] Documentation updated
- [x] Schema versioning explicit
- [x] Mixed format test exists
- [x] Backward compatibility verified

### Post-Deployment Validation

- [ ] Verify hints include `hint_schema_version: "1.0.0"` in production
- [ ] Verify backward compatibility with existing string-format hints
- [ ] Verify mixed format ingestion works in real calibration runs
- [ ] Verify status JSON extraction works with both formats
- [ ] Verify evidence pack attachment includes schema version

---

## Summary

**Status:** ✅ **READY FOR DEPLOYMENT**

All requirements met:
- ✅ Explicit schema versioning (`hint_schema_version: "1.0.0"`)
- ✅ Mixed format ingestion test exists
- ✅ Documentation enumerates allowed `basis` values
- ✅ Documentation states `scope_note` is constant in v1.0.0
- ✅ All 23 tests passing
- ✅ Backward compatibility verified
- ✅ No linter errors

**Next Steps:**
1. Deploy to production
2. Execute post-deployment validation checklist
3. Monitor for backward compatibility issues
4. Document any production observations

---

**Signed Off By:** A4 — Performance Ratchet Custodian  
**Date:** 2025-12-11  
**Version:** 1.0.0

