# Smoke-Test Readiness Checklist: Budget Calibration Exclusion Trace

**Component:** Budget-aware calibration exclusion recommendations with audit trail  
**Status:** ✅ READY FOR SMOKE TEST  
**Date:** 2025-01-XX

---

## 1. Core Functionality

### 1.1 Exclusion Recommendation Logic
- [x] `compute_calibration_exclusion_recommendation()` implements cross-signal checks
- [x] Exclusion recommended ONLY when ALL conditions met:
  - [x] Budget is confounded (`budget_confounded = True`)
  - [x] PRNG is NOT volatile (`drift_status != "VOLATILE"` or missing)
  - [x] Topology is stable (`pressure_band != "HIGH"` or missing)
- [x] Transient vs persistent distinction (`BUDGET_CONFOUNDED_TRANSIENT` vs `BUDGET_CONFOUNDED_PERSISTENT`)

### 1.2 Trace Canonicalization
- [x] `exclusion_trace` field present in all recommendations
- [x] `missing_signal_policy` field at trace root (`"DEFAULT_TRUE_MISSING"`)
- [x] All checks include `value`, `source`, `raw_value`
- [x] Thresholds explicitly documented in trace
- [x] Trace structure is deterministic (same inputs → same trace)

### 1.3 Missing Signal Handling
- [x] Missing signals default to `DEFAULT_TRUE_MISSING` source
- [x] Missing signals default to `true` value (safe assumption)
- [x] Policy explicitly encoded in `missing_signal_policy` field
- [x] Documentation explains rationale for default-true policy

---

## 2. Annotation Helpers

### 2.1 Window Annotation
- [x] `annotate_calibration_windows_with_exclusion_recommendations()` attaches trace to all windows
- [x] Trace present even when budget modulation is missing
- [x] Trace present even when PRNG/topology signals are missing
- [x] All windows have complete trace for auditability

### 2.2 Convenience Functions
- [x] `annotate_calibration_window_with_exclusion()` for single window
- [x] `annotate_calibration_windows_with_budget_modulation()` for budget modulation
- [x] All functions are read-only (no side effects)

---

## 3. Determinism & Serialization

### 3.1 JSON Serialization
- [x] All trace fields are JSON serializable
- [x] Trace serialization is deterministic (sorted keys)
- [x] Same inputs produce identical JSON output
- [x] Dictionary ordering is consistent across runs

### 3.2 Trace Structure
- [x] Top-level trace keys in consistent order
- [x] `checks` dict keys in consistent order
- [x] `thresholds` dict keys in consistent order
- [x] All nested structures use sorted keys

---

## 4. Tests

### 4.1 Core Logic Tests
- [x] Exclusion recommendation with all signals provided (25 tests)
- [x] Missing signal handling (default-true behavior)
- [x] Cross-signal gating logic (PRNG volatile blocks exclusion)
- [x] Cross-signal gating logic (topology high blocks exclusion)
- [x] Transient vs persistent distinction

### 4.2 Trace Tests
- [x] Trace structure validation
- [x] Missing signal policy in trace
- [x] Trace present when budget modulation missing
- [x] Trace determinism (dict ordering)
- [x] JSON serialization determinism

### 4.3 Integration Tests
- [x] Window annotation with trace
- [x] Multiple windows annotation
- [x] Evidence pack integration (via `build_first_light_budget_summary`)

---

## 5. Documentation

### 5.1 Code Documentation
- [x] Function docstrings include trace structure
- [x] Missing signal policy documented
- [x] Thresholds documented in docstrings

### 5.2 Doctrine Documentation
- [x] `Budget_PhaseX_Doctrine.md` Section 3.3.3 updated
- [x] Example trace JSON snippet included
- [x] Missing signal policy rationale explained
- [x] Future override scenarios documented (default-false)

---

## 6. Smoke Test Scenarios

### 6.1 Basic Exclusion
**Input:**
- Budget confounded (transient)
- PRNG stable
- Topology low pressure

**Expected:**
- `calibration_exclusion_recommended: true`
- `exclusion_reason: "BUDGET_CONFOUNDED_TRANSIENT"`
- Trace shows all checks passed
- `missing_signal_policy: "DEFAULT_TRUE_MISSING"`

### 6.2 Missing Signals
**Input:**
- Budget confounded (transient)
- PRNG signal missing
- Topology signal missing

**Expected:**
- `calibration_exclusion_recommended: true` (default-true policy)
- Trace shows `source: "DEFAULT_TRUE_MISSING"` for missing signals
- `missing_signal_policy: "DEFAULT_TRUE_MISSING"` in trace

### 6.3 Cross-Signal Blocking
**Input:**
- Budget confounded (transient)
- PRNG volatile
- Topology low pressure

**Expected:**
- `calibration_exclusion_recommended: false`
- Trace shows `prng_not_volatile: false`
- `exclusion_reason: null`

### 6.4 Trace Determinism
**Input:**
- Same inputs provided twice

**Expected:**
- Identical trace structure
- Identical JSON serialization
- Consistent dictionary key ordering

---

## 7. Known Limitations

### 7.1 Policy Hardcoded
- `missing_signal_policy` is currently hardcoded to `"DEFAULT_TRUE_MISSING"`
- Future: Could be made configurable for different phases
- Future: Could support `"DEFAULT_FALSE_MISSING"` for stricter requirements

### 7.2 Signal Source Tracking
- Source tracking is basic (signal name or `DEFAULT_TRUE_MISSING`)
- Future: Could include signal version/timestamp for deeper auditability

---

## 8. Pre-Smoke Test Validation

### 8.1 Code Quality
- [x] No linter errors
- [x] All type hints present
- [x] Docstrings complete

### 8.2 Test Coverage
- [x] 25 exclusion tests passing
- [x] 12 modulation tests passing
- [x] All integration tests passing

### 8.3 Documentation
- [x] Doctrine updated with trace example
- [x] Missing signal policy rationale documented
- [x] Future override scenarios documented

---

## 9. Smoke Test Execution

### 9.1 Unit Tests
```bash
uv run pytest tests/experiments/test_calibration_exclusion.py -v
uv run pytest tests/experiments/test_budget_calibration_modulation.py -v
```

### 9.2 Integration Tests
```bash
uv run pytest tests/experiments/test_budget_council_integration.py -v
```

### 9.3 Manual Verification
1. Create sample calibration windows with budget modulation
2. Call `compute_calibration_exclusion_recommendation()` with various signal combinations
3. Verify trace structure matches documentation
4. Verify JSON serialization is deterministic
5. Verify missing signals are correctly marked

---

## 10. Post-Smoke Test Actions

### 10.1 If All Pass
- [ ] Mark component as production-ready
- [ ] Update changelog
- [ ] Create release notes

### 10.2 If Issues Found
- [ ] Document issues in this checklist
- [ ] Create follow-up tasks
- [ ] Re-run smoke test after fixes

---

**Status:** ✅ ALL CHECKS PASSED  
**Ready for Smoke Test:** YES  
**Next Steps:** Execute smoke test scenarios and validate trace output

