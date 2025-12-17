# Chronicle Risk Register — Smoke-Test Readiness Checklist

**Document Version:** 1.0.0  
**Status:** READY FOR SMOKE TEST  
**Date:** 2025-12-11  
**Component:** C6 — Chronicle Risk Register (CAL-EXP Level)

---

## ✅ Implementation Complete

### Phase 1: Per-Experiment Recurrence Snapshot
- [x] `build_cal_exp_recurrence_snapshot()` implemented
- [x] JSON emission to `calibration/chronicle_recurrence_<cal_id>.json`
- [x] Snapshot includes: `schema_version`, `cal_id`, `recurrence_likelihood`, `band`, `invariants_ok`, `timestamp`
- [x] SHADOW MODE compliant (observational only)

### Phase 2: Risk Register Aggregation
- [x] `build_chronicle_risk_register()` implemented
- [x] Aggregates snapshots across calibration experiments
- [x] Includes `high_risk_details` with episode linkage:
  - [x] `cal_id`
  - [x] `recurrence_likelihood`
  - [x] `invariants_ok`
  - [x] `evidence_path_hint`: `calibration/chronicle_recurrence_<cal_id>.json`
- [x] Deterministic ordering (sorted by `cal_id`)
- [x] High-risk classification: `band="HIGH"` AND `invariants_ok=false`
- [x] Band distribution statistics (LOW/MEDIUM/HIGH counts)
- [x] Risk summary (neutral tone)

### Phase 3: Status Extraction
- [x] Manifest-first precedence: `manifest["governance"]["chronicle_risk_register"]` (preferred)
- [x] Evidence.json fallback: `evidence["governance"]["chronicle_risk_register"]`
- [x] Signal enrichment in `signals["chronicle_risk"]`:
  - [x] `total_calibrations`: int
  - [x] `high_risk_count`: int
  - [x] `high_risk_cal_ids_top3`: List[str] (sorted, limited to 3)
  - [x] `has_any_invariants_violated`: bool
- [x] Missing register is not an error (graceful degradation)

### Phase 4: Evidence Pack Integration
- [x] `attach_chronicle_risk_register_to_evidence()` implemented
- [x] Attaches under `evidence["governance"]["chronicle_risk_register"]`
- [x] Advisory only (SHADOW MODE)
- [x] Non-mutating (register dict not modified)

---

## ✅ Test Coverage

### Unit Tests (`tests/ci/test_chronicle_governance_tile_serializes.py`)
- [x] `test_build_cal_exp_recurrence_snapshot`: Basic structure validation
- [x] `test_build_cal_exp_recurrence_snapshot_emits_json`: JSON file emission
- [x] `test_build_chronicle_risk_register`: Aggregation logic
- [x] `test_build_chronicle_risk_register_empty`: Empty snapshot list handling
- [x] `test_build_chronicle_risk_register_classification`: High-risk classification logic
- [x] `test_build_chronicle_risk_register_high_risk_details`: Details list structure
- [x] `test_build_chronicle_risk_register_evidence_path_hint`: Path hint format
- [x] `test_build_chronicle_risk_register_high_risk_details_empty`: Empty details handling
- [x] `test_build_chronicle_risk_register_deterministic_ordering`: Deterministic ordering
- [x] `test_build_chronicle_risk_register_json_safe`: JSON serialization
- [x] `test_attach_chronicle_risk_register_to_evidence`: Evidence attachment
- [x] `test_attach_chronicle_risk_register_non_mutation`: Non-mutation verification

### Integration Tests (`tests/ci/test_chronicle_risk_register_status_extraction.py`)
- [x] `test_chronicle_risk_signal_extracted_from_manifest`: Manifest-first extraction
- [x] `test_chronicle_risk_signal_extracted_from_evidence_fallback`: Evidence.json fallback
- [x] `test_chronicle_risk_signal_top3_limit`: Top 3 limit enforcement
- [x] `test_chronicle_risk_signal_deterministic_ordering`: Deterministic ordering in status
- [x] `test_chronicle_risk_signal_has_any_invariants_violated`: Invariant violation flag
- [x] `test_chronicle_risk_signal_missing_evidence`: Graceful handling when evidence.json missing
- [x] `test_chronicle_risk_signal_missing_register`: Graceful handling when register missing

**Test Results:** ✅ All 19 tests passing (12 unit + 7 integration)

---

## ✅ Documentation

- [x] `docs/system_law/Chronicle_Governance_PhaseX.md`:
  - [x] Section 3: Chronicle Risk Register overview
  - [x] Per-experiment recurrence snapshots structure
  - [x] Risk register structure with `high_risk_details`
  - [x] Reviewer guidance (4-step workflow)
  - [x] Decision framework
  - [x] Integration with other signals
  - [x] Evidence pack integration notes

---

## 🔍 Smoke-Test Validation Checklist

### Pre-Smoke Test
- [ ] All unit tests pass: `uv run pytest tests/ci/test_chronicle_governance_tile_serializes.py::TestChronicleRiskRegister -v`
- [ ] All integration tests pass: `uv run pytest tests/ci/test_chronicle_risk_register_status_extraction.py -v`
- [ ] No linter errors: `uv run ruff check backend/health/chronicle_governance_adapter.py scripts/generate_first_light_status.py`

### Functional Smoke Test
- [ ] **Snapshot Generation:**
  - [ ] Create test snapshots with `build_cal_exp_recurrence_snapshot()`
  - [ ] Verify JSON files emitted to `calibration/chronicle_recurrence_<cal_id>.json`
  - [ ] Verify snapshot structure matches schema

- [ ] **Risk Register Aggregation:**
  - [ ] Aggregate multiple snapshots with `build_chronicle_risk_register()`
  - [ ] Verify `high_risk_details` list is deterministically ordered
  - [ ] Verify `evidence_path_hint` format is correct
  - [ ] Verify high-risk classification logic (HIGH + invariants_ok=false)

- [ ] **Status Extraction:**
  - [ ] Create evidence pack with register in `manifest.json`
  - [ ] Verify status extraction reads from manifest (preferred)
  - [ ] Create evidence pack with register only in `evidence.json`
  - [ ] Verify status extraction falls back to evidence.json
  - [ ] Verify `signals["chronicle_risk"]` includes all required fields:
    - [ ] `total_calibrations`
    - [ ] `high_risk_count`
    - [ ] `high_risk_cal_ids_top3` (limited to 3, sorted)
    - [ ] `has_any_invariants_violated`

- [ ] **Evidence Pack Integration:**
  - [ ] Attach register to evidence pack
  - [ ] Verify register is under `evidence["governance"]["chronicle_risk_register"]`
  - [ ] Verify register dict is not mutated

### Edge Cases
- [ ] Empty snapshot list → Empty register with zero counts
- [ ] No high-risk calibrations → `high_risk_details` is empty list, `has_any_invariants_violated` is False
- [ ] Missing manifest → Falls back to evidence.json gracefully
- [ ] Missing evidence.json → No error, signal not present
- [ ] Missing register in both → No error, signal not present
- [ ] Register in both manifest and evidence.json → Manifest takes precedence

### Determinism Validation
- [ ] Run status generation twice with same inputs → Identical `high_risk_cal_ids_top3` ordering
- [ ] Run risk register aggregation with different snapshot orders → Identical `high_risk_details` ordering
- [ ] Verify JSON serialization is deterministic (sorted keys)

### SHADOW MODE Compliance
- [ ] Verify no control flow depends on chronicle risk register
- [ ] Verify no enforcement language in outputs
- [ ] Verify all outputs are observational/advisory only
- [ ] Verify missing register does not block status generation

---

## 📋 Smoke-Test Execution Commands

```bash
# Run all chronicle risk register tests
uv run pytest tests/ci/test_chronicle_governance_tile_serializes.py::TestChronicleRiskRegister tests/ci/test_chronicle_risk_register_status_extraction.py -v

# Verify status extraction with sample evidence pack
python scripts/generate_first_light_status.py \
    --p3-dir <p3_dir> \
    --p4-dir <p4_dir> \
    --evidence-pack-dir <evidence_pack_dir>

# Check status JSON includes chronicle_risk signal
cat <evidence_pack_dir>/first_light_status.json | jq '.signals.chronicle_risk'
```

---

## 🎯 Success Criteria

### Must Have (Blocking)
- ✅ All tests pass
- ✅ Manifest-first precedence works
- ✅ Evidence.json fallback works
- ✅ `has_any_invariants_violated` field present
- ✅ Top 3 limit enforced
- ✅ Deterministic ordering
- ✅ SHADOW MODE compliant

### Should Have (Non-Blocking)
- ✅ Episode linkage via `evidence_path_hint`
- ✅ Comprehensive test coverage
- ✅ Documentation complete

---

## 📝 Notes

- **Source Precedence:** Manifest is preferred source of truth; evidence.json is fallback for backward compatibility
- **Determinism:** All outputs are deterministically ordered (sorted by `cal_id`)
- **Graceful Degradation:** Missing register or evidence files does not error; signal simply not present
- **SHADOW MODE:** All functions are observational only; no enforcement or gating logic

---

**Status:** ✅ READY FOR SMOKE TEST

