# Topology Hotspot Ledger — Smoke-Test Readiness Checklist

**STATUS:** PHASE X — TOPOLOGY HOTSPOT LEDGER STATUS SIGNAL + TOP-DRIVER EXPORT

**Date:** 2025-12-11

**Component:** `backend/health/topology_pressure_adapter.py`

---

## ✅ Implementation Checklist

### Core Functions
- [x] `extract_topology_hotspot_ledger_signal(ledger)` — Extracts compact signal from ledger
- [x] `extract_topology_hotspot_ledger_signal_from_evidence(evidence)` — Extracts signal from evidence pack
- [x] `extract_topology_hotspot_ledger_warnings(ledger)` — Generates warnings for recurring hotspots (count >= 2)
- [x] `attach_topology_hotspot_ledger_signal_to_evidence(evidence)` — Attaches signal to evidence["signals"]

### Signal Structure
- [x] `schema_version: "1.0.0"`
- [x] `num_experiments: int`
- [x] `unique_hotspot_count: int`
- [x] `top_hotspots_top3: List[str]` (names only, top 3)
- [x] `top_hotspot_counts_top3: List[int]` (corresponding counts)

### Warning Logic
- [x] Warnings generated when any hotspot count >= 2 (recurs across experiments)
- [x] Warnings include hotspot name and count
- [x] Warnings sorted deterministically (alphabetically by hotspot name)
- [x] Warnings attached to signal when present

### Safety & Determinism
- [x] Missing ledger handled gracefully (returns None, no signal attached)
- [x] Missing panel handled gracefully (returns None, no signal attached)
- [x] Non-mutating functions (all return new dicts)
- [x] Deterministic output (sorted keys, deterministic ordering)
- [x] JSON serializable (all outputs)

---

## ✅ Test Coverage

### Test Class: `TestTopologyHotspotLedgerSignal`
- [x] **19 tests** covering:
  - Signal extraction correctness (required fields, correct values)
  - Status integration (evidence attachment)
  - Warning threshold (count >= 2 triggers warnings)
  - Missing ledger safe behavior (graceful degradation)
  - Determinism (sorted output, consistent results)
  - Non-mutation (input evidence unchanged)

### Test Results
- [x] **77 total tests passing** (including 19 new signal tests)
- [x] **0 linter errors**
- [x] **All functions follow SHADOW MODE contract**

---

## ✅ Integration Points

### Evidence Pack Structure
- [x] Signal attached under `evidence["signals"]["topology_hotspot_ledger"]`
- [x] Warnings attached under `evidence["signals"]["topology_hotspot_ledger"]["warnings"]` (when present)
- [x] Ledger source: `evidence["governance"]["topology_stress_panel"]["hotspot_ledger"]`

### SHADOW MODE Compliance
- [x] All functions read-only (aside from dict construction)
- [x] Signals purely observational
- [x] No control flow depends on signal contents
- [x] Warnings are advisory only (no gating semantics)

---

## ✅ Smoke Test Scenarios

### Scenario 1: Normal Operation (Ledger Present)
**Input:** Evidence pack with complete topology stress panel and hotspot ledger
**Expected:**
- Signal extracted successfully
- Signal attached to `evidence["signals"]["topology_hotspot_ledger"]`
- If any hotspot count >= 2, warnings included in signal
- All required fields present in signal

**Test:** `test_14_attach_signal_to_evidence_includes_signal`
**Test:** `test_15_attach_signal_to_evidence_includes_warnings_when_recurring`

### Scenario 2: Missing Ledger (Graceful Degradation)
**Input:** Evidence pack without topology stress panel or hotspot ledger
**Expected:**
- No signal extracted (returns None)
- No signal attached to evidence
- No errors raised
- Evidence pack unchanged

**Test:** `test_08_extract_from_evidence_missing_panel`
**Test:** `test_09_extract_from_evidence_missing_ledger`
**Test:** `test_17_attach_signal_to_evidence_missing_ledger_safe`

### Scenario 3: Recurring Hotspots (Warning Generation)
**Input:** Ledger with hotspots having count >= 2
**Expected:**
- Warnings generated for all recurring hotspots
- Warnings include hotspot name and count
- Warnings sorted alphabetically
- Warnings attached to signal

**Test:** `test_10_warnings_extracted_for_recurring_hotspots`
**Test:** `test_11_warnings_include_count`
**Test:** `test_13_warnings_sorted_deterministic`

### Scenario 4: No Recurring Hotspots (No Warnings)
**Input:** Ledger with all hotspots having count < 2
**Expected:**
- No warnings generated
- Signal attached without warnings field
- No errors raised

**Test:** `test_12_warnings_empty_when_no_recurring_hotspots`
**Test:** `test_16_attach_signal_to_evidence_no_warnings_when_no_recurring`

### Scenario 5: Determinism Verification
**Input:** Same ledger processed multiple times
**Expected:**
- Identical output each time
- Sorted keys in hotspot_counts
- Deterministic ordering in top_hotspots_top3
- Deterministic warning ordering

**Test:** `test_19_signal_deterministic_output`
**Test:** `test_13_warnings_sorted_deterministic`

---

## ✅ Code Quality

### Documentation
- [x] All functions have docstrings
- [x] SHADOW MODE contract documented
- [x] Examples provided in docstrings
- [x] Type hints included

### Error Handling
- [x] Missing data handled gracefully (returns None)
- [x] No exceptions raised for missing ledger/panel
- [x] Empty lists handled correctly
- [x] Edge cases covered (fewer than 3 hotspots, etc.)

### Code Style
- [x] Follows repository conventions
- [x] Functions exported in `__all__`
- [x] No linter errors
- [x] Consistent naming conventions

---

## ✅ Smoke Test Execution

### Quick Verification
```bash
# Run all topology pressure integration tests
uv run python -m pytest tests/health/test_topology_pressure_integration.py -v

# Expected: 77 tests passing
```

### Manual Smoke Test
```python
from backend.health.topology_pressure_adapter import (
    build_topology_hotspot_ledger,
    extract_topology_hotspot_ledger_signal,
    attach_topology_hotspot_ledger_signal_to_evidence,
)

# Create test ledger
ledger = {
    "schema_version": "1.0.0",
    "hotspot_counts": {"slice_a": 3, "slice_b": 2, "slice_c": 1},
    "top_hotspots": ["slice_a", "slice_b", "slice_c"],
    "num_experiments": 3,
}

# Extract signal
signal = extract_topology_hotspot_ledger_signal(ledger)
assert signal["num_experiments"] == 3
assert signal["unique_hotspot_count"] == 3
assert len(signal["top_hotspots_top3"]) == 3

# Attach to evidence
evidence = {
    "governance": {
        "topology_stress_panel": {
            "hotspot_ledger": ledger,
        }
    }
}
enriched = attach_topology_hotspot_ledger_signal_to_evidence(evidence)
assert "signals" in enriched
assert "topology_hotspot_ledger" in enriched["signals"]
assert "warnings" in enriched["signals"]["topology_hotspot_ledger"]
assert len(enriched["signals"]["topology_hotspot_ledger"]["warnings"]) == 2
```

---

## ✅ Readiness Status

**STATUS: READY FOR SMOKE TEST**

- ✅ All implementation tasks completed
- ✅ All tests passing (77/77)
- ✅ No linter errors
- ✅ SHADOW MODE contract verified
- ✅ Determinism verified
- ✅ Graceful degradation verified
- ✅ Warning logic verified
- ✅ Integration points verified

**Next Steps:**
1. Execute smoke test scenarios
2. Verify signal appears in First Light status files
3. Verify warnings appear when hotspots recur (count >= 2)
4. Verify graceful degradation when ledger missing

---

**End of Checklist**

