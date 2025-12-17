# Topology P5 Trust Boundary Examples

**STATUS:** REFERENCE (examples) — Phase X P5 GGFL Alignment Adapter
**VERSION:** 1.0.0
**DATE:** 2025-12-13

> Adapter behavior is enforced by `tests/health/test_p5_topology_reality_adapter.py`; this document is not normative.

## Overview

This document provides reference examples demonstrating the **TRUST BOUNDARY CONTRACT** for the P5 Topology Reality Adapter. The trust boundary ensures that when `schema_ok=False`, no derived scenario fields are fabricated or surfaced.

## Trust Boundary Contract

```
IF schema_ok=False:
  - Status is "warn"
  - Only path + sha256 are surfaced
  - NO scenario, joint_status, validation_passed, or scenario_confidence fields
  - Driver is always [DRIVER_SCHEMA_NOT_OK]

IF schema_ok=True:
  - All derived scenario fields are included
  - Drivers reflect actual validation state
```

---

## Example 1: `schema_ok=False` (Trust Boundary Enforced)

When the P5 topology auditor report has invalid JSON or schema violations, the trust boundary prevents any scenario fabrication.

### Input

```json
{
  "schema_ok": false,
  "path": "corrupted/report.json",
  "sha256": "deadbeef1234",
  "scenario": "FAKE_SCENARIO",
  "validation_passed": true,
  "extraction_source": "P4_SHADOW"
}
```

### GGFL View Output (`topology_p5_for_alignment_view`)

```json
{
  "signal_type": "SIG-TOP5",
  "status": "warn",
  "conflict": false,
  "drivers": ["DRIVER_SCHEMA_NOT_OK"],
  "summary": "Topology P5 report at corrupted/report.json has invalid schema.",
  "path": "corrupted/report.json",
  "sha256": "deadbeef1234",
  "extraction_source": "P4_SHADOW",
  "shadow_mode_invariants": {
    "advisory_only": true,
    "no_enforcement": true,
    "conflict_invariant": false
  }
}
```

**Trust Boundary Verification:**
- `scenario` field: **NOT PRESENT** (trust boundary enforced)
- `joint_status` field: **NOT PRESENT**
- `validation_passed` field: **NOT PRESENT**
- `scenario_confidence` field: **NOT PRESENT**

### Status Signal Output (`build_topology_p5_status_signal`)

```json
{
  "present": true,
  "schema_ok": false,
  "extraction_source": "P4_SHADOW",
  "path": "corrupted/report.json",
  "sha256": "deadbeef1234",
  "advisory_warning": null
}
```

---

## Example 2: `schema_ok=True` (Full Fields Surfaced)

When the P5 topology auditor report has valid schema, all derived scenario fields are included.

### Input

```json
{
  "schema_ok": true,
  "path": "evidence/p5_report.json",
  "sha256": "abc123def456",
  "scenario": "HEALTHY",
  "validation_passed": true,
  "joint_status": "ALIGNED",
  "scenario_confidence": 0.95,
  "extraction_source": "MANIFEST"
}
```

### GGFL View Output (`topology_p5_for_alignment_view`)

```json
{
  "signal_type": "SIG-TOP5",
  "status": "ok",
  "conflict": false,
  "drivers": [],
  "summary": "Topology P5 scenario HEALTHY validated successfully.",
  "path": "evidence/p5_report.json",
  "sha256": "abc123def456",
  "scenario": "HEALTHY",
  "scenario_confidence": 0.95,
  "joint_status": "ALIGNED",
  "validation_passed": true,
  "extraction_source": "MANIFEST",
  "shadow_mode_invariants": {
    "advisory_only": true,
    "no_enforcement": true,
    "conflict_invariant": false
  }
}
```

**All scenario fields present:** `scenario`, `joint_status`, `validation_passed`, `scenario_confidence`

### Status Signal Output (`build_topology_p5_status_signal`)

```json
{
  "present": true,
  "schema_ok": true,
  "extraction_source": "MANIFEST",
  "path": "evidence/p5_report.json",
  "sha256": "abc123def456",
  "scenario": "HEALTHY",
  "scenario_confidence": 0.95,
  "joint_status": "ALIGNED",
  "shadow_mode_invariant_ok": true,
  "validation_passed": true,
  "mode": null,
  "advisory_warning": null
}
```

---

## Reason Code Drivers

The GGFL adapter uses **fixed reason codes only** (no free text). All drivers are:

| Reason Code | Trigger Condition |
|-------------|-------------------|
| `DRIVER_SCHEMA_NOT_OK` | `schema_ok=False` |
| `DRIVER_VALIDATION_NOT_PASSED` | `validation_passed=False` |
| `DRIVER_SCENARIO_MISMATCH` | `scenario="MISMATCH"` |
| `DRIVER_SCENARIO_XCOR_ANOMALY` | `scenario="XCOR_ANOMALY"` |

### Deterministic Ordering Rule

1. **Drivers are sorted alphabetically** for deterministic output
2. **Maximum 3 drivers** per signal (capped)
3. **Each driver appears at most once**

Example with multiple conditions:

```json
// Input: scenario="MISMATCH", validation_passed=False
// Output drivers (sorted):
["DRIVER_SCENARIO_MISMATCH", "DRIVER_VALIDATION_NOT_PASSED"]
```

---

## Extraction Source Provenance

The `extraction_source` field tracks where the P5 report was loaded from:

| Value | Description |
|-------|-------------|
| `MANIFEST` | Loaded from manifest.json governance block |
| `EVIDENCE_JSON` | Loaded from evidence pack JSON file |
| `RUN_DIR_ROOT` | Loaded from run directory root |
| `P4_SHADOW` | Loaded from p4_shadow subdirectory |
| `MISSING` | No report found |

---

## Shadow Mode Invariants

The `shadow_mode_invariants` block is **always present** and **immutable**:

```json
{
  "advisory_only": true,
  "no_enforcement": true,
  "conflict_invariant": false
}
```

These values are constant across all scenarios and schema states.

---

## How to Verify Locally

```bash
pytest tests/health/test_p5_topology_reality_adapter.py -v -k "trust_boundary or extraction_source or shadow_mode_invariants or deterministic"
```

Or run the full test suite (90 tests):

```bash
pytest tests/health/test_p5_topology_reality_adapter.py -v
```

---

## Related Documents

- [Topology_Bundle_PhaseX_Requirements.md](Topology_Bundle_PhaseX_Requirements.md)
- [Phase_X_P5_Implementation_Blueprint.md](Phase_X_P5_Implementation_Blueprint.md)

---

*Last verified: `pytest tests/health/test_p5_topology_reality_adapter.py -v` → 90 passed*

**SAVE TO REPO: YES**
