# Harmonic Curriculum Status Signal - Smoke Test Checklist

**STATUS:** PHASE X — CAL-EXP CURRICULUM HARMONIC GRID

**Purpose:** Verification checklist for harmonic curriculum panel status signal extraction and GGFL adapter.

---

## Smoke Test Commands

```bash
# Run all harmonic curriculum tests
uv run pytest tests/health/test_curriculum_harmonic_grid.py tests/health/test_harmonic_curriculum_status.py -q

# Expected: All tests pass (57+ tests)
```

---

## Example Status Signal Output

### With Panel from Manifest

```json
{
  "harmonic_curriculum_panel": {
    "schema_version": "1.0.0",
    "mode": "SHADOW",
    "extraction_source": "MANIFEST",
    "band_counts": {
      "COHERENT": 2,
      "PARTIAL": 1,
      "MISMATCHED": 0
    },
    "top_driver_concepts_top5": ["slice_a", "slice_b", "slice_c"],
    "delta": {
      "top_driver_overlap_top5": ["slice_a", "slice_b"],
      "frequency_shift_top3": [
        {"concept": "slice_a", "delta": 1},
        {"concept": "slice_b", "delta": 0}
      ]
    }
  }
}
```

### With Panel from Evidence JSON (Fallback)

```json
{
  "harmonic_curriculum_panel": {
    "schema_version": "1.0.0",
    "mode": "SHADOW",
    "extraction_source": "EVIDENCE_JSON",
    "band_counts": {
      "COHERENT": 1,
      "PARTIAL": 1,
      "MISMATCHED": 1
    },
    "top_driver_concepts_top5": ["slice_b", "slice_a", "slice_d"]
  }
}
```

### Missing Panel

```json
{
  "harmonic_curriculum_panel": {
    "schema_version": "1.0.0",
    "mode": "SHADOW",
    "extraction_source": "MISSING",
    "band_counts": {
      "COHERENT": 0,
      "PARTIAL": 0,
      "MISMATCHED": 0
    },
    "top_driver_concepts_top5": []
  }
}
```

**Key Properties:**
- `schema_version`: Always present (from panel or "1.0.0" default)
- `mode`: Always "SHADOW" (constant)
- `extraction_source`: Always present ("MANIFEST" | "EVIDENCE_JSON" | "MISSING")
- `delta`: Only present when panel contains delta data
- All fields are JSON-serializable and deterministic

---

## Example GGFL Adapter Output

### With MISMATCHED Present

```json
{
  "signal_type": "SIG-HAR",
  "status": "warn",
  "conflict": false,
  "drivers": [
    "DRIVER_MISMATCHED_PRESENT",
    "DRIVER_TOP_CONCEPTS_PRESENT"
  ],
  "summary": "Harmonic curriculum alignment: 1 of 3 experiment(s) show MISMATCHED harmonic band",
  "shadow_mode_invariants": {
    "observational_only": true,
    "no_control_flow": true,
    "advisory_weight": "LOW"
  },
  "weight_hint": "LOW"
}
```

### With Delta Shift Triggered

```json
{
  "signal_type": "SIG-HAR",
  "status": "ok",
  "conflict": false,
  "drivers": [
    "DRIVER_DELTA_SHIFT_PRESENT",
    "DRIVER_TOP_CONCEPTS_PRESENT"
  ],
  "summary": "Harmonic curriculum alignment: 2 of 3 experiment(s) show COHERENT harmonic band",
  "shadow_mode_invariants": {
    "observational_only": true,
    "no_control_flow": true,
    "advisory_weight": "LOW"
  },
  "weight_hint": "LOW"
}
```

### Clean (No Drivers)

```json
{
  "signal_type": "SIG-HAR",
  "status": "ok",
  "conflict": false,
  "drivers": [],
  "summary": "Harmonic curriculum alignment: 3 of 3 experiment(s) show COHERENT harmonic band",
  "shadow_mode_invariants": {
    "observational_only": true,
    "no_control_flow": true,
    "advisory_weight": "LOW"
  },
  "weight_hint": "LOW"
}
```

**Key Properties:**
- `signal_type`: Always "SIG-HAR" (constant)
- `status`: "ok" | "warn" (warn if any MISMATCHED count > 0)
- `conflict`: Always `false` (constant)
- `drivers`: List of reason codes (max 3), deterministic ordering:
  1. `DRIVER_MISMATCHED_PRESENT` (if mismatched_count > 0)
  2. `DRIVER_DELTA_SHIFT_PRESENT` (if delta present and threshold triggered)
  3. `DRIVER_TOP_CONCEPTS_PRESENT` (if top concepts present)
- `shadow_mode_invariants`: Always present with SHADOW MODE contract indicators
- `weight_hint`: Always "LOW" (constant)
- All fields are JSON-serializable and deterministic

---

## Warning Hygiene Verification

**Single Warning Cap:** At most one warning is generated per panel:

1. **MISMATCHED Warning** (takes priority):
   ```
   "Harmonic curriculum panel: 1 experiment(s) with MISMATCHED harmonic band (out of 3 total)"
   ```

2. **Delta Shift Warning** (only if no MISMATCHED):
   ```
   "Harmonic curriculum delta: concept 'slice_a' shows frequency shift of 3 (threshold: 2)"
   ```

**Verification:**
- If both conditions are met, only MISMATCHED warning is generated
- Warning is a single string, not a list
- Warning is advisory only (does not gate status generation)

---

## Schema Stability

**Resistant to String Drift:**
- All driver codes use constant strings: `DRIVER_*_PRESENT`
- No concept names or dynamic strings in drivers
- `extraction_source` uses enum values: `MANIFEST` | `EVIDENCE_JSON` | `MISSING`
- `mode` and `weight_hint` are constants
- `schema_version` is always present and passthrough from panel

**Deterministic Ordering:**
- Drivers follow fixed order: mismatched → delta shift → top concepts
- Same inputs always produce identical outputs
- JSON serialization is deterministic

---

## Related Documentation

- `backend/health/harmonic_alignment_p3p4_integration.py`: Core implementation
- `scripts/generate_first_light_status.py`: Status generation integration
- `tests/health/test_harmonic_curriculum_status.py`: Test suite
- `docs/system_law/Harmonic_PhaseX_Binding.md`: System law documentation

---

**Last Updated:** Phase X — Harmonic Curriculum Status Signal (Provenance + Driver Codes)

