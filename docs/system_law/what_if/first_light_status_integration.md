# What-If Status Signal Integration in first_light_status.json

**Schema Version**: 1.7.0
**Status**: SHADOW MODE (Observational Only)
**Mode**: Always HYPOTHETICAL (never SHADOW)

> **Interpretation guidance**: What-If is informational; uplift discussions are typically deferred until P4 divergence is characterized.

## Overview

The What-If status signal is integrated into `first_light_status.json` extraction with:
- Manifest-first resolution (no evidence fallback needed if bound)
- `extraction_source` field tracking
- GGFL consistency cross-check

## Extraction Resolution Order

1. `manifest.signals.what_if` (preferred - compact status)
2. `manifest.governance.what_if_analysis.status` (fallback)
3. `manifest.governance.what_if_analysis.report` (last resort - extract status)

## extraction_source Values

| Value | Description |
|-------|-------------|
| `MANIFEST_SIGNALS` | Extracted from `manifest.signals.what_if` (preferred path) |
| `MANIFEST_GOVERNANCE` | Extracted from `manifest.governance.what_if_analysis.status` |
| `DERIVED_FROM_REPORT` | Derived from `manifest.governance.what_if_analysis.report.summary` |
| `MISSING` | No What-If data found in manifest |

## GGFL Consistency Cross-Check

The signal includes a `ggfl_consistency` block verifying alignment with GGFL SIG-WIF:

```json
"ggfl_consistency": {
  "status_consistent": true,
  "mode_consistent": true,
  "conflict_consistent": true,
  "all_consistent": true,
  "ggfl_signal_id": "SIG-WIF"
}
```

### Consistency Rules

| Check | Rule |
|-------|------|
| `status_consistent` | `block_rate > 0` → GGFL status = "warn", else "ok" |
| `mode_consistent` | GGFL mode must equal "HYPOTHETICAL" |
| `conflict_consistent` | GGFL conflict must be `false` (What-If cannot conflict) |

## Example Status Snippet

```json
{
  "signals": {
    "what_if": {
      "schema_version": "1.0.0",
      "mode": "HYPOTHETICAL",
      "hypothetical_block_rate": 0.15,
      "blocking_gate_distribution": {
        "G2_INVARIANT": 5,
        "G3_SAFE_REGION": 7,
        "G4_SOFT": 3
      },
      "top_blocking_gate": "G3_SAFE_REGION",
      "first_block_cycle": 12,
      "total_cycles": 100,
      "hypothetical_blocks": 15,
      "extraction_source": "MANIFEST_SIGNALS",
      "ggfl_consistency": {
        "status_consistent": true,
        "mode_consistent": true,
        "conflict_consistent": true,
        "all_consistent": true,
        "ggfl_signal_id": "SIG-WIF"
      }
    }
  },
  "warnings": [
    "What-If (HYPOTHETICAL): 15.0% hypothetical block rate; top_gate=G3_SAFE_REGION"
  ]
}
```

## Golden Bundle Entry

Canonical `signals.what_if` entry for evidence bundle:

```json
"what_if": {
  "schema_version": "1.0.0",
  "mode": "HYPOTHETICAL",
  "hypothetical_block_rate": 0.12,
  "blocking_gate_distribution": {
    "G3_SAFE_REGION": 8,
    "G4_SOFT": 4
  },
  "top_blocking_gate": "G3_SAFE_REGION",
  "first_block_cycle": 15,
  "total_cycles": 100,
  "hypothetical_blocks": 12,
  "extraction_source": "MANIFEST_SIGNALS",
  "ggfl_consistency": {
    "status_consistent": true,
    "mode_consistent": true,
    "conflict_consistent": true,
    "all_consistent": true,
    "ggfl_signal_id": "SIG-WIF"
  }
}
```

### Field Notes

| Field | Meaning |
|-------|---------|
| `extraction_source` | Provenance: `MANIFEST_SIGNALS` (preferred), `MANIFEST_GOVERNANCE`, `DERIVED_FROM_REPORT`, or `MISSING` |
| `ggfl_consistency` | Informational only. Verifies alignment with GGFL SIG-WIF. Does not gate or fail status. |
| `mode` | Always `HYPOTHETICAL`. Never `SHADOW`. |

### Warning Line

```
What-If (HYPOTHETICAL): 12.0% hypothetical block rate; top_gate=G3_SAFE_REGION
```

Emitted only when `hypothetical_block_rate > 0`. Single line, non-gating.

---

## Warning Hygiene

Single-line warning format when `hypothetical_block_rate > 0`:

```
What-If (HYPOTHETICAL): 15.0% hypothetical block rate; top_gate=G3_SAFE_REGION
```

### Warning Components

- **Mode indicator**: Always `(HYPOTHETICAL)`
- **Block rate**: Percentage format with 1 decimal
- **Top gate**: Gate with highest block count (derived from distribution if not provided)

## Smoke Test Checklist

### Pre-flight

- [ ] `manifest.json` exists in evidence pack
- [ ] One of these paths has What-If data:
  - [ ] `manifest.signals.what_if`
  - [ ] `manifest.governance.what_if_analysis.status`
  - [ ] `manifest.governance.what_if_analysis.report`

### Extraction Verification

- [ ] Run: `python scripts/generate_first_light_status.py --evidence-pack-dir <path>`
- [ ] Check `first_light_status.json` contains:
  - [ ] `signals.what_if` block present
  - [ ] `extraction_source` is one of: `MANIFEST_SIGNALS`, `MANIFEST_GOVERNANCE`, `DERIVED_FROM_REPORT`
  - [ ] `mode` equals `"HYPOTHETICAL"`
  - [ ] `ggfl_consistency.all_consistent` is `true`

### Warning Verification

- [ ] If `hypothetical_block_rate > 0`:
  - [ ] Warning present in `warnings` array
  - [ ] Warning is single line (no newlines)
  - [ ] Warning contains "HYPOTHETICAL"
  - [ ] Warning contains block rate percentage
  - [ ] Warning contains `top_gate=<gate_id>` if blocks present

### GGFL Consistency Verification

- [ ] `ggfl_consistency.status_consistent`:
  - [ ] `true` if (block_rate > 0 AND GGFL status = "warn") OR (block_rate = 0 AND GGFL status = "ok")
- [ ] `ggfl_consistency.mode_consistent`: `true` if mode = "HYPOTHETICAL"
- [ ] `ggfl_consistency.conflict_consistent`: `true` if conflict = `false`

### Edge Cases

- [ ] Missing What-If: Signal absent, no error, extraction_source would be `MISSING`
- [ ] Empty distribution: `top_blocking_gate` absent when no blocks
- [ ] Wrong mode: Warning in extraction warnings, `mode_consistent` = `false`

## Test Coverage

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `tests/scripts/test_first_light_what_if_integration.py` | 23 | Extraction, GGFL, warnings |
| `tests/governance/test_what_if_manifest_binding.py` | 25 | Manifest binding |
| `tests/governance/test_what_if_ggfl_adapter.py` | 27 | GGFL adapter |
| `tests/governance/test_what_if_field_mapping.py` | 34 | Field mapping |
| `tests/governance/test_what_if_auto_detection.py` | 22 | Auto-detection |
| `tests/scripts/test_generate_what_if_report.py` | 32 | CLI adapter |

**Total What-If Tests**: 163

## SHADOW MODE Contract

This integration adheres to SHADOW MODE:
- **Observational only**: Does not gate status generation
- **No enforcement**: Missing signal is not an error
- **Advisory warnings**: Single-line format, non-blocking
- **GGFL precedence 11**: Lowest advisory signal
- **Conflict always false**: What-If cannot conflict with other signals

---

## Cross-Check Footer

- [x] `ggfl_consistency` is informational-only; never gates status or CI.

### Run Integration Tests

```bash
pytest tests/scripts/test_first_light_what_if_integration.py -v
```

### Run All What-If Tests

```bash
pytest tests/governance/test_what_if*.py tests/scripts/test_generate_what_if_report.py tests/scripts/test_first_light_what_if_integration.py -v
```
