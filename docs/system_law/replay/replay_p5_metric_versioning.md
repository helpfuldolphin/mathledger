# Replay P5 Metric Versioning

> **Status**: SHADOW MODE SPECIFICATION
> **Version**: 1.3.0
> **Date**: 2025-12-12

## Versioning Policy

**NEVER DELETE METRICS.** Old metrics remain for backward compatibility.
New metrics are added alongside, not replacing, existing ones.

## Metric Evolution

### v1.0.0 (Initial)
- `determinism_rate`: Raw any-mismatch rate (0.0-1.0)
- `determinism_band`: GREEN (≥85%) | YELLOW (70-85%) | RED (<70%)

### v1.1.0 (Robustness)
- `schema_ok`: Boolean for schema validation status
- `skipped_gz_count`: Number of gzip files skipped
- `malformed_line_count`: Number of malformed lines skipped
- `advisory_warnings`: List of advisory warnings (sorted)

### v1.2.0 (True Divergence v1)
- `legacy_metric_label`: "RAW_ANY_MISMATCH" (clarifies old metric semantics)
- `true_divergence_v1`: Vector with breakdown metrics:
  - `outcome_mismatch_rate`: General outcome mismatches
  - `safety_mismatch_rate`: Ω/blocked mismatches only
  - `state_mismatch_rate`: H/rho/tau/beta mismatches only
  - `brier_score_success`: Optional prob(success) calibration

### v1.3.0 (Canonicalization)
- `extraction_source`: Provenance enum (MANIFEST|EVIDENCE_JSON|DIRECT_LOG|MISSING)
- `input_schema_version`: Passthrough schema version (or "UNKNOWN")
- `driver_codes`: Frozen driver reason codes (NO PROSE, cap=3, deterministic order)
  - `DRIVER_SCHEMA_NOT_OK`: Schema validation failed
  - `DRIVER_SAFETY_MISMATCH_PRESENT`: Ω/blocked mismatch detected
  - `DRIVER_STATE_MISMATCH_PRESENT`: H/rho/tau/beta mismatch detected
  - `DRIVER_DETERMINISM_RED_BAND`: Fallback legacy RED band

## Warning Logic (v1.2.0)

Warnings are generated with single-cap policy (no spam):
1. `schema_ok=false` → Single schema warning
2. `safety_mismatch_rate > 0` → Single safety warning
3. `determinism_band=RED` → Fallback legacy warning (only if no above warnings)

## Driver Code Ordering (v1.3.0)

Driver codes are emitted in priority order (cap=3):
1. `DRIVER_SCHEMA_NOT_OK` (highest priority)
2. `DRIVER_SAFETY_MISMATCH_PRESENT`
3. `DRIVER_STATE_MISMATCH_PRESENT`
4. `DRIVER_DETERMINISM_RED_BAND` (lowest priority)

## Frozen Labels (v1.3.0)

The following labels are **FROZEN** and must not be changed:

| Label | Status | Notes |
|-------|--------|-------|
| `legacy_metric_label="RAW_ANY_MISMATCH"` | FROZEN | Non-equivalent to `true_divergence_v1` |
| Driver codes (`DRIVER_*`) | FROZEN | Enum values, NO PROSE |

**WARNING**: `legacy_metric_label` and `true_divergence_v1` are NOT equivalent metrics.
The legacy metric is a raw any-mismatch rate that does not distinguish between
safety, state, and outcome mismatches. Do not use them interchangeably.

## Backward Compatibility

All consumers should:
1. Check for field existence before use
2. Fall back to `determinism_rate` if `true_divergence_v1` missing
3. Treat missing `legacy_metric_label` as "RAW_ANY_MISMATCH"
4. Treat missing `extraction_source` as "DIRECT_LOG"
5. Treat missing `input_schema_version` as "UNKNOWN"
6. Treat missing `driver_codes` as empty list `[]`

**SHADOW MODE: All metrics are observational. No gating logic is implemented.**

## Implementation Trace (v1.3.0)

This section documents where each frozen contract is implemented and tested.

### Authoritative Constants

The single source of truth for frozen constants is:

| Constant | Location | Symbol |
|----------|----------|--------|
| GREEN threshold | `backend/health/replay_governance_adapter.py` | `P5_DETERMINISM_GREEN_THRESHOLD = 0.85` |
| YELLOW threshold | `backend/health/replay_governance_adapter.py` | `P5_DETERMINISM_YELLOW_THRESHOLD = 0.70` |
| Required fields | `scripts/generate_first_light_status.py:extract_p5_replay_signal()` | `P5_REQUIRED_FIELDS = ["trace_hash"]` |

### Invariant → Test Mapping

| Invariant | Function Location | Test Suite |
|-----------|-------------------|------------|
| `.gz` files counted, never parsed | `scripts/generate_first_light_status.py:extract_p5_replay_signal()` | `test_gz_file_skipped_with_warning` |
| `P5_REQUIRED_FIELDS = ["trace_hash"]` | `scripts/generate_first_light_status.py:extract_p5_replay_signal()` | `test_missing_fields_schema_ok_false` |
| `determinism_band` thresholds (GREEN≥0.85, YELLOW≥0.70, RED<0.70) | `scripts/generate_first_light_status.py:extract_p5_replay_signal()` | `test_extract_p5_replay_signal_from_jsonl`, `TestTrueDivergenceV1Vector` |
| `advisory_warnings` sorted, uncapped | `scripts/generate_first_light_status.py:extract_p5_replay_signal()` | `test_replay_p5_top_reasons_sorted_deterministically` |
| Warning cap precedence (schema > safety > RED) | `scripts/generate_first_light_status.py:generate_warnings()` | `test_single_warning_cap`, `test_single_warning_cap_precedence_unchanged` |
| Driver codes frozen enum (cap=3) | `backend/health/replay_governance_adapter.py:compute_driver_codes()` | `test_driver_code_constraint_no_prose`, `test_driver_code_ordering_determinism` |
| Extraction provenance tracking | `backend/health/replay_governance_adapter.py` | `test_extraction_provenance_values` |

**Test file**: `tests/first_light/test_p5_replay_wiring_integration.py` (38 tests)
