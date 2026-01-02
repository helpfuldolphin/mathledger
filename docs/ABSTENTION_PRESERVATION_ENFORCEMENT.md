# Abstention Preservation Enforcement (v0.1)

**FM Reference**: Section 4.1 - "abstention is a typed outcome... first-class ledger artifact"

**Enforcement Status**: Tier A (Structurally Enforced)

**Implementation Date**: 2026-01-02

---

## Current Enforcement Jurisdiction

**This gate enforces that ABSTAINED outcomes are never silently dropped, converted to null, or treated as missing values. ABSTAINED is a first-class typed outcome that must be explicitly preserved through all data transformations.**

The scope in v0.1:

| What Is Gated | Location | Status |
|---------------|----------|--------|
| Missing `validation_outcome` field | Reasoning artifact validation | **Enforced** |
| Null/None `validation_outcome` | Reasoning artifact validation | **Enforced** |
| Invalid outcome values | Reasoning artifact validation | **Enforced** |
| Outcome aggregation | `validate_outcome_aggregation()` | **Enforced** |

---

## What "Abstention Preservation" Means in v0.1

In v0.1, the abstention preservation invariant is enforced as follows:

1. **ABSTAINED is Typed** - ABSTAINED must be the explicit string "ABSTAINED", never null, undefined, or missing
2. **ABSTAINED is First-Class** - ABSTAINED is treated equally with VERIFIED and REFUTED in all processing
3. **No Silent Dropping** - Downstream code cannot filter out or ignore ABSTAINED outcomes
4. **Aggregation Rules** - When aggregating outcomes, ABSTAINED takes precedence over VERIFIED (but not REFUTED)

---

## Valid Outcomes

| Outcome | Meaning | R_t Inclusion |
|---------|---------|---------------|
| VERIFIED | Validator confirmed the claim | Yes |
| REFUTED | Validator disproved the claim | Yes |
| ABSTAINED | No validator could confirm or refute | Yes |

**Invalid values that trigger violations**:
- `null` / `None`
- Missing field
- `"UNKNOWN"`, `"PENDING"`, `"SKIPPED"`, `"N/A"`
- Empty string `""`
- Boolean values (`true`, `false`)
- Numeric values (`0`, `1`)

---

## What Is Enforced

| Gate Location | What It Protects | Failure Mode |
|---------------|------------------|--------------|
| `governance/abstention_preservation.py:verify_outcome_present()` | Single artifact validation | `AbstentionPreservationViolation` exception |
| `governance/abstention_preservation.py:require_abstention_preservation()` | Batch validation | `AbstentionPreservationViolation` exception |
| `governance/abstention_preservation.py:validate_outcome_aggregation()` | Aggregation logic | `AbstentionPreservationViolation` exception |
| `governance/abstention_preservation.py:verify_not_coerced_to_null()` | Serialization bugs | `AbstentionPreservationViolation` exception |

**Enforcement Logic**:

```
require_abstention_preservation(reasoning_artifacts)
    for each artifact:
        if "validation_outcome" not in artifact:
            raise AbstentionPreservationViolation("MISSING_FIELD")
        if artifact["validation_outcome"] is None:
            raise AbstentionPreservationViolation("NULL_VALUE")
        if artifact["validation_outcome"] not in {"VERIFIED", "REFUTED", "ABSTAINED"}:
            raise AbstentionPreservationViolation("INVALID_VALUE")
```

The gate verifies:
- Every reasoning artifact has a `validation_outcome` field
- The field is not null/None
- The value is one of: VERIFIED, REFUTED, ABSTAINED

On any validation failure, the gate raises `AbstentionPreservationViolation` and **processing is rejected**.

**Error Response** (HTTP 422):
```json
{
  "error_code": "ABSTENTION_PRESERVATION_VIOLATION",
  "message": "Abstention preservation FAILED: validation_outcome is null. Use 'ABSTAINED' explicitly, not null/None. Null outcomes cannot be distinguished from missing data.",
  "artifact_index": 2,
  "claim_id": "sha256:...",
  "violation_type": "NULL_VALUE",
  "details": {
    "received_value": null
  }
}
```

---

## Aggregation Rules

When multiple outcomes must be aggregated (e.g., computing overall verification status):

| Input Outcomes | Aggregate Result | Reason |
|----------------|------------------|--------|
| All VERIFIED | VERIFIED | All claims verified |
| Any REFUTED | REFUTED | Refutation takes precedence |
| Any ABSTAINED (no REFUTED) | ABSTAINED | Cannot claim verified if some abstained |
| Empty list | ABSTAINED | Cannot verify nothing |
| Any null | VIOLATION | Null is not a valid outcome |

**Key Principle**: ABSTAINED propagates upward. If ANY claim is ABSTAINED, the aggregate cannot be VERIFIED (unless there's a REFUTED, which takes precedence).

---

## Audit Trail: Blocked Attempts

When the abstention preservation gate blocks processing, an audit entry is recorded:

```json
{
  "artifact_kind": "ABSTENTION_PRESERVATION_VIOLATION",
  "artifact_index": 2,
  "claim_id": "sha256:...",
  "reason": "Reasoning artifact has null validation_outcome. ABSTAINED must be explicit string 'ABSTAINED', not null.",
  "violation_type": "NULL_VALUE",
  "request_payload_hash": "sha256:...",
  "timestamp_epoch": 1704153600
}
```

**Key properties**:
- This audit entry does NOT contaminate R_t or U_t
- It is stored in a separate `_abstention_violations` log
- It is replay-visible but not authority-bearing
- It enables forensic analysis of blocked attempts

---

## Why This Matters

A potential hostile critique: "Your system could silently drop ABSTAINED outcomes and treat them as verified."

**Response**: This is prevented by the abstention preservation gate:

1. **Missing field is a violation** - Cannot omit the outcome to imply verification
2. **Null is a violation** - Cannot use null as a default that gets ignored
3. **Aggregation preserves ABSTAINED** - Cannot aggregate to VERIFIED if any claim abstained
4. **Serialization bugs are caught** - JSON `null` coercion is detected

The system cannot claim verification when verification was not performed.

---

## Test Coverage

`tests/governance/test_abstention_preservation.py` - 25+ tests

| Test Class | Coverage |
|------------|----------|
| `TestMissingOutcome` | Missing field detection, batch validation |
| `TestNullOutcome` | Null value detection, batch validation |
| `TestInvalidOutcome` | Invalid value detection, error messages |
| `TestValidOutcomes` | VERIFIED, REFUTED, ABSTAINED all pass |
| `TestAggregationPreservation` | ABSTAINED propagation, REFUTED precedence |
| `TestCoercedNullDetection` | Serialization bug detection |
| `TestErrorResponseFormat` | Structured error response |
| `TestNoContamination` | Violations don't enter U_t, R_t, H_t |

---

## Bypass Audit

**Reasoning artifact processing paths enumerated**:

| Operation | Can Drop ABSTAINED? | Gate Applied? |
|-----------|---------------------|---------------|
| Evidence pack generation | **Attempts blocked** | **YES** |
| R_t computation | **Attempts blocked** | **YES** |
| Verification outcome aggregation | **Attempts blocked** | **YES** |
| JSON serialization | **Coercion detected** | **YES** |

**Conclusion**: All paths that process reasoning artifacts are gated by abstention preservation. No bypass path exists.

---

## How to Verify

```bash
uv run pytest tests/governance/test_abstention_preservation.py -v
```

All tests must pass. Any failure indicates gate regression.

---

**Author**: Claude A (v0.1 Abstention Preservation Gate Implementation)
**Audit Date**: 2026-01-02
