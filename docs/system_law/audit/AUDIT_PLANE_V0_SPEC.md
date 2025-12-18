# Audit Plane v0 Specification

**Version**: 1.0.0
**Status**: CANONICAL
**Date**: 2025-12-18
**Trust Class**: Advisory (non-blocking)

---

## 1. Purpose

Audit Plane v0 is an **event recording layer** that captures externally meaningful actions during MathLedger operations. It produces evidence artifacts without influencing verification outcomes.

### What Audit Plane v0 Is

- A stream of structured audit events conforming to `audit_event.schema.json`
- A parallel audit root `A_t` (Merkle root over canonicalized event digests)
- An evidence-producing mechanism for post-hoc analysis

### What Audit Plane v0 Is NOT

- NOT a gating mechanism (no blocking, no exit-nonzero behavior)
- NOT an input to `H_t = SHA256(R_t || U_t)` (A_t is kept parallel)
- NOT a verifier or validator (records only, does not evaluate correctness)
- NOT a learning signal (audit outputs do not influence training or curriculum)

---

## 2. Threat Motivation

Agents operating within MathLedger may attempt to probe verifiers, access unexpected files, or execute commands outside expected boundaries. Audit Plane v0 records these behaviors as evidence for human review.

**Recorded behaviors include**:
- File system access patterns (FS_TOUCH events)
- Command executions and exit codes (CMD_RUN events)
- Hash emissions and verification attempts (HASH_EMITTED events)
- Test results and policy evaluations (TEST_RESULT, POLICY_CHECK events)
- Anomalous patterns flagged by monitoring agents (FUZZ_FINDING events)

Audit Plane v0 **does not block** any of these behaviors. It records and emits evidence. **It does not detect, prevent, or respond to these behaviors.**

---

## 3. Trust Class

| Attribute | Value |
|-----------|-------|
| Trust Class | **Advisory** |
| Blocking Behavior | None |
| Exit Codes | Always 0 in normal operation |
| Influence on H_t | None (A_t is parallel) |
| Influence on Verifiers | None |
| Influence on Learning | None |

Audit outputs are informational artifacts. They may be consumed by external analysis tools but have no authority over MathLedger operations.

---

## 4. Event Model

### 4.1 Schema Reference

All audit events conform to `schemas/audit/audit_event.schema.json` (v1.0.0).

### 4.2 Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | const "1.0.0" | Schema version identifier |
| `event_id` | string (64 hex) | SHA-256 of canonical event bytes (excluding event_id) |
| `event_type` | enum | Event category (FS_TOUCH, CMD_RUN, etc.) |
| `subject` | object | Entity being audited (kind + ref) |
| `digest` | object | SHA-256 of subject content (alg + hex) |
| `timestamp` | ISO 8601 | Event occurrence time (non-attestation-critical) |
| `severity` | enum | INFO or WARN |
| `source` | string | Emitting component identifier |

### 4.3 Canonicalization Rule

For deterministic event digests:

1. Serialize event as JSON with keys sorted alphabetically
2. Use compact representation (no whitespace)
3. Exclude `event_id` and `timestamp` from digest computation
4. Compute SHA-256 of the resulting bytes
5. Encode as 64 lowercase hex characters

### 4.4 Determinism Policy

| Element | Deterministic? |
|---------|----------------|
| `event_id` | Yes (derived from canonical content) |
| `event_type` | Yes |
| `subject` | Yes |
| `digest` | Yes |
| `timestamp` | **No** (metadata only, excluded from A_t) |
| `meta.note` | **No** (excluded from canonicalization) |

---

## 5. Audit Root (A_t)

The audit root `A_t` is a Merkle root computed over audit events for a given run or epoch.

### 5.1 Computation

```
1. Collect all audit events for the epoch
2. Extract event_id from each event (already a SHA-256 digest)
3. Sort event_ids lexicographically (ascending)
4. Compute Merkle root over sorted digests using SHA-256
5. A_t = resulting 64-character hex string
```

### 5.2 Properties

| Property | Value |
|----------|-------|
| Algorithm | SHA-256 Merkle tree |
| Leaf nodes | Sorted event_id values |
| Empty tree | A_t = SHA-256("") = `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

### 5.3 Relationship to H_t

**A_t is kept parallel to H_t.** The dual-root attestation formula remains:

```
H_t = SHA256(R_t || U_t)
```

A_t is recorded alongside H_t but is NOT braided into H_t. This separation ensures:
- Existing attestation semantics are unchanged
- Audit plane can evolve independently
- No retroactive claims about prior attestations

---

## 6. Storage

Audit artifacts are stored at:

```
results/<run_id>/audit/
├── audit_events.jsonl       # One JSON event per line
├── audit_root.json          # { "A_t": "<hex>", "event_count": N }
└── audit_manifest.json      # Optional: index of events by type
```

**Note**: This is the proposed path structure. Storage is additive; no existing paths are modified.

---

## 7. Non-Claims

**Audit Plane v0 produces evidence only; it has no authority, no gating capability, and does not alter truth claims, verification outcomes, or learning signals.**

Audit Plane v0 explicitly makes **no claims** about:

| Non-Claim | Rationale |
|-----------|-----------|
| Completeness | Not all actions may be instrumented |
| Tamper-resistance | A_t can be recomputed from events; no trusted timestamping |
| Real-time detection | Events are recorded, not blocked |
| Correctness validation | Audit records actions, not outcomes |
| Learning influence | Audit outputs are not training signals |

---

## 7.1 Non-Claims and Explicit Exclusions (Audit Plane v0)

The following exclusions are **unconditional and non-negotiable** for Audit Plane v0:

- **No authority**: Audit Plane v0 has no decision-making power over any MathLedger operation.
- **No blocking**: Audit Plane v0 does not block, halt, or gate any execution path.
- **No learning influence**: Audit outputs are not training signals, curriculum inputs, or feedback loops.
- **No H_t participation**: A_t is not braided into `H_t = SHA256(R_t || U_t)`. The attestation formula is unchanged.
- **No truth alteration**: Audit Plane v0 does not modify, validate, or influence truth claims.
- **No contract changes**: This version introduces no changes to existing governance contracts.

Audit Plane v0 produces evidence only.

---

## 8. Future Work (Deferred)

The following are explicitly deferred to future versions:

| Item | Status |
|------|--------|
| SHADOW-GATED audit gates | Deferred |
| Braiding A_t into H_t | Deferred |
| Separate audit epoch root | Deferred |
| Trusted timestamping | Deferred |
| Real-time alerting | Deferred |

---

## 9. References

- `schemas/audit/audit_event.schema.json` — Authoritative schema
- `docs/system_law/SHADOW_MODE_CONTRACT.md` — SHADOW MODE semantics
- `docs/DETERMINISM_CLAIMS.md` — Determinism scope

---

**Version**: 1.0.0
**Owner**: STRATCOM Audit Engineering
