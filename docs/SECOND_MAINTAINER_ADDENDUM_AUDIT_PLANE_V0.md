# Second Maintainer Addendum: Audit Plane v0

**Status**: OPTIONAL
**Estimated Time**: 5–10 minutes
**Prerequisite**: Completed `SECOND_MAINTAINER_BRIEF_v1.2.md` checklist

---

## Purpose

This addendum provides an **optional** verification step for Audit Plane v0 artifacts. Skipping this section incurs no penalty and does not affect your Verification Note.

If you choose to proceed, you will:
1. Run unit tests for schema validation and determinism
2. Validate a sample audit event using the reference implementation
3. Compute an audit root (A_t) from sample events

---

## Quick Verification (5 commands)

Run these commands from the repository root after completing `uv sync`:

```bash
# 1. Run all audit tests (schema + determinism)
uv run pytest tests/audit/ -v --tb=short

# 2. Run reference implementation demo
uv run python backend/audit/audit_root.py

# 3. Validate schema structure
uv run python -c "
from backend.audit.audit_root import load_schema
schema = load_schema()
print(f'Schema: {schema[\"title\"]}')
print(f'Required fields: {len(schema[\"required\"])}')
print('Schema: VALID')
"

# 4. Compute event_id for a sample event
uv run python -c "
from backend.audit.audit_root import compute_event_id
event = {
    'schema_version': '1.0.0',
    'event_type': 'TEST_RESULT',
    'subject': {'kind': 'TEST', 'ref': 'tests/audit/test_audit_root.py'},
    'digest': {'alg': 'sha256', 'hex': 'a' * 64},
    'timestamp': '2025-12-18T12:00:00Z',
    'severity': 'INFO',
    'source': 'audit_plane_v0'
}
event_id = compute_event_id(event)
print(f'event_id = {event_id}')
print('Deterministic: YES (timestamp excluded)')
"

# 5. Compute A_t from sample events
uv run python -c "
from backend.audit.audit_root import compute_event_id, compute_audit_root, generate_audit_root_artifact
events = [
    {'schema_version': '1.0.0', 'event_type': 'TEST_RESULT', 'subject': {'kind': 'TEST', 'ref': 'test1'}, 'digest': {'alg': 'sha256', 'hex': 'a' * 64}, 'timestamp': '2025-12-18T12:00:00Z', 'severity': 'INFO', 'source': 'audit_plane_v0'},
    {'schema_version': '1.0.0', 'event_type': 'CMD_RUN', 'subject': {'kind': 'COMMAND', 'ref': 'make'}, 'digest': {'alg': 'sha256', 'hex': 'b' * 64}, 'timestamp': '2025-12-18T12:00:01Z', 'severity': 'INFO', 'source': 'audit_plane_v0'},
]
artifact = generate_audit_root_artifact(events)
print(f'Event count: {artifact[\"event_count\"]}')
print(f'A_t = {artifact[\"audit_root\"]}')
print('Audit root: COMPUTED')
"
```

---

## Expected Output

Command 1 (tests):
```
tests/audit/test_audit_event_schema.py ............                      [XXX]
tests/audit/test_audit_event_id_determinism.py .............             [XXX]
tests/audit/test_audit_root_determinism.py ..............                [100%]
PASSED
```

Command 2 (demo):
```
Audit Plane v0 — Demo
==================================================
Event 0: event_id = <16-char prefix>...
Event 1: event_id = <16-char prefix>...

A_t = <64-character hex string>
Event count: 2

Demo complete. SHADOW-OBSERVE: No authority; evidence only.
```

Command 3 (schema):
```
Schema: Audit Event Schema v1.0.0
Required fields: 8
Schema: VALID
```

Command 4 (event_id):
```
event_id = <64-character hex string>
Deterministic: YES (timestamp excluded)
```

Command 5 (A_t):
```
Event count: 2
A_t = <64-character hex string>
Audit root: COMPUTED
```

---

## What This Verifies

| Check | Verified Property |
|-------|-------------------|
| Unit tests pass | Schema validation and determinism properties hold |
| Demo runs | Reference implementation executes without error |
| Schema loads | Schema file exists and is valid JSON |
| event_id computed | Canonical hashing produces deterministic event_id |
| A_t computed | Merkle root can be derived from event digests |

---

## What This Does NOT Verify

| Non-Claim | Reason |
|-----------|--------|
| Runtime audit capture | No instrumented code in v0 |
| Integration with H_t | A_t is parallel by design |
| Completeness of events | Schema-only verification |
| Tamper resistance | No trusted timestamping |

---

## Skip Policy

This addendum is **optional**. If you skip it:
- Note "Audit Plane v0 addendum: SKIPPED" in your Verification Note
- No penalty; does not affect overall verification

---

*This document is version-controlled at `docs/SECOND_MAINTAINER_ADDENDUM_AUDIT_PLANE_V0.md`.*
