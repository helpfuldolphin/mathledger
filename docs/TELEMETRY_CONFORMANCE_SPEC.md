# Telemetry Conformance Specification

**Document Version:** 1.0
**Author:** CLAUDE H (Telemetry Canonist)
**Date:** 2025-12-06
**Phase:** II — Operation Asymmetry
**Status:** PHASE II — NOT IMPLEMENTED
**Companion Document:** `TELEMETRY_CANONICAL_FORM.md`

```
+===============================================================================+
|                                                                               |
|              NOT IMPLEMENTED — FUTURE PHASE II                                |
|                                                                               |
|  This document defines conformance levels, quarantine protocols, and          |
|  test plans for U2 telemetry validation. It governs how telemetry             |
|  records are classified, how non-conforming records are handled,              |
|  and how conformance is verified.                                             |
|                                                                               |
|              PHASE II — NOT RUN IN PHASE I                                    |
|                                                                               |
+===============================================================================+
```

---

## 1. Purpose & Scope

### 1.1 Purpose

This specification establishes:

1. **Conformance Levels** — A tiered classification system for telemetry quality
2. **Consumer Behavior** — What consumers may do with each conformance level
3. **Quarantine Protocol** — Handling of non-conforming records
4. **Test Plan** — Verification procedures for conformance

### 1.2 Scope

This specification applies to:

- All U2 telemetry record types (`cycle_metric`, `experiment_summary`, `uplift_result`)
- All telemetry emitters (U2 runner, summary generator, uplift calculator)
- All telemetry consumers (governance gates, dashboards, archivers)
- All telemetry storage (JSONL files, quarantine directories)

### 1.3 Normative References

| Document | Relationship |
|----------|--------------|
| `TELEMETRY_CANONICAL_FORM.md` | Defines canonical form, serialization rules, prohibited fields |
| `first_organism_telemetry_plan_v2.md` | Defines record schemas, governance mappings |
| `PREREG_UPLIFT_U2.yaml` | Defines governance gate requirements |

---

## 2. Conformance Levels

### 2.1 Level Definitions

```
+-----------------------------------------------------------------------+
|                     CONFORMANCE LEVEL HIERARCHY                       |
+-----------------------------------------------------------------------+
|                                                                       |
|   L2: CANONICAL                                                       |
|   +---------------------------------------------------------------+   |
|   |  - Schema valid (L1 satisfied)                                |   |
|   |  - Field ordering: lexicographic                              |   |
|   |  - Serialization: canonical (no whitespace, proper escapes)   |   |
|   |  - Hashes: valid chain, correct format                        |   |
|   |  - Timestamps: monotonic, correct precision                   |   |
|   |  - Cross-field: consistent (e.g., uplift_delta = p_rfl-p_base)|   |
|   +---------------------------------------------------------------+   |
|                              ▲                                        |
|                              | Promotes to                            |
|                              |                                        |
|   L1: SCHEMA-VALID                                                    |
|   +---------------------------------------------------------------+   |
|   |  - All required fields present                                |   |
|   |  - No prohibited fields                                       |   |
|   |  - Field types correct (string, number, boolean, array)       |   |
|   |  - Enum values valid                                          |   |
|   |  - Value ranges respected                                     |   |
|   +---------------------------------------------------------------+   |
|                              ▲                                        |
|                              | Promotes to                            |
|                              |                                        |
|   L0: RAW                                                             |
|   +---------------------------------------------------------------+   |
|   |  - Parseable as JSON                                          |   |
|   |  - No schema guarantees                                       |   |
|   |  - May contain any fields                                     |   |
|   |  - May have any structure                                     |   |
|   +---------------------------------------------------------------+   |
|                                                                       |
+-----------------------------------------------------------------------+
```

### 2.2 L0: Raw Telemetry

**Definition:** A record that is valid JSON but has not been validated against any schema.

**Characteristics:**

| Property | Requirement |
|----------|-------------|
| JSON validity | MUST parse as valid JSON |
| Schema adherence | NOT guaranteed |
| Field presence | NOT guaranteed |
| Field types | NOT guaranteed |
| Field ordering | NOT guaranteed |
| Serialization | NOT guaranteed |

**Source:** Any external system, legacy emitters, or records prior to validation.

**Lifecycle:** L0 records MUST be validated before use. They cannot remain L0 indefinitely in the active pipeline.

### 2.3 L1: Schema-Valid Telemetry

**Definition:** A record that conforms to the JSON Schema for its record type.

**Characteristics:**

| Property | Requirement |
|----------|-------------|
| JSON validity | MUST be valid JSON |
| Required fields | ALL required fields MUST be present |
| Prohibited fields | NO prohibited fields may be present |
| Field types | ALL fields MUST have correct types |
| Enum values | ALL enum fields MUST have valid values |
| Value ranges | ALL numeric fields MUST be in valid ranges |
| Field ordering | NOT required (any order acceptable) |
| Serialization format | NOT required (whitespace allowed) |
| Hash chain | NOT verified |
| Cross-field consistency | NOT verified |

**Promotion Criteria:** To promote L0 → L1:

1. Parse record as JSON
2. Identify record type from field signature
3. Validate against corresponding JSON Schema (Appendix A of `TELEMETRY_CANONICAL_FORM.md`)
4. If validation passes, mark as L1

### 2.4 L2: Canonical Telemetry

**Definition:** A record that fully conforms to the Canonical Form specification.

**Characteristics:**

| Property | Requirement |
|----------|-------------|
| Schema validity | MUST satisfy all L1 requirements |
| Field ordering | MUST be lexicographic (alphabetical by key) |
| JSON serialization | MUST be compact (no whitespace between tokens) |
| Numeric format | MUST use decimal notation (no scientific) |
| Timestamp format | MUST be ISO 8601 with microseconds and Z suffix |
| Hash format | MUST be 64 lowercase hex characters |
| Hash chain | MUST be valid (each ht derives from previous) |
| Timestamp sequence | MUST be monotonically increasing within run |
| Cycle sequence | MUST be sequential starting from 0 |
| Cross-field consistency | ALL derived fields MUST be correct |

**Promotion Criteria:** To promote L1 → L2:

1. Verify field ordering matches canonical order
2. Verify serialization format (no extraneous whitespace)
3. Verify numeric format (no scientific notation)
4. Verify timestamp format and sequence
5. Verify hash format and chain validity
6. Verify cross-field consistency
7. If all pass, mark as L2

---

## 3. Consumer Behavior by Level

### 3.1 Allowed Operations Matrix

| Consumer | L0 | L1 | L2 |
|----------|-----|-----|-----|
| **Validator** | MAY validate | MAY re-validate | MAY verify canonical hash |
| **Governance Gate** | MUST NOT consume | MUST NOT consume | MAY consume |
| **Dashboard (real-time)** | MUST NOT display | MAY display with warning | MAY display |
| **Dashboard (historical)** | MUST NOT display | MUST NOT display | MAY display |
| **Archiver** | MUST quarantine | MAY archive with L1 tag | MAY archive |
| **Summary Generator** | MUST NOT aggregate | MUST NOT aggregate | MAY aggregate |
| **Uplift Calculator** | MUST NOT compute | MUST NOT compute | MAY compute |
| **Hash Verifier** | MUST NOT verify | MUST NOT verify | MAY verify |
| **Audit Trail** | MAY log receipt | MAY log validation | MAY log as authoritative |

### 3.2 Consumer Requirements

#### 3.2.1 Governance Gate Consumers

Governance gates (G1-G6 as defined in `first_organism_telemetry_plan_v2.md`) are **L2-only consumers**.

**Requirements:**

- MUST reject any record not at L2
- MUST verify L2 conformance before gate evaluation
- MUST log rejection reason for non-L2 records
- MUST NOT infer or interpolate missing data

#### 3.2.2 Dashboard Consumers

Real-time dashboards display current telemetry state.

**L1 Behavior:**
- MAY display L1 records
- MUST display visual indicator that data is "unverified"
- MUST NOT persist L1 records to historical storage
- MUST NOT use L1 data for trend calculation

**L2 Behavior:**
- MAY display without warning
- MAY persist to historical storage
- MAY use for trend calculation

#### 3.2.3 Archiver Consumers

Archivers persist telemetry for long-term storage.

**L0 Behavior:**
- MUST quarantine (see Section 4)
- MUST NOT archive to primary storage

**L1 Behavior:**
- MAY archive with conformance level tag
- MUST store in separate L1 partition
- MUST NOT mix with L2 records

**L2 Behavior:**
- MAY archive to primary storage
- SHOULD verify canonical hash before archival
- MUST include conformance attestation

#### 3.2.4 Aggregation Consumers

Summary generators and uplift calculators are **L2-only consumers**.

**Requirements:**

- MUST reject non-L2 input
- MUST verify all input records are from same run
- MUST verify cycle sequence completeness before aggregation
- MUST emit L2-conformant output

### 3.3 Level Downgrade Conditions

A record's conformance level may be downgraded:

| Condition | Downgrade |
|-----------|-----------|
| Hash chain break detected | L2 → Quarantine |
| Duplicate cycle number | L2 → Quarantine |
| Timestamp regression | L2 → L1 (flag for review) |
| Unknown field discovered | L2 → L1 |
| Cross-field inconsistency | L2 → Quarantine |
| Storage corruption detected | Any → Quarantine |

---

## 4. Quarantine Protocol

### 4.1 Quarantine Triggers

A record MUST be quarantined when:

| Trigger | Severity | Description |
|---------|----------|-------------|
| JSON parse failure | CRITICAL | Cannot parse as JSON |
| Missing required field | CRITICAL | L1 validation failure |
| Prohibited field present | ERROR | Contains banned field name |
| Type mismatch | ERROR | Field has wrong JSON type |
| Enum violation | ERROR | Invalid enum value |
| Range violation | ERROR | Numeric value out of bounds |
| Hash format invalid | CRITICAL | Hash not 64 lowercase hex |
| Hash chain broken | CRITICAL | ht does not chain from previous |
| Cycle sequence gap | CRITICAL | Missing or duplicate cycle numbers |
| Cross-field mismatch | CRITICAL | Derived fields inconsistent |
| Timestamp in future | WARNING → ERROR | Beyond tolerance threshold |

### 4.2 Quarantine File Layout

```
results/
└── quarantine/
    ├── index.jsonl                    # Master index of all quarantined records
    ├── by_run/
    │   ├── U2-aaa-bbb/
    │   │   ├── manifest.json          # Run-level quarantine summary
    │   │   ├── records/
    │   │   │   ├── 00042.json         # Individual quarantined record (cycle 42)
    │   │   │   ├── 00043.json
    │   │   │   └── ...
    │   │   └── violations.jsonl       # All violations for this run
    │   └── U2-ccc-ddd/
    │       └── ...
    ├── by_date/
    │   ├── 2025-12-06/
    │   │   └── <symlinks to by_run records>
    │   └── ...
    └── by_severity/
        ├── critical/
        │   └── <symlinks to by_run records>
        ├── error/
        │   └── <symlinks to by_run records>
        └── warning/
            └── <symlinks to by_run records>
```

### 4.3 Required Metadata per Quarantined Record

Each quarantined record MUST be wrapped with metadata:

```
+-----------------------------------------------------------------------+
|                    QUARANTINE RECORD ENVELOPE                         |
+-----------------------------------------------------------------------+

{
  "quarantine": {
    "id": "<uuid>",                          # Unique quarantine record ID
    "quarantined_at": "<ISO 8601>",          # When quarantine occurred
    "quarantine_reason": "<summary>",        # Human-readable reason
    "severity": "critical|error|warning",    # Severity classification
    "source_file": "<path>",                 # Original file path
    "source_line": <integer>,                # Line number in source file
    "source_byte_offset": <integer>,         # Byte offset in source file
    "detector_version": "<semver>",          # Drift detector version
    "canonical_form_version": "1.0"          # TELEMETRY_CANONICAL_FORM.md version
  },
  "violations": [
    {
      "check_name": "<check identifier>",    # e.g., "check_required_fields"
      "check_category": "<category>",        # "schema", "format", "semantic", "temporal"
      "severity": "critical|error|warning",
      "field": "<field name or null>",       # Affected field, if applicable
      "message": "<detailed message>",
      "expected": "<expected value/format>",
      "actual": "<actual value/format>"
    }
  ],
  "record": {
    "raw_json": "<original JSON string>",    # Exact bytes received
    "parsed": { ... },                       # Parsed JSON (if parseable)
    "record_type": "<type or unknown>",      # Detected record type
    "run_id": "<run_id or unknown>",         # Extracted run_id if available
    "cycle": <integer or null>               # Extracted cycle if available
  },
  "context": {
    "preceding_record_hash": "<hash>",       # Hash of previous valid record
    "following_record_hash": "<hash>",       # Hash of next valid record (if known)
    "run_record_count": <integer>,           # Total records in run at quarantine time
    "run_quarantine_count": <integer>        # Quarantined records in run
  },
  "disposition": {
    "status": "pending|diagnosed|purged|restored",
    "diagnosed_at": "<ISO 8601 or null>",
    "diagnosed_by": "<identifier or null>",
    "diagnosis": "<explanation or null>",
    "resolved_at": "<ISO 8601 or null>",
    "resolution": "purged|restored|null"
  }
}
```

### 4.4 Quarantine Index Schema

The master index (`quarantine/index.jsonl`) contains one line per quarantine event:

```
{
  "id": "<uuid>",
  "quarantined_at": "<ISO 8601>",
  "run_id": "<run_id or unknown>",
  "cycle": <integer or null>,
  "severity": "critical|error|warning",
  "primary_violation": "<check_name>",
  "record_path": "by_run/<run_id>/records/<cycle>.json",
  "status": "pending|diagnosed|purged|restored"
}
```

### 4.5 Run-Level Manifest Schema

Each run's manifest (`by_run/<run_id>/manifest.json`) summarizes quarantine state:

```
{
  "run_id": "<run_id>",
  "first_quarantine_at": "<ISO 8601>",
  "last_quarantine_at": "<ISO 8601>",
  "total_quarantined": <integer>,
  "by_severity": {
    "critical": <integer>,
    "error": <integer>,
    "warning": <integer>
  },
  "by_check": {
    "<check_name>": <count>,
    ...
  },
  "affected_cycles": [<list of cycle numbers>],
  "run_invalidated": <boolean>,
  "invalidation_reason": "<reason or null>"
}
```

### 4.6 Diagnosis Procedure

To diagnose a quarantined record:

```
+-----------------------------------------------------------------------+
|                      DIAGNOSIS WORKFLOW                               |
+-----------------------------------------------------------------------+

1. IDENTIFY
   - Locate record in quarantine/by_run/<run_id>/records/
   - Read violations from envelope
   - Note severity and affected fields

2. CLASSIFY
   - Emitter bug: Systematic issue in telemetry emitter
   - Data corruption: Storage or transmission error
   - Schema evolution: Record from different schema version
   - Environmental: Clock skew, resource exhaustion
   - Unknown: Requires deeper investigation

3. ROOT CAUSE
   - For emitter bugs: Identify code path, check logs
   - For corruption: Check storage integrity, network logs
   - For schema issues: Compare against schema version history
   - For environmental: Check system state at quarantine time

4. DOCUMENT
   - Update disposition.diagnosed_at
   - Update disposition.diagnosed_by
   - Update disposition.diagnosis with findings

5. DECIDE
   - If fixable: Correct and restore (Section 4.7)
   - If unfixable: Purge (Section 4.8)
   - If unclear: Escalate to run invalidation review
```

### 4.7 Inspection Interface

Quarantined records SHOULD be inspectable via:

**Command-line interface (specification):**

```
# List all quarantined records
quarantine list [--run <run_id>] [--severity <level>] [--status <status>]

# Show details of a quarantined record
quarantine show <quarantine_id>

# Show all violations for a run
quarantine violations <run_id>

# Show quarantine statistics
quarantine stats [--run <run_id>] [--date <date>]
```

**Output format for `quarantine show`:**

```
Quarantine Record: <uuid>
========================
Quarantined At: 2025-12-06T10:30:00Z
Severity: CRITICAL
Run ID: U2-aaa-bbb
Cycle: 42

Violations:
  1. [CRITICAL] check_hash_chain
     Field: ht
     Expected: abc123... (derived from previous)
     Actual: xyz789...
     Message: Hash chain broken at cycle 42

  2. [ERROR] check_cross_field_consistency
     Field: uplift_delta
     Expected: 0.16 (p_rfl - p_base)
     Actual: 0.15
     Message: uplift_delta does not equal p_rfl - p_base

Original Record:
  {"cycle":42,"ht":"xyz789...","metric_type":"goal_hit",...}

Context:
  Preceding valid record: cycle 41, hash abc123...
  Run total records: 500
  Run quarantined: 3

Disposition:
  Status: pending
  Diagnosed: no
```

### 4.8 Purge Protocol

Quarantined records may be purged (permanently deleted) when:

| Condition | Purge Allowed |
|-----------|---------------|
| Diagnosed as unrecoverable | YES |
| Run fully invalidated | YES |
| Age > 90 days AND status = diagnosed | YES |
| Age > 180 days (any status) | YES, with audit log |
| Status = pending AND age < 30 days | NO |
| Status = pending AND age >= 30 days | YES, after final review |

**Purge procedure:**

1. Verify purge conditions are met
2. Log purge intent to audit trail
3. Update index.jsonl: set status = "purged"
4. Archive envelope to `archives/quarantine/<year>/<month>/`
5. Remove from active quarantine directories
6. Update run manifest quarantine counts

**Purge audit record:**

```
{
  "action": "purge",
  "quarantine_id": "<uuid>",
  "purged_at": "<ISO 8601>",
  "purged_by": "<identifier>",
  "reason": "<justification>",
  "archive_path": "archives/quarantine/2025/12/<uuid>.json"
}
```

### 4.9 Restoration Protocol

A quarantined record may be restored to active telemetry if:

1. Root cause is identified and resolved
2. Record can be corrected to L2 conformance
3. Correction does not break hash chain
4. Run has not been invalidated

**Restoration procedure:**

1. Create corrected record
2. Validate corrected record achieves L2
3. Verify corrected record maintains hash chain
4. Insert corrected record into run's JSONL
5. Update quarantine envelope: status = "restored"
6. Log restoration to audit trail

**Restoration constraints:**

- MUST NOT restore if hash chain cannot be maintained
- MUST NOT restore if run is invalidated
- MUST NOT alter cycle number during restoration
- MUST preserve original quarantine record in archive

---

## 5. Conformance Test Plan

### 5.1 Test Categories

```
+-----------------------------------------------------------------------+
|                      CONFORMANCE TEST TAXONOMY                        |
+-----------------------------------------------------------------------+

CONFORMANCE TESTS
├── SCHEMA ADHERENCE TESTS (Section 5.2)
│   ├── Required Field Tests
│   ├── Prohibited Field Tests
│   ├── Type Validation Tests
│   ├── Enum Validation Tests
│   └── Range Validation Tests
│
├── CANONICALIZATION TESTS (Section 5.3)
│   ├── Field Ordering Tests
│   ├── Serialization Format Tests
│   ├── Numeric Format Tests
│   ├── Timestamp Format Tests
│   └── Hash Format Tests
│
└── DRIFT DETECTION TESTS (Section 5.4)
    ├── Hash Chain Tests
    ├── Temporal Sequence Tests
    ├── Cross-Field Consistency Tests
    └── Anomaly Detection Tests
```

### 5.2 Schema Adherence Tests

#### 5.2.1 Required Field Tests

**Test Category:** Verify all required fields are validated correctly.

| Test ID | Description | Input | Expected Result |
|---------|-------------|-------|-----------------|
| REQ-001 | All required fields present | Valid L2 record | PASS: L1+ |
| REQ-002 | Missing one required field | Record minus `ht` | FAIL: Quarantine |
| REQ-003 | Missing multiple required fields | Record minus `ht`, `ts` | FAIL: Quarantine |
| REQ-004 | Required field is null | `"ht": null` | FAIL: Quarantine |
| REQ-005 | Required field is empty string | `"ht": ""` | FAIL: Quarantine |
| REQ-006 | Required field has wrong type | `"cycle": "42"` (string) | FAIL: Quarantine |

**Test fixture generation:**

For each record type, for each required field:
- Generate valid record
- Remove field
- Validate → expect quarantine

#### 5.2.2 Prohibited Field Tests

**Test Category:** Verify prohibited fields are rejected.

| Test ID | Description | Input | Expected Result |
|---------|-------------|-------|-----------------|
| PRO-001 | No prohibited fields | Valid L2 record | PASS: L1+ |
| PRO-002 | Single prohibited field | Add `"_id": "..."` | FAIL: Quarantine |
| PRO-003 | Multiple prohibited fields | Add `"timestamp"`, `"metadata"` | FAIL: Quarantine |
| PRO-004 | Prohibited field with null value | `"metadata": null` | FAIL: Quarantine |
| PRO-005 | Nested prohibited pattern | `"data": {"x": 1}` | FAIL: Quarantine |

**Test fixture generation:**

For each prohibited field from `TELEMETRY_CANONICAL_FORM.md` Section 4:
- Generate valid record
- Add prohibited field
- Validate → expect quarantine

#### 5.2.3 Type Validation Tests

**Test Category:** Verify field types are enforced.

| Test ID | Description | Field | Invalid Type | Expected |
|---------|-------------|-------|--------------|----------|
| TYP-001 | Integer as string | `cycle` | `"42"` | FAIL |
| TYP-002 | Boolean as string | `success` | `"true"` | FAIL |
| TYP-003 | Boolean as integer | `success` | `1` | FAIL |
| TYP-004 | Float as string | `p_success` | `"0.42"` | FAIL |
| TYP-005 | Array as object | `ci_95` | `{"low": 0.3}` | FAIL |
| TYP-006 | String as integer | `run_id` | `12345` | FAIL |
| TYP-007 | Null for non-nullable | `ht` | `null` | FAIL |

#### 5.2.4 Enum Validation Tests

**Test Category:** Verify enum constraints are enforced.

| Test ID | Field | Valid Values | Invalid Input | Expected |
|---------|-------|--------------|---------------|----------|
| ENM-001 | `mode` | baseline, rfl, comparison | `"test"` | FAIL |
| ENM-002 | `slice` | slice_uplift_goal, ... | `"custom_slice"` | FAIL |
| ENM-003 | `metric_type` | goal_hit, throughput, ... | `"custom"` | FAIL |
| ENM-004 | `phase` | I, II, III | `"IV"` | FAIL |
| ENM-005 | `mode` | (case sensitive) | `"BASELINE"` | FAIL |

#### 5.2.5 Range Validation Tests

**Test Category:** Verify numeric ranges are enforced.

| Test ID | Field | Valid Range | Invalid Input | Expected |
|---------|-------|-------------|---------------|----------|
| RNG-001 | `p_success` | [0.0, 1.0] | `1.5` | FAIL |
| RNG-002 | `p_success` | [0.0, 1.0] | `-0.1` | FAIL |
| RNG-003 | `cycle` | [0, ∞) | `-1` | FAIL |
| RNG-004 | `n_cycles` | [1, ∞) | `0` | FAIL |
| RNG-005 | `p_value` | [0.0, 1.0] | `1.001` | FAIL |
| RNG-006 | `ci_95[0]` | ≤ ci_95[1] | `[0.6, 0.4]` | FAIL |

### 5.3 Canonicalization Tests

#### 5.3.1 Field Ordering Tests

**Test Category:** Verify lexicographic field ordering requirement.

| Test ID | Description | Input | Expected |
|---------|-------------|-------|----------|
| ORD-001 | Canonical order | Fields in alpha order | PASS: L2 |
| ORD-002 | Reverse order | Fields in reverse alpha | FAIL: L1 only |
| ORD-003 | Random order | Fields randomized | FAIL: L1 only |
| ORD-004 | Partial order | Some fields out of order | FAIL: L1 only |

**Test procedure:**

1. Create record with correct field values
2. Serialize with specified field order
3. Validate conformance level
4. Verify L2 only for canonical order

#### 5.3.2 Serialization Format Tests

**Test Category:** Verify JSON serialization rules.

| Test ID | Description | Input | Expected |
|---------|-------------|-------|----------|
| SER-001 | Compact (no whitespace) | `{"a":1,"b":2}` | PASS: L2 |
| SER-002 | Spaces after colons | `{"a": 1,"b": 2}` | FAIL: L1 only |
| SER-003 | Spaces after commas | `{"a":1, "b":2}` | FAIL: L1 only |
| SER-004 | Pretty-printed | Multi-line JSON | FAIL: L1 only |
| SER-005 | Trailing comma | `{"a":1,}` | FAIL: Parse error |
| SER-006 | Single quotes | `{'a':1}` | FAIL: Parse error |

#### 5.3.3 Numeric Format Tests

**Test Category:** Verify numeric serialization rules.

| Test ID | Description | Input | Expected |
|---------|-------------|-------|----------|
| NUM-001 | Decimal notation | `0.123456` | PASS: L2 |
| NUM-002 | Scientific notation | `1.23e-4` | FAIL: L1 only |
| NUM-003 | Integer (no decimal) | `42` | PASS: L2 |
| NUM-004 | Float with trailing zeros | `0.500000` | PASS: L2 |
| NUM-005 | NaN | `NaN` | FAIL: Parse error |
| NUM-006 | Infinity | `Infinity` | FAIL: Parse error |
| NUM-007 | Negative zero | `-0` | PASS: L2 (equivalent to 0) |

#### 5.3.4 Timestamp Format Tests

**Test Category:** Verify timestamp serialization rules.

| Test ID | Description | Input | Expected |
|---------|-------------|-------|----------|
| TST-001 | Full precision with Z | `2025-12-06T10:30:00.123456Z` | PASS: L2 |
| TST-002 | Missing microseconds | `2025-12-06T10:30:00Z` | FAIL: L1 only |
| TST-003 | Milliseconds only | `2025-12-06T10:30:00.123Z` | FAIL: L1 only |
| TST-004 | Offset instead of Z | `2025-12-06T10:30:00.123456+00:00` | FAIL: L1 only |
| TST-005 | No timezone | `2025-12-06T10:30:00.123456` | FAIL: L1 only |
| TST-006 | Date only | `2025-12-06` | FAIL: Quarantine |
| TST-007 | Invalid date | `2025-13-45T10:30:00.123456Z` | FAIL: Quarantine |

#### 5.3.5 Hash Format Tests

**Test Category:** Verify hash serialization rules.

| Test ID | Description | Input | Expected |
|---------|-------------|-------|----------|
| HSH-001 | 64 lowercase hex | `abc123...` (64 chars) | PASS: L2 |
| HSH-002 | Uppercase hex | `ABC123...` | FAIL: L1 only |
| HSH-003 | Mixed case | `AbC123...` | FAIL: L1 only |
| HSH-004 | Too short (63 chars) | 63 hex chars | FAIL: Quarantine |
| HSH-005 | Too long (65 chars) | 65 hex chars | FAIL: Quarantine |
| HSH-006 | Non-hex characters | `xyz123...` | FAIL: Quarantine |
| HSH-007 | With 0x prefix | `0xabc123...` | FAIL: Quarantine |

### 5.4 Drift Detection Tests

#### 5.4.1 Hash Chain Tests

**Test Category:** Verify hash chain validation.

| Test ID | Description | Setup | Expected |
|---------|-------------|-------|----------|
| HCH-001 | Valid chain | Correct ht derivation | PASS |
| HCH-002 | Chain break | ht[42] incorrect | FAIL: Quarantine cycle 42+ |
| HCH-003 | Chain break recovery | Correct ht after break | FAIL: Chain already broken |
| HCH-004 | First record invalid | ht[0] not derived from run_id | FAIL: Quarantine |
| HCH-005 | Duplicate ht | ht[42] == ht[41] | FAIL: Quarantine |

**Test procedure for HCH-002:**

1. Generate valid 100-cycle run
2. Modify ht at cycle 42 to incorrect value
3. Run validation
4. Verify cycles 0-41 pass
5. Verify cycle 42+ quarantined

#### 5.4.2 Temporal Sequence Tests

**Test Category:** Verify timestamp monotonicity.

| Test ID | Description | Setup | Expected |
|---------|-------------|-------|----------|
| TMP-001 | Monotonic timestamps | Each ts > previous | PASS |
| TMP-002 | Timestamp regression | ts[42] < ts[41] | FAIL: Flag cycle 42 |
| TMP-003 | Duplicate timestamp | ts[42] == ts[41] | WARN: Log but accept |
| TMP-004 | Large gap | ts[42] > ts[41] + 1 hour | WARN: Anomaly flag |
| TMP-005 | Future timestamp | ts > now + 60s | FAIL: Quarantine |

#### 5.4.3 Cycle Sequence Tests

**Test Category:** Verify cycle number sequence.

| Test ID | Description | Setup | Expected |
|---------|-------------|-------|----------|
| CYC-001 | Sequential from 0 | 0, 1, 2, 3, ... | PASS |
| CYC-002 | Missing cycle | 0, 1, 3, 4 (no 2) | FAIL: Quarantine |
| CYC-003 | Duplicate cycle | 0, 1, 2, 2, 3 | FAIL: Quarantine |
| CYC-004 | Out of order | 0, 2, 1, 3 | FAIL: Quarantine |
| CYC-005 | Negative cycle | -1, 0, 1 | FAIL: Quarantine |
| CYC-006 | Non-integer cycle | 0, 1.5, 2 | FAIL: Quarantine |

#### 5.4.4 Cross-Field Consistency Tests

**Test Category:** Verify derived field correctness.

| Test ID | Record Type | Check | Setup | Expected |
|---------|-------------|-------|-------|----------|
| XFC-001 | uplift_result | delta = p_rfl - p_base | Correct values | PASS |
| XFC-002 | uplift_result | delta ≠ p_rfl - p_base | Off by 0.01 | FAIL: Quarantine |
| XFC-003 | uplift_result | significant matches CI | ci_lower > 0, sig=true | PASS |
| XFC-004 | uplift_result | significant mismatch | ci_lower > 0, sig=false | FAIL: Quarantine |
| XFC-005 | experiment_summary | p within ci_95 | p_success in bounds | PASS |
| XFC-006 | experiment_summary | p outside ci_95 | p_success out of bounds | FAIL: Quarantine |

### 5.5 Test Execution Requirements

#### 5.5.1 Test Coverage Requirements

| Category | Minimum Coverage |
|----------|------------------|
| Schema adherence | 100% of required fields |
| Schema adherence | 100% of prohibited fields |
| Schema adherence | All type combinations |
| Schema adherence | All enum values + boundaries |
| Canonicalization | All format rules |
| Drift detection | All chain/sequence checks |

#### 5.5.2 Test Data Requirements

| Requirement | Specification |
|-------------|---------------|
| Valid record corpus | ≥ 100 valid L2 records per type |
| Invalid record corpus | ≥ 10 invalid variants per check |
| Edge cases | Boundary values for all numerics |
| Unicode | Records with non-ASCII strings |
| Large records | Records at size limits |

#### 5.5.3 Test Environment Requirements

| Requirement | Specification |
|-------------|---------------|
| Isolation | Tests must not affect production data |
| Determinism | Same input → same result |
| Speed | Full suite < 60 seconds |
| Reporting | Machine-readable test results |
| CI integration | Run on every commit |

### 5.6 Test Result Classification

| Result | Meaning | Action |
|--------|---------|--------|
| PASS | Validation behaves correctly | None |
| FAIL | Validation does not match spec | Fix implementation |
| ERROR | Test execution failed | Fix test infrastructure |
| SKIP | Test not applicable | Document reason |

---

## 6. Conformance Attestation

### 6.1 Attestation Record

When a record or run achieves L2 conformance, an attestation record SHOULD be generated:

```
{
  "attestation_type": "conformance",
  "attestation_id": "<uuid>",
  "attested_at": "<ISO 8601>",
  "attested_by": "conformance_validator_v1.0",
  "subject": {
    "type": "run|record",
    "run_id": "<run_id>",
    "cycle": <integer or null>,
    "canonical_hash": "<hash>"
  },
  "conformance": {
    "level": "L2",
    "schema_version": "1.0",
    "canonical_form_version": "1.0",
    "checks_passed": [
      "schema_adherence",
      "field_ordering",
      "serialization_format",
      "hash_chain",
      "temporal_sequence",
      "cross_field_consistency"
    ]
  }
}
```

### 6.2 Run-Level Attestation

For a complete run, attestation includes aggregate metrics:

```
{
  "attestation_type": "run_conformance",
  "attestation_id": "<uuid>",
  "attested_at": "<ISO 8601>",
  "run_id": "<run_id>",
  "conformance": {
    "level": "L2",
    "total_records": <integer>,
    "l2_records": <integer>,
    "quarantined_records": <integer>,
    "hash_chain_valid": <boolean>,
    "temporal_sequence_valid": <boolean>,
    "cycle_sequence_complete": <boolean>
  },
  "run_hash": "<hash of all canonical record hashes>"
}
```

---

## Appendix A: Conformance Level Decision Tree

```
                         ┌─────────────────┐
                         │  Receive Record │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ Parse as JSON?  │
                         └────────┬────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │ NO                        │ YES
                    ▼                           ▼
           ┌────────────────┐          ┌─────────────────┐
           │  QUARANTINE    │          │ L0: Raw         │
           │  (parse error) │          └────────┬────────┘
           └────────────────┘                   │
                                                ▼
                                       ┌─────────────────┐
                                       │ Schema valid?   │
                                       │ (JSON Schema)   │
                                       └────────┬────────┘
                                                │
                                  ┌─────────────┴─────────────┐
                                  │ NO                        │ YES
                                  ▼                           ▼
                         ┌────────────────┐          ┌─────────────────┐
                         │  QUARANTINE    │          │ L1: Schema-Valid│
                         │  (schema fail) │          └────────┬────────┘
                         └────────────────┘                   │
                                                              ▼
                                                     ┌─────────────────┐
                                                     │ Field order     │
                                                     │ canonical?      │
                                                     └────────┬────────┘
                                                              │
                                                ┌─────────────┴─────────────┐
                                                │ NO                        │ YES
                                                ▼                           │
                                       ┌────────────────┐                   │
                                       │ Remains L1     │                   │
                                       └────────────────┘                   │
                                                                            ▼
                                                                   ┌─────────────────┐
                                                                   │ Serialization   │
                                                                   │ canonical?      │
                                                                   └────────┬────────┘
                                                                            │
                                                              ┌─────────────┴─────────────┐
                                                              │ NO                        │ YES
                                                              ▼                           │
                                                     ┌────────────────┐                   │
                                                     │ Remains L1     │                   │
                                                     └────────────────┘                   │
                                                                                          ▼
                                                                                 ┌─────────────────┐
                                                                                 │ Hash chain &    │
                                                                                 │ sequences valid?│
                                                                                 └────────┬────────┘
                                                                                          │
                                                                            ┌─────────────┴─────────────┐
                                                                            │ NO                        │ YES
                                                                            ▼                           ▼
                                                                   ┌────────────────┐          ┌────────────────┐
                                                                   │  QUARANTINE    │          │ L2: Canonical  │
                                                                   │  (chain break) │          └────────────────┘
                                                                   └────────────────┘
```

---

## Appendix B: Quick Reference

### B.1 Conformance Level Summary

| Level | Schema | Ordering | Format | Chain | Governance |
|-------|--------|----------|--------|-------|------------|
| L0 | No | No | No | No | Not allowed |
| L1 | Yes | No | No | No | Not allowed |
| L2 | Yes | Yes | Yes | Yes | Allowed |

### B.2 Quarantine Severity Summary

| Severity | Consequence | Auto-purge |
|----------|-------------|------------|
| CRITICAL | Run invalidated | After 90 days |
| ERROR | Record quarantined | After 180 days |
| WARNING | Flagged for review | After 180 days |

### B.3 Consumer Permission Summary

| Consumer | L0 | L1 | L2 |
|----------|-----|-----|-----|
| Governance | No | No | Yes |
| Dashboard (live) | No | Warn | Yes |
| Dashboard (historical) | No | No | Yes |
| Archiver | Quarantine | Tag | Archive |
| Aggregator | No | No | Yes |

---

*End of Document*
