# Telemetry Canonical Form Specification

**Document Version:** 1.0
**Author:** CLAUDE H (Telemetry Lawyer)
**Date:** 2025-12-06
**Phase:** II — Operation Asymmetry
**Status:** PHASE II — NOT IMPLEMENTED

```
+===============================================================================+
|                                                                               |
|              NOT IMPLEMENTED — FUTURE PHASE II                                |
|                                                                               |
|  This document defines the canonical form for U2 telemetry records.           |
|  All emitters MUST conform to this specification. Drift from canonical        |
|  form invalidates records for governance purposes.                            |
|                                                                               |
|              PHASE II — NOT RUN IN PHASE I                                    |
|                                                                               |
+===============================================================================+
```

---

## 1. Purpose

This document establishes the **canonical form** for all U2 telemetry records. Canonical form ensures:

1. **Deterministic hashing** — Identical logical records produce identical hashes
2. **Schema stability** — Field ordering is fixed and versioned
3. **Audit compatibility** — Records can be verified across systems
4. **Governance validity** — Only canonical records pass governance gates

---

## 2. Stable Field Ordering

### 2.1 Ordering Principle

All JSON records MUST serialize fields in **lexicographic order by key name**. This ensures deterministic output regardless of the emitter's internal data structure ordering.

### 2.2 Record Type: `cycle_metric`

**Canonical field order (alphabetical):**

```
1.  cycle
2.  ht
3.  metric_type
4.  metric_value
5.  mode
6.  r_t
7.  run_id
8.  slice
9.  success
10. ts
11. u_t
```

**Canonical example:**

```json
{"cycle":123,"ht":"abc123...","metric_type":"goal_hit","metric_value":1.0,"mode":"baseline","r_t":"def456...","run_id":"U2-aaa-bbb","slice":"slice_uplift_goal","success":true,"ts":"2025-12-05T14:30:00.123456Z","u_t":"789abc..."}
```

### 2.3 Record Type: `experiment_summary`

**Canonical field order (alphabetical):**

```
1.  ci_95
2.  mode
3.  n_cycles
4.  p_success
5.  phase
6.  run_id
7.  slice
8.  uplift_delta
```

**Canonical example:**

```json
{"ci_95":[0.377,0.464],"mode":"baseline","n_cycles":500,"p_success":0.42,"phase":"II","run_id":"U2-baseline-001","slice":"slice_uplift_goal","uplift_delta":null}
```

### 2.4 Record Type: `uplift_result`

**Canonical field order (alphabetical):**

```
1.  baseline_run_id
2.  ci_95
3.  n_base
4.  n_rfl
5.  p_base
6.  p_rfl
7.  p_value
8.  phase
9.  rfl_run_id
10. significant
11. slice
12. ts
13. uplift_delta
```

**Canonical example:**

```json
{"baseline_run_id":"U2-baseline-001","ci_95":[0.089,0.231],"n_base":500,"n_rfl":500,"p_base":0.42,"p_rfl":0.58,"p_value":0.00023,"phase":"II","rfl_run_id":"U2-rfl-001","significant":true,"slice":"slice_uplift_goal","ts":"2025-12-05T16:00:00Z","uplift_delta":0.16}
```

---

## 3. Canonical Serialization Rules

### 3.1 JSON Serialization

| Rule | Specification | Example |
|------|---------------|---------|
| **Encoding** | UTF-8, no BOM | — |
| **Whitespace** | None between tokens | `{"a":1}` not `{ "a" : 1 }` |
| **Key ordering** | Lexicographic (ASCII) | `a` < `b` < `ci_95` < `cycle` |
| **Key quoting** | Double quotes only | `"key"` not `'key'` |
| **String quoting** | Double quotes only | `"value"` not `'value'` |
| **Escape sequences** | Standard JSON escapes | `\n`, `\t`, `\\`, `\"` |
| **Unicode** | Unescaped UTF-8 preferred | `"π"` not `"\u03C0"` |
| **Line terminator** | Single `\n` (LF) per record | No trailing whitespace |

### 3.2 Numeric Serialization

| Type | Format | Precision | Example |
|------|--------|-----------|---------|
| **Integer** | No decimal point | Exact | `123`, `0`, `-5` |
| **Float** | Decimal notation | 6 significant digits | `0.123456`, `1.5` |
| **Probability** | Decimal, 0-1 range | 6 decimal places max | `0.42`, `0.123456` |
| **Scientific** | NOT allowed | — | `1.23e5` → `123000` |
| **Infinity/NaN** | NOT allowed | — | Use `null` instead |

### 3.3 Timestamp Serialization

| Component | Format | Example |
|-----------|--------|---------|
| **Full timestamp** | ISO 8601 with microseconds | `2025-12-05T14:30:00.123456Z` |
| **Timezone** | Always UTC (`Z` suffix) | Not `+00:00` |
| **Precision** | Microseconds (6 digits) | `.123456` not `.123` |
| **Date-only** | NOT allowed for `ts` | Use full timestamp |

### 3.4 Hash Serialization

| Field | Format | Length | Example |
|-------|--------|--------|---------|
| `ht` | Lowercase hex | 64 chars | `abc123def456...` (64 chars) |
| `r_t` | Lowercase hex | 64 chars | `def456789012...` (64 chars) |
| `u_t` | Lowercase hex | 64 chars | `789012345678...` (64 chars) |

**Validation regex:** `^[0-9a-f]{64}$`

### 3.5 Array Serialization

| Rule | Specification | Example |
|------|---------------|---------|
| **Brackets** | No internal whitespace | `[0.35,0.49]` not `[ 0.35, 0.49 ]` |
| **Separator** | Comma, no trailing | `[1,2,3]` not `[1,2,3,]` |
| **ci_95 order** | `[lower, upper]` always | `[0.35,0.49]` |

### 3.6 Null Handling

| Context | Rule | Example |
|---------|------|---------|
| **Optional field absent** | Omit field entirely | `{}` not `{"field":null}` |
| **Explicit null value** | Include as `null` | `"uplift_delta":null` |
| **Empty string** | NOT equivalent to null | `""` ≠ `null` |

---

## 4. Prohibited Fields

The following field names are **PROHIBITED** in canonical records. Their presence invalidates the record.

### 4.1 Reserved Field Names

| Field | Reason | Alternative |
|-------|--------|-------------|
| `_id` | MongoDB artifact | Use `run_id` |
| `id` | Ambiguous identifier | Use `run_id` |
| `timestamp` | Non-canonical name | Use `ts` |
| `time` | Non-canonical name | Use `ts` |
| `datetime` | Non-canonical name | Use `ts` |
| `created_at` | Database artifact | Use `ts` |
| `updated_at` | Implies mutability | Records are immutable |
| `version` | Ambiguous | Use `phase` for phase version |
| `schema_version` | Redundant | Derive from `phase` |
| `type` | Reserved keyword | Use `metric_type` |
| `status` | Ambiguous | Use `success` (boolean) |
| `result` | Ambiguous | Use specific metric fields |
| `data` | Wrapper anti-pattern | Flatten into record |
| `payload` | Wrapper anti-pattern | Flatten into record |
| `metadata` | Encourages schema drift | Define explicit fields |
| `extra` | Encourages schema drift | Not allowed |
| `custom` | Encourages schema drift | Not allowed |
| `tags` | Unstructured data | Use explicit labels |
| `labels` | Non-canonical wrapper | Flatten into record |

### 4.2 Prohibited Patterns

| Pattern | Reason | Example |
|---------|--------|---------|
| Nested objects | Increases complexity | `{"config":{"depth":4}}` |
| Arrays (except `ci_95`) | Schema instability | `{"values":[1,2,3]}` |
| Underscore prefix | Reserved for internal | `_internal_field` |
| Dollar prefix | MongoDB artifact | `$oid`, `$date` |
| CamelCase keys | Inconsistent style | `runId` → `run_id` |

### 4.3 Validation Regex for Field Names

```
^[a-z][a-z0-9_]*$
```

- Must start with lowercase letter
- Only lowercase letters, digits, underscores
- No consecutive underscores
- No trailing underscores

---

## 5. Versioning Scheme

### 5.1 Phase-Based Versioning

Telemetry schema is versioned by **Phase**, not by semantic version. This ties schema changes to governance milestones.

| Phase | Schema Features | Compatibility |
|-------|-----------------|---------------|
| `I` | FO metrics only (fo_baseline_wide.jsonl) | Legacy, not U2 |
| `II` | U2 canonical form (this document) | Current |
| `III` | Future extensions (TBD) | Forward-compatible |

### 5.2 Version Identification

Records identify their version via the `phase` field:

```json
{"phase": "II", ...}
```

**Rules:**
- `phase` is REQUIRED in `experiment_summary` and `uplift_result`
- `phase` is IMPLIED as `II` in `cycle_metric` (omit for compactness)
- Unknown phase values → reject record

### 5.3 Schema Evolution Rules

| Change Type | Allowed? | Procedure |
|-------------|----------|-----------|
| Add optional field | Yes | Document in next phase |
| Add required field | No | Requires new phase |
| Remove field | No | Deprecate, keep parsing |
| Rename field | No | Add new, deprecate old |
| Change field type | No | Requires new phase |
| Change field semantics | No | Requires new phase |
| Reorder fields | No | Order is canonical |

### 5.4 Deprecation Process

1. Mark field as deprecated in documentation
2. Continue emitting deprecated field for 2 phases
3. Continue parsing deprecated field indefinitely
4. Never remove parsing support

---

## 6. Canonical Hash Computation

### 6.1 Record Hash Algorithm

To compute a deterministic hash of a record:

```python
# PHASE II — NOT IMPLEMENTED

import hashlib
import json

def canonical_hash(record: dict) -> str:
    """
    Compute SHA-256 hash of record in canonical form.

    NOT IMPLEMENTED — FUTURE PHASE II
    """
    # 1. Sort keys lexicographically
    canonical = json.dumps(
        record,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False
    )

    # 2. Encode as UTF-8
    canonical_bytes = canonical.encode('utf-8')

    # 3. Compute SHA-256
    return hashlib.sha256(canonical_bytes).hexdigest()
```

### 6.2 Hash Chain Verification

For a sequence of `cycle_metric` records, verify hash chain:

```python
# PHASE II — NOT IMPLEMENTED

def verify_hash_chain(records: list[dict]) -> bool:
    """
    Verify that ht values form a valid chain.

    NOT IMPLEMENTED — FUTURE PHASE II
    """
    for i, record in enumerate(records):
        if i == 0:
            # First record: ht should be derivable from run_id
            expected = hashlib.sha256(
                f"{record['run_id']}:0".encode()
            ).hexdigest()
        else:
            # Subsequent: ht chains from previous
            prev = records[i - 1]
            expected = hashlib.sha256(
                f"{prev['ht']}:{record['cycle']}".encode()
            ).hexdigest()

        if record['ht'] != expected:
            return False

    return True
```

---

## 7. Telemetry Drift Detection Specification

### 7.1 Overview

Telemetry drift occurs when emitted records deviate from canonical form. Drift detection ensures governance validity by catching:

1. **Schema drift** — New/missing/renamed fields
2. **Format drift** — Non-canonical serialization
3. **Semantic drift** — Valid schema but invalid semantics
4. **Temporal drift** — Timestamp anomalies

### 7.2 Drift Detection Architecture

```
                    +---------------------------+
                    |     Telemetry Emitter     |
                    +-------------+-------------+
                                  |
                                  | JSONL records
                                  v
                    +---------------------------+
                    |    Drift Detection Layer  |
                    |  +---------------------+  |
                    |  | Schema Validator    |  |
                    |  +---------------------+  |
                    |  | Format Validator    |  |
                    |  +---------------------+  |
                    |  | Semantic Validator  |  |
                    |  +---------------------+  |
                    |  | Temporal Validator  |  |
                    |  +---------------------+  |
                    +-------------+-------------+
                                  |
                    +-------------+-------------+
                    |     PASS     |    FAIL    |
                    +-------------+-------------+
                          |              |
                          v              v
                    +-----------+  +-----------+
                    | Governance |  | Quarantine|
                    | Pipeline   |  | + Alert   |
                    +-----------+  +-----------+
```

### 7.3 Schema Drift Detection

#### 7.3.1 Required Fields Check

```python
# PHASE II — NOT IMPLEMENTED

REQUIRED_FIELDS = {
    "cycle_metric": {
        "ts", "run_id", "slice", "mode", "cycle",
        "success", "metric_type", "metric_value",
        "ht", "r_t", "u_t"
    },
    "experiment_summary": {
        "run_id", "slice", "mode", "n_cycles",
        "p_success", "ci_95", "uplift_delta", "phase"
    },
    "uplift_result": {
        "ts", "slice", "baseline_run_id", "rfl_run_id",
        "p_base", "p_rfl", "uplift_delta", "ci_95",
        "significant", "p_value", "n_base", "n_rfl", "phase"
    }
}

def check_required_fields(record: dict, record_type: str) -> list[str]:
    """Return list of missing required fields."""
    required = REQUIRED_FIELDS.get(record_type, set())
    present = set(record.keys())
    return list(required - present)
```

#### 7.3.2 Prohibited Fields Check

```python
# PHASE II — NOT IMPLEMENTED

PROHIBITED_FIELDS = {
    "_id", "id", "timestamp", "time", "datetime",
    "created_at", "updated_at", "version", "schema_version",
    "type", "status", "result", "data", "payload",
    "metadata", "extra", "custom", "tags", "labels"
}

def check_prohibited_fields(record: dict) -> list[str]:
    """Return list of prohibited fields present."""
    present = set(record.keys())
    return list(present & PROHIBITED_FIELDS)
```

#### 7.3.3 Unknown Fields Check

```python
# PHASE II — NOT IMPLEMENTED

KNOWN_FIELDS = {
    "cycle_metric": REQUIRED_FIELDS["cycle_metric"],
    "experiment_summary": REQUIRED_FIELDS["experiment_summary"],
    "uplift_result": REQUIRED_FIELDS["uplift_result"]
}

def check_unknown_fields(record: dict, record_type: str) -> list[str]:
    """Return list of unknown fields (potential schema drift)."""
    known = KNOWN_FIELDS.get(record_type, set())
    present = set(record.keys())
    return list(present - known)
```

### 7.4 Format Drift Detection

#### 7.4.1 Field Ordering Check

```python
# PHASE II — NOT IMPLEMENTED

import json

def check_field_ordering(record_json: str) -> bool:
    """
    Check if JSON string has fields in canonical order.
    Returns True if canonical, False if drift detected.
    """
    # Parse and re-serialize canonically
    record = json.loads(record_json)
    canonical = json.dumps(record, sort_keys=True, separators=(',', ':'))

    # Compare original to canonical
    return record_json.strip() == canonical
```

#### 7.4.2 Numeric Format Check

```python
# PHASE II — NOT IMPLEMENTED

import re

def check_numeric_format(record: dict) -> list[str]:
    """Return list of fields with non-canonical numeric format."""
    errors = []

    for key, value in record.items():
        if isinstance(value, float):
            # Check for scientific notation in original JSON
            # (This requires access to raw JSON string)
            if abs(value) >= 1e6 or (abs(value) < 1e-4 and value != 0):
                errors.append(f"{key}: value may require scientific notation")

            # Check for NaN/Infinity
            if value != value:  # NaN check
                errors.append(f"{key}: NaN not allowed")
            if value == float('inf') or value == float('-inf'):
                errors.append(f"{key}: Infinity not allowed")

    return errors
```

#### 7.4.3 Timestamp Format Check

```python
# PHASE II — NOT IMPLEMENTED

import re
from datetime import datetime

TS_REGEX = re.compile(
    r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$'
)

def check_timestamp_format(record: dict) -> list[str]:
    """Return list of timestamp fields with non-canonical format."""
    errors = []

    for key in ['ts']:
        if key in record:
            value = record[key]
            if not TS_REGEX.match(value):
                errors.append(f"{key}: must match YYYY-MM-DDTHH:MM:SS.ffffffZ")
            else:
                # Validate it's a real timestamp
                try:
                    datetime.fromisoformat(value.replace('Z', '+00:00'))
                except ValueError:
                    errors.append(f"{key}: invalid timestamp value")

    return errors
```

#### 7.4.4 Hash Format Check

```python
# PHASE II — NOT IMPLEMENTED

import re

HASH_REGEX = re.compile(r'^[0-9a-f]{64}$')

def check_hash_format(record: dict) -> list[str]:
    """Return list of hash fields with non-canonical format."""
    errors = []

    for key in ['ht', 'r_t', 'u_t']:
        if key in record:
            value = record[key]
            if not HASH_REGEX.match(value):
                errors.append(f"{key}: must be 64 lowercase hex chars")

    return errors
```

### 7.5 Semantic Drift Detection

#### 7.5.1 Value Range Check

```python
# PHASE II — NOT IMPLEMENTED

VALUE_RANGES = {
    "p_success": (0.0, 1.0),
    "p_base": (0.0, 1.0),
    "p_rfl": (0.0, 1.0),
    "p_value": (0.0, 1.0),
    "metric_value": (0.0, float('inf')),  # Depends on metric_type
    "n_cycles": (1, float('inf')),
    "n_base": (1, float('inf')),
    "n_rfl": (1, float('inf')),
    "cycle": (0, float('inf')),
}

def check_value_ranges(record: dict) -> list[str]:
    """Return list of fields with out-of-range values."""
    errors = []

    for key, (min_val, max_val) in VALUE_RANGES.items():
        if key in record and record[key] is not None:
            value = record[key]
            if value < min_val or value > max_val:
                errors.append(f"{key}: {value} not in [{min_val}, {max_val}]")

    return errors
```

#### 7.5.2 Enum Value Check

```python
# PHASE II — NOT IMPLEMENTED

ENUM_VALUES = {
    "mode": {"baseline", "rfl", "comparison"},
    "slice": {
        "slice_uplift_goal", "slice_throughput",
        "slice_depth_advance", "slice_novelty"
    },
    "metric_type": {
        "goal_hit", "throughput", "depth_reached", "novel_count"
    },
    "phase": {"I", "II", "III"},
}

def check_enum_values(record: dict) -> list[str]:
    """Return list of fields with invalid enum values."""
    errors = []

    for key, valid_values in ENUM_VALUES.items():
        if key in record:
            value = record[key]
            if value not in valid_values:
                errors.append(f"{key}: '{value}' not in {valid_values}")

    return errors
```

#### 7.5.3 Cross-Field Consistency Check

```python
# PHASE II — NOT IMPLEMENTED

def check_cross_field_consistency(record: dict, record_type: str) -> list[str]:
    """Return list of cross-field consistency violations."""
    errors = []

    if record_type == "experiment_summary":
        # ci_95[0] <= p_success <= ci_95[1]
        if record.get("ci_95") and record.get("p_success"):
            ci = record["ci_95"]
            p = record["p_success"]
            if not (ci[0] <= p <= ci[1]):
                errors.append("p_success must be within ci_95 bounds")

    if record_type == "uplift_result":
        # uplift_delta == p_rfl - p_base
        if all(k in record for k in ["uplift_delta", "p_rfl", "p_base"]):
            expected = record["p_rfl"] - record["p_base"]
            actual = record["uplift_delta"]
            if abs(expected - actual) > 1e-6:
                errors.append("uplift_delta must equal p_rfl - p_base")

        # ci_95[0] > 0 iff significant == true
        if record.get("ci_95") and "significant" in record:
            ci_lower = record["ci_95"][0]
            sig = record["significant"]
            if (ci_lower > 0) != sig:
                errors.append("significant must match ci_95[0] > 0")

    return errors
```

### 7.6 Temporal Drift Detection

#### 7.6.1 Timestamp Monotonicity Check

```python
# PHASE II — NOT IMPLEMENTED

from datetime import datetime

def check_timestamp_monotonicity(records: list[dict]) -> list[str]:
    """
    Check that timestamps are monotonically increasing within a run.
    Returns list of violations.
    """
    errors = []
    prev_ts = None

    for i, record in enumerate(records):
        if "ts" not in record:
            continue

        ts = datetime.fromisoformat(record["ts"].replace('Z', '+00:00'))

        if prev_ts and ts < prev_ts:
            errors.append(
                f"Record {i}: timestamp {record['ts']} < previous {prev_ts.isoformat()}"
            )

        prev_ts = ts

    return errors
```

#### 7.6.2 Cycle Sequence Check

```python
# PHASE II — NOT IMPLEMENTED

def check_cycle_sequence(records: list[dict]) -> list[str]:
    """
    Check that cycle numbers are sequential starting from 0.
    Returns list of violations.
    """
    errors = []
    expected_cycle = 0

    for i, record in enumerate(records):
        if "cycle" not in record:
            continue

        actual = record["cycle"]
        if actual != expected_cycle:
            errors.append(
                f"Record {i}: cycle {actual} expected {expected_cycle}"
            )

        expected_cycle = actual + 1

    return errors
```

#### 7.6.3 Future Timestamp Check

```python
# PHASE II — NOT IMPLEMENTED

from datetime import datetime, timezone

def check_future_timestamp(record: dict, tolerance_seconds: int = 60) -> list[str]:
    """
    Check that timestamp is not in the future (with tolerance).
    Returns list of violations.
    """
    errors = []

    if "ts" in record:
        ts = datetime.fromisoformat(record["ts"].replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)

        if ts > now + timedelta(seconds=tolerance_seconds):
            errors.append(f"ts {record['ts']} is in the future")

    return errors
```

### 7.7 Drift Detection API

```python
# PHASE II — NOT IMPLEMENTED

from dataclasses import dataclass
from enum import Enum
from typing import Optional

class DriftSeverity(Enum):
    INFO = "info"          # Non-blocking, logged
    WARNING = "warning"    # Non-blocking, alerted
    ERROR = "error"        # Blocking, quarantined
    CRITICAL = "critical"  # Blocking, run invalidated

@dataclass
class DriftReport:
    """Report of drift detection results."""
    record_index: int
    record_type: str
    run_id: str
    violations: list[tuple[str, DriftSeverity, str]]  # (check_name, severity, message)
    is_valid: bool
    canonical_hash: Optional[str]

class TelemetryDriftDetector:
    """
    Drift detection engine for U2 telemetry.

    NOT IMPLEMENTED — FUTURE PHASE II
    """

    def validate_record(
        self,
        record_json: str,
        record_type: str
    ) -> DriftReport:
        """
        Validate a single record against canonical form.

        Args:
            record_json: Raw JSON string of the record
            record_type: One of 'cycle_metric', 'experiment_summary', 'uplift_result'

        Returns:
            DriftReport with all violations found
        """
        ...

    def validate_run(
        self,
        jsonl_path: str
    ) -> list[DriftReport]:
        """
        Validate all records in a JSONL file.

        Args:
            jsonl_path: Path to JSONL file

        Returns:
            List of DriftReports, one per record
        """
        ...

    def get_run_validity(
        self,
        reports: list[DriftReport]
    ) -> tuple[bool, str]:
        """
        Determine overall run validity from drift reports.

        Returns:
            (is_valid, reason) tuple
        """
        ...
```

### 7.8 Drift Severity Classification

| Check | Severity | Consequence |
|-------|----------|-------------|
| Missing required field | CRITICAL | Invalidate run |
| Prohibited field present | ERROR | Quarantine record |
| Unknown field present | WARNING | Log and alert |
| Non-canonical field order | WARNING | Log, reserialize |
| Numeric format violation | ERROR | Quarantine record |
| Timestamp format violation | ERROR | Quarantine record |
| Hash format violation | CRITICAL | Invalidate run |
| Value out of range | ERROR | Quarantine record |
| Invalid enum value | ERROR | Quarantine record |
| Cross-field inconsistency | CRITICAL | Invalidate run |
| Timestamp non-monotonic | ERROR | Flag for review |
| Cycle sequence gap | CRITICAL | Invalidate run |
| Future timestamp | WARNING | Log and alert |

### 7.9 Quarantine Protocol

When a record fails drift detection with ERROR or CRITICAL severity:

1. **Quarantine:** Move record to `results/quarantine/<run_id>/`
2. **Alert:** Emit alert to monitoring system
3. **Log:** Record violation details in `logs/drift_violations.jsonl`
4. **Block:** Prevent governance gate evaluation

**Quarantine record format:**

```json
{
  "quarantined_at": "2025-12-06T10:00:00Z",
  "original_record": { ... },
  "violations": [
    {
      "check": "check_required_fields",
      "severity": "CRITICAL",
      "message": "Missing field: ht"
    }
  ],
  "run_id": "U2-xxx",
  "record_index": 42
}
```

---

## 8. Implementation Checklist

```
+===============================================================================+
|                                                                               |
|              NOT IMPLEMENTED — FUTURE PHASE II                                |
|                                                                               |
+===============================================================================+
```

### 8.1 Canonical Form Implementation

- [ ] Create `backend/metrics/canonical_form.py` module
- [ ] Implement `canonicalize_record()` function
- [ ] Implement `canonical_hash()` function
- [ ] Add JSON schema files for each record type
- [ ] Validate field ordering in emitters

### 8.2 Drift Detection Implementation

- [ ] Create `backend/metrics/drift_detector.py` module
- [ ] Implement all schema checks (7.3)
- [ ] Implement all format checks (7.4)
- [ ] Implement all semantic checks (7.5)
- [ ] Implement all temporal checks (7.6)
- [ ] Add quarantine directory structure
- [ ] Wire drift detection to ingestion pipeline

### 8.3 Testing

- [ ] Unit tests for each check function
- [ ] Integration test with sample JSONL files
- [ ] Fuzzing with malformed records
- [ ] Performance benchmark (target: <1ms per record)

---

## Appendix A: JSON Schema Definitions

### A.1 cycle_metric JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://mathledger.io/schemas/cycle_metric.json",
  "title": "cycle_metric",
  "type": "object",
  "required": [
    "cycle", "ht", "metric_type", "metric_value", "mode",
    "r_t", "run_id", "slice", "success", "ts", "u_t"
  ],
  "additionalProperties": false,
  "properties": {
    "cycle": {
      "type": "integer",
      "minimum": 0
    },
    "ht": {
      "type": "string",
      "pattern": "^[0-9a-f]{64}$"
    },
    "metric_type": {
      "type": "string",
      "enum": ["goal_hit", "throughput", "depth_reached", "novel_count"]
    },
    "metric_value": {
      "type": "number",
      "minimum": 0
    },
    "mode": {
      "type": "string",
      "enum": ["baseline", "rfl"]
    },
    "r_t": {
      "type": "string",
      "pattern": "^[0-9a-f]{64}$"
    },
    "run_id": {
      "type": "string",
      "pattern": "^U2-[0-9a-f-]+$"
    },
    "slice": {
      "type": "string",
      "enum": [
        "slice_uplift_goal", "slice_throughput",
        "slice_depth_advance", "slice_novelty"
      ]
    },
    "success": {
      "type": "boolean"
    },
    "ts": {
      "type": "string",
      "pattern": "^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}Z$"
    },
    "u_t": {
      "type": "string",
      "pattern": "^[0-9a-f]{64}$"
    }
  }
}
```

### A.2 experiment_summary JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://mathledger.io/schemas/experiment_summary.json",
  "title": "experiment_summary",
  "type": "object",
  "required": [
    "ci_95", "mode", "n_cycles", "p_success",
    "phase", "run_id", "slice", "uplift_delta"
  ],
  "additionalProperties": false,
  "properties": {
    "ci_95": {
      "type": "array",
      "items": { "type": "number" },
      "minItems": 2,
      "maxItems": 2
    },
    "mode": {
      "type": "string",
      "enum": ["baseline", "rfl", "comparison"]
    },
    "n_cycles": {
      "type": "integer",
      "minimum": 1
    },
    "p_success": {
      "type": ["number", "null"],
      "minimum": 0,
      "maximum": 1
    },
    "phase": {
      "type": "string",
      "enum": ["I", "II", "III"]
    },
    "run_id": {
      "type": "string",
      "pattern": "^U2-[0-9a-z-]+$"
    },
    "slice": {
      "type": "string",
      "enum": [
        "slice_uplift_goal", "slice_throughput",
        "slice_depth_advance", "slice_novelty"
      ]
    },
    "uplift_delta": {
      "type": ["number", "null"]
    }
  }
}
```

### A.3 uplift_result JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://mathledger.io/schemas/uplift_result.json",
  "title": "uplift_result",
  "type": "object",
  "required": [
    "baseline_run_id", "ci_95", "n_base", "n_rfl",
    "p_base", "p_rfl", "p_value", "phase", "rfl_run_id",
    "significant", "slice", "ts", "uplift_delta"
  ],
  "additionalProperties": false,
  "properties": {
    "baseline_run_id": {
      "type": "string",
      "pattern": "^U2-[0-9a-z-]+$"
    },
    "ci_95": {
      "type": "array",
      "items": { "type": "number" },
      "minItems": 2,
      "maxItems": 2
    },
    "n_base": {
      "type": "integer",
      "minimum": 1
    },
    "n_rfl": {
      "type": "integer",
      "minimum": 1
    },
    "p_base": {
      "type": "number",
      "minimum": 0,
      "maximum": 1
    },
    "p_rfl": {
      "type": "number",
      "minimum": 0,
      "maximum": 1
    },
    "p_value": {
      "type": "number",
      "minimum": 0,
      "maximum": 1
    },
    "phase": {
      "type": "string",
      "enum": ["I", "II", "III"]
    },
    "rfl_run_id": {
      "type": "string",
      "pattern": "^U2-[0-9a-z-]+$"
    },
    "significant": {
      "type": "boolean"
    },
    "slice": {
      "type": "string",
      "enum": [
        "slice_uplift_goal", "slice_throughput",
        "slice_depth_advance", "slice_novelty"
      ]
    },
    "ts": {
      "type": "string",
      "pattern": "^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}(\\.\\d+)?Z$"
    },
    "uplift_delta": {
      "type": "number"
    }
  }
}
```

---

*End of Document*
