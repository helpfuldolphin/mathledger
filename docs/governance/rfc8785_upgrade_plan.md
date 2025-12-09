# RFC 8785 Canonicalizer Upgrade Plan

**Author**: MANUS-G (CI/Governance Systems Architect)
**Version**: 1.0.0
**Status**: **DRAFT**

## 1. Overview

The current implementation of RFC 8785 canonicalization in MathLedger is **simplified** and does not handle all edge cases specified in the RFC. This document outlines a plan to upgrade to a **production-grade** implementation that provides full RFC 8785 compliance.

RFC 8785, also known as **JSON Canonicalization Scheme (JCS)**, is a standard for producing a deterministic, byte-for-byte reproducible representation of JSON data [1]. This is critical for cryptographic applications where the hash of a JSON document must be stable across different implementations and platforms.

## 2. Current Implementation Limitations

The current canonicalization function used across all generator scripts and tools is:

```python
def canonicalize_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(',', ':'))
```

This implementation handles the most common requirements (sorted keys, no whitespace, ASCII encoding), but it **does not** handle:

1.  **Unicode Normalization**: RFC 8785 requires that all Unicode strings be normalized to **NFC (Normalization Form C)** before serialization. The current implementation does not perform this normalization.

2.  **Number Formatting**: RFC 8785 specifies precise rules for number formatting:
    -   No leading zeros (except for `0.x`)
    -   No trailing zeros in the fractional part
    -   No exponent notation for numbers that can be represented exactly in decimal
    -   The current implementation relies on Python's default `json.dumps()` behavior, which is mostly correct but not guaranteed to be RFC 8785 compliant for all edge cases.

3.  **Escape Sequences**: RFC 8785 requires specific escape sequences for control characters (e.g., `\u0000` through `\u001F`). The current implementation uses `ensure_ascii=True`, which is mostly correct, but does not explicitly validate the escape sequences.

## 3. Proposed Solution: Use `canonicaljson` Library

Instead of maintaining a custom implementation, we will adopt the **`canonicaljson`** library, which provides full RFC 8785 compliance [2].

### 3.1. Library Selection

| Library | RFC 8785 Compliance | Python Support | Maintenance | License |
|---|---|---|---|---|
| `canonicaljson` | **Full** | Python 3.6+ | Active | Apache 2.0 |
| `jcs` | **Full** | Python 3.7+ | Active | MIT |
| Custom (current) | **Partial** | Python 3.11+ | Manual | N/A |

**Recommendation**: Use `canonicaljson`. It is the most widely used library for this purpose and has been battle-tested in production systems (e.g., Matrix.org [3]).

### 3.2. Installation

The library will be added to the project's dependencies:

```bash
pip3 install canonicaljson
```

### 3.3. API Usage

The `canonicaljson` library provides a simple API:

```python
import canonicaljson

# Serialize to canonical JSON (returns bytes)
canonical_bytes = canonicaljson.encode_canonical_json(obj)

# Convert to string (UTF-8)
canonical_str = canonical_bytes.decode('utf-8')
```

## 4. Unicode Normalization Strategy

RFC 8785 requires that all Unicode strings be normalized to **NFC (Normalization Form C)** [4]. This ensures that equivalent Unicode representations (e.g., precomposed vs. decomposed characters) produce the same canonical form.

**Example**:
- The string "é" can be represented as:
  - Precomposed: U+00E9 (LATIN SMALL LETTER E WITH ACUTE)
  - Decomposed: U+0065 U+0301 (LATIN SMALL LETTER E + COMBINING ACUTE ACCENT)

RFC 8785 requires that both representations be normalized to the precomposed form (U+00E9) before serialization.

The `canonicaljson` library handles this automatically by applying `unicodedata.normalize('NFC', string)` to all string values before serialization.

## 5. Number Formatting Policy

RFC 8785 specifies the following rules for number formatting:

1.  **No Leading Zeros**: Numbers MUST NOT have leading zeros (except for `0.x`).
    -   **Valid**: `0`, `123`, `0.5`
    -   **Invalid**: `01`, `00.5`

2.  **No Trailing Zeros**: Fractional numbers MUST NOT have trailing zeros.
    -   **Valid**: `1.5`, `2.0` → `2`
    -   **Invalid**: `1.50`, `2.00`

3.  **No Exponent Notation**: Numbers MUST be represented in decimal form unless they cannot be represented exactly (e.g., very large or very small numbers).
    -   **Valid**: `1000000`, `0.000001`
    -   **Invalid**: `1e6`, `1e-6` (unless necessary)

The `canonicaljson` library enforces these rules by using a custom JSON encoder that formats numbers according to RFC 8785.

## 6. Implementation Plan

### 6.1. Update All Scripts

All scripts that currently use the simplified `canonicalize_json()` function will be updated to use `canonicaljson`.

**Files to Update**:
-   `scripts/generators/generate_curriculum_snapshot.py`
-   `scripts/generators/generate_telemetry_snapshot.py`
-   `scripts/generators/generate_ledger_snapshot.py`
-   `scripts/generators/generate_attestation_snapshot.py`
-   `scripts/validation/governance_validator.py`
-   `scripts/evidence_pack.py`
-   `scripts/radars/curriculum_drift_radar.py`
-   `scripts/radars/telemetry_drift_radar.py`
-   `scripts/radars/ledger_drift_radar.py`
-   `scripts/radars/ht_triangle_drift_radar.py`

**Change Pattern**:

```diff
- def canonicalize_json(obj: Any) -> str:
-     return json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(',', ':'))
+ import canonicaljson
+ 
+ def canonicalize_json(obj: Any) -> str:
+     return canonicaljson.encode_canonical_json(obj).decode('utf-8')
```

### 6.2. Create Shared Utility Module

To avoid code duplication, a new shared utility module will be created:

**File**: `scripts/lib/canonicalization.py`

```python
"""
Shared canonicalization utilities for MathLedger governance.
"""

import canonicaljson
from typing import Any


def canonicalize_json(obj: Any) -> str:
    """
    Serialize an object to RFC 8785 canonical JSON.
    
    Args:
        obj: A JSON-serializable Python object.
    
    Returns:
        A UTF-8 string containing the canonical JSON representation.
    """
    canonical_bytes = canonicaljson.encode_canonical_json(obj)
    return canonical_bytes.decode('utf-8')
```

All scripts will then import from this module:

```python
from scripts.lib.canonicalization import canonicalize_json
```

## 7. Test Suite for Canonicalization Correctness

To ensure that the upgrade does not introduce regressions, a comprehensive test suite will be created.

**File**: `tests/test_canonicalization.py`

```python
import pytest
from scripts.lib.canonicalization import canonicalize_json


def test_sorted_keys():
    """Test that object keys are sorted."""
    obj = {"z": 1, "a": 2, "m": 3}
    result = canonicalize_json(obj)
    assert result == '{"a":2,"m":3,"z":1}'


def test_no_whitespace():
    """Test that there is no insignificant whitespace."""
    obj = {"key": "value"}
    result = canonicalize_json(obj)
    assert result == '{"key":"value"}'
    assert ' ' not in result


def test_unicode_normalization():
    """Test that Unicode strings are normalized to NFC."""
    # Decomposed form: U+0065 U+0301
    decomposed = "e\u0301"
    # Precomposed form: U+00E9
    precomposed = "\u00E9"
    
    obj1 = {"text": decomposed}
    obj2 = {"text": precomposed}
    
    result1 = canonicalize_json(obj1)
    result2 = canonicalize_json(obj2)
    
    # Both should produce the same canonical form
    assert result1 == result2


def test_number_formatting():
    """Test that numbers are formatted correctly."""
    obj = {"int": 123, "float": 1.5, "zero": 0.0}
    result = canonicalize_json(obj)
    # Note: 0.0 should be serialized as 0
    assert result == '{"float":1.5,"int":123,"zero":0}'


def test_escape_sequences():
    """Test that control characters are escaped."""
    obj = {"text": "line1\nline2\ttab"}
    result = canonicalize_json(obj)
    # Newline and tab should be escaped
    assert '\\n' in result
    assert '\\t' in result


def test_determinism():
    """Test that the same input always produces the same output."""
    obj = {"a": 1, "b": [2, 3], "c": {"nested": True}}
    result1 = canonicalize_json(obj)
    result2 = canonicalize_json(obj)
    assert result1 == result2
```

This test suite will be run in CI to ensure that all canonicalization is correct.

## 8. References

[1] **RFC 8785**: JSON Canonicalization Scheme (JCS), [https://tools.ietf.org/html/rfc8785](https://tools.ietf.org/html/rfc8785)

[2] **canonicaljson**: Python library for RFC 8785, [https://github.com/matrix-org/python-canonicaljson](https://github.com/matrix-org/python-canonicaljson)

[3] **Matrix.org**: Matrix uses canonical JSON for cryptographic signing, [https://matrix.org/docs/spec/appendices#canonical-json](https://matrix.org/docs/spec/appendices#canonical-json)

[4] **Unicode Normalization**: Unicode Standard Annex #15, [https://unicode.org/reports/tr15/](https://unicode.org/reports/tr15/)
