# RTTS Provenance + Warning Normal Form Smoke Checklist

**Phase X P5.2 VALIDATE Stage**
**Date**: 2025-12-12
**Schema Version**: 1.2.0
**Status**: SHADOW MODE (LOGGED_ONLY)

---

## Overview

This checklist verifies the RTTS provenance tracking (`extraction_source`) and warning normal form enhancements.

### Changes Implemented

1. **Extraction Source Provenance** - Tracks where RTTS data came from
2. **Frozen Enums** - Immutable enum values for driver categories and extraction sources
3. **Warning Normal Form** - Deterministic single-line warning format

---

## Frozen Enums

### ExtractionSource
| Value | Description |
|-------|-------------|
| `MANIFEST_REFERENCE` | Loaded via manifest reference with sha256 integrity check |
| `DIRECT_DISCOVERY` | Loaded via direct file discovery from run directory |
| `MISSING` | No RTTS validation data found |

### DriverCategory
| Value | Description |
|-------|-------------|
| `STATISTICAL` | MOCK-001, MOCK-002, MOCK-005, MOCK-006, MOCK-007, MOCK-008 |
| `CORRELATION` | MOCK-003, MOCK-004 |
| `CONTINUITY` | MOCK-009, MOCK-010 |
| `UNKNOWN` | No flags or unrecognized codes |

---

## Warning Normal Form

**Format:**
```
RTTS {overall_status}: {violation_count} violations | driver={top_driver_category} | flags=[{top3_mock_codes}]
```

**Examples:**
```
RTTS WARN: 3 violations | driver=STATISTICAL | flags=[MOCK-001, MOCK-003]
RTTS CRITICAL: 10 violations | driver=CONTINUITY | flags=[MOCK-009, MOCK-010]
RTTS WARN: 1 violations | driver=UNKNOWN | flags=[none]
```

---

## Smoke Tests

### 1. Extraction Source Provenance

| Test | Condition | Expected extraction_source |
|------|-----------|---------------------------|
| Manifest preferred | Manifest ref + run dir both exist | `MANIFEST_REFERENCE` |
| Direct fallback | No manifest ref, run dir exists | `DIRECT_DISCOVERY` |
| Missing | No file anywhere | `MISSING` |

```bash
# Test extraction_source tracking
uv run python -c "
from backend.health.rtts_status_adapter import (
    extract_rtts_status_signal,
    ExtractionSource,
)

# Test DIRECT_DISCOVERY
rtts = {'overall_status': 'OK', 'warning_count': 0, 'mock_detection_flags': []}
signal = extract_rtts_status_signal(rtts)
assert signal['extraction_source'] == ExtractionSource.DIRECT_DISCOVERY
print('PASS: DIRECT_DISCOVERY default')

# Test MISSING
signal = extract_rtts_status_signal(None)
assert signal['extraction_source'] == ExtractionSource.MISSING
print('PASS: MISSING when None')

# Test explicit MANIFEST_REFERENCE
signal = extract_rtts_status_signal(rtts, extraction_source=ExtractionSource.MANIFEST_REFERENCE)
assert signal['extraction_source'] == ExtractionSource.MANIFEST_REFERENCE
print('PASS: MANIFEST_REFERENCE explicit')
"
```

### 2. Frozen Enum Values

```bash
# Verify frozen enum values
uv run python -c "
from backend.health.rtts_status_adapter import (
    ExtractionSource,
    DriverCategory,
    VALID_DRIVER_CATEGORIES,
)

# ExtractionSource values
assert ExtractionSource.MANIFEST_REFERENCE == 'MANIFEST_REFERENCE'
assert ExtractionSource.DIRECT_DISCOVERY == 'DIRECT_DISCOVERY'
assert ExtractionSource.MISSING == 'MISSING'
print('PASS: ExtractionSource enum values frozen')

# DriverCategory values
assert DriverCategory.STATISTICAL == 'STATISTICAL'
assert DriverCategory.CORRELATION == 'CORRELATION'
assert DriverCategory.CONTINUITY == 'CONTINUITY'
assert DriverCategory.UNKNOWN == 'UNKNOWN'
print('PASS: DriverCategory enum values frozen')

# Valid categories set
assert len(VALID_DRIVER_CATEGORIES) == 4
print('PASS: VALID_DRIVER_CATEGORIES has 4 members')
"
```

### 3. Warning Normal Form

| Test | Input | Expected Warning |
|------|-------|------------------|
| WARN with flags | status=WARN, count=3, driver=STATISTICAL, flags=[MOCK-001,MOCK-003] | `RTTS WARN: 3 violations \| driver=STATISTICAL \| flags=[MOCK-001, MOCK-003]` |
| CRITICAL | status=CRITICAL, count=10, driver=CONTINUITY | Full normal form |
| No flags | status=WARN, flags=[] | `flags=[none]` |
| OK status | status=OK | `None` (no warning) |

```bash
# Test warning normal form
uv run python -c "
from backend.health.rtts_status_adapter import (
    generate_rtts_warning,
    DriverCategory,
)

# Test normal form structure
signal = {
    'available': True,
    'overall_status': 'WARN',
    'violation_count': 3,
    'top3_mock_codes': ['MOCK-001', 'MOCK-003'],
    'top_driver_category': DriverCategory.STATISTICAL,
}
warning = generate_rtts_warning(signal)
expected = 'RTTS WARN: 3 violations | driver=STATISTICAL | flags=[MOCK-001, MOCK-003]'
assert warning == expected, f'Got: {warning}'
print('PASS: Warning normal form correct')

# Test no flags shows 'none'
signal['top3_mock_codes'] = []
signal['top_driver_category'] = DriverCategory.UNKNOWN
warning = generate_rtts_warning(signal)
assert 'flags=[none]' in warning
print('PASS: Empty flags shows [none]')

# Test OK status returns None
signal['overall_status'] = 'OK'
warning = generate_rtts_warning(signal)
assert warning is None
print('PASS: OK status returns None')
"
```

### 4. Warning Determinism

```bash
# Test warning string determinism
uv run python -c "
from backend.health.rtts_status_adapter import (
    generate_rtts_warning,
    DriverCategory,
)

signal = {
    'available': True,
    'overall_status': 'WARN',
    'violation_count': 5,
    'top3_mock_codes': ['MOCK-001', 'MOCK-003', 'MOCK-005'],
    'top_driver_category': DriverCategory.STATISTICAL,
}

# Generate 100 times
warnings = [generate_rtts_warning(signal) for _ in range(100)]

# All must be identical
first = warnings[0]
for i, w in enumerate(warnings):
    assert w == first, f'Mismatch at iteration {i}'

print('PASS: Warning string deterministic across 100 iterations')
print(f'Value: {first}')
"
```

### 5. Full Pipeline Test

```bash
# Test full extraction pipeline with provenance
uv run python -c "
from backend.health.rtts_status_adapter import (
    extract_rtts_status_for_first_light,
    ExtractionSource,
    DriverCategory,
)
from pathlib import Path
import tempfile
import json

with tempfile.TemporaryDirectory() as tmpdir:
    run_dir = Path(tmpdir)

    # Create RTTS validation file
    rtts_data = {
        'overall_status': 'WARN',
        'warning_count': 4,
        'mock_detection_flags': ['MOCK-001', 'MOCK-002', 'MOCK-005', 'MOCK-003'],
    }
    (run_dir / 'rtts_validation.json').write_text(json.dumps(rtts_data))

    # Extract
    signal = extract_rtts_status_for_first_light(run_dir)

    # Verify all fields
    assert signal['available'] is True
    assert signal['overall_status'] == 'WARN'
    assert signal['violation_count'] == 4
    assert signal['extraction_source'] == ExtractionSource.DIRECT_DISCOVERY
    assert signal['top_driver_category'] == DriverCategory.STATISTICAL
    assert signal['top3_mock_codes'] == ['MOCK-001', 'MOCK-002', 'MOCK-003']
    assert signal['mode'] == 'SHADOW'
    assert signal['action'] == 'LOGGED_ONLY'

    print('PASS: Full pipeline extraction correct')
    print(f'extraction_source: {signal[\"extraction_source\"]}')
    print(f'top_driver_category: {signal[\"top_driver_category\"]}')
"
```

---

## Unit Test Summary

Run all tests:
```bash
uv run python -m pytest tests/scripts/test_generate_first_light_status_rtts_signal.py -v
```

| Test Class | Count | Description |
|------------|-------|-------------|
| TestExtractRTTSStatusSignal | 5 | Basic signal extraction + extraction_source |
| TestTopDriverComputation | 6 | Driver category (frozen enum) |
| TestGenerateRTTSWarning | 7 | Warning normal form + determinism |
| TestLoadRTTSValidationForStatus | 3 | Direct file loading |
| TestManifestReferenceLoading | 5 | Manifest reference + integrity |
| TestExtractRTTSStatusForFirstLight | 4 | Full pipeline + provenance |
| **Total** | **30** | |

---

## Schema Changes (v1.1.0 → v1.2.0)

| Field | v1.1.0 | v1.2.0 |
|-------|--------|--------|
| `extraction_source` | N/A | `MANIFEST_REFERENCE \| DIRECT_DISCOVERY \| MISSING` |
| `top_driver_category` | `statistical \| correlation \| continuity \| null` | `STATISTICAL \| CORRELATION \| CONTINUITY \| UNKNOWN` |

---

## SHADOW MODE Compliance

All outputs verified to include:
- `mode: "SHADOW"`
- `action: "LOGGED_ONLY"`

No gating or enforcement is performed. All operations are observational only.

---

## Golden Bundle Entry

### Golden Bundle Minimal Tree

```
evidence_pack/
├── manifest.json
│   └── governance:
│       └── rtts_validation_reference:
│           ├── path: "governance/rtts_validation.json"
│           ├── sha256: "a1b2c3..."
│           ├── mode: "SHADOW"
│           └── action: "LOGGED_ONLY"
├── governance/
│   └── rtts_validation.json          ← integrity-checked file
└── first_light_status.json
    └── signals:
        └── rtts:
            ├── extraction_source: "MANIFEST_REFERENCE"
            ├── overall_status: "WARN"
            ├── top_driver_category: "STATISTICAL"
            └── ...
```

### One-Liner Smoke Verification (Determinism Check)

```bash
uv run python -c "
from backend.health.rtts_status_adapter import extract_rtts_status_for_first_light, compute_rtts_file_sha256, ExtractionSource
from pathlib import Path; import tempfile, json
def strip_time(d): return {k:v for k,v in d.items() if 'time' not in k.lower() and 'timestamp' not in k.lower()}
with tempfile.TemporaryDirectory() as t:
    p,g,r=Path(t),Path(t)/'governance',Path(t)/'run'; g.mkdir(); r.mkdir()
    (g/'rtts_validation.json').write_text(json.dumps({'overall_status':'WARN','warning_count':3,'mock_detection_flags':['MOCK-001','MOCK-005']}))
    sha=compute_rtts_file_sha256(g/'rtts_validation.json')
    ref={'path':'governance/rtts_validation.json','sha256':sha}
    s1,s2=strip_time(extract_rtts_status_for_first_light(r,ref,p)),strip_time(extract_rtts_status_for_first_light(r,ref,p))
    assert s1==s2,'DETERMINISM FAIL: runs differ'
    assert s1['extraction_source']==ExtractionSource.MANIFEST_REFERENCE
    print(f'SMOKE: repeat_match={s1==s2} src={s1[\"extraction_source\"]} status={s1[\"overall_status\"]}')"
```

**Expected output:** `SMOKE: repeat_match=True src=MANIFEST_REFERENCE status=WARN`

> **Note:** Determinism check is content-level only; not correctness. This confirms identical output across two runs with the same input—no semantic validation is performed.
>
> `strip_time` removes keys containing `'time'` or `'timestamp'` only; all other fields must match byte-for-byte.

---

### Example Signal JSON (with extraction_source)

```json
{
  "schema_version": "1.2.0",
  "mode": "SHADOW",
  "action": "LOGGED_ONLY",
  "available": true,
  "overall_status": "WARN",
  "violation_count": 4,
  "top3_mock_codes": ["MOCK-001", "MOCK-002", "MOCK-003"],
  "top_driver_category": "STATISTICAL",
  "top_driver_codes_top3": ["MOCK-001", "MOCK-002", "MOCK-005"],
  "extraction_source": "MANIFEST_REFERENCE"
}
```

### Warning Normal-Form Example Line

```
RTTS WARN: 4 violations | driver=STATISTICAL | flags=[MOCK-001, MOCK-002, MOCK-003]
```

### 3-Step Smoke Command Sequence

**Step 1: Setup test environment**
```bash
uv run python -c "
from pathlib import Path
import tempfile, json

# Create test directory structure
tmpdir = tempfile.mkdtemp()
pack_dir = Path(tmpdir)
gov_dir = pack_dir / 'governance'
gov_dir.mkdir()
run_dir = pack_dir / 'run'
run_dir.mkdir()

# Create RTTS validation file
rtts_data = {'overall_status': 'WARN', 'warning_count': 4, 'mock_detection_flags': ['MOCK-001', 'MOCK-002', 'MOCK-005', 'MOCK-003']}
rtts_path = gov_dir / 'rtts_validation.json'
rtts_path.write_text(json.dumps(rtts_data))

# Create fallback file with different status
fallback_data = {'overall_status': 'CRITICAL', 'warning_count': 10, 'mock_detection_flags': ['MOCK-009']}
(run_dir / 'rtts_validation.json').write_text(json.dumps(fallback_data))

print(f'TEST_DIR={tmpdir}')
print('Setup complete: governance/ has WARN, run/ has CRITICAL')
"
```

**Step 2: SHA256 matches → MANIFEST_REFERENCE**
```bash
uv run python -c "
from backend.health.rtts_status_adapter import (
    extract_rtts_status_for_first_light,
    compute_rtts_file_sha256,
    generate_rtts_warning,
    ExtractionSource,
)
from pathlib import Path
import tempfile, json

with tempfile.TemporaryDirectory() as tmpdir:
    pack_dir = Path(tmpdir)
    gov_dir = pack_dir / 'governance'
    gov_dir.mkdir()
    run_dir = pack_dir / 'run'
    run_dir.mkdir()

    # Manifest file (WARN)
    rtts_manifest = {'overall_status': 'WARN', 'warning_count': 4, 'mock_detection_flags': ['MOCK-001', 'MOCK-002', 'MOCK-005', 'MOCK-003']}
    rtts_path = gov_dir / 'rtts_validation.json'
    rtts_path.write_text(json.dumps(rtts_manifest))

    # Fallback file (CRITICAL)
    (run_dir / 'rtts_validation.json').write_text(json.dumps({'overall_status': 'CRITICAL', 'warning_count': 10, 'mock_detection_flags': ['MOCK-009']}))

    # VALID sha256
    valid_sha = compute_rtts_file_sha256(rtts_path)
    manifest_ref = {'path': 'governance/rtts_validation.json', 'sha256': valid_sha}

    signal = extract_rtts_status_for_first_light(run_dir, manifest_ref, pack_dir)
    warning = generate_rtts_warning(signal)

    print('=== SHA256 MATCH ===')
    print(f'extraction_source: {signal[\"extraction_source\"]}')
    print(f'overall_status:    {signal[\"overall_status\"]}')
    print(f'top_driver:        {signal[\"top_driver_category\"]}')
    print(f'warning:           {warning}')
    assert signal['extraction_source'] == ExtractionSource.MANIFEST_REFERENCE
    assert signal['overall_status'] == 'WARN'  # From manifest, not fallback
    print('PASS: Loaded from MANIFEST_REFERENCE')
"
```

**Step 3: SHA256 mismatch → soft fail + DIRECT_DISCOVERY fallback**
```bash
uv run python -c "
from backend.health.rtts_status_adapter import (
    extract_rtts_status_for_first_light,
    generate_rtts_warning,
    ExtractionSource,
)
from pathlib import Path
import tempfile, json

with tempfile.TemporaryDirectory() as tmpdir:
    pack_dir = Path(tmpdir)
    gov_dir = pack_dir / 'governance'
    gov_dir.mkdir()
    run_dir = pack_dir / 'run'
    run_dir.mkdir()

    # Manifest file (WARN)
    (gov_dir / 'rtts_validation.json').write_text(json.dumps({'overall_status': 'WARN', 'warning_count': 4, 'mock_detection_flags': ['MOCK-001']}))

    # Fallback file (CRITICAL)
    (run_dir / 'rtts_validation.json').write_text(json.dumps({'overall_status': 'CRITICAL', 'warning_count': 10, 'mock_detection_flags': ['MOCK-009', 'MOCK-010']}))

    # INVALID sha256 - triggers soft fail
    manifest_ref = {'path': 'governance/rtts_validation.json', 'sha256': '0' * 64}

    signal = extract_rtts_status_for_first_light(run_dir, manifest_ref, pack_dir)
    warning = generate_rtts_warning(signal)

    print('=== SHA256 MISMATCH (soft fail) ===')
    print(f'extraction_source: {signal[\"extraction_source\"]}')
    print(f'overall_status:    {signal[\"overall_status\"]}')
    print(f'top_driver:        {signal[\"top_driver_category\"]}')
    print(f'warning:           {warning}')
    print('')
    print('NOTE: Integrity check failed silently (SHADOW MODE)')
    print('NOTE: Fell back to run_dir discovery (CRITICAL status)')
    assert signal['extraction_source'] == ExtractionSource.DIRECT_DISCOVERY
    assert signal['overall_status'] == 'CRITICAL'  # From fallback, not manifest
    print('PASS: Soft fail with DIRECT_DISCOVERY fallback')
"
```

### Integrity Check Behavior Summary

| Condition | Behavior | extraction_source | Notes |
|-----------|----------|-------------------|-------|
| sha256 matches | Load manifest file | `MANIFEST_REFERENCE` | Integrity verified |
| sha256 mismatch | **Soft fail** → try run_dir | `DIRECT_DISCOVERY` | No exception, no gating |
| sha256 mismatch + no fallback | **Soft fail** → unavailable | `MISSING` | SHADOW MODE: silent |

---

## Sign-off

- [ ] All 30 unit tests pass
- [ ] ExtractionSource enum values frozen (uppercase strings)
- [ ] DriverCategory enum values frozen (uppercase strings)
- [ ] Warning normal form deterministic
- [ ] extraction_source tracked correctly for all paths
- [ ] SHADOW MODE contract maintained
