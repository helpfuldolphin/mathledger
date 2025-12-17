# RTTS Top-Driver + Manifest Hook Smoke Checklist

**Phase X P5.2 VALIDATE Stage**
**Date**: 2025-12-12
**Status**: SHADOW MODE (LOGGED_ONLY)

---

## Overview

This checklist verifies the RTTS "Top-Driver" explainability extension and evidence pack manifest hook integration.

### Components Tested

1. **Top-Driver Explainability** (`backend/health/rtts_status_adapter.py`)
   - `top_driver_category`: statistical | correlation | continuity
   - `top_driver_codes_top3`: Top 3 MOCK codes from driver category (sorted)

2. **Manifest Hook** (`scripts/build_first_light_evidence_pack.py`)
   - Copies `rtts_validation.json` to `governance/` directory
   - Records reference with path + sha256 in manifest

3. **Status Generator Integration** (`scripts/generate_first_light_status.py`)
   - Prefers manifest reference when available
   - Validates sha256 integrity before loading
   - Falls back to direct file discovery

---

## Smoke Tests

### 1. Top-Driver Computation

| Test | Command | Expected |
|------|---------|----------|
| Statistical dominance | Run with MOCK-001,002,005,003 | `top_driver_category="statistical"` |
| Correlation dominance | Run with MOCK-003,004,001 | `top_driver_category="correlation"` |
| Continuity dominance | Run with MOCK-009,010,001 | `top_driver_category="continuity"` |
| Tie-breaking | Run with MOCK-001,009 (1 each) | `top_driver_category="continuity"` (alphabetic) |
| Determinism | Run same input 5× | Identical output each time |

```bash
# Quick validation
uv run python -c "
from backend.health.rtts_status_adapter import extract_rtts_status_signal
rtts = {'overall_status': 'WARN', 'warning_count': 4, 'mock_detection_flags': ['MOCK-001', 'MOCK-002', 'MOCK-005', 'MOCK-003']}
signal = extract_rtts_status_signal(rtts)
print(f'top_driver_category: {signal[\"top_driver_category\"]}')
print(f'top_driver_codes_top3: {signal[\"top_driver_codes_top3\"]}')
assert signal['top_driver_category'] == 'statistical'
assert signal['top_driver_codes_top3'] == ['MOCK-001', 'MOCK-002', 'MOCK-005']
print('PASS: Top-driver computation correct')
"
```

### 2. Manifest Reference Hook

| Test | Condition | Expected |
|------|-----------|----------|
| Hook triggers | rtts_validation.json exists in P4 run | File copied to `governance/rtts_validation.json` |
| Manifest entry | After evidence pack build | `manifest.governance.rtts_validation_reference` contains path + sha256 |
| SHA256 recorded | After hook runs | Hash matches file content |
| Mode recorded | After hook runs | `mode="SHADOW"`, `action="LOGGED_ONLY"` |

```bash
# Verify manifest structure (example)
uv run python -c "
import json
from pathlib import Path
# Replace with actual evidence pack path
pack_dir = Path('results/first_light_evidence')
manifest_path = pack_dir / 'manifest.json'
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
    rtts_ref = manifest.get('governance', {}).get('rtts_validation_reference')
    if rtts_ref:
        print(f'path: {rtts_ref.get(\"path\")}')
        print(f'sha256: {rtts_ref.get(\"sha256\")}')
        print(f'mode: {rtts_ref.get(\"mode\")}')
        assert rtts_ref.get('mode') == 'SHADOW'
        print('PASS: Manifest hook structure correct')
    else:
        print('INFO: No rtts_validation_reference in manifest (rtts_validation.json may not exist)')
else:
    print('SKIP: No evidence pack manifest found')
"
```

### 3. Integrity Verification

| Test | Condition | Expected |
|------|-----------|----------|
| Valid sha256 | Hash matches file | Data loaded successfully |
| Invalid sha256 | Hash mismatch | Returns None (no error) |
| Missing sha256 | No hash in reference | Loads without check |
| Missing file | File doesn't exist | Returns None (no error) |

```bash
# Test integrity check
uv run python -c "
from backend.health.rtts_status_adapter import load_rtts_validation_from_manifest_reference, compute_rtts_file_sha256
from pathlib import Path
import tempfile, json

with tempfile.TemporaryDirectory() as tmpdir:
    pack_dir = Path(tmpdir)
    gov_dir = pack_dir / 'governance'
    gov_dir.mkdir()

    # Create test file
    rtts_data = {'overall_status': 'WARN'}
    rtts_path = gov_dir / 'rtts_validation.json'
    rtts_path.write_text(json.dumps(rtts_data))

    # Test valid hash
    valid_hash = compute_rtts_file_sha256(rtts_path)
    ref_valid = {'path': 'governance/rtts_validation.json', 'sha256': valid_hash}
    result = load_rtts_validation_from_manifest_reference(ref_valid, pack_dir)
    assert result is not None and result['overall_status'] == 'WARN'
    print('PASS: Valid sha256 loads data')

    # Test invalid hash
    ref_invalid = {'path': 'governance/rtts_validation.json', 'sha256': '0' * 64}
    result = load_rtts_validation_from_manifest_reference(ref_invalid, pack_dir)
    assert result is None
    print('PASS: Invalid sha256 returns None (SHADOW MODE)')
"
```

### 4. Status Generator Integration

| Test | Condition | Expected |
|------|-----------|----------|
| Manifest preferred | Both manifest ref and direct file exist | Uses manifest version |
| Fallback works | No manifest ref, direct file exists | Uses direct discovery |
| Unavailable | No file anywhere | `available=False`, `overall_status="UNKNOWN"` |

```bash
# Test full pipeline
uv run python -c "
from backend.health.rtts_status_adapter import extract_rtts_status_for_first_light, compute_rtts_file_sha256
from pathlib import Path
import tempfile, json

with tempfile.TemporaryDirectory() as tmpdir:
    pack_dir = Path(tmpdir)
    gov_dir = pack_dir / 'governance'
    gov_dir.mkdir()

    # Manifest version: WARN
    rtts_manifest = {'overall_status': 'WARN', 'warning_count': 3, 'mock_detection_flags': ['MOCK-001']}
    manifest_path = gov_dir / 'rtts_validation.json'
    manifest_path.write_text(json.dumps(rtts_manifest))

    # Direct version: CRITICAL
    run_dir = Path(tmpdir) / 'run'
    run_dir.mkdir()
    rtts_direct = {'overall_status': 'CRITICAL', 'warning_count': 10, 'mock_detection_flags': ['MOCK-009']}
    (run_dir / 'rtts_validation.json').write_text(json.dumps(rtts_direct))

    # Test: manifest reference should be preferred
    manifest_ref = {'path': 'governance/rtts_validation.json', 'sha256': compute_rtts_file_sha256(manifest_path)}
    signal = extract_rtts_status_for_first_light(run_dir, manifest_ref, pack_dir)
    assert signal['overall_status'] == 'WARN', f'Expected WARN (manifest), got {signal[\"overall_status\"]}'
    print('PASS: Manifest reference preferred over direct discovery')
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
| TestExtractRTTSStatusSignal | 4 | Basic signal extraction |
| TestTopDriverComputation | 5 | Top-driver category/codes |
| TestGenerateRTTSWarning | 5 | Warning generation |
| TestLoadRTTSValidationForStatus | 3 | Direct file loading |
| TestManifestReferenceLoading | 5 | Manifest reference + integrity |
| TestExtractRTTSStatusForFirstLight | 3 | Full pipeline |
| **Total** | **25** | |

---

## SHADOW MODE Compliance

All outputs verified to include:
- `mode: "SHADOW"`
- `action: "LOGGED_ONLY"`

No gating or enforcement is performed. All failures (integrity, missing files) return None silently per SHADOW MODE contract.

---

## MOCK Code Category Reference

| Code | Category | Detection Criterion |
|------|----------|---------------------|
| MOCK-001 | statistical | Var(H) below threshold |
| MOCK-002 | statistical | Var(rho) below threshold |
| MOCK-003 | correlation | Low correlation \|Cor(H,rho)\| |
| MOCK-004 | correlation | High correlation \|Cor(H,rho)\| |
| MOCK-005 | statistical | ACF below threshold |
| MOCK-006 | statistical | ACF above threshold |
| MOCK-007 | statistical | Kurtosis below threshold |
| MOCK-008 | statistical | Kurtosis above threshold |
| MOCK-009 | continuity | Jump in H (max delta) |
| MOCK-010 | continuity | Discrete rho values |

---

## Sign-off

- [ ] All 25 unit tests pass
- [ ] Top-driver computation deterministic
- [ ] Manifest hook copies file to governance/
- [ ] SHA256 integrity check works (valid → load, invalid → None)
- [ ] Status generator prefers manifest reference
- [ ] SHADOW MODE contract maintained (mode="SHADOW", action="LOGGED_ONLY")
