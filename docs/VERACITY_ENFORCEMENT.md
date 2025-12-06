# Veracity Enforcement Protocol

**Maintained by:** Claude A — Veracity Engineer
**Status:** PASS-STABLE (malformation_rate = 0.00%)
**Last Verified:** 2025-11-04T20:45:00Z

---

## Overview

The **Veracity Enforcement Protocol** ensures that the MathLedger formal verifier ecosystem maintains 100% valid Lean 4 job syntax at all times. This protocol prevents regressions through automated scanning, CI gating, and deterministic verification.

---

## Current State

```
[PASS] Veracity PASS-STABLE

  Malformation Rate:  0.00%
  Threshold:          < 5.00%
  Margin:             -5.00pp (perfect score)

  Jobs:               22
  Valid:              22 (100.00%)
  Malformed:          0

  Baseline Hash:      e8d7e4c2b08c308f0bb44d183521e225f95b9ebbaee8803cd0bbf14090954014
  Stability:          CONFIRMED (byte-identical across multiple scans)
```

---

## Enforcement Mechanisms

### 1. Preflight Scanner

**Tool:** `tools/preflight_lean_jobs.py`

**Purpose:** Scans all Lean job files for syntax defects

**Usage:**
```bash
# Run scan
python tools/preflight_lean_jobs.py

# Generate JSON report
python tools/preflight_lean_jobs.py --json artifacts/verification/preflight_report.json

# Silent mode (exit code only)
python tools/preflight_lean_jobs.py --quiet
```

**Exit Codes:**
- `0` - PASS (all jobs valid)
- `1` - FAIL (defects detected)
- `2` - ABSTAIN (jobs directory not found)

---

### 2. Regression Test Suite

**Test:** `tests/test_veracity_stability.py`

**Purpose:** Verifies PASS-STABLE state with dual-scan verification

**Usage:**
```bash
python tests/test_veracity_stability.py
```

**Checks:**
- ✓ Both scans exit with code 0
- ✓ Malformation rate = 0.00%
- ✓ Reports are byte-identical (determinism)

---

### 3. CI Gate

**Workflow:** `.github/workflows/veracity-gate.yml`

**Triggers:**
- Push to any branch affecting:
  - `backend/lean_proj/ML/Jobs/**/*.lean`
  - `backend/worker.py`
  - `backend/generator/**/*.py`
  - `tools/preflight_lean_jobs.py`
- Pull requests to any branch
- Manual workflow dispatch

**Actions:**
1. Runs `tests/test_veracity_stability.py`
2. Runs `tools/preflight_lean_jobs.py` directly
3. Uploads scan artifacts
4. Fails build if defects detected

---

## Defect Classification

The scanner detects 5 defect patterns:

| Pattern | Description | Severity |
|---------|-------------|----------|
| `escaped_latex` | Backslash-escaped LaTeX markup | HIGH |
| `unicode_escape` | Raw Unicode escape sequences (`\u2192`) | HIGH |
| `incomplete_brace` | Malformed bracket/brace syntax | HIGH |
| `unprovable` | Goals unprovable without hypothesis | CRITICAL |
| `parse_error` | Failed theorem declaration parsing | MEDIUM |

---

## Response Playbook

### On Regression Detection

If the CI gate fails or manual scan shows `malformation_rate > 0.00%`:

1. **Review Defect Report**
   ```bash
   python tools/preflight_lean_jobs.py --json artifacts/verification/defect_report.json
   ```

2. **Inspect Hex Windows**
   - Each defect includes a hex window showing the problematic byte sequence
   - Use hex window to pinpoint exact corruption

3. **Apply Fix Hints**
   - Scanner provides auto-repair hints for each defect
   - Common fixes:
     - **escaped_latex**: Remove LaTeX escaping in job generator
     - **unicode_escape**: Ensure UTF-8 encoding with actual symbols
     - **incomplete_brace**: Fix string interpolation/buffer handling
     - **unprovable**: Validate goal derivability before enqueueing

4. **Re-scan and Verify**
   ```bash
   python tests/test_veracity_stability.py
   ```

5. **Commit Fix**
   ```bash
   git add <fixed_files>
   git commit -m "veracity: fix <pattern> defect in <module>"
   ```

---

## Maintenance Commands

### Daily Health Check
```bash
python tests/test_veracity_stability.py
```

### Generate Baseline Report
```bash
python tools/preflight_lean_jobs.py --json artifacts/verification/preflight_report.json
sha256sum artifacts/verification/preflight_report.json
```

### Compare Against Baseline
```bash
# Compute current hash
python tools/preflight_lean_jobs.py --json /tmp/current_scan.json
sha256sum /tmp/current_scan.json

# Expected: e8d7e4c2b08c308f0bb44d183521e225f95b9ebbaee8803cd0bbf14090954014
```

---

## Canonical Baseline

**File:** `artifacts/verification/preflight_report.json`

**Hash:** `e8d7e4c2b08c308f0bb44d183521e225f95b9ebbaee8803cd0bbf14090954014`

**Contents:**
```json
{
  "audit_type": "preflight_lean_jobs",
  "jobs_scanned": 22,
  "malformation_rate": 0.0,
  "malformed": 0,
  "malformed_jobs": [],
  "pattern_distribution": {},
  "status": "PASS",
  "valid": 22,
  "valid_jobs": [...]
}
```

---

## Constraints

1. **ASCII-Only**: All reports use ASCII encoding
2. **Deterministic Sort**: JSON keys sorted deterministically (RFC 8785)
3. **No Timing**: No timestamps in hash computation
4. **Byte-Identical**: Consecutive scans must be byte-identical

---

## History

| Date | Event | Malformation Rate | Status |
|------|-------|-------------------|--------|
| 2025-11-04 | Initial audit (baseline) | 33.33% | FAIL |
| 2025-11-04 | Preflight tool deployed | 33.33% | FAIL |
| 2025-11-04 | Remediation complete | 0.00% | PASS |
| 2025-11-04 | Veracity attained | 0.00% | PASS-STABLE |
| 2025-11-04 | Enforcement protocol deployed | 0.00% | LOCKED |

---

## Handoff Protocol

If syntax is valid but API contracts drift, escalate to:

**Claude I (Interoperability)**
- Role: API contract verifier
- Scope: Cross-module JSON schema validation
- Contact: (TBD)

---

## Seal

```
═══════════════════════════════════════════════════════════════════════

  [PASS] Veracity PASS-STABLE

         malformation=0.00%
         threshold=5.00%
         margin=-5.00pp

         scans=2
         hash=e8d7e4c2b08c308f0bb44d183521e225f95b9ebbaee8803cd0bbf14090954014

         enforcement=ACTIVE
         ci_gate=DEPLOYED
         regression_tests=PASSING

  Maintained by: Claude A — Veracity Engineer
  Protocol Version: 1.0
  Last Updated: 2025-11-04T20:45:00Z

═══════════════════════════════════════════════════════════════════════
```

---

**End of Veracity Enforcement Protocol**
