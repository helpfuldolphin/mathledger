# SHADOW AUDIT Test Strategy & CI Discipline (CANONICAL)
## `run_shadow_audit.py` v0.1

**Author:** CLAUDE V (Gatekeeper)
**Status:** Engineering Ready
**Phase:** X (Shadow Mode Only)
**Last Updated:** 2025-12-12

---

## 1. Canonical Contract Reference

This test strategy enforces the **SINGLE SOURCE OF TRUTH**:
- `docs/system_law/calibration/RUN_SHADOW_AUDIT_V0_1_CONTRACT.md` (Claude P - CANONICAL)

### 1.1 CLI Specification (FROZEN)

```
usage: run_shadow_audit.py [-h] --input INPUT --output OUTPUT
                           [--seed SEED] [--verbose] [--dry-run]
```

| Flag | Required | Description |
|------|----------|-------------|
| `--input` | **YES** | Input shadow log directory |
| `--output` | **YES** | Output directory for evidence bundle |
| `--seed` | NO | Seed for deterministic run_id |
| `--verbose`, `-v` | NO | Extended stderr logging |
| `--dry-run` | NO | Validate inputs, print plan, exit without writing |

### 1.2 Exit Code Contract (FROZEN)

| Code | Name | Meaning | CI Behavior |
|------|------|---------|-------------|
| **0** | OK | Script completed (success or warnings) | PASS |
| **1** | FATAL | Script failed (missing input, crash, exception) | BLOCK |
| **2** | RESERVED | Unused in v0.1 | N/A |

**Rule:** Warnings do NOT affect exit code. Exit 0 means "ran to completion."

### 1.2 SHADOW MODE Contract (Phase X)

```
PHASE X SHADOW MODE CONTRACT

1. The USLA simulator NEVER modifies real governance decisions
2. Disagreements are LOGGED, not ACTED upon
3. No cycle is blocked or allowed based on simulator output
4. The simulator runs AFTER the real governance decision
5. All USLA state is written to shadow logs only

Violations of this contract require explicit Phase XI approval.
```

---

## 2. Minimal Test Matrix

### 2.1 Unit Tests (6 canonical)

| # | Test Name | Enforces | Files |
|---|-----------|----------|-------|
| U1 | `test_schema_version_fixed` | schema_version="1.0.0" immutable | `test_run_shadow_audit.py` |
| U2 | `test_mode_shadow_everywhere` | mode="SHADOW" in all outputs | `test_run_shadow_audit.py` |
| U3 | `test_exit_codes_per_contract` | Exit 0/1/2 per Claude P | `test_run_shadow_audit.py` |
| U4 | `test_determinism_flag_behavior` | --deterministic produces sorted JSON | `test_run_shadow_audit.py` |
| U5 | `test_graceful_degradation_no_fabrication` | Missing data -> schema_ok/warnings, NOT fake values | `test_run_shadow_audit.py` |
| U6 | `test_windows_safe_encoding` | UTF-8 writes, no BOM, no emojis | `test_run_shadow_audit.py` |

### 2.2 Integration Tests (8 canonical)

| # | Test Name | Enforces | Files |
|---|-----------|----------|-------|
| I1 | `test_e2e_basic_run` | --input/--output produces run_summary.json | `test_shadow_audit_e2e.py` |
| I2 | `test_e2e_deterministic_seed` | Same --seed → same run_id (INV-04) | `test_shadow_audit_e2e.py` |
| I3 | `test_e2e_dry_run_no_files` | --dry-run creates no files (INV-05) | `test_shadow_audit_e2e.py` |
| I4 | `test_e2e_missing_input_exit_1` | Missing input → exit 1 (INV-07) | `test_shadow_audit_e2e.py` |
| I5 | `test_e2e_empty_input_exit_0_warn` | Empty input → exit 0, status=WARN (INV-06) | `test_shadow_audit_e2e.py` |
| I6 | `test_e2e_required_artifacts` | run_summary.json + first_light_status.json (MC-03) | `test_shadow_audit_e2e.py` |
| I7 | `test_e2e_help_output` | --help shows canonical flags | `test_shadow_audit_e2e.py` |
| I8 | `test_e2e_verbose_mode` | --verbose produces extended output | `test_shadow_audit_e2e.py` |

### 2.3 Sentinel Tests (Merge Blockers)

| # | Test Name | Enforces | Files |
|---|-----------|----------|-------|
| S1 | `test_sentinel_shadow_audit_contract` | Full contract compliance | `test_shadow_audit_sentinel.py` |
| S2 | `test_sentinel_no_enforcement_true` | No enforcement=True in source | `test_shadow_audit_sentinel.py` |
| S3 | `test_sentinel_utf8_encoding` | Windows-safe encoding | `test_shadow_audit_sentinel.py` |
| S4 | `test_sentinel_frozen_cli_flags` | Canonical CLI flags only | `test_shadow_audit_e2e.py` |

---

## 3. Determinism Policy

### 3.1 MUST Match (Byte-Stable with --deterministic)

| Field | Rule |
|-------|------|
| `mode` | Always "SHADOW" |
| `schema_version` | Fixed "1.0.0" |
| `enforcement` | Always false |
| `artifacts[]` | Sorted by name |
| `divergences[]` | Sorted by (cycle, field) |

### 3.2 IGNORED (Non-Deterministic)

| Field | Reason |
|-------|--------|
| `timestamp` | Runtime |
| `created_at` | Runtime |
| `bundle_id` | Contains timestamp |
| `generated_at` | Runtime |

### 3.3 Comparison Function

```python
IGNORED_KEYS = {"timestamp", "created_at", "bundle_id", "generated_at", "started_at", "ended_at"}

def normalize(obj):
    if isinstance(obj, dict):
        return {k: normalize(v) for k, v in sorted(obj.items()) if k not in IGNORED_KEYS}
    if isinstance(obj, list):
        normalized = [normalize(item) for item in obj]
        if normalized and all(isinstance(x, dict) for x in normalized):
            return sorted(normalized, key=lambda x: json.dumps(x, sort_keys=True))
        return normalized
    return obj
```

---

## 4. Graceful Degradation Policy

### 4.1 NO Value Fabrication

When data is missing, the script MUST:
- Report `schema_ok: true` if schema structure is valid
- Add `advisory_warnings: ["<description>"]` for missing optional data
- NEVER invent placeholder values (no `0.0`, `"unknown"`, or similar)

### 4.2 Missing Required Artifacts

```
Exit Code: 1
Output: Human-readable error listing missing artifacts
Evidence Pack: NOT created
```

### 4.3 Missing Optional Artifacts

```
Exit Code: 0
Output: "Optional: X/Y present (missing: <list>)"
Evidence Pack: Created with available artifacts
```

---

## 5. Windows-Safe Encoding Requirements

### 5.1 File Writes

```python
# CORRECT: Explicit UTF-8, no BOM
with open(path, "w", encoding="utf-8") as f:
    f.write(content)

# WRONG: Default encoding (varies by platform)
with open(path, "w") as f:
    f.write(content)
```

### 5.2 JSON Serialization

```python
# CORRECT: ensure_ascii=False for Unicode, explicit encoding
json.dump(data, f, ensure_ascii=False, indent=2)

# WRONG: ensure_ascii=True (escapes Unicode unnecessarily)
json.dump(data, f, ensure_ascii=True)
```

### 5.3 No Emojis Unless Explicit

Script output MUST NOT contain emojis unless:
1. User explicitly enables via `--emoji` flag, OR
2. Operating in a context where encoding is guaranteed

---

## 6. CI Workflow (Non-Gating)

### 6.1 Alignment with Phase X Workflows

Consistent with `usla-shadow-gate.yml`:
- Uses `USLA_SHADOW_ENABLED` environment variable
- Same trigger patterns (topology/health paths)
- Same artifact retention (30 days)

### 6.2 Jobs Structure

```yaml
jobs:
  unit-tests:
    continue-on-error: false  # Script failures BLOCK

  integration-tests:
    continue-on-error: false  # Script failures BLOCK
    needs: unit-tests

  guardrail-tests:
    continue-on-error: false  # Contract violations BLOCK
    needs: unit-tests

  nightly-audit:
    continue-on-error: true   # Audit RESULTS are advisory
    if: schedule || manual
```

### 6.3 Artifact Upload

```yaml
- uses: actions/upload-artifact@v4
  if: always()
  with:
    name: shadow-audit-${{ github.run_number }}
    path: results/shadow_audit/
    retention-days: 30
```

---

## 7. Smoke-Test Readiness Checklist

### 7.1 Pre-Flight Verification

```powershell
# Verify paths exist before testing
Test-Path scripts/run_shadow_audit.py
Test-Path tests/scripts/test_run_shadow_audit.py
Test-Path tests/integration/test_shadow_audit_e2e.py
Test-Path tests/ci/test_shadow_audit_guardrails.py
Test-Path .github/workflows/shadow-audit-gate.yml
```

### 7.2 Unit Test Smoke

```powershell
# Run minimal unit tests
uv run pytest tests/scripts/test_run_shadow_audit.py -v --tb=short -k "schema_version or mode_shadow"
```

### 7.3 Integration Smoke

```powershell
# Create mock input
mkdir -p results/smoke_input
echo '{"_header":true,"mode":"SHADOW","schema_version":"1.0.0"}' > results/smoke_input/shadow_log_test.jsonl

# Run with canonical flags
uv run python scripts/run_shadow_audit.py --input results/smoke_input --output results/smoke_test --seed 42

# Verify outputs
Test-Path results/smoke_test/sha_42_*/run_summary.json
Get-Content results/smoke_test/sha_42_*/run_summary.json | Select-String '"mode": "SHADOW"'
```

### 7.4 Exit Code Verification

```powershell
# Exit 0: Success (completed, possibly with warnings)
uv run python scripts/run_shadow_audit.py --input results/smoke_input --output results/out1; echo "Exit: $LASTEXITCODE"
# Expected: Exit: 0

# Exit 1: Missing input directory (FATAL)
uv run python scripts/run_shadow_audit.py --input /nonexistent/ --output results/out2; echo "Exit: $LASTEXITCODE"
# Expected: Exit: 1

# Exit 0: Dry-run validates without writing
uv run python scripts/run_shadow_audit.py --input results/smoke_input --output results/dry --dry-run; echo "Exit: $LASTEXITCODE"
# Expected: Exit: 0
```

### 7.5 Encoding Verification

```powershell
# Check no BOM in output files
$bytes = [System.IO.File]::ReadAllBytes("results/smoke_test/evidence_pack/manifest.json")
if ($bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
    Write-Error "BOM detected!"
} else {
    Write-Host "No BOM - OK"
}
```

### 7.6 Determinism Verification

```powershell
# Run twice with same --seed, compare
uv run python scripts/run_shadow_audit.py --input results/smoke_input --output results/det1 --seed 42
uv run python scripts/run_shadow_audit.py --input results/smoke_input --output results/det2 --seed 42

# Verify same run_id
dir results/det1  # Should show sha_42_*
dir results/det2  # Should show sha_42_* with same suffix

# Compare summary (ignoring timestamps)
uv run python -c "
import json, hashlib
from pathlib import Path
def h(p):
    d = json.loads(p.read_text())
    d.pop('generated_at', None)
    d.pop('timestamp', None)
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]
r1 = list(Path('results/det1').glob('sha_42_*'))[0] / 'run_summary.json'
r2 = list(Path('results/det2').glob('sha_42_*'))[0] / 'run_summary.json'
assert h(r1) == h(r2), 'Determinism failed'
print('Determinism OK')
"
```

---

## 8. Summary Table

| Requirement | Test Coverage | CI Enforcement |
|-------------|---------------|----------------|
| schema_version="1.0.0" (INV-02) | U1, S1 | Unit + Sentinel |
| mode="SHADOW" everywhere (INV-01) | U2, S1 | Guardrails + Sentinel |
| Exit codes 0/1 (INV-06, INV-07) | U3, I4, I5 | Integration |
| Determinism (INV-04) | U4, I2 | Unit + Integration |
| No fabrication | U5 | Unit tests |
| UTF-8 encoding | U6, S3 | Unit + Sentinel |
| Canonical CLI flags | S1, S4 | Sentinel (BLOCKS MERGE) |
| Basic run E2E | I1 | Integration |
| Required artifacts (MC-03) | I6 | Integration |
| --dry-run no files (INV-05) | I3 | Integration |

---

## 9. Sentinel Gate & Post-Merge Unskip

### 9.1 Sentinel Tests

Located at: `tests/ci/test_shadow_audit_sentinel.py`

These tests guard against non-compliant implementations:
- `test_sentinel_shadow_audit_contract` - Core contract compliance
- `test_sentinel_no_enforcement_true` - No enforcement=True in source
- `test_sentinel_utf8_encoding` - Windows-safe encoding patterns

**Behavior:**
- SKIP if script doesn't exist (pre-implementation, expected)
- FAIL if script exists but violates contract (blocks merge)
- PASS if script exists and is compliant

### 9.2 Post-Merge: Unskip E2E Tests

Once Claude S lands PR-1 (`scripts/run_shadow_audit.py`):

```bash
# Run all tests including integration (no longer skipped)
uv run pytest tests/scripts/test_run_shadow_audit.py \
             tests/integration/test_shadow_audit_e2e.py \
             tests/ci/test_shadow_audit_sentinel.py \
             tests/ci/test_shadow_audit_guardrails.py \
             -v --tb=short

# Run only integration tests
uv run pytest tests/integration/test_shadow_audit_e2e.py -v -m integration

# Run sentinel + integration together
uv run pytest tests/ci/test_shadow_audit_sentinel.py \
             tests/integration/test_shadow_audit_e2e.py \
             -v --tb=short
```

### 9.3 Pytest Markers Reference

```bash
# Unit tests only (always run)
uv run pytest -m unit tests/scripts/test_run_shadow_audit.py

# Integration tests (run post-merge)
uv run pytest -m integration tests/integration/test_shadow_audit_e2e.py

# Determinism tests
uv run pytest -m determinism

# All shadow audit tests
uv run pytest tests/scripts/test_run_shadow_audit.py \
             tests/integration/test_shadow_audit_e2e.py \
             tests/ci/test_shadow_audit_sentinel.py \
             tests/ci/test_shadow_audit_guardrails.py
```

---

## 10. Expected Test Counts

### 10.1 State 1: Script Missing (Pre-Implementation)

```bash
uv run pytest tests/ci/test_shadow_audit_guardrails.py \
             tests/ci/test_shadow_audit_sentinel.py \
             tests/integration/test_shadow_audit_e2e.py \
             -v --tb=short
```

**Expected:**
- **9 PASSED** (guardrail tests - contract mocks)
- **11 SKIPPED** (sentinels + E2E - script not deployed yet)
- **0 FAILED**

### 10.2 State 2: Script Exists, Non-Canonical CLI (Transition)

Current state: Script uses `--p3-dir`/`--p4-dir` instead of canonical `--input`/`--output`.

```bash
uv run pytest tests/ci/test_shadow_audit_guardrails.py \
             tests/ci/test_shadow_audit_sentinel.py \
             tests/integration/test_shadow_audit_e2e.py \
             -v --tb=short
```

**Expected:**
- **11 PASSED** (guardrails + source checks)
- **8 SKIPPED** (E2E - awaiting canonical CLI)
- **2 FAILED** (sentinels - correctly catching non-canonical CLI) → **BLOCKS MERGE**

### 10.3 State 3: Script Compliant (Post-Implementation)

Once script uses canonical CLI (`--input`/`--output`):

```bash
uv run pytest tests/ci/test_shadow_audit_guardrails.py \
             tests/ci/test_shadow_audit_sentinel.py \
             tests/integration/test_shadow_audit_e2e.py \
             -v --tb=short
```

**Expected:**
- **21 PASSED** (all tests)
- **0 SKIPPED**
- **0 FAILED**

---

## 11. Reality Lock

| Role | Owner | Scope |
|------|-------|-------|
| Source of Truth | Claude P | `docs/system_law/calibration/RUN_SHADOW_AUDIT_V0_1_CONTRACT.md` |
| Implementation | Claude S | `scripts/run_shadow_audit.py` (PR-1) |
| Tests & CI | Claude V | This doc + test files |

**No parallel implementations. Single source of truth.**
