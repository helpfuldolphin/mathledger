# Shadow Audit Contract↔Reality Reconciliation

**Document Version:** 1.0.0-FINAL
**Status:** RECONCILIATION COMPLETE
**Date:** 2025-12-12
**Owner:** CLAUDE P (Contract Steward)

---

## 1. Contract↔Reality Reconciliation Table

### 1.1 CLI Flags Comparison

| Flag | CONTRACT (Canonical) | IMPLEMENTATION (Actual) | SENTINEL (Tests) | STATUS |
|------|----------------------|------------------------|------------------|--------|
| Input (required) | `--input INPUT` | `--p3-dir`, `--p4-dir` (both optional, default values) | `--input` (required) | **MISMATCH** |
| Output (required) | `--output OUTPUT` | `--output-dir` (optional, default) | `--output` (required) | **MISMATCH** |
| Seed (optional) | `--seed SEED` | `--seed SEED` | `--seed` | **MATCH** |
| Verbose (optional) | `--verbose`, `-v` | `--verbose`, `-v` | `--verbose` | **MATCH** |
| Dry-run (optional) | `--dry-run` | `--dry-run` | `--dry-run` | **MATCH** |
| Deterministic | NOT IN CONTRACT | `--deterministic` | **FORBIDDEN** | **VIOLATION** |
| P4 Harness | NOT IN CONTRACT | `--run-p4-harness` | — | **EXTRA** |
| P4 Cycles | NOT IN CONTRACT | `--p4-cycles` | — | **EXTRA** |
| Telemetry Adapter | NOT IN CONTRACT | `--telemetry-adapter` | — | **EXTRA** |
| Evidence Pack | NOT IN CONTRACT | `--build-evidence-pack` | — | **EXTRA** |
| Alignment View | NOT IN CONTRACT | `--alignment-view` | — | **EXTRA** |
| P5 Replay Logs | NOT IN CONTRACT | `--p5-replay-logs` | — | **EXTRA** |

### 1.2 Exit Code Comparison

| Code | CONTRACT Semantics | IMPLEMENTATION Semantics | STATUS |
|------|-------------------|-------------------------|--------|
| **0** | OK (script completed, success or warnings) | Always returns 0 (except crash) | **MATCH** |
| **1** | FATAL (missing input, crash, exception) | Uncaught exception only | **PARTIAL** |
| **2** | RESERVED (unused in v0.1) | Not implemented | **MATCH** |

**Note:** Contract says exit 1 on "missing input" — implementation exits 0 with `status: "fail"` in JSON.

### 1.3 run_id Format Comparison

| Condition | CONTRACT Format | IMPLEMENTATION Format | STATUS |
|-----------|-----------------|----------------------|--------|
| With `--seed N` | `sha_{N}_{YYYYMMDD}_{HHMMSS}` | `shadow_audit_{YYYYMMDD}_{HHMMSS}_seed{N}` | **MISMATCH** |
| Without seed | `run_{YYYYMMDD}_{HHMMSS}_{rand4}` | `shadow_audit_{YYYYMMDD}_{HHMMSS}_noseed` | **MISMATCH** |

### 1.4 Output File Comparison

| File | CONTRACT (Required) | IMPLEMENTATION | STATUS |
|------|---------------------|----------------|--------|
| `run_summary.json` | **REQUIRED** | Writes `run_summary.json` | **MATCH** |
| `first_light_status.json` | **REQUIRED** | Writes via subprocess to status script | **CONDITIONAL** |
| `manifest.json` | optional | Via `--build-evidence-pack` | **MATCH** |

### 1.5 Schema Markers Comparison

| Marker | CONTRACT | IMPLEMENTATION | STATUS |
|--------|----------|----------------|--------|
| `"mode": "SHADOW"` | Required in all outputs | Present in `run_summary` | **MATCH** |
| `"schema_version": "1.0.0"` | Required in all outputs | Present (`SCHEMA_VERSION = "1.0.0"`) | **MATCH** |
| `"enforcement": false` | Implied (no_enforcement) | `"enforcement": False` | **MATCH** |

---

## 2. Decisive Ruling

### 2.1 Canonical Source of Truth

**RULING: The CONTRACT is canonical. The implementation MUST be updated to match.**

Rationale:
1. The contract was explicitly frozen as "SINGLE SOURCE OF TRUTH" (line 4 of contract)
2. The sentinel tests enforce the contract, not the implementation
3. The current implementation is a "shadow orchestrator" that exceeds v0.1 scope

### 2.2 Required Changes

| Item | Current State | Required State | Action |
|------|---------------|----------------|--------|
| CLI flags | `--p3-dir`, `--p4-dir`, `--output-dir` | `--input`, `--output` | **REFACTOR** |
| `--deterministic` | Present | Remove | **DELETE** |
| Extra flags | 6 non-canonical flags | None | **DELETE or hide** |
| run_id format | `shadow_audit_*_seed{N}` | `sha_{N}_{YYYYMMDD}_{HHMMSS}` | **FIX** |
| Exit 1 on missing input | Exits 0 with fail status | Must exit 1 | **FIX** |

### 2.3 Allowed Extensions (Backward-Compatible)

The following MAY remain if they don't conflict with canonical flags:
- Internal orchestration stages (as implementation detail)
- Additional optional output files beyond the 2 required

---

## 3. Merge Gate Checklist

### 3.1 Pre-Flight Commands

```powershell
# 1. Run sentinel test (MUST PASS after fix)
uv run pytest tests/ci/test_shadow_audit_sentinel.py -v

# 2. Verify --help shows canonical flags only
uv run python scripts/run_shadow_audit.py --help 2>&1 | findstr /i "input output seed verbose dry-run"

# 3. Verify forbidden flags are absent
uv run python scripts/run_shadow_audit.py --help 2>&1 | findstr /i "p3-dir p4-dir output-dir deterministic"
# Expected: no matches (empty output)

# 4. Test dry-run with canonical flags
$tmpIn = New-TemporaryFile | ForEach-Object { Remove-Item $_; New-Item -ItemType Directory -Path $_ }
$tmpOut = New-TemporaryFile | ForEach-Object { Remove-Item $_; $_.FullName }
uv run python scripts/run_shadow_audit.py --input $tmpIn --output $tmpOut --dry-run
echo "Exit code: $LASTEXITCODE"
# Expected: Exit code: 0

# 5. Test missing input exits 1
$missing = "C:\nonexistent_path_$(Get-Random)"
uv run python scripts/run_shadow_audit.py --input $missing --output $tmpOut
echo "Exit code: $LASTEXITCODE"
# Expected: Exit code: 1 (per contract INV-07)

# 6. Test empty input exits 0 with WARN
uv run python scripts/run_shadow_audit.py --input $tmpIn --output $tmpOut --seed 42
echo "Exit code: $LASTEXITCODE"
# Expected: Exit code: 0
# Check run_summary.json has "status": "WARN"
```

### 3.2 Output Verification

```powershell
# After successful run with --seed 42:
$runDir = Get-ChildItem "$tmpOut\sha_42_*" | Select-Object -First 1

# Check required files exist
Test-Path "$runDir\run_summary.json"        # MUST be True
Test-Path "$runDir\first_light_status.json" # MUST be True

# Check SHADOW markers
Get-Content "$runDir\run_summary.json" | findstr '"mode": "SHADOW"'
Get-Content "$runDir\run_summary.json" | findstr '"schema_version": "1.0.0"'
```

### 3.3 Final Gate Criteria

| ID | Criterion | Command | Expected |
|----|-----------|---------|----------|
| MG-01 | Sentinel test passes | `pytest tests/ci/test_shadow_audit_sentinel.py` | PASS |
| MG-02 | `--input` flag exists | `--help \| findstr input` | Match |
| MG-03 | `--output` flag exists | `--help \| findstr output` | Match |
| MG-04 | `--p3-dir` flag absent | `--help \| findstr p3-dir` | No match |
| MG-05 | `--deterministic` absent | `--help \| findstr deterministic` | No match |
| MG-06 | Missing input → exit 1 | Run with nonexistent path | Exit 1 |
| MG-07 | Dry-run → exit 0, no files | Run with `--dry-run` | Exit 0, no output |
| MG-08 | run_id format correct | Check output directory name | `sha_42_*` |
| MG-09 | Both required files exist | `Test-Path` | True |
| MG-10 | SHADOW markers present | `findstr` | Match |

---

## 4. Summary

| Category | Contract | Implementation | Verdict |
|----------|----------|----------------|---------|
| CLI Interface | Flags-only, minimal | Orchestrator-style, extended | **NON-COMPLIANT** |
| Exit Codes | 0/1/2 semantics | 0-always except crash | **PARTIAL** |
| Output Files | 2 required | 1 guaranteed, 1 conditional | **PARTIAL** |
| Schema Markers | Correct | Correct | **COMPLIANT** |

**Overall Status: NON-COMPLIANT — Requires refactor before merge**

---

## 5. Decision Ledger

| Decision | Made By | Date | Rationale |
|----------|---------|------|-----------|
| Contract is canonical | CLAUDE P | 2025-12-12 | Contract was frozen first; sentinel enforces it |
| Implementation must change | CLAUDE P | 2025-12-12 | v0.1 scope creep must be reversed |
| Orchestrator features deferred to v0.2 | CLAUDE P | 2025-12-12 | Non-goal for v0.1 per contract §6 |

---

**CANON LOCK READY**

*Pending: Implementation refactor to match contract CLI specification.*
