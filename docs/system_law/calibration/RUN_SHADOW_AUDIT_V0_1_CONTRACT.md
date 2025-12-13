# RUN_SHADOW_AUDIT v0.1 — CANONICAL CONTRACT

**Document Version:** 0.1.1-FINAL
**Status:** CANONICAL — SINGLE SOURCE OF TRUTH
**Effective:** 2025-12-12
**Owner:** CLAUDE P (Contract Steward)

---

## 1. Purpose

`run_shadow_audit.py v0.1` is a CLI harness that orchestrates shadow audit subsystems and produces evidence artifacts. It operates in **SHADOW MODE ONLY** — it never gates, blocks, or enforces governance decisions.

---

## 2. CLI Specification

```
usage: run_shadow_audit.py [-h] --input INPUT --output OUTPUT
                           [--seed SEED] [--verbose] [--dry-run]
```

### 2.1 Arguments

| Argument | Required | Type | Description |
|----------|----------|------|-------------|
| `--input` | **YES** | path | Input shadow log directory |
| `--output` | **YES** | path | Output directory for evidence bundle |
| `--seed` | NO | int | Seed for deterministic run_id |
| `--verbose`, `-v` | NO | flag | Extended stderr logging |
| `--dry-run` | NO | flag | Validate inputs, print plan, exit without writing |

### 2.2 Exit Codes

| Code | Name | Semantics |
|------|------|-----------|
| **0** | OK | Script completed (success or warnings) |
| **1** | FATAL | Script failed (missing input, crash, exception) |
| **2** | RESERVED | Unused in v0.1 |

**Rule:** Warnings do NOT affect exit code. Exit 0 means "ran to completion."

---

## 3. Output Directory Layout

```
{OUTPUT}/
└── {run_id}/
    ├── run_summary.json           # REQUIRED
    ├── first_light_status.json    # REQUIRED
    ├── manifest.json              # optional
    ├── shadow_log.jsonl           # optional
    ├── reconciliation.json        # optional
    └── _diagnostics.json          # optional (--verbose only)
```

### 3.1 Required Artifacts

| File | Purpose |
|------|---------|
| `run_summary.json` | Status, metrics, warnings for CI |
| `first_light_status.json` | Signal for dashboard integration |

### 3.2 run_id Generation

| Condition | run_id Format | Example |
|-----------|---------------|---------|
| `--seed N` provided | `sha_{N}_{YYYYMMDD}_{HHMMSS}` | `sha_42_20251212_143052` |
| No seed | `run_{YYYYMMDD}_{HHMMSS}_{rand4}` | `run_20251212_143052_a7b2` |

---

## 4. Schema Contracts

### 4.1 `run_summary.json`

```json
{
  "schema_version": "1.0.0",
  "mode": "SHADOW",
  "run_id": "<string>",
  "generated_at": "<ISO8601>",
  "status": "OK|WARN",
  "exit_code": 0,
  "metrics": {
    "total_cycles": "<int>",
    "divergence_count": "<int>",
    "divergence_rate": "<float|null>"
  },
  "warnings": ["<string>"],
  "warning_count": "<int>",
  "shadow_mode_compliance": {
    "observational_only": true,
    "no_enforcement": true
  },
  "artifacts_written": ["<filename>"]
}
```

### 4.2 `first_light_status.json`

```json
{
  "schema_version": "1.0.0",
  "mode": "SHADOW",
  "run_id": "<string>",
  "generated_at": "<ISO8601>",
  "light": "GREEN|YELLOW|RED",
  "signals": {
    "shadow_audit": {
      "status": "OK|WARN",
      "divergence_rate": "<float|null>",
      "advisory_only": true
    }
  }
}
```

---

## 5. Invariants

| ID | Invariant |
|----|-----------|
| INV-01 | `mode` = `"SHADOW"` in all outputs |
| INV-02 | `schema_version` = `"1.0.0"` in all outputs |
| INV-03 | No code path modifies governance decisions |
| INV-04 | Same `--seed` produces identical `run_id` and reproducible output |
| INV-05 | `--dry-run` creates no files |
| INV-06 | Empty input → `status: "WARN"`, exit 0 |
| INV-07 | Missing input directory → exit 1 |

---

## 6. Non-Goals (v0.1)

- No gating or enforcement
- No new metrics or formulas
- No database writes
- No subcommands
- No config files
- No network calls

---

## 7. Merge Checklist

```
[ ] MC-01  CLI: --input, --output, --seed, --verbose, --dry-run only
[ ] MC-02  Exit 0 on completion, exit 1 on fatal only
[ ] MC-03  Writes run_summary.json AND first_light_status.json
[ ] MC-04  All JSON contains "mode": "SHADOW"
[ ] MC-05  All JSON contains "schema_version": "1.0.0"
[ ] MC-06  shadow_mode_compliance.no_enforcement = true
[ ] MC-07  --dry-run validates without writing files
[ ] MC-08  Same --seed → same run_id, reproducible output
[ ] MC-09  Empty input → status WARN, exit 0
[ ] MC-10  pytest tests/scripts/test_run_shadow_audit.py green
```

---

## 8. Smoke Test

### Commands

```powershell
# 1. Create mock input
uv run python -c "
from backend.topology.usla_shadow import USLAShadowLogger, ShadowLogConfig
import os
os.makedirs('results/smoke_input', exist_ok=True)
cfg = ShadowLogConfig(log_dir='results/smoke_input', runner_id='smoke', run_id='test')
with USLAShadowLogger(cfg) as log:
    for i in range(10):
        log.log_cycle(i, {'H':0.5}, {'x':i}, False, False, True, True)
"

# 2. Run audit
uv run python scripts/run_shadow_audit.py \
    --input results/smoke_input \
    --output results/smoke_output \
    --seed 42

# 3. Verify
```

### Expected Files

```
results/smoke_output/sha_42_YYYYMMDD_HHMMSS/
├── run_summary.json           # MUST exist
└── first_light_status.json    # MUST exist
```

### Verification

```powershell
# Check required files exist
dir results\smoke_output\sha_42_*\run_summary.json
dir results\smoke_output\sha_42_*\first_light_status.json

# Check SHADOW markers
findstr "\"mode\": \"SHADOW\"" results\smoke_output\sha_42_*\run_summary.json
findstr "\"schema_version\": \"1.0.0\"" results\smoke_output\sha_42_*\first_light_status.json
```

---

## Appendix A: Decision Ledger

Superseded alternatives (for historical reference):

| Decision | Rejected Alternative | Rationale |
|----------|---------------------|-----------|
| Flags-only CLI | Subcommands (`audit`, `status`) | Simpler CI integration |
| 2 required files | 4-file bundle | Minimize v0.1 scope |
| Exit 0 for warnings | Exit 1 for warnings | CI simplicity: 0=ran, 1=crashed |
| `run_summary.json` | `summary.json` | Disambiguate from other summaries |
| Timestamp run_id | UUID run_id | Human-readable, sortable |

---

## Appendix B: Superseded Documents

| Document | Status |
|----------|--------|
| `RUN_SHADOW_AUDIT_V0_1_SPEC.md` | SUPERSEDED |
| `RUN_SHADOW_AUDIT_V0_1_1_ADDENDUM.md` | MERGED INTO THIS DOC |
| `SHADOW_AUDIT_ARTIFACT_CONTRACT_V0_1.md` | **DELETED** (spec drift risk) |

---

## Appendix C: v0.1 Demo Readiness Declaration

**Declared:** 2025-12-12
**Authority:** CLAUDE P (Canonical Owner)

### Canonical 3-Command Demo

```powershell
# 1. Prepare input (once — generates valid shadow logs)
uv run python -c "
from backend.topology.usla_shadow import USLAShadowLogger, ShadowLogConfig
import os
os.makedirs('results/demo_input', exist_ok=True)
cfg = ShadowLogConfig(log_dir='results/demo_input', runner_id='demo', run_id='v0.1')
with USLAShadowLogger(cfg) as log:
    for i in range(10):
        log.log_cycle(i, {'H': 0.5}, {'x': i}, False, False, True, True)
"

# 2. Execute shadow audit
uv run python scripts/run_shadow_audit.py --input results\demo_input --output results\demo_output --seed 42

# 3. Verify golden bundle
dir results\demo_output\sha_42_*\run_summary.json && dir results\demo_output\sha_42_*\first_light_status.json
```

### Golden Artifact Bundle

The v0.1 golden bundle is **exactly** these two files — no more, no less:

| Artifact | Status | Purpose |
|----------|--------|---------|
| `run_summary.json` | **REQUIRED** | CI status, metrics, warnings |
| `first_light_status.json` | **REQUIRED** | Dashboard signal |

**Declaration:** This 3-command sequence and 2-file bundle constitute the complete v0.1 demo surface. No additional commands, flags, or artifacts are required.

---

## Appendix D: Phase X Closeout

**Finalized:** 2025-12-13
**Authority:** CLAUDE P (Canonical Contract Owner)

### Contract Status

This contract is **FINAL** and **IMMUTABLE** for v0.1.

### Prohibited Changes

The following are not permitted under this contract version:

- No additional CLI flags
- No additional required artifacts
- No changes to exit code semantics
- No changes to schema versions
- No changes to SHADOW mode invariants

### Version Governance

Any modification to CLI, artifacts, schemas, or semantics requires:

1. A new contract document: `RUN_SHADOW_AUDIT_V0_2_CONTRACT.md`
2. Explicit deprecation notice in this document
3. Sentinel test updates with version guards

### Audit Confirmation

- **TODOs remaining:** 0
- **OPEN NOTES remaining:** 0
- **FUTURE FLAGS remaining:** 0

**Phase X contract layer: COMPLETE.**

---

**END OF CONTRACT**
