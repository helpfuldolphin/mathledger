> **SUPERSEDED** (2025-12-12): CLI shape, exit codes, and artifact layout are governed by the canonical contract.
> Canonical reference: `docs/system_law/calibration/RUN_SHADOW_AUDIT_V0_1_CONTRACT.md`
> This doc is retained for UX exploration context only — do not implement CLI from this file.

---

# run_shadow_audit.py v0.1 — UX Specification

**Canonical Reference:** `docs/system_law/calibration/RUN_SHADOW_AUDIT_V0_1_CONTRACT.md`
**Status:** SUPERSEDED for CLI/exit semantics; retained for UX context

---

## 1. Quickstart (3 Commands)

```bash
# 1. Dry-run: validate inputs, print plan, exit without writing
python scripts/run_shadow_audit.py --input results/shadow_logs/ --output results/audit/ --dry-run

# 2. Minimal run: produce evidence bundle
python scripts/run_shadow_audit.py --input results/shadow_logs/ --output results/audit/

# 3. Deterministic run: reproducible output for CI
python scripts/run_shadow_audit.py --input results/shadow_logs/ --output results/audit/ --seed 42
```

---

## 2. CLI Flags (FROZEN)

```
usage: run_shadow_audit.py [-h] --input INPUT --output OUTPUT
                           [--seed SEED] [--verbose] [--dry-run]
```

| Flag | Required | Type | Description |
|------|----------|------|-------------|
| `--input` | **YES** | path | Input shadow log directory |
| `--output` | **YES** | path | Output directory for evidence bundle |
| `--seed` | NO | int | Seed for deterministic run_id |
| `--verbose`, `-v` | NO | flag | Extended stderr logging |
| `--dry-run` | NO | flag | Validate inputs, print plan, exit without writing |

---

## 3. Exit Codes (FROZEN)

| Code | Name | Semantics |
|------|------|-----------|
| **0** | OK | Script completed (success or warnings) |
| **1** | FATAL | Script failed (missing input, crash, exception) |
| **2** | RESERVED | Unused in v0.1 |

**Rule:** Warnings do NOT affect exit code. Exit 0 means "ran to completion."

---

## 4. Required Artifacts (FROZEN)

| File | Required | Purpose |
|------|----------|---------|
| `run_summary.json` | **YES** | Status, metrics, warnings for CI |
| `first_light_status.json` | **YES** | Signal for dashboard integration |
| `manifest.json` | NO | Optional bundle inventory |
| `shadow_log.jsonl` | NO | Optional per-cycle records |
| `reconciliation.json` | NO | Optional metric consistency |
| `_diagnostics.json` | NO | Optional, `--verbose` only |

---

## 5. Output Layout (FROZEN)

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

**run_id generation:**
- With `--seed N`: `sha_{N}_{YYYYMMDD}_{HHMMSS}` (e.g., `sha_42_20251212_143052`)
- Without seed: `run_{YYYYMMDD}_{HHMMSS}_{rand4}` (e.g., `run_20251212_143052_a7b2`)

---

## 6. Common Mistakes

1. **Looking for output in `{OUTPUT}/` root**
   Outputs are in `{OUTPUT}/{run_id}/`, not `{OUTPUT}/`. The run_id subdirectory is always created.

2. **Expecting exit 1 for warnings**
   Warnings do NOT change exit code. Exit 0 = "ran to completion" (even with warnings). Exit 1 = crash/fatal only.

3. **Thinking `--seed` is optional for determinism**
   Without `--seed`, run_id includes random bytes. For reproducible CI, always pass `--seed N`.

4. **Providing wrong input path**
   `--input` must contain `shadow_log*.jsonl` from `USLAShadowLogger`. If missing, run `usla_first_light_harness.py` first.

5. **Expecting subcommands, prompts, or config files**
   The CLI is FLAGS-ONLY, non-interactive, no config files. Missing `--input` or `--output` = immediate exit 1.

---

## 7. Smoke-Test Checklist

### Setup
```powershell
# Generate mock shadow logs
uv run python -c "
from backend.topology.usla_shadow import USLAShadowLogger, ShadowLogConfig
import os
os.makedirs('results/smoke_input', exist_ok=True)
cfg = ShadowLogConfig(log_dir='results/smoke_input', runner_id='smoke', run_id='test')
with USLAShadowLogger(cfg) as log:
    for i in range(10):
        log.log_cycle(i, {'H':0.5}, {'x':i}, False, False, True, True)
"
```

### Test Commands
```powershell
# Dry-run (exit 0, no files)
uv run python scripts/run_shadow_audit.py --input results/smoke_input --output results/smoke_out --dry-run
echo Exit: %ERRORLEVEL%  # Expected: 0

# Deterministic run
uv run python scripts/run_shadow_audit.py --input results/smoke_input --output results/smoke_out --seed 42
echo Exit: %ERRORLEVEL%  # Expected: 0

# Fatal error (missing input)
uv run python scripts/run_shadow_audit.py --input /nonexistent --output results/smoke_out
echo Exit: %ERRORLEVEL%  # Expected: 1
```

### Verify Required Artifacts
```powershell
# These two files MUST exist
dir results\smoke_out\sha_42_*\run_summary.json
dir results\smoke_out\sha_42_*\first_light_status.json

# Check SHADOW markers
findstr "\"mode\": \"SHADOW\"" results\smoke_out\sha_42_*\run_summary.json
findstr "\"schema_version\": \"1.0.0\"" results\smoke_out\sha_42_*\first_light_status.json
```

### Verify Determinism
```powershell
# Run twice with same seed
uv run python scripts/run_shadow_audit.py --input results/smoke_input --output results/det1 --seed 99
uv run python scripts/run_shadow_audit.py --input results/smoke_input --output results/det2 --seed 99

# Same run_id produced
dir results\det1\sha_99_*
dir results\det2\sha_99_*
```

---

## 8. UX Freeze Statement

**UX IS FROZEN FOR v0.1.**

| Frozen | Items |
|--------|-------|
| CLI Flags | `--input`, `--output`, `--seed`, `--verbose`, `--dry-run` |
| Exit Codes | 0 (OK), 1 (FATAL), 2 (RESERVED) |
| Required Artifacts | `run_summary.json`, `first_light_status.json` |
| Output Layout | `{OUTPUT}/{run_id}/` |
| Schema Version | `"1.0.0"` |
| Mode Marker | `mode: "SHADOW"` |

Changes require v0.2+ spec and `MIGRATION.md`.

---

*Canonical: `docs/system_law/calibration/RUN_SHADOW_AUDIT_V0_1_CONTRACT.md`*
