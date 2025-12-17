# First Light CI Integration Design Stub

**Document Version:** 1.0.0
**Status:** Design specification only — no CI files modified
**Mode:** SHADOW (advisory only, no gating)

---

## Overview

This document specifies how First Light verification should be integrated into CI pipelines. All checks remain **SHADOW MODE** — they report status but do not gate merges or deployments.

---

## Pipeline Architecture

### Fast Path: Per-PR Checks

**Trigger:** Every pull request to `master` or `integrate/*` branches

**Job:** `first-light-shadow-compliance`

**Duration:** ~2 minutes

**Steps:**
1. Install dependencies (`uv sync`)
2. Run SHADOW MODE compliance tests
3. Export status to `first_light_status.json`

**Command:**
```bash
uv run python -m pytest tests/integration/test_shadow_mode_compliance.py -v --tb=short
```

**Success criteria:** 14/14 tests pass

**Failure behavior:** Report failure in PR status check. **Do not block merge** (SHADOW MODE).

---

### Slow Path: Nightly Full Verification

**Trigger:** Scheduled nightly (e.g., 02:00 UTC) or manual dispatch

**Job:** `first-light-golden-run`

**Duration:** ~5-10 minutes

**Steps:**
1. Install dependencies (`uv sync`)
2. Run P3 harness (1000 cycles, seed=42)
3. Run P4 harness (1000 cycles, seed=42)
4. Build evidence pack
5. Run determinism check (100 cycles x 4 runs)
6. Run SHADOW MODE compliance tests
7. Export comprehensive status to `first_light_status.json`
8. Archive evidence pack as build artifact

**Commands:**
```bash
# P3 Golden Run
uv run python scripts/usla_first_light_harness.py \
    --cycles 1000 --seed 42 --slice arithmetic_simple \
    --runner-type u2 --tau-0 0.20 --window-size 50 \
    --output-dir results/ci/p3

# P4 Golden Run
uv run python scripts/usla_first_light_p4_harness.py \
    --cycles 1000 --seed 42 --slice arithmetic_simple \
    --runner-type u2 --tau-0 0.20 \
    --output-dir results/ci/p4

# Build Evidence Pack
uv run python scripts/build_first_light_evidence_pack.py \
    --p3-dir results/ci/p3 \
    --p4-dir results/ci/p4 \
    --output-dir results/ci/evidence_pack

# Determinism Check
uv run python scripts/verify_first_light_determinism.py

# Compliance Tests
uv run python -m pytest tests/integration/test_shadow_mode_compliance.py -v
```

**Success criteria:**
- P3 harness exits 0, produces 6 artifacts
- P4 harness exits 0, produces 6 artifacts
- Evidence pack has 14 files with valid manifest
- Determinism check reports PASSED
- Compliance tests: 14/14 pass

**Failure behavior:** Report failure in CI dashboard. Send notification (Slack/Discord webhook). **Do not block anything** (SHADOW MODE).

---

## Status Export Schema

After each CI run, export `first_light_status.json`:

```json
{
  "schema_version": "1.2.0",
  "timestamp": "2025-12-11T12:00:00Z",
  "mode": "SHADOW",
  "pipeline": "nightly",
  "git_sha": "abc123...",
  "git_branch": "master",
  "telemetry_source": "mock",
  "proof_snapshot_present": false,

  "shadow_mode_ok": true,
  "determinism_ok": true,
  "p3_harness_ok": true,
  "p4_harness_ok": true,
  "evidence_pack_ok": true,

  "last_run_id": {
    "p3": "fl_20251211_020000_seed42",
    "p4": "p4_20251211_020015"
  },

  "metrics_snapshot": {
    "p3_success_rate": 0.852,
    "p3_omega_occupancy": 0.851,
    "p4_success_rate": 0.927,
    "p4_divergence_rate": 0.972,
    "p4_twin_accuracy": 0.886
  },

  "compliance_tests": {
    "total": 14,
    "passed": 14,
    "failed": 0
  },

  "warnings": [
    "p3_omega_occupancy below 90% threshold (hypothetical abort)"
  ],

  "artifacts": {
    "evidence_pack": "results/ci/evidence_pack/",
    "manifest_sha256": "..."
  }
}
```

**Field definitions:**

| Field | Type | Description |
|-------|------|-------------|
| `shadow_mode_ok` | bool | All SHADOW MODE compliance tests passed |
| `determinism_ok` | bool | Determinism check passed (slow path only) |
| `p3_harness_ok` | bool | P3 harness completed successfully |
| `p4_harness_ok` | bool | P4 harness completed successfully |
| `evidence_pack_ok` | bool | Evidence pack built with valid manifest |
| `telemetry_source` | string | `"mock"`, `"real_synthetic"`, or `"real_trace"` (P5 status single source of truth) |
| `proof_snapshot_present` | bool | Proof snapshot artifact detected in evidence pack |
| `last_run_id` | object | Run IDs for P3 and P4 golden runs |
| `metrics_snapshot` | object | Key metrics from latest run |
| `warnings` | array | Non-blocking issues detected |

**Migration note:** Evidence packs with `schema_version < 1.2.0` do not include `telemetry_source`; auditors interpret these as `mock` (legacy default).

---

## CI Status Artifact (Phase X Evidence workflow)

- The `CODEX M — Phase X Evidence (SHADOW TESTS ONLY)` workflow uploads `phase-x-evidence-artifacts-<run>` containing `artifacts/phase_x/first_light_status.json` plus JUnit XMLs.
- Download from the GitHub Actions run page (Artifacts panel) to inspect the SHADOW-only status snapshot without changing gates.
- Use `first_light_status.json` to debug failures by checking `shadow_mode_ok`, `p3_harness_ok`, `p4_harness_ok`, `evidence_pack_ok`, `warnings`, and `errors`, then cross-reference `last_run_id` with `results/first_light/golden_run` outputs.

---

## CI Job Definitions (Pseudocode)

### GitHub Actions Structure

```yaml
# .github/workflows/first-light-shadow.yml (DO NOT CREATE YET)

name: First Light Shadow Checks

on:
  pull_request:
    branches: [master, 'integrate/*']
  schedule:
    - cron: '0 2 * * *'  # Nightly at 02:00 UTC
  workflow_dispatch:

jobs:
  shadow-compliance:
    # Fast path - runs on every PR
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync
      - run: uv run python -m pytest tests/integration/test_shadow_mode_compliance.py -v
      - run: |
          # Export minimal status
          echo '{"shadow_mode_ok": true, "pipeline": "pr"}' > first_light_status.json
      - uses: actions/upload-artifact@v4
        with:
          name: first-light-status
          path: first_light_status.json

  golden-run:
    # Slow path - nightly only
    if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync

      # P3 Harness
      - run: |
          uv run python scripts/usla_first_light_harness.py \
            --cycles 1000 --seed 42 --slice arithmetic_simple \
            --runner-type u2 --tau-0 0.20 --window-size 50 \
            --output-dir results/ci/p3

      # P4 Harness
      - run: |
          uv run python scripts/usla_first_light_p4_harness.py \
            --cycles 1000 --seed 42 --slice arithmetic_simple \
            --runner-type u2 --tau-0 0.20 \
            --output-dir results/ci/p4

      # Evidence Pack
      - run: |
          uv run python scripts/build_first_light_evidence_pack.py \
            --p3-dir results/ci/p3 \
            --p4-dir results/ci/p4 \
            --output-dir results/ci/evidence_pack

      # Determinism Check
      - run: uv run python scripts/verify_first_light_determinism.py

      # Compliance Tests
      - run: uv run python -m pytest tests/integration/test_shadow_mode_compliance.py -v

      # Export comprehensive status
      - run: |
          uv run python -c "
          import json
          from pathlib import Path
          from datetime import datetime, timezone

          # ... status generation logic ...

          status = {
            'schema_version': '1.0.0',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'mode': 'SHADOW',
            'pipeline': 'nightly',
            'shadow_mode_ok': True,
            'determinism_ok': True,
            # ... etc
          }

          with open('first_light_status.json', 'w') as f:
            json.dump(status, f, indent=2)
          "

      # Archive artifacts
      - uses: actions/upload-artifact@v4
        with:
          name: first-light-evidence-pack
          path: results/ci/evidence_pack/

      - uses: actions/upload-artifact@v4
        with:
          name: first-light-status
          path: first_light_status.json
```

### System Law Index Freshness Check (spec only, shadow)

```yaml
# .github/workflows/system-law-index-check.yml (DO NOT CREATE YET)
# name: System Law Index Freshness
# on:
#   pull_request:
#     branches: [master, 'integrate/*']
#   workflow_dispatch:
# jobs:
#   system-law-index-check:
#     runs-on: ubuntu-latest
#     steps:
#       - uses: actions/checkout@v4
#       - uses: actions/setup-python@v5
#         with:
#           python-version: '3.11'
#       - run: python scripts/check_system_law_index.py
#         # Status is advisory; script always exits 0 and prints:
#         # - "system-law-index: up to date (no action needed)" OR
#         # - "system-law-index: out of date; run `python tools/generate_system_law_index.py`"
```

---

## Governance Integration (Future)

When authorized to move from SHADOW to ACTIVE mode:

### Phase 1: Advisory Gating
- CI reports `shadow_mode_ok: false` as **warning** on PR
- Human reviewer must acknowledge before merge
- No automatic blocking

### Phase 2: Soft Gating
- `shadow_mode_ok: false` blocks merge **unless** maintainer override
- Override requires explicit comment (e.g., `/first-light-override reason: ...`)
- All overrides logged to audit trail

### Phase 3: Hard Gating
- `shadow_mode_ok: false` blocks merge unconditionally
- Requires code fix or threshold adjustment (via governance process)

### Status Signals for Governance

The `first_light_status.json` provides machine-readable signals:

```python
# Example governance hook (pseudocode)
def check_first_light_gate(status: dict) -> GateDecision:
    if not status["shadow_mode_ok"]:
        return GateDecision.BLOCK("SHADOW MODE compliance failed")

    if status["metrics_snapshot"]["p3_omega_occupancy"] < 0.85:
        return GateDecision.WARN("Omega occupancy critically low")

    if status["metrics_snapshot"]["p4_divergence_rate"] > 0.99:
        return GateDecision.WARN("Divergence rate near 100%")

    return GateDecision.PASS()
```

---

## Current Constraints

1. **No CI files created** — this document is specification only
2. **SHADOW MODE enforced** — all checks are advisory, not blocking
3. **No governance integration** — `first_light_status.json` is exported but not consumed
4. **Manual trigger available** — nightly can be run on-demand via `workflow_dispatch`

---

## Golden Run Version Tagging

### Evidence Pack Naming Convention

To distinguish mock and real telemetry evidence packs:

| Telemetry Source | Evidence Pack Name | Run Directory Pattern |
|------------------|--------------------|-----------------------|
| Mock (current) | `evidence_pack_first_light_mock` | `fl_<timestamp>_seed42_mock` |
| Real (P5) | `evidence_pack_first_light_real` | `fl_<timestamp>_seed42_real` |

**Migration path:**
1. Current `evidence_pack_first_light/` remains as-is (implicit mock)
2. P5 creates new `evidence_pack_first_light_real/`
3. Future runs explicitly suffix with `_mock` or `_real`

### Status JSON Extension for P5 (implemented)

Schema is bumped to **1.2.0** and now carries two observational fields:

- `telemetry_source`: "mock", "real_synthetic", or "real_trace" (defaults to `mock` for legacy runs, derived from P4 run config or directory name)
- `proof_snapshot_present`: `true`/`false` indicating whether a proof snapshot artifact exists in the evidence pack (non-gating)

### Comparison Report (P5 vs Mock)

After P5 execution, generate a comparison report:

```json
{
  "comparison_type": "mock_vs_real",
  "mock_run_id": "fl_20251211_044905_seed42",
  "real_run_id": "fl_20251215_120000_seed42_real",
  "metrics_comparison": {
    "p4_divergence_rate": {
      "mock": 0.972,
      "real": 0.XXX,
      "delta": -0.XXX,
      "improved": true
    },
    "p4_twin_accuracy": {
      "mock": 0.886,
      "real": 0.XXX,
      "delta": +0.XXX,
      "improved": true
    }
  },
  "assessment": "acceptable" | "marginal" | "concerning" | "unstable"
}
```

This report will be generated by a future `compare_first_light_runs.py` script.

---

## Implementation Checklist (For Future Agent)

When authorized to implement CI integration:

```
[ ] Create .github/workflows/first-light-shadow.yml
[ ] Add uv setup step with caching
[ ] Implement status JSON generation script
[ ] Configure artifact retention (30 days recommended)
[ ] Add Slack/Discord notification on nightly failure
[ ] Wire first_light_status.json to PR status check
[ ] Document manual override process for soft gating
[ ] Create governance dashboard to visualize historical status
```

---

**Document generated:** 2025-12-11
**Authorization required for:** CI file creation, governance binding
