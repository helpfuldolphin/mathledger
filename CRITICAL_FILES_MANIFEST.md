# CRITICAL FILES MANIFEST

**Version:** 1.5.0
**Last Updated:** 2025-12-13
**Hygiene Audit:** CAL-EXP-2 reproducibility verified with CI gate (import-level GATING)
**Purpose:** Enumerate files that MUST be under version control to prevent operational failures.

---

## Overview

This manifest identifies files critical to MathLedger's operation. Any file listed here:
1. MUST be tracked in git (not untracked)
2. MUST NOT be corrupted or accidentally deleted
3. MUST be validated by CI on every PR

The `generate_first_light_status.py` incident (2025-12-13) demonstrated that untracked critical files can become corrupted without detection, causing test cascades and trust erosion.

---

## Tier 1: ABSOLUTELY CRITICAL (System Cannot Function Without)

These files, if missing or corrupted, will cause immediate system failure.

### Core Infrastructure

| File | Purpose | Verified |
|------|---------|----------|
| `backend/worker.py` | Lean verification worker | YES |
| `backend/lean_mode.py` | Lean mode configuration | YES |
| `backend/lean_interface.py` | Lean statement interface | YES |
| `attestation/dual_root.py` | Dual-root attestation | YES |
| `normalization/canon.py` | Formula canonicalization | YES |
| `normalization/taut.py` | Tautology verification | YES |
| `backend/axiom_engine/derive_core.py` | Derivation engine | YES |
| `backend/axiom_engine/axioms.py` | Axiom schemas | YES |
| `backend/axiom_engine/rules.py` | Inference rules | YES |
| `backend/crypto/core.py` | Cryptographic primitives | YES |
| `ledger/ingest.py` | Ledger ingestion | YES |

### Database & Migrations

| File | Purpose | Verified |
|------|---------|----------|
| `migrations/*.sql` | Database schema | YES |
| `backend/ledger/blockchain.py` | Block sealing | YES |

### Configuration

| File | Purpose | Verified |
|------|---------|----------|
| `backend/lean_proj/lakefile.lean` | Lean build config | YES |
| `backend/lean_proj/lean-toolchain` | Lean version pin | YES |

---

## Tier 2: OPERATIONALLY CRITICAL (Tests/CI Will Fail)

These files, if missing or corrupted, will cause test failures and block CI.

### First Light / Calibration

| File | Purpose | Tracked |
|------|---------|---------|
| `scripts/generate_first_light_status.py` | Status generation | YES |
| `scripts/build_first_light_evidence_pack.py` | Evidence pack builder | **NO - MUST TRACK** |
| `scripts/generate_first_light_alignment_view.py` | Alignment view | **NO - MUST TRACK** |
| `scripts/first_light_p3_harness.py` | P3 harness | YES |

### CAL-EXP-1 Harnesses (CRITICAL FOR REPRODUCIBILITY)

| File | Purpose | Tracked |
|------|---------|---------|
| `scripts/run_p5_cal_exp1.py` | Main P5 CAL-EXP-1 harness | **MUST TRACK** |
| `scripts/first_light_cal_exp1_warm_start.py` | Warm-start harness (200-cycle) | **MUST TRACK** |
| `scripts/first_light_cal_exp1_runtime_stability.py` | Runtime stability metrics | **MUST TRACK** |
| `scripts/first_light_cal_exp2_convergence.py` | CAL-EXP-2 harness | **MUST TRACK** |
| `scripts/first_light_cal_exp3_regime_change.py` | CAL-EXP-3 harness | **MUST TRACK** |
| `scripts/first_light_proof_hash_snapshot.py` | Proof hash snapshot | **MUST TRACK** |
| `results/cal_exp_1/cal_exp_1_harness.py` | CAL-EXP-1 execution harness | YES |

### CAL-EXP-1 Backend Modules (CRITICAL)

| File/Directory | Purpose | Tracked |
|----------------|---------|---------|
| `experiments/u2/runtime/__init__.py` | Runtime profile system | **MUST TRACK** |
| `experiments/u2/runtime/profile_guard.py` | Profile drift detection | **MUST TRACK** |
| `experiments/u2/runtime/calibration_correlation.py` | Calibration correlation | **MUST TRACK** |
| `backend/telemetry/__init__.py` | Telemetry module | **MUST TRACK** |
| `backend/telemetry/rtts_cal_exp_window_join.py` | RTTS CAL-EXP window join | **MUST TRACK** |
| `backend/telemetry/governance_signal.py` | Governance signal emission | **MUST TRACK** |
| `derivation/budget_cal_exp_integration.py` | Budget calibration | **MUST TRACK** |
| `derivation/budget_invariants.py` | Budget invariants | **MUST TRACK** |
| `experiments/u2/cal_exp1_reconciliation.py` | CAL-EXP-1 reconciliation | **MUST TRACK** |

### CAL-EXP-1 Configuration

| File | Purpose | Tracked |
|------|---------|---------|
| `config/p5_synthetic.json` | P5 synthetic adapter config | **MUST TRACK** |

### CAL-EXP-1 Test Directories (CRITICAL)

| Directory | Purpose | Tracked |
|-----------|---------|---------|
| `tests/telemetry/` | Telemetry tests | **MUST TRACK** |
| `tests/derivation/` | Derivation tests | **MUST TRACK** |
| `tests/evidence/` | Evidence tests | **MUST TRACK** |
| `tests/experiments/` | Experiments tests | **MUST TRACK** |

> **HYGIENE AUDIT 2025-12-13:** These files were identified as untracked by the Commit Hygiene Sentinel. See `COMMIT_HYGIENE_REPORT_CAL_EXP_1.md` for details.

### CAL-EXP-2 Minimal Set (GATING)

> **CI VALIDATION:** These files are validated by `.github/workflows/cal_exp_hygiene_gate.yml` (GATING).
> **Test Coverage:** `tests/ci/test_cal_exp2_reproducibility.py::TestCalExp2MinimalSet`

The following files constitute the **minimum required set** for CAL-EXP-2 reproducibility.
A clean checkout with only these files tracked MUST be able to import and initialize CAL-EXP-2.

| File | Purpose | Import Required | Tracked |
|------|---------|-----------------|---------|
| `scripts/first_light_cal_exp2_convergence.py` | **HARNESS** - Main entry point | YES | ✅ |
| `backend/topology/first_light/runner_p4.py` | P4 shadow runner | YES | ✅ |
| `backend/topology/first_light/config_p4.py` | P4 configuration | YES | ✅ |
| `backend/topology/first_light/telemetry_adapter.py` | Mock telemetry provider | YES | ✅ |
| `backend/topology/first_light/divergence_analyzer.py` | Divergence analysis | YES | ✅ |
| `backend/topology/first_light/data_structures_p4.py` | P4 data structures | YES | ✅ |
| `backend/topology/first_light/p5_pattern_classifier.py` | Pattern classifier | YES | ✅ |
| `tests/first_light/test_cal_exp2_exp3_scaffolds.py` | CAL-EXP-2 test suite | NO | ✅ |

**Constraints:**
- ❌ `results/` directory is **NOT REQUIRED** for import
- ❌ No 1000-cycle execution in CI (import-level only)
- ✅ Fresh checkout + `uv sync` must succeed
- ✅ All imports must resolve without runtime dependencies

> **CAL-EXP-2 REPRODUCIBILITY RESTORED 2025-12-13:** All dependencies verified TRACKED. Clean checkout can reproduce CAL-EXP-2.
> **CI GATE ADDED 2025-12-13:** Import-level gating via `cal_exp_hygiene_gate.yml`.

### Governance

| File | Purpose | Tracked |
|------|---------|---------|
| `backend/governance/fusion.py` | Multi-signal governance | **NO - MUST TRACK** |
| `backend/governance/evidence_pack.py` | Evidence pack core | **NO - MUST TRACK** |
| `backend/governance/last_mile_checker.py` | Final validation | **NO - MUST TRACK** |

### TDA / Calibration Experiment Support

| File | Purpose | Tracked |
|------|---------|---------|
| `backend/tda/metrics.py` | TDA metrics computation | YES |
| `backend/tda/monitor.py` | TDA monitoring and red-flags | YES |
| `backend/tda/console_tile.py` | TDA console tile | YES |
| `backend/tda/evidence.py` | TDA evidence artifacts | YES |
| `backend/tda/pattern_classifier.py` | TDA pattern classification | YES |
| `backend/tda/patterns_from_windows.py` | TDA windowed patterns | YES |

> **Note:** These files are required by CAL-EXP-1 (calibration experiment). Missing any of these will cause CAL-EXP-1 replication to fail.

### Health / USLA

| File | Purpose | Tracked |
|------|---------|---------|
| `backend/topology/usla_simulator.py` | USLA simulator | YES |
| `backend/topology/usla_integration.py` | USLA integration | YES |
| `backend/health/*.py` | Health adapters | PARTIAL |

### Curriculum

| File | Purpose | Tracked |
|------|---------|---------|
| `curriculum/gates.py` | Curriculum gates | YES |
| `curriculum/enforcement.py` | Gate enforcement | **NO - MUST TRACK** |
| `curriculum/integration.py` | Integration layer | **NO - MUST TRACK** |

---

## Tier 3: IMPORTANT (Functionality Degraded)

These files, if missing, will degrade functionality but not cause complete failure.

### Scripts

| File | Purpose | Tracked |
|------|---------|---------|
| `scripts/generate_p5_divergence_real_report.py` | P5 divergence | **NO - MUST TRACK** |
| `scripts/generate_what_if_report.py` | What-if scenarios | **NO - MUST TRACK** |
| `scripts/compute_ctrpk_from_signals.py` | CTRPK computation | **NO - MUST TRACK** |

### Backend Modules

| File | Purpose | Tracked |
|------|---------|---------|
| `backend/orchestrator/app.py` | API server | **NO - MUST TRACK** |
| `backend/ledger/monotone_guard.py` | Monotone invariants | **NO - MUST TRACK** |
| `backend/dag/invariant_guard.py` | DAG invariants | **NO - MUST TRACK** |

---

## CI Validation Requirements

The following CI checks MUST be implemented:

### 1. Critical File Existence Check

```yaml
# .github/workflows/critical-files-check.yml
name: Critical Files Check
on: [push, pull_request]
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Verify critical files exist
        run: |
          CRITICAL_FILES=(
            "backend/worker.py"
            "backend/lean_mode.py"
            "attestation/dual_root.py"
            "normalization/canon.py"
            "scripts/generate_first_light_status.py"
            "backend/tda/metrics.py"
            "backend/tda/monitor.py"
            "backend/tda/console_tile.py"
            "backend/tda/evidence.py"
            "backend/tda/pattern_classifier.py"
            "backend/tda/patterns_from_windows.py"
          )
          for file in "${CRITICAL_FILES[@]}"; do
            if [ ! -f "$file" ]; then
              echo "CRITICAL: Missing $file"
              exit 1
            fi
          done
```

### 2. Untracked Critical Files Check

```yaml
- name: Check for untracked critical files
  run: |
    UNTRACKED=$(git status --porcelain | grep "^??" | grep -E "(scripts|backend|attestation|curriculum)/.*\.py$" || true)
    if [ -n "$UNTRACKED" ]; then
      echo "WARNING: Untracked Python files in critical paths:"
      echo "$UNTRACKED"
      # Optionally: exit 1
    fi
```

### 3. File Integrity Check

```yaml
- name: Verify critical files are valid Python
  run: |
    python -m py_compile backend/worker.py
    python -m py_compile attestation/dual_root.py
    python -m py_compile scripts/generate_first_light_status.py
```

---

## Recovery Procedures

### If a Tier 1 file is corrupted:

1. **STOP** - Do not attempt to run tests
2. Check `git stash list` for recent stashes
3. Check `git log --all --name-only` for file history
4. If file was never committed: reconstruct from test contracts (as done for `generate_first_light_status.py`)
5. **COMMIT IMMEDIATELY** after recovery

### If a Tier 2 file is corrupted:

1. Check if tests can run without it
2. Reconstruct from test imports and assertions
3. Commit and add to Tier 1 if critical

---

## Maintenance

This manifest MUST be updated when:
- New critical files are created
- Files are promoted from Tier 3 to Tier 2 or Tier 1
- CI validation rules change

**Owner:** Repository maintainers
**Review Cadence:** Every major release
