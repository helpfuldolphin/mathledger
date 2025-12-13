# CRITICAL FILES MANIFEST

**Version:** 1.0.0
**Last Updated:** 2025-12-13
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
| `scripts/first_light_p3_harness.py` | P3 harness | **NO - MUST TRACK** |

### Governance

| File | Purpose | Tracked |
|------|---------|---------|
| `backend/governance/fusion.py` | Multi-signal governance | **NO - MUST TRACK** |
| `backend/governance/evidence_pack.py` | Evidence pack core | **NO - MUST TRACK** |
| `backend/governance/last_mile_checker.py` | Final validation | **NO - MUST TRACK** |

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
