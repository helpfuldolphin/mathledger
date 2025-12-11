# 7-Week Execution Tracker with Milestone Gates

**Author**: Manus-B (Ledger Integrity & PQ Migration Engineer)  
**Date**: 2025-12-09  
**Status**: REAL-READY

---

## OVERVIEW

This tracker defines a **7-week deterministic rollout** for consensus-replay integration, drift radar deployment, and PQ migration readiness.

Each week has:
- **Milestone Gate** - Must-pass criteria to proceed to next week
- **Deliverables** - Concrete artifacts to produce
- **Verification Commands** - Commands to verify milestone completion
- **Rollback Procedure** - How to rollback if milestone fails

---

## WEEK 1: Foundation & Consensus Integration

### Milestone Gate

**Criteria**:
- ✅ Consensus integration diffs applied successfully
- ✅ All imports work without errors
- ✅ Syntax verification passes
- ✅ Unit tests pass (consensus rules, validators, violations)

**Verification Command**:
```bash
# Verify imports
python3 -c "
from backend.consensus.rules import validate_block_structure, validate_attestation_roots
from backend.consensus.validators import BlockValidator
from backend.consensus.violations import RuleViolation
from backend.ledger.replay.checker import ReplayResult, verify_block_replay
from backend.ledger.replay.engine import replay_blocks
print('✓ All imports successful')
"

# Run unit tests
python3 -m pytest tests/unit/test_consensus_rules.py -v
python3 -m pytest tests/unit/test_consensus_validators.py -v
python3 -m pytest tests/unit/test_replay_verification.py -v
```

**Expected Output**:
```
✓ All imports successful
PASSED tests/unit/test_consensus_rules.py::test_validate_block_structure
PASSED tests/unit/test_consensus_rules.py::test_validate_attestation_roots
PASSED tests/unit/test_consensus_validators.py::test_block_validator
PASSED tests/unit/test_replay_verification.py::test_verify_block_replay
```

---

### Deliverables

1. **Applied Diffs**:
   - `backend/ledger/replay/checker.py` - Consensus integration (~80 lines)
   - `backend/ledger/replay/engine.py` - Consensus-first vetting (~40 lines)

2. **Backup Files**:
   - `backend/ledger/replay/checker.py.backup`
   - `backend/ledger/replay/engine.py.backup`

3. **Unit Tests**:
   - `tests/unit/test_consensus_replay_integration.py` - Integration tests

---

### Tasks

| Day | Task | Owner | Status |
|-----|------|-------|--------|
| Mon | Apply Δ-DIFF 1 (checker.py) | Engineer A | ⏳ |
| Mon | Apply Δ-DIFF 2 (engine.py) | Engineer A | ⏳ |
| Tue | Write unit tests (consensus integration) | Engineer B | ⏳ |
| Wed | Run unit tests | Engineer A | ⏳ |
| Thu | Fix test failures (if any) | Engineer A | ⏳ |
| Fri | Milestone Gate verification | Tech Lead | ⏳ |

---

### Rollback Procedure

```bash
# If milestone gate fails, rollback to original code
./scripts/rollback_consensus_integration.sh

# Verify rollback
python3 -m pytest tests/unit/test_replay_verification.py -v
```

---

## WEEK 2: Integration Testing & Drift Radar Setup

### Milestone Gate

**Criteria**:
- ✅ Integration tests pass (20+ tests)
- ✅ Replay verification succeeds on 1000 test blocks
- ✅ Drift radar scanner deployed
- ✅ Drift radar governance adaptor deployed

**Verification Command**:
```bash
# Run integration tests
python3 -m pytest tests/integration/test_consensus_replay_integration.py -v

# Run replay verification on test database
python3 scripts/replay_verify.py \\
    --database-url $TEST_DATABASE_URL \\
    --start-block 0 \\
    --end-block 1000 \\
    --strategy full_chain

# Test drift radar scanner
python3 scripts/ci/drift_radar_scan.py \\
    --database-url $TEST_DATABASE_URL \\
    --start-block 0 \\
    --end-block 1000 \\
    --scan-types schema,hash-delta \\
    --governance-policy moderate \\
    --output drift_signals.json \\
    --evidence-pack evidence_pack.json
```

**Expected Output**:
```
PASSED tests/integration/test_consensus_replay_integration.py (20/20 tests)
Replayed 1000 blocks, Success rate: 100.00%
GOVERNANCE SIGNAL: OK
```

---

### Deliverables

1. **Integration Tests**:
   - `tests/integration/test_consensus_replay_integration.py` - 20+ tests

2. **CI Scripts**:
   - `scripts/ci/drift_radar_scan.py` - Drift radar scanner
   - `scripts/replay_verify.py` - Replay verification CLI

3. **Test Results**:
   - `test_results_week2.json` - Integration test results
   - `drift_signals.json` - Drift signals from test scan
   - `evidence_pack.json` - Governance evidence pack

---

### Tasks

| Day | Task | Owner | Status |
|-----|------|-------|--------|
| Mon | Write integration tests (10 tests) | Engineer B | ⏳ |
| Tue | Write integration tests (10 tests) | Engineer B | ⏳ |
| Wed | Deploy drift radar scanner | Engineer C | ⏳ |
| Thu | Run integration tests + replay verification | Engineer A | ⏳ |
| Fri | Milestone Gate verification | Tech Lead | ⏳ |

---

### Rollback Procedure

```bash
# If integration tests fail, rollback consensus integration
./scripts/rollback_consensus_integration.sh

# If drift radar fails, adjust governance policy
# (No code rollback needed, drift radar is read-only)
```

---

## WEEK 3: Database Migration & Epoch System

### Milestone Gate

**Criteria**:
- ✅ Migration 018 (epoch root system) applied successfully
- ✅ Epoch sealer deployed
- ✅ Epoch verifier deployed
- ✅ Backfill script tested on test database

**Verification Command**:
```bash
# Verify migration 018 applied
psql $TEST_DATABASE_URL -c "\\d epochs"

# Expected output:
# Table "public.epochs"
# Column          | Type    | ...
# ----------------+---------+-----
# epoch_number    | integer | ...
# epoch_root      | varchar | ...
# ...

# Test epoch sealer
python3 -c "
from backend.ledger.epoch.sealer import seal_epoch
epoch_root = seal_epoch(epoch_number=0, blocks=[...])
print(f'✓ Epoch sealer works: {epoch_root}')
"

# Test backfill script
python3 scripts/backfill_epochs.py \\
    --database-url $TEST_DATABASE_URL \\
    --start-block 0 \\
    --end-block 1000 \\
    --dry-run
```

**Expected Output**:
```
✓ Epoch sealer works: abc123...
Dry-run: Would create 10 epochs (0-9)
```

---

### Deliverables

1. **Database Migration**:
   - `migrations/018_epoch_root_system.sql` - Epoch table schema

2. **Backfill Script**:
   - `scripts/backfill_epochs.py` - Epoch backfill script

3. **Test Results**:
   - `backfill_dry_run_results.json` - Dry-run results

---

### Tasks

| Day | Task | Owner | Status |
|-----|------|-------|--------|
| Mon | Apply migration 018 on test database | DBA | ⏳ |
| Tue | Test epoch sealer | Engineer A | ⏳ |
| Wed | Test epoch verifier | Engineer A | ⏳ |
| Thu | Run backfill script (dry-run) | Engineer A | ⏳ |
| Fri | Milestone Gate verification | Tech Lead | ⏳ |

---

### Rollback Procedure

```bash
# Rollback migration 018
psql $TEST_DATABASE_URL -c "DROP TABLE IF EXISTS epochs CASCADE;"

# Verify rollback
psql $TEST_DATABASE_URL -c "\\d epochs"
# Expected output: Did not find any relation named "epochs"
```

---

## WEEK 4: CI Helper Scripts & Governance Chain

### Milestone Gate

**Criteria**:
- ✅ 10 CI helper scripts implemented
- ✅ Governance chain script runs successfully
- ✅ All CI scripts tested locally
- ✅ CI scripts integrated into GitHub Actions (manual setup)

**Verification Command**:
```bash
# Test all CI helper scripts
for script in scripts/ci/*.py; do
    echo "Testing $script..."
    python3 "$script" --help
done

# Test governance chain
./scripts/ci/run_governance_chain.sh \\
    --database-url $TEST_DATABASE_URL \\
    --start-block 0 \\
    --end-block 1000
```

**Expected Output**:
```
Testing scripts/ci/drift_radar_scan.py... OK
Testing scripts/ci/replay_verify.py... OK
...
Governance Chain: PASSED (6/6 jobs)
```

---

### Deliverables

1. **CI Helper Scripts** (10 scripts):
   - `scripts/ci/drift_radar_scan.py`
   - `scripts/ci/replay_verify.py`
   - `scripts/ci/validate_migrations.py`
   - `scripts/ci/audit_monotonicity.py`
   - `scripts/ci/validate_chain.py`
   - `scripts/ci/epoch_validation.py`
   - `scripts/ci/schema_migration_check.py`
   - `scripts/ci/backfill_dry_run.py`
   - `scripts/ci/attestation_integrity_sweep.py`
   - `scripts/ci/pq_activation_readiness.py`

2. **Governance Chain Script**:
   - `scripts/ci/run_governance_chain.sh` - Orchestrates all CI jobs

3. **GitHub Actions Workflow** (manual setup):
   - `.github/workflows/governance-chain.yml` - CI workflow definition

---

### Tasks

| Day | Task | Owner | Status |
|-----|------|-------|--------|
| Mon | Implement CI scripts 1-3 | Engineer C | ⏳ |
| Tue | Implement CI scripts 4-7 | Engineer C | ⏳ |
| Wed | Implement CI scripts 8-10 | Engineer C | ⏳ |
| Thu | Test governance chain locally | Engineer C | ⏳ |
| Fri | Milestone Gate verification | Tech Lead | ⏳ |

---

### Rollback Procedure

```bash
# No rollback needed (CI scripts are standalone)
# If governance chain fails, fix individual scripts
```

---

## WEEK 5: PQ Migration Preparation

### Milestone Gate

**Criteria**:
- ✅ Migration 019 (dual-commitment) applied successfully
- ✅ Dual-commitment hash algorithm implemented
- ✅ Activation script tested on test database
- ✅ Cross-algorithm prev_hash validation implemented

**Verification Command**:
```bash
# Verify migration 019 applied
psql $TEST_DATABASE_URL -c "\\d blocks"

# Expected output includes SHA-3 columns:
# reasoning_attestation_root_sha3 | varchar | ...
# ui_attestation_root_sha3        | varchar | ...
# composite_attestation_root_sha3 | varchar | ...

# Test dual-commitment algorithm
python3 -c "
from backend.crypto.hashing import get_hash_algorithm
dual = get_hash_algorithm('dual-v1')
sha256_hash, sha3_hash = dual.hash(b'test')
print(f'✓ SHA-256: {sha256_hash}')
print(f'✓ SHA-3: {sha3_hash}')
"

# Test activation script (dry-run)
python3 scripts/activate_dual_commitment.py \\
    --activation-block 100000 \\
    --database-url $TEST_DATABASE_URL \\
    --dry-run
```

**Expected Output**:
```
✓ SHA-256: 9f86d081...
✓ SHA-3: 36f02858...
Dry-run: Would activate dual-commitment at block 100000
```

---

### Deliverables

1. **Database Migration**:
   - `migrations/019_dual_commitment.sql` - SHA-3 columns

2. **Hash Algorithm**:
   - `backend/crypto/hashing.py` - Dual-commitment algorithm

3. **Activation Scripts**:
   - `scripts/activate_dual_commitment.py` - Activate dual-commitment
   - `scripts/verify_dual_commitment.py` - Verify activation

---

### Tasks

| Day | Task | Owner | Status |
|-----|------|-------|--------|
| Mon | Apply migration 019 on test database | DBA | ⏳ |
| Tue | Implement dual-commitment algorithm | Engineer A | ⏳ |
| Wed | Implement activation script | Engineer A | ⏳ |
| Thu | Test activation script (dry-run) | Engineer A | ⏳ |
| Fri | Milestone Gate verification | Tech Lead | ⏳ |

---

### Rollback Procedure

```bash
# Rollback migration 019
psql $TEST_DATABASE_URL -c "
ALTER TABLE blocks
DROP COLUMN IF EXISTS reasoning_attestation_root_sha3,
DROP COLUMN IF EXISTS ui_attestation_root_sha3,
DROP COLUMN IF EXISTS composite_attestation_root_sha3;
"

# Rollback code changes
git checkout backend/crypto/hashing.py
git checkout backend/ledger/blockchain.py
```

---

## WEEK 6: Staging Deployment & End-to-End Testing

### Milestone Gate

**Criteria**:
- ✅ All components deployed to staging environment
- ✅ End-to-end replay verification succeeds on staging data
- ✅ Drift radar scan completes without errors
- ✅ Governance chain passes all 6 jobs

**Verification Command**:
```bash
# Run end-to-end replay verification on staging
python3 scripts/replay_verify.py \\
    --database-url $STAGING_DATABASE_URL \\
    --start-block 0 \\
    --end-block 10000 \\
    --strategy full_chain

# Run drift radar scan on staging
python3 scripts/ci/drift_radar_scan.py \\
    --database-url $STAGING_DATABASE_URL \\
    --start-block 0 \\
    --end-block 10000 \\
    --scan-types schema,hash-delta,metadata,statement \\
    --governance-policy moderate \\
    --output drift_signals_staging.json \\
    --evidence-pack evidence_pack_staging.json

# Run governance chain on staging
./scripts/ci/run_governance_chain.sh \\
    --database-url $STAGING_DATABASE_URL \\
    --start-block 0 \\
    --end-block 10000
```

**Expected Output**:
```
Replayed 10000 blocks, Success rate: 100.00%
GOVERNANCE SIGNAL: OK
Governance Chain: PASSED (6/6 jobs)
```

---

### Deliverables

1. **Staging Deployment**:
   - All code deployed to staging environment
   - All migrations applied to staging database

2. **Test Results**:
   - `staging_replay_results.json` - Replay verification results
   - `staging_drift_signals.json` - Drift signals
   - `staging_governance_chain_results.json` - Governance chain results

---

### Tasks

| Day | Task | Owner | Status |
|-----|------|-------|--------|
| Mon | Deploy code to staging | DevOps | ⏳ |
| Tue | Apply migrations to staging database | DBA | ⏳ |
| Wed | Run end-to-end replay verification | Engineer A | ⏳ |
| Thu | Run drift radar + governance chain | Engineer C | ⏳ |
| Fri | Milestone Gate verification | Tech Lead | ⏳ |

---

### Rollback Procedure

```bash
# Rollback staging deployment
git checkout main
# Re-deploy previous version to staging
```

---

## WEEK 7: Production Validation (Read-Only)

### Milestone Gate

**Criteria**:
- ✅ Read-only replay verification succeeds on production data
- ✅ Read-only drift radar scan completes without errors
- ✅ Production audit report generated
- ✅ No critical violations detected

**Verification Command**:
```bash
# Run read-only replay verification on production
python3 scripts/replay_verify.py \\
    --database-url $PRODUCTION_DATABASE_URL \\
    --start-block 0 \\
    --end-block 50000 \\
    --strategy sliding_window \\
    --read-only

# Run read-only drift radar scan on production
python3 scripts/ci/drift_radar_scan.py \\
    --database-url $PRODUCTION_DATABASE_URL \\
    --start-block 0 \\
    --end-block 50000 \\
    --scan-types schema,hash-delta,metadata,statement \\
    --governance-policy moderate \\
    --output drift_signals_prod.json \\
    --evidence-pack evidence_pack_prod.json \\
    --read-only

# Generate production audit report
python3 scripts/generate_audit_report.py \\
    --replay-results replay_results_prod.json \\
    --drift-signals drift_signals_prod.json \\
    --output production_audit_report.md
```

**Expected Output**:
```
Replayed 50000 blocks, Success rate: 100.00%
GOVERNANCE SIGNAL: OK
Production Audit Report: production_audit_report.md
```

---

### Deliverables

1. **Production Validation Results**:
   - `replay_results_prod.json` - Replay verification results
   - `drift_signals_prod.json` - Drift signals
   - `evidence_pack_prod.json` - Governance evidence pack

2. **Production Audit Report**:
   - `production_audit_report.md` - Comprehensive audit report

---

### Tasks

| Day | Task | Owner | Status |
|-----|------|-------|--------|
| Mon | Run read-only replay verification | Engineer A | ⏳ |
| Tue | Run read-only drift radar scan | Engineer C | ⏳ |
| Wed | Generate production audit report | Engineer B | ⏳ |
| Thu | Review audit report with team | Tech Lead | ⏳ |
| Fri | Milestone Gate verification | CTO | ⏳ |

---

### Rollback Procedure

```bash
# No rollback needed (read-only operations)
```

---

## MILESTONE GATE SUMMARY

| Week | Milestone | Status | Gate Criteria |
|------|-----------|--------|---------------|
| 1 | Foundation & Consensus Integration | ⏳ | Imports work, unit tests pass |
| 2 | Integration Testing & Drift Radar | ⏳ | 20+ tests pass, 1000 blocks replayed |
| 3 | Database Migration & Epoch System | ⏳ | Migration 018 applied, backfill tested |
| 4 | CI Helper Scripts & Governance Chain | ⏳ | 10 scripts work, governance chain passes |
| 5 | PQ Migration Preparation | ⏳ | Migration 019 applied, dual-commitment works |
| 6 | Staging Deployment & E2E Testing | ⏳ | 10000 blocks replayed, governance chain passes |
| 7 | Production Validation (Read-Only) | ⏳ | 50000 blocks replayed, audit report generated |

---

## CRITICAL PATH

**Minimum Duration**: 7 weeks (no parallelization)

**Optimized Duration**: 5 weeks (with parallelization)

**Parallelization Opportunities**:
- Week 2: Integration tests (Engineer B) + Drift radar (Engineer C) in parallel
- Week 3: Epoch system (Engineer A) + CI scripts (Engineer C) in parallel
- Week 5: PQ migration (Engineer A) + CI integration (Engineer C) in parallel

---

## SMOKE-TEST READINESS CHECKLIST

### Files to Create

1. **Tracking Spreadsheet**:
   - `7_week_execution_tracker.xlsx` - Excel tracker with tasks, owners, status

2. **Milestone Gate Scripts**:
   - `scripts/verify_milestone_gate_week1.sh` - Week 1 verification
   - `scripts/verify_milestone_gate_week2.sh` - Week 2 verification
   - ... (7 scripts total)

3. **Audit Report Template**:
   - `templates/production_audit_report_template.md` - Audit report template

### Commands to Run Locally

**Create Milestone Gate Script (Week 1)**:
```bash
cat > scripts/verify_milestone_gate_week1.sh <<'EOF'
#!/bin/bash
set -e

echo "Verifying Week 1 Milestone Gate..."

# Verify imports
python3 -c "
from backend.consensus.rules import validate_block_structure
from backend.ledger.replay.checker import ReplayResult
print('✓ Imports successful')
"

# Run unit tests
python3 -m pytest tests/unit/test_consensus_rules.py -v
python3 -m pytest tests/unit/test_replay_verification.py -v

echo "✓ Week 1 Milestone Gate: PASSED"
EOF

chmod +x scripts/verify_milestone_gate_week1.sh

# Test milestone gate script
./scripts/verify_milestone_gate_week1.sh
```

**Expected Output**:
```
Verifying Week 1 Milestone Gate...
✓ Imports successful
PASSED tests/unit/test_consensus_rules.py
PASSED tests/unit/test_replay_verification.py
✓ Week 1 Milestone Gate: PASSED
```

---

## REALITY LOCK VERIFICATION

**Files Referenced** (all REAL):
- ✅ `backend/consensus/rules.py` - EXISTS
- ✅ `backend/ledger/replay/checker.py` - EXISTS
- ✅ `backend/ledger/replay/engine.py` - EXISTS
- ✅ `backend/ledger/epoch/sealer.py` - EXISTS
- ✅ `backend/crypto/hashing.py` - EXISTS

**Migrations Referenced**:
- ✅ `migrations/018_epoch_root_system.sql` - EXISTS (created in Phase II)
- ⏳ `migrations/019_dual_commitment.sql` - TO BE CREATED (Week 5)

**Scripts Referenced**:
- ✅ `scripts/backfill_epochs.py` - EXISTS (created in Phase II)
- ⏳ `scripts/ci/drift_radar_scan.py` - EXISTS (created in Phase IV)
- ⏳ `scripts/replay_verify.py` - TO BE CREATED (Week 2)

**Status**: # REAL-READY

---

**"Keep it blue, keep it clean, keep it sealed."**

— Manus-B, Ledger Integrity & PQ Migration Engineer
