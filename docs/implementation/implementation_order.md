# Implementation Order: Deterministic Rollout Sequence

**Author**: Manus-B (Ledger Integrity & PQ Migration Engineer)  
**Date**: 2025-12-09  
**Version**: 1.0  
**Status**: Operational

---

## Purpose

This document defines a **deterministic ordering** for integrating consensus-replay-governance components into MathLedger without breaking the system.

**Critical Principle**: Dependencies must be satisfied before dependents are deployed.

---

## Dependency Graph

```
┌─────────────────────────────────────────────────────────────┐
│                    Layer 0: Foundation                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Hash         │  │ Database     │  │ Merkle       │      │
│  │ Abstraction  │  │ Schema       │  │ Trees        │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Layer 1: Consensus Runtime                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Rules        │  │ Validators   │  │ Violations   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Layer 2: Replay Engine                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Recompute    │  │ Checker      │  │ Engine       │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Layer 3: Drift Radar                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Scanner      │  │ Classifier   │  │ Governance   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Layer 4: CI Enforcement                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Helper       │  │ Governance   │  │ GitHub       │      │
│  │ Scripts      │  │ Chain        │  │ Actions      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### **Phase 0: Foundation** (Week 1)

**Objective**: Ensure all foundation components are in place.

#### Step 0.1: Verify Hash Abstraction (Manus-H Integration)

**Precondition**: None

**Action**:
```bash
# Check if hash abstraction exists
python3 -c "from backend.crypto.hashing import get_hash_algorithm; print('OK')"
```

**Expected Output**: `OK`

**If Missing**:
```bash
# File issue for Manus-H integration
gh issue create \
  --title "Implement hash abstraction layer (Manus-H)" \
  --body "Required for PQ migration. See docs/architecture/hash_abstraction.md" \
  --label "blocker,manus-h"
```

**Verification**:
```bash
# Test hash abstraction
python3 -c "
from backend.crypto.hashing import get_hash_algorithm
sha256 = get_hash_algorithm('sha256-v1')
sha3 = get_hash_algorithm('sha3-v1')
print('SHA-256:', sha256.hash(b'test'))
print('SHA-3:', sha3.hash(b'test'))
"
```

**Expected Output**: Two different hash values

---

#### Step 0.2: Apply Database Schema Migration 018

**Precondition**: Step 0.1 complete

**Action**:
```bash
# Apply migration 018 (epochs table)
python3 scripts/run-migrations.py \
  --database-url $DATABASE_URL \
  --migration 018_epoch_root_system.sql
```

**Expected Output**: Migration 018 applied successfully

**Verification**:
```bash
# Check epochs table exists
psql $DATABASE_URL -c "\d epochs"
```

**Expected Output**: Table schema displayed

---

#### Step 0.3: Verify Merkle Tree Implementation

**Precondition**: Step 0.1 complete

**Action**:
```bash
# Check if Merkle tree implementation exists
python3 -c "from backend.dag.merkle import merkle_root; print('OK')"
```

**Expected Output**: `OK`

**Verification**:
```bash
# Test Merkle tree
python3 -c "
from backend.dag.merkle import merkle_root
leaves = ['a', 'b', 'c', 'd']
root = merkle_root(leaves)
print('Merkle root:', root)
"
```

**Expected Output**: Merkle root hash

---

### **Phase 1: Consensus Runtime** (Week 2)

**Objective**: Deploy consensus rules, validators, and violations tracking.

#### Step 1.1: Deploy Consensus Rules

**Precondition**: Phase 0 complete

**Action**:
```bash
# Copy consensus rules module
cp backend/consensus/rules.py /path/to/production/backend/consensus/

# Verify import
python3 -c "from backend.consensus.rules import validate_block_structure; print('OK')"
```

**Expected Output**: `OK`

**Verification**:
```bash
# Test consensus rules
python3 -c "
from backend.consensus.rules import validate_block_structure
block = {'id': 1, 'block_number': 1, 'prev_hash': 'abc', 'reasoning_attestation_root': 'def'}
is_valid, violations = validate_block_structure(block)
print('Valid:', is_valid)
print('Violations:', violations)
"
```

**Expected Output**: `Valid: True` (for valid block)

---

#### Step 1.2: Deploy Consensus Validators

**Precondition**: Step 1.1 complete

**Action**:
```bash
# Copy consensus validators module
cp backend/consensus/validators.py /path/to/production/backend/consensus/

# Verify import
python3 -c "from backend.consensus.validators import BlockValidator; print('OK')"
```

**Expected Output**: `OK`

**Verification**:
```bash
# Test block validator
python3 -c "
from backend.consensus.validators import BlockValidator
validator = BlockValidator()
# Test with sample block
"
```

---

#### Step 1.3: Deploy Violations Tracking

**Precondition**: Step 1.2 complete

**Action**:
```bash
# Copy violations module
cp backend/consensus/violations.py /path/to/production/backend/consensus/

# Verify import
python3 -c "from backend.consensus.violations import RuleViolation; print('OK')"
```

**Expected Output**: `OK`

---

### **Phase 2: Replay Engine Integration** (Week 3)

**Objective**: Wire consensus rules into replay verification engine.

#### Step 2.1: Apply Recompute Module Diff

**Precondition**: Phase 1 complete

**Action**:
```bash
# Backup original file
cp backend/ledger/replay/recompute.py backend/ledger/replay/recompute.py.backup

# Apply diff (see docs/integration/consensus_replay_integration.md)
# Manually edit backend/ledger/replay/recompute.py
# Add: consensus_violations, consensus_passed, consensus_severity fields to ReplayResult

# Verify syntax
python3 -m py_compile backend/ledger/replay/recompute.py
```

**Expected Output**: No syntax errors

**Verification**:
```bash
# Test ReplayResult schema
python3 -c "
from backend.ledger/replay/recompute import ReplayResult
result = ReplayResult(
    block_id=1, block_number=1, hash_version='sha256-v1',
    r_t_recomputed='abc', u_t_recomputed='def', h_t_recomputed='ghi',
    r_t_stored='abc', u_t_stored='def', h_t_stored='ghi',
    r_t_match=True, u_t_match=True, h_t_match=True,
    consensus_violations=[], consensus_passed=True, consensus_severity=None
)
print('ReplayResult created successfully')
print('Has critical violations:', result.has_critical_violations())
"
```

**Expected Output**: `ReplayResult created successfully`, `Has critical violations: False`

---

#### Step 2.2: Apply Checker Module Diff

**Precondition**: Step 2.1 complete

**Action**:
```bash
# Backup original file
cp backend/ledger/replay/checker.py backend/ledger/replay/checker.py.backup

# Apply diff (see docs/integration/consensus_replay_integration.md)
# Manually edit backend/ledger/replay/checker.py
# Add: consensus validation logic to verify_block_replay()

# Verify syntax
python3 -m py_compile backend/ledger/replay/checker.py
```

**Expected Output**: No syntax errors

**Verification**:
```bash
# Test checker with sample block
python3 -c "
from backend.ledger.replay.checker import verify_block_replay
block = {
    'id': 1, 'block_number': 1, 'prev_hash': 'abc',
    'reasoning_attestation_root': 'def',
    'ui_attestation_root': 'ghi',
    'composite_attestation_root': 'jkl',
    'attestation_metadata': {'hash_version': 'sha256-v1'}
}
result = verify_block_replay(block)
print('Consensus passed:', result.consensus_passed)
"
```

**Expected Output**: `Consensus passed: True` (for valid block)

---

#### Step 2.3: Apply Engine Module Diff

**Precondition**: Step 2.2 complete

**Action**:
```bash
# Backup original file
cp backend/ledger/replay/engine.py backend/ledger/replay/engine.py.backup

# Apply diff (see docs/integration/consensus_replay_integration.md)
# Manually edit backend/ledger/replay/engine.py
# Add: consensus_first parameter to replay_blocks()

# Verify syntax
python3 -m py_compile backend/ledger/replay/engine.py
```

**Expected Output**: No syntax errors

**Verification**:
```bash
# Test engine with sample blocks
python3 -c "
from backend.ledger.replay.engine import replay_blocks
blocks = [
    {'id': 1, 'block_number': 1, 'prev_hash': 'genesis', ...},
    {'id': 2, 'block_number': 2, 'prev_hash': 'abc', ...}
]
results = replay_blocks(blocks, consensus_first=True, fail_fast=False)
print('Replayed', len(results), 'blocks')
"
```

**Expected Output**: `Replayed 2 blocks`

---

#### Step 2.4: Run Integration Tests

**Precondition**: Steps 2.1-2.3 complete

**Action**:
```bash
# Run integration tests
pytest tests/integration/test_consensus_replay_integration.py -v
```

**Expected Output**: All tests pass (20/20)

**If Tests Fail**:
```bash
# Rollback changes
cp backend/ledger/replay/recompute.py.backup backend/ledger/replay/recompute.py
cp backend/ledger/replay/checker.py.backup backend/ledger/replay/checker.py
cp backend/ledger/replay/engine.py.backup backend/ledger/replay/engine.py

# Debug and retry
```

---

### **Phase 3: Drift Radar & Governance** (Week 4)

**Objective**: Deploy drift radar scanner, classifier, and governance adaptor.

#### Step 3.1: Deploy Drift Scanner

**Precondition**: Phase 2 complete

**Action**:
```bash
# Copy drift scanner module
cp backend/ledger/drift/scanner.py /path/to/production/backend/ledger/drift/

# Verify import
python3 -c "from backend.ledger.drift.scanner import DriftScanner; print('OK')"
```

**Expected Output**: `OK`

---

#### Step 3.2: Deploy Drift Classifier

**Precondition**: Step 3.1 complete

**Action**:
```bash
# Copy drift classifier module
cp backend/ledger/drift/classifier.py /path/to/production/backend/ledger/drift/

# Verify import
python3 -c "from backend.ledger.drift.classifier import DriftClassifier; print('OK')"
```

**Expected Output**: `OK`

---

#### Step 3.3: Deploy Governance Adaptor

**Precondition**: Step 3.2 complete

**Action**:
```bash
# Copy governance adaptor module
cp backend/ledger/drift/governance.py /path/to/production/backend/ledger/drift/

# Verify import
python3 -c "from backend.ledger.drift.governance import create_governance_adaptor; print('OK')"
```

**Expected Output**: `OK`

**Verification**:
```bash
# Test governance adaptor
python3 -c "
from backend.ledger.drift.governance import create_governance_adaptor
adaptor = create_governance_adaptor('strict')
drift_signals = [
    {'type': 'SCHEMA_DRIFT', 'severity': 'CRITICAL', 'message': 'Test'}
]
evidence_pack = adaptor.evaluate_drift_signals(drift_signals)
print('Governance signal:', evidence_pack.signal.value)
"
```

**Expected Output**: `Governance signal: BLOCK`

---

### **Phase 4: CI Helper Scripts** (Week 5)

**Objective**: Implement CI helper scripts for governance chain.

#### Step 4.1: Implement Validation Scripts

**Precondition**: Phase 3 complete

**Action**:
```bash
# Implement scripts (see docs/ci/implementation_guide.md for specifications)
# 1. validate_migrations.py
# 2. check_replay_success_rate.py
# 3. check_governance_signal.py
# 4. attestation_integrity_sweep.py
# 5. check_integrity_violations.py
# 6. check_pq_migration_code.py
# 7. aggregate_governance_results.py
# 8. check_governance_gate.py
# 9. load_test_data.py

# Test each script individually
python3 scripts/ci/validate_migrations.py --help
```

**Expected Output**: Help text for each script

---

#### Step 4.2: Implement Drift Radar Scan Script

**Precondition**: Step 4.1 complete

**Action**:
```bash
# Implement drift_radar_scan.py (see readiness_gate_checklist.md)
# Test script
python3 scripts/ci/drift_radar_scan.py \
  --database-url $TEST_DATABASE_URL \
  --start-block 0 \
  --end-block 100 \
  --scan-types schema,hash-delta \
  --governance-policy strict \
  --output drift_signals.json \
  --evidence-pack evidence_pack.json
```

**Expected Output**: Drift signals and evidence pack written

---

### **Phase 5: CI Chain Integration** (Week 6)

**Objective**: Deploy governance chain to CI environment.

#### Step 5.1: Test Governance Chain Locally

**Precondition**: Phase 4 complete

**Action**:
```bash
# Set environment variables
export DATABASE_URL="postgresql://postgres:test@localhost:5432/mathledger_test"
export START_BLOCK=0
export END_BLOCK=1000
export REPLAY_MODE="sliding_window"
export GOVERNANCE_POLICY="strict"
export OUTPUT_DIR="./reports"

# Run governance chain
./scripts/ci/run_governance_chain.sh
```

**Expected Output**: All 7 jobs pass

---

#### Step 5.2: Deploy to CI Environment

**Precondition**: Step 5.1 complete

**Action**:
```bash
# Option 1: GitHub Actions (if permissions available)
cp docs/ci/governance-chain.yml .github/workflows/

# Option 2: GitLab CI
cp docs/ci/governance-chain.yml .gitlab-ci.yml

# Option 3: CircleCI
cp docs/ci/governance-chain.yml .circleci/config.yml

# Commit and push
git add .github/workflows/governance-chain.yml  # or .gitlab-ci.yml, etc.
git commit -m "feat(ci): Add governance chain workflow"
git push origin main
```

**Expected Output**: CI pipeline triggered

---

#### Step 5.3: Monitor First CI Run

**Precondition**: Step 5.2 complete

**Action**:
```bash
# Monitor CI run
gh run watch

# Check reports
gh run download --name governance-reports
ls reports/
```

**Expected Output**: All 7 jobs pass, reports generated

---

### **Phase 6: Production Validation** (Week 7)

**Objective**: Validate governance chain on production data (read-only).

#### Step 6.1: Run Full-Chain Replay on Production

**Precondition**: Phase 5 complete

**Action**:
```bash
# Run full-chain replay (read-only)
python3 scripts/replay_verify.py \
  --database-url $PRODUCTION_DATABASE_URL_READONLY \
  --mode full_chain \
  --start-block 0 \
  --end-block 100000 \
  --output production_replay_report.json
```

**Expected Output**: 100% replay success rate

---

#### Step 6.2: Run Drift Radar on Production

**Precondition**: Step 6.1 complete

**Action**:
```bash
# Run drift radar (read-only)
python3 scripts/ci/drift_radar_scan.py \
  --database-url $PRODUCTION_DATABASE_URL_READONLY \
  --start-block 0 \
  --end-block 100000 \
  --scan-types schema,hash-delta,metadata,statement \
  --governance-policy strict \
  --output production_drift_signals.json \
  --evidence-pack production_evidence_pack.json
```

**Expected Output**: Governance signal OK or WARN (not BLOCK)

---

#### Step 6.3: Generate Production Audit Report

**Precondition**: Steps 6.1-6.2 complete

**Action**:
```bash
# Generate audit report
python3 scripts/generate_audit_report.py \
  --replay-report production_replay_report.json \
  --drift-report production_drift_signals.json \
  --evidence-pack production_evidence_pack.json \
  --output production_audit_report.pdf
```

**Expected Output**: PDF audit report generated

---

## Rollback Procedures

### Rollback Phase 2 (Replay Engine Integration)

**Scenario**: Integration tests fail after applying diffs.

**Action**:
```bash
# Restore backups
cp backend/ledger/replay/recompute.py.backup backend/ledger/replay/recompute.py
cp backend/ledger/replay/checker.py.backup backend/ledger/replay/checker.py
cp backend/ledger/replay/engine.py.backup backend/ledger/replay/engine.py

# Verify rollback
python3 -m pytest tests/integration/test_replay_engine.py -v
```

**Expected Output**: All tests pass (original behavior restored)

---

### Rollback Phase 5 (CI Chain Integration)

**Scenario**: CI pipeline fails after deployment.

**Action**:
```bash
# Remove workflow file
git rm .github/workflows/governance-chain.yml
git commit -m "revert: Remove governance chain workflow (rollback)"
git push origin main
```

**Expected Output**: CI pipeline no longer runs

---

## Critical Path

**Critical Path** (minimum time to production):

1. Phase 0: Foundation (1 week) - **CRITICAL**
2. Phase 1: Consensus Runtime (1 week) - **CRITICAL**
3. Phase 2: Replay Engine Integration (1 week) - **CRITICAL**
4. Phase 3: Drift Radar & Governance (1 week) - **CRITICAL**
5. Phase 4: CI Helper Scripts (1 week) - **CRITICAL**
6. Phase 5: CI Chain Integration (1 week) - **CRITICAL**
7. Phase 6: Production Validation (1 week) - **OPTIONAL** (can be done in parallel with Phase 5)

**Total Critical Path**: 6 weeks (7 weeks with production validation)

---

## Parallelization Opportunities

**Parallel Track 1**: Phase 0 + Phase 1 (if Manus-H integration is complete)

**Parallel Track 2**: Phase 4 (CI Helper Scripts) can start during Phase 3

**Parallel Track 3**: Phase 6 (Production Validation) can run in parallel with Phase 5

**Optimized Timeline**: 5 weeks (with parallelization)

---

## Sign-Off

**Integration Engineer**: _________________ Date: _______

**QA Engineer**: _________________ Date: _______

**Release Manager**: _________________ Date: _______

---

**"Keep it blue, keep it clean, keep it sealed."**

— Manus-B, Ledger Integrity & PQ Migration Engineer
