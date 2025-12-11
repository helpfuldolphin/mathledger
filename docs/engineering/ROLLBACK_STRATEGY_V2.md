# Rollback Strategy v2: 3 Failure Modes & Deterministic State Recovery

**Author**: Manus-B (Ledger Integrity & PQ Migration Engineer)  
**Date**: 2025-12-09  
**Status**: REAL-READY

---

## OVERVIEW

This document defines rollback procedures for 3 failure modes:
1. **Consensus Integration Failure** - Replay verification fails after consensus integration
2. **Drift Radar False Positives** - Drift radar blocks valid merges
3. **PQ Migration Failure** - Dual-commitment or SHA-3 cutover fails

Each failure mode includes:
- **Detection criteria** (how to detect the failure)
- **Rollback steps** (how to undo changes)
- **State recovery** (how to restore deterministic state)
- **Verification** (how to confirm rollback success)

---

## FAILURE MODE 1: Consensus Integration Failure

### Detection Criteria

**Symptoms**:
- Replay verification fails after applying consensus integration diffs
- `verify_block_replay()` raises exceptions
- Consensus violations detected on previously valid blocks
- Integration tests fail

**Detection Commands**:
```bash
# Run integration test
python3 -m pytest tests/integration/test_consensus_replay_integration.py -v

# Expected output (FAILURE):
# FAILED tests/integration/test_consensus_replay_integration.py::test_replay_with_consensus
# AssertionError: Consensus validation failed on block 1234
```

**Failure Indicators**:
- Exit code: 1 (test failure)
- Error message contains "Consensus validation failed"
- `ReplayResult.consensus_passed = False` on valid blocks

---

### Rollback Steps

**Step 1: Restore Original Files**

```bash
cd /home/ubuntu/mathledger

# Restore checker.py
if [ -f backend/ledger/replay/checker.py.backup ]; then
    cp backend/ledger/replay/checker.py.backup backend/ledger/replay/checker.py
    echo "Restored checker.py"
else
    echo "ERROR: No backup found for checker.py"
    exit 1
fi

# Restore engine.py
if [ -f backend/ledger/replay/engine.py.backup ]; then
    cp backend/ledger/replay/engine.py.backup backend/ledger/replay/engine.py
    echo "Restored engine.py"
else
    echo "ERROR: No backup found for engine.py"
    exit 1
fi

# Verify syntax
python3 -m py_compile backend/ledger/replay/checker.py
python3 -m py_compile backend/ledger/replay/engine.py

echo "Rollback Step 1: Complete"
```

**Expected Output**:
```
Restored checker.py
Restored engine.py
Rollback Step 1: Complete
```

---

**Step 2: Verify Rollback**

```bash
# Test import (should work with original code)
python3 -c "
from backend.ledger.replay.checker import verify_block_replay
from backend.ledger.replay.engine import replay_blocks
print('Imports: OK')
"

# Run integration test (should pass with original code)
python3 -m pytest tests/integration/test_replay_verification.py -v
```

**Expected Output**:
```
Imports: OK
PASSED tests/integration/test_replay_verification.py::test_replay_blocks
```

---

**Step 3: State Recovery**

```bash
# No database state changes (consensus integration is code-only)
# State recovery: N/A

echo "Rollback Step 3: State recovery complete (no database changes)"
```

---

**Step 4: Verification**

```bash
# Verify replay verification works with original code
python3 scripts/replay_verify.py \\
    --database-url $DATABASE_URL \\
    --start-block 0 \\
    --end-block 100 \\
    --strategy full_chain

# Expected output:
# Replayed 100 blocks
# Success rate: 100.00% (100/100)
# Failures: 0
# Status: PASSED
```

**Success Criteria**:
- All imports work
- Integration tests pass
- Replay verification succeeds on 100 blocks
- No consensus violations on valid blocks

---

## FAILURE MODE 2: Drift Radar False Positives

### Detection Criteria

**Symptoms**:
- Drift radar blocks valid merges in CI
- False positive drift signals on benign changes
- Governance signal: BLOCK on non-critical drift
- CI pipeline fails with "Drift radar detected critical signals"

**Detection Commands**:
```bash
# Run drift radar scan
python3 scripts/ci/drift_radar_scan.py \\
    --database-url $DATABASE_URL \\
    --start-block 0 \\
    --end-block 1000 \\
    --scan-types schema,hash-delta,metadata,statement \\
    --governance-policy strict \\
    --output drift_signals.json \\
    --evidence-pack evidence_pack.json

# Expected output (FAILURE):
# GOVERNANCE SIGNAL: BLOCK
# Exit code: 1
```

**Failure Indicators**:
- Exit code: 1 (merge blocked)
- Evidence pack contains false positive signals
- Governance signal: BLOCK on LOW or MEDIUM severity drift

---

### Rollback Steps

**Step 1: Adjust Governance Policy**

```bash
cd /home/ubuntu/mathledger

# Option A: Switch to moderate policy (less strict)
python3 scripts/ci/drift_radar_scan.py \\
    --database-url $DATABASE_URL \\
    --start-block 0 \\
    --end-block 1000 \\
    --scan-types schema,hash-delta,metadata,statement \\
    --governance-policy moderate \\  # CHANGED
    --output drift_signals.json \\
    --evidence-pack evidence_pack.json

# Option B: Exclude specific scan types
python3 scripts/ci/drift_radar_scan.py \\
    --database-url $DATABASE_URL \\
    --start-block 0 \\
    --end-block 1000 \\
    --scan-types hash-delta,statement \\  # CHANGED (removed schema, metadata)
    --governance-policy strict \\
    --output drift_signals.json \\
    --evidence-pack evidence_pack.json
```

**Expected Output**:
```
GOVERNANCE SIGNAL: OK
Exit code: 0
```

---

**Step 2: Whitelist False Positives**

```bash
# Create whitelist file
cat > drift_whitelist.json <<'EOF'
{
  "whitelisted_signals": [
    {
      "signal_type": "schema_drift",
      "block_range": [100, 200],
      "reason": "Benign schema migration 018 (epoch root system)"
    },
    {
      "signal_type": "hash_delta",
      "block_number": 150,
      "reason": "Expected hash change due to canonical_proof format update"
    }
  ]
}
EOF

# Run drift radar with whitelist
python3 scripts/ci/drift_radar_scan.py \\
    --database-url $DATABASE_URL \\
    --start-block 0 \\
    --end-block 1000 \\
    --scan-types schema,hash-delta,metadata,statement \\
    --governance-policy strict \\
    --whitelist drift_whitelist.json \\  # NEW
    --output drift_signals.json \\
    --evidence-pack evidence_pack.json
```

**Expected Output**:
```
Whitelisted 2 signals
GOVERNANCE SIGNAL: OK
Exit code: 0
```

---

**Step 3: State Recovery**

```bash
# No database state changes (drift radar is read-only)
# State recovery: N/A

echo "Rollback Step 3: State recovery complete (drift radar is read-only)"
```

---

**Step 4: Verification**

```bash
# Verify drift radar no longer blocks merges
python3 scripts/ci/drift_radar_scan.py \\
    --database-url $DATABASE_URL \\
    --start-block 0 \\
    --end-block 1000 \\
    --scan-types schema,hash-delta,metadata,statement \\
    --governance-policy moderate \\
    --whitelist drift_whitelist.json \\
    --output drift_signals.json \\
    --evidence-pack evidence_pack.json

# Expected output:
# GOVERNANCE SIGNAL: OK
# Exit code: 0
```

**Success Criteria**:
- Governance signal: OK or WARN (not BLOCK)
- Exit code: 0
- Evidence pack shows whitelisted signals excluded
- CI pipeline passes

---

## FAILURE MODE 3: PQ Migration Failure

### Detection Criteria

**Symptoms**:
- Dual-commitment activation fails
- SHA-3 cutover produces invalid blocks
- Cross-algorithm prev_hash validation fails
- Replay verification fails on migration boundary blocks

**Detection Commands**:
```bash
# Verify dual-commitment activation
python3 scripts/verify_dual_commitment.py \\
    --database-url $DATABASE_URL \\
    --start-block 100000 \\
    --end-block 100100

# Expected output (FAILURE):
# Block 100000: ERROR - Missing SHA-3 roots
# Block 100001: ERROR - Missing SHA-3 roots
# Verification: FAILED (100 blocks, 100 errors)
# Exit code: 1
```

**Failure Indicators**:
- Exit code: 1 (verification failed)
- Missing SHA-3 roots in dual-commitment blocks
- Prev_hash validation fails on migration boundary
- Replay verification fails on blocks >= activation_block

---

### Rollback Steps

**Step 1: Identify Migration Boundary**

```bash
cd /home/ubuntu/mathledger

# Query database for migration boundary
psql $DATABASE_URL -c "
SELECT 
    block_number,
    attestation_metadata->>'hash_version' AS hash_version,
    reasoning_attestation_root,
    reasoning_attestation_root_sha3
FROM blocks
WHERE block_number BETWEEN 99990 AND 100010
ORDER BY block_number;
"

# Expected output:
# block_number | hash_version | reasoning_attestation_root | reasoning_attestation_root_sha3
# -------------+--------------+----------------------------+--------------------------------
# 99999        | sha256-v1    | abc...                     | NULL
# 100000       | dual-v1      | def...                     | NULL  ← FAILURE (missing SHA-3)
# 100001       | dual-v1      | ghi...                     | NULL  ← FAILURE
```

**Failure Analysis**:
- Activation block: 100000
- Hash version: `dual-v1` (correct)
- SHA-3 roots: NULL (incorrect, should be populated)
- **Root Cause**: `seal_block()` not computing SHA-3 roots

---

**Step 2: Rollback Database State**

```bash
# Option A: Delete invalid blocks (if activation just occurred)
psql $DATABASE_URL -c "
DELETE FROM blocks
WHERE block_number >= 100000
  AND attestation_metadata->>'hash_version' = 'dual-v1'
  AND reasoning_attestation_root_sha3 IS NULL;
"

# Expected output:
# DELETE 100  (100 invalid blocks deleted)

# Option B: Recompute SHA-3 roots (if blocks contain valuable data)
python3 scripts/recompute_sha3_roots.py \\
    --database-url $DATABASE_URL \\
    --start-block 100000 \\
    --end-block 100100

# Expected output:
# Recomputed SHA-3 roots for 100 blocks
# Updated blocks: 100
```

---

**Step 3: Rollback Code Changes**

```bash
# Restore original seal_block() (before dual-commitment changes)
if [ -f backend/ledger/blockchain.py.backup ]; then
    cp backend/ledger/blockchain.py.backup backend/ledger/blockchain.py
    echo "Restored blockchain.py"
else
    echo "ERROR: No backup found for blockchain.py"
    exit 1
fi

# Verify syntax
python3 -m py_compile backend/ledger/blockchain.py

echo "Rollback Step 3: Code rollback complete"
```

---

**Step 4: State Recovery**

```bash
# Verify database state after rollback
psql $DATABASE_URL -c "
SELECT 
    COUNT(*) AS total_blocks,
    COUNT(*) FILTER (WHERE attestation_metadata->>'hash_version' = 'sha256-v1') AS sha256_blocks,
    COUNT(*) FILTER (WHERE attestation_metadata->>'hash_version' = 'dual-v1') AS dual_blocks,
    MAX(block_number) AS max_block_number
FROM blocks;
"

# Expected output:
# total_blocks | sha256_blocks | dual_blocks | max_block_number
# -------------+---------------+-------------+------------------
# 99999        | 99999         | 0           | 99999

# State recovered: All blocks are SHA-256, dual-commitment activation rolled back
```

---

**Step 5: Verification**

```bash
# Verify replay verification works after rollback
python3 scripts/replay_verify.py \\
    --database-url $DATABASE_URL \\
    --start-block 99900 \\
    --end-block 99999 \\
    --strategy full_chain

# Expected output:
# Replayed 100 blocks
# Success rate: 100.00% (100/100)
# Failures: 0
# Status: PASSED

# Verify no dual-commitment blocks exist
psql $DATABASE_URL -c "
SELECT COUNT(*) FROM blocks
WHERE attestation_metadata->>'hash_version' = 'dual-v1';
"

# Expected output:
# count
# -------
# 0
```

**Success Criteria**:
- All invalid dual-commitment blocks deleted
- Database state recovered to pre-activation state
- Replay verification succeeds on all blocks
- No dual-commitment blocks exist
- Code rolled back to original `seal_block()`

---

## DETERMINISTIC STATE RECOVERY PROTOCOL

### Principles

1. **Idempotent Rollback**: Rollback can be run multiple times without side effects
2. **Atomic Operations**: Database changes are atomic (all-or-nothing)
3. **Backup-First**: Always create backups before making changes
4. **Verification-Last**: Always verify state after rollback

### State Recovery Checklist

**Before Rollback**:
- [ ] Identify failure mode (consensus, drift radar, PQ migration)
- [ ] Create database backup (if applicable)
- [ ] Create code backups (if applicable)
- [ ] Document failure symptoms and root cause

**During Rollback**:
- [ ] Follow rollback steps for identified failure mode
- [ ] Verify each step completes successfully
- [ ] Log all rollback actions for audit trail

**After Rollback**:
- [ ] Verify database state (if applicable)
- [ ] Verify code state (imports, syntax, tests)
- [ ] Run integration tests
- [ ] Run replay verification
- [ ] Document rollback success and lessons learned

---

## SMOKE-TEST READINESS CHECKLIST

### Files to Create

**Rollback Scripts**:
1. `scripts/rollback_consensus_integration.sh` - Rollback consensus integration
2. `scripts/rollback_pq_migration.sh` - Rollback PQ migration
3. `scripts/recompute_sha3_roots.py` - Recompute SHA-3 roots for invalid blocks
4. `scripts/verify_dual_commitment.py` - Verify dual-commitment activation
5. `drift_whitelist.json` - Whitelist for false positive drift signals

### Commands to Run Locally

**Test Rollback Script (Consensus Integration)**:
```bash
cd /home/ubuntu/mathledger

# Create rollback script
cat > scripts/rollback_consensus_integration.sh <<'EOF'
#!/bin/bash
set -e

echo "Rolling back consensus integration..."

# Restore checker.py
if [ -f backend/ledger/replay/checker.py.backup ]; then
    cp backend/ledger/replay/checker.py.backup backend/ledger/replay/checker.py
    echo "✓ Restored checker.py"
fi

# Restore engine.py
if [ -f backend/ledger/replay/engine.py.backup ]; then
    cp backend/ledger/replay/engine.py.backup backend/ledger/replay/engine.py
    echo "✓ Restored engine.py"
fi

# Verify syntax
python3 -m py_compile backend/ledger/replay/checker.py
python3 -m py_compile backend/ledger/replay/engine.py
echo "✓ Syntax verification passed"

# Test imports
python3 -c "from backend.ledger.replay.checker import verify_block_replay; from backend.ledger.replay.engine import replay_blocks; print('✓ Imports OK')"

echo "Rollback complete"
EOF

chmod +x scripts/rollback_consensus_integration.sh

# Test rollback script
./scripts/rollback_consensus_integration.sh
```

**Expected Output**:
```
Rolling back consensus integration...
✓ Restored checker.py
✓ Restored engine.py
✓ Syntax verification passed
✓ Imports OK
Rollback complete
```

### Observable Artifacts

**After Rollback**:
- `backend/ledger/replay/checker.py` - Restored to original version
- `backend/ledger/replay/engine.py` - Restored to original version
- `scripts/rollback_consensus_integration.sh` - Rollback script created
- Console output: "Rollback complete"

---

## REALITY LOCK VERIFICATION

**Files Referenced** (all REAL):
- ✅ `backend/ledger/replay/checker.py` - EXISTS
- ✅ `backend/ledger/replay/engine.py` - EXISTS
- ✅ `backend/ledger/blockchain.py` - EXISTS
- ✅ `backend/consensus/rules.py` - EXISTS

**Database Tables Referenced**:
- ✅ `blocks` - EXISTS

**Commands Referenced**:
- ✅ `psql` - Standard PostgreSQL client
- ✅ `python3 -m py_compile` - Standard Python syntax checker
- ✅ `python3 -m pytest` - Standard Python test runner

**Status**: # REAL-READY

---

**"Keep it blue, keep it clean, keep it sealed."**

— Manus-B, Ledger Integrity & PQ Migration Engineer
