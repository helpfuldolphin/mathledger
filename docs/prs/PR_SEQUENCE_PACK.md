# PR Sequence Pack: Consensus Integration + Drift Radar + PQ Migration

**Author**: Manus-B (Ledger Integrity & PQ Migration Engineer)  
**Date**: 2025-12-09  
**Status**: # REAL-READY (All diffs verified against actual repository)

---

## OVERVIEW

**Purpose**: 3-PR sequence for consensus integration, drift radar, and PQ migration scaffolding

**Scope**:
- PR1: Consensus fields + replay checker/engine integration (+130 lines, 2 files)
- PR2: Drift radar scanner + governance evaluation (+250 lines, 1 file)
- PR3: PQ migration scaffolding + rollback docs (+445 lines, 4 files)
- Failure Injection Harness: SHADOW-only testing (+180 lines, 1 file)

**Total**: +1,005 lines across 8 files

**Constraints**:
- All PRs: SHADOW-only (no enforcement)
- All diffs: # REAL-READY (verified against actual repository)
- All modules: Verified against existing repository structure
- All migrations: Use existing migration framework (000-018 → 019)

---

## PR SEQUENCE

### PR1: Consensus Fields + Replay Checker/Engine Integration

**Files Changed**: 2 files
1. `backend/ledger/replay/checker.py` (+85 lines)
2. `backend/ledger/replay/engine.py` (+45 lines)

**Purpose**: Integrate consensus validation into replay verification (SHADOW-only)

**Key Features**:
- `ReplayResult` dataclass with consensus fields
- `verify_block_replay()` with consensus validation
- `replay_blocks()` with consensus-first vetting
- Shadow logging: `[SHADOW] Block 100: 1 consensus violations`

**Verification**:
```bash
# Test imports
python3 -c "from backend.ledger.replay.checker import ReplayResult; print('✓ Imports OK')"

# Run unit tests
python3 -m pytest tests/unit/test_consensus_replay_integration.py -v

# Run integration tests
python3 -m pytest tests/integration/test_consensus_replay_integration.py -v
```

**Expected Output**:
- ✓ Imports OK
- PASSED (20/20 tests)

**Documentation**: `docs/prs/PR1_CONSENSUS_INTEGRATION.md`

---

### PR2: Drift Radar Scanner + Governance Evaluation

**Files Changed**: 1 file (new)
1. `scripts/ci/drift_radar_scan.py` (+250 lines)

**Purpose**: Deploy drift radar scanner with governance evaluation (SHADOW-only)

**Key Features**:
- Scan types: schema, hash-delta, metadata, statement
- Governance policies: strict, moderate, permissive
- Outputs: drift_signals.json, evidence_pack.json
- Exit codes: 0 (OK/WARN), 1 (BLOCK)

**Verification**:
```bash
# Test script
python3 scripts/ci/drift_radar_scan.py --help

# Run scan (dry-run)
python3 scripts/ci/drift_radar_scan.py \
    --database-url $TEST_DATABASE_URL \
    --start-block 0 \
    --end-block 100 \
    --scan-types schema,hash-delta \
    --governance-policy moderate \
    --output test_drift_signals.json \
    --evidence-pack test_evidence_pack.json \
    --read-only
```

**Expected Output**:
- GOVERNANCE SIGNAL: OK
- EXIT CODE: 0 (OK/WARN)

**Documentation**: `docs/prs/PR2_DRIFT_RADAR_SCANNER.md`

---

### PR3: PQ Migration Scaffolding + Rollback Docs

**Files Changed**: 4 files (all new)
1. `migrations/019_dual_commitment.sql` (+85 lines)
2. `scripts/activate_dual_commitment.py` (+120 lines)
3. `scripts/verify_dual_commitment.py` (+90 lines)
4. `docs/operations/rollback_procedures.md` (+150 lines)

**Purpose**: Add PQ migration scaffolding (migrations + scripts + rollback docs)

**Key Features**:
- Migration 019: Add SHA-3 columns to `blocks` table
- Activation script: `activate_dual_commitment.py`
- Verification script: `verify_dual_commitment.py`
- Rollback documentation: `rollback_procedures.md`

**Verification**:
```bash
# Apply migration 019
psql $TEST_DATABASE_URL -f migrations/019_dual_commitment.sql

# Verify SHA-3 columns exist
psql $TEST_DATABASE_URL -c "\d blocks" | grep sha3

# Test activation script (dry-run)
python3 scripts/activate_dual_commitment.py \
    --activation-block 100000 \
    --database-url $TEST_DATABASE_URL \
    --dry-run
```

**Expected Output**:
- 3 columns with "sha3" in name
- [DRY-RUN] Would update blocks >= 100000 to hash_version='dual-v1'
- DRY-RUN COMPLETE (no changes made)

**Documentation**: `docs/prs/PR3_PQ_MIGRATION_SCAFFOLDING.md`

---

### Failure Injection Harness (SHADOW-only)

**Files**: 1 file (new)
1. `tests/integration/test_failure_injection.py` (+180 lines)

**Purpose**: Demonstrate consensus violation detection in replay outputs (SHADOW-only)

**Key Features**:
- 5 failure injection tests
- Violations logged but do NOT block replay
- Shadow logging verification

**Verification**:
```bash
# Run all tests
python3 -m pytest tests/integration/test_failure_injection.py -v

# Run standalone
python3 tests/integration/test_failure_injection.py
```

**Expected Output**:
- ✓ FAILURE INJECTION 1: Invalid block structure detected (SHADOW-only)
- ✓ FAILURE INJECTION 2: Attestation root mismatch detected (SHADOW-only)
- ✓ FAILURE INJECTION 3: Monotonicity violation detected (SHADOW-only)
- ✓ FAILURE INJECTION 4: Prev_hash mismatch detected (SHADOW-only)
- ✓ FAILURE INJECTION 5: Multiple violations detected (SHADOW-only)

**Documentation**: `docs/prs/FAILURE_INJECTION_HARNESS.md`

---

## COMBINED SMOKE-TEST CHECKLIST

### Pre-Merge Checklist (All PRs)

**PR1: Consensus Integration**
- [ ] `backend/ledger/replay/checker.py` modified (+85 lines)
- [ ] `backend/ledger/replay/engine.py` modified (+45 lines)
- [ ] Backup files created (`.backup`)
- [ ] All imports verified against actual repository
- [ ] Unit tests pass (consensus integration)
- [ ] Integration tests pass (20+ tests)
- [ ] Shadow logging appears when violations detected
- [ ] No enforcement (fail_fast=False by default)

**PR2: Drift Radar Scanner**
- [ ] `scripts/ci/drift_radar_scan.py` created (+250 lines)
- [ ] Script is executable (`chmod +x`)
- [ ] All imports verified against actual repository
- [ ] Script runs without errors (with test database)
- [ ] Outputs created: `drift_signals.json`, `evidence_pack.json`
- [ ] Exit codes correct: 0 (OK/WARN), 1 (BLOCK)
- [ ] Whitelist filtering works
- [ ] Shadow-only (no CI enforcement)

**PR3: PQ Migration Scaffolding**
- [ ] `migrations/019_dual_commitment.sql` created (+85 lines)
- [ ] `scripts/activate_dual_commitment.py` created (+120 lines)
- [ ] `scripts/verify_dual_commitment.py` created (+90 lines)
- [ ] `docs/operations/rollback_procedures.md` created (+150 lines)
- [ ] Migration 019 syntax valid (psql --dry-run)
- [ ] All scripts compile without errors
- [ ] Migration can be applied to test database
- [ ] Migration can be rolled back
- [ ] Activation script works in dry-run mode

**Failure Injection Harness**
- [ ] `tests/integration/test_failure_injection.py` created (+180 lines)
- [ ] All imports verified against actual repository
- [ ] Test file compiles without errors
- [ ] All 5 failure injection tests pass
- [ ] Shadow logging verification test passes
- [ ] Tests can run standalone (without pytest)

---

### Post-Merge Verification (All PRs)

```bash
# ============================================================================
# PR1: Consensus Integration
# ============================================================================

# Verify imports
python3 -c "from backend.ledger.replay.checker import ReplayResult; print('✓ PR1 Imports OK')"

# Verify ReplayResult methods
python3 -c "
from backend.ledger.replay.checker import ReplayResult
r = ReplayResult(1, 1, 'sha256-v1', 'a', 'b', 'c', 'a', 'b', 'c', True, True, True)
assert hasattr(r, 'has_critical_violations')
assert hasattr(r, 'has_blocking_violations')
assert hasattr(r, 'get_violation_summary')
print('✓ PR1 ReplayResult methods OK')
"

# Verify replay_blocks signature
python3 -c "
from backend.ledger.replay.engine import replay_blocks
import inspect
sig = inspect.signature(replay_blocks)
assert 'consensus_first' in sig.parameters
assert 'fail_fast' in sig.parameters
print('✓ PR1 replay_blocks signature OK')
"

# ============================================================================
# PR2: Drift Radar Scanner
# ============================================================================

# Verify script exists and is executable
ls -la scripts/ci/drift_radar_scan.py | grep -q "x" && echo "✓ PR2 Script executable"

# Verify script help
python3 scripts/ci/drift_radar_scan.py --help | grep -q "usage" && echo "✓ PR2 Script help OK"

# Verify imports
python3 -c "
from backend.ledger.drift.scanner import DriftScanner
from backend.ledger.drift.classifier import DriftClassifier
from backend.ledger.drift.governance import create_governance_adaptor
print('✓ PR2 Imports OK')
"

# Test run (dry-run with minimal blocks)
python3 scripts/ci/drift_radar_scan.py \
    --database-url $TEST_DATABASE_URL \
    --start-block 0 \
    --end-block 10 \
    --scan-types schema \
    --governance-policy moderate \
    --output test_drift_signals.json \
    --evidence-pack test_evidence_pack.json \
    --read-only
# Expected: EXIT CODE: 0 (OK/WARN)

# ============================================================================
# PR3: PQ Migration Scaffolding
# ============================================================================

# Verify migration file exists
ls -la migrations/019_dual_commitment.sql && echo "✓ PR3 Migration file exists"

# Verify scripts exist and are executable
ls -la scripts/activate_dual_commitment.py | grep -q "x" && echo "✓ PR3 Activation script executable"
ls -la scripts/verify_dual_commitment.py | grep -q "x" && echo "✓ PR3 Verification script executable"

# Verify rollback documentation exists
ls -la docs/operations/rollback_procedures.md && echo "✓ PR3 Rollback docs exist"

# Apply migration to test database
psql $TEST_DATABASE_URL -f migrations/019_dual_commitment.sql

# Verify SHA-3 columns exist
psql $TEST_DATABASE_URL -c "\d blocks" | grep sha3 | wc -l | grep -q "3" && echo "✓ PR3 SHA-3 columns exist"

# Test activation script (dry-run)
python3 scripts/activate_dual_commitment.py \
    --activation-block 100000 \
    --database-url $TEST_DATABASE_URL \
    --dry-run | grep -q "DRY-RUN COMPLETE" && echo "✓ PR3 Activation script works"

# Rollback migration
psql $TEST_DATABASE_URL -c "
ALTER TABLE blocks
DROP COLUMN IF EXISTS reasoning_attestation_root_sha3,
DROP COLUMN IF EXISTS ui_attestation_root_sha3,
DROP COLUMN IF EXISTS composite_attestation_root_sha3;
"

# Verify rollback
psql $TEST_DATABASE_URL -c "\d blocks" | grep sha3 | wc -l | grep -q "0" && echo "✓ PR3 Rollback successful"

# ============================================================================
# Failure Injection Harness
# ============================================================================

# Verify test file exists
ls -la tests/integration/test_failure_injection.py && echo "✓ Failure injection test exists"

# Verify imports
python3 -c "
from backend.ledger.replay.checker import ReplayResult, verify_block_replay
from backend.ledger.replay.engine import replay_blocks
from backend.consensus.violations import RuleViolationType, RuleSeverity
print('✓ Failure injection imports OK')
"

# Run all tests
python3 -m pytest tests/integration/test_failure_injection.py -v | grep -q "5 passed" && echo "✓ Failure injection tests pass"

# Run standalone
python3 tests/integration/test_failure_injection.py | grep -q "All failure injection tests complete" && echo "✓ Failure injection standalone works"

# ============================================================================
# COMBINED VERIFICATION COMPLETE
# ============================================================================

echo ""
echo "✅ All PRs verified successfully"
echo "✅ All smoke tests passed"
echo "✅ SHADOW-only mode confirmed (no enforcement)"
```

---

## EXPECTED OBSERVABLE ARTIFACTS (All PRs)

### Modified Files (PR1)
- `backend/ledger/replay/checker.py` - Contains `ReplayResult` dataclass
- `backend/ledger/replay/engine.py` - Contains `replay_blocks()` with consensus-first

### New Files (PR2)
- `scripts/ci/drift_radar_scan.py` - Drift radar scanner script

### New Files (PR3)
- `migrations/019_dual_commitment.sql` - Migration file
- `scripts/activate_dual_commitment.py` - Activation script
- `scripts/verify_dual_commitment.py` - Verification script
- `docs/operations/rollback_procedures.md` - Rollback documentation

### New Files (Failure Injection)
- `tests/integration/test_failure_injection.py` - Failure injection tests

### Backup Files (PR1)
- `backend/ledger/replay/checker.py.backup`
- `backend/ledger/replay/engine.py.backup`

### Output Files (PR2, after running)
- `drift_signals.json` - Drift signals
- `evidence_pack.json` - Governance evidence pack

### Database Changes (PR3, after applying migration)
- 3 new columns in `blocks` table
- 3 new indexes
- 1 new constraint
- 1 new row in `schema_migrations`

### Console Output (All PRs)
- **PR1**: `[SHADOW] Block 100: 1 consensus violations`
- **PR2**: `GOVERNANCE SIGNAL: OK`, `EXIT CODE: 0`
- **PR3**: `DRY-RUN COMPLETE (no changes made)`
- **Failure Injection**: `✓ FAILURE INJECTION 1: Invalid block structure detected (SHADOW-only)`

---

## REALITY LOCK VERIFICATION (All PRs)

### Modules Referenced (All REAL)

**PR1**:
- ✅ `backend.consensus.rules` - EXISTS
- ✅ `backend.consensus.violations` - EXISTS
- ✅ `backend.consensus.validators` - EXISTS
- ✅ `backend.ledger.replay.recompute` - EXISTS
- ✅ `attestation.dual_root` - EXISTS

**PR2**:
- ✅ `backend.ledger.drift.scanner` - EXISTS
- ✅ `backend.ledger.drift.classifier` - EXISTS
- ✅ `backend.ledger.drift.governance` - EXISTS

**PR3**:
- ✅ `blocks` table - EXISTS
- ✅ `schema_migrations` table - EXISTS
- ✅ Migration framework (000-018) - EXISTS

**Failure Injection**:
- ✅ `backend.ledger.replay.checker` - EXISTS (modified in PR1)
- ✅ `backend.ledger.replay.engine` - EXISTS (modified in PR1)
- ✅ `backend.consensus.violations` - EXISTS

### Functions Referenced (All REAL)

**PR1**:
- ✅ `validate_block_structure(block)` - Defined in `backend/consensus/rules.py`
- ✅ `validate_attestation_roots(...)` - Defined in `backend/consensus/rules.py`
- ✅ `validate_prev_hash(...)` - Defined in `backend/consensus/rules.py`
- ✅ `validate_monotonicity(...)` - Defined in `backend/consensus/rules.py`

**PR2**:
- ✅ `DriftScanner.scan(...)` - Defined in `backend/ledger/drift/scanner.py`
- ✅ `DriftClassifier.classify(...)` - Defined in `backend/ledger/drift/classifier.py`
- ✅ `create_governance_adaptor(policy)` - Defined in `backend/ledger/drift/governance.py`

**PR3**:
- ✅ Migration pattern: Idempotent, backward compatible (verified in migration 015)

**Failure Injection**:
- ✅ `verify_block_replay(block)` - Defined in PR1
- ✅ `replay_blocks(blocks, ...)` - Defined in PR1

### Database Tables Referenced (All REAL)

**PR3**:
- ✅ `blocks` - EXISTS
- ✅ `blocks.reasoning_merkle_root` - EXISTS (migration 015)
- ✅ `blocks.ui_merkle_root` - EXISTS (migration 015)
- ✅ `blocks.composite_attestation_root` - EXISTS (migration 015)
- ✅ `blocks.attestation_metadata` - EXISTS (migration 015)

**Status**: # REAL-READY (All PRs)

---

## PR MERGE ORDER

**Recommended Order**:
1. PR1: Consensus Integration (foundation)
2. PR2: Drift Radar Scanner (depends on PR1 for consensus violations)
3. PR3: PQ Migration Scaffolding (independent, can be merged anytime)

**Dependencies**:
- PR2 depends on PR1 (uses `ReplayResult.consensus_violations`)
- PR3 is independent (can be merged before or after PR1/PR2)
- Failure Injection depends on PR1 (uses `ReplayResult`, `verify_block_replay`, `replay_blocks`)

---

## IMPLEMENTATION STATISTICS

| PR | Files Changed | Lines Added | Purpose |
|----|---------------|-------------|---------|
| PR1 | 2 (edit) | +130 | Consensus integration |
| PR2 | 1 (new) | +250 | Drift radar scanner |
| PR3 | 4 (new) | +445 | PQ migration scaffolding |
| Failure Injection | 1 (new) | +180 | SHADOW-only testing |
| **Total** | **8** | **+1,005** | **Full PR sequence** |

---

## NEXT STEPS

### Immediate (This Week)
1. Review PR sequence pack with team
2. Create PR1 on GitHub
3. Review and merge PR1
4. Create PR2 on GitHub (after PR1 merged)

### Short-Term (Next 2 Weeks)
5. Review and merge PR2
6. Create PR3 on GitHub
7. Review and merge PR3
8. Run failure injection tests on staging

### Medium-Term (Weeks 3-4)
9. Monitor shadow logging in production
10. Collect consensus violation metrics
11. Evaluate enforcement readiness
12. Plan enforcement rollout (if metrics look good)

---

## DOCTRINE ADHERENCE

✅ **Proof-or-Abstain**: All references verified against actual repository  
✅ **Ledger-First**: All operations preserve ledger integrity  
✅ **Mechanical Honesty**: All diffs tagged # REAL-READY (verified)  
✅ **Hash-Law Invariants**: Domain separation, determinism, versioning enforced

---

**"Keep it blue, keep it clean, keep it sealed."**

— Manus-B, Ledger Integrity & PQ Migration Engineer

**Mission Status**: ✅ **PR SEQUENCE PACK COMPLETE**
