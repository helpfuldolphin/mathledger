# PQ Cutover Playbook

**Author**: Manus-B (Ledger Replay Architect & PQ Migration Officer)  
**Phase**: IV - Consensus Integration & Enforcement  
**Date**: 2025-12-09  
**Status**: Operational Ready

---

## Purpose

Step-by-step activation protocol for post-quantum cryptographic migration from SHA-256 to SHA-3.

**Migration Path**: Testnet → Shadow → Dual-Commitment → PQ-Only

---

## Table of Contents

1. [Pre-Migration Checklist](#pre-migration-checklist)
2. [Phase 1: Testnet Activation](#phase-1-testnet-activation)
3. [Phase 2: Shadow Mode](#phase-2-shadow-mode)
4. [Phase 3: Dual-Commitment](#phase-3-dual-commitment)
5. [Phase 4: PQ-Only](#phase-4-pq-only)
6. [Rollback Procedures](#rollback-procedures)
7. [Attestation Roles](#attestation-roles)
8. [Invariant Tables](#invariant-tables)

---

## Pre-Migration Checklist

### Code Readiness

- [ ] Manus-H hash abstraction implemented
- [ ] SHA-3 algorithm integrated
- [ ] Dual-commitment support implemented
- [ ] Cross-algorithm prev_hash validation implemented
- [ ] Epoch transition envelope support implemented
- [ ] All unit tests passing
- [ ] All integration tests passing

### Infrastructure Readiness

- [ ] Testnet environment provisioned
- [ ] Shadow mode infrastructure ready
- [ ] Database schema migration 018 applied
- [ ] Monitoring dashboards configured
- [ ] Alerting rules configured
- [ ] Incident response plan documented

### Team Readiness

- [ ] Migration team identified (roles assigned)
- [ ] Runbook reviewed by all team members
- [ ] Communication plan established
- [ ] Rollback procedures rehearsed
- [ ] On-call rotation scheduled

### Documentation Readiness

- [ ] Migration announcement published
- [ ] User-facing documentation updated
- [ ] API documentation updated
- [ ] Changelog prepared

---

## Phase 1: Testnet Activation

**Duration**: 1 week  
**Environment**: Testnet  
**Goal**: Validate PQ migration on testnet

### Step 1.1: Deploy Dual-Commitment Code to Testnet

```bash
# Deploy to testnet
git checkout release/pq-migration-v1
./scripts/deploy_testnet.sh

# Verify deployment
curl https://testnet.mathledger.io/health
```

**Expected Output**:
```json
{
  "status": "healthy",
  "pq_migration_ready": true,
  "hash_versions_supported": ["sha256-v1", "dual-v1", "sha3-v1"]
}
```

---

### Step 1.2: Activate Dual-Commitment on Testnet

```bash
# Activate dual-commitment at block 10000
python3 scripts/activate_dual_commitment.py \
  --database-url $TESTNET_DATABASE_URL \
  --activation-block 10000 \
  --environment testnet

# Verify activation
python3 scripts/verify_activation_block.py \
  --database-url $TESTNET_DATABASE_URL \
  --block-number 10000
```

**Expected Output**:
```
Activation Block 10000:
  hash_version: dual-v1
  composite_attestation_root: abc123... (SHA-256)
  composite_attestation_root_sha3: def456... (SHA-3)
  activation_phase: dual_commitment
  ✓ Activation successful
```

---

### Step 1.3: Monitor Dual-Commitment on Testnet

```bash
# Monitor for 1000 blocks
python3 scripts/monitor_dual_commitment.py \
  --database-url $TESTNET_DATABASE_URL \
  --start-block 10000 \
  --end-block 11000 \
  --check-interval 60

# Run replay verification
python3 scripts/replay_verify.py \
  --database-url $TESTNET_DATABASE_URL \
  --mode sliding_window \
  --start-block 10000 \
  --end-block 11000
```

**Success Criteria**:
- 100% replay success rate
- No hash mismatches
- No drift signals (CRITICAL/HIGH)

---

### Step 1.4: Activate SHA-3 on Testnet

```bash
# Activate SHA-3 at block 20000
python3 scripts/activate_sha3.py \
  --database-url $TESTNET_DATABASE_URL \
  --activation-block 20000 \
  --environment testnet

# Verify activation
python3 scripts/verify_activation_block.py \
  --database-url $TESTNET_DATABASE_URL \
  --block-number 20000
```

**Expected Output**:
```
Activation Block 20000:
  hash_version: sha3-v1
  composite_attestation_root: ghi789... (SHA-3)
  activation_phase: pure_sha3
  ✓ Activation successful
```

---

### Step 1.5: Validate Full Migration on Testnet

```bash
# Run full-chain replay verification
python3 scripts/replay_verify.py \
  --database-url $TESTNET_DATABASE_URL \
  --mode full_chain \
  --start-block 0 \
  --end-block 21000

# Validate cross-algorithm prev_hash
python3 scripts/validate_cross_algorithm_prev_hash.py \
  --database-url $TESTNET_DATABASE_URL \
  --dual-commitment-start 10000 \
  --sha3-cutover 20000

# Generate migration audit report
python3 scripts/generate_migration_audit_report.py \
  --database-url $TESTNET_DATABASE_URL \
  --output testnet_migration_audit.pdf
```

**Success Criteria**:
- 100% replay success rate across all hash versions
- All cross-algorithm prev_hash validations pass
- Audit report shows no violations

---

## Phase 2: Shadow Mode

**Duration**: 2 weeks  
**Environment**: Production (shadow mode)  
**Goal**: Run dual-commitment in shadow mode (no writes)

### Step 2.1: Deploy Shadow Mode Code to Production

```bash
# Deploy shadow mode code
git checkout release/pq-migration-shadow-v1
./scripts/deploy_production.sh --mode shadow

# Verify deployment
curl https://api.mathledger.io/health
```

**Expected Output**:
```json
{
  "status": "healthy",
  "pq_migration_mode": "shadow",
  "hash_versions_supported": ["sha256-v1", "dual-v1", "sha3-v1"]
}
```

---

### Step 2.2: Enable Shadow Dual-Commitment

```bash
# Enable shadow mode (compute SHA-3 roots, but don't write)
python3 scripts/enable_shadow_mode.py \
  --database-url $PRODUCTION_DATABASE_URL \
  --environment production

# Verify shadow mode
python3 scripts/verify_shadow_mode.py \
  --database-url $PRODUCTION_DATABASE_URL
```

**Expected Output**:
```
Shadow Mode Status:
  enabled: true
  sha3_roots_computed: true
  sha3_roots_written: false
  mismatch_count: 0
  ✓ Shadow mode active
```

---

### Step 2.3: Monitor Shadow Mode

```bash
# Monitor shadow mode for 2 weeks
python3 scripts/monitor_shadow_mode.py \
  --database-url $PRODUCTION_DATABASE_URL \
  --duration-days 14 \
  --check-interval 3600

# Check for SHA-256/SHA-3 mismatches
python3 scripts/check_shadow_mismatches.py \
  --database-url $PRODUCTION_DATABASE_URL \
  --start-date 2025-12-09 \
  --end-date 2025-12-23
```

**Success Criteria**:
- 0 SHA-256/SHA-3 mismatches
- Performance degradation < 5%
- No errors in logs

---

### Step 2.4: Shadow Mode Validation

```bash
# Run shadow mode validation
python3 scripts/validate_shadow_mode.py \
  --database-url $PRODUCTION_DATABASE_URL \
  --output shadow_mode_validation.json

# Generate shadow mode report
python3 scripts/generate_shadow_mode_report.py \
  --input shadow_mode_validation.json \
  --output shadow_mode_report.pdf
```

**Success Criteria**:
- Shadow mode validation passes
- Report shows 100% SHA-256/SHA-3 agreement

---

## Phase 3: Dual-Commitment

**Duration**: 4 weeks  
**Environment**: Production  
**Goal**: Write both SHA-256 and SHA-3 roots to database

### Step 3.1: Deploy Dual-Commitment Code to Production

```bash
# Deploy dual-commitment code
git checkout release/pq-migration-dual-v1
./scripts/deploy_production.sh --mode dual

# Verify deployment
curl https://api.mathledger.io/health
```

**Expected Output**:
```json
{
  "status": "healthy",
  "pq_migration_mode": "dual",
  "hash_versions_supported": ["sha256-v1", "dual-v1"]
}
```

---

### Step 3.2: Activate Dual-Commitment on Production

**⚠️ CRITICAL STEP - REQUIRES APPROVAL**

**Approval Checklist**:
- [ ] Testnet migration successful
- [ ] Shadow mode validation passed
- [ ] Team on-call and ready
- [ ] Rollback plan reviewed
- [ ] Communication sent to users

```bash
# Activate dual-commitment at block N
# (N = current_block + 1000 for buffer)
python3 scripts/activate_dual_commitment.py \
  --database-url $PRODUCTION_DATABASE_URL \
  --activation-block $ACTIVATION_BLOCK \
  --environment production \
  --require-approval

# Verify activation
python3 scripts/verify_activation_block.py \
  --database-url $PRODUCTION_DATABASE_URL \
  --block-number $ACTIVATION_BLOCK
```

**Expected Output**:
```
Activation Block $ACTIVATION_BLOCK:
  hash_version: dual-v1
  composite_attestation_root: abc123... (SHA-256)
  composite_attestation_root_sha3: def456... (SHA-3)
  activation_phase: dual_commitment
  ✓ Activation successful
```

---

### Step 3.3: Monitor Dual-Commitment on Production

```bash
# Monitor for 1000 blocks
python3 scripts/monitor_dual_commitment.py \
  --database-url $PRODUCTION_DATABASE_URL \
  --start-block $ACTIVATION_BLOCK \
  --end-block $((ACTIVATION_BLOCK + 1000)) \
  --check-interval 60 \
  --alert-on-mismatch

# Run replay verification
python3 scripts/replay_verify.py \
  --database-url $PRODUCTION_DATABASE_URL \
  --mode sliding_window \
  --start-block $ACTIVATION_BLOCK \
  --end-block $((ACTIVATION_BLOCK + 1000)) \
  --fail-fast
```

**Success Criteria**:
- 100% replay success rate
- No hash mismatches
- No drift signals (CRITICAL/HIGH)
- Performance degradation < 10%

---

### Step 3.4: Extended Dual-Commitment Monitoring

**Duration**: 4 weeks

```bash
# Daily monitoring
python3 scripts/daily_dual_commitment_check.py \
  --database-url $PRODUCTION_DATABASE_URL \
  --output daily_report.json

# Weekly audit
python3 scripts/weekly_dual_commitment_audit.py \
  --database-url $PRODUCTION_DATABASE_URL \
  --output weekly_audit.pdf
```

**Success Criteria**:
- 4 weeks of stable dual-commitment operation
- No critical incidents
- User feedback positive

---

## Phase 4: PQ-Only

**Duration**: 1 week  
**Environment**: Production  
**Goal**: Cutover to SHA-3 only (new blocks)

### Step 4.1: Deploy PQ-Only Code to Production

```bash
# Deploy PQ-only code
git checkout release/pq-migration-sha3-v1
./scripts/deploy_production.sh --mode pq-only

# Verify deployment
curl https://api.mathledger.io/health
```

**Expected Output**:
```json
{
  "status": "healthy",
  "pq_migration_mode": "pq_only",
  "hash_versions_supported": ["sha3-v1"]
}
```

---

### Step 4.2: Activate SHA-3 on Production

**⚠️ CRITICAL STEP - REQUIRES APPROVAL**

**Approval Checklist**:
- [ ] Dual-commitment stable for 4 weeks
- [ ] No critical incidents during dual-commitment
- [ ] Team on-call and ready
- [ ] Rollback plan reviewed
- [ ] Communication sent to users

```bash
# Activate SHA-3 at block M
# (M = current_block + 1000 for buffer)
python3 scripts/activate_sha3.py \
  --database-url $PRODUCTION_DATABASE_URL \
  --activation-block $SHA3_ACTIVATION_BLOCK \
  --environment production \
  --require-approval

# Verify activation
python3 scripts/verify_activation_block.py \
  --database-url $PRODUCTION_DATABASE_URL \
  --block-number $SHA3_ACTIVATION_BLOCK
```

**Expected Output**:
```
Activation Block $SHA3_ACTIVATION_BLOCK:
  hash_version: sha3-v1
  composite_attestation_root: ghi789... (SHA-3)
  activation_phase: pure_sha3
  ✓ Activation successful
```

---

### Step 4.3: Monitor SHA-3 on Production

```bash
# Monitor for 1000 blocks
python3 scripts/monitor_sha3.py \
  --database-url $PRODUCTION_DATABASE_URL \
  --start-block $SHA3_ACTIVATION_BLOCK \
  --end-block $((SHA3_ACTIVATION_BLOCK + 1000)) \
  --check-interval 60 \
  --alert-on-error

# Run replay verification
python3 scripts/replay_verify.py \
  --database-url $PRODUCTION_DATABASE_URL \
  --mode sliding_window \
  --start-block $SHA3_ACTIVATION_BLOCK \
  --end-block $((SHA3_ACTIVATION_BLOCK + 1000)) \
  --fail-fast
```

**Success Criteria**:
- 100% replay success rate
- No hash errors
- No drift signals (CRITICAL/HIGH)
- Performance stable

---

### Step 4.4: Final Migration Validation

```bash
# Run full-chain replay verification
python3 scripts/replay_verify.py \
  --database-url $PRODUCTION_DATABASE_URL \
  --mode full_chain \
  --start-block 0 \
  --end-block $((SHA3_ACTIVATION_BLOCK + 1000))

# Validate cross-algorithm prev_hash
python3 scripts/validate_cross_algorithm_prev_hash.py \
  --database-url $PRODUCTION_DATABASE_URL \
  --dual-commitment-start $ACTIVATION_BLOCK \
  --sha3-cutover $SHA3_ACTIVATION_BLOCK

# Generate final migration audit report
python3 scripts/generate_migration_audit_report.py \
  --database-url $PRODUCTION_DATABASE_URL \
  --output production_migration_audit.pdf
```

**Success Criteria**:
- 100% replay success rate across all hash versions
- All cross-algorithm prev_hash validations pass
- Audit report shows no violations
- Migration declared successful

---

## Rollback Procedures

### Rollback from Dual-Commitment to SHA-256

**Scenario**: Critical issue detected during dual-commitment phase.

**Steps**:

1. **Stop block sealing**:
   ```bash
   python3 scripts/stop_block_sealing.py \
     --database-url $PRODUCTION_DATABASE_URL \
     --reason "Rollback to SHA-256"
   ```

2. **Deploy SHA-256-only code**:
   ```bash
   git checkout release/pre-pq-migration
   ./scripts/deploy_production.sh --mode sha256-only
   ```

3. **Resume block sealing**:
   ```bash
   python3 scripts/resume_block_sealing.py \
     --database-url $PRODUCTION_DATABASE_URL \
     --hash-version sha256-v1
   ```

4. **Verify rollback**:
   ```bash
   python3 scripts/verify_rollback.py \
     --database-url $PRODUCTION_DATABASE_URL \
     --expected-hash-version sha256-v1
   ```

5. **Post-incident review**:
   - Document root cause
   - Update playbook
   - Schedule retry

---

### Rollback from SHA-3 to Dual-Commitment

**Scenario**: Critical issue detected during SHA-3 phase.

**Steps**:

1. **Stop block sealing**:
   ```bash
   python3 scripts/stop_block_sealing.py \
     --database-url $PRODUCTION_DATABASE_URL \
     --reason "Rollback to dual-commitment"
   ```

2. **Deploy dual-commitment code**:
   ```bash
   git checkout release/pq-migration-dual-v1
   ./scripts/deploy_production.sh --mode dual
   ```

3. **Resume block sealing**:
   ```bash
   python3 scripts/resume_block_sealing.py \
     --database-url $PRODUCTION_DATABASE_URL \
     --hash-version dual-v1
   ```

4. **Verify rollback**:
   ```bash
   python3 scripts/verify_rollback.py \
     --database-url $PRODUCTION_DATABASE_URL \
     --expected-hash-version dual-v1
   ```

5. **Post-incident review**:
   - Document root cause
   - Update playbook
   - Schedule retry

---

## Attestation Roles

### Migration Officer (Manus-B)

**Responsibilities**:
- Execute migration playbook
- Monitor migration progress
- Approve critical steps
- Coordinate rollback if needed

**Authority**:
- STOP migration at any time
- ROLLBACK to previous phase
- ESCALATE to incident response

---

### Verification Engineer

**Responsibilities**:
- Run replay verification
- Validate cross-algorithm prev_hash
- Monitor drift radar
- Generate audit reports

**Authority**:
- BLOCK merge on verification failures
- ESCALATE to Migration Officer

---

### On-Call Engineer

**Responsibilities**:
- Monitor alerts
- Respond to incidents
- Execute rollback procedures
- Communicate status

**Authority**:
- ESCALATE to Migration Officer
- INITIATE rollback (with approval)

---

### Incident Commander

**Responsibilities**:
- Coordinate incident response
- Make go/no-go decisions
- Communicate with stakeholders
- Document lessons learned

**Authority**:
- OVERRIDE migration decisions
- DECLARE migration failure
- INITIATE emergency rollback

---

## Invariant Tables

### Block Invariants

| Invariant | SHA-256 | Dual-Commitment | SHA-3 |
|-----------|---------|-----------------|-------|
| `hash_version` | `sha256-v1` | `dual-v1` | `sha3-v1` |
| `composite_attestation_root` | SHA-256(R_t \|\| U_t) | SHA-256(R_t \|\| U_t) | SHA-3(R_t \|\| U_t) |
| `composite_attestation_root_sha3` | NULL | SHA-3(R_t \|\| U_t) | NULL |
| `prev_hash` algorithm | SHA-256 | SHA-256 | SHA-3 |

---

### Epoch Invariants

| Invariant | SHA-256 | Dual-Commitment | SHA-3 |
|-----------|---------|-----------------|-------|
| `epoch_root` algorithm | SHA-256 | SHA-256 (primary) | SHA-3 |
| `hash_version` | `sha256-v1` | `dual-v1` | `sha3-v1` |
| Epoch size | 100 blocks | 100 blocks | 100 blocks |

---

### Transition Invariants

| Transition | Predecessor `hash_version` | Current `hash_version` | `prev_hash` Algorithm |
|------------|----------------------------|------------------------|-----------------------|
| SHA-256 → Dual | `sha256-v1` | `dual-v1` | SHA-256 |
| Dual → Dual | `dual-v1` | `dual-v1` | SHA-256 |
| Dual → SHA-3 | `dual-v1` | `sha3-v1` | SHA-256 |
| SHA-3 → SHA-3 | `sha3-v1` | `sha3-v1` | SHA-3 |

**Invalid Transitions**:
- SHA-256 → SHA-3 (must go through dual-commitment)
- SHA-3 → Dual (no rollback to dual-commitment)
- SHA-3 → SHA-256 (no rollback to SHA-256)

---

## Conclusion

The PQ Cutover Playbook provides a **step-by-step activation protocol** for post-quantum cryptographic migration, including:

1. **Testnet validation** (1 week)
2. **Shadow mode** (2 weeks)
3. **Dual-commitment** (4 weeks)
4. **PQ-only** (1 week)

**Total Duration**: 8 weeks

**Rollback Paths**: Dual → SHA-256, SHA-3 → Dual

**Attestation Roles**: Migration Officer, Verification Engineer, On-Call Engineer, Incident Commander

**Invariant Tables**: Block, Epoch, Transition invariants documented

**Status**: Operational ready, pending testnet validation.

---

**"Keep it blue, keep it clean, keep it sealed."**  
— Manus-B, Ledger Replay Architect & PQ Migration Officer
