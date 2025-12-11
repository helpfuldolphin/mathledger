# Consensus–Replay–Governance Integration Checklist v1.0

**Author**: Manus-B (Ledger Integrity & PQ Migration Engineer)  
**Date**: 2025-12-09  
**Version**: 1.0  
**Status**: Operational

---

## Purpose

This checklist ensures safe integration of consensus rules, replay verification, and governance enforcement into MathLedger's ledger runtime.

**Scope**: Phases I-IV deliverables (replay verification, epoch sealing, consensus runtime, drift radar, governance adaptor)

**Target Audience**: Integration engineers, QA engineers, release managers

---

## Preconditions

### 1. Module Dependencies

**Required Modules** (must exist before integration):

- [ ] `backend/crypto/hashing.py` - Hash abstraction layer (Manus-H)
  - Provides: `get_hash_algorithm(version: str) -> HashAlgorithm`
  - Supports: `sha256-v1`, `dual-v1`, `sha3-v1`

- [ ] `backend/ledger/blockchain.py` - Block sealing runtime
  - Provides: `seal_block(block: Dict) -> int`
  - Supports: Block creation, attestation root computation

- [ ] `backend/dag/merkle.py` - Merkle tree implementation
  - Provides: `merkle_root(leaves: List[str]) -> str`
  - Supports: Domain-separated hashing

- [ ] `attestation/dual_root.py` - Dual attestation module
  - Provides: `compute_reasoning_root(...)`, `compute_ui_root(...)`
  - Supports: R_t and U_t computation

**Missing Module Handling**: If any module is missing, **BLOCK integration** and file an issue for module implementation.

---

### 2. Database Schema

**Required Tables** (must exist before integration):

- [ ] `blocks` table with columns:
  - `id` (primary key)
  - `block_number` (unique, monotone)
  - `prev_hash` (foreign key to previous block)
  - `reasoning_attestation_root` (R_t)
  - `ui_attestation_root` (U_t)
  - `composite_attestation_root` (H_t)
  - `attestation_metadata` (JSONB, contains `hash_version`)
  - `sealed_at` (timestamp)

- [ ] `epochs` table (from migration 018):
  - `epoch_number` (primary key)
  - `start_block` (foreign key to blocks)
  - `end_block` (foreign key to blocks)
  - `epoch_root` (Merkle root of composite roots)
  - `hash_version` (hash algorithm used)
  - `sealed_at` (timestamp)

- [ ] `canonical_proofs` table:
  - `id` (primary key)
  - `block_id` (foreign key to blocks)
  - `proof_data` (JSONB)
  - `statement_hash` (hash of statement)

- [ ] `canonical_statements` table:
  - `id` (primary key)
  - `statement_text` (text)
  - `statement_hash` (hash of statement)

**Schema Validation**:
```bash
python3 scripts/ci/validate_migrations.py \
  --database-url $DATABASE_URL \
  --migrations-dir migrations \
  --check-schema
```

**Expected Output**: All required tables and columns exist.

---

### 3. Helper Scripts

**Required Scripts** (must be implemented before CI integration):

- [ ] `scripts/ci/validate_migrations.py` - Validate SQL migrations
- [ ] `scripts/ci/check_replay_success_rate.py` - Check replay success rate
- [ ] `scripts/ci/drift_radar_scan.py` - Run drift radar with governance
- [ ] `scripts/ci/check_governance_signal.py` - Check governance signal
- [ ] `scripts/ci/attestation_integrity_sweep.py` - Verify attestation integrity
- [ ] `scripts/ci/check_integrity_violations.py` - Check integrity violations
- [ ] `scripts/ci/check_pq_migration_code.py` - Audit PQ migration code
- [ ] `scripts/ci/aggregate_governance_results.py` - Aggregate results
- [ ] `scripts/ci/check_governance_gate.py` - Check governance gate
- [ ] `scripts/load_test_data.py` - Load test data for CI

**Implementation Status**: See `docs/ci/implementation_guide.md` for specifications.

---

## Integration Steps

### Step 1: Apply Consensus-Replay Integration Diffs

**Objective**: Wire consensus rules into replay verification engine.

**Files to Modify**:

#### 1.1 `backend/ledger/replay/recompute.py` (+45 lines)

**Location**: After `ReplayResult` dataclass definition (line ~30)

**Diff**:
```python
# ADD: Import consensus violations
from backend.consensus.violations import RuleViolation
from typing import List, Optional

# MODIFY: ReplayResult dataclass
@dataclass
class ReplayResult:
    # Existing fields (unchanged)
    block_id: int
    block_number: int
    hash_version: str
    r_t_recomputed: str
    u_t_recomputed: str
    h_t_recomputed: str
    r_t_stored: str
    u_t_stored: str
    h_t_stored: str
    r_t_match: bool
    u_t_match: bool
    h_t_match: bool
    
    # NEW: Consensus integration fields
    consensus_violations: List[RuleViolation] = field(default_factory=list)
    consensus_passed: bool = True
    consensus_severity: Optional[str] = None  # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    
    # NEW: Helper methods
    def has_critical_violations(self) -> bool:
        """Check if any violations are CRITICAL."""
        return any(v.severity == "CRITICAL" for v in self.consensus_violations)
    
    def has_blocking_violations(self) -> bool:
        """Check if any violations should block replay."""
        return any(v.severity in ["CRITICAL", "ERROR"] for v in self.consensus_violations)
    
    def get_violation_summary(self) -> Dict[str, int]:
        """Get violation count by severity."""
        from collections import Counter
        return dict(Counter(v.severity for v in self.consensus_violations))
```

**Verification**:
```bash
python3 -c "from backend.ledger.replay.recompute import ReplayResult; print('OK')"
```

**Expected Output**: `OK` (no import errors)

---

#### 1.2 `backend/ledger/replay/checker.py` (+30 lines)

**Location**: Inside `verify_block_replay()` function (line ~50)

**Diff**:
```python
# ADD: Import consensus validators
from backend.consensus.rules import validate_block_structure, validate_attestation_roots
from backend.consensus.violations import RuleViolation

def verify_block_replay(block: Dict[str, Any]) -> ReplayResult:
    """
    Verify block replay with consensus integration.
    
    Args:
        block: Block dictionary
    
    Returns:
        ReplayResult with consensus violations
    """
    # NEW: Step 1 - Consensus-first block vetting
    consensus_violations = []
    
    # Validate block structure
    is_valid, structure_violations = validate_block_structure(block)
    if not is_valid:
        consensus_violations.extend(structure_violations)
    
    # EXISTING: Step 2 - Recompute attestation roots (unchanged)
    r_t = block.get("reasoning_attestation_root")
    u_t = block.get("ui_attestation_root")
    h_t_stored = block.get("composite_attestation_root")
    hash_version = block.get("attestation_metadata", {}).get("hash_version", "sha256-v1")
    
    r_t_recomputed = recompute_reasoning_root(block, hash_version)
    u_t_recomputed = recompute_ui_root(block, hash_version)
    h_t_recomputed = recompute_composite_root(r_t_recomputed, u_t_recomputed, hash_version)
    
    # NEW: Step 3 - Validate attestation roots (consensus)
    is_valid, attestation_violations = validate_attestation_roots(
        block, r_t_recomputed, u_t_recomputed, h_t_recomputed
    )
    if not is_valid:
        consensus_violations.extend(attestation_violations)
    
    # NEW: Step 4 - Determine consensus severity
    consensus_severity = None
    if consensus_violations:
        severities = [v.severity for v in consensus_violations]
        if "CRITICAL" in severities:
            consensus_severity = "CRITICAL"
        elif "ERROR" in severities:
            consensus_severity = "ERROR"
        elif "WARNING" in severities:
            consensus_severity = "WARNING"
        else:
            consensus_severity = "INFO"
    
    # MODIFY: Return ReplayResult with consensus fields
    return ReplayResult(
        block_id=block["id"],
        block_number=block["block_number"],
        hash_version=hash_version,
        r_t_recomputed=r_t_recomputed,
        u_t_recomputed=u_t_recomputed,
        h_t_recomputed=h_t_recomputed,
        r_t_stored=r_t,
        u_t_stored=u_t,
        h_t_stored=h_t_stored,
        r_t_match=(r_t == r_t_recomputed),
        u_t_match=(u_t == u_t_recomputed),
        h_t_match=(h_t_stored == h_t_recomputed),
        consensus_violations=consensus_violations,  # NEW
        consensus_passed=(len(consensus_violations) == 0),  # NEW
        consensus_severity=consensus_severity,  # NEW
    )
```

**Verification**:
```bash
python3 -c "from backend.ledger.replay.checker import verify_block_replay; print('OK')"
```

**Expected Output**: `OK` (no import errors)

---

#### 1.3 `backend/ledger/replay/engine.py` (+60 lines)

**Location**: Inside `replay_blocks()` function (line ~100)

**Diff**:
```python
# ADD: Import consensus validators
from backend.consensus.validators import BlockValidator
from backend.consensus.rules import validate_prev_hash, validate_monotonicity

def replay_blocks(
    blocks: List[Dict[str, Any]],
    consensus_first: bool = True,  # NEW parameter
    fail_fast: bool = False,  # NEW parameter
) -> List[ReplayResult]:
    """
    Replay blocks with consensus-first vetting.
    
    Args:
        blocks: List of blocks to replay
        consensus_first: If True, validate consensus before replay
        fail_fast: If True, stop on first critical violation
    
    Returns:
        List of ReplayResult
    """
    # NEW: Pre-flight consensus checks
    if consensus_first:
        validator = BlockValidator()
        for i, block in enumerate(blocks):
            # Validate monotonicity (if not first block)
            if i > 0:
                is_valid, violations = validate_monotonicity(blocks[i-1], block)
                if not is_valid and fail_fast:
                    raise ValueError(f"Monotonicity violation at block {block['block_number']}: {violations}")
            
            # Validate prev_hash (if not first block)
            if i > 0:
                is_valid, violations = validate_prev_hash(block, blocks[i-1])
                if not is_valid and fail_fast:
                    raise ValueError(f"Prev_hash violation at block {block['block_number']}: {violations}")
    
    # EXISTING: Replay each block (unchanged)
    results = []
    for block in blocks:
        result = verify_block_replay(block)
        results.append(result)
        
        # NEW: Fail fast on critical violations
        if fail_fast and result.has_critical_violations():
            raise ValueError(f"Critical violation at block {block['block_number']}")
    
    return results
```

**Verification**:
```bash
python3 -c "from backend.ledger.replay.engine import replay_blocks; print('OK')"
```

**Expected Output**: `OK` (no import errors)

---

### Step 2: Integration Tests

**Objective**: Verify consensus-replay integration works correctly.

**Test Suite**: `tests/integration/test_consensus_replay_integration.py`

**Required Tests** (20+ tests):

#### 2.1 Consensus Violation Detection

- [ ] `test_detect_block_structure_violation()` - Detect missing required fields
- [ ] `test_detect_monotonicity_violation()` - Detect non-monotone block numbers
- [ ] `test_detect_prev_hash_violation()` - Detect prev_hash mismatch
- [ ] `test_detect_attestation_root_violation()` - Detect attestation root mismatch

#### 2.2 Consensus-First Vetting

- [ ] `test_consensus_first_vetting_valid_blocks()` - Pass valid blocks
- [ ] `test_consensus_first_vetting_invalid_blocks()` - Reject invalid blocks
- [ ] `test_consensus_first_fail_fast()` - Stop on first critical violation

#### 2.3 Replay Result Schema

- [ ] `test_replay_result_with_consensus_violations()` - ReplayResult contains violations
- [ ] `test_replay_result_has_critical_violations()` - has_critical_violations() works
- [ ] `test_replay_result_has_blocking_violations()` - has_blocking_violations() works
- [ ] `test_replay_result_get_violation_summary()` - get_violation_summary() works

#### 2.4 Mixed Hash Version Epochs

- [ ] `test_replay_sha256_blocks()` - Replay SHA-256 blocks
- [ ] `test_replay_dual_commitment_blocks()` - Replay dual-commitment blocks
- [ ] `test_replay_sha3_blocks()` - Replay SHA-3 blocks
- [ ] `test_replay_mixed_epoch()` - Replay epoch with mixed hash versions
- [ ] `test_cross_algorithm_prev_hash_validation()` - Validate SHA-256→Dual→SHA-3 transitions

#### 2.5 Edge Cases

- [ ] `test_replay_empty_block_list()` - Handle empty block list
- [ ] `test_replay_single_block()` - Handle single block
- [ ] `test_replay_with_missing_fields()` - Handle blocks with missing fields
- [ ] `test_replay_with_invalid_hash_version()` - Handle invalid hash version

**Test Execution**:
```bash
pytest tests/integration/test_consensus_replay_integration.py -v
```

**Expected Output**: All tests pass (20/20)

---

### Step 3: Expected Outputs & Invariants

**Objective**: Define expected outputs and invariants for verification.

#### 3.1 Replay Result Invariants

**Invariant 1: Replay Determinism**
```
∀ blocks b: replay(b) = replay(b)
```
Replaying the same block twice produces identical results.

**Invariant 2: Consensus Monotonicity**
```
∀ consecutive blocks b₁, b₂: 
  b₁.block_number < b₂.block_number ⟹ 
  b₁.sealed_at ≤ b₂.sealed_at
```
Block numbers are monotone increasing.

**Invariant 3: Prev-Hash Lineage**
```
∀ consecutive blocks b₁, b₂: 
  b₂.prev_hash = Hash(b₁.block_identity)
```
Prev-hash forms a valid chain.

**Invariant 4: Attestation Root Consistency**
```
∀ blocks b: 
  b.composite_attestation_root = Hash("EPOCH:" || b.r_t || b.u_t)
```
Composite root is deterministic function of R_t and U_t.

**Invariant 5: Hash Version Consistency**
```
∀ blocks b: 
  b.hash_version ∈ {"sha256-v1", "dual-v1", "sha3-v1"}
```
Hash version is one of the supported versions.

---

#### 3.2 Merkle Root Invariants

**Invariant 6: Merkle Root Determinism**
```
∀ leaves L: MerkleRoot(L) = MerkleRoot(L)
```
Merkle root computation is deterministic.

**Invariant 7: Merkle Root Domain Separation**
```
∀ leaves L: 
  MerkleRoot(L) uses domain-separated hashing
  (DOMAIN_LEAF for leaves, DOMAIN_NODE for internal nodes)
```
Merkle tree uses domain separation to prevent second preimage attacks.

---

#### 3.3 Block Structure Invariants

**Invariant 8: Required Fields**
```
∀ blocks b: 
  b has fields {id, block_number, prev_hash, r_t, u_t, h_t, attestation_metadata, sealed_at}
```
All blocks have required fields.

**Invariant 9: Attestation Metadata**
```
∀ blocks b: 
  b.attestation_metadata contains {hash_version, algorithm, domain_tags}
```
Attestation metadata is complete.

---

#### 3.4 Epoch Invariants

**Invariant 10: Epoch Size**
```
∀ epochs e: 
  e.end_block - e.start_block + 1 = 100
```
All epochs contain exactly 100 blocks (except possibly the last epoch).

**Invariant 11: Epoch Root Computation**
```
∀ epochs e: 
  e.epoch_root = MerkleRoot([H_0, H_1, ..., H_99])
  where H_i = blocks[e.start_block + i].composite_attestation_root
```
Epoch root is Merkle root of composite roots.

---

### Step 4: Governance Adaptor Integration

**Objective**: Integrate drift radar governance adaptor into CI pipeline.

**Files to Modify**:

#### 4.1 `scripts/ci/drift_radar_scan.py` (new file, ~300 lines)

**Implementation**:
```python
#!/usr/bin/env python3
"""
Drift Radar Scan with Governance Evaluation

Scans blocks for drift signals and evaluates governance signal.
"""

import argparse
import json
from backend.ledger.drift.scanner import DriftScanner
from backend.ledger.drift.governance import create_governance_adaptor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--start-block", type=int, required=True)
    parser.add_argument("--end-block", type=int, required=True)
    parser.add_argument("--scan-types", default="schema,hash-delta,metadata,statement")
    parser.add_argument("--governance-policy", default="strict")
    parser.add_argument("--output", required=True)
    parser.add_argument("--evidence-pack", required=True)
    args = parser.parse_args()
    
    # Create scanner
    scanner = DriftScanner(database_url=args.database_url)
    
    # Scan for drift
    scan_types = args.scan_types.split(",")
    drift_signals = scanner.scan(
        start_block=args.start_block,
        end_block=args.end_block,
        scan_types=scan_types,
    )
    
    # Evaluate governance signal
    adaptor = create_governance_adaptor(args.governance_policy)
    evidence_pack = adaptor.evaluate_drift_signals(drift_signals)
    
    # Write outputs
    with open(args.output, "w") as f:
        json.dump([s.to_dict() for s in drift_signals], f, indent=2)
    
    with open(args.evidence_pack, "w") as f:
        json.dump(evidence_pack.to_dict(), f, indent=2)
    
    # Print evidence pack
    print(evidence_pack.to_console_output())
    
    # Exit with appropriate code
    if adaptor.should_block_merge(evidence_pack):
        print("GOVERNANCE SIGNAL: BLOCK")
        return 1
    elif adaptor.should_warn(evidence_pack):
        print("GOVERNANCE SIGNAL: WARN")
        return 0
    else:
        print("GOVERNANCE SIGNAL: OK")
        return 0

if __name__ == "__main__":
    exit(main())
```

**Verification**:
```bash
python3 scripts/ci/drift_radar_scan.py \
  --database-url $DATABASE_URL \
  --start-block 0 \
  --end-block 100 \
  --scan-types schema,hash-delta \
  --governance-policy strict \
  --output drift_signals.json \
  --evidence-pack evidence_pack.json
```

**Expected Output**: Drift signals and evidence pack written to files.

---

### Step 5: CI Chain Integration

**Objective**: Integrate governance chain into CI pipeline.

**Files to Create**:

#### 5.1 `.github/workflows/governance-chain.yml` (optional, if using GitHub Actions)

**Implementation**: See `docs/ci/implementation_guide.md` for full YAML.

**Alternative**: Use standalone script `scripts/ci/run_governance_chain.sh` (no GitHub Actions permissions required).

---

## Verification Checklist

### Pre-Integration Verification

- [ ] All preconditions met (modules, schemas, helpers)
- [ ] All diffs reviewed by at least 2 engineers
- [ ] All integration tests written (20+ tests)
- [ ] Test database provisioned with test data

### Post-Integration Verification

- [ ] All integration tests pass (20/20)
- [ ] Replay verification runs successfully on test data
- [ ] Drift radar scan produces expected evidence packs
- [ ] CI chain completes in < 35 minutes
- [ ] All invariants hold (10 invariants verified)

### Production Readiness Verification

- [ ] Full-chain replay verification on production data (read-only)
- [ ] Drift radar scan on production data (read-only)
- [ ] Governance adaptor tested with real drift signals
- [ ] CI chain deployed to staging environment
- [ ] Rollback procedures tested

---

## Rollback Procedure

**If integration fails**, rollback using:

```bash
# 1. Revert commits
git revert <commit-sha>

# 2. Redeploy previous version
git checkout <previous-tag>
./scripts/deploy.sh

# 3. Verify rollback
python3 scripts/verify_rollback.py --expected-version <previous-version>
```

---

## Sign-Off

**Integration Engineer**: _________________ Date: _______

**QA Engineer**: _________________ Date: _______

**Release Manager**: _________________ Date: _______

---

**"Keep it blue, keep it clean, keep it sealed."**

— Manus-B, Ledger Integrity & PQ Migration Engineer
