# Post-Quantum Consensus Rules Specification

## Document Status

**Version**: 1.0  
**Status**: Draft  
**Author**: Manus-H  
**Date**: 2024-12-06

## Executive Summary

This document specifies the consensus rules for post-quantum (PQ) migration in MathLedger. These rules define how nodes validate blocks during each migration phase, handle epoch transitions, and detect consensus violations. The specification ensures that all nodes agree on block validity throughout the migration process, preventing chain splits and maintaining network integrity.

## Consensus Rule Architecture

### Rule Versioning

Consensus rules are versioned according to migration phases. Each phase introduces new validation requirements while maintaining backward compatibility with historical blocks.

| Phase | Rule Version | Consensus Mode | Description |
|-------|--------------|----------------|-------------|
| Phase 0 | v1-legacy | SHA-256 only | Pre-migration baseline |
| Phase 1 | v1-legacy | SHA-256 only | Scaffolding deployed, PQ fields ignored |
| Phase 2 | v2-dual-optional | SHA-256 primary | Dual commitments optional, not validated |
| Phase 3 | v2-dual-required | Dual validation | Dual commitments required and validated |
| Phase 4 | v2-pq-primary | PQ primary | PQ hashes canonical, legacy maintained |
| Phase 5 | v3-pq-only | PQ only | Legacy fields optional, PQ canonical |

### Epoch Boundary Semantics

An **epoch** is a contiguous range of blocks using the same canonical hash algorithm. Epoch boundaries are defined by governance-approved activation blocks.

**Epoch Definition**:
```python
@dataclass(frozen=True)
class HashEpoch:
    start_block: int              # First block of this epoch
    end_block: Optional[int]      # Last block (None = ongoing)
    algorithm_id: int             # Canonical hash algorithm
    algorithm_name: str           # Human-readable name
    rule_version: str             # Consensus rule version
    activation_timestamp: float   # Unix timestamp of activation
    governance_hash: str          # Hash of governance proposal
```

**Epoch Transition Rules**:

1. **Immutability**: Once an epoch is activated, its parameters are immutable
2. **Monotonicity**: Epoch start blocks must be strictly increasing
3. **Continuity**: No gaps allowed between epochs (end_block[N] + 1 = start_block[N+1])
4. **Finality**: Epoch transitions require finality confirmation (e.g., 100 blocks)

## Phase-Specific Consensus Rules

### Phase 0 & Phase 1: Legacy Mode (v1-legacy)

**Block Validation Rules**:

```
RULE L1: Block Header Structure
  - MUST contain: block_number, prev_hash, merkle_root, timestamp, version
  - version MUST be "v1"
  - All fields MUST be present and non-null

RULE L2: Merkle Root Validation
  - merkle_root MUST equal merkle_root_versioned(statements, algorithm_id=0x00)
  - Statements MUST be normalized and sorted lexicographically
  - Empty statement list MUST produce domain-separated empty hash

RULE L3: Previous Hash Validation
  - prev_hash MUST equal hash_block_versioned(prev_block, algorithm_id=0x00)
  - Genesis block (block_number=0) MUST have prev_hash="0"*64
  - prev_hash MUST reference a known, valid block

RULE L4: Timestamp Validation
  - timestamp MUST be greater than prev_block.timestamp
  - timestamp MUST NOT be more than 2 hours in the future
  - timestamp MUST be Unix epoch seconds (float)

RULE L5: Block Number Validation
  - block_number MUST equal prev_block.block_number + 1
  - Genesis block MUST have block_number=0
```

**PQ Field Handling**:
- PQ fields (pq_prev_hash, pq_merkle_root, pq_algorithm, dual_commitment) are IGNORED
- Blocks MAY contain PQ fields but they are NOT validated
- Presence of PQ fields does NOT affect block validity

### Phase 2: Dual Commitment Optional (v2-dual-optional)

**Block Validation Rules**:

All Phase 1 rules apply, PLUS:

```
RULE D1: Dual Commitment Structure (if present)
  - IF pq_merkle_root is present, THEN:
    - pq_algorithm MUST be present and valid (0x01-0xFF)
    - pq_prev_hash MUST be present
    - dual_commitment MUST be present
  - IF any PQ field is present, ALL PQ fields MUST be present

RULE D2: Dual Commitment Validation (if present)
  - dual_commitment MUST equal:
    compute_dual_commitment(merkle_root, pq_merkle_root, pq_algorithm)
  - This validates the cryptographic binding

RULE D3: PQ Merkle Root Validation (if present)
  - pq_merkle_root MUST equal:
    merkle_root_versioned(statements, algorithm_id=pq_algorithm)
  - Uses same statements as legacy merkle_root

RULE D4: PQ Previous Hash Validation (if present)
  - IF prev_block has pq_merkle_root, THEN:
    - pq_prev_hash MUST equal:
      hash_block_versioned(prev_block, algorithm_id=pq_algorithm)
  - IF prev_block does NOT have pq_merkle_root:
    - pq_prev_hash validation is SKIPPED (transition block)
```

**Consensus Behavior**:
- Legacy merkle_root and prev_hash are CANONICAL for consensus
- PQ fields are validated IF present but NOT required
- Blocks without PQ fields are VALID
- Blocks with invalid PQ fields are REJECTED

### Phase 3: Dual Commitment Required (v2-dual-required)

**Activation Condition**:
- Activated at governance-approved epoch boundary block
- Requires supermajority node support (e.g., 80% of stake)

**Block Validation Rules**:

All Phase 2 rules apply, with modifications:

```
RULE R1: Dual Commitment Mandatory
  - ALL blocks at or after activation_block MUST contain PQ fields
  - pq_merkle_root, pq_algorithm, pq_prev_hash, dual_commitment MUST be present
  - Blocks without PQ fields are INVALID

RULE R2: Algorithm Consistency
  - pq_algorithm MUST match the epoch's canonical PQ algorithm
  - For epoch with algorithm_id=0x01, pq_algorithm MUST be 0x01
  - Mismatched algorithms are INVALID

RULE R3: Epoch Transition Block
  - First block of new epoch (block_number = epoch.start_block):
    - MUST have valid dual commitment
    - MUST have pq_prev_hash linking to prev_block
    - MUST use new epoch's pq_algorithm
    - prev_hash MUST still use prev_epoch's algorithm

RULE R4: Cross-Algorithm Validation
  - Both legacy and PQ hashes MUST be valid
  - merkle_root MUST validate with SHA-256
  - pq_merkle_root MUST validate with pq_algorithm
  - dual_commitment MUST bind both correctly
```

**Consensus Behavior**:
- Legacy merkle_root and prev_hash remain CANONICAL
- PQ fields are REQUIRED and VALIDATED
- Invalid PQ fields cause block REJECTION
- Both hash chains must be valid

### Phase 4: PQ Primary (v2-pq-primary)

**Activation Condition**:
- Activated after grace period (e.g., 6-12 months)
- Requires governance approval

**Block Validation Rules**:

```
RULE P1: PQ Canonical
  - pq_merkle_root becomes CANONICAL for consensus
  - Legacy merkle_root is VALIDATED but not canonical
  - Consensus decisions use PQ hashes

RULE P2: Dual Commitment Still Required
  - ALL blocks MUST contain both legacy and PQ fields
  - Both hash chains MUST be valid
  - dual_commitment MUST be valid

RULE P3: PQ Previous Hash Canonical
  - Chain linkage uses pq_prev_hash
  - prev_hash is validated but not canonical for consensus
  - Reorg decisions use PQ hash chain

RULE P4: Legacy Hash Validation
  - merkle_root MUST still validate with SHA-256
  - prev_hash MUST still link correctly
  - Legacy chain must remain valid for historical verification
```

**Consensus Behavior**:
- PQ hashes are CANONICAL for all consensus decisions
- Legacy hashes are MAINTAINED for backward compatibility
- Historical verification uses epoch-appropriate algorithms

### Phase 5: PQ Only (v3-pq-only)

**Activation Condition**:
- Activated after extended grace period
- Requires governance approval

**Block Validation Rules**:

```
RULE O1: PQ Fields Mandatory
  - pq_merkle_root, pq_algorithm, pq_prev_hash MUST be present
  - These are now the primary fields (no longer "pq_" prefix in logic)

RULE O2: Legacy Fields Optional
  - merkle_root, prev_hash MAY be omitted
  - IF present, they MUST be valid
  - dual_commitment MAY be omitted

RULE O3: Historical Verification
  - Historical blocks MUST be verified with epoch-appropriate algorithms
  - Blocks from SHA-256 epoch use SHA-256
  - Blocks from PQ epoch use PQ algorithm

RULE O4: Simplified Validation
  - Only PQ hash chain is validated for new blocks
  - Legacy validation only for historical blocks
```

**Consensus Behavior**:
- PQ hashes are the ONLY canonical hashes
- Legacy fields are OPTIONAL and used only for backward compatibility
- Storage overhead reduced for new blocks

## Cross-Algorithm Previous Hash Validation

### Challenge

During epoch transitions, prev_hash must link blocks across different hash algorithms. Block N uses SHA-256, Block N+1 uses SHA3-256. How does Block N+1's prev_hash reference Block N?

### Solution: Dual Previous Hash

**Epoch Transition Block Structure**:

```python
# Block N (last block of SHA-256 epoch)
block_n = BlockHeaderPQ(
    block_number=N,
    merkle_root=<SHA-256 Merkle root>,
    prev_hash=<SHA-256 hash of block N-1>,
    pq_merkle_root=<SHA3-256 Merkle root>,
    pq_prev_hash=<SHA3-256 hash of block N-1>,
    pq_algorithm=0x01,
    dual_commitment=<binding>,
)

# Block N+1 (first block of SHA3-256 epoch)
block_n1 = BlockHeaderPQ(
    block_number=N+1,
    merkle_root=<SHA-256 Merkle root>,  # Still computed for compatibility
    prev_hash=<SHA-256 hash of block N>,  # Links to SHA-256 chain
    pq_merkle_root=<SHA3-256 Merkle root>,
    pq_prev_hash=<SHA3-256 hash of block N>,  # Links to SHA3-256 chain
    pq_algorithm=0x01,
    dual_commitment=<binding>,
)
```

**Validation Algorithm**:

```python
def validate_prev_hash_cross_epoch(block: BlockHeaderPQ, prev_block: BlockHeaderPQ) -> bool:
    """
    Validate prev_hash linkage across epoch boundaries.
    """
    # Determine epochs
    block_epoch = get_epoch_for_block(block.block_number)
    prev_epoch = get_epoch_for_block(prev_block.block_number)
    
    # Same epoch: standard validation
    if block_epoch.algorithm_id == prev_epoch.algorithm_id:
        expected_prev_hash = hash_block_versioned(
            prev_block,
            algorithm_id=prev_epoch.algorithm_id
        )
        return block.prev_hash == expected_prev_hash
    
    # Cross-epoch: validate both chains
    else:
        # Legacy chain: use prev_epoch algorithm
        expected_legacy_prev = hash_block_versioned(
            prev_block,
            algorithm_id=prev_epoch.algorithm_id
        )
        
        # PQ chain: use block_epoch algorithm
        expected_pq_prev = hash_block_versioned(
            prev_block,
            algorithm_id=block_epoch.algorithm_id
        )
        
        # Both must be valid
        return (
            block.prev_hash == expected_legacy_prev and
            block.pq_prev_hash == expected_pq_prev
        )
```

### Epoch Cutover Semantics

**Cutover Block Requirements**:

1. **Block Number**: Must equal epoch.start_block exactly
2. **Dual Commitment**: Must contain valid dual commitment
3. **Algorithm Switch**: pq_algorithm must match new epoch's algorithm_id
4. **Linkage**: Both prev_hash and pq_prev_hash must link to previous block
5. **Validation**: Both hash chains must validate correctly

**Cutover Validation Pseudocode**:

```python
def validate_epoch_cutover(block: BlockHeaderPQ, prev_block: BlockHeaderPQ) -> bool:
    """
    Validate epoch cutover block.
    """
    # Check if this is a cutover block
    block_epoch = get_epoch_for_block(block.block_number)
    prev_epoch = get_epoch_for_block(prev_block.block_number)
    
    if block_epoch.algorithm_id == prev_epoch.algorithm_id:
        return True  # Not a cutover block
    
    # This is a cutover block
    if block.block_number != block_epoch.start_block:
        return False  # Cutover block must be at epoch start
    
    # Validate dual commitment
    if not block.has_dual_commitment():
        return False
    
    if not block.verify_dual_commitment():
        return False
    
    # Validate algorithm switch
    if block.pq_algorithm != block_epoch.algorithm_id:
        return False
    
    # Validate both prev_hash chains
    if not validate_prev_hash_cross_epoch(block, prev_block):
        return False
    
    # Validate both Merkle roots
    legacy_valid = verify_merkle_root_historical(
        block.block_number,
        block.statements,
        block.merkle_root
    )
    
    pq_valid = verify_merkle_root_historical(
        block.block_number,
        block.statements,
        block.pq_merkle_root
    )
    
    return legacy_valid and pq_valid
```

## Reorganization Handling Policy

### Reorg Challenges in Dual-Hash Environment

During the grace period (Phase 3-4), blocks contain both SHA-256 and PQ hashes. Reorganizations must handle both hash chains consistently.

### Reorg Policy

**POLICY R1: Canonical Chain Selection**

During Phase 1-3 (legacy canonical):
- Chain selection uses legacy prev_hash
- Longest valid SHA-256 chain wins
- PQ hashes are validated but not used for chain selection

During Phase 4-5 (PQ canonical):
- Chain selection uses pq_prev_hash
- Longest valid PQ chain wins
- Legacy hashes are validated but not used for chain selection

**POLICY R2: Dual Chain Consistency**

All reorg candidates must satisfy:
- Both hash chains must be valid
- Both hash chains must have consistent block numbers
- Dual commitments must be valid for all blocks

**POLICY R3: Epoch Boundary Protection**

Reorgs that cross epoch boundaries are RESTRICTED:
- Cannot reorg past an epoch cutover block after finality (e.g., 100 blocks)
- Epoch cutover blocks are treated as checkpoints
- This prevents consensus confusion about which algorithm is canonical

**POLICY R4: Reorg Validation Algorithm**

```python
def validate_reorg_candidate(
    current_chain: List[BlockHeaderPQ],
    candidate_chain: List[BlockHeaderPQ],
    fork_point: int
) -> bool:
    """
    Validate a reorg candidate chain.
    """
    # Check if reorg crosses epoch boundary
    fork_epoch = get_epoch_for_block(fork_point)
    candidate_epochs = [get_epoch_for_block(b.block_number) for b in candidate_chain]
    
    # Reject reorg past epoch cutover after finality
    for epoch in candidate_epochs:
        if epoch.algorithm_id != fork_epoch.algorithm_id:
            # Check if cutover block has finality
            cutover_block = epoch.start_block
            if fork_point < cutover_block - FINALITY_DEPTH:
                return False  # Reorg past finalized cutover
    
    # Validate all blocks in candidate chain
    for i, block in enumerate(candidate_chain):
        if i == 0:
            prev_block = get_block(fork_point)
        else:
            prev_block = candidate_chain[i - 1]
        
        # Validate block
        if not validate_block_full(block, prev_block):
            return False
    
    # Check chain length (longest chain wins)
    current_length = len(current_chain)
    candidate_length = len(candidate_chain)
    
    # Use canonical hash for comparison
    current_epoch = get_epoch_for_block(current_chain[-1].block_number)
    if current_epoch.rule_version in ["v1-legacy", "v2-dual-optional", "v2-dual-required"]:
        # Legacy canonical: use SHA-256 chain length
        return candidate_length > current_length
    else:
        # PQ canonical: use PQ chain length
        return candidate_length > current_length
```

### Reorg Attack Mitigation

**Attack Vector**: Attacker creates a longer chain with invalid PQ hashes but valid legacy hashes (or vice versa).

**Mitigation**: Dual chain consistency requirement (POLICY R2) prevents this. Both chains must be valid, so attacker must break both hash functions.

**Attack Vector**: Attacker creates a reorg that crosses epoch boundary to confuse nodes about canonical algorithm.

**Mitigation**: Epoch boundary protection (POLICY R3) prevents reorgs past finalized cutover blocks.

## Consensus Violation Detection

### Violation Types

**V1: Invalid Merkle Root**
- merkle_root does not match computed value
- Severity: CRITICAL
- Action: REJECT block

**V2: Invalid Previous Hash**
- prev_hash does not link to previous block
- Severity: CRITICAL
- Action: REJECT block

**V3: Invalid Dual Commitment**
- dual_commitment does not match computed value
- Severity: CRITICAL (Phase 3+)
- Action: REJECT block

**V4: Missing PQ Fields**
- PQ fields absent when required (Phase 3+)
- Severity: CRITICAL
- Action: REJECT block

**V5: Algorithm Mismatch**
- pq_algorithm does not match epoch's canonical algorithm
- Severity: CRITICAL
- Action: REJECT block

**V6: Epoch Cutover Violation**
- Cutover block does not satisfy cutover requirements
- Severity: CRITICAL
- Action: REJECT block, ALERT network

**V7: Timestamp Violation**
- Timestamp out of acceptable range
- Severity: HIGH
- Action: REJECT block

**V8: Block Number Discontinuity**
- Block number does not follow prev_block
- Severity: CRITICAL
- Action: REJECT block

### Violation Response Protocol

```python
def handle_consensus_violation(
    block: BlockHeaderPQ,
    violation_type: str,
    details: str
) -> None:
    """
    Handle detected consensus violation.
    """
    # Log violation
    log_consensus_violation(
        block_number=block.block_number,
        block_hash=hash_block_versioned(block, algorithm_id=0x00),
        violation_type=violation_type,
        details=details,
        timestamp=time.time()
    )
    
    # Reject block
    reject_block(block)
    
    # Alert network (for critical violations)
    if violation_type in ["V3", "V4", "V5", "V6"]:
        broadcast_violation_alert(
            block_number=block.block_number,
            violation_type=violation_type,
            node_id=get_node_id()
        )
    
    # Penalize peer (if applicable)
    if block_source := get_block_source(block):
        penalize_peer(
            peer_id=block_source,
            violation_type=violation_type,
            severity=get_violation_severity(violation_type)
        )
```

## Implementation Requirements

### Consensus Module Structure

```
backend/consensus/
├── __init__.py
├── rules.py              # Consensus rule implementations
├── validation.py         # Block validation logic
├── reorg.py              # Reorganization handling
├── epoch.py              # Epoch management
└── violations.py         # Violation detection and handling
```

### Required Functions

```python
# rules.py
def validate_block_phase1(block: BlockHeaderPQ, prev_block: BlockHeaderPQ) -> bool
def validate_block_phase2(block: BlockHeaderPQ, prev_block: BlockHeaderPQ) -> bool
def validate_block_phase3(block: BlockHeaderPQ, prev_block: BlockHeaderPQ) -> bool
def validate_block_phase4(block: BlockHeaderPQ, prev_block: BlockHeaderPQ) -> bool
def validate_block_phase5(block: BlockHeaderPQ, prev_block: BlockHeaderPQ) -> bool

# validation.py
def validate_block_full(block: BlockHeaderPQ, prev_block: BlockHeaderPQ) -> Tuple[bool, Optional[str]]
def validate_merkle_root(block: BlockHeaderPQ) -> bool
def validate_prev_hash(block: BlockHeaderPQ, prev_block: BlockHeaderPQ) -> bool
def validate_dual_commitment(block: BlockHeaderPQ) -> bool
def validate_epoch_cutover(block: BlockHeaderPQ, prev_block: BlockHeaderPQ) -> bool

# reorg.py
def validate_reorg_candidate(current_chain: List[BlockHeaderPQ], candidate_chain: List[BlockHeaderPQ], fork_point: int) -> bool
def select_canonical_chain(chains: List[List[BlockHeaderPQ]]) -> List[BlockHeaderPQ]
def handle_reorg(new_chain: List[BlockHeaderPQ]) -> None

# epoch.py
def get_current_epoch() -> HashEpoch
def get_epoch_for_block(block_number: int) -> HashEpoch
def validate_epoch_transition(block: BlockHeaderPQ) -> bool
def activate_epoch(epoch: HashEpoch) -> None

# violations.py
def detect_violation(block: BlockHeaderPQ, prev_block: BlockHeaderPQ) -> Optional[Tuple[str, str]]
def handle_consensus_violation(block: BlockHeaderPQ, violation_type: str, details: str) -> None
def log_consensus_violation(...) -> None
```

## Testing Requirements

### Consensus Rule Tests

1. **Phase Transition Tests**: Validate correct behavior at each phase boundary
2. **Epoch Cutover Tests**: Validate epoch transition blocks
3. **Cross-Algorithm Tests**: Validate prev_hash linkage across algorithms
4. **Dual Commitment Tests**: Validate dual commitment computation and verification
5. **Reorg Tests**: Validate reorganization handling in dual-hash environment
6. **Violation Detection Tests**: Validate detection of all violation types
7. **Edge Case Tests**: Empty blocks, single-statement blocks, large blocks

### Integration Tests

1. **End-to-End Migration**: Simulate complete migration from Phase 1 to Phase 5
2. **Multi-Node Consensus**: Validate consensus across multiple nodes
3. **Network Partition**: Validate behavior under network partition
4. **Byzantine Nodes**: Validate behavior with malicious nodes

## Security Considerations

### Consensus Attack Vectors

**Attack 1: Dual Hash Inconsistency**
- Attacker creates block with valid legacy hash but invalid PQ hash
- Mitigation: Dual chain consistency requirement rejects such blocks

**Attack 2: Epoch Confusion**
- Attacker creates reorg across epoch boundary to confuse nodes
- Mitigation: Epoch boundary protection prevents reorgs past finalized cutover

**Attack 3: Algorithm Downgrade**
- Attacker tries to force network back to SHA-256
- Mitigation: Epoch immutability prevents downgrade

**Attack 4: Selective Forgery**
- Attacker forges one hash chain while keeping other valid
- Mitigation: Dual commitment binding requires breaking both chains

### Consensus Safety Properties

**Property 1: Agreement**
- All honest nodes agree on the canonical chain
- Guaranteed by: Deterministic validation rules, epoch immutability

**Property 2: Validity**
- All blocks in canonical chain satisfy consensus rules
- Guaranteed by: Comprehensive validation, violation detection

**Property 3: Termination**
- Consensus eventually reaches agreement
- Guaranteed by: Longest chain rule, finality checkpoints

**Property 4: Historical Consistency**
- Historical blocks remain verifiable under original algorithms
- Guaranteed by: Epoch-aware verification, algorithm versioning

## Governance Integration

Consensus rule changes require governance approval:

1. **Proposal**: Community member proposes epoch activation
2. **Review**: Technical review of activation parameters
3. **Vote**: Governance vote on activation block number
4. **Activation**: Epoch activated at approved block number

See "PQ Activation Governance Process" document for details.

---

**Document Version**: 1.0  
**Last Updated**: 2024-12-06  
**Author**: Manus-H (Quantum-Migration Engineer)  
**Status**: Draft - Pending Community Review
