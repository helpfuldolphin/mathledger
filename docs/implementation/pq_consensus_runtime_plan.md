## PQ Consensus Runtime Implementation Plan

**Document Version**: 1.0  
**Author**: Manus-H  
**Date**: 2024-12-06

### Executive Summary

This document provides the complete implementation plan for the post-quantum consensus runtime. All core modules have been implemented and are ready for integration testing and deployment.

### Module Structure

```
backend/consensus_pq/
├── __init__.py              # Module exports and version info
├── rules.py                 # Phase-specific consensus rules (198 lines)
├── epoch.py                 # Epoch management and resolution (219 lines)
├── prev_hash.py             # Prev-hash validation (154 lines)
├── validation.py            # Block validation logic (193 lines)
├── violations.py            # Violation detection and classification (284 lines)
└── reorg.py                 # Reorganization handling (249 lines)

Total: 1,297 lines of production code
```

### Module Signatures

#### 1. Rules Module (`rules.py`)

**Core Types**:
```python
class ConsensusRuleVersion(Enum):
    V1_LEGACY = "v1-legacy"
    V2_DUAL_OPTIONAL = "v2-dual-optional"
    V2_DUAL_REQUIRED = "v2-dual-required"
    V2_PQ_PRIMARY = "v2-pq-primary"
    V3_PQ_ONLY = "v3-pq-only"

@dataclass(frozen=True)
class ConsensusRules:
    version: ConsensusRuleVersion
    pq_fields_required: bool
    pq_fields_validated: bool
    legacy_fields_required: bool
    legacy_fields_canonical: bool
    pq_fields_canonical: bool
    dual_commitment_required: bool
    dual_commitment_validated: bool
```

**Key Functions**:
```python
def get_consensus_rules_for_phase(rule_version: ConsensusRuleVersion) -> ConsensusRules
def validate_consensus_rules(block: BlockHeaderPQ, rules: ConsensusRules) -> tuple[bool, Optional[str]]
def get_canonical_hash_fields(block: BlockHeaderPQ, rules: ConsensusRules) -> tuple[str, Optional[str]]
def requires_dual_validation(rules: ConsensusRules) -> bool
```

#### 2. Epoch Module (`epoch.py`)

**Core Types**:
```python
@dataclass(frozen=True)
class HashEpoch:
    start_block: int
    end_block: Optional[int]
    algorithm_id: int
    algorithm_name: str
    rule_version: str
    activation_timestamp: float
    governance_hash: str
```

**Key Functions**:
```python
def register_epoch(epoch: HashEpoch) -> None
def get_epoch_for_block(block_number: int) -> Optional[HashEpoch]
def get_current_epoch() -> Optional[HashEpoch]
def list_epochs() -> List[HashEpoch]
def get_epoch_by_governance_hash(governance_hash: str) -> Optional[HashEpoch]
def is_epoch_transition(block_number: int) -> bool
def get_next_epoch_transition(current_block: int) -> Optional[int]
def initialize_genesis_epoch() -> None
```

**Epoch Invariants**:
- Immutability: Once registered, epochs cannot be modified
- Monotonicity: Epoch start blocks must be strictly increasing
- Continuity: No gaps allowed between epochs
- Finality: Epoch transitions require 100 blocks confirmation

#### 3. Prev-Hash Module (`prev_hash.py`)

**Key Functions**:
```python
def compute_prev_hash(prev_block: BlockHeaderPQ, algorithm_id: int) -> str
def validate_prev_hash_linkage(block: BlockHeaderPQ, prev_block: BlockHeaderPQ) -> Tuple[bool, Optional[str]]
def validate_dual_prev_hash(block: BlockHeaderPQ, prev_block: BlockHeaderPQ) -> Tuple[bool, bool, Optional[str], Optional[str]]
def validate_genesis_block(block: BlockHeaderPQ) -> Tuple[bool, Optional[str]]
```

**Prev-Hash Rules**:
- **Legacy Chain**: `prev_hash` computed using previous epoch's algorithm
- **PQ Chain**: `pq_prev_hash` computed using current block's PQ algorithm
- **Epoch Transitions**: Both chains must be valid across boundaries
- **Genesis Block**: `prev_hash` must be all zeros

#### 4. Validation Module (`validation.py`)

**Key Functions**:
```python
def validate_merkle_root(block: BlockHeaderPQ, algorithm_id: int) -> Tuple[bool, Optional[str]]
def validate_dual_commitment(block: BlockHeaderPQ) -> Tuple[bool, Optional[str]]
def validate_block_header(block: BlockHeaderPQ, prev_block: Optional[BlockHeaderPQ]) -> Tuple[bool, Optional[str]]
def validate_block_full(block: BlockHeaderPQ, prev_block: Optional[BlockHeaderPQ]) -> Tuple[bool, Optional[str]]
def validate_block_batch(blocks: list[BlockHeaderPQ]) -> Tuple[bool, Optional[str], Optional[int]]
```

**Validation Sequence**:
1. **Header Validation**: Check structural requirements (fields present, types correct)
2. **Merkle Root Validation**: Recompute and verify both legacy and PQ Merkle roots
3. **Dual Commitment Validation**: Verify binding of legacy and PQ hashes
4. **Prev-Hash Validation**: Verify chain linkage for both legacy and PQ chains
5. **Consensus Rules Validation**: Verify block conforms to epoch's consensus rules

#### 5. Violations Module (`violations.py`)

**Core Types**:
```python
class ViolationType(Enum):
    INVALID_MERKLE_ROOT = "invalid_merkle_root"
    INVALID_PQ_MERKLE_ROOT = "invalid_pq_merkle_root"
    INVALID_PREV_HASH = "invalid_prev_hash"
    INVALID_PQ_PREV_HASH = "invalid_pq_prev_hash"
    INVALID_DUAL_COMMITMENT = "invalid_dual_commitment"
    MISSING_PQ_FIELDS = "missing_pq_fields"
    MISSING_LEGACY_FIELDS = "missing_legacy_fields"
    ALGORITHM_MISMATCH = "algorithm_mismatch"
    EPOCH_CUTOVER_VIOLATION = "epoch_cutover_violation"
    TIMESTAMP_VIOLATION = "timestamp_violation"
    BLOCK_NUMBER_DISCONTINUITY = "block_number_discontinuity"

class ViolationSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class ConsensusViolation:
    violation_type: ViolationType
    severity: ViolationSeverity
    block_number: int
    block_hash: Optional[str]
    message: str
    details: Optional[dict]
```

**Key Functions**:
```python
def classify_violation(violation_type: ViolationType) -> ViolationSeverity
def detect_violations(block: BlockHeaderPQ, prev_block: Optional[BlockHeaderPQ]) -> List[ConsensusViolation]
def has_critical_violations(violations: List[ConsensusViolation]) -> bool
def format_violation_report(violations: List[ConsensusViolation]) -> str
```

**Violation Classification**:
- **CRITICAL**: Consensus-breaking (invalid hashes, algorithm mismatch, epoch violation)
- **HIGH**: Serious issues (missing required fields, block discontinuity)
- **MEDIUM**: Warnings (timestamp violations)
- **LOW**: Informational

#### 6. Reorg Module (`reorg.py`)

**Core Types**:
```python
@dataclass
class ReorgEvaluation:
    can_reorg: bool
    fork_point: int
    current_chain_length: int
    candidate_chain_length: int
    reason: str
```

**Key Functions**:
```python
def find_fork_point(current_chain: List[BlockHeaderPQ], candidate_chain: List[BlockHeaderPQ]) -> Optional[int]
def validate_reorg_constraints(current_chain: List[BlockHeaderPQ], candidate_chain: List[BlockHeaderPQ], fork_point: int) -> Tuple[bool, Optional[str]]
def evaluate_reorg(current_chain: List[BlockHeaderPQ], candidate_chain: List[BlockHeaderPQ]) -> ReorgEvaluation
def can_reorg_to_chain(current_chain: List[BlockHeaderPQ], candidate_chain: List[BlockHeaderPQ]) -> bool
def validate_candidate_chain(candidate_chain: List[BlockHeaderPQ]) -> Tuple[bool, Optional[str]]
```

**Reorg Constraints**:
- **Finality Depth**: 100 blocks (configurable constant)
- **Epoch Boundary Protection**: Cannot reorg across finalized epoch boundaries
- **Chain Length**: Candidate must be longer than current chain
- **Full Validation**: Candidate chain must be fully valid

**Fork Choice Rule**:
- **Phases 1-3** (legacy canonical): Longest valid legacy chain wins
- **Phases 4-5** (PQ canonical): Longest valid PQ chain wins

### Epoch Cutover Validation

**Cutover Rules**:

1. **Phase 0 → Phase 1** (Scaffolding Deployment):
   - No validation changes
   - PQ fields ignored if present

2. **Phase 1 → Phase 2** (Dual Optional):
   - PQ fields validated if present
   - Blocks without PQ fields still valid

3. **Phase 2 → Phase 3** (Dual Required):
   - **Cutover Block**: First block in new epoch MUST have valid PQ fields
   - Both legacy and PQ chains MUST be valid
   - Dual commitment MUST be present and valid

4. **Phase 3 → Phase 4** (PQ Primary):
   - **Cutover Block**: PQ chain becomes canonical for fork choice
   - Legacy chain still validated for compatibility
   - Dual commitment still required

5. **Phase 4 → Phase 5** (PQ Only):
   - **Cutover Block**: Legacy fields become optional
   - Only PQ chain validated for new blocks
   - Historical blocks still verified with original algorithms

**Cutover Validation Logic**:
```python
def validate_epoch_cutover(block: BlockHeaderPQ, prev_block: BlockHeaderPQ) -> Tuple[bool, Optional[str]]:
    """Validate that a block correctly transitions between epochs."""
    
    block_epoch = get_epoch_for_block(block.block_number)
    prev_epoch = get_epoch_for_block(prev_block.block_number)
    
    if block_epoch == prev_epoch:
        return True, None  # Not a cutover block
    
    # This is a cutover block
    if block.block_number != block_epoch.start_block:
        return False, "Cutover block number mismatch"
    
    # Validate cutover-specific requirements
    new_rule_version = ConsensusRuleVersion(block_epoch.rule_version)
    new_rules = get_consensus_rules_for_phase(new_rule_version)
    
    # Check that block satisfies new epoch's requirements
    rules_valid, rules_error = validate_consensus_rules(block, new_rules)
    if not rules_valid:
        return False, f"Cutover block violates new epoch rules: {rules_error}"
    
    return True, None
```

### Reorg Boundary Enforcement

**Finality Rules**:

1. **Finality Depth**: 100 blocks
   - Reorgs deeper than 100 blocks are forbidden
   - This provides economic finality

2. **Epoch Boundary Finality**:
   - Once an epoch has been active for >100 blocks, its boundary is finalized
   - Reorgs cannot cross finalized epoch boundaries
   - This prevents malicious reorgs from undoing algorithm transitions

3. **Reorg Validation**:
```python
def validate_reorg_constraints(current_chain, candidate_chain, fork_point):
    # Check finality depth
    reorg_depth = current_tip - fork_point
    if reorg_depth > FINALITY_DEPTH:
        return False, "Reorg too deep"
    
    # Check epoch boundary crossing
    fork_epoch = get_epoch_for_block(fork_point)
    current_epoch = get_epoch_for_block(current_tip)
    
    if fork_epoch != current_epoch:
        blocks_since_epoch_start = current_tip - current_epoch.start_block
        if blocks_since_epoch_start > FINALITY_DEPTH:
            return False, "Cannot reorg across finalized epoch boundary"
    
    return True, None
```

### Test Suite Catalogue

#### Unit Tests (`tests/unit/consensus_pq/`)

**test_rules.py** (Testing consensus rules):
- `test_get_consensus_rules_for_phase`: Verify rule retrieval
- `test_validate_consensus_rules_v1_legacy`: Test Phase 0-1 rules
- `test_validate_consensus_rules_v2_dual_optional`: Test Phase 2 rules
- `test_validate_consensus_rules_v2_dual_required`: Test Phase 3 rules
- `test_validate_consensus_rules_v2_pq_primary`: Test Phase 4 rules
- `test_validate_consensus_rules_v3_pq_only`: Test Phase 5 rules
- `test_get_canonical_hash_fields`: Test canonical field selection
- `test_requires_dual_validation`: Test dual validation requirement

**test_epoch.py** (Testing epoch management):
- `test_register_epoch`: Test epoch registration
- `test_epoch_overlap_detection`: Test overlap prevention
- `test_get_epoch_for_block`: Test epoch resolution
- `test_get_current_epoch`: Test current epoch retrieval
- `test_is_epoch_transition`: Test transition detection
- `test_get_next_epoch_transition`: Test future transition lookup
- `test_initialize_genesis_epoch`: Test genesis initialization

**test_prev_hash.py** (Testing prev-hash validation):
- `test_compute_prev_hash`: Test prev-hash computation
- `test_validate_prev_hash_linkage`: Test linkage validation
- `test_validate_dual_prev_hash`: Test dual chain validation
- `test_validate_genesis_block`: Test genesis validation
- `test_prev_hash_across_epoch_boundary`: Test epoch transition

**test_validation.py** (Testing block validation):
- `test_validate_merkle_root`: Test Merkle root validation
- `test_validate_dual_commitment`: Test dual commitment validation
- `test_validate_block_header`: Test header validation
- `test_validate_block_full`: Test full block validation
- `test_validate_block_batch`: Test batch validation

**test_violations.py** (Testing violation detection):
- `test_classify_violation`: Test severity classification
- `test_detect_violations_invalid_merkle_root`: Test Merkle root violation
- `test_detect_violations_missing_pq_fields`: Test missing fields
- `test_detect_violations_algorithm_mismatch`: Test algorithm mismatch
- `test_has_critical_violations`: Test critical violation detection
- `test_format_violation_report`: Test report formatting

**test_reorg.py** (Testing reorganization):
- `test_find_fork_point`: Test fork point detection
- `test_validate_reorg_constraints`: Test reorg constraints
- `test_evaluate_reorg_longer_chain`: Test fork choice
- `test_evaluate_reorg_finality_depth`: Test finality enforcement
- `test_evaluate_reorg_epoch_boundary`: Test epoch boundary protection
- `test_can_reorg_to_chain`: Test reorg decision

#### Integration Tests (`tests/integration/consensus_pq/`)

**test_phase_transitions.py** (Testing phase transitions):
- `test_phase_1_to_phase_2_transition`: Test scaffolding to dual-optional
- `test_phase_2_to_phase_3_transition`: Test dual-optional to dual-required
- `test_phase_3_to_phase_4_transition`: Test dual-required to PQ-primary
- `test_phase_4_to_phase_5_transition`: Test PQ-primary to PQ-only

**test_epoch_cutover.py** (Testing epoch cutover):
- `test_first_pq_block_validation`: Test first PQ block
- `test_cutover_block_requirements`: Test cutover requirements
- `test_cutover_with_invalid_pq_fields`: Test invalid cutover
- `test_cutover_across_multiple_epochs`: Test multi-epoch chain

**test_reorg_scenarios.py** (Testing reorg scenarios):
- `test_reorg_within_epoch`: Test reorg within single epoch
- `test_reorg_across_epoch_boundary`: Test reorg across epochs
- `test_reorg_beyond_finality_depth`: Test deep reorg rejection
- `test_reorg_with_invalid_candidate`: Test invalid candidate rejection

**test_historical_verification.py** (Testing historical verification):
- `test_verify_legacy_blocks_after_pq_activation`: Test legacy verification
- `test_verify_blocks_across_all_epochs`: Test multi-epoch verification
- `test_verify_genesis_to_current`: Test full chain verification

### Integration Requirements

**Dependencies**:
- `basis.ledger.block_pq`: Block header structures
- `basis.crypto.hash_versioned`: Versioned hashing primitives
- `basis.crypto.hash_registry`: Hash algorithm registry

**Database Integration**:
- Epoch registry should be persisted to database
- Current implementation uses in-memory storage (for testing)
- Production deployment requires database backend

**API Integration**:
- Consensus validation should be called from block sealing logic
- Reorg evaluation should be called from chain synchronization logic
- Violation detection should feed into monitoring/alerting system

### Deployment Checklist

- [ ] Run full unit test suite (target: 100% pass rate)
- [ ] Run integration test suite (target: 100% pass rate)
- [ ] Deploy to testnet for validation
- [ ] Monitor testnet for 1000 blocks
- [ ] Conduct security audit of consensus module
- [ ] Deploy to mainnet with genesis epoch
- [ ] Monitor mainnet consensus health

### Performance Considerations

**Optimization Opportunities**:
1. **Merkle Root Caching**: Cache computed Merkle roots to avoid recomputation
2. **Epoch Lookup Caching**: Cache epoch lookups for recent blocks
3. **Validation Parallelization**: Validate legacy and PQ chains in parallel
4. **Batch Validation**: Optimize batch validation with vectorized operations

**Expected Performance**:
- Block validation: <100ms per block
- Reorg evaluation: <500ms for 100-block reorg
- Violation detection: <50ms per block

---

**Document Version**: 1.0  
**Status**: Implementation Complete  
**Next Steps**: Test suite implementation and integration testing
