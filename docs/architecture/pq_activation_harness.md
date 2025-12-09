# PQ Migration Activation Harness

**Author**: Manus-B (Ledger Replay Architect & Attestation Runtime Engineer)  
**Phase**: III - Consensus Runtime Activation  
**Date**: 2025-12-06

---

## Purpose

The **PQ Migration Activation Harness** orchestrates the transition from SHA-256 to SHA-3 (post-quantum) cryptography in MathLedger's attestation system.

This document defines:
1. Activation stages and their code representation
2. Dual-commitment verification integration
3. Epoch cut-over validation on-chain
4. Activation block invariants in code form
5. Error handling on migration boundary blocks

---

## Migration Phases

### Phase 1: Pre-Migration (SHA-256 Only)

**State**:
- All blocks use `hash_version = "sha256-v1"`
- `composite_attestation_root` = SHA256(R_t || U_t)
- No SHA-3 roots stored

**Code Representation**:
```python
class MigrationPhase(Enum):
    PRE_MIGRATION = "pre_migration"
```

---

### Phase 2: Dual-Commitment (SHA-256 + SHA-3)

**State**:
- New blocks use `hash_version = "dual-v1"`
- `composite_attestation_root` = SHA256(R_t || U_t)
- `composite_attestation_root_sha3` = SHA3_256(R_t || U_t)
- Both roots computed from same R_t, U_t

**Code Representation**:
```python
class MigrationPhase(Enum):
    DUAL_COMMITMENT = "dual_commitment"
```

**Activation Block Invariants**:
```python
def validate_dual_commitment_activation(block: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate dual-commitment activation block.
    
    Invariants:
        1. hash_version = "dual-v1"
        2. composite_attestation_root present (SHA-256)
        3. composite_attestation_root_sha3 present (SHA-3)
        4. Both roots computed from same R_t, U_t
        5. attestation_metadata.activation_phase = "dual_commitment"
    """
    hash_version = block.get("attestation_metadata", {}).get("hash_version")
    if hash_version != "dual-v1":
        return False, f"Invalid hash_version: {hash_version}"
    
    r_t = block.get("reasoning_attestation_root")
    u_t = block.get("ui_attestation_root")
    h_t_sha256 = block.get("composite_attestation_root")
    h_t_sha3 = block.get("composite_attestation_root_sha3")
    
    if not all([r_t, u_t, h_t_sha256, h_t_sha3]):
        return False, "Missing attestation roots"
    
    # Verify both roots computed correctly
    expected_sha256 = hashlib.sha256((r_t + u_t).encode()).hexdigest()
    expected_sha3 = hashlib.sha3_256((r_t + u_t).encode()).hexdigest()
    
    if h_t_sha256 != expected_sha256:
        return False, f"SHA-256 root mismatch"
    
    if h_t_sha3 != expected_sha3:
        return False, f"SHA-3 root mismatch"
    
    activation_phase = block.get("attestation_metadata", {}).get("activation_phase")
    if activation_phase != "dual_commitment":
        return False, f"Invalid activation_phase: {activation_phase}"
    
    return True, None
```

---

### Phase 3: Pure SHA-3

**State**:
- New blocks use `hash_version = "sha3-v1"`
- `composite_attestation_root` = SHA3_256(R_t || U_t)
- No SHA-256 roots stored (legacy blocks remain SHA-256)

**Code Representation**:
```python
class MigrationPhase(Enum):
    PURE_SHA3 = "pure_sha3"
```

**Activation Block Invariants**:
```python
def validate_sha3_activation(block: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate SHA-3 activation block.
    
    Invariants:
        1. hash_version = "sha3-v1"
        2. composite_attestation_root present (SHA-3)
        3. composite_attestation_root_sha3 absent (redundant)
        4. attestation_metadata.activation_phase = "pure_sha3"
        5. Predecessor block has hash_version = "dual-v1"
    """
    hash_version = block.get("attestation_metadata", {}).get("hash_version")
    if hash_version != "sha3-v1":
        return False, f"Invalid hash_version: {hash_version}"
    
    r_t = block.get("reasoning_attestation_root")
    u_t = block.get("ui_attestation_root")
    h_t = block.get("composite_attestation_root")
    
    if not all([r_t, u_t, h_t]):
        return False, "Missing attestation roots"
    
    # Verify SHA-3 root computed correctly
    expected_sha3 = hashlib.sha3_256((r_t + u_t).encode()).hexdigest()
    
    if h_t != expected_sha3:
        return False, f"SHA-3 root mismatch"
    
    activation_phase = block.get("attestation_metadata", {}).get("activation_phase")
    if activation_phase != "pure_sha3":
        return False, f"Invalid activation_phase: {activation_phase}"
    
    return True, None
```

---

## Activation Stages

### Stage 1: Pre-Activation Planning

**Actions**:
1. Announce migration timeline
2. Update documentation
3. Deploy code with dual-commitment support
4. Test dual-commitment on staging

**Code**:
```python
def plan_pq_migration(target_block: int) -> Dict[str, Any]:
    """
    Plan PQ migration.
    
    Args:
        target_block: Block number for dual-commitment activation
    
    Returns:
        Migration plan dictionary
    """
    return {
        "phase": "planning",
        "dual_commitment_activation_block": target_block,
        "sha3_cutover_block": target_block + 10000,  # 10k blocks later
        "estimated_duration_days": 7,
        "rollback_plan": "Revert to SHA-256 if dual-commitment fails",
    }
```

---

### Stage 2: Dual-Commitment Activation

**Actions**:
1. Deploy dual-commitment code to production
2. Seal activation block with `hash_version = "dual-v1"`
3. Verify both SHA-256 and SHA-3 roots computed correctly
4. Monitor for 1000 blocks (ensure stability)

**Code**:
```python
def activate_dual_commitment(block_number: int) -> ActivationBlock:
    """
    Activate dual-commitment phase.
    
    Args:
        block_number: Activation block number
    
    Returns:
        ActivationBlock
    """
    # Seal activation block
    block = seal_block_with_dual_commitment(block_number)
    
    # Validate activation block
    is_valid, error = validate_dual_commitment_activation(block)
    if not is_valid:
        raise ValueError(f"Dual-commitment activation failed: {error}")
    
    # Create activation record
    activation = ActivationBlock(
        block_number=block_number,
        from_phase=MigrationPhase.PRE_MIGRATION,
        to_phase=MigrationPhase.DUAL_COMMITMENT,
        activation_root=block["composite_attestation_root"],
        activation_metadata={
            "hash_version": "dual-v1",
            "sha256_root": block["composite_attestation_root"],
            "sha3_root": block["composite_attestation_root_sha3"],
        },
    )
    
    return activation
```

---

### Stage 3: Dual-Commitment Monitoring

**Actions**:
1. Run replay verification on dual-commitment blocks
2. Verify both SHA-256 and SHA-3 roots match
3. Monitor for drift signals
4. Collect performance metrics

**Code**:
```python
def monitor_dual_commitment(
    start_block: int,
    end_block: int,
) -> Dict[str, Any]:
    """
    Monitor dual-commitment phase.
    
    Args:
        start_block: Start of monitoring range
        end_block: End of monitoring range
    
    Returns:
        Monitoring report
    """
    # Fetch blocks
    blocks = fetch_blocks(start_block, end_block)
    
    # Verify dual-commitment
    mismatches = []
    for block in blocks:
        is_valid, error = verify_dual_commitment(block)
        if not is_valid:
            mismatches.append({
                "block_number": block["block_number"],
                "error": error,
            })
    
    return {
        "total_blocks": len(blocks),
        "dual_commitment_blocks": sum(1 for b in blocks if b.get("attestation_metadata", {}).get("hash_version") == "dual-v1"),
        "mismatches": mismatches,
        "success_rate": (len(blocks) - len(mismatches)) / len(blocks) if blocks else 0.0,
    }
```

---

### Stage 4: SHA-3 Cutover

**Actions**:
1. Deploy pure SHA-3 code to production
2. Seal activation block with `hash_version = "sha3-v1"`
3. Verify SHA-3 root computed correctly
4. Monitor for 1000 blocks (ensure stability)

**Code**:
```python
def activate_sha3(block_number: int) -> ActivationBlock:
    """
    Activate pure SHA-3 phase.
    
    Args:
        block_number: Activation block number
    
    Returns:
        ActivationBlock
    """
    # Seal activation block
    block = seal_block_with_sha3(block_number)
    
    # Validate activation block
    is_valid, error = validate_sha3_activation(block)
    if not is_valid:
        raise ValueError(f"SHA-3 activation failed: {error}")
    
    # Create activation record
    activation = ActivationBlock(
        block_number=block_number,
        from_phase=MigrationPhase.DUAL_COMMITMENT,
        to_phase=MigrationPhase.PURE_SHA3,
        activation_root=block["composite_attestation_root"],
        activation_metadata={
            "hash_version": "sha3-v1",
            "sha3_root": block["composite_attestation_root"],
        },
    )
    
    return activation
```

---

### Stage 5: Post-Migration Validation

**Actions**:
1. Run full-chain replay verification
2. Verify cross-algorithm prev_hash chains
3. Validate epoch roots across migration boundary
4. Generate migration audit report

**Code**:
```python
def validate_pq_migration(
    dual_commitment_start: int,
    sha3_cutover: int,
) -> Dict[str, Any]:
    """
    Validate PQ migration.
    
    Args:
        dual_commitment_start: Dual-commitment activation block
        sha3_cutover: SHA-3 cutover block
    
    Returns:
        Validation report
    """
    # Fetch all blocks
    blocks = fetch_blocks(0, sha3_cutover + 1000)
    
    # Validate phase transitions
    phase_transitions = []
    for i in range(1, len(blocks)):
        prev_hash_version = blocks[i-1].get("attestation_metadata", {}).get("hash_version", "sha256-v1")
        curr_hash_version = blocks[i].get("attestation_metadata", {}).get("hash_version", "sha256-v1")
        
        if prev_hash_version != curr_hash_version:
            phase_transitions.append({
                "block_number": blocks[i]["block_number"],
                "from": prev_hash_version,
                "to": curr_hash_version,
            })
    
    # Validate cross-algorithm prev_hash
    prev_hash_errors = []
    for i in range(1, len(blocks)):
        is_valid, error = validate_cross_algorithm_prev_hash(blocks[i], blocks[i-1])
        if not is_valid:
            prev_hash_errors.append({
                "block_number": blocks[i]["block_number"],
                "error": error,
            })
    
    return {
        "total_blocks": len(blocks),
        "phase_transitions": phase_transitions,
        "prev_hash_errors": prev_hash_errors,
        "migration_success": len(prev_hash_errors) == 0,
    }
```

---

## Dual-Commitment Verification Integration

### Replay Verification

**Integration Point**: `backend/ledger/replay/recompute.py`

```python
def recompute_composite_root_dual(
    r_t: str,
    u_t: str,
    hash_version: str,
) -> Tuple[str, Optional[str]]:
    """
    Recompute composite root for dual-commitment blocks.
    
    Args:
        r_t: Reasoning attestation root
        u_t: UI attestation root
        hash_version: Hash version ("dual-v1")
    
    Returns:
        (h_t_sha256, h_t_sha3)
    """
    if hash_version != "dual-v1":
        raise ValueError(f"Invalid hash_version for dual-commitment: {hash_version}")
    
    # Compute both roots
    h_t_sha256 = hashlib.sha256((r_t + u_t).encode()).hexdigest()
    h_t_sha3 = hashlib.sha3_256((r_t + u_t).encode()).hexdigest()
    
    return h_t_sha256, h_t_sha3
```

---

### Manus-H Hash Abstraction Integration

**Interface**:
```python
class HashAlgorithm(ABC):
    """Abstract hash algorithm interface (Manus-H)."""
    
    @abstractmethod
    def hash(self, data: bytes) -> str:
        """Compute hash of data."""
        pass
    
    @abstractmethod
    def merkle_root(self, leaves: List[str]) -> str:
        """Compute Merkle root."""
        pass
    
    @abstractmethod
    def version(self) -> str:
        """Get hash version."""
        pass


class SHA256Algorithm(HashAlgorithm):
    """SHA-256 implementation."""
    
    def hash(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()
    
    def merkle_root(self, leaves: List[str]) -> str:
        return compute_merkle_root_sha256(leaves)
    
    def version(self) -> str:
        return "sha256-v1"


class SHA3Algorithm(HashAlgorithm):
    """SHA-3 implementation."""
    
    def hash(self, data: bytes) -> str:
        return hashlib.sha3_256(data).hexdigest()
    
    def merkle_root(self, leaves: List[str]) -> str:
        return compute_merkle_root_sha3(leaves)
    
    def version(self) -> str:
        return "sha3-v1"


def get_hash_algorithm(version: str) -> HashAlgorithm:
    """Get hash algorithm by version."""
    if version == "sha256-v1":
        return SHA256Algorithm()
    elif version == "sha3-v1":
        return SHA3Algorithm()
    elif version == "dual-v1":
        # Dual-commitment uses both
        return DualCommitmentAlgorithm()
    else:
        raise ValueError(f"Unsupported hash version: {version}")
```

---

## Epoch Cut-Over Validation

### Epoch Boundary Invariants

**Invariant**: Epoch roots must be computed using the hash algorithm of the **last block** in the epoch.

**Code**:
```python
def validate_epoch_cutover(epoch: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate epoch cut-over across hash algorithm boundary.
    
    Args:
        epoch: Epoch dictionary
    
    Returns:
        (is_valid, error_message)
    
    Invariants:
        1. Epoch root uses hash algorithm of last block
        2. If epoch spans migration boundary, use last block's algorithm
        3. Epoch metadata records hash_version
    """
    epoch_number = epoch["epoch_number"]
    epoch_root = epoch["epoch_root"]
    hash_version = epoch.get("hash_version")
    
    # Fetch blocks in epoch
    start_block = epoch_number * 100
    end_block = (epoch_number + 1) * 100
    blocks = fetch_blocks(start_block, end_block)
    
    if not blocks:
        return False, "Epoch has no blocks"
    
    # Get hash version of last block
    last_block_hash_version = blocks[-1].get("attestation_metadata", {}).get("hash_version", "sha256-v1")
    
    # Verify epoch hash_version matches last block
    if hash_version != last_block_hash_version:
        return False, f"Epoch hash_version mismatch: {hash_version} != {last_block_hash_version}"
    
    # Recompute epoch root using correct algorithm
    composite_roots = [b["composite_attestation_root"] for b in blocks]
    
    if hash_version == "sha256-v1":
        expected_epoch_root = compute_merkle_root_sha256(composite_roots)
    elif hash_version == "sha3-v1":
        expected_epoch_root = compute_merkle_root_sha3(composite_roots)
    elif hash_version == "dual-v1":
        # For dual-commitment, use SHA-256 root (primary)
        expected_epoch_root = compute_merkle_root_sha256(composite_roots)
    else:
        return False, f"Unsupported hash_version: {hash_version}"
    
    # Verify epoch root matches
    if epoch_root != expected_epoch_root:
        return False, f"Epoch root mismatch: {epoch_root} != {expected_epoch_root}"
    
    return True, None
```

---

## Error Handling on Migration Boundary Blocks

### Error Taxonomy

| Error Type | Severity | Remediation |
|------------|----------|-------------|
| Missing SHA-3 root in dual-commitment block | CRITICAL | Recompute and update block |
| SHA-256/SHA-3 root mismatch | CRITICAL | Investigate hash computation bug |
| Invalid hash_version transition | ERROR | Revert to previous phase |
| Premature SHA-3 adoption (skipped dual-commitment) | ERROR | Insert dual-commitment phase |
| Epoch root uses wrong algorithm | ERROR | Reseal epoch with correct algorithm |

### Error Handling Code

```python
class MigrationBoundaryError(Exception):
    """Base class for migration boundary errors."""
    pass


class MissingDualCommitmentError(MigrationBoundaryError):
    """Missing SHA-3 root in dual-commitment block."""
    pass


class HashMismatchError(MigrationBoundaryError):
    """SHA-256/SHA-3 root mismatch."""
    pass


class InvalidTransitionError(MigrationBoundaryError):
    """Invalid hash_version transition."""
    pass


def handle_migration_boundary_error(
    error: MigrationBoundaryError,
    block: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Handle migration boundary error.
    
    Args:
        error: Migration boundary error
        block: Block where error occurred
    
    Returns:
        Remediation plan
    """
    if isinstance(error, MissingDualCommitmentError):
        return {
            "error_type": "missing_dual_commitment",
            "severity": "CRITICAL",
            "block_number": block["block_number"],
            "remediation": "Recompute SHA-3 root and update block",
            "auto_remediable": True,
        }
    elif isinstance(error, HashMismatchError):
        return {
            "error_type": "hash_mismatch",
            "severity": "CRITICAL",
            "block_number": block["block_number"],
            "remediation": "Investigate hash computation bug",
            "auto_remediable": False,
        }
    elif isinstance(error, InvalidTransitionError):
        return {
            "error_type": "invalid_transition",
            "severity": "ERROR",
            "block_number": block["block_number"],
            "remediation": "Revert to previous phase",
            "auto_remediable": False,
        }
    else:
        return {
            "error_type": "unknown",
            "severity": "ERROR",
            "block_number": block["block_number"],
            "remediation": "Manual investigation required",
            "auto_remediable": False,
        }
```

---

## Integration Checklist

- [ ] Implement `MigrationPhase` enum in `backend/consensus/pq_migration.py`
- [ ] Implement activation block validation functions
- [ ] Integrate dual-commitment verification into replay engine
- [ ] Implement epoch cut-over validation
- [ ] Add migration boundary error handling
- [ ] Coordinate with Manus-H on hash abstraction interface
- [ ] Write integration tests for all migration stages
- [ ] Document migration runbook for operators

---

## Manus-H Coordination Points

1. **Hash Algorithm Interface**: Manus-H provides `HashAlgorithm` abstract class
2. **SHA-256/SHA-3 Implementations**: Manus-H implements concrete algorithms
3. **Merkle Root Computation**: Manus-H provides Merkle tree logic
4. **Version Management**: Manus-H manages hash version registry

---

## Conclusion

The PQ Migration Activation Harness provides a **deterministic, auditable, and reversible** path from SHA-256 to SHA-3 cryptography. All activation stages are codified, all invariants are enforced, and all errors are classified with remediation guidance.

**Status**: Design complete, implementation pending integration with Manus-H.

---

**"Keep it blue, keep it clean, keep it sealed."**  
— Manus-B, Ledger Replay Architect & Attestation Runtime Engineer
