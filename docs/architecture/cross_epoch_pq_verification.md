# Cross-Epoch Chain Verification with Post-Quantum Integration

**Author**: Manus-B (Ledger Replay Architect & Attestation Runtime Engineer)  
**Phase**: II - Epochization & Governance  
**Date**: 2025-12-06  
**Status**: Design Specification  
**Integration**: Manus-H (PQ Migration Primitives)

---

## Executive Summary

This specification defines how MathLedger's ledger verification system handles **heterogeneous hash chains** during the post-quantum (PQ) cryptographic migration. It ensures that:

1. **Legacy SHA-256 blocks** remain verifiable indefinitely
2. **PQ SHA-3 blocks** integrate seamlessly with legacy chain
3. **Dual-commitment transitions** maintain cryptographic continuity
4. **Cross-epoch verification** works across hash algorithm boundaries
5. **Composite-root invariants** hold during and after migration

---

## Problem Statement

### Current State (Pre-PQ Migration)

All blocks use SHA-256:
```
Block 0 (SHA-256) → Block 1 (SHA-256) → Block 2 (SHA-256) → ...
     ↓                   ↓                   ↓
   H_0 (SHA-256)      H_1 (SHA-256)      H_2 (SHA-256)
```

Replay verification:
```python
recomputed_h_t = SHA256(r_t || u_t)
assert recomputed_h_t == stored_h_t  # Always SHA-256
```

---

### Target State (Post-PQ Migration)

Mixed hash chain:
```
Block 0-999 (SHA-256) → Block 1000-1999 (DUAL) → Block 2000+ (SHA-3)
       ↓                       ↓                        ↓
   H_0 (SHA-256)    H_1000 (SHA-256 + SHA-3)      H_2000 (SHA-3)
```

Replay verification must:
```python
# Detect hash algorithm from metadata
hash_version = block["attestation_metadata"]["hash_version"]

if hash_version == "sha256-v1":
    recomputed_h_t = SHA256(r_t || u_t)
elif hash_version == "sha3-v1":
    recomputed_h_t = SHA3_256(r_t || u_t)
elif hash_version == "dual-v1":
    recomputed_h_t_sha256 = SHA256(r_t || u_t)
    recomputed_h_t_sha3 = SHA3_256(r_t || u_t)
    assert recomputed_h_t_sha256 == stored_h_t_sha256
    assert recomputed_h_t_sha3 == stored_h_t_sha3
else:
    raise UnsupportedHashVersion(hash_version)
```

---

## Design Principles

### Principle 1: Hash Algorithm Versioning

**Every block MUST declare its hash algorithm version.**

**Implementation**:
```python
attestation_metadata = {
    "hash_version": "sha256-v1",  # or "sha3-v1", "dual-v1"
    "hash_algorithm": "SHA-256",  # or "SHA-3-256", "DUAL"
    "domain_separation_version": "v1",
    "ui_leaves": [...],
    "proof_count": 5,
}
```

**Supported Versions**:
- `sha256-v1`: Legacy SHA-256 (pre-PQ migration)
- `dual-v1`: Dual-commitment (SHA-256 + SHA-3) during transition
- `sha3-v1`: Pure SHA-3-256 (post-PQ migration)

---

### Principle 2: Dual-Commitment Transition

**During PQ migration, blocks MUST commit to both SHA-256 and SHA-3 roots.**

**Schema**:
```python
# Dual-commitment block
block = {
    "composite_attestation_root": h_t_sha256,  # Primary (for backward compat)
    "composite_attestation_root_sha3": h_t_sha3,  # Secondary (PQ-safe)
    "attestation_metadata": {
        "hash_version": "dual-v1",
        "hash_algorithm": "DUAL",
        "dual_roots": {
            "sha256": h_t_sha256,
            "sha3": h_t_sha3,
        },
    },
}
```

**Replay Verification**:
```python
if block["attestation_metadata"]["hash_version"] == "dual-v1":
    # Recompute both roots
    r_t_sha256, u_t_sha256, h_t_sha256 = recompute_roots_sha256(block)
    r_t_sha3, u_t_sha3, h_t_sha3 = recompute_roots_sha3(block)
    
    # Verify both match
    assert h_t_sha256 == block["composite_attestation_root"]
    assert h_t_sha3 == block["composite_attestation_root_sha3"]
```

---

### Principle 3: Backward Compatibility

**Legacy blocks (SHA-256) MUST remain verifiable after PQ migration.**

**Implementation**:
```python
def recompute_attestation_roots(block: Dict) -> Tuple[str, str, str]:
    """
    Recompute attestation roots with hash algorithm detection.
    
    Returns:
        (r_t, u_t, h_t) using the block's declared hash algorithm
    """
    hash_version = block["attestation_metadata"].get("hash_version", "sha256-v1")
    
    if hash_version == "sha256-v1":
        return recompute_roots_sha256(block)
    elif hash_version == "sha3-v1":
        return recompute_roots_sha3(block)
    elif hash_version == "dual-v1":
        # For dual-commitment, return primary (SHA-256) for backward compat
        r_t, u_t, h_t = recompute_roots_sha256(block)
        return r_t, u_t, h_t
    else:
        raise UnsupportedHashVersion(f"Unknown hash_version: {hash_version}")
```

---

### Principle 4: Cross-Epoch Prev-Hash Validation

**Prev-hash linkage MUST work across hash algorithm boundaries.**

**Challenge**:
```
Block 999 (SHA-256) → Block 1000 (DUAL)
     ↓                       ↓
 block_identity_999    prev_hash_1000 = ?
```

**Solution**: Prev-hash uses the **primary hash algorithm** of the predecessor block.

```python
def compute_prev_hash(predecessor_block: Dict) -> str:
    """
    Compute prev_hash for next block.
    
    Uses predecessor's primary hash algorithm.
    """
    hash_version = predecessor_block["attestation_metadata"].get("hash_version", "sha256-v1")
    
    if hash_version in ["sha256-v1", "dual-v1"]:
        # Use SHA-256 for prev_hash
        return SHA256(predecessor_block["block_identity"])
    elif hash_version == "sha3-v1":
        # Use SHA-3 for prev_hash
        return SHA3_256(predecessor_block["block_identity"])
    else:
        raise UnsupportedHashVersion(hash_version)
```

**Validation**:
```python
def validate_prev_hash_linkage(block: Dict, predecessor: Dict) -> bool:
    """
    Validate prev_hash linkage across hash algorithm boundaries.
    """
    expected_prev_hash = compute_prev_hash(predecessor)
    return block["prev_hash"] == expected_prev_hash
```

---

### Principle 5: Composite-Root Invariants

**The composite root invariant H_t = Hash(R_t || U_t) MUST hold for all hash algorithms.**

**Invariant**:
```
∀ blocks b:
  b.composite_attestation_root = Hash(b.reasoning_attestation_root || b.ui_attestation_root)
  where Hash ∈ {SHA256, SHA3_256} depending on b.hash_version
```

**Verification**:
```python
def verify_composite_consistency(block: Dict) -> bool:
    """
    Verify composite root consistency across hash algorithms.
    """
    hash_version = block["attestation_metadata"].get("hash_version", "sha256-v1")
    r_t = block["reasoning_attestation_root"]
    u_t = block["ui_attestation_root"]
    h_t = block["composite_attestation_root"]
    
    if hash_version == "sha256-v1":
        expected_h_t = SHA256(r_t + u_t)
    elif hash_version == "sha3-v1":
        expected_h_t = SHA3_256(r_t + u_t)
    elif hash_version == "dual-v1":
        # Verify both SHA-256 and SHA-3 composite roots
        expected_h_t_sha256 = SHA256(r_t + u_t)
        expected_h_t_sha3 = SHA3_256(r_t + u_t)
        h_t_sha3 = block["composite_attestation_root_sha3"]
        return h_t == expected_h_t_sha256 and h_t_sha3 == expected_h_t_sha3
    else:
        raise UnsupportedHashVersion(hash_version)
    
    return h_t == expected_h_t
```

---

## Cross-Epoch Verification Logic

### Epoch Root Computation (Heterogeneous Hash Chains)

**Challenge**: Epoch contains blocks with different hash algorithms.

**Example**:
```
Epoch 10: Blocks 1000-1099
  - Blocks 1000-1049: SHA-256
  - Blocks 1050-1099: DUAL
```

**Solution**: Epoch root uses **SHA-256** until all blocks in epoch are SHA-3.

```python
def compute_epoch_root_heterogeneous(blocks: List[Dict]) -> str:
    """
    Compute epoch root for heterogeneous hash chain.
    
    Algorithm:
    1. Extract composite roots (H_t) from all blocks
    2. Detect hash algorithms used in epoch
    3. If all blocks use SHA-3, use SHA-3 for epoch root
    4. Otherwise, use SHA-256 for backward compatibility
    """
    composite_roots = [b["composite_attestation_root"] for b in blocks]
    hash_versions = [b["attestation_metadata"].get("hash_version", "sha256-v1") for b in blocks]
    
    # Determine epoch hash algorithm
    if all(v == "sha3-v1" for v in hash_versions):
        # Pure SHA-3 epoch
        epoch_hash_algorithm = "sha3-v1"
        epoch_root = merkle_root_sha3(composite_roots)
    else:
        # Mixed or legacy epoch → use SHA-256
        epoch_hash_algorithm = "sha256-v1"
        epoch_root = merkle_root_sha256(composite_roots)
    
    return epoch_root, epoch_hash_algorithm
```

**Epoch Metadata**:
```python
epoch_metadata = {
    "composite_roots": composite_roots,
    "block_ids": block_ids,
    "epoch_size": 100,
    "hash_version": epoch_hash_algorithm,  # "sha256-v1" or "sha3-v1"
    "heterogeneous": len(set(hash_versions)) > 1,  # True if mixed
    "hash_version_distribution": {
        "sha256-v1": 50,
        "dual-v1": 30,
        "sha3-v1": 20,
    },
}
```

---

### Epoch Replay Verification (Cross-Algorithm)

**Verification Algorithm**:
```python
def replay_epoch_heterogeneous(epoch: Dict, blocks: List[Dict]) -> bool:
    """
    Replay epoch verification for heterogeneous hash chain.
    
    Steps:
    1. Recompute composite roots for all blocks (using their declared hash algorithms)
    2. Compute epoch root using same algorithm as original sealing
    3. Verify epoch root matches stored value
    """
    # Recompute composite roots
    recomputed_roots = []
    for block in blocks:
        hash_version = block["attestation_metadata"].get("hash_version", "sha256-v1")
        
        if hash_version == "sha256-v1":
            _, _, h_t = recompute_roots_sha256(block)
        elif hash_version == "sha3-v1":
            _, _, h_t = recompute_roots_sha3(block)
        elif hash_version == "dual-v1":
            _, _, h_t = recompute_roots_sha256(block)  # Use primary
        else:
            raise UnsupportedHashVersion(hash_version)
        
        recomputed_roots.append(h_t)
    
    # Compute epoch root using epoch's hash algorithm
    epoch_hash_version = epoch["epoch_metadata"]["hash_version"]
    
    if epoch_hash_version == "sha256-v1":
        recomputed_epoch_root = merkle_root_sha256(recomputed_roots)
    elif epoch_hash_version == "sha3-v1":
        recomputed_epoch_root = merkle_root_sha3(recomputed_roots)
    else:
        raise UnsupportedHashVersion(epoch_hash_version)
    
    # Verify
    return recomputed_epoch_root == epoch["epoch_root"]
```

---

## PQ Migration Phases

### Phase 1: Pre-Migration (Legacy SHA-256)

**State**: All blocks use SHA-256.

**Block Structure**:
```python
block = {
    "composite_attestation_root": h_t_sha256,
    "attestation_metadata": {
        "hash_version": "sha256-v1",
        "hash_algorithm": "SHA-256",
    },
}
```

**Epoch Structure**:
```python
epoch = {
    "epoch_root": e_t_sha256,
    "epoch_metadata": {
        "hash_version": "sha256-v1",
        "heterogeneous": False,
    },
}
```

---

### Phase 2: Dual-Commitment Transition

**State**: New blocks use dual-commitment (SHA-256 + SHA-3).

**Block Structure**:
```python
block = {
    "composite_attestation_root": h_t_sha256,  # Primary
    "composite_attestation_root_sha3": h_t_sha3,  # Secondary
    "reasoning_attestation_root": r_t_sha256,
    "reasoning_attestation_root_sha3": r_t_sha3,
    "ui_attestation_root": u_t_sha256,
    "ui_attestation_root_sha3": u_t_sha3,
    "attestation_metadata": {
        "hash_version": "dual-v1",
        "hash_algorithm": "DUAL",
        "dual_roots": {
            "sha256": {"r_t": r_t_sha256, "u_t": u_t_sha256, "h_t": h_t_sha256},
            "sha3": {"r_t": r_t_sha3, "u_t": u_t_sha3, "h_t": h_t_sha3},
        },
    },
}
```

**Epoch Structure** (Mixed):
```python
epoch = {
    "epoch_root": e_t_sha256,  # Still SHA-256 for backward compat
    "epoch_metadata": {
        "hash_version": "sha256-v1",
        "heterogeneous": True,
        "hash_version_distribution": {
            "sha256-v1": 50,
            "dual-v1": 50,
        },
    },
}
```

---

### Phase 3: Pure SHA-3 (Post-Migration)

**State**: All new blocks use SHA-3. Legacy blocks remain SHA-256.

**Block Structure**:
```python
block = {
    "composite_attestation_root": h_t_sha3,
    "attestation_metadata": {
        "hash_version": "sha3-v1",
        "hash_algorithm": "SHA-3-256",
    },
}
```

**Epoch Structure** (Pure SHA-3):
```python
epoch = {
    "epoch_root": e_t_sha3,  # SHA-3 epoch root
    "epoch_metadata": {
        "hash_version": "sha3-v1",
        "heterogeneous": False,
    },
}
```

**Epoch Structure** (Mixed, during transition):
```python
epoch = {
    "epoch_root": e_t_sha256,  # Still SHA-256 until all blocks are SHA-3
    "epoch_metadata": {
        "hash_version": "sha256-v1",
        "heterogeneous": True,
        "hash_version_distribution": {
            "sha256-v1": 10,
            "dual-v1": 20,
            "sha3-v1": 70,
        },
    },
}
```

---

## Integration with Manus-H (PQ Migration Primitives)

### Manus-H Responsibilities

1. **Hash Algorithm Abstraction**:
   - Provide `HashAlgorithm` interface
   - Implement `SHA256v1`, `SHA3v1`, `DualHashv1`
   - Version management

2. **Migration Orchestration**:
   - Trigger dual-commitment phase
   - Monitor migration progress
   - Cutover to pure SHA-3

3. **Backward Compatibility**:
   - Ensure legacy blocks remain verifiable
   - Provide hash algorithm detection

### Manus-B Responsibilities

1. **Replay Verification**:
   - Detect hash algorithm from metadata
   - Recompute roots using correct algorithm
   - Verify composite-root invariants

2. **Epoch Verification**:
   - Handle heterogeneous epochs
   - Compute epoch roots with correct algorithm
   - Verify cross-algorithm linkage

3. **Governance Enforcement**:
   - Block sealing with wrong hash version
   - Detect hash algorithm drift
   - Alert on PQ migration violations

### Interface Contract

**Manus-H provides**:
```python
class HashAlgorithm(ABC):
    @abstractmethod
    def hash(self, data: bytes) -> str:
        """Compute hash of data."""
        ...
    
    @abstractmethod
    def merkle_root(self, leaves: List[str]) -> str:
        """Compute Merkle root of leaves."""
        ...
    
    @abstractmethod
    def version(self) -> str:
        """Return hash version string."""
        ...

def get_hash_algorithm(version: str) -> HashAlgorithm:
    """Get hash algorithm by version."""
    if version == "sha256-v1":
        return SHA256v1()
    elif version == "sha3-v1":
        return SHA3v1()
    elif version == "dual-v1":
        return DualHashv1()
    else:
        raise UnsupportedHashVersion(version)
```

**Manus-B uses**:
```python
from manus_h.hash import get_hash_algorithm

def recompute_attestation_roots(block: Dict) -> Tuple[str, str, str]:
    hash_version = block["attestation_metadata"].get("hash_version", "sha256-v1")
    hash_algo = get_hash_algorithm(hash_version)
    
    # Recompute using Manus-H's hash algorithm
    r_t = hash_algo.compute_reasoning_root(block["canonical_proofs"])
    u_t = hash_algo.compute_ui_root(block["attestation_metadata"]["ui_leaves"])
    h_t = hash_algo.hash((r_t + u_t).encode())
    
    return r_t, u_t, h_t
```

---

## Testing Strategy

### Test 1: Legacy Block Replay (SHA-256)

```python
def test_replay_legacy_sha256_block():
    """Test replay of legacy SHA-256 block."""
    block = create_legacy_block_sha256()
    result = replay_block(block)
    assert result.is_valid
    assert result.hash_version == "sha256-v1"
```

### Test 2: Dual-Commitment Block Replay

```python
def test_replay_dual_commitment_block():
    """Test replay of dual-commitment block."""
    block = create_dual_commitment_block()
    result = replay_block(block)
    assert result.is_valid
    assert result.hash_version == "dual-v1"
    assert result.sha256_roots_match
    assert result.sha3_roots_match
```

### Test 3: Pure SHA-3 Block Replay

```python
def test_replay_pure_sha3_block():
    """Test replay of pure SHA-3 block."""
    block = create_pure_sha3_block()
    result = replay_block(block)
    assert result.is_valid
    assert result.hash_version == "sha3-v1"
```

### Test 4: Cross-Algorithm Prev-Hash Validation

```python
def test_cross_algorithm_prev_hash():
    """Test prev_hash linkage across hash algorithm boundary."""
    block_sha256 = create_legacy_block_sha256(block_number=999)
    block_dual = create_dual_commitment_block(block_number=1000, prev_block=block_sha256)
    
    # Validate prev_hash uses SHA-256 (predecessor's algorithm)
    expected_prev_hash = SHA256(block_sha256["block_identity"])
    assert block_dual["prev_hash"] == expected_prev_hash
    
    # Validate chain
    assert validate_prev_hash_linkage(block_dual, block_sha256)
```

### Test 5: Heterogeneous Epoch Replay

```python
def test_heterogeneous_epoch_replay():
    """Test epoch replay with mixed hash algorithms."""
    blocks = [
        *[create_legacy_block_sha256(i) for i in range(50)],
        *[create_dual_commitment_block(i) for i in range(50, 100)],
    ]
    
    epoch = seal_epoch(0, blocks, "system-uuid")
    
    # Epoch root should use SHA-256 (backward compat)
    assert epoch["epoch_metadata"]["hash_version"] == "sha256-v1"
    assert epoch["epoch_metadata"]["heterogeneous"] is True
    
    # Replay should succeed
    assert replay_epoch(epoch, blocks)
```

---

## Migration Checklist

### Pre-Migration

- [ ] Implement hash algorithm abstraction (Manus-H)
- [ ] Add `hash_version` field to `attestation_metadata`
- [ ] Backfill `hash_version` for existing blocks (`"sha256-v1"`)
- [ ] Update replay verification to detect hash algorithm
- [ ] Test replay of legacy blocks with new code

### Dual-Commitment Phase

- [ ] Implement dual-hash block sealing
- [ ] Add `composite_attestation_root_sha3` column to blocks table
- [ ] Update epoch sealing to handle heterogeneous epochs
- [ ] Test cross-algorithm prev_hash validation
- [ ] Monitor dual-commitment block sealing

### Post-Migration

- [ ] Cutover to pure SHA-3 block sealing
- [ ] Verify all new blocks use `"sha3-v1"`
- [ ] Test pure SHA-3 epoch sealing
- [ ] Archive dual-commitment code (keep for replay)
- [ ] Document migration in ledger history

---

## Status

**Design**: ✅ Complete  
**Manus-H Integration**: ⏳ Pending (requires Manus-H hash abstraction)  
**Implementation**: ⏳ Pending  
**Testing**: ⏳ Pending

---

**Next**: Deliver Comprehensive Technical Blueprint
