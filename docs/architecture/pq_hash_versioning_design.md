# Post-Quantum Hash Versioning Design

## Executive Summary

This document specifies the architecture for **post-quantum (PQ) migration** in MathLedger through versioned hash domains, dual-commitment block headers, and compatibility layers for historical epoch root verification. The design ensures that MathLedger can transition to quantum-resistant hash functions without breaking historical verification or requiring hard forks.

## Design Principles

1. **Hash Algorithm Versioning**: Hash algorithms are never baked into data structures; they are always identified by version tags
2. **Canonical Encoding Invariance**: Canonical encodings (normalization, serialization) remain untouched across hash migrations
3. **Historical Verifiability**: All historical roots must remain verifiable under their original algorithm identifiers
4. **Dual Commitment**: During migration periods, blocks carry both legacy and PQ commitments
5. **Domain Separation**: Hash domain prefixes are extended to include algorithm version identifiers

## Hash Algorithm Registry

### Algorithm Identifiers

```python
# Hash algorithm version identifiers
HASH_ALG_SHA256 = 0x00      # Current: SHA-256
HASH_ALG_PQ1 = 0x01         # Reserved: First PQ hash (e.g., SHA-3, BLAKE3)
HASH_ALG_PQ2 = 0x02         # Reserved: Second PQ hash (e.g., future NIST standard)
HASH_ALG_PQ3 = 0x03         # Reserved: Third PQ hash
# 0x04-0xFF reserved for future algorithms
```

### Algorithm Metadata

Each algorithm version maintains:

- **Algorithm ID**: Single byte identifier (0x00-0xFF)
- **Algorithm Name**: Human-readable name (e.g., "SHA-256", "SHA3-256")
- **Output Length**: Digest size in bytes (e.g., 32 for SHA-256)
- **Security Level**: Classical/PQ security bits
- **Activation Epoch**: Block number when algorithm becomes valid
- **Deprecation Epoch**: Block number when algorithm is deprecated (optional)
- **Implementation**: Python callable implementing the hash function

## Versioned Domain Separation

### Current Domain Tags (SHA-256 Era)

```
DOMAIN_LEAF = b'\x00'           # Merkle tree leaves
DOMAIN_NODE = b'\x01'           # Merkle tree internal nodes
DOMAIN_STMT = b'\x02'           # Statement content
DOMAIN_BLCK = b'\x03'           # Block headers
DOMAIN_FED = b'\x04'            # Federation namespace
DOMAIN_NODE_ATTEST = b'\x05'    # Node attestation
DOMAIN_DOSSIER = b'\x06'        # Celestial dossier
DOMAIN_ROOT = b'\x07'           # Root hash namespace
```

### Versioned Domain Tags (PQ Era)

Domain tags are extended to include algorithm version:

```
Format: <algorithm_id:1 byte><domain_tag:1 byte>

Examples:
- SHA-256 leaf: b'\x00\x00'
- SHA-256 node: b'\x00\x01'
- PQ1 leaf: b'\x01\x00'
- PQ1 node: b'\x01\x01'
```

This ensures that:
- Different hash algorithms produce different digests for the same input
- Domain separation is preserved within each algorithm
- Historical hashes remain valid under their original algorithm ID

## Dual-Commitment Block Headers

### Legacy Block Header (Pre-PQ)

```python
@dataclass(frozen=True)
class BlockHeader:
    block_number: int
    prev_hash: HexDigest
    merkle_root: HexDigest
    timestamp: float
    version: str = "v1"
```

### PQ-Ready Block Header (Migration Era)

```python
@dataclass(frozen=True)
class BlockHeaderPQ:
    block_number: int
    prev_hash: HexDigest              # Legacy SHA-256 hash
    merkle_root: HexDigest            # Legacy SHA-256 Merkle root
    timestamp: float
    version: str = "v2-pq"
    
    # Dual commitment fields
    hash_algorithm: int = 0x00        # Current algorithm ID
    pq_prev_hash: Optional[HexDigest] = None      # PQ hash of previous block
    pq_merkle_root: Optional[HexDigest] = None    # PQ Merkle root
    pq_algorithm: Optional[int] = None            # PQ algorithm ID
    
    # Composite binding
    dual_commitment: Optional[HexDigest] = None   # SHA256(legacy || pq)
```

### Dual Commitment Construction

During the migration period, blocks maintain both legacy and PQ commitments:

```python
def compute_dual_commitment(
    legacy_hash: str,
    pq_hash: str,
    algorithm_id: int
) -> str:
    """
    Compute dual commitment binding legacy and PQ hashes.
    
    Format: SHA256(algorithm_id || legacy_hash || pq_hash)
    """
    payload = bytes([algorithm_id]) + legacy_hash.encode('ascii') + pq_hash.encode('ascii')
    return hashlib.sha256(payload).hexdigest()
```

## Migration Phases

### Phase 0: Pre-Migration (Current State)
- All hashes use SHA-256
- Domain separation with single-byte tags
- Block headers contain only legacy fields

### Phase 1: Scaffolding (Preparation)
- Add hash algorithm registry
- Implement versioned domain separation
- Add dual-commitment block header structure
- Add PQ placeholder implementations (identity functions)
- **No consensus changes**: New fields are optional and ignored

### Phase 2: Parallel Computation (Testing)
- Compute both legacy and PQ hashes for all operations
- Store both in block headers (dual commitment)
- Verify consistency between implementations
- **Consensus still uses legacy hashes**

### Phase 3: PQ Activation (Transition)
- Consensus switches to PQ hashes at predetermined epoch
- Legacy hashes maintained for backward compatibility
- All nodes must support both algorithms

### Phase 4: Legacy Deprecation (Cleanup)
- Legacy hashes become optional after grace period
- Historical verification still uses original algorithm IDs
- New blocks may drop legacy fields

## Historical Verification Compatibility Layer

### Epoch-Based Algorithm Resolution

```python
@dataclass(frozen=True)
class HashEpoch:
    """Defines which hash algorithm was canonical at a given block range."""
    start_block: int
    end_block: Optional[int]  # None = ongoing
    algorithm_id: int
    algorithm_name: str

# Example epoch registry
HASH_EPOCHS = [
    HashEpoch(0, 999999, HASH_ALG_SHA256, "SHA-256"),
    HashEpoch(1000000, None, HASH_ALG_PQ1, "SHA3-256"),
]
```

### Historical Root Verification

```python
def verify_historical_merkle_root(
    block_number: int,
    leaves: List[str],
    expected_root: str
) -> bool:
    """
    Verify Merkle root using the algorithm that was canonical
    at the given block number.
    """
    epoch = get_epoch_for_block(block_number)
    algorithm = get_hash_algorithm(epoch.algorithm_id)
    
    computed_root = merkle_root_versioned(
        leaves,
        algorithm_id=epoch.algorithm_id,
        hash_fn=algorithm
    )
    
    return computed_root == expected_root
```

### Cross-Epoch Verification

For verifying chains that span multiple hash epochs:

```python
def verify_block_chain(blocks: List[Block]) -> bool:
    """
    Verify a chain of blocks that may span multiple hash epochs.
    """
    for i, block in enumerate(blocks):
        # Determine canonical algorithm for this block
        epoch = get_epoch_for_block(block.header.block_number)
        algorithm = get_hash_algorithm(epoch.algorithm_id)
        
        # Verify Merkle root
        if not verify_merkle_root_with_algorithm(
            block.statements,
            block.header.merkle_root,
            algorithm
        ):
            return False
        
        # Verify chain linkage
        if i > 0:
            prev_block = blocks[i - 1]
            prev_epoch = get_epoch_for_block(prev_block.header.block_number)
            
            # Use the algorithm from the previous block's epoch
            prev_algorithm = get_hash_algorithm(prev_epoch.algorithm_id)
            computed_prev_hash = hash_block_with_algorithm(
                prev_block,
                prev_algorithm
            )
            
            if computed_prev_hash != block.header.prev_hash:
                return False
    
    return True
```

## Implementation Modules

### 1. `basis/crypto/hash_registry.py`
- Hash algorithm registry and metadata
- Algorithm lookup by ID and epoch
- Algorithm activation/deprecation logic

### 2. `basis/crypto/hash_versioned.py`
- Versioned hash functions (wrapping SHA-256, PQ placeholders)
- Versioned domain separation
- Dual-commitment computation

### 3. `basis/ledger/block_pq.py`
- PQ-ready block header structure
- Dual-commitment block sealing
- Migration helpers

### 4. `basis/ledger/verification.py`
- Historical verification using epoch-based algorithm resolution
- Cross-epoch chain verification
- Merkle proof verification with versioned algorithms

### 5. `backend/crypto/pq_placeholders.py`
- Placeholder implementations for PQ hash functions
- Initially return SHA-256 hashes with PQ domain tags
- To be replaced with actual PQ algorithms (SHA-3, BLAKE3, etc.)

## Testing Strategy

### Unit Tests
- Hash algorithm registry operations
- Versioned domain separation
- Dual commitment computation
- Epoch-based algorithm resolution

### Integration Tests
- Block sealing with dual commitments
- Historical verification across epochs
- Chain verification spanning multiple algorithms

### Migration Tests
- Simulate migration from SHA-256 to PQ1
- Verify all historical blocks remain valid
- Verify new blocks use PQ algorithm
- Verify dual-commitment period works correctly

### Regression Tests
- Ensure canonical encodings unchanged
- Ensure existing SHA-256 hashes still verify
- Ensure no breaking changes to existing APIs

## Security Considerations

### Domain Separation
- Versioned domain tags prevent cross-algorithm collisions
- Each algorithm has its own domain namespace
- Prevents second-preimage attacks across algorithms

### Dual Commitment Binding
- Cryptographically binds legacy and PQ hashes
- Prevents selective forgery during migration
- Uses SHA-256 for binding (conservative choice)

### Epoch Boundaries
- Clear activation blocks prevent ambiguity
- Grace periods allow node upgrades
- Historical epochs are immutable

### Algorithm Agility
- Registry allows adding new algorithms without code changes
- Deprecation mechanism for compromised algorithms
- Multiple PQ algorithms for hedging bets

## Migration Timeline (Proposed)

| Phase | Duration | Activities |
|-------|----------|------------|
| Phase 1: Scaffolding | 2-4 weeks | Implement registry, versioned hashing, dual headers |
| Phase 2: Testing | 4-8 weeks | Parallel computation, consistency verification |
| Phase 3: Activation | 1 block | Switch consensus to PQ at predetermined epoch |
| Phase 4: Grace Period | 6-12 months | Maintain dual commitments |
| Phase 5: Deprecation | Ongoing | Legacy hashes become optional |

## Open Questions

1. **PQ Algorithm Selection**: Which specific PQ hash function(s) to use?
   - SHA3-256 (NIST standard, conservative)
   - BLAKE3 (fast, modern, not NIST)
   - Multiple algorithms for redundancy?

2. **Activation Epoch**: How to determine the PQ activation block?
   - Governance vote?
   - Predetermined timeline?
   - Triggered by quantum threat assessment?

3. **Performance Impact**: What is the overhead of dual computation?
   - Benchmark SHA-256 vs SHA-3 vs BLAKE3
   - Optimize hot paths
   - Consider async computation

4. **Backward Compatibility**: How far back should historical verification go?
   - All blocks from genesis?
   - Checkpointed epochs?
   - Pruning strategy?

## References

- [NIST Post-Quantum Cryptography](https://csrc.nist.gov/projects/post-quantum-cryptography)
- [SHA-3 Standard (FIPS 202)](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.202.pdf)
- [BLAKE3 Specification](https://github.com/BLAKE3-team/BLAKE3-specs)
- Bitcoin CVE-2012-2459: Merkle tree second preimage attack
- Ethereum EIP-2929: Gas cost increases for state access opcodes

## Changelog

- **2024-12-06**: Initial design document (Manus-H)
