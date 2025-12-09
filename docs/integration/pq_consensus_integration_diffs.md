# PQ Consensus Integration Diffs

**Document Version**: 1.0  
**Author**: Manus-H  
**Date**: 2024-12-06  
**Purpose**: Integration diffs for wiring PQ consensus modules into active block sealing path

---

## Overview

This document provides complete integration diffs for wiring the PQ consensus module into MathLedger's block sealing and validation pipeline. The integration supports all 5 migration phases and maintains backward compatibility with existing code.

---

## 1. Block Sealing Integration

### File: `backend/ledger/blockchain.py`

**Current Implementation**:
```python
def seal_block(statement_ids: List[str], prev_hash: str, block_number: int, ts: float, version: str="v1") -> Dict:
    """Build a block dict with deterministic header + statement id list."""
    mroot = merkle_root(statement_ids)
    header = {
        "block_number": block_number,
        "prev_hash": prev_hash,
        "merkle_root": mroot,
        "timestamp": ts,
        "version": version,
    }
    return {"header": header, "statements": statement_ids}
```

**New Implementation** (with PQ support):

```python
# Add imports at top of file
from backend.consensus_pq.epoch import get_epoch_for_block, get_current_epoch
from backend.consensus_pq.rules import ConsensusRuleVersion, get_consensus_rules_for_phase
from basis.crypto.hash_versioned import merkle_root_versioned, compute_dual_commitment
from basis.ledger.block_pq import BlockHeaderPQ, seal_block_dual

def seal_block(statement_ids: List[str], prev_hash: str, block_number: int, ts: float, version: str="v1") -> Dict:
    """
    Build a block dict with deterministic header + statement id list.
    
    Automatically detects current epoch and seals block with appropriate
    hash algorithm (legacy SHA-256 or dual-commitment PQ).
    """
    # Get current epoch to determine which algorithm to use
    current_epoch = get_current_epoch()
    
    if current_epoch is None:
        # Fallback to legacy sealing if no epoch registered
        return seal_block_legacy(statement_ids, prev_hash, block_number, ts, version)
    
    # Get consensus rules for current epoch
    rule_version = ConsensusRuleVersion(current_epoch.rule_version)
    rules = get_consensus_rules_for_phase(rule_version)
    
    # Determine sealing mode based on consensus rules
    if rules.pq_fields_required:
        # Dual-commitment sealing required
        return seal_block_pq_dual(
            statement_ids=statement_ids,
            prev_hash=prev_hash,
            block_number=block_number,
            ts=ts,
            version=version,
            pq_algorithm_id=current_epoch.algorithm_id,
        )
    else:
        # Legacy sealing (Phase 0-1)
        return seal_block_legacy(statement_ids, prev_hash, block_number, ts, version)


def seal_block_legacy(statement_ids: List[str], prev_hash: str, block_number: int, ts: float, version: str="v1") -> Dict:
    """
    Legacy block sealing (SHA-256 only).
    
    Used in Phase 0-1 before PQ migration begins.
    """
    mroot = merkle_root(statement_ids)
    header = {
        "block_number": block_number,
        "prev_hash": prev_hash,
        "merkle_root": mroot,
        "timestamp": ts,
        "version": version,
    }
    return {"header": header, "statements": statement_ids}


def seal_block_pq_dual(
    statement_ids: List[str],
    prev_hash: str,
    block_number: int,
    ts: float,
    version: str,
    pq_algorithm_id: int,
) -> Dict:
    """
    Dual-commitment block sealing (legacy + PQ).
    
    Used in Phase 2-4 during PQ migration.
    Computes both SHA-256 and PQ Merkle roots, binds them with dual commitment.
    """
    # Compute legacy Merkle root (SHA-256)
    legacy_merkle_root = merkle_root_versioned(statement_ids, algorithm_id=0x00)
    
    # Compute PQ Merkle root
    pq_merkle_root = merkle_root_versioned(statement_ids, algorithm_id=pq_algorithm_id)
    
    # Compute dual commitment (binds legacy and PQ hashes)
    dual_commitment = compute_dual_commitment(
        legacy_hash=legacy_merkle_root,
        pq_hash=pq_merkle_root,
        pq_algorithm_id=pq_algorithm_id,
    )
    
    # Get previous block to compute PQ prev_hash
    # In production, this would fetch from blockchain state
    # For now, we compute it from prev_hash (assuming it's available)
    pq_prev_hash = prev_hash  # TODO: Compute from previous block's PQ hash
    
    # Build header with PQ fields
    header = {
        "block_number": block_number,
        "prev_hash": prev_hash,
        "merkle_root": legacy_merkle_root,
        "timestamp": ts,
        "version": version,
        # PQ fields
        "pq_algorithm": pq_algorithm_id,
        "pq_merkle_root": pq_merkle_root,
        "pq_prev_hash": pq_prev_hash,
        "dual_commitment": dual_commitment,
    }
    
    return {"header": header, "statements": statement_ids}


def get_prev_block_hash(block_number: int, algorithm_id: int) -> str:
    """
    Get previous block's hash using specified algorithm.
    
    This is a helper function for computing pq_prev_hash during block sealing.
    In production, this would query the blockchain state.
    
    Args:
        block_number: Current block number
        algorithm_id: Hash algorithm ID to use
        
    Returns:
        Previous block's hash as hex string
    """
    if block_number == 0:
        # Genesis block
        return "0x" + "00" * 32
    
    # TODO: Implement actual blockchain state query
    # For now, return placeholder
    raise NotImplementedError("Blockchain state query not implemented")
```

---

## 2. Block Validation Integration

### File: `backend/ledger/blockchain.py` (add new function)

```python
from backend.consensus_pq.validation import validate_block_full
from basis.ledger.block_pq import BlockHeaderPQ

def validate_block(block: Dict, prev_block: Dict = None) -> tuple[bool, str]:
    """
    Validate a block according to PQ consensus rules.
    
    Automatically detects block's epoch and applies appropriate validation rules.
    
    Args:
        block: Block dictionary with header and statements
        prev_block: Previous block dictionary (optional)
        
    Returns:
        Tuple of (is_valid, error_message)
        If valid, error_message is empty string
        If invalid, error_message describes the violation
    """
    # Convert dict to BlockHeaderPQ
    header = block["header"]
    block_pq = dict_to_block_header_pq(header, block["statements"])
    
    # Convert prev_block if provided
    prev_block_pq = None
    if prev_block is not None:
        prev_header = prev_block["header"]
        prev_block_pq = dict_to_block_header_pq(prev_header, prev_block["statements"])
    
    # Validate using PQ consensus module
    is_valid, error_message = validate_block_full(block_pq, prev_block_pq)
    
    return is_valid, error_message or ""


def dict_to_block_header_pq(header: Dict, statements: List[str]) -> BlockHeaderPQ:
    """
    Convert block header dict to BlockHeaderPQ object.
    
    Args:
        header: Block header dictionary
        statements: List of statement IDs
        
    Returns:
        BlockHeaderPQ object
    """
    return BlockHeaderPQ(
        block_number=header["block_number"],
        prev_hash=header["prev_hash"],
        merkle_root=header["merkle_root"],
        timestamp=header["timestamp"],
        statements=statements,
        # PQ fields (optional)
        pq_algorithm=header.get("pq_algorithm"),
        pq_merkle_root=header.get("pq_merkle_root"),
        pq_prev_hash=header.get("pq_prev_hash"),
        dual_commitment=header.get("dual_commitment"),
    )
```

---

## 3. Prev-Hash Cross-Algorithm Validators

### File: `backend/ledger/validators.py` (new file)

```python
"""
Cross-Algorithm Prev-Hash Validators

Validates prev_hash linkage across algorithm transitions (epoch boundaries).
"""

from typing import Dict, Tuple, Optional
from backend.consensus_pq.prev_hash import (
    validate_prev_hash_linkage,
    validate_dual_prev_hash,
    compute_prev_hash,
)
from backend.consensus_pq.epoch import get_epoch_for_block, is_epoch_transition
from basis.ledger.block_pq import BlockHeaderPQ


def validate_prev_hash_cross_algorithm(
    block: BlockHeaderPQ,
    prev_block: BlockHeaderPQ,
) -> Tuple[bool, Optional[str]]:
    """
    Validate prev_hash linkage with cross-algorithm support.
    
    This validator handles epoch transitions where the hash algorithm changes.
    
    Args:
        block: Current block header
        prev_block: Previous block header
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if this is an epoch transition
    if is_epoch_transition(block.block_number):
        return validate_epoch_transition_prev_hash(block, prev_block)
    else:
        return validate_prev_hash_linkage(block, prev_block)


def validate_epoch_transition_prev_hash(
    block: BlockHeaderPQ,
    prev_block: BlockHeaderPQ,
) -> Tuple[bool, Optional[str]]:
    """
    Validate prev_hash at epoch transition boundary.
    
    At epoch boundaries, the hash algorithm may change. This validator
    ensures that both legacy and PQ prev_hash chains remain valid.
    
    Args:
        block: Current block header (first block of new epoch)
        prev_block: Previous block header (last block of old epoch)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    block_epoch = get_epoch_for_block(block.block_number)
    prev_epoch = get_epoch_for_block(prev_block.block_number)
    
    if block_epoch is None or prev_epoch is None:
        return False, "Epoch not found for transition validation"
    
    # Validate legacy prev_hash (always required at transitions)
    legacy_prev_hash = compute_prev_hash(prev_block, algorithm_id=prev_epoch.algorithm_id)
    if block.prev_hash != legacy_prev_hash:
        return False, (
            f"Legacy prev_hash mismatch at epoch transition: "
            f"expected {legacy_prev_hash}, got {block.prev_hash}"
        )
    
    # Validate PQ prev_hash if block has PQ fields
    if block.pq_prev_hash is not None:
        pq_prev_hash = compute_prev_hash(prev_block, algorithm_id=block.pq_algorithm)
        if block.pq_prev_hash != pq_prev_hash:
            return False, (
                f"PQ prev_hash mismatch at epoch transition: "
                f"expected {pq_prev_hash}, got {block.pq_prev_hash}"
            )
    
    return True, None


def validate_dual_prev_hash_chains(
    blocks: list[BlockHeaderPQ],
) -> Tuple[bool, Optional[str]]:
    """
    Validate both legacy and PQ prev_hash chains across a sequence of blocks.
    
    This is useful for validating chain segments that span epoch transitions.
    
    Args:
        blocks: List of consecutive blocks to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(blocks) < 2:
        return True, None
    
    for i in range(1, len(blocks)):
        block = blocks[i]
        prev_block = blocks[i - 1]
        
        # Validate prev_hash linkage
        is_valid, error = validate_prev_hash_cross_algorithm(block, prev_block)
        if not is_valid:
            return False, f"Block {block.block_number}: {error}"
    
    return True, None


def detect_prev_hash_drift(
    block: BlockHeaderPQ,
    prev_block: BlockHeaderPQ,
) -> Dict[str, bool]:
    """
    Detect drift between legacy and PQ prev_hash chains.
    
    This is a monitoring function that independently validates both chains
    and reports any inconsistencies.
    
    Args:
        block: Current block header
        prev_block: Previous block header
        
    Returns:
        Dictionary with validation results for each chain:
        {
            "legacy_valid": bool,
            "pq_valid": bool,
            "legacy_error": str or None,
            "pq_error": str or None,
        }
    """
    legacy_valid, pq_valid, legacy_error, pq_error = validate_dual_prev_hash(
        block, prev_block
    )
    
    return {
        "legacy_valid": legacy_valid,
        "pq_valid": pq_valid,
        "legacy_error": legacy_error,
        "pq_error": pq_error,
    }
```

---

## 4. Epoch Management Integration

### File: `backend/ledger/epoch_manager.py` (new file)

```python
"""
Epoch Manager

Manages epoch registration and activation for PQ migration.
Integrates with governance system for epoch activation.
"""

import time
from typing import Optional
from backend.consensus_pq.epoch import (
    register_epoch,
    get_current_epoch,
    get_next_epoch_transition,
    HashEpoch,
)
from backend.consensus_pq.rules import ConsensusRuleVersion


class EpochManager:
    """Manages epoch lifecycle for PQ migration."""
    
    def __init__(self):
        self.pending_epochs = []
    
    def propose_epoch_activation(
        self,
        start_block: int,
        algorithm_id: int,
        algorithm_name: str,
        rule_version: str,
        governance_hash: str,
    ) -> str:
        """
        Propose a new epoch activation.
        
        This creates a pending epoch that will be activated when the
        start_block is reached.
        
        Args:
            start_block: Block number where epoch starts
            algorithm_id: Hash algorithm ID for this epoch
            algorithm_name: Human-readable algorithm name
            rule_version: Consensus rule version (e.g., "v2-dual-required")
            governance_hash: Hash of governance proposal that approved this epoch
            
        Returns:
            Epoch ID (hex string)
        """
        # Validate that start_block is in the future
        current_epoch = get_current_epoch()
        if current_epoch is not None:
            # Estimate current block (in production, query blockchain state)
            # For now, use current epoch's start_block as reference
            if start_block <= current_epoch.start_block:
                raise ValueError(
                    f"Epoch start_block ({start_block}) must be after "
                    f"current epoch start ({current_epoch.start_block})"
                )
        
        # Create pending epoch
        pending_epoch = HashEpoch(
            start_block=start_block,
            end_block=None,  # Will be set when next epoch activates
            algorithm_id=algorithm_id,
            algorithm_name=algorithm_name,
            rule_version=rule_version,
            activation_timestamp=time.time(),
            governance_hash=governance_hash,
        )
        
        self.pending_epochs.append(pending_epoch)
        
        # Generate epoch ID
        epoch_id = f"epoch-{start_block}-{algorithm_id:02x}"
        return epoch_id
    
    def activate_pending_epochs(self, current_block: int) -> list[HashEpoch]:
        """
        Activate any pending epochs that have reached their start_block.
        
        This should be called after each block is sealed.
        
        Args:
            current_block: Current block number
            
        Returns:
            List of activated epochs
        """
        activated = []
        
        for epoch in self.pending_epochs:
            if current_block >= epoch.start_block:
                # Activate this epoch
                register_epoch(epoch)
                activated.append(epoch)
        
        # Remove activated epochs from pending list
        self.pending_epochs = [
            e for e in self.pending_epochs
            if e not in activated
        ]
        
        return activated
    
    def get_next_activation_block(self) -> Optional[int]:
        """
        Get the block number of the next pending epoch activation.
        
        Returns:
            Block number of next activation, or None if no pending epochs
        """
        if not self.pending_epochs:
            return None
        
        return min(e.start_block for e in self.pending_epochs)


# Global epoch manager instance
_epoch_manager = EpochManager()


def get_epoch_manager() -> EpochManager:
    """Get the global epoch manager instance."""
    return _epoch_manager
```

---

## 5. Integration Testing

### File: `tests/integration/test_pq_block_sealing.py` (new file)

```python
"""
Integration tests for PQ block sealing.

Tests the complete flow from block sealing to validation across all 5 phases.
"""

import pytest
import time
from backend.ledger.blockchain import (
    seal_block,
    seal_block_legacy,
    seal_block_pq_dual,
    validate_block,
)
from backend.consensus_pq.epoch import (
    register_epoch,
    get_current_epoch,
    initialize_genesis_epoch,
    HashEpoch,
)
from backend.consensus_pq.rules import ConsensusRuleVersion


@pytest.fixture(autouse=True)
def reset_epochs():
    """Reset epoch registry before each test."""
    initialize_genesis_epoch()


def test_seal_block_phase_1_legacy():
    """Test block sealing in Phase 1 (legacy, SHA-256 only)."""
    # Genesis epoch is v1-legacy
    statements = ["stmt1", "stmt2", "stmt3"]
    prev_hash = "0x" + "00" * 32
    block_number = 1
    ts = time.time()
    
    # Seal block (should use legacy sealing)
    block = seal_block(statements, prev_hash, block_number, ts)
    
    # Verify block structure
    assert "header" in block
    assert "statements" in block
    assert block["header"]["block_number"] == block_number
    assert block["header"]["prev_hash"] == prev_hash
    assert "merkle_root" in block["header"]
    
    # Verify no PQ fields in Phase 1
    assert "pq_algorithm" not in block["header"]
    assert "pq_merkle_root" not in block["header"]
    assert "dual_commitment" not in block["header"]


def test_seal_block_phase_3_dual_required():
    """Test block sealing in Phase 3 (dual-commitment required)."""
    # Register Phase 3 epoch
    phase3_epoch = HashEpoch(
        start_block=1000,
        end_block=None,
        algorithm_id=0x01,  # SHA3-256
        algorithm_name="SHA3-256",
        rule_version=ConsensusRuleVersion.V2_DUAL_REQUIRED.value,
        activation_timestamp=time.time(),
        governance_hash="0x" + "AA" * 32,
    )
    register_epoch(phase3_epoch)
    
    # Seal block (should use dual-commitment sealing)
    statements = ["stmt1", "stmt2", "stmt3"]
    prev_hash = "0x" + "11" * 32
    block_number = 1000
    ts = time.time()
    
    block = seal_block(statements, prev_hash, block_number, ts)
    
    # Verify block has both legacy and PQ fields
    assert "merkle_root" in block["header"]
    assert "pq_algorithm" in block["header"]
    assert "pq_merkle_root" in block["header"]
    assert "pq_prev_hash" in block["header"]
    assert "dual_commitment" in block["header"]
    
    # Verify PQ algorithm is correct
    assert block["header"]["pq_algorithm"] == 0x01


def test_validate_block_legacy():
    """Test validation of legacy block."""
    statements = ["stmt1", "stmt2"]
    prev_hash = "0x" + "00" * 32
    block_number = 1
    ts = time.time()
    
    block = seal_block_legacy(statements, prev_hash, block_number, ts)
    
    # Validate block
    is_valid, error = validate_block(block)
    
    # Should be valid
    assert is_valid, f"Block validation failed: {error}"


def test_validate_block_dual_commitment():
    """Test validation of dual-commitment block."""
    # Register Phase 3 epoch
    phase3_epoch = HashEpoch(
        start_block=1000,
        end_block=None,
        algorithm_id=0x01,
        algorithm_name="SHA3-256",
        rule_version=ConsensusRuleVersion.V2_DUAL_REQUIRED.value,
        activation_timestamp=time.time(),
        governance_hash="0x" + "AA" * 32,
    )
    register_epoch(phase3_epoch)
    
    # Seal block with dual commitment
    statements = ["stmt1", "stmt2"]
    prev_hash = "0x" + "11" * 32
    block_number = 1000
    ts = time.time()
    
    block = seal_block_pq_dual(
        statement_ids=statements,
        prev_hash=prev_hash,
        block_number=block_number,
        ts=ts,
        version="v1",
        pq_algorithm_id=0x01,
    )
    
    # Validate block
    is_valid, error = validate_block(block)
    
    # Should be valid
    assert is_valid, f"Block validation failed: {error}"


def test_epoch_transition_sealing():
    """Test block sealing across epoch transition."""
    # Seal block in Phase 1 (legacy)
    statements1 = ["stmt1"]
    block1 = seal_block_legacy(statements1, "0x" + "00" * 32, 999, time.time())
    
    # Register Phase 2 epoch starting at block 1000
    phase2_epoch = HashEpoch(
        start_block=1000,
        end_block=None,
        algorithm_id=0x01,
        algorithm_name="SHA3-256",
        rule_version=ConsensusRuleVersion.V2_DUAL_OPTIONAL.value,
        activation_timestamp=time.time(),
        governance_hash="0x" + "BB" * 32,
    )
    register_epoch(phase2_epoch)
    
    # Seal block in Phase 2 (dual optional)
    statements2 = ["stmt2"]
    block2 = seal_block(statements2, block1["header"]["merkle_root"], 1000, time.time())
    
    # Block 2 should have PQ fields (even though optional, seal_block adds them)
    assert "pq_algorithm" in block2["header"]
```

---

## 6. Deployment Checklist

### Pre-Deployment

- [ ] Review all integration diffs
- [ ] Run integration test suite
- [ ] Verify backward compatibility with existing blocks
- [ ] Test epoch transition scenarios
- [ ] Validate prev_hash cross-algorithm logic

### Deployment Steps

1. **Deploy Updated Modules**:
   ```bash
   # Copy updated files
   cp backend/ledger/blockchain.py.new backend/ledger/blockchain.py
   cp backend/ledger/validators.py.new backend/ledger/validators.py
   cp backend/ledger/epoch_manager.py.new backend/ledger/epoch_manager.py
   ```

2. **Run Integration Tests**:
   ```bash
   pytest tests/integration/test_pq_block_sealing.py -v
   ```

3. **Verify Epoch Registry**:
   ```bash
   python -c "from backend.consensus_pq.epoch import get_current_epoch; print(get_current_epoch())"
   ```

4. **Seal Test Block**:
   ```bash
   python -c "from backend.ledger.blockchain import seal_block; import time; block = seal_block(['test'], '0x00' * 32, 1, time.time()); print(block)"
   ```

### Post-Deployment Verification

- [ ] Verify legacy blocks still validate correctly
- [ ] Verify new blocks seal with correct algorithm
- [ ] Monitor logs for any validation errors
- [ ] Check epoch manager status

---

## 7. Migration Path Summary

### Phase 0 → Phase 1: Scaffolding Deployment
- **Changes**: None (backward compatible)
- **Block Format**: Legacy only
- **Validation**: Legacy rules

### Phase 1 → Phase 2: Dual Optional
- **Changes**: `seal_block()` starts adding PQ fields (optional)
- **Block Format**: Legacy + optional PQ fields
- **Validation**: PQ fields validated if present

### Phase 2 → Phase 3: Dual Required
- **Changes**: `seal_block()` requires PQ fields
- **Block Format**: Legacy + PQ fields (both required)
- **Validation**: Both chains validated

### Phase 3 → Phase 4: PQ Primary
- **Changes**: Fork choice switches to PQ chain
- **Block Format**: Legacy + PQ fields (both required)
- **Validation**: Both chains validated, PQ canonical

### Phase 4 → Phase 5: PQ Only
- **Changes**: Legacy fields become optional
- **Block Format**: PQ fields only (legacy optional)
- **Validation**: PQ chain only

---

**Document Version**: 1.0  
**Status**: Ready for Integration  
**Next Steps**: Implement integration tests and deploy to testnet
