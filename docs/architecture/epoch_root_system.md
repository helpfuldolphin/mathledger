# Epoch Root System Design

**Author**: Manus-B (Ledger & Attestation Runtime Engineer)  
**Date**: 2025-12-06  
**Status**: Design Proposal

---

## Overview

The **Epoch Root System** provides periodic checkpointing of the ledger by aggregating block attestation roots into epoch-level commitments. This enables efficient verification of large ledger segments and provides natural boundaries for archival and pruning.

---

## Motivation

### Current State

- Individual blocks have dual attestation roots (R_t, U_t, H_t)
- No aggregation mechanism for multiple blocks
- Difficult to verify large ledger segments efficiently
- No natural boundaries for archival or pruning

### Desired State

- Epoch-level aggregation of block roots
- Efficient verification of epoch integrity
- Natural boundaries for ledger management
- Hierarchical Merkle structure: blocks → epochs → super-epochs

---

## Design

### Epoch Definition

An **epoch** is a fixed-size sequence of blocks (e.g., 100 blocks) with a deterministic boundary.

```
Epoch N: Blocks [N*100, (N+1)*100)
Epoch 0: Blocks [0, 100)
Epoch 1: Blocks [100, 200)
Epoch 2: Blocks [200, 300)
```

### Epoch Root Computation

The **epoch root** (E_t) is the Merkle root of all composite attestation roots (H_t) in the epoch:

```
E_t = MerkleRoot([H_0, H_1, H_2, ..., H_99])
```

Where each H_i is the composite attestation root of block i in the epoch.

### Hierarchical Structure

```
┌─────────────────────────────────────────────────────────┐
│                     Super-Epoch Root                     │
│                    SE_t = MerkleRoot(E_0, E_1, ...)     │
└─────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Epoch 0     │    │  Epoch 1     │    │  Epoch 2     │
│  E_0         │    │  E_1         │    │  E_2         │
└──────────────┘    └──────────────┘    └──────────────┘
        │                   │                   │
   ┌────┼────┐         ┌────┼────┐         ┌────┼────┐
   ▼    ▼    ▼         ▼    ▼    ▼         ▼    ▼    ▼
  H_0  H_1  H_2      H_100 H_101 H_102   H_200 H_201 H_202
   │    │    │         │    │    │         │    │    │
   ▼    ▼    ▼         ▼    ▼    ▼         ▼    ▼    ▼
 Block Block Block   Block Block Block   Block Block Block
   0    1    2       100  101  102       200  201  202
```

---

## Schema Design

### Epochs Table

```sql
CREATE TABLE epochs (
    id BIGSERIAL PRIMARY KEY,
    epoch_number BIGINT NOT NULL UNIQUE,
    start_block_number BIGINT NOT NULL,
    end_block_number BIGINT NOT NULL,
    block_count INT NOT NULL,
    
    -- Epoch root: Merkle root of all H_t in epoch
    epoch_root TEXT NOT NULL,
    
    -- Aggregate statistics
    total_proofs INT NOT NULL DEFAULT 0,
    total_ui_events INT NOT NULL DEFAULT 0,
    
    -- Metadata
    sealed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    sealed_by TEXT DEFAULT 'epoch_sealer',
    
    -- Epoch attestation metadata
    epoch_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    
    -- System reference
    system_id UUID REFERENCES theories(id) ON DELETE CASCADE,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX epochs_epoch_number_idx ON epochs(epoch_number DESC);
CREATE INDEX epochs_system_id_idx ON epochs(system_id);
CREATE INDEX epochs_epoch_root_idx ON epochs(epoch_root);
CREATE INDEX epochs_block_range_idx ON epochs(start_block_number, end_block_number);

-- Constraints
ALTER TABLE epochs ADD CONSTRAINT epochs_block_range_valid
CHECK (end_block_number > start_block_number);

ALTER TABLE epochs ADD CONSTRAINT epochs_block_count_valid
CHECK (block_count = end_block_number - start_block_number);

-- Comments
COMMENT ON TABLE epochs IS 
'Epoch-level aggregation of block attestation roots for efficient verification';

COMMENT ON COLUMN epochs.epoch_root IS 
'E_t: Merkle root of all composite attestation roots (H_t) in epoch';
```

### Block-Epoch Linkage

```sql
-- Add epoch reference to blocks table
ALTER TABLE blocks ADD COLUMN IF NOT EXISTS epoch_id BIGINT REFERENCES epochs(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS blocks_epoch_id_idx ON blocks(epoch_id);
```

---

## Implementation

### Core Module: `backend/ledger/epoch/`

#### File: `backend/ledger/epoch/__init__.py`

```python
"""
Epoch root system for ledger checkpointing.
"""

from .sealer import seal_epoch, compute_epoch_root
from .verifier import verify_epoch_integrity, replay_epoch

__all__ = [
    "seal_epoch",
    "compute_epoch_root",
    "verify_epoch_integrity",
    "replay_epoch",
]
```

#### File: `backend/ledger/epoch/sealer.py`

```python
"""
Epoch sealing logic.
"""

from typing import List, Dict, Any
from substrate.crypto.hashing import merkle_root


def compute_epoch_root(composite_roots: List[str]) -> str:
    """
    Compute epoch root from composite attestation roots.
    
    Args:
        composite_roots: List of H_t values from blocks in epoch
        
    Returns:
        E_t: Epoch root (Merkle root of H_t values)
    """
    if not composite_roots:
        raise ValueError("Cannot compute epoch root from empty list")
    
    # Use same Merkle tree construction as block sealing
    return merkle_root(composite_roots)


def seal_epoch(
    epoch_number: int,
    blocks: List[Dict[str, Any]],
    system_id: str,
) -> Dict[str, Any]:
    """
    Seal an epoch with aggregated block roots.
    
    Args:
        epoch_number: Epoch number
        blocks: List of block data in epoch
        system_id: System identifier
        
    Returns:
        Epoch metadata dictionary
    """
    if not blocks:
        raise ValueError(f"Cannot seal empty epoch {epoch_number}")
    
    # Extract composite roots
    composite_roots = [
        block["composite_attestation_root"]
        for block in blocks
    ]
    
    # Compute epoch root
    epoch_root = compute_epoch_root(composite_roots)
    
    # Aggregate statistics
    total_proofs = sum(block.get("proof_count", 0) for block in blocks)
    total_ui_events = sum(block.get("ui_event_count", 0) for block in blocks)
    
    # Extract block range
    block_numbers = [block["block_number"] for block in blocks]
    start_block = min(block_numbers)
    end_block = max(block_numbers) + 1  # Exclusive end
    
    return {
        "epoch_number": epoch_number,
        "epoch_root": epoch_root,
        "start_block_number": start_block,
        "end_block_number": end_block,
        "block_count": len(blocks),
        "total_proofs": total_proofs,
        "total_ui_events": total_ui_events,
        "system_id": system_id,
        "epoch_metadata": {
            "composite_roots": composite_roots,
            "block_ids": [block["id"] for block in blocks],
        },
    }
```

#### File: `backend/ledger/epoch/verifier.py`

```python
"""
Epoch verification logic.
"""

from typing import Dict, Any, List
from .sealer import compute_epoch_root


def verify_epoch_integrity(epoch_data: Dict[str, Any], blocks: List[Dict[str, Any]]) -> bool:
    """
    Verify epoch root matches recomputed root from blocks.
    
    Args:
        epoch_data: Epoch metadata from database
        blocks: List of blocks in epoch
        
    Returns:
        True if epoch root is valid
    """
    stored_epoch_root = epoch_data["epoch_root"]
    
    # Recompute epoch root from blocks
    composite_roots = [
        block["composite_attestation_root"]
        for block in blocks
    ]
    recomputed_epoch_root = compute_epoch_root(composite_roots)
    
    return stored_epoch_root == recomputed_epoch_root


def replay_epoch(epoch_data: Dict[str, Any], blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Replay epoch verification with detailed results.
    
    Args:
        epoch_data: Epoch metadata
        blocks: List of blocks in epoch
        
    Returns:
        Verification result dictionary
    """
    is_valid = verify_epoch_integrity(epoch_data, blocks)
    
    return {
        "epoch_number": epoch_data["epoch_number"],
        "is_valid": is_valid,
        "block_count": len(blocks),
        "stored_epoch_root": epoch_data["epoch_root"],
    }
```

---

## Epoch Sealing Strategy

### Automatic Sealing

Seal epochs automatically when block count reaches threshold:

```python
def maybe_seal_epoch(system_id: str, current_block_number: int):
    """
    Check if epoch should be sealed and seal if necessary.
    """
    EPOCH_SIZE = 100
    
    if current_block_number % EPOCH_SIZE == 0:
        epoch_number = current_block_number // EPOCH_SIZE - 1
        start_block = epoch_number * EPOCH_SIZE
        end_block = (epoch_number + 1) * EPOCH_SIZE
        
        # Fetch blocks in epoch
        blocks = fetch_blocks_in_range(system_id, start_block, end_block)
        
        # Seal epoch
        epoch_data = seal_epoch(epoch_number, blocks, system_id)
        
        # Store in database
        store_epoch(epoch_data)
```

### Manual Sealing

Provide CLI tool for manual epoch sealing:

```bash
python scripts/seal_epoch.py --epoch-number 5 --system-id <uuid>
```

---

## Benefits

1. **Efficient Verification**: Verify large ledger segments by checking epoch roots
2. **Natural Boundaries**: Clear boundaries for archival, pruning, backup
3. **Hierarchical Proofs**: Enable Merkle proofs at multiple levels
4. **Scalability**: Reduce verification cost from O(n) to O(log n) for n blocks
5. **Auditability**: Epoch roots provide natural audit checkpoints

---

## Future Extensions

### Super-Epochs

Aggregate epochs into super-epochs (e.g., 100 epochs = 10,000 blocks):

```
SE_t = MerkleRoot([E_0, E_1, ..., E_99])
```

### Cross-Epoch Proofs

Enable Merkle proofs spanning multiple epochs:

```
Proof: Block → Epoch → Super-Epoch → Root
```

### Epoch Signatures

Add cryptographic signatures to epoch roots for external verification:

```
Signature = Sign(E_t, private_key)
```

---

## Testing Strategy

1. **Unit Tests**: Test epoch root computation determinism
2. **Integration Tests**: Test epoch sealing with real blocks
3. **Replay Tests**: Verify epoch integrity after sealing
4. **Performance Tests**: Measure epoch sealing performance

---

**Status**: Ready for implementation
