# Replay Verification System Design

**Author**: Manus-B (Ledger & Attestation Runtime Engineer)  
**Date**: 2025-12-06  
**Status**: Design Proposal

---

## Overview

The **Replay Verification System** ensures that historical blocks can be deterministically reconstructed, producing identical attestation roots. This is a critical invariant for ledger integrity and auditability.

---

## Requirements

### Functional Requirements

1. **Deterministic Reconstruction**: Given block inputs (proofs, UI events), recompute attestation roots
2. **Historical Replay**: Replay any historical block and verify H_t matches stored value
3. **Chain Replay**: Replay entire chain from genesis to current height
4. **Failure Detection**: Detect and report any replay mismatches

### Non-Functional Requirements

1. **Performance**: Replay should complete in O(n) time per block
2. **Isolation**: Replay must not modify ledger state
3. **Auditability**: All replay operations must be logged
4. **Testability**: Replay verification must be testable in CI

---

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────┐
│                  Replay Verification                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────┐         ┌──────────────────┐       │
│  │ Block Fetcher  │────────▶│ Replay Engine    │       │
│  └────────────────┘         └──────────────────┘       │
│         │                            │                  │
│         │                            ▼                  │
│         │                   ┌──────────────────┐       │
│         │                   │ Root Recomputer  │       │
│         │                   └──────────────────┘       │
│         │                            │                  │
│         │                            ▼                  │
│         │                   ┌──────────────────┐       │
│         └──────────────────▶│ Integrity Checker│       │
│                             └──────────────────┘       │
│                                      │                  │
│                                      ▼                  │
│                             ┌──────────────────┐       │
│                             │ Audit Logger     │       │
│                             └──────────────────┘       │
└─────────────────────────────────────────────────────────┘
```

### Module Responsibilities

#### 1. Block Fetcher
- **Purpose**: Retrieve historical block data from ledger
- **Inputs**: Block ID or block number
- **Outputs**: Block metadata + canonical payloads
- **Location**: `backend/ledger/replay/fetcher.py`

#### 2. Replay Engine
- **Purpose**: Orchestrate replay process
- **Inputs**: Block data from fetcher
- **Outputs**: Replay result (success/failure + diagnostics)
- **Location**: `backend/ledger/replay/engine.py`

#### 3. Root Recomputer
- **Purpose**: Recompute attestation roots from canonical payloads
- **Inputs**: Canonical statements, proofs, UI events
- **Outputs**: Recomputed R_t, U_t, H_t
- **Location**: `backend/ledger/replay/recompute.py`

#### 4. Integrity Checker
- **Purpose**: Compare recomputed roots against stored values
- **Inputs**: Stored roots, recomputed roots
- **Outputs**: Verification result + mismatch details
- **Location**: `backend/ledger/replay/checker.py`

#### 5. Audit Logger
- **Purpose**: Log all replay operations and results
- **Inputs**: Replay events, results, errors
- **Outputs**: Structured audit logs
- **Location**: `backend/ledger/replay/audit.py`

---

## Data Flow

### Single Block Replay

```
1. fetch_block(block_id)
   ↓
2. extract_canonical_payloads(block)
   ↓
3. recompute_roots(canonical_statements, canonical_proofs, ui_events)
   ↓
4. verify_roots(stored_roots, recomputed_roots)
   ↓
5. log_replay_result(block_id, verification_status)
```

### Chain Replay

```
1. fetch_chain_blocks(start_height, end_height)
   ↓
2. for each block:
   ↓
3.   replay_block(block)
   ↓
4.   if mismatch: record_failure(block_id, details)
   ↓
5. aggregate_results(all_blocks)
   ↓
6. report_chain_integrity(pass_count, fail_count)
```

---

## Implementation Plan

### Phase 1: Core Replay Infrastructure

**File**: `backend/ledger/replay/__init__.py`
```python
"""
Replay verification system for ledger integrity.
"""

from .engine import replay_block, replay_chain
from .recompute import recompute_attestation_roots
from .checker import verify_block_integrity

__all__ = [
    "replay_block",
    "replay_chain",
    "recompute_attestation_roots",
    "verify_block_integrity",
]
```

**File**: `backend/ledger/replay/recompute.py`
```python
"""
Root recomputation from canonical block payloads.
"""

from typing import Any, Dict, List, Sequence, Tuple
from attestation.dual_root import (
    build_reasoning_attestation,
    build_ui_attestation,
    compute_composite_root,
)


def recompute_attestation_roots(
    canonical_statements: Sequence[Dict[str, Any]],
    canonical_proofs: Sequence[Dict[str, Any]],
    ui_events: Sequence[Any],
) -> Tuple[str, str, str]:
    """
    Recompute attestation roots from canonical block payloads.
    
    Args:
        canonical_statements: Canonical statement payloads
        canonical_proofs: Canonical proof payloads (reasoning events)
        ui_events: UI event payloads
        
    Returns:
        Tuple of (R_t, U_t, H_t)
        
    Invariant:
        For any historical block, recomputing roots from canonical payloads
        MUST produce identical R_t, U_t, H_t values as originally sealed.
    """
    # Build reasoning tree from proofs (reasoning events)
    reasoning_tree = build_reasoning_attestation(canonical_proofs)
    r_t = reasoning_tree.root
    
    # Build UI tree from UI events
    ui_tree = build_ui_attestation(ui_events)
    u_t = ui_tree.root
    
    # Compute composite root
    h_t = compute_composite_root(r_t, u_t)
    
    return r_t, u_t, h_t
```

**File**: `backend/ledger/replay/checker.py`
```python
"""
Integrity verification for replayed blocks.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class IntegrityResult:
    """Result of block integrity verification."""
    
    block_id: int
    block_number: int
    is_valid: bool
    
    stored_r_t: str
    stored_u_t: str
    stored_h_t: str
    
    recomputed_r_t: str
    recomputed_u_t: str
    recomputed_h_t: str
    
    r_t_match: bool
    u_t_match: bool
    h_t_match: bool
    
    error: Optional[str] = None
    
    def to_dict(self):
        return {
            "block_id": self.block_id,
            "block_number": self.block_number,
            "is_valid": self.is_valid,
            "stored_roots": {
                "r_t": self.stored_r_t,
                "u_t": self.stored_u_t,
                "h_t": self.stored_h_t,
            },
            "recomputed_roots": {
                "r_t": self.recomputed_r_t,
                "u_t": self.recomputed_u_t,
                "h_t": self.recomputed_h_t,
            },
            "matches": {
                "r_t": self.r_t_match,
                "u_t": self.u_t_match,
                "h_t": self.h_t_match,
            },
            "error": self.error,
        }


def verify_block_integrity(
    block_id: int,
    block_number: int,
    stored_r_t: str,
    stored_u_t: str,
    stored_h_t: str,
    recomputed_r_t: str,
    recomputed_u_t: str,
    recomputed_h_t: str,
) -> IntegrityResult:
    """
    Verify that recomputed roots match stored roots.
    
    Args:
        block_id: Database block ID
        block_number: Block number in chain
        stored_r_t: Stored reasoning root
        stored_u_t: Stored UI root
        stored_h_t: Stored composite root
        recomputed_r_t: Recomputed reasoning root
        recomputed_u_t: Recomputed UI root
        recomputed_h_t: Recomputed composite root
        
    Returns:
        IntegrityResult with detailed comparison
    """
    r_t_match = stored_r_t == recomputed_r_t
    u_t_match = stored_u_t == recomputed_u_t
    h_t_match = stored_h_t == recomputed_h_t
    
    is_valid = r_t_match and u_t_match and h_t_match
    
    error = None
    if not is_valid:
        mismatches = []
        if not r_t_match:
            mismatches.append("R_t")
        if not u_t_match:
            mismatches.append("U_t")
        if not h_t_match:
            mismatches.append("H_t")
        error = f"Root mismatch: {', '.join(mismatches)}"
    
    return IntegrityResult(
        block_id=block_id,
        block_number=block_number,
        is_valid=is_valid,
        stored_r_t=stored_r_t,
        stored_u_t=stored_u_t,
        stored_h_t=stored_h_t,
        recomputed_r_t=recomputed_r_t,
        recomputed_u_t=recomputed_u_t,
        recomputed_h_t=recomputed_h_t,
        r_t_match=r_t_match,
        u_t_match=u_t_match,
        h_t_match=h_t_match,
        error=error,
    )
```

**File**: `backend/ledger/replay/engine.py`
```python
"""
Replay engine for block verification.
"""

import json
from typing import Any, Dict, List, Optional
from .recompute import recompute_attestation_roots
from .checker import verify_block_integrity, IntegrityResult


def replay_block(block_data: Dict[str, Any]) -> IntegrityResult:
    """
    Replay a single block and verify integrity.
    
    Args:
        block_data: Block data from database (must include canonical payloads)
        
    Returns:
        IntegrityResult with verification status
        
    Raises:
        ValueError: If block data is missing required fields
    """
    # Extract block metadata
    block_id = block_data.get("id")
    block_number = block_data.get("block_number")
    
    if not block_id or not block_number:
        raise ValueError("Block data missing id or block_number")
    
    # Extract stored roots
    stored_r_t = block_data.get("reasoning_merkle_root")
    stored_u_t = block_data.get("ui_merkle_root")
    stored_h_t = block_data.get("composite_attestation_root")
    
    if not stored_r_t or not stored_u_t or not stored_h_t:
        raise ValueError(f"Block {block_id} missing attestation roots")
    
    # Extract canonical payloads
    canonical_statements = block_data.get("canonical_statements", [])
    canonical_proofs = block_data.get("canonical_proofs", [])
    
    # Extract UI events from attestation metadata
    attestation_metadata = block_data.get("attestation_metadata", {})
    ui_leaves = attestation_metadata.get("ui_leaves", [])
    ui_events = [leaf["canonical_value"] for leaf in ui_leaves]
    
    # Recompute roots
    recomputed_r_t, recomputed_u_t, recomputed_h_t = recompute_attestation_roots(
        canonical_statements=canonical_statements,
        canonical_proofs=canonical_proofs,
        ui_events=ui_events,
    )
    
    # Verify integrity
    result = verify_block_integrity(
        block_id=block_id,
        block_number=block_number,
        stored_r_t=stored_r_t,
        stored_u_t=stored_u_t,
        stored_h_t=stored_h_t,
        recomputed_r_t=recomputed_r_t,
        recomputed_u_t=recomputed_u_t,
        recomputed_h_t=recomputed_h_t,
    )
    
    return result


def replay_chain(
    blocks: List[Dict[str, Any]],
    stop_on_failure: bool = False,
) -> Dict[str, Any]:
    """
    Replay multiple blocks and aggregate results.
    
    Args:
        blocks: List of block data dictionaries
        stop_on_failure: If True, stop on first failure
        
    Returns:
        Dictionary with aggregated results:
        - total_blocks: Total blocks replayed
        - valid_blocks: Number of valid blocks
        - invalid_blocks: Number of invalid blocks
        - results: List of IntegrityResult dictionaries
        - first_failure: First failure details (if any)
    """
    results = []
    valid_count = 0
    invalid_count = 0
    first_failure = None
    
    for block in blocks:
        try:
            result = replay_block(block)
            results.append(result.to_dict())
            
            if result.is_valid:
                valid_count += 1
            else:
                invalid_count += 1
                if first_failure is None:
                    first_failure = result.to_dict()
                
                if stop_on_failure:
                    break
                    
        except Exception as e:
            error_result = {
                "block_id": block.get("id"),
                "block_number": block.get("block_number"),
                "is_valid": False,
                "error": str(e),
            }
            results.append(error_result)
            invalid_count += 1
            
            if first_failure is None:
                first_failure = error_result
            
            if stop_on_failure:
                break
    
    return {
        "total_blocks": len(blocks),
        "valid_blocks": valid_count,
        "invalid_blocks": invalid_count,
        "success_rate": valid_count / len(blocks) if blocks else 0.0,
        "results": results,
        "first_failure": first_failure,
    }
```

### Phase 2: Testing Infrastructure

**File**: `tests/unit/test_replay_verification.py`
```python
"""
Unit tests for replay verification system.
"""

import pytest
from backend.ledger.replay import (
    replay_block,
    replay_chain,
    recompute_attestation_roots,
    verify_block_integrity,
)
from attestation.dual_root import (
    compute_reasoning_root,
    compute_ui_root,
    compute_composite_root,
)


class TestRootRecomputation:
    """Test root recomputation from canonical payloads."""
    
    def test_recompute_roots_deterministic(self):
        """Test that recomputation is deterministic."""
        proofs = [{"statement": "p -> p", "method": "axiom"}]
        ui_events = ["event1"]
        
        r_t1, u_t1, h_t1 = recompute_attestation_roots([], proofs, ui_events)
        r_t2, u_t2, h_t2 = recompute_attestation_roots([], proofs, ui_events)
        
        assert r_t1 == r_t2
        assert u_t1 == u_t2
        assert h_t1 == h_t2
    
    def test_recompute_roots_matches_original(self):
        """Test that recomputation matches original sealing."""
        proofs = [{"statement": "p -> p", "method": "axiom"}]
        ui_events = ["event1"]
        
        # Original computation
        original_r_t = compute_reasoning_root(proofs)
        original_u_t = compute_ui_root(ui_events)
        original_h_t = compute_composite_root(original_r_t, original_u_t)
        
        # Recomputation
        recomputed_r_t, recomputed_u_t, recomputed_h_t = recompute_attestation_roots(
            [], proofs, ui_events
        )
        
        assert recomputed_r_t == original_r_t
        assert recomputed_u_t == original_u_t
        assert recomputed_h_t == original_h_t


class TestIntegrityVerification:
    """Test integrity verification logic."""
    
    def test_verify_valid_block(self):
        """Test verification of valid block."""
        result = verify_block_integrity(
            block_id=1,
            block_number=1,
            stored_r_t="a" * 64,
            stored_u_t="b" * 64,
            stored_h_t="c" * 64,
            recomputed_r_t="a" * 64,
            recomputed_u_t="b" * 64,
            recomputed_h_t="c" * 64,
        )
        
        assert result.is_valid is True
        assert result.r_t_match is True
        assert result.u_t_match is True
        assert result.h_t_match is True
        assert result.error is None
    
    def test_verify_invalid_r_t(self):
        """Test verification detects R_t mismatch."""
        result = verify_block_integrity(
            block_id=1,
            block_number=1,
            stored_r_t="a" * 64,
            stored_u_t="b" * 64,
            stored_h_t="c" * 64,
            recomputed_r_t="x" * 64,
            recomputed_u_t="b" * 64,
            recomputed_h_t="c" * 64,
        )
        
        assert result.is_valid is False
        assert result.r_t_match is False
        assert result.u_t_match is True
        assert result.h_t_match is True
        assert "R_t" in result.error


class TestReplayEngine:
    """Test replay engine."""
    
    def test_replay_block_valid(self):
        """Test replaying a valid block."""
        # This would require a real block from DB or mock
        # Placeholder for integration test
        pass
    
    def test_replay_chain_all_valid(self):
        """Test replaying chain with all valid blocks."""
        # Placeholder for integration test
        pass
    
    def test_replay_chain_with_failure(self):
        """Test replaying chain with invalid block."""
        # Placeholder for integration test
        pass
```

### Phase 3: CLI Tool

**File**: `scripts/replay_verify.py`
```python
#!/usr/bin/env python3
"""
CLI tool for replay verification.

Usage:
    python scripts/replay_verify.py --block-id 123
    python scripts/replay_verify.py --block-range 1-100
    python scripts/replay_verify.py --all
"""

import argparse
import json
import sys
from backend.ledger.replay import replay_block, replay_chain


def main():
    parser = argparse.ArgumentParser(description="Replay verification tool")
    parser.add_argument("--block-id", type=int, help="Replay single block by ID")
    parser.add_argument("--block-range", help="Replay block range (e.g., 1-100)")
    parser.add_argument("--all", action="store_true", help="Replay entire chain")
    parser.add_argument("--stop-on-failure", action="store_true", help="Stop on first failure")
    parser.add_argument("--output", help="Output file for results (JSON)")
    
    args = parser.parse_args()
    
    # TODO: Implement DB fetching logic
    # For now, placeholder
    
    print("Replay verification tool - Implementation pending")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

---

## Testing Strategy

### Unit Tests
- Test root recomputation determinism
- Test integrity verification logic
- Test error handling

### Integration Tests
- Test replay of real blocks from test database
- Test chain replay with multiple blocks
- Test failure detection and reporting

### CI Integration
- Add replay verification to CI pipeline
- Run replay verification on every commit
- Block merges if replay verification fails

---

## Success Criteria

1. ✅ All historical blocks replay successfully
2. ✅ Recomputed roots match stored roots
3. ✅ Replay tests pass in CI
4. ✅ Replay tool available for manual verification
5. ✅ Audit logs capture all replay operations

---

## Future Enhancements

1. **Parallel Replay**: Replay multiple blocks in parallel for performance
2. **Incremental Replay**: Only replay blocks since last verification
3. **Replay Metrics**: Track replay performance and failure rates
4. **Automatic Remediation**: Suggest fixes for failed replays
5. **Replay API**: Expose replay verification via API endpoint

---

**Status**: Ready for implementation
