# Consensus → Replay Integration Plan

**Author**: Manus-B (Ledger Replay Architect & PQ Migration Officer)  
**Phase**: IV - Consensus Integration & Enforcement  
**Date**: 2025-12-09  
**Status**: Implementation Ready

---

## Purpose

Wire consensus rules (`rules.py`, `validators.py`, `pq_migration.py`) into the existing replay engine to enforce ledger integrity at replay time.

**Integration Points**:
1. Violation objects into replay results
2. Consensus-first block vetting
3. Fallback pathways for mixed SHA-256/SHA-3 epochs

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Replay Verification Engine                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Recompute    │  │ Checker      │  │ Engine       │      │
│  │ Module       │  │ Module       │  │ Module       │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            ▲
                            │ Consensus Integration Layer
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Consensus Runtime                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Rules        │  │ Validators   │  │ PQ Migration │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

## Integration 1: Violation Objects into Replay Results

### Current Replay Result Schema

**File**: `backend/ledger/replay/recompute.py`

```python
# Current (Phase III)
@dataclass
class ReplayResult:
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
```

---

### Enhanced Replay Result Schema (Phase IV)

**File**: `backend/ledger/replay/recompute.py`

```python
# Enhanced (Phase IV)
from backend.consensus.violations import RuleViolation
from typing import List, Optional

@dataclass
class ReplayResult:
    # Existing fields
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

---

### Code Diff: Integrating Violations into Replay

**File**: `backend/ledger/replay/checker.py`

```diff
# backend/ledger/replay/checker.py

from backend.ledger.replay.recompute import recompute_composite_root, ReplayResult
+from backend.consensus.rules import validate_block_structure, validate_attestation_roots
+from backend.consensus.violations import RuleViolation
from typing import Dict, Any, Tuple, Optional, List

def verify_block_replay(block: Dict[str, Any]) -> ReplayResult:
    """
    Verify block replay with consensus integration.
    
    Args:
        block: Block dictionary
    
    Returns:
        ReplayResult with consensus violations
    """
+    # Step 1: Consensus-first block vetting
+    consensus_violations = []
+    
+    # Validate block structure
+    is_valid, structure_violations = validate_block_structure(block)
+    if not is_valid:
+        consensus_violations.extend(structure_violations)
+    
    # Step 2: Recompute attestation roots
    r_t = block.get("reasoning_attestation_root")
    u_t = block.get("ui_attestation_root")
    h_t_stored = block.get("composite_attestation_root")
    hash_version = block.get("attestation_metadata", {}).get("hash_version", "sha256-v1")
    
    r_t_recomputed = recompute_reasoning_root(block, hash_version)
    u_t_recomputed = recompute_ui_root(block, hash_version)
    h_t_recomputed = recompute_composite_root(r_t_recomputed, u_t_recomputed, hash_version)
    
+    # Step 3: Validate attestation roots (consensus)
+    is_valid, attestation_violations = validate_attestation_roots(
+        block, r_t_recomputed, u_t_recomputed, h_t_recomputed
+    )
+    if not is_valid:
+        consensus_violations.extend(attestation_violations)
+    
+    # Step 4: Determine consensus severity
+    consensus_severity = None
+    if consensus_violations:
+        severities = [v.severity for v in consensus_violations]
+        if "CRITICAL" in severities:
+            consensus_severity = "CRITICAL"
+        elif "ERROR" in severities:
+            consensus_severity = "ERROR"
+        elif "WARNING" in severities:
+            consensus_severity = "WARNING"
+        else:
+            consensus_severity = "INFO"
+    
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
+        consensus_violations=consensus_violations,
+        consensus_passed=(len(consensus_violations) == 0),
+        consensus_severity=consensus_severity,
    )
```

---

## Integration 2: Consensus-First Block Vetting

### Vetting Pipeline

**Order of Operations**:
1. **Consensus validation** (structure, monotonicity, prev_hash)
2. **Replay verification** (recompute roots)
3. **Attestation validation** (verify roots match)

**Rationale**: Fail fast on consensus violations before expensive replay computation.

---

### Code Diff: Consensus-First Vetting

**File**: `backend/ledger/replay/engine.py`

```diff
# backend/ledger/replay/engine.py

from backend.ledger.replay.checker import verify_block_replay
+from backend.consensus.validators import BlockValidator
+from backend.consensus.rules import validate_prev_hash, validate_monotonicity
from typing import Dict, Any, List

def replay_blocks(
    blocks: List[Dict[str, Any]],
+    consensus_first: bool = True,
+    fail_fast: bool = False,
) -> List[ReplayResult]:
    """
    Replay blocks with consensus-first vetting.
    
    Args:
        blocks: List of blocks to replay
+        consensus_first: If True, validate consensus before replay
+        fail_fast: If True, stop on first critical violation
    
    Returns:
        List of ReplayResult
    """
+    if consensus_first:
+        # Pre-flight consensus checks
+        validator = BlockValidator()
+        for i, block in enumerate(blocks):
+            # Validate monotonicity (if not first block)
+            if i > 0:
+                is_valid, violations = validate_monotonicity(blocks[i-1], block)
+                if not is_valid and fail_fast:
+                    raise ValueError(f"Monotonicity violation at block {block['block_number']}: {violations}")
+            
+            # Validate prev_hash (if not first block)
+            if i > 0:
+                is_valid, violations = validate_prev_hash(block, blocks[i-1])
+                if not is_valid and fail_fast:
+                    raise ValueError(f"Prev_hash violation at block {block['block_number']}: {violations}")
    
    results = []
    for block in blocks:
        result = verify_block_replay(block)
        results.append(result)
+        
+        # Fail fast on critical violations
+        if fail_fast and result.has_critical_violations():
+            raise ValueError(f"Critical violation at block {block['block_number']}")
    
    return results
```

---

## Integration 3: Fallback Pathways for Mixed SHA-256/SHA-3 Epochs

### Problem

During PQ migration, epochs may contain blocks with different hash versions:
- Blocks 0-50: `sha256-v1`
- Blocks 51-100: `dual-v1`
- Blocks 101-150: `sha3-v1`

**Challenge**: Replay engine must handle heterogeneous hash versions within a single epoch.

---

### Solution: Hash Version Detection + Fallback

**File**: `backend/ledger/replay/recompute.py`

```diff
# backend/ledger/replay/recompute.py

from backend.crypto.hash_abstraction import get_hash_algorithm
+from backend.consensus.pq_migration import detect_migration_state, validate_cross_algorithm_prev_hash

def recompute_composite_root(
    r_t: str,
    u_t: str,
    hash_version: str,
+    fallback_version: Optional[str] = None,
) -> str:
    """
    Recompute composite attestation root with fallback support.
    
    Args:
        r_t: Reasoning attestation root
        u_t: UI attestation root
        hash_version: Primary hash version
+        fallback_version: Fallback hash version (for migration)
    
    Returns:
        Composite attestation root (hex string)
    """
+    try:
+        # Try primary hash version
+        hash_algo = get_hash_algorithm(hash_version)
+        composite_data = f"EPOCH:{r_t}{u_t}".encode()
+        h_t = hash_algo.domain_hash("EPOCH:", composite_data)
+        return h_t
+    except ValueError as e:
+        # Fallback to fallback_version
+        if fallback_version:
+            hash_algo = get_hash_algorithm(fallback_version)
+            composite_data = f"EPOCH:{r_t}{u_t}".encode()
+            h_t = hash_algo.domain_hash("EPOCH:", composite_data)
+            return h_t
+        else:
+            raise ValueError(f"Unsupported hash_version: {hash_version}") from e
```

---

### Code Diff: Mixed Epoch Replay

**File**: `backend/ledger/replay/engine.py`

```diff
# backend/ledger/replay/engine.py

+from backend.consensus.pq_migration import detect_migration_state

def replay_heterogeneous_epoch(
    epoch_number: int,
+    enable_fallback: bool = True,
) -> Dict[str, Any]:
    """
    Replay heterogeneous epoch with fallback support.
    
    Args:
        epoch_number: Epoch number to replay
+        enable_fallback: Enable hash version fallback
    
    Returns:
        Replay result dictionary
    """
    # Fetch epoch metadata
    epoch = fetch_epoch(epoch_number)
    epoch_hash_version = epoch["hash_version"]
    start_block = epoch["start_block"]
    end_block = epoch["end_block"]
    
    # Fetch blocks
    blocks = fetch_blocks(start_block, end_block)
    
+    # Detect migration state
+    migration_state = detect_migration_state(blocks)
+    
+    # Determine fallback strategy
+    fallback_version = None
+    if migration_state == "DUAL_COMMITMENT" and enable_fallback:
+        fallback_version = "sha256-v1"  # Fallback to SHA-256
+    elif migration_state == "PURE_SHA3" and enable_fallback:
+        fallback_version = "sha256-v1"  # Fallback for legacy blocks
+    
    # Group blocks by hash_version
    block_groups = {}
    for block in blocks:
        block_hash_version = block.get("attestation_metadata", {}).get("hash_version", "sha256-v1")
        if block_hash_version not in block_groups:
            block_groups[block_hash_version] = []
        block_groups[block_hash_version].append(block)
    
    # Replay each group
    all_results = []
    for hash_version, group_blocks in block_groups.items():
-        group_results = replay_blocks_with_algorithm(group_blocks, hash_version)
+        group_results = replay_blocks_with_algorithm(
+            group_blocks, hash_version, fallback_version=fallback_version
+        )
        all_results.extend(group_results)
    
    # Compute epoch root
    composite_roots = [r.h_t_recomputed for r in all_results]
    epoch_algo = get_hash_algorithm(epoch_hash_version)
    recomputed_epoch_root = epoch_algo.merkle_root(composite_roots)
    
    # Compare with stored epoch root
    stored_epoch_root = epoch["epoch_root"]
    epoch_match = (recomputed_epoch_root == stored_epoch_root)
    
    return {
        "epoch_number": epoch_number,
        "epoch_hash_version": epoch_hash_version,
+        "migration_state": migration_state,
+        "fallback_enabled": enable_fallback,
+        "fallback_version": fallback_version,
        "block_groups": {hv: len(blocks) for hv, blocks in block_groups.items()},
        "total_blocks": len(blocks),
        "recomputed_epoch_root": recomputed_epoch_root,
        "stored_epoch_root": stored_epoch_root,
        "epoch_match": epoch_match,
        "block_results": all_results,
    }
```

---

## Integration Summary

### Files Modified

| File | Changes | Lines Added |
|------|---------|-------------|
| `backend/ledger/replay/recompute.py` | Enhanced ReplayResult schema, fallback support | +45 |
| `backend/ledger/replay/checker.py` | Consensus-first vetting | +30 |
| `backend/ledger/replay/engine.py` | Consensus validation, mixed epoch replay | +60 |

**Total**: 135 lines added

---

### Integration Checklist

- [ ] Update `ReplayResult` schema with consensus fields
- [ ] Integrate `validate_block_structure()` into `verify_block_replay()`
- [ ] Integrate `validate_attestation_roots()` into `verify_block_replay()`
- [ ] Add consensus-first vetting to `replay_blocks()`
- [ ] Add fallback support to `recompute_composite_root()`
- [ ] Add mixed epoch replay to `replay_heterogeneous_epoch()`
- [ ] Write integration tests (10+ tests)
- [ ] Update documentation

---

### Testing Strategy

**Unit Tests**:
1. Test `ReplayResult` with consensus violations
2. Test consensus-first vetting with valid blocks
3. Test consensus-first vetting with invalid blocks
4. Test fallback support with unsupported hash versions
5. Test mixed epoch replay with SHA-256/dual/SHA-3 blocks

**Integration Tests**:
1. Test full-chain replay with consensus violations
2. Test epoch replay with mixed hash versions
3. Test fail-fast behavior on critical violations

---

## Conclusion

The consensus-replay integration wires consensus rules into the replay engine, enabling:
1. **Violation tracking** in replay results
2. **Consensus-first vetting** for fail-fast behavior
3. **Fallback pathways** for mixed SHA-256/SHA-3 epochs

**Status**: Code-ready diffs provided, implementation pending.

---

**"Keep it blue, keep it clean, keep it sealed."**  
— Manus-B, Ledger Replay Architect & PQ Migration Officer
