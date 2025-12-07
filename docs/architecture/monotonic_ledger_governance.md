# Monotonic Ledger Governance Layer

**Author**: Manus-B (Ledger Replay Architect & Attestation Runtime Engineer)  
**Phase**: II - Epochization & Governance  
**Date**: 2025-12-06  
**Status**: Design Specification

---

## Executive Summary

The **Monotonic Ledger Governance Layer** transforms MathLedger from a ledger with replay verification capabilities into a **self-governing chain** where replay determinism is not just verified but **enforced at every mutation point**. This layer interlocks with the Phase I replay engine to create a closed-loop system: blocks can only be sealed if they preserve monotonicity, and existing blocks can only be trusted if they pass replay verification.

---

## Core Invariants

### Invariant 1: Block Number Monotonicity

**Statement**: Block numbers MUST form a strictly increasing sequence with no gaps.

**Formal Definition**:
```
∀ blocks b₁, b₂ where b₂ is sealed after b₁:
  b₂.block_number = b₁.block_number + 1
```

**Enforcement Points**:
1. **Pre-seal validation**: Before `seal_block_with_dual_roots()`, verify `new_block_number == max(existing_block_numbers) + 1`
2. **Database constraint**: `CHECK (block_number >= 0)`
3. **Unique constraint**: `UNIQUE (system_id, block_number)`

**Violation Detection**:
- Gap detection: `SELECT block_number FROM blocks ORDER BY block_number` → check for gaps
- Duplicate detection: `SELECT block_number, COUNT(*) FROM blocks GROUP BY block_number HAVING COUNT(*) > 1`

---

### Invariant 2: Prev-Hash Lineage

**Statement**: Every block MUST cryptographically commit to its predecessor via `prev_hash`.

**Formal Definition**:
```
∀ blocks b where b.block_number > 0:
  b.prev_hash = SHA256(b_prev.block_identity)
  where b_prev.block_number = b.block_number - 1
```

**Current State**: `prev_hash` exists in schema but is **NOT included in block identity derivation**.

**Required Changes**:

#### 1. Modify Block Identity Derivation

**Current** (`ledger/blocking.py`):
```python
def _derive_block_identity(system, proof_count, ui_event_count, r_t, u_t):
    material = f"{system}|{proof_count}|{ui_event_count}|{r_t}|{u_t}"
    return hashlib.sha256(material.encode()).hexdigest()
```

**Proposed**:
```python
def _derive_block_identity(
    system: str,
    block_number: int,
    prev_hash: str,
    proof_count: int,
    ui_event_count: int,
    r_t: str,
    u_t: str,
    h_t: str,
) -> str:
    """
    Derive block identity with prev_hash linkage.
    
    Block identity now commits to:
    - System identifier
    - Block number (position in chain)
    - Previous block hash (chain linkage)
    - Proof count (reasoning stream size)
    - UI event count (interaction stream size)
    - R_t (reasoning root)
    - U_t (UI root)
    - H_t (composite root)
    
    This ensures block identity is a function of both:
    1. Block content (proofs, UI events, attestation roots)
    2. Chain position (block_number, prev_hash)
    
    Breaking the chain (changing prev_hash) invalidates all descendant blocks.
    """
    material = f"{system}|{block_number}|{prev_hash}|{proof_count}|{ui_event_count}|{r_t}|{u_t}|{h_t}"
    return hashlib.sha256(material.encode()).hexdigest()
```

**Migration Path**:
- Add `hash_version` field to blocks: `"v1"` (current), `"v2"` (with prev_hash)
- Compute both identities during transition
- Store as `block_identity_v1` and `block_identity_v2`
- After full migration, deprecate v1

#### 2. Chain Validation Function

```python
def validate_chain_lineage(blocks: List[Dict[str, Any]]) -> ChainValidationResult:
    """
    Validate prev_hash linkage across entire chain.
    
    Returns:
        ChainValidationResult with:
        - is_valid: bool
        - broken_links: List[Tuple[int, int]] (block_number pairs where link breaks)
        - orphan_blocks: List[int] (blocks with no valid predecessor)
        - fork_points: List[int] (blocks with multiple valid successors)
    """
    blocks_by_number = {b["block_number"]: b for b in blocks}
    broken_links = []
    orphan_blocks = []
    
    for block in sorted(blocks, key=lambda b: b["block_number"]):
        if block["block_number"] == 0:
            # Genesis block has no predecessor
            if block["prev_hash"] is not None:
                broken_links.append((None, 0))
            continue
        
        prev_block_number = block["block_number"] - 1
        if prev_block_number not in blocks_by_number:
            orphan_blocks.append(block["block_number"])
            continue
        
        prev_block = blocks_by_number[prev_block_number]
        expected_prev_hash = SHA256(prev_block["block_identity"])
        
        if block["prev_hash"] != expected_prev_hash:
            broken_links.append((prev_block_number, block["block_number"]))
    
    # Detect forks (multiple blocks claiming same prev_hash)
    prev_hash_map = {}
    for block in blocks:
        if block["prev_hash"] in prev_hash_map:
            prev_hash_map[block["prev_hash"]].append(block["block_number"])
        else:
            prev_hash_map[block["prev_hash"]] = [block["block_number"]]
    
    fork_points = [
        blocks_by_number[min(block_numbers) - 1]["block_number"]
        for prev_hash, block_numbers in prev_hash_map.items()
        if len(block_numbers) > 1 and prev_hash is not None
    ]
    
    return ChainValidationResult(
        is_valid=len(broken_links) == 0 and len(orphan_blocks) == 0 and len(fork_points) == 0,
        broken_links=broken_links,
        orphan_blocks=orphan_blocks,
        fork_points=fork_points,
    )
```

---

### Invariant 3: Non-Deletability

**Statement**: Once sealed, blocks MUST NOT be deleted or modified.

**Enforcement**:

#### 1. Database Triggers

```sql
-- Prevent block deletion
CREATE OR REPLACE FUNCTION prevent_block_deletion()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'Ledger violation: Block deletion forbidden (block_id=%, block_number=%)', 
        OLD.id, OLD.block_number;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER enforce_block_non_deletability
BEFORE DELETE ON blocks
FOR EACH ROW
EXECUTE FUNCTION prevent_block_deletion();

-- Prevent block modification (except metadata updates)
CREATE OR REPLACE FUNCTION prevent_block_mutation()
RETURNS TRIGGER AS $$
BEGIN
    -- Allow metadata updates (e.g., adding epoch_id)
    IF NEW.block_number != OLD.block_number OR
       NEW.prev_hash != OLD.prev_hash OR
       NEW.reasoning_attestation_root != OLD.reasoning_attestation_root OR
       NEW.ui_attestation_root != OLD.ui_attestation_root OR
       NEW.composite_attestation_root != OLD.composite_attestation_root OR
       NEW.canonical_proofs != OLD.canonical_proofs OR
       NEW.canonical_statements != OLD.canonical_statements THEN
        RAISE EXCEPTION 'Ledger violation: Block mutation forbidden (block_id=%)', OLD.id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER enforce_block_immutability
BEFORE UPDATE ON blocks
FOR EACH ROW
EXECUTE FUNCTION prevent_block_mutation();
```

#### 2. Application-Level Guards

```python
def delete_block(block_id: int):
    """
    Attempt to delete a block.
    
    This function ALWAYS raises an exception.
    It exists only to document the prohibition.
    """
    raise LedgerViolationError(
        f"Ledger violation: Block deletion forbidden (block_id={block_id}). "
        "Blocks are immutable once sealed. "
        "If you need to correct an error, seal a new block with the correction."
    )

def update_block_content(block_id: int, **updates):
    """
    Attempt to update block content.
    
    Only metadata updates are allowed (e.g., epoch_id).
    Content fields (proofs, roots, etc.) are immutable.
    """
    forbidden_fields = {
        "block_number", "prev_hash", "reasoning_attestation_root",
        "ui_attestation_root", "composite_attestation_root",
        "canonical_proofs", "canonical_statements",
    }
    
    if any(field in updates for field in forbidden_fields):
        raise LedgerViolationError(
            f"Ledger violation: Cannot update block content (block_id={block_id}). "
            f"Forbidden fields: {forbidden_fields & updates.keys()}"
        )
    
    # Allow metadata updates
    allowed_fields = {"epoch_id", "metadata"}
    safe_updates = {k: v for k, v in updates.items() if k in allowed_fields}
    
    # Execute update with safe fields only
    db.execute(
        "UPDATE blocks SET " + ", ".join(f"{k} = %s" for k in safe_updates.keys()) + " WHERE id = %s",
        list(safe_updates.values()) + [block_id]
    )
```

---

### Invariant 4: Append-Only Proofs

**Statement**: The ledger MUST only grow; no reordering, no gaps, no deletions.

**Formal Definition**:
```
∀ time t₁ < t₂:
  blocks(t₂) ⊇ blocks(t₁)
  ∧ ∀ b ∈ blocks(t₁): b.content(t₂) = b.content(t₁)
```

**Enforcement**:

#### 1. Pre-Seal Validation

```python
def seal_block_with_governance(
    system_id: str,
    proofs: List[Dict],
    ui_events: List[str],
    **kwargs
) -> Dict[str, Any]:
    """
    Seal block with full governance checks.
    
    Governance checks:
    1. Block number monotonicity
    2. Prev-hash lineage
    3. No concurrent sealing (race condition detection)
    4. Replay verification of last N blocks
    """
    # 1. Get current chain head
    current_head = get_chain_head(system_id)
    
    if current_head is None:
        # Genesis block
        new_block_number = 0
        prev_hash = None
        prev_block_identity = None
    else:
        # Validate monotonicity
        new_block_number = current_head["block_number"] + 1
        prev_hash = SHA256(current_head["block_identity"])
        prev_block_identity = current_head["block_identity"]
        
        # Detect race condition: another block sealed since we fetched head
        recheck_head = get_chain_head(system_id)
        if recheck_head["id"] != current_head["id"]:
            raise ConcurrentSealingError(
                f"Race condition detected: chain head changed during sealing. "
                f"Expected block {current_head['id']}, found {recheck_head['id']}. "
                f"Retry sealing."
            )
    
    # 2. Compute attestation roots
    r_t = compute_reasoning_root(proofs)
    u_t = compute_ui_root(ui_events)
    h_t = compute_composite_root(r_t, u_t)
    
    # 3. Derive block identity (with prev_hash linkage)
    block_identity = _derive_block_identity(
        system=system_id,
        block_number=new_block_number,
        prev_hash=prev_hash,
        proof_count=len(proofs),
        ui_event_count=len(ui_events),
        r_t=r_t,
        u_t=u_t,
        h_t=h_t,
    )
    
    # 4. Seal block
    block = {
        "system_id": system_id,
        "block_number": new_block_number,
        "prev_hash": prev_hash,
        "prev_block_id": current_head["id"] if current_head else None,
        "block_identity": block_identity,
        "reasoning_attestation_root": r_t,
        "ui_attestation_root": u_t,
        "composite_attestation_root": h_t,
        "canonical_proofs": proofs,
        "canonical_statements": extract_statements(proofs),
        "attestation_metadata": {
            "ui_leaves": ui_events,
            "proof_count": len(proofs),
            "ui_event_count": len(ui_events),
            "sealed_at": datetime.utcnow().isoformat(),
        },
    }
    
    # 5. Store block (triggers will enforce immutability)
    block_id = store_block(block)
    
    # 6. Post-seal verification: replay last N blocks
    verify_recent_chain(system_id, window_size=10)
    
    return {**block, "id": block_id}
```

#### 2. Periodic Chain Audit

```python
def audit_chain_append_only(system_id: str, snapshot_interval: timedelta) -> AuditReport:
    """
    Verify ledger has only grown since last snapshot.
    
    Algorithm:
    1. Load last snapshot (block_count, max_block_number, chain_head_hash)
    2. Fetch current chain state
    3. Verify:
       - block_count(now) >= block_count(snapshot)
       - max_block_number(now) >= max_block_number(snapshot)
       - All blocks from snapshot still exist with identical content
    4. Store new snapshot
    """
    last_snapshot = load_snapshot(system_id)
    current_blocks = fetch_all_blocks(system_id)
    
    violations = []
    
    # Check block count
    if len(current_blocks) < last_snapshot["block_count"]:
        violations.append(f"Block count decreased: {last_snapshot['block_count']} → {len(current_blocks)}")
    
    # Check max block number
    current_max = max(b["block_number"] for b in current_blocks)
    if current_max < last_snapshot["max_block_number"]:
        violations.append(f"Max block number decreased: {last_snapshot['max_block_number']} → {current_max}")
    
    # Check all snapshot blocks still exist
    snapshot_blocks = {b["block_number"]: b for b in last_snapshot["blocks"]}
    for block_number, snapshot_block in snapshot_blocks.items():
        current_block = next((b for b in current_blocks if b["block_number"] == block_number), None)
        
        if current_block is None:
            violations.append(f"Block {block_number} deleted")
        elif current_block["composite_attestation_root"] != snapshot_block["composite_attestation_root"]:
            violations.append(f"Block {block_number} mutated: H_t changed")
    
    # Store new snapshot
    store_snapshot(system_id, {
        "block_count": len(current_blocks),
        "max_block_number": current_max,
        "chain_head_hash": current_blocks[-1]["block_identity"] if current_blocks else None,
        "blocks": current_blocks,
        "timestamp": datetime.utcnow(),
    })
    
    return AuditReport(
        is_valid=len(violations) == 0,
        violations=violations,
        snapshot_interval=snapshot_interval,
    )
```

---

### Invariant 5: Hash-Chain Violation Detection

**Statement**: Any break in the hash chain MUST be detected and reported.

**Detection Algorithm**:

```python
def detect_hash_chain_violations(blocks: List[Dict[str, Any]]) -> List[Violation]:
    """
    Detect all hash-chain violations.
    
    Violation types:
    1. Missing predecessor: block.prev_hash points to non-existent block
    2. Incorrect prev_hash: block.prev_hash != SHA256(predecessor.block_identity)
    3. Orphan block: block has no predecessor but block_number > 0
    4. Fork: multiple blocks with same prev_hash
    5. Cycle: prev_hash chain forms a cycle
    6. Genesis violation: block_number=0 but prev_hash != None
    """
    violations = []
    blocks_by_number = {b["block_number"]: b for b in blocks}
    blocks_by_identity = {b["block_identity"]: b for b in blocks}
    
    # Build prev_hash → blocks map
    children_map = {}
    for block in blocks:
        if block["prev_hash"] not in children_map:
            children_map[block["prev_hash"]] = []
        children_map[block["prev_hash"]].append(block)
    
    for block in blocks:
        block_number = block["block_number"]
        prev_hash = block["prev_hash"]
        
        # Genesis block checks
        if block_number == 0:
            if prev_hash is not None:
                violations.append(Violation(
                    type="genesis_violation",
                    block_number=block_number,
                    message=f"Genesis block has non-null prev_hash: {prev_hash}",
                ))
            continue
        
        # Non-genesis block checks
        if prev_hash is None:
            violations.append(Violation(
                type="orphan_block",
                block_number=block_number,
                message=f"Block {block_number} has null prev_hash but is not genesis",
            ))
            continue
        
        # Find predecessor
        predecessor = blocks_by_identity.get(prev_hash)
        if predecessor is None:
            violations.append(Violation(
                type="missing_predecessor",
                block_number=block_number,
                message=f"Block {block_number} prev_hash points to non-existent block: {prev_hash}",
            ))
            continue
        
        # Verify prev_hash correctness
        expected_prev_hash = SHA256(predecessor["block_identity"])
        if prev_hash != expected_prev_hash:
            violations.append(Violation(
                type="incorrect_prev_hash",
                block_number=block_number,
                message=f"Block {block_number} prev_hash mismatch: expected {expected_prev_hash}, got {prev_hash}",
            ))
        
        # Verify sequential block numbers
        if predecessor["block_number"] != block_number - 1:
            violations.append(Violation(
                type="non_sequential",
                block_number=block_number,
                message=f"Block {block_number} predecessor has block_number {predecessor['block_number']}, expected {block_number - 1}",
            ))
    
    # Detect forks
    for prev_hash, children in children_map.items():
        if len(children) > 1 and prev_hash is not None:
            violations.append(Violation(
                type="fork",
                block_number=blocks_by_identity[prev_hash]["block_number"],
                message=f"Fork detected: {len(children)} blocks with same prev_hash: {[c['block_number'] for c in children]}",
            ))
    
    # Detect cycles
    visited = set()
    for block in blocks:
        path = []
        current = block
        while current is not None and current["block_identity"] not in visited:
            if current["block_identity"] in path:
                violations.append(Violation(
                    type="cycle",
                    block_number=current["block_number"],
                    message=f"Cycle detected in chain: {[blocks_by_identity[h]['block_number'] for h in path]}",
                ))
                break
            path.append(current["block_identity"])
            current = blocks_by_identity.get(current["prev_hash"])
        visited.update(path)
    
    return violations
```

---

## Governance Enforcement Architecture

### Layer 1: Database Constraints

```sql
-- Monotonicity
ALTER TABLE blocks ADD CONSTRAINT block_number_non_negative CHECK (block_number >= 0);
CREATE UNIQUE INDEX blocks_system_block_number_unique ON blocks(system_id, block_number);

-- Immutability
CREATE TRIGGER enforce_block_non_deletability BEFORE DELETE ON blocks ...;
CREATE TRIGGER enforce_block_immutability BEFORE UPDATE ON blocks ...;

-- Referential integrity
ALTER TABLE blocks ADD CONSTRAINT prev_block_exists 
    FOREIGN KEY (prev_block_id) REFERENCES blocks(id) ON DELETE RESTRICT;
```

### Layer 2: Application Logic

```python
# Pre-seal validation
def seal_block_with_governance(...):
    validate_monotonicity()
    validate_prev_hash_lineage()
    detect_race_conditions()
    seal_block()
    verify_recent_chain()
```

### Layer 3: Periodic Audits

```python
# Cron job: every 1 hour
def hourly_chain_audit():
    for system in all_systems():
        audit_chain_append_only(system)
        validate_chain_lineage(system)
        detect_hash_chain_violations(system)
        verify_epoch_integrity(system)
```

### Layer 4: CI Governance Gate

```yaml
# .github/workflows/replay-governance.yml
- name: Replay Verification
  run: python scripts/replay_verify.py --all --strict
- name: Chain Validation
  run: python scripts/validate_chain.py --all
- name: Monotonicity Audit
  run: python scripts/audit_monotonicity.py --all
```

---

## Integration with Replay Engine

### Closed-Loop Governance

```
┌─────────────────────────────────────────────────────────────┐
│                     Seal New Block                          │
│                                                             │
│  1. Validate monotonicity (block_number = head + 1)        │
│  2. Compute prev_hash = SHA256(head.block_identity)        │
│  3. Compute attestation roots (R_t, U_t, H_t)              │
│  4. Derive block_identity (includes prev_hash)             │
│  5. Store block (triggers enforce immutability)            │
│  6. Replay last N blocks (verify determinism)              │
└─────────────────────────────────────────────────────────────┘
                            ↓
                    ┌───────────────┐
                    │  Replay Pass? │
                    └───────────────┘
                      ↙           ↘
                   YES             NO
                    ↓               ↓
            ┌──────────────┐  ┌──────────────────┐
            │ Block Valid  │  │ ROLLBACK BLOCK   │
            │ Commit TX    │  │ Investigate Drift│
            └──────────────┘  └──────────────────┘
```

### Replay-Governed Sealing

```python
def seal_block_with_replay_governance(
    system_id: str,
    proofs: List[Dict],
    ui_events: List[str],
    replay_window: int = 10,
) -> Dict[str, Any]:
    """
    Seal block with replay verification gate.
    
    If replay fails, block sealing is ROLLED BACK.
    """
    # Begin transaction
    with db.transaction() as tx:
        # Seal block
        block = seal_block_with_governance(system_id, proofs, ui_events)
        
        # Replay last N blocks (including new block)
        recent_blocks = fetch_recent_blocks(system_id, limit=replay_window)
        replay_result = replay_chain(recent_blocks)
        
        if not replay_result["is_valid"]:
            # ROLLBACK: replay failed
            tx.rollback()
            raise ReplayGovernanceViolation(
                f"Replay verification failed after sealing block {block['block_number']}. "
                f"Valid: {replay_result['valid_blocks']}/{replay_result['total_blocks']}. "
                f"First failure: block {replay_result['first_failure']['block_number']}. "
                f"Block sealing ROLLED BACK."
            )
        
        # Commit transaction
        tx.commit()
        
        return block
```

---

## Violation Classification

### Severity Levels

| Level | Name | Description | Response |
|-------|------|-------------|----------|
| 0 | **CRITICAL** | Chain integrity compromised (fork, cycle, deletion) | Halt all operations, forensic investigation |
| 1 | **HIGH** | Monotonicity violated (gap, duplicate block_number) | Block new sealing, repair chain |
| 2 | **MEDIUM** | Replay failure (H_t mismatch) | Investigate drift, reseal if needed |
| 3 | **LOW** | Metadata inconsistency (epoch_id mismatch) | Log warning, schedule repair |

### Violation Response Matrix

| Violation Type | Severity | Auto-Remediation | Manual Investigation |
|----------------|----------|------------------|----------------------|
| Block deletion | CRITICAL | ❌ None | ✅ Required |
| Fork detected | CRITICAL | ❌ None | ✅ Required |
| Cycle detected | CRITICAL | ❌ None | ✅ Required |
| Block number gap | HIGH | ⚠️ Attempt backfill | ✅ If backfill fails |
| Duplicate block_number | HIGH | ❌ None | ✅ Required |
| Incorrect prev_hash | HIGH | ⚠️ Recompute if deterministic | ✅ If non-deterministic |
| Replay failure (H_t) | MEDIUM | ⚠️ Reseal if canonical data intact | ✅ If data corrupted |
| Epoch_id mismatch | LOW | ✅ Recompute epoch_id | ❌ Optional |

---

## Tooling Recommendations

### 1. Ledger Governance CLI

```bash
# Validate entire chain
mathledger-gov validate-chain --system-id <uuid> --strict

# Audit monotonicity
mathledger-gov audit-monotonicity --system-id <uuid> --snapshot-interval 1h

# Detect violations
mathledger-gov detect-violations --system-id <uuid> --output violations.json

# Repair chain (interactive)
mathledger-gov repair-chain --system-id <uuid> --dry-run
```

### 2. Governance Dashboard

```python
# Real-time governance metrics
class GovernanceDashboard:
    def get_metrics(self, system_id: str) -> Dict:
        return {
            "chain_health": {
                "total_blocks": count_blocks(system_id),
                "max_block_number": get_max_block_number(system_id),
                "chain_head": get_chain_head(system_id),
                "last_sealed_at": get_last_sealed_timestamp(system_id),
            },
            "violations": {
                "critical": count_violations(system_id, severity="CRITICAL"),
                "high": count_violations(system_id, severity="HIGH"),
                "medium": count_violations(system_id, severity="MEDIUM"),
                "low": count_violations(system_id, severity="LOW"),
            },
            "replay_stats": {
                "last_replay_at": get_last_replay_timestamp(system_id),
                "replay_success_rate": get_replay_success_rate(system_id),
                "avg_replay_time": get_avg_replay_time(system_id),
            },
            "governance_status": get_governance_status(system_id),  # HEALTHY | DEGRADED | CRITICAL
        }
```

### 3. Violation Alerting

```python
# Slack/PagerDuty integration
def alert_on_violation(violation: Violation):
    if violation.severity == "CRITICAL":
        pagerduty.trigger_incident(
            title=f"CRITICAL: Ledger violation in system {violation.system_id}",
            description=violation.message,
            urgency="high",
        )
    elif violation.severity == "HIGH":
        slack.send_message(
            channel="#ledger-alerts",
            text=f"⚠️ HIGH: {violation.message}",
        )
```

---

## Status

**Design**: ✅ Complete  
**Implementation**: ⏳ Pending  
**Testing**: ⏳ Pending  
**Deployment**: ⏳ Pending

---

**Next**: Epoch Schema Migration & Backfill Strategy
