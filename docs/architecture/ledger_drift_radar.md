# Ledger Drift Radar System

**Author**: Manus-B (Ledger Replay Architect & Attestation Runtime Engineer)  
**Phase**: II - Epochization & Governance  
**Date**: 2025-12-06  
**Status**: Design Specification

---

## Executive Summary

The **Ledger Drift Radar** is a continuous monitoring and forensic analysis system that detects, classifies, and diagnoses deviations from expected ledger behavior. It operates as an early-warning system for:

- **Schema drift**: Changes in canonical payload formats
- **Hash drift**: Unexpected changes in attestation root computation
- **Metadata drift**: Inconsistencies in attestation_metadata structure
- **Statement drift**: Changes in canonical_statements format or content

The Drift Radar transforms replay failures from opaque errors into actionable forensic intelligence.

---

## Motivation

### Problem

When replay verification fails (H_t mismatch), the error message is:
```
Block 1234 failed: H_t mismatch (stored: abc..., recomputed: def...)
```

This tells us **WHAT** failed, but not **WHY**. Possible causes:
- Code change in hash computation
- Schema migration changed payload format
- Data corruption in database
- Bug in canonicalization logic
- Intentional malicious tampering

### Solution

The Drift Radar provides:
1. **Root cause analysis**: Classify drift into known categories
2. **Forensic artifacts**: Capture evidence for debugging
3. **Drift severity**: Distinguish benign from critical drift
4. **Remediation guidance**: Suggest fixes based on drift type

---

## Drift Signal Taxonomy

### Level 1: Schema Drift

**Definition**: Changes in the structure or format of canonical payloads.

**Detection Signals**:
- `canonical_proofs` field type changed (dict → list, or vice versa)
- New fields added to proof objects
- Required fields removed from proof objects
- Field renamed (e.g., `statement` → `stmt`)

**Example**:
```python
# Old format
canonical_proofs = [
    {"statement": "p -> p", "method": "axiom"}
]

# New format (drift)
canonical_proofs = [
    {"stmt": "p -> p", "proof_method": "axiom", "version": "v2"}
]
```

**Severity**: **MEDIUM** (requires migration, but deterministic)

**Remediation**:
- Implement schema version detection
- Add backward compatibility layer
- Migrate old blocks to new schema (if safe)
- Or: maintain dual-schema support

---

### Level 2: Hash-Delta Drift

**Definition**: Changes in hash computation logic that affect attestation roots.

**Detection Signals**:
- R_t mismatch but canonical_proofs unchanged
- U_t mismatch but ui_events unchanged
- H_t mismatch but R_t and U_t match
- Domain separation tags changed
- Hash algorithm changed (SHA-256 → SHA-3)

**Example**:
```python
# Old: domain tag "STMT"
leaf_hash = SHA256("STMT:" + statement)

# New: domain tag "STATEMENT" (drift)
leaf_hash = SHA256("STATEMENT:" + statement)
```

**Severity**: **HIGH** (breaks replay determinism)

**Remediation**:
- Identify code change that altered hash computation
- Add hash_version field to metadata
- Implement dual-hash during migration
- Reseal affected blocks with corrected hashes

---

### Level 3: Metadata Drift

**Definition**: Inconsistencies in attestation_metadata structure or content.

**Detection Signals**:
- `ui_leaves` field missing or renamed
- `proof_count` mismatch with len(canonical_proofs)
- `ui_event_count` mismatch with len(ui_leaves)
- New metadata fields added without version bump

**Example**:
```python
# Old metadata
attestation_metadata = {
    "ui_leaves": ["event1", "event2"],
    "proof_count": 5
}

# New metadata (drift)
attestation_metadata = {
    "ui_events": ["event1", "event2"],  # Renamed field
    "proof_count": 5,
    "hash_version": "v2"  # New field
}
```

**Severity**: **LOW** (usually doesn't affect roots, but indicates schema evolution)

**Remediation**:
- Document metadata schema changes
- Add version field to track evolution
- Ensure backward compatibility

---

### Level 4: Statement Drift

**Definition**: Changes in canonical_statements format or content.

**Detection Signals**:
- Statement hash changed but statement text unchanged
- Canonicalization order changed (sorted → unsorted)
- Statement deduplication logic changed
- Whitespace normalization changed

**Example**:
```python
# Old: statements sorted
canonical_statements = ["p -> p", "q -> q"]

# New: statements unsorted (drift)
canonical_statements = ["q -> q", "p -> p"]
```

**Severity**: **HIGH** (affects R_t computation)

**Remediation**:
- Enforce deterministic statement ordering
- Document canonicalization rules
- Add tests for statement normalization

---

## Drift Detection Architecture

### Component 1: Drift Scanner

**Purpose**: Continuously scan ledger for drift signals.

**Algorithm**:
```python
def scan_for_drift(blocks: List[Dict]) -> List[DriftSignal]:
    """
    Scan blocks for drift signals.
    
    Returns list of detected drift signals with:
    - drift_type: schema | hash_delta | metadata | statement
    - severity: LOW | MEDIUM | HIGH | CRITICAL
    - affected_blocks: List[int]
    - evidence: Dict (forensic artifacts)
    """
    signals = []
    
    # Scan for schema drift
    schema_versions = detect_schema_versions(blocks)
    if len(schema_versions) > 1:
        signals.append(DriftSignal(
            drift_type="schema",
            severity="MEDIUM",
            affected_blocks=get_blocks_with_schema_version(blocks, schema_versions[1]),
            evidence={"schema_versions": schema_versions},
        ))
    
    # Scan for hash-delta drift
    for block in blocks:
        replay_result = replay_block(block)
        if not replay_result.is_valid:
            # Classify mismatch
            if replay_result.r_t_match and replay_result.u_t_match and not replay_result.h_t_match:
                # H_t mismatch but R_t and U_t match → composite hash changed
                signals.append(DriftSignal(
                    drift_type="hash_delta",
                    severity="HIGH",
                    affected_blocks=[block["block_number"]],
                    evidence={
                        "stored_h_t": replay_result.stored_h_t,
                        "recomputed_h_t": replay_result.recomputed_h_t,
                        "r_t_match": True,
                        "u_t_match": True,
                    },
                ))
            elif not replay_result.r_t_match:
                # R_t mismatch → reasoning hash changed
                signals.append(DriftSignal(
                    drift_type="hash_delta",
                    severity="HIGH",
                    affected_blocks=[block["block_number"]],
                    evidence={
                        "stored_r_t": replay_result.stored_r_t,
                        "recomputed_r_t": replay_result.recomputed_r_t,
                        "canonical_proofs": block["canonical_proofs"],
                    },
                ))
    
    # Scan for metadata drift
    metadata_schemas = detect_metadata_schemas(blocks)
    if len(metadata_schemas) > 1:
        signals.append(DriftSignal(
            drift_type="metadata",
            severity="LOW",
            affected_blocks=get_blocks_with_metadata_schema(blocks, metadata_schemas[1]),
            evidence={"metadata_schemas": metadata_schemas},
        ))
    
    return signals
```

---

### Component 2: Drift Classifier

**Purpose**: Classify drift signals into known categories.

**Categories**:

| Category | Description | Severity | Auto-Remediation |
|----------|-------------|----------|------------------|
| **Benign Schema Evolution** | Backward-compatible schema changes | LOW | ✅ Yes (add compatibility layer) |
| **Breaking Schema Change** | Non-backward-compatible schema changes | MEDIUM | ⚠️ Partial (requires migration) |
| **Hash Algorithm Upgrade** | Intentional hash algorithm change (e.g., SHA-256 → SHA-3) | HIGH | ✅ Yes (dual-hash migration) |
| **Unintentional Hash Change** | Bug or code change that altered hash computation | HIGH | ❌ No (requires code fix) |
| **Data Corruption** | Database corruption or bit flips | CRITICAL | ❌ No (requires restore from backup) |
| **Malicious Tampering** | Intentional modification of sealed blocks | CRITICAL | ❌ No (forensic investigation) |

**Classification Algorithm**:
```python
def classify_drift(signal: DriftSignal, historical_data: Dict) -> DriftCategory:
    """
    Classify drift signal into category.
    
    Uses:
    - Drift type and severity
    - Historical drift patterns
    - Code change history (git log)
    - Schema migration history
    """
    if signal.drift_type == "schema":
        # Check if schema change is documented in migrations
        if is_documented_migration(signal.evidence["schema_versions"]):
            return DriftCategory.BENIGN_SCHEMA_EVOLUTION
        else:
            return DriftCategory.BREAKING_SCHEMA_CHANGE
    
    elif signal.drift_type == "hash_delta":
        # Check if hash change is intentional (e.g., PQ migration)
        if is_intentional_hash_upgrade(signal.affected_blocks):
            return DriftCategory.HASH_ALGORITHM_UPGRADE
        else:
            # Check git history for hash computation changes
            recent_commits = get_recent_commits_affecting_hash_code()
            if recent_commits:
                return DriftCategory.UNINTENTIONAL_HASH_CHANGE
            else:
                # No code changes → likely data corruption or tampering
                if has_corruption_signature(signal):
                    return DriftCategory.DATA_CORRUPTION
                else:
                    return DriftCategory.MALICIOUS_TAMPERING
    
    elif signal.drift_type == "metadata":
        return DriftCategory.BENIGN_SCHEMA_EVOLUTION
    
    else:
        return DriftCategory.UNKNOWN
```

---

### Component 3: Forensic Artifact Collector

**Purpose**: Capture detailed evidence for drift investigation.

**Artifacts Collected**:

1. **Block Snapshot**:
   - Full block data (all fields)
   - Stored attestation roots (R_t, U_t, H_t)
   - Canonical payloads (proofs, statements, UI events)
   - Attestation metadata

2. **Replay Trace**:
   - Recomputed attestation roots
   - Intermediate hash values
   - Canonicalization output
   - Merkle tree structure

3. **Code Context**:
   - Git commit SHA at time of block sealing
   - Recent commits affecting hash/attestation code
   - Schema migration history

4. **Environment Context**:
   - Database schema version
   - Python version
   - Dependency versions (hashlib, etc.)

**Artifact Format**:
```json
{
  "drift_signal_id": "drift_20251206_001",
  "detected_at": "2025-12-06T12:34:56Z",
  "drift_type": "hash_delta",
  "severity": "HIGH",
  "category": "UNINTENTIONAL_HASH_CHANGE",
  "affected_blocks": [1234, 1235, 1236],
  "evidence": {
    "block_snapshot": {
      "block_id": 1234,
      "block_number": 1234,
      "stored_r_t": "abc...",
      "stored_u_t": "def...",
      "stored_h_t": "ghi...",
      "canonical_proofs": [...],
      "attestation_metadata": {...}
    },
    "replay_trace": {
      "recomputed_r_t": "xyz...",
      "recomputed_u_t": "def...",
      "recomputed_h_t": "uvw...",
      "intermediate_hashes": {
        "proof_leaf_0": "...",
        "proof_leaf_1": "...",
        "merkle_node_0": "..."
      }
    },
    "code_context": {
      "git_sha_at_sealing": "a1b2c3d4",
      "recent_commits": [
        {
          "sha": "e5f6g7h8",
          "message": "Refactor hash computation",
          "files_changed": ["backend/crypto/hashing.py"]
        }
      ]
    },
    "environment_context": {
      "db_schema_version": "018",
      "python_version": "3.11.0",
      "hashlib_version": "..."
    }
  },
  "remediation_guidance": "Code change detected in backend/crypto/hashing.py (commit e5f6g7h8). Review changes and add hash_version field to metadata."
}
```

---

### Component 4: Drift Dashboard

**Purpose**: Visualize drift signals and trends over time.

**Metrics Tracked**:
- Drift signal count (by type, severity, category)
- Affected block count
- Drift detection rate (signals per 1000 blocks)
- Mean time to remediation (MTTR)
- Drift recurrence rate

**Dashboard Views**:

1. **Real-Time Drift Monitor**:
   - Live feed of new drift signals
   - Color-coded by severity (green/yellow/red)
   - Click to view forensic artifacts

2. **Drift Trend Analysis**:
   - Time series of drift signal count
   - Breakdown by drift type
   - Correlation with code deployments

3. **Affected Blocks Map**:
   - Heatmap of blocks with drift
   - Identify drift clusters (consecutive blocks)
   - Epoch-level aggregation

4. **Remediation Tracker**:
   - Open drift signals (unresolved)
   - Remediation status (in progress, resolved, wontfix)
   - Assigned owner

---

## Drift Severity Levels

### CRITICAL (Severity 0)

**Characteristics**:
- Chain integrity compromised
- Potential malicious tampering
- Data corruption detected

**Response**:
- **Immediate**: Halt all block sealing
- **Alert**: Page on-call engineer
- **Action**: Forensic investigation required
- **Timeline**: Resolve within 1 hour

**Examples**:
- Multiple blocks deleted
- Fork detected (multiple blocks with same prev_hash)
- Cycle in prev_hash chain

---

### HIGH (Severity 1)

**Characteristics**:
- Replay determinism broken
- Hash computation changed
- Schema incompatibility

**Response**:
- **Immediate**: Block code merges
- **Alert**: Notify #ledger-alerts
- **Action**: Code review and fix required
- **Timeline**: Resolve within 24 hours

**Examples**:
- R_t or U_t mismatch
- Domain separation tag changed
- Hash algorithm changed without migration

---

### MEDIUM (Severity 2)

**Characteristics**:
- Schema evolution detected
- Backward compatibility at risk
- Migration required

**Response**:
- **Immediate**: Log warning
- **Alert**: Create GitHub issue
- **Action**: Plan migration
- **Timeline**: Resolve within 1 week

**Examples**:
- canonical_proofs format changed
- New required fields added
- Field renamed

---

### LOW (Severity 3)

**Characteristics**:
- Metadata inconsistency
- Benign schema evolution
- No impact on replay

**Response**:
- **Immediate**: Log info
- **Alert**: None
- **Action**: Document change
- **Timeline**: Resolve when convenient

**Examples**:
- attestation_metadata field added
- Metadata field renamed (but not used in hash computation)
- Comment or documentation field changed

---

## Implementation

### Module: `backend/ledger/drift/`

**`backend/ledger/drift/scanner.py`**:
```python
def scan_for_drift(blocks: List[Dict]) -> List[DriftSignal]:
    """Scan blocks for drift signals."""
    ...

def detect_schema_versions(blocks: List[Dict]) -> List[str]:
    """Detect distinct schema versions in blocks."""
    ...

def detect_metadata_schemas(blocks: List[Dict]) -> List[Dict]:
    """Detect distinct metadata schemas."""
    ...
```

**`backend/ledger/drift/classifier.py`**:
```python
def classify_drift(signal: DriftSignal, context: Dict) -> DriftCategory:
    """Classify drift signal into category."""
    ...

def is_documented_migration(schema_versions: List[str]) -> bool:
    """Check if schema change is documented in migrations."""
    ...

def is_intentional_hash_upgrade(affected_blocks: List[int]) -> bool:
    """Check if hash change is part of planned upgrade."""
    ...
```

**`backend/ledger/drift/forensics.py`**:
```python
def collect_artifacts(signal: DriftSignal) -> Dict:
    """Collect forensic artifacts for drift signal."""
    ...

def capture_block_snapshot(block_id: int) -> Dict:
    """Capture full block snapshot."""
    ...

def capture_replay_trace(block: Dict) -> Dict:
    """Capture detailed replay trace."""
    ...

def capture_code_context() -> Dict:
    """Capture git and code context."""
    ...
```

**`backend/ledger/drift/dashboard.py`**:
```python
class DriftDashboard:
    def get_metrics(self) -> Dict:
        """Get drift metrics."""
        ...
    
    def get_active_signals(self) -> List[DriftSignal]:
        """Get unresolved drift signals."""
        ...
    
    def get_trend_data(self, time_range: str) -> Dict:
        """Get drift trend data."""
        ...
```

---

## Integration with Replay Governance

### Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                  Replay Verification                        │
│                                                             │
│  1. Fetch blocks                                           │
│  2. Replay each block                                      │
│  3. Detect failures (H_t mismatch)                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
                    ┌───────────────┐
                    │  Failure?     │
                    └───────────────┘
                      ↙           ↘
                   YES             NO
                    ↓               ↓
            ┌──────────────┐  ┌──────────────┐
            │ Drift Radar  │  │ Success      │
            │ Activated    │  │ Continue     │
            └──────────────┘  └──────────────┘
                    ↓
            ┌──────────────┐
            │ Scan for     │
            │ Drift Signals│
            └──────────────┘
                    ↓
            ┌──────────────┐
            │ Classify     │
            │ Drift        │
            └──────────────┘
                    ↓
            ┌──────────────┐
            │ Collect      │
            │ Forensics    │
            └──────────────┘
                    ↓
            ┌──────────────┐
            │ Generate     │
            │ Report       │
            └──────────────┘
                    ↓
            ┌──────────────┐
            │ Alert Team   │
            │ Block Merge  │
            └──────────────┘
```

---

## Drift Radar CLI

```bash
# Scan for drift
mathledger-drift scan --system-id <uuid> --output drift_report.json

# Classify drift signals
mathledger-drift classify --input drift_report.json

# Collect forensics for specific signal
mathledger-drift forensics --signal-id drift_20251206_001

# View dashboard
mathledger-drift dashboard --port 8080

# Export metrics
mathledger-drift metrics --format prometheus > drift_metrics.prom
```

---

## Future Enhancements

1. **Machine Learning Drift Prediction**:
   - Train model on historical drift patterns
   - Predict likelihood of drift before it occurs
   - Proactive alerts

2. **Automated Remediation**:
   - Auto-fix benign schema evolution
   - Auto-migrate blocks with documented schema changes
   - Self-healing ledger

3. **Cross-System Drift Detection**:
   - Compare drift patterns across multiple systems
   - Detect systemic issues (e.g., library upgrade causing drift)

4. **Drift Simulation**:
   - Simulate code changes and predict drift impact
   - Pre-deployment drift testing

---

**Status**: Design complete, implementation pending

**Next**: Cross-Epoch Chain Verification with PQ Integration
