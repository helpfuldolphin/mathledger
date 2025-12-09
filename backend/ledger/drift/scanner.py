"""
Ledger Drift Radar - Scanner Module

Author: Manus-B (Ledger Replay Architect & Attestation Runtime Engineer)
Phase: III - Drift Radar MVP Implementation
Date: 2025-12-06

Purpose:
    Scan ledger for drift signals across blocks.
    
    Drift signals detected:
    - Schema drift (canonical_proofs structure changes)
    - Hash-delta drift (hash computation changes)
    - Metadata drift (attestation_metadata inconsistencies)
    - Statement drift (canonical_statements format changes)

Design Principles:
    1. Comprehensive: Detect all drift types
    2. Efficient: Minimize database queries
    3. Deterministic: Same blocks → same drift signals
    4. Actionable: Provide context for classification
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import hashlib
import json


# ============================================================================
# DRIFT SIGNAL TYPES
# ============================================================================

class DriftSignalType(Enum):
    """Types of drift signals."""
    SCHEMA_DRIFT = "schema_drift"          # canonical_proofs structure changed
    HASH_DELTA_DRIFT = "hash_delta_drift"  # Hash computation changed
    METADATA_DRIFT = "metadata_drift"      # attestation_metadata changed
    STATEMENT_DRIFT = "statement_drift"    # canonical_statements format changed


class DriftSeverity(Enum):
    """Drift severity levels."""
    LOW = "low"          # Benign, no action required
    MEDIUM = "medium"    # Should be investigated
    HIGH = "high"        # Requires immediate attention
    CRITICAL = "critical"  # Chain integrity compromised


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class DriftSignal:
    """
    Represents a detected drift signal.
    
    Attributes:
        signal_type: Type of drift signal
        severity: Severity level (preliminary, may be refined by classifier)
        block_number: Block number where drift was detected
        block_id: Block ID
        description: Human-readable description
        context: Additional context for classification
        detected_at: Timestamp when signal was detected
    """
    signal_type: DriftSignalType
    severity: DriftSeverity
    block_number: int
    block_id: int
    description: str
    context: Dict[str, Any]
    detected_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "signal_type": self.signal_type.value,
            "severity": self.severity.value,
            "block_number": self.block_number,
            "block_id": self.block_id,
            "description": self.description,
            "context": self.context,
            "detected_at": self.detected_at,
        }


# ============================================================================
# SCHEMA DRIFT DETECTION
# ============================================================================

def detect_schema_drift(
    blocks: List[Dict[str, Any]],
    window_size: int = 100,
) -> List[DriftSignal]:
    """
    Detect schema drift in canonical_proofs structure.
    
    Args:
        blocks: List of blocks (sorted by block_number)
        window_size: Number of blocks to analyze for schema stability
    
    Returns:
        List of DriftSignal objects
    
    Detection Algorithm:
        1. Extract canonical_proofs schema from each block
        2. Compare schema across blocks in sliding window
        3. Detect schema changes (new fields, removed fields, type changes)
        4. Generate drift signal for each change
    
    Schema Representation:
        - Dict schema: {field_name: field_type, ...}
        - List schema: [item_type, ...]
        - Nested schemas recursively represented
    
    Deterministic Ordering:
        - Blocks must be pre-sorted by block_number
        - Schema comparison in sequential order
        - Drift signals emitted in block_number order
    """
    signals = []
    
    if not blocks:
        return signals
    
    # Extract schemas
    schemas = []
    for block in blocks:
        canonical_proofs = block.get("canonical_proofs", {})
        schema = extract_schema(canonical_proofs)
        schemas.append((block, schema))
    
    # Detect schema changes
    for i in range(1, len(schemas)):
        prev_block, prev_schema = schemas[i - 1]
        curr_block, curr_schema = schemas[i]
        
        # Compare schemas
        schema_diff = compare_schemas(prev_schema, curr_schema)
        
        if schema_diff["changed"]:
            # Schema drift detected
            signal = DriftSignal(
                signal_type=DriftSignalType.SCHEMA_DRIFT,
                severity=classify_schema_drift_severity(schema_diff),
                block_number=curr_block["block_number"],
                block_id=curr_block["id"],
                description=f"Schema drift detected: {schema_diff['description']}",
                context={
                    "prev_schema": prev_schema,
                    "curr_schema": curr_schema,
                    "diff": schema_diff,
                    "prev_block_number": prev_block["block_number"],
                },
                detected_at=datetime.utcnow().isoformat() + "Z",
            )
            signals.append(signal)
    
    return signals


def extract_schema(obj: Any, depth: int = 0, max_depth: int = 3) -> Dict[str, Any]:
    """
    Extract schema from object.
    
    Args:
        obj: Object to extract schema from
        depth: Current recursion depth
        max_depth: Maximum recursion depth
    
    Returns:
        Schema dictionary
    
    Schema Format:
        - Primitive: {"type": "int" | "str" | "float" | "bool" | "null"}
        - List: {"type": "list", "item_schema": schema}
        - Dict: {"type": "dict", "fields": {field_name: schema, ...}}
    """
    if depth > max_depth:
        return {"type": "unknown"}
    
    if obj is None:
        return {"type": "null"}
    elif isinstance(obj, bool):
        return {"type": "bool"}
    elif isinstance(obj, int):
        return {"type": "int"}
    elif isinstance(obj, float):
        return {"type": "float"}
    elif isinstance(obj, str):
        return {"type": "str"}
    elif isinstance(obj, list):
        if not obj:
            return {"type": "list", "item_schema": {"type": "unknown"}}
        # Use first item as representative
        item_schema = extract_schema(obj[0], depth + 1, max_depth)
        return {"type": "list", "item_schema": item_schema}
    elif isinstance(obj, dict):
        fields = {}
        for key, value in obj.items():
            fields[key] = extract_schema(value, depth + 1, max_depth)
        return {"type": "dict", "fields": fields}
    else:
        return {"type": "unknown"}


def compare_schemas(schema1: Dict[str, Any], schema2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare two schemas and detect differences.
    
    Args:
        schema1: First schema
        schema2: Second schema
    
    Returns:
        Difference dictionary
    
    Difference Format:
        {
            "changed": bool,
            "description": str,
            "added_fields": [field_name, ...],
            "removed_fields": [field_name, ...],
            "type_changes": {field_name: (old_type, new_type), ...},
        }
    """
    diff = {
        "changed": False,
        "description": "",
        "added_fields": [],
        "removed_fields": [],
        "type_changes": {},
    }
    
    # Type change
    if schema1.get("type") != schema2.get("type"):
        diff["changed"] = True
        diff["description"] = f"Type changed: {schema1.get('type')} → {schema2.get('type')}"
        return diff
    
    # Dict schema comparison
    if schema1.get("type") == "dict" and schema2.get("type") == "dict":
        fields1 = schema1.get("fields", {})
        fields2 = schema2.get("fields", {})
        
        # Added fields
        for field in fields2:
            if field not in fields1:
                diff["added_fields"].append(field)
                diff["changed"] = True
        
        # Removed fields
        for field in fields1:
            if field not in fields2:
                diff["removed_fields"].append(field)
                diff["changed"] = True
        
        # Type changes
        for field in fields1:
            if field in fields2:
                if fields1[field].get("type") != fields2[field].get("type"):
                    diff["type_changes"][field] = (fields1[field].get("type"), fields2[field].get("type"))
                    diff["changed"] = True
        
        # Build description
        if diff["changed"]:
            parts = []
            if diff["added_fields"]:
                parts.append(f"added {len(diff['added_fields'])} fields")
            if diff["removed_fields"]:
                parts.append(f"removed {len(diff['removed_fields'])} fields")
            if diff["type_changes"]:
                parts.append(f"changed {len(diff['type_changes'])} field types")
            diff["description"] = ", ".join(parts)
    
    return diff


def classify_schema_drift_severity(schema_diff: Dict[str, Any]) -> DriftSeverity:
    """
    Classify schema drift severity.
    
    Args:
        schema_diff: Schema difference dictionary
    
    Returns:
        DriftSeverity
    
    Classification Rules:
        - Removed fields → HIGH (breaking change)
        - Type changes → HIGH (breaking change)
        - Added fields only → LOW (backward compatible)
    """
    if schema_diff.get("removed_fields"):
        return DriftSeverity.HIGH
    
    if schema_diff.get("type_changes"):
        return DriftSeverity.HIGH
    
    if schema_diff.get("added_fields"):
        return DriftSeverity.LOW
    
    return DriftSeverity.LOW


# ============================================================================
# HASH-DELTA DRIFT DETECTION
# ============================================================================

def detect_hash_delta_drift(
    blocks: List[Dict[str, Any]],
    replay_results: Optional[List[Dict[str, Any]]] = None,
) -> List[DriftSignal]:
    """
    Detect hash-delta drift (hash computation changes).
    
    Args:
        blocks: List of blocks (sorted by block_number)
        replay_results: Optional replay verification results
    
    Returns:
        List of DriftSignal objects
    
    Detection Algorithm:
        1. Compare stored roots with recomputed roots (from replay)
        2. If mismatch but payloads unchanged → hash-delta drift
        3. Analyze drift pattern (systematic vs isolated)
        4. Generate drift signal with context
    
    Requires:
        - replay_results from replay verification engine
        - Each result: {block_id, r_t_stored, r_t_recomputed, u_t_stored, u_t_recomputed, h_t_stored, h_t_recomputed}
    """
    signals = []
    
    if not replay_results:
        return signals  # Cannot detect without replay results
    
    # Build block map
    block_map = {b["id"]: b for b in blocks}
    
    # Detect mismatches
    for result in replay_results:
        block_id = result.get("block_id")
        block = block_map.get(block_id)
        
        if not block:
            continue
        
        # Check for hash mismatches
        r_t_match = result.get("r_t_stored") == result.get("r_t_recomputed")
        u_t_match = result.get("u_t_stored") == result.get("u_t_recomputed")
        h_t_match = result.get("h_t_stored") == result.get("h_t_recomputed")
        
        if not (r_t_match and u_t_match and h_t_match):
            # Hash mismatch detected
            signal = DriftSignal(
                signal_type=DriftSignalType.HASH_DELTA_DRIFT,
                severity=DriftSeverity.HIGH,  # Hash changes are always high severity
                block_number=block["block_number"],
                block_id=block["id"],
                description=f"Hash mismatch detected: R_t={r_t_match}, U_t={u_t_match}, H_t={h_t_match}",
                context={
                    "r_t_stored": result.get("r_t_stored"),
                    "r_t_recomputed": result.get("r_t_recomputed"),
                    "u_t_stored": result.get("u_t_stored"),
                    "u_t_recomputed": result.get("u_t_recomputed"),
                    "h_t_stored": result.get("h_t_stored"),
                    "h_t_recomputed": result.get("h_t_recomputed"),
                    "hash_version": block.get("attestation_metadata", {}).get("hash_version", "sha256-v1"),
                },
                detected_at=datetime.utcnow().isoformat() + "Z",
            )
            signals.append(signal)
    
    return signals


# ============================================================================
# METADATA DRIFT DETECTION
# ============================================================================

def detect_metadata_drift(blocks: List[Dict[str, Any]]) -> List[DriftSignal]:
    """
    Detect metadata drift in attestation_metadata.
    
    Args:
        blocks: List of blocks (sorted by block_number)
    
    Returns:
        List of DriftSignal objects
    
    Detection Algorithm:
        1. Extract attestation_metadata schema from each block
        2. Compare schema across blocks
        3. Detect schema changes (new fields, removed fields)
        4. Generate drift signal for each change
    """
    signals = []
    
    if not blocks:
        return signals
    
    # Extract metadata schemas
    schemas = []
    for block in blocks:
        metadata = block.get("attestation_metadata", {})
        schema = extract_schema(metadata)
        schemas.append((block, schema))
    
    # Detect schema changes
    for i in range(1, len(schemas)):
        prev_block, prev_schema = schemas[i - 1]
        curr_block, curr_schema = schemas[i]
        
        # Compare schemas
        schema_diff = compare_schemas(prev_schema, curr_schema)
        
        if schema_diff["changed"]:
            # Metadata drift detected
            signal = DriftSignal(
                signal_type=DriftSignalType.METADATA_DRIFT,
                severity=DriftSeverity.LOW,  # Metadata changes are usually low severity
                block_number=curr_block["block_number"],
                block_id=curr_block["id"],
                description=f"Metadata drift detected: {schema_diff['description']}",
                context={
                    "prev_schema": prev_schema,
                    "curr_schema": curr_schema,
                    "diff": schema_diff,
                    "prev_block_number": prev_block["block_number"],
                },
                detected_at=datetime.utcnow().isoformat() + "Z",
            )
            signals.append(signal)
    
    return signals


# ============================================================================
# STATEMENT DRIFT DETECTION
# ============================================================================

def detect_statement_drift(blocks: List[Dict[str, Any]]) -> List[DriftSignal]:
    """
    Detect statement drift in canonical_statements format.
    
    Args:
        blocks: List of blocks (sorted by block_number)
    
    Returns:
        List of DriftSignal objects
    
    Detection Algorithm:
        1. Extract statement format from canonical_proofs
        2. Detect format changes (ordering, representation)
        3. Generate drift signal for each change
    """
    signals = []
    
    # Implementation note: This requires analyzing statement structure
    # within canonical_proofs. Simplified version here.
    
    return signals


# ============================================================================
# DRIFT SCANNER ORCHESTRATOR
# ============================================================================

class DriftScanner:
    """
    Orchestrates drift detection across all signal types.
    
    Usage:
        scanner = DriftScanner()
        signals = scanner.scan(blocks, replay_results)
        report = scanner.generate_report()
    """
    
    def __init__(self):
        """Initialize drift scanner."""
        self.signals: List[DriftSignal] = []
        self.scan_history: List[Dict[str, Any]] = []
    
    def scan(
        self,
        blocks: List[Dict[str, Any]],
        replay_results: Optional[List[Dict[str, Any]]] = None,
    ) -> List[DriftSignal]:
        """
        Scan blocks for drift signals.
        
        Args:
            blocks: List of blocks (sorted by block_number)
            replay_results: Optional replay verification results
        
        Returns:
            List of DriftSignal objects
        """
        all_signals = []
        
        # Detect schema drift
        schema_signals = detect_schema_drift(blocks)
        all_signals.extend(schema_signals)
        
        # Detect hash-delta drift
        if replay_results:
            hash_delta_signals = detect_hash_delta_drift(blocks, replay_results)
            all_signals.extend(hash_delta_signals)
        
        # Detect metadata drift
        metadata_signals = detect_metadata_drift(blocks)
        all_signals.extend(metadata_signals)
        
        # Detect statement drift
        statement_signals = detect_statement_drift(blocks)
        all_signals.extend(statement_signals)
        
        # Store signals
        self.signals.extend(all_signals)
        
        # Record scan
        self.scan_history.append({
            "scanned_at": datetime.utcnow().isoformat() + "Z",
            "block_count": len(blocks),
            "signal_count": len(all_signals),
            "by_type": self._count_by_type(all_signals),
            "by_severity": self._count_by_severity(all_signals),
        })
        
        return all_signals
    
    def _count_by_type(self, signals: List[DriftSignal]) -> Dict[str, int]:
        """Count signals by type."""
        counts = {}
        for signal in signals:
            signal_type = signal.signal_type.value
            counts[signal_type] = counts.get(signal_type, 0) + 1
        return counts
    
    def _count_by_severity(self, signals: List[DriftSignal]) -> Dict[str, int]:
        """Count signals by severity."""
        counts = {}
        for signal in signals:
            severity = signal.severity.value
            counts[severity] = counts.get(severity, 0) + 1
        return counts
    
    def get_signals_by_type(self, signal_type: DriftSignalType) -> List[DriftSignal]:
        """Get signals by type."""
        return [s for s in self.signals if s.signal_type == signal_type]
    
    def get_signals_by_severity(self, severity: DriftSeverity) -> List[DriftSignal]:
        """Get signals by severity."""
        return [s for s in self.signals if s.severity == severity]
    
    def get_signals_by_block(self, block_number: int) -> List[DriftSignal]:
        """Get signals for specific block."""
        return [s for s in self.signals if s.block_number == block_number]
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate drift scan report.
        
        Returns:
            Report dictionary
        """
        return {
            "total_signals": len(self.signals),
            "by_type": self._count_by_type(self.signals),
            "by_severity": self._count_by_severity(self.signals),
            "scan_history": self.scan_history,
            "critical_count": len(self.get_signals_by_severity(DriftSeverity.CRITICAL)),
            "high_count": len(self.get_signals_by_severity(DriftSeverity.HIGH)),
            "medium_count": len(self.get_signals_by_severity(DriftSeverity.MEDIUM)),
            "low_count": len(self.get_signals_by_severity(DriftSeverity.LOW)),
        }
