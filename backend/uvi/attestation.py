"""
User Verified Input Attestation

Integrates UVI events into the dual-root attestation framework.

INTEGRATION FLOW:
1. UVI events are recorded via events.py
2. Events are converted to attestation leaves
3. Leaves flow into the UI Root (U_t) Merkle tree
4. U_t combines with Reasoning Root (R_t) for composite H_t

SCHEMA VERSION: 1.0.0 (Stub)
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence

from backend.uvi.events import UVIEvent, get_all_events


# =============================================================================
# Constants
# =============================================================================

SCHEMA_VERSION = "1.0.0"
DOMAIN_SEPARATOR = b"MathLedger:UVI:v1:"


# =============================================================================
# Digest Computation
# =============================================================================


def compute_uvi_digest(event: UVIEvent) -> str:
    """
    Compute attestation digest for a UVI event.

    Uses domain separation to prevent cross-protocol attacks.

    Args:
        event: The UVI event to digest

    Returns:
        SHA-256 hex digest with domain separation
    """
    payload = event.canonical_json().encode("utf-8")
    return hashlib.sha256(DOMAIN_SEPARATOR + payload).hexdigest()


def compute_batch_digest(events: Sequence[UVIEvent]) -> str:
    """
    Compute aggregate digest over multiple UVI events.

    Events are sorted by event_id for deterministic ordering.

    Args:
        events: Sequence of UVI events

    Returns:
        SHA-256 hex digest of concatenated individual digests
    """
    if not events:
        # Empty batch has a defined sentinel digest
        return hashlib.sha256(DOMAIN_SEPARATOR + b"EMPTY_BATCH").hexdigest()

    # Sort by event_id for determinism
    sorted_events = sorted(events, key=lambda e: e.event_id)

    # Concatenate individual digests
    combined = b""
    for event in sorted_events:
        combined += bytes.fromhex(compute_uvi_digest(event))

    return hashlib.sha256(DOMAIN_SEPARATOR + combined).hexdigest()


# =============================================================================
# Attestation Leaf Generation
# =============================================================================


def build_uvi_attestation_leaf(event: UVIEvent) -> Dict[str, Any]:
    """
    Build an attestation leaf for a UVI event.

    The leaf structure is compatible with the UI Root (U_t) Merkle tree.

    Args:
        event: The UVI event

    Returns:
        Dictionary representing the attestation leaf
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "leaf_type": "uvi_event",
        "event_id": event.event_id,
        "event_type": event.event_type.value,
        "target_hash": event.target_hash,
        "target_type": event.target_type,
        "user_id_hash": hashlib.sha256(
            (DOMAIN_SEPARATOR + event.user_id.encode("utf-8"))
        ).hexdigest()[:16],  # Anonymized user reference
        "timestamp": event.timestamp,
        "digest": compute_uvi_digest(event),
    }


def build_uvi_summary_leaf(events: Sequence[UVIEvent]) -> Dict[str, Any]:
    """
    Build a summary attestation leaf for a batch of UVI events.

    Used when aggregating multiple events into a single attestation.

    Args:
        events: Sequence of UVI events

    Returns:
        Dictionary representing the summary attestation leaf
    """
    # Count by event type
    type_counts = {
        "CONFIRMATION": 0,
        "CORRECTION": 0,
        "FLAG": 0,
    }
    for event in events:
        type_counts[event.event_type.value] += 1

    # Collect unique targets
    unique_targets = set(e.target_hash for e in events)

    return {
        "schema_version": SCHEMA_VERSION,
        "leaf_type": "uvi_summary",
        "event_count": len(events),
        "type_counts": type_counts,
        "unique_target_count": len(unique_targets),
        "batch_digest": compute_batch_digest(events),
        "mode": "SHADOW",  # UVI is observational in Phase I
    }


# =============================================================================
# Integration with Dual-Root Attestation
# =============================================================================


def get_uvi_leaves_for_attestation() -> List[Dict[str, Any]]:
    """
    Get all UVI events as attestation leaves.

    These leaves should be included in the UI Root (U_t) computation.

    Returns:
        List of attestation leaf dictionaries
    """
    events = get_all_events()
    return [build_uvi_attestation_leaf(e) for e in events]


def get_uvi_summary_for_attestation() -> Dict[str, Any]:
    """
    Get a summary of all UVI events for attestation.

    Returns:
        Summary attestation leaf dictionary
    """
    events = get_all_events()
    return build_uvi_summary_leaf(events)


# =============================================================================
# Verification (Stub)
# =============================================================================


def verify_uvi_leaf(leaf: Dict[str, Any], event: UVIEvent) -> bool:
    """
    Verify that an attestation leaf matches the original event.

    Args:
        leaf: The attestation leaf to verify
        event: The original UVI event

    Returns:
        True if the leaf is valid, False otherwise
    """
    if leaf.get("schema_version") != SCHEMA_VERSION:
        return False

    if leaf.get("leaf_type") != "uvi_event":
        return False

    if leaf.get("event_id") != event.event_id:
        return False

    if leaf.get("digest") != compute_uvi_digest(event):
        return False

    return True
