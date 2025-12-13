"""
User Verified Input Events

Defines the event types and recording functions for user epistemic actions.

DESIGN PRINCIPLES:
1. Events are immutable once recorded
2. All events include timestamps and user identifiers
3. Events are JSON-serializable for attestation
4. No enforcement - purely observational (Shadow Mode)

EVENT TYPES:
- CONFIRMATION: User confirms a theorem/statement is correct
- CORRECTION: User marks a proof as suspicious/incorrect
- FLAG: User flags content for review without making a judgment

INTEGRATION:
- Events flow into the UI Root (U_t) Merkle tree
- Combined with Reasoning Root (R_t) for composite attestation (H_t)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4


class UVIEventType(str, Enum):
    """Types of user verification events."""

    CONFIRMATION = "CONFIRMATION"  # User confirms correctness
    CORRECTION = "CORRECTION"  # User marks as suspicious/incorrect
    FLAG = "FLAG"  # User flags for review


@dataclass(frozen=True)
class UVIEvent:
    """
    Immutable record of a user verification event.

    Attributes:
        event_id: Unique identifier for this event
        event_type: Type of verification action
        target_hash: SHA-256 hash of the target statement/proof
        target_type: Type of target ("statement", "proof", "derivation")
        user_id: Identifier of the user (anonymized if needed)
        timestamp: UTC timestamp of the event
        rationale: Optional user-provided explanation
        metadata: Additional context (non-identifying)
    """

    event_id: str
    event_type: UVIEventType
    target_hash: str
    target_type: str
    user_id: str
    timestamp: str
    rationale: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "target_hash": self.target_hash,
            "target_type": self.target_type,
            "user_id": self.user_id,
            "timestamp": self.timestamp,
            "rationale": self.rationale,
            "metadata": self.metadata,
        }

    def canonical_json(self) -> str:
        """
        Produce canonical JSON representation for hashing.

        Uses RFC 8785 principles:
        - Sorted keys
        - No whitespace
        - Deterministic ordering
        """
        data = self.to_dict()
        # Remove None values for canonical form
        data = {k: v for k, v in data.items() if v is not None}
        return json.dumps(data, sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        """Compute SHA-256 digest of canonical representation."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


# =============================================================================
# Event Storage (In-Memory for Stub)
# =============================================================================

# In production, this would be database-backed
_event_store: List[UVIEvent] = []


def _generate_event_id() -> str:
    """Generate a unique event ID."""
    return f"uvi_{uuid4().hex[:16]}"


def _current_timestamp() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


# =============================================================================
# Public API
# =============================================================================


def record_confirmation(
    target_hash: str,
    target_type: str,
    user_id: str,
    rationale: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> UVIEvent:
    """
    Record a user confirmation event.

    The user asserts that the target statement/proof is correct.

    Args:
        target_hash: SHA-256 hash of the target
        target_type: Type of target ("statement", "proof", "derivation")
        user_id: User identifier
        rationale: Optional explanation
        metadata: Additional context

    Returns:
        The recorded UVIEvent
    """
    event = UVIEvent(
        event_id=_generate_event_id(),
        event_type=UVIEventType.CONFIRMATION,
        target_hash=target_hash,
        target_type=target_type,
        user_id=user_id,
        timestamp=_current_timestamp(),
        rationale=rationale,
        metadata=metadata or {},
    )
    _event_store.append(event)
    return event


def record_correction(
    target_hash: str,
    target_type: str,
    user_id: str,
    rationale: str,  # Required for corrections
    metadata: Optional[Dict[str, Any]] = None,
) -> UVIEvent:
    """
    Record a user correction event.

    The user marks the target as suspicious or incorrect.

    Args:
        target_hash: SHA-256 hash of the target
        target_type: Type of target ("statement", "proof", "derivation")
        user_id: User identifier
        rationale: Explanation of the issue (required)
        metadata: Additional context

    Returns:
        The recorded UVIEvent
    """
    event = UVIEvent(
        event_id=_generate_event_id(),
        event_type=UVIEventType.CORRECTION,
        target_hash=target_hash,
        target_type=target_type,
        user_id=user_id,
        timestamp=_current_timestamp(),
        rationale=rationale,
        metadata=metadata or {},
    )
    _event_store.append(event)
    return event


def record_flag(
    target_hash: str,
    target_type: str,
    user_id: str,
    rationale: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> UVIEvent:
    """
    Record a user flag event.

    The user flags the target for review without asserting correctness.

    Args:
        target_hash: SHA-256 hash of the target
        target_type: Type of target ("statement", "proof", "derivation")
        user_id: User identifier
        rationale: Optional explanation
        metadata: Additional context

    Returns:
        The recorded UVIEvent
    """
    event = UVIEvent(
        event_id=_generate_event_id(),
        event_type=UVIEventType.FLAG,
        target_hash=target_hash,
        target_type=target_type,
        user_id=user_id,
        timestamp=_current_timestamp(),
        rationale=rationale,
        metadata=metadata or {},
    )
    _event_store.append(event)
    return event


def get_events_for_target(target_hash: str) -> List[UVIEvent]:
    """Get all UVI events for a specific target."""
    return [e for e in _event_store if e.target_hash == target_hash]


def get_all_events() -> List[UVIEvent]:
    """Get all recorded UVI events."""
    return list(_event_store)


def clear_events() -> None:
    """Clear all events (for testing only)."""
    _event_store.clear()
