"""
RFL Event Verification & Filtering Module
==========================================

Implements dual-attested event verification and filtering for RFL.

Ensures that RFL policy updates consume only events that have been
verified through dual attestation (reasoning + UI roots).

Invariants:
1. RFL must never read unverifiable events
2. RFL must never read unattested events
3. Events must pass integrity checks before policy updates

Usage:
    from rfl.event_verification import (
        AttestedEvent,
        EventVerifier,
        verify_dual_attestation,
        filter_attested_events,
    )

    # Verify single event
    verifier = EventVerifier()
    is_valid = verifier.verify_event(attested_event)

    # Filter event stream
    valid_events = filter_attested_events(event_stream, verifier)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum


class VerificationStatus(Enum):
    """Status of event verification."""
    VALID = "valid"
    INVALID_REASONING_ROOT = "invalid_reasoning_root"
    INVALID_UI_ROOT = "invalid_ui_root"
    INVALID_COMPOSITE_ROOT = "invalid_composite_root"
    MISSING_ATTESTATION = "missing_attestation"
    MALFORMED_EVENT = "malformed_event"


@dataclass(frozen=True)
class AttestedEvent:
    """
    Dual-attested event for RFL consumption.
    
    Represents a verified event with both reasoning and UI attestation.
    
    Attributes:
        reasoning_root: SHA256 hash of reasoning event stream (R_t)
        ui_root: SHA256 hash of UI event stream (U_t)
        composite_root: SHA256(R_t || U_t) = H_t
        reasoning_event_count: Number of reasoning events
        ui_event_count: Number of UI events
        metadata: Additional context (verified_count, abstention_rate, etc.)
        statement_hash: Hash of the statement that triggered this event
    """
    reasoning_root: str
    ui_root: str
    composite_root: str
    reasoning_event_count: int
    ui_event_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    statement_hash: str = ""
    
    def __post_init__(self):
        """Validate attestation structure."""
        # Validate hex digest format
        for root_name, root_value in [
            ("reasoning_root", self.reasoning_root),
            ("ui_root", self.ui_root),
            ("composite_root", self.composite_root),
        ]:
            if len(root_value) != 64:
                raise ValueError(f"{root_name} must be 64-character hex digest, got {len(root_value)}")
            try:
                int(root_value, 16)
            except ValueError:
                raise ValueError(f"{root_name} must be valid hex string: {root_value}")
        
        # Validate event counts
        if self.reasoning_event_count < 0:
            raise ValueError(f"reasoning_event_count must be >= 0, got {self.reasoning_event_count}")
        if self.ui_event_count < 0:
            raise ValueError(f"ui_event_count must be >= 0, got {self.ui_event_count}")
    
    def verify_composite_root(self) -> bool:
        """
        Verify that composite_root = SHA256(reasoning_root || ui_root).
        
        Returns:
            True if composite root is valid
        """
        expected = compute_composite_root(self.reasoning_root, self.ui_root)
        return expected == self.composite_root
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "reasoning_root": self.reasoning_root,
            "ui_root": self.ui_root,
            "composite_root": self.composite_root,
            "reasoning_event_count": self.reasoning_event_count,
            "ui_event_count": self.ui_event_count,
            "metadata": self.metadata,
            "statement_hash": self.statement_hash,
            "verified": self.verify_composite_root(),
        }


def compute_composite_root(reasoning_root: str, ui_root: str) -> str:
    """
    Compute composite root: H_t = SHA256(R_t || U_t).
    
    Args:
        reasoning_root: R_t (64-character hex digest)
        ui_root: U_t (64-character hex digest)
    
    Returns:
        H_t (64-character hex digest)
    
    Raises:
        ValueError: If roots are not valid hex digests
    """
    if len(reasoning_root) != 64 or len(ui_root) != 64:
        raise ValueError("Reasoning and UI roots must be 64-character hex digests")
    
    # Validate hex format
    try:
        int(reasoning_root, 16)
        int(ui_root, 16)
    except ValueError as e:
        raise ValueError(f"Invalid hex digest: {e}")
    
    # Compute composite: SHA256(R_t || U_t)
    payload = f"{reasoning_root}{ui_root}".encode("ascii")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class VerificationResult:
    """
    Result of event verification.
    
    Attributes:
        status: Verification status
        is_valid: True if event passed all checks
        error_message: Human-readable error message (if invalid)
        details: Additional verification details
    """
    status: VerificationStatus
    is_valid: bool
    error_message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "status": self.status.value,
            "is_valid": self.is_valid,
            "error_message": self.error_message,
            "details": self.details,
        }


class EventVerifier:
    """
    Verifier for dual-attested events.
    
    Performs comprehensive verification of event attestation:
    1. Structural validation (hex digests, event counts)
    2. Composite root verification (H_t = SHA256(R_t || U_t))
    3. Optional: Reasoning root verification against event stream
    4. Optional: UI root verification against event stream
    
    Attributes:
        strict_mode: If True, reject events with any warnings
        verify_event_streams: If True, verify roots against event streams
    """
    
    def __init__(
        self,
        strict_mode: bool = True,
        verify_event_streams: bool = False,
    ):
        """
        Initialize event verifier.
        
        Args:
            strict_mode: Reject events with warnings
            verify_event_streams: Verify roots against event streams
        """
        self.strict_mode = strict_mode
        self.verify_event_streams = verify_event_streams
        
        # Statistics
        self.total_verified = 0
        self.total_rejected = 0
        self.rejection_reasons: Dict[str, int] = {}
    
    def verify_event(
        self,
        event: AttestedEvent,
        reasoning_events: Optional[List[str]] = None,
        ui_events: Optional[List[str]] = None,
    ) -> VerificationResult:
        """
        Verify dual-attested event.
        
        Args:
            event: Attested event to verify
            reasoning_events: Optional reasoning event stream for verification
            ui_events: Optional UI event stream for verification
        
        Returns:
            VerificationResult with status and details
        """
        # Step 1: Structural validation (already done in __post_init__)
        try:
            # This will raise if structure is invalid
            _ = event.to_dict()
        except Exception as e:
            result = VerificationResult(
                status=VerificationStatus.MALFORMED_EVENT,
                is_valid=False,
                error_message=f"Malformed event: {e}",
            )
            self._record_rejection(result.status)
            return result
        
        # Step 2: Verify composite root
        if not event.verify_composite_root():
            result = VerificationResult(
                status=VerificationStatus.INVALID_COMPOSITE_ROOT,
                is_valid=False,
                error_message=f"Composite root mismatch: expected {compute_composite_root(event.reasoning_root, event.ui_root)}, got {event.composite_root}",
                details={
                    "reasoning_root": event.reasoning_root,
                    "ui_root": event.ui_root,
                    "composite_root": event.composite_root,
                }
            )
            self._record_rejection(result.status)
            return result
        
        # Step 3: Verify reasoning root against event stream (if provided)
        if self.verify_event_streams and reasoning_events is not None:
            expected_reasoning_root = self._compute_reasoning_root(reasoning_events)
            if expected_reasoning_root != event.reasoning_root:
                result = VerificationResult(
                    status=VerificationStatus.INVALID_REASONING_ROOT,
                    is_valid=False,
                    error_message=f"Reasoning root mismatch: expected {expected_reasoning_root}, got {event.reasoning_root}",
                    details={
                        "expected": expected_reasoning_root,
                        "actual": event.reasoning_root,
                        "event_count": len(reasoning_events),
                    }
                )
                self._record_rejection(result.status)
                return result
        
        # Step 4: Verify UI root against event stream (if provided)
        if self.verify_event_streams and ui_events is not None:
            expected_ui_root = self._compute_ui_root(ui_events)
            if expected_ui_root != event.ui_root:
                result = VerificationResult(
                    status=VerificationStatus.INVALID_UI_ROOT,
                    is_valid=False,
                    error_message=f"UI root mismatch: expected {expected_ui_root}, got {event.ui_root}",
                    details={
                        "expected": expected_ui_root,
                        "actual": event.ui_root,
                        "event_count": len(ui_events),
                    }
                )
                self._record_rejection(result.status)
                return result
        
        # All checks passed
        result = VerificationResult(
            status=VerificationStatus.VALID,
            is_valid=True,
            details={
                "composite_root": event.composite_root,
                "reasoning_event_count": event.reasoning_event_count,
                "ui_event_count": event.ui_event_count,
            }
        )
        self.total_verified += 1
        return result
    
    def _compute_reasoning_root(self, events: List[str]) -> str:
        """Compute reasoning root from event stream."""
        # Merkle tree computation (simplified: hash of concatenated events)
        # In production, use proper Merkle tree from basis.crypto.hash
        content = "\n".join(events).encode("utf-8")
        return hashlib.sha256(content).hexdigest()
    
    def _compute_ui_root(self, events: List[str]) -> str:
        """Compute UI root from event stream."""
        # Merkle tree computation (simplified: hash of concatenated events)
        # In production, use proper Merkle tree from basis.crypto.hash
        content = "\n".join(events).encode("utf-8")
        return hashlib.sha256(content).hexdigest()
    
    def _record_rejection(self, status: VerificationStatus) -> None:
        """Record rejection statistics."""
        self.total_rejected += 1
        reason = status.value
        self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get verification statistics."""
        total = self.total_verified + self.total_rejected
        return {
            "total_events": total,
            "verified": self.total_verified,
            "rejected": self.total_rejected,
            "rejection_rate": self.total_rejected / total if total > 0 else 0.0,
            "rejection_reasons": self.rejection_reasons,
        }


def verify_dual_attestation(
    reasoning_root: str,
    ui_root: str,
    composite_root: str,
) -> Tuple[bool, str]:
    """
    Verify dual attestation roots.
    
    Args:
        reasoning_root: R_t
        ui_root: U_t
        composite_root: H_t
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        expected = compute_composite_root(reasoning_root, ui_root)
        if expected != composite_root:
            return False, f"Composite root mismatch: expected {expected}, got {composite_root}"
        return True, ""
    except ValueError as e:
        return False, str(e)


def filter_attested_events(
    events: List[AttestedEvent],
    verifier: EventVerifier,
) -> Tuple[List[AttestedEvent], List[Tuple[AttestedEvent, VerificationResult]]]:
    """
    Filter event stream to only verified events.
    
    Args:
        events: List of attested events
        verifier: Event verifier instance
    
    Returns:
        Tuple of (valid_events, rejected_events_with_reasons)
    """
    valid_events = []
    rejected_events = []
    
    for event in events:
        result = verifier.verify_event(event)
        if result.is_valid:
            valid_events.append(event)
        else:
            rejected_events.append((event, result))
    
    return valid_events, rejected_events


def create_attested_event_from_context(context: Any) -> AttestedEvent:
    """
    Create AttestedEvent from AttestedRunContext.
    
    Args:
        context: AttestedRunContext from substrate.bridge.context
    
    Returns:
        AttestedEvent instance
    """
    return AttestedEvent(
        reasoning_root=context.reasoning_root,
        ui_root=context.ui_root,
        composite_root=context.composite_root,
        reasoning_event_count=context.metadata.get("reasoning_event_count", 0),
        ui_event_count=context.metadata.get("ui_event_count", 0),
        metadata=context.metadata,
        statement_hash=context.statement_hash,
    )


# -----------------------------------------------------------------------------
# RFL Integration
# -----------------------------------------------------------------------------

class RFLEventGate:
    """
    Gate for RFL event ingestion.
    
    Ensures that only dual-attested, verified events are consumed by RFL.
    Implements the invariant: RFL must never read unverifiable or unattested events.
    
    Attributes:
        verifier: Event verifier instance
        fail_closed: If True, reject events on verification errors
    """
    
    def __init__(
        self,
        verifier: Optional[EventVerifier] = None,
        fail_closed: bool = True,
    ):
        """
        Initialize RFL event gate.
        
        Args:
            verifier: Optional custom verifier (creates default if None)
            fail_closed: Reject events on verification errors
        """
        self.verifier = verifier or EventVerifier(strict_mode=True)
        self.fail_closed = fail_closed
        
        # Statistics
        self.events_passed = 0
        self.events_blocked = 0
    
    def admit_event(self, event: AttestedEvent) -> Tuple[bool, str]:
        """
        Admit event for RFL consumption.
        
        Args:
            event: Attested event to admit
        
        Returns:
            Tuple of (admitted, reason)
        """
        result = self.verifier.verify_event(event)
        
        if result.is_valid:
            self.events_passed += 1
            return True, "Event verified and admitted"
        else:
            self.events_blocked += 1
            if self.fail_closed:
                return False, f"Event rejected (fail-closed): {result.error_message}"
            else:
                # Fail-open mode: log warning but admit
                return True, f"Event admitted with warning: {result.error_message}"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get gate statistics."""
        total = self.events_passed + self.events_blocked
        return {
            "total_events": total,
            "passed": self.events_passed,
            "blocked": self.events_blocked,
            "block_rate": self.events_blocked / total if total > 0 else 0.0,
            "verifier_stats": self.verifier.get_statistics(),
        }


__all__ = [
    "VerificationStatus",
    "AttestedEvent",
    "VerificationResult",
    "EventVerifier",
    "verify_dual_attestation",
    "filter_attested_events",
    "create_attested_event_from_context",
    "compute_composite_root",
    "RFLEventGate",
]
