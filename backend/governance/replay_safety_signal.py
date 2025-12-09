#!/usr/bin/env python3
"""
Replay Safety Governance Signal Adapter
========================================

Collapses Replay Safety envelope and Replay Governance radar into a single,
normalized GovernanceSignal for upstream consumption.

This module provides a clean, unambiguous "go/no-go" governance vector that
can be used by the Lawkeeper, CI/CD gates, and other enforcement mechanisms.

Signal Logic:
- BLOCK: Either Safety or Radar reports BLOCK, or alignment is DIVERGENT
- WARN: Either Safety or Radar reports WARN, or alignment is TENSION
- OK: Both Safety and Radar report OK, and alignment is ALIGNED

Author: Claude C (Replay Safety × Radar Fusion)
Date: 2025-12-09
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
from enum import Enum


class AlignmentStatus(Enum):
    """Alignment between Safety and Radar assessments."""
    ALIGNED = "ALIGNED"        # Both agree on the same conclusion
    TENSION = "TENSION"        # Minor disagreement, but both allow progress
    DIVERGENT = "DIVERGENT"    # Major disagreement, block for safety


class GovernanceStatus(Enum):
    """Final governance decision."""
    OK = "OK"          # Safe to proceed
    WARN = "WARN"      # Proceed with caution
    BLOCK = "BLOCK"    # Must not proceed


@dataclass
class SafetyEvaluation:
    """
    Replay Safety envelope evaluation result.
    
    Attributes:
        status: Safety status (OK/WARN/BLOCK)
        determinism_score: Determinism score (0-100)
        hash_match: Whether hashes match expected values
        reasons: List of reasons for the status
    """
    status: str
    determinism_score: float
    hash_match: bool
    reasons: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class RadarView:
    """
    Replay Governance radar assessment.
    
    Attributes:
        status: Radar status (OK/WARN/BLOCK)
        threading_intact: Whether chain threading is intact
        signature_valid: Whether signatures are valid
        reasons: List of reasons for the status
    """
    status: str
    threading_intact: bool
    signature_valid: bool
    reasons: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class GovernanceSignal:
    """
    Unified governance signal for replay safety.
    
    Attributes:
        alignment: Alignment between Safety and Radar
        final_status: Final governance decision (OK/WARN/BLOCK)
        reasons: Consolidated list of reasons with prefixes
        safety_status: Original safety status
        radar_status: Original radar status
        metadata: Additional metadata for debugging
    """
    alignment: str
    final_status: str
    reasons: List[str]
    safety_status: str
    radar_status: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


def assess_alignment(
    safety_eval: SafetyEvaluation,
    radar_view: RadarView,
) -> AlignmentStatus:
    """
    Assess alignment between Safety and Radar evaluations.
    
    Args:
        safety_eval: Safety evaluation result
        radar_view: Radar assessment result
    
    Returns:
        AlignmentStatus indicating the level of agreement
    """
    safety_status = safety_eval.status.upper()
    radar_status = radar_view.status.upper()
    
    # Perfect alignment: both agree
    if safety_status == radar_status:
        return AlignmentStatus.ALIGNED
    
    # Divergent: one says BLOCK, the other doesn't
    if "BLOCK" in (safety_status, radar_status):
        return AlignmentStatus.DIVERGENT
    
    # Tension: one says WARN, the other says OK
    if (safety_status == "WARN" and radar_status == "OK") or \
       (safety_status == "OK" and radar_status == "WARN"):
        return AlignmentStatus.TENSION
    
    # Default to DIVERGENT for any other mismatch
    return AlignmentStatus.DIVERGENT


def consolidate_reasons(
    safety_eval: SafetyEvaluation,
    radar_view: RadarView,
    alignment: AlignmentStatus,
) -> List[str]:
    """
    Consolidate reasons from Safety and Radar with appropriate prefixes.
    
    Args:
        safety_eval: Safety evaluation result
        radar_view: Radar assessment result
        alignment: Alignment status
    
    Returns:
        List of prefixed reasons
    """
    reasons = []
    
    # Add safety reasons with [Safety] prefix
    for reason in safety_eval.reasons:
        reasons.append(f"[Safety] {reason}")
    
    # Add radar reasons with [Radar] prefix
    for reason in radar_view.reasons:
        reasons.append(f"[Radar] {reason}")
    
    # Add conflict reason if divergent
    if alignment == AlignmentStatus.DIVERGENT:
        reasons.append(
            f"[CONFLICT] Safety={safety_eval.status}, Radar={radar_view.status} (DIVERGENT)"
        )
    elif alignment == AlignmentStatus.TENSION:
        reasons.append(
            f"[CONFLICT] Safety={safety_eval.status}, Radar={radar_view.status} (TENSION)"
        )
    
    return reasons


def determine_final_status(
    safety_eval: SafetyEvaluation,
    radar_view: RadarView,
    alignment: AlignmentStatus,
) -> GovernanceStatus:
    """
    Determine the final governance status.
    
    Logic:
    - BLOCK if either side BLOCKs OR alignment is DIVERGENT
    - WARN if either side WARNs OR alignment is TENSION
    - OK if both OK AND alignment is ALIGNED
    
    Args:
        safety_eval: Safety evaluation result
        radar_view: Radar assessment result
        alignment: Alignment status
    
    Returns:
        GovernanceStatus (OK/WARN/BLOCK)
    """
    safety_status = safety_eval.status.upper()
    radar_status = radar_view.status.upper()
    
    # BLOCK conditions
    if safety_status == "BLOCK" or radar_status == "BLOCK":
        return GovernanceStatus.BLOCK
    
    if alignment == AlignmentStatus.DIVERGENT:
        return GovernanceStatus.BLOCK
    
    # WARN conditions
    if safety_status == "WARN" or radar_status == "WARN":
        return GovernanceStatus.WARN
    
    if alignment == AlignmentStatus.TENSION:
        return GovernanceStatus.WARN
    
    # OK condition (both OK and aligned)
    if safety_status == "OK" and radar_status == "OK" and alignment == AlignmentStatus.ALIGNED:
        return GovernanceStatus.OK
    
    # Default to BLOCK for safety
    return GovernanceStatus.BLOCK


def to_governance_signal_for_replay_safety(
    safety_eval: SafetyEvaluation,
    radar_view: RadarView,
) -> Dict[str, Any]:
    """
    Convert Replay Safety and Radar assessments into a unified GovernanceSignal.
    
    This is the primary API for collapsing the two subsystems into a single,
    normalized signal that can be consumed by upstream governance mechanisms.
    
    Args:
        safety_eval: Replay Safety envelope evaluation
        radar_view: Replay Governance radar assessment
    
    Returns:
        Dictionary representation of GovernanceSignal
    
    Example:
        >>> safety = SafetyEvaluation(
        ...     status="OK",
        ...     determinism_score=98.5,
        ...     hash_match=True,
        ...     reasons=["All hashes verified"]
        ... )
        >>> radar = RadarView(
        ...     status="OK",
        ...     threading_intact=True,
        ...     signature_valid=True,
        ...     reasons=["Chain threading intact"]
        ... )
        >>> signal = to_governance_signal_for_replay_safety(safety, radar)
        >>> signal["final_status"]
        'OK'
        >>> signal["alignment"]
        'ALIGNED'
    """
    # Assess alignment
    alignment = assess_alignment(safety_eval, radar_view)
    
    # Determine final status
    final_status = determine_final_status(safety_eval, radar_view, alignment)
    
    # Consolidate reasons
    reasons = consolidate_reasons(safety_eval, radar_view, alignment)
    
    # Build metadata
    metadata = {
        "safety_determinism_score": safety_eval.determinism_score,
        "safety_hash_match": safety_eval.hash_match,
        "radar_threading_intact": radar_view.threading_intact,
        "radar_signature_valid": radar_view.signature_valid,
    }
    
    # Create governance signal
    signal = GovernanceSignal(
        alignment=alignment.value,
        final_status=final_status.value,
        reasons=reasons,
        safety_status=safety_eval.status,
        radar_status=radar_view.status,
        metadata=metadata,
    )
    
    return signal.to_dict()


def extend_evidence_pack_with_governance_status(
    evidence_pack: Dict[str, Any],
    governance_signal: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Harmonize evidence pack with governance_status field.
    
    This extends the existing evidence pack contract by adding a top-level
    governance_status field that mirrors the consolidated status from the
    governance signal.
    
    Args:
        evidence_pack: Existing evidence pack dictionary
        governance_signal: Governance signal from to_governance_signal_for_replay_safety()
    
    Returns:
        Extended evidence pack with governance_status field
    
    Example:
        >>> evidence = {"replay_hash": "abc123", "determinism_score": 98.5}
        >>> signal = {"final_status": "OK", "alignment": "ALIGNED"}
        >>> extended = extend_evidence_pack_with_governance_status(evidence, signal)
        >>> extended["governance_status"]
        {'final_status': 'OK', 'alignment': 'ALIGNED', 'reasons': [...]}
    """
    # Create a copy to avoid mutating the original
    extended_pack = evidence_pack.copy()
    
    # Add governance_status field
    extended_pack["governance_status"] = {
        "final_status": governance_signal["final_status"],
        "alignment": governance_signal["alignment"],
        "reasons": governance_signal["reasons"],
        "safety_status": governance_signal["safety_status"],
        "radar_status": governance_signal["radar_status"],
    }
    
    return extended_pack


# Convenience function for creating SafetyEvaluation from dict
def safety_eval_from_dict(data: Dict[str, Any]) -> SafetyEvaluation:
    """Create SafetyEvaluation from dictionary."""
    return SafetyEvaluation(
        status=data["status"],
        determinism_score=data["determinism_score"],
        hash_match=data["hash_match"],
        reasons=data["reasons"],
    )


# Convenience function for creating RadarView from dict
def radar_view_from_dict(data: Dict[str, Any]) -> RadarView:
    """Create RadarView from dictionary."""
    return RadarView(
        status=data["status"],
        threading_intact=data["threading_intact"],
        signature_valid=data["signature_valid"],
        reasons=data["reasons"],
    )
