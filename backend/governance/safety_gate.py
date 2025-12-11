"""
Safety Gate Module for Phase X Neural Link.

Provides data structures and utilities for surfacing safety gate decisions
into First Light summaries, global health tiles, and evidence packs.

DESIGN CONSTRAINT: This module is behavior-preserving. It does not implement
new gating logic, only structures for surfacing existing gate decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class SafetyGateStatus(str, Enum):
    """Safety gate decision status."""
    
    PASS = "PASS"
    WARN = "WARN"
    BLOCK = "BLOCK"


@dataclass
class SafetyGateDecision:
    """Single safety gate decision record."""
    
    cycle: int
    status: SafetyGateStatus
    reason: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyEnvelope:
    """
    Safety gate envelope aggregating decisions across a run.
    
    This structure is populated by safety modules that evaluate gate
    conditions. It serves as the source of truth for surfacing gate
    decisions into observability systems.
    """
    
    final_status: SafetyGateStatus
    total_decisions: int
    blocked_cycles: int
    advisory_cycles: int  # WARN status
    decisions: List[SafetyGateDecision] = field(default_factory=list)
    
    def get_reasons(self, limit: Optional[int] = None) -> List[str]:
        """
        Extract reasons from decisions, deterministically ordered.
        
        Args:
            limit: Maximum number of reasons to return (default: all)
            
        Returns:
            List of unique reasons, sorted alphabetically for determinism
        """
        reasons = []
        for decision in self.decisions:
            if decision.reason:
                reasons.append(decision.reason)
        
        # Deduplicate and sort for determinism
        unique_reasons = sorted(set(reasons))
        
        if limit is not None:
            return unique_reasons[:limit]
        return unique_reasons
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "final_status": self.final_status.value,
            "total_decisions": self.total_decisions,
            "blocked_cycles": self.blocked_cycles,
            "advisory_cycles": self.advisory_cycles,
            "reasons": self.get_reasons(),
        }


def build_safety_gate_summary_for_first_light(envelope: SafetyEnvelope) -> Dict[str, Any]:
    """
    Build safety gate summary for First Light (First Organism) output.
    
    Args:
        envelope: Safety gate envelope with aggregated decisions
        
    Returns:
        Dictionary suitable for inclusion in First Light summary.json
    """
    return {
        "final_status": envelope.final_status.value,
        "total_decisions": envelope.total_decisions,
        "blocked_cycles": envelope.blocked_cycles,
        "advisory_cycles": envelope.advisory_cycles,
        "reasons": envelope.get_reasons(),  # Deterministically ordered
    }


def build_safety_gate_tile_for_global_health(envelope: SafetyEnvelope) -> Dict[str, Any]:
    """
    Build safety gate tile for global health surface.
    
    Maps gate status to traffic light indicators:
    - PASS -> GREEN
    - WARN -> YELLOW
    - BLOCK -> RED
    
    Args:
        envelope: Safety gate envelope with aggregated decisions
        
    Returns:
        Dictionary with global health tile structure
    """
    # Map status to status_light
    status_light_map = {
        SafetyGateStatus.PASS: "GREEN",
        SafetyGateStatus.WARN: "YELLOW",
        SafetyGateStatus.BLOCK: "RED",
    }
    
    status_light = status_light_map[envelope.final_status]
    
    # Compute blocked fraction
    blocked_fraction = 0.0
    if envelope.total_decisions > 0:
        blocked_fraction = envelope.blocked_cycles / envelope.total_decisions
    
    # Generate neutral headline
    headline = f"Safety gate: {envelope.final_status.value}"
    if envelope.blocked_cycles > 0:
        headline += f" ({envelope.blocked_cycles} blocked)"
    elif envelope.advisory_cycles > 0:
        headline += f" ({envelope.advisory_cycles} advisory)"
    
    return {
        "schema_version": "1.0.0",
        "status_light": status_light,
        "blocked_fraction": round(blocked_fraction, 4),
        "headline": headline,
        "total_decisions": envelope.total_decisions,
        "blocked_cycles": envelope.blocked_cycles,
        "advisory_cycles": envelope.advisory_cycles,
    }


def attach_safety_gate_to_evidence(
    evidence: Dict[str, Any],
    envelope: SafetyEnvelope,
) -> Dict[str, Any]:
    """
    Attach safety gate data to evidence pack.
    
    Adds gate decisions under evidence["governance"]["safety_gate"].
    Does not mutate input dictionary - returns new dictionary.
    
    Args:
        evidence: Evidence pack dictionary
        envelope: Safety gate envelope with aggregated decisions
        
    Returns:
        New evidence dictionary with safety gate attached
    """
    # Copy evidence to avoid mutation
    evidence_copy = evidence.copy()
    
    # Ensure governance section exists
    if "governance" not in evidence_copy:
        evidence_copy["governance"] = {}
    else:
        evidence_copy["governance"] = evidence_copy["governance"].copy()
    
    # Attach safety gate data
    evidence_copy["governance"]["safety_gate"] = {
        "final_status": envelope.final_status.value,
        "blocked_cycles": envelope.blocked_cycles,
        "advisory_cycles": envelope.advisory_cycles,
        "reasons": envelope.get_reasons(limit=3),  # Top 3 reasons
    }
    
    return evidence_copy


def build_global_health_surface(
    tiles: Dict[str, Dict[str, Any]],
    safety_envelope: Optional[SafetyEnvelope] = None,
) -> Dict[str, Any]:
    """
    Build global health surface with safety gate tile.
    
    This is a placeholder/adapter that integrates the safety gate tile
    into a global health surface structure. In production, this would
    be called by the main health aggregator.
    
    Args:
        tiles: Existing health tiles
        safety_envelope: Optional safety gate envelope to include
        
    Returns:
        Global health surface dictionary
    """
    health = {
        "schema_version": "1.0.0",
        "tiles": tiles.copy(),
    }
    
    # Add safety gate tile if envelope provided
    if safety_envelope is not None:
        health["tiles"]["safety_gate"] = build_safety_gate_tile_for_global_health(
            safety_envelope
        )
    
    return health
