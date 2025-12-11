"""
Cortex Telemetry — Surface Cortex Gate Decisions

This module provides data structures and utilities for surfacing Cortex
(hard gate + TDA) decisions into First Light summaries, Uplift Safety Engine,
and Evidence Packs. It does NOT implement gating logic—only telemetry.

Cortex owns the gate; this module only exposes gate outcomes.

Author: rfl-policy-engineer
Date: 2025-12-11
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Any, Optional


class TDAMode(str, Enum):
    """TDA (Telemetry, Diagnosis, Advisory) gating mode."""
    BLOCK = "BLOCK"        # Hard block on violations
    DRY_RUN = "DRY_RUN"    # Log violations but don't block
    SHADOW = "SHADOW"       # Silent monitoring only


class HardGateStatus(str, Enum):
    """Derived hard gate status for summary reporting."""
    OK = "OK"         # No violations detected
    WARN = "WARN"     # Advisory violations (DRY_RUN/SHADOW mode)
    BLOCK = "BLOCK"   # Hard blocking violations (BLOCK mode)


@dataclass
class CortexDecision:
    """Single Cortex gate decision record."""
    
    decision_id: str
    cycle_id: Optional[int]
    item: str
    blocked: bool
    advisory: bool
    rationale: str
    tda_mode: TDAMode
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "decision_id": self.decision_id,
            "cycle_id": self.cycle_id,
            "item": self.item,
            "blocked": self.blocked,
            "advisory": self.advisory,
            "rationale": self.rationale,
            "tda_mode": self.tda_mode.value,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class CortexEnvelope:
    """Envelope of Cortex decisions for a run or experiment."""
    
    tda_mode: TDAMode
    decisions: List[CortexDecision] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_decision(self, decision: CortexDecision) -> None:
        """Add a decision to the envelope."""
        self.decisions.append(decision)
    
    def total_decisions(self) -> int:
        """Count total decisions."""
        return len(self.decisions)
    
    def blocked_decisions(self) -> int:
        """Count blocked decisions."""
        return sum(1 for d in self.decisions if d.blocked)
    
    def advisory_decisions(self) -> int:
        """Count advisory-only decisions."""
        return sum(1 for d in self.decisions if d.advisory and not d.blocked)
    
    def compute_hard_gate_status(self) -> HardGateStatus:
        """
        Derive hard gate status from decisions and TDA mode.
        
        Logic:
        - If BLOCK mode and blocked_decisions > 0: BLOCK
        - If (DRY_RUN or SHADOW) and any violations: WARN
        - Otherwise: OK
        """
        if self.tda_mode == TDAMode.BLOCK and self.blocked_decisions() > 0:
            return HardGateStatus.BLOCK
        elif self.tda_mode in (TDAMode.DRY_RUN, TDAMode.SHADOW) and (
            self.advisory_decisions() > 0 or self.blocked_decisions() > 0
        ):
            return HardGateStatus.WARN
        return HardGateStatus.OK
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "tda_mode": self.tda_mode.value,
            "decisions": [d.to_dict() for d in self.decisions],
            "metadata": self.metadata,
        }


@dataclass
class CortexSummary:
    """First Light summary for Cortex gate outcomes."""
    
    total_decisions: int
    blocked_decisions: int
    advisory_decisions: int
    tda_mode: str
    hard_gate_status: str
    
    @classmethod
    def from_envelope(cls, envelope: CortexEnvelope) -> CortexSummary:
        """Create summary from envelope."""
        return cls(
            total_decisions=envelope.total_decisions(),
            blocked_decisions=envelope.blocked_decisions(),
            advisory_decisions=envelope.advisory_decisions(),
            tda_mode=envelope.tda_mode.value,
            hard_gate_status=envelope.compute_hard_gate_status().value,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return asdict(self)


def compute_cortex_summary(envelope: CortexEnvelope) -> Dict[str, Any]:
    """
    Compute First Light cortex_summary from envelope.
    
    Args:
        envelope: Cortex decision envelope
        
    Returns:
        Dictionary with cortex_summary structure
    """
    summary = CortexSummary.from_envelope(envelope)
    return {
        "cortex_summary": summary.to_dict()
    }


@dataclass
class UpliftSafetyCortexAdapter:
    """
    Adapter for Uplift Safety Engine v6.
    
    Converts Cortex envelope to uplift safety tensor contribution.
    This is advisory-only weight; does not alter core uplift computation.
    """
    
    cortex_gate_band: str  # LOW, MEDIUM, HIGH
    hypothetical_block_rate: float
    advisory_only: bool
    
    @classmethod
    def from_envelope(cls, envelope: CortexEnvelope) -> UpliftSafetyCortexAdapter:
        """
        Create adapter from envelope.
        
        Band logic:
        - HIGH: blocked_decisions > 0 (in BLOCK mode)
        - MEDIUM: advisory_decisions > 0 (any mode)
        - LOW: no violations
        """
        total = envelope.total_decisions()
        blocked = envelope.blocked_decisions()
        advisory = envelope.advisory_decisions()
        
        # Compute hypothetical block rate
        if total > 0:
            hypothetical_block_rate = (blocked + advisory) / total
        else:
            hypothetical_block_rate = 0.0
        
        # Determine band
        if blocked > 0 and envelope.tda_mode == TDAMode.BLOCK:
            band = "HIGH"
        elif advisory > 0 or blocked > 0:
            band = "MEDIUM"
        else:
            band = "LOW"
        
        # advisory_only indicates whether the adapter's output is informational only
        # (i.e., TDA mode is not BLOCK). When False, it means BLOCK mode is active
        # and violations would result in hard blocks.
        advisory_only = envelope.tda_mode != TDAMode.BLOCK
        
        return cls(
            cortex_gate_band=band,
            hypothetical_block_rate=hypothetical_block_rate,
            advisory_only=advisory_only,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return asdict(self)


def attach_cortex_governance_to_evidence(
    evidence: Dict[str, Any],
    cortex_envelope: CortexEnvelope
) -> Dict[str, Any]:
    """
    Attach Cortex governance tile to evidence pack.
    
    Adds evidence["governance"]["cortex_gate"] with:
    - hard_gate_status
    - blocked_decisions
    - tda_mode
    - rationale strings (top 3)
    
    Args:
        evidence: Evidence pack dict (will NOT be mutated)
        cortex_envelope: Cortex decision envelope
        
    Returns:
        New evidence dict with cortex_gate added
    """
    # Deep copy to avoid mutation
    evidence_copy = copy.deepcopy(evidence)
    
    # Ensure governance key exists
    if "governance" not in evidence_copy:
        evidence_copy["governance"] = {}
    
    # Extract top rationales (limit to 3)
    rationales = [d.rationale for d in cortex_envelope.decisions if d.rationale]
    top_rationales = rationales[:3] if rationales else []
    
    # Build cortex_gate tile
    cortex_gate = {
        "hard_gate_status": cortex_envelope.compute_hard_gate_status().value,
        "blocked_decisions": cortex_envelope.blocked_decisions(),
        "tda_mode": cortex_envelope.tda_mode.value,
        "rationales": top_rationales,
        "total_decisions": cortex_envelope.total_decisions(),
        "advisory_decisions": cortex_envelope.advisory_decisions(),
    }
    
    evidence_copy["governance"]["cortex_gate"] = cortex_gate
    
    return evidence_copy
