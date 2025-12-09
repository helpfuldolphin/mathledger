"""
U2 Safety Enforcement Layer

Implements runtime safety gates for U2Runner:
- Hard gate decision evaluation (Cortex approval)
- Safety SLO envelope tracking
- TDA attitude integration hooks
- Deterministic safety decisions

INVARIANTS:
- evaluate_hard_gate_decision() is BLOCKING
- NO candidate executes without approval
- All decisions are deterministic given same input
- Safety state is serializable for snapshots
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from enum import Enum

from rfl.prng import DeterministicPRNG


class GateDecision(Enum):
    """Hard gate decision outcomes."""
    APPROVED = "approved"
    REJECTED = "rejected"
    ABSTAINED = "abstained"


@dataclass
class U2SafetyContext:
    """
    Runtime safety context for U2 execution.
    
    Tracks safety-critical metrics and state for gate evaluation.
    """
    
    # Execution metrics
    total_candidates_evaluated: int = 0
    total_approvals: int = 0
    total_rejections: int = 0
    total_abstentions: int = 0
    
    # Safety SLO tracking
    approval_rate: float = 0.0
    rejection_rate: float = 0.0
    abstention_rate: float = 0.0
    
    # TDA attitude integration (placeholder for future integration)
    tda_attitudes: Dict[str, Any] = field(default_factory=dict)
    
    # Runtime state
    safety_violations: int = 0
    last_decision: Optional[GateDecision] = None
    
    def record_decision(self, decision: GateDecision) -> None:
        """
        Record a gate decision and update metrics.
        
        Args:
            decision: Gate decision outcome
        """
        self.total_candidates_evaluated += 1
        self.last_decision = decision
        
        if decision == GateDecision.APPROVED:
            self.total_approvals += 1
        elif decision == GateDecision.REJECTED:
            self.total_rejections += 1
        elif decision == GateDecision.ABSTAINED:
            self.total_abstentions += 1
        
        # Update rates
        total = self.total_candidates_evaluated
        if total > 0:
            self.approval_rate = self.total_approvals / total
            self.rejection_rate = self.total_rejections / total
            self.abstention_rate = self.total_abstentions / total
    
    def to_dict(self) -> Dict[str, Any]:
        """Export safety context as dictionary."""
        return {
            "total_candidates_evaluated": self.total_candidates_evaluated,
            "total_approvals": self.total_approvals,
            "total_rejections": self.total_rejections,
            "total_abstentions": self.total_abstentions,
            "approval_rate": self.approval_rate,
            "rejection_rate": self.rejection_rate,
            "abstention_rate": self.abstention_rate,
            "safety_violations": self.safety_violations,
            "last_decision": self.last_decision.value if self.last_decision else None,
            "tda_attitudes": dict(self.tda_attitudes),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "U2SafetyContext":
        """Restore safety context from dictionary."""
        ctx = cls(
            total_candidates_evaluated=data.get("total_candidates_evaluated", 0),
            total_approvals=data.get("total_approvals", 0),
            total_rejections=data.get("total_rejections", 0),
            total_abstentions=data.get("total_abstentions", 0),
            approval_rate=data.get("approval_rate", 0.0),
            rejection_rate=data.get("rejection_rate", 0.0),
            abstention_rate=data.get("abstention_rate", 0.0),
            safety_violations=data.get("safety_violations", 0),
            tda_attitudes=data.get("tda_attitudes", {}),
        )
        
        last_decision_str = data.get("last_decision")
        if last_decision_str:
            ctx.last_decision = GateDecision(last_decision_str)
        
        return ctx


@dataclass
class SafetyEnvelope:
    """
    Safety envelope metadata for gate decisions.
    
    Contains decision rationale and compliance attestation.
    """
    
    decision: GateDecision
    candidate_id: str
    cycle: int
    
    # Decision rationale
    reason: str
    confidence: float  # 0.0 to 1.0
    
    # SLO compliance
    slo_compliant: bool
    slo_violations: Dict[str, Any] = field(default_factory=dict)
    
    # Provenance
    gate_version: str = "v1.0.0"
    prng_state: Optional[Dict[str, Any]] = None  # PRNG state for reproducibility
    
    def to_dict(self) -> Dict[str, Any]:
        """Export envelope as dictionary."""
        return {
            "decision": self.decision.value,
            "candidate_id": self.candidate_id,
            "cycle": self.cycle,
            "reason": self.reason,
            "confidence": self.confidence,
            "slo_compliant": self.slo_compliant,
            "slo_violations": dict(self.slo_violations),
            "gate_version": self.gate_version,
            "prng_state": self.prng_state,
        }


def _extract_candidate_id(candidate: Any) -> str:
    """
    Extract candidate ID from various candidate formats.
    
    Supports:
    - Dict with "item" key
    - Dict with other keys (use string representation)
    - String candidates
    - Other types (use string representation)
    
    Args:
        candidate: Candidate in any supported format
    
    Returns:
        String identifier for the candidate
    """
    if isinstance(candidate, dict):
        if "item" in candidate:
            return str(candidate["item"])
        # Use dict representation if no item key
        return str(candidate)
    # For non-dict types, use string representation
    return str(candidate)


def evaluate_hard_gate_decision(
    candidate: Any,
    cycle: int,
    safety_context: U2SafetyContext,
    prng: DeterministicPRNG,
    max_depth: int = 10,
    max_complexity: float = 1000.0,
) -> SafetyEnvelope:
    """
    BLOCKING call to evaluate whether a candidate should execute.
    
    This is the Cortex approval mechanism - NO candidate executes
    without passing this gate.
    
    INVARIANTS:
    - Decision is deterministic given same inputs
    - Blocking call (synchronous evaluation)
    - No side effects except safety_context update
    
    Args:
        candidate: Candidate to evaluate
        cycle: Current cycle number
        safety_context: Runtime safety context
        prng: Deterministic PRNG for tie-breaking
        max_depth: Maximum allowed depth
        max_complexity: Maximum allowed complexity
    
    Returns:
        SafetyEnvelope with decision and metadata
    """
    
    # Extract candidate features
    candidate_id = _extract_candidate_id(candidate)
    depth = candidate.get("depth", 0) if isinstance(candidate, dict) else 0
    complexity = len(str(candidate))
    
    # Initialize decision state
    decision = GateDecision.APPROVED
    reason = "passed_all_checks"
    confidence = 1.0
    slo_compliant = True
    slo_violations = {}
    
    # Check 1: Depth limit
    if depth > max_depth:
        decision = GateDecision.REJECTED
        reason = f"depth_exceeded: {depth} > {max_depth}"
        confidence = 1.0
        slo_compliant = False
        slo_violations["depth_limit"] = {
            "observed": depth,
            "limit": max_depth,
        }
    
    # Check 2: Complexity limit
    elif complexity > max_complexity:
        decision = GateDecision.REJECTED
        reason = f"complexity_exceeded: {complexity} > {max_complexity}"
        confidence = 1.0
        slo_compliant = False
        slo_violations["complexity_limit"] = {
            "observed": complexity,
            "limit": max_complexity,
        }
    
    # Check 3: Safety SLO envelope check
    # If rejection rate is too high, start abstaining to preserve SLO
    elif safety_context.rejection_rate > 0.5 and safety_context.total_candidates_evaluated > 10:
        # Use PRNG for deterministic tie-breaking
        abstention_threshold = 0.3
        random_value = prng.for_path("safety_gate", candidate_id, str(cycle)).random()
        
        if random_value < abstention_threshold:
            decision = GateDecision.ABSTAINED
            reason = "slo_protection: high_rejection_rate"
            confidence = 0.5
            slo_compliant = True  # Abstention is compliant behavior
        else:
            # Allow through with reduced confidence
            decision = GateDecision.APPROVED
            reason = "conditional_approval: slo_warning"
            confidence = 0.6
            slo_compliant = True
    
    # Check 4: TDA attitude integration (placeholder)
    # Future: integrate topological data analysis attitudes here
    # For now, this is a no-op that preserves determinism
    tda_signal = safety_context.tda_attitudes.get("approval_signal", 1.0)
    if tda_signal < 0.3:
        # TDA suggests high risk
        decision = GateDecision.REJECTED
        reason = "tda_risk_signal"
        confidence = 0.9
        slo_compliant = True
        slo_violations["tda_attitude"] = {
            "signal": tda_signal,
            "threshold": 0.3,
        }
    
    # Create envelope
    envelope = SafetyEnvelope(
        decision=decision,
        candidate_id=candidate_id,
        cycle=cycle,
        reason=reason,
        confidence=confidence,
        slo_compliant=slo_compliant,
        slo_violations=slo_violations,
        prng_state=prng.get_state(),
    )
    
    # Record decision in safety context
    safety_context.record_decision(decision)
    
    # Track SLO violations
    if not slo_compliant:
        safety_context.safety_violations += 1
    
    return envelope


def validate_safety_envelope(envelope: SafetyEnvelope) -> bool:
    """
    Validate safety envelope integrity.
    
    Args:
        envelope: Safety envelope to validate
    
    Returns:
        True if envelope is valid
    """
    # Check required fields
    if not envelope.candidate_id:
        return False
    
    if envelope.cycle < 0:
        return False
    
    if not (0.0 <= envelope.confidence <= 1.0):
        return False
    
    # Validate SLO compliance consistency
    # If NOT slo_compliant, there should be violations recorded
    if not envelope.slo_compliant and not envelope.slo_violations:
        return False
    
    return True
