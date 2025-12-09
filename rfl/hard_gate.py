"""
TDA Hard Gate Decision Evaluation

Implements evaluate_hard_gate_decision() for integration with U2Runner and RFLRunner.
Produces HSS (Hash Stability Score) traces and Δp (policy delta) metrics.

PHASE II — U2 Uplift Experiments Extension
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

from .evidence_fusion import TDAOutcome, TDAFields

logger = logging.getLogger(__name__)


# Hard Gate Thresholds
THRESHOLD_CRITICAL_BLOCK_RATE = 0.95  # Block rate above which promotion is blocked
THRESHOLD_HIGH_BLOCK_RATE = 0.80      # Block rate above which warning is issued
THRESHOLD_CRITICAL_HSS = 0.30         # HSS below which structural instability is flagged
THRESHOLD_MODERATE_HSS = 0.70         # HSS below which moderate instability is flagged
THRESHOLD_STABILITY_TREND = 0.10      # HSS difference for trend detection


class HardGateMode(Enum):
    """Hard gate evaluation mode."""
    SHADOW = "shadow"  # Log only, don't block
    ENFORCE = "enforce"  # Block on violations
    DISABLED = "disabled"  # No evaluation


@dataclass
class HSSTrace:
    """
    Hash Stability Score trace entry.
    
    HSS measures the stability/consistency of policy hashes over time,
    detecting structural changes that might indicate instability.
    """
    cycle: int
    policy_hash: str
    stability_score: float  # 0.0 (unstable) to 1.0 (stable)
    hash_delta: Optional[str] = None  # Hash difference from previous cycle
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "cycle": self.cycle,
            "policy_hash": self.policy_hash,
            "stability_score": self.stability_score,
            "hash_delta": self.hash_delta,
        }


@dataclass
class PolicyDelta:
    """
    Policy delta (Δp) metrics between cycles or runs.
    
    Tracks changes in policy parameters/weights to detect drift.
    """
    delta_magnitude: float  # L2 norm of parameter changes
    delta_direction: str  # "increase", "decrease", "stable"
    affected_parameters: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "delta_magnitude": self.delta_magnitude,
            "delta_direction": self.delta_direction,
            "affected_parameters": self.affected_parameters,
        }


@dataclass
class HardGateDecision:
    """
    Result of TDA hard gate evaluation.
    
    Contains decision, TDA fields, HSS traces, and policy deltas.
    """
    outcome: TDAOutcome
    tda_fields: TDAFields
    hss_traces: List[HSSTrace] = field(default_factory=list)
    policy_delta: Optional[PolicyDelta] = None
    decision_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "outcome": self.outcome.value,
            "tda_fields": self.tda_fields.to_dict(),
            "hss_traces": [trace.to_dict() for trace in self.hss_traces],
            "policy_delta": self.policy_delta.to_dict() if self.policy_delta else None,
            "decision_reason": self.decision_reason,
            "metadata": self.metadata,
        }


def evaluate_hard_gate_decision(
    cycle: int,
    policy_state: Dict[str, Any],
    event_stats: Dict[str, Any],
    previous_policy_hash: Optional[str] = None,
    mode: HardGateMode = HardGateMode.SHADOW,
) -> HardGateDecision:
    """
    Evaluate TDA hard gate decision for a single cycle.
    
    Computes:
    - HSS (Hash Stability Score) traces
    - Δp (policy delta) metrics
    - TDA fields (HSS value, block_rate, tda_outcome)
    - Hard gate decision (PASS/WARN/BLOCK/SHADOW)
    
    Args:
        cycle: Current cycle number
        policy_state: Policy state dictionary (weights, parameters, etc.)
        event_stats: Event verification statistics (blocked, passed, etc.)
        previous_policy_hash: Hash from previous cycle for delta computation
        mode: Hard gate mode (SHADOW, ENFORCE, DISABLED)
    
    Returns:
        HardGateDecision with outcome and traces
    """
    logger.debug(f"Evaluating hard gate for cycle {cycle}, mode={mode.value}")
    
    if mode == HardGateMode.DISABLED:
        return HardGateDecision(
            outcome=TDAOutcome.UNKNOWN,
            tda_fields=TDAFields(),
            decision_reason="Hard gate disabled",
        )
    
    # Compute current policy hash
    policy_hash = _compute_policy_hash(policy_state)
    
    # Compute HSS (Hash Stability Score)
    hss_score, hash_delta = _compute_hss(policy_hash, previous_policy_hash)
    
    hss_trace = HSSTrace(
        cycle=cycle,
        policy_hash=policy_hash,
        stability_score=hss_score,
        hash_delta=hash_delta,
    )
    
    # Compute policy delta (Δp)
    policy_delta = None
    if previous_policy_hash:
        policy_delta = _compute_policy_delta(policy_state)
    
    # Extract event blocking statistics
    total_events = event_stats.get("total_events", 0)
    blocked_events = event_stats.get("blocked", 0)
    block_rate = blocked_events / max(1, total_events)
    
    # Determine TDA outcome based on thresholds
    outcome, reason = _determine_outcome(
        hss_score=hss_score,
        block_rate=block_rate,
        policy_delta=policy_delta,
        mode=mode,
    )
    
    # Build TDA fields
    tda_fields = TDAFields(
        HSS=hss_score,
        block_rate=block_rate,
        tda_outcome=outcome,
    )
    
    decision = HardGateDecision(
        outcome=outcome,
        tda_fields=tda_fields,
        hss_traces=[hss_trace],
        policy_delta=policy_delta,
        decision_reason=reason,
        metadata={
            "cycle": cycle,
            "mode": mode.value,
            "total_events": total_events,
            "blocked_events": blocked_events,
        },
    )
    
    logger.info(f"Hard gate decision for cycle {cycle}: {outcome.value} - {reason}")
    
    return decision


def _compute_policy_hash(policy_state: Dict[str, Any]) -> str:
    """
    Compute deterministic hash of policy state.
    
    Uses canonical JSON encoding for reproducibility.
    """
    import json
    
    # Extract relevant policy parameters (ignore metadata)
    policy_params = {
        k: v for k, v in policy_state.items()
        if k not in ["timestamp", "run_id", "metadata"]
    }
    
    canonical = json.dumps(policy_params, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def _compute_hss(
    current_hash: str,
    previous_hash: Optional[str],
) -> Tuple[float, Optional[str]]:
    """
    Compute Hash Stability Score (HSS).
    
    HSS = 1.0 if hashes match (stable)
    HSS = 0.0 if hashes differ significantly (unstable)
    HSS = 0.5 for first cycle (no previous hash)
    
    Returns:
        Tuple of (hss_score, hash_delta)
    """
    if previous_hash is None:
        # First cycle - neutral score
        return 0.5, None
    
    if current_hash == previous_hash:
        # Hashes match - perfectly stable
        return 1.0, None
    
    # Hashes differ - compute similarity
    # Simple approach: count matching hex characters
    matches = sum(c1 == c2 for c1, c2 in zip(current_hash, previous_hash))
    similarity = matches / len(current_hash)
    
    hash_delta = f"{current_hash[:8]}...{current_hash[-8:]}"
    
    return similarity, hash_delta


def _compute_policy_delta(policy_state: Dict[str, Any]) -> PolicyDelta:
    """
    Compute policy delta (Δp) metrics.
    
    Simplified version - in production would track parameter changes.
    """
    # Mock implementation - would need previous policy state for real delta
    # For now, return neutral delta
    return PolicyDelta(
        delta_magnitude=0.0,
        delta_direction="stable",
        affected_parameters=[],
    )


def _determine_outcome(
    hss_score: float,
    block_rate: float,
    policy_delta: Optional[PolicyDelta],
    mode: HardGateMode,
) -> Tuple[TDAOutcome, str]:
    """
    Determine TDA outcome based on metrics and thresholds.
    
    Uses module-level threshold constants for consistency.
    
    Returns:
        Tuple of (TDAOutcome, reason)
    """
    # Check for critical blocking
    if block_rate > THRESHOLD_CRITICAL_BLOCK_RATE:
        outcome = TDAOutcome.BLOCK if mode == HardGateMode.ENFORCE else TDAOutcome.SHADOW
        reason = f"Critical event blocking rate: {block_rate:.2%}"
        return outcome, reason
    
    # Check for structural instability
    if hss_score < THRESHOLD_CRITICAL_HSS:
        outcome = TDAOutcome.BLOCK if mode == HardGateMode.ENFORCE else TDAOutcome.SHADOW
        reason = f"Structural instability detected (HSS={hss_score:.2f})"
        return outcome, reason
    
    # Check for excessive blocking
    if block_rate > THRESHOLD_HIGH_BLOCK_RATE:
        reason = f"High event blocking rate: {block_rate:.2%}"
        return TDAOutcome.WARN, reason
    
    # Check for moderate instability
    if hss_score < THRESHOLD_MODERATE_HSS:
        reason = f"Moderate instability (HSS={hss_score:.2f})"
        return TDAOutcome.WARN, reason
    
    # All checks passed
    reason = f"Metrics within acceptable ranges (HSS={hss_score:.2f}, block_rate={block_rate:.2%})"
    return TDAOutcome.PASS, reason


def aggregate_hss_traces(traces: List[HSSTrace]) -> Dict[str, Any]:
    """
    Aggregate HSS traces across multiple cycles.
    
    Computes:
    - Mean stability score
    - Min/max stability
    - Stability trend
    """
    if not traces:
        return {
            "mean_stability": 0.0,
            "min_stability": 0.0,
            "max_stability": 0.0,
            "trend": "unknown",
        }
    
    scores = [t.stability_score for t in traces]
    
    # Compute trend (simple: compare first half vs second half)
    if len(scores) >= 4:
        mid = len(scores) // 2
        first_half_mean = sum(scores[:mid]) / mid
        second_half_mean = sum(scores[mid:]) / (len(scores) - mid)
        
        if second_half_mean > first_half_mean + THRESHOLD_STABILITY_TREND:
            trend = "improving"
        elif second_half_mean < first_half_mean - THRESHOLD_STABILITY_TREND:
            trend = "degrading"
        else:
            trend = "stable"
    else:
        trend = "insufficient_data"
    
    return {
        "mean_stability": sum(scores) / len(scores),
        "min_stability": min(scores),
        "max_stability": max(scores),
        "trend": trend,
        "trace_count": len(traces),
    }


__all__ = [
    "HardGateMode",
    "HSSTrace",
    "PolicyDelta",
    "HardGateDecision",
    "evaluate_hard_gate_decision",
    "aggregate_hss_traces",
]
