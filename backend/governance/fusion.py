"""
Global Governance Fusion Layer (GGFL) — Phase X SHADOW MODE Implementation.

This module implements the unified governance signal fusion layer as specified in:
docs/system_law/Global_Governance_Fusion_PhaseX.md

SHADOW MODE CONTRACT:
1. The GGFL NEVER influences any real governance decisions
2. All outputs are purely observational and logged
3. No control flow depends on fusion results
4. This is reversible via GGFL_ENABLED=false

Signal Precedence (highest to lowest):
1. Identity (SIG-IDN) - Cryptographic integrity is paramount
2. Structure (SIG-STR) - DAG consistency required for correctness
3. Telemetry (SIG-TEL) - System must be operational
4. Replay (SIG-RPL) - Verification integrity
5. Topology (SIG-TOP) - Stability assessment
6. Budget (SIG-BUD) - Resource constraints
7. Metrics (SIG-MET) - Performance indicators
8. Narrative (SIG-NAR) - Strategic guidance (advisory)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

# Schema version for fusion output
FUSION_SCHEMA_VERSION = "1.0.0"

# Default bias toward ALLOW in weighted voting
DEFAULT_ALLOW_BIAS = 10.0

# Staleness tolerance (cycles)
DEFAULT_STALENESS_TOLERANCE = 1


class EscalationLevel(IntEnum):
    """Escalation levels from L0 (nominal) to L5 (emergency)."""
    L0_NOMINAL = 0
    L1_WARNING = 1
    L2_DEGRADED = 2
    L3_CRITICAL = 3
    L4_CONFLICT = 4
    L5_EMERGENCY = 5


class GovernanceAction(str):
    """Governance action types."""
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    HARD_BLOCK = "HARD_BLOCK"
    THROTTLE = "THROTTLE"
    EASE = "EASE"
    PAUSE = "PAUSE"
    WARNING = "WARNING"


# Signal precedence order (index = priority, lower = higher precedence)
SIGNAL_PRECEDENCE: Dict[str, int] = {
    "identity": 1,    # SIG-IDN - highest precedence
    "structure": 2,   # SIG-STR
    "telemetry": 3,   # SIG-TEL
    "replay": 4,      # SIG-RPL
    "topology": 5,    # SIG-TOP
    "budget": 6,      # SIG-BUD
    "metrics": 7,     # SIG-MET
    "narrative": 8,   # SIG-NAR
    "p5_patterns": 9, # SIG-P5P - lowest precedence (SHADOW advisory only)
    "risk": 10,       # SIG-RSK - new SHADOW advisory signal
    "what_if": 11,    # SIG-WIF - What-If hypothetical (SHADOW advisory only)
}


@dataclass
class Recommendation:
    """Individual governance recommendation from a signal."""
    signal_id: str
    action: str
    confidence: float
    reason: str
    priority: int
    field_trigger: Optional[str] = None
    threshold_value: Optional[Any] = None
    actual_value: Optional[Any] = None


@dataclass
class ConflictDetection:
    """Cross-signal consistency violation."""
    rule_id: str
    description: str
    signals_involved: List[str]
    severity: str = "MEDIUM"


@dataclass
class FusionResult:
    """Result of governance signal fusion."""
    decision: str
    is_hard: bool
    primary_reason: str
    block_score: float = 0.0
    allow_score: float = 0.0
    determining_signal: Optional[str] = None


@dataclass
class EscalationState:
    """Current escalation level and state."""
    level: int
    level_name: str
    trigger_reason: str = ""
    consecutive_cycles_at_level: int = 0
    cooldown_remaining: int = 0
    alerts_emitted: List[str] = field(default_factory=list)


@dataclass
class SignalValidation:
    """Result of signal validation."""
    valid: bool
    reason: str = ""


def _is_ggfl_enabled() -> bool:
    """Check if GGFL is enabled via environment."""
    return os.getenv("GGFL_ENABLED", "").lower() in ("1", "true", "yes")


def _get_signal_precedence(signal_id: str) -> int:
    """Get precedence value for a signal (lower = higher precedence)."""
    return SIGNAL_PRECEDENCE.get(signal_id, 999)


def _validate_signal(signal_id: str, signal: Optional[Dict[str, Any]]) -> SignalValidation:
    """
    Validate a single signal for schema compliance and freshness.

    Args:
        signal_id: Signal identifier
        signal: Signal data dict

    Returns:
        SignalValidation with valid=True/False and reason
    """
    if signal is None:
        return SignalValidation(valid=False, reason=f"{signal_id}: missing signal")

    if not isinstance(signal, dict):
        return SignalValidation(valid=False, reason=f"{signal_id}: not a dict")

    # Check for explicit valid flag
    if "valid" in signal and not signal["valid"]:
        return SignalValidation(valid=False, reason=f"{signal_id}: marked invalid")

    return SignalValidation(valid=True)


def _extract_topology_recommendations(signal: Dict[str, Any]) -> List[Recommendation]:
    """Extract governance recommendations from topology signal."""
    recommendations = []

    # Check safe region
    if not signal.get("within_omega", True):
        recommendations.append(Recommendation(
            signal_id="topology",
            action=GovernanceAction.BLOCK,
            confidence=0.9,
            reason="State outside safe region Omega",
            priority=8,
            field_trigger="within_omega",
            actual_value=False,
        ))

    # Check convergence class
    convergence = signal.get("C")
    if convergence == 2:  # DIVERGING
        recommendations.append(Recommendation(
            signal_id="topology",
            action=GovernanceAction.BLOCK,
            confidence=0.85,
            reason="System in DIVERGING convergence class",
            priority=7,
            field_trigger="C",
            actual_value=convergence,
        ))

    # Check RSI
    rho = signal.get("rho")
    if rho is not None and rho < 0.4:
        recommendations.append(Recommendation(
            signal_id="topology",
            action=GovernanceAction.BLOCK,
            confidence=0.8,
            reason=f"RSI below minimum threshold: {rho:.2f} < 0.4",
            priority=7,
            field_trigger="rho",
            threshold_value=0.4,
            actual_value=rho,
        ))

    # Check active CDIs
    active_cdis = signal.get("active_cdis", [])
    if active_cdis:
        recommendations.append(Recommendation(
            signal_id="topology",
            action=GovernanceAction.WARNING,
            confidence=0.7,
            reason=f"Active CDIs detected: {active_cdis}",
            priority=5,
            field_trigger="active_cdis",
            actual_value=active_cdis,
        ))

    # Check invariant violations
    violations = signal.get("invariant_violations", [])
    if violations:
        recommendations.append(Recommendation(
            signal_id="topology",
            action=GovernanceAction.BLOCK,
            confidence=0.9,
            reason=f"Invariant violations: {violations}",
            priority=8,
            field_trigger="invariant_violations",
            actual_value=violations,
        ))

    # Default ALLOW if no issues
    if not recommendations:
        recommendations.append(Recommendation(
            signal_id="topology",
            action=GovernanceAction.ALLOW,
            confidence=0.8,
            reason="Topology nominal",
            priority=5,
        ))

    return recommendations


def _extract_replay_recommendations(signal: Dict[str, Any]) -> List[Recommendation]:
    """Extract governance recommendations from replay signal."""
    recommendations = []

    # Check replay verification
    if not signal.get("replay_verified", True):
        recommendations.append(Recommendation(
            signal_id="replay",
            action=GovernanceAction.BLOCK,
            confidence=0.95,
            reason="Replay verification failed",
            priority=9,
            field_trigger="replay_verified",
            actual_value=False,
        ))

    # Check hash match (security critical)
    if not signal.get("replay_hash_match", True):
        recommendations.append(Recommendation(
            signal_id="replay",
            action=GovernanceAction.HARD_BLOCK,
            confidence=1.0,
            reason="Replay hash mismatch - security critical",
            priority=10,
            field_trigger="replay_hash_match",
            actual_value=False,
        ))

    # Check divergence
    divergence = signal.get("replay_divergence", 0.0)
    if divergence > 0.1:
        recommendations.append(Recommendation(
            signal_id="replay",
            action=GovernanceAction.WARNING,
            confidence=0.7,
            reason=f"Replay divergence elevated: {divergence:.2f}",
            priority=6,
            field_trigger="replay_divergence",
            threshold_value=0.1,
            actual_value=divergence,
        ))

    # Default ALLOW if no issues
    if not recommendations:
        recommendations.append(Recommendation(
            signal_id="replay",
            action=GovernanceAction.ALLOW,
            confidence=0.9,
            reason="Replay verification passed",
            priority=5,
        ))

    return recommendations


def _extract_metrics_recommendations(signal: Dict[str, Any]) -> List[Recommendation]:
    """Extract governance recommendations from metrics signal."""
    recommendations = []

    # Check block rate
    block_rate = signal.get("block_rate", 0.0)
    if block_rate > 0.5:
        recommendations.append(Recommendation(
            signal_id="metrics",
            action=GovernanceAction.EASE,
            confidence=0.7,
            reason=f"High block rate: {block_rate:.2f}",
            priority=4,
            field_trigger="block_rate",
            threshold_value=0.5,
            actual_value=block_rate,
        ))

    # Check abstention rate
    abstention_rate = signal.get("abstention_rate", 0.0)
    if abstention_rate > 0.3:
        recommendations.append(Recommendation(
            signal_id="metrics",
            action=GovernanceAction.WARNING,
            confidence=0.6,
            reason=f"High abstention rate: {abstention_rate:.2f}",
            priority=4,
            field_trigger="abstention_rate",
            threshold_value=0.3,
            actual_value=abstention_rate,
        ))

    # Check queue depth
    queue_depth = signal.get("queue_depth", 0)
    if queue_depth > 1000:
        recommendations.append(Recommendation(
            signal_id="metrics",
            action=GovernanceAction.THROTTLE,
            confidence=0.8,
            reason=f"Queue depth exceeded: {queue_depth}",
            priority=6,
            field_trigger="queue_depth",
            threshold_value=1000,
            actual_value=queue_depth,
        ))

    # Default ALLOW if no issues
    if not recommendations:
        recommendations.append(Recommendation(
            signal_id="metrics",
            action=GovernanceAction.ALLOW,
            confidence=0.7,
            reason="Metrics nominal",
            priority=3,
        ))

    return recommendations


def _extract_budget_recommendations(signal: Dict[str, Any]) -> List[Recommendation]:
    """Extract governance recommendations from budget signal."""
    recommendations = []

    # Check compute budget
    compute_remaining = signal.get("compute_budget_remaining", 1.0)
    if compute_remaining < 0.1:
        recommendations.append(Recommendation(
            signal_id="budget",
            action=GovernanceAction.THROTTLE,
            confidence=0.9,
            reason=f"Compute budget low: {compute_remaining:.1%}",
            priority=7,
            field_trigger="compute_budget_remaining",
            threshold_value=0.1,
            actual_value=compute_remaining,
        ))

    # Check verification quota
    quota_remaining = signal.get("verification_quota_remaining")
    if quota_remaining is not None and quota_remaining == 0:
        recommendations.append(Recommendation(
            signal_id="budget",
            action=GovernanceAction.HARD_BLOCK,
            confidence=1.0,
            reason="Verification quota exhausted",
            priority=10,
            field_trigger="verification_quota_remaining",
            threshold_value=0,
            actual_value=0,
        ))

    # Check budget exhaustion ETA
    eta_cycles = signal.get("budget_exhaustion_eta_cycles")
    if eta_cycles is not None and eta_cycles < 10:
        recommendations.append(Recommendation(
            signal_id="budget",
            action=GovernanceAction.WARNING,
            confidence=0.8,
            reason=f"Budget exhaustion in {eta_cycles} cycles",
            priority=5,
            field_trigger="budget_exhaustion_eta_cycles",
            threshold_value=10,
            actual_value=eta_cycles,
        ))

    # Default ALLOW if no issues
    if not recommendations:
        recommendations.append(Recommendation(
            signal_id="budget",
            action=GovernanceAction.ALLOW,
            confidence=0.8,
            reason="Budget healthy",
            priority=4,
        ))

    return recommendations


def _extract_structure_recommendations(signal: Dict[str, Any]) -> List[Recommendation]:
    """Extract governance recommendations from structure signal."""
    recommendations = []

    # Check DAG coherence
    if not signal.get("dag_coherent", True):
        recommendations.append(Recommendation(
            signal_id="structure",
            action=GovernanceAction.HARD_BLOCK,
            confidence=1.0,
            reason="DAG coherence check failed",
            priority=10,
            field_trigger="dag_coherent",
            actual_value=False,
        ))

    # Check cycle detection (critical)
    if signal.get("cycle_detected", False):
        recommendations.append(Recommendation(
            signal_id="structure",
            action=GovernanceAction.HARD_BLOCK,
            confidence=1.0,
            reason="Cyclic dependency detected in DAG",
            priority=10,
            field_trigger="cycle_detected",
            actual_value=True,
        ))

    # Check orphan count
    orphan_count = signal.get("orphan_count", 0)
    if orphan_count > 100:
        recommendations.append(Recommendation(
            signal_id="structure",
            action=GovernanceAction.WARNING,
            confidence=0.6,
            reason=f"High orphan count: {orphan_count}",
            priority=4,
            field_trigger="orphan_count",
            threshold_value=100,
            actual_value=orphan_count,
        ))

    # Check min cut capacity
    min_cut = signal.get("min_cut_capacity")
    if min_cut is not None and min_cut < 0.1:
        recommendations.append(Recommendation(
            signal_id="structure",
            action=GovernanceAction.BLOCK,
            confidence=0.8,
            reason=f"Low min-cut capacity: {min_cut:.2f}",
            priority=7,
            field_trigger="min_cut_capacity",
            threshold_value=0.1,
            actual_value=min_cut,
        ))

    # Default ALLOW if no issues
    if not recommendations:
        recommendations.append(Recommendation(
            signal_id="structure",
            action=GovernanceAction.ALLOW,
            confidence=0.9,
            reason="DAG structure healthy",
            priority=5,
        ))

    return recommendations


def _extract_telemetry_recommendations(signal: Dict[str, Any]) -> List[Recommendation]:
    """Extract governance recommendations from telemetry signal."""
    recommendations = []

    # Check Lean verifier health
    if not signal.get("lean_healthy", True):
        recommendations.append(Recommendation(
            signal_id="telemetry",
            action=GovernanceAction.HARD_BLOCK,
            confidence=1.0,
            reason="Lean verifier unhealthy",
            priority=10,
            field_trigger="lean_healthy",
            actual_value=False,
        ))

    # Check database health
    if not signal.get("db_healthy", True):
        recommendations.append(Recommendation(
            signal_id="telemetry",
            action=GovernanceAction.HARD_BLOCK,
            confidence=1.0,
            reason="Database connection unhealthy",
            priority=10,
            field_trigger="db_healthy",
            actual_value=False,
        ))

    # Check Redis health
    if not signal.get("redis_healthy", True):
        recommendations.append(Recommendation(
            signal_id="telemetry",
            action=GovernanceAction.HARD_BLOCK,
            confidence=1.0,
            reason="Redis connection unhealthy",
            priority=10,
            field_trigger="redis_healthy",
            actual_value=False,
        ))

    # Check worker count
    worker_count = signal.get("worker_count")
    if worker_count is not None and worker_count == 0:
        recommendations.append(Recommendation(
            signal_id="telemetry",
            action=GovernanceAction.HARD_BLOCK,
            confidence=1.0,
            reason="No active workers",
            priority=10,
            field_trigger="worker_count",
            actual_value=0,
        ))

    # Check error rate
    error_rate = signal.get("error_rate", 0.0)
    if error_rate > 0.1:
        recommendations.append(Recommendation(
            signal_id="telemetry",
            action=GovernanceAction.WARNING,
            confidence=0.7,
            reason=f"Elevated error rate: {error_rate:.1%}",
            priority=5,
            field_trigger="error_rate",
            threshold_value=0.1,
            actual_value=error_rate,
        ))

    # Default ALLOW if no issues
    if not recommendations:
        recommendations.append(Recommendation(
            signal_id="telemetry",
            action=GovernanceAction.ALLOW,
            confidence=0.95,
            reason="All systems operational",
            priority=5,
        ))

    return recommendations


def _extract_identity_recommendations(signal: Dict[str, Any]) -> List[Recommendation]:
    """Extract governance recommendations from identity signal."""
    recommendations = []

    # Check block hash validity (security critical)
    if not signal.get("block_hash_valid", True):
        recommendations.append(Recommendation(
            signal_id="identity",
            action=GovernanceAction.HARD_BLOCK,
            confidence=1.0,
            reason="Block hash validation failed - security critical",
            priority=10,
            field_trigger="block_hash_valid",
            actual_value=False,
        ))

    # Check chain continuity
    if not signal.get("chain_continuous", True):
        recommendations.append(Recommendation(
            signal_id="identity",
            action=GovernanceAction.HARD_BLOCK,
            confidence=1.0,
            reason="Chain continuity broken - security critical",
            priority=10,
            field_trigger="chain_continuous",
            actual_value=False,
        ))

    # Check dual root consistency
    if not signal.get("dual_root_consistent", True):
        recommendations.append(Recommendation(
            signal_id="identity",
            action=GovernanceAction.HARD_BLOCK,
            confidence=1.0,
            reason="Dual-root attestation inconsistent",
            priority=10,
            field_trigger="dual_root_consistent",
            actual_value=False,
        ))

    # Check Merkle root validity
    if not signal.get("merkle_root_valid", True):
        recommendations.append(Recommendation(
            signal_id="identity",
            action=GovernanceAction.HARD_BLOCK,
            confidence=1.0,
            reason="Merkle root validation failed",
            priority=10,
            field_trigger="merkle_root_valid",
            actual_value=False,
        ))

    # Check signature validity
    if not signal.get("signature_valid", True):
        recommendations.append(Recommendation(
            signal_id="identity",
            action=GovernanceAction.HARD_BLOCK,
            confidence=1.0,
            reason="Block signature invalid",
            priority=10,
            field_trigger="signature_valid",
            actual_value=False,
        ))

    # PQ attestation is warning only (not mandatory yet)
    if not signal.get("pq_attestation_valid", True):
        recommendations.append(Recommendation(
            signal_id="identity",
            action=GovernanceAction.WARNING,
            confidence=0.6,
            reason="PQ attestation invalid (advisory)",
            priority=3,
            field_trigger="pq_attestation_valid",
            actual_value=False,
        ))

    # Default ALLOW if no issues
    if not recommendations:
        recommendations.append(Recommendation(
            signal_id="identity",
            action=GovernanceAction.ALLOW,
            confidence=1.0,
            reason="Cryptographic identity verified",
            priority=5,
        ))

    return recommendations


def _extract_narrative_recommendations(signal: Dict[str, Any]) -> List[Recommendation]:
    """Extract governance recommendations from narrative signal."""
    recommendations = []

    # Check curriculum health
    curriculum_health = signal.get("curriculum_health")
    if curriculum_health == "CRITICAL":
        recommendations.append(Recommendation(
            signal_id="narrative",
            action=GovernanceAction.PAUSE,
            confidence=0.8,
            reason="Curriculum health CRITICAL",
            priority=6,
            field_trigger="curriculum_health",
            actual_value=curriculum_health,
        ))

    # Check drift detection
    if signal.get("drift_detected", False):
        recommendations.append(Recommendation(
            signal_id="narrative",
            action=GovernanceAction.WARNING,
            confidence=0.6,
            reason="Curriculum drift detected",
            priority=4,
            field_trigger="drift_detected",
            actual_value=True,
        ))

    # Check narrative coherence
    coherence = signal.get("narrative_coherence")
    if coherence is not None and coherence < 0.5:
        recommendations.append(Recommendation(
            signal_id="narrative",
            action=GovernanceAction.WARNING,
            confidence=0.5,
            reason=f"Low narrative coherence: {coherence:.2f}",
            priority=3,
            field_trigger="narrative_coherence",
            threshold_value=0.5,
            actual_value=coherence,
        ))

    # Default ALLOW (narrative is advisory)
    if not recommendations:
        recommendations.append(Recommendation(
            signal_id="narrative",
            action=GovernanceAction.ALLOW,
            confidence=0.6,
            reason="Narrative nominal (advisory)",
            priority=2,
        ))

    return recommendations


def _extract_p5_patterns_recommendations(signal: Dict[str, Any]) -> List[Recommendation]:
    """
    Extract governance recommendations from P5 patterns signal.

    SHADOW MODE CONTRACT:
    - P5 patterns are OBSERVATIONAL ONLY
    - Never produces HARD_BLOCK on its own (solo)
    - Only contributes to escalation when combined with other conflicts
    - All recommendations are advisory

    Patterns and their escalation contributions:
    - DRIFT: WARNING (calibration drift)
    - NOISE_AMPLIFICATION: WARNING (twin over-fitting)
    - PHASE_LAG: WARNING (temporal misalignment)
    - ATTRACTOR_MISS: BLOCK (soft) when streak >= 3
    - TRANSIENT_MISS: BLOCK (soft) when streak >= 5
    - STRUCTURAL_BREAK: BLOCK (soft) when streak >= 2 - NEVER HARD_BLOCK alone
    """
    recommendations = []

    # Get pattern classification
    pattern = signal.get("final_pattern") or signal.get("divergence_pattern")
    streak = signal.get("final_streak") or signal.get("divergence_pattern_streak", 0)
    recalibration_triggered = signal.get("recalibration_triggered", False)

    if not pattern or pattern == "NOMINAL":
        # No pattern detected - nominal
        recommendations.append(Recommendation(
            signal_id="p5_patterns",
            action=GovernanceAction.ALLOW,
            confidence=0.7,
            reason="P5 pattern classification nominal (SHADOW)",
            priority=1,
        ))
        return recommendations

    # WARNING-level patterns (always warning, never block)
    warning_patterns = {"DRIFT", "NOISE_AMPLIFICATION", "PHASE_LAG", "UNCLASSIFIED"}
    if pattern in warning_patterns:
        recommendations.append(Recommendation(
            signal_id="p5_patterns",
            action=GovernanceAction.WARNING,
            confidence=0.6,
            reason=f"P5 {pattern} detected (streak {streak}) - advisory only",
            priority=2,
            field_trigger="divergence_pattern",
            actual_value=pattern,
        ))

    # SOFT BLOCK patterns - only when streak threshold met
    # Note: NEVER HARD_BLOCK - P5 patterns cannot solo-block
    elif pattern == "ATTRACTOR_MISS" and streak >= 3:
        recommendations.append(Recommendation(
            signal_id="p5_patterns",
            action=GovernanceAction.BLOCK,  # Soft block, NOT hard
            confidence=0.7,
            reason=f"P5 ATTRACTOR_MISS streak {streak} >= 3: twin misaligned (soft block)",
            priority=3,
            field_trigger="divergence_pattern",
            threshold_value=3,
            actual_value=streak,
        ))

    elif pattern == "TRANSIENT_MISS" and streak >= 5:
        recommendations.append(Recommendation(
            signal_id="p5_patterns",
            action=GovernanceAction.BLOCK,  # Soft block, NOT hard
            confidence=0.65,
            reason=f"P5 TRANSIENT_MISS streak {streak} >= 5: transient fidelity concern (soft block)",
            priority=3,
            field_trigger="divergence_pattern",
            threshold_value=5,
            actual_value=streak,
        ))

    elif pattern == "STRUCTURAL_BREAK" and streak >= 2:
        # IMPORTANT: STRUCTURAL_BREAK produces SOFT block only
        # It requires combination with other signals to escalate to HARD_BLOCK
        recommendations.append(Recommendation(
            signal_id="p5_patterns",
            action=GovernanceAction.BLOCK,  # Soft block, NOT hard - cannot solo-block
            confidence=0.8,
            reason=f"P5 STRUCTURAL_BREAK streak {streak} >= 2: regime change (soft block, requires conflict)",
            priority=4,
            field_trigger="divergence_pattern",
            threshold_value=2,
            actual_value=streak,
        ))

    else:
        # Pattern detected but streak not met - warning
        recommendations.append(Recommendation(
            signal_id="p5_patterns",
            action=GovernanceAction.WARNING,
            confidence=0.5,
            reason=f"P5 {pattern} detected (streak {streak}) - below threshold",
            priority=2,
            field_trigger="divergence_pattern",
            actual_value=pattern,
        ))

    # Add warning if recalibration was triggered
    if recalibration_triggered:
        recommendations.append(Recommendation(
            signal_id="p5_patterns",
            action=GovernanceAction.WARNING,
            confidence=0.7,
            reason="P5 recalibration triggered (SHADOW observation)",
            priority=3,
            field_trigger="recalibration_triggered",
            actual_value=True,
        ))

    return recommendations


def _extract_risk_recommendations(signal: Dict[str, Any]) -> List[Recommendation]:
    """
    Extract governance recommendations from the risk tile signal.

    SHADOW MODE CONTRACT:
    - Risk signal is OBSERVATIONAL ONLY
    - Never produces HARD_BLOCK
    - Only contributes WARNING or soft BLOCK recommendations
    """
    recommendations = []
    risk_band = signal.get("risk_band")

    if risk_band == "CRITICAL":
        recommendations.append(Recommendation(
            signal_id="risk",
            action=GovernanceAction.BLOCK, # Soft block
            confidence=0.8,
            reason=f"Overall risk band is CRITICAL (SHADOW)",
            priority=4,
            field_trigger="risk_band",
            actual_value=risk_band,
        ))
    elif risk_band == "HIGH":
        recommendations.append(Recommendation(
            signal_id="risk",
            action=GovernanceAction.WARNING,
            confidence=0.7,
            reason=f"Overall risk band is HIGH (SHADOW)",
            priority=3,
            field_trigger="risk_band",
            actual_value=risk_band,
        ))
    elif risk_band == "MEDIUM":
        recommendations.append(Recommendation(
            signal_id="risk",
            action=GovernanceAction.WARNING,
            confidence=0.5,
            reason=f"Overall risk band is MEDIUM (SHADOW)",
            priority=2,
            field_trigger="risk_band",
            actual_value=risk_band,
        ))
    else: # LOW or unknown
        recommendations.append(Recommendation(
            signal_id="risk",
            action=GovernanceAction.ALLOW,
            confidence=0.8,
            reason="Overall risk band is LOW or nominal (SHADOW)",
            priority=1,
        ))

    return recommendations


def _extract_recommendations(signal_id: str, signal: Dict[str, Any]) -> List[Recommendation]:
    """Extract recommendations from a signal based on its type."""
    extractors = {
        "topology": _extract_topology_recommendations,
        "replay": _extract_replay_recommendations,
        "metrics": _extract_metrics_recommendations,
        "budget": _extract_budget_recommendations,
        "structure": _extract_structure_recommendations,
        "telemetry": _extract_telemetry_recommendations,
        "identity": _extract_identity_recommendations,
        "narrative": _extract_narrative_recommendations,
        "p5_patterns": _extract_p5_patterns_recommendations,
        "risk": _extract_risk_recommendations, # Hook for the new risk signal
    }

    extractor = extractors.get(signal_id)
    if extractor:
        return extractor(signal)

    # Unknown signal type - return warning
    return [Recommendation(
        signal_id=signal_id,
        action=GovernanceAction.WARNING,
        confidence=0.5,
        reason=f"Unknown signal type: {signal_id}",
        priority=1,
    )]


def _detect_cross_signal_conflicts(signals: Dict[str, Dict[str, Any]]) -> List[ConflictDetection]:
    """
    Detect cross-signal consistency violations.

    Rules:
    - CSC-001: chain_continuous=false AND dag_coherent=true → conflict
    - CSC-002: within_omega=true AND block_rate>0.5 → conflict
    - CSC-003: lean_healthy=true AND replay_verified=false → conflict
    - CSC-004: verification_quota_remaining>0 AND worker_count=0 → conflict
    """
    conflicts = []

    identity = signals.get("identity", {})
    structure = signals.get("structure", {})
    topology = signals.get("topology", {})
    metrics = signals.get("metrics", {})
    telemetry = signals.get("telemetry", {})
    replay = signals.get("replay", {})
    budget = signals.get("budget", {})

    # CSC-001: Chain continuity vs DAG coherence
    if not identity.get("chain_continuous", True) and structure.get("dag_coherent", False):
        conflicts.append(ConflictDetection(
            rule_id="CSC-001",
            description="Chain discontinuous but DAG marked coherent - possible data inconsistency",
            signals_involved=["identity", "structure"],
            severity="HIGH",
        ))

    # CSC-002: Topology health vs metrics block rate
    if topology.get("within_omega", False) and metrics.get("block_rate", 0.0) > 0.5:
        conflicts.append(ConflictDetection(
            rule_id="CSC-002",
            description="Within safe region but high block rate - metrics disagree with topology",
            signals_involved=["topology", "metrics"],
            severity="MEDIUM",
        ))

    # CSC-003: Lean healthy vs replay failed
    if telemetry.get("lean_healthy", False) and not replay.get("replay_verified", True):
        conflicts.append(ConflictDetection(
            rule_id="CSC-003",
            description="Lean verifier healthy but replay verification failed",
            signals_involved=["telemetry", "replay"],
            severity="HIGH",
        ))

    # CSC-004: Budget available vs no workers
    quota = budget.get("verification_quota_remaining")
    workers = telemetry.get("worker_count")
    if quota is not None and quota > 0 and workers is not None and workers == 0:
        conflicts.append(ConflictDetection(
            rule_id="CSC-004",
            description="Verification quota available but no workers running",
            signals_involved=["budget", "telemetry"],
            severity="MEDIUM",
        ))

    # P5 Pattern Cross-Signal Rules (SHADOW MODE - observational only)
    # These conflicts are advisory and only escalate when combined with other issues
    p5_patterns = signals.get("p5_patterns", {})
    p5_pattern = p5_patterns.get("final_pattern") or p5_patterns.get("divergence_pattern")
    p5_streak = p5_patterns.get("final_streak") or p5_patterns.get("divergence_pattern_streak", 0)

    # CSC-P5-001: STRUCTURAL_BREAK + low min_cut_capacity (DAG tension)
    # Reference: docs/system_law/GGFL_P5_Pattern_Test_Plan.md
    min_cut = structure.get("min_cut_capacity")
    if p5_pattern == "STRUCTURAL_BREAK" and p5_streak >= 2 and min_cut is not None and min_cut < 0.2:
        conflicts.append(ConflictDetection(
            rule_id="CSC-P5-001",
            description="P5 STRUCTURAL_BREAK with DAG tension (min_cut < 0.2): regime change under stress",
            signals_involved=["p5_patterns", "structure"],
            severity="CRITICAL",
        ))

    # CSC-P5-002: DRIFT + replay_divergence elevated
    # Pattern drift combined with replay verification concerns
    replay_div = replay.get("replay_divergence", 0.0)
    if p5_pattern == "DRIFT" and p5_streak >= 3 and replay_div > 0.1:
        conflicts.append(ConflictDetection(
            rule_id="CSC-P5-002",
            description="P5 DRIFT with elevated replay divergence: twin calibration + verification concerns",
            signals_involved=["p5_patterns", "replay"],
            severity="MEDIUM",
        ))

    # CSC-P5-003: ATTRACTOR_MISS + within_omega=true
    # Twin fails to track safe region despite real system being safe
    p5_twin = topology.get("p5_twin", {})
    attractor_miss_rate = p5_twin.get("attractor_miss_rate", 0.0)
    if p5_pattern == "ATTRACTOR_MISS" and topology.get("within_omega", False) and attractor_miss_rate > 0.2:
        conflicts.append(ConflictDetection(
            rule_id="CSC-P5-003",
            description="P5 ATTRACTOR_MISS while system in safe region: twin fundamentally misaligned",
            signals_involved=["p5_patterns", "topology"],
            severity="HIGH",
        ))

    # CSC-P5-004: NOISE_AMPLIFICATION + high error_rate
    # Twin oversensitivity combined with system errors
    error_rate = telemetry.get("error_rate", 0.0)
    if p5_pattern == "NOISE_AMPLIFICATION" and p5_streak >= 5 and error_rate > 0.05:
        conflicts.append(ConflictDetection(
            rule_id="CSC-P5-004",
            description="P5 NOISE_AMPLIFICATION with elevated error rate: twin + system instability",
            signals_involved=["p5_patterns", "telemetry"],
            severity="MEDIUM",
        ))

    return conflicts


def _compute_fusion_result(
    recommendations: List[Recommendation],
    allow_bias: float = DEFAULT_ALLOW_BIAS,
) -> FusionResult:
    """
    Compute fusion result from recommendations using weighted voting.

    Args:
        recommendations: All extracted recommendations
        allow_bias: Bias toward ALLOW in voting

    Returns:
        FusionResult with decision and scores
    """
    # Check for HARD_BLOCK first (unconditional)
    hard_blocks = [r for r in recommendations if r.action == GovernanceAction.HARD_BLOCK]
    if hard_blocks:
        # Sort by precedence (lower = higher precedence)
        hard_blocks.sort(key=lambda r: _get_signal_precedence(r.signal_id))
        primary = hard_blocks[0]
        return FusionResult(
            decision=GovernanceAction.BLOCK,
            is_hard=True,
            primary_reason=primary.reason,
            determining_signal=primary.signal_id,
        )

    # Weighted voting for soft decisions
    blocks = [r for r in recommendations if r.action == GovernanceAction.BLOCK]
    allows = [r for r in recommendations if r.action == GovernanceAction.ALLOW]

    block_score = sum(r.confidence * r.priority for r in blocks)
    allow_score = sum(r.confidence * r.priority for r in allows) + allow_bias

    if block_score > allow_score:
        # Sort blocks by weighted score
        blocks.sort(key=lambda r: r.confidence * r.priority, reverse=True)
        primary = blocks[0]
        return FusionResult(
            decision=GovernanceAction.BLOCK,
            is_hard=False,
            primary_reason=primary.reason,
            block_score=block_score,
            allow_score=allow_score,
            determining_signal=primary.signal_id,
        )

    return FusionResult(
        decision=GovernanceAction.ALLOW,
        is_hard=False,
        primary_reason="All signals nominal",
        block_score=block_score,
        allow_score=allow_score,
    )


def _compute_escalation_level(
    recommendations: List[Recommendation],
    conflicts: List[ConflictDetection],
) -> EscalationState:
    """
    Compute escalation level from recommendations and conflicts.

    Levels:
    - L0 NOMINAL: All signals healthy
    - L1 WARNING: Any WARNING recommendation
    - L2 DEGRADED: Any BLOCK (soft) recommendation
    - L3 CRITICAL: Any HARD_BLOCK recommendation
    - L4 CONFLICT: Cross-signal consistency failure
    - L5 EMERGENCY: Multiple L3 conditions OR identity/structure failure
    """
    hard_blocks = [r for r in recommendations if r.action == GovernanceAction.HARD_BLOCK]
    soft_blocks = [r for r in recommendations if r.action == GovernanceAction.BLOCK]
    warnings = [r for r in recommendations if r.action == GovernanceAction.WARNING]

    # L5: Identity or structure HARD_BLOCK, or multiple HARD_BLOCKs
    identity_or_structure_hard = any(
        r.signal_id in ("identity", "structure") and r.action == GovernanceAction.HARD_BLOCK
        for r in recommendations
    )
    if identity_or_structure_hard:
        return EscalationState(
            level=EscalationLevel.L5_EMERGENCY,
            level_name="L5_EMERGENCY",
            trigger_reason="Identity or structure failure",
        )

    if len(hard_blocks) >= 2:
        return EscalationState(
            level=EscalationLevel.L5_EMERGENCY,
            level_name="L5_EMERGENCY",
            trigger_reason=f"Multiple HARD_BLOCK conditions: {len(hard_blocks)}",
        )

    # L4: Cross-signal conflict
    if conflicts:
        return EscalationState(
            level=EscalationLevel.L4_CONFLICT,
            level_name="L4_CONFLICT",
            trigger_reason=f"Cross-signal conflicts detected: {[c.rule_id for c in conflicts]}",
        )

    # L3: Any HARD_BLOCK
    if hard_blocks:
        return EscalationState(
            level=EscalationLevel.L3_CRITICAL,
            level_name="L3_CRITICAL",
            trigger_reason=hard_blocks[0].reason,
        )

    # L2: Multiple soft blocks
    if len(soft_blocks) >= 2:
        return EscalationState(
            level=EscalationLevel.L2_DEGRADED,
            level_name="L2_DEGRADED",
            trigger_reason=f"Multiple BLOCK recommendations: {len(soft_blocks)}",
        )

    # L1: Any warning
    if warnings:
        return EscalationState(
            level=EscalationLevel.L1_WARNING,
            level_name="L1_WARNING",
            trigger_reason=warnings[0].reason,
        )

    # L0: Nominal
    return EscalationState(
        level=EscalationLevel.L0_NOMINAL,
        level_name="L0_NOMINAL",
        trigger_reason="All systems nominal",
    )


def build_global_alignment_view(
    *,
    topology: Optional[Dict[str, Any]] = None,
    replay: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    budget: Optional[Dict[str, Any]] = None,
    structure: Optional[Dict[str, Any]] = None,
    telemetry: Optional[Dict[str, Any]] = None,
    identity: Optional[Dict[str, Any]] = None,
    narrative: Optional[Dict[str, Any]] = None,
    p5_patterns: Optional[Dict[str, Any]] = None,
    risk: Optional[Dict[str, Any]] = None, # New optional risk signal
    cycle: int = 0,
    allow_bias: float = DEFAULT_ALLOW_BIAS,
) -> Dict[str, Any]:
    """
    Build the unified global alignment view from all governance signals.

    SHADOW MODE CONTRACT:
    - This function is purely observational
    - Output NEVER influences real governance decisions
    - All outputs are for logging and analysis only

    Args:
        topology: SIG-TOP topology signal
        replay: SIG-RPL replay signal
        metrics: SIG-MET metrics signal
        budget: SIG-BUD budget signal
        structure: SIG-STR structure signal
        telemetry: SIG-TEL telemetry signal
        identity: SIG-IDN identity signal
        narrative: SIG-NAR narrative signal
        p5_patterns: SIG-P5P P5 pattern classification (SHADOW advisory only)
        cycle: Current cycle number
        allow_bias: Bias toward ALLOW in weighted voting

    Returns:
        Unified governance signal envelope conforming to governance_signal_unified.schema.json
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    # Collect all signals
    signals_raw = {
        "topology": topology,
        "replay": replay,
        "metrics": metrics,
        "budget": budget,
        "structure": structure,
        "telemetry": telemetry,
        "identity": identity,
        "narrative": narrative,
        "p5_patterns": p5_patterns,
        "risk": risk,
    }

    # Validate signals and build validated dict
    signals_validated: Dict[str, Dict[str, Any]] = {}
    validation_results: Dict[str, SignalValidation] = {}

    for signal_id, signal in signals_raw.items():
        validation = _validate_signal(signal_id, signal)
        validation_results[signal_id] = validation
        if validation.valid and signal is not None:
            signals_validated[signal_id] = signal

    # Extract recommendations from all valid signals
    all_recommendations: List[Recommendation] = []
    for signal_id, signal in signals_validated.items():
        recs = _extract_recommendations(signal_id, signal)
        all_recommendations.extend(recs)

    # Add validation failure warnings
    for signal_id, validation in validation_results.items():
        if not validation.valid:
            all_recommendations.append(Recommendation(
                signal_id=signal_id,
                action=GovernanceAction.WARNING,
                confidence=1.0,
                reason=validation.reason,
                priority=5,
            ))

    # Detect cross-signal conflicts
    conflicts = _detect_cross_signal_conflicts(signals_validated)

    # Compute fusion result
    fusion_result = _compute_fusion_result(all_recommendations, allow_bias)

    # Compute escalation level
    escalation = _compute_escalation_level(all_recommendations, conflicts)

    # Build signal summary
    signal_summary = {}
    for signal_id in signals_raw.keys():
        signal_recs = [r for r in all_recommendations if r.signal_id == signal_id]
        has_block = any(r.action in (GovernanceAction.BLOCK, GovernanceAction.HARD_BLOCK) for r in signal_recs)
        has_warning = any(r.action == GovernanceAction.WARNING for r in signal_recs)

        status = "healthy"
        if has_block:
            status = "unhealthy"
        elif has_warning:
            status = "degraded"
        elif not validation_results[signal_id].valid:
            status = "missing"

        signal_summary[signal_id] = {
            "status": status,
            "recommendations": len(signal_recs),
        }

    # Build signal dicts with validation status
    signals_output = {}
    for signal_id, signal in signals_raw.items():
        if signal is not None:
            signal_output = dict(signal)
            signal_output["signal_id"] = f"SIG-{signal_id.upper()[:3]}"
            signal_output["timestamp"] = timestamp
            signal_output["valid"] = validation_results[signal_id].valid
            signals_output[signal_id] = signal_output
        else:
            signals_output[signal_id] = {
                "signal_id": f"SIG-{signal_id.upper()[:3]}",
                "timestamp": timestamp,
                "valid": False,
            }

    # Build final envelope
    envelope = {
        "schema_version": FUSION_SCHEMA_VERSION,
        "timestamp": timestamp,
        "cycle": cycle,
        "mode": "shadow",
        "signals": signals_output,
        "fusion_result": {
            "decision": fusion_result.decision,
            "is_hard": fusion_result.is_hard,
            "primary_reason": fusion_result.primary_reason,
            "block_score": fusion_result.block_score,
            "allow_score": fusion_result.allow_score,
            "determining_signal": fusion_result.determining_signal,
        },
        "escalation": {
            "level": escalation.level,
            "level_name": escalation.level_name,
            "trigger_reason": escalation.trigger_reason,
            "consecutive_cycles_at_level": escalation.consecutive_cycles_at_level,
            "cooldown_remaining": escalation.cooldown_remaining,
            "alerts_emitted": escalation.alerts_emitted,
        },
        "conflict_detections": [
            {
                "rule_id": c.rule_id,
                "description": c.description,
                "signals_involved": c.signals_involved,
                "severity": c.severity,
            }
            for c in conflicts
        ],
        "recommendations": [
            {
                "signal_id": r.signal_id,
                "action": r.action,
                "confidence": r.confidence,
                "reason": r.reason,
                "priority": r.priority,
                "field_trigger": r.field_trigger,
                "threshold_value": r.threshold_value,
                "actual_value": r.actual_value,
            }
            for r in all_recommendations
        ],
        "metadata": {
            "fusion_latency_ms": 0,  # Would be measured in real implementation
            "signals_received": sum(1 for s in signals_raw.values() if s is not None),
            "signals_valid": len(signals_validated),
        },
        "signal_summary": signal_summary,
        "headline": _generate_headline(fusion_result, escalation, conflicts),
    }

    return envelope


def _generate_headline(
    fusion_result: FusionResult,
    escalation: EscalationState,
    conflicts: List[ConflictDetection],
) -> str:
    """Generate human-readable headline for the fusion result."""
    if escalation.level == EscalationLevel.L5_EMERGENCY:
        return f"EMERGENCY: {escalation.trigger_reason}"

    if escalation.level == EscalationLevel.L4_CONFLICT:
        return f"CONFLICT: Cross-signal consistency violations detected"

    if escalation.level == EscalationLevel.L3_CRITICAL:
        return f"CRITICAL: {escalation.trigger_reason}"

    if fusion_result.is_hard:
        return f"HARD BLOCK: {fusion_result.primary_reason}"

    if fusion_result.decision == GovernanceAction.BLOCK:
        return f"BLOCK: {fusion_result.primary_reason}"

    if escalation.level == EscalationLevel.L2_DEGRADED:
        return "Governance degraded; multiple concerns detected"

    if escalation.level == EscalationLevel.L1_WARNING:
        return "Governance nominal with warnings"

    return "All signals nominal; governance fusion ALLOW"


# =============================================================================
# WHAT-IF GGFL ADAPTER
# =============================================================================

@dataclass
class WhatIfAlignmentSignal:
    """
    What-If signal for GGFL alignment view.

    SHADOW MODE CONTRACT:
    - Mode is always HYPOTHETICAL (never SHADOW)
    - This signal is purely advisory
    - conflict is always False (What-If cannot conflict)
    - No enforcement action taken
    """
    status: str  # "ok" or "warn"
    hypothetical_block_rate: float
    top_blocking_gate: Optional[str]
    drivers: List[str]
    conflict: bool = False  # Always False for What-If
    mode: str = "HYPOTHETICAL"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_id": "SIG-WIF",
            "status": self.status,
            "hypothetical_block_rate": round(self.hypothetical_block_rate, 4),
            "top_blocking_gate": self.top_blocking_gate,
            "drivers": self.drivers,
            "conflict": self.conflict,
            "mode": self.mode,
        }


def what_if_for_alignment_view(
    status_signal_or_report: Dict[str, Any],
) -> WhatIfAlignmentSignal:
    """
    Convert What-If status signal or report to GGFL alignment view format.

    SHADOW MODE CONTRACT:
    - Mode is always HYPOTHETICAL
    - Status is "warn" if hypothetical_block_rate > 0
    - Drivers include top blocking gate and block rate
    - conflict is always False (What-If reports don't conflict)

    Args:
        status_signal_or_report: Either:
            - WhatIfStatusSignal.to_dict() output
            - Full WhatIfReport.to_dict() output
            - Dict with summary.hypothetical_block_rate

    Returns:
        WhatIfAlignmentSignal for GGFL consumption
    """
    # Extract block rate from various possible structures
    if "summary" in status_signal_or_report:
        # Full report format
        summary = status_signal_or_report["summary"]
        block_rate = summary.get("hypothetical_block_rate", 0.0)
        gate_dist = summary.get("blocking_gate_distribution", {})
    else:
        # Status signal format
        block_rate = status_signal_or_report.get("hypothetical_block_rate", 0.0)
        gate_dist = status_signal_or_report.get("blocking_gate_distribution", {})

    # Determine status: warn if any hypothetical blocks
    status = "warn" if block_rate > 0 else "ok"

    # Find top blocking gate
    top_gate = None
    if gate_dist:
        top_gate = max(gate_dist.items(), key=lambda x: x[1])[0]

    # Build drivers list
    drivers: List[str] = []

    if block_rate > 0:
        drivers.append(f"hypothetical_block_rate={block_rate:.2%}")

    if top_gate:
        gate_count = gate_dist.get(top_gate, 0)
        drivers.append(f"top_blocking_gate={top_gate} ({gate_count} blocks)")

    # Add mode driver
    mode = status_signal_or_report.get("mode", "HYPOTHETICAL")
    if mode != "HYPOTHETICAL":
        drivers.append(f"mode={mode} (expected HYPOTHETICAL)")

    return WhatIfAlignmentSignal(
        status=status,
        hypothetical_block_rate=block_rate,
        top_blocking_gate=top_gate,
        drivers=drivers,
        conflict=False,  # Always False - What-If cannot conflict
        mode=mode,
    )


def _extract_what_if_recommendations(signal: Dict[str, Any]) -> List[Recommendation]:
    """
    Extract governance recommendations from What-If signal.

    SHADOW MODE CONTRACT:
    - What-If is purely advisory (lowest precedence)
    - Never produces BLOCK or HARD_BLOCK
    - Only produces WARNING when hypothetical_block_rate > 0
    """
    recommendations = []

    block_rate = signal.get("hypothetical_block_rate", 0.0)
    top_gate = signal.get("top_blocking_gate")
    mode = signal.get("mode", "HYPOTHETICAL")

    # Mode validation warning
    if mode != "HYPOTHETICAL":
        recommendations.append(Recommendation(
            signal_id="what_if",
            action=GovernanceAction.WARNING,
            confidence=0.9,
            reason=f"What-If mode is '{mode}', expected 'HYPOTHETICAL'",
            priority=2,
            field_trigger="mode",
            actual_value=mode,
        ))

    # Warning if any hypothetical blocks
    if block_rate > 0:
        reason = f"What-If analysis shows {block_rate:.1%} hypothetical block rate"
        if top_gate:
            reason += f" (top gate: {top_gate})"

        recommendations.append(Recommendation(
            signal_id="what_if",
            action=GovernanceAction.WARNING,
            confidence=0.6,
            reason=reason,
            priority=2,  # Low priority - advisory only
            field_trigger="hypothetical_block_rate",
            threshold_value=0.0,
            actual_value=block_rate,
        ))

    # Default ALLOW (What-If is advisory)
    if not recommendations:
        recommendations.append(Recommendation(
            signal_id="what_if",
            action=GovernanceAction.ALLOW,
            confidence=0.7,
            reason="What-If analysis shows no hypothetical blocks (SHADOW)",
            priority=1,
        ))

    return recommendations


__all__ = [
    "build_global_alignment_view",
    "EscalationLevel",
    "GovernanceAction",
    "SIGNAL_PRECEDENCE",
    "FUSION_SCHEMA_VERSION",
    "Recommendation",
    "ConflictDetection",
    "FusionResult",
    "EscalationState",
    # What-If GGFL Adapter
    "WhatIfAlignmentSignal",
    "what_if_for_alignment_view",
]
