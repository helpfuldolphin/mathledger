"""
Telemetry P4 Integration — Bridging Telemetry to First Light and Evidence

Phase X: Telemetry Canonical Interface

This module provides integration functions that connect the TelemetryGovernanceSignal
to Phase X P4 calibration reports and evidence packs.

SHADOW MODE CONTRACT:
- All integrations are OBSERVATIONAL ONLY
- No modification of real runner execution
- All outputs are for logging and analysis only

See: docs/system_law/Telemetry_PhaseX_Contract.md
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from backend.telemetry.governance_signal import TelemetryGovernanceSignal

__all__ = [
    "build_telemetry_summary_for_p4",
    "attach_telemetry_governance_to_evidence",
    "TelemetryP4Summary",
    "TelemetryHealthSummary",
    "TDAFeedbackSummary",
]


@dataclass
class TelemetryHealthSummary:
    """Health summary for P4 calibration report."""
    lean_health: str  # HEALTHY, DEGRADED, UNHEALTHY, CRITICAL
    db_health: str
    redis_health: str
    overall_health: str
    health_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TDAFeedbackSummary:
    """TDA feedback summary for P4 calibration report."""
    topology_alert_level: str
    betti_anomaly_detected: bool
    persistence_anomaly_detected: bool
    min_cut_capacity_degraded: bool
    recommended_actions_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TelemetryP4Summary:
    """
    Telemetry summary designed for p4_calibration_report.json integration.

    This summary provides:
    - Component health states (Lean, DB, Redis)
    - Anomaly rate from telemetry window
    - TDA feedback summary
    - Governance recommendation

    SHADOW MODE: All fields are observational only.
    """
    # Health metrics
    lean_health: str
    db_health: str
    redis_health: str
    overall_health: str
    health_score: float

    # Anomaly metrics
    anomaly_rate: float  # anomalies per cycle
    anomaly_count: int
    anomalies_by_severity: Dict[str, int]

    # TDA feedback
    tda_feedback_summary: Dict[str, Any]

    # Governance
    governance_status: str
    governance_recommendation: str
    safe_for_p4_coupling: bool

    # Metadata
    timestamp: str
    signal_id: str
    mode: str = "SHADOW"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "lean_health": self.lean_health,
            "db_health": self.db_health,
            "redis_health": self.redis_health,
            "overall_health": self.overall_health,
            "health_score": round(self.health_score, 4),
            "anomaly_rate": round(self.anomaly_rate, 4),
            "anomaly_count": self.anomaly_count,
            "anomalies_by_severity": self.anomalies_by_severity,
            "tda_feedback_summary": self.tda_feedback_summary,
            "governance_status": self.governance_status,
            "governance_recommendation": self.governance_recommendation,
            "safe_for_p4_coupling": self.safe_for_p4_coupling,
            "timestamp": self.timestamp,
            "signal_id": self.signal_id,
            "mode": self.mode,
        }


def _extract_component_health(emitter_status: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract component health from emitter status.

    Maps emitter counts to health status for individual components.
    """
    total = emitter_status.get("total_emitters", 0)
    healthy = emitter_status.get("healthy_emitters", 0)
    degraded = emitter_status.get("degraded_emitters", 0)

    # Default to HEALTHY if no data
    if total == 0:
        return {
            "lean_health": "HEALTHY",
            "db_health": "HEALTHY",
            "redis_health": "HEALTHY",
        }

    # Calculate health ratio
    health_ratio = healthy / total if total > 0 else 1.0

    # Map to health status
    if health_ratio >= 0.9:
        status = "HEALTHY"
    elif health_ratio >= 0.7:
        status = "DEGRADED"
    elif health_ratio >= 0.5:
        status = "UNHEALTHY"
    else:
        status = "CRITICAL"

    return {
        "lean_health": status,
        "db_health": status,
        "redis_health": status,
    }


def _build_tda_summary(tda_feedback: Optional[Any]) -> Dict[str, Any]:
    """Build TDA feedback summary from TDAFeedback object."""
    if tda_feedback is None:
        return {
            "topology_alert_level": "NORMAL",
            "betti_anomaly_detected": False,
            "persistence_anomaly_detected": False,
            "min_cut_capacity_degraded": False,
            "recommended_actions_count": 0,
        }

    return {
        "topology_alert_level": getattr(tda_feedback, "topology_alert_level", "NORMAL"),
        "betti_anomaly_detected": getattr(tda_feedback, "betti_anomaly_detected", False),
        "persistence_anomaly_detected": getattr(tda_feedback, "persistence_anomaly_detected", False),
        "min_cut_capacity_degraded": getattr(tda_feedback, "min_cut_capacity_degraded", False),
        "recommended_actions_count": len(getattr(tda_feedback, "recommended_actions", [])),
    }


def build_telemetry_summary_for_p4(
    telemetry_signal: "TelemetryGovernanceSignal",
    cycles_observed: int = 0,
) -> TelemetryP4Summary:
    """
    Build a telemetry summary designed for p4_calibration_report.json integration.

    This function extracts key telemetry metrics from a TelemetryGovernanceSignal
    and packages them in a format suitable for inclusion in P4 calibration reports.

    SHADOW MODE: All outputs are observational only.

    Args:
        telemetry_signal: The TelemetryGovernanceSignal to summarize
        cycles_observed: Number of cycles observed (for anomaly rate calculation)

    Returns:
        TelemetryP4Summary with health, anomaly, and TDA metrics

    Example:
        >>> from backend.telemetry import TelemetryGovernanceSignalEmitter
        >>> emitter = TelemetryGovernanceSignalEmitter()
        >>> signal = emitter.emit_signal()
        >>> summary = build_telemetry_summary_for_p4(signal, cycles_observed=1000)
        >>> print(summary.lean_health)
        'HEALTHY'
    """
    # Extract component health
    component_health = _extract_component_health(telemetry_signal.emitter_status)

    # Calculate anomaly rate
    anomaly_count = 0
    anomalies_by_severity: Dict[str, int] = {}

    if telemetry_signal.anomaly_summary is not None:
        anomaly_count = telemetry_signal.anomaly_summary.anomaly_count
        anomalies_by_severity = telemetry_signal.anomaly_summary.by_severity

    anomaly_rate = anomaly_count / cycles_observed if cycles_observed > 0 else 0.0

    # Build TDA summary
    tda_summary = _build_tda_summary(telemetry_signal.tda_feedback)

    # Extract governance info
    recommendation = "PROCEED"
    safe_for_p4 = True

    if telemetry_signal.recommendation is not None:
        recommendation = telemetry_signal.recommendation.action
        safe_for_p4 = telemetry_signal.recommendation.safe_for_p4_coupling

    return TelemetryP4Summary(
        lean_health=component_health["lean_health"],
        db_health=component_health["db_health"],
        redis_health=component_health["redis_health"],
        overall_health=telemetry_signal.overall_health,
        health_score=telemetry_signal.health_score,
        anomaly_rate=anomaly_rate,
        anomaly_count=anomaly_count,
        anomalies_by_severity=anomalies_by_severity,
        tda_feedback_summary=tda_summary,
        governance_status=telemetry_signal.status,
        governance_recommendation=recommendation,
        safe_for_p4_coupling=safe_for_p4,
        timestamp=telemetry_signal.timestamp,
        signal_id=telemetry_signal.signal_id,
        mode="SHADOW",
    )


def attach_telemetry_governance_to_evidence(
    evidence: Dict[str, Any],
    telemetry_signal: "TelemetryGovernanceSignal",
) -> Dict[str, Any]:
    """
    Attach telemetry governance data to an evidence pack dictionary.

    This function adds telemetry governance information under the
    evidence["governance"]["telemetry"] path, preserving any existing
    governance data.

    SHADOW MODE: All outputs are observational only.

    Args:
        evidence: The evidence pack dictionary to modify
        telemetry_signal: The TelemetryGovernanceSignal to attach

    Returns:
        The modified evidence dictionary with telemetry governance attached

    Example:
        >>> evidence = {"governance": {}}
        >>> signal = emitter.emit_signal()
        >>> evidence = attach_telemetry_governance_to_evidence(evidence, signal)
        >>> print(evidence["governance"]["telemetry"]["status"])
        'OK'
    """
    # Ensure governance section exists
    if "governance" not in evidence:
        evidence["governance"] = {}

    # Build telemetry governance block
    telemetry_governance = {
        "schema_version": telemetry_signal.schema_version,
        "signal_id": telemetry_signal.signal_id,
        "timestamp": telemetry_signal.timestamp,
        "mode": telemetry_signal.mode,
        "status": telemetry_signal.status,
        "overall_health": telemetry_signal.overall_health,
        "health_score": round(telemetry_signal.health_score, 4),
    }

    # Add recommendation if present
    if telemetry_signal.recommendation is not None:
        telemetry_governance["recommendation"] = {
            "action": telemetry_signal.recommendation.action,
            "confidence": round(telemetry_signal.recommendation.confidence, 4),
            "reasons": telemetry_signal.recommendation.reasons,
            "safe_for_p4_coupling": telemetry_signal.recommendation.safe_for_p4_coupling,
            "safe_for_promotion": telemetry_signal.recommendation.safe_for_promotion,
            "enforcement_status": telemetry_signal.recommendation.enforcement_status,
        }

    # Add anomaly summary if present
    if telemetry_signal.anomaly_summary is not None:
        telemetry_governance["anomaly_summary"] = {
            "anomalies_detected": telemetry_signal.anomaly_summary.anomalies_detected,
            "anomaly_count": telemetry_signal.anomaly_summary.anomaly_count,
            "by_severity": telemetry_signal.anomaly_summary.by_severity,
            "by_type": telemetry_signal.anomaly_summary.by_type,
            "tda_alert_triggered": telemetry_signal.anomaly_summary.tda_alert_triggered,
        }

    # Add TDA feedback if present
    if telemetry_signal.tda_feedback is not None:
        telemetry_governance["tda_feedback"] = {
            "feedback_available": telemetry_signal.tda_feedback.feedback_available,
            "topology_alert_level": telemetry_signal.tda_feedback.topology_alert_level,
            "betti_anomaly_detected": telemetry_signal.tda_feedback.betti_anomaly_detected,
            "persistence_anomaly_detected": telemetry_signal.tda_feedback.persistence_anomaly_detected,
            "min_cut_capacity_degraded": telemetry_signal.tda_feedback.min_cut_capacity_degraded,
            "feedback_cycle": telemetry_signal.tda_feedback.feedback_cycle,
            "recommended_actions": telemetry_signal.tda_feedback.recommended_actions,
        }

    # Add P4 coupling context
    telemetry_governance["p4_coupling"] = {
        "supported": telemetry_signal.p4_coupling_supported,
        "adapter_ready": telemetry_signal.p4_adapter_ready,
        "snapshot_availability": telemetry_signal.p4_snapshot_availability,
        "divergence_tracking": telemetry_signal.p4_divergence_tracking,
        "last_snapshot_cycle": telemetry_signal.p4_last_snapshot_cycle,
    }

    # Add hash for integrity
    telemetry_governance["hash"] = {
        "algorithm": telemetry_signal.hash_algorithm,
        "value": telemetry_signal.hash_value or telemetry_signal.compute_hash(),
    }

    # Attach to evidence
    evidence["governance"]["telemetry"] = telemetry_governance

    return evidence


def telemetry_signal_to_ggfl_telemetry(
    telemetry_signal: "TelemetryGovernanceSignal",
) -> Dict[str, Any]:
    """
    Convert TelemetryGovernanceSignal to GGFL telemetry signal format.

    This adapter translates the telemetry signal into the format expected
    by build_global_alignment_view() for the telemetry signal slot.

    SHADOW MODE: Output is observational only.

    The GGFL telemetry signal expects:
    - lean_healthy: bool
    - db_healthy: bool
    - redis_healthy: bool
    - worker_count: int
    - error_rate: float

    Args:
        telemetry_signal: The TelemetryGovernanceSignal to convert

    Returns:
        Dict in GGFL telemetry signal format

    Example:
        >>> signal = emitter.emit_signal()
        >>> ggfl_telemetry = telemetry_signal_to_ggfl_telemetry(signal)
        >>> view = build_global_alignment_view(telemetry=ggfl_telemetry, ...)
    """
    # Map overall health to boolean flags
    healthy = telemetry_signal.overall_health in ("HEALTHY", "DEGRADED")

    # Extract error rate from anomaly summary
    error_rate = 0.0
    if telemetry_signal.anomaly_summary is not None:
        total_anomalies = telemetry_signal.anomaly_summary.anomaly_count
        critical_anomalies = telemetry_signal.anomaly_summary.by_severity.get("CRITICAL", 0)
        # Error rate = critical anomalies / total if any
        if total_anomalies > 0:
            error_rate = critical_anomalies / total_anomalies

    # Extract worker count proxy from emitter status
    worker_count = telemetry_signal.emitter_status.get("healthy_emitters", 1)

    return {
        "lean_healthy": healthy,
        "db_healthy": healthy,
        "redis_healthy": healthy,
        "worker_count": worker_count,
        "error_rate": error_rate,
        # Additional telemetry fields that GGFL can use
        "overall_health": telemetry_signal.overall_health,
        "health_score": telemetry_signal.health_score,
        "anomaly_count": (
            telemetry_signal.anomaly_summary.anomaly_count
            if telemetry_signal.anomaly_summary else 0
        ),
        "tda_alert_level": (
            telemetry_signal.tda_feedback.topology_alert_level
            if telemetry_signal.tda_feedback else "NORMAL"
        ),
        "signal_id": telemetry_signal.signal_id,
        "timestamp": telemetry_signal.timestamp,
        "mode": telemetry_signal.mode,
    }
