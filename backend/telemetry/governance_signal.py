"""
Telemetry Governance Signal Emitter

Phase X: Telemetry Canonical Interface

This module generates telemetry-derived governance signals that inform Phase X P4
coupling and TDA feedback loops. Signals synthesize telemetry health, conformance
status, and anomaly detection into actionable governance recommendations.

SHADOW MODE CONTRACT:
- All signals are OBSERVATIONAL ONLY
- Recommendations have enforcement_status="LOGGED_ONLY"
- No modification of real runner execution
- No governance state changes

See: docs/system_law/schemas/telemetry/telemetry_governance_signal.schema.json
See: docs/system_law/Telemetry_PhaseX_Contract.md

Status: P4 IMPLEMENTATION
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.topology.tda_telemetry_feedback import TDAFeedback, TDAFeedbackProvider

__all__ = [
    "TelemetryGovernanceSignal",
    "TelemetryGovernanceSignalEmitter",
    "EmitterHealth",
    "AnomalySummary",
    "GovernanceRecommendation",
    "MockIndicatorSummary",
    "RTTSCorrelationResult",
]


@dataclass
class EmitterHealth:
    """Health status of a telemetry emitter."""
    component: str
    status: str  # HEALTHY, DEGRADED, SILENT, ERROR
    last_emit: str
    emit_rate_per_minute: float = 0.0
    error_rate: float = 0.0
    schema_violations: int = 0
    instance_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "status": self.status,
            "last_emit": self.last_emit,
            "emit_rate_per_minute": round(self.emit_rate_per_minute, 2),
            "error_rate": round(self.error_rate, 4),
            "schema_violations": self.schema_violations,
            "instance_count": self.instance_count,
        }


@dataclass
class AnomalySummary:
    """Summary of detected telemetry anomalies."""
    anomalies_detected: bool = False
    anomaly_count: int = 0
    by_severity: Dict[str, int] = field(default_factory=dict)
    by_type: Dict[str, int] = field(default_factory=dict)
    tda_alert_triggered: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anomalies_detected": self.anomalies_detected,
            "anomaly_count": self.anomaly_count,
            "anomalies_by_severity": self.by_severity,
            "anomalies_by_type": self.by_type,
            "tda_alert_triggered": self.tda_alert_triggered,
        }


@dataclass
class GovernanceRecommendation:
    """Governance recommendation from telemetry analysis."""
    action: str  # PROCEED, CAUTION, REVIEW, HALT_RECOMMENDED
    confidence: float
    reasons: List[str]
    safe_for_p4_coupling: bool = True
    safe_for_promotion: bool = True
    enforcement_status: str = "LOGGED_ONLY"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "confidence": round(self.confidence, 4),
            "reasons": self.reasons,
            "safe_for_p4_coupling": self.safe_for_p4_coupling,
            "safe_for_promotion": self.safe_for_promotion,
            "enforcement_status": self.enforcement_status,
        }


@dataclass
class MockIndicatorSummary:
    """
    Summary of RTTS mock detection indicators.

    Maps to RTTS Section 2.1 MOCK-001 through MOCK-010.

    RTTS-GAP-002: Mock Detection Status (P5.1 LOG-ONLY)
    See: docs/system_law/RTTS_Gap_Closure_Blueprint.md
    See: docs/system_law/Real_Telemetry_Topology_Spec.md Section 2.1

    # REAL-READY: Populated by RTTSMockDetector
    """

    # High severity indicators (any triggers SUSPECTED_MOCK)
    mock_001_var_H_low: bool = False         # Var(H) < 0.0001
    mock_002_var_rho_low: bool = False       # Var(ρ) < 0.00005
    mock_009_jump_H: bool = False            # max(|ΔH|) > δ_H_max
    mock_010_discrete_rho: bool = False      # unique(ρ) < 10 over 100 cycles

    # Medium severity indicators
    mock_003_cor_low: bool = False           # |Cor(H, ρ)| < 0.1
    mock_004_cor_high: bool = False          # |Cor(H, ρ)| > 0.99
    mock_005_acf_low: bool = False           # autocorr(H, lag=1) < 0.05
    mock_006_acf_high: bool = False          # autocorr(H, lag=1) > 0.95

    # Low severity indicators
    mock_007_kurtosis_low: bool = False      # kurtosis(H) < -1.0
    mock_008_kurtosis_high: bool = False     # kurtosis(H) > 5.0

    # Computed scores
    high_severity_count: int = 0
    medium_severity_count: int = 0
    low_severity_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "indicators": {
                "MOCK_001": self.mock_001_var_H_low,
                "MOCK_002": self.mock_002_var_rho_low,
                "MOCK_003": self.mock_003_cor_low,
                "MOCK_004": self.mock_004_cor_high,
                "MOCK_005": self.mock_005_acf_low,
                "MOCK_006": self.mock_006_acf_high,
                "MOCK_007": self.mock_007_kurtosis_low,
                "MOCK_008": self.mock_008_kurtosis_high,
                "MOCK_009": self.mock_009_jump_H,
                "MOCK_010": self.mock_010_discrete_rho,
            },
            "severity_counts": {
                "high": self.high_severity_count,
                "medium": self.medium_severity_count,
                "low": self.low_severity_count,
            },
        }

    def compute_counts(self) -> None:
        """Compute severity counts from indicator flags."""
        self.high_severity_count = sum([
            self.mock_001_var_H_low,
            self.mock_002_var_rho_low,
            self.mock_009_jump_H,
            self.mock_010_discrete_rho,
        ])
        self.medium_severity_count = sum([
            self.mock_003_cor_low,
            self.mock_004_cor_high,
            self.mock_005_acf_low,
            self.mock_006_acf_high,
        ])
        self.low_severity_count = sum([
            self.mock_007_kurtosis_low,
            self.mock_008_kurtosis_high,
        ])


@dataclass
class RTTSCorrelationResult:
    """
    RTTS cross-correlation results.

    Tracks correlations per RTTS Section 1.2.3:
    - Cor(H, ρ) ∈ [0.3, 0.9]
    - Cor(ρ, ω) ∈ [0.5, 1.0]
    - Cor(β, 1-ω) ∈ [0.2, 0.8]

    RTTS-GAP-004: Cross-Correlation Tracking
    P5.2: VALIDATE (NO ENFORCEMENT) - generates warnings
    See: docs/system_law/RTTS_Gap_Closure_Blueprint.md
    See: docs/system_law/Real_Telemetry_Topology_Spec.md Section 1.2.3

    # REAL-READY: Computed by RTTSCorrelationTracker
    """

    # Computed correlations (None = not computed yet)
    cor_H_rho: Optional[float] = None        # Cor(H, ρ)
    cor_rho_omega: Optional[float] = None    # Cor(ρ, ω)
    cor_beta_not_omega: Optional[float] = None  # Cor(β, 1-ω)

    # Violation flags
    cor_H_rho_violated: bool = False
    cor_rho_omega_violated: bool = False
    cor_beta_not_omega_violated: bool = False

    # Mock detection inference
    zero_correlation_detected: bool = False      # Independent random mock
    perfect_correlation_detected: bool = False   # Deterministic coupling mock
    inverted_correlation_detected: bool = False  # Negative where positive expected

    # Window metadata
    window_size: int = 0
    window_start_cycle: int = 0

    # P5.2: Warnings (LOGGED_ONLY)
    warnings: List[str] = field(default_factory=list)

    # SHADOW MODE
    mode: str = "SHADOW"
    action: str = "LOGGED_ONLY"

    # RTTS expected bounds (class constants)
    COR_H_RHO_MIN: float = field(default=0.3, repr=False)
    COR_H_RHO_MAX: float = field(default=0.9, repr=False)
    COR_RHO_OMEGA_MIN: float = field(default=0.5, repr=False)
    COR_RHO_OMEGA_MAX: float = field(default=1.0, repr=False)
    COR_BETA_NOT_OMEGA_MIN: float = field(default=0.2, repr=False)
    COR_BETA_NOT_OMEGA_MAX: float = field(default=0.8, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "correlations": {
                "H_rho": round(self.cor_H_rho, 4) if self.cor_H_rho is not None else None,
                "rho_omega": round(self.cor_rho_omega, 4) if self.cor_rho_omega is not None else None,
                "beta_not_omega": round(self.cor_beta_not_omega, 4) if self.cor_beta_not_omega is not None else None,
            },
            "bounds": {
                "H_rho": {"min": self.COR_H_RHO_MIN, "max": self.COR_H_RHO_MAX},
                "rho_omega": {"min": self.COR_RHO_OMEGA_MIN, "max": self.COR_RHO_OMEGA_MAX},
                "beta_not_omega": {"min": self.COR_BETA_NOT_OMEGA_MIN, "max": self.COR_BETA_NOT_OMEGA_MAX},
            },
            "violations": {
                "H_rho": self.cor_H_rho_violated,
                "rho_omega": self.cor_rho_omega_violated,
                "beta_not_omega": self.cor_beta_not_omega_violated,
            },
            "mock_patterns": {
                "zero_correlation": self.zero_correlation_detected,
                "perfect_correlation": self.perfect_correlation_detected,
                "inverted_correlation": self.inverted_correlation_detected,
            },
            "window": {
                "size": self.window_size,
                "start_cycle": self.window_start_cycle,
            },
            "warnings": self.warnings,
            "mode": self.mode,
            "action": self.action,
        }

    @property
    def has_violations(self) -> bool:
        """Check if any correlation bounds violated."""
        return any([
            self.cor_H_rho_violated,
            self.cor_rho_omega_violated,
            self.cor_beta_not_omega_violated,
        ])

    @property
    def violation_count(self) -> int:
        """Count correlation bound violations."""
        return sum([
            self.cor_H_rho_violated,
            self.cor_rho_omega_violated,
            self.cor_beta_not_omega_violated,
        ])


@dataclass
class TelemetryGovernanceSignal:
    """
    Telemetry-derived governance signal.

    SHADOW MODE: This signal is OBSERVATIONAL ONLY.
    All recommendations are LOGGED, not ENFORCED.

    Conforms to: telemetry_governance_signal.schema.json
    """
    # Required fields per schema
    schema_version: str = "1.0.0"
    signal_type: str = "telemetry_governance"
    signal_id: str = ""
    timestamp: str = ""
    mode: str = "SHADOW"
    status: str = "OK"  # OK, ATTENTION, WARN, CRITICAL

    # Governance recommendation
    recommendation: Optional[GovernanceRecommendation] = None

    # Telemetry health
    overall_health: str = "HEALTHY"  # HEALTHY, DEGRADED, UNHEALTHY, CRITICAL
    health_score: float = 1.0
    emitter_status: Dict[str, int] = field(default_factory=dict)
    schema_conformance: Dict[str, Any] = field(default_factory=dict)
    data_freshness_seconds: float = 0.0

    # Anomaly summary
    anomaly_summary: Optional[AnomalySummary] = None

    # P4 coupling context
    p4_coupling_supported: bool = True
    p4_adapter_ready: bool = True
    p4_snapshot_availability: str = "AVAILABLE"
    p4_divergence_tracking: bool = True
    p4_last_snapshot_cycle: int = 0

    # TDA feedback
    tda_feedback: Optional[TDAFeedback] = None

    # Correlation
    conformance_snapshot_id: Optional[str] = None
    replay_safety_signal_id: Optional[str] = None
    p4_divergence_cycle: Optional[int] = None

    # Hash
    hash_algorithm: str = "sha256"
    hash_value: str = ""

    # RTTS-GAP-002: Mock Detection Status (P5.1 LOG-ONLY)
    # REAL-READY: Populate from RTTSMockDetector
    mock_detection_status: str = "UNKNOWN"  # VALIDATED_REAL | SUSPECTED_MOCK | UNKNOWN
    mock_detection_confidence: float = 0.0  # [0.0, 1.0]
    mock_indicators: Optional[MockIndicatorSummary] = None
    rtts_validation_passed: bool = False
    rtts_validation_violations: List[str] = field(default_factory=list)

    # RTTS-GAP-004: Cross-Correlation Analysis (P5.1 LOG-ONLY)
    # REAL-READY: Populate from RTTSCorrelationTracker
    correlation_analysis: Optional[RTTSCorrelationResult] = None

    def __post_init__(self):
        """Initialize computed fields."""
        if not self.signal_id:
            self.signal_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S.%f+00:00"
            )

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of signal content."""
        # Build hashable content (excluding hash fields)
        content = {
            "schema_version": self.schema_version,
            "signal_type": self.signal_type,
            "signal_id": self.signal_id,
            "timestamp": self.timestamp,
            "status": self.status,
            "overall_health": self.overall_health,
            "health_score": self.health_score,
        }
        canonical = json.dumps(content, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary conforming to schema."""
        # Compute hash if not set
        if not self.hash_value:
            self.hash_value = self.compute_hash()

        return {
            "schema_version": self.schema_version,
            "signal_type": self.signal_type,
            "signal_id": self.signal_id,
            "timestamp": self.timestamp,
            "mode": self.mode,
            "status": self.status,
            "governance_recommendation": (
                self.recommendation.to_dict() if self.recommendation else None
            ),
            "telemetry_health": {
                "overall_health": self.overall_health,
                "health_score": round(self.health_score, 4),
                "emitter_status": self.emitter_status,
                "schema_conformance": self.schema_conformance,
                "data_freshness_seconds": round(self.data_freshness_seconds, 2),
            },
            "anomaly_summary": (
                self.anomaly_summary.to_dict() if self.anomaly_summary else None
            ),
            "p4_coupling_context": {
                "coupling_supported": self.p4_coupling_supported,
                "adapter_ready": self.p4_adapter_ready,
                "snapshot_availability": self.p4_snapshot_availability,
                "divergence_tracking_active": self.p4_divergence_tracking,
                "last_snapshot_cycle": self.p4_last_snapshot_cycle,
            },
            "tda_feedback": (
                self.tda_feedback.to_dict() if self.tda_feedback else None
            ),
            "correlation": {
                "conformance_snapshot_id": self.conformance_snapshot_id,
                "replay_safety_signal_id": self.replay_safety_signal_id,
                "p4_divergence_cycle": self.p4_divergence_cycle,
            },
            "hash": {
                "algorithm": self.hash_algorithm,
                "value": self.hash_value,
            },
            # RTTS-GAP-002: Mock detection (P5.1 LOG-ONLY)
            "mock_detection": {
                "status": self.mock_detection_status,
                "confidence": round(self.mock_detection_confidence, 4),
                "indicators": (
                    self.mock_indicators.to_dict() if self.mock_indicators else None
                ),
            },
            "rtts_validation": {
                "passed": self.rtts_validation_passed,
                "violations": self.rtts_validation_violations,
            },
            # RTTS-GAP-004: Correlation analysis (P5.1 LOG-ONLY)
            "correlation_analysis": (
                self.correlation_analysis.to_dict() if self.correlation_analysis else None
            ),
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def to_jsonl(self) -> str:
        """Convert to JSON line (no indent)."""
        return json.dumps(self.to_dict())


class TelemetryGovernanceSignalEmitter:
    """
    Emitter for telemetry governance signals.

    SHADOW MODE CONTRACT:
    - All signals are OBSERVATIONAL ONLY
    - Recommendations have enforcement_status="LOGGED_ONLY"
    - No modification of real runner execution
    - No governance state changes

    See: docs/system_law/Telemetry_PhaseX_Contract.md Section 8
    """

    def __init__(
        self,
        tda_provider: Optional[TDAFeedbackProvider] = None,
        health_threshold: float = 0.7,
        p4_enabled: bool = True,
    ):
        """
        Initialize signal emitter.

        Args:
            tda_provider: Optional TDA feedback provider
            health_threshold: Threshold for healthy status
            p4_enabled: Whether P4 coupling is enabled
        """
        self._tda_provider = tda_provider or TDAFeedbackProvider()
        self._health_threshold = health_threshold
        self._p4_enabled = p4_enabled

        # Emitter tracking
        self._emitters: Dict[str, EmitterHealth] = {}
        self._last_signal: Optional[TelemetryGovernanceSignal] = None
        self._signal_history: List[TelemetryGovernanceSignal] = []

        # Conformance tracking
        self._schema_conformant = True
        self._drift_detected = False
        self._violations_count = 0

        # P4 state
        self._p4_last_cycle = 0

    def register_emitter(
        self,
        component: str,
        status: str = "HEALTHY",
        emit_rate: float = 0.0,
    ) -> None:
        """
        Register a telemetry emitter.

        Args:
            component: Component name
            status: Initial status
            emit_rate: Emission rate per minute
        """
        self._emitters[component] = EmitterHealth(
            component=component,
            status=status,
            last_emit=datetime.now(timezone.utc).isoformat(),
            emit_rate_per_minute=emit_rate,
        )

    def update_emitter(
        self,
        component: str,
        status: Optional[str] = None,
        emit_rate: Optional[float] = None,
        error_rate: Optional[float] = None,
        schema_violations: Optional[int] = None,
    ) -> None:
        """
        Update emitter status.

        Args:
            component: Component name
            status: New status (if changed)
            emit_rate: New emit rate
            error_rate: New error rate
            schema_violations: New violation count
        """
        if component not in self._emitters:
            self.register_emitter(component)

        emitter = self._emitters[component]
        if status is not None:
            emitter.status = status
        if emit_rate is not None:
            emitter.emit_rate_per_minute = emit_rate
        if error_rate is not None:
            emitter.error_rate = error_rate
        if schema_violations is not None:
            emitter.schema_violations = schema_violations
            self._violations_count += schema_violations

        emitter.last_emit = datetime.now(timezone.utc).isoformat()

    def record_anomaly(
        self,
        cycle: int,
        anomaly_type: str,
        severity: str,
        component: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record an anomaly for TDA analysis.

        Args:
            cycle: Cycle number
            anomaly_type: Type of anomaly
            severity: Severity level
            component: Source component
            details: Additional details
        """
        self._tda_provider.add_anomaly(
            cycle=cycle,
            anomaly_type=anomaly_type,
            severity=severity,
            component=component,
            details=details,
        )

    def set_conformance_status(
        self,
        conformant: bool,
        drift_detected: bool = False,
        violations: int = 0,
    ) -> None:
        """
        Set schema conformance status.

        Args:
            conformant: Whether schemas are conformant
            drift_detected: Whether drift was detected
            violations: Number of violations
        """
        self._schema_conformant = conformant
        self._drift_detected = drift_detected
        self._violations_count = violations

    def set_p4_state(
        self,
        enabled: bool = True,
        last_cycle: int = 0,
        adapter_ready: bool = True,
    ) -> None:
        """
        Set P4 coupling state.

        Args:
            enabled: Whether P4 is enabled
            last_cycle: Last snapshot cycle
            adapter_ready: Whether adapter is ready
        """
        self._p4_enabled = enabled
        self._p4_last_cycle = last_cycle

    def _compute_health_score(self) -> Tuple[str, float]:
        """
        Compute overall health status and score.

        Returns:
            Tuple of (status, score)
        """
        if not self._emitters:
            return "HEALTHY", 1.0

        # Count by status
        healthy = sum(1 for e in self._emitters.values() if e.status == "HEALTHY")
        degraded = sum(1 for e in self._emitters.values() if e.status == "DEGRADED")
        silent = sum(1 for e in self._emitters.values() if e.status == "SILENT")
        error = sum(1 for e in self._emitters.values() if e.status == "ERROR")
        total = len(self._emitters)

        # Compute score
        score = (healthy * 1.0 + degraded * 0.5 + silent * 0.3) / total if total > 0 else 1.0

        # Determine status
        if error > 0 or score < 0.3:
            status = "CRITICAL"
        elif degraded > total / 2 or score < 0.5:
            status = "UNHEALTHY"
        elif degraded > 0 or silent > 0 or score < self._health_threshold:
            status = "DEGRADED"
        else:
            status = "HEALTHY"

        return status, score

    def _compute_signal_status(
        self,
        health_status: str,
        anomaly_summary: AnomalySummary,
        tda_feedback: TDAFeedback,
    ) -> str:
        """
        Compute overall signal status.

        Returns:
            Signal status: OK, ATTENTION, WARN, CRITICAL
        """
        # Start from health status
        if health_status == "CRITICAL":
            return "CRITICAL"

        # Check TDA feedback
        if tda_feedback.topology_alert_level == "CRITICAL":
            return "CRITICAL"
        if tda_feedback.topology_alert_level == "WARNING":
            return "WARN"

        # Check anomalies
        if anomaly_summary.by_severity.get("CRITICAL", 0) > 0:
            return "CRITICAL"
        if anomaly_summary.by_severity.get("WARN", 0) > 3:
            return "WARN"

        # Check conformance
        if not self._schema_conformant or self._drift_detected:
            return "WARN"

        if health_status == "UNHEALTHY":
            return "WARN"
        if health_status == "DEGRADED":
            return "ATTENTION"

        if anomaly_summary.anomalies_detected:
            return "ATTENTION"

        return "OK"

    def _compute_recommendation(
        self,
        signal_status: str,
        health_score: float,
        tda_feedback: TDAFeedback,
    ) -> GovernanceRecommendation:
        """
        Compute governance recommendation.

        SHADOW MODE: All recommendations are LOGGED_ONLY.

        Returns:
            GovernanceRecommendation
        """
        reasons = []

        # Determine action based on status
        if signal_status == "CRITICAL":
            action = "HALT_RECOMMENDED"
            safe_for_p4 = False
            safe_for_promotion = False
            reasons.append("Critical telemetry status detected")
        elif signal_status == "WARN":
            action = "REVIEW"
            safe_for_p4 = True
            safe_for_promotion = False
            reasons.append("Warning conditions require review")
        elif signal_status == "ATTENTION":
            action = "CAUTION"
            safe_for_p4 = True
            safe_for_promotion = True
            reasons.append("Minor concerns detected")
        else:
            action = "PROCEED"
            safe_for_p4 = True
            safe_for_promotion = True
            reasons.append("All telemetry checks passed")

        # Add TDA-specific reasons
        if tda_feedback.betti_anomaly_detected:
            reasons.append("TDA: Betti anomaly detected")
        if tda_feedback.persistence_anomaly_detected:
            reasons.append("TDA: Persistence anomaly detected")
        if tda_feedback.min_cut_capacity_degraded:
            reasons.append("TDA: Telemetry flow capacity degraded")

        # Compute confidence based on data quality
        confidence = min(1.0, health_score * 0.7 + 0.3)
        if self._drift_detected:
            confidence *= 0.8

        return GovernanceRecommendation(
            action=action,
            confidence=confidence,
            reasons=reasons,
            safe_for_p4_coupling=safe_for_p4,
            safe_for_promotion=safe_for_promotion,
            enforcement_status="LOGGED_ONLY",
        )

    def emit_signal(
        self,
        conformance_snapshot_id: Optional[str] = None,
        p4_divergence_cycle: Optional[int] = None,
    ) -> TelemetryGovernanceSignal:
        """
        Emit a telemetry governance signal.

        SHADOW MODE: Signal is OBSERVATIONAL ONLY.

        Args:
            conformance_snapshot_id: Optional correlation ID
            p4_divergence_cycle: Optional P4 cycle correlation

        Returns:
            TelemetryGovernanceSignal
        """
        # Compute health
        health_status, health_score = self._compute_health_score()

        # Get TDA feedback
        tda_feedback = self._tda_provider.generate_feedback()

        # Build anomaly summary
        anomaly_data = self._tda_provider.get_anomaly_summary()
        anomaly_summary = AnomalySummary(
            anomalies_detected=anomaly_data["anomaly_count"] > 0,
            anomaly_count=anomaly_data["anomaly_count"],
            by_severity=anomaly_data.get("by_severity", {}),
            by_type=anomaly_data.get("by_type", {}),
            tda_alert_triggered=tda_feedback.topology_alert_level != "NORMAL",
        )

        # Compute signal status
        signal_status = self._compute_signal_status(
            health_status, anomaly_summary, tda_feedback
        )

        # Compute recommendation
        recommendation = self._compute_recommendation(
            signal_status, health_score, tda_feedback
        )

        # Build emitter status summary
        emitter_status = {
            "total_emitters": len(self._emitters),
            "healthy_emitters": sum(1 for e in self._emitters.values() if e.status == "HEALTHY"),
            "degraded_emitters": sum(1 for e in self._emitters.values() if e.status == "DEGRADED"),
            "silent_emitters": sum(1 for e in self._emitters.values() if e.status == "SILENT"),
        }

        # Build signal
        signal = TelemetryGovernanceSignal(
            status=signal_status,
            recommendation=recommendation,
            overall_health=health_status,
            health_score=health_score,
            emitter_status=emitter_status,
            schema_conformance={
                "conformant": self._schema_conformant,
                "drift_detected": self._drift_detected,
                "violations_count": self._violations_count,
            },
            anomaly_summary=anomaly_summary,
            p4_coupling_supported=self._p4_enabled,
            p4_adapter_ready=self._p4_enabled,
            p4_snapshot_availability="AVAILABLE" if self._p4_enabled else "UNAVAILABLE",
            p4_divergence_tracking=self._p4_enabled,
            p4_last_snapshot_cycle=self._p4_last_cycle,
            tda_feedback=tda_feedback,
            conformance_snapshot_id=conformance_snapshot_id,
            p4_divergence_cycle=p4_divergence_cycle,
        )

        # Store in history
        self._last_signal = signal
        self._signal_history.append(signal)

        return signal

    def get_last_signal(self) -> Optional[TelemetryGovernanceSignal]:
        """Get the last emitted signal."""
        return self._last_signal

    def get_signal_history(self, limit: int = 100) -> List[TelemetryGovernanceSignal]:
        """Get signal history."""
        return self._signal_history[-limit:]

    def reset(self) -> None:
        """Reset emitter state."""
        self._emitters.clear()
        self._last_signal = None
        self._signal_history.clear()
        self._tda_provider.reset()
        self._schema_conformant = True
        self._drift_detected = False
        self._violations_count = 0
        self._p4_last_cycle = 0


# Type alias for convenience
from typing import Tuple
