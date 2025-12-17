"""
DivergenceMonitor — Real-time divergence tracking between real and simulated governance.

Phase X: SHADOW MODE ONLY

This module monitors divergence between real TDA governance decisions and
USLA simulator predictions. It tracks consecutive divergence cycles and
raises alerts according to configurable thresholds.

SHADOW MODE CONTRACT:
1. The USLA simulator NEVER modifies real governance decisions
2. Disagreements are LOGGED, not ACTED upon
3. No cycle is blocked or allowed based on simulator output
4. The simulator runs AFTER the real governance decision
5. All USLA state is written to shadow logs only

Alert Thresholds (from Phase X Spec Section 3.2):
| Field               | Threshold | Consecutive for WARNING | Consecutive for CRITICAL |
|---------------------|-----------|------------------------|-------------------------|
| Governance decision | exact     | 3 cycles               | 10 cycles               |
| HSS                 | ε = 0.1   | 3 cycles               | 10 cycles               |
| Threshold τ         | ε = 0.05  | 3 cycles               | 10 cycles               |
| RSI ρ               | ε = 0.15  | N/A (INFO only)        | N/A                     |
| Block rate β        | ε = 0.1   | N/A (INFO only)        | N/A                     |

Usage:
    from backend.topology.divergence_monitor import DivergenceMonitor

    monitor = DivergenceMonitor()
    alerts = monitor.check(cycle, real_state, sim_state)
    for alert in alerts:
        logger.log_divergence_alert(...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = [
    "DivergenceMonitor",
    "DivergenceAlert",
    "AlertSeverity",
    "DivergenceConfig",
]


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class DivergenceAlert:
    """
    A divergence alert.
    """
    cycle: int
    timestamp: str
    severity: AlertSeverity
    field: str
    description: str
    real_value: Any
    sim_value: Any
    epsilon: Optional[float]
    consecutive_cycles: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle": self.cycle,
            "timestamp": self.timestamp,
            "severity": self.severity.value,
            "field": self.field,
            "description": self.description,
            "real_value": self.real_value,
            "sim_value": self.sim_value,
            "epsilon": self.epsilon,
            "consecutive_cycles": self.consecutive_cycles,
        }


@dataclass
class DivergenceConfig:
    """Configuration for divergence monitoring."""
    # Governance decision (exact match)
    governance_warning_cycles: int = 3
    governance_critical_cycles: int = 10

    # HSS (H)
    hss_epsilon: float = 0.1
    hss_warning_cycles: int = 3
    hss_critical_cycles: int = 10

    # Threshold (τ)
    threshold_epsilon: float = 0.05
    threshold_warning_cycles: int = 3
    threshold_critical_cycles: int = 10

    # RSI (ρ) - INFO only
    rsi_epsilon: float = 0.15

    # Block rate (β) - INFO only
    beta_epsilon: float = 0.1

    # Abort conditions
    abort_on_critical_governance: bool = False
    abort_governance_cycles: int = 20

    @classmethod
    def default(cls) -> "DivergenceConfig":
        return cls()


class DivergenceMonitor:
    """
    Monitor divergence between real and simulated governance state.

    Tracks consecutive divergence cycles for each monitored field
    and generates alerts according to configured thresholds.
    """

    def __init__(self, config: Optional[DivergenceConfig] = None):
        self.config = config or DivergenceConfig.default()

        # Consecutive divergence counters
        self._governance_divergence: int = 0
        self._hss_divergence: int = 0
        self._threshold_divergence: int = 0
        self._rsi_divergence: int = 0
        self._beta_divergence: int = 0

        # Statistics
        self._total_cycles: int = 0
        self._total_governance_divergence: int = 0
        self._total_hss_divergence: int = 0
        self._total_threshold_divergence: int = 0

        # Alert history
        self._alerts: List[DivergenceAlert] = []
        self._max_severity_seen: AlertSeverity = AlertSeverity.INFO

        # Abort flag
        self._should_abort: bool = False
        self._abort_reason: Optional[str] = None

    def check(
        self,
        cycle: int,
        real_blocked: bool,
        sim_blocked: bool,
        real_hss: Optional[float] = None,
        sim_hss: Optional[float] = None,
        real_threshold: Optional[float] = None,
        sim_threshold: Optional[float] = None,
        real_rsi: Optional[float] = None,
        sim_rsi: Optional[float] = None,
        real_beta: Optional[float] = None,
        sim_beta: Optional[float] = None,
    ) -> List[DivergenceAlert]:
        """
        Check for divergence and return any alerts.

        Returns list of DivergenceAlert objects for this cycle.
        """
        self._total_cycles += 1
        alerts: List[DivergenceAlert] = []
        timestamp = datetime.now(timezone.utc).isoformat()

        # 1. Governance decision (exact match required)
        if real_blocked != sim_blocked:
            self._governance_divergence += 1
            self._total_governance_divergence += 1

            severity = self._get_severity(
                self._governance_divergence,
                self.config.governance_warning_cycles,
                self.config.governance_critical_cycles,
            )

            alerts.append(DivergenceAlert(
                cycle=cycle,
                timestamp=timestamp,
                severity=severity,
                field="governance",
                description=f"Governance mismatch: real={real_blocked}, sim={sim_blocked}",
                real_value=real_blocked,
                sim_value=sim_blocked,
                epsilon=None,
                consecutive_cycles=self._governance_divergence,
            ))

            # Check abort condition
            if (self.config.abort_on_critical_governance and
                self._governance_divergence >= self.config.abort_governance_cycles):
                self._should_abort = True
                self._abort_reason = f"Governance divergence exceeded {self.config.abort_governance_cycles} consecutive cycles"
        else:
            self._governance_divergence = 0

        # 2. HSS (within epsilon)
        if real_hss is not None and sim_hss is not None:
            if abs(real_hss - sim_hss) > self.config.hss_epsilon:
                self._hss_divergence += 1
                self._total_hss_divergence += 1

                severity = self._get_severity(
                    self._hss_divergence,
                    self.config.hss_warning_cycles,
                    self.config.hss_critical_cycles,
                )

                alerts.append(DivergenceAlert(
                    cycle=cycle,
                    timestamp=timestamp,
                    severity=severity,
                    field="hss",
                    description=f"HSS divergence: |{real_hss:.4f} - {sim_hss:.4f}| > {self.config.hss_epsilon}",
                    real_value=real_hss,
                    sim_value=sim_hss,
                    epsilon=self.config.hss_epsilon,
                    consecutive_cycles=self._hss_divergence,
                ))
            else:
                self._hss_divergence = 0

        # 3. Threshold τ (within epsilon)
        if real_threshold is not None and sim_threshold is not None:
            if abs(real_threshold - sim_threshold) > self.config.threshold_epsilon:
                self._threshold_divergence += 1
                self._total_threshold_divergence += 1

                severity = self._get_severity(
                    self._threshold_divergence,
                    self.config.threshold_warning_cycles,
                    self.config.threshold_critical_cycles,
                )

                alerts.append(DivergenceAlert(
                    cycle=cycle,
                    timestamp=timestamp,
                    severity=severity,
                    field="threshold",
                    description=f"Threshold divergence: |{real_threshold:.4f} - {sim_threshold:.4f}| > {self.config.threshold_epsilon}",
                    real_value=real_threshold,
                    sim_value=sim_threshold,
                    epsilon=self.config.threshold_epsilon,
                    consecutive_cycles=self._threshold_divergence,
                ))
            else:
                self._threshold_divergence = 0

        # 4. RSI ρ (INFO only)
        if real_rsi is not None and sim_rsi is not None:
            if abs(real_rsi - sim_rsi) > self.config.rsi_epsilon:
                self._rsi_divergence += 1

                alerts.append(DivergenceAlert(
                    cycle=cycle,
                    timestamp=timestamp,
                    severity=AlertSeverity.INFO,
                    field="rsi",
                    description=f"RSI divergence: |{real_rsi:.4f} - {sim_rsi:.4f}| > {self.config.rsi_epsilon}",
                    real_value=real_rsi,
                    sim_value=sim_rsi,
                    epsilon=self.config.rsi_epsilon,
                    consecutive_cycles=self._rsi_divergence,
                ))
            else:
                self._rsi_divergence = 0

        # 5. Block rate β (INFO only)
        if real_beta is not None and sim_beta is not None:
            if abs(real_beta - sim_beta) > self.config.beta_epsilon:
                self._beta_divergence += 1

                alerts.append(DivergenceAlert(
                    cycle=cycle,
                    timestamp=timestamp,
                    severity=AlertSeverity.INFO,
                    field="beta",
                    description=f"Block rate divergence: |{real_beta:.4f} - {sim_beta:.4f}| > {self.config.beta_epsilon}",
                    real_value=real_beta,
                    sim_value=sim_beta,
                    epsilon=self.config.beta_epsilon,
                    consecutive_cycles=self._beta_divergence,
                ))
            else:
                self._beta_divergence = 0

        # Update max severity
        for alert in alerts:
            if alert.severity.value > self._max_severity_seen.value:
                self._max_severity_seen = alert.severity

        self._alerts.extend(alerts)
        return alerts

    def _get_severity(
        self,
        consecutive: int,
        warning_threshold: int,
        critical_threshold: int,
    ) -> AlertSeverity:
        """Determine severity based on consecutive divergence count."""
        if consecutive >= critical_threshold:
            return AlertSeverity.CRITICAL
        elif consecutive >= warning_threshold:
            return AlertSeverity.WARNING
        return AlertSeverity.INFO

    @property
    def should_abort(self) -> bool:
        """Check if monitoring recommends abort."""
        return self._should_abort

    @property
    def abort_reason(self) -> Optional[str]:
        """Get abort reason if should_abort is True."""
        return self._abort_reason

    @property
    def governance_aligned(self) -> bool:
        """Check if governance is currently aligned."""
        return self._governance_divergence == 0

    @property
    def max_severity(self) -> AlertSeverity:
        """Get maximum severity seen so far."""
        return self._max_severity_seen

    @property
    def consecutive_governance_divergence(self) -> int:
        """Get current consecutive governance divergence count."""
        return self._governance_divergence

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "total_cycles": self._total_cycles,
            "governance_divergence_cycles": self._total_governance_divergence,
            "governance_divergence_rate": (
                self._total_governance_divergence / self._total_cycles
                if self._total_cycles > 0 else 0.0
            ),
            "hss_divergence_cycles": self._total_hss_divergence,
            "threshold_divergence_cycles": self._total_threshold_divergence,
            "current_governance_streak": self._governance_divergence,
            "current_hss_streak": self._hss_divergence,
            "current_threshold_streak": self._threshold_divergence,
            "max_severity_seen": self._max_severity_seen.value,
            "total_alerts": len(self._alerts),
            "should_abort": self._should_abort,
            "abort_reason": self._abort_reason,
        }

    def reset(self) -> None:
        """Reset all counters and state."""
        self._governance_divergence = 0
        self._hss_divergence = 0
        self._threshold_divergence = 0
        self._rsi_divergence = 0
        self._beta_divergence = 0
        self._total_cycles = 0
        self._total_governance_divergence = 0
        self._total_hss_divergence = 0
        self._total_threshold_divergence = 0
        self._alerts.clear()
        self._max_severity_seen = AlertSeverity.INFO
        self._should_abort = False
        self._abort_reason = None

    def get_alerts(
        self,
        min_severity: AlertSeverity = AlertSeverity.INFO,
    ) -> List[DivergenceAlert]:
        """Get alerts filtered by minimum severity."""
        severity_order = {
            AlertSeverity.INFO: 0,
            AlertSeverity.WARNING: 1,
            AlertSeverity.CRITICAL: 2,
        }
        min_level = severity_order[min_severity]
        return [a for a in self._alerts if severity_order[a.severity] >= min_level]
