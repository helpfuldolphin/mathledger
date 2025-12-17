"""
USLAHealthTileProducer — Minimal USLA health tile for observability.

Phase X P1: Health Tile Producer

This module produces a health tile dictionary summarizing USLA simulator state
for human consumption. The tile is designed to be dropped into global_health.json
or displayed on a dashboard.

SHADOW MODE CONTRACT:
This is a pure, side-effect-free producer. It only reads state and produces
a dictionary. It never modifies governance decisions or system behavior.

Usage:
    from backend.topology.usla_health_tile import USLAHealthTileProducer

    producer = USLAHealthTileProducer()

    # From USLAIntegration:
    tile = producer.produce(
        state=integration._bridge.state,
        hard_ok=integration._bridge.is_hard_ok(),
        monitor=integration._monitor,
    )

    # tile is a JSON-serializable dict ready for dashboard display
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

__all__ = [
    "USLAHealthTileProducer",
    "USLAHealthTile",
]


# Schema version for health tile
HEALTH_TILE_SCHEMA_VERSION = "1.0.0"


@dataclass
class USLAHealthTile:
    """
    USLA Health Tile data structure.

    This is the canonical output format for the health tile producer.
    All fields are JSON-serializable.
    """
    # Schema metadata
    schema_version: str = HEALTH_TILE_SCHEMA_VERSION
    tile_type: str = "usla_health"
    timestamp: str = ""
    mode: str = "SHADOW"

    # Core state summary
    cycle: int = 0
    H: float = 1.0          # HSS
    rho: float = 1.0        # Rolling Stability Index
    tau: float = 0.2        # Effective threshold
    beta: float = 0.0       # Block rate
    J: float = 0.0          # Jacobian sensitivity
    C: str = "CONVERGING"   # Convergence class

    # HARD mode status
    hard_ok: bool = True
    hard_mode_status: str = "OK"

    # Safe region
    in_safe_region: bool = True

    # Defects and invariants
    active_cdis: List[str] = None
    invariant_violations: List[str] = None
    delta: int = 0          # CDI defect count

    # Readiness
    Gamma: float = 1.0      # TGRS score

    # Divergence summary
    governance_aligned: bool = True
    consecutive_divergence: int = 0
    max_severity: str = "NONE"

    # Human-readable headline
    headline: str = "Topology stable; monitoring active"

    # Optional alerts
    alerts: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.active_cdis is None:
            self.active_cdis = []
        if self.invariant_violations is None:
            self.invariant_violations = []
        if self.alerts is None:
            self.alerts = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "schema_version": self.schema_version,
            "tile_type": self.tile_type,
            "timestamp": self.timestamp,
            "mode": self.mode,
            "cycle": self.cycle,
            "state_summary": {
                "H": round(self.H, 4),
                "rho": round(self.rho, 4),
                "tau": round(self.tau, 4),
                "beta": round(self.beta, 4),
                "J": round(self.J, 4),
                "C": self.C,
                "Gamma": round(self.Gamma, 4),
            },
            "hard_mode_status": self.hard_mode_status,
            "safe_region": {
                "within_omega": self.in_safe_region,
            },
            "active_cdis": self.active_cdis,
            "invariant_violations": self.invariant_violations,
            "delta": self.delta,
            "divergence_summary": {
                "governance_aligned": self.governance_aligned,
                "consecutive_divergence": self.consecutive_divergence,
                "max_severity": self.max_severity,
            },
            "headline": self.headline,
            "alerts": self.alerts,
        }


class USLAHealthTileProducer:
    """
    Producer for USLA health tiles.

    This is a pure, stateless producer that creates health tile dicts
    from USLA state and related components.

    SHADOW MODE: This producer is read-only. It never modifies state.
    """

    # Headline templates based on system state
    HEADLINES = {
        "nominal": "Topology stable; monitoring active",
        "degraded_cdis": "Topology degraded; active CDIs detected",
        "degraded_invariants": "Topology degraded; invariant violations detected",
        "hard_fail": "HARD mode inactive; system outside safe region",
        "divergence": "Governance divergence detected; monitoring continues",
        "critical": "CRITICAL: Multiple topology anomalies detected",
    }

    def produce(
        self,
        state: Any,
        hard_ok: bool = True,
        monitor: Any = None,
        params: Any = None,
    ) -> Dict[str, Any]:
        """
        Produce a health tile from USLA state.

        Args:
            state: USLAState instance
            hard_ok: Whether HARD mode is OK
            monitor: Optional DivergenceMonitor instance
            params: Optional USLAParams for safe region check

        Returns:
            JSON-serializable health tile dict
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        # Extract state values
        H = getattr(state, 'H', 1.0)
        rho = getattr(state, 'rho', 1.0)
        tau = getattr(state, 'tau', 0.2)
        beta = getattr(state, 'beta', 0.0)
        J = getattr(state, 'J', 0.0)
        Gamma = getattr(state, 'Gamma', 1.0)
        delta = getattr(state, 'delta', 0)
        cycle = getattr(state, 'cycle', 0)

        # Convergence class
        C = getattr(state, 'C', None)
        if C is not None:
            if hasattr(C, 'name'):
                C_str = C.name
            else:
                C_str = str(C)
        else:
            C_str = "UNKNOWN"

        # Active CDIs and invariant violations
        active_cdis = getattr(state, 'active_cdis', []) or []
        invariant_violations = getattr(state, 'invariant_violations', []) or []

        # Safe region check
        in_safe_region = True
        if params is not None and hasattr(state, 'is_in_safe_region'):
            in_safe_region = state.is_in_safe_region(params)

        # HARD mode status string
        hard_mode_status = "OK" if hard_ok else "FAIL"

        # Divergence summary from monitor
        governance_aligned = True
        consecutive_divergence = 0
        max_severity = "NONE"
        alerts = []

        if monitor is not None:
            governance_aligned = getattr(monitor, 'governance_aligned', True)
            consecutive_divergence = getattr(monitor, 'consecutive_governance_divergence', 0)
            max_severity_enum = getattr(monitor, 'max_severity', None)
            if max_severity_enum is not None:
                if hasattr(max_severity_enum, 'value'):
                    max_severity = max_severity_enum.value
                else:
                    max_severity = str(max_severity_enum)

            # Get recent alerts
            if hasattr(monitor, 'get_alerts'):
                recent_alerts = monitor.get_alerts()
                alerts = [a.to_dict() for a in recent_alerts[-5:]]  # Last 5 alerts

        # Determine headline
        headline = self._compute_headline(
            hard_ok=hard_ok,
            in_safe_region=in_safe_region,
            active_cdis=active_cdis,
            invariant_violations=invariant_violations,
            governance_aligned=governance_aligned,
            max_severity=max_severity,
        )

        # Build tile
        tile = USLAHealthTile(
            timestamp=timestamp,
            cycle=cycle,
            H=H,
            rho=rho,
            tau=tau,
            beta=beta,
            J=J,
            C=C_str,
            hard_ok=hard_ok,
            hard_mode_status=hard_mode_status,
            in_safe_region=in_safe_region,
            active_cdis=list(active_cdis),
            invariant_violations=list(invariant_violations),
            delta=delta,
            Gamma=Gamma,
            governance_aligned=governance_aligned,
            consecutive_divergence=consecutive_divergence,
            max_severity=max_severity,
            headline=headline,
            alerts=alerts,
        )

        return tile.to_dict()

    def _compute_headline(
        self,
        hard_ok: bool,
        in_safe_region: bool,
        active_cdis: List[str],
        invariant_violations: List[str],
        governance_aligned: bool,
        max_severity: str,
    ) -> str:
        """
        Compute human-readable headline based on system state.

        Priority order:
        1. CRITICAL (multiple issues)
        2. HARD fail
        3. Invariant violations
        4. Active CDIs
        5. Divergence
        6. Nominal
        """
        issues = 0

        if not hard_ok:
            issues += 1
        if not in_safe_region:
            issues += 1
        if active_cdis:
            issues += 1
        if invariant_violations:
            issues += 1
        if not governance_aligned:
            issues += 1
        if max_severity == "CRITICAL":
            issues += 1

        # Multiple issues = CRITICAL
        if issues >= 3:
            return self.HEADLINES["critical"]

        # Single issue headlines in priority order
        if not hard_ok or not in_safe_region:
            return self.HEADLINES["hard_fail"]

        if invariant_violations:
            return self.HEADLINES["degraded_invariants"]

        if active_cdis:
            return self.HEADLINES["degraded_cdis"]

        if not governance_aligned:
            return self.HEADLINES["divergence"]

        # All clear
        return self.HEADLINES["nominal"]

    def produce_from_integration(
        self,
        integration: Any,
    ) -> Optional[Dict[str, Any]]:
        """
        Produce health tile from USLAIntegration instance.

        Convenience method that extracts all required components from
        the integration object.

        Args:
            integration: USLAIntegration instance

        Returns:
            Health tile dict, or None if integration is disabled
        """
        if not getattr(integration, 'enabled', False):
            return None

        bridge = getattr(integration, '_bridge', None)
        monitor = getattr(integration, '_monitor', None)

        if bridge is None:
            return None

        state = getattr(bridge, 'state', None)
        params = getattr(bridge, 'params', None)
        hard_ok = bridge.is_hard_ok() if hasattr(bridge, 'is_hard_ok') else True

        if state is None:
            return None

        return self.produce(
            state=state,
            hard_ok=hard_ok,
            monitor=monitor,
            params=params,
        )
