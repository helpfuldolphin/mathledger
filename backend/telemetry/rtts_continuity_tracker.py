"""
RTTS Continuity Tracker

Phase X P5.2: VALIDATE Continuity Tracking (NO ENFORCEMENT)

This module tracks cycle-to-cycle continuity per RTTS Section 1.2.2.
Detects Lipschitz bound violations (TELEMETRY_JUMP events).

SHADOW MODE CONTRACT:
- All tracking is OBSERVATIONAL ONLY
- Violations generate WARNINGS, not enforcement
- Results are logged, not enforced
- No modification of telemetry flow

RTTS-GAP-003: Cycle-to-Cycle Continuity Tracking
See: docs/system_law/RTTS_Gap_Closure_Blueprint.md
See: docs/system_law/Real_Telemetry_Topology_Spec.md Section 1.2.2

Status: P5.2 VALIDATE (NO ENFORCEMENT)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from backend.topology.first_light.data_structures_p4 import TelemetrySnapshot, ContinuityCheck

__all__ = [
    "RTTSContinuityTracker",
    "ContinuityStats",
]


@dataclass
class ContinuityStats:
    """
    Aggregated continuity statistics.

    SHADOW MODE: Stats are for logging only.
    P5.2: Includes warnings and threshold bounds.
    """

    # Counts
    total_checks: int = 0
    violation_count: int = 0
    consecutive_violations: int = 0
    max_consecutive_violations: int = 0

    # Per-component violations
    H_violations: int = 0
    rho_violations: int = 0
    tau_violations: int = 0
    beta_violations: int = 0

    # Max deltas observed
    max_delta_H: float = 0.0
    max_delta_rho: float = 0.0
    max_delta_tau: float = 0.0
    max_delta_beta: float = 0.0

    # Window info
    window_start_cycle: int = 0
    window_end_cycle: int = 0

    # P5.2: Warnings (LOGGED_ONLY)
    warnings: List[str] = field(default_factory=list)

    # SHADOW MODE
    mode: str = "SHADOW"
    action: str = "LOGGED_ONLY"

    # RTTS Lipschitz bounds (for reference in output)
    DELTA_H_MAX: float = field(default=0.15, repr=False)
    DELTA_RHO_MAX: float = field(default=0.10, repr=False)
    DELTA_TAU_MAX: float = field(default=0.05, repr=False)
    DELTA_BETA_MAX: float = field(default=0.20, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "counts": {
                "total_checks": self.total_checks,
                "violation_count": self.violation_count,
                "consecutive_violations": self.consecutive_violations,
                "max_consecutive_violations": self.max_consecutive_violations,
            },
            "violations_by_component": {
                "H": self.H_violations,
                "rho": self.rho_violations,
                "tau": self.tau_violations,
                "beta": self.beta_violations,
            },
            "max_deltas": {
                "H": round(self.max_delta_H, 6),
                "rho": round(self.max_delta_rho, 6),
                "tau": round(self.max_delta_tau, 6),
                "beta": round(self.max_delta_beta, 6),
            },
            "bounds": {
                "H": self.DELTA_H_MAX,
                "rho": self.DELTA_RHO_MAX,
                "tau": self.DELTA_TAU_MAX,
                "beta": self.DELTA_BETA_MAX,
            },
            "window": {
                "start_cycle": self.window_start_cycle,
                "end_cycle": self.window_end_cycle,
            },
            "violation_rate": round(self.violation_rate, 4),
            "warnings": self.warnings,
            "mode": self.mode,
            "action": self.action,
        }

    @property
    def violation_rate(self) -> float:
        """Compute violation rate."""
        if self.total_checks == 0:
            return 0.0
        return self.violation_count / self.total_checks

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)

    def clear_warnings(self) -> None:
        """Clear all warnings."""
        self.warnings.clear()


class RTTSContinuityTracker:
    """
    RTTS cycle-to-cycle continuity tracker.

    Implements Lipschitz continuity validation from
    Real_Telemetry_Topology_Spec.md Section 1.2.2.

    SHADOW MODE: Violations are logged, not enforced.

    # REAL-READY: Hook point for production continuity tracking

    RTTS Lipschitz bounds:
    - |H(t+1) - H(t)| ≤ 0.15 (δ_H_max)
    - |ρ(t+1) - ρ(t)| ≤ 0.10 (δ_ρ_max)
    - |τ(t+1) - τ(t)| ≤ 0.05 (δ_τ_max)
    - |β(t+1) - β(t)| ≤ 0.20 (δ_β_max)
    """

    # RTTS Lipschitz bounds
    DELTA_H_MAX = 0.15
    DELTA_RHO_MAX = 0.10
    DELTA_TAU_MAX = 0.05
    DELTA_BETA_MAX = 0.20

    def __init__(self, history_size: int = 100):
        """
        Initialize continuity tracker.

        Args:
            history_size: Number of checks to keep in history
        """
        self.history_size = history_size
        self._previous_snapshot: Optional["TelemetrySnapshot"] = None
        self._check_history: List["ContinuityCheck"] = []
        self._stats = ContinuityStats()
        self._consecutive_violations: int = 0

    # REAL-READY: Call from TelemetryProviderInterface after each snapshot
    def check(self, snapshot: "TelemetrySnapshot") -> Optional["ContinuityCheck"]:
        """
        Check continuity against previous snapshot.

        Args:
            snapshot: Current TelemetrySnapshot

        Returns:
            ContinuityCheck with violation flags, or None if no previous snapshot
        """
        from backend.topology.first_light.data_structures_p4 import ContinuityCheck

        if self._previous_snapshot is None:
            self._previous_snapshot = snapshot
            self._stats.window_start_cycle = snapshot.cycle
            return None

        # Create continuity check
        check = ContinuityCheck.from_snapshots(snapshot, self._previous_snapshot)

        # Update stats
        self._update_stats(check)

        # Store in history
        self._check_history.append(check)
        if len(self._check_history) > self.history_size:
            self._check_history.pop(0)

        # Update previous snapshot
        self._previous_snapshot = snapshot
        self._stats.window_end_cycle = snapshot.cycle

        return check

    def _update_stats(self, check: "ContinuityCheck") -> None:
        """
        Update aggregated statistics from check.

        P5.2: Generates warnings for violations.
        """
        self._stats.total_checks += 1

        # Track max deltas
        if check.delta_H > self._stats.max_delta_H:
            self._stats.max_delta_H = check.delta_H
        if check.delta_rho > self._stats.max_delta_rho:
            self._stats.max_delta_rho = check.delta_rho
        if check.delta_tau > self._stats.max_delta_tau:
            self._stats.max_delta_tau = check.delta_tau
        if check.delta_beta > self._stats.max_delta_beta:
            self._stats.max_delta_beta = check.delta_beta

        # Track violations and generate warnings
        if check.any_violation:
            self._stats.violation_count += 1
            self._consecutive_violations += 1

            if self._consecutive_violations > self._stats.max_consecutive_violations:
                self._stats.max_consecutive_violations = self._consecutive_violations

            # P5.2: Generate warnings for each violated component
            if check.H_violated:
                self._stats.H_violations += 1
                self._stats.add_warning(
                    f"TELEMETRY_JUMP: delta_H={check.delta_H:.4f} exceeds bound {self.DELTA_H_MAX} at cycle {check.cycle}"
                )
            if check.rho_violated:
                self._stats.rho_violations += 1
                self._stats.add_warning(
                    f"TELEMETRY_JUMP: delta_rho={check.delta_rho:.4f} exceeds bound {self.DELTA_RHO_MAX} at cycle {check.cycle}"
                )
            if check.tau_violated:
                self._stats.tau_violations += 1
                self._stats.add_warning(
                    f"TELEMETRY_JUMP: delta_tau={check.delta_tau:.4f} exceeds bound {self.DELTA_TAU_MAX} at cycle {check.cycle}"
                )
            if check.beta_violated:
                self._stats.beta_violations += 1
                self._stats.add_warning(
                    f"TELEMETRY_JUMP: delta_beta={check.delta_beta:.4f} exceeds bound {self.DELTA_BETA_MAX} at cycle {check.cycle}"
                )

            # Warn on consecutive violations
            if self._consecutive_violations >= 3:
                self._stats.add_warning(
                    f"WARNING: {self._consecutive_violations} consecutive Lipschitz violations ending at cycle {check.cycle}"
                )
        else:
            self._consecutive_violations = 0

        self._stats.consecutive_violations = self._consecutive_violations

    # REAL-READY: Get violation statistics
    def get_stats(self) -> ContinuityStats:
        """
        Return aggregated continuity statistics.

        Returns:
            ContinuityStats with violation counts and max deltas
        """
        return self._stats

    def get_recent_violations(self, limit: int = 10) -> List["ContinuityCheck"]:
        """
        Get recent continuity violations.

        Args:
            limit: Maximum number of violations to return

        Returns:
            List of ContinuityCheck with violations
        """
        violations = [c for c in self._check_history if c.any_violation]
        return violations[-limit:]

    def get_check_history(self, limit: int = 100) -> List["ContinuityCheck"]:
        """
        Get recent continuity checks.

        Args:
            limit: Maximum number of checks to return

        Returns:
            List of recent ContinuityCheck
        """
        return self._check_history[-limit:]

    def has_recent_violations(self, window: int = 10) -> bool:
        """
        Check if there are violations in recent window.

        Args:
            window: Number of recent checks to examine

        Returns:
            True if any violations in window
        """
        recent = self._check_history[-window:] if self._check_history else []
        return any(c.any_violation for c in recent)

    def reset(self) -> None:
        """Reset tracker state."""
        self._previous_snapshot = None
        self._check_history.clear()
        self._stats = ContinuityStats()
        self._consecutive_violations = 0

    def get_violation_rate(self) -> float:
        """Get overall violation rate."""
        return self._stats.violation_rate

    def get_max_delta_H(self) -> float:
        """Get maximum H delta observed."""
        return self._stats.max_delta_H
