"""
Phase X P3: Red-Flag Observation Layer

This module implements the red-flag observation layer for detecting anomalous
conditions. See docs/system_law/Phase_X_P3_Spec.md for full specification.

SHADOW MODE CONTRACT:
- observe() NEVER returns an abort signal
- observe() NEVER modifies control flow
- All observations are logged only
- hypothetical_should_abort() is for analysis ONLY
- This code runs OFFLINE only, never in production governance paths

Status: P3 IMPLEMENTATION (OFFLINE, SHADOW-ONLY)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "RedFlagType",
    "RedFlagSeverity",
    "RedFlagObservation",
    "RedFlagSummary",
    "RedFlagObserver",
]


class RedFlagType(Enum):
    """
    Types of red-flag observations.

    See: docs/system_law/Phase_X_P3_Spec.md Section 3.2
    """

    CDI_010 = "CDI-010"                    # Fixed-Point Multiplicity
    CDI_007 = "CDI-007"                    # Exception Exhaustion
    RSI_COLLAPSE = "RSI_COLLAPSE"          # rho < rho_min
    OMEGA_EXIT = "OMEGA_EXIT"              # State outside safe region Omega
    BLOCK_RATE_EXPLOSION = "BLOCK_RATE_EXPLOSION"  # beta > beta_max
    THRESHOLD_DRIFT = "THRESHOLD_DRIFT"    # |tau - tau_0| > epsilon
    GOVERNANCE_DIVERGENCE = "GOVERNANCE_DIVERGENCE"  # real != sim
    HARD_FAIL = "HARD_FAIL"                # HARD_OK = False


class RedFlagSeverity(Enum):
    """
    Severity levels for red-flag observations.

    See: docs/system_law/Phase_X_P3_Spec.md Section 3.2
    """

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class RedFlagObservation:
    """
    A single red-flag observation.

    SHADOW MODE: This is an observation only. It does NOT trigger
    any control flow changes or abort conditions.

    See: docs/system_law/Phase_X_P3_Spec.md Section 3.3
    """

    cycle: int = 0
    timestamp: str = ""
    flag_type: RedFlagType = RedFlagType.CDI_007
    severity: RedFlagSeverity = RedFlagSeverity.INFO

    # Observation details
    observed_value: float = 0.0
    threshold: float = 0.0
    consecutive_cycles: int = 0

    # Context
    state_snapshot: Dict[str, Any] = field(default_factory=dict)

    # SHADOW MODE marker - always "LOGGED_ONLY" in P3
    action_taken: str = "LOGGED_ONLY"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "cycle": self.cycle,
            "timestamp": self.timestamp,
            "flag_type": self.flag_type.value,
            "severity": self.severity.value,
            "observed_value": self.observed_value,
            "threshold": self.threshold,
            "consecutive_cycles": self.consecutive_cycles,
            "state_snapshot": self.state_snapshot,
            "action_taken": self.action_taken,
        }

    def to_jsonl(self) -> str:
        """Serialize to JSONL format."""
        return json.dumps(self.to_dict())


@dataclass
class RedFlagSummary:
    """
    Summary of all red-flag observations in a run.

    SHADOW MODE: All data is for logging/analysis only.
    hypothetical_abort fields are for analysis, NEVER for control flow.

    See: docs/system_law/Phase_X_P3_Spec.md Section 3.3
    """

    total_observations: int = 0
    observations_by_type: Dict[str, int] = field(default_factory=dict)
    observations_by_severity: Dict[str, int] = field(default_factory=dict)

    # Streak tracking
    max_cdi_007_streak: int = 0
    max_rsi_collapse_streak: int = 0
    max_omega_exit_streak: int = 0
    max_block_rate_streak: int = 0
    max_divergence_streak: int = 0
    max_hard_fail_streak: int = 0

    # CDI-010 is special (any activation is notable)
    cdi_010_activations: int = 0

    # Would-have-aborted analysis (SHADOW MODE: hypothetical only)
    hypothetical_abort_cycles: List[int] = field(default_factory=list)
    hypothetical_abort_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "total_observations": self.total_observations,
            "observations_by_type": self.observations_by_type,
            "observations_by_severity": self.observations_by_severity,
            "max_streaks": {
                "cdi_007": self.max_cdi_007_streak,
                "rsi_collapse": self.max_rsi_collapse_streak,
                "omega_exit": self.max_omega_exit_streak,
                "block_rate": self.max_block_rate_streak,
                "divergence": self.max_divergence_streak,
                "hard_fail": self.max_hard_fail_streak,
            },
            "cdi_010_activations": self.cdi_010_activations,
            "hypothetical_aborts": {
                "count": len(self.hypothetical_abort_cycles),
                "cycles": self.hypothetical_abort_cycles,
                "reasons": self.hypothetical_abort_reasons,
            },
        }


@dataclass
class RedFlagConfig:
    """Configuration thresholds for red-flag detection."""

    # RSI collapse threshold
    rsi_collapse_threshold: float = 0.2

    # Block rate explosion threshold
    block_rate_threshold: float = 0.6

    # Threshold drift epsilon
    threshold_drift_epsilon: float = 0.1
    tau_0: float = 0.20

    # Streak thresholds for hypothetical abort analysis
    cdi_007_streak_threshold: int = 10
    omega_exit_streak_threshold: int = 100
    divergence_streak_threshold: int = 20
    hard_fail_streak_threshold: int = 50

    # Enable CDI-010 observation
    cdi_010_observation_enabled: bool = True


class RedFlagObserver:
    """
    Observes red-flag conditions without enforcing them.

    SHADOW MODE CONTRACT:
    - observe() NEVER returns an abort signal
    - observe() NEVER modifies control flow
    - All observations are logged only
    - hypothetical_should_abort() is for analysis ONLY
    - This code runs OFFLINE only

    See: docs/system_law/Phase_X_P3_Spec.md Section 3.4
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """
        Initialize red-flag observer.

        Args:
            config: Optional configuration (RedFlagConfig or FirstLightConfig)
        """
        # Extract thresholds from config
        if config is None:
            self._config = RedFlagConfig()
        elif hasattr(config, "rsi_collapse_threshold"):
            # FirstLightConfig or similar
            self._config = RedFlagConfig(
                rsi_collapse_threshold=getattr(config, "rsi_collapse_threshold", 0.2),
                block_rate_threshold=getattr(config, "block_rate_threshold", 0.6),
                threshold_drift_epsilon=0.1,
                tau_0=getattr(config, "tau_0", 0.20),
                cdi_007_streak_threshold=getattr(config, "cdi_007_streak_threshold", 10),
                omega_exit_streak_threshold=getattr(config, "omega_exit_threshold", 100),
                divergence_streak_threshold=getattr(config, "divergence_streak_threshold", 20),
                hard_fail_streak_threshold=getattr(config, "hard_fail_streak_threshold", 50),
                cdi_010_observation_enabled=getattr(config, "cdi_010_observation_enabled", True),
            )
        else:
            self._config = config if isinstance(config, RedFlagConfig) else RedFlagConfig()

        # Observation storage
        self._observations: List[RedFlagObservation] = []

        # Current streak counters
        self._cdi_007_streak: int = 0
        self._rsi_collapse_streak: int = 0
        self._omega_exit_streak: int = 0
        self._block_rate_streak: int = 0
        self._divergence_streak: int = 0
        self._hard_fail_streak: int = 0

        # Maximum streaks observed
        self._max_cdi_007_streak: int = 0
        self._max_rsi_collapse_streak: int = 0
        self._max_omega_exit_streak: int = 0
        self._max_block_rate_streak: int = 0
        self._max_divergence_streak: int = 0
        self._max_hard_fail_streak: int = 0

        # CDI-010 counter
        self._cdi_010_count: int = 0

        # Hypothetical abort tracking
        self._hypothetical_abort_cycles: List[int] = []
        self._hypothetical_abort_reasons: List[str] = []

        # Block rate tracking (rolling window)
        self._block_history: List[bool] = []
        self._block_window_size: int = 50

    def observe(
        self,
        cycle: int,
        state: Any,
        hard_ok: bool,
        governance_aligned: bool,
    ) -> List[RedFlagObservation]:
        """
        Observe current state for red-flag conditions.

        SHADOW MODE: Returns observations for LOGGING only.
        Does NOT trigger any control flow changes.

        Args:
            cycle: Current cycle number
            state: USLAState instance or dict with H, rho, tau, beta fields
            hard_ok: Whether HARD mode is OK
            governance_aligned: Whether real and sim governance agree

        Returns:
            List of observations (may be empty)
        """
        observations: List[RedFlagObservation] = []
        timestamp = datetime.now(timezone.utc).isoformat()

        # Extract state values
        if hasattr(state, "H"):
            H = state.H
            rho = state.rho
            tau = state.tau
            beta = getattr(state, "beta", 0.0)
            in_omega = getattr(state, "in_omega", True)
        elif isinstance(state, dict):
            H = state.get("H", 0.5)
            rho = state.get("rho", 0.8)
            tau = state.get("tau", 0.2)
            beta = state.get("beta", 0.0)
            in_omega = state.get("in_omega", True)
        else:
            # Default safe values
            H = 0.5
            rho = 0.8
            tau = 0.2
            beta = 0.0
            in_omega = True

        state_snapshot = {"H": H, "rho": rho, "tau": tau, "beta": beta, "in_omega": in_omega}

        # Check CDI-010 (Fixed-Point Multiplicity)
        # CDI-010 is triggered when the system enters a fixed-point state
        # For synthetic mode, we simulate it when H is very close to tau
        if self._config.cdi_010_observation_enabled:
            if abs(H - tau) < 0.01 and rho < 0.5:
                self._cdi_010_count += 1
                obs = RedFlagObservation(
                    cycle=cycle,
                    timestamp=timestamp,
                    flag_type=RedFlagType.CDI_010,
                    severity=RedFlagSeverity.CRITICAL,
                    observed_value=H,
                    threshold=tau,
                    consecutive_cycles=self._cdi_010_count,
                    state_snapshot=state_snapshot,
                )
                observations.append(obs)

        # Check CDI-007 (Exception Exhaustion)
        # We don't have direct exception count, so use proxy: high beta + low rho
        if beta > 0.5 and rho < 0.5:
            self._cdi_007_streak += 1
            self._max_cdi_007_streak = max(self._max_cdi_007_streak, self._cdi_007_streak)
            obs = RedFlagObservation(
                cycle=cycle,
                timestamp=timestamp,
                flag_type=RedFlagType.CDI_007,
                severity=RedFlagSeverity.WARNING if self._cdi_007_streak < 5 else RedFlagSeverity.CRITICAL,
                observed_value=beta,
                threshold=0.5,
                consecutive_cycles=self._cdi_007_streak,
                state_snapshot=state_snapshot,
            )
            observations.append(obs)
        else:
            self._cdi_007_streak = 0

        # Check RSI Collapse
        if rho < self._config.rsi_collapse_threshold:
            self._rsi_collapse_streak += 1
            self._max_rsi_collapse_streak = max(self._max_rsi_collapse_streak, self._rsi_collapse_streak)
            obs = RedFlagObservation(
                cycle=cycle,
                timestamp=timestamp,
                flag_type=RedFlagType.RSI_COLLAPSE,
                severity=RedFlagSeverity.WARNING,
                observed_value=rho,
                threshold=self._config.rsi_collapse_threshold,
                consecutive_cycles=self._rsi_collapse_streak,
                state_snapshot=state_snapshot,
            )
            observations.append(obs)
        else:
            self._rsi_collapse_streak = 0

        # Check Omega Exit
        if not in_omega:
            self._omega_exit_streak += 1
            self._max_omega_exit_streak = max(self._max_omega_exit_streak, self._omega_exit_streak)
            obs = RedFlagObservation(
                cycle=cycle,
                timestamp=timestamp,
                flag_type=RedFlagType.OMEGA_EXIT,
                severity=RedFlagSeverity.INFO if self._omega_exit_streak < 10 else RedFlagSeverity.WARNING,
                observed_value=0.0,
                threshold=1.0,
                consecutive_cycles=self._omega_exit_streak,
                state_snapshot=state_snapshot,
            )
            observations.append(obs)
        else:
            self._omega_exit_streak = 0

        # Check Block Rate Explosion (using rolling window)
        # For synthetic mode, estimate from beta
        blocked = beta > self._config.block_rate_threshold
        self._block_history.append(blocked)
        if len(self._block_history) > self._block_window_size:
            self._block_history.pop(0)

        if len(self._block_history) >= 10:
            block_rate = sum(self._block_history) / len(self._block_history)
            if block_rate > self._config.block_rate_threshold:
                self._block_rate_streak += 1
                self._max_block_rate_streak = max(self._max_block_rate_streak, self._block_rate_streak)
                obs = RedFlagObservation(
                    cycle=cycle,
                    timestamp=timestamp,
                    flag_type=RedFlagType.BLOCK_RATE_EXPLOSION,
                    severity=RedFlagSeverity.WARNING,
                    observed_value=block_rate,
                    threshold=self._config.block_rate_threshold,
                    consecutive_cycles=self._block_rate_streak,
                    state_snapshot=state_snapshot,
                )
                observations.append(obs)
            else:
                self._block_rate_streak = 0

        # Check Threshold Drift
        if abs(tau - self._config.tau_0) > self._config.threshold_drift_epsilon:
            obs = RedFlagObservation(
                cycle=cycle,
                timestamp=timestamp,
                flag_type=RedFlagType.THRESHOLD_DRIFT,
                severity=RedFlagSeverity.INFO,
                observed_value=tau,
                threshold=self._config.tau_0,
                consecutive_cycles=1,
                state_snapshot=state_snapshot,
            )
            observations.append(obs)

        # Check Governance Divergence
        if not governance_aligned:
            self._divergence_streak += 1
            self._max_divergence_streak = max(self._max_divergence_streak, self._divergence_streak)
            obs = RedFlagObservation(
                cycle=cycle,
                timestamp=timestamp,
                flag_type=RedFlagType.GOVERNANCE_DIVERGENCE,
                severity=RedFlagSeverity.WARNING if self._divergence_streak < 10 else RedFlagSeverity.CRITICAL,
                observed_value=float(self._divergence_streak),
                threshold=1.0,
                consecutive_cycles=self._divergence_streak,
                state_snapshot=state_snapshot,
            )
            observations.append(obs)
        else:
            self._divergence_streak = 0

        # Check HARD Fail
        if not hard_ok:
            self._hard_fail_streak += 1
            self._max_hard_fail_streak = max(self._max_hard_fail_streak, self._hard_fail_streak)
            obs = RedFlagObservation(
                cycle=cycle,
                timestamp=timestamp,
                flag_type=RedFlagType.HARD_FAIL,
                severity=RedFlagSeverity.CRITICAL,
                observed_value=0.0,
                threshold=1.0,
                consecutive_cycles=self._hard_fail_streak,
                state_snapshot=state_snapshot,
            )
            observations.append(obs)
        else:
            self._hard_fail_streak = 0

        # Store observations
        self._observations.extend(observations)

        # Check for hypothetical abort conditions (LOGGING only, NEVER enforced)
        self._check_hypothetical_abort(cycle)

        return observations

    def _check_hypothetical_abort(self, cycle: int) -> None:
        """
        Check if hypothetical abort would be triggered.

        SHADOW MODE: This is for analysis only. Results are LOGGED,
        never used for control flow.
        """
        abort_reason: Optional[str] = None

        if self._cdi_010_count > 0:
            abort_reason = f"CDI-010 activated ({self._cdi_010_count} times)"
        elif self._cdi_007_streak >= self._config.cdi_007_streak_threshold:
            abort_reason = f"CDI-007 streak reached {self._cdi_007_streak}"
        elif self._omega_exit_streak >= self._config.omega_exit_streak_threshold:
            abort_reason = f"Omega exit streak reached {self._omega_exit_streak}"
        elif self._divergence_streak >= self._config.divergence_streak_threshold:
            abort_reason = f"Governance divergence streak reached {self._divergence_streak}"
        elif self._hard_fail_streak >= self._config.hard_fail_streak_threshold:
            abort_reason = f"HARD fail streak reached {self._hard_fail_streak}"

        if abort_reason and cycle not in self._hypothetical_abort_cycles:
            self._hypothetical_abort_cycles.append(cycle)
            self._hypothetical_abort_reasons.append(abort_reason)

    def hypothetical_should_abort(self) -> Tuple[bool, Optional[str]]:
        """
        Check if abort WOULD be triggered (hypothetical analysis only).

        SHADOW MODE: This is for analysis/logging only. The return value
        MUST NOT be used to control experiment flow.

        Returns:
            (would_abort, reason) tuple for logging purposes
        """
        if self._hypothetical_abort_cycles:
            return (True, self._hypothetical_abort_reasons[-1])
        return (False, None)

    def get_summary(self) -> RedFlagSummary:
        """
        Get summary of all observations.

        Returns:
            RedFlagSummary instance
        """
        # Count by type
        by_type: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}

        for obs in self._observations:
            type_name = obs.flag_type.value
            severity_name = obs.severity.value

            by_type[type_name] = by_type.get(type_name, 0) + 1
            by_severity[severity_name] = by_severity.get(severity_name, 0) + 1

        return RedFlagSummary(
            total_observations=len(self._observations),
            observations_by_type=by_type,
            observations_by_severity=by_severity,
            max_cdi_007_streak=self._max_cdi_007_streak,
            max_rsi_collapse_streak=self._max_rsi_collapse_streak,
            max_omega_exit_streak=self._max_omega_exit_streak,
            max_block_rate_streak=self._max_block_rate_streak,
            max_divergence_streak=self._max_divergence_streak,
            max_hard_fail_streak=self._max_hard_fail_streak,
            cdi_010_activations=self._cdi_010_count,
            hypothetical_abort_cycles=list(self._hypothetical_abort_cycles),
            hypothetical_abort_reasons=list(self._hypothetical_abort_reasons),
        )

    def get_observations(self) -> List[RedFlagObservation]:
        """Get all recorded observations."""
        return list(self._observations)

    def reset(self) -> None:
        """Reset observer state."""
        self._observations.clear()
        self._cdi_007_streak = 0
        self._rsi_collapse_streak = 0
        self._omega_exit_streak = 0
        self._block_rate_streak = 0
        self._divergence_streak = 0
        self._hard_fail_streak = 0
        self._max_cdi_007_streak = 0
        self._max_rsi_collapse_streak = 0
        self._max_omega_exit_streak = 0
        self._max_block_rate_streak = 0
        self._max_divergence_streak = 0
        self._max_hard_fail_streak = 0
        self._cdi_010_count = 0
        self._hypothetical_abort_cycles.clear()
        self._hypothetical_abort_reasons.clear()
        self._block_history.clear()
