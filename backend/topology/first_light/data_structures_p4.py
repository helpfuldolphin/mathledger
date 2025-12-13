"""
Phase X P4: Data Structures for Real Runner Shadow Coupling

This module defines the data structures for P4 shadow observation.
See docs/system_law/Phase_X_P4_Spec.md for full specification.

SHADOW MODE CONTRACT:
- All structures are for observation only
- No structure influences real runner execution
- All structures include mode="SHADOW" markers

Status: P4 IMPLEMENTATION
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

__all__ = [
    "TelemetrySnapshot",
    "RealCycleObservation",
    "TwinCycleObservation",
    "DivergenceSnapshot",
    "ContinuityCheck",
]


@dataclass(frozen=True)
class TelemetrySnapshot:
    """
    Immutable snapshot of runner telemetry.

    SHADOW MODE: This is a READ-ONLY capture of real runner state.
    No methods modify any external state.

    See: docs/system_law/Phase_X_P4_Spec.md Section 5.1
    """

    # Cycle identification
    cycle: int = 0
    timestamp: str = ""
    runner_type: str = ""  # "u2" or "rfl"
    slice_name: str = ""

    # Runner outcome
    success: bool = False
    depth: Optional[int] = None
    proof_hash: Optional[str] = None

    # USLA state
    H: float = 0.0           # Health metric
    rho: float = 0.0         # RSI (Running Stability Index)
    tau: float = 0.0         # Current threshold
    beta: float = 0.0        # Block rate
    in_omega: bool = False   # Safe region membership

    # Governance state
    real_blocked: bool = False
    governance_aligned: bool = True
    governance_reason: Optional[str] = None

    # HARD mode
    hard_ok: bool = True

    # Abstention (RFL only)
    abstained: Optional[bool] = None
    abstention_reason: Optional[str] = None

    # Extended metrics (optional)
    reasoning_graph_hash: Optional[str] = None
    proof_dag_size: int = 0

    # Snapshot hash for integrity verification
    snapshot_hash: str = ""

    # RTTS-GAP-001: Statistical Validation Fields (P5.1 LOG-ONLY)
    # REAL-READY: Populate from RTTSStatisticalValidator
    # See: docs/system_law/RTTS_Gap_Closure_Blueprint.md

    # Variance metrics (rolling window)
    variance_H: Optional[float] = None       # Var(H) over validation window
    variance_rho: Optional[float] = None     # Var(ρ) over validation window
    variance_tau: Optional[float] = None     # Var(τ) over validation window
    variance_beta: Optional[float] = None    # Var(β) over validation window

    # Autocorrelation metrics (lag-1)
    autocorr_H_lag1: Optional[float] = None  # ACF(H, lag=1)
    autocorr_rho_lag1: Optional[float] = None  # ACF(ρ, lag=1)

    # Distribution shape metrics
    kurtosis_H: Optional[float] = None       # Excess kurtosis of H
    kurtosis_rho: Optional[float] = None     # Excess kurtosis of ρ

    # Validation window metadata
    stats_window_size: int = 0               # Cycles used for stats computation
    stats_window_start_cycle: int = 0        # First cycle in window

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "cycle": self.cycle,
            "timestamp": self.timestamp,
            "runner_type": self.runner_type,
            "slice_name": self.slice_name,
            "success": self.success,
            "depth": self.depth,
            "proof_hash": self.proof_hash,
            "H": round(self.H, 6),
            "rho": round(self.rho, 6),
            "tau": round(self.tau, 6),
            "beta": round(self.beta, 6),
            "in_omega": self.in_omega,
            "real_blocked": self.real_blocked,
            "governance_aligned": self.governance_aligned,
            "governance_reason": self.governance_reason,
            "hard_ok": self.hard_ok,
            "abstained": self.abstained,
            "abstention_reason": self.abstention_reason,
            "reasoning_graph_hash": self.reasoning_graph_hash,
            "proof_dag_size": self.proof_dag_size,
            "snapshot_hash": self.snapshot_hash,
            # RTTS-GAP-001: Statistical validation (P5.1 LOG-ONLY)
            "statistical_validation": {
                "variance": {
                    "H": round(self.variance_H, 8) if self.variance_H is not None else None,
                    "rho": round(self.variance_rho, 8) if self.variance_rho is not None else None,
                    "tau": round(self.variance_tau, 8) if self.variance_tau is not None else None,
                    "beta": round(self.variance_beta, 8) if self.variance_beta is not None else None,
                },
                "autocorrelation": {
                    "H_lag1": round(self.autocorr_H_lag1, 6) if self.autocorr_H_lag1 is not None else None,
                    "rho_lag1": round(self.autocorr_rho_lag1, 6) if self.autocorr_rho_lag1 is not None else None,
                },
                "kurtosis": {
                    "H": round(self.kurtosis_H, 6) if self.kurtosis_H is not None else None,
                    "rho": round(self.kurtosis_rho, 6) if self.kurtosis_rho is not None else None,
                },
                "window": {
                    "size": self.stats_window_size,
                    "start_cycle": self.stats_window_start_cycle,
                },
            },
        }

    @staticmethod
    def compute_hash(data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of snapshot data for integrity."""
        # Sort keys for deterministic output
        canonical = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


@dataclass
class RealCycleObservation:
    """
    Observation of a real runner cycle.

    SHADOW MODE: This captures what actually happened.
    It does NOT influence what happens next.

    See: docs/system_law/Phase_X_P4_Spec.md Section 6.1
    """

    # Source identification
    source: str = "REAL_RUNNER"
    mode: str = "SHADOW"

    # Cycle data
    cycle: int = 0
    timestamp: str = ""

    # Runner outcome
    runner_type: str = ""
    slice_name: str = ""
    success: bool = False
    depth: Optional[int] = None

    # USLA state snapshot
    H: float = 0.0
    rho: float = 0.0
    tau: float = 0.0
    beta: float = 0.0
    in_omega: bool = False

    # Governance
    real_blocked: bool = False
    governance_aligned: bool = True

    # HARD mode
    hard_ok: bool = True

    # Abstention
    abstained: bool = False
    abstention_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSONL serialization."""
        return {
            "source": self.source,
            "mode": self.mode,
            "cycle": self.cycle,
            "timestamp": self.timestamp,
            "runner_type": self.runner_type,
            "slice_name": self.slice_name,
            "success": self.success,
            "depth": self.depth,
            "usla_state": {
                "H": round(self.H, 6),
                "rho": round(self.rho, 6),
                "tau": round(self.tau, 6),
                "beta": round(self.beta, 6),
                "in_omega": self.in_omega,
            },
            "governance": {
                "real_blocked": self.real_blocked,
                "governance_aligned": self.governance_aligned,
            },
            "hard_ok": self.hard_ok,
            "abstained": self.abstained,
            "abstention_reason": self.abstention_reason,
        }

    @classmethod
    def from_snapshot(cls, snapshot: TelemetrySnapshot) -> "RealCycleObservation":
        """Create observation from telemetry snapshot."""
        return cls(
            source="REAL_RUNNER",
            mode="SHADOW",
            cycle=snapshot.cycle,
            timestamp=snapshot.timestamp,
            runner_type=snapshot.runner_type,
            slice_name=snapshot.slice_name,
            success=snapshot.success,
            depth=snapshot.depth,
            H=snapshot.H,
            rho=snapshot.rho,
            tau=snapshot.tau,
            beta=snapshot.beta,
            in_omega=snapshot.in_omega,
            real_blocked=snapshot.real_blocked,
            governance_aligned=snapshot.governance_aligned,
            hard_ok=snapshot.hard_ok,
            abstained=snapshot.abstained or False,
            abstention_reason=snapshot.abstention_reason,
        )


@dataclass
class TwinCycleObservation:
    """
    Shadow twin prediction for a cycle.

    SHADOW MODE: This is what the twin PREDICTED would happen,
    computed without influencing the real execution.

    See: docs/system_law/Phase_X_P4_Spec.md Section 6.2
    """

    # Source identification
    source: str = "SHADOW_TWIN"
    mode: str = "SHADOW"

    # Corresponding real cycle
    real_cycle: int = 0
    timestamp: str = ""

    # Twin predictions
    predicted_success: bool = False
    predicted_blocked: bool = False
    predicted_in_omega: bool = False
    predicted_hard_ok: bool = True

    # Twin state
    twin_H: float = 0.0
    twin_rho: float = 0.0
    twin_tau: float = 0.0
    twin_beta: float = 0.0

    # Confidence metrics
    prediction_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSONL serialization."""
        return {
            "source": self.source,
            "mode": self.mode,
            "real_cycle": self.real_cycle,
            "timestamp": self.timestamp,
            "predictions": {
                "success": self.predicted_success,
                "blocked": self.predicted_blocked,
                "in_omega": self.predicted_in_omega,
                "hard_ok": self.predicted_hard_ok,
            },
            "twin_state": {
                "H": round(self.twin_H, 6),
                "rho": round(self.twin_rho, 6),
                "tau": round(self.twin_tau, 6),
                "beta": round(self.twin_beta, 6),
            },
            "prediction_confidence": round(self.prediction_confidence, 4),
        }


@dataclass
class DivergenceSnapshot:
    """
    Comparison between real and twin observations.

    SHADOW MODE: This analysis is for logging only.
    Divergences do NOT trigger any remediation.

    See: docs/system_law/Phase_X_P4_Spec.md Section 6.3
    See: docs/system_law/Structural_Cohesion_PhaseX.md Section 4.4
    """

    cycle: int = 0
    timestamp: str = ""

    # Divergence flags
    success_diverged: bool = False
    blocked_diverged: bool = False
    omega_diverged: bool = False
    hard_ok_diverged: bool = False

    # Magnitude metrics
    H_delta: float = 0.0
    rho_delta: float = 0.0
    tau_delta: float = 0.0
    beta_delta: float = 0.0

    # Composite divergence metric
    delta_p: float = 0.0
    delta_p_percent: float = 0.0

    # Classification
    divergence_severity: str = "NONE"  # NONE, INFO, WARN, CRITICAL
    divergence_type: str = "NONE"      # NONE, STATE, OUTCOME, BOTH

    # Analysis
    consecutive_divergences: int = 0
    divergence_streak_start: Optional[int] = None

    # Action taken
    action: str = "LOGGED_ONLY"  # Always "LOGGED_ONLY" in P4

    # Structural governance fields (CLAUDE G integration)
    # See: docs/system_law/Structural_Cohesion_PhaseX.md
    structural_conflict: bool = False  # True if SI-001 or SI-010 violated
    cohesion_degraded: bool = False    # True if cohesion_score < 0.8
    cohesion_score: float = 1.0        # Structural cohesion score [0,1]
    original_severity: str = "NONE"    # Severity before structural adjustment
    severity_escalated: bool = False   # True if severity was upgraded due to structural

    # TDA Context (CLAUDE O - TDA Mind Scanner integration)
    # See: docs/system_law/TDA_PhaseX_Binding.md Section 4
    drs: float = 0.0                   # Drift Rate Score (L2 norm of state drift)
    drs_severity: str = "NONE"         # DRS-specific severity: NONE/INFO/WARN/CRITICAL
    drs_H_drift: float = 0.0           # Per-component: |real_H - twin_H|
    drs_rho_drift: float = 0.0         # Per-component: |real_rho - twin_rho|
    drs_tau_drift: float = 0.0         # Per-component: |real_tau - twin_tau|
    drs_beta_drift: float = 0.0        # Per-component: |real_beta - twin_beta|
    tda_sns: Optional[float] = None    # SNS at divergence (if available)
    tda_pcs: Optional[float] = None    # PCS at divergence (if available)
    tda_hss: Optional[float] = None    # HSS at divergence (if available)
    in_tda_envelope: bool = True       # TDA envelope membership

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSONL serialization."""
        return {
            "cycle": self.cycle,
            "timestamp": self.timestamp,
            "flags": {
                "success_diverged": self.success_diverged,
                "blocked_diverged": self.blocked_diverged,
                "omega_diverged": self.omega_diverged,
                "hard_ok_diverged": self.hard_ok_diverged,
            },
            "magnitudes": {
                "H_delta": round(self.H_delta, 6),
                "rho_delta": round(self.rho_delta, 6),
                "tau_delta": round(self.tau_delta, 6),
                "beta_delta": round(self.beta_delta, 6),
            },
            "delta_p": round(self.delta_p, 6),
            "delta_p_percent": round(self.delta_p_percent, 4),
            "classification": {
                "severity": self.divergence_severity,
                "type": self.divergence_type,
            },
            "streak": {
                "consecutive": self.consecutive_divergences,
                "start_cycle": self.divergence_streak_start,
            },
            "action": self.action,
            "mode": "SHADOW",
            # Structural governance fields
            "structural": {
                "conflict": self.structural_conflict,
                "cohesion_degraded": self.cohesion_degraded,
                "cohesion_score": round(self.cohesion_score, 4),
                "original_severity": self.original_severity,
                "severity_escalated": self.severity_escalated,
            },
            # TDA context (CLAUDE O - TDA Mind Scanner)
            "tda_context": {
                "drs": round(self.drs, 6),
                "drs_severity": self.drs_severity,
                "drs_components": {
                    "H_drift": round(self.drs_H_drift, 6),
                    "rho_drift": round(self.drs_rho_drift, 6),
                    "tau_drift": round(self.drs_tau_drift, 6),
                    "beta_drift": round(self.drs_beta_drift, 6),
                },
                "sns_at_divergence": round(self.tda_sns, 6) if self.tda_sns is not None else None,
                "pcs_at_divergence": round(self.tda_pcs, 6) if self.tda_pcs is not None else None,
                "hss_at_divergence": round(self.tda_hss, 6) if self.tda_hss is not None else None,
                "in_tda_envelope": self.in_tda_envelope,
            },
        }

    def is_diverged(self) -> bool:
        """Check if any divergence occurred."""
        return (
            self.success_diverged or
            self.blocked_diverged or
            self.omega_diverged or
            self.hard_ok_diverged or
            self.divergence_severity != "NONE"
        )

    @classmethod
    def from_observations(
        cls,
        real: RealCycleObservation,
        twin: TwinCycleObservation,
        thresholds: Optional[Dict[str, float]] = None,
        consecutive: int = 0,
        streak_start: Optional[int] = None,
    ) -> "DivergenceSnapshot":
        """Create divergence snapshot from real and twin observations."""
        thresholds = thresholds or {}

        # Compute deltas
        H_delta = abs(real.H - twin.twin_H)
        rho_delta = abs(real.rho - twin.twin_rho)
        tau_delta = abs(real.tau - twin.twin_tau)
        beta_delta = abs(real.beta - twin.twin_beta)

        # Composite delta_p (weighted average)
        delta_p = (H_delta + rho_delta + tau_delta + beta_delta) / 4.0

        # Delta_p as percentage
        delta_p_percent = delta_p * 100.0

        # Compute DRS (Drift Rate Score) - L2 norm of state drift
        # See: docs/system_law/TDA_PhaseX_Binding.md Section 2.3
        import math
        drs = math.sqrt(H_delta**2 + rho_delta**2 + tau_delta**2 + beta_delta**2)

        # Classify DRS severity per TDA spec
        if drs <= 0.05:
            drs_severity = "NONE"
        elif drs <= 0.10:
            drs_severity = "INFO"
        elif drs <= 0.20:
            drs_severity = "WARN"
        else:
            drs_severity = "CRITICAL"

        # Check outcome divergences
        success_diverged = real.success != twin.predicted_success
        blocked_diverged = real.real_blocked != twin.predicted_blocked
        omega_diverged = real.in_omega != twin.predicted_in_omega
        hard_ok_diverged = real.hard_ok != twin.predicted_hard_ok

        # Classify divergence type
        outcome_diverged = success_diverged or blocked_diverged
        state_diverged = delta_p > thresholds.get("state_threshold", 0.05)

        if outcome_diverged and state_diverged:
            divergence_type = "BOTH"
        elif outcome_diverged:
            divergence_type = "OUTCOME"
        elif state_diverged:
            divergence_type = "STATE"
        else:
            divergence_type = "NONE"

        # Classify severity based on delta_p per spec
        epsilon = thresholds.get("epsilon", 0.001)
        threshold_none = thresholds.get("threshold_none", 0.01)
        threshold_info = thresholds.get("threshold_info", 0.05)
        threshold_warn = thresholds.get("threshold_warn", 0.15)

        if delta_p <= threshold_none:
            divergence_severity = "NONE"
        elif delta_p <= threshold_info:
            divergence_severity = "INFO"
        elif delta_p <= threshold_warn:
            divergence_severity = "WARN"
        else:
            divergence_severity = "CRITICAL"

        # Track streaks
        if divergence_severity in ("WARN", "CRITICAL"):
            consecutive += 1
            if streak_start is None:
                streak_start = real.cycle
        else:
            consecutive = 0
            streak_start = None

        return cls(
            cycle=real.cycle,
            timestamp=real.timestamp or datetime.now(timezone.utc).isoformat(),
            success_diverged=success_diverged,
            blocked_diverged=blocked_diverged,
            omega_diverged=omega_diverged,
            hard_ok_diverged=hard_ok_diverged,
            H_delta=H_delta,
            rho_delta=rho_delta,
            tau_delta=tau_delta,
            beta_delta=beta_delta,
            delta_p=delta_p,
            delta_p_percent=delta_p_percent,
            divergence_severity=divergence_severity,
            divergence_type=divergence_type,
            consecutive_divergences=consecutive,
            divergence_streak_start=streak_start,
            action="LOGGED_ONLY",
            # TDA DRS fields
            drs=drs,
            drs_severity=drs_severity,
            drs_H_drift=H_delta,
            drs_rho_drift=rho_delta,
            drs_tau_drift=tau_delta,
            drs_beta_drift=beta_delta,
        )


@dataclass
class ContinuityCheck:
    """
    RTTS cycle-to-cycle continuity validation.

    Tracks |S(t) - S(t-1)| per RTTS Section 1.2.2.

    SHADOW MODE: Continuity violations are logged, not enforced.

    RTTS-GAP-003: Cycle-to-Cycle Continuity Tracking (P5.1 LOG-ONLY)
    See: docs/system_law/RTTS_Gap_Closure_Blueprint.md
    See: docs/system_law/Real_Telemetry_Topology_Spec.md Section 1.2.2

    # REAL-READY: Computed by RTTSContinuityTracker
    """

    cycle: int = 0
    prev_cycle: int = 0
    timestamp: str = ""

    # Per-component deltas
    delta_H: float = 0.0          # |H(t) - H(t-1)|
    delta_rho: float = 0.0        # |ρ(t) - ρ(t-1)|
    delta_tau: float = 0.0        # |τ(t) - τ(t-1)|
    delta_beta: float = 0.0       # |β(t) - β(t-1)|

    # Violation flags
    H_violated: bool = False      # delta_H > DELTA_H_MAX
    rho_violated: bool = False    # delta_rho > DELTA_RHO_MAX
    tau_violated: bool = False    # delta_tau > DELTA_TAU_MAX
    beta_violated: bool = False   # delta_beta > DELTA_BETA_MAX

    # Aggregate
    any_violation: bool = False
    continuity_flag: str = "OK"   # OK | TELEMETRY_JUMP

    # SHADOW MODE marker
    mode: str = "SHADOW"
    action: str = "LOGGED_ONLY"

    # RTTS bounds (from spec Section 1.2.2)
    DELTA_H_MAX: float = field(default=0.15, repr=False)
    DELTA_RHO_MAX: float = field(default=0.10, repr=False)
    DELTA_TAU_MAX: float = field(default=0.05, repr=False)
    DELTA_BETA_MAX: float = field(default=0.20, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSONL serialization."""
        return {
            "cycle": self.cycle,
            "prev_cycle": self.prev_cycle,
            "timestamp": self.timestamp,
            "deltas": {
                "H": round(self.delta_H, 6),
                "rho": round(self.delta_rho, 6),
                "tau": round(self.delta_tau, 6),
                "beta": round(self.delta_beta, 6),
            },
            "bounds": {
                "H_max": self.DELTA_H_MAX,
                "rho_max": self.DELTA_RHO_MAX,
                "tau_max": self.DELTA_TAU_MAX,
                "beta_max": self.DELTA_BETA_MAX,
            },
            "violations": {
                "H": self.H_violated,
                "rho": self.rho_violated,
                "tau": self.tau_violated,
                "beta": self.beta_violated,
            },
            "any_violation": self.any_violation,
            "continuity_flag": self.continuity_flag,
            "mode": self.mode,
            "action": self.action,
        }

    @classmethod
    def from_snapshots(
        cls,
        current: TelemetrySnapshot,
        previous: TelemetrySnapshot,
    ) -> "ContinuityCheck":
        """Create continuity check from consecutive snapshots."""
        delta_H = abs(current.H - previous.H)
        delta_rho = abs(current.rho - previous.rho)
        delta_tau = abs(current.tau - previous.tau)
        delta_beta = abs(current.beta - previous.beta)

        # Check against RTTS bounds
        H_violated = delta_H > 0.15
        rho_violated = delta_rho > 0.10
        tau_violated = delta_tau > 0.05
        beta_violated = delta_beta > 0.20

        any_violation = H_violated or rho_violated or tau_violated or beta_violated

        return cls(
            cycle=current.cycle,
            prev_cycle=previous.cycle,
            timestamp=current.timestamp or datetime.now(timezone.utc).isoformat(),
            delta_H=delta_H,
            delta_rho=delta_rho,
            delta_tau=delta_tau,
            delta_beta=delta_beta,
            H_violated=H_violated,
            rho_violated=rho_violated,
            tau_violated=tau_violated,
            beta_violated=beta_violated,
            any_violation=any_violation,
            continuity_flag="TELEMETRY_JUMP" if any_violation else "OK",
        )
