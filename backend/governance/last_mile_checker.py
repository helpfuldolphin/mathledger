"""
Last-Mile Governance Checker — CLAUDE K Final Pass Engine

Phase X: Final gating layer before any governance decision, action, or artifact
is committed. Implements the 6-gate hierarchy from LastMile_Governance_Spec.md.

SHADOW MODE CONTRACT:
- All gates are evaluated as in production
- Verdicts are logged but NOT enforced
- No control paths exist from checker to real governance
- Divergence with real governance is tracked for analysis

Gate Hierarchy (earlier gates have precedence):
- G0: Catastrophic (CDI-010) — No override
- G1: Hard (HARD_OK failure streak) — No override
- G2: Invariant (INV violations) — Waiver possible
- G3: Safe Region (Ω exit streak) — Waiver possible
- G4: Soft (ρ collapse, β explosion) — Override possible
- G5: Advisory (TDA degradation) — Informational only

Usage:
    from backend.governance.last_mile_checker import (
        GovernanceFinalChecker,
        GovernanceFinalCheckInput,
        GovernanceFinalCheckResult,
    )

    checker = GovernanceFinalChecker(config=GovernanceFinalCheckConfig())

    # Build input from USLA state
    input_signals = GovernanceFinalCheckInput(
        cycle=42,
        timestamp="2025-12-10T12:00:00Z",
        usla_state=usla_state,
        hard_ok=True,
        hard_fail_streak=0,
        ...
    )

    # Run final check
    result = checker.run_governance_final_check(input_signals)

    # SHADOW MODE: result.verdict is for logging only, NOT enforcement
    audit_logger.log(result)
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

__all__ = [
    "GovernanceFinalChecker",
    "GovernanceFinalCheckInput",
    "GovernanceFinalCheckResult",
    "GovernanceFinalCheckConfig",
    "GateResult",
    "GateEvaluations",
    "GovernanceWaiver",
    "GovernanceOverride",
    "TDAMetrics",
    "GateId",
    "GateStatus",
    "Severity",
    "Verdict",
]


# =============================================================================
# ENUMERATIONS
# =============================================================================

class GateId(Enum):
    """Gate identifiers in evaluation order."""
    G0_CATASTROPHIC = "G0_CATASTROPHIC"
    G1_HARD = "G1_HARD"
    G2_INVARIANT = "G2_INVARIANT"
    G3_SAFE_REGION = "G3_SAFE_REGION"
    G4_SOFT = "G4_SOFT"
    G5_ADVISORY = "G5_ADVISORY"


class GateStatus(Enum):
    """Gate evaluation status."""
    PASS = "PASS"
    FAIL = "FAIL"
    WAIVED = "WAIVED"
    OVERRIDDEN = "OVERRIDDEN"


class Severity(Enum):
    """Severity levels."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class Verdict(Enum):
    """Final governance verdict."""
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TDAMetrics:
    """Topological Data Analysis metrics."""
    sns: float = 0.5   # Structural Non-Triviality Score [0, 1]
    pcs: float = 0.5   # Persistence Coherence Score [0, 1]
    drs: float = 0.0   # Deviation-from-Reference Score [0, 1]
    hss: float = 0.5   # Hallucination Stability Score [0, 1]

    def to_dict(self) -> Dict[str, float]:
        return {
            "sns": round(self.sns, 4),
            "pcs": round(self.pcs, 4),
            "drs": round(self.drs, 4),
            "hss": round(self.hss, 4),
        }


@dataclass
class GovernanceWaiver:
    """Waiver for gate bypass with audit trail."""
    waiver_id: str
    gate_id: str  # "G2_INVARIANT" or "G3_SAFE_REGION"
    issued_by: str  # "human_operator" or "policy_engine"
    issued_at: str  # ISO 8601
    expires_at: str  # ISO 8601
    justification: str
    conditions: List[str] = field(default_factory=list)
    max_cycles: int = 100
    signature: str = ""

    def is_valid(self, current_time: str, cycle: int, start_cycle: int = 0) -> bool:
        """Check if waiver is currently valid."""
        if current_time > self.expires_at:
            return False
        if cycle - start_cycle > self.max_cycles:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "waiver_id": self.waiver_id,
            "gate_id": self.gate_id,
            "issued_by": self.issued_by,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "justification": self.justification,
            "conditions": self.conditions,
            "max_cycles": self.max_cycles,
        }


@dataclass
class GovernanceOverride:
    """Override for soft gate with audit trail."""
    override_id: str
    gate_id: str  # "G4_SOFT"
    issued_by: str
    issued_at: str
    reason: str
    valid_for_cycles: int = 10
    auto_revoke_conditions: List[str] = field(default_factory=list)
    audit_required: bool = True

    def is_valid(self, cycle: int, start_cycle: int) -> bool:
        """Check if override is currently valid."""
        return cycle - start_cycle < self.valid_for_cycles

    def to_dict(self) -> Dict[str, Any]:
        return {
            "override_id": self.override_id,
            "gate_id": self.gate_id,
            "issued_by": self.issued_by,
            "issued_at": self.issued_at,
            "reason": self.reason,
            "valid_for_cycles": self.valid_for_cycles,
        }


@dataclass
class GateResult:
    """Single gate evaluation result."""
    gate_id: GateId
    status: GateStatus
    severity: Severity
    trigger_value: Optional[Any] = None
    threshold: Optional[float] = None
    streak: Optional[int] = None
    details: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_id": self.gate_id.value,
            "status": self.status.value,
            "severity": self.severity.value,
            "trigger_value": self.trigger_value,
            "threshold": self.threshold,
            "streak": self.streak,
            "details": self.details,
        }


@dataclass
class GateEvaluations:
    """All gate evaluation results."""
    g0_catastrophic: GateResult
    g1_hard: GateResult
    g2_invariant: GateResult
    g3_safe_region: GateResult
    g4_soft: GateResult
    g5_advisory: GateResult

    def to_dict(self) -> Dict[str, Any]:
        return {
            "g0_catastrophic": self.g0_catastrophic.to_dict(),
            "g1_hard": self.g1_hard.to_dict(),
            "g2_invariant": self.g2_invariant.to_dict(),
            "g3_safe_region": self.g3_safe_region.to_dict(),
            "g4_soft": self.g4_soft.to_dict(),
            "g5_advisory": self.g5_advisory.to_dict(),
        }


@dataclass
class GovernanceFinalCheckInput:
    """Input signals for Last-Mile Governance Check."""

    # Cycle identification
    cycle: int
    timestamp: str  # ISO 8601

    # USLA State (simplified - can also accept full USLAState object)
    H: float = 1.0          # HSS
    D: int = 0              # Depth
    D_dot: float = 0.0      # Depth velocity
    B: float = 1.0          # Branch factor
    S: float = 0.0          # Shear
    C: int = 0              # Convergence class
    rho: float = 1.0        # RSI
    tau: float = 0.2        # Threshold
    J: float = 0.0          # Jacobian
    W: bool = False         # Exception window
    beta: float = 0.0       # Block rate
    kappa: float = 1.0      # Coupling
    nu: float = 0.0         # Variance velocity
    delta: int = 0          # CDI defect count
    Gamma: float = 1.0      # TGRS

    # HARD mode status
    hard_ok: bool = True
    hard_fail_streak: int = 0

    # Safe region status
    in_omega: bool = True
    omega_exit_streak: int = 0

    # Invariant status
    invariant_violations: List[str] = field(default_factory=list)
    invariant_all_pass: bool = True

    # CDI status
    active_cdis: List[str] = field(default_factory=list)
    cdi_010_active: bool = False

    # Stability metrics
    rho_collapse_streak: int = 0
    beta_explosion_streak: int = 0

    # TDA metrics (optional)
    tda_metrics: Optional[TDAMetrics] = None

    # Override/Waiver inputs
    waivers: List[GovernanceWaiver] = field(default_factory=list)
    overrides: List[GovernanceOverride] = field(default_factory=list)

    @classmethod
    def from_usla_state(
        cls,
        cycle: int,
        timestamp: str,
        usla_state: Any,
        hard_ok: bool,
        hard_fail_streak: int,
        in_omega: bool,
        omega_exit_streak: int,
        rho_collapse_streak: int = 0,
        beta_explosion_streak: int = 0,
        tda_metrics: Optional[TDAMetrics] = None,
        waivers: Optional[List[GovernanceWaiver]] = None,
        overrides: Optional[List[GovernanceOverride]] = None,
    ) -> "GovernanceFinalCheckInput":
        """Create input from USLAState object."""
        return cls(
            cycle=cycle,
            timestamp=timestamp,
            H=usla_state.H,
            D=usla_state.D,
            D_dot=usla_state.D_dot,
            B=usla_state.B,
            S=usla_state.S,
            C=usla_state.C.value if hasattr(usla_state.C, 'value') else usla_state.C,
            rho=usla_state.rho,
            tau=usla_state.tau,
            J=usla_state.J,
            W=usla_state.W,
            beta=usla_state.beta,
            kappa=usla_state.kappa,
            nu=usla_state.nu,
            delta=usla_state.delta,
            Gamma=usla_state.Gamma,
            hard_ok=hard_ok,
            hard_fail_streak=hard_fail_streak,
            in_omega=in_omega,
            omega_exit_streak=omega_exit_streak,
            invariant_violations=usla_state.invariant_violations,
            invariant_all_pass=len(usla_state.invariant_violations) == 0,
            active_cdis=usla_state.active_cdis,
            cdi_010_active="CDI-010" in usla_state.active_cdis,
            rho_collapse_streak=rho_collapse_streak,
            beta_explosion_streak=beta_explosion_streak,
            tda_metrics=tda_metrics,
            waivers=waivers or [],
            overrides=overrides or [],
        )

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of input signals for audit."""
        data = {
            "cycle": self.cycle,
            "timestamp": self.timestamp,
            "H": self.H,
            "D": self.D,
            "D_dot": self.D_dot,
            "B": self.B,
            "S": self.S,
            "C": self.C,
            "rho": self.rho,
            "tau": self.tau,
            "J": self.J,
            "W": self.W,
            "beta": self.beta,
            "kappa": self.kappa,
            "nu": self.nu,
            "delta": self.delta,
            "Gamma": self.Gamma,
            "hard_ok": self.hard_ok,
            "hard_fail_streak": self.hard_fail_streak,
            "in_omega": self.in_omega,
            "omega_exit_streak": self.omega_exit_streak,
            "invariant_violations": sorted(self.invariant_violations),
            "active_cdis": sorted(self.active_cdis),
            "cdi_010_active": self.cdi_010_active,
            "rho_collapse_streak": self.rho_collapse_streak,
            "beta_explosion_streak": self.beta_explosion_streak,
        }
        serialized = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(serialized.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "usla_state": {
                "H": round(self.H, 4),
                "D": self.D,
                "D_dot": round(self.D_dot, 4),
                "B": round(self.B, 4),
                "S": round(self.S, 4),
                "C": self.C,
                "rho": round(self.rho, 4),
                "tau": round(self.tau, 4),
                "J": round(self.J, 4),
                "W": self.W,
                "beta": round(self.beta, 4),
                "kappa": round(self.kappa, 4),
                "nu": round(self.nu, 4),
                "delta": self.delta,
                "Gamma": round(self.Gamma, 4),
            },
            "hard_ok": self.hard_ok,
            "hard_fail_streak": self.hard_fail_streak,
            "in_omega": self.in_omega,
            "omega_exit_streak": self.omega_exit_streak,
            "invariant_all_pass": self.invariant_all_pass,
            "invariant_violations": self.invariant_violations,
            "cdi_010_active": self.cdi_010_active,
            "active_cdis": self.active_cdis,
            "rho": round(self.rho, 4),
            "rho_collapse_streak": self.rho_collapse_streak,
            "beta": round(self.beta, 4),
            "beta_explosion_streak": self.beta_explosion_streak,
            "tda_metrics": self.tda_metrics.to_dict() if self.tda_metrics else None,
        }


@dataclass
class GovernanceFinalCheckResult:
    """Result of Last-Mile Governance Check."""

    # Identification
    check_id: str
    cycle: int
    timestamp: str
    check_version: str = "1.0.0"
    mode: str = "SHADOW"  # "SHADOW" or "ACTIVE"

    # Final verdict
    verdict: Verdict = Verdict.ALLOW
    verdict_confidence: float = 1.0

    # Gate evaluations
    gates: Optional[GateEvaluations] = None

    # Blocking gate (if verdict = BLOCK)
    blocking_gate: Optional[str] = None
    blocking_reason: Optional[str] = None

    # Active waivers/overrides applied
    waivers_applied: List[str] = field(default_factory=list)
    overrides_applied: List[str] = field(default_factory=list)

    # Audit hashes
    input_hash: str = ""
    output_hash: str = ""
    previous_check_hash: Optional[str] = None
    chain_height: int = 1

    # Input snapshot for replay
    input_signals: Optional[GovernanceFinalCheckInput] = None

    def compute_output_hash(self) -> str:
        """Compute SHA-256 hash of output for audit chain."""
        data = {
            "check_id": self.check_id,
            "cycle": self.cycle,
            "timestamp": self.timestamp,
            "verdict": self.verdict.value,
            "verdict_confidence": self.verdict_confidence,
            "blocking_gate": self.blocking_gate,
            "waivers_applied": sorted(self.waivers_applied),
            "overrides_applied": sorted(self.overrides_applied),
            "input_hash": self.input_hash,
        }
        serialized = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(serialized.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/schema compliance."""
        result = {
            "schema_version": self.check_version,
            "check_id": self.check_id,
            "cycle": self.cycle,
            "timestamp": self.timestamp,
            "mode": self.mode,
            "verdict": {
                "decision": self.verdict.value,
                "confidence": round(self.verdict_confidence, 4),
                "blocking_gate": self.blocking_gate,
                "blocking_reason": self.blocking_reason,
            },
            "gates": self.gates.to_dict() if self.gates else None,
            "waivers_applied": [
                {"waiver_id": w, "gate_id": w.split("_")[0] if "_" in w else w}
                for w in self.waivers_applied
            ],
            "overrides_applied": [
                {"override_id": o, "gate_id": "G4_SOFT"}
                for o in self.overrides_applied
            ],
            "audit": {
                "input_hash": self.input_hash,
                "output_hash": self.output_hash,
                "previous_check_hash": self.previous_check_hash,
                "chain_height": self.chain_height,
            },
        }
        if self.input_signals:
            result["input_signals"] = self.input_signals.to_dict()
        return result


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class GovernanceFinalCheckConfig:
    """Configuration for Last-Mile Governance Checker."""

    # Gate thresholds
    hard_fail_threshold: int = 50       # Cycles before G1 blocks
    invariant_tolerance: int = 0        # Violations allowed before G2
    omega_exit_threshold: int = 100     # Cycles outside Ω before G3
    rho_min: float = 0.4                # RSI floor for G4
    beta_max: float = 0.6               # Block rate ceiling for G4
    rho_streak_threshold: int = 10      # Cycles of ρ < ρ_min for G4
    beta_streak_threshold: int = 20     # Cycles of β > β_max for G4

    # TDA advisory thresholds
    sns_min: float = 0.5                # SNS below triggers advisory
    pcs_min: float = 0.6                # PCS below triggers advisory
    drs_max: float = 0.3                # DRS above triggers advisory
    hss_min: float = 0.5                # HSS below triggers advisory

    # Mode
    shadow_mode: bool = True            # SHADOW mode - log only, no enforcement
    include_input_in_result: bool = True  # Include input signals in result

    @classmethod
    def default(cls) -> "GovernanceFinalCheckConfig":
        return cls()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hard_fail_threshold": self.hard_fail_threshold,
            "invariant_tolerance": self.invariant_tolerance,
            "omega_exit_threshold": self.omega_exit_threshold,
            "rho_min": self.rho_min,
            "beta_max": self.beta_max,
            "rho_streak_threshold": self.rho_streak_threshold,
            "beta_streak_threshold": self.beta_streak_threshold,
            "sns_min": self.sns_min,
            "pcs_min": self.pcs_min,
            "drs_max": self.drs_max,
            "hss_min": self.hss_min,
        }


# =============================================================================
# MAIN CHECKER
# =============================================================================

class GovernanceFinalChecker:
    """
    Last-Mile Governance Checker — CLAUDE K Final Pass Engine.

    SHADOW MODE CONTRACT:
    - All gates are evaluated as in production
    - Verdicts are computed but NOT enforced
    - Results are for logging and analysis only
    - No control paths exist to real governance
    """

    def __init__(
        self,
        config: Optional[GovernanceFinalCheckConfig] = None,
    ) -> None:
        self.config = config or GovernanceFinalCheckConfig.default()

        # Audit chain tracking
        self._previous_check_hash: Optional[str] = None
        self._chain_height: int = 0

        # Waiver/override start cycles (for validity tracking)
        self._waiver_start_cycles: Dict[str, int] = {}
        self._override_start_cycles: Dict[str, int] = {}

    def run_governance_final_check(
        self,
        input_signals: GovernanceFinalCheckInput,
    ) -> GovernanceFinalCheckResult:
        """
        Execute Last-Mile Governance Check.

        SHADOW MODE: Computes verdict but does NOT enforce.
        The returned verdict is for logging and analysis only.

        Args:
            input_signals: All input signals for gate evaluation

        Returns:
            GovernanceFinalCheckResult with verdict and gate evaluations
        """
        # Generate check ID
        check_id = str(uuid.uuid4())

        # Compute input hash for audit
        input_hash = input_signals.compute_hash()

        # Evaluate all gates
        g0 = self._evaluate_g0_catastrophic(input_signals)
        g1 = self._evaluate_g1_hard(input_signals)
        g2 = self._evaluate_g2_invariant(input_signals)
        g3 = self._evaluate_g3_safe_region(input_signals)
        g4 = self._evaluate_g4_soft(input_signals)
        g5 = self._evaluate_g5_advisory(input_signals)

        gates = GateEvaluations(
            g0_catastrophic=g0,
            g1_hard=g1,
            g2_invariant=g2,
            g3_safe_region=g3,
            g4_soft=g4,
            g5_advisory=g5,
        )

        # Determine final verdict
        verdict, blocking_gate, blocking_reason, waivers_applied, overrides_applied = (
            self._compute_verdict(gates, input_signals)
        )

        # Compute confidence based on gate margins
        confidence = self._compute_confidence(input_signals, gates)

        # Update audit chain
        self._chain_height += 1

        # Build result
        result = GovernanceFinalCheckResult(
            check_id=check_id,
            cycle=input_signals.cycle,
            timestamp=input_signals.timestamp,
            check_version="1.0.0",
            mode="SHADOW" if self.config.shadow_mode else "ACTIVE",
            verdict=verdict,
            verdict_confidence=confidence,
            gates=gates,
            blocking_gate=blocking_gate,
            blocking_reason=blocking_reason,
            waivers_applied=waivers_applied,
            overrides_applied=overrides_applied,
            input_hash=input_hash,
            previous_check_hash=self._previous_check_hash,
            chain_height=self._chain_height,
            input_signals=input_signals if self.config.include_input_in_result else None,
        )

        # Compute output hash
        result.output_hash = result.compute_output_hash()

        # Update previous hash for chain
        self._previous_check_hash = result.output_hash

        return result

    # =========================================================================
    # GATE EVALUATORS
    # =========================================================================

    def _evaluate_g0_catastrophic(
        self,
        input_signals: GovernanceFinalCheckInput,
    ) -> GateResult:
        """
        G0: Catastrophic Gate (CDI-010 detection).

        Trigger: CDI-010 (Fixed-Point Multiplicity) activated
        Override: NOT POSSIBLE
        """
        if input_signals.cdi_010_active:
            return GateResult(
                gate_id=GateId.G0_CATASTROPHIC,
                status=GateStatus.FAIL,
                severity=Severity.CRITICAL,
                trigger_value=True,
                threshold=0,  # Any activation triggers
                details="CDI-010 (Fixed-Point Multiplicity) active - system has no stable fixed point",
            )

        return GateResult(
            gate_id=GateId.G0_CATASTROPHIC,
            status=GateStatus.PASS,
            severity=Severity.CRITICAL,
            trigger_value=False,
            details="No CDI-010 activation",
        )

    def _evaluate_g1_hard(
        self,
        input_signals: GovernanceFinalCheckInput,
    ) -> GateResult:
        """
        G1: Hard Gate (HARD_OK failure streak).

        Trigger: HARD_OK = False for > hard_fail_threshold consecutive cycles
        Override: NOT POSSIBLE
        """
        threshold = self.config.hard_fail_threshold
        streak = input_signals.hard_fail_streak

        if not input_signals.hard_ok and streak > threshold:
            return GateResult(
                gate_id=GateId.G1_HARD,
                status=GateStatus.FAIL,
                severity=Severity.CRITICAL,
                trigger_value=streak,
                threshold=threshold,
                streak=streak,
                details=f"HARD mode failure streak ({streak}) exceeds threshold ({threshold})",
            )

        return GateResult(
            gate_id=GateId.G1_HARD,
            status=GateStatus.PASS,
            severity=Severity.CRITICAL,
            trigger_value=streak,
            threshold=threshold,
            streak=streak,
            details=f"HARD mode OK or streak ({streak}) within threshold ({threshold})",
        )

    def _evaluate_g2_invariant(
        self,
        input_signals: GovernanceFinalCheckInput,
    ) -> GateResult:
        """
        G2: Invariant Gate (INV-001 through INV-008).

        Trigger: Any invariant violated beyond tolerance
        Override: Waiver possible
        """
        violations = input_signals.invariant_violations
        tolerance = self.config.invariant_tolerance
        violation_count = len(violations)

        # Check for applicable waiver
        waiver = self._find_applicable_waiver(
            input_signals, "G2_INVARIANT"
        )

        if violation_count > tolerance:
            if waiver:
                return GateResult(
                    gate_id=GateId.G2_INVARIANT,
                    status=GateStatus.WAIVED,
                    severity=Severity.HIGH,
                    trigger_value=violation_count,
                    threshold=tolerance,
                    details=f"Invariant violations ({violations}) waived by {waiver.waiver_id}",
                )
            return GateResult(
                gate_id=GateId.G2_INVARIANT,
                status=GateStatus.FAIL,
                severity=Severity.HIGH,
                trigger_value=violation_count,
                threshold=tolerance,
                details=f"Invariant violations: {violations}",
            )

        return GateResult(
            gate_id=GateId.G2_INVARIANT,
            status=GateStatus.PASS,
            severity=Severity.HIGH,
            trigger_value=violation_count,
            threshold=tolerance,
            details="All invariants satisfied" if violation_count == 0 else f"Violations ({violation_count}) within tolerance ({tolerance})",
        )

    def _evaluate_g3_safe_region(
        self,
        input_signals: GovernanceFinalCheckInput,
    ) -> GateResult:
        """
        G3: Safe Region Gate (Ω membership).

        Trigger: State x outside Ω for > omega_exit_threshold consecutive cycles
        Override: Waiver possible
        """
        threshold = self.config.omega_exit_threshold
        streak = input_signals.omega_exit_streak

        # Check for applicable waiver
        waiver = self._find_applicable_waiver(
            input_signals, "G3_SAFE_REGION"
        )

        if not input_signals.in_omega and streak > threshold:
            if waiver:
                return GateResult(
                    gate_id=GateId.G3_SAFE_REGION,
                    status=GateStatus.WAIVED,
                    severity=Severity.HIGH,
                    trigger_value=streak,
                    threshold=threshold,
                    streak=streak,
                    details=f"Omega exit streak ({streak}) waived by {waiver.waiver_id}",
                )
            return GateResult(
                gate_id=GateId.G3_SAFE_REGION,
                status=GateStatus.FAIL,
                severity=Severity.HIGH,
                trigger_value=streak,
                threshold=threshold,
                streak=streak,
                details=f"Outside safe region Ω for {streak} cycles (threshold: {threshold})",
            )

        return GateResult(
            gate_id=GateId.G3_SAFE_REGION,
            status=GateStatus.PASS,
            severity=Severity.HIGH,
            trigger_value=streak,
            threshold=threshold,
            streak=streak,
            details="Within safe region Ω" if input_signals.in_omega else f"Omega exit streak ({streak}) within threshold ({threshold})",
        )

    def _evaluate_g4_soft(
        self,
        input_signals: GovernanceFinalCheckInput,
    ) -> GateResult:
        """
        G4: Soft Gate (RSI collapse or block rate explosion).

        Trigger: (ρ < ρ_min for streak) OR (β > β_max for streak)
        Override: Possible with audit
        """
        rho = input_signals.rho
        rho_min = self.config.rho_min
        rho_streak = input_signals.rho_collapse_streak
        rho_streak_threshold = self.config.rho_streak_threshold

        beta = input_signals.beta
        beta_max = self.config.beta_max
        beta_streak = input_signals.beta_explosion_streak
        beta_streak_threshold = self.config.beta_streak_threshold

        # Check for applicable override
        override = self._find_applicable_override(input_signals, "G4_SOFT")

        # RSI collapse check
        rho_failure = rho < rho_min and rho_streak >= rho_streak_threshold
        # Beta explosion check
        beta_failure = beta > beta_max and beta_streak >= beta_streak_threshold

        if rho_failure or beta_failure:
            if override:
                return GateResult(
                    gate_id=GateId.G4_SOFT,
                    status=GateStatus.OVERRIDDEN,
                    severity=Severity.MEDIUM,
                    trigger_value={"rho": rho, "beta": beta},
                    threshold={"rho_min": rho_min, "beta_max": beta_max},
                    streak=max(rho_streak, beta_streak),
                    details=f"Soft gate failure overridden by {override.override_id}",
                )

            reason_parts = []
            if rho_failure:
                reason_parts.append(f"RSI collapse (ρ={rho:.3f} < {rho_min} for {rho_streak} cycles)")
            if beta_failure:
                reason_parts.append(f"Block rate explosion (β={beta:.3f} > {beta_max} for {beta_streak} cycles)")

            return GateResult(
                gate_id=GateId.G4_SOFT,
                status=GateStatus.FAIL,
                severity=Severity.MEDIUM,
                trigger_value={"rho": rho, "beta": beta},
                threshold={"rho_min": rho_min, "beta_max": beta_max},
                streak=max(rho_streak, beta_streak),
                details="; ".join(reason_parts),
            )

        return GateResult(
            gate_id=GateId.G4_SOFT,
            status=GateStatus.PASS,
            severity=Severity.MEDIUM,
            trigger_value={"rho": rho, "beta": beta},
            threshold={"rho_min": rho_min, "beta_max": beta_max},
            streak=max(rho_streak, beta_streak),
            details=f"RSI (ρ={rho:.3f}) and block rate (β={beta:.3f}) within bounds",
        )

    def _evaluate_g5_advisory(
        self,
        input_signals: GovernanceFinalCheckInput,
    ) -> GateResult:
        """
        G5: Advisory Gate (TDA metrics degradation).

        Trigger: TDA metrics below thresholds
        Action: LOG only, never blocks
        """
        tda = input_signals.tda_metrics

        if tda is None:
            return GateResult(
                gate_id=GateId.G5_ADVISORY,
                status=GateStatus.PASS,
                severity=Severity.LOW,
                details="No TDA metrics available",
            )

        issues = []
        if tda.sns < self.config.sns_min:
            issues.append(f"SNS={tda.sns:.3f} < {self.config.sns_min}")
        if tda.pcs < self.config.pcs_min:
            issues.append(f"PCS={tda.pcs:.3f} < {self.config.pcs_min}")
        if tda.drs > self.config.drs_max:
            issues.append(f"DRS={tda.drs:.3f} > {self.config.drs_max}")
        if tda.hss < self.config.hss_min:
            issues.append(f"HSS={tda.hss:.3f} < {self.config.hss_min}")

        if issues:
            # G5 never blocks - always PASS but with advisory info
            return GateResult(
                gate_id=GateId.G5_ADVISORY,
                status=GateStatus.PASS,  # Advisory never blocks
                severity=Severity.LOW,
                trigger_value=tda.to_dict(),
                details=f"ADVISORY: TDA degradation detected - {'; '.join(issues)}",
            )

        return GateResult(
            gate_id=GateId.G5_ADVISORY,
            status=GateStatus.PASS,
            severity=Severity.LOW,
            trigger_value=tda.to_dict(),
            details="TDA metrics nominal",
        )

    # =========================================================================
    # VERDICT COMPUTATION
    # =========================================================================

    def _compute_verdict(
        self,
        gates: GateEvaluations,
        input_signals: GovernanceFinalCheckInput,
    ) -> Tuple[Verdict, Optional[str], Optional[str], List[str], List[str]]:
        """
        Compute final verdict from gate evaluations.

        FINAL_VERDICT = ALLOW if and only if:
            (G0 = PASS) ∧
            (G1 = PASS) ∧
            (G2 = PASS ∨ G2_waiver) ∧
            (G3 = PASS ∨ G3_waiver) ∧
            (G4 = PASS ∨ G4_override)

        G5 is advisory-only and does not affect FINAL_VERDICT.

        Returns:
            (verdict, blocking_gate, blocking_reason, waivers_applied, overrides_applied)
        """
        waivers_applied: List[str] = []
        overrides_applied: List[str] = []

        # G0: No override possible
        if gates.g0_catastrophic.status == GateStatus.FAIL:
            return (
                Verdict.BLOCK,
                GateId.G0_CATASTROPHIC.value,
                gates.g0_catastrophic.details,
                waivers_applied,
                overrides_applied,
            )

        # G1: No override possible
        if gates.g1_hard.status == GateStatus.FAIL:
            return (
                Verdict.BLOCK,
                GateId.G1_HARD.value,
                gates.g1_hard.details,
                waivers_applied,
                overrides_applied,
            )

        # G2: Waiver possible
        if gates.g2_invariant.status == GateStatus.FAIL:
            return (
                Verdict.BLOCK,
                GateId.G2_INVARIANT.value,
                gates.g2_invariant.details,
                waivers_applied,
                overrides_applied,
            )
        if gates.g2_invariant.status == GateStatus.WAIVED:
            waiver = self._find_applicable_waiver(input_signals, "G2_INVARIANT")
            if waiver:
                waivers_applied.append(waiver.waiver_id)

        # G3: Waiver possible
        if gates.g3_safe_region.status == GateStatus.FAIL:
            return (
                Verdict.BLOCK,
                GateId.G3_SAFE_REGION.value,
                gates.g3_safe_region.details,
                waivers_applied,
                overrides_applied,
            )
        if gates.g3_safe_region.status == GateStatus.WAIVED:
            waiver = self._find_applicable_waiver(input_signals, "G3_SAFE_REGION")
            if waiver:
                waivers_applied.append(waiver.waiver_id)

        # G4: Override possible
        if gates.g4_soft.status == GateStatus.FAIL:
            return (
                Verdict.BLOCK,
                GateId.G4_SOFT.value,
                gates.g4_soft.details,
                waivers_applied,
                overrides_applied,
            )
        if gates.g4_soft.status == GateStatus.OVERRIDDEN:
            override = self._find_applicable_override(input_signals, "G4_SOFT")
            if override:
                overrides_applied.append(override.override_id)

        # G5 is advisory-only, does not affect verdict

        return (
            Verdict.ALLOW,
            None,
            None,
            waivers_applied,
            overrides_applied,
        )

    def _compute_confidence(
        self,
        input_signals: GovernanceFinalCheckInput,
        gates: GateEvaluations,
    ) -> float:
        """
        Compute confidence score for verdict.

        Higher confidence when:
        - All gates pass cleanly (no waivers/overrides)
        - Metrics are well within thresholds
        - No advisory warnings
        """
        confidence = 1.0

        # Reduce confidence for waivers/overrides
        if gates.g2_invariant.status == GateStatus.WAIVED:
            confidence -= 0.1
        if gates.g3_safe_region.status == GateStatus.WAIVED:
            confidence -= 0.1
        if gates.g4_soft.status == GateStatus.OVERRIDDEN:
            confidence -= 0.15

        # Reduce confidence for advisory warnings
        if gates.g5_advisory.details and "ADVISORY" in gates.g5_advisory.details:
            confidence -= 0.05

        # Reduce confidence for near-threshold metrics
        rho_margin = (input_signals.rho - self.config.rho_min) / self.config.rho_min
        if 0 < rho_margin < 0.2:
            confidence -= 0.05

        beta_margin = (self.config.beta_max - input_signals.beta) / self.config.beta_max
        if 0 < beta_margin < 0.2:
            confidence -= 0.05

        return max(0.0, min(1.0, confidence))

    # =========================================================================
    # WAIVER/OVERRIDE HELPERS
    # =========================================================================

    def _find_applicable_waiver(
        self,
        input_signals: GovernanceFinalCheckInput,
        gate_id: str,
    ) -> Optional[GovernanceWaiver]:
        """Find applicable waiver for a gate."""
        for waiver in input_signals.waivers:
            if waiver.gate_id != gate_id:
                continue

            # Track start cycle
            if waiver.waiver_id not in self._waiver_start_cycles:
                self._waiver_start_cycles[waiver.waiver_id] = input_signals.cycle

            start_cycle = self._waiver_start_cycles[waiver.waiver_id]
            if waiver.is_valid(input_signals.timestamp, input_signals.cycle, start_cycle):
                return waiver

        return None

    def _find_applicable_override(
        self,
        input_signals: GovernanceFinalCheckInput,
        gate_id: str,
    ) -> Optional[GovernanceOverride]:
        """Find applicable override for a gate."""
        for override in input_signals.overrides:
            if override.gate_id != gate_id:
                continue

            # Track start cycle
            if override.override_id not in self._override_start_cycles:
                self._override_start_cycles[override.override_id] = input_signals.cycle

            start_cycle = self._override_start_cycles[override.override_id]
            if override.is_valid(input_signals.cycle, start_cycle):
                return override

        return None

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def reset(self) -> None:
        """Reset checker state."""
        self._previous_check_hash = None
        self._chain_height = 0
        self._waiver_start_cycles.clear()
        self._override_start_cycles.clear()

    def get_chain_info(self) -> Dict[str, Any]:
        """Get audit chain information."""
        return {
            "chain_height": self._chain_height,
            "previous_check_hash": self._previous_check_hash,
            "active_waivers": len(self._waiver_start_cycles),
            "active_overrides": len(self._override_start_cycles),
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_governance_final_check(
    input_signals: GovernanceFinalCheckInput,
    config: Optional[GovernanceFinalCheckConfig] = None,
) -> GovernanceFinalCheckResult:
    """
    Convenience function to run a single governance final check.

    SHADOW MODE: Returns verdict for logging only, NOT enforcement.

    Args:
        input_signals: Input signals for gate evaluation
        config: Optional configuration

    Returns:
        GovernanceFinalCheckResult
    """
    checker = GovernanceFinalChecker(config=config)
    return checker.run_governance_final_check(input_signals)
