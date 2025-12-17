"""
USLA Simulator — Unified System Law Abstraction Simulator

Phase IX: Core dynamical system simulator implementing the canonical
update operator F for the MathLedger governance-topology organism.

This simulator enables:
- Stability boundary analysis
- Bifurcation detection
- Fixed-point computation
- Curriculum stress testing
- Governance drift simulation
- Defect triggering frequency analysis
- Topology coupling effect studies

Usage:
    from backend.topology.usla_simulator import USLASimulator, USLAState

    sim = USLASimulator()
    state = USLAState.initial()

    for cycle_result in cycle_results:
        state = sim.step(state, cycle_result)
        if sim.is_red_flag(state):
            break
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = [
    "USLAState",
    "USLAParams",
    "ConvergenceClass",
    "USLASimulator",
    "SimulationResult",
    "StabilityAnalysis",
]


class ConvergenceClass(Enum):
    """Convergence classification."""
    CONVERGING = 0
    OSCILLATING = 1
    DIVERGING = 2


@dataclass
class USLAState:
    """
    Canonical state vector x ∈ ℝ¹⁵.

    This is the minimal sufficient representation of the full
    governance-topology system state.
    """
    # Primary observables
    H: float = 1.0          # HSS (health signal) [0, 1]
    D: int = 0              # Proof depth ℤ⁺
    D_dot: float = 0.0      # Depth velocity ℝ
    B: float = 1.0          # Branch factor ℝ⁺
    S: float = 0.0          # Semantic shear [0, 1]

    # Dynamics classification
    C: ConvergenceClass = ConvergenceClass.CONVERGING

    # Derived state
    rho: float = 1.0        # Rolling Stability Index [0, 1]
    tau: float = 0.2        # Effective threshold [0.1, 0.5]
    J: float = 0.0          # Jacobian max sensitivity ℝ⁺
    W: bool = False         # Exception window active
    beta: float = 0.0       # Block rate (rolling) [0, 1]
    kappa: float = 1.0      # Coupling strength [0, 1]
    nu: float = 0.0         # Variance velocity ℝ
    delta: int = 0          # CDI defect count ℤ⁺
    Gamma: float = 1.0      # TGRS (readiness score) [0, 1]

    # Internal tracking (not part of canonical state)
    cycle: int = 0
    blocked: bool = False
    active_cdis: List[str] = field(default_factory=list)
    invariant_violations: List[str] = field(default_factory=list)

    @classmethod
    def initial(cls) -> "USLAState":
        """Create initial optimistic state."""
        return cls()

    def to_vector(self) -> List[float]:
        """Convert to numeric vector."""
        return [
            self.H, float(self.D), self.D_dot, self.B, self.S,
            float(self.C.value), self.rho, self.tau, self.J,
            float(self.W), self.beta, self.kappa, self.nu,
            float(self.delta), self.Gamma,
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "H": round(self.H, 4),
            "D": self.D,
            "D_dot": round(self.D_dot, 4),
            "B": round(self.B, 4),
            "S": round(self.S, 4),
            "C": self.C.name,
            "rho": round(self.rho, 4),
            "tau": round(self.tau, 4),
            "J": round(self.J, 4),
            "W": self.W,
            "beta": round(self.beta, 4),
            "kappa": round(self.kappa, 4),
            "nu": round(self.nu, 4),
            "delta": self.delta,
            "Gamma": round(self.Gamma, 4),
            "cycle": self.cycle,
            "blocked": self.blocked,
            "active_cdis": self.active_cdis,
            "invariant_violations": self.invariant_violations,
        }

    def is_in_safe_region(self, params: "USLAParams") -> bool:
        """Check if state is in safe control region Ω."""
        return (
            self.H >= params.H_min and
            abs(self.D_dot) <= params.D_dot_max and
            self.B <= params.B_max and
            self.S <= params.S_max and
            self.C != ConvergenceClass.DIVERGING
        )


@dataclass
class USLAParams:
    """
    Parameter manifold Θ.

    Contains all tunable parameters of the USLA system.
    """
    # Threshold parameters
    tau_0: float = 0.2
    alpha_D: float = 0.02
    alpha_B: float = 0.01
    alpha_S: float = 0.1
    B_0: float = 2.0

    # Convergence modifiers
    gamma_converging: float = 1.0
    gamma_oscillating: float = 1.1
    gamma_diverging: float = 1.3

    # Stability parameters
    alpha_rho: float = 0.9
    rho_min: float = 0.4

    # Block rate smoothing
    alpha_beta: float = 0.9

    # Safe region bounds
    H_min: float = 0.3
    D_dot_max: float = 2.0
    B_max: float = 8.0
    S_max: float = 0.4

    # Jacobian computation
    epsilon: float = 0.01
    J_threshold: float = 10.0

    # Stability weights
    w_H: float = 0.3
    w_D: float = 0.2
    w_B: float = 0.2
    w_S: float = 0.2
    w_u: float = 0.1

    # TGRS weights
    tgrs_w_H: float = 0.25
    tgrs_w_C: float = 0.25
    tgrs_w_S: float = 0.15
    tgrs_w_B: float = 0.15
    tgrs_w_P: float = 0.20

    # Exception window
    exception_shear_threshold: float = 0.4
    exception_block_rate_threshold: float = 0.5
    exception_stable_cycles: int = 5

    # Variance window for convergence classification
    variance_window: int = 10
    convergence_slope_threshold: float = 0.05
    oscillation_variance_threshold: float = 0.15

    # Invariant tolerances (INV-001 through INV-008)
    inv_001_shear_delta_max: float = 0.05       # Max shear change per cycle
    inv_002_bf_depth_gradient_max: float = 1.0  # Max BF-depth gradient
    inv_003_variance_lipschitz: float = 0.02    # Variance velocity bound
    inv_004_cut_coherence_min: float = 0.1      # Min cut coherence (stub)
    inv_005_rho_delta_max: float = 0.1          # Max RSI change per cycle
    inv_006_beta_delta_max: float = 0.1         # Max block rate change
    inv_007_beta_max: float = 0.2               # Exception conservation bound
    inv_008_depth_max: int = 20                 # Depth boundedness

    # CDI trigger thresholds
    cdi_002_shear_gradient_max: float = 0.4     # Asymmetric shear threshold
    cdi_005_depth_runaway: int = 15             # Depth runaway threshold
    cdi_005_accel_threshold: float = 0.5        # Depth acceleration threshold
    cdi_006_stagnation_window: int = 20         # Cycles for stagnation detection
    cdi_006_d_dot_avg_min: float = 0.1          # Min avg depth velocity
    cdi_008_kappa_oscillation: float = 0.2      # Coupling oscillation threshold
    cdi_008_oscillation_cycles: int = 3         # Consecutive cycles for CDI-008
    cdi_009_nu_threshold: float = 0.02          # Variance blowup threshold
    cdi_009_sustained_cycles: int = 3           # Cycles for sustained variance

    def gamma(self, C: ConvergenceClass) -> float:
        """Get convergence modifier."""
        return {
            ConvergenceClass.CONVERGING: self.gamma_converging,
            ConvergenceClass.OSCILLATING: self.gamma_oscillating,
            ConvergenceClass.DIVERGING: self.gamma_diverging,
        }[C]


@dataclass
class CycleInput:
    """Input data for a single cycle."""
    hss: float
    depth: int
    branch_factor: float
    shear: Optional[float] = None
    success: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CycleInput":
        return cls(
            hss=d.get("hss", d.get("H", 1.0)),
            depth=d.get("depth", d.get("max_depth", 0)),
            branch_factor=d.get("branch_factor", 1.0),
            shear=d.get("shear", d.get("S")),
            success=d.get("success", True),
        )


@dataclass
class SimulationResult:
    """Result of running a simulation."""
    states: List[USLAState]
    final_state: USLAState
    total_cycles: int
    blocks: int
    block_rate: float
    mean_rho: float
    min_rho: float
    safe_region_violations: int
    defect_cycles: int
    red_flags: List[Tuple[int, str]]
    # P0 additions
    hard_mode_failures: List[Tuple[int, str]] = field(default_factory=list)
    invariant_violations: Dict[str, int] = field(default_factory=dict)
    cdi_activations: Dict[str, int] = field(default_factory=dict)
    first_hard_failure_cycle: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_cycles": self.total_cycles,
            "blocks": self.blocks,
            "block_rate": round(self.block_rate, 4),
            "mean_rho": round(self.mean_rho, 4),
            "min_rho": round(self.min_rho, 4),
            "safe_region_violations": self.safe_region_violations,
            "defect_cycles": self.defect_cycles,
            "red_flags": [{"cycle": c, "reason": r} for c, r in self.red_flags],
            "hard_mode_failures": [{"cycle": c, "reason": r} for c, r in self.hard_mode_failures],
            "invariant_violations": self.invariant_violations,
            "cdi_activations": self.cdi_activations,
            "first_hard_failure_cycle": self.first_hard_failure_cycle,
            "final_state": self.final_state.to_dict(),
        }


@dataclass
class StabilityAnalysis:
    """Result of stability analysis."""
    fixed_points: List[USLAState]
    stable_fixed_points: List[USLAState]
    unstable_fixed_points: List[USLAState]
    bifurcation_params: List[Tuple[str, float]]
    jacobian_at_fixed_points: List[float]
    stability_margin: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_fixed_points": len(self.fixed_points),
            "num_stable": len(self.stable_fixed_points),
            "num_unstable": len(self.unstable_fixed_points),
            "bifurcation_params": [{"param": p, "value": v} for p, v in self.bifurcation_params],
            "stability_margin": round(self.stability_margin, 4),
        }


class USLASimulator:
    """
    USLA Simulator implementing the canonical update operator F.

    F: X × U × Θ → X

    This simulator provides:
    - Single-step state updates
    - Full trajectory simulation
    - Stability analysis
    - Bifurcation detection
    - Safe region monitoring
    """

    def __init__(self, params: Optional[USLAParams] = None) -> None:
        self.params = params or USLAParams()

        # History for variance computation
        self._hss_history: List[float] = []
        self._variance_history: List[float] = []
        self._depth_history: List[int] = []
        self._stable_cycle_counter: int = 0

        # Extended history for invariant and CDI checking
        self._state_history: List[USLAState] = []
        self._shear_history: List[float] = []
        self._rho_history: List[float] = []
        self._beta_history: List[float] = []
        self._kappa_history: List[float] = []
        self._nu_history: List[float] = []
        self._d_dot_history: List[float] = []
        self._bf_history: List[float] = []

    def reset(self) -> None:
        """Reset simulator state."""
        self._hss_history = []
        self._variance_history = []
        self._depth_history = []
        self._stable_cycle_counter = 0
        self._state_history = []
        self._shear_history = []
        self._rho_history = []
        self._beta_history = []
        self._kappa_history = []
        self._nu_history = []
        self._d_dot_history = []
        self._bf_history = []

    def step(
        self,
        state: USLAState,
        cycle_input: CycleInput,
    ) -> USLAState:
        """
        Execute one step of the canonical update operator F.

        xₜ₊₁ = F(xₜ, uₜ, θ)

        Args:
            state: Current state xₜ
            cycle_input: Cycle observation data

        Returns:
            Next state xₜ₊₁
        """
        p = self.params

        # ═══════════════════════════════════════════════════════════
        # PHASE 1: OBSERVATION UPDATE
        # ═══════════════════════════════════════════════════════════

        H_new = cycle_input.hss
        D_new = cycle_input.depth
        D_dot_new = D_new - state.D
        B_new = cycle_input.branch_factor
        S_new = cycle_input.shear if cycle_input.shear is not None else self._compute_shear(state)

        # Update histories
        self._hss_history.append(H_new)
        self._depth_history.append(D_new)
        if len(self._hss_history) > p.variance_window * 2:
            self._hss_history = self._hss_history[-p.variance_window * 2:]
            self._depth_history = self._depth_history[-p.variance_window * 2:]

        # ═══════════════════════════════════════════════════════════
        # PHASE 2: DYNAMICS CLASSIFICATION
        # ═══════════════════════════════════════════════════════════

        C_new = self._classify_convergence()
        kappa_new = self._estimate_coupling()
        nu_new = self._compute_variance_velocity()

        # ═══════════════════════════════════════════════════════════
        # PHASE 3: GOVERNANCE COMPUTATION
        # ═══════════════════════════════════════════════════════════

        # Adaptive threshold τ(x)
        tau_new = self._compute_threshold(D_dot_new, B_new, S_new, C_new)

        # Jacobian
        J_new = self._compute_jacobian(H_new, tau_new, D_dot_new, B_new, S_new, C_new)

        # Exception window
        W_new = self._update_exception_window(
            state.W, C_new, S_new, state.beta, H_new
        )

        # Governance decision: G(x) = 𝟙[H < τ ∧ ¬W]
        u = (H_new < tau_new) and (not W_new)

        # Rolling block rate
        beta_new = p.alpha_beta * state.beta + (1 - p.alpha_beta) * (1.0 if u else 0.0)

        # ═══════════════════════════════════════════════════════════
        # PHASE 4: STABILITY ASSESSMENT
        # ═══════════════════════════════════════════════════════════

        S_inst = self._compute_instantaneous_stability(H_new, D_dot_new, B_new, S_new, u)
        rho_new = p.alpha_rho * state.rho + (1 - p.alpha_rho) * S_inst

        # ═══════════════════════════════════════════════════════════
        # PHASE 5: DEFECT DETECTION (full CDI coverage)
        # ═══════════════════════════════════════════════════════════

        # Update extended histories for CDI detection
        self._shear_history.append(S_new)
        self._rho_history.append(rho_new)
        self._beta_history.append(beta_new)
        self._kappa_history.append(kappa_new)
        self._nu_history.append(nu_new)
        self._d_dot_history.append(D_dot_new)
        self._bf_history.append(B_new)

        # Trim histories to reasonable window
        max_hist = 50
        if len(self._shear_history) > max_hist:
            self._shear_history = self._shear_history[-max_hist:]
            self._rho_history = self._rho_history[-max_hist:]
            self._beta_history = self._beta_history[-max_hist:]
            self._kappa_history = self._kappa_history[-max_hist:]
            self._nu_history = self._nu_history[-max_hist:]
            self._d_dot_history = self._d_dot_history[-max_hist:]
            self._bf_history = self._bf_history[-max_hist:]

        delta_new, active_cdis = self._count_defects_full(
            H_new, D_new, D_dot_new, B_new, S_new, C_new, beta_new, J_new, kappa_new, nu_new
        )

        # ═══════════════════════════════════════════════════════════
        # PHASE 6: READINESS COMPUTATION
        # ═══════════════════════════════════════════════════════════

        Gamma_new = self._compute_tgrs(H_new, C_new, S_new, B_new, delta_new)

        # ═══════════════════════════════════════════════════════════
        # PHASE 7: INVARIANT VERIFICATION
        # ═══════════════════════════════════════════════════════════

        invariant_violations = self._check_invariants(
            state, H_new, D_new, D_dot_new, B_new, S_new, rho_new, beta_new, nu_new
        )

        # ═══════════════════════════════════════════════════════════
        # CONSTRUCT NEW STATE
        # ═══════════════════════════════════════════════════════════

        new_state = USLAState(
            H=H_new,
            D=D_new,
            D_dot=D_dot_new,
            B=B_new,
            S=S_new,
            C=C_new,
            rho=rho_new,
            tau=tau_new,
            J=J_new,
            W=W_new,
            beta=beta_new,
            kappa=kappa_new,
            nu=nu_new,
            delta=delta_new,
            Gamma=Gamma_new,
            cycle=state.cycle + 1,
            blocked=u,
            active_cdis=active_cdis,
            invariant_violations=invariant_violations,
        )

        # Store state in history
        self._state_history.append(new_state)
        if len(self._state_history) > max_hist:
            self._state_history = self._state_history[-max_hist:]

        return new_state

    def _compute_threshold(
        self,
        D_dot: float,
        B: float,
        S: float,
        C: ConvergenceClass,
    ) -> float:
        """Compute adaptive threshold τ(x)."""
        p = self.params
        tau = p.tau_0
        tau *= (1 + p.alpha_D * D_dot)
        tau *= (1 + p.alpha_B * (B - p.B_0))
        tau *= (1 - p.alpha_S * S)
        tau *= p.gamma(C)
        return max(0.1, min(0.5, tau))

    def _compute_jacobian(
        self,
        H: float,
        tau: float,
        D_dot: float,
        B: float,
        S: float,
        C: ConvergenceClass,
    ) -> float:
        """Compute max Jacobian sensitivity."""
        p = self.params

        # Sigmoid derivative at operating point
        z = -(H - tau) / p.epsilon
        z = max(-20, min(20, z))  # Clip to avoid overflow
        exp_neg_z = math.exp(-z)
        sigma_prime = exp_neg_z / ((1 + exp_neg_z) ** 2)

        # ∂g/∂H
        dg_dH = abs(sigma_prime / p.epsilon)

        # Other partial derivatives
        gamma_C = p.gamma(C)
        factor1 = (1 + p.alpha_B * (B - p.B_0)) * (1 - p.alpha_S * S) * gamma_C
        factor2 = (1 + p.alpha_D * D_dot) * (1 - p.alpha_S * S) * gamma_C
        factor3 = (1 + p.alpha_D * D_dot) * (1 + p.alpha_B * (B - p.B_0)) * gamma_C

        dg_dD_dot = abs(sigma_prime * p.tau_0 * p.alpha_D * factor1 / p.epsilon)
        dg_dB = abs(sigma_prime * p.tau_0 * p.alpha_B * factor2 / p.epsilon)
        dg_dS = abs(sigma_prime * p.tau_0 * p.alpha_S * factor3 / p.epsilon)

        return max(dg_dH, dg_dD_dot, dg_dB, dg_dS)

    def _classify_convergence(self) -> ConvergenceClass:
        """Classify convergence from variance history."""
        p = self.params

        if len(self._hss_history) < p.variance_window:
            return ConvergenceClass.CONVERGING

        # Compute windowed variances
        variances = []
        for i in range(p.variance_window, len(self._hss_history)):
            window = self._hss_history[i - p.variance_window:i]
            mean_w = sum(window) / p.variance_window
            var_w = sum((x - mean_w) ** 2 for x in window) / p.variance_window
            variances.append(var_w)

        if len(variances) < 2:
            return ConvergenceClass.CONVERGING

        # Linear fit for slope
        n = len(variances)
        x_mean = (n - 1) / 2
        y_mean = sum(variances) / n

        numerator = sum((i - x_mean) * (variances[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        slope = numerator / denominator if denominator > 0 else 0

        mean_var = y_mean

        if slope < -p.convergence_slope_threshold and mean_var < p.oscillation_variance_threshold:
            return ConvergenceClass.CONVERGING
        elif abs(slope) < p.convergence_slope_threshold and mean_var > p.oscillation_variance_threshold:
            return ConvergenceClass.OSCILLATING
        else:
            return ConvergenceClass.DIVERGING

    def _estimate_coupling(self) -> float:
        """Estimate topology coupling κ."""
        if len(self._hss_history) < 5 or len(self._depth_history) < 5:
            return 1.0

        H = self._hss_history[-20:] if len(self._hss_history) >= 20 else self._hss_history
        D = [float(d) for d in (self._depth_history[-20:] if len(self._depth_history) >= 20 else self._depth_history)]

        if len(H) != len(D):
            return 1.0

        n = len(H)
        mean_H = sum(H) / n
        mean_D = sum(D) / n

        cov = sum((H[i] - mean_H) * (D[i] - mean_D) for i in range(n)) / n
        var_H = sum((h - mean_H) ** 2 for h in H) / n
        var_D = sum((d - mean_D) ** 2 for d in D) / n

        if var_H * var_D == 0:
            return 1.0

        return abs(cov / ((var_H * var_D) ** 0.5))

    def _compute_variance_velocity(self) -> float:
        """Compute second derivative of variance."""
        p = self.params

        if len(self._hss_history) < p.variance_window * 2:
            return 0.0

        variances = []
        for i in range(p.variance_window, len(self._hss_history)):
            window = self._hss_history[i - p.variance_window:i]
            mean_w = sum(window) / p.variance_window
            var_w = sum((x - mean_w) ** 2 for x in window) / p.variance_window
            variances.append(var_w)

        if len(variances) < 3:
            return 0.0

        # Second derivative
        d2 = variances[-1] - 2 * variances[-2] + variances[-3]
        return d2

    def _update_exception_window(
        self,
        W: bool,
        C: ConvergenceClass,
        S: float,
        beta: float,
        H: float,
    ) -> bool:
        """Update exception window state."""
        p = self.params

        if not W:
            # Activation check
            if C == ConvergenceClass.DIVERGING:
                self._stable_cycle_counter = 0
                return True
            if S > p.exception_shear_threshold:
                self._stable_cycle_counter = 0
                return True
            if beta > p.exception_block_rate_threshold:
                self._stable_cycle_counter = 0
                return True
            return False
        else:
            # Deactivation check
            if C == ConvergenceClass.CONVERGING and H >= 0.5:
                self._stable_cycle_counter += 1
                if self._stable_cycle_counter >= p.exception_stable_cycles:
                    self._stable_cycle_counter = 0
                    return False
            else:
                self._stable_cycle_counter = 0
            return True

    def _compute_instantaneous_stability(
        self,
        H: float,
        D_dot: float,
        B: float,
        S: float,
        blocked: bool,
    ) -> float:
        """Compute instantaneous stability S(x)."""
        p = self.params

        s_H = min(1.0, H / 0.5) if H > 0.2 else 0.0
        s_D = max(0, 1 - abs(D_dot) / 3.0)
        s_B = max(0, 1 - B / 10.0)
        s_S = max(0, 1 - S / 0.5)
        s_u = 0.8 if blocked else 1.0

        return (
            p.w_H * s_H +
            p.w_D * s_D +
            p.w_B * s_B +
            p.w_S * s_S +
            p.w_u * s_u
        )

    def _compute_shear(self, state: USLAState) -> float:
        """Estimate shear from current state (simplified)."""
        # In full implementation, would compute from depth-binned HSS
        return state.S

    def _count_defects(
        self,
        H: float,
        D_dot: float,
        B: float,
        S: float,
        C: ConvergenceClass,
        beta: float,
        J: float,
    ) -> int:
        """Count active CDI defects (simplified)."""
        count = 0

        # CDI-001: Dynamical Brittleness (J > 10)
        if J > 10:
            count += 1

        # CDI-003: Region instability (low H)
        if H < 0.2:
            count += 1

        # CDI-004: Shear attractor (high S)
        if S > 0.3:
            count += 1

        # CDI-007: High exception usage (tracked via beta)
        if beta > 0.3:
            count += 1

        # CDI-010: Diverging (C = DIVERGING)
        if C == ConvergenceClass.DIVERGING:
            count += 1

        return count

    def _compute_tgrs(
        self,
        H: float,
        C: ConvergenceClass,
        S: float,
        B: float,
        delta: int,
    ) -> float:
        """Compute TGRS readiness score."""
        p = self.params

        # H_score
        H_score = min(1.0, H / 0.5) if H > 0.2 else 0.0

        # C_score
        c_map = {
            ConvergenceClass.CONVERGING: 1.0,
            ConvergenceClass.OSCILLATING: 0.5,
            ConvergenceClass.DIVERGING: 0.0,
        }
        C_score = c_map[C]

        # S_score
        S_score = max(0, 1 - 2 * S)

        # B_score
        B_score = max(0, 1 - B / 10.0)

        # P_score (based on defects)
        P_score = max(0, 1 - delta / 5.0)

        return (
            p.tgrs_w_H * H_score +
            p.tgrs_w_C * C_score +
            p.tgrs_w_S * S_score +
            p.tgrs_w_B * B_score +
            p.tgrs_w_P * P_score
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # INVARIANT CHECKING (INV-001 through INV-008)
    # ═══════════════════════════════════════════════════════════════════════════

    def _check_invariants(
        self,
        prev_state: USLAState,
        H: float,
        D: int,
        D_dot: float,
        B: float,
        S: float,
        rho: float,
        beta: float,
        nu: float,
    ) -> List[str]:
        """
        Check all 8 invariants from USLA v0.1.

        Returns list of violated invariant IDs.
        """
        violations = []

        if not self._check_inv_001_shear_monotonicity(prev_state.S, S):
            violations.append("INV-001")

        if not self._check_inv_002_bf_depth_gradient(B, D):
            violations.append("INV-002")

        if not self._check_inv_003_variance_lipschitz(nu):
            violations.append("INV-003")

        if not self._check_inv_004_cut_coherence(H, S, B):
            violations.append("INV-004")

        if not self._check_inv_005_stability_of_stability(prev_state.rho, rho):
            violations.append("INV-005")

        if not self._check_inv_006_block_rate_stationarity(prev_state.beta, beta):
            violations.append("INV-006")

        if not self._check_inv_007_exception_conservation(beta):
            violations.append("INV-007")

        if not self._check_inv_008_depth_boundedness(D):
            violations.append("INV-008")

        return violations

    def _check_inv_001_shear_monotonicity(self, prev_S: float, curr_S: float) -> bool:
        """
        INV-001: Shear Monotonicity.

        |S_{t+1} - S_t| <= ε_shear (default 0.05)

        Shear should not jump discontinuously between cycles.
        """
        return abs(curr_S - prev_S) <= self.params.inv_001_shear_delta_max

    def _check_inv_002_bf_depth_gradient(self, B: float, D: int) -> bool:
        """
        INV-002: BF-Depth Cross-Gradient.

        The branch factor should scale reasonably with depth.
        |B - B_expected(D)| <= ε_bf (default 1.0)

        Expected: B grows sub-linearly with D (sqrt approximation).
        """
        # Approximate expected BF at depth D
        B_expected = self.params.B_0 * (1 + 0.1 * math.sqrt(max(0, D)))
        return abs(B - B_expected) <= self.params.inv_002_bf_depth_gradient_max

    def _check_inv_003_variance_lipschitz(self, nu: float) -> bool:
        """
        INV-003: HSS-Variance Lipschitz.

        |ν| = |d²Var(H)/dt²| <= ε_lip (default 0.02)

        Variance velocity should be bounded.
        """
        return abs(nu) <= self.params.inv_003_variance_lipschitz

    def _check_inv_004_cut_coherence(self, H: float, S: float, B: float) -> bool:
        """
        INV-004: Minimal-Cut Coherence.

        STUB: In full implementation, requires topology data (min-cut capacity).
        For now, approximate as: high HSS + low shear implies good cut coherence.

        TODO: Attach to real TDA min-cut computation when available.
        Real implementation would compute:
            cut_capacity = min_cut(proof_dag_at_depth_d)
            return cut_capacity >= ε_cut
        """
        # Synthetic coherence: coherent if healthy and low shear
        synthetic_coherence = H * (1 - S) * (1 / max(1, B / 3))
        return synthetic_coherence >= self.params.inv_004_cut_coherence_min

    def _check_inv_005_stability_of_stability(self, prev_rho: float, curr_rho: float) -> bool:
        """
        INV-005: Stability-of-Stability.

        |ρ_{t+1} - ρ_t| <= ε_rho (default 0.1)

        RSI should not jump discontinuously.
        """
        return abs(curr_rho - prev_rho) <= self.params.inv_005_rho_delta_max

    def _check_inv_006_block_rate_stationarity(self, prev_beta: float, curr_beta: float) -> bool:
        """
        INV-006: Block Rate Stationarity.

        |β_{t+1} - β_t| <= ε_beta (default 0.1)

        Block rate should evolve smoothly.
        """
        return abs(curr_beta - prev_beta) <= self.params.inv_006_beta_delta_max

    def _check_inv_007_exception_conservation(self, beta: float) -> bool:
        """
        INV-007: Exception Conservation.

        β <= β_max (default 0.2)

        Exception window usage should be bounded.
        """
        return beta <= self.params.inv_007_beta_max

    def _check_inv_008_depth_boundedness(self, D: int) -> bool:
        """
        INV-008: Depth Boundedness.

        D <= D_max (default 20)

        Proof depth should remain bounded.
        """
        return D <= self.params.inv_008_depth_max

    def get_invariant_status(self, state: USLAState, prev_state: Optional[USLAState] = None) -> Dict[str, bool]:
        """
        Get status of all invariants for a state.

        Returns dict mapping invariant ID to pass/fail.
        """
        if prev_state is None:
            prev_state = state

        violations = self._check_invariants(
            prev_state, state.H, state.D, state.D_dot, state.B,
            state.S, state.rho, state.beta, state.nu
        )

        return {
            "INV-001": "INV-001" not in violations,
            "INV-002": "INV-002" not in violations,
            "INV-003": "INV-003" not in violations,
            "INV-004": "INV-004" not in violations,
            "INV-005": "INV-005" not in violations,
            "INV-006": "INV-006" not in violations,
            "INV-007": "INV-007" not in violations,
            "INV-008": "INV-008" not in violations,
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # CDI TRIGGER FUNCTIONS (CDI-001 through CDI-010)
    # ═══════════════════════════════════════════════════════════════════════════

    def _count_defects_full(
        self,
        H: float,
        D: int,
        D_dot: float,
        B: float,
        S: float,
        C: ConvergenceClass,
        beta: float,
        J: float,
        kappa: float,
        nu: float,
    ) -> Tuple[int, List[str]]:
        """
        Count active CDI defects with full coverage (all 10 CDIs).

        Returns (count, list of active CDI IDs).
        """
        active = []

        # CDI-001: Dynamical Brittleness
        if self._trigger_cdi_001_dynamical_brittleness(J):
            active.append("CDI-001")

        # CDI-002: Asymmetric Shear
        if self._trigger_cdi_002_asymmetric_shear():
            active.append("CDI-002")

        # CDI-003: Region Instability
        if self._trigger_cdi_003_region_instability(H):
            active.append("CDI-003")

        # CDI-004: Shear Attractor
        if self._trigger_cdi_004_shear_attractor(S):
            active.append("CDI-004")

        # CDI-005: Runaway Depth
        if self._trigger_cdi_005_runaway_depth(D, D_dot):
            active.append("CDI-005")

        # CDI-006: Complexity Avoidance
        if self._trigger_cdi_006_complexity_avoidance():
            active.append("CDI-006")

        # CDI-007: Exception Exhaustion
        if self._trigger_cdi_007_exception_exhaustion(beta):
            active.append("CDI-007")

        # CDI-008: Coupling Pathology
        if self._trigger_cdi_008_coupling_pathology():
            active.append("CDI-008")

        # CDI-009: Variance Blowup
        if self._trigger_cdi_009_variance_blowup():
            active.append("CDI-009")

        # CDI-010: Fixed-Point Multiplicity
        if self._trigger_cdi_010_fixed_point_multiplicity(C):
            active.append("CDI-010")

        return len(active), active

    def _trigger_cdi_001_dynamical_brittleness(self, J: float) -> bool:
        """
        CDI-001: Dynamical Brittleness.

        Trigger: J > 10 (Jacobian indicates high sensitivity).
        """
        return J > self.params.J_threshold

    def _trigger_cdi_002_asymmetric_shear(self) -> bool:
        """
        CDI-002: Asymmetric Shear.

        Trigger: Large gradient in HSS across depth bins.
        Approximated as: max(shear_history) - min(shear_history) > threshold
        over recent window.

        In full implementation, would compute from depth-stratified HSS.
        """
        p = self.params
        if len(self._shear_history) < 5:
            return False

        window = self._shear_history[-10:] if len(self._shear_history) >= 10 else self._shear_history
        shear_range = max(window) - min(window)
        return shear_range > p.cdi_002_shear_gradient_max

    def _trigger_cdi_003_region_instability(self, H: float) -> bool:
        """
        CDI-003: Region Instability.

        Trigger: H < 0.2 (HSS critically low).
        """
        return H < 0.2

    def _trigger_cdi_004_shear_attractor(self, S: float) -> bool:
        """
        CDI-004: Shear Attractor.

        Trigger: S > 0.3 with increasing trend.
        """
        if S <= 0.3:
            return False

        # Check for increasing trend
        if len(self._shear_history) < 3:
            return S > 0.3

        recent = self._shear_history[-3:]
        increasing = all(recent[i] <= recent[i+1] for i in range(len(recent)-1))
        return S > 0.3 and increasing

    def _trigger_cdi_005_runaway_depth(self, D: int, D_dot: float) -> bool:
        """
        CDI-005: Runaway Depth.

        Trigger: D > 15 AND depth is accelerating (D_dot increasing).
        """
        p = self.params
        if D <= p.cdi_005_depth_runaway:
            return False

        if len(self._d_dot_history) < 3:
            return D > p.cdi_005_depth_runaway

        # Check for acceleration
        recent = self._d_dot_history[-3:]
        d_dot_accel = recent[-1] - recent[0]
        return D > p.cdi_005_depth_runaway and d_dot_accel > p.cdi_005_accel_threshold

    def _trigger_cdi_006_complexity_avoidance(self) -> bool:
        """
        CDI-006: Complexity Avoidance.

        Trigger: Depth stagnant (avg D_dot < 0.1) over stagnation window.
        System is not exploring complexity.
        """
        p = self.params
        window = p.cdi_006_stagnation_window

        if len(self._d_dot_history) < window:
            return False

        recent = self._d_dot_history[-window:]
        avg_d_dot = sum(abs(d) for d in recent) / len(recent)
        return avg_d_dot < p.cdi_006_d_dot_avg_min

    def _trigger_cdi_007_exception_exhaustion(self, beta: float) -> bool:
        """
        CDI-007: Exception Exhaustion.

        Trigger: β > 0.3 (high sustained blocking rate).
        """
        return beta > 0.3

    def _trigger_cdi_008_coupling_pathology(self) -> bool:
        """
        CDI-008: Coupling Pathology.

        Trigger: κ oscillation amplitude > 0.2 for consecutive cycles.
        """
        p = self.params
        cycles = p.cdi_008_oscillation_cycles

        if len(self._kappa_history) < cycles + 1:
            return False

        # Check for oscillation: alternating large changes
        recent = self._kappa_history[-(cycles + 1):]
        deltas = [abs(recent[i+1] - recent[i]) for i in range(len(recent)-1)]

        # All deltas must exceed threshold
        return all(d > p.cdi_008_kappa_oscillation for d in deltas)

    def _trigger_cdi_009_variance_blowup(self) -> bool:
        """
        CDI-009: Variance Blowup.

        Trigger: |ν| > 0.02 sustained for multiple cycles.
        """
        p = self.params
        cycles = p.cdi_009_sustained_cycles

        if len(self._nu_history) < cycles:
            return False

        recent = self._nu_history[-cycles:]
        return all(abs(n) > p.cdi_009_nu_threshold for n in recent)

    def _trigger_cdi_010_fixed_point_multiplicity(self, C: ConvergenceClass) -> bool:
        """
        CDI-010: Fixed-Point Multiplicity.

        Trigger: C = DIVERGING (system has no stable fixed point).
        """
        return C == ConvergenceClass.DIVERGING

    def get_active_cdis(self, state: USLAState) -> List[str]:
        """Get list of active CDI IDs for a state."""
        return state.active_cdis

    # ═══════════════════════════════════════════════════════════════════════════
    # HARD MODE GATE (Law 7)
    # ═══════════════════════════════════════════════════════════════════════════

    def is_hard_ok(self, state: USLAState, prev_state: Optional[USLAState] = None) -> bool:
        """
        Evaluate HARD mode activation envelope (Law 7).

        HARD_OK(x) ⟺ (x ∈ Ω) ∧ (I(x)) ∧ (D(x) = ∅) ∧ (ρ ≥ ρ_min)

        Args:
            state: Current state to evaluate
            prev_state: Previous state for invariant checking (optional)

        Returns:
            True if HARD mode can be safely activated
        """
        p = self.params

        # Condition 1: x ∈ Ω (safe region)
        if not state.is_in_safe_region(p):
            return False

        # Condition 2: I(x) - all invariants satisfied
        if prev_state is None:
            prev_state = state
        invariant_violations = self._check_invariants(
            prev_state, state.H, state.D, state.D_dot, state.B,
            state.S, state.rho, state.beta, state.nu
        )
        if invariant_violations:
            return False

        # Condition 3: D(x) = ∅ (no active defects)
        if state.delta > 0:
            return False

        # Condition 4: ρ ≥ ρ_min
        if state.rho < p.rho_min:
            return False

        return True

    def get_hard_mode_status(self, state: USLAState, prev_state: Optional[USLAState] = None) -> Dict[str, Any]:
        """
        Get detailed HARD mode status breakdown.

        Returns dict with each condition's status.
        """
        p = self.params
        if prev_state is None:
            prev_state = state

        in_safe_region = state.is_in_safe_region(p)
        invariant_violations = self._check_invariants(
            prev_state, state.H, state.D, state.D_dot, state.B,
            state.S, state.rho, state.beta, state.nu
        )
        no_defects = state.delta == 0
        rho_ok = state.rho >= p.rho_min

        return {
            "hard_ok": in_safe_region and not invariant_violations and no_defects and rho_ok,
            "in_safe_region": in_safe_region,
            "invariants_satisfied": not invariant_violations,
            "invariant_violations": invariant_violations,
            "no_defects": no_defects,
            "active_cdis": state.active_cdis,
            "rho_ok": rho_ok,
            "rho": state.rho,
            "rho_min": p.rho_min,
        }

    def simulate(
        self,
        cycle_inputs: List[CycleInput],
        initial_state: Optional[USLAState] = None,
    ) -> SimulationResult:
        """
        Run full simulation over cycle inputs.

        Args:
            cycle_inputs: List of cycle input data
            initial_state: Starting state (default: initial optimistic)

        Returns:
            SimulationResult with trajectory and analysis
        """
        self.reset()

        state = initial_state or USLAState.initial()
        states = [state]
        red_flags = []
        hard_mode_failures: List[Tuple[int, str]] = []
        inv_violation_counts: Dict[str, int] = {}
        cdi_activation_counts: Dict[str, int] = {}
        first_hard_failure: Optional[int] = None
        prev_hard_ok = True

        for i, cycle_input in enumerate(cycle_inputs):
            prev_state = state
            state = self.step(state, cycle_input)
            states.append(state)

            # Check for red flags
            if flag := self._check_red_flag(state):
                red_flags.append((i + 1, flag))

            # Track HARD mode transitions
            curr_hard_ok = self.is_hard_ok(state, prev_state)
            if prev_hard_ok and not curr_hard_ok:
                # HARD mode just failed
                reason = self._get_hard_failure_reason(state, prev_state)
                hard_mode_failures.append((i + 1, reason))
                if first_hard_failure is None:
                    first_hard_failure = i + 1
            prev_hard_ok = curr_hard_ok

            # Track invariant violations
            for inv_id in state.invariant_violations:
                inv_violation_counts[inv_id] = inv_violation_counts.get(inv_id, 0) + 1

            # Track CDI activations
            for cdi_id in state.active_cdis:
                cdi_activation_counts[cdi_id] = cdi_activation_counts.get(cdi_id, 0) + 1

        # Compute summary statistics
        blocks = sum(1 for s in states if s.blocked)
        rho_values = [s.rho for s in states]
        safe_violations = sum(1 for s in states if not s.is_in_safe_region(self.params))
        defect_cycles = sum(1 for s in states if s.delta > 0)

        return SimulationResult(
            states=states,
            final_state=state,
            total_cycles=len(cycle_inputs),
            blocks=blocks,
            block_rate=blocks / len(states) if states else 0,
            mean_rho=sum(rho_values) / len(rho_values) if rho_values else 0,
            min_rho=min(rho_values) if rho_values else 0,
            safe_region_violations=safe_violations,
            defect_cycles=defect_cycles,
            red_flags=red_flags,
            hard_mode_failures=hard_mode_failures,
            invariant_violations=inv_violation_counts,
            cdi_activations=cdi_activation_counts,
            first_hard_failure_cycle=first_hard_failure,
        )

    def _get_hard_failure_reason(self, state: USLAState, prev_state: USLAState) -> str:
        """Get reason for HARD mode failure."""
        reasons = []

        if not state.is_in_safe_region(self.params):
            reasons.append("OUTSIDE_SAFE_REGION")

        if state.invariant_violations:
            reasons.append(f"INV_VIOLATIONS({','.join(state.invariant_violations)})")

        if state.delta > 0:
            reasons.append(f"CDIS_ACTIVE({','.join(state.active_cdis)})")

        if state.rho < self.params.rho_min:
            reasons.append(f"LOW_RSI({state.rho:.3f})")

        return "; ".join(reasons) if reasons else "UNKNOWN"

    def _check_red_flag(self, state: USLAState) -> Optional[str]:
        """Check for red-flag conditions."""
        p = self.params

        if state.C == ConvergenceClass.DIVERGING:
            return "DIVERGING"

        if state.J > 15:
            return f"JACOBIAN_UNSTABLE (J={state.J:.2f})"

        if state.beta > 0.5:
            return f"EXCESSIVE_BLOCKING (beta={state.beta:.2f})"

        if state.rho < p.rho_min:
            return f"LOW_STABILITY (rho={state.rho:.2f})"

        return None

    def is_red_flag(self, state: USLAState) -> bool:
        """Check if state triggers red flag."""
        return self._check_red_flag(state) is not None

    def find_fixed_points(
        self,
        hss_range: Tuple[float, float] = (0.0, 1.0),
        samples: int = 20,
        tolerance: float = 0.01,
        max_iterations: int = 100,
    ) -> List[USLAState]:
        """
        Find fixed points of the system.

        Uses iterative refinement from multiple starting points.

        Args:
            hss_range: Range of HSS values to sample
            samples: Number of starting points
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations per starting point

        Returns:
            List of fixed point states
        """
        fixed_points = []

        for i in range(samples):
            # Sample starting HSS
            hss_start = hss_range[0] + (hss_range[1] - hss_range[0]) * i / (samples - 1)

            self.reset()
            state = USLAState(H=hss_start, D=5, B=2.0, S=0.1)

            # Iterate until convergence or max iterations
            for _ in range(max_iterations):
                cycle_input = CycleInput(
                    hss=state.H,
                    depth=state.D,
                    branch_factor=state.B,
                    shear=state.S,
                )
                next_state = self.step(state, cycle_input)

                # Check convergence
                diff = sum(abs(a - b) for a, b in zip(state.to_vector(), next_state.to_vector()))
                if diff < tolerance:
                    # Found fixed point
                    is_new = True
                    for fp in fixed_points:
                        fp_diff = sum(abs(a - b) for a, b in zip(fp.to_vector(), next_state.to_vector()))
                        if fp_diff < tolerance * 10:
                            is_new = False
                            break

                    if is_new:
                        fixed_points.append(next_state)
                    break

                state = next_state

        return fixed_points

    def analyze_stability(self) -> StabilityAnalysis:
        """
        Perform stability analysis of the system.

        Returns:
            StabilityAnalysis with fixed points and stability information
        """
        fixed_points = self.find_fixed_points()

        stable = []
        unstable = []
        jacobians = []

        for fp in fixed_points:
            # Compute Jacobian at fixed point
            J = fp.J
            jacobians.append(J)

            # Stability criterion: J < threshold
            if J < self.params.J_threshold:
                stable.append(fp)
            else:
                unstable.append(fp)

        # Find bifurcation parameters (simplified)
        bifurcations = []

        # Test tau_0 sensitivity
        for tau_0_test in [0.1, 0.15, 0.25, 0.3, 0.35, 0.4]:
            old_tau = self.params.tau_0
            self.params.tau_0 = tau_0_test
            fps = self.find_fixed_points(samples=5, max_iterations=50)
            if len(fps) != len(fixed_points):
                bifurcations.append(("tau_0", tau_0_test))
            self.params.tau_0 = old_tau

        # Compute stability margin
        if jacobians:
            margin = 1 - max(jacobians) / 15  # Relative to unstable threshold
        else:
            margin = 1.0

        return StabilityAnalysis(
            fixed_points=fixed_points,
            stable_fixed_points=stable,
            unstable_fixed_points=unstable,
            bifurcation_params=bifurcations,
            jacobian_at_fixed_points=jacobians,
            stability_margin=margin,
        )


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def create_stress_test_inputs(
    num_cycles: int = 100,
    stress_pattern: str = "ramp",
) -> List[CycleInput]:
    """
    Create cycle inputs for stress testing.

    Patterns:
    - "ramp": Gradually decreasing HSS
    - "oscillate": Oscillating HSS
    - "spike": Sudden HSS drops
    - "stable": Constant healthy HSS
    """
    inputs = []

    for i in range(num_cycles):
        if stress_pattern == "ramp":
            hss = max(0.1, 1.0 - i / num_cycles)
        elif stress_pattern == "oscillate":
            hss = 0.5 + 0.4 * math.sin(2 * math.pi * i / 20)
        elif stress_pattern == "spike":
            hss = 0.1 if i % 20 == 0 else 0.8
        else:  # stable
            hss = 0.8

        inputs.append(CycleInput(
            hss=hss,
            depth=5 + i % 3,
            branch_factor=2.0 + 0.5 * math.sin(i / 10),
            shear=0.1 + 0.05 * math.sin(i / 15),
        ))

    return inputs


def run_parameter_sweep(
    param_name: str,
    param_range: Tuple[float, float],
    steps: int = 10,
    num_cycles: int = 50,
) -> List[Tuple[float, SimulationResult]]:
    """
    Sweep a parameter and analyze system response.

    Args:
        param_name: Name of USLAParams attribute to sweep
        param_range: (min, max) range
        steps: Number of steps in sweep
        num_cycles: Cycles per simulation

    Returns:
        List of (param_value, SimulationResult) tuples
    """
    results = []
    inputs = create_stress_test_inputs(num_cycles, "oscillate")

    for i in range(steps):
        value = param_range[0] + (param_range[1] - param_range[0]) * i / (steps - 1)

        params = USLAParams()
        setattr(params, param_name, value)

        sim = USLASimulator(params)
        result = sim.simulate(inputs)
        results.append((value, result))

    return results
