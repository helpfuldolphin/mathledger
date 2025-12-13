"""
Phase X P3: First-Light Shadow Runner

This module implements the FirstLightShadowRunner class for executing shadow
experiments. See docs/system_law/Phase_X_P3_Spec.md for full specification.

SHADOW MODE CONTRACT:
- NEVER modifies governance decisions
- NEVER enforces abort conditions
- All outputs are observational only
- Red-flags are LOGGED, not ACTED upon
- This code runs OFFLINE only, never in production governance paths

Status: P3 IMPLEMENTATION (OFFLINE, SHADOW-ONLY)

TDA Integration (PhaseX-TDA-P3):
- TDAMonitor is called at each window boundary in finalize()
- Computes SNS, PCS, DRS, HSS trajectories from window data
- TDA metrics are included in FirstLightResult.tda_metrics
- See: docs/system_law/Phase_X_Prelaunch_Review.md Section 3.3
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

from backend.ht.tda_monitor import TDAMonitor
from backend.synthetic.pathology_injection import (
    inject_drift,
    inject_oscillation,
    inject_spike,
)
from backend.topology.first_light.config import FirstLightConfig, FirstLightResult
from backend.topology.first_light.delta_p_computer import DeltaPComputer, compute_slope
from backend.topology.first_light.red_flag_observer import RedFlagObserver
from backend.topology.first_light.metrics_window import MetricsAccumulator
from backend.topology.first_light.noise_harness import (
    P3NoiseConfig,
    P3NoiseHarness,
)

__all__ = [
    "FirstLightShadowRunner",
    "CycleObservation",
    "SyntheticStateGenerator",
]


@dataclass
class CycleObservation:
    """
    Observation from a single cycle.

    SHADOW MODE: This is an observation only. It does NOT trigger
    any control flow changes.
    """

    cycle: int
    timestamp: str

    # Runner outcome (synthetic)
    success: bool
    depth: Optional[int]

    # USLA state snapshot
    H: float
    rho: float
    tau: float
    beta: float
    in_omega: bool

    # Governance
    real_blocked: bool
    sim_blocked: bool
    governance_aligned: bool

    # HARD mode
    hard_ok: bool

    # RFL (abstention)
    abstained: bool

    # Slice Identity (Phase X pre-execution blocker)
    identity_stable: bool = True

    # Noise contribution (P3 Noise Model)
    noise_decision: Optional[str] = None
    noise_caused_failure: bool = False
    noise_delta_p_contribution: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cycle": self.cycle,
            "timestamp": self.timestamp,
            "runner": {
                "success": self.success,
                "depth": self.depth,
            },
            "usla_state": {
                "H": round(self.H, 4),
                "rho": round(self.rho, 4),
                "tau": round(self.tau, 4),
                "beta": round(self.beta, 4),
                "in_omega": self.in_omega,
            },
            "governance": {
                "real_blocked": self.real_blocked,
                "sim_blocked": self.sim_blocked,
                "aligned": self.governance_aligned,
            },
            "hard_ok": self.hard_ok,
            "abstained": self.abstained,
            "identity_stable": self.identity_stable,
            "noise": {
                "decision": self.noise_decision,
                "caused_failure": self.noise_caused_failure,
                "delta_p_contribution": round(self.noise_delta_p_contribution, 6),
            } if self.noise_decision else None,
        }


class SyntheticStateGenerator:
    """
    Generates synthetic USLA state for offline testing.

    SHADOW MODE: This generates synthetic data only for OFFLINE testing.
    It does NOT connect to any real USLA or governance system.

    The generator simulates realistic state transitions with:
    - Gradual learning (improving success rate)
    - RSI fluctuations
    - Occasional Omega exits
    - Rare HARD failures
    - P3 Noise Model integration (per P3_Noise_Model_Spec.md)
    """

    def __init__(
        self,
        tau_0: float = 0.20,
        seed: Optional[int] = None,
        learning_rate: float = 0.001,
        noise_scale: float = 0.05,
        noise_config: Optional[P3NoiseConfig] = None,
        pathology_mode: str = "none",
        pathology_params: Optional[Dict[str, float]] = None,
        planned_cycles: Optional[int] = None,
    ) -> None:
        """
        Initialize state generator.

        Args:
            tau_0: Initial threshold
            seed: Random seed for reproducibility
            learning_rate: Rate of improvement over cycles
            noise_scale: Scale of random noise
            noise_config: Optional P3 noise configuration (None = disabled)
            pathology_mode: Optional pathology injection mode (TEST-ONLY, default: none)
            pathology_params: Optional parameters for pathology injection
            planned_cycles: Optional planned cycle count (used to place spike default)
        """
        self.tau_0 = tau_0
        self.learning_rate = learning_rate
        self.noise_scale = noise_scale
        self._seed = seed

        # Use isolated Random instance for reproducibility
        self._rng = random.Random(seed)

        # Initial state
        self._H = 0.5
        self._rho = 0.7
        self._tau = tau_0
        self._beta = 0.1
        self._cycle = 0

        # Success probability starts low and improves
        self._base_success_prob = 0.6

        # P3 Noise Harness integration
        self._noise_harness: Optional[P3NoiseHarness] = None
        if noise_config is not None:
            self._noise_harness = P3NoiseHarness(
                noise_config=noise_config,
                seed=seed if seed is not None else 42,
            )

        # Pathology injection (TEST-ONLY)
        self._pathology_mode = pathology_mode
        self._pathology_params = pathology_params or {}
        self._planned_cycles = planned_cycles or 0
        self._effective_pathology_params = self._resolve_pathology_params(
            pathology_mode,
            self._pathology_params,
            self._planned_cycles,
        )
        self._H_history: List[float] = []

    def step(self) -> Dict[str, Any]:
        """
        Generate next state.

        Returns:
            Dictionary with H, rho, tau, beta, in_omega, success, hard_ok, etc.
            Includes noise contribution if noise harness is enabled.
        """
        self._cycle += 1

        # Step noise harness if enabled
        if self._noise_harness:
            self._noise_harness.step_cycle()

        # Simulate learning: success probability improves over time
        learning_bonus = min(0.3, self._cycle * self.learning_rate)
        success_prob = min(0.95, self._base_success_prob + learning_bonus)

        # Determine success
        success = self._rng.random() < success_prob

        # Update H based on success
        if success:
            self._H = min(1.0, self._H + 0.02 + self._rng.gauss(0, self.noise_scale * 0.5))
        else:
            self._H = max(0.0, self._H - 0.05 + self._rng.gauss(0, self.noise_scale * 0.5))

        # Update rho (RSI) - tends toward stable value with noise
        target_rho = 0.8 if success else 0.6
        self._rho = self._rho + 0.1 * (target_rho - self._rho) + self._rng.gauss(0, self.noise_scale)
        self._rho = max(0.0, min(1.0, self._rho))

        # Update tau (threshold) - small drift
        self._tau = self._tau + self._rng.gauss(0, self.noise_scale * 0.1)
        self._tau = max(0.0, min(1.0, self._tau))

        # Update beta (block rate proxy) - inversely related to success
        if success:
            self._beta = max(0.0, self._beta - 0.01 + self._rng.gauss(0, self.noise_scale * 0.2))
        else:
            self._beta = min(1.0, self._beta + 0.02 + self._rng.gauss(0, self.noise_scale * 0.2))

        # Determine Omega membership
        # In Omega when H > tau and rho > 0.5
        in_omega = self._H > self._tau and self._rho > 0.5

        # HARD mode check - fails rarely
        hard_ok = self._rng.random() > 0.02  # 2% HARD failure rate

        # Governance simulation
        # Real and sim usually agree, occasional divergence
        real_blocked = self._beta > 0.7
        sim_blocked = real_blocked if self._rng.random() > 0.05 else not real_blocked
        governance_aligned = real_blocked == sim_blocked

        # Abstention (RFL mode) - abstain when uncertain
        abstained = self._H < 0.3 or self._rho < 0.4

        # Depth - varies with success
        depth = self._rng.randint(3, 8) if success else self._rng.randint(1, 4)

        # Build base state dict
        state = {
            "H": self._H,
            "rho": self._rho,
            "tau": self._tau,
            "beta": self._beta,
            "in_omega": in_omega,
            "success": success,
            "hard_ok": hard_ok,
            "real_blocked": real_blocked,
            "sim_blocked": sim_blocked,
            "governance_aligned": governance_aligned,
            "abstained": abstained,
            "depth": depth,
            # Noise fields (will be populated below if noise enabled)
            "noise_decision": None,
            "noise_caused_failure": False,
            "noise_delta_p_contribution": 0.0,
        }

        # Apply P3 noise if harness is enabled
        if self._noise_harness:
            state = self._noise_harness.apply_noise(
                state,
                item=f"item_{self._cycle}",
            )
            # Extract noise fields
            noise_info = state.get("noise", {})
            state["noise_decision"] = noise_info.get("decision")
            state["noise_caused_failure"] = state.get("noise_caused_failure", False)
            state["noise_delta_p_contribution"] = noise_info.get("delta_p_contribution", 0.0)

        # Apply pathology injection to H trajectory (TEST-ONLY; default disabled)
        adjusted_H = self._apply_pathology(state["H"])
        if adjusted_H is not None:
            state["H"] = adjusted_H
            state["in_omega"] = state["H"] > state["tau"] and state["rho"] > 0.5

        return state

    def get_noise_stability_report(self) -> Optional[Dict[str, Any]]:
        """
        Get noise stability report if noise harness is enabled.

        Returns:
            Stability report dict or None if noise disabled
        """
        if self._noise_harness:
            return self._noise_harness.get_stability_report()
        return None

    def get_noise_delta_p_contribution(self, window_size: int = 50) -> float:
        """
        Get cumulative delta_p contribution from noise.

        Args:
            window_size: Number of recent cycles to consider

        Returns:
            Sum of delta_p contributions from noise (0.0 if noise disabled)
        """
        if self._noise_harness:
            return self._noise_harness.get_delta_p_noise_contribution(window_size)
        return 0.0

    def reset(self) -> None:
        """Reset generator to initial state."""
        self._H = 0.5
        self._rho = 0.7
        self._tau = self.tau_0
        self._beta = 0.1
        self._cycle = 0
        # Reset RNG to original seed for reproducibility
        self._rng = random.Random(self._seed)
        # Reset noise harness if enabled
        if self._noise_harness:
            self._noise_harness.reset()

        # Reset pathology tracking
        self._H_history = []
        self._effective_pathology_params = self._resolve_pathology_params(
            self._pathology_mode,
            self._pathology_params,
            self._planned_cycles,
        )

    def _apply_pathology(self, H_value: float) -> Optional[float]:
        """
        Apply pathology injection to the observed H trajectory.

        Returns:
            Adjusted H value for this cycle (or None if pathology disabled)
        """
        if self._pathology_mode == "none":
            return None

        # Track baseline trajectory (never mutated in place)
        self._H_history.append(H_value)
        baseline = list(self._H_history)

        try:
            if self._pathology_mode == "spike":
                target = int(self._effective_pathology_params.get("at", 0))
                magnitude = self._effective_pathology_params.get("magnitude", 0.75)
                if target >= len(baseline):
                    return baseline[-1]
                adjusted = inject_spike(baseline, at=target, magnitude=magnitude)
            elif self._pathology_mode == "drift":
                slope = self._effective_pathology_params.get("slope", 0.0025)
                adjusted = inject_drift(baseline, slope=slope)
            elif self._pathology_mode == "oscillation":
                period = int(self._effective_pathology_params.get("period", 10))
                amplitude = self._effective_pathology_params.get("amplitude", 0.1)
                adjusted = inject_oscillation(baseline, period=period, amplitude=amplitude)
            else:
                return None
        except Exception:
            # Fail closed: if injection fails, return baseline to avoid breaking harness
            return baseline[-1]

        return adjusted[-1]

    def _resolve_pathology_params(
        self,
        mode: str,
        params: Dict[str, float],
        planned_cycles: int,
    ) -> Dict[str, float]:
        """Resolve pathology parameters with defaults for reporting/annotation."""
        if mode == "spike":
            default_at = max(0, int(0.5 * planned_cycles)) if planned_cycles else 0
            return {
                "magnitude": params.get("magnitude", 0.75),
                "at": int(params.get("at", default_at)),
            }
        if mode == "drift":
            return {"slope": params.get("slope", 0.0025)}
        if mode == "oscillation":
            return {
                "period": int(params.get("period", 10)),
                "amplitude": params.get("amplitude", 0.1),
            }
        return {}

    def get_pathology_params(self) -> Dict[str, float]:
        """Return effective pathology parameters (resolved with defaults)."""
        return dict(self._effective_pathology_params)


class FirstLightShadowRunner:
    """
    First-Light shadow experiment runner.

    SHADOW MODE CONTRACT:
    - NEVER modifies governance decisions
    - NEVER enforces abort conditions
    - All outputs are observational only
    - Red-flags are LOGGED, not ACTED upon
    - This runs OFFLINE only with synthetic data

    See: docs/system_law/Phase_X_P3_Spec.md Section 2.4
    """

    def __init__(
        self,
        config: FirstLightConfig,
        seed: Optional[int] = None,
        pathology_mode: str = "none",
        pathology_params: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize runner with validated config.

        SHADOW MODE: Enforces shadow_mode=True at initialization.

        Args:
            config: FirstLightConfig instance
            seed: Optional random seed for reproducibility

        Raises:
            ValueError: If shadow_mode is False
        """
        # SHADOW MODE: Enforce at initialization
        if not config.shadow_mode:
            raise ValueError(
                "SHADOW MODE VIOLATION: FirstLightShadowRunner requires "
                "shadow_mode=True. Active mode is not authorized in Phase X P3."
            )

        # Validate config
        config.validate_or_raise()

        self.config = config
        self._seed = seed

        # Initialize components
        self._state_generator = SyntheticStateGenerator(
            tau_0=config.tau_0,
            seed=seed,
            pathology_mode=pathology_mode,
            pathology_params=pathology_params,
            planned_cycles=config.total_cycles,
        )
        self._red_flag_observer = RedFlagObserver(config)
        self._delta_p_computer = DeltaPComputer(window_size=config.success_window)
        self._metrics_accumulator = MetricsAccumulator(window_size=config.success_window)
        self._tda_monitor = TDAMonitor()

        # Run state
        self._cycle = 0
        self._observations: List[CycleObservation] = []
        self._start_time: Optional[str] = None
        self._end_time: Optional[str] = None
        self._run_id: str = ""

    def run(self) -> FirstLightResult:
        """
        Execute shadow experiment.

        SHADOW MODE: This runs the experiment in observation-only mode.
        No governance decisions are modified. All outputs are logged
        but never influence actual system behavior.

        Returns:
            FirstLightResult with all metrics and observations
        """
        # Generate run ID
        self._run_id = f"fl_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        self._start_time = datetime.now(timezone.utc).isoformat()

        # Run all cycles
        for obs in self.run_cycles(self.config.total_cycles):
            self._observations.append(obs)

        self._end_time = datetime.now(timezone.utc).isoformat()

        return self.finalize()

    def run_cycles(self, n: int) -> Iterator[CycleObservation]:
        """
        Run N cycles and yield observations.

        SHADOW MODE: Each cycle is observed but not controlled.

        Args:
            n: Number of cycles to run

        Yields:
            CycleObservation for each cycle
        """
        for i in range(n):
            self._cycle += 1
            obs = self._run_single_cycle()
            yield obs

    def _run_single_cycle(self) -> CycleObservation:
        """
        Execute a single cycle and return observation.

        SHADOW MODE: This is observation only. No control flow changes.
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        # Generate synthetic state
        state = self._state_generator.step()

        # Create observation (including noise fields from P3 Noise Model)
        obs = CycleObservation(
            cycle=self._cycle,
            timestamp=timestamp,
            success=state["success"],
            depth=state["depth"],
            H=state["H"],
            rho=state["rho"],
            tau=state["tau"],
            beta=state["beta"],
            in_omega=state["in_omega"],
            real_blocked=state["real_blocked"],
            sim_blocked=state["sim_blocked"],
            governance_aligned=state["governance_aligned"],
            hard_ok=state["hard_ok"],
            abstained=state["abstained"],
            noise_decision=state.get("noise_decision"),
            noise_caused_failure=state.get("noise_caused_failure", False),
            noise_delta_p_contribution=state.get("noise_delta_p_contribution", 0.0),
        )

        # Update red-flag observer (LOGGING only)
        self._red_flag_observer.observe(
            cycle=self._cycle,
            state=state,
            hard_ok=state["hard_ok"],
            governance_aligned=state["governance_aligned"],
        )

        # Update delta-p computer
        self._delta_p_computer.update(
            cycle=self._cycle,
            success=state["success"],
            in_omega=state["in_omega"],
            hard_ok=state["hard_ok"],
            rsi=state["rho"],
            abstained=state["abstained"],
        )

        # Update metrics accumulator (pass H for TDA tracking)
        self._metrics_accumulator.add(
            success=state["success"],
            abstained=state["abstained"],
            in_omega=state["in_omega"],
            hard_ok=state["hard_ok"],
            rsi=state["rho"],
            blocked=state["real_blocked"],
            H=state["H"],
        )

        return obs

    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics snapshot.

        SHADOW MODE: Metrics are observational only.

        Returns:
            MetricsSnapshot dictionary
        """
        cumulative = self._metrics_accumulator.get_cumulative_rates()
        delta_p = self._delta_p_computer.compute()

        # Get noise contribution to delta_p
        noise_delta_p = self._state_generator.get_noise_delta_p_contribution(
            window_size=self.config.success_window
        )

        return {
            "cycle": self._cycle,
            "mode": "SHADOW",
            "cumulative": cumulative,
            "delta_p": {
                "success": delta_p.delta_p_success,
                "abstention": delta_p.delta_p_abstention,
                "noise_contribution": round(noise_delta_p, 6),
            },
            "windows_completed": self._metrics_accumulator.total_windows,
        }

    def get_red_flag_status(self) -> Dict[str, Any]:
        """
        Get current red-flag observation status.

        SHADOW MODE: This is observational only. The status does NOT
        trigger any control flow changes.

        Returns:
            RedFlagStatus dictionary
        """
        summary = self._red_flag_observer.get_summary()
        would_abort, reason = self._red_flag_observer.hypothetical_should_abort()

        return {
            "mode": "SHADOW",
            "total_observations": summary.total_observations,
            "observations_by_type": summary.observations_by_type,
            "cdi_010_activations": summary.cdi_010_activations,
            "hypothetical_abort": {
                "would_abort": would_abort,
                "reason": reason,
            },
        }

    def get_noise_stability_report(self) -> Optional[Dict[str, Any]]:
        """
        Get noise stability report from P3 Noise Harness.

        SHADOW MODE: Observational only.

        Returns:
            Stability report dict or None if noise not enabled
        """
        return self._state_generator.get_noise_stability_report()

    def finalize(
        self,
        convergence_pressure_tile: Optional[Dict[str, Any]] = None,
    ) -> FirstLightResult:
        """
        Finalize experiment and create result.

        Args:
            convergence_pressure_tile: Optional convergence pressure tile from
                build_convergence_pressure_tile() (Phase X — observational only).

        Returns:
            FirstLightResult with final metrics
        """
        # Finalize any partial metrics window
        self._metrics_accumulator.finalize_partial()

        # Get final metrics
        delta_p = self._delta_p_computer.compute()
        cumulative = self._metrics_accumulator.get_cumulative_rates()
        trajectories = self._metrics_accumulator.get_trajectories()
        red_flag_summary = self._red_flag_observer.get_summary()
        would_abort, abort_reason = self._red_flag_observer.hypothetical_should_abort()

        # Compute TDA metrics at window boundaries
        self._tda_monitor.reset()
        for window in self._metrics_accumulator.get_all_windows():
            tda_inputs = window.get("tda_inputs", {})
            if tda_inputs:
                self._tda_monitor.compute(
                    H_series=tda_inputs.get("H_trajectory", []),
                    rho_series=tda_inputs.get("rho_trajectory", []),
                    success_series=tda_inputs.get("success_trajectory", []),
                    window_index=window.get("window_index", 0),
                )
        tda_snapshots = [s.to_dict() for s in self._tda_monitor.get_computed_snapshots()]

        # Phase X: Extract convergence pressure summary if available
        # (SHADOW MODE: observational only, no gating)
        convergence_summary = None
        if convergence_pressure_tile:
            convergence_summary = {
                "global_pressure_norm": convergence_pressure_tile.get("global_pressure_norm", 0.0),
                "transition_likelihood_band": convergence_pressure_tile.get("transition_likelihood_band", "LOW"),
                "slices_at_risk": sorted(convergence_pressure_tile.get("slices_at_risk", [])),
                "pressure_drivers": convergence_pressure_tile.get("pressure_drivers", []),
            }

        # Build result
        result = FirstLightResult(
            run_id=self._run_id,
            config_slice=self.config.slice_name,
            config_runner_type=self.config.runner_type,
            start_time=self._start_time or "",
            end_time=self._end_time or datetime.now(timezone.utc).isoformat(),
            total_cycles_requested=self.config.total_cycles,
            cycles_completed=self._cycle,
            # U2 metrics
            u2_success_rate_final=cumulative["success_rate"],
            u2_success_rate_trajectory=trajectories["success_rate"],
            delta_p_success=delta_p.delta_p_success,
            # RFL metrics
            rfl_abstention_rate_final=cumulative["abstention_rate"],
            rfl_abstention_trajectory=trajectories["abstention_rate"],
            delta_p_abstention=delta_p.delta_p_abstention,
            # Stability metrics
            mean_rsi=cumulative["mean_rsi"],
            min_rsi=delta_p.min_rsi,
            max_rsi=delta_p.max_rsi,
            rsi_trajectory=trajectories["mean_rsi"],
            # Safe region metrics
            omega_occupancy=cumulative["omega_occupancy"],
            omega_exit_count=red_flag_summary.observations_by_type.get("OMEGA_EXIT", 0),
            max_omega_exit_streak=red_flag_summary.max_omega_exit_streak,
            omega_occupancy_trajectory=trajectories["omega_occupancy"],
            # HARD mode metrics
            hard_ok_rate=cumulative["hard_ok_rate"],
            hard_fail_count=red_flag_summary.observations_by_type.get("HARD_FAIL", 0),
            max_hard_fail_streak=red_flag_summary.max_hard_fail_streak,
            hard_ok_trajectory=trajectories["hard_ok_rate"],
            # CDI metrics
            cdi_010_activations=red_flag_summary.cdi_010_activations,
            cdi_007_activations=red_flag_summary.observations_by_type.get("CDI-007", 0),
            max_cdi_007_streak=red_flag_summary.max_cdi_007_streak,
            # Divergence metrics
            divergence_count=red_flag_summary.observations_by_type.get("GOVERNANCE_DIVERGENCE", 0),
            max_divergence_streak=red_flag_summary.max_divergence_streak,
            # Red-flag summary
            total_red_flags=red_flag_summary.total_observations,
            red_flags_by_type=red_flag_summary.observations_by_type,
            red_flags_by_severity=red_flag_summary.observations_by_severity,
            # Hypothetical abort (SHADOW MODE: for analysis only)
            hypothetical_abort_cycle=red_flag_summary.hypothetical_abort_cycles[0] if red_flag_summary.hypothetical_abort_cycles else None,
            hypothetical_abort_reason=abort_reason,
            # TDA metrics
            tda_metrics=tda_snapshots,
            # Phase X: Convergence pressure summary (SHADOW MODE: observational only)
            convergence_summary=convergence_summary,
        )

        return result

    def get_observations(self) -> List[CycleObservation]:
        """Get all recorded observations."""
        return list(self._observations)

    def reset(self) -> None:
        """Reset runner for a new run."""
        self._state_generator.reset()
        self._red_flag_observer.reset()
        self._delta_p_computer.reset()
        self._metrics_accumulator.reset()
        self._tda_monitor.reset()
        self._cycle = 0
        self._observations.clear()
        self._start_time = None
        self._end_time = None
        self._run_id = ""

    def get_pathology_summary(self) -> Dict[str, Any]:
        """Return pathology selection and resolved parameters for reporting."""
        return {
            "mode": getattr(self, "_state_generator", None) and self._state_generator._pathology_mode or "none",
            "params": self._state_generator.get_pathology_params() if hasattr(self, "_state_generator") else {},
        }
