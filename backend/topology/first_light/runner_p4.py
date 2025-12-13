"""
Phase X P4: Shadow Runner with Real Runner Coupling

This module implements the FirstLightShadowRunnerP4 class for executing
shadow experiments coupled to real runner telemetry.
See docs/system_law/Phase_X_P4_Spec.md for full specification.

SHADOW MODE CONTRACT:
- NEVER modifies governance decisions
- NEVER enforces abort conditions
- All outputs are observational only
- Red-flags are LOGGED, not ACTED upon
- Read-only coupling to real runners via adapter

Status: P4 IMPLEMENTATION

TODO[PhaseX-TDA-P4]: Integrate TDA metrics for real vs twin comparison.
    - Compute SNS, PCS, DRS, HSS for both real and twin trajectories
    - Emit comparison to p4_tda_metrics.json
    - Plot: overlay P3 synthetic envelope with P4 real trajectory
    - See: docs/system_law/Phase_X_Prelaunch_Review.md Section 3.3
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from backend.topology.first_light.config_p4 import FirstLightConfigP4, FirstLightResultP4
from backend.topology.first_light.data_structures_p4 import (
    DivergenceSnapshot,
    RealCycleObservation,
    TelemetrySnapshot,
    TwinCycleObservation,
)
from backend.topology.first_light.divergence_analyzer import DivergenceAnalyzer
from backend.topology.first_light.telemetry_adapter import TelemetryProviderInterface
from backend.topology.first_light.p5_pattern_classifier import (
    TDAPatternClassifier,
    DivergencePattern,
    PatternClassification,
    P5_PATTERN_SCHEMA_VERSION,
)

__all__ = [
    "FirstLightShadowRunnerP4",
    "TwinRunner",
]


class TwinRunner:
    """
    Shadow twin that generates predictions without influencing real execution.

    SHADOW MODE CONTRACT:
    - Twin predictions are computed locally
    - No external calls that could affect real runners
    - All predictions are for comparison only

    See: docs/system_law/Phase_X_P4_Spec.md Section 4.1
    """

    def __init__(
        self,
        tau_0: float = 0.20,
        seed: Optional[int] = None,
        learning_rate: float = 0.1,
        noise_scale: float = 0.02,
        *,
        use_decoupled_success: bool = False,
        lr_overrides: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize twin runner.

        Args:
            tau_0: Initial threshold for twin model
            seed: Random seed for reproducibility
            learning_rate: Rate at which twin learns from real observations
            noise_scale: Scale of noise in state dynamics
        """
        self._tau_0 = tau_0
        self._seed = seed
        self._learning_rate = learning_rate
        self._noise_scale = noise_scale
        self._rng = random.Random(seed)
        self._use_decoupled_success = use_decoupled_success
        self._lr_overrides = lr_overrides or {}

        # Internal state
        self._H = 0.5
        self._rho = 0.7
        self._tau = tau_0
        self._beta = 0.1
        self._cycle = 0

        # Prediction history
        self._predictions: List[TwinCycleObservation] = []

    def initialize_from_snapshot(self, snapshot: TelemetrySnapshot) -> None:
        """
        Initialize twin state from a telemetry snapshot.

        Args:
            snapshot: Initial state from real runner
        """
        self._H = snapshot.H
        self._rho = snapshot.rho
        self._tau = snapshot.tau
        self._beta = snapshot.beta
        self._cycle = snapshot.cycle

    def predict(
        self,
        real_observation: RealCycleObservation,
    ) -> TwinCycleObservation:
        """
        Generate twin prediction for a cycle.

        SHADOW MODE: This prediction is computed locally and does NOT
        influence the real runner in any way.

        Args:
            real_observation: The real observation to predict against

        Returns:
            TwinCycleObservation with twin's predictions
        """
        # Predict based on current twin state
        predicted_in_omega = self._H > self._tau and self._rho > 0.5
        predicted_blocked = self._beta > 0.7

        if self._use_decoupled_success:
            # Use real observation state directly to estimate success probability
            base_success_prob = 0.4 + 0.35 * real_observation.H + 0.25 * real_observation.rho - 0.25 * real_observation.beta
        else:
            # Success prediction based on learned state (matches RealTelemetryAdapter logic)
            base_success_prob = 0.5 + 0.3 * self._H + 0.2 * self._rho - 0.3 * self._beta

        base_success_prob = max(0.1, min(0.95, base_success_prob))
        predicted_success = base_success_prob > 0.5

        # Hard OK prediction: predict True when probability > 0.5
        hard_ok_prob = 0.9 + 0.1 * self._rho
        predicted_hard_ok = hard_ok_prob > 0.5

        # Confidence based on state stability
        confidence = min(1.0, self._rho * (1.0 - abs(self._H - real_observation.H)))

        timestamp = datetime.now(timezone.utc).isoformat()

        prediction = TwinCycleObservation(
            source="SHADOW_TWIN",
            mode="SHADOW",
            real_cycle=real_observation.cycle,
            timestamp=timestamp,
            predicted_success=predicted_success,
            predicted_blocked=predicted_blocked,
            predicted_in_omega=predicted_in_omega,
            predicted_hard_ok=predicted_hard_ok,
            twin_H=self._H,
            twin_rho=self._rho,
            twin_tau=self._tau,
            twin_beta=self._beta,
            prediction_confidence=confidence,
        )

        self._predictions.append(prediction)
        return prediction

    def update_state(
        self,
        real_observation: RealCycleObservation,
    ) -> None:
        """
        Update twin internal state based on real observation.

        SHADOW MODE: This updates the twin's model to track real behavior,
        but does NOT send any feedback to the real runner.

        Args:
            real_observation: Real observation to learn from
        """
        self._cycle = real_observation.cycle

        # Blend twin state toward real state using learning rate
        lr = self._learning_rate
        noise = self._noise_scale

        lr_H = self._lr_overrides.get("H", lr)
        lr_rho = self._lr_overrides.get("rho", lr)
        lr_tau = self._lr_overrides.get("tau", lr * 0.5)
        lr_beta = self._lr_overrides.get("beta", lr)

        self._H = self._H * (1 - lr_H) + real_observation.H * lr_H
        self._H += self._rng.gauss(0, noise)
        self._H = max(0.0, min(1.0, self._H))

        self._rho = self._rho * (1 - lr_rho) + real_observation.rho * lr_rho
        self._rho += self._rng.gauss(0, noise)
        self._rho = max(0.0, min(1.0, self._rho))

        self._tau = self._tau * (1 - lr_tau) + real_observation.tau * lr_tau
        self._tau += self._rng.gauss(0, noise * 0.5)
        self._tau = max(0.0, min(1.0, self._tau))

        # Beta learns from blocked status
        if real_observation.real_blocked:
            self._beta = min(1.0, self._beta + 0.05 * lr_beta / max(lr, 1e-6))
        else:
            self._beta = max(0.0, self._beta - 0.01 * lr_beta / max(lr, 1e-6) + self._rng.gauss(0, noise))

    def get_current_state(self) -> Dict[str, float]:
        """
        Get current twin state.

        Returns:
            Dictionary with H, rho, tau, beta values
        """
        return {
            "H": self._H,
            "rho": self._rho,
            "tau": self._tau,
            "beta": self._beta,
            "cycle": self._cycle,
        }

    def get_predictions(self) -> List[TwinCycleObservation]:
        """Get all predictions made by twin."""
        return list(self._predictions)

    def reset(self) -> None:
        """Reset twin to initial state."""
        self._H = 0.5
        self._rho = 0.7
        self._tau = self._tau_0
        self._beta = 0.1
        self._cycle = 0
        self._rng = random.Random(self._seed)
        self._predictions.clear()


class FirstLightShadowRunnerP4:
    """
    First-Light shadow experiment runner with real telemetry coupling.

    SHADOW MODE CONTRACT:
    - NEVER modifies governance decisions
    - NEVER enforces abort conditions
    - All outputs are observational only
    - Red-flags are LOGGED, not ACTED upon
    - Reads from real runner via read-only adapter

    INVARIANT ENFORCEMENT:
    - INV-01: Shadow mode enforced at initialization
    - INV-02: Read-only coupling enforced by adapter type
    - INV-07: Observer effect avoided by read-only design

    See: docs/system_law/Phase_X_P4_Spec.md Section 4
    """

    def __init__(
        self,
        config: FirstLightConfigP4,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize P4 runner with validated config.

        SHADOW MODE: Enforces shadow_mode=True and read-only adapter.

        Args:
            config: FirstLightConfigP4 instance
            seed: Optional random seed for twin reproducibility

        Raises:
            ValueError: If shadow_mode is False
            TypeError: If adapter doesn't implement TelemetryProviderInterface
        """
        self._validate_config(config)
        self._validate_adapter(config.telemetry_adapter)

        self._config = config
        self._seed = seed
        self._adapter = config.telemetry_adapter

        # Initialize components
        self._twin = TwinRunner(
            tau_0=config.tau_0,
            seed=seed,
            use_decoupled_success=config.use_decoupled_success,
            lr_overrides=config.twin_lr_overrides,
        )
        self._divergence_analyzer = DivergenceAnalyzer()

        # P5 Pattern Classifier (SHADOW MODE - observational only)
        self._pattern_classifier = TDAPatternClassifier(window_size=50)
        self._pattern_classifications: List[PatternClassification] = []

        # Storage
        self._real_observations: List[RealCycleObservation] = []
        self._twin_observations: List[TwinCycleObservation] = []
        self._divergence_snapshots: List[DivergenceSnapshot] = []

        # Metrics tracking
        self._cycles_completed = 0
        self._start_time: Optional[str] = None
        self._end_time: Optional[str] = None

        # Red-flag tracking (observation only)
        self._red_flag_count = 0
        self._red_flags_by_type: Dict[str, int] = {}
        self._red_flags_by_severity: Dict[str, int] = {}

    def run(self) -> FirstLightResultP4:
        """
        Execute P4 shadow experiment.

        SHADOW MODE: This runs the experiment in observation-only mode.
        No governance decisions are modified. All outputs are logged
        but never influence actual system behavior.

        Returns:
            FirstLightResultP4 with all metrics, observations, and divergence analysis
        """
        self._start_time = datetime.now(timezone.utc).isoformat()

        # Run all cycles
        for _ in self.run_cycles(self._config.total_cycles):
            pass  # Observations recorded internally

        self._end_time = datetime.now(timezone.utc).isoformat()

        return self.finalize()

    def run_cycles(self, n: int) -> Iterator[RealCycleObservation]:
        """
        Run N cycles and yield real observations.

        SHADOW MODE: Each cycle is observed but not controlled.
        Twin predictions and divergence analysis happen internally.

        Args:
            n: Number of cycles to observe

        Yields:
            RealCycleObservation for each cycle
        """
        for _ in range(n):
            observation = self.observe_single_cycle()
            if observation is not None:
                yield observation
            else:
                # Telemetry unavailable, stop
                break

    def observe_single_cycle(self) -> Optional[RealCycleObservation]:
        """
        Observe a single cycle from the real runner.

        SHADOW MODE: This is observation only.

        Returns:
            RealCycleObservation, or None if telemetry unavailable
        """
        if self._adapter is None or not self._adapter.is_available():
            return None

        # Get telemetry snapshot (READ-ONLY)
        snapshot = self._adapter.get_snapshot()
        if snapshot is None:
            return None

        # Convert to real observation
        real_obs = RealCycleObservation.from_snapshot(snapshot)
        self._real_observations.append(real_obs)

        # Generate twin prediction
        twin_pred = self._twin.predict(real_obs)
        self._twin_observations.append(twin_pred)

        # Analyze divergence
        divergence = self._divergence_analyzer.analyze(real_obs, twin_pred)
        self._divergence_snapshots.append(divergence)

        # P5 Pattern Classification (SHADOW MODE - observational only)
        delta_p = twin_pred.twin_H - real_obs.H if hasattr(twin_pred, 'twin_H') else 0.0
        classification = self._pattern_classifier.classify(
            delta_p=delta_p,
            p_real=real_obs.H,
            p_twin=getattr(twin_pred, 'twin_H', real_obs.H),
            omega_real=real_obs.in_omega,
            omega_twin=twin_pred.predicted_in_omega,
            is_excursion=not real_obs.in_omega,
        )
        self._pattern_classifications.append(classification)

        # Update twin state based on real observation
        self._twin.update_state(real_obs)

        # Check for red-flags (observation only)
        self._check_red_flags(real_obs, divergence)

        self._cycles_completed += 1

        return real_obs

    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics snapshot.

        SHADOW MODE: Metrics are observational only.

        Returns:
            Dictionary with current metrics
        """
        summary = self._divergence_analyzer.get_summary()

        # Compute success rate from observations
        if self._real_observations:
            success_count = sum(1 for obs in self._real_observations if obs.success)
            success_rate = success_count / len(self._real_observations)
        else:
            success_rate = 0.0

        return {
            "cycles_completed": self._cycles_completed,
            "success_rate": round(success_rate, 4),
            "divergence_rate": round(summary.divergence_rate, 4),
            "twin_accuracy": {
                "success": round(summary.success_accuracy, 4),
                "blocked": round(summary.blocked_accuracy, 4),
                "omega": round(summary.omega_accuracy, 4),
                "hard_ok": round(summary.hard_ok_accuracy, 4),
            },
            "red_flag_count": self._red_flag_count,
        }

    def get_red_flag_status(self) -> Dict[str, Any]:
        """
        Get current red-flag observation status.

        SHADOW MODE: This is observational only. The status does NOT
        trigger any control flow changes.

        Returns:
            Dictionary with red-flag status
        """
        return {
            "mode": "SHADOW",
            "total_flags": self._red_flag_count,
            "by_type": dict(self._red_flags_by_type),
            "by_severity": dict(self._red_flags_by_severity),
            "action": "LOGGED_ONLY",
        }

    def get_divergence_status(self) -> Dict[str, Any]:
        """
        Get current divergence analysis status.

        SHADOW MODE: This is observational only. Divergences do NOT
        trigger any remediation.

        Returns:
            Dictionary with divergence status
        """
        summary = self._divergence_analyzer.get_summary()
        would_alert, alert_reason = self._divergence_analyzer.hypothetical_should_alert()

        return {
            "mode": "SHADOW",
            "summary": summary.to_dict(),
            "current_streak": self._divergence_analyzer.get_current_streak(),
            "hypothetical_alert": {
                "would_trigger": would_alert,
                "reason": alert_reason,
            },
            "action": "LOGGED_ONLY",
        }

    def get_twin_status(self) -> Dict[str, Any]:
        """
        Get current twin runner status.

        Returns:
            Dictionary with twin state and accuracy metrics
        """
        summary = self._divergence_analyzer.get_summary()
        twin_state = self._twin.get_current_state()

        return {
            "twin_state": twin_state,
            "predictions_made": len(self._twin_observations),
            "accuracy": {
                "success": round(summary.success_accuracy, 4),
                "blocked": round(summary.blocked_accuracy, 4),
                "omega": round(summary.omega_accuracy, 4),
                "hard_ok": round(summary.hard_ok_accuracy, 4),
            },
        }

    def finalize(
        self,
        convergence_pressure_tile: Optional[Dict[str, Any]] = None,
        convergence_early_warning: Optional[Dict[str, Any]] = None,
    ) -> FirstLightResultP4:
        """
        Finalize experiment and create result.

        Args:
            convergence_pressure_tile: Optional convergence pressure tile from
                build_convergence_pressure_tile() (Phase X — observational only).
            convergence_early_warning: Optional early-warning radar from
                build_phase_transition_early_warning_radar() (Phase X — observational only).

        Returns:
            FirstLightResultP4 with final metrics including convergence_calibration
        """
        if self._end_time is None:
            self._end_time = datetime.now(timezone.utc).isoformat()

        summary = self._divergence_analyzer.get_summary()

        # Compute metrics from observations
        success_rate = 0.0
        mean_rsi = 0.0
        omega_occupancy = 0.0
        hard_ok_rate = 0.0

        if self._real_observations:
            n = len(self._real_observations)
            success_rate = sum(1 for o in self._real_observations if o.success) / n
            mean_rsi = sum(o.rho for o in self._real_observations) / n
            omega_occupancy = sum(1 for o in self._real_observations if o.in_omega) / n
            hard_ok_rate = sum(1 for o in self._real_observations if o.hard_ok) / n

        # Build convergence calibration if provided
        adversarial_calibration = None
        if convergence_pressure_tile or convergence_early_warning:
            adversarial_calibration = {
                "global_pressure_norm": (
                    convergence_pressure_tile.get("global_pressure_norm", 0.0)
                    if convergence_pressure_tile else 0.0
                ),
                "transition_likelihood_band": (
                    convergence_early_warning.get("transition_likelihood_band", "LOW")
                    if convergence_early_warning else "LOW"
                ),
            }

        return FirstLightResultP4(
            run_id=self._config.run_id or "",
            config_slice=self._config.slice_name,
            config_runner_type=self._config.runner_type,
            start_time=self._start_time or "",
            end_time=self._end_time,
            total_cycles_requested=self._config.total_cycles,
            cycles_completed=self._cycles_completed,
            u2_success_rate_final=success_rate,
            mean_rsi=mean_rsi,
            omega_occupancy=omega_occupancy,
            hard_ok_rate=hard_ok_rate,
            total_red_flags=self._red_flag_count,
            red_flags_by_type=dict(self._red_flags_by_type),
            red_flags_by_severity=dict(self._red_flags_by_severity),
            total_divergences=summary.total_divergences,
            divergences_by_type={
                "state": summary.state_divergences,
                "outcome": summary.outcome_divergences,
                "combined": summary.combined_divergences,
            },
            divergences_by_severity={
                "minor": summary.minor_divergences,
                "moderate": summary.moderate_divergences,
                "severe": summary.severe_divergences,
            },
            max_divergence_streak=summary.max_divergence_streak,
            divergence_rate=summary.divergence_rate,
            twin_success_prediction_accuracy=summary.success_accuracy,
            twin_blocked_prediction_accuracy=summary.blocked_accuracy,
            twin_omega_prediction_accuracy=summary.omega_accuracy,
            twin_hard_ok_prediction_accuracy=summary.hard_ok_accuracy,
            adversarial_calibration=adversarial_calibration,
        )

    def get_observations(self) -> List[RealCycleObservation]:
        """Get all recorded real observations."""
        return list(self._real_observations)

    def get_twin_observations(self) -> List[TwinCycleObservation]:
        """Get all recorded twin predictions."""
        return list(self._twin_observations)

    def get_divergence_snapshots(self) -> List[DivergenceSnapshot]:
        """Get all recorded divergence snapshots."""
        return list(self._divergence_snapshots)

    def get_pattern_classifications(self) -> List[PatternClassification]:
        """Get all recorded pattern classifications."""
        return list(self._pattern_classifications)

    def get_p5_pattern_status(self) -> Dict[str, Any]:
        """
        Get current P5 pattern classification status.

        SHADOW MODE: This is observational only. Pattern status does NOT
        influence any governance decisions.

        Returns:
            Dictionary with pattern classification summary
        """
        current_pattern = self._pattern_classifier.get_current_pattern()
        streak = self._pattern_classifier.get_pattern_streak()

        # Compute pattern distribution
        pattern_counts: Dict[str, int] = {}
        for classification in self._pattern_classifications:
            pattern_name = classification.pattern.value
            pattern_counts[pattern_name] = pattern_counts.get(pattern_name, 0) + 1

        return {
            "mode": "SHADOW",
            "schema_version": P5_PATTERN_SCHEMA_VERSION,
            "current_pattern": current_pattern.value,
            "pattern_streak": streak,
            "total_classifications": len(self._pattern_classifications),
            "pattern_distribution": pattern_counts,
            "recalibration_triggered": self._pattern_classifier._should_trigger_recalibration(),
            "action": "LOGGED_ONLY",
        }

    def build_p5_pattern_tags(self) -> Dict[str, Any]:
        """
        Build P5 pattern tags artifact for evidence pack integration.

        SHADOW MODE CONTRACT:
        - Tags are for ANALYSIS only
        - Does NOT modify governance decisions
        - All outputs include mode="SHADOW"

        Returns:
            Dictionary suitable for writing to p5_pattern_tags.json
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        current_pattern = self._pattern_classifier.get_current_pattern()
        streak = self._pattern_classifier.get_pattern_streak()

        # Get signal extensions for GGFL integration
        p5_telemetry = self._pattern_classifier.get_p5_telemetry_extension(
            validation_status="VALIDATED_REAL" if self._adapter else "SYNTHETIC",
            validation_confidence=0.9,
        )
        p5_topology = self._pattern_classifier.get_p5_topology_extension()
        p5_replay = self._pattern_classifier.get_p5_replay_extension()

        # Compute pattern window summaries (10-cycle windows)
        window_summaries = []
        window_size = 10
        for i in range(0, len(self._pattern_classifications), window_size):
            window = self._pattern_classifications[i:i + window_size]
            if window:
                dominant = max(
                    set(c.pattern for c in window),
                    key=lambda p: sum(1 for c in window if c.pattern == p),
                )
                mean_confidence = sum(c.confidence for c in window) / len(window)
                window_summaries.append({
                    "window_start": i,
                    "window_end": min(i + window_size, len(self._pattern_classifications)),
                    "dominant_pattern": dominant.value,
                    "mean_confidence": round(mean_confidence, 4),
                })

        return {
            "schema_version": P5_PATTERN_SCHEMA_VERSION,
            "mode": "SHADOW",
            "timestamp": timestamp,
            "run_id": self._config.run_id or "",
            "cycles_analyzed": len(self._pattern_classifications),
            "classification_summary": {
                "final_pattern": current_pattern.value,
                "final_streak": streak,
                "recalibration_triggered": self._pattern_classifier._should_trigger_recalibration(),
            },
            "signal_extensions": {
                "p5_telemetry": p5_telemetry.to_dict(),
                "p5_topology": p5_topology.to_dict(),
                "p5_replay": p5_replay.to_dict(),
            },
            "window_summaries": window_summaries,
            "shadow_mode_invariants": {
                "no_enforcement": True,
                "logged_only": True,
                "observation_only": True,
            },
        }

    def write_p5_pattern_tags(self, output_dir: Path) -> Optional[Path]:
        """
        Write P5 pattern tags to output directory.

        SHADOW MODE: Pattern tags are written for analysis only.

        Args:
            output_dir: Directory to write p5_pattern_tags.json

        Returns:
            Path to written file, or None if no classifications
        """
        if not self._pattern_classifications:
            return None

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        tags = self.build_p5_pattern_tags()
        output_path = output_dir / "p5_pattern_tags.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(tags, f, indent=2)

        return output_path

    def reset(self) -> None:
        """Reset runner for a new run."""
        self._twin.reset()
        self._divergence_analyzer.reset()
        self._pattern_classifier.reset()
        self._real_observations.clear()
        self._twin_observations.clear()
        self._divergence_snapshots.clear()
        self._pattern_classifications.clear()
        self._cycles_completed = 0
        self._start_time = None
        self._end_time = None
        self._red_flag_count = 0
        self._red_flags_by_type.clear()
        self._red_flags_by_severity.clear()

    def _validate_config(self, config: FirstLightConfigP4) -> None:
        """
        Validate P4 configuration.

        Raises:
            ValueError: If configuration violates P4 invariants
        """
        config.validate_or_raise()

    def _validate_adapter(
        self,
        adapter: Optional[TelemetryProviderInterface],
    ) -> None:
        """
        Validate telemetry adapter is read-only.

        Raises:
            TypeError: If adapter doesn't implement TelemetryProviderInterface
        """
        if adapter is not None and not isinstance(adapter, TelemetryProviderInterface):
            raise TypeError(
                f"telemetry_adapter must implement TelemetryProviderInterface, "
                f"got {type(adapter).__name__}"
            )

    def _check_red_flags(
        self,
        real_obs: RealCycleObservation,
        divergence: DivergenceSnapshot,
    ) -> None:
        """
        Check for red-flag conditions (observation only).

        SHADOW MODE: Flags are logged but never trigger control flow.

        Args:
            real_obs: Real observation
            divergence: Divergence analysis
        """
        # Check for HARD mode failure
        if not real_obs.hard_ok:
            self._record_red_flag("HARD_FAIL", "WARN")

        # Check for blocked state
        if real_obs.real_blocked:
            self._record_red_flag("BLOCKED", "INFO")

        # Check for severe divergence
        if divergence.divergence_severity == "CRITICAL":
            self._record_red_flag("DIVERGENCE_CRITICAL", "CRITICAL")
        elif divergence.divergence_severity == "WARN":
            self._record_red_flag("DIVERGENCE_WARN", "WARN")

        # Check for omega exit
        if not real_obs.in_omega:
            self._record_red_flag("OMEGA_EXIT", "INFO")

    def _record_red_flag(self, flag_type: str, severity: str) -> None:
        """Record a red-flag observation."""
        self._red_flag_count += 1
        self._red_flags_by_type[flag_type] = (
            self._red_flags_by_type.get(flag_type, 0) + 1
        )
        self._red_flags_by_severity[severity] = (
            self._red_flags_by_severity.get(severity, 0) + 1
        )
