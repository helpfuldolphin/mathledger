"""
Phase X P4: Configuration for Real Runner Shadow Coupling

This module defines configuration and result classes for P4 shadow experiments.
See docs/system_law/Phase_X_P4_Spec.md for full specification.

SHADOW MODE CONTRACT:
- shadow_mode must always be True
- No configuration enables governance modification
- No configuration enables abort enforcement

Status: P4 DESIGN FREEZE (STUBS ONLY)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from backend.topology.first_light.telemetry_adapter import TelemetryProviderInterface

__all__ = [
    "FirstLightConfigP4",
    "FirstLightResultP4",
]


@dataclass
class FirstLightConfigP4:
    """
    Configuration for P4 shadow experiment with real runner coupling.

    SHADOW MODE CONTRACT:
    - shadow_mode must always be True
    - telemetry_adapter must be read-only
    - No configuration enables governance modification

    See: docs/system_law/Phase_X_P4_Spec.md Section 6.4
    """

    # Slice and runner selection
    slice_name: str = "arithmetic_simple"
    runner_type: str = "u2"  # "u2" or "rfl"

    # Run parameters
    total_cycles: int = 1000
    tau_0: float = 0.20

    # Telemetry adapter (P4-specific)
    # Must implement TelemetryProviderInterface (read-only)
    telemetry_adapter: Optional["TelemetryProviderInterface"] = None

    # Logging configuration
    log_dir: str = "results/first_light_p4"
    run_id: Optional[str] = None
    log_every_n_cycles: int = 1

    # Windows
    success_window: int = 50
    rsi_window: int = 20

    # P5 Twin upgrades (SHADOW-ONLY flags)
    use_decoupled_success: bool = False
    twin_lr_overrides: Optional[Dict[str, float]] = None

    # Divergence thresholds (for LOGGING only, NOT enforcement)
    divergence_H_threshold: float = 0.1
    divergence_rho_threshold: float = 0.1
    divergence_tau_threshold: float = 0.05
    divergence_beta_threshold: float = 0.1
    divergence_streak_threshold: int = 20

    # Red-flag thresholds (P3-inherited, LOGGING only)
    cdi_010_observation_enabled: bool = True
    cdi_007_streak_threshold: int = 10
    rsi_collapse_threshold: float = 0.2
    omega_exit_threshold: int = 100
    block_rate_threshold: float = 0.6

    # SHADOW MODE: Must always be True
    shadow_mode: bool = True

    def validate(self) -> List[str]:
        """
        Validate configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors: List[str] = []

        # SHADOW MODE enforcement
        if not self.shadow_mode:
            errors.append(
                "SHADOW MODE VIOLATION: shadow_mode must be True in Phase X P4"
            )

        # tau_0 bounds
        if self.tau_0 < 0.0 or self.tau_0 > 1.0:
            errors.append(f"tau_0 must be in [0.0, 1.0], got {self.tau_0}")

        # Cycle count
        if self.total_cycles <= 0:
            errors.append(f"total_cycles must be > 0, got {self.total_cycles}")

        # Window sizes
        if self.success_window <= 0:
            errors.append(f"success_window must be > 0, got {self.success_window}")

        if self.rsi_window <= 0:
            errors.append(f"rsi_window must be > 0, got {self.rsi_window}")

        # Runner type validation
        if self.runner_type not in ("u2", "rfl"):
            errors.append(
                f"runner_type must be 'u2' or 'rfl', got '{self.runner_type}'"
            )

        if self.twin_lr_overrides:
            for key in self.twin_lr_overrides:
                if key not in {"H", "rho", "tau", "beta"}:
                    errors.append(
                        f"twin_lr_overrides key '{key}' is invalid; allowed keys: H, rho, tau, beta"
                    )
                elif self.twin_lr_overrides[key] < 0 or self.twin_lr_overrides[key] > 1:
                    errors.append(
                        f"twin_lr_overrides[{key}] must be in [0,1], got {self.twin_lr_overrides[key]}"
                    )

        # Divergence thresholds
        if self.divergence_H_threshold < 0.0:
            errors.append(
                f"divergence_H_threshold must be >= 0, got {self.divergence_H_threshold}"
            )
        if self.divergence_rho_threshold < 0.0:
            errors.append(
                f"divergence_rho_threshold must be >= 0, got {self.divergence_rho_threshold}"
            )

        return errors

    def validate_or_raise(self) -> None:
        """
        Validate and raise ValueError on first error.

        Raises:
            ValueError: If configuration is invalid
        """
        errors = self.validate()
        if errors:
            raise ValueError(f"Invalid FirstLightConfigP4: {'; '.join(errors)}")


@dataclass
class FirstLightResultP4:
    """
    Result of P4 shadow experiment with real runner coupling.

    Extends P3 result with divergence analysis and twin accuracy metrics.

    SHADOW MODE: All results are observational only.

    See: docs/system_law/Phase_X_P4_Spec.md Section 6.5
    """

    # Run metadata
    run_id: str = ""
    config_slice: str = ""
    config_runner_type: str = ""
    start_time: str = ""
    end_time: str = ""

    # Cycle summary
    total_cycles_requested: int = 0
    cycles_completed: int = 0

    # P3 metrics (inherited) - U2 success
    u2_success_rate_final: float = 0.0
    u2_success_rate_trajectory: List[float] = field(default_factory=list)
    delta_p_success: Optional[float] = None

    # P3 metrics (inherited) - RFL abstention
    rfl_abstention_rate_final: float = 0.0
    rfl_abstention_trajectory: List[float] = field(default_factory=list)
    delta_p_abstention: Optional[float] = None

    # P3 metrics (inherited) - Stability
    mean_rsi: float = 0.0
    min_rsi: float = 0.0
    max_rsi: float = 0.0
    rsi_trajectory: List[float] = field(default_factory=list)

    # P3 metrics (inherited) - Safe region
    omega_occupancy: float = 0.0
    omega_exit_count: int = 0
    max_omega_exit_streak: int = 0
    omega_occupancy_trajectory: List[float] = field(default_factory=list)

    # P3 metrics (inherited) - HARD mode
    hard_ok_rate: float = 0.0
    hard_fail_count: int = 0
    max_hard_fail_streak: int = 0
    hard_ok_trajectory: List[float] = field(default_factory=list)

    # P3 metrics (inherited) - CDI
    cdi_010_activations: int = 0
    cdi_007_activations: int = 0
    max_cdi_007_streak: int = 0

    # P3 metrics (inherited) - Red-flag summary
    total_red_flags: int = 0
    red_flags_by_type: Dict[str, int] = field(default_factory=dict)
    red_flags_by_severity: Dict[str, int] = field(default_factory=dict)

    hypothetical_abort_cycle: Optional[int] = None
    hypothetical_abort_reason: Optional[str] = None

    # P4-specific: Divergence analysis
    total_divergences: int = 0
    divergences_by_type: Dict[str, int] = field(default_factory=dict)
    divergences_by_severity: Dict[str, int] = field(default_factory=dict)
    max_divergence_streak: int = 0
    divergence_rate: float = 0.0

    # P4-specific: Twin accuracy metrics
    twin_success_prediction_accuracy: float = 0.0
    twin_blocked_prediction_accuracy: float = 0.0
    twin_omega_prediction_accuracy: float = 0.0
    twin_hard_ok_prediction_accuracy: float = 0.0

    # P4-specific: Adversarial calibration (Phase X — observational only)
    adversarial_calibration: Optional[Dict[str, Any]] = None

    # P4-specific: Output paths
    real_cycles_log_path: str = ""
    twin_cycles_log_path: str = ""
    divergence_log_path: str = ""
    red_flags_log_path: str = ""
    metrics_log_path: str = ""
    summary_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "schema": "first-light-p4-result/1.0.0",
            "run_id": self.run_id,
            "mode": "SHADOW",
            "config": {
                "slice_name": self.config_slice,
                "runner_type": self.config_runner_type,
                "total_cycles": self.total_cycles_requested,
            },
            "execution": {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "cycles_completed": self.cycles_completed,
            },
            "uplift_metrics": {
                "u2_success_rate_final": round(self.u2_success_rate_final, 4),
                "delta_p_success": self.delta_p_success,
                "rfl_abstention_rate_final": round(self.rfl_abstention_rate_final, 4),
                "delta_p_abstention": self.delta_p_abstention,
            },
            "stability_metrics": {
                "mean_rsi": round(self.mean_rsi, 4),
                "min_rsi": round(self.min_rsi, 4),
                "max_rsi": round(self.max_rsi, 4),
            },
            "safe_region_metrics": {
                "omega_occupancy": round(self.omega_occupancy, 4),
                "omega_exit_count": self.omega_exit_count,
                "max_omega_exit_streak": self.max_omega_exit_streak,
            },
            "hard_mode_metrics": {
                "hard_ok_rate": round(self.hard_ok_rate, 4),
                "hard_fail_count": self.hard_fail_count,
                "max_hard_fail_streak": self.max_hard_fail_streak,
            },
            "cdi_metrics": {
                "cdi_010_activations": self.cdi_010_activations,
                "cdi_007_activations": self.cdi_007_activations,
                "max_cdi_007_streak": self.max_cdi_007_streak,
            },
            "red_flag_summary": {
                "total_observations": self.total_red_flags,
                "by_type": self.red_flags_by_type,
                "by_severity": self.red_flags_by_severity,
            },
            "hypothetical_abort": {
                "would_have_aborted": self.hypothetical_abort_cycle is not None,
                "abort_cycle": self.hypothetical_abort_cycle,
                "abort_reason": self.hypothetical_abort_reason,
            },
            "divergence_analysis": {
                "total_divergences": self.total_divergences,
                "divergences_by_type": self.divergences_by_type,
                "divergences_by_severity": self.divergences_by_severity,
                "max_divergence_streak": self.max_divergence_streak,
                "divergence_rate": round(self.divergence_rate, 4),
            },
            "twin_accuracy": {
                "success_prediction_accuracy": round(self.twin_success_prediction_accuracy, 4),
                "blocked_prediction_accuracy": round(self.twin_blocked_prediction_accuracy, 4),
                "omega_prediction_accuracy": round(self.twin_omega_prediction_accuracy, 4),
                "hard_ok_prediction_accuracy": round(self.twin_hard_ok_prediction_accuracy, 4),
            },
            "adversarial_calibration": self.adversarial_calibration,
            "output_files": {
                "real_cycles_log": self.real_cycles_log_path,
                "twin_cycles_log": self.twin_cycles_log_path,
                "divergence_log": self.divergence_log_path,
                "red_flags_log": self.red_flags_log_path,
                "metrics_log": self.metrics_log_path,
            },
        }

    def meets_success_criteria(self) -> Dict[str, bool]:
        """
        Check success criteria (for LOGGING only, not control flow).

        Returns:
            Dictionary mapping criterion name to pass/fail
        """
        return {
            "divergence_rate_acceptable": self.divergence_rate < 0.20,
            "twin_success_accuracy_acceptable": self.twin_success_prediction_accuracy > 0.70,
            "twin_blocked_accuracy_acceptable": self.twin_blocked_prediction_accuracy > 0.70,
            "no_severe_divergence_streak": self.max_divergence_streak < 20,
        }
