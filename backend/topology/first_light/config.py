"""
Phase X P3: First-Light Configuration

This module defines configuration and result dataclasses for the First-Light
shadow experiment. See docs/system_law/Phase_X_P3_Spec.md for full specification.

SHADOW MODE CONTRACT:
- shadow_mode must always be True
- No governance control or abort logic
- All fields are for observational/logging purposes only
- This code runs OFFLINE only, never in production governance paths

Status: P3 IMPLEMENTATION (OFFLINE, SHADOW-ONLY)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = [
    "FirstLightConfig",
    "FirstLightResult",
]


@dataclass
class FirstLightConfig:
    """
    Configuration for First-Light shadow experiment.

    SHADOW MODE CONTRACT:
    - shadow_mode must always be True in Phase X P3
    - All thresholds are for OBSERVATION only, not enforcement
    - No abort logic is triggered based on these values

    See: docs/system_law/Phase_X_P3_Spec.md Section 2.2
    """

    # Slice selection
    slice_name: str = "arithmetic_simple"
    runner_type: str = "u2"  # "u2" or "rfl"

    # Run parameters
    total_cycles: int = 1000
    tau_0: float = 0.20  # Initial threshold (Goldilocks: [0.16, 0.24])

    # Logging configuration
    log_dir: str = "results/first_light"
    run_id: Optional[str] = None
    log_every_n_cycles: int = 1

    # Metrics windows
    success_window: int = 50
    rsi_window: int = 20

    # Red-flag observation thresholds (LOGGING only, NOT enforcement)
    cdi_010_observation_enabled: bool = True
    cdi_007_streak_threshold: int = 10
    rsi_collapse_threshold: float = 0.2
    omega_exit_threshold: int = 100
    block_rate_threshold: float = 0.6
    divergence_streak_threshold: int = 20

    # HARD mode thresholds
    hard_fail_streak_threshold: int = 50

    # SHADOW MODE: Must always be True in P3
    shadow_mode: bool = True

    # Slice Identity (Phase X pre-execution blocker)
    baseline_slice_fingerprint: Optional[str] = None
    identity_verified: bool = False
    curriculum_fingerprint: Optional[str] = None

    def validate(self) -> List[str]:
        """
        Validate configuration.

        Enforces:
        - shadow_mode must be True (SHADOW MODE only)
        - tau_0 must be in [0.0, 1.0]
        - total_cycles must be > 0
        - Window sizes must be > 0

        Returns:
            List of validation errors (empty if valid)
        """
        errors: List[str] = []

        # SHADOW MODE enforcement
        if not self.shadow_mode:
            errors.append(
                "SHADOW MODE VIOLATION: shadow_mode must be True in Phase X P3"
            )

        # tau_0 bounds
        if self.tau_0 < 0.0 or self.tau_0 > 1.0:
            errors.append(
                f"tau_0 must be in [0.0, 1.0], got {self.tau_0}"
            )

        # Cycle count
        if self.total_cycles <= 0:
            errors.append(
                f"total_cycles must be > 0, got {self.total_cycles}"
            )

        # Window sizes
        if self.success_window <= 0:
            errors.append(
                f"success_window must be > 0, got {self.success_window}"
            )

        if self.rsi_window <= 0:
            errors.append(
                f"rsi_window must be > 0, got {self.rsi_window}"
            )

        # Runner type validation
        if self.runner_type not in ("u2", "rfl"):
            errors.append(
                f"runner_type must be 'u2' or 'rfl', got '{self.runner_type}'"
            )

        # Threshold bounds
        if self.rsi_collapse_threshold < 0.0 or self.rsi_collapse_threshold > 1.0:
            errors.append(
                f"rsi_collapse_threshold must be in [0.0, 1.0], got {self.rsi_collapse_threshold}"
            )

        if self.block_rate_threshold < 0.0 or self.block_rate_threshold > 1.0:
            errors.append(
                f"block_rate_threshold must be in [0.0, 1.0], got {self.block_rate_threshold}"
            )

        return errors

    def validate_or_raise(self) -> None:
        """
        Validate configuration and raise ValueError if invalid.

        Raises:
            ValueError: If configuration is invalid
        """
        errors = self.validate()
        if errors:
            raise ValueError(f"Invalid FirstLightConfig: {'; '.join(errors)}")


@dataclass
class FirstLightResult:
    """
    Result of First-Light shadow experiment.

    SHADOW MODE CONTRACT:
    - All metrics are observational only
    - No governance decisions depend on these values
    - Red-flag summary is for logging, not enforcement

    See: docs/system_law/Phase_X_P3_Spec.md Section 2.3
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

    # Success metrics (U2) — observational only
    u2_success_rate_final: Optional[float] = None
    u2_success_rate_trajectory: List[float] = field(default_factory=list)
    delta_p_success: Optional[float] = None

    # Abstention metrics (RFL) — observational only
    rfl_abstention_rate_final: Optional[float] = None
    rfl_abstention_trajectory: List[float] = field(default_factory=list)
    delta_p_abstention: Optional[float] = None

    # Stability metrics — observational only
    mean_rsi: float = 0.0
    min_rsi: float = 1.0
    max_rsi: float = 0.0
    rsi_trajectory: List[float] = field(default_factory=list)

    # Safe region metrics — observational only
    omega_occupancy: float = 0.0
    omega_exit_count: int = 0
    max_omega_exit_streak: int = 0
    omega_occupancy_trajectory: List[float] = field(default_factory=list)

    # HARD mode metrics — observational only
    hard_ok_rate: float = 0.0
    hard_fail_count: int = 0
    max_hard_fail_streak: int = 0
    hard_ok_trajectory: List[float] = field(default_factory=list)

    # CDI metrics — observational only
    cdi_010_activations: int = 0
    cdi_007_activations: int = 0
    max_cdi_007_streak: int = 0

    # Governance divergence — observational only
    divergence_count: int = 0
    max_divergence_streak: int = 0

    # Red-flag observations (LOGGED, not enforced)
    total_red_flags: int = 0
    red_flags_by_type: Dict[str, int] = field(default_factory=dict)
    red_flags_by_severity: Dict[str, int] = field(default_factory=dict)

    # Hypothetical abort analysis (SHADOW MODE: for analysis only)
    hypothetical_abort_cycle: Optional[int] = None
    hypothetical_abort_reason: Optional[str] = None

    # TDA metrics (computed at window boundaries by TDAMonitor)
    tda_metrics: List[Dict[str, Any]] = field(default_factory=list)

    # Convergence pressure summary (Phase X — observational only)
    convergence_summary: Optional[Dict[str, Any]] = None

    # Metric readiness summary (Phase X — observational only)
    metric_readiness_summary: Optional[Dict[str, Any]] = None

    # Telemetry governance summary (Phase X — observational only)
    telemetry_governance_summary: Optional[Dict[str, Any]] = None

    # Evidence quality summary (Phase X — observational only)
    evidence_quality_summary: Optional[Dict[str, Any]] = None

    # Output paths
    cycles_log_path: str = ""
    red_flags_log_path: str = ""
    metrics_log_path: str = ""
    summary_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to JSON-serializable dictionary.

        Structure matches Phase_X_P3_Spec.md Section 5.4 (summary.json schema).

        Returns:
            JSON-serializable dictionary
        """
        return {
            "schema": "first-light-summary/1.0.0",
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
                "u2_success_rate_final": self.u2_success_rate_final,
                "delta_p_success": self.delta_p_success,
                "rfl_abstention_rate_final": self.rfl_abstention_rate_final,
                "delta_p_abstention": self.delta_p_abstention,
            },
            "stability_metrics": {
                "mean_rsi": round(self.mean_rsi, 4),
                "min_rsi": round(self.min_rsi, 4),
                "max_rsi": round(self.max_rsi, 4),
                "rsi_trajectory": [round(r, 4) for r in self.rsi_trajectory],
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
            "divergence_metrics": {
                "divergence_count": self.divergence_count,
                "max_divergence_streak": self.max_divergence_streak,
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
            "trajectories": {
                "success_rate": [round(s, 4) for s in self.u2_success_rate_trajectory],
                "omega_occupancy": [round(o, 4) for o in self.omega_occupancy_trajectory],
                "hard_ok_rate": [round(h, 4) for h in self.hard_ok_trajectory],
            },
            "tda_metrics": self.tda_metrics,
            "convergence_summary": self.convergence_summary,  # Phase X — observational only
            "telemetry_governance_summary": self.telemetry_governance_summary,  # Phase X — observational only
            "metric_readiness_summary": self.metric_readiness_summary,  # Phase X — observational only
            "evidence_quality_summary": self.evidence_quality_summary,  # Phase X — observational only
            "output_files": {
                "cycles_log": self.cycles_log_path,
                "red_flags_log": self.red_flags_log_path,
                "metrics_log": self.metrics_log_path,
            },
        }
