"""
Phase X P4: JSONL Schema Definitions for Real Runner Shadow Coupling

This module defines the JSONL schema dataclasses for P4 logging.
See docs/system_law/Phase_X_P4_Spec.md for full specification.

SHADOW MODE CONTRACT:
- All schemas include mode="SHADOW" marker
- All schemas include action="LOGGED_ONLY" where applicable
- Schemas are for serialization only, no enforcement logic

Status: P4 DESIGN FREEZE (STUBS ONLY)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = [
    "RealCycleLogEntry",
    "TwinCycleLogEntry",
    "DivergenceLogEntry",
    "P4MetricsLogEntry",
    "P4SummarySchema",
    # Schema version constants
    "REAL_CYCLE_SCHEMA_VERSION",
    "TWIN_CYCLE_SCHEMA_VERSION",
    "DIVERGENCE_SCHEMA_VERSION",
    "P4_METRICS_SCHEMA_VERSION",
    "P4_SUMMARY_SCHEMA_VERSION",
]

# Schema version constants
REAL_CYCLE_SCHEMA_VERSION = "first-light-p4-real-cycle/1.0.0"
TWIN_CYCLE_SCHEMA_VERSION = "first-light-p4-twin-cycle/1.0.0"
DIVERGENCE_SCHEMA_VERSION = "first-light-p4-divergence/1.0.0"
P4_METRICS_SCHEMA_VERSION = "first-light-p4-metrics/1.0.0"
P4_SUMMARY_SCHEMA_VERSION = "first-light-p4-summary/1.0.0"


@dataclass
class RealCycleLogEntry:
    """
    JSONL log entry for real runner cycle observation.

    SHADOW MODE: This is an observation log. mode="SHADOW" is always set.

    See: docs/system_law/Phase_X_P4_Spec.md Section 7.1
    """

    # Schema identification
    schema: str = REAL_CYCLE_SCHEMA_VERSION
    source: str = "REAL_RUNNER"
    mode: str = "SHADOW"  # Always "SHADOW" in P4

    # Cycle identification
    cycle: int = 0
    timestamp: str = ""

    # Runner outcome
    runner_type: str = ""
    slice_name: str = ""
    success: bool = False
    depth: Optional[int] = None
    proof_hash: Optional[str] = None

    # USLA state
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
        raise NotImplementedError("P4 implementation not yet activated")

    def to_json_line(self) -> str:
        """Convert to JSON string for JSONL file."""
        raise NotImplementedError("P4 implementation not yet activated")


@dataclass
class TwinCycleLogEntry:
    """
    JSONL log entry for shadow twin cycle prediction.

    SHADOW MODE: This is a prediction log. mode="SHADOW" is always set.

    See: docs/system_law/Phase_X_P4_Spec.md Section 7.2
    """

    # Schema identification
    schema: str = TWIN_CYCLE_SCHEMA_VERSION
    source: str = "SHADOW_TWIN"
    mode: str = "SHADOW"  # Always "SHADOW" in P4

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

    # Confidence
    prediction_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSONL serialization."""
        raise NotImplementedError("P4 implementation not yet activated")

    def to_json_line(self) -> str:
        """Convert to JSON string for JSONL file."""
        raise NotImplementedError("P4 implementation not yet activated")


@dataclass
class DivergenceLogEntry:
    """
    JSONL log entry for divergence analysis.

    SHADOW MODE: This is a divergence observation. action="LOGGED_ONLY" always.

    See: docs/system_law/Phase_X_P4_Spec.md Section 7.3
    """

    # Schema identification
    schema: str = DIVERGENCE_SCHEMA_VERSION
    mode: str = "SHADOW"  # Always "SHADOW" in P4
    action: str = "LOGGED_ONLY"  # Always "LOGGED_ONLY" in P4

    # Cycle identification
    cycle: int = 0
    timestamp: str = ""

    # Divergence flags
    success_diverged: bool = False
    blocked_diverged: bool = False
    omega_diverged: bool = False
    hard_ok_diverged: bool = False

    # Magnitude
    H_delta: float = 0.0
    rho_delta: float = 0.0
    tau_delta: float = 0.0
    beta_delta: float = 0.0

    # Classification
    severity: str = "NONE"  # NONE, MINOR, MODERATE, SEVERE
    divergence_type: str = "NONE"  # NONE, STATE, OUTCOME, BOTH
    consecutive_count: int = 0

    # Chronicle drift markers (Phase X)
    chronicle_drift: Optional[Dict[str, Any]] = None  # From extract_chronicle_drift_signal()

    # Drift governance signal (Phase X)
    drift_governance_signal: Optional[Dict[str, Any]] = None  # From extract_drift_signal_for_shadow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSONL serialization."""
        raise NotImplementedError("P4 implementation not yet activated")

    def to_json_line(self) -> str:
        """Convert to JSON string for JSONL file."""
        raise NotImplementedError("P4 implementation not yet activated")


@dataclass
class P4MetricsLogEntry:
    """
    JSONL log entry for P4 windowed metrics.

    SHADOW MODE: This is a metrics observation. mode="SHADOW" always.

    Extends P3 MetricsLogEntry with divergence and twin accuracy metrics.

    See: docs/system_law/Phase_X_P4_Spec.md Section 7
    """

    # Schema identification
    schema: str = P4_METRICS_SCHEMA_VERSION
    mode: str = "SHADOW"  # Always "SHADOW" in P4

    # Window identification
    window_index: int = 0
    window_start_cycle: int = 0
    window_end_cycle: int = 0
    timestamp: str = ""

    # P3 metrics (inherited)
    window_success_rate: float = 0.0
    cumulative_success_rate: float = 0.0
    delta_p_success: Optional[float] = None

    window_abstention_rate: float = 0.0
    cumulative_abstention_rate: float = 0.0
    delta_p_abstention: Optional[float] = None

    window_mean_rsi: float = 0.0
    cumulative_mean_rsi: float = 0.0

    window_omega_occupancy: float = 0.0
    cumulative_omega_occupancy: float = 0.0

    window_hard_ok_rate: float = 0.0
    cumulative_hard_ok_rate: float = 0.0

    # P4 metrics (divergence)
    window_divergence_count: int = 0
    window_divergence_rate: float = 0.0
    cumulative_divergence_rate: float = 0.0

    # P4 metrics (twin accuracy)
    window_twin_success_accuracy: float = 0.0
    window_twin_blocked_accuracy: float = 0.0
    window_twin_omega_accuracy: float = 0.0

    # Red-flag counts (window)
    red_flag_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSONL serialization."""
        raise NotImplementedError("P4 implementation not yet activated")

    def to_json_line(self) -> str:
        """Convert to JSON string for JSONL file."""
        raise NotImplementedError("P4 implementation not yet activated")


@dataclass
class P4SummarySchema:
    """
    JSON schema for P4 experiment summary.

    SHADOW MODE: This is a run summary. mode="SHADOW" always.

    Extends P3 summary with divergence analysis and twin accuracy.

    See: docs/system_law/Phase_X_P4_Spec.md Section 7.4
    """

    # Schema identification
    schema: str = P4_SUMMARY_SCHEMA_VERSION
    mode: str = "SHADOW"  # Always "SHADOW" in P4

    # Run identification
    run_id: str = ""
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0

    # Configuration
    slice_name: str = ""
    runner_type: str = ""
    total_cycles_requested: int = 0
    cycles_completed: int = 0
    tau_0: float = 0.20

    # P3 success criteria (inherited)
    u2_success_rate_target: float = 0.75
    u2_success_rate_actual: float = 0.0
    u2_success_rate_passed: bool = False

    delta_p_target: float = 0.0
    delta_p_actual: Optional[float] = None
    delta_p_passed: bool = False

    mean_rsi_target: float = 0.60
    mean_rsi_actual: float = 0.0
    mean_rsi_passed: bool = False

    omega_occupancy_target: float = 0.90
    omega_occupancy_actual: float = 0.0
    omega_occupancy_passed: bool = False

    cdi_010_target: int = 0
    cdi_010_actual: int = 0
    cdi_010_passed: bool = False

    cdi_007_target: int = 50
    cdi_007_actual: int = 0
    cdi_007_passed: bool = False

    hard_ok_target: float = 0.80
    hard_ok_actual: float = 0.0
    hard_ok_passed: bool = False

    # P4 divergence analysis
    total_divergences: int = 0
    divergences_by_type: Dict[str, int] = field(default_factory=dict)
    divergences_by_severity: Dict[str, int] = field(default_factory=dict)
    max_divergence_streak: int = 0
    divergence_rate: float = 0.0

    # P4 twin accuracy
    twin_success_accuracy: float = 0.0
    twin_blocked_accuracy: float = 0.0
    twin_omega_accuracy: float = 0.0
    twin_hard_ok_accuracy: float = 0.0

    # Red-flag summary
    total_red_flags: int = 0
    hypothetical_aborts: int = 0

    # Consensus governance (Phase X — observational only)
    consensus_governance: Optional[Dict[str, Any]] = None

    # Output paths
    real_cycles_path: str = ""
    twin_cycles_path: str = ""
    divergence_path: str = ""
    red_flags_path: str = ""
    metrics_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        raise NotImplementedError("P4 implementation not yet activated")

    def to_json(self, indent: int = 2) -> str:
        """Convert to formatted JSON string."""
        raise NotImplementedError("P4 implementation not yet activated")
