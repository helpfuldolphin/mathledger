"""
Phase X P3: JSONL Log Schemas — Design Stubs

This module defines the JSONL schema structures for First-Light experiment logs.
See docs/system_law/Phase_X_P3_Spec.md for full specification.

IMPORTANT: This is a Phase X P3 **design stub**, not a complete implementation.

SHADOW MODE CONTRACT:
- All schemas include "mode": "SHADOW" field
- Red-flag schemas include "action": "LOGGED_ONLY"
- No parsing or validation logic is implemented yet

Status: DESIGN FREEZE / STUBS ONLY

JSONL Schema Definitions:
========================

1. cycles.jsonl (CycleLogEntry)
   - schema: "first-light-cycle/1.0.0"
   - Contains: cycle number, runner output, USLA state, governance alignment

2. red_flags.jsonl (RedFlagLogEntry)
   - schema: "first-light-red-flag/1.0.0"
   - Contains: flag type, severity, observation details
   - MUST include: "action": "LOGGED_ONLY"

3. metrics.jsonl (MetricsLogEntry)
   - schema: "first-light-metrics/1.0.0"
   - Contains: window metrics, Δp values, red-flag counts

4. summary.json (SummarySchema)
   - schema: "first-light-summary/1.0.0"
   - Contains: run config, execution stats, success criteria results
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = [
    "CYCLE_LOG_SCHEMA_VERSION",
    "RED_FLAG_LOG_SCHEMA_VERSION",
    "METRICS_LOG_SCHEMA_VERSION",
    "SUMMARY_SCHEMA_VERSION",
    "CycleLogEntry",
    "RedFlagLogEntry",
    "MetricsLogEntry",
    "SummarySchema",
]


# Schema versions
CYCLE_LOG_SCHEMA_VERSION = "first-light-cycle/1.0.0"
RED_FLAG_LOG_SCHEMA_VERSION = "first-light-red-flag/1.0.0"
METRICS_LOG_SCHEMA_VERSION = "first-light-metrics/1.0.0"
SUMMARY_SCHEMA_VERSION = "first-light-summary/1.0.0"


@dataclass
class CycleLogEntry:
    """
    Schema for cycles.jsonl entries.

    SHADOW MODE: All entries include "mode": "SHADOW".

    See: docs/system_law/Phase_X_P3_Spec.md Section 5.1

    Example:
        {
          "schema": "first-light-cycle/1.0.0",
          "cycle": 42,
          "timestamp": "2025-12-09T12:00:00Z",
          "mode": "SHADOW",
          "runner": {"type": "u2", "slice": "arithmetic_simple", "success": true},
          "usla_state": {"H": 0.75, "rho": 0.85, "tau": 0.21},
          "governance": {"real_blocked": false, "sim_blocked": false, "aligned": true},
          "metrics": {"hard_ok": true, "in_omega": true}
        }
    """

    schema: str = CYCLE_LOG_SCHEMA_VERSION
    cycle: int = 0
    timestamp: str = ""
    mode: str = "SHADOW"  # Always "SHADOW" in P3

    # Runner output
    runner_type: str = ""
    runner_slice: str = ""
    runner_success: bool = False
    runner_depth: Optional[int] = None

    # USLA state snapshot
    usla_H: float = 0.0
    usla_rho: float = 0.0
    usla_tau: float = 0.0
    usla_beta: float = 0.0

    # Governance
    real_blocked: bool = False
    sim_blocked: bool = False
    governance_aligned: bool = True

    # Metrics
    hard_ok: bool = True
    in_omega: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "schema": self.schema,
            "cycle": self.cycle,
            "timestamp": self.timestamp,
            "mode": self.mode,
            "runner": {
                "type": self.runner_type,
                "slice": self.runner_slice,
                "success": self.runner_success,
                "depth": self.runner_depth,
            },
            "usla_state": {
                "H": self.usla_H,
                "rho": self.usla_rho,
                "tau": self.usla_tau,
                "beta": self.usla_beta,
            },
            "governance": {
                "real_blocked": self.real_blocked,
                "sim_blocked": self.sim_blocked,
                "aligned": self.governance_aligned,
            },
            "metrics": {
                "hard_ok": self.hard_ok,
                "in_omega": self.in_omega,
            },
        }


@dataclass
class RedFlagLogEntry:
    """
    Schema for red_flags.jsonl entries.

    SHADOW MODE: All entries include:
    - "mode": "SHADOW"
    - "action": "LOGGED_ONLY"

    See: docs/system_law/Phase_X_P3_Spec.md Section 5.2

    Example:
        {
          "schema": "first-light-red-flag/1.0.0",
          "cycle": 142,
          "timestamp": "2025-12-09T12:01:42Z",
          "mode": "SHADOW",
          "flag": {"type": "RSI_COLLAPSE", "severity": "WARNING", "observed_value": 0.18},
          "action": "LOGGED_ONLY",
          "hypothetical": {"would_abort": false}
        }
    """

    schema: str = RED_FLAG_LOG_SCHEMA_VERSION
    cycle: int = 0
    timestamp: str = ""
    mode: str = "SHADOW"  # Always "SHADOW" in P3

    # Flag details
    flag_type: str = ""
    flag_severity: str = "INFO"
    observed_value: float = 0.0
    threshold: float = 0.0
    consecutive_cycles: int = 0

    # SHADOW MODE: Action is always "LOGGED_ONLY"
    action: str = "LOGGED_ONLY"

    # Hypothetical analysis (for logging only, NEVER for control)
    hypothetical_would_abort: bool = False
    hypothetical_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "schema": self.schema,
            "cycle": self.cycle,
            "timestamp": self.timestamp,
            "mode": self.mode,
            "flag": {
                "type": self.flag_type,
                "severity": self.flag_severity,
                "observed_value": self.observed_value,
                "threshold": self.threshold,
                "consecutive_cycles": self.consecutive_cycles,
            },
            "action": self.action,
            "hypothetical": {
                "would_abort": self.hypothetical_would_abort,
                "reason": self.hypothetical_reason,
            },
        }


@dataclass
class MetricsLogEntry:
    """
    Schema for metrics.jsonl entries.

    SHADOW MODE: All entries include "mode": "SHADOW".

    See: docs/system_law/Phase_X_P3_Spec.md Section 5.3

    Example:
        {
          "schema": "first-light-metrics/1.0.0",
          "window_index": 8,
          "window_start_cycle": 400,
          "window_end_cycle": 449,
          "mode": "SHADOW",
          "success_metrics": {"window_success_rate": 0.82, "delta_p_success": 0.0012},
          "stability_metrics": {"window_mean_rsi": 0.84}
        }
    """

    schema: str = METRICS_LOG_SCHEMA_VERSION
    window_index: int = 0
    window_start_cycle: int = 0
    window_end_cycle: int = 0
    timestamp: str = ""
    mode: str = "SHADOW"  # Always "SHADOW" in P3

    # Success metrics
    window_success_rate: Optional[float] = None
    cumulative_success_rate: Optional[float] = None
    delta_p_success: Optional[float] = None

    # Abstention metrics
    window_abstention_rate: Optional[float] = None
    cumulative_abstention_rate: Optional[float] = None
    delta_p_abstention: Optional[float] = None

    # Stability metrics
    window_mean_rsi: Optional[float] = None
    cumulative_mean_rsi: Optional[float] = None

    # Safe region metrics
    window_omega_occupancy: Optional[float] = None
    cumulative_omega_occupancy: Optional[float] = None

    # HARD mode metrics
    window_hard_ok_rate: Optional[float] = None
    cumulative_hard_ok_rate: Optional[float] = None

    # Red-flag counts for this window
    red_flag_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "schema": self.schema,
            "window_index": self.window_index,
            "window_start_cycle": self.window_start_cycle,
            "window_end_cycle": self.window_end_cycle,
            "timestamp": self.timestamp,
            "mode": self.mode,
            "success_metrics": {
                "window_success_rate": self.window_success_rate,
                "cumulative_success_rate": self.cumulative_success_rate,
                "delta_p_success": self.delta_p_success,
            },
            "abstention_metrics": {
                "window_abstention_rate": self.window_abstention_rate,
                "cumulative_abstention_rate": self.cumulative_abstention_rate,
                "delta_p_abstention": self.delta_p_abstention,
            },
            "stability_metrics": {
                "window_mean_rsi": self.window_mean_rsi,
                "cumulative_mean_rsi": self.cumulative_mean_rsi,
            },
            "safe_region_metrics": {
                "window_omega_occupancy": self.window_omega_occupancy,
                "cumulative_omega_occupancy": self.cumulative_omega_occupancy,
            },
            "hard_mode_metrics": {
                "window_hard_ok_rate": self.window_hard_ok_rate,
                "cumulative_hard_ok_rate": self.cumulative_hard_ok_rate,
            },
            "red_flag_counts": self.red_flag_counts,
        }


@dataclass
class SummarySchema:
    """
    Schema for summary.json.

    SHADOW MODE: Includes "mode": "SHADOW" at top level.

    See: docs/system_law/Phase_X_P3_Spec.md Section 5.4

    Example:
        {
          "schema": "first-light-summary/1.0.0",
          "run_id": "fl_20251209_120000_abc123",
          "mode": "SHADOW",
          "config": {...},
          "execution": {...},
          "success_criteria": {...},
          "red_flag_summary": {...}
        }
    """

    schema: str = SUMMARY_SCHEMA_VERSION
    run_id: str = ""
    mode: str = "SHADOW"  # Always "SHADOW" in P3

    # Config summary
    slice_name: str = ""
    runner_type: str = ""
    total_cycles: int = 0
    tau_0: float = 0.0

    # Execution summary
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0
    cycles_completed: int = 0

    # Success criteria results (observational only)
    success_criteria: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Red-flag summary
    red_flag_total: int = 0
    red_flag_by_type: Dict[str, int] = field(default_factory=dict)
    hypothetical_aborts: int = 0

    # Evidence quality summary (Phase X — observational only)
    evidence_quality_summary: Optional[Dict[str, Any]] = None

    # Consensus governance summary (Phase X — observational only)
    consensus_governance_summary: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "mode": self.mode,
            "config": {
                "slice_name": self.slice_name,
                "runner_type": self.runner_type,
                "total_cycles": self.total_cycles,
                "tau_0": self.tau_0,
            },
            "execution": {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "duration_seconds": self.duration_seconds,
                "cycles_completed": self.cycles_completed,
            },
            "success_criteria": self.success_criteria,
            "red_flag_summary": {
                "total_observations": self.red_flag_total,
                "by_type": self.red_flag_by_type,
                "hypothetical_aborts": self.hypothetical_aborts,
            },
            "consensus_governance_summary": self.consensus_governance_summary,
        }
