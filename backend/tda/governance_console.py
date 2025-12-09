"""
TDA Governance Console — Phase V Operator Interface

Operation CORTEX: Phase V Operator Console & Self-Audit Harness
================================================================

This module provides a read-only API layer over existing TDA governance
primitives (Phases II-IV), enabling operators and auditors to inspect
TDA hard-gate behavior over time.

Design Principles:
- Pure, deterministic functions with no side effects
- All outputs are JSON-serializable
- No normative/vibe language — fields are neutral and structural
- Composes from existing Phase III/IV functions (no duplicated logic)

Usage:
    from backend.tda.governance_console import (
        TDAGovernanceSnapshot,
        build_governance_console_snapshot,
    )
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

from backend.tda.governance import (
    TDAHardGateMode,
    CalibrationResult,
    GovernanceAlignmentResult,
    ExceptionWindowManager,
    ExceptionWindowState,
    summarize_tda_for_global_health,
    evaluate_hard_gate_calibration,
    evaluate_tda_governance_alignment,
    build_tda_hard_gate_evidence_tile,
    TDA_GOVERNANCE_SCHEMA_VERSION,
)

if TYPE_CHECKING:
    from backend.tda.runtime_monitor import TDAMonitorResult, TDAMonitorConfig

# Schema versions
GOVERNANCE_CONSOLE_SCHEMA_VERSION = "tda-governance-console-1.0.0"
BLOCK_EXPLANATION_SCHEMA_VERSION = "tda-block-explanation-1.0.0"
LONGHORIZON_DRIFT_SCHEMA_VERSION = "tda-longhorizon-drift-1.0.0"


# ============================================================================
# HSS Trend Classification
# ============================================================================

class HSSTrend(Enum):
    """Classification of HSS trend over time."""
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    UNKNOWN = "unknown"


def classify_hss_trend(
    hss_values: Sequence[float],
    improving_threshold: float = 0.01,
    degrading_threshold: float = -0.01,
    min_samples: int = 5,
) -> HSSTrend:
    """
    Classify HSS trend based on linear regression slope.

    Args:
        hss_values: Sequence of HSS values over time.
        improving_threshold: Slope above this is "improving".
        degrading_threshold: Slope below this is "degrading".
        min_samples: Minimum samples for trend analysis.

    Returns:
        HSSTrend classification.
    """
    if len(hss_values) < min_samples:
        return HSSTrend.UNKNOWN

    x = np.arange(len(hss_values))
    try:
        slope = float(np.polyfit(x, hss_values, 1)[0])
    except (np.linalg.LinAlgError, ValueError):
        return HSSTrend.UNKNOWN

    if slope > improving_threshold:
        return HSSTrend.IMPROVING
    elif slope < degrading_threshold:
        return HSSTrend.DEGRADING
    else:
        return HSSTrend.STABLE


# ============================================================================
# Golden Alignment Status
# ============================================================================

class GoldenAlignmentStatus(Enum):
    """Status of golden set calibration alignment."""
    ALIGNED = "ALIGNED"
    DRIFTING = "DRIFTING"
    BROKEN = "BROKEN"
    UNKNOWN = "UNKNOWN"


def classify_golden_alignment(calibration_status: str) -> GoldenAlignmentStatus:
    """
    Map calibration status string to golden alignment enum.

    Args:
        calibration_status: Status from CalibrationResult ("OK", "DRIFTING", "BROKEN").

    Returns:
        GoldenAlignmentStatus enum value.
    """
    mapping = {
        "OK": GoldenAlignmentStatus.ALIGNED,
        "DRIFTING": GoldenAlignmentStatus.DRIFTING,
        "BROKEN": GoldenAlignmentStatus.BROKEN,
    }
    return mapping.get(calibration_status, GoldenAlignmentStatus.UNKNOWN)


# ============================================================================
# Exception Window Descriptor
# ============================================================================

@dataclass(frozen=True)
class ExceptionWindowDescriptor:
    """Compact descriptor of an exception window activation."""
    active: bool
    runs_remaining: int
    total_runs: int
    activation_reason: Optional[str]
    activated_at: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active": self.active,
            "runs_remaining": self.runs_remaining,
            "total_runs": self.total_runs,
            "activation_reason": self.activation_reason,
            "activated_at": self.activated_at,
        }


def build_exception_descriptor(
    state: ExceptionWindowState,
) -> ExceptionWindowDescriptor:
    """
    Build exception window descriptor from state.

    Args:
        state: ExceptionWindowState from manager.

    Returns:
        Frozen ExceptionWindowDescriptor.
    """
    return ExceptionWindowDescriptor(
        active=state.active,
        runs_remaining=state.runs_remaining,
        total_runs=state.total_runs,
        activation_reason=state.activation_reason,
        activated_at=state.activated_at,
    )


# ============================================================================
# TDA Governance Snapshot
# ============================================================================

@dataclass(frozen=True)
class TDAGovernanceSnapshot:
    """
    Immutable snapshot of TDA governance state.

    This is the primary output of the governance console API,
    providing a complete picture of TDA hard-gate behavior.
    """
    schema_version: str
    mode: TDAHardGateMode
    cycle_count: int
    block_rate: float
    warn_rate: float
    mean_hss: float
    hss_trend: str  # "improving" | "stable" | "degrading" | "unknown"
    golden_alignment: str  # "ALIGNED" | "DRIFTING" | "BROKEN" | "UNKNOWN"
    exception_windows_active: int
    recent_exceptions: Tuple[Dict[str, Any], ...]
    governance_signal: str  # "HEALTHY" | "DEGRADED" | "CRITICAL"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "schema_version": self.schema_version,
            "mode": self.mode.value,
            "cycle_count": self.cycle_count,
            "block_rate": round(self.block_rate, 4),
            "warn_rate": round(self.warn_rate, 4),
            "mean_hss": round(self.mean_hss, 4),
            "hss_trend": self.hss_trend,
            "golden_alignment": self.golden_alignment,
            "exception_windows_active": self.exception_windows_active,
            "recent_exceptions": list(self.recent_exceptions),
            "governance_signal": self.governance_signal,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


# ============================================================================
# Governance Console Snapshot Builder
# ============================================================================

def build_governance_console_snapshot(
    tda_results: Sequence["TDAMonitorResult"],
    hard_gate_decisions: Sequence[Dict[str, Any]],
    golden_state: Optional[Dict[str, Any]],
    exception_manager: Optional[ExceptionWindowManager],
    mode: TDAHardGateMode = TDAHardGateMode.HARD,
) -> TDAGovernanceSnapshot:
    """
    Build governance console snapshot from TDA state.

    Composes from existing Phase III/IV functions without duplicating logic.

    Args:
        tda_results: Sequence of TDAMonitorResult from evaluations.
        hard_gate_decisions: Sequence of decision dicts with gate outcomes.
        golden_state: Dict with calibration_status from golden set evaluation.
            Expected keys: "calibration_status", "false_block_rate", "false_pass_rate"
        exception_manager: Optional ExceptionWindowManager instance.
        mode: Current hard gate mode.

    Returns:
        Frozen TDAGovernanceSnapshot with all governance metrics.

    Example:
        >>> snapshot = build_governance_console_snapshot(
        ...     tda_results=results,
        ...     hard_gate_decisions=decisions,
        ...     golden_state={"calibration_status": "OK"},
        ...     exception_manager=manager,
        ... )
        >>> print(snapshot.to_json())
    """
    # Handle empty inputs
    if not tda_results:
        return TDAGovernanceSnapshot(
            schema_version=GOVERNANCE_CONSOLE_SCHEMA_VERSION,
            mode=mode,
            cycle_count=0,
            block_rate=0.0,
            warn_rate=0.0,
            mean_hss=0.0,
            hss_trend=HSSTrend.UNKNOWN.value,
            golden_alignment=GoldenAlignmentStatus.UNKNOWN.value,
            exception_windows_active=0,
            recent_exceptions=(),
            governance_signal="HEALTHY",
        )

    # Extract HSS values for trend analysis
    hss_values = [r.hss for r in tda_results]
    hss_trend = classify_hss_trend(hss_values)

    # Compute block/warn rates from results
    block_count = sum(1 for r in tda_results if r.block)
    warn_count = sum(1 for r in tda_results if r.warn and not r.block)
    cycle_count = len(tda_results)

    block_rate = block_count / cycle_count if cycle_count > 0 else 0.0
    warn_rate = warn_count / cycle_count if cycle_count > 0 else 0.0
    mean_hss = float(np.mean(hss_values)) if hss_values else 0.0

    # Determine governance signal
    if block_rate > 0.2 or mean_hss < 0.3:
        governance_signal = "CRITICAL"
    elif block_rate > 0.1 or mean_hss < 0.5:
        governance_signal = "DEGRADED"
    else:
        governance_signal = "HEALTHY"

    # Golden alignment from calibration state
    if golden_state and "calibration_status" in golden_state:
        golden_alignment = classify_golden_alignment(
            golden_state["calibration_status"]
        )
    else:
        golden_alignment = GoldenAlignmentStatus.UNKNOWN

    # Exception window state
    exception_windows_active = 0
    recent_exceptions: List[Dict[str, Any]] = []

    if exception_manager:
        state = exception_manager.get_state()
        if state.active:
            exception_windows_active = 1
            recent_exceptions.append(build_exception_descriptor(state).to_dict())

    return TDAGovernanceSnapshot(
        schema_version=GOVERNANCE_CONSOLE_SCHEMA_VERSION,
        mode=mode,
        cycle_count=cycle_count,
        block_rate=block_rate,
        warn_rate=warn_rate,
        mean_hss=mean_hss,
        hss_trend=hss_trend.value,
        golden_alignment=golden_alignment.value,
        exception_windows_active=exception_windows_active,
        recent_exceptions=tuple(recent_exceptions),
        governance_signal=governance_signal,
    )


# ============================================================================
# Block Explanation Builder
# ============================================================================

@dataclass(frozen=True)
class BlockExplanation:
    """
    Structured explanation of why a cycle was blocked/warned.

    No free-text judgment — only structured reasons and flags.
    """
    schema_version: str
    run_id: str
    cycle_id: int
    tda_mode: str
    hss: float
    scores: Dict[str, float]
    gate_decision: Dict[str, Any]
    effects: Dict[str, bool]
    status: str  # "BLOCK" | "WARN" | "OK" | "UNKNOWN"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "cycle_id": self.cycle_id,
            "tda_mode": self.tda_mode,
            "hss": round(self.hss, 4),
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "gate_decision": self.gate_decision,
            "effects": self.effects,
            "status": self.status,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def build_reason_codes(
    hss: float,
    block_threshold: float = 0.2,
    warn_threshold: float = 0.4,
    golden_alignment: str = "ALIGNED",
    exception_window_applied: bool = False,
) -> List[str]:
    """
    Build structured reason codes for a gate decision.

    Args:
        hss: HSS score for the cycle.
        block_threshold: Threshold for blocking.
        warn_threshold: Threshold for warning.
        golden_alignment: Current golden alignment status.
        exception_window_applied: Whether exception window was active.

    Returns:
        List of reason code strings (e.g., ["HSS_BELOW_THRESHOLD"]).
    """
    codes = []

    if hss < block_threshold:
        codes.append("HSS_BELOW_THRESHOLD")
    elif hss < warn_threshold:
        codes.append("HSS_BELOW_WARN_THRESHOLD")

    if golden_alignment in ("DRIFTING", "BROKEN"):
        codes.append("GOLDEN_MISALIGNED")

    if exception_window_applied:
        codes.append("EXCEPTION_WINDOW_APPLIED")

    if not codes:
        codes.append("HSS_ACCEPTABLE")

    return codes


def build_block_explanation(
    run_id: str,
    cycle_id: int,
    tda_mode: TDAHardGateMode,
    hss: float,
    sns: float,
    pcs: float,
    drs: float,
    block: bool,
    warn: bool,
    exception_window_applied: bool = False,
    lean_submission_avoided: bool = False,
    policy_update_avoided: bool = False,
    golden_alignment: str = "ALIGNED",
    block_threshold: float = 0.2,
    warn_threshold: float = 0.4,
) -> BlockExplanation:
    """
    Build structured block explanation for a cycle.

    Args:
        run_id: Identifier for the run.
        cycle_id: Cycle index or unique ID.
        tda_mode: Hard gate mode at time of decision.
        hss: Hallucination Stability Score.
        sns: Structural Novelty Score.
        pcs: Proof Coherence Score.
        drs: Derivation Risk Score.
        block: Whether the cycle was blocked.
        warn: Whether the cycle triggered a warning.
        exception_window_applied: Whether exception window was active.
        lean_submission_avoided: Whether Lean submission was skipped.
        policy_update_avoided: Whether policy update was skipped.
        golden_alignment: Current golden alignment status.
        block_threshold: Threshold for blocking.
        warn_threshold: Threshold for warning.

    Returns:
        Frozen BlockExplanation with all decision context.
    """
    # Determine status
    if block:
        status = "BLOCK"
    elif warn:
        status = "WARN"
    else:
        status = "OK"

    # Build reason codes
    reason_codes = build_reason_codes(
        hss=hss,
        block_threshold=block_threshold,
        warn_threshold=warn_threshold,
        golden_alignment=golden_alignment,
        exception_window_applied=exception_window_applied,
    )

    return BlockExplanation(
        schema_version=BLOCK_EXPLANATION_SCHEMA_VERSION,
        run_id=run_id,
        cycle_id=cycle_id,
        tda_mode=tda_mode.value,
        hss=hss,
        scores={
            "sns": sns,
            "pcs": pcs,
            "drs": drs,
        },
        gate_decision={
            "status": status,
            "reason_codes": reason_codes,
            "exception_window_applied": exception_window_applied,
        },
        effects={
            "lean_submission_avoided": lean_submission_avoided,
            "policy_update_avoided": policy_update_avoided,
        },
        status=status,
    )


def build_block_explanation_from_ledger_entry(
    run_id: str,
    cycle_id: int,
    ledger_entry: Dict[str, Any],
    tda_mode: TDAHardGateMode = TDAHardGateMode.HARD,
    golden_alignment: str = "ALIGNED",
) -> BlockExplanation:
    """
    Build block explanation from a run ledger entry.

    Robust to missing TDA data — emits UNKNOWN status when appropriate.

    Args:
        run_id: Identifier for the run.
        cycle_id: Cycle index.
        ledger_entry: Dict with TDA fields from RunLedgerEntry.
        tda_mode: Hard gate mode (if not in entry).
        golden_alignment: Golden alignment status.

    Returns:
        BlockExplanation, with status="UNKNOWN" if TDA data missing.
    """
    # Check for TDA data presence
    if "tda_hss" not in ledger_entry and "hss" not in ledger_entry:
        return BlockExplanation(
            schema_version=BLOCK_EXPLANATION_SCHEMA_VERSION,
            run_id=run_id,
            cycle_id=cycle_id,
            tda_mode=tda_mode.value,
            hss=0.0,
            scores={"sns": 0.0, "pcs": 0.0, "drs": 0.0},
            gate_decision={
                "status": "UNKNOWN",
                "reason_codes": ["TDA_DATA_MISSING"],
                "exception_window_applied": False,
            },
            effects={
                "lean_submission_avoided": False,
                "policy_update_avoided": False,
            },
            status="UNKNOWN",
        )

    # Extract TDA scores (support multiple field naming conventions)
    hss = ledger_entry.get("tda_hss", ledger_entry.get("hss", 0.0))
    sns = ledger_entry.get("tda_sns", ledger_entry.get("sns", 0.0))
    pcs = ledger_entry.get("tda_pcs", ledger_entry.get("pcs", 0.0))
    drs = ledger_entry.get("tda_drs", ledger_entry.get("drs", 0.0))

    # Extract gate decision
    tda_outcome = ledger_entry.get("tda_outcome", "OK")
    block = tda_outcome == "BLOCK" or ledger_entry.get("tda_gate_enforced", False)
    warn = tda_outcome == "WARN"

    # Extract effects
    lean_submission_avoided = ledger_entry.get("lean_submission_avoided", False)
    policy_update_avoided = ledger_entry.get("policy_update_avoided", False)

    # Exception window
    exception_window_applied = ledger_entry.get("exception_window_applied", False)

    # Mode from entry or default
    entry_mode = ledger_entry.get("tda_mode", tda_mode.value)
    if isinstance(entry_mode, str):
        try:
            tda_mode = TDAHardGateMode(entry_mode.lower())
        except ValueError:
            pass

    return build_block_explanation(
        run_id=run_id,
        cycle_id=cycle_id,
        tda_mode=tda_mode,
        hss=hss,
        sns=sns,
        pcs=pcs,
        drs=drs,
        block=block,
        warn=warn,
        exception_window_applied=exception_window_applied,
        lean_submission_avoided=lean_submission_avoided,
        policy_update_avoided=policy_update_avoided,
        golden_alignment=golden_alignment,
    )


# ============================================================================
# Long-Horizon Drift Analysis
# ============================================================================

class TrendDirection(Enum):
    """Direction of a metric trend over time."""
    INCREASING = "increasing"
    STABLE = "stable"
    DECREASING = "decreasing"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class LongHorizonDriftReport:
    """
    Report on TDA governance drift over multiple runs.

    Captures long-term trends in hard gate behavior.
    """
    schema_version: str
    runs_analyzed: int
    first_run_timestamp: Optional[str]
    last_run_timestamp: Optional[str]
    block_rate_trend: str  # "increasing" | "stable" | "decreasing"
    mean_hss_trend: str    # "improving" | "stable" | "degrading"
    golden_alignment_trend: str  # "stable" | "drifting" | "broken"
    exception_usage: Dict[str, Any]
    governance_signal: str  # "OK" | "ATTENTION" | "ALERT"
    recommendations: Tuple[str, ...]
    metrics: Dict[str, Any]  # Raw metrics for further analysis

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "runs_analyzed": self.runs_analyzed,
            "first_run_timestamp": self.first_run_timestamp,
            "last_run_timestamp": self.last_run_timestamp,
            "block_rate_trend": self.block_rate_trend,
            "mean_hss_trend": self.mean_hss_trend,
            "golden_alignment_trend": self.golden_alignment_trend,
            "exception_usage": self.exception_usage,
            "governance_signal": self.governance_signal,
            "recommendations": list(self.recommendations),
            "metrics": self.metrics,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def classify_trend(
    values: Sequence[float],
    increasing_threshold: float = 0.01,
    decreasing_threshold: float = -0.01,
    min_samples: int = 3,
) -> TrendDirection:
    """
    Classify trend direction from a sequence of values.

    Args:
        values: Sequence of values over time.
        increasing_threshold: Slope above this is "increasing".
        decreasing_threshold: Slope below this is "decreasing".
        min_samples: Minimum samples for trend analysis.

    Returns:
        TrendDirection classification.
    """
    if len(values) < min_samples:
        return TrendDirection.UNKNOWN

    x = np.arange(len(values))
    try:
        slope = float(np.polyfit(x, values, 1)[0])
    except (np.linalg.LinAlgError, ValueError):
        return TrendDirection.UNKNOWN

    if slope > increasing_threshold:
        return TrendDirection.INCREASING
    elif slope < decreasing_threshold:
        return TrendDirection.DECREASING
    else:
        return TrendDirection.STABLE


def analyze_golden_alignment_trend(
    alignment_statuses: Sequence[str],
) -> str:
    """
    Analyze trend in golden alignment statuses.

    Args:
        alignment_statuses: Sequence of alignment status strings.

    Returns:
        Trend classification: "stable", "drifting", or "broken".
    """
    if not alignment_statuses:
        return "stable"

    # Count status occurrences in recent half
    mid = len(alignment_statuses) // 2
    recent = alignment_statuses[mid:] if mid > 0 else alignment_statuses

    broken_count = sum(1 for s in recent if s == "BROKEN")
    drifting_count = sum(1 for s in recent if s == "DRIFTING")

    if broken_count > len(recent) * 0.3:
        return "broken"
    elif drifting_count > len(recent) * 0.3:
        return "drifting"
    else:
        return "stable"


def build_longhorizon_drift_report(
    governance_tiles: Sequence[Dict[str, Any]],
) -> LongHorizonDriftReport:
    """
    Build long-horizon drift report from governance tiles.

    Args:
        governance_tiles: Sequence of TDA governance tile dicts from multiple runs.
            Each tile should have: block_rate, mean_hss, golden_alignment,
            exception_active, timestamp (optional).

    Returns:
        LongHorizonDriftReport with trend analysis and recommendations.
    """
    if not governance_tiles:
        return LongHorizonDriftReport(
            schema_version=LONGHORIZON_DRIFT_SCHEMA_VERSION,
            runs_analyzed=0,
            first_run_timestamp=None,
            last_run_timestamp=None,
            block_rate_trend=TrendDirection.UNKNOWN.value,
            mean_hss_trend=HSSTrend.UNKNOWN.value,
            golden_alignment_trend="stable",
            exception_usage={
                "total_windows": 0,
                "per_run_mean": 0.0,
                "trend": TrendDirection.UNKNOWN.value,
            },
            governance_signal="OK",
            recommendations=(),
            metrics={},
        )

    # Extract time series
    block_rates = [t.get("block_rate", 0.0) for t in governance_tiles]
    mean_hss_values = [t.get("mean_hss", 0.0) for t in governance_tiles]
    golden_alignments = [t.get("golden_alignment", "ALIGNED") for t in governance_tiles]
    exception_actives = [1 if t.get("exception_active", False) else 0 for t in governance_tiles]

    # Extract timestamps if available
    timestamps = [t.get("timestamp") for t in governance_tiles]
    first_timestamp = next((t for t in timestamps if t), None)
    last_timestamp = next((t for t in reversed(timestamps) if t), None)

    # Compute trends
    block_rate_trend = classify_trend(block_rates)
    mean_hss_trend = classify_hss_trend(mean_hss_values)
    golden_alignment_trend = analyze_golden_alignment_trend(golden_alignments)

    # Exception usage analysis
    total_exception_windows = sum(exception_actives)
    per_run_mean = total_exception_windows / len(governance_tiles)
    exception_trend = classify_trend(exception_actives, increasing_threshold=0.05)

    exception_usage = {
        "total_windows": total_exception_windows,
        "per_run_mean": round(per_run_mean, 4),
        "trend": exception_trend.value,
    }

    # Determine governance signal
    recommendations = []

    # Logic: block_rate rising + mean_hss falling → ATTENTION/ALERT
    block_rate_rising = block_rate_trend == TrendDirection.INCREASING
    mean_hss_falling = mean_hss_trend == HSSTrend.DEGRADING

    if block_rate_rising and mean_hss_falling:
        governance_signal = "ALERT"
        recommendations.append("Block rate increasing while HSS degrading — investigate TDA calibration")
    elif block_rate_rising or mean_hss_falling:
        governance_signal = "ATTENTION"
        if block_rate_rising:
            recommendations.append("Block rate trend is increasing")
        if mean_hss_falling:
            recommendations.append("Mean HSS trend is degrading")
    else:
        governance_signal = "OK"

    # Exception window + golden alignment check
    if exception_trend == TrendDirection.INCREASING and golden_alignment_trend in ("drifting", "broken"):
        governance_signal = "ALERT"
        recommendations.append("Exception window usage increasing with golden alignment drift — recalibration recommended")
    elif golden_alignment_trend == "broken":
        governance_signal = "ALERT"
        recommendations.append("Golden alignment broken — immediate review required")
    elif golden_alignment_trend == "drifting":
        if governance_signal == "OK":
            governance_signal = "ATTENTION"
        recommendations.append("Golden alignment drifting — consider recalibration")

    # Compute summary metrics
    metrics = {
        "block_rate_series": [round(v, 4) for v in block_rates],
        "mean_hss_series": [round(v, 4) for v in mean_hss_values],
        "block_rate_mean": round(float(np.mean(block_rates)), 4) if block_rates else 0.0,
        "block_rate_std": round(float(np.std(block_rates)), 4) if len(block_rates) > 1 else 0.0,
        "mean_hss_mean": round(float(np.mean(mean_hss_values)), 4) if mean_hss_values else 0.0,
        "mean_hss_std": round(float(np.std(mean_hss_values)), 4) if len(mean_hss_values) > 1 else 0.0,
    }

    return LongHorizonDriftReport(
        schema_version=LONGHORIZON_DRIFT_SCHEMA_VERSION,
        runs_analyzed=len(governance_tiles),
        first_run_timestamp=first_timestamp,
        last_run_timestamp=last_timestamp,
        block_rate_trend=block_rate_trend.value,
        mean_hss_trend=mean_hss_trend.value,
        golden_alignment_trend=golden_alignment_trend,
        exception_usage=exception_usage,
        governance_signal=governance_signal,
        recommendations=tuple(recommendations),
        metrics=metrics,
    )


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Schema versions
    "GOVERNANCE_CONSOLE_SCHEMA_VERSION",
    "BLOCK_EXPLANATION_SCHEMA_VERSION",
    "LONGHORIZON_DRIFT_SCHEMA_VERSION",
    # Enums
    "HSSTrend",
    "GoldenAlignmentStatus",
    "TrendDirection",
    # Data classes
    "TDAGovernanceSnapshot",
    "ExceptionWindowDescriptor",
    "BlockExplanation",
    "LongHorizonDriftReport",
    # Functions
    "classify_hss_trend",
    "classify_golden_alignment",
    "classify_trend",
    "build_governance_console_snapshot",
    "build_block_explanation",
    "build_block_explanation_from_ledger_entry",
    "build_reason_codes",
    "analyze_golden_alignment_trend",
    "build_longhorizon_drift_report",
]
