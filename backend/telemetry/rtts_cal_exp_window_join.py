"""
RTTS × CAL-EXP Window Join Helper

Phase X P5.2: Observational join of RTTS validation with CAL-EXP windows.

This module provides a join helper that annotates CAL-EXP windows with
RTTS mock detection flags and continuity violation rates.

SHADOW MODE CONTRACT:
- All operations are OBSERVATIONAL ONLY
- No gating or enforcement
- Missing RTTS data is handled gracefully
- Results are advisory only

RTTS Gap Closure: P5.2 VALIDATE stage extension
See: docs/system_law/RTTS_Gap_Closure_Blueprint.md

Status: P5.2 VALIDATE (NO ENFORCEMENT)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from backend.telemetry.rtts_window_validator import RTTSValidationBlock
    from backend.topology.first_light.metrics_window import MetricsWindow

__all__ = [
    "RTTSAnnotatedWindow",
    "join_rtts_to_cal_exp_windows",
    "annotate_window_with_rtts",
]


# Schema version for annotated windows
RTTS_ANNOTATED_WINDOW_SCHEMA_VERSION = "1.0.0"


@dataclass
class RTTSAnnotatedWindow:
    """
    CAL-EXP window annotated with RTTS validation results.

    SHADOW MODE: Annotations are observational only.
    """

    # Window identification
    window_index: int = 0
    start_cycle: int = 0
    end_cycle: int = 0

    # Original CAL-EXP metrics (copied from MetricsWindow)
    success_rate: Optional[float] = None
    omega_occupancy: Optional[float] = None
    mean_rsi: Optional[float] = None
    block_rate: Optional[float] = None

    # RTTS annotations
    mock_flags_count: int = 0
    mock_flags: List[str] = field(default_factory=list)
    continuity_violation_rate: float = 0.0
    rtts_overall_status: str = "UNKNOWN"
    rtts_warning_count: int = 0

    # Join metadata
    rtts_available: bool = False
    schema_version: str = RTTS_ANNOTATED_WINDOW_SCHEMA_VERSION
    mode: str = "SHADOW"
    action: str = "LOGGED_ONLY"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "schema_version": self.schema_version,
            "window_index": self.window_index,
            "start_cycle": self.start_cycle,
            "end_cycle": self.end_cycle,
            "cal_exp_metrics": {
                "success_rate": round(self.success_rate, 6) if self.success_rate is not None else None,
                "omega_occupancy": round(self.omega_occupancy, 6) if self.omega_occupancy is not None else None,
                "mean_rsi": round(self.mean_rsi, 6) if self.mean_rsi is not None else None,
                "block_rate": round(self.block_rate, 6) if self.block_rate is not None else None,
            },
            "rtts_annotations": {
                "mock_flags_count": self.mock_flags_count,
                "mock_flags": self.mock_flags,
                "continuity_violation_rate": round(self.continuity_violation_rate, 6),
                "overall_status": self.rtts_overall_status,
                "warning_count": self.rtts_warning_count,
                "rtts_available": self.rtts_available,
            },
            "mode": self.mode,
            "action": self.action,
        }


def annotate_window_with_rtts(
    window: "MetricsWindow",
    rtts_block: Optional["RTTSValidationBlock"] = None,
) -> RTTSAnnotatedWindow:
    """
    Annotate a single CAL-EXP window with RTTS validation results.

    SHADOW MODE CONTRACT:
    - Non-mutating (creates new RTTSAnnotatedWindow)
    - Missing RTTS is handled gracefully
    - No gating or enforcement

    Args:
        window: MetricsWindow from CAL-EXP run
        rtts_block: Optional RTTSValidationBlock from RTTS validation

    Returns:
        RTTSAnnotatedWindow with CAL-EXP metrics and RTTS annotations
    """
    annotated = RTTSAnnotatedWindow(
        window_index=window.window_index,
        start_cycle=window.start_cycle,
        end_cycle=window.end_cycle,
        success_rate=window.success_rate,
        omega_occupancy=window.omega_occupancy,
        mean_rsi=window.mean_rsi,
        block_rate=window.block_rate,
    )

    if rtts_block is not None:
        annotated.rtts_available = True
        annotated.rtts_overall_status = rtts_block.overall_status
        annotated.rtts_warning_count = rtts_block.warning_count

        # Extract mock flags
        if rtts_block.mock_detection is not None:
            mock_flags = []
            md = rtts_block.mock_detection
            if md.mock_001_var_H_low:
                mock_flags.append("MOCK-001")
            if md.mock_002_var_rho_low:
                mock_flags.append("MOCK-002")
            if md.mock_003_cor_low:
                mock_flags.append("MOCK-003")
            if md.mock_004_cor_high:
                mock_flags.append("MOCK-004")
            if md.mock_005_acf_low:
                mock_flags.append("MOCK-005")
            if md.mock_006_acf_high:
                mock_flags.append("MOCK-006")
            if md.mock_007_kurtosis_low:
                mock_flags.append("MOCK-007")
            if md.mock_008_kurtosis_high:
                mock_flags.append("MOCK-008")
            if md.mock_009_jump_H:
                mock_flags.append("MOCK-009")
            if md.mock_010_discrete_rho:
                mock_flags.append("MOCK-010")

            annotated.mock_flags = mock_flags
            annotated.mock_flags_count = len(mock_flags)

        # Extract continuity violation rate
        if rtts_block.continuity is not None:
            annotated.continuity_violation_rate = rtts_block.continuity.violation_rate

    return annotated


def join_rtts_to_cal_exp_windows(
    windows: List["MetricsWindow"],
    rtts_block: Optional["RTTSValidationBlock"] = None,
) -> List[RTTSAnnotatedWindow]:
    """
    Join RTTS validation results to CAL-EXP windows.

    SHADOW MODE CONTRACT:
    - Non-mutating (creates new list of RTTSAnnotatedWindow)
    - Missing RTTS is handled gracefully (windows get rtts_available=False)
    - Deterministic ordering (preserves window order)
    - No gating or enforcement

    Args:
        windows: List of MetricsWindow from CAL-EXP run
        rtts_block: Optional RTTSValidationBlock from RTTS validation

    Returns:
        List of RTTSAnnotatedWindow with CAL-EXP metrics and RTTS annotations

    Example:
        from backend.telemetry.rtts_cal_exp_window_join import join_rtts_to_cal_exp_windows
        from backend.telemetry.rtts_window_validator import rtts_validate_window

        # Get RTTS validation
        rtts_block = rtts_validate_window(snapshots)

        # Join to CAL-EXP windows
        annotated = join_rtts_to_cal_exp_windows(
            windows=accumulator.get_all_windows(),
            rtts_block=rtts_block,
        )

        for w in annotated:
            print(f"Window {w.window_index}: mock_flags={w.mock_flags_count}")
    """
    annotated_windows: List[RTTSAnnotatedWindow] = []

    for window in windows:
        annotated = annotate_window_with_rtts(window, rtts_block)
        annotated_windows.append(annotated)

    return annotated_windows


def join_rtts_dict_to_cal_exp_windows(
    windows: List["MetricsWindow"],
    rtts_dict: Optional[Dict[str, Any]] = None,
) -> List[RTTSAnnotatedWindow]:
    """
    Join RTTS validation dict (from rtts_validation.json) to CAL-EXP windows.

    This is a convenience function that works with the JSON-serialized form
    of RTTS validation results.

    SHADOW MODE CONTRACT:
    - Non-mutating (creates new list of RTTSAnnotatedWindow)
    - Missing RTTS is handled gracefully
    - Deterministic ordering

    Args:
        windows: List of MetricsWindow from CAL-EXP run
        rtts_dict: Optional dict from rtts_validation.json

    Returns:
        List of RTTSAnnotatedWindow with CAL-EXP metrics and RTTS annotations
    """
    annotated_windows: List[RTTSAnnotatedWindow] = []

    for window in windows:
        annotated = RTTSAnnotatedWindow(
            window_index=window.window_index,
            start_cycle=window.start_cycle,
            end_cycle=window.end_cycle,
            success_rate=window.success_rate,
            omega_occupancy=window.omega_occupancy,
            mean_rsi=window.mean_rsi,
            block_rate=window.block_rate,
        )

        if rtts_dict is not None:
            annotated.rtts_available = True
            annotated.rtts_overall_status = rtts_dict.get("overall_status", "UNKNOWN")
            annotated.rtts_warning_count = rtts_dict.get("warning_count", 0)

            # Extract mock flags from standardized format
            mock_flags = rtts_dict.get("mock_detection_flags", [])
            annotated.mock_flags = mock_flags
            annotated.mock_flags_count = len(mock_flags)

            # Extract continuity violation rate
            continuity = rtts_dict.get("continuity")
            if continuity and isinstance(continuity, dict):
                annotated.continuity_violation_rate = continuity.get("violation_rate", 0.0)

        annotated_windows.append(annotated)

    return annotated_windows
