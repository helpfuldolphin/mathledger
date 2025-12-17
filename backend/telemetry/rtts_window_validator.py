"""
RTTS Window Validator Orchestrator

Phase X P5.2: VALIDATE Window Validation (NO ENFORCEMENT)

This module provides the rtts_validate_window() orchestrator that coordinates
all RTTS validation components to produce a schema-versioned validation block
for use by the TelemetryGovernanceSignalEmitter.

SHADOW MODE CONTRACT:
- All validation is OBSERVATIONAL ONLY
- Results are surfaced as WARNINGS in governance signals
- No blocking, no enforcement
- All fields remain optional

RTTS Gap Closure: P5.2 VALIDATE stage
See: docs/system_law/RTTS_Gap_Closure_Blueprint.md

Status: P5.2 VALIDATE (NO ENFORCEMENT)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from backend.telemetry.rtts_statistical_validator import (
    RTTSStatisticalValidator,
    RTTSStatisticalResult,
)
from backend.telemetry.rtts_mock_detector import (
    RTTSMockDetector,
    MockDetectionResult,
)
from backend.telemetry.rtts_continuity_tracker import (
    RTTSContinuityTracker,
    ContinuityStats,
)
from backend.telemetry.rtts_correlation_tracker import (
    RTTSCorrelationTracker,
)
from backend.telemetry.governance_signal import (
    RTTSCorrelationResult,
    MockIndicatorSummary,
)

if TYPE_CHECKING:
    from backend.topology.first_light.data_structures_p4 import TelemetrySnapshot

__all__ = [
    "RTTSValidationBlock",
    "rtts_validate_window",
    "RTTSWindowValidator",
]


# Schema version for validation blocks
RTTS_VALIDATION_SCHEMA_VERSION = "1.0.0"


@dataclass
class RTTSValidationBlock:
    """
    Schema-versioned RTTS validation result block.

    This block aggregates results from all RTTS validation components
    into a single structure for use by TelemetryGovernanceSignalEmitter.

    SHADOW MODE: All results are observational only.
    """

    # Schema metadata
    schema_version: str = RTTS_VALIDATION_SCHEMA_VERSION
    block_id: str = ""
    timestamp: str = ""

    # Component results
    statistical: Optional[RTTSStatisticalResult] = None
    correlation: Optional[RTTSCorrelationResult] = None
    continuity: Optional[ContinuityStats] = None
    mock_detection: Optional[MockDetectionResult] = None

    # Aggregate status
    overall_status: str = "UNKNOWN"  # OK | ATTENTION | WARN | CRITICAL
    validation_passed: bool = False

    # Aggregate warnings (collected from all components)
    all_warnings: List[str] = field(default_factory=list)
    warning_count: int = 0

    # Window metadata
    window_size: int = 0
    window_start_cycle: int = 0
    window_end_cycle: int = 0

    # SHADOW MODE
    mode: str = "SHADOW"
    action: str = "LOGGED_ONLY"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "schema_version": self.schema_version,
            "block_id": self.block_id,
            "timestamp": self.timestamp,
            "statistical": self.statistical.to_dict() if self.statistical else None,
            "correlation": self.correlation.to_dict() if self.correlation else None,
            "continuity": self.continuity.to_dict() if self.continuity else None,
            "mock_detection": self.mock_detection.to_dict() if self.mock_detection else None,
            "overall_status": self.overall_status,
            "validation_passed": self.validation_passed,
            "all_warnings": self.all_warnings,
            "warning_count": self.warning_count,
            "window": {
                "size": self.window_size,
                "start_cycle": self.window_start_cycle,
                "end_cycle": self.window_end_cycle,
            },
            "mode": self.mode,
            "action": self.action,
        }

    @property
    def has_warnings(self) -> bool:
        """Check if any warnings were generated."""
        return self.warning_count > 0


class RTTSWindowValidator:
    """
    RTTS window validation orchestrator.

    Coordinates all RTTS validation components to produce a unified
    validation block for each telemetry window.

    P5.2: Computes real statistics and validates against RTTS thresholds,
    but all results are LOGGED_ONLY (no enforcement).

    Usage:
        validator = RTTSWindowValidator(window_size=200)

        # Feed snapshots
        for snapshot in telemetry_stream:
            validator.update(snapshot)

        # Get validation block when ready
        if validator.is_window_ready():
            block = validator.validate()
    """

    def __init__(self, window_size: int = 200):
        """
        Initialize RTTS window validator.

        Args:
            window_size: Number of cycles for validation window
        """
        self.window_size = window_size

        # Initialize component validators
        self._stat_validator = RTTSStatisticalValidator(window_size=window_size)
        self._correlation_tracker = RTTSCorrelationTracker(window_size=window_size)
        self._continuity_tracker = RTTSContinuityTracker(history_size=window_size)
        self._mock_detector = RTTSMockDetector()

        # Track snapshots for window metadata
        self._snapshots: List["TelemetrySnapshot"] = []

    def update(self, snapshot: "TelemetrySnapshot") -> None:
        """
        Update all validators with new snapshot.

        Args:
            snapshot: TelemetrySnapshot to add to window
        """
        # Update all component validators
        self._stat_validator.update(snapshot)
        self._correlation_tracker.update(snapshot)
        continuity_check = self._continuity_tracker.check(snapshot)

        # Track delta_H for mock detection (MOCK-009)
        if continuity_check is not None:
            self._mock_detector.record_delta_H(continuity_check.delta_H)

        # Track rho value for mock detection (MOCK-010)
        self._mock_detector.record_rho_value(snapshot.rho)

        # Store snapshot reference
        self._snapshots.append(snapshot)
        if len(self._snapshots) > self.window_size:
            self._snapshots.pop(0)

    def is_window_ready(self) -> bool:
        """Check if window has enough data for validation."""
        return self._stat_validator.is_window_full()

    def validate(self) -> RTTSValidationBlock:
        """
        Run full RTTS validation and produce validation block.

        Returns:
            RTTSValidationBlock with all component results
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        block_id = str(uuid.uuid4())

        # Get statistical results
        stat_result = self._stat_validator.get_current_result()

        # Get correlation results
        correlation_result = self._correlation_tracker.compute()

        # Get continuity stats
        continuity_stats = self._continuity_tracker.get_stats()

        # Run mock detection
        mock_result = self._mock_detector.detect(
            stats=stat_result,
            correlations=correlation_result,
        )

        # Collect all warnings
        all_warnings: List[str] = []
        if stat_result.warnings:
            all_warnings.extend(stat_result.warnings)
        if correlation_result.warnings:
            all_warnings.extend(correlation_result.warnings)
        if continuity_stats.warnings:
            all_warnings.extend(continuity_stats.warnings)
        if mock_result.violations:
            all_warnings.extend(mock_result.violations)

        # Determine overall status
        overall_status = self._compute_overall_status(
            stat_result, correlation_result, continuity_stats, mock_result
        )

        # Determine if validation passed (no critical issues)
        validation_passed = (
            overall_status in ("OK", "ATTENTION")
            and mock_result.status != "SUSPECTED_MOCK"
        )

        # Get window metadata
        window_start = self._snapshots[0].cycle if self._snapshots else 0
        window_end = self._snapshots[-1].cycle if self._snapshots else 0

        return RTTSValidationBlock(
            schema_version=RTTS_VALIDATION_SCHEMA_VERSION,
            block_id=block_id,
            timestamp=timestamp,
            statistical=stat_result,
            correlation=correlation_result,
            continuity=continuity_stats,
            mock_detection=mock_result,
            overall_status=overall_status,
            validation_passed=validation_passed,
            all_warnings=all_warnings,
            warning_count=len(all_warnings),
            window_size=len(self._snapshots),
            window_start_cycle=window_start,
            window_end_cycle=window_end,
        )

    def _compute_overall_status(
        self,
        stat_result: RTTSStatisticalResult,
        correlation_result: RTTSCorrelationResult,
        continuity_stats: ContinuityStats,
        mock_result: MockDetectionResult,
    ) -> str:
        """
        Compute overall validation status.

        Status levels:
        - OK: No issues detected
        - ATTENTION: Minor issues (low severity violations)
        - WARN: Moderate issues (threshold violations)
        - CRITICAL: Severe issues (suspected mock, high violation rate)
        """
        # CRITICAL: Suspected mock telemetry
        if mock_result.status == "SUSPECTED_MOCK":
            return "CRITICAL"

        # CRITICAL: High violation rate
        if continuity_stats.violation_rate > 0.2:
            return "CRITICAL"

        # WARN: Multiple threshold violations
        stat_violations = stat_result.violation_count if stat_result.computed else 0
        cor_violations = correlation_result.violation_count
        total_violations = stat_violations + cor_violations

        if total_violations >= 3:
            return "WARN"

        # WARN: Consecutive continuity violations
        if continuity_stats.consecutive_violations >= 3:
            return "WARN"

        # ATTENTION: Any violations
        if total_violations > 0 or continuity_stats.violation_count > 0:
            return "ATTENTION"

        # OK: No issues
        return "OK"

    def reset(self) -> None:
        """Reset all validators."""
        self._stat_validator.reset()
        self._correlation_tracker.reset()
        self._continuity_tracker.reset()
        self._mock_detector.reset()
        self._snapshots.clear()

    def get_warning_count(self) -> int:
        """Get total warning count from all components."""
        count = 0
        count += len(self._stat_validator.get_current_result().warnings)
        count += len(self._correlation_tracker.compute().warnings)
        count += len(self._continuity_tracker.get_stats().warnings)
        return count


def rtts_validate_window(
    snapshots: List["TelemetrySnapshot"],
    window_size: Optional[int] = None,
) -> RTTSValidationBlock:
    """
    Validate a window of telemetry snapshots against RTTS thresholds.

    This is the main orchestrator function for P5.2 VALIDATE stage.
    It processes a batch of snapshots and returns a schema-versioned
    validation block with all RTTS component results.

    SHADOW MODE: All results are LOGGED_ONLY, no enforcement.

    Args:
        snapshots: List of TelemetrySnapshot to validate
        window_size: Optional window size override (defaults to len(snapshots))

    Returns:
        RTTSValidationBlock with validation results and warnings

    Example:
        snapshots = [provider.get_snapshot() for _ in range(200)]
        block = rtts_validate_window(snapshots)
        if block.has_warnings:
            for warning in block.all_warnings:
                log.warning(warning)
    """
    if not snapshots:
        # Return empty block for empty input
        return RTTSValidationBlock(
            block_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            overall_status="UNKNOWN",
            validation_passed=False,
            all_warnings=["No snapshots provided for validation"],
            warning_count=1,
        )

    # Use provided window_size or default to snapshot count
    effective_window_size = window_size if window_size else len(snapshots)

    # Create validator with appropriate window size
    validator = RTTSWindowValidator(window_size=effective_window_size)

    # Feed all snapshots
    for snapshot in snapshots:
        validator.update(snapshot)

    # Run validation
    return validator.validate()
