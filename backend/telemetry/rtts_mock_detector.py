"""
RTTS Mock Telemetry Detector

Phase X P5.1: LOG-ONLY Mock Detection

This module implements MOCK-001 through MOCK-010 criteria from
Real_Telemetry_Topology_Spec.md Section 2.1 for detecting mock telemetry.

SHADOW MODE CONTRACT:
- Detection is OBSERVATIONAL ONLY
- Results are logged, not enforced
- No modification of telemetry flow

RTTS-GAP-002: Mock Detection Status
See: docs/system_law/RTTS_Gap_Closure_Blueprint.md
See: docs/system_law/Real_Telemetry_Topology_Spec.md Section 2.1

Status: P5.1 LOG-ONLY PLACEHOLDER
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from backend.telemetry.rtts_statistical_validator import RTTSStatisticalResult
    from backend.telemetry.governance_signal import RTTSCorrelationResult, MockIndicatorSummary

__all__ = [
    "RTTSMockDetector",
    "MockDetectionResult",
]


@dataclass
class MockDetectionResult:
    """
    Result of RTTS mock detection analysis.

    SHADOW MODE: Results are for logging only.
    """

    # Overall status
    status: str = "UNKNOWN"  # VALIDATED_REAL | SUSPECTED_MOCK | UNKNOWN
    confidence: float = 0.0  # [0.0, 1.0]

    # Indicator flags (MOCK-001 through MOCK-010)
    mock_001_var_H_low: bool = False
    mock_002_var_rho_low: bool = False
    mock_003_cor_low: bool = False
    mock_004_cor_high: bool = False
    mock_005_acf_low: bool = False
    mock_006_acf_high: bool = False
    mock_007_kurtosis_low: bool = False
    mock_008_kurtosis_high: bool = False
    mock_009_jump_H: bool = False
    mock_010_discrete_rho: bool = False

    # Severity counts
    high_severity_count: int = 0
    medium_severity_count: int = 0
    low_severity_count: int = 0

    # Validation
    rtts_validation_passed: bool = False
    violations: List[str] = None

    # SHADOW MODE
    mode: str = "SHADOW"

    def __post_init__(self):
        if self.violations is None:
            self.violations = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status,
            "confidence": round(self.confidence, 4),
            "indicators": {
                "MOCK_001": self.mock_001_var_H_low,
                "MOCK_002": self.mock_002_var_rho_low,
                "MOCK_003": self.mock_003_cor_low,
                "MOCK_004": self.mock_004_cor_high,
                "MOCK_005": self.mock_005_acf_low,
                "MOCK_006": self.mock_006_acf_high,
                "MOCK_007": self.mock_007_kurtosis_low,
                "MOCK_008": self.mock_008_kurtosis_high,
                "MOCK_009": self.mock_009_jump_H,
                "MOCK_010": self.mock_010_discrete_rho,
            },
            "severity_counts": {
                "high": self.high_severity_count,
                "medium": self.medium_severity_count,
                "low": self.low_severity_count,
            },
            "validation": {
                "passed": self.rtts_validation_passed,
                "violations": self.violations,
            },
            "mode": self.mode,
        }


class RTTSMockDetector:
    """
    RTTS mock telemetry detector.

    Implements MOCK-001 through MOCK-010 criteria from
    Real_Telemetry_Topology_Spec.md Section 2.1.

    SHADOW MODE: Detection is OBSERVATIONAL ONLY.
    Results are logged, not enforced.

    # REAL-READY: Hook point for production mock detection
    """

    # RTTS threshold constants
    VAR_H_THRESHOLD = 0.0001      # MOCK-001
    VAR_RHO_THRESHOLD = 0.00005   # MOCK-002
    COR_LOW_THRESHOLD = 0.1       # MOCK-003
    COR_HIGH_THRESHOLD = 0.99     # MOCK-004
    ACF_LOW_THRESHOLD = 0.05      # MOCK-005
    ACF_HIGH_THRESHOLD = 0.95     # MOCK-006
    KURTOSIS_LOW_THRESHOLD = -1.0 # MOCK-007
    KURTOSIS_HIGH_THRESHOLD = 5.0 # MOCK-008
    DELTA_H_MAX = 0.15            # MOCK-009
    UNIQUE_RHO_MIN = 10           # MOCK-010

    def __init__(self):
        """Initialize mock detector."""
        self._max_delta_H: float = 0.0
        self._unique_rho_values: set = set()
        self._cycle_count: int = 0

    # REAL-READY: Call from TelemetryGovernanceSignalEmitter.emit_signal()
    def detect(
        self,
        stats: Optional["RTTSStatisticalResult"] = None,
        correlations: Optional["RTTSCorrelationResult"] = None,
    ) -> MockDetectionResult:
        """
        Run all MOCK-001 through MOCK-010 checks.

        P5.1 LOG-ONLY: Placeholder implementation that checks
        available data and returns detection result.

        Args:
            stats: Statistical validation result (variance, autocorr, kurtosis)
            correlations: Cross-correlation result

        Returns:
            MockDetectionResult with detection flags and status
        """
        result = MockDetectionResult()
        violations = []

        # Check MOCK-001: Var(H) too low
        if stats and stats.variance_H is not None:
            if stats.variance_H < self.VAR_H_THRESHOLD:
                result.mock_001_var_H_low = True
                violations.append("MOCK-001: Var(H) below threshold")

        # Check MOCK-002: Var(ρ) too low
        if stats and stats.variance_rho is not None:
            if stats.variance_rho < self.VAR_RHO_THRESHOLD:
                result.mock_002_var_rho_low = True
                violations.append("MOCK-002: Var(rho) below threshold")

        # Check MOCK-003: |Cor(H, ρ)| too low
        if correlations and correlations.cor_H_rho is not None:
            if abs(correlations.cor_H_rho) < self.COR_LOW_THRESHOLD:
                result.mock_003_cor_low = True
                violations.append("MOCK-003: |Cor(H, rho)| below threshold")

        # Check MOCK-004: |Cor(H, ρ)| too high
        if correlations and correlations.cor_H_rho is not None:
            if abs(correlations.cor_H_rho) > self.COR_HIGH_THRESHOLD:
                result.mock_004_cor_high = True
                violations.append("MOCK-004: |Cor(H, rho)| above threshold")

        # Check MOCK-005: Autocorrelation too low
        if stats and stats.autocorr_H_lag1 is not None:
            if stats.autocorr_H_lag1 < self.ACF_LOW_THRESHOLD:
                result.mock_005_acf_low = True
                violations.append("MOCK-005: ACF(H, lag=1) below threshold")

        # Check MOCK-006: Autocorrelation too high
        if stats and stats.autocorr_H_lag1 is not None:
            if stats.autocorr_H_lag1 > self.ACF_HIGH_THRESHOLD:
                result.mock_006_acf_high = True
                violations.append("MOCK-006: ACF(H, lag=1) above threshold")

        # Check MOCK-007: Kurtosis too low
        if stats and stats.kurtosis_H is not None:
            if stats.kurtosis_H < self.KURTOSIS_LOW_THRESHOLD:
                result.mock_007_kurtosis_low = True
                violations.append("MOCK-007: Kurtosis(H) below threshold")

        # Check MOCK-008: Kurtosis too high
        if stats and stats.kurtosis_H is not None:
            if stats.kurtosis_H > self.KURTOSIS_HIGH_THRESHOLD:
                result.mock_008_kurtosis_high = True
                violations.append("MOCK-008: Kurtosis(H) above threshold")

        # Check MOCK-009: Jump in H (tracked separately)
        if self._max_delta_H > self.DELTA_H_MAX:
            result.mock_009_jump_H = True
            violations.append("MOCK-009: max(|ΔH|) exceeds threshold")

        # Check MOCK-010: Discrete ρ values (tracked separately)
        if self._cycle_count >= 100 and len(self._unique_rho_values) < self.UNIQUE_RHO_MIN:
            result.mock_010_discrete_rho = True
            violations.append("MOCK-010: Too few unique rho values")

        # Compute severity counts
        result.high_severity_count = sum([
            result.mock_001_var_H_low,
            result.mock_002_var_rho_low,
            result.mock_009_jump_H,
            result.mock_010_discrete_rho,
        ])
        result.medium_severity_count = sum([
            result.mock_003_cor_low,
            result.mock_004_cor_high,
            result.mock_005_acf_low,
            result.mock_006_acf_high,
        ])
        result.low_severity_count = sum([
            result.mock_007_kurtosis_low,
            result.mock_008_kurtosis_high,
        ])

        # Store violations
        result.violations = violations

        # Compute status and confidence
        status, confidence = self._compute_status(result)
        result.status = status
        result.confidence = confidence
        result.rtts_validation_passed = (status == "VALIDATED_REAL")

        return result

    # REAL-READY: Compute overall mock detection status
    def _compute_status(self, result: MockDetectionResult) -> Tuple[str, float]:
        """
        Compute mock_detection_status and confidence.

        Args:
            result: MockDetectionResult with indicator flags

        Returns:
            Tuple of (status, confidence)
        """
        # Any high severity indicator → SUSPECTED_MOCK
        if result.high_severity_count > 0:
            confidence = 0.9 - (0.1 * result.medium_severity_count)
            return "SUSPECTED_MOCK", max(0.5, confidence)

        # Multiple medium severity indicators → SUSPECTED_MOCK
        if result.medium_severity_count >= 2:
            confidence = 0.7 + (0.05 * result.medium_severity_count)
            return "SUSPECTED_MOCK", min(0.9, confidence)

        # Single medium or low severity → UNKNOWN (need more data)
        if result.medium_severity_count > 0 or result.low_severity_count > 0:
            confidence = 0.5 - (0.1 * (result.medium_severity_count + result.low_severity_count))
            return "UNKNOWN", max(0.3, confidence)

        # No indicators triggered → VALIDATED_REAL
        # Confidence depends on how much data we have
        base_confidence = 0.8
        if self._cycle_count >= 200:
            base_confidence = 0.95
        elif self._cycle_count >= 100:
            base_confidence = 0.85

        return "VALIDATED_REAL", base_confidence

    def record_delta_H(self, delta_H: float) -> None:
        """
        Record H delta for MOCK-009 tracking.

        Args:
            delta_H: Absolute change in H from previous cycle
        """
        if delta_H > self._max_delta_H:
            self._max_delta_H = delta_H

    def record_rho_value(self, rho: float) -> None:
        """
        Record ρ value for MOCK-010 tracking.

        Args:
            rho: Current ρ value
        """
        # Round to 4 decimal places for uniqueness check
        self._unique_rho_values.add(round(rho, 4))
        self._cycle_count += 1

    def to_mock_indicator_summary(self, result: MockDetectionResult) -> "MockIndicatorSummary":
        """
        Convert detection result to MockIndicatorSummary.

        Args:
            result: MockDetectionResult

        Returns:
            MockIndicatorSummary dataclass
        """
        from backend.telemetry.governance_signal import MockIndicatorSummary

        summary = MockIndicatorSummary(
            mock_001_var_H_low=result.mock_001_var_H_low,
            mock_002_var_rho_low=result.mock_002_var_rho_low,
            mock_003_cor_low=result.mock_003_cor_low,
            mock_004_cor_high=result.mock_004_cor_high,
            mock_005_acf_low=result.mock_005_acf_low,
            mock_006_acf_high=result.mock_006_acf_high,
            mock_007_kurtosis_low=result.mock_007_kurtosis_low,
            mock_008_kurtosis_high=result.mock_008_kurtosis_high,
            mock_009_jump_H=result.mock_009_jump_H,
            mock_010_discrete_rho=result.mock_010_discrete_rho,
            high_severity_count=result.high_severity_count,
            medium_severity_count=result.medium_severity_count,
            low_severity_count=result.low_severity_count,
        )
        return summary

    def reset(self) -> None:
        """Reset detector state."""
        self._max_delta_H = 0.0
        self._unique_rho_values.clear()
        self._cycle_count = 0

    def get_cycle_count(self) -> int:
        """Get number of cycles recorded."""
        return self._cycle_count
