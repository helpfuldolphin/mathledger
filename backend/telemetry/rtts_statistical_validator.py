"""
RTTS Statistical Validator

Phase X P5.2: VALIDATE Statistical Validation (NO ENFORCEMENT)

This module provides statistical validation for RTTS mock detection.
Implements variance, autocorrelation, and kurtosis computation with
real threshold validation and warning generation.

SHADOW MODE CONTRACT:
- All computations are OBSERVATIONAL ONLY
- Threshold violations generate WARNINGS, not enforcement
- Results are logged, not enforced
- No modification of telemetry flow

RTTS-GAP-001: Statistical Validation Fields
See: docs/system_law/RTTS_Gap_Closure_Blueprint.md
See: docs/system_law/Real_Telemetry_Topology_Spec.md Section 1.3

Status: P5.2 VALIDATE (NO ENFORCEMENT)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from backend.topology.first_light.data_structures_p4 import TelemetrySnapshot

__all__ = [
    "RTTSStatisticalValidator",
    "RTTSStatisticalResult",
]


@dataclass
class RTTSStatisticalResult:
    """
    Result of RTTS statistical validation.

    Contains variance, autocorrelation, and kurtosis metrics
    computed over a rolling window, with threshold validation.

    SHADOW MODE: Results are for logging only.
    P5.2: Includes threshold violations and warnings.
    """

    # Variance metrics
    variance_H: Optional[float] = None
    variance_rho: Optional[float] = None
    variance_tau: Optional[float] = None
    variance_beta: Optional[float] = None

    # Autocorrelation metrics (lag-1)
    autocorr_H_lag1: Optional[float] = None
    autocorr_rho_lag1: Optional[float] = None

    # Kurtosis metrics
    kurtosis_H: Optional[float] = None
    kurtosis_rho: Optional[float] = None

    # Window metadata
    window_size: int = 0
    window_start_cycle: int = 0
    window_end_cycle: int = 0

    # Computation status
    computed: bool = False
    insufficient_data: bool = True

    # P5.2: Threshold validation
    var_H_below_threshold: bool = False
    var_rho_below_threshold: bool = False
    acf_below_threshold: bool = False
    acf_above_threshold: bool = False
    kurtosis_below_threshold: bool = False
    kurtosis_above_threshold: bool = False

    # P5.2: Warnings (LOGGED_ONLY)
    warnings: List[str] = field(default_factory=list)

    # SHADOW MODE
    mode: str = "SHADOW"
    action: str = "LOGGED_ONLY"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "variance": {
                "H": round(self.variance_H, 8) if self.variance_H is not None else None,
                "rho": round(self.variance_rho, 8) if self.variance_rho is not None else None,
                "tau": round(self.variance_tau, 8) if self.variance_tau is not None else None,
                "beta": round(self.variance_beta, 8) if self.variance_beta is not None else None,
            },
            "autocorrelation": {
                "H_lag1": round(self.autocorr_H_lag1, 6) if self.autocorr_H_lag1 is not None else None,
                "rho_lag1": round(self.autocorr_rho_lag1, 6) if self.autocorr_rho_lag1 is not None else None,
            },
            "kurtosis": {
                "H": round(self.kurtosis_H, 6) if self.kurtosis_H is not None else None,
                "rho": round(self.kurtosis_rho, 6) if self.kurtosis_rho is not None else None,
            },
            "window": {
                "size": self.window_size,
                "start_cycle": self.window_start_cycle,
                "end_cycle": self.window_end_cycle,
            },
            "status": {
                "computed": self.computed,
                "insufficient_data": self.insufficient_data,
            },
            "threshold_violations": {
                "var_H_below": self.var_H_below_threshold,
                "var_rho_below": self.var_rho_below_threshold,
                "acf_below": self.acf_below_threshold,
                "acf_above": self.acf_above_threshold,
                "kurtosis_below": self.kurtosis_below_threshold,
                "kurtosis_above": self.kurtosis_above_threshold,
            },
            "warnings": self.warnings,
            "mode": self.mode,
            "action": self.action,
        }

    @property
    def has_violations(self) -> bool:
        """Check if any threshold violations detected."""
        return any([
            self.var_H_below_threshold,
            self.var_rho_below_threshold,
            self.acf_below_threshold,
            self.acf_above_threshold,
            self.kurtosis_below_threshold,
            self.kurtosis_above_threshold,
        ])

    @property
    def violation_count(self) -> int:
        """Count number of threshold violations."""
        return sum([
            self.var_H_below_threshold,
            self.var_rho_below_threshold,
            self.acf_below_threshold,
            self.acf_above_threshold,
            self.kurtosis_below_threshold,
            self.kurtosis_above_threshold,
        ])


class RTTSStatisticalValidator:
    """
    RTTS statistical validation for mock detection.

    SHADOW MODE: Computes statistics for observation only.
    Does not modify telemetry or governance.

    # REAL-READY: Hook point for production telemetry validation

    Usage:
        validator = RTTSStatisticalValidator(window_size=200)
        for snapshot in telemetry_stream:
            result = validator.update(snapshot)
            if result.computed:
                # Use result for mock detection
                pass
    """

    # RTTS threshold constants for mock detection
    VAR_H_THRESHOLD = 0.0001      # MOCK-001: Var(H) < threshold → suspected mock
    VAR_RHO_THRESHOLD = 0.00005   # MOCK-002: Var(ρ) < threshold → suspected mock
    ACF_LOW_THRESHOLD = 0.05      # MOCK-005: autocorr < threshold → suspected mock
    ACF_HIGH_THRESHOLD = 0.95     # MOCK-006: autocorr > threshold → suspected mock
    KURTOSIS_LOW_THRESHOLD = -1.0 # MOCK-007: kurtosis < threshold → suspected mock
    KURTOSIS_HIGH_THRESHOLD = 5.0 # MOCK-008: kurtosis > threshold → suspected mock

    def __init__(self, window_size: int = 200):
        """
        Initialize statistical validator.

        Args:
            window_size: Number of cycles for rolling statistics
        """
        self.window_size = window_size
        self._history: List["TelemetrySnapshot"] = []
        self._H_values: List[float] = []
        self._rho_values: List[float] = []
        self._tau_values: List[float] = []
        self._beta_values: List[float] = []

    # REAL-READY: Call from TelemetryProviderInterface.get_snapshot()
    def update(self, snapshot: "TelemetrySnapshot") -> RTTSStatisticalResult:
        """
        Update rolling statistics with new snapshot.

        Args:
            snapshot: TelemetrySnapshot to add to window

        Returns:
            RTTSStatisticalResult with computed metrics (if window full)
        """
        # Add to history
        self._history.append(snapshot)
        self._H_values.append(snapshot.H)
        self._rho_values.append(snapshot.rho)
        self._tau_values.append(snapshot.tau)
        self._beta_values.append(snapshot.beta)

        # Trim to window size
        if len(self._history) > self.window_size:
            self._history.pop(0)
            self._H_values.pop(0)
            self._rho_values.pop(0)
            self._tau_values.pop(0)
            self._beta_values.pop(0)

        # Check if we have enough data
        if len(self._history) < self.window_size:
            return RTTSStatisticalResult(
                window_size=len(self._history),
                window_start_cycle=self._history[0].cycle if self._history else 0,
                window_end_cycle=snapshot.cycle,
                computed=False,
                insufficient_data=True,
            )

        # Compute statistics
        return self._compute_statistics()

    def _compute_statistics(self) -> RTTSStatisticalResult:
        """
        Compute all statistical metrics over current window.

        P5.2: Validates against RTTS thresholds and generates warnings.
        """
        start_cycle = self._history[0].cycle
        end_cycle = self._history[-1].cycle

        # Compute metrics
        variance_H = self._compute_variance(self._H_values)
        variance_rho = self._compute_variance(self._rho_values)
        variance_tau = self._compute_variance(self._tau_values)
        variance_beta = self._compute_variance(self._beta_values)
        autocorr_H_lag1 = self._compute_autocorrelation(self._H_values, lag=1)
        autocorr_rho_lag1 = self._compute_autocorrelation(self._rho_values, lag=1)
        kurtosis_H = self._compute_kurtosis(self._H_values)
        kurtosis_rho = self._compute_kurtosis(self._rho_values)

        # P5.2: Validate against thresholds
        warnings: List[str] = []

        var_H_below = False
        if variance_H is not None and variance_H < self.VAR_H_THRESHOLD:
            var_H_below = True
            warnings.append(
                f"MOCK-001: Var(H)={variance_H:.6f} below threshold {self.VAR_H_THRESHOLD}"
            )

        var_rho_below = False
        if variance_rho is not None and variance_rho < self.VAR_RHO_THRESHOLD:
            var_rho_below = True
            warnings.append(
                f"MOCK-002: Var(rho)={variance_rho:.6f} below threshold {self.VAR_RHO_THRESHOLD}"
            )

        acf_below = False
        if autocorr_H_lag1 is not None and autocorr_H_lag1 < self.ACF_LOW_THRESHOLD:
            acf_below = True
            warnings.append(
                f"MOCK-005: ACF(H,lag=1)={autocorr_H_lag1:.4f} below threshold {self.ACF_LOW_THRESHOLD}"
            )

        acf_above = False
        if autocorr_H_lag1 is not None and autocorr_H_lag1 > self.ACF_HIGH_THRESHOLD:
            acf_above = True
            warnings.append(
                f"MOCK-006: ACF(H,lag=1)={autocorr_H_lag1:.4f} above threshold {self.ACF_HIGH_THRESHOLD}"
            )

        kurtosis_below = False
        if kurtosis_H is not None and kurtosis_H < self.KURTOSIS_LOW_THRESHOLD:
            kurtosis_below = True
            warnings.append(
                f"MOCK-007: Kurtosis(H)={kurtosis_H:.4f} below threshold {self.KURTOSIS_LOW_THRESHOLD}"
            )

        kurtosis_above = False
        if kurtosis_H is not None and kurtosis_H > self.KURTOSIS_HIGH_THRESHOLD:
            kurtosis_above = True
            warnings.append(
                f"MOCK-008: Kurtosis(H)={kurtosis_H:.4f} above threshold {self.KURTOSIS_HIGH_THRESHOLD}"
            )

        return RTTSStatisticalResult(
            variance_H=variance_H,
            variance_rho=variance_rho,
            variance_tau=variance_tau,
            variance_beta=variance_beta,
            autocorr_H_lag1=autocorr_H_lag1,
            autocorr_rho_lag1=autocorr_rho_lag1,
            kurtosis_H=kurtosis_H,
            kurtosis_rho=kurtosis_rho,
            window_size=len(self._history),
            window_start_cycle=start_cycle,
            window_end_cycle=end_cycle,
            computed=True,
            insufficient_data=False,
            var_H_below_threshold=var_H_below,
            var_rho_below_threshold=var_rho_below,
            acf_below_threshold=acf_below,
            acf_above_threshold=acf_above,
            kurtosis_below_threshold=kurtosis_below,
            kurtosis_above_threshold=kurtosis_above,
            warnings=warnings,
        )

    # REAL-READY: Variance computation
    def _compute_variance(self, values: List[float]) -> Optional[float]:
        """
        Compute variance for a list of values.

        P5.1 LOG-ONLY: Placeholder implementation.

        Args:
            values: List of float values

        Returns:
            Variance or None if insufficient data
        """
        if len(values) < 2:
            return None

        n = len(values)
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)  # Sample variance
        return variance

    # REAL-READY: Autocorrelation computation
    def _compute_autocorrelation(self, values: List[float], lag: int = 1) -> Optional[float]:
        """
        Compute lag-k autocorrelation for a list of values.

        P5.1 LOG-ONLY: Placeholder implementation.

        Args:
            values: List of float values
            lag: Lag for autocorrelation (default: 1)

        Returns:
            Autocorrelation coefficient or None if insufficient data
        """
        if len(values) < lag + 2:
            return None

        n = len(values)
        mean = sum(values) / n

        # Compute autocovariance at lag k
        autocovariance = sum(
            (values[i] - mean) * (values[i - lag] - mean)
            for i in range(lag, n)
        ) / (n - lag)

        # Compute variance
        variance = sum((x - mean) ** 2 for x in values) / n

        if variance == 0:
            return 0.0  # No variation → zero autocorrelation

        return autocovariance / variance

    # REAL-READY: Kurtosis computation
    def _compute_kurtosis(self, values: List[float]) -> Optional[float]:
        """
        Compute excess kurtosis for a list of values.

        P5.1 LOG-ONLY: Placeholder implementation.

        Args:
            values: List of float values

        Returns:
            Excess kurtosis or None if insufficient data
        """
        if len(values) < 4:
            return None

        n = len(values)
        mean = sum(values) / n

        # Compute variance
        variance = sum((x - mean) ** 2 for x in values) / n
        if variance == 0:
            return 0.0  # No variation

        std_dev = variance ** 0.5

        # Compute fourth moment
        fourth_moment = sum((x - mean) ** 4 for x in values) / n

        # Kurtosis = fourth_moment / variance^2 - 3 (excess kurtosis)
        kurtosis = (fourth_moment / (variance ** 2)) - 3.0
        return kurtosis

    def get_current_result(self) -> RTTSStatisticalResult:
        """Get current statistical result without adding new data."""
        if len(self._history) < self.window_size:
            return RTTSStatisticalResult(
                window_size=len(self._history),
                window_start_cycle=self._history[0].cycle if self._history else 0,
                window_end_cycle=self._history[-1].cycle if self._history else 0,
                computed=False,
                insufficient_data=True,
            )
        return self._compute_statistics()

    def reset(self) -> None:
        """Reset validator state."""
        self._history.clear()
        self._H_values.clear()
        self._rho_values.clear()
        self._tau_values.clear()
        self._beta_values.clear()

    def get_window_size(self) -> int:
        """Get current window size."""
        return len(self._history)

    def is_window_full(self) -> bool:
        """Check if window has enough data for computation."""
        return len(self._history) >= self.window_size
