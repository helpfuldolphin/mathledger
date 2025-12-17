"""
RTTS Correlation Tracker

Phase X P5.2: VALIDATE Cross-Correlation Tracking (NO ENFORCEMENT)

This module tracks cross-correlations per RTTS Section 1.2.3:
- Cor(H, ρ) ∈ [0.3, 0.9]
- Cor(ρ, ω) ∈ [0.5, 1.0]
- Cor(β, 1-ω) ∈ [0.2, 0.8]

SHADOW MODE CONTRACT:
- All computations are OBSERVATIONAL ONLY
- Threshold violations generate WARNINGS, not enforcement
- Results are logged, not enforced
- No modification of telemetry flow

RTTS-GAP-004: Cross-Correlation Tracking
See: docs/system_law/RTTS_Gap_Closure_Blueprint.md
See: docs/system_law/Real_Telemetry_Topology_Spec.md Section 1.2.3

Status: P5.2 VALIDATE (NO ENFORCEMENT)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from backend.topology.first_light.data_structures_p4 import TelemetrySnapshot
    from backend.telemetry.governance_signal import RTTSCorrelationResult

__all__ = [
    "RTTSCorrelationTracker",
]


class RTTSCorrelationTracker:
    """
    RTTS cross-correlation tracker.

    Implements correlation structure validation from
    Real_Telemetry_Topology_Spec.md Section 1.2.3.

    SHADOW MODE: Correlation analysis is OBSERVATIONAL ONLY.

    # REAL-READY: Hook point for production correlation tracking

    RTTS expected correlation bounds:
    - Cor(H, ρ) ∈ [0.3, 0.9] — Health and stability positively coupled
    - Cor(ρ, ω) ∈ [0.5, 1.0] — High stability implies safe region
    - Cor(β, 1-ω) ∈ [0.2, 0.8] — Blocking correlates with unsafe regions
    """

    # RTTS correlation bounds
    COR_H_RHO_MIN = 0.3
    COR_H_RHO_MAX = 0.9
    COR_RHO_OMEGA_MIN = 0.5
    COR_RHO_OMEGA_MAX = 1.0
    COR_BETA_NOT_OMEGA_MIN = 0.2
    COR_BETA_NOT_OMEGA_MAX = 0.8

    def __init__(self, window_size: int = 200):
        """
        Initialize correlation tracker.

        Args:
            window_size: Number of cycles for correlation window
        """
        self.window_size = window_size
        self._H_history: List[float] = []
        self._rho_history: List[float] = []
        self._omega_history: List[bool] = []
        self._beta_history: List[float] = []
        self._cycle_history: List[int] = []

    # REAL-READY: Call from TelemetryProviderInterface at validation intervals
    def update(self, snapshot: "TelemetrySnapshot") -> None:
        """
        Add snapshot to correlation window.

        Args:
            snapshot: TelemetrySnapshot to add
        """
        self._H_history.append(snapshot.H)
        self._rho_history.append(snapshot.rho)
        self._omega_history.append(snapshot.in_omega)
        self._beta_history.append(snapshot.beta)
        self._cycle_history.append(snapshot.cycle)

        # Trim to window size
        if len(self._H_history) > self.window_size:
            self._H_history.pop(0)
            self._rho_history.pop(0)
            self._omega_history.pop(0)
            self._beta_history.pop(0)
            self._cycle_history.pop(0)

    # REAL-READY: Compute correlations when window is full
    def compute(self) -> "RTTSCorrelationResult":
        """
        Compute cross-correlations over current window.

        P5.2: Validates against RTTS bounds and generates warnings.

        Returns:
            RTTSCorrelationResult with correlation values, violations, and warnings
        """
        from backend.telemetry.governance_signal import RTTSCorrelationResult

        # Check if we have enough data
        if len(self._H_history) < 10:  # Minimum for meaningful correlation
            return RTTSCorrelationResult(
                window_size=len(self._H_history),
                window_start_cycle=self._cycle_history[0] if self._cycle_history else 0,
            )

        # Compute Cor(H, ρ)
        cor_H_rho = self._pearson_correlation(self._H_history, self._rho_history)

        # Compute Cor(ρ, ω) - point-biserial for continuous vs binary
        cor_rho_omega = self._point_biserial(self._rho_history, self._omega_history)

        # Compute Cor(β, 1-ω)
        not_omega = [not w for w in self._omega_history]
        cor_beta_not_omega = self._point_biserial(self._beta_history, not_omega)

        # P5.2: Check violations and generate warnings
        warnings: List[str] = []

        cor_H_rho_violated = False
        if cor_H_rho is not None:
            if cor_H_rho < self.COR_H_RHO_MIN:
                cor_H_rho_violated = True
                warnings.append(
                    f"Cor(H,rho)={cor_H_rho:.4f} below min bound {self.COR_H_RHO_MIN}"
                )
            elif cor_H_rho > self.COR_H_RHO_MAX:
                cor_H_rho_violated = True
                warnings.append(
                    f"Cor(H,rho)={cor_H_rho:.4f} above max bound {self.COR_H_RHO_MAX}"
                )

        cor_rho_omega_violated = False
        if cor_rho_omega is not None:
            if cor_rho_omega < self.COR_RHO_OMEGA_MIN:
                cor_rho_omega_violated = True
                warnings.append(
                    f"Cor(rho,omega)={cor_rho_omega:.4f} below min bound {self.COR_RHO_OMEGA_MIN}"
                )
            elif cor_rho_omega > self.COR_RHO_OMEGA_MAX:
                cor_rho_omega_violated = True
                warnings.append(
                    f"Cor(rho,omega)={cor_rho_omega:.4f} above max bound {self.COR_RHO_OMEGA_MAX}"
                )

        cor_beta_not_omega_violated = False
        if cor_beta_not_omega is not None:
            if cor_beta_not_omega < self.COR_BETA_NOT_OMEGA_MIN:
                cor_beta_not_omega_violated = True
                warnings.append(
                    f"Cor(beta,1-omega)={cor_beta_not_omega:.4f} below min bound {self.COR_BETA_NOT_OMEGA_MIN}"
                )
            elif cor_beta_not_omega > self.COR_BETA_NOT_OMEGA_MAX:
                cor_beta_not_omega_violated = True
                warnings.append(
                    f"Cor(beta,1-omega)={cor_beta_not_omega:.4f} above max bound {self.COR_BETA_NOT_OMEGA_MAX}"
                )

        # Detect mock patterns
        zero_correlation = (
            cor_H_rho is not None and abs(cor_H_rho) < 0.1
        )
        perfect_correlation = (
            cor_H_rho is not None and abs(cor_H_rho) > 0.99
        )
        inverted_correlation = (
            cor_H_rho is not None and cor_H_rho < 0
        )

        # Add mock pattern warnings
        if zero_correlation:
            warnings.append(
                f"MOCK-003: Near-zero correlation Cor(H,rho)={cor_H_rho:.4f} suggests independent mock"
            )
        if perfect_correlation:
            warnings.append(
                f"MOCK-004: Perfect correlation |Cor(H,rho)|={abs(cor_H_rho):.4f} suggests deterministic mock"
            )
        if inverted_correlation:
            warnings.append(
                f"Inverted correlation Cor(H,rho)={cor_H_rho:.4f} (expected positive)"
            )

        return RTTSCorrelationResult(
            cor_H_rho=cor_H_rho,
            cor_rho_omega=cor_rho_omega,
            cor_beta_not_omega=cor_beta_not_omega,
            cor_H_rho_violated=cor_H_rho_violated,
            cor_rho_omega_violated=cor_rho_omega_violated,
            cor_beta_not_omega_violated=cor_beta_not_omega_violated,
            zero_correlation_detected=zero_correlation,
            perfect_correlation_detected=perfect_correlation,
            inverted_correlation_detected=inverted_correlation,
            window_size=len(self._H_history),
            window_start_cycle=self._cycle_history[0] if self._cycle_history else 0,
            warnings=warnings,
        )

    # REAL-READY: Pearson correlation computation
    def _pearson_correlation(self, x: List[float], y: List[float]) -> Optional[float]:
        """
        Compute Pearson correlation coefficient.

        Args:
            x: First variable values
            y: Second variable values

        Returns:
            Correlation coefficient or None if cannot compute
        """
        if len(x) != len(y) or len(x) < 2:
            return None

        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n

        # Compute covariance and standard deviations
        cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / n
        std_x = (sum((xi - mean_x) ** 2 for xi in x) / n) ** 0.5
        std_y = (sum((yi - mean_y) ** 2 for yi in y) / n) ** 0.5

        if std_x == 0 or std_y == 0:
            return 0.0  # No variation

        return cov / (std_x * std_y)

    # REAL-READY: Point-biserial correlation for bool fields
    def _point_biserial(self, continuous: List[float], binary: List[bool]) -> Optional[float]:
        """
        Compute point-biserial correlation for continuous vs binary.

        This is mathematically equivalent to Pearson correlation when
        one variable is dichotomous (0/1).

        Args:
            continuous: Continuous variable values
            binary: Binary variable values (bool)

        Returns:
            Correlation coefficient or None if cannot compute
        """
        if len(continuous) != len(binary) or len(continuous) < 2:
            return None

        # Convert binary to float (True=1, False=0)
        binary_float = [1.0 if b else 0.0 for b in binary]

        return self._pearson_correlation(continuous, binary_float)

    def is_window_full(self) -> bool:
        """Check if window has enough data for computation."""
        return len(self._H_history) >= self.window_size

    def get_window_size(self) -> int:
        """Get current window fill level."""
        return len(self._H_history)

    def reset(self) -> None:
        """Reset tracker state."""
        self._H_history.clear()
        self._rho_history.clear()
        self._omega_history.clear()
        self._beta_history.clear()
        self._cycle_history.clear()
