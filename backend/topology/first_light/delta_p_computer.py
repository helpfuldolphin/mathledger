"""
Phase X P3: Δp Computation

This module implements the Δp (learning curve) computation for measuring
success rate slopes. See docs/system_law/Phase_X_P3_Spec.md for full specification.

SHADOW MODE CONTRACT:
- All computations are read-only
- No side effects on governance
- Results are for observation/logging only
- No control flow depends on computed values
- This code runs OFFLINE only, never in production governance paths

Status: P3 IMPLEMENTATION (OFFLINE, SHADOW-ONLY)

TODO[P3-BUDGET-NOISE-001]: Budget risk feed into P3 Δp noise injection
    When budget drift affects cycle measurements, the noise floor must be adjusted.
    See docs/system_law/Budget_PhaseX_Doctrine.md Section 2.4

    Implementation (when authorized):
    1. Import budget_drift_trajectory from derivation.budget_invariants
    2. Compute drift-adjusted noise floor: 1.0 / sqrt(1 - |drift_value|)
    3. Apply noise floor multiplier to Δp slope calculation uncertainty
    4. Add uncertainty bands to RSI readings when drift_classification != "STABLE"
    5. Emit "budget_confounded" markers in cycle logs when |drift| > 0.05

    Dependencies:
    - budget_drift_trajectory.schema.json
    - budget_governance_signal.schema.json
    - GovernanceSignal from governance_verifier.py

    Status: NOT AUTHORIZED (requires P3 execution auth)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.topology.first_light.budget_binding import (
    BudgetRiskSignal,
    build_budget_risk_signal,
    extend_stability_report_with_budget,
)

__all__ = [
    "DeltaPMetrics",
    "DeltaPComputer",
    "compute_slope",
    "BudgetAwareDeltaPMetrics",
]


def compute_slope(values: List[float]) -> Optional[float]:
    """
    Compute slope via simple linear regression.

    Uses least squares: slope = Σ((x - x̄)(y - ȳ)) / Σ((x - x̄)²)

    Args:
        values: Y values (X is implicit: 0, 1, 2, ...)

    Returns:
        Slope, or None if insufficient data (< 2 points)

    See: docs/system_law/Phase_X_P3_Spec.md Section 4.3
    """
    if len(values) < 2:
        return None

    n = len(values)
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n

    numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))

    if denominator == 0:
        return 0.0

    return numerator / denominator


@dataclass
class DeltaPMetrics:
    """
    Learning curve metrics computed in SHADOW mode.

    SHADOW MODE CONTRACT:
    - All values are observational only
    - No governance decisions depend on these metrics
    - Used for analysis and logging purposes

    See: docs/system_law/Phase_X_P3_Spec.md Section 4.2
    """

    # U2 metrics
    delta_p_success: Optional[float] = None       # Slope of success rate
    success_rate_trajectory: List[float] = field(default_factory=list)
    success_rate_final: Optional[float] = None
    success_count: int = 0
    total_count: int = 0

    # RFL metrics (abstention)
    delta_p_abstention: Optional[float] = None    # Slope of abstention rate
    abstention_trajectory: List[float] = field(default_factory=list)
    abstention_rate_final: Optional[float] = None
    abstention_count: int = 0

    # Safe region metrics
    omega_occupancy: float = 0.0                  # Overall Ω occupancy
    omega_count: int = 0
    omega_occupancy_trajectory: List[float] = field(default_factory=list)

    # HARD mode metrics
    hard_ok_rate: float = 0.0                     # Overall HARD-OK rate
    hard_ok_count: int = 0
    hard_ok_trajectory: List[float] = field(default_factory=list)

    # RSI metrics
    mean_rsi: float = 0.0                         # Mean ρ over all cycles
    min_rsi: float = 1.0
    max_rsi: float = 0.0
    rsi_sum: float = 0.0
    rsi_trajectory: List[float] = field(default_factory=list)

    # Window configuration
    window_size: int = 50
    total_windows: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "delta_p_success": self.delta_p_success,
            "delta_p_abstention": self.delta_p_abstention,
            "success_rate_final": self.success_rate_final,
            "abstention_rate_final": self.abstention_rate_final,
            "omega_occupancy": round(self.omega_occupancy, 4),
            "hard_ok_rate": round(self.hard_ok_rate, 4),
            "mean_rsi": round(self.mean_rsi, 4),
            "min_rsi": round(self.min_rsi, 4),
            "max_rsi": round(self.max_rsi, 4),
            "total_count": self.total_count,
            "window_size": self.window_size,
            "total_windows": self.total_windows,
            "trajectories": {
                "success_rate": [round(s, 4) for s in self.success_rate_trajectory],
                "abstention_rate": [round(a, 4) for a in self.abstention_trajectory],
                "omega_occupancy": [round(o, 4) for o in self.omega_occupancy_trajectory],
                "hard_ok_rate": [round(h, 4) for h in self.hard_ok_trajectory],
                "rsi": [round(r, 4) for r in self.rsi_trajectory],
            },
        }

    def meets_success_criteria(self) -> Dict[str, bool]:
        """
        Check if metrics meet success criteria.

        SHADOW MODE: This is for LOGGING only. The result does NOT
        influence any control flow.

        Returns:
            Dict mapping criterion name to pass/fail
        """
        return {
            "delta_p_success_positive": (
                self.delta_p_success is not None and
                self.delta_p_success > 0
            ),
            "delta_p_abstention_negative": (
                self.delta_p_abstention is not None and
                self.delta_p_abstention < 0
            ),
            "omega_occupancy_90": self.omega_occupancy >= 0.90,
            "hard_ok_80": self.hard_ok_rate >= 0.80,
            "mean_rsi_60": self.mean_rsi >= 0.60,
        }


@dataclass
class BudgetAwareDeltaPMetrics:
    """
    Extended Δp metrics with budget risk integration.

    SHADOW MODE CONTRACT:
    - Budget multipliers are computed but do NOT alter behavior
    - All values are observational only
    - Used for logging and analysis purposes

    See: docs/system_law/Budget_PhaseX_Doctrine.md Section 2.4
    """

    # Base metrics
    metrics: DeltaPMetrics = field(default_factory=DeltaPMetrics)

    # Budget risk signal
    budget_signal: BudgetRiskSignal = field(
        default_factory=lambda: build_budget_risk_signal()
    )

    # Computed adjustments (SHADOW: not enforced)
    noise_multiplier: float = 1.0
    rsi_correction_factor: float = 1.0
    confounded_windows: int = 0

    # Adjusted metrics (for analysis only)
    adjusted_mean_rsi: float = 0.0
    uncertainty_band_rsi: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary with budget risk."""
        base = self.metrics.to_dict()
        return extend_stability_report_with_budget(base, self.budget_signal)

    def build_stability_report(self) -> Dict[str, Any]:
        """
        Build stability report with budget risk section.

        Returns a stability_report.json compatible structure with:
        - All base DeltaPMetrics fields
        - budget_risk section with drift_class, noise_multiplier, admissibility_hint

        SHADOW MODE: This is for logging only.
        """
        report = {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "metrics": self.metrics.to_dict(),
            "success_criteria": self.metrics.meets_success_criteria(),
            "budget_risk": {
                "drift_class": self.budget_signal.drift_class.value,
                "noise_multiplier": round(self.noise_multiplier, 4),
                "admissibility_hint": self.budget_signal.admissibility_hint,
                "stability_class": self.budget_signal.stability_class.value,
                "health_score": round(self.budget_signal.health_score, 2),
                "stability_index": round(self.budget_signal.stability_index, 4),
                "budget_confounded": self.budget_signal.budget_confounded,
                "confounded_windows": self.confounded_windows,
                "rsi_correction_factor": round(self.rsi_correction_factor, 4),
                "adjusted_mean_rsi": round(self.adjusted_mean_rsi, 4),
                "uncertainty_band_rsi": round(self.uncertainty_band_rsi, 4),
            },
        }
        return report

    @classmethod
    def from_metrics_and_budget(
        cls,
        metrics: DeltaPMetrics,
        drift_value: float = 0.0,
        health_score: float = 100.0,
        stability_index: float = 1.0,
        inv_bud_failures: Optional[List[str]] = None,
    ) -> "BudgetAwareDeltaPMetrics":
        """
        Create budget-aware metrics from base metrics and budget data.

        Args:
            metrics: Base DeltaPMetrics
            drift_value: Current budget drift
            health_score: Budget health score
            stability_index: Stability index
            inv_bud_failures: List of failing invariants

        Returns:
            BudgetAwareDeltaPMetrics with computed adjustments
        """
        budget_signal = build_budget_risk_signal(
            drift_value=drift_value,
            health_score=health_score,
            stability_index=stability_index,
            inv_bud_failures=inv_bud_failures,
        )

        noise_multiplier = budget_signal.noise_multiplier
        rsi_correction = budget_signal.rsi_correction_factor

        # Compute adjusted RSI (SHADOW: for analysis only)
        adjusted_mean_rsi = metrics.mean_rsi * rsi_correction

        # Compute uncertainty band based on noise multiplier
        # Higher noise multiplier = larger uncertainty band
        uncertainty_band = (noise_multiplier - 1.0) * 0.1  # 10% per multiplier unit

        # Count confounded windows (where drift would affect measurement)
        confounded_windows = 0
        if budget_signal.budget_confounded:
            confounded_windows = metrics.total_windows

        return cls(
            metrics=metrics,
            budget_signal=budget_signal,
            noise_multiplier=noise_multiplier,
            rsi_correction_factor=rsi_correction,
            confounded_windows=confounded_windows,
            adjusted_mean_rsi=adjusted_mean_rsi,
            uncertainty_band_rsi=uncertainty_band,
        )


class DeltaPComputer:
    """
    Computes Δp learning curve metrics.

    SHADOW MODE CONTRACT:
    - All computations are read-only
    - No side effects on governance
    - Results are for observation/logging only
    - This is used OFFLINE only from tests

    See: docs/system_law/Phase_X_P3_Spec.md Section 4.2
    """

    def __init__(self, window_size: int = 50) -> None:
        """
        Initialize Δp computer.

        Args:
            window_size: Size of metrics windows for trajectory computation
        """
        self.window_size = window_size

        # Cumulative counters
        self._success_count: int = 0
        self._abstention_count: int = 0
        self._omega_count: int = 0
        self._hard_ok_count: int = 0
        self._total_count: int = 0
        self._rsi_sum: float = 0.0
        self._min_rsi: float = 1.0
        self._max_rsi: float = 0.0

        # Window accumulators
        self._window_success: int = 0
        self._window_abstention: int = 0
        self._window_omega: int = 0
        self._window_hard_ok: int = 0
        self._window_count: int = 0
        self._window_rsi_sum: float = 0.0

        # Trajectory storage
        self._success_trajectory: List[float] = []
        self._abstention_trajectory: List[float] = []
        self._omega_trajectory: List[float] = []
        self._hard_ok_trajectory: List[float] = []
        self._rsi_trajectory: List[float] = []

    def update(
        self,
        cycle: int,
        success: bool,
        in_omega: bool,
        hard_ok: bool,
        rsi: float,
        abstained: bool = False,
    ) -> None:
        """
        Update metrics with cycle observation.

        Args:
            cycle: Current cycle number
            success: U2 success or RFL non-abstention
            in_omega: State in safe region
            hard_ok: HARD mode OK
            rsi: Current RSI (ρ)
            abstained: Whether this cycle abstained (for RFL)
        """
        # Update cumulative counters
        self._total_count += 1
        if success:
            self._success_count += 1
        if abstained:
            self._abstention_count += 1
        if in_omega:
            self._omega_count += 1
        if hard_ok:
            self._hard_ok_count += 1
        self._rsi_sum += rsi
        self._min_rsi = min(self._min_rsi, rsi)
        self._max_rsi = max(self._max_rsi, rsi)

        # Update window accumulators
        self._window_count += 1
        if success:
            self._window_success += 1
        if abstained:
            self._window_abstention += 1
        if in_omega:
            self._window_omega += 1
        if hard_ok:
            self._window_hard_ok += 1
        self._window_rsi_sum += rsi

        # Check if window is complete
        if self._window_count >= self.window_size:
            self._finalize_window()

    def _finalize_window(self) -> None:
        """Finalize current window and store trajectory point."""
        if self._window_count == 0:
            return

        # Compute window rates
        success_rate = self._window_success / self._window_count
        abstention_rate = self._window_abstention / self._window_count
        omega_rate = self._window_omega / self._window_count
        hard_ok_rate = self._window_hard_ok / self._window_count
        mean_rsi = self._window_rsi_sum / self._window_count

        # Store in trajectories
        self._success_trajectory.append(success_rate)
        self._abstention_trajectory.append(abstention_rate)
        self._omega_trajectory.append(omega_rate)
        self._hard_ok_trajectory.append(hard_ok_rate)
        self._rsi_trajectory.append(mean_rsi)

        # Reset window
        self._window_success = 0
        self._window_abstention = 0
        self._window_omega = 0
        self._window_hard_ok = 0
        self._window_count = 0
        self._window_rsi_sum = 0.0

    def compute(self) -> DeltaPMetrics:
        """
        Compute current Δp metrics.

        Returns:
            DeltaPMetrics instance with computed values
        """
        # Finalize any partial window
        if self._window_count > 0:
            self._finalize_window()

        # Compute cumulative rates
        if self._total_count > 0:
            success_rate_final = self._success_count / self._total_count
            abstention_rate_final = self._abstention_count / self._total_count
            omega_occupancy = self._omega_count / self._total_count
            hard_ok_rate = self._hard_ok_count / self._total_count
            mean_rsi = self._rsi_sum / self._total_count
        else:
            success_rate_final = 0.0
            abstention_rate_final = 0.0
            omega_occupancy = 0.0
            hard_ok_rate = 0.0
            mean_rsi = 0.0

        # Compute slopes (Δp)
        delta_p_success = compute_slope(self._success_trajectory)
        delta_p_abstention = compute_slope(self._abstention_trajectory)

        return DeltaPMetrics(
            delta_p_success=delta_p_success,
            success_rate_trajectory=list(self._success_trajectory),
            success_rate_final=success_rate_final,
            success_count=self._success_count,
            total_count=self._total_count,
            delta_p_abstention=delta_p_abstention,
            abstention_trajectory=list(self._abstention_trajectory),
            abstention_rate_final=abstention_rate_final,
            abstention_count=self._abstention_count,
            omega_occupancy=omega_occupancy,
            omega_count=self._omega_count,
            omega_occupancy_trajectory=list(self._omega_trajectory),
            hard_ok_rate=hard_ok_rate,
            hard_ok_count=self._hard_ok_count,
            hard_ok_trajectory=list(self._hard_ok_trajectory),
            mean_rsi=mean_rsi,
            min_rsi=self._min_rsi if self._total_count > 0 else 0.0,
            max_rsi=self._max_rsi if self._total_count > 0 else 0.0,
            rsi_sum=self._rsi_sum,
            rsi_trajectory=list(self._rsi_trajectory),
            window_size=self.window_size,
            total_windows=len(self._success_trajectory),
        )

    def get_trajectory_point(self, window_index: int) -> Dict[str, float]:
        """
        Get metrics for a specific window.

        Args:
            window_index: Index of the window

        Returns:
            Dict with metrics for that window

        Raises:
            IndexError: If window_index is out of range
        """
        if window_index < 0 or window_index >= len(self._success_trajectory):
            raise IndexError(f"Window index {window_index} out of range")

        return {
            "window_index": window_index,
            "success_rate": self._success_trajectory[window_index],
            "abstention_rate": self._abstention_trajectory[window_index],
            "omega_occupancy": self._omega_trajectory[window_index],
            "hard_ok_rate": self._hard_ok_trajectory[window_index],
            "mean_rsi": self._rsi_trajectory[window_index],
        }

    def reset(self) -> None:
        """Reset all counters and trajectories."""
        self._success_count = 0
        self._abstention_count = 0
        self._omega_count = 0
        self._hard_ok_count = 0
        self._total_count = 0
        self._rsi_sum = 0.0
        self._min_rsi = 1.0
        self._max_rsi = 0.0

        self._window_success = 0
        self._window_abstention = 0
        self._window_omega = 0
        self._window_hard_ok = 0
        self._window_count = 0
        self._window_rsi_sum = 0.0

        self._success_trajectory.clear()
        self._abstention_trajectory.clear()
        self._omega_trajectory.clear()
        self._hard_ok_trajectory.clear()
        self._rsi_trajectory.clear()
