"""
Phase X P3: TDA Monitor for First-Light Experiments

This module implements the TDAMonitor class for computing Topological Data Analysis
(TDA) metrics at window boundaries during First-Light shadow experiments.

SHADOW MODE CONTRACT:
- All metrics are observational only
- No governance decisions depend on TDA values
- TDA metrics are computed for analysis and logging purposes
- This code runs OFFLINE only, never in production governance paths

Status: P3 IMPLEMENTATION (OFFLINE, SHADOW-ONLY)

Metrics Computed:
- SNS (State Neighborhood Stability): Measures local state coherence
- PCS (Path Connectivity Score): Measures trajectory continuity
- DRS (Divergence Rate Score): Measures deviation from expected behavior
- HSS (Health State Score): Composite health indicator

See: docs/system_law/Phase_X_Prelaunch_Review.md Section 3.3
Binding: first_light_tda_metrics.json schema
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "TDAMetricsSnapshot",
    "TDAMonitor",
]


@dataclass
class TDAMetricsSnapshot:
    """
    Immutable snapshot of TDA metrics computed at a window boundary.

    SHADOW MODE: This is observational data only. No governance
    decisions depend on these values.

    Attributes:
        window_index: Index of the metrics window (0-indexed)
        SNS: State Neighborhood Stability [0, 1]
             Measures local coherence of state trajectories
        PCS: Path Connectivity Score [0, 1]
             Measures trajectory continuity and smoothness
        DRS: Divergence Rate Score [0, 1]
             Measures deviation rate from expected behavior (lower is better)
        HSS: Health State Score [0, 1]
             Composite indicator combining all TDA metrics

    See: docs/system_law/Phase_X_Prelaunch_Review.md Section 3.3
    """

    window_index: int
    SNS: float  # State Neighborhood Stability
    PCS: float  # Path Connectivity Score
    DRS: float  # Divergence Rate Score
    HSS: float  # Health State Score

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert snapshot to JSON-serializable dictionary.

        Returns:
            Dictionary with all TDA metrics
        """
        return {
            "window_index": self.window_index,
            "SNS": round(self.SNS, 6),
            "PCS": round(self.PCS, 6),
            "DRS": round(self.DRS, 6),
            "HSS": round(self.HSS, 6),
        }

    def is_healthy(self, threshold: float = 0.5) -> bool:
        """
        Check if HSS meets health threshold.

        SHADOW MODE: This check is for logging/analysis only.

        Args:
            threshold: Minimum HSS value to consider healthy

        Returns:
            True if HSS >= threshold
        """
        return self.HSS >= threshold


class TDAMonitor:
    """
    Computes TDA metrics for First-Light shadow experiments.

    SHADOW MODE CONTRACT:
    - All computations are observational only
    - No governance decisions depend on monitor output
    - Stateless design: same inputs always produce same outputs
    - Reset() clears any accumulated state for test isolation

    The monitor computes four key metrics:
    1. SNS (State Neighborhood Stability): Local state coherence
    2. PCS (Path Connectivity Score): Trajectory continuity
    3. DRS (Divergence Rate Score): Deviation rate (lower is better)
    4. HSS (Health State Score): Composite health indicator

    For MVP, deterministic mock values are returned. Future versions
    will implement actual TDA algorithms (persistent homology, etc.).

    See: docs/system_law/Phase_X_Prelaunch_Review.md Section 3.3
    """

    # Default weights for HSS computation
    DEFAULT_SNS_WEIGHT: float = 0.3
    DEFAULT_PCS_WEIGHT: float = 0.3
    DEFAULT_DRS_WEIGHT: float = 0.2
    DEFAULT_SUCCESS_WEIGHT: float = 0.2

    def __init__(
        self,
        sns_weight: float = DEFAULT_SNS_WEIGHT,
        pcs_weight: float = DEFAULT_PCS_WEIGHT,
        drs_weight: float = DEFAULT_DRS_WEIGHT,
        success_weight: float = DEFAULT_SUCCESS_WEIGHT,
    ) -> None:
        """
        Initialize TDA monitor.

        Args:
            sns_weight: Weight for SNS in HSS computation
            pcs_weight: Weight for PCS in HSS computation
            drs_weight: Weight for DRS in HSS computation (negative contribution)
            success_weight: Weight for success rate in HSS computation

        Raises:
            ValueError: If weights don't sum to approximately 1.0
        """
        total_weight = sns_weight + pcs_weight + drs_weight + success_weight
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(
                f"TDAMonitor weights must sum to 1.0, got {total_weight}"
            )

        self._sns_weight = sns_weight
        self._pcs_weight = pcs_weight
        self._drs_weight = drs_weight
        self._success_weight = success_weight

        # Track computed snapshots for analysis
        self._computed_snapshots: List[TDAMetricsSnapshot] = []

    def compute(
        self,
        H_series: Sequence[float],
        rho_series: Sequence[float],
        success_series: Sequence[bool],
        window_index: int,
    ) -> TDAMetricsSnapshot:
        """
        Compute TDA metrics from window trajectories.

        SHADOW MODE: This computation is for analysis only.
        Results are logged but never influence governance.

        Args:
            H_series: Health metric trajectory for the window
            rho_series: RSI (rho) trajectory for the window
            success_series: Success/failure trajectory for the window
            window_index: Index of this window (0-indexed)

        Returns:
            TDAMetricsSnapshot with computed metrics

        For MVP, this returns deterministic values based on input statistics.
        Future versions will implement actual TDA algorithms.
        """
        # Handle empty series gracefully
        if not H_series or not rho_series or not success_series:
            snapshot = TDAMetricsSnapshot(
                window_index=window_index,
                SNS=0.5,  # Neutral default
                PCS=0.5,
                DRS=0.5,
                HSS=0.5,
            )
            self._computed_snapshots.append(snapshot)
            return snapshot

        # Compute SNS: State Neighborhood Stability
        # Based on variance of state values (lower variance = higher stability)
        sns = self._compute_sns(H_series, rho_series)

        # Compute PCS: Path Connectivity Score
        # Based on smoothness of trajectories
        pcs = self._compute_pcs(H_series, rho_series)

        # Compute DRS: Divergence Rate Score
        # Based on rate of change (lower is better, so we invert for HSS)
        drs = self._compute_drs(H_series, rho_series)

        # Compute success rate for HSS
        success_rate = sum(1 for s in success_series if s) / len(success_series)

        # Compute HSS: Health State Score (composite)
        # DRS contribution is inverted (1 - DRS) because lower divergence is better
        hss = (
            self._sns_weight * sns +
            self._pcs_weight * pcs +
            self._drs_weight * (1.0 - drs) +
            self._success_weight * success_rate
        )

        # Clamp to [0, 1]
        hss = max(0.0, min(1.0, hss))

        snapshot = TDAMetricsSnapshot(
            window_index=window_index,
            SNS=sns,
            PCS=pcs,
            DRS=drs,
            HSS=hss,
        )

        self._computed_snapshots.append(snapshot)
        return snapshot

    def _compute_sns(
        self,
        H_series: Sequence[float],
        rho_series: Sequence[float],
    ) -> float:
        """
        Compute State Neighborhood Stability (SNS).

        SNS measures local coherence of state trajectories.
        Higher values indicate more stable neighborhood structure.

        For MVP: Based on inverse normalized variance of H and rho.
        Future: Use persistent homology to measure topological stability.

        Args:
            H_series: Health metric trajectory
            rho_series: RSI trajectory

        Returns:
            SNS value in [0, 1]
        """
        if len(H_series) < 2 or len(rho_series) < 2:
            return 0.5  # Neutral for insufficient data

        # Compute variance of both series
        h_mean = sum(H_series) / len(H_series)
        h_var = sum((h - h_mean) ** 2 for h in H_series) / len(H_series)

        rho_mean = sum(rho_series) / len(rho_series)
        rho_var = sum((r - rho_mean) ** 2 for r in rho_series) / len(rho_series)

        # Combined variance (normalized to [0, 1] range assuming max var ~0.25)
        combined_var = (h_var + rho_var) / 2.0
        max_expected_var = 0.25

        # SNS = 1 - normalized_variance (higher stability = lower variance)
        sns = 1.0 - min(1.0, combined_var / max_expected_var)

        return max(0.0, min(1.0, sns))

    def _compute_pcs(
        self,
        H_series: Sequence[float],
        rho_series: Sequence[float],
    ) -> float:
        """
        Compute Path Connectivity Score (PCS).

        PCS measures trajectory continuity and smoothness.
        Higher values indicate smoother, more connected paths.

        For MVP: Based on inverse mean absolute difference between steps.
        Future: Use path topology analysis.

        Args:
            H_series: Health metric trajectory
            rho_series: RSI trajectory

        Returns:
            PCS value in [0, 1]
        """
        if len(H_series) < 2 or len(rho_series) < 2:
            return 0.5  # Neutral for insufficient data

        # Compute mean absolute step changes
        h_steps = [abs(H_series[i + 1] - H_series[i]) for i in range(len(H_series) - 1)]
        rho_steps = [abs(rho_series[i + 1] - rho_series[i]) for i in range(len(rho_series) - 1)]

        mean_h_step = sum(h_steps) / len(h_steps) if h_steps else 0.0
        mean_rho_step = sum(rho_steps) / len(rho_steps) if rho_steps else 0.0

        # Combined step size (normalized assuming max step ~0.5)
        combined_step = (mean_h_step + mean_rho_step) / 2.0
        max_expected_step = 0.3

        # PCS = 1 - normalized_step (smoother paths = smaller steps)
        pcs = 1.0 - min(1.0, combined_step / max_expected_step)

        return max(0.0, min(1.0, pcs))

    def _compute_drs(
        self,
        H_series: Sequence[float],
        rho_series: Sequence[float],
    ) -> float:
        """
        Compute Divergence Rate Score (DRS).

        DRS measures the rate of deviation from stable behavior.
        LOWER values indicate BETTER behavior (less divergence).

        For MVP: Based on trend direction and magnitude.
        Future: Use topological divergence measures.

        Args:
            H_series: Health metric trajectory
            rho_series: RSI trajectory

        Returns:
            DRS value in [0, 1] (lower is better)
        """
        if len(H_series) < 2 or len(rho_series) < 2:
            return 0.5  # Neutral for insufficient data

        # Compute trend: difference between end and start
        h_trend = H_series[-1] - H_series[0]
        rho_trend = rho_series[-1] - rho_series[0]

        # Negative trends indicate divergence from stable state
        # Positive trends indicate convergence toward stable state
        # DRS penalizes negative trends

        # Normalize trends to [-1, 1] assuming max change of 1.0
        h_divergence = max(0.0, -h_trend)  # Only penalize negative trends
        rho_divergence = max(0.0, -rho_trend)

        # Combined divergence
        drs = (h_divergence + rho_divergence) / 2.0

        return max(0.0, min(1.0, drs))

    def get_computed_snapshots(self) -> List[TDAMetricsSnapshot]:
        """
        Get all computed TDA snapshots.

        Returns:
            List of TDAMetricsSnapshot objects computed so far
        """
        return list(self._computed_snapshots)

    def get_latest_snapshot(self) -> Optional[TDAMetricsSnapshot]:
        """
        Get the most recently computed snapshot.

        Returns:
            Latest TDAMetricsSnapshot, or None if none computed
        """
        return self._computed_snapshots[-1] if self._computed_snapshots else None

    def get_trajectory(self) -> Dict[str, List[float]]:
        """
        Get trajectories of all TDA metrics across windows.

        Returns:
            Dictionary with SNS, PCS, DRS, HSS trajectories
        """
        return {
            "SNS": [s.SNS for s in self._computed_snapshots],
            "PCS": [s.PCS for s in self._computed_snapshots],
            "DRS": [s.DRS for s in self._computed_snapshots],
            "HSS": [s.HSS for s in self._computed_snapshots],
        }

    def reset(self) -> None:
        """
        Reset monitor state.

        Clears all computed snapshots for test isolation.
        """
        self._computed_snapshots.clear()
