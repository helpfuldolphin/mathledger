"""
TDA Metric Computations

Implements the four core TDA metrics:
- SNS: Structural Novelty Score - measures unexpected proof structure emergence
- PCS: Proof Coherence Score - measures consistency of derivation patterns
- DRS: Drift Rate Score - measures rate of state trajectory deviation
- HSS: Homological Stability Score - measures persistence of topological features

See: docs/system_law/TDA_PhaseX_Binding.md for full specification.

SHADOW MODE CONTRACT:
- All computations are pure functions with no side effects
- No governance modification based on computed values
- Results are for logging and analysis only
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "SNSComputer",
    "PCSComputer",
    "DRSComputer",
    "HSSComputer",
    "TDAMetrics",
    "TDAWindowMetrics",
    "compute_sns",
    "compute_pcs",
    "compute_drs",
    "compute_hss",
]


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TDAMetrics:
    """
    Single-cycle TDA metrics snapshot.

    SHADOW MODE: Observational only, no governance modification.
    """

    sns: float = 0.0  # Structural Novelty Score [0, 1]
    pcs: float = 1.0  # Proof Coherence Score [0, 1]
    drs: float = 0.0  # Drift Rate Score [0, inf)
    hss: float = 1.0  # Homological Stability Score [0, 1]

    # Envelope membership
    in_tda_envelope: bool = True

    # Thresholds used
    sns_threshold: float = 0.4
    pcs_threshold: float = 0.6
    hss_threshold: float = 0.6

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "sns": round(self.sns, 6),
            "pcs": round(self.pcs, 6),
            "drs": round(self.drs, 6),
            "hss": round(self.hss, 6),
            "in_tda_envelope": self.in_tda_envelope,
        }

    def check_envelope(self) -> bool:
        """
        Check if metrics are within TDA stability envelope Omega_TDA.

        Omega_TDA = { (SNS, PCS, HSS) : SNS <= 0.4 AND PCS >= 0.6 AND HSS >= 0.6 }
        """
        return (
            self.sns <= self.sns_threshold and
            self.pcs >= self.pcs_threshold and
            self.hss >= self.hss_threshold
        )


@dataclass
class TDAWindowMetrics:
    """
    Window-aggregated TDA metrics for P3 stability envelopes.

    See: docs/system_law/schemas/tda/tda_metrics_p3.schema.json
    """

    window_index: int = 0
    window_start_cycle: int = 0
    window_end_cycle: int = 0

    # SNS statistics
    sns_mean: float = 0.0
    sns_max: float = 0.0
    sns_min: float = 1.0
    sns_std: float = 0.0
    sns_anomaly_count: int = 0  # SNS > 0.6
    sns_elevated_count: int = 0  # SNS > 0.4

    # PCS statistics
    pcs_mean: float = 1.0
    pcs_max: float = 1.0
    pcs_min: float = 1.0
    pcs_std: float = 0.0
    pcs_low_coherence_count: int = 0  # PCS < 0.6
    pcs_incoherent_count: int = 0  # PCS < 0.4

    # HSS statistics
    hss_mean: float = 1.0
    hss_max: float = 1.0
    hss_min: float = 1.0
    hss_std: float = 0.0
    hss_degradation_count: int = 0  # HSS < 0.4
    hss_unstable_count: int = 0  # HSS < 0.6

    # Betti numbers (optional)
    betti_b0: int = 1
    betti_b1: int = 0
    betti_b2: int = 0

    # Envelope metrics
    envelope_occupancy_rate: float = 1.0
    envelope_exit_count: int = 0
    max_envelope_exit_streak: int = 0

    # Red-flag counts
    tda_sns_anomaly_flags: int = 0
    tda_pcs_collapse_flags: int = 0
    tda_hss_degradation_flags: int = 0
    tda_envelope_exit_flags: int = 0

    # Trajectories (optional, for detailed analysis)
    sns_trajectory: List[float] = field(default_factory=list)
    pcs_trajectory: List[float] = field(default_factory=list)
    hss_trajectory: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary per schema."""
        return {
            "schema_version": "1.0.0",
            "window_index": self.window_index,
            "window_start_cycle": self.window_start_cycle,
            "window_end_cycle": self.window_end_cycle,
            "mode": "SHADOW",
            "sns": {
                "mean": round(self.sns_mean, 6),
                "max": round(self.sns_max, 6),
                "min": round(self.sns_min, 6),
                "std": round(self.sns_std, 6),
                "anomaly_count": self.sns_anomaly_count,
                "elevated_count": self.sns_elevated_count,
            },
            "pcs": {
                "mean": round(self.pcs_mean, 6),
                "max": round(self.pcs_max, 6),
                "min": round(self.pcs_min, 6),
                "std": round(self.pcs_std, 6),
                "low_coherence_count": self.pcs_low_coherence_count,
                "incoherent_count": self.pcs_incoherent_count,
            },
            "hss": {
                "mean": round(self.hss_mean, 6),
                "max": round(self.hss_max, 6),
                "min": round(self.hss_min, 6),
                "std": round(self.hss_std, 6),
                "degradation_count": self.hss_degradation_count,
                "unstable_count": self.hss_unstable_count,
                "betti_snapshot": {
                    "b0": self.betti_b0,
                    "b1": self.betti_b1,
                    "b2": self.betti_b2,
                },
            },
            "envelope": {
                "occupancy_rate": round(self.envelope_occupancy_rate, 6),
                "exit_count": self.envelope_exit_count,
                "max_exit_streak": self.max_envelope_exit_streak,
            },
            "red_flags": {
                "tda_sns_anomaly": self.tda_sns_anomaly_flags,
                "tda_pcs_collapse": self.tda_pcs_collapse_flags,
                "tda_hss_degradation": self.tda_hss_degradation_flags,
                "tda_envelope_exit": self.tda_envelope_exit_flags,
            },
        }


# =============================================================================
# SNS: Structural Novelty Score
# =============================================================================

class SNSComputer:
    """
    Computes Structural Novelty Score based on proof structure patterns.

    SNS(t) = 1 - similarity(P_t, P_historical)

    Where:
    - P_t: proof structure at cycle t (represented by success/depth patterns)
    - P_historical: accumulated historical patterns
    - similarity(): pattern similarity measure [0, 1]

    SHADOW MODE: Pure computation, no side effects.
    """

    def __init__(self, history_window: int = 100) -> None:
        """
        Initialize SNS computer.

        Args:
            history_window: Number of historical patterns to maintain
        """
        self._history_window = history_window
        self._pattern_history: List[Tuple[bool, int]] = []
        self._depth_histogram: Dict[int, int] = {}
        self._total_patterns: int = 0

    def compute(
        self,
        success: bool,
        depth: int,
        proof_hash: Optional[str] = None,
    ) -> float:
        """
        Compute SNS for current cycle.

        Args:
            success: Whether proof succeeded
            depth: Proof depth
            proof_hash: Optional proof hash for exact match detection

        Returns:
            SNS value in [0, 1] where 1 = completely novel
        """
        pattern = (success, depth)

        # Check for exact historical match
        if self._total_patterns == 0:
            # First pattern is novel by definition
            sns = 0.5  # Moderate novelty for bootstrap
        else:
            # Compute similarity based on depth distribution
            depth_freq = self._depth_histogram.get(depth, 0) / max(1, self._total_patterns)
            success_match_rate = sum(
                1 for s, d in self._pattern_history
                if s == success
            ) / max(1, len(self._pattern_history))

            # Similarity combines depth frequency and success match
            similarity = 0.6 * depth_freq + 0.4 * success_match_rate
            sns = 1.0 - similarity

        # Update history
        self._pattern_history.append(pattern)
        if len(self._pattern_history) > self._history_window:
            old_pattern = self._pattern_history.pop(0)
            old_depth = old_pattern[1]
            if old_depth in self._depth_histogram:
                self._depth_histogram[old_depth] -= 1
                if self._depth_histogram[old_depth] <= 0:
                    del self._depth_histogram[old_depth]

        self._depth_histogram[depth] = self._depth_histogram.get(depth, 0) + 1
        self._total_patterns += 1

        # Clamp to [0, 1]
        return max(0.0, min(1.0, sns))

    def reset(self) -> None:
        """Reset computer state."""
        self._pattern_history.clear()
        self._depth_histogram.clear()
        self._total_patterns = 0


def compute_sns(
    success: bool,
    depth: int,
    historical_patterns: List[Tuple[bool, int]],
) -> float:
    """
    Compute SNS as standalone function.

    Args:
        success: Whether proof succeeded
        depth: Proof depth
        historical_patterns: List of (success, depth) tuples

    Returns:
        SNS value in [0, 1]
    """
    if not historical_patterns:
        return 0.5

    # Build depth histogram
    depth_counts: Dict[int, int] = {}
    success_count = 0
    for s, d in historical_patterns:
        depth_counts[d] = depth_counts.get(d, 0) + 1
        if s == success:
            success_count += 1

    total = len(historical_patterns)
    depth_freq = depth_counts.get(depth, 0) / total
    success_rate = success_count / total

    similarity = 0.6 * depth_freq + 0.4 * success_rate
    return max(0.0, min(1.0, 1.0 - similarity))


# =============================================================================
# PCS: Proof Coherence Score
# =============================================================================

class PCSComputer:
    """
    Computes Proof Coherence Score based on derivation pattern consistency.

    PCS(t) = coherence(D_window)

    Where:
    - D_window: derivation sequences in observation window
    - coherence(): ratio of consistent inference patterns

    SHADOW MODE: Pure computation, no side effects.
    """

    def __init__(self, coherence_window: int = 20) -> None:
        """
        Initialize PCS computer.

        Args:
            coherence_window: Window size for coherence measurement
        """
        self._coherence_window = coherence_window
        self._success_history: List[bool] = []
        self._transition_history: List[Tuple[bool, bool]] = []

    def compute(
        self,
        success: bool,
        rho: float,
        H: float,
    ) -> float:
        """
        Compute PCS for current cycle.

        Coherence is measured by:
        1. Consistency of success patterns (no rapid oscillation)
        2. Correlation between rho and success
        3. Health (H) stability

        Args:
            success: Whether proof succeeded
            rho: RSI value (stability indicator)
            H: Health metric

        Returns:
            PCS value in [0, 1] where 1 = fully coherent
        """
        # Track success transitions
        if self._success_history:
            transition = (self._success_history[-1], success)
            self._transition_history.append(transition)
            if len(self._transition_history) > self._coherence_window:
                self._transition_history.pop(0)

        self._success_history.append(success)
        if len(self._success_history) > self._coherence_window:
            self._success_history.pop(0)

        if len(self._success_history) < 3:
            return 1.0  # Not enough data for coherence

        # Coherence factors:
        # 1. Transition stability (fewer oscillations = more coherent)
        if self._transition_history:
            oscillations = sum(
                1 for t in self._transition_history
                if t[0] != t[1]
            )
            transition_coherence = 1.0 - (oscillations / len(self._transition_history))
        else:
            transition_coherence = 1.0

        # 2. RSI correlation (high rho with success = coherent)
        rho_contribution = rho if success else (1.0 - rho) * 0.5 + 0.5

        # 3. Health contribution
        health_contribution = H

        # Weighted combination
        pcs = (
            0.5 * transition_coherence +
            0.3 * rho_contribution +
            0.2 * health_contribution
        )

        return max(0.0, min(1.0, pcs))

    def reset(self) -> None:
        """Reset computer state."""
        self._success_history.clear()
        self._transition_history.clear()


def compute_pcs(
    success_trajectory: List[bool],
    rho_trajectory: List[float],
    H_trajectory: List[float],
) -> float:
    """
    Compute PCS as standalone function over trajectories.

    Args:
        success_trajectory: List of success values
        rho_trajectory: List of RSI values
        H_trajectory: List of health values

    Returns:
        PCS value in [0, 1]
    """
    if len(success_trajectory) < 3:
        return 1.0

    # Transition stability
    oscillations = sum(
        1 for i in range(1, len(success_trajectory))
        if success_trajectory[i] != success_trajectory[i-1]
    )
    transition_coherence = 1.0 - (oscillations / (len(success_trajectory) - 1))

    # Mean rho
    mean_rho = sum(rho_trajectory) / len(rho_trajectory) if rho_trajectory else 0.5

    # Mean H
    mean_H = sum(H_trajectory) / len(H_trajectory) if H_trajectory else 0.5

    pcs = (
        0.5 * transition_coherence +
        0.3 * mean_rho +
        0.2 * mean_H
    )

    return max(0.0, min(1.0, pcs))


# =============================================================================
# DRS: Drift Rate Score
# =============================================================================

class DRSComputer:
    """
    Computes Drift Rate Score based on state trajectory deviation.

    DRS(t) = ||S_real(t) - S_twin(t)|| / delta_t

    Where:
    - S_real: real runner state vector [H, rho, tau, beta]
    - S_twin: shadow twin predicted state vector
    - ||.||: L2 norm

    Used in P4 divergence analysis.

    SHADOW MODE: Pure computation, no side effects.
    """

    def __init__(self) -> None:
        """Initialize DRS computer."""
        self._previous_state: Optional[Tuple[float, float, float, float]] = None

    def compute(
        self,
        real_H: float,
        real_rho: float,
        real_tau: float,
        real_beta: float,
        twin_H: float,
        twin_rho: float,
        twin_tau: float,
        twin_beta: float,
        delta_t: float = 1.0,
    ) -> float:
        """
        Compute DRS between real and twin states.

        Args:
            real_*: Real runner state components
            twin_*: Twin predicted state components
            delta_t: Time delta for rate normalization

        Returns:
            DRS value in [0, inf) where 0 = no drift
        """
        # Compute L2 norm of state difference
        dH = real_H - twin_H
        drho = real_rho - twin_rho
        dtau = real_tau - twin_tau
        dbeta = real_beta - twin_beta

        l2_norm = math.sqrt(dH**2 + drho**2 + dtau**2 + dbeta**2)

        # Normalize by time delta
        drs = l2_norm / max(delta_t, 0.001)

        return drs

    def compute_components(
        self,
        real_H: float,
        real_rho: float,
        real_tau: float,
        real_beta: float,
        twin_H: float,
        twin_rho: float,
        twin_tau: float,
        twin_beta: float,
    ) -> Dict[str, float]:
        """
        Compute DRS with per-component breakdown.

        Returns:
            Dictionary with drs, H_drift, rho_drift, tau_drift, beta_drift
        """
        dH = abs(real_H - twin_H)
        drho = abs(real_rho - twin_rho)
        dtau = abs(real_tau - twin_tau)
        dbeta = abs(real_beta - twin_beta)

        l2_norm = math.sqrt(dH**2 + drho**2 + dtau**2 + dbeta**2)

        return {
            "drs": l2_norm,
            "H_drift": dH,
            "rho_drift": drho,
            "tau_drift": dtau,
            "beta_drift": dbeta,
        }

    def reset(self) -> None:
        """Reset computer state."""
        self._previous_state = None


def compute_drs(
    real_state: Tuple[float, float, float, float],
    twin_state: Tuple[float, float, float, float],
    delta_t: float = 1.0,
) -> float:
    """
    Compute DRS as standalone function.

    Args:
        real_state: (H, rho, tau, beta) real state
        twin_state: (H, rho, tau, beta) twin state
        delta_t: Time delta for rate normalization

    Returns:
        DRS value in [0, inf)
    """
    diffs = [r - t for r, t in zip(real_state, twin_state)]
    l2_norm = math.sqrt(sum(d**2 for d in diffs))
    return l2_norm / max(delta_t, 0.001)


# =============================================================================
# HSS: Homological Stability Score
# =============================================================================

class HSSComputer:
    """
    Computes Homological Stability Score based on topological feature persistence.

    HSS(t) = persistence_ratio(B_t, B_0)

    Where:
    - B_t: Betti numbers of proof DAG at cycle t
    - B_0: baseline Betti numbers at initialization
    - persistence_ratio(): fraction of preserved features

    SHADOW MODE: Pure computation, no side effects.
    """

    def __init__(self) -> None:
        """Initialize HSS computer."""
        self._baseline_betti: Optional[Tuple[int, int, int]] = None
        self._accumulated_features: int = 0
        self._preserved_features: int = 0

    def compute(
        self,
        b0: int,
        b1: int,
        b2: int = 0,
    ) -> float:
        """
        Compute HSS from current Betti numbers.

        For simplified computation without full persistent homology:
        - b0: connected components (proofs)
        - b1: 1-dimensional holes (cycles in DAG)
        - b2: 2-dimensional voids (rarely non-zero)

        Args:
            b0: Betti-0 (connected components)
            b1: Betti-1 (1-cycles)
            b2: Betti-2 (2-voids)

        Returns:
            HSS value in [0, 1] where 1 = fully preserved
        """
        current_betti = (b0, b1, b2)

        if self._baseline_betti is None:
            self._baseline_betti = current_betti
            return 1.0  # Full stability at baseline

        # Compute feature preservation
        # Features are "preserved" if they don't decrease unexpectedly
        baseline_total = sum(self._baseline_betti)
        current_total = sum(current_betti)

        if baseline_total == 0:
            return 1.0 if current_total == 0 else 0.5

        # Ratio of preserved features
        # Allow growth (new proofs) but penalize unexpected loss
        if current_total >= baseline_total:
            # Growth or stable - high HSS
            hss = 1.0 - min(0.3, (current_total - baseline_total) / (baseline_total + 1) * 0.3)
        else:
            # Loss of features - proportional penalty
            loss_ratio = (baseline_total - current_total) / baseline_total
            hss = 1.0 - loss_ratio

        return max(0.0, min(1.0, hss))

    def compute_from_dag_size(
        self,
        dag_size: int,
        proof_count: int,
    ) -> float:
        """
        Simplified HSS computation from DAG statistics.

        Uses proxy metrics when full Betti numbers unavailable:
        - dag_size: total nodes in proof DAG
        - proof_count: number of completed proofs

        Approximates:
        - b0 ~ proof_count (connected components)
        - b1 ~ dag_size - proof_count - edges (cycles)

        Args:
            dag_size: Number of nodes in proof DAG
            proof_count: Number of proofs

        Returns:
            HSS value in [0, 1]
        """
        # Approximate b0 as proof_count
        # Assume b1 is small relative to dag_size
        b0 = proof_count
        b1 = max(0, dag_size - proof_count * 2)  # Rough cycle estimate
        b2 = 0

        return self.compute(b0, b1, b2)

    def get_baseline(self) -> Optional[Tuple[int, int, int]]:
        """Get baseline Betti numbers."""
        return self._baseline_betti

    def reset(self) -> None:
        """Reset computer state."""
        self._baseline_betti = None
        self._accumulated_features = 0
        self._preserved_features = 0


def compute_hss(
    current_betti: Tuple[int, int, int],
    baseline_betti: Tuple[int, int, int],
) -> float:
    """
    Compute HSS as standalone function.

    Args:
        current_betti: (b0, b1, b2) current Betti numbers
        baseline_betti: (b0, b1, b2) baseline Betti numbers

    Returns:
        HSS value in [0, 1]
    """
    baseline_total = sum(baseline_betti)
    current_total = sum(current_betti)

    if baseline_total == 0:
        return 1.0 if current_total == 0 else 0.5

    if current_total >= baseline_total:
        hss = 1.0 - min(0.3, (current_total - baseline_total) / (baseline_total + 1) * 0.3)
    else:
        loss_ratio = (baseline_total - current_total) / baseline_total
        hss = 1.0 - loss_ratio

    return max(0.0, min(1.0, hss))


# =============================================================================
# Utility Functions
# =============================================================================

def compute_window_statistics(values: List[float]) -> Dict[str, float]:
    """
    Compute mean, min, max, std for a list of values.

    Args:
        values: List of numeric values

    Returns:
        Dictionary with mean, min, max, std
    """
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}

    n = len(values)
    mean = sum(values) / n
    min_val = min(values)
    max_val = max(values)

    if n > 1:
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0

    return {
        "mean": mean,
        "min": min_val,
        "max": max_val,
        "std": std,
    }


def classify_drs_severity(drs: float) -> str:
    """
    Classify DRS severity per TDA_PhaseX_Binding.md.

    Args:
        drs: Drift Rate Score value

    Returns:
        Severity string: "NONE", "INFO", "WARN", "CRITICAL"
    """
    if drs <= 0.05:
        return "NONE"
    elif drs <= 0.10:
        return "INFO"
    elif drs <= 0.20:
        return "WARN"
    else:
        return "CRITICAL"
