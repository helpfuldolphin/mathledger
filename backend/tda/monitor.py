"""
TDA Monitor for Phase X Integration

Provides integrated TDA monitoring for P3 (MetricsWindow) and P4 (DivergenceAnalyzer).

See: docs/system_law/TDA_PhaseX_Binding.md Section 3-4

SHADOW MODE CONTRACT:
- All monitoring is observational only
- No governance modification based on TDA values
- Red-flags are logged only, never enforced
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from backend.tda.metrics import (
    SNSComputer,
    PCSComputer,
    DRSComputer,
    HSSComputer,
    TDAMetrics,
    TDAWindowMetrics,
    compute_window_statistics,
    classify_drs_severity,
)

__all__ = [
    "TDAMonitor",
    "TDARedFlag",
    "TDASummary",
]


@dataclass
class TDARedFlag:
    """
    TDA red-flag observation (LOGGED ONLY).

    SHADOW MODE: Red-flags are observational only, never enforced.
    """

    cycle: int
    timestamp: str
    flag_type: str  # TDA_SNS_ANOMALY, TDA_PCS_COLLAPSE, TDA_HSS_DEGRADATION, TDA_ENVELOPE_EXIT
    severity: str  # INFO, WARN, CRITICAL
    observed_value: float
    threshold: float
    consecutive_cycles: int = 0
    action: str = "LOGGED_ONLY"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "cycle": self.cycle,
            "timestamp": self.timestamp,
            "flag_type": self.flag_type,
            "severity": self.severity,
            "observed_value": round(self.observed_value, 6),
            "threshold": round(self.threshold, 6),
            "consecutive_cycles": self.consecutive_cycles,
            "action": self.action,
            "mode": "SHADOW",
        }


@dataclass
class TDASummary:
    """
    Summary of TDA metrics across a run.

    SHADOW MODE: Summary is observational only.
    """

    # Aggregate statistics
    total_cycles: int = 0

    # SNS summary
    sns_mean: float = 0.0
    sns_max: float = 0.0
    sns_anomaly_count: int = 0

    # PCS summary
    pcs_mean: float = 1.0
    pcs_min: float = 1.0
    pcs_collapse_count: int = 0

    # DRS summary (P4 only)
    drs_mean: float = 0.0
    drs_max: float = 0.0
    drs_critical_count: int = 0

    # HSS summary
    hss_mean: float = 1.0
    hss_min: float = 1.0
    hss_degradation_count: int = 0

    # Envelope summary
    envelope_occupancy: float = 1.0
    envelope_exit_total: int = 0
    max_envelope_exit_streak: int = 0

    # Red-flag counts
    total_red_flags: int = 0
    red_flags_by_type: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "total_cycles": self.total_cycles,
            "sns": {
                "mean": round(self.sns_mean, 6),
                "max": round(self.sns_max, 6),
                "anomaly_count": self.sns_anomaly_count,
            },
            "pcs": {
                "mean": round(self.pcs_mean, 6),
                "min": round(self.pcs_min, 6),
                "collapse_count": self.pcs_collapse_count,
            },
            "drs": {
                "mean": round(self.drs_mean, 6),
                "max": round(self.drs_max, 6),
                "critical_count": self.drs_critical_count,
            },
            "hss": {
                "mean": round(self.hss_mean, 6),
                "min": round(self.hss_min, 6),
                "degradation_count": self.hss_degradation_count,
            },
            "envelope": {
                "occupancy": round(self.envelope_occupancy, 6),
                "exit_total": self.envelope_exit_total,
                "max_exit_streak": self.max_envelope_exit_streak,
            },
            "red_flags": {
                "total": self.total_red_flags,
                "by_type": dict(self.red_flags_by_type),
            },
            "mode": "SHADOW",
        }


class TDAMonitor:
    """
    Integrated TDA monitoring for Phase X.

    Provides:
    1. Per-cycle TDA metric computation (SNS, PCS, DRS, HSS)
    2. Window-level aggregation for P3 stability envelopes
    3. DRS-based divergence context for P4
    4. Red-flag observation (LOGGED ONLY)

    SHADOW MODE CONTRACT:
    - All monitoring is observational only
    - No governance modification
    - Red-flags are logged, never enforced
    """

    # TDA Thresholds per spec
    SNS_ELEVATED_THRESHOLD = 0.4
    SNS_ANOMALY_THRESHOLD = 0.6
    PCS_LOW_COHERENCE_THRESHOLD = 0.6
    PCS_INCOHERENT_THRESHOLD = 0.4
    DRS_INFO_THRESHOLD = 0.05
    DRS_WARN_THRESHOLD = 0.10
    DRS_CRITICAL_THRESHOLD = 0.20
    HSS_UNSTABLE_THRESHOLD = 0.6
    HSS_DEGRADATION_THRESHOLD = 0.4

    # Envelope thresholds (Omega_TDA)
    ENVELOPE_SNS_MAX = 0.4
    ENVELOPE_PCS_MIN = 0.6
    ENVELOPE_HSS_MIN = 0.6

    def __init__(
        self,
        sns_history_window: int = 100,
        pcs_coherence_window: int = 20,
    ) -> None:
        """
        Initialize TDA monitor.

        Args:
            sns_history_window: Pattern history window for SNS
            pcs_coherence_window: Coherence window for PCS
        """
        self._sns_computer = SNSComputer(history_window=sns_history_window)
        self._pcs_computer = PCSComputer(coherence_window=pcs_coherence_window)
        self._drs_computer = DRSComputer()
        self._hss_computer = HSSComputer()

        # Cycle tracking
        self._cycle_count = 0
        self._metrics_history: List[TDAMetrics] = []

        # Window tracking
        self._current_window_metrics: List[TDAMetrics] = []

        # Envelope streak tracking
        self._envelope_exit_streak = 0
        self._max_envelope_exit_streak = 0

        # Red-flag tracking
        self._red_flags: List[TDARedFlag] = []
        self._red_flag_counts: Dict[str, int] = {
            "TDA_SNS_ANOMALY": 0,
            "TDA_PCS_COLLAPSE": 0,
            "TDA_HSS_DEGRADATION": 0,
            "TDA_ENVELOPE_EXIT": 0,
        }

        # Accumulator for summary
        self._sns_sum = 0.0
        self._sns_max = 0.0
        self._pcs_sum = 0.0
        self._pcs_min = 1.0
        self._drs_sum = 0.0
        self._drs_max = 0.0
        self._hss_sum = 0.0
        self._hss_min = 1.0
        self._envelope_in_count = 0

    def observe_cycle(
        self,
        cycle: int,
        success: bool,
        depth: int,
        H: float,
        rho: float,
        tau: float,
        beta: float,
        dag_size: int = 0,
        proof_count: int = 0,
        proof_hash: Optional[str] = None,
    ) -> TDAMetrics:
        """
        Observe a P3 cycle and compute TDA metrics.

        Args:
            cycle: Cycle number
            success: Whether proof succeeded
            depth: Proof depth
            H: Health metric
            rho: RSI value
            tau: Threshold
            beta: Block rate
            dag_size: Size of proof DAG (for HSS)
            proof_count: Number of proofs (for HSS)
            proof_hash: Optional proof hash

        Returns:
            TDAMetrics for this cycle
        """
        self._cycle_count = cycle

        # Compute SNS
        sns = self._sns_computer.compute(success, depth, proof_hash)

        # Compute PCS
        pcs = self._pcs_computer.compute(success, rho, H)

        # Compute HSS (simplified without full Betti numbers)
        if dag_size > 0 or proof_count > 0:
            hss = self._hss_computer.compute_from_dag_size(dag_size, proof_count)
        else:
            # Use proxy based on H and rho stability
            hss = 0.5 * H + 0.5 * rho

        # DRS not computed in P3 (no twin)
        drs = 0.0

        # Check envelope membership
        in_envelope = (
            sns <= self.ENVELOPE_SNS_MAX and
            pcs >= self.ENVELOPE_PCS_MIN and
            hss >= self.ENVELOPE_HSS_MIN
        )

        metrics = TDAMetrics(
            sns=sns,
            pcs=pcs,
            drs=drs,
            hss=hss,
            in_tda_envelope=in_envelope,
            sns_threshold=self.ENVELOPE_SNS_MAX,
            pcs_threshold=self.ENVELOPE_PCS_MIN,
            hss_threshold=self.ENVELOPE_HSS_MIN,
        )

        # Track in window
        self._current_window_metrics.append(metrics)
        self._metrics_history.append(metrics)

        # Update accumulators
        self._sns_sum += sns
        self._sns_max = max(self._sns_max, sns)
        self._pcs_sum += pcs
        self._pcs_min = min(self._pcs_min, pcs)
        self._hss_sum += hss
        self._hss_min = min(self._hss_min, hss)
        if in_envelope:
            self._envelope_in_count += 1

        # Track envelope streaks
        if not in_envelope:
            self._envelope_exit_streak += 1
            self._max_envelope_exit_streak = max(
                self._max_envelope_exit_streak,
                self._envelope_exit_streak
            )
        else:
            self._envelope_exit_streak = 0

        # Check for red-flags (LOGGED ONLY)
        self._check_red_flags(cycle, metrics)

        return metrics

    def observe_divergence(
        self,
        cycle: int,
        real_H: float,
        real_rho: float,
        real_tau: float,
        real_beta: float,
        twin_H: float,
        twin_rho: float,
        twin_tau: float,
        twin_beta: float,
    ) -> Dict[str, Any]:
        """
        Observe P4 divergence and compute DRS with TDA context.

        Args:
            cycle: Cycle number
            real_*: Real runner state components
            twin_*: Twin predicted state components

        Returns:
            Dictionary with DRS and component breakdown
        """
        self._cycle_count = cycle

        # Compute DRS with components
        drs_result = self._drs_computer.compute_components(
            real_H, real_rho, real_tau, real_beta,
            twin_H, twin_rho, twin_tau, twin_beta,
        )

        drs = drs_result["drs"]

        # Update accumulators
        self._drs_sum += drs
        self._drs_max = max(self._drs_max, drs)

        # Classify severity
        severity = classify_drs_severity(drs)

        return {
            "drs": drs,
            "severity": severity,
            "components": {
                "H_drift": drs_result["H_drift"],
                "rho_drift": drs_result["rho_drift"],
                "tau_drift": drs_result["tau_drift"],
                "beta_drift": drs_result["beta_drift"],
            },
        }

    def finalize_window(self, window_index: int) -> TDAWindowMetrics:
        """
        Finalize current window and compute aggregated TDA metrics.

        Called by MetricsWindow.finalize() or at window boundaries.

        Args:
            window_index: Index of the window being finalized

        Returns:
            TDAWindowMetrics for the window
        """
        if not self._current_window_metrics:
            # Empty window
            return TDAWindowMetrics(window_index=window_index)

        # Extract trajectories
        sns_values = [m.sns for m in self._current_window_metrics]
        pcs_values = [m.pcs for m in self._current_window_metrics]
        hss_values = [m.hss for m in self._current_window_metrics]

        # Compute statistics
        sns_stats = compute_window_statistics(sns_values)
        pcs_stats = compute_window_statistics(pcs_values)
        hss_stats = compute_window_statistics(hss_values)

        # Count threshold violations
        sns_anomaly_count = sum(1 for v in sns_values if v > self.SNS_ANOMALY_THRESHOLD)
        sns_elevated_count = sum(1 for v in sns_values if v > self.SNS_ELEVATED_THRESHOLD)
        pcs_low_count = sum(1 for v in pcs_values if v < self.PCS_LOW_COHERENCE_THRESHOLD)
        pcs_incoherent_count = sum(1 for v in pcs_values if v < self.PCS_INCOHERENT_THRESHOLD)
        hss_unstable_count = sum(1 for v in hss_values if v < self.HSS_UNSTABLE_THRESHOLD)
        hss_degradation_count = sum(1 for v in hss_values if v < self.HSS_DEGRADATION_THRESHOLD)

        # Envelope metrics
        envelope_in_count = sum(1 for m in self._current_window_metrics if m.in_tda_envelope)
        envelope_occupancy = envelope_in_count / len(self._current_window_metrics)

        # Count exit events and max streak within window
        exit_count = 0
        current_streak = 0
        max_streak = 0
        for m in self._current_window_metrics:
            if not m.in_tda_envelope:
                if current_streak == 0:
                    exit_count += 1
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        # Determine start/end cycles
        n_metrics = len(self._current_window_metrics)
        base_cycle = self._cycle_count - n_metrics + 1
        start_cycle = max(1, base_cycle)
        end_cycle = self._cycle_count

        # Get Betti numbers from HSS computer
        baseline_betti = self._hss_computer.get_baseline()
        if baseline_betti:
            b0, b1, b2 = baseline_betti
        else:
            b0, b1, b2 = 1, 0, 0

        window_metrics = TDAWindowMetrics(
            window_index=window_index,
            window_start_cycle=start_cycle,
            window_end_cycle=end_cycle,
            # SNS
            sns_mean=sns_stats["mean"],
            sns_max=sns_stats["max"],
            sns_min=sns_stats["min"],
            sns_std=sns_stats["std"],
            sns_anomaly_count=sns_anomaly_count,
            sns_elevated_count=sns_elevated_count,
            # PCS
            pcs_mean=pcs_stats["mean"],
            pcs_max=pcs_stats["max"],
            pcs_min=pcs_stats["min"],
            pcs_std=pcs_stats["std"],
            pcs_low_coherence_count=pcs_low_count,
            pcs_incoherent_count=pcs_incoherent_count,
            # HSS
            hss_mean=hss_stats["mean"],
            hss_max=hss_stats["max"],
            hss_min=hss_stats["min"],
            hss_std=hss_stats["std"],
            hss_degradation_count=hss_degradation_count,
            hss_unstable_count=hss_unstable_count,
            betti_b0=b0,
            betti_b1=b1,
            betti_b2=b2,
            # Envelope
            envelope_occupancy_rate=envelope_occupancy,
            envelope_exit_count=exit_count,
            max_envelope_exit_streak=max_streak,
            # Red-flags (window-specific)
            tda_sns_anomaly_flags=sns_anomaly_count,
            tda_pcs_collapse_flags=pcs_incoherent_count,
            tda_hss_degradation_flags=hss_degradation_count,
            tda_envelope_exit_flags=exit_count,
            # Trajectories
            sns_trajectory=sns_values,
            pcs_trajectory=pcs_values,
            hss_trajectory=hss_values,
        )

        # Clear window buffer
        self._current_window_metrics.clear()

        return window_metrics

    def get_summary(self) -> TDASummary:
        """
        Get summary of all TDA observations.

        Returns:
            TDASummary with accumulated statistics
        """
        n = self._cycle_count
        if n == 0:
            return TDASummary()

        return TDASummary(
            total_cycles=n,
            sns_mean=self._sns_sum / n,
            sns_max=self._sns_max,
            sns_anomaly_count=self._red_flag_counts.get("TDA_SNS_ANOMALY", 0),
            pcs_mean=self._pcs_sum / n,
            pcs_min=self._pcs_min,
            pcs_collapse_count=self._red_flag_counts.get("TDA_PCS_COLLAPSE", 0),
            drs_mean=self._drs_sum / n if self._drs_sum > 0 else 0.0,
            drs_max=self._drs_max,
            drs_critical_count=0,  # Would track in P4
            hss_mean=self._hss_sum / n,
            hss_min=self._hss_min,
            hss_degradation_count=self._red_flag_counts.get("TDA_HSS_DEGRADATION", 0),
            envelope_occupancy=self._envelope_in_count / n,
            envelope_exit_total=n - self._envelope_in_count,
            max_envelope_exit_streak=self._max_envelope_exit_streak,
            total_red_flags=sum(self._red_flag_counts.values()),
            red_flags_by_type=dict(self._red_flag_counts),
        )

    def get_red_flags(self) -> List[TDARedFlag]:
        """Get all observed red-flags."""
        return list(self._red_flags)

    def get_current_metrics(self) -> Optional[TDAMetrics]:
        """Get most recent TDA metrics."""
        if self._metrics_history:
            return self._metrics_history[-1]
        return None

    def reset(self) -> None:
        """Reset all monitor state."""
        self._sns_computer.reset()
        self._pcs_computer.reset()
        self._drs_computer.reset()
        self._hss_computer.reset()

        self._cycle_count = 0
        self._metrics_history.clear()
        self._current_window_metrics.clear()

        self._envelope_exit_streak = 0
        self._max_envelope_exit_streak = 0

        self._red_flags.clear()
        self._red_flag_counts = {
            "TDA_SNS_ANOMALY": 0,
            "TDA_PCS_COLLAPSE": 0,
            "TDA_HSS_DEGRADATION": 0,
            "TDA_ENVELOPE_EXIT": 0,
        }

        self._sns_sum = 0.0
        self._sns_max = 0.0
        self._pcs_sum = 0.0
        self._pcs_min = 1.0
        self._drs_sum = 0.0
        self._drs_max = 0.0
        self._hss_sum = 0.0
        self._hss_min = 1.0
        self._envelope_in_count = 0

    def _check_red_flags(self, cycle: int, metrics: TDAMetrics) -> None:
        """
        Check for TDA red-flag conditions (LOGGED ONLY).

        SHADOW MODE: Red-flags are observed and logged, never enforced.
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        # SNS anomaly check
        if metrics.sns > self.SNS_ANOMALY_THRESHOLD:
            flag = TDARedFlag(
                cycle=cycle,
                timestamp=timestamp,
                flag_type="TDA_SNS_ANOMALY",
                severity="CRITICAL",
                observed_value=metrics.sns,
                threshold=self.SNS_ANOMALY_THRESHOLD,
            )
            self._red_flags.append(flag)
            self._red_flag_counts["TDA_SNS_ANOMALY"] += 1

        # PCS collapse check
        if metrics.pcs < self.PCS_INCOHERENT_THRESHOLD:
            flag = TDARedFlag(
                cycle=cycle,
                timestamp=timestamp,
                flag_type="TDA_PCS_COLLAPSE",
                severity="CRITICAL",
                observed_value=metrics.pcs,
                threshold=self.PCS_INCOHERENT_THRESHOLD,
            )
            self._red_flags.append(flag)
            self._red_flag_counts["TDA_PCS_COLLAPSE"] += 1

        # HSS degradation check
        if metrics.hss < self.HSS_DEGRADATION_THRESHOLD:
            flag = TDARedFlag(
                cycle=cycle,
                timestamp=timestamp,
                flag_type="TDA_HSS_DEGRADATION",
                severity="CRITICAL",
                observed_value=metrics.hss,
                threshold=self.HSS_DEGRADATION_THRESHOLD,
            )
            self._red_flags.append(flag)
            self._red_flag_counts["TDA_HSS_DEGRADATION"] += 1

        # Envelope exit check (100+ consecutive)
        if self._envelope_exit_streak >= 100:
            flag = TDARedFlag(
                cycle=cycle,
                timestamp=timestamp,
                flag_type="TDA_ENVELOPE_EXIT",
                severity="WARN",
                observed_value=float(self._envelope_exit_streak),
                threshold=100.0,
                consecutive_cycles=self._envelope_exit_streak,
            )
            self._red_flags.append(flag)
            self._red_flag_counts["TDA_ENVELOPE_EXIT"] += 1
