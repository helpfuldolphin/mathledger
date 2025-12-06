"""
Causal variable extraction from run data.

Computes deltas: Δabstain, Δproof, Δpolicy between consecutive runs.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from datetime import datetime
import numpy as np


@dataclass
class RunMetrics:
    """Metrics from a single derivation run."""
    run_id: int
    started_at: datetime
    ended_at: Optional[datetime]
    policy_hash: Optional[str]
    abstain_pct: float
    proofs_success: int
    proofs_per_sec: float
    depth_max_reached: int
    system: str
    slice_name: str

    def duration_seconds(self) -> float:
        """Compute run duration in seconds."""
        if self.ended_at is None:
            return 0.0
        return (self.ended_at - self.started_at).total_seconds()


@dataclass
class RunDelta:
    """
    Causal variables representing change between two runs.

    Encodes:
    - Δpolicy: whether policy changed (0 = same, 1 = different)
    - Δabstain: change in abstention percentage (percentage points)
    - Δproof: change in successful proofs
    - Δthroughput: change in proofs/sec
    - Δdepth: change in max depth reached
    """
    baseline_run: RunMetrics
    comparison_run: RunMetrics

    # Deltas
    delta_policy: int  # 0 or 1 (binary indicator)
    delta_abstain: float  # percentage points
    delta_proof: int  # absolute count
    delta_throughput: float  # proofs/sec
    delta_depth: int  # depth units

    # Relative changes
    abstain_ratio: float  # comparison / baseline
    proof_ratio: float
    throughput_ratio: float

    @property
    def policy_changed(self) -> bool:
        """True if policy hash differs between runs."""
        return self.delta_policy == 1

    def to_feature_vector(self) -> np.ndarray:
        """
        Convert to feature vector for regression/estimation.

        Returns:
            [delta_policy, delta_abstain, delta_depth, delta_throughput]
        """
        return np.array([
            self.delta_policy,
            self.delta_abstain,
            self.delta_depth,
            self.delta_throughput
        ], dtype=np.float32)


def compute_policy_delta(baseline: RunMetrics, comparison: RunMetrics) -> int:
    """
    Compute policy change indicator.

    Returns:
        1 if policies differ, 0 if same
    """
    if baseline.policy_hash is None or comparison.policy_hash is None:
        # If either is None, conservatively treat as same
        return 0

    return 1 if baseline.policy_hash != comparison.policy_hash else 0


def compute_abstention_delta(baseline: RunMetrics, comparison: RunMetrics) -> float:
    """
    Compute change in abstention percentage.

    Returns:
        Δabstain = comparison.abstain_pct - baseline.abstain_pct (in percentage points)
    """
    return comparison.abstain_pct - baseline.abstain_pct


def compute_throughput_delta(baseline: RunMetrics, comparison: RunMetrics) -> float:
    """
    Compute change in proof throughput.

    Returns:
        Δthroughput = comparison.proofs_per_sec - baseline.proofs_per_sec
    """
    return comparison.proofs_per_sec - baseline.proofs_per_sec


def compute_run_delta(baseline: RunMetrics, comparison: RunMetrics) -> RunDelta:
    """
    Compute all deltas between two runs.

    Args:
        baseline: Earlier run (or control condition)
        comparison: Later run (or treatment condition)

    Returns:
        RunDelta with all computed differences
    """
    delta_policy = compute_policy_delta(baseline, comparison)
    delta_abstain = compute_abstention_delta(baseline, comparison)
    delta_throughput = compute_throughput_delta(baseline, comparison)

    delta_proof = comparison.proofs_success - baseline.proofs_success
    delta_depth = comparison.depth_max_reached - baseline.depth_max_reached

    # Compute ratios (avoid division by zero)
    abstain_ratio = (
        comparison.abstain_pct / baseline.abstain_pct
        if baseline.abstain_pct > 0 else 1.0
    )

    proof_ratio = (
        comparison.proofs_success / baseline.proofs_success
        if baseline.proofs_success > 0 else 1.0
    )

    throughput_ratio = (
        comparison.proofs_per_sec / baseline.proofs_per_sec
        if baseline.proofs_per_sec > 0 else 1.0
    )

    return RunDelta(
        baseline_run=baseline,
        comparison_run=comparison,
        delta_policy=delta_policy,
        delta_abstain=delta_abstain,
        delta_proof=delta_proof,
        delta_throughput=delta_throughput,
        delta_depth=delta_depth,
        abstain_ratio=abstain_ratio,
        proof_ratio=proof_ratio,
        throughput_ratio=throughput_ratio
    )


def extract_run_deltas(runs: List[RunMetrics]) -> List[RunDelta]:
    """
    Extract consecutive run deltas from a time series.

    Args:
        runs: List of runs ordered by started_at (chronological)

    Returns:
        List of deltas between consecutive runs
    """
    if len(runs) < 2:
        return []

    deltas = []
    for i in range(len(runs) - 1):
        baseline = runs[i]
        comparison = runs[i + 1]

        # Only compute delta if they're in the same system and slice
        if (baseline.system == comparison.system and
            baseline.slice_name == comparison.slice_name):
            delta = compute_run_delta(baseline, comparison)
            deltas.append(delta)

    return deltas


def stratify_by_policy_change(deltas: List[RunDelta]) -> Tuple[List[RunDelta], List[RunDelta]]:
    """
    Stratify deltas by whether policy changed.

    Returns:
        (policy_changed_deltas, policy_unchanged_deltas)
    """
    changed = [d for d in deltas if d.policy_changed]
    unchanged = [d for d in deltas if not d.policy_changed]
    return changed, unchanged


def compute_mean_deltas(deltas: List[RunDelta]) -> dict:
    """
    Compute mean values across a set of deltas.

    Returns:
        Dictionary with mean values for all delta metrics
    """
    if not deltas:
        return {
            'mean_delta_abstain': 0.0,
            'mean_delta_throughput': 0.0,
            'mean_delta_proof': 0.0,
            'mean_delta_depth': 0.0,
            'n_deltas': 0
        }

    return {
        'mean_delta_abstain': np.mean([d.delta_abstain for d in deltas]),
        'mean_delta_throughput': np.mean([d.delta_throughput for d in deltas]),
        'mean_delta_proof': np.mean([d.delta_proof for d in deltas]),
        'mean_delta_depth': np.mean([d.delta_depth for d in deltas]),
        'mean_abstain_ratio': np.mean([d.abstain_ratio for d in deltas]),
        'mean_proof_ratio': np.mean([d.proof_ratio for d in deltas]),
        'mean_throughput_ratio': np.mean([d.throughput_ratio for d in deltas]),
        'n_deltas': len(deltas)
    }


def summary_statistics(deltas: List[RunDelta]) -> dict:
    """
    Compute summary statistics for a set of deltas.

    Returns:
        Dictionary with mean, std, min, max for key metrics
    """
    if not deltas:
        return {}

    abstain_deltas = [d.delta_abstain for d in deltas]
    throughput_deltas = [d.delta_throughput for d in deltas]
    proof_deltas = [d.delta_proof for d in deltas]

    return {
        'abstain': {
            'mean': np.mean(abstain_deltas),
            'std': np.std(abstain_deltas),
            'min': np.min(abstain_deltas),
            'max': np.max(abstain_deltas),
        },
        'throughput': {
            'mean': np.mean(throughput_deltas),
            'std': np.std(throughput_deltas),
            'min': np.min(throughput_deltas),
            'max': np.max(throughput_deltas),
        },
        'proof': {
            'mean': np.mean(proof_deltas),
            'std': np.std(proof_deltas),
            'min': np.min(proof_deltas),
            'max': np.max(proof_deltas),
        },
        'n': len(deltas)
    }
