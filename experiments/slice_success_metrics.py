"""
Re-export slice_success_metrics from backend.substrate for backward compatibility.

This module provides a stable import path for experiments and tests.
The canonical implementation lives in backend/substrate/slice_success_metrics.py.
"""

from backend.substrate.slice_success_metrics import (
    compute_goal_hit,
    compute_sparse_success,
    compute_chain_success,
    compute_multi_goal_success,
)

# Also re-export compute_metric if it exists
try:
    from backend.substrate.slice_success_metrics import compute_metric
except ImportError:
    # Provide a shim for compute_metric if not available
    def compute_metric(kind: str, **kwargs):
        """
        Backward-compatible shim for compute_metric.
        Dispatches to the appropriate compute_* function based on kind.
        """
        if kind == "goal_hit":
            return compute_goal_hit(
                kwargs.get("verified_statements", []),
                kwargs.get("target_hashes", set()),
                kwargs.get("min_total_verified", 1),
            )
        elif kind == "sparse_success":
            return compute_sparse_success(
                kwargs.get("verified_count", 0),
                kwargs.get("attempted_count", 0),
                kwargs.get("min_verified", 1),
            )
        elif kind == "chain_success":
            return compute_chain_success(
                kwargs.get("verified_statements", []),
                kwargs.get("dependency_graph", {}),
                kwargs.get("chain_target_hash", ""),
                kwargs.get("min_chain_length", 1),
            )
        elif kind == "multi_goal_success":
            return compute_multi_goal_success(
                kwargs.get("verified_hashes", set()),
                kwargs.get("required_goal_hashes", set()),
            )
        else:
            raise ValueError(f"Unknown metric kind: {kind}")

__all__ = [
    "compute_goal_hit",
    "compute_sparse_success",
    "compute_chain_success",
    "compute_multi_goal_success",
    "compute_metric",
]

