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
    pass

__all__ = [
    "compute_goal_hit",
    "compute_sparse_success", 
    "compute_chain_success",
    "compute_multi_goal_success",
]

