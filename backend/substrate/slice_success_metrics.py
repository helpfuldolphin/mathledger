"""Slice success metrics shim.

Re-exports from experiments.slice_success_metrics for backwards compatibility.
"""

import warnings

warnings.warn(
    "backend.substrate.slice_success_metrics is deprecated; "
    "import experiments.slice_success_metrics instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from canonical location
from experiments.slice_success_metrics import (
    METRIC_SCHEMA_VERSION,
    MetricResult,
    compute_coverage_metric,
    compute_goal_hit,
    compute_metric,
    compute_sparse_success,
    compute_chain_success,
    compute_multi_goal_success,
    compute_velocity_metric,
    validate_metric_params,
)

__all__ = [
    "METRIC_SCHEMA_VERSION",
    "MetricResult",
    "compute_coverage_metric",
    "compute_goal_hit",
    "compute_metric",
    "compute_sparse_success",
    "compute_chain_success",
    "compute_multi_goal_success",
    "compute_velocity_metric",
    "validate_metric_params",
]
