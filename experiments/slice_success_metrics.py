"""
Slice success metrics module.

Provides metric computation for slice success analysis.
This is the canonical location for slice success metrics.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


METRIC_SCHEMA_VERSION = "1.0.0"


@dataclass
class MetricResult:
    """Result of a metric computation."""
    value: float
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


def compute_goal_hit(
    achieved: float,
    target: float,
    tolerance: float = 0.0,
) -> Dict[str, Any]:
    """Compute goal hit metric."""
    hit = achieved >= (target - tolerance)
    return {
        "achieved": achieved,
        "target": target,
        "tolerance": tolerance,
        "hit": hit,
        "gap": target - achieved if not hit else 0.0,
    }


def compute_sparse_success(
    successes: int,
    total: int,
    min_rate: float = 0.0,
) -> Dict[str, Any]:
    """Compute sparse success metric."""
    rate = successes / total if total > 0 else 0.0
    return {
        "successes": successes,
        "total": total,
        "rate": rate,
        "passes_threshold": rate >= min_rate,
    }


def compute_chain_success(
    chain_results: List[bool],
) -> Dict[str, Any]:
    """Compute chain success metric."""
    success_count = sum(1 for r in chain_results if r)
    total = len(chain_results)
    all_success = all(chain_results) if chain_results else False
    return {
        "success_count": success_count,
        "total": total,
        "all_success": all_success,
        "chain_rate": success_count / total if total > 0 else 0.0,
    }


def compute_multi_goal_success(
    goals: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute multi-goal success metric."""
    results = []
    for goal in goals:
        achieved = goal.get("achieved", 0.0)
        target = goal.get("target", 0.0)
        results.append(compute_goal_hit(achieved, target))

    hits = sum(1 for r in results if r["hit"])
    return {
        "goals": results,
        "total_goals": len(goals),
        "goals_hit": hits,
        "all_hit": hits == len(goals) if goals else True,
    }


def compute_coverage_metric(
    covered: int,
    total: int,
) -> Dict[str, Any]:
    """Compute coverage metric."""
    rate = covered / total if total > 0 else 0.0
    return {
        "covered": covered,
        "total": total,
        "coverage_rate": rate,
    }


def compute_velocity_metric(
    count: int,
    duration_seconds: float,
) -> Dict[str, Any]:
    """Compute velocity metric."""
    rate = count / duration_seconds if duration_seconds > 0 else 0.0
    return {
        "count": count,
        "duration_seconds": duration_seconds,
        "rate_per_second": rate,
        "rate_per_hour": rate * 3600,
    }


def compute_metric(kind: str, **kwargs: Any) -> Dict[str, Any]:
    """Compute metric by kind."""
    if kind == "goal_hit":
        return compute_goal_hit(
            achieved=kwargs.get("achieved", 0.0),
            target=kwargs.get("target", 0.0),
            tolerance=kwargs.get("tolerance", 0.0),
        )
    elif kind == "sparse_success":
        return compute_sparse_success(
            successes=kwargs.get("successes", 0),
            total=kwargs.get("total", 0),
            min_rate=kwargs.get("min_rate", 0.0),
        )
    elif kind == "chain_success":
        return compute_chain_success(
            chain_results=kwargs.get("chain_results", []),
        )
    elif kind == "multi_goal":
        return compute_multi_goal_success(
            goals=kwargs.get("goals", []),
        )
    elif kind == "coverage":
        return compute_coverage_metric(
            covered=kwargs.get("covered", 0),
            total=kwargs.get("total", 0),
        )
    elif kind == "velocity":
        return compute_velocity_metric(
            count=kwargs.get("count", 0),
            duration_seconds=kwargs.get("duration_seconds", 0.0),
        )
    else:
        return {"error": f"Unknown metric kind: {kind}"}


def validate_metric_params(kind: str, params: Dict[str, Any]) -> bool:
    """Validate metric parameters."""
    required = {
        "goal_hit": ["achieved", "target"],
        "sparse_success": ["successes", "total"],
        "chain_success": ["chain_results"],
        "multi_goal": ["goals"],
        "coverage": ["covered", "total"],
        "velocity": ["count", "duration_seconds"],
    }
    required_keys = required.get(kind, [])
    return all(k in params for k in required_keys)


__all__ = [
    "METRIC_SCHEMA_VERSION",
    "MetricResult",
    "compute_goal_hit",
    "compute_sparse_success",
    "compute_chain_success",
    "compute_multi_goal_success",
    "compute_coverage_metric",
    "compute_velocity_metric",
    "compute_metric",
    "validate_metric_params",
]
