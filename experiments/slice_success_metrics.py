"""
Slice success metrics for curriculum experiments.

Provides metric computation functions for different curriculum slice types.
"""

from typing import Any, Dict, List, Optional


def compute_goal_hit(
    target_hash: str,
    verified_hashes: List[str],
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute goal-hit success metric."""
    hit = target_hash in verified_hashes
    return {
        "metric": "goal_hit",
        "target_hash": target_hash,
        "hit": hit,
        "verified_count": len(verified_hashes),
        "metadata": metadata or {},
    }


def compute_sparse_success(
    verified_hashes: List[str],
    total_candidates: int,
    *,
    threshold: float = 0.1,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute sparse success metric for wide proof spaces."""
    rate = len(verified_hashes) / max(total_candidates, 1)
    return {
        "metric": "sparse_success",
        "verified_count": len(verified_hashes),
        "total_candidates": total_candidates,
        "success_rate": rate,
        "threshold": threshold,
        "passed": rate >= threshold,
        "metadata": metadata or {},
    }


def compute_chain_success(
    chain_depths: List[int],
    *,
    min_depth: int = 2,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute chain success metric for proof trees."""
    max_depth = max(chain_depths) if chain_depths else 0
    avg_depth = sum(chain_depths) / len(chain_depths) if chain_depths else 0
    return {
        "metric": "chain_success",
        "max_depth": max_depth,
        "avg_depth": avg_depth,
        "chain_count": len(chain_depths),
        "min_depth": min_depth,
        "passed": max_depth >= min_depth,
        "metadata": metadata or {},
    }


def compute_multi_goal_success(
    goals: List[str],
    verified_hashes: List[str],
    *,
    min_goals: int = 1,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute multi-goal coordination success metric."""
    goals_hit = [g for g in goals if g in verified_hashes]
    return {
        "metric": "multi_goal_success",
        "goals_total": len(goals),
        "goals_hit": len(goals_hit),
        "goals_hit_list": goals_hit,
        "min_goals": min_goals,
        "passed": len(goals_hit) >= min_goals,
        "metadata": metadata or {},
    }


def compute_metric(kind: str, **kwargs: Any) -> Dict[str, Any]:
    """
    Unified metric computation dispatcher.

    Args:
        kind: Metric type - one of "goal_hit", "sparse_success",
              "chain_success", "multi_goal_success"
        **kwargs: Arguments passed to the specific metric function

    Returns:
        Metric result dictionary
    """
    if kind == "goal_hit":
        return compute_goal_hit(
            kwargs.get("target_hash", ""),
            kwargs.get("verified_hashes", []),
            metadata=kwargs.get("metadata"),
        )
    elif kind == "sparse_success":
        return compute_sparse_success(
            kwargs.get("verified_hashes", []),
            kwargs.get("total_candidates", 0),
            threshold=kwargs.get("threshold", 0.1),
            metadata=kwargs.get("metadata"),
        )
    elif kind == "chain_success":
        return compute_chain_success(
            kwargs.get("chain_depths", []),
            min_depth=kwargs.get("min_depth", 2),
            metadata=kwargs.get("metadata"),
        )
    elif kind == "multi_goal_success":
        return compute_multi_goal_success(
            kwargs.get("goals", []),
            kwargs.get("verified_hashes", []),
            min_goals=kwargs.get("min_goals", 1),
            metadata=kwargs.get("metadata"),
        )
    else:
        return {"metric": kind, "error": f"Unknown metric kind: {kind}"}


__all__ = [
    "compute_goal_hit",
    "compute_sparse_success",
    "compute_chain_success",
    "compute_multi_goal_success",
    "compute_metric",
]

