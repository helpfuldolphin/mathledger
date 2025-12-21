"""
U2 Planner Search Policies and Lean failure summarizers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend.lean_interface import LeanFailureSignal

from rfl.prng import DeterministicPRNG


class SearchPolicy(ABC):
    """
    Abstract base class for search policies.
    
    INVARIANTS:
    - rank() must be deterministic given same PRNG seed
    - Higher scores indicate higher priority
    """
    
    def __init__(self, prng: DeterministicPRNG):
        """
        Initialize policy with PRNG.
        
        Args:
            prng: Deterministic PRNG for policy decisions
        """
        self.prng = prng
    
    @abstractmethod
    def rank(self, candidates: List[Any]) -> List[Tuple[Any, float]]:
        """
        Rank candidates by priority.
        
        Args:
            candidates: List of candidate items
            
        Returns:
            List of (candidate, score) tuples, sorted by score descending
        """
        pass
    
    @abstractmethod
    def get_priority(self, candidate: Any, score: float) -> float:
        """
        Convert score to priority (lower = higher priority).
        
        Args:
            candidate: Candidate item
            score: Policy score
            
        Returns:
            Priority value for frontier queue
        """
        pass


class BaselinePolicy(SearchPolicy):
    """
    Baseline random policy.
    
    Assigns random scores to candidates for unbiased exploration.
    Used as control in RFL experiments.
    """
    
    def rank(self, candidates: List[Any]) -> List[Tuple[Any, float]]:
        """
        Rank candidates randomly.
        
        Args:
            candidates: List of candidate items
            
        Returns:
            List of (candidate, score) tuples with random scores
        """
        scored = [(c, self.prng.random()) for c in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
    
    def get_priority(self, candidate: Any, score: float) -> float:
        """
        Convert score to priority.
        
        For baseline, priority is just inverse of score.
        
        Args:
            candidate: Candidate item
            score: Random score in [0, 1]
            
        Returns:
            Priority (lower = higher priority)
        """
        return 1.0 - score


class RFLPolicy(SearchPolicy):
    """
    RFL (Reflexive Formal Learning) policy.
    
    Uses verifiable feedback to guide search:
    - Success rate of similar candidates
    - Depth-based heuristics
    - Structural features
    
    IMPORTANT: Only uses verifiable feedback (no RLHF, no preferences)
    """
    
    def __init__(
        self,
        prng: DeterministicPRNG,
        feedback_data: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize RFL policy.
        
        Args:
            prng: Deterministic PRNG
            feedback_data: Verifiable feedback from previous runs
        """
        super().__init__(prng)
        self.feedback_data = feedback_data or {}
        
        # Feature weights (learned from verifiable feedback)
        self.weights = {
            "depth": -0.3,  # Prefer shallower
            "complexity": -0.2,  # Prefer simpler
            "success_rate": 0.5,  # Prefer high success rate
        }
    
    def rank(self, candidates: List[Any]) -> List[Tuple[Any, float]]:
        """
        Rank candidates using RFL policy.
        
        Args:
            candidates: List of candidate items
            
        Returns:
            List of (candidate, score) tuples sorted by RFL score
        """
        scored = []
        
        for candidate in candidates:
            # Extract features
            features = self._extract_features(candidate)
            
            # Compute score
            score = self._compute_score(features)
            
            scored.append((candidate, score))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return scored
    
    def get_priority(self, candidate: Any, score: float) -> float:
        """
        Convert RFL score to priority.
        
        Args:
            candidate: Candidate item
            score: RFL score
            
        Returns:
            Priority (lower = higher priority)
        """
        # Invert score to get priority
        # Add small noise for tie-breaking
        noise = self.prng.random() * 1e-6
        return -score + noise
    
    def _extract_features(self, candidate: Any) -> Dict[str, float]:
        """
        Extract features from candidate.
        
        Args:
            candidate: Candidate item
            
        Returns:
            Feature dictionary
        """
        # Default features
        features = {
            "depth": 0.0,
            "complexity": 0.0,
            "success_rate": 0.5,  # Default to neutral
        }
        
        # Extract from candidate if it's a dict
        if isinstance(candidate, dict):
            features["depth"] = float(candidate.get("depth", 0))
            features["complexity"] = float(len(str(candidate)))
            
            # Look up success rate from feedback
            candidate_key = str(candidate.get("item", candidate))
            if candidate_key in self.feedback_data:
                features["success_rate"] = self.feedback_data[candidate_key].get("success_rate", 0.5)
        else:
            # Simple heuristics for non-dict candidates
            features["complexity"] = float(len(str(candidate)))
        
        return features
    
    def _compute_score(self, features: Dict[str, float]) -> float:
        """
        Compute RFL score from features.
        
        Args:
            features: Feature dictionary
            
        Returns:
            RFL score (higher = better)
        """
        score = 0.0
        
        for feature_name, feature_value in features.items():
            weight = self.weights.get(feature_name, 0.0)
            score += weight * feature_value
        
        # Normalize to [0, 1]
        score = 1.0 / (1.0 + abs(score))
        
        return score


def create_policy(
    mode: str,
    prng: DeterministicPRNG,
    feedback_data: Optional[Dict[str, Any]] = None,
) -> SearchPolicy:
    """
    Factory function to create search policy.
    
    Args:
        mode: "baseline" or "rfl"
        prng: Deterministic PRNG
        feedback_data: Feedback data for RFL policy
        
    Returns:
        SearchPolicy instance
    """
    if mode == "baseline":
        return BaselinePolicy(prng)
    elif mode == "rfl":
        return RFLPolicy(prng, feedback_data)
    else:
        raise ValueError(f"Unknown policy mode: {mode}")


# --------------------------------------------------------------------------- #
# Lean failure telemetry summarizers
# --------------------------------------------------------------------------- #

LEAN_FAILURE_TILE_SCHEMA = "lean-failure-tile.v1"
LEAN_FAILURE_TRENDS_SCHEMA = "lean-failure-trend.v1"
_FAILURE_KIND_ORDER = ("timeout", "type_error", "tactic_failure", "unknown")
_REVIEW_THRESHOLD = 0.25


def _normalize_kind(kind: str) -> str:
    kind = str(kind).lower()
    if kind in _FAILURE_KIND_ORDER:
        return kind
    return "unknown"


def summarize_lean_failures_for_global_health(
    signals: Sequence[LeanFailureSignal],
) -> Dict[str, Any]:
    """
    Produce a compact Lean failure tile for global_health.json.
    """
    counts = {kind: 0 for kind in _FAILURE_KIND_ORDER}
    total_elapsed = 0
    for signal in signals:
        counts[_normalize_kind(signal.kind)] += 1
        total_elapsed += max(0, int(signal.elapsed_ms))

    total = sum(counts.values())
    rates = {
        kind: round(counts[kind] / total, 4) if total else 0.0
        for kind in _FAILURE_KIND_ORDER
    }

    status = "OK"
    headline = "Lean failures nominal"
    alerts: List[str] = []

    timeout_rate = rates["timeout"]
    type_rate = rates["type_error"]
    tactic_rate = rates["tactic_failure"]

    if total:
        if timeout_rate >= 0.4 or type_rate >= 0.3:
            status = "BLOCK"
            headline = "Lean failures blocking policy updates"
            if timeout_rate >= 0.4:
                alerts.append("timeout_spike")
            if type_rate >= 0.3:
                alerts.append("type_error_spike")
        elif timeout_rate >= 0.2 or type_rate >= 0.15 or tactic_rate >= 0.3:
            status = "ATTENTION"
            headline = "Lean failure rate rising"
            if timeout_rate >= 0.2:
                alerts.append("timeout_trend")
            if type_rate >= 0.15:
                alerts.append("type_error_trend")
            if tactic_rate >= 0.3:
                alerts.append("tactic_failure_trend")

    avg_duration = round(total_elapsed / total, 2) if total else 0.0

    return {
        "schema_version": LEAN_FAILURE_TILE_SCHEMA,
        "status": status,
        "headline": headline,
        "total_invocations": total,
        "counts": counts,
        "rates": rates,
        "avg_duration_ms": avg_duration,
        "alerts": alerts,
    }


def summarize_lean_failure_trends(
    tiles: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate per-run tiles into high-level trend flags.
    """
    if not tiles:
        return {
            "schema_version": LEAN_FAILURE_TRENDS_SCHEMA,
            "sampled_runs": 0,
            "trend_flags": [],
            "notes": ["no_runs_observed"],
        }

    trend_flags: List[str] = []
    notes: List[str] = []

    def _series(kind: str) -> List[float]:
        return [
            float(tile.get("rates", {}).get(kind, 0.0))
            for tile in tiles
        ]

    timeout_series = _series("timeout")
    type_series = _series("type_error")

    if len(timeout_series) >= 2 and timeout_series[-1] - timeout_series[0] >= 0.1:
        trend_flags.append("timeout_increasing")
    if len(type_series) >= 2 and type_series[-1] - type_series[0] >= 0.08:
        trend_flags.append("type_error_increasing")

    notes.append(f"latest_status:{tiles[-1].get('status', 'UNKNOWN')}")

    return {
        "schema_version": LEAN_FAILURE_TRENDS_SCHEMA,
        "sampled_runs": len(tiles),
        "trend_flags": trend_flags,
        "notes": notes,
    }


def slices_needing_review(
    per_slice_tiles: Dict[str, Dict[str, Any]],
    *,
    threshold: float = _REVIEW_THRESHOLD,
) -> List[str]:
    """
    Identify slices whose Lean failure rate exceeds the threshold.
    """
    flagged: List[str] = []
    for slice_name, tile in per_slice_tiles.items():
        rates = tile.get("rates", {})
        peak = max(
            rates.get("timeout", 0.0),
            rates.get("type_error", 0.0),
            rates.get("tactic_failure", 0.0),
        )
        if peak >= threshold:
            flagged.append(slice_name)
    return sorted(flagged)
