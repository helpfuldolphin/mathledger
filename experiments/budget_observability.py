"""
Minimal Budget Observability for Phase II Uplift Experiments.

Provides read-only budget health classification and summary aggregation.
All functions are pure (no side effects) and JSON-serializable.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, Sequence
import json


class BudgetHealthStatus(str, Enum):
    """Neutral budget health classification."""
    SAFE = "SAFE"
    TIGHT = "TIGHT"
    STARVED = "STARVED"
    INVALID = "INVALID"


@dataclass(frozen=True)
class BudgetSummary:
    """
    Read-only budget summary for a single experiment run.
    
    Aggregates budget enforcement statistics from JSONL logs.
    All fields are safe for JSON serialization.
    """
    total_cycles: int
    budget_exhausted_count: int
    max_candidates_hit_count: int
    timeout_abstentions_total: int
    
    @property
    def budget_exhausted_pct(self) -> float:
        """Percentage of cycles that exhausted budget."""
        if self.total_cycles == 0:
            return 0.0
        return (self.budget_exhausted_count / self.total_cycles) * 100.0
    
    @property
    def max_candidates_hit_pct(self) -> float:
        """Percentage of cycles that hit max_candidates limit."""
        if self.total_cycles == 0:
            return 0.0
        return (self.max_candidates_hit_count / self.total_cycles) * 100.0
    
    @property
    def timeout_abstentions_avg(self) -> float:
        """Average timeout abstentions per cycle."""
        if self.total_cycles == 0:
            return 0.0
        return self.timeout_abstentions_total / self.total_cycles
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "total_cycles": self.total_cycles,
            "budget_exhausted_count": self.budget_exhausted_count,
            "max_candidates_hit_count": self.max_candidates_hit_count,
            "timeout_abstentions_total": self.timeout_abstentions_total,
            "budget_exhausted_pct": self.budget_exhausted_pct,
            "max_candidates_hit_pct": self.max_candidates_hit_pct,
            "timeout_abstentions_avg": self.timeout_abstentions_avg,
        }


def classify_budget_health(
    summary: BudgetSummary,
    exhausted_threshold_safe: float = 1.0,
    exhausted_threshold_starved: float = 5.0,
    timeout_threshold_safe: float = 0.1,
    timeout_threshold_starved: float = 1.0,
) -> BudgetHealthStatus:
    """
    Classify budget health status from summary.
    
    Classification rules (neutral, non-judgmental):
    - SAFE: budget_exhausted_pct < 1% AND timeout_abstentions_avg < 0.1
    - TIGHT: (1% ≤ exhausted < 5%) OR (0.1 ≤ timeout < 1.0)
    - STARVED: exhausted ≥ 5% OR timeout ≥ 1.0
    - INVALID: total_cycles == 0 (no data)
    
    Args:
        summary: Budget summary to classify
        exhausted_threshold_safe: Percentage threshold for SAFE (default 1.0%)
        exhausted_threshold_starved: Percentage threshold for STARVED (default 5.0%)
        timeout_threshold_safe: Average timeout threshold for SAFE (default 0.1)
        timeout_threshold_starved: Average timeout threshold for STARVED (default 1.0)
    
    Returns:
        BudgetHealthStatus classification
    """
    if summary.total_cycles == 0:
        return BudgetHealthStatus.INVALID
    
    exhausted_pct = summary.budget_exhausted_pct
    timeout_avg = summary.timeout_abstentions_avg
    
    # STARVED: High budget pressure
    if exhausted_pct >= exhausted_threshold_starved or timeout_avg >= timeout_threshold_starved:
        return BudgetHealthStatus.STARVED
    
    # SAFE: Low budget pressure
    if exhausted_pct < exhausted_threshold_safe and timeout_avg < timeout_threshold_safe:
        return BudgetHealthStatus.SAFE
    
    # TIGHT: Moderate budget pressure
    return BudgetHealthStatus.TIGHT


def summarize_budget_from_logs(
    log_lines: Sequence[str],
) -> Dict[str, Any]:
    """
    Summarize budget usage from JSONL log lines.
    
    Parses JSONL format, extracting budget fields:
    - budget.budget_exhausted (bool per cycle)
    - budget.max_candidates_hit (bool per cycle)
    - budget.timeout_abstentions (int per cycle)
    
    Handles both formats:
    1. Direct budget fields: {"budget": {"budget_exhausted": true, ...}}
    2. Nested in metrics: {"metrics": {"budget": {...}}}
    
    Args:
        log_lines: Sequence of JSONL log lines (strings)
    
    Returns:
        Dictionary with schema_version, status, and summary fields
    """
    total_cycles = 0
    budget_exhausted_count = 0
    max_candidates_hit_count = 0
    timeout_abstentions_total = 0
    
    for line in log_lines:
        line = line.strip()
        if not line:
            continue
        
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        
        # Extract budget data from various possible locations
        budget_data = None
        if "budget" in data:
            budget_data = data["budget"]
        elif "metrics" in data and isinstance(data["metrics"], dict) and "budget" in data["metrics"]:
            budget_data = data["metrics"]["budget"]
        elif "cycle" in data or "cycle_index" in data:
            # Cycle log entry - check for budget fields at top level
            if any(key.startswith("budget_") for key in data.keys()):
                budget_data = {k.replace("budget_", ""): v for k, v in data.items() if k.startswith("budget_")}
        
        if budget_data is None:
            continue
        
        # Count this as a cycle if we found budget data
        total_cycles += 1
        
        # Extract fields
        if budget_data.get("budget_exhausted", False):
            budget_exhausted_count += 1
        
        if budget_data.get("max_candidates_hit", False):
            max_candidates_hit_count += 1
        
        timeout_count = budget_data.get("timeout_abstentions", 0)
        if isinstance(timeout_count, (int, float)):
            timeout_abstentions_total += int(timeout_count)
    
    # Build summary
    summary = BudgetSummary(
        total_cycles=total_cycles,
        budget_exhausted_count=budget_exhausted_count,
        max_candidates_hit_count=max_candidates_hit_count,
        timeout_abstentions_total=timeout_abstentions_total,
    )
    
    # Classify health
    status = classify_budget_health(summary)
    
    return {
        "schema_version": "1.0.0",
        "status": status.value,
        "total_cycles": summary.total_cycles,
        "budget_exhausted_pct": summary.budget_exhausted_pct,
        "timeout_abstentions_avg": summary.timeout_abstentions_avg,
        "summary": summary.to_dict(),
    }

