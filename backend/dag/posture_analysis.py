# backend/dag/posture_analysis.py
"""
PHASE III - Functions for analyzing DAG posture history and drift.
"""
from typing import Any, Dict, List, Sequence
import operator
import yaml

# Add project root for local imports
import sys
from pathlib import Path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.dag.preflight_check import compare_dag_postures

# Configuration for timeline trend detection
TIMELINE_THRESHOLDS = {
    "sustained_regression_duration": 3,
    "explosive_growth_threshold": 1.5,
    "explosive_growth_duration": 2,
}

# --- Policy-Driven Drift Gate ---

OPERATORS = {
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    ">": operator.gt,
}

def _get_nested_metric(metric_name: str, old, new, comp):
    """Safely gets a value from old, new, or comparison dicts using dot notation."""
    source_str, key = metric_name.split('.', 1)
    source_map = {"old": old, "new": new, "comparison": comp}
    
    if source_str not in source_map:
        raise KeyError(f"Invalid metric source '{source_str}' in policy rule.")
        
    # In case the key itself has dots, we don't support that for now.
    return source_map[source_str].get(key)

def _rule_conditions_met(rule: Dict, old: Dict, new: Dict, comp: Dict) -> bool:
    """Evaluates if all conditions for a given rule are met."""
    for condition in rule.get("conditions", []):
        metric_val = _get_nested_metric(condition["metric"], old, new, comp)
        op_str = condition["operator"]
        policy_val = condition["value"]

        if op_str not in OPERATORS:
            raise ValueError(f"Unsupported operator '{op_str}' in policy rule.")
        
        op_func = OPERATORS[op_str]

        if metric_val is None: # Metric not present in data, condition fails
            return False
            
        if not op_func(metric_val, policy_val):
            return False # A single failed condition means the rule doesn't match
            
    return True # All conditions passed

def load_drift_policy(path: Path) -> Dict[str, Any]:
    """Loads a drift policy from a YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def evaluate_dag_drift_acceptability(
    old_posture: Dict[str, Any],
    new_posture: Dict[str, Any],
    policy: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluates DAG drift acceptability against a declarative policy.
    """
    comparison = compare_dag_postures(old_posture, new_posture)
    
    # Add computed metrics to the comparison dict for policy evaluation
    old_vertices = old_posture.get("vertex_count", 0)
    if old_vertices > 0 and comparison["vertex_count_delta"] > 0:
        comparison["vertex_growth_ratio"] = comparison["vertex_count_delta"] / old_vertices
    else:
        comparison["vertex_growth_ratio"] = 0.0

    # Evaluate rules sequentially
    for rule in policy.get("rules", []):
        if _rule_conditions_met(rule, old_posture, new_posture, comparison):
            # First matching rule determines the outcome
            return {
                "drift_status": rule["status"],
                "reasons": [rule["name"]],
                "comparison": comparison,
            }
            
    # If no rules match, return the default status
    return {
        "drift_status": policy.get("default_status", "OK"),
        "reasons": ["Default status: No rules matched."],
        "comparison": comparison,
    }


# --- Historical Timeline Analysis ---

def build_dag_posture_timeline(
    snapshots: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Builds a historical ledger and trend analysis from a sequence of posture snapshots.
    """
    if not snapshots:
        return {"timeline": [], "aggregates": {}, "trend_flags": {}}

    sorted_snapshots = sorted(snapshots, key=lambda s: s.get("timestamp", 0))

    eligible_count = sum(1 for s in sorted_snapshots if s.get("drift_eligible", False))
    
    timeline_entries = []
    positive_depth_deltas = 0
    negative_depth_deltas = 0
    
    for i in range(len(sorted_snapshots)):
        current_posture = sorted_snapshots[i]
        entry = {"posture": current_posture}
        
        if i > 0:
            previous_posture = sorted_snapshots[i-1]
            comparison = compare_dag_postures(previous_posture, current_posture)
            entry["comparison_from_previous"] = comparison
            
            if comparison["depth_delta"] > 0:
                positive_depth_deltas += 1
            elif comparison["depth_delta"] < 0:
                negative_depth_deltas += 1
        
        timeline_entries.append(entry)

    # Compute trend flags
    sustained_depth_regression = False
    regression_streak = 0
    for entry in timeline_entries:
        delta = entry.get("comparison_from_previous", {}).get("depth_delta", 0)
        regression_streak = regression_streak + 1 if delta < 0 else 0
        if regression_streak >= TIMELINE_THRESHOLDS["sustained_regression_duration"]:
            sustained_depth_regression = True
            break
            
    explosive_vertex_growth = False
    growth_streak = 0
    for entry in timeline_entries:
        comp = entry.get("comparison_from_previous", {})
        v_delta = comp.get("vertex_count_delta", 0)
        old_v_count = entry["posture"].get("vertex_count", 0) - v_delta
        
        growth_ratio = v_delta / old_v_count if old_v_count > 0 and v_delta > 0 else 0
        growth_streak = growth_streak + 1 if growth_ratio > TIMELINE_THRESHOLDS["explosive_growth_threshold"] else 0
        if growth_streak >= TIMELINE_THRESHOLDS["explosive_growth_duration"]:
            explosive_vertex_growth = True
            break

    return {
        "timeline": timeline_entries,
        "aggregates": {
            "total_snapshots": len(sorted_snapshots),
            "eligible_count": eligible_count,
            "ineligible_count": len(sorted_snapshots) - eligible_count,
            "positive_depth_delta_periods": positive_depth_deltas,
            "negative_depth_delta_periods": negative_depth_deltas,
        },
        "trend_flags": {
            "sustained_depth_regression": sustained_depth_regression,
            "explosive_vertex_growth": explosive_vertex_growth,
        }
    }
