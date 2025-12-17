"""
PHASE II — NOT USED IN PHASE I

Snapshot History & Multi-Run Analytics
========================================

Provides:
- Single-run snapshot history ledger
- Multi-run snapshot aggregation
- Run planning advisor
- Orchestrator adapter
- Global console health adapter

All functions are read-only and advisory (no mutation of snapshots).
"""

import json
import logging
import statistics
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .snapshots import (
    SnapshotData,
    SnapshotValidationError,
    load_snapshot,
    find_latest_snapshot,
)

logger = logging.getLogger(__name__)

# Schema versions for forward compatibility
SNAPSHOT_HISTORY_SCHEMA_VERSION = "1.0.0"
MULTI_RUN_SCHEMA_VERSION = "1.0.0"


class SnapshotHealthStatus(str, Enum):
    """Health status for a snapshot or run."""
    OK = "OK"
    WARN = "WARN"
    BLOCK = "BLOCK"
    EMPTY = "EMPTY"


@dataclass
class SnapshotInfo:
    """Information about a single snapshot."""
    path: Path
    cycle_index: int
    total_cycles: int
    health: str  # "VALID", "CORRUPTED", "INVALID", "UNREADABLE"
    error: Optional[str] = None


def build_snapshot_history(
    run_dir: Path,
    include_manifest_validation: bool = True,
) -> Dict[str, Any]:
    """
    Build snapshot history ledger for a single run.
    
    Args:
        run_dir: Directory containing snapshots
        include_manifest_validation: Whether to validate against manifest
        
    Returns:
        History dict with schema_version, snapshots, coverage metrics, status
    """
    snapshots_dir = run_dir / "snapshots"
    if not snapshots_dir.exists():
        return {
            "schema_version": SNAPSHOT_HISTORY_SCHEMA_VERSION,
            "run_dir": str(run_dir),
            "snapshots": [],
            "valid_count": 0,
            "corrupted_count": 0,
            "coverage_pct": 0.0,
            "max_gap": 0,
            "avg_gap": 0.0,
            "status": SnapshotHealthStatus.EMPTY.value,
            "recommended_resume_point": None,
        }
    
    # Discover snapshots (try multiple patterns)
    snapshot_files = (
        list(snapshots_dir.glob("*.json")) + 
        list(snapshots_dir.glob("*.snap")) +
        list(snapshots_dir.glob("*_cycle_*.json"))  # Pattern from create_snapshot_name
    )
    # Remove duplicates
    snapshot_files = list(set(snapshot_files))
    snapshot_infos: List[SnapshotInfo] = []
    
    for snapshot_path in sorted(snapshot_files):
        try:
            snapshot = load_snapshot(snapshot_path, verify_hash=True)
            snapshot_infos.append(SnapshotInfo(
                path=snapshot_path,
                cycle_index=snapshot.current_cycle,
                total_cycles=snapshot.total_cycles,
                health="VALID",
            ))
        except SnapshotValidationError as e:
            snapshot_infos.append(SnapshotInfo(
                path=snapshot_path,
                cycle_index=0,
                total_cycles=0,
                health="CORRUPTED",
                error=str(e),
            ))
        except Exception as e:
            snapshot_infos.append(SnapshotInfo(
                path=snapshot_path,
                cycle_index=0,
                total_cycles=0,
                health="UNREADABLE",
                error=str(e),
            ))
    
    # Compute metrics
    valid_snapshots = [s for s in snapshot_infos if s.health == "VALID"]
    corrupted_count = len([s for s in snapshot_infos if s.health == "CORRUPTED"])
    
    if not valid_snapshots:
        return {
            "schema_version": SNAPSHOT_HISTORY_SCHEMA_VERSION,
            "run_dir": str(run_dir),
            "snapshots": [{"path": str(s.path), "cycle": s.cycle_index, "health": s.health} for s in snapshot_infos],
            "valid_count": 0,
            "corrupted_count": corrupted_count,
            "coverage_pct": 0.0,
            "max_gap": 0,
            "avg_gap": 0.0,
            "status": SnapshotHealthStatus.EMPTY.value,
            "recommended_resume_point": None,
        }
    
    # Compute coverage
    total_cycles = valid_snapshots[0].total_cycles
    cycles_with_snapshots = len(valid_snapshots)
    coverage_pct = (cycles_with_snapshots / total_cycles * 100.0) if total_cycles > 0 else 0.0
    
    # Compute gaps
    cycles = sorted([s.cycle_index for s in valid_snapshots])
    gaps = [cycles[i+1] - cycles[i] for i in range(len(cycles) - 1)]
    max_gap = max(gaps) if gaps else 0
    avg_gap = sum(gaps) / len(gaps) if gaps else 0.0
    
    # Determine status
    if corrupted_count > 0 or max_gap > total_cycles * 0.5:
        status = SnapshotHealthStatus.WARN.value
    elif coverage_pct < 10.0:
        status = SnapshotHealthStatus.WARN.value
    else:
        status = SnapshotHealthStatus.OK.value
    
    # Recommend resume point (latest valid snapshot)
    latest = valid_snapshots[-1]
    recommended_resume_point = {
        "path": str(latest.path),
        "cycle": latest.cycle_index,
        "total_cycles": latest.total_cycles,
    }
    
    return {
        "schema_version": SNAPSHOT_HISTORY_SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "snapshots": [{"path": str(s.path), "cycle": s.cycle_index, "health": s.health} for s in snapshot_infos],
        "valid_count": len(valid_snapshots),
        "corrupted_count": corrupted_count,
        "coverage_pct": round(coverage_pct, 1),
        "max_gap": max_gap,
        "avg_gap": round(avg_gap, 1),
        "status": status,
        "recommended_resume_point": recommended_resume_point,
    }


def build_multi_run_snapshot_history(
    run_dirs: Sequence[str],
    include_manifest_validation: bool = True,
) -> Dict[str, Any]:
    """
    Build multi-run snapshot history aggregating multiple experiment runs.
    
    Args:
        run_dirs: List of run directory paths
        include_manifest_validation: Whether to validate against manifests
        
    Returns:
        Multi-run history with per-run summaries and global metrics
    """
    runs: List[Dict[str, Any]] = []
    runs_with_block_status = 0
    global_max_gap = 0
    status_counts: Dict[str, int] = {}
    total_valid = 0
    total_corrupted = 0
    coverage_values: List[float] = []
    
    for run_dir_str in run_dirs:
        run_dir = Path(run_dir_str)
        if not run_dir.exists() or not run_dir.is_dir():
            logger.warning(f"Skipping non-existent or non-directory: {run_dir_str}")
            continue
        
        try:
            history = build_snapshot_history(run_dir, include_manifest_validation)
            runs.append(history)
            
            # Aggregate metrics
            status = history.get("status", SnapshotHealthStatus.EMPTY.value)
            status_counts[status] = status_counts.get(status, 0) + 1
            
            if status == SnapshotHealthStatus.BLOCK.value:
                runs_with_block_status += 1
            
            global_max_gap = max(global_max_gap, history.get("max_gap", 0))
            total_valid += history.get("valid_count", 0)
            total_corrupted += history.get("corrupted_count", 0)
            
            coverage_pct = history.get("coverage_pct", 0.0)
            if coverage_pct > 0:
                coverage_values.append(coverage_pct)
        
        except Exception as e:
            # Log with context about which run failed
            error_context = f"Error analyzing run {run_dir_str}: {e}"
            logger.warning(error_context)
            # Add empty entry for failed analysis with error details
            runs.append({
                "schema_version": SNAPSHOT_HISTORY_SCHEMA_VERSION,
                "run_dir": str(run_dir),
                "status": SnapshotHealthStatus.EMPTY.value,
                "error": str(e),
                "error_type": type(e).__name__,
            })
    
    # Compute overall status
    if not runs or all(r.get("status") == SnapshotHealthStatus.EMPTY.value for r in runs):
        overall_status = SnapshotHealthStatus.EMPTY.value
    elif runs_with_block_status > 0:
        overall_status = SnapshotHealthStatus.BLOCK.value
    elif any(r.get("status") == SnapshotHealthStatus.WARN.value for r in runs):
        overall_status = SnapshotHealthStatus.WARN.value
    else:
        overall_status = SnapshotHealthStatus.OK.value
    
    avg_coverage = sum(coverage_values) / len(coverage_values) if coverage_values else 0.0
    
    return {
        "schema_version": MULTI_RUN_SCHEMA_VERSION,
        "run_count": len(runs),
        "runs": runs,
        "runs_with_block_status": runs_with_block_status,
        "global_max_gap": global_max_gap,
        "overall_status": overall_status,
        "status_counts": status_counts,
        "summary": {
            "total_valid_snapshots": total_valid,
            "total_corrupted_snapshots": total_corrupted,
            "average_coverage_pct": round(avg_coverage, 1),
            "runs_with_resume_points": sum(
                1 for r in runs 
                if r.get("recommended_resume_point", {}).get("path")
            ),
        },
    }


def plan_future_runs(
    multi_history: Dict[str, Any],
    target_coverage: float = 10.0,
) -> Dict[str, Any]:
    """
    Plan future runs based on multi-run snapshot history.
    
    Args:
        multi_history: Output from build_multi_run_snapshot_history
        target_coverage: Target coverage percentage
        
    Returns:
        Plan dict with runs_to_extend, suggested_new_runs, message
    """
    runs = multi_history.get("runs", [])
    if not runs:
        return {
            "runs_to_extend": [],
            "suggested_new_runs": 1,
            "message": "No runs found. Suggest creating new run.",
            "priority": [],
            "target_coverage": target_coverage,
            "current_avg_coverage": 0.0,
        }
    
    # Identify runs to extend
    runs_to_extend: List[Dict[str, Any]] = []
    
    for run in runs:
        if run.get("status") == SnapshotHealthStatus.EMPTY.value:
            continue
        
        resume_point = run.get("recommended_resume_point")
        if not resume_point or not resume_point.get("path"):
            continue
        
        coverage_pct = run.get("coverage_pct", 0.0)
        max_gap = run.get("max_gap", 0)
        total_cycles = resume_point.get("total_cycles", 0)
        current_cycle = resume_point.get("cycle", 0)
        
        # Calculate priority score (higher = more important to extend)
        coverage_deficit = max(0, target_coverage - coverage_pct)
        gap_penalty = max_gap * 0.1
        progress_bonus = (current_cycle / total_cycles) * 0.2 if total_cycles > 0 else 0
        
        priority_score = coverage_deficit * 2.0 - gap_penalty + progress_bonus
        
        runs_to_extend.append({
            "run_dir": run.get("run_dir"),
            "resume_point": resume_point,
            "coverage_pct": coverage_pct,
            "max_gap": max_gap,
            "priority_score": round(priority_score, 2),
        })
    
    # Sort by priority (highest first)
    runs_to_extend.sort(key=lambda r: r["priority_score"], reverse=True)
    
    # Determine if new runs are needed
    avg_coverage = multi_history.get("summary", {}).get("average_coverage_pct", 0.0)
    suggested_new_runs = 0
    
    if avg_coverage < target_coverage * 0.5:
        suggested_new_runs = 1
    
    message = f"Found {len(runs_to_extend)} runs to extend. "
    if suggested_new_runs > 0:
        message += f"Suggest creating {suggested_new_runs} new run(s) for better coverage."
    else:
        message += "Coverage is adequate."
    
    return {
        "runs_to_extend": runs_to_extend,
        "suggested_new_runs": suggested_new_runs,
        "message": message,
        "priority": runs_to_extend[:5],  # Top 5
        "target_coverage": target_coverage,
        "current_avg_coverage": round(avg_coverage, 1),
    }


def summarize_snapshot_plans_for_u2_orchestrator(
    plan: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Summarize snapshot plans for U2 orchestrator consumption.
    
    Args:
        plan: Output from plan_future_runs
        
    Returns:
        Orchestrator summary with has_resume_targets, preferred_run_id, preferred_snapshot_path, status
    """
    runs_to_extend = plan.get("runs_to_extend", [])
    has_resume_targets = len(runs_to_extend) > 0
    
    if not has_resume_targets:
        return {
            "has_resume_targets": False,
            "preferred_run_id": None,
            "preferred_snapshot_path": None,
            "status": "NEW_RUN",
            "details": {
                "runs_available": 0,
                "suggested_new_runs": plan.get("suggested_new_runs", 0),
                "message": plan.get("message", "No viable resume points"),
            },
        }
    
    # Select highest priority run
    top_run = runs_to_extend[0]
    resume_point = top_run.get("resume_point", {})
    
    # Extract run_id from run_dir path
    run_dir = top_run.get("run_dir", "")
    run_id = Path(run_dir).name if run_dir else None
    
    return {
        "has_resume_targets": True,
        "preferred_run_id": run_id,
        "preferred_snapshot_path": resume_point.get("path"),
        "status": "RESUME",
        "details": {
            "runs_available": len(runs_to_extend),
            "suggested_new_runs": plan.get("suggested_new_runs", 0),
            "message": plan.get("message", ""),
        },
    }


def summarize_snapshot_plans_for_global_console(
    history: Dict[str, Any],
    plan: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Summarize snapshot plans for global console/health dashboard.
    
    Args:
        history: Output from build_multi_run_snapshot_history
        plan: Output from plan_future_runs
        
    Returns:
        Console summary with status_light, metrics, headline
    """
    runs_analyzed = history.get("run_count", 0)
    has_resume_targets = len(plan.get("runs_to_extend", [])) > 0
    mean_coverage_pct = history.get("summary", {}).get("average_coverage_pct", 0.0)
    max_gap = history.get("global_max_gap", 0)
    
    # Determine status light
    if has_resume_targets and mean_coverage_pct >= 70.0:
        status_light = "GREEN"
    elif has_resume_targets and mean_coverage_pct >= 30.0:
        status_light = "YELLOW"
    elif has_resume_targets:
        status_light = "YELLOW"  # Low coverage but has targets
    elif mean_coverage_pct < 10.0:
        status_light = "RED"
    else:
        status_light = "YELLOW"
    
    # Generate headline
    if has_resume_targets:
        headline = f"{len(plan.get('runs_to_extend', []))} runs resumable, {mean_coverage_pct:.1f}% avg coverage"
    else:
        headline = f"No resumable runs, {mean_coverage_pct:.1f}% avg coverage"
    
    return {
        "schema_version": "1.0.0",
        "tile_type": "snapshot_health",
        "status_light": status_light,
        "has_resume_targets": has_resume_targets,
        "runs_analyzed": runs_analyzed,
        "mean_coverage_pct": round(mean_coverage_pct, 1),
        "max_gap": max_gap,
        "headline": headline,
    }


def build_calibration_experiment_runbook(
    multi_history: Dict[str, Any],
    plan: Dict[str, Any],
    experiment_type: str,
) -> Dict[str, Any]:
    """
    Build runbook summary for a specific calibration experiment (CAL-EXP-1, 2, or 3).
    
    This extends the base runbook with experiment-specific metrics and analysis.
    
    Args:
        multi_history: Output from build_multi_run_snapshot_history
        plan: Output from plan_future_runs
        experiment_type: "CAL-EXP-1", "CAL-EXP-2", or "CAL-EXP-3"
        
    Returns:
        Calibration experiment runbook with experiment-specific fields
    """
    base_runbook = build_snapshot_runbook_summary(multi_history, plan)
    
    # Add experiment-specific fields
    base_runbook["experiment_type"] = experiment_type
    base_runbook["calibration_metrics"] = {
        "runs_analyzed": multi_history.get("run_count", 0),
        "mean_coverage_pct": multi_history.get("summary", {}).get("average_coverage_pct", 0.0),
        "max_gap": multi_history.get("global_max_gap", 0),
        "overall_status": multi_history.get("overall_status", "EMPTY"),
    }
    
    # Experiment-specific analysis
    if experiment_type == "CAL-EXP-1":
        # Warm-start: focus on short-window stability
        base_runbook["experiment_focus"] = "short_window_stability"
        base_runbook["stability_indicators"] = {
            "coverage_consistency": "high" if base_runbook["mean_coverage_pct"] > 50.0 else "low",
            "gap_risk": "low" if base_runbook["max_gap"] < 20 else "medium" if base_runbook["max_gap"] < 50 else "high",
        }
    elif experiment_type == "CAL-EXP-2":
        # Long-window convergence: focus on coverage progression
        base_runbook["experiment_focus"] = "long_window_convergence"
        base_runbook["convergence_indicators"] = {
            "coverage_trend": "stable" if base_runbook["mean_coverage_pct"] > 70.0 else "improving" if base_runbook["mean_coverage_pct"] > 40.0 else "degrading",
            "gap_stability": "stable" if base_runbook["max_gap"] < 30 else "unstable",
        }
    elif experiment_type == "CAL-EXP-3":
        # Regime-change probes: focus on resilience
        base_runbook["experiment_focus"] = "regime_change_resilience"
        base_runbook["resilience_indicators"] = {
            "coverage_robustness": "high" if base_runbook["mean_coverage_pct"] > 60.0 else "medium" if base_runbook["mean_coverage_pct"] > 30.0 else "low",
            "gap_tolerance": "acceptable" if base_runbook["max_gap"] < 40 else "concerning",
        }
    else:
        base_runbook["experiment_focus"] = "unknown"
    
    return base_runbook


def compare_multi_run_snapshots(
    multi_history: Dict[str, Any],
    previous_multi_history: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compare multiple runs to detect stability deltas, max_gap issues, and coverage regression.
    
    This function provides multi-run comparison analysis for calibration experiments.
    
    Args:
        multi_history: Current multi-run snapshot history
        previous_multi_history: Optional previous history for delta comparison
        
    Returns:
        Comparison dict with stability_deltas, max_gap_analysis, coverage_regression
    """
    runs = multi_history.get("runs", [])
    current_mean_coverage = multi_history.get("summary", {}).get("average_coverage_pct", 0.0)
    current_max_gap = multi_history.get("global_max_gap", 0)
    
    # Stability deltas: compare coverage across runs
    coverage_values = []
    gap_values = []
    status_counts: Dict[str, int] = {}
    
    for run in runs:
        coverage = run.get("coverage_pct", 0.0)
        max_gap = run.get("max_gap", 0)
        status = run.get("status", "EMPTY")
        
        if coverage > 0:
            coverage_values.append(coverage)
        if max_gap > 0:
            gap_values.append(max_gap)
        status_counts[status] = status_counts.get(status, 0) + 1
    
    # Compute stability metrics
    coverage_std = 0.0
    coverage_mean = 0.0
    if coverage_values:
        coverage_mean = statistics.mean(coverage_values)
        if len(coverage_values) > 1:
            coverage_std = statistics.stdev(coverage_values)
    
    gap_mean = 0.0
    if gap_values:
        gap_mean = statistics.mean(gap_values)
    
    # Max gap detection: identify runs with problematic gaps
    problematic_gaps: List[Dict[str, Any]] = []
    for run in runs:
        max_gap = run.get("max_gap", 0)
        total_cycles = run.get("recommended_resume_point", {}).get("total_cycles", 0)
        if max_gap > 0 and total_cycles > 0:
            gap_ratio = max_gap / total_cycles
            if gap_ratio > 0.5:  # Gap is more than 50% of total cycles
                problematic_gaps.append({
                    "run_dir": run.get("run_dir"),
                    "max_gap": max_gap,
                    "total_cycles": total_cycles,
                    "gap_ratio": round(gap_ratio, 3),
                    "risk_level": "high" if gap_ratio > 0.7 else "medium",
                })
    
    # Coverage regression: detect decreasing coverage
    coverage_regression = {
        "detected": False,
        "severity": "none",
        "runs_affected": 0,
        "mean_degradation": 0.0,
    }
    
    if previous_multi_history:
        previous_mean_coverage = previous_multi_history.get("summary", {}).get("average_coverage_pct", 0.0)
        if previous_mean_coverage > 0 and current_mean_coverage < previous_mean_coverage:
            degradation = previous_mean_coverage - current_mean_coverage
            coverage_regression = {
                "detected": True,
                "severity": "high" if degradation > 20.0 else "medium" if degradation > 10.0 else "low",
                "runs_affected": multi_history.get("run_count", 0),
                "mean_degradation": round(degradation, 1),
                "previous_mean": round(previous_mean_coverage, 1),
                "current_mean": round(current_mean_coverage, 1),
            }
    
    # Stability deltas
    stability_deltas = {
        "coverage_mean": round(coverage_mean, 1),
        "coverage_std": round(coverage_std, 1),
        "coverage_stability": "stable" if coverage_std < 5.0 else "moderate" if coverage_std < 15.0 else "unstable",
        "gap_mean": round(gap_mean, 1),
        "status_distribution": status_counts,
    }
    
    return {
        "schema_version": "1.0.0",
        "stability_deltas": stability_deltas,
        "max_gap_analysis": {
            "global_max_gap": current_max_gap,
            "problematic_gaps": problematic_gaps,
            "problematic_count": len(problematic_gaps),
        },
        "coverage_regression": coverage_regression,
        "comparison_timestamp": multi_history.get("schema_version", "1.0.0"),  # Use schema version as proxy
    }


def classify_calibration_trend(
    comparison_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Classify calibration trend from multi-run comparison analysis.
    
    Returns a verdict (IMPROVING | STABLE | REGRESSING | INCONCLUSIVE) with
    explainability: contributing metrics, confidence score, and human-readable rationale.
    
    SHADOW MODE CONTRACT:
    - Verdicts are advisory only; no acceptance wiring
    - Purely observational and explanatory
    
    Args:
        comparison_result: Output from compare_multi_run_snapshots
        
    Returns:
        Trend verdict dict with:
        - verdict: "IMPROVING" | "STABLE" | "REGRESSING" | "INCONCLUSIVE"
        - confidence: float in [0, 1]
        - contributing_metrics: List of metric contributions
        - rationale: Human-readable explanation
        - top_signals: Top 3 contributing signals
    """
    stability_deltas = comparison_result.get("stability_deltas", {})
    max_gap_analysis = comparison_result.get("max_gap_analysis", {})
    coverage_regression = comparison_result.get("coverage_regression", {})
    
    # Extract key metrics
    coverage_mean = stability_deltas.get("coverage_mean", 0.0)
    coverage_std = stability_deltas.get("coverage_std", 0.0)
    coverage_stability = stability_deltas.get("coverage_stability", "unknown")
    gap_mean = stability_deltas.get("gap_mean", 0.0)
    global_max_gap = max_gap_analysis.get("global_max_gap", 0)
    problematic_count = max_gap_analysis.get("problematic_count", 0)
    
    regression_detected = coverage_regression.get("detected", False)
    regression_severity = coverage_regression.get("severity", "none")
    regression_degradation = coverage_regression.get("mean_degradation", 0.0)
    
    # Collect contributing metrics
    contributing_metrics: List[Dict[str, Any]] = []
    signals: List[Dict[str, Any]] = []
    
    # Signal 1: Coverage regression
    if regression_detected:
        weight = 0.4 if regression_severity == "high" else 0.3 if regression_severity == "medium" else 0.2
        contributing_metrics.append({
            "metric": "coverage_regression",
            "value": regression_degradation,
            "weight": weight,
            "direction": "negative",
            "description": f"Coverage decreased by {regression_degradation:.1f}% ({regression_severity} severity)",
        })
        signals.append({
            "signal": "coverage_regression",
            "strength": regression_severity,
            "impact": weight,
            "message": f"Coverage regression detected: {regression_degradation:.1f}% degradation",
        })
    else:
        # Check if coverage is improving (no previous history, but current is good)
        if coverage_mean > 70.0 and coverage_std < 5.0:
            weight = 0.25
            contributing_metrics.append({
                "metric": "coverage_high_stable",
                "value": coverage_mean,
                "weight": weight,
                "direction": "positive",
                "description": f"High and stable coverage: {coverage_mean:.1f}% (std: {coverage_std:.1f}%)",
            })
            signals.append({
                "signal": "coverage_high_stable",
                "strength": "strong",
                "impact": weight,
                "message": f"High stable coverage: {coverage_mean:.1f}%",
            })
    
    # Signal 2: Coverage stability
    if coverage_stability == "stable":
        weight = 0.2
        contributing_metrics.append({
            "metric": "coverage_stability",
            "value": coverage_std,
            "weight": weight,
            "direction": "positive",
            "description": f"Stable coverage across runs (std: {coverage_std:.1f}%)",
        })
        signals.append({
            "signal": "coverage_stability",
            "strength": "strong",
            "impact": weight,
            "message": f"Coverage stability: std {coverage_std:.1f}%",
        })
    elif coverage_stability == "unstable":
        weight = 0.25
        contributing_metrics.append({
            "metric": "coverage_instability",
            "value": coverage_std,
            "weight": weight,
            "direction": "negative",
            "description": f"Unstable coverage across runs (std: {coverage_std:.1f}%)",
        })
        signals.append({
            "signal": "coverage_instability",
            "strength": "strong",
            "impact": weight,
            "message": f"Coverage instability: std {coverage_std:.1f}%",
        })
    
    # Signal 3: Gap analysis
    if problematic_count > 0:
        weight = 0.2
        contributing_metrics.append({
            "metric": "problematic_gaps",
            "value": problematic_count,
            "weight": weight,
            "direction": "negative",
            "description": f"{problematic_count} run(s) with problematic gaps (>50% of cycles)",
        })
        signals.append({
            "signal": "problematic_gaps",
            "strength": "medium" if problematic_count == 1 else "strong",
            "impact": weight,
            "message": f"{problematic_count} run(s) with problematic gaps",
        })
    elif global_max_gap < 20:
        weight = 0.15
        contributing_metrics.append({
            "metric": "gap_health",
            "value": global_max_gap,
            "weight": weight,
            "direction": "positive",
            "description": f"Healthy gap distribution (max gap: {global_max_gap} cycles)",
        })
        signals.append({
            "signal": "gap_health",
            "strength": "medium",
            "impact": weight,
            "message": f"Healthy gaps: max {global_max_gap} cycles",
        })
    
    # Signal 4: Gap mean trend
    if gap_mean < 15.0:
        weight = 0.1
        contributing_metrics.append({
            "metric": "gap_mean_low",
            "value": gap_mean,
            "weight": weight,
            "direction": "positive",
            "description": f"Low average gap: {gap_mean:.1f} cycles",
        })
        signals.append({
            "signal": "gap_mean_low",
            "strength": "weak",
            "impact": weight,
            "message": f"Low average gap: {gap_mean:.1f} cycles",
        })
    
    # Classify verdict
    positive_weight = sum(
        m["weight"] for m in contributing_metrics if m["direction"] == "positive"
    )
    negative_weight = sum(
        m["weight"] for m in contributing_metrics if m["direction"] == "negative"
    )
    
    # Confidence calculation
    total_weight = positive_weight + negative_weight
    if total_weight == 0:
        confidence = 0.0
        verdict = "INCONCLUSIVE"
        rationale = "Insufficient data to determine trend"
    else:
        # Confidence based on signal strength and agreement
        signal_agreement = abs(positive_weight - negative_weight) / total_weight
        confidence = min(0.95, 0.5 + (signal_agreement * 0.45))
        
        # Verdict logic
        if regression_detected and regression_severity in ["high", "medium"]:
            verdict = "REGRESSING"
            rationale = (
                f"Coverage regression detected ({regression_severity} severity: "
                f"{regression_degradation:.1f}% degradation). "
                f"Problematic gaps: {problematic_count}. "
                f"Coverage stability: {coverage_stability}."
            )
        elif negative_weight > positive_weight + 0.1:
            verdict = "REGRESSING"
            rationale = (
                f"Negative signals outweigh positive ones. "
                f"Coverage stability: {coverage_stability}. "
                f"Problematic gaps: {problematic_count}. "
                f"Gap mean: {gap_mean:.1f} cycles."
            )
        elif positive_weight > negative_weight + 0.15:
            verdict = "IMPROVING"
            rationale = (
                f"Positive signals indicate improvement. "
                f"Coverage: {coverage_mean:.1f}% (stability: {coverage_stability}). "
                f"Gap health: max {global_max_gap} cycles, mean {gap_mean:.1f} cycles. "
                f"No regression detected."
            )
        elif abs(positive_weight - negative_weight) < 0.1:
            verdict = "STABLE"
            rationale = (
                f"Balanced signals indicate stability. "
                f"Coverage: {coverage_mean:.1f}% (stability: {coverage_stability}). "
                f"Gap mean: {gap_mean:.1f} cycles. "
                f"Problematic gaps: {problematic_count}."
            )
        else:
            verdict = "INCONCLUSIVE"
            rationale = (
                f"Mixed signals prevent clear classification. "
                f"Coverage: {coverage_mean:.1f}% (stability: {coverage_stability}). "
                f"Gap mean: {gap_mean:.1f} cycles."
            )
    
    # Sort signals by impact (descending) and take top 3
    signals_sorted = sorted(signals, key=lambda s: s["impact"], reverse=True)
    top_signals = signals_sorted[:3]
    
    return {
        "schema_version": "1.0.0",
        "verdict": verdict,
        "confidence": round(confidence, 3),
        "contributing_metrics": contributing_metrics,
        "rationale": rationale,
        "top_signals": top_signals,
        "weight_summary": {
            "positive_weight": round(positive_weight, 3),
            "negative_weight": round(negative_weight, 3),
            "total_weight": round(total_weight, 3),
        },
    }


def build_snapshot_runbook_summary(
    multi_history: Dict[str, Any],
    plan: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a compact runbook summary explaining why a particular resume choice was made.
    
    This summary is designed for inclusion in First Light evidence packs and internal
    ops runbooks. It provides machine-readable explanation of snapshot planning decisions.
    
    SHADOW MODE CONTRACT:
    - This is explanation-only; no changes to snapshot selection behavior
    - Purely observational and advisory
    
    Args:
        multi_history: Output from build_multi_run_snapshot_history
        plan: Output from plan_future_runs
        
    Returns:
        Runbook summary dict with schema_version, runs_analyzed, preferred_run_id,
        preferred_snapshot_path, mean_coverage_pct, max_gap, and reason
    """
    # Get orchestrator summary to extract preferred choices
    orchestrator_summary = summarize_snapshot_plans_for_u2_orchestrator(plan)
    
    # Extract metrics
    runs_analyzed = multi_history.get("run_count", 0)
    mean_coverage_pct = multi_history.get("summary", {}).get("average_coverage_pct", 0.0)
    max_gap = multi_history.get("global_max_gap", 0)
    preferred_run_id = orchestrator_summary.get("preferred_run_id")
    preferred_snapshot_path = orchestrator_summary.get("preferred_snapshot_path")
    status = orchestrator_summary.get("status", "NO_ACTION")
    has_resume_targets = orchestrator_summary.get("has_resume_targets", False)
    
    # Generate reason string
    if status == "RESUME" and preferred_run_id:
        # Find the preferred run's details for context
        runs_to_extend = plan.get("runs_to_extend", [])
        preferred_run = next(
            (r for r in runs_to_extend if preferred_run_id in r.get("run_dir", "")),
            None
        )
        
        if preferred_run:
            coverage = preferred_run.get("coverage_pct", 0.0)
            priority_score = preferred_run.get("priority_score", 0.0)
            reason = (
                f"Selected run '{preferred_run_id}' for resume: "
                f"coverage {coverage:.1f}% (priority score {priority_score:.2f}), "
                f"mean coverage across all runs {mean_coverage_pct:.1f}%, "
                f"max gap {max_gap} cycles"
            )
        else:
            reason = (
                f"Selected run '{preferred_run_id}' for resume: "
                f"mean coverage {mean_coverage_pct:.1f}%, max gap {max_gap} cycles"
            )
    elif status == "NEW_RUN":
        if runs_analyzed == 0:
            reason = "No runs found in snapshot root; starting new run"
        elif not has_resume_targets:
            reason = (
                f"No viable resume targets found among {runs_analyzed} runs; "
                f"mean coverage {mean_coverage_pct:.1f}%, max gap {max_gap} cycles"
            )
        else:
            reason = (
                f"Decision: NEW_RUN despite {runs_analyzed} runs analyzed; "
                f"mean coverage {mean_coverage_pct:.1f}%, max gap {max_gap} cycles"
            )
    else:
        reason = (
            f"Status: {status}; {runs_analyzed} runs analyzed, "
            f"mean coverage {mean_coverage_pct:.1f}%, max gap {max_gap} cycles"
        )
    
    return {
        "schema_version": "1.0.0",
        "runs_analyzed": runs_analyzed,
        "preferred_run_id": preferred_run_id,
        "preferred_snapshot_path": preferred_snapshot_path,
        "mean_coverage_pct": round(mean_coverage_pct, 1),
        "max_gap": max_gap,
        "reason": reason,
        "status": status,
        "has_resume_targets": has_resume_targets,
    }

