"""
PHASE II — NOT USED IN PHASE I

U2 Snapshot Health Tile Adapter
================================

Provides snapshot continuity tile for global health surface.
This tile is advisory only and does NOT influence run decisions.
"""

from typing import Any, Dict, Optional

from experiments.u2.snapshot_history import (
    build_multi_run_snapshot_history,
    plan_future_runs,
    summarize_snapshot_plans_for_global_console,
)


def build_u2_snapshot_tile_for_global_health(
    snapshot_root: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Build U2 snapshot health tile for global health surface.
    
    This tile provides:
    - Snapshot coverage metrics across runs
    - Resume target availability
    - Status light (GREEN/YELLOW/RED)
    - Headline summary
    
    SHADOW MODE CONTRACT:
    - The snapshot tile is purely observational
    - It does NOT influence any other tiles or system health classification
    - No control flow depends on this tile
    - This is advisory only; no run blocking
    
    Args:
        snapshot_root: Optional path to snapshot root directory. If None or invalid,
                      returns a safe "NO_DATA" tile with YELLOW status.
        
    Returns:
        Snapshot tile dict, or None if snapshot_root is not provided.
        If snapshot_root is provided but invalid/empty, returns a safe NO_DATA tile.
    """
    if snapshot_root is None:
        return None
    
    from pathlib import Path
    
    snapshot_root_path = Path(snapshot_root)
    
    # If snapshot root doesn't exist, return safe NO_DATA tile
    if not snapshot_root_path.exists() or not snapshot_root_path.is_dir():
        return {
            "schema_version": "1.0.0",
            "tile_type": "u2_snapshot_plans",
            "status_light": "YELLOW",
            "has_resume_targets": False,
            "runs_analyzed": 0,
            "mean_coverage_pct": 0.0,
            "max_gap": 0,
            "headline": "Snapshot root not found or invalid",
            "data_status": "NO_DATA",
        }
    
    try:
        # Discover run directories
        run_dirs: list[str] = []
        try:
            for item in snapshot_root_path.iterdir():
                if item.is_dir():
                    run_dirs.append(str(item))
        except (OSError, PermissionError) as e:
            # Return safe tile with error info
            return {
                "schema_version": "1.0.0",
                "tile_type": "u2_snapshot_plans",
                "status_light": "YELLOW",
                "has_resume_targets": False,
                "runs_analyzed": 0,
                "mean_coverage_pct": 0.0,
                "max_gap": 0,
                "headline": f"Error scanning snapshot root: {e}",
                "data_status": "ERROR",
                "error": str(e),
            }
        
        if not run_dirs:
            # No runs found - return safe NO_DATA tile
            return {
                "schema_version": "1.0.0",
                "tile_type": "u2_snapshot_plans",
                "status_light": "GREEN",  # GREEN because no data is not a problem
                "has_resume_targets": False,
                "runs_analyzed": 0,
                "mean_coverage_pct": 0.0,
                "max_gap": 0,
                "headline": "No runs found in snapshot root",
                "data_status": "NO_DATA",
            }
        
        # Build multi-run history
        try:
            multi_history = build_multi_run_snapshot_history(run_dirs)
        except Exception as e:
            # Return safe tile with error info
            return {
                "schema_version": "1.0.0",
                "tile_type": "u2_snapshot_plans",
                "status_light": "YELLOW",
                "has_resume_targets": False,
                "runs_analyzed": 0,
                "mean_coverage_pct": 0.0,
                "max_gap": 0,
                "headline": f"Error building snapshot history: {e}",
                "data_status": "ERROR",
                "error": str(e),
            }
        
        # Plan future runs
        try:
            plan = plan_future_runs(multi_history, target_coverage=10.0)
        except Exception as e:
            # Return safe tile with error info
            return {
                "schema_version": "1.0.0",
                "tile_type": "u2_snapshot_plans",
                "status_light": "YELLOW",
                "has_resume_targets": False,
                "runs_analyzed": multi_history.get("run_count", 0),
                "mean_coverage_pct": multi_history.get("summary", {}).get("average_coverage_pct", 0.0),
                "max_gap": multi_history.get("global_max_gap", 0),
                "headline": f"Error planning future runs: {e}",
                "data_status": "ERROR",
                "error": str(e),
            }
        
        # Get console summary
        try:
            console_tile = summarize_snapshot_plans_for_global_console(multi_history, plan)
            # Add data_status to indicate successful analysis
            console_tile["data_status"] = "OK"
            return console_tile
        except Exception as e:
            # Return safe tile with error info
            return {
                "schema_version": "1.0.0",
                "tile_type": "u2_snapshot_plans",
                "status_light": "YELLOW",
                "has_resume_targets": False,
                "runs_analyzed": multi_history.get("run_count", 0),
                "mean_coverage_pct": multi_history.get("summary", {}).get("average_coverage_pct", 0.0),
                "max_gap": multi_history.get("global_max_gap", 0),
                "headline": f"Error summarizing for console: {e}",
                "data_status": "ERROR",
                "error": str(e),
            }
    
    except Exception as e:
        # Catch-all for any unexpected errors
        return {
            "schema_version": "1.0.0",
            "tile_type": "u2_snapshot_plans",
            "status_light": "YELLOW",
            "has_resume_targets": False,
            "runs_analyzed": 0,
            "mean_coverage_pct": 0.0,
            "max_gap": 0,
            "headline": f"Unexpected error: {e}",
            "data_status": "ERROR",
            "error": str(e),
        }


__all__ = [
    "build_u2_snapshot_tile_for_global_health",
]

