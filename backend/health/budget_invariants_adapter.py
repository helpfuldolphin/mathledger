"""
Budget Invariants Adapter for Global Health Surface.

Maps budget invariant governance output into a global health tile that provides
budget health signals for P3/P4 First-Light runs and evidence chains.

SHADOW MODE CONTRACT:
- This function is read-only (aside from dict construction)
- The returned tile is purely observational
- No control flow depends on the tile contents
- Budget invariants = "Energy Law" of First-Light runs
- Storyline + BNH-Φ = temporal coherence evidence
- These appear in P3 stability reports and P4 calibration bundles

The Budget Invariants tile represents the "Energy Law" of First-Light runs:
- INV-BUD-1 through INV-BUD-5 enforce deterministic budget boundaries
- Timeline aggregation tracks stability across runs
- Storyline + BNH-Φ projection provide temporal coherence evidence
- This signal is essential for P3 stability reports and P4 calibration bundles
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from derivation.budget_invariants import (
    summarize_budget_invariants_for_global_health,
)


def build_budget_invariants_tile_for_global_health(
    invariant_timeline: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build budget invariants tile for global health surface.
    
    Maps budget invariant timeline from build_budget_invariant_timeline() into
    a global health tile format with status_light mapping.
    
    Budget Invariants represent the "Energy Law" of First-Light runs:
    - They enforce deterministic budget boundaries (no post-exhaustion processing,
      hard caps on candidates, monotonic budget remaining)
    - Timeline aggregation tracks stability across runs (stability_index)
    - Storyline + BNH-Φ projection provide temporal coherence evidence
    - These signals appear in P3 stability reports and P4 calibration bundles
    
    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned tile is purely observational
    - No control flow depends on the tile contents
    - Exceptions degrade gracefully (returns minimal tile)
    
    Args:
        invariant_timeline: Budget invariant timeline from build_budget_invariant_timeline().
            Must contain:
            - schema_version: str
            - total_runs: int
            - ok_count, warn_count, fail_count: int
            - inv_bud_1_failures through inv_bud_5_failures: int
            - recent_status: "OK" | "WARN" | "FAIL"
            - stability_index: float [0.0, 1.0]
    
    Returns:
        Budget invariants health tile dictionary with:
        - schema_version: "1.0.0"
        - tile_type: "budget_invariants"
        - status_light: "GREEN" | "YELLOW" | "RED"
        - health_index: float [0.0, 1.0] (derived from stability_index)
        - inv_bud_failures: List[str] (e.g., ["INV-BUD-1", "INV-BUD-3"])
        - stability_trend: "STABLE" | "DRIFTING" | "VOLATILE"
        - headline: str (neutral, factual)
        - total_runs: int
    
    Example:
        >>> timeline = build_budget_invariant_timeline(snapshots)
        >>> health_tile = build_budget_invariants_tile_for_global_health(timeline)
        >>> health_tile["status_light"]
        'GREEN'
    """
    try:
        # Get global health summary (includes invariant failures and status)
        health_summary = summarize_budget_invariants_for_global_health(invariant_timeline)
        
        status = health_summary.get("status", "OK")
        inv_bud_failures = health_summary.get("inv_bud_failures", [])
        stability_index = invariant_timeline.get("stability_index", 1.0)
        total_runs = invariant_timeline.get("total_runs", 0)
        recent_status = invariant_timeline.get("recent_status", "OK")
        
        # Map status to status_light
        # BLOCK → RED, WARN → YELLOW, OK → GREEN
        if status == "BLOCK":
            status_light = "RED"
        elif status == "WARN":
            status_light = "YELLOW"
        else:
            status_light = "GREEN"
        
        # Derive health_index from stability_index (0.0-1.0 scale)
        health_index = stability_index
        
        # Determine stability_trend from recent_status and stability_index
        if stability_index >= 0.95 and recent_status == "OK":
            stability_trend = "STABLE"
        elif stability_index < 0.7 or recent_status == "FAIL":
            stability_trend = "VOLATILE"
        else:
            stability_trend = "DRIFTING"
        
        # Build neutral headline
        headline_parts = []
        headline_parts.append(f"Budget invariants: {status}")
        
        if inv_bud_failures:
            headline_parts.append(f"{len(inv_bud_failures)} invariant(s) flagged")
        
        if stability_trend != "STABLE":
            headline_parts.append(f"stability trend: {stability_trend}")
        
        if total_runs > 0:
            headline_parts.append(f"{total_runs} run(s) analyzed")
        
        headline = ". ".join(headline_parts) + "."
        
        return {
            "schema_version": "1.0.0",
            "tile_type": "budget_invariants",
            "status_light": status_light,
            "health_index": health_index,
            "inv_bud_failures": inv_bud_failures,
            "stability_trend": stability_trend,
            "headline": headline,
            "total_runs": total_runs,
        }
    
    except Exception:
        # SHADOW MODE: Degrade gracefully, return minimal tile
        return {
            "schema_version": "1.0.0",
            "tile_type": "budget_invariants",
            "status_light": "YELLOW",
            "health_index": 0.5,
            "inv_bud_failures": [],
            "stability_trend": "UNKNOWN",
            "headline": "Budget invariants: status unavailable",
            "total_runs": 0,
        }

