"""
Budget Invariant Governance — Snapshot, Timeline, Storyline, and Release Gating

This module provides pure, deterministic functions for:
- Building budget invariant snapshots from PipelineStats
- Aggregating snapshots into timelines
- Summarizing for global health surfaces (MAAS, Director Console)
- Building governance views for cross-layer aggregation
- Evaluating release readiness signals
- Generating storylines and projections (BNH-Φ)

All functions are:
- Pure (no I/O, no side effects)
- Deterministic (same input → same output)
- JSON-serializable (return Dict[str, Any])

Author: Agent B1 (Budget Enforcement Architect)
Date: 2025-01-XX
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

# Schema version for invariant snapshots and governance objects
BUDGET_SCHEMA_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Core Invariant Snapshot
# ---------------------------------------------------------------------------


def build_budget_invariant_snapshot(
    stats: Any,  # PipelineStats or compatible dict
    cycle_budget_s: Optional[float] = None,
    max_candidates_per_cycle: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build a budget invariant snapshot for a single run.
    
    Evaluates INV-BUD-1 through INV-BUD-5 and produces a structured snapshot
    that is deterministic and JSON-serializable.
    
    Invariants:
        INV-BUD-1: No processing after budget exhaustion (budget_exhausted → no post-exhaustion candidates)
        INV-BUD-2: Hard cap on candidates (max_candidates_hit → candidates_considered == max_candidates)
        INV-BUD-3: Budget remaining is monotonic (if not exhausted, remaining >= 0)
        INV-BUD-4: Timeout abstentions are tracked (timeout_abstentions >= 0)
        INV-BUD-5: Statements skipped is non-negative (statements_skipped >= 0)
    
    Args:
        stats: PipelineStats object or dict with budget fields:
               - budget_exhausted: bool
               - max_candidates_hit: bool
               - timeout_abstentions: int
               - statements_skipped: int
               - candidates_considered: int
               - budget_remaining_s: float (optional)
               - post_exhaustion_candidates: int (optional)
        cycle_budget_s: Original cycle budget in seconds (for validation)
        max_candidates_per_cycle: Original max candidates limit (for validation)
    
    Returns:
        Dict with schema:
            {
                "schema_version": "1.0.0",
                "inv_bud_1_ok": bool,
                "inv_bud_2_ok": bool,
                "inv_bud_3_ok": bool,
                "inv_bud_4_ok": bool,
                "inv_bud_5_ok": bool,
                "timeout_abstentions": int,
                "post_exhaustion_candidates": int,
                "max_candidates_hit": bool,
                "summary_status": "OK" | "WARN" | "FAIL",
            }
    """
    # Extract fields (handle both object and dict)
    if hasattr(stats, "__dict__"):
        budget_exhausted = getattr(stats, "budget_exhausted", False)
        max_candidates_hit = getattr(stats, "max_candidates_hit", False)
        timeout_abstentions = getattr(stats, "timeout_abstentions", 0)
        statements_skipped = getattr(stats, "statements_skipped", 0)
        candidates_considered = getattr(stats, "candidates_considered", 0)
        budget_remaining_s = getattr(stats, "budget_remaining_s", None)
        post_exhaustion_candidates = getattr(stats, "post_exhaustion_candidates", 0)
    else:
        budget_exhausted = stats.get("budget_exhausted", False)
        max_candidates_hit = stats.get("max_candidates_hit", False)
        timeout_abstentions = stats.get("timeout_abstentions", 0)
        statements_skipped = stats.get("statements_skipped", 0)
        candidates_considered = stats.get("candidates_considered", 0)
        budget_remaining_s = stats.get("budget_remaining_s", None)
        post_exhaustion_candidates = stats.get("post_exhaustion_candidates", 0)
    
    # INV-BUD-1: No processing after budget exhaustion
    inv_bud_1_ok = not budget_exhausted or post_exhaustion_candidates == 0
    
    # INV-BUD-2: Hard cap on candidates (if hit, must match)
    if max_candidates_hit and max_candidates_per_cycle is not None:
        inv_bud_2_ok = candidates_considered >= max_candidates_per_cycle
    elif max_candidates_hit:
        # If hit but no limit provided, we can't validate
        inv_bud_2_ok = True
    else:
        inv_bud_2_ok = True
    
    # INV-BUD-3: Budget remaining is monotonic (non-negative if tracked)
    if budget_remaining_s is not None:
        inv_bud_3_ok = budget_remaining_s >= 0.0
    else:
        # Not tracked, assume OK
        inv_bud_3_ok = True
    
    # INV-BUD-4: Timeout abstentions are tracked (non-negative)
    inv_bud_4_ok = timeout_abstentions >= 0
    
    # INV-BUD-5: Statements skipped is non-negative
    inv_bud_5_ok = statements_skipped >= 0
    
    # Determine summary status
    all_ok = inv_bud_1_ok and inv_bud_2_ok and inv_bud_3_ok and inv_bud_4_ok and inv_bud_5_ok
    has_warn = timeout_abstentions > 0 or statements_skipped > 0
    
    if not all_ok:
        summary_status = "FAIL"
    elif has_warn:
        summary_status = "WARN"
    else:
        summary_status = "OK"
    
    return {
        "schema_version": BUDGET_SCHEMA_VERSION,
        "inv_bud_1_ok": inv_bud_1_ok,
        "inv_bud_2_ok": inv_bud_2_ok,
        "inv_bud_3_ok": inv_bud_3_ok,
        "inv_bud_4_ok": inv_bud_4_ok,
        "inv_bud_5_ok": inv_bud_5_ok,
        "timeout_abstentions": timeout_abstentions,
        "post_exhaustion_candidates": post_exhaustion_candidates,
        "max_candidates_hit": max_candidates_hit,
        "summary_status": summary_status,
    }


def build_budget_invariant_timeline(
    snapshots: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate multiple budget invariant snapshots into a timeline.
    
    Tracks per-invariant failure counts, overall status distribution,
    recent status trends, and stability index.
    
    Args:
        snapshots: Sequence of snapshots from build_budget_invariant_snapshot()
    
    Returns:
        Dict with schema:
            {
                "schema_version": "1.0.0",
                "total_runs": int,
                "ok_count": int,
                "warn_count": int,
                "fail_count": int,
                "inv_bud_1_failures": int,
                "inv_bud_2_failures": int,
                "inv_bud_3_failures": int,
                "inv_bud_4_failures": int,
                "inv_bud_5_failures": int,
                "recent_status": "OK" | "WARN" | "FAIL",
                "stability_index": float,  # [0.0, 1.0]
            }
    """
    if not snapshots:
        return {
            "schema_version": BUDGET_SCHEMA_VERSION,
            "total_runs": 0,
            "ok_count": 0,
            "warn_count": 0,
            "fail_count": 0,
            "inv_bud_1_failures": 0,
            "inv_bud_2_failures": 0,
            "inv_bud_3_failures": 0,
            "inv_bud_4_failures": 0,
            "inv_bud_5_failures": 0,
            "recent_status": "OK",
            "stability_index": 1.0,
        }
    
    total_runs = len(snapshots)
    ok_count = 0
    warn_count = 0
    fail_count = 0
    
    inv_bud_1_failures = 0
    inv_bud_2_failures = 0
    inv_bud_3_failures = 0
    inv_bud_4_failures = 0
    inv_bud_5_failures = 0
    
    for snapshot in snapshots:
        status = snapshot.get("summary_status", "OK")
        if status == "OK":
            ok_count += 1
        elif status == "WARN":
            warn_count += 1
        else:
            fail_count += 1
        
        if not snapshot.get("inv_bud_1_ok", True):
            inv_bud_1_failures += 1
        if not snapshot.get("inv_bud_2_ok", True):
            inv_bud_2_failures += 1
        if not snapshot.get("inv_bud_3_ok", True):
            inv_bud_3_failures += 1
        if not snapshot.get("inv_bud_4_ok", True):
            inv_bud_4_failures += 1
        if not snapshot.get("inv_bud_5_ok", True):
            inv_bud_5_failures += 1
    
    # Recent status: last 5 runs (or all if fewer)
    recent_snapshots = list(snapshots[-5:])
    recent_statuses = [s.get("summary_status", "OK") for s in recent_snapshots]
    
    # Most common status in recent runs
    if recent_statuses:
        recent_status_counts = {"OK": 0, "WARN": 0, "FAIL": 0}
        for status in recent_statuses:
            if status in recent_status_counts:
                recent_status_counts[status] += 1
        
        # Prefer most severe
        if recent_status_counts["FAIL"] > 0:
            recent_status = "FAIL"
        elif recent_status_counts["WARN"] > 0:
            recent_status = "WARN"
        else:
            recent_status = "OK"
    else:
        recent_status = "OK"
    
    # Stability index: consistency of recent statuses
    if len(recent_statuses) >= 2:
        unique_statuses = len(set(recent_statuses))
        # More consistent = higher stability (1.0 if all same, lower if mixed)
        stability_index = 1.0 - ((unique_statuses - 1) / len(recent_statuses))
    else:
        stability_index = 1.0
    
    return {
        "schema_version": BUDGET_SCHEMA_VERSION,
        "total_runs": total_runs,
        "ok_count": ok_count,
        "warn_count": warn_count,
        "fail_count": fail_count,
        "inv_bud_1_failures": inv_bud_1_failures,
        "inv_bud_2_failures": inv_bud_2_failures,
        "inv_bud_3_failures": inv_bud_3_failures,
        "inv_bud_4_failures": inv_bud_4_failures,
        "inv_bud_5_failures": inv_bud_5_failures,
        "recent_status": recent_status,
        "stability_index": stability_index,
    }


# ---------------------------------------------------------------------------
# Global Health Summary
# ---------------------------------------------------------------------------


def summarize_budget_invariants_for_global_health(
    timeline: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Summarize budget invariants for global health surfaces (MAAS, Director Console).
    
    Produces a compact, high-level summary suitable for aggregation into
    system-wide health signals.
    
    Args:
        timeline: Output from build_budget_invariant_timeline()
    
    Returns:
        Dict with schema:
            {
                "schema_version": "1.0.0",
                "invariants_ok": bool,
                "recent_status": "OK" | "WARN" | "FAIL",
                "inv_bud_failures": List[str],  # e.g., ["INV-BUD-1", "INV-BUD-3"]
                "status": "OK" | "WARN" | "BLOCK",
                "stability_index": float,
                "total_runs": int,
            }
    """
    recent_status = timeline.get("recent_status", "OK")
    stability_index = timeline.get("stability_index", 1.0)
    total_runs = timeline.get("total_runs", 0)
    
    # Collect failed invariants
    inv_bud_failures: List[str] = []
    if timeline.get("inv_bud_1_failures", 0) > 0:
        inv_bud_failures.append("INV-BUD-1")
    if timeline.get("inv_bud_2_failures", 0) > 0:
        inv_bud_failures.append("INV-BUD-2")
    if timeline.get("inv_bud_3_failures", 0) > 0:
        inv_bud_failures.append("INV-BUD-3")
    if timeline.get("inv_bud_4_failures", 0) > 0:
        inv_bud_failures.append("INV-BUD-4")
    if timeline.get("inv_bud_5_failures", 0) > 0:
        inv_bud_failures.append("INV-BUD-5")
    
    invariants_ok = len(inv_bud_failures) == 0
    
    # Determine overall status
    # BLOCK if recent FAIL or stability very low
    if recent_status == "FAIL" or stability_index < 0.5:
        status = "BLOCK"
    elif recent_status == "WARN" or stability_index < 0.95 or inv_bud_failures:
        status = "WARN"
    else:
        status = "OK"
    
    return {
        "schema_version": BUDGET_SCHEMA_VERSION,
        "invariants_ok": invariants_ok,
        "recent_status": recent_status,
        "inv_bud_failures": inv_bud_failures,
        "status": status,
        "stability_index": stability_index,
        "total_runs": total_runs,
    }


# ---------------------------------------------------------------------------
# Governance View and Release Readiness
# ---------------------------------------------------------------------------


def build_budget_invariants_governance_view(
    invariant_timeline: Dict[str, Any],
    budget_health: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build cross-layer budget governance view combining invariants and budget health.
    
    Aggregates invariant timeline with A5 budget health metrics to produce
    a unified governance view for release gating.
    
    Args:
        invariant_timeline: Output from build_budget_invariant_timeline()
        budget_health: Budget health object from A5 (should have health_score, trend_status)
    
    Returns:
        Dict with schema:
            {
                "schema_version": "1.0.0",
                "total_runs": int,
                "stability_index": float,
                "health_score": float,  # From budget_health
                "combined_status": "OK" | "WARN" | "BLOCK",
            }
    """
    health_summary = summarize_budget_invariants_for_global_health(invariant_timeline)
    invariants_status = health_summary.get("status", "OK")
    inv_bud_failures = health_summary.get("inv_bud_failures", [])
    stability_index = invariant_timeline.get("stability_index", 1.0)
    total_runs = invariant_timeline.get("total_runs", 0)
    
    health_score = budget_health.get("health_score", 100.0)
    trend_status = budget_health.get("trend_status", "STABLE")
    
    # Combined status logic
    # BLOCK if invariants BLOCK
    if invariants_status == "BLOCK":
        combined_status = "BLOCK"
    # WARN if invariants WARN or budget health degrading
    elif invariants_status == "WARN" or trend_status == "DEGRADING":
        combined_status = "WARN"
    else:
        combined_status = "OK"
    
    return {
        "schema_version": BUDGET_SCHEMA_VERSION,
        "total_runs": total_runs,
        "stability_index": stability_index,
        "health_score": health_score,
        "combined_status": combined_status,
        "inv_bud_failures": inv_bud_failures,  # Include for explain_budget_release_decision
    }


def evaluate_budget_release_readiness(
    governance_view: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate budget-based release readiness signal for CI gating.
    
    Pure function designed for CI workflows to gate releases based on
    budget invariant health and budget health metrics.
    
    Args:
        governance_view: Output from build_budget_invariants_governance_view()
    
    Returns:
        Dict with schema:
            {
                "release_ok": bool,
                "blocking_reasons": List[str],
                "status": "OK" | "WARN" | "BLOCK",
            }
    """
    combined_status = governance_view.get("combined_status", "UNKNOWN")
    stability_index = governance_view.get("stability_index", 0.0)
    health_score = governance_view.get("health_score", 0.0)
    
    blocking_reasons: List[str] = []
    
    # Check for BLOCK conditions
    if combined_status == "BLOCK":
        blocking_reasons.append(f"combined_status is BLOCK")
    
    if health_score < 70.0:
        blocking_reasons.append(f"health_score < 70.0 (current: {health_score:.1f})")
    
    # Determine final status
    if combined_status == "BLOCK" or health_score < 70.0:
        status = "BLOCK"
        release_ok = False
    elif combined_status == "WARN" or stability_index < 0.95:
        status = "WARN"
        release_ok = True  # WARN doesn't block, but flags concern
        if stability_index < 0.95:
            blocking_reasons.append(f"stability_index < 0.95 (current: {stability_index:.3f})")
    else:
        status = "OK"
        release_ok = True
    
    return {
        "release_ok": release_ok,
        "blocking_reasons": blocking_reasons,
        "status": status,
    }


# ---------------------------------------------------------------------------
# Director Console Panel
# ---------------------------------------------------------------------------


def build_budget_invariants_director_panel(
    governance_view: Dict[str, Any],
    readiness: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build Budget Invariants Director Panel tile.
    
    Creates a compact, human-readable panel for high-level dashboards.
    Uses neutral language suitable for Director Console.
    
    Args:
        governance_view: Output from build_budget_invariants_governance_view()
        readiness: Output from evaluate_budget_release_readiness()
    
    Returns:
        Dict with schema:
            {
                "status_light": "GREEN" | "YELLOW" | "RED",
                "recent_status": str,
                "stability_index": float,
                "health_score": float,
                "headline": str,  # Neutral, factual sentence
                "key_invariants_with_failures": List[str],
            }
    """
    combined_status = governance_view.get("combined_status", "OK")
    stability_index = governance_view.get("stability_index", 1.0)
    health_score = governance_view.get("health_score", 100.0)
    readiness_status = readiness.get("status", "OK")
    
    # Map status to status_light
    if combined_status == "BLOCK" or readiness_status == "BLOCK":
        status_light = "RED"
    elif combined_status == "WARN" or readiness_status == "WARN":
        status_light = "YELLOW"
    else:
        status_light = "GREEN"
    
    # Extract invariant failures from blocking reasons
    key_invariants_with_failures: List[str] = []
    blocking_reasons = readiness.get("blocking_reasons", [])
    for reason in blocking_reasons:
        if "INV-BUD-" in reason:
            # Extract invariant ID
            for inv_id in ["INV-BUD-1", "INV-BUD-2", "INV-BUD-3", "INV-BUD-4", "INV-BUD-5"]:
                if inv_id in reason and inv_id not in key_invariants_with_failures:
                    key_invariants_with_failures.append(inv_id)
    
    # Generate neutral headline
    headline_parts = []
    headline_parts.append(f"Budget invariants status: {combined_status}")
    
    if health_score < 100.0:
        headline_parts.append(f"health score {health_score:.1f}")
    
    if stability_index < 1.0:
        headline_parts.append(f"stability index {stability_index:.3f}")
    
    if key_invariants_with_failures:
        headline_parts.append(f"{len(key_invariants_with_failures)} invariant(s) flagged")
    
    headline = ". ".join(headline_parts) + "."
    
    return {
        "status_light": status_light,
        "recent_status": combined_status,
        "stability_index": stability_index,
        "health_score": health_score,
        "headline": headline,
        "key_invariants_with_failures": key_invariants_with_failures,
    }


# ---------------------------------------------------------------------------
# Storyline Builder
# ---------------------------------------------------------------------------


def build_budget_storyline(
    invariant_timeline: Dict[str, Any],
    budget_health_history: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build a narrative storyline from invariant timeline and budget health history.
    
    Produces ordered episodes, structural events, and a stability classification.
    All descriptions use neutral, factual language.
    
    Args:
        invariant_timeline: Output from build_budget_invariant_timeline()
        budget_health_history: Sequence of budget_health objects (chronological)
                              Each dict should have:
                              - "health_score": float
                              - "trend_status": "IMPROVING" | "STABLE" | "DEGRADING"
                              - (optional) "run_index": int
    
    Returns:
        Dict with schema:
            {
                "schema_version": "1.0.0",
                "runs_analyzed": int,
                "episodes": [
                    {
                        "run_range": "runs 40-55",
                        "status": "WARN" | "FAIL",
                        "invariants_affected": ["INV-BUD-2"],
                        "health_score_range": [72.0, 79.0],
                        "description": "neutral factual description"
                    },
                ],
                "structural_events": [
                    "INV-BUD-1 first failure at run 42",
                    ...
                ],
                "stability_class": "STABLE" | "DRIFTING" | "VOLATILE",
                "summary": "neutral narrative summary"
            }
    """
    total_runs = invariant_timeline.get("total_runs", 0)
    
    if total_runs == 0:
        return {
            "schema_version": BUDGET_SCHEMA_VERSION,
            "runs_analyzed": 0,
            "episodes": [],
            "structural_events": [],
            "stability_class": "STABLE",
            "summary": "No runs analyzed.",
        }
    
    episodes: List[Dict[str, Any]] = []
    structural_events: List[str] = []
    
    stability_index = invariant_timeline.get("stability_index", 1.0)
    recent_status = invariant_timeline.get("recent_status", "OK")
    
    # Classify stability
    if stability_index >= 0.95 and recent_status == "OK":
        stability_class = "STABLE"
    elif stability_index < 0.7 or recent_status == "FAIL":
        stability_class = "VOLATILE"
    else:
        stability_class = "DRIFTING"
    
    # Detect structural events (first failures)
    inv_failures = {
        "INV-BUD-1": invariant_timeline.get("inv_bud_1_failures", 0),
        "INV-BUD-2": invariant_timeline.get("inv_bud_2_failures", 0),
        "INV-BUD-3": invariant_timeline.get("inv_bud_3_failures", 0),
        "INV-BUD-4": invariant_timeline.get("inv_bud_4_failures", 0),
        "INV-BUD-5": invariant_timeline.get("inv_bud_5_failures", 0),
    }
    
    for inv_id, count in inv_failures.items():
        if count > 0:
            structural_events.append(f"{inv_id} failure detected ({count} occurrence(s))")
    
    # Generate episodes from health history (simplified)
    if budget_health_history and len(budget_health_history) > 0:
        # Find ranges of concern
        episode_start = None
        episode_status = None
        episode_invariants: List[str] = []
        episode_health_scores: List[float] = []
        
        for i, health in enumerate(budget_health_history):
            health_score = health.get("health_score", 100.0)
            trend = health.get("trend_status", "STABLE")
            
            # Detect episode boundaries
            if health_score < 70.0 or trend == "DEGRADING":
                if episode_start is None:
                    episode_start = i
                    episode_status = "WARN"
                    episode_invariants = []
                    episode_health_scores = []
                
                episode_health_scores.append(health_score)
                
                if episode_status == "WARN" and health_score < 50.0:
                    episode_status = "FAIL"
            else:
                # End current episode
                if episode_start is not None:
                    if len(episode_health_scores) > 0:
                        episodes.append({
                            "run_range": f"runs {episode_start}-{i-1}",
                            "status": episode_status,
                            "invariants_affected": episode_invariants,
                            "health_score_range": [
                                min(episode_health_scores),
                                max(episode_health_scores),
                            ],
                            "description": f"Health score range {min(episode_health_scores):.1f}-{max(episode_health_scores):.1f}, status {episode_status}",
                        })
                    episode_start = None
                    episode_status = None
                    episode_invariants = []
                    episode_health_scores = []
        
        # Close final episode if open
        if episode_start is not None:
            episodes.append({
                "run_range": f"runs {episode_start}-{len(budget_health_history)-1}",
                "status": episode_status or "WARN",
                "invariants_affected": episode_invariants,
                "health_score_range": [
                    min(episode_health_scores) if episode_health_scores else 0.0,
                    max(episode_health_scores) if episode_health_scores else 0.0,
                ],
                "description": f"Health score range {min(episode_health_scores):.1f}-{max(episode_health_scores):.1f}",
            })
    
    # Generate summary
    summary_parts = []
    summary_parts.append(f"Analyzed {total_runs} runs")
    summary_parts.append(f"stability classification: {stability_class}")
    
    if episodes:
        summary_parts.append(f"{len(episodes)} episode(s) identified")
    
    if structural_events:
        summary_parts.append(f"{len(structural_events)} structural event(s) detected")
    
    summary = ". ".join(summary_parts) + "."
    
    return {
        "schema_version": BUDGET_SCHEMA_VERSION,
        "runs_analyzed": total_runs,
        "episodes": episodes,
        "structural_events": structural_events,
        "stability_class": stability_class,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# BNH-Φ Projection Engine
# ---------------------------------------------------------------------------


def project_budget_stability_horizon(
    budget_health_history: Sequence[Dict[str, Any]],
    horizon_length: int = 10,
) -> Dict[str, Any]:
    """
    Project budget stability trajectory forward using deterministic rules.
    
    Uses only the last 5 health_scores and trend_status entries to project
    the expected risk band trajectory. No ML, simple linear/threshold rules.
    
    Args:
        budget_health_history: Sequence of budget_health objects (chronological).
                              Each dict should have:
                              - "health_score": float
                              - "trend_status": "IMPROVING" | "STABLE" | "DEGRADING"
        horizon_length: Number of future runs to project (default: 10).
    
    Returns:
        Dict with schema:
            {
                "horizon_length": int,
                "projected_stability_class": "STABLE" | "DRIFTING" | "VOLATILE",
                "converging_invariants": bool,
                "diverging_invariants": bool,
                "projection_method": "linear_extrapolation" | "threshold_rule",
                "risk_trajectory": ["STABLE", "DRIFTING", ...],  # per-horizon step
            }
    """
    if not budget_health_history:
        return {
            "horizon_length": horizon_length,
            "projected_stability_class": "UNKNOWN",
            "converging_invariants": False,
            "diverging_invariants": False,
            "projection_method": "insufficient_data",
            "risk_trajectory": ["UNKNOWN"] * horizon_length,
        }
    
    # Use last 5 entries
    recent_history = list(budget_health_history[-5:])
    num_samples = len(recent_history)
    
    if num_samples < 2:
        return {
            "horizon_length": horizon_length,
            "projected_stability_class": "UNKNOWN",
            "converging_invariants": False,
            "diverging_invariants": False,
            "projection_method": "insufficient_data",
            "risk_trajectory": ["UNKNOWN"] * horizon_length,
        }
    
    # Extract health scores and trends
    health_scores = [h.get("health_score", 0.0) for h in recent_history]
    trends = [h.get("trend_status", "STABLE") for h in recent_history]
    
    # Determine projection method
    # Simple linear extrapolation: calculate average rate of change
    if num_samples >= 2:
        score_deltas = [health_scores[i+1] - health_scores[i] for i in range(num_samples-1)]
        avg_delta = sum(score_deltas) / len(score_deltas)
        projection_method = "linear_extrapolation"
    else:
        avg_delta = 0.0
        projection_method = "threshold_rule"
    
    # Project trajectory
    risk_trajectory = []
    current_health = health_scores[-1]
    
    for step in range(horizon_length):
        # Project health score
        projected_health = current_health + (avg_delta * (step + 1))
        
        # Classify projected stability
        if projected_health >= 80.0:
            risk_class = "STABLE"
        elif projected_health >= 70.0:
            risk_class = "DRIFTING"
        else:
            risk_class = "VOLATILE"
        
        # Adjust based on trend pattern
        recent_trend = trends[-1] if trends else "STABLE"
        if recent_trend == "DEGRADING" and projected_health > 70.0:
            risk_class = "DRIFTING"  # Degrading trend moves toward volatility
        elif recent_trend == "IMPROVING" and projected_health < 80.0:
            risk_class = "DRIFTING"  # Improving trend moves toward stability
        
        risk_trajectory.append(risk_class)
    
    # Determine overall projected stability class (most common in trajectory)
    stability_counts = {"STABLE": 0, "DRIFTING": 0, "VOLATILE": 0}
    for risk in risk_trajectory:
        if risk in stability_counts:
            stability_counts[risk] += 1
    
    # Prefer most severe classification if ties
    if stability_counts["VOLATILE"] > 0:
        projected_stability_class = "VOLATILE"
    elif stability_counts["DRIFTING"] > stability_counts["STABLE"]:
        projected_stability_class = "DRIFTING"
    else:
        projected_stability_class = "STABLE"
    
    # Detect converging vs diverging
    converging_invariants = avg_delta > 0.0  # Health improving
    diverging_invariants = avg_delta < -1.0  # Health degrading significantly
    
    # Check trend pattern
    if trends:
        degrading_count = sum(1 for t in trends if t == "DEGRADING")
        improving_count = sum(1 for t in trends if t == "IMPROVING")
        
        if improving_count > degrading_count:
            converging_invariants = True
        elif degrading_count > improving_count:
            diverging_invariants = True
    
    return {
        "horizon_length": horizon_length,
        "projected_stability_class": projected_stability_class,
        "converging_invariants": converging_invariants,
        "diverging_invariants": diverging_invariants,
        "projection_method": projection_method,
        "risk_trajectory": risk_trajectory,
    }


def build_budget_episode_ledger_tile(
    storyline: Dict[str, Any],
    projection: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build Budget Episode Ledger Tile for Director Console.
    
    Creates a console-ready structure that aggregates episodes, projection,
    and key metrics into a queryable, cross-tile compatible format.
    
    Args:
        storyline: Output from build_budget_storyline().
        projection: Output from project_budget_stability_horizon().
    
    Returns:
        Dict with schema:
            {
                "schema_version": "1.0.0",
                "episodes_count": int,
                "episode_status_distribution": {"OK": int, "WARN": int, "FAIL": int},
                "first_hard_violation": str | None,
                "stability_class": "STABLE" | "DRIFTING" | "VOLATILE",
                "projection": {
                    "projected_stability_class": str,
                    "converging_invariants": bool,
                    "diverging_invariants": bool,
                },
                "status_light": "GREEN" | "YELLOW" | "RED",
                "headline": str,  # Neutral summary
            }
    """
    episodes = storyline.get("episodes", [])
    stability_class = storyline.get("stability_class", "UNKNOWN")
    structural_events = storyline.get("structural_events", [])
    
    # Count episodes by status
    status_counts = {"OK": 0, "WARN": 0, "FAIL": 0, "VOLATILE": 0}
    for episode in episodes:
        ep_status = episode.get("status", "OK")
        if ep_status in status_counts:
            status_counts[ep_status] += 1
        # Check if episode has volatile characteristics
        invariants = episode.get("invariants_affected", [])
        hard_invariants = [inv for inv in invariants if inv in ["INV-BUD-1", "INV-BUD-2", "INV-BUD-3"]]
        if len(hard_invariants) > 0:
            status_counts["VOLATILE"] += 1
    
    # Find first hard violation
    first_hard_violation = None
    for event in structural_events:
        if "INV-BUD-1" in event or "INV-BUD-2" in event or "INV-BUD-3" in event:
            first_hard_violation = event
            break
    
    # Determine status light (VOLATILE takes precedence)
    projected_class = projection.get("projected_stability_class", "UNKNOWN")
    
    if stability_class == "VOLATILE" or projected_class == "VOLATILE" or status_counts["VOLATILE"] > 0:
        status_light = "RED"
    elif stability_class == "DRIFTING" or projected_class == "DRIFTING" or status_counts["WARN"] > 0:
        status_light = "YELLOW"
    else:
        status_light = "GREEN"
    
    # Generate neutral headline
    headline_parts = []
    headline_parts.append(f"{len(episodes)} episodes analyzed")
    
    if first_hard_violation:
        headline_parts.append(f"hard violation detected")
    
    if projected_class != "UNKNOWN":
        headline_parts.append(f"projected trajectory: {projected_class}")
    
    if projection.get("diverging_invariants", False):
        headline_parts.append("diverging pattern observed")
    elif projection.get("converging_invariants", False):
        headline_parts.append("converging pattern observed")
    
    headline = ". ".join(headline_parts) + "."
    
    return {
        "schema_version": BUDGET_SCHEMA_VERSION,
        "episodes_count": len(episodes),
        "episode_status_distribution": {
            "OK": status_counts["OK"],
            "WARN": status_counts["WARN"],
            "FAIL": status_counts["FAIL"],
        },
        "first_hard_violation": first_hard_violation,
        "stability_class": stability_class,
        "projection": {
            "projected_stability_class": projected_class,
            "converging_invariants": projection.get("converging_invariants", False),
            "diverging_invariants": projection.get("diverging_invariants", False),
        },
        "status_light": status_light,
        "headline": headline,
    }


# ---------------------------------------------------------------------------
# CI Release Explanation Capsule
# ---------------------------------------------------------------------------


def explain_budget_release_decision(
    governance_view: Dict[str, Any],
    readiness: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Explain budget release decision with forensic details.
    
    Extended version with CI capsule fields: root_cause_vector, trigger_index,
    fault_surface, and CI-safe summary string.
    
    Args:
        governance_view: Output from build_budget_invariants_governance_view()
        readiness: Output from evaluate_budget_release_readiness()
    
    Returns:
        Dict with schema:
            {
                "decision": "OK" | "WARN" | "BLOCK",
                "primary_causes": List[str],
                "contributing_factors": List[str],
                "recommended_followups": List[str],
                "root_cause_vector": List[str],  # Ordered, prioritized
                "trigger_index": int,  # Index of first failing invariant (0-4)
                "fault_surface": str,  # Neutral description
                "ci_summary": str,  # <=160 chars, neutral, factual
            }
    """
    decision = readiness.get("status", "OK")
    blocking_reasons = readiness.get("blocking_reasons", [])
    combined_status = governance_view.get("combined_status", "OK")
    health_score = governance_view.get("health_score", 100.0)
    stability_index = governance_view.get("stability_index", 1.0)
    inv_bud_failures = governance_view.get("inv_bud_failures", [])
    
    # Build root cause vector (ordered, prioritized)
    root_cause_vector: List[str] = []
    
    # Add invariant failures from governance_view (most direct source)
    for inv_id in inv_bud_failures:
        root_cause_vector.append(f"{inv_id} failure detected")
    
    # Extract other reasons from blocking_reasons
    for reason in blocking_reasons:
        if "INV-BUD-" not in reason:  # Skip invariants already added
            if "health_score" in reason:
                root_cause_vector.append(f"health_score below threshold: {health_score:.1f}")
            elif "stability_index" in reason:
                root_cause_vector.append(f"stability_index below threshold: {stability_index:.3f}")
            else:
                root_cause_vector.append(reason)
    
    # If no specific causes, add status
    if not root_cause_vector:
        root_cause_vector.append(f"combined_status: {combined_status}")
    
    # Determine trigger_index (first failing invariant)
    trigger_index = -1
    for i, inv_id in enumerate(["INV-BUD-1", "INV-BUD-2", "INV-BUD-3", "INV-BUD-4", "INV-BUD-5"]):
        if any(inv_id in cause for cause in root_cause_vector):
            trigger_index = i
            break
    
    # Build fault surface description
    fault_surface_parts = []
    if trigger_index >= 0:
        fault_surface_parts.append(f"Invariant {trigger_index + 1} failure detected")
    if health_score < 70.0:
        fault_surface_parts.append(f"budget health score {health_score:.1f} below 70.0 threshold")
    if stability_index < 0.95:
        fault_surface_parts.append(f"stability index {stability_index:.3f} below 0.95 threshold")
    
    fault_surface = ". ".join(fault_surface_parts) if fault_surface_parts else "No fault surface identified"
    
    # Primary causes (most significant)
    primary_causes = root_cause_vector[:3] if len(root_cause_vector) > 0 else [f"Status: {decision}"]
    
    # Contributing factors
    contributing_factors = []
    if stability_index < 1.0:
        contributing_factors.append(f"stability_index: {stability_index:.3f}")
    if health_score < 100.0:
        contributing_factors.append(f"health_score: {health_score:.1f}")
    
    # Recommended followups
    recommended_followups = []
    if decision == "BLOCK":
        recommended_followups.append("Investigate root cause vector entries")
        if trigger_index >= 0:
            recommended_followups.append(f"Review invariant {trigger_index + 1} enforcement logic")
    elif decision == "WARN":
        recommended_followups.append("Monitor stability index trend")
    
    # Build CI-safe summary (<=160 chars, neutral, factual)
    ci_summary_parts = []
    ci_summary_parts.append(f"Budget status: {decision}")
    
    if trigger_index >= 0:
        ci_summary_parts.append(f"INV-BUD-{trigger_index + 1} failed")
    
    if health_score < 70.0:
        ci_summary_parts.append(f"health {health_score:.1f}")
    
    ci_summary = ". ".join(ci_summary_parts)
    
    # Trim to 160 chars if needed
    if len(ci_summary) > 160:
        ci_summary = ci_summary[:157] + "..."
    
    return {
        "decision": decision,
        "primary_causes": primary_causes,
        "contributing_factors": contributing_factors,
        "recommended_followups": recommended_followups,
        "root_cause_vector": root_cause_vector,
        "trigger_index": trigger_index,
        "fault_surface": fault_surface,
        "ci_summary": ci_summary,
    }


# ---------------------------------------------------------------------------
# Global Health Adapter
# ---------------------------------------------------------------------------


def summarize_storyline_for_global_health(
    storyline: Dict[str, Any],
    projection: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Adapt storyline and projection for global health surfaces (MAAS, Director Console).
    
    Produces a compact, queryable structure suitable for aggregation into
    system-wide health signals.
    
    Args:
        storyline: Output from build_budget_storyline()
        projection: Output from project_budget_stability_horizon()
    
    Returns:
        Dict with schema:
            {
                "schema_version": "1.0.0",
                "status": "OK" | "WARN" | "BLOCK",
                "stability_class": str,
                "projected_stability_class": str,
                "episodes_count": int,
                "stability_index": float,  # Derived from storyline/projection
                "summary": str,  # Compact neutral summary
            }
    """
    stability_class = storyline.get("stability_class", "STABLE")
    projected_class = projection.get("projected_stability_class", "STABLE")
    episodes_count = len(storyline.get("episodes", []))
    
    # Map stability to status
    if stability_class == "VOLATILE" or projected_class == "VOLATILE":
        status = "BLOCK"
    elif stability_class == "DRIFTING" or projected_class == "DRIFTING":
        status = "WARN"
    else:
        status = "OK"
    
    # Derive stability_index from projection convergence
    converging = projection.get("converging_invariants", False)
    diverging = projection.get("diverging_invariants", False)
    
    if converging and not diverging:
        stability_index = 0.95
    elif diverging:
        stability_index = 0.5
    else:
        stability_index = 0.85
    
    # Build summary
    summary_parts = []
    summary_parts.append(f"Stability: {stability_class}")
    if projected_class != stability_class:
        summary_parts.append(f"projected: {projected_class}")
    if episodes_count > 0:
        summary_parts.append(f"{episodes_count} episode(s)")
    
    summary = ". ".join(summary_parts) + "."
    
    return {
        "schema_version": BUDGET_SCHEMA_VERSION,
        "status": status,
        "stability_class": stability_class,
        "projected_stability_class": projected_class,
        "episodes_count": episodes_count,
        "stability_index": stability_index,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Evidence Pack Integration
# ---------------------------------------------------------------------------


def attach_budget_invariants_to_evidence(
    evidence: Dict[str, Any],
    governance_view: Dict[str, Any],
    projection: Optional[Dict[str, Any]] = None,
    ci_capsule: Optional[Dict[str, Any]] = None,
    timeline: Optional[Dict[str, Any]] = None,
    storyline: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach budget invariants governance data to an evidence pack.
    
    Non-mutating: returns a new dict with budget invariants attached under
    evidence["governance"]["budget_invariants"].
    
    Budget Invariants represent the "Energy Law" of First-Light runs:
    - They enforce deterministic budget boundaries (INV-BUD-1 through INV-BUD-5)
    - Timeline aggregation tracks stability across runs
    - Storyline + BNH-Φ projection provide temporal coherence evidence
    - These signals appear in P3 stability reports and P4 calibration bundles
    
    Args:
        evidence: Existing evidence pack dict (read-only, not modified).
        governance_view: Output from build_budget_invariants_governance_view().
        projection: Optional BNH-Φ projection from project_budget_stability_horizon().
        ci_capsule: Optional CI capsule from explain_budget_release_decision().
        timeline: Optional budget invariant timeline (for First Light storyline summary).
        storyline: Optional budget storyline (for First Light storyline summary).
    
    Returns:
        New dict with evidence contents plus budget_invariants attached under
        evidence["governance"]["budget_invariants"].
    
    Structure:
        {
            ...existing evidence fields...,
            "governance": {
                ...existing governance fields...,
                "budget_invariants": {
                    "schema_version": "1.0.0",
                    "invariant_failures": List[str],  # e.g., ["INV-BUD-1"]
                    "stability_index": float,
                    "combined_status": "OK" | "WARN" | "BLOCK",
                    "projected_horizon": Optional[Dict],  # If projection provided
                    "ci_trigger_index": Optional[int],  # If ci_capsule provided
                    "ci_fault_surface": Optional[str],  # If ci_capsule provided
                }
            }
        }
    """
    # Create a copy to avoid mutating the original
    enriched = evidence.copy()
    
    # Ensure governance section exists
    if "governance" not in enriched:
        enriched["governance"] = {}
    
    # Build budget invariants attachment
    budget_data: Dict[str, Any] = {
        "schema_version": BUDGET_SCHEMA_VERSION,
        "invariant_failures": governance_view.get("inv_bud_failures", []),
        "stability_index": governance_view.get("stability_index", 0.0),
        "combined_status": governance_view.get("combined_status", "OK"),
    }
    
    # Add projection data if provided
    if projection is not None:
        budget_data["projected_horizon"] = {
            "projected_stability_class": projection.get("projected_stability_class", "UNKNOWN"),
            "converging_invariants": projection.get("converging_invariants", False),
            "diverging_invariants": projection.get("diverging_invariants", False),
            "horizon_length": projection.get("horizon_length", 10),
        }
    
    # Add CI capsule data if provided
    if ci_capsule is not None:
        budget_data["ci_trigger_index"] = ci_capsule.get("trigger_index", -1)
        budget_data["ci_fault_surface"] = ci_capsule.get("fault_surface", "")
        budget_data["ci_summary"] = ci_capsule.get("ci_summary", "")
    
    # Attach to governance section
    enriched["governance"]["budget_invariants"] = budget_data
    
    # Add First Light storyline summary if timeline, storyline, and projection provided
    if timeline is not None and storyline is not None and projection is not None:
        first_light_storyline = build_first_light_budget_storyline(
            timeline=timeline,
            storyline=storyline,
            projection=projection,
        )
        enriched["governance"]["budget_storyline_summary"] = first_light_storyline
    
    return enriched


# ---------------------------------------------------------------------------
# First Light Budget Storyline
# ---------------------------------------------------------------------------


def build_first_light_budget_storyline(
    timeline: Dict[str, Any],
    storyline: Dict[str, Any],
    projection: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a compact, evidence-ready budget storyline summary for First Light reports.
    
    This function creates a condensed summary suitable for inclusion in P3 stability
    reports and P4 calibration bundles. It combines timeline stability metrics,
    storyline episodes, and BNH-Φ projection data into a single compact structure.
    
    Budget Invariants represent the "Energy Law" of First-Light runs:
    - They enforce deterministic budget boundaries (INV-BUD-1 through INV-BUD-5)
    - Timeline aggregation tracks stability across runs
    - Storyline + BNH-Φ projection provide temporal coherence evidence
    - These signals appear in P3 stability reports and P4 calibration bundles
    
    Args:
        timeline: Budget invariant timeline from build_budget_invariant_timeline().
            Must contain: stability_index, total_runs
        storyline: Budget storyline from build_budget_storyline().
            Must contain: episodes, structural_events
        projection: BNH-Φ projection from project_budget_stability_horizon().
            Must contain: projected_stability_class
    
    Returns:
        Compact evidence-ready summary dict with:
        {
            "schema_version": "1.0.0",
            "combined_status": "OK" | "WARN" | "BLOCK",  # Derived from timeline
            "stability_index": float,  # From timeline
            "episodes_count": int,  # From storyline
            "projection_class": str,  # From projection
            "key_structural_events": List[str],  # First 5 events from storyline
        }
    
    Pure function, deterministic, JSON-serializable.
    """
    # Extract stability_index from timeline
    stability_index = timeline.get("stability_index", 0.0)
    total_runs = timeline.get("total_runs", 0)
    recent_status = timeline.get("recent_status", "OK")
    
    # Determine combined_status from timeline
    # If recent FAIL or very low stability, classify as BLOCK
    # If recent WARN or moderate stability, classify as WARN
    # Otherwise OK
    if recent_status == "FAIL" or stability_index < 0.5:
        combined_status = "BLOCK"
    elif recent_status == "WARN" or stability_index < 0.95:
        combined_status = "WARN"
    else:
        combined_status = "OK"
    
    # Extract episodes and structural events from storyline
    episodes = storyline.get("episodes", [])
    episodes_count = len(episodes)
    
    structural_events = storyline.get("structural_events", [])
    # Take first 5 events (most significant)
    key_structural_events = structural_events[:5]
    
    # Extract projection class
    projection_class = projection.get("projected_stability_class", "UNKNOWN")
    
    return {
        "schema_version": BUDGET_SCHEMA_VERSION,
        "combined_status": combined_status,
        "stability_index": stability_index,
        "episodes_count": episodes_count,
        "projection_class": projection_class,
        "key_structural_events": key_structural_events,
    }

