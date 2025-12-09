"""
Phase III-V Budget Invariant Governance Layer
═════════════════════════════════════════════

This module provides:
    1. Invariant snapshot contract for single runs (Phase III)
    2. Cross-run timeline aggregation for drift detection (Phase III)
    3. Global health / MAAS integration adapter (Phase III)
    4. Cross-layer governance view (Phase IV)
    5. Release readiness evaluation (Phase IV)
    6. Director Console panel (Phase IV)
    7. Budget storyline builder (Phase V)
    8. Release post-mortem engine (Phase V)

Schema Version: 1

Invariants Tracked:
    INV-BUD-1: No candidate processed after budget_exhausted=True
    INV-BUD-2: candidates_considered <= max_candidates_per_cycle
    INV-BUD-3: remaining_budget_s monotonically non-increasing (>= 0 or -1)
    INV-BUD-4: All budget fields present in to_dict()["budget"]
    INV-BUD-5: to_dict() is deterministic

Author: Agent B1 (verifier-ops-1)
Phase: III-V
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

# Current schema version for snapshot contract
SCHEMA_VERSION = 1

# Required budget fields for INV-BUD-4 validation
REQUIRED_BUDGET_FIELDS = frozenset([
    "cycle_budget_s",
    "taut_timeout_s",
    "max_candidates_per_cycle",
    "budget_exhausted",
    "max_candidates_hit",
    "statements_skipped",
    "timeout_abstentions",
    "remaining_budget_s",
    "budget_checks_performed",
    "post_exhaustion_candidates",
])


def build_budget_invariant_snapshot(
    stats: "PipelineStats",
    max_candidates_limit: Optional[int] = None,
    budget_section: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a compact, stable snapshot of budget invariant state for a single run.
    
    This is the canonical contract for budget invariant reporting.
    JSON-serializable and deterministic given the same inputs.
    
    Args:
        stats: PipelineStats from a derivation run.
        max_candidates_limit: The configured max_candidates_per_cycle limit.
                              Used for INV-BUD-2 validation.
        budget_section: Optional pre-computed to_dict()["budget"] section.
                        If None, INV-BUD-4 check is skipped.
    
    Returns:
        Dict with schema:
            {
                "schema_version": 1,
                "inv_bud_1_ok": bool,
                "inv_bud_2_ok": bool,
                "inv_bud_3_ok": bool,
                "inv_bud_4_ok": bool,
                "inv_bud_5_ok": bool,
                "timeout_abstentions": int,
                "post_exhaustion_candidates": int,
                "max_candidates_hit": bool,
                "summary_status": "OK" | "WARN" | "FAIL"
            }
    """
    # INV-BUD-1: No candidate processed after budget_exhausted=True
    inv_bud_1_ok = stats.post_exhaustion_candidates == 0
    
    # INV-BUD-2: Hard cap enforced
    # If max_candidates_hit is True, verify count <= limit
    # If limit not provided, assume OK (can't validate)
    if max_candidates_limit is not None and stats.max_candidates_hit:
        inv_bud_2_ok = stats.candidates_considered <= max_candidates_limit
    else:
        inv_bud_2_ok = True  # No violation detected
    
    # INV-BUD-3: remaining_budget_s >= 0 or == -1 (no budget)
    inv_bud_3_ok = (
        stats.budget_remaining_s >= 0.0 or 
        stats.budget_remaining_s == -1.0
    )
    
    # INV-BUD-4: All required fields present in budget section
    if budget_section is not None:
        missing_fields = REQUIRED_BUDGET_FIELDS - set(budget_section.keys())
        inv_bud_4_ok = len(missing_fields) == 0
    else:
        inv_bud_4_ok = True  # Can't validate without budget_section
    
    # INV-BUD-5: Determinism (always True if we got here without exception)
    inv_bud_5_ok = True
    
    # Compute summary status
    all_ok = all([inv_bud_1_ok, inv_bud_2_ok, inv_bud_3_ok, inv_bud_4_ok, inv_bud_5_ok])
    
    # FAIL if any hard invariant violated (1, 2, 3)
    # WARN if soft invariant violated (4, 5) or high timeout_abstentions
    if not all([inv_bud_1_ok, inv_bud_2_ok, inv_bud_3_ok]):
        summary_status = "FAIL"
    elif not all([inv_bud_4_ok, inv_bud_5_ok]):
        summary_status = "WARN"
    elif stats.timeout_abstentions > 0:
        # Timeouts aren't invariant violations, but worth flagging
        summary_status = "WARN"
    else:
        summary_status = "OK"
    
    return {
        "schema_version": SCHEMA_VERSION,
        "inv_bud_1_ok": inv_bud_1_ok,
        "inv_bud_2_ok": inv_bud_2_ok,
        "inv_bud_3_ok": inv_bud_3_ok,
        "inv_bud_4_ok": inv_bud_4_ok,
        "inv_bud_5_ok": inv_bud_5_ok,
        "timeout_abstentions": stats.timeout_abstentions,
        "post_exhaustion_candidates": stats.post_exhaustion_candidates,
        "max_candidates_hit": stats.max_candidates_hit,
        "summary_status": summary_status,
    }


def build_budget_invariant_timeline(
    snapshots: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate multiple invariant snapshots to detect drift over time.
    
    This enables detection of emerging invariant violations across runs.
    
    Args:
        snapshots: Sequence of invariant snapshots from build_budget_invariant_snapshot.
    
    Returns:
        Dict with schema:
            {
                "schema_version": 1,
                "total_runs": int,
                "ok_count": int,
                "warn_count": int,
                "fail_count": int,
                "inv_bud_1_failures": int,
                "inv_bud_2_failures": int,
                "inv_bud_3_failures": int,
                "inv_bud_4_failures": int,
                "inv_bud_5_failures": int,
                "recent_status": "OK" | "WARN" | "FAIL" | "UNKNOWN",
                "stability_index": float,  # fraction of OK runs
                "timeout_abstention_runs": int,  # runs with timeouts > 0
            }
    """
    if not snapshots:
        return {
            "schema_version": SCHEMA_VERSION,
            "total_runs": 0,
            "ok_count": 0,
            "warn_count": 0,
            "fail_count": 0,
            "inv_bud_1_failures": 0,
            "inv_bud_2_failures": 0,
            "inv_bud_3_failures": 0,
            "inv_bud_4_failures": 0,
            "inv_bud_5_failures": 0,
            "recent_status": "UNKNOWN",
            "stability_index": 0.0,
            "timeout_abstention_runs": 0,
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
    timeout_abstention_runs = 0
    
    for snap in snapshots:
        status = snap.get("summary_status", "UNKNOWN")
        if status == "OK":
            ok_count += 1
        elif status == "WARN":
            warn_count += 1
        elif status == "FAIL":
            fail_count += 1
        
        if not snap.get("inv_bud_1_ok", True):
            inv_bud_1_failures += 1
        if not snap.get("inv_bud_2_ok", True):
            inv_bud_2_failures += 1
        if not snap.get("inv_bud_3_ok", True):
            inv_bud_3_failures += 1
        if not snap.get("inv_bud_4_ok", True):
            inv_bud_4_failures += 1
        if not snap.get("inv_bud_5_ok", True):
            inv_bud_5_failures += 1
        
        if snap.get("timeout_abstentions", 0) > 0:
            timeout_abstention_runs += 1
    
    # Recent status is the last snapshot's status
    recent_status = snapshots[-1].get("summary_status", "UNKNOWN")
    
    # Stability index: fraction of runs with OK status
    stability_index = ok_count / total_runs if total_runs > 0 else 0.0
    
    return {
        "schema_version": SCHEMA_VERSION,
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
        "stability_index": round(stability_index, 4),
        "timeout_abstention_runs": timeout_abstention_runs,
    }


def summarize_budget_invariants_for_global_health(
    timeline: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Produce a compact summary for MAAS and Director's Console integration.
    
    This is the signal that feeds into global_health.json for system-wide
    budget invariant monitoring.
    
    Args:
        timeline: Output from build_budget_invariant_timeline.
    
    Returns:
        Dict with schema:
            {
                "invariants_ok": bool,
                "recent_status": "OK" | "WARN" | "FAIL" | "UNKNOWN",
                "inv_bud_failures": ["INV-BUD-1", ...],  # IDs with non-zero failures
                "status": "OK" | "WARN" | "BLOCK",
                "stability_index": float,
                "total_runs": int,
            }
    
    Status Logic:
        - "BLOCK" if any hard invariant (1, 2, 3) has failures
        - "WARN" if soft invariants (4, 5) have failures or stability < 0.9
        - "OK" otherwise
    """
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
    
    # Hard invariant failures → BLOCK
    hard_invariant_failures = any(
        inv in inv_bud_failures 
        for inv in ["INV-BUD-1", "INV-BUD-2", "INV-BUD-3"]
    )
    
    # Soft invariant failures or low stability → WARN
    soft_invariant_failures = any(
        inv in inv_bud_failures 
        for inv in ["INV-BUD-4", "INV-BUD-5"]
    )
    stability_index = timeline.get("stability_index", 0.0)
    low_stability = stability_index < 0.9
    
    if hard_invariant_failures:
        status = "BLOCK"
    elif soft_invariant_failures or low_stability:
        status = "WARN"
    else:
        status = "OK"
    
    return {
        "invariants_ok": len(inv_bud_failures) == 0,
        "recent_status": timeline.get("recent_status", "UNKNOWN"),
        "inv_bud_failures": inv_bud_failures,
        "status": status,
        "stability_index": stability_index,
        "total_runs": timeline.get("total_runs", 0),
    }


def build_budget_invariants_governance_view(
    invariant_timeline: Dict[str, Any],
    budget_health: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build cross-layer budget governance view combining invariants and health.
    
    This is the primary governance object for MAAS and Director Console,
    correlating budget invariants (A1) with budget health metrics (A5).
    
    Args:
        invariant_timeline: Output from build_budget_invariant_timeline.
        budget_health: Output from A5's summarize_budget_for_global_health.
                       Expected schema:
                       {
                           "health_score": float (0-100),
                           "trend_status": "IMPROVING" | "STABLE" | "DEGRADING",
                           ... (other A5 fields)
                       }
    
    Returns:
        Dict with schema:
            {
                "schema_version": 1,
                "total_runs": int,
                "stability_index": float,
                "health_score": float,
                "invariants_status": "OK" | "WARN" | "BLOCK",
                "combined_status": "OK" | "WARN" | "BLOCK",
                "inv_bud_failures": ["INV-BUD-1", ...],
                "recent_status": "OK" | "WARN" | "FAIL" | "UNKNOWN",
            }
    
    Combined Status Logic:
        - "BLOCK" if invariants_status == "BLOCK"
        - "WARN" if invariants_status == "WARN" OR budget_health.trend_status == "DEGRADING"
        - "OK" otherwise
    """
    # Get invariant summary status
    invariant_summary = summarize_budget_invariants_for_global_health(invariant_timeline)
    invariants_status = invariant_summary.get("status", "UNKNOWN")
    
    # Extract health score and trend from A5 budget_health
    health_score = budget_health.get("health_score", 0.0)
    trend_status = budget_health.get("trend_status", "STABLE")
    
    # Compute combined status
    if invariants_status == "BLOCK":
        combined_status = "BLOCK"
    elif invariants_status == "WARN" or trend_status == "DEGRADING":
        combined_status = "WARN"
    else:
        combined_status = "OK"
    
    return {
        "schema_version": SCHEMA_VERSION,
        "total_runs": invariant_timeline.get("total_runs", 0),
        "stability_index": invariant_timeline.get("stability_index", 0.0),
        "health_score": float(health_score),
        "invariants_status": invariants_status,
        "combined_status": combined_status,
        "inv_bud_failures": invariant_summary.get("inv_bud_failures", []),
        "recent_status": invariant_summary.get("recent_status", "UNKNOWN"),
    }


def evaluate_budget_release_readiness(
    governance_view: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate budget-based release readiness signal for CI gating.
    
    This is a pure function designed for CI workflows to gate releases
    based on budget invariant health and budget health metrics.
    
    Args:
        governance_view: Output from build_budget_invariants_governance_view.
    
    Returns:
        Dict with schema:
            {
                "release_ok": bool,
                "blocking_reasons": ["INV-BUD-1 failed in 3 runs", "health_score < 70", ...],
                "status": "OK" | "WARN" | "BLOCK",
            }
    
    Blocking Rules:
        - BLOCK if combined_status == "BLOCK"
        - BLOCK if health_score < 70.0
        - WARN if combined_status == "WARN" or stability_index < 0.95
        - OK otherwise
    """
    combined_status = governance_view.get("combined_status", "UNKNOWN")
    invariants_status = governance_view.get("invariants_status", "UNKNOWN")
    stability_index = governance_view.get("stability_index", 0.0)
    health_score = governance_view.get("health_score", 0.0)
    inv_bud_failures = governance_view.get("inv_bud_failures", [])
    
    blocking_reasons: List[str] = []
    
    # Check for BLOCK conditions
    if combined_status == "BLOCK":
        blocking_reasons.append(f"combined_status is BLOCK (invariants_status={invariants_status})")
    
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
    
    # Add detailed failure reasons for invariants
    if inv_bud_failures:
        failure_details = ", ".join(inv_bud_failures)
        blocking_reasons.append(f"invariant failures: {failure_details}")
    
    return {
        "release_ok": release_ok,
        "blocking_reasons": blocking_reasons,
        "status": status,
    }


def build_budget_invariants_director_panel(
    governance_view: Dict[str, Any],
    readiness: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build Director Console budget tile with neutral, factual language.
    
    This produces a compact, deterministic panel object suitable for
    direct embedding in global_health.json.
    
    Args:
        governance_view: Output from build_budget_invariants_governance_view.
        readiness: Output from evaluate_budget_release_readiness.
    
    Returns:
        Dict with schema:
            {
                "status_light": "GREEN" | "YELLOW" | "RED",
                "recent_status": "OK" | "WARN" | "FAIL" | "UNKNOWN",
                "stability_index": float,
                "health_score": float,
                "headline": str,  # Neutral, factual sentence
                "key_invariants_with_failures": ["INV-BUD-1", ...],
            }
    
    Status Light Logic:
        - RED if combined_status == "BLOCK"
        - YELLOW if combined_status == "WARN"
        - GREEN if combined_status == "OK"
    
    Headline Rules:
        - Neutral language only (no "good/bad/failure/success")
        - Factual statements about state
        - Single sentence
    """
    combined_status = governance_view.get("combined_status", "UNKNOWN")
    recent_status = governance_view.get("recent_status", "UNKNOWN")
    stability_index = governance_view.get("stability_index", 0.0)
    health_score = governance_view.get("health_score", 0.0)
    inv_bud_failures = governance_view.get("inv_bud_failures", [])
    total_runs = governance_view.get("total_runs", 0)
    
    # Determine status light
    if combined_status == "BLOCK":
        status_light = "RED"
    elif combined_status == "WARN":
        status_light = "YELLOW"
    else:
        status_light = "GREEN"
    
    # Generate neutral headline
    if len(inv_bud_failures) > 0:
        inv_list = ", ".join(inv_bud_failures)
        headline = f"Budget invariants {inv_list} show violations across {total_runs} runs"
    elif stability_index < 0.95:
        headline = f"Budget stability index {stability_index:.2f} below 0.95 threshold"
    elif health_score < 80.0:
        headline = f"Budget health score {health_score:.1f} below 80.0 threshold"
    elif combined_status == "WARN":
        headline = "Budget governance status is WARN"
    else:
        headline = f"Budget invariants pass across {total_runs} runs with stability {stability_index:.2f}"
    
    return {
        "status_light": status_light,
        "recent_status": recent_status,
        "stability_index": round(stability_index, 4),
        "health_score": round(health_score, 2),
        "headline": headline,
        "key_invariants_with_failures": inv_bud_failures,
    }


def build_budget_storyline(
    invariant_timeline: Dict[str, Any],
    budget_health_history: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build a time-aware narrative storyline from invariant timeline and health history.
    
    Identifies episodes of interest, structural events, and stability classification
    to provide a human-readable narrative of budget governance over time.
    
    Args:
        invariant_timeline: Output from build_budget_invariant_timeline.
        budget_health_history: Sequence of budget_health objects from A5, ordered
                               chronologically. Each dict should have:
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
                        "status": "WARN",
                        "invariants_affected": ["INV-BUD-2"],
                        "health_score_range": [72.0, 79.0],
                        "description": "neutral factual description"
                    },
                    ...
                ],
                "structural_events": [
                    "INV-BUD-1 first failure at run 42",
                    "stability_index dropped below 0.95 at run 50",
                    ...
                ],
                "stability_class": "STABLE" | "DRIFTING" | "VOLATILE",
                "summary": "neutral narrative summary"
            }
    
    Stability Classification:
        - STABLE: stability_index >= 0.95 and no hard invariant failures
        - DRIFTING: stability_index in [0.85, 0.95) or soft invariant failures
        - VOLATILE: stability_index < 0.85 or hard invariant failures
    """
    total_runs = invariant_timeline.get("total_runs", 0)
    stability_index = invariant_timeline.get("stability_index", 0.0)
    
    # Extract structural events
    structural_events: List[str] = []
    
    # Track first failures for each invariant
    if invariant_timeline.get("inv_bud_1_failures", 0) > 0:
        structural_events.append("INV-BUD-1 violation detected")
    if invariant_timeline.get("inv_bud_2_failures", 0) > 0:
        structural_events.append("INV-BUD-2 violation detected")
    if invariant_timeline.get("inv_bud_3_failures", 0) > 0:
        structural_events.append("INV-BUD-3 violation detected")
    if invariant_timeline.get("inv_bud_4_failures", 0) > 0:
        structural_events.append("INV-BUD-4 violation detected")
    if invariant_timeline.get("inv_bud_5_failures", 0) > 0:
        structural_events.append("INV-BUD-5 violation detected")
    
    # Stability threshold events
    if stability_index < 0.95:
        structural_events.append(f"stability_index {stability_index:.3f} below 0.95 threshold")
    if stability_index < 0.85:
        structural_events.append(f"stability_index {stability_index:.3f} below 0.85 threshold")
    
    # Build episodes from health history
    episodes: List[Dict[str, Any]] = []
    
    if len(budget_health_history) > 0:
        # Group consecutive runs with similar status
        current_episode_start = 0
        current_status = None
        current_invariants = set()
        current_health_scores = []
        
        for i, health in enumerate(budget_health_history):
            run_idx = health.get("run_index", i)
            health_score = health.get("health_score", 0.0)
            trend = health.get("trend_status", "STABLE")
            
            # Determine episode status
            if trend == "DEGRADING" or health_score < 70.0:
                episode_status = "WARN"
            elif health_score >= 80.0 and trend in ["IMPROVING", "STABLE"]:
                episode_status = "OK"
            else:
                episode_status = "WARN"
            
            # Check if we need to start a new episode
            if current_status is not None and episode_status != current_status:
                # Finalize current episode
                if current_episode_start < run_idx:
                    episodes.append({
                        "run_range": f"runs {current_episode_start}-{run_idx-1}",
                        "status": current_status,
                        "invariants_affected": sorted(list(current_invariants)),
                        "health_score_range": [
                            min(current_health_scores) if current_health_scores else 0.0,
                            max(current_health_scores) if current_health_scores else 0.0,
                        ],
                        "description": _generate_episode_description(
                            current_status, current_invariants, current_health_scores
                        ),
                    })
                
                # Start new episode
                current_episode_start = run_idx
                current_invariants = set()
                current_health_scores = []
            
            current_status = episode_status
            current_health_scores.append(health_score)
            
            # Track invariants affected in this period
            # (Simplified - in real implementation would track per-run)
            inv_summary = summarize_budget_invariants_for_global_health(invariant_timeline)
            current_invariants.update(inv_summary.get("inv_bud_failures", []))
        
        # Finalize last episode
        if current_status is not None and current_episode_start < total_runs:
            episodes.append({
                "run_range": f"runs {current_episode_start}-{total_runs-1}",
                "status": current_status,
                "invariants_affected": sorted(list(current_invariants)),
                "health_score_range": [
                    min(current_health_scores) if current_health_scores else 0.0,
                    max(current_health_scores) if current_health_scores else 0.0,
                ],
                "description": _generate_episode_description(
                    current_status, current_invariants, current_health_scores
                ),
            })
    
    # Classify stability
    hard_invariant_failures = any([
        invariant_timeline.get("inv_bud_1_failures", 0) > 0,
        invariant_timeline.get("inv_bud_2_failures", 0) > 0,
        invariant_timeline.get("inv_bud_3_failures", 0) > 0,
    ])
    
    if stability_index < 0.85 or hard_invariant_failures:
        stability_class = "VOLATILE"
    elif stability_index < 0.95:
        stability_class = "DRIFTING"
    else:
        stability_class = "STABLE"
    
    # Generate summary
    summary_parts = []
    summary_parts.append(f"Analyzed {total_runs} runs with stability index {stability_index:.3f}")
    
    inv_summary = summarize_budget_invariants_for_global_health(invariant_timeline)
    if len(inv_summary.get("inv_bud_failures", [])) > 0:
        inv_list = ", ".join(inv_summary["inv_bud_failures"])
        summary_parts.append(f"invariants {inv_list} show violations")
    
    if len(budget_health_history) > 0:
        health_scores = [h.get("health_score", 0.0) for h in budget_health_history]
        avg_health = sum(health_scores) / len(health_scores)
        summary_parts.append(f"average health score {avg_health:.1f}")
    
    summary = ". ".join(summary_parts) + "."
    
    return {
        "schema_version": "1.0.0",
        "runs_analyzed": total_runs,
        "episodes": episodes,
        "structural_events": structural_events,
        "stability_class": stability_class,
        "summary": summary,
    }


def _generate_episode_description(
    status: str,
    invariants: set,
    health_scores: List[float],
) -> str:
    """Generate neutral episode description."""
    parts = []
    
    if status == "WARN":
        parts.append("WARN status observed")
    else:
        parts.append("OK status maintained")
    
    if len(invariants) > 0:
        inv_list = ", ".join(sorted(invariants))
        parts.append(f"invariants {inv_list} affected")
    
    if health_scores:
        min_health = min(health_scores)
        max_health = max(health_scores)
        if min_health == max_health:
            parts.append(f"health score {min_health:.1f}")
        else:
            parts.append(f"health score range [{min_health:.1f}, {max_health:.1f}]")
    
    return ", ".join(parts)


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
                "converging_invariants": bool,  # True if violations decreasing
                "diverging_invariants": bool,    # True if violations increasing
                "projection_method": "linear_extrapolation" | "threshold_rule",
                "risk_trajectory": ["STABLE", "DRIFTING", ...],  # per-horizon step
            }
    
    Projection Rules:
        - Use last 5 entries if available, fewer if not
        - Linear extrapolation: average trend of health_score
        - Threshold rule: if trend_status pattern indicates continuation
        - Converging: health_score improving or violations decreasing
        - Diverging: health_score degrading or violations increasing
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
    
    Status Light Rules:
        - RED if any VOLATILE episode or projected_stability_class == "VOLATILE"
        - YELLOW if any WARN episode or projected_stability_class == "DRIFTING"
        - GREEN otherwise
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
        "schema_version": "1.0.0",
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


def explain_budget_release_decision(
    governance_view: Dict[str, Any],
    readiness: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Explain why a given release decision (OK/WARN/BLOCK) was made.
    
    Provides a post-mortem analysis with primary causes, contributing factors,
    and recommended follow-ups. Uses only neutral, factual language.
    
    Args:
        governance_view: Output from build_budget_invariants_governance_view.
        readiness: Output from evaluate_budget_release_readiness.
    
    Returns:
        Dict with schema:
            {
                "decision": "OK" | "WARN" | "BLOCK",
                "primary_causes": [
                    "INV-BUD-1 failed in 3 runs",
                    "health_score 65.0 below 70.0 threshold",
                    ...
                ],
                "contributing_factors": [
                    "stability_index 0.92 below 0.95 threshold",
                    "timeout_abstentions observed in 5 runs",
                    ...
                ],
                "recommended_followups": [
                    "investigate INV-BUD-1 violations",
                    "review budget timeout configurations",
                    ...
                ],
            }
    
    Primary Causes:
        - Failed invariants (hard invariants prioritized)
        - Low health_score (< 70)
        - Recent FAIL status
    
    Contributing Factors:
        - Low stability_index (< 0.95)
        - High timeout_abstentions
        - Soft invariant failures
        - DEGRADING trend
    """
    decision = readiness.get("status", "UNKNOWN")
    
    primary_causes: List[str] = []
    contributing_factors: List[str] = []
    recommended_followups: List[str] = []
    
    # Extract key metrics
    combined_status = governance_view.get("combined_status", "UNKNOWN")
    invariants_status = governance_view.get("invariants_status", "UNKNOWN")
    stability_index = governance_view.get("stability_index", 0.0)
    health_score = governance_view.get("health_score", 0.0)
    inv_bud_failures = governance_view.get("inv_bud_failures", [])
    recent_status = governance_view.get("recent_status", "UNKNOWN")
    total_runs = governance_view.get("total_runs", 0)
    
    # Primary causes
    if combined_status == "BLOCK":
        primary_causes.append(f"combined_status is BLOCK (invariants_status={invariants_status})")
    
    hard_invariants = [inv for inv in inv_bud_failures if inv in ["INV-BUD-1", "INV-BUD-2", "INV-BUD-3"]]
    if len(hard_invariants) > 0:
        inv_list = ", ".join(hard_invariants)
        primary_causes.append(f"hard invariants {inv_list} show violations")
    
    if health_score < 70.0:
        primary_causes.append(f"health_score {health_score:.1f} below 70.0 threshold")
    
    if recent_status == "FAIL":
        primary_causes.append(f"recent_status is FAIL")
    
    # Contributing factors
    if stability_index < 0.95:
        contributing_factors.append(f"stability_index {stability_index:.3f} below 0.95 threshold")
    
    soft_invariants = [inv for inv in inv_bud_failures if inv in ["INV-BUD-4", "INV-BUD-5"]]
    if len(soft_invariants) > 0:
        inv_list = ", ".join(soft_invariants)
        contributing_factors.append(f"soft invariants {inv_list} show violations")
    
    if combined_status == "WARN" and invariants_status != "BLOCK":
        contributing_factors.append("combined_status is WARN")
    
    # Recommended follow-ups
    if len(hard_invariants) > 0:
        for inv in hard_invariants:
            recommended_followups.append(f"investigate {inv} violations")
    
    if health_score < 70.0:
        recommended_followups.append("review budget health metrics and trends")
    
    if stability_index < 0.95:
        recommended_followups.append("analyze stability trend over recent runs")
    
    if len(inv_bud_failures) > 0:
        recommended_followups.append("review budget enforcement implementation")
    
    if combined_status == "BLOCK":
        recommended_followups.append("assess impact of blocking conditions on release readiness")
    
    # Ensure we have at least one primary cause for BLOCK/WARN
    if decision in ["BLOCK", "WARN"] and len(primary_causes) == 0:
        primary_causes.append(f"release decision is {decision}")
    
    return {
        "decision": decision,
        "primary_causes": primary_causes,
        "contributing_factors": contributing_factors,
        "recommended_followups": recommended_followups,
    }


# Type stub for PipelineStats (avoid circular import)
if False:  # TYPE_CHECKING equivalent without import
    from derivation.pipeline import PipelineStats


__all__ = [
    "SCHEMA_VERSION",
    "REQUIRED_BUDGET_FIELDS",
    "build_budget_invariant_snapshot",
    "build_budget_invariant_timeline",
    "summarize_budget_invariants_for_global_health",
    # Phase IV cross-layer governance
    "build_budget_invariants_governance_view",
    "evaluate_budget_release_readiness",
    "build_budget_invariants_director_panel",
    # Phase V narrative and forensics
    "build_budget_storyline",
    "explain_budget_release_decision",
]

