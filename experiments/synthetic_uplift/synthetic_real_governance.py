#!/usr/bin/env python3
"""
==============================================================================
PHASE II — SYNTHETIC TEST DATA ONLY
==============================================================================

Synthetic vs Real Governance & RFL-Coupled Scenario Control
------------------------------------------------------------

This module provides governance functions for positioning the synthetic
universe as a governed, deliberately sculpted frontier that integrates
with real logs and RFL.

Functions:
    1. build_synthetic_real_consistency_view - Compare synthetic with real
    2. derive_synthetic_scenario_policy - RFL-coupled policy derivation
    3. build_synthetic_director_panel - Executive summary panel

Must NOT:
    - Produce claims about real uplift
    - Mix synthetic and real data in outputs
    - Imply empirical validity

==============================================================================
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Set

from experiments.synthetic_uplift.noise_models import SAFETY_LABEL


# ==============================================================================
# CONSTANTS
# ==============================================================================

SCHEMA_VERSION = "synthetic_real_governance_v1.0"


# ==============================================================================
# ENUMS
# ==============================================================================

class ConsistencyStatus(Enum):
    """Consistency status between synthetic and real."""
    ALIGNED = "ALIGNED"
    PARTIAL = "PARTIAL"
    MISALIGNED = "MISALIGNED"


class PolicyStatus(Enum):
    """Policy status for RFL-coupled scenarios."""
    OK = "OK"
    ATTENTION = "ATTENTION"
    BLOCK = "BLOCK"


class StatusLight(Enum):
    """Status light for director panel."""
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


# ==============================================================================
# TASK 1: SYNTHETIC vs REAL CONSISTENCY VIEW
# ==============================================================================

def build_synthetic_real_consistency_view(
    synthetic_timeline: Dict[str, Any],
    real_topology_health: Dict[str, Any],
    real_metric_health: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a consistency view comparing synthetic scenarios with real-world data.
    
    Args:
        synthetic_timeline: Timeline from build_realism_envelope_timeline
        real_topology_health: Real-world topology health metrics
        real_metric_health: Real-world metric health metrics
    
    Returns:
        Consistency view with:
            - schema_version
            - scenarios_consistent_with_real: List of scenario names
            - scenarios_more_aggressive_than_real: List of scenario names
            - scenarios_under_exploring: List of scenario names
            - consistency_status: ALIGNED | PARTIAL | MISALIGNED
    """
    per_scenario = synthetic_timeline.get("per_scenario", {})
    
    # Extract real-world characteristics
    real_breach_rate = real_topology_health.get("envelope_breach_rate", 0.0)
    real_variance_level = real_metric_health.get("variance_level", "moderate")
    real_correlation_level = real_metric_health.get("correlation_level", "low")
    real_drift_present = real_topology_health.get("has_temporal_drift", False)
    
    # Classify scenarios
    consistent = []
    more_aggressive = []
    under_exploring = []
    
    for scenario_name, record in per_scenario.items():
        pass_rate = record.get("pass_rate", 1.0)
        times_failed = record.get("times_failed", 0)
        
        # Check consistency based on breach patterns
        synthetic_breach_rate = 1.0 - pass_rate
        breach_diff = synthetic_breach_rate - real_breach_rate
        
        # Consistent: Similar breach rate to real (within 10%, exclusive)
        if abs(breach_diff) < 0.10:
            consistent.append(scenario_name)
        
        # More aggressive: Higher breach rate than real (stress-testing)
        elif breach_diff >= 0.10:
            more_aggressive.append(scenario_name)
        
        # Under-exploring: Much lower breach rate (too conservative)
        elif breach_diff <= -0.10:
            under_exploring.append(scenario_name)
        
        # Default: If no clear pattern, consider consistent
        else:
            consistent.append(scenario_name)
    
    # Determine overall consistency status
    total_scenarios = len(per_scenario)
    consistent_ratio = len(consistent) / total_scenarios if total_scenarios > 0 else 1.0
    
    if consistent_ratio >= 0.7:
        consistency_status = ConsistencyStatus.ALIGNED
    elif consistent_ratio >= 0.4:
        consistency_status = ConsistencyStatus.PARTIAL
    else:
        consistency_status = ConsistencyStatus.MISALIGNED
    
    return {
        "label": SAFETY_LABEL,
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scenarios_consistent_with_real": sorted(consistent),
        "scenarios_more_aggressive_than_real": sorted(more_aggressive),
        "scenarios_under_exploring": sorted(under_exploring),
        "consistency_status": consistency_status.value,
        "consistent_ratio": consistent_ratio,
        "real_breach_rate": real_breach_rate,
        "synthetic_breach_rate": synthetic_timeline.get("global", {}).get("envelope_breach_rate", 0.0),
    }


# ==============================================================================
# TASK 2: RFL-COUPLED SCENARIO POLICY
# ==============================================================================

def derive_synthetic_scenario_policy(
    consistency_view: Dict[str, Any],
    realism_timeline: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Derive advisory policy for RFL-coupled scenario experiments.
    
    Args:
        consistency_view: Output from build_synthetic_real_consistency_view
        realism_timeline: Timeline from build_realism_envelope_timeline
    
    Returns:
        Policy with:
            - scenarios_recommended_for_rfl_experiments: List of scenario names
            - scenarios_needing_tuning: List of scenario names
            - status: OK | ATTENTION | BLOCK
            - policy_notes: Neutral rationale description
    """
    per_scenario = realism_timeline.get("per_scenario", {})
    global_stats = realism_timeline.get("global", {})
    
    # Extract consistency classifications
    consistent = set(consistency_view.get("scenarios_consistent_with_real", []))
    more_aggressive = set(consistency_view.get("scenarios_more_aggressive_than_real", []))
    under_exploring = set(consistency_view.get("scenarios_under_exploring", []))
    
    # Recommended: Consistent scenarios with good pass rates
    recommended = []
    for name in consistent:
        record = per_scenario.get(name, {})
        pass_rate = record.get("pass_rate", 0.0)
        times_failed = record.get("times_failed", 0)
        
        # Recommend if pass rate > 80% and no repeated failures
        if pass_rate >= 0.8 and times_failed <= 1:
            recommended.append(name)
    
    # Also recommend some aggressive scenarios for stress-testing
    for name in more_aggressive:
        record = per_scenario.get(name, {})
        times_failed = record.get("times_failed", 0)
        
        # Recommend if failures are controlled (not repeated)
        if times_failed <= 2:
            recommended.append(name)
    
    # Needing tuning: Under-exploring or repeated failures
    needing_tuning = []
    
    # Under-exploring scenarios need tuning to be more realistic
    needing_tuning.extend(under_exploring)
    
    # Scenarios with repeated failures need tuning
    repeated_breaches = set(global_stats.get("scenarios_with_repeated_breaches", []))
    needing_tuning.extend(repeated_breaches)
    
    # Remove duplicates
    needing_tuning = sorted(set(needing_tuning))
    
    # Determine policy status
    envelope_breach_rate = global_stats.get("envelope_breach_rate", 0.0)
    consistency_status = consistency_view.get("consistency_status", "MISALIGNED")
    
    # BLOCK: High breach rate or misaligned
    if envelope_breach_rate > 0.3 or consistency_status == "MISALIGNED":
        status = PolicyStatus.BLOCK
    
    # ATTENTION: Moderate issues
    elif envelope_breach_rate > 0.1 or consistency_status == "PARTIAL" or len(needing_tuning) > 3:
        status = PolicyStatus.ATTENTION
    
    # OK: Everything looks good
    else:
        status = PolicyStatus.OK
    
    # Build policy notes
    policy_notes = _build_policy_notes(
        recommended=recommended,
        needing_tuning=needing_tuning,
        status=status,
        consistency_status=consistency_status,
        breach_rate=envelope_breach_rate,
    )
    
    return {
        "label": SAFETY_LABEL,
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scenarios_recommended_for_rfl_experiments": sorted(recommended),
        "scenarios_needing_tuning": needing_tuning,
        "status": status.value,
        "policy_notes": policy_notes,
        "recommendation_count": len(recommended),
        "tuning_count": len(needing_tuning),
    }


def _build_policy_notes(
    recommended: List[str],
    needing_tuning: List[str],
    status: PolicyStatus,
    consistency_status: str,
    breach_rate: float,
) -> str:
    """Build neutral policy notes describing rationale."""
    notes = []
    
    notes.append(f"Synthetic universe governance status: {status.value}.")
    
    if len(recommended) > 0:
        notes.append(f"{len(recommended)} scenarios recommended for RFL experiments based on consistency and envelope compliance.")
    
    if len(needing_tuning) > 0:
        notes.append(f"{len(needing_tuning)} scenarios identified for parameter tuning to improve realism or reduce envelope breaches.")
    
    notes.append(f"Consistency status with real-world patterns: {consistency_status}.")
    notes.append(f"Global envelope breach rate: {breach_rate:.1%}.")
    
    if status == PolicyStatus.BLOCK:
        notes.append("Policy recommends blocking RFL experiments until envelope compliance improves or consistency is restored.")
    elif status == PolicyStatus.ATTENTION:
        notes.append("Policy recommends attention to identified scenarios before proceeding with RFL experiments.")
    else:
        notes.append("Policy indicates synthetic universe is ready for RFL experiment integration.")
    
    return " ".join(notes)


# ==============================================================================
# TASK 3: DIRECTOR SYNTHETIC PANEL
# ==============================================================================

def build_synthetic_director_panel(
    realism_summary: Dict[str, Any],
    consistency_view: Dict[str, Any],
    scenario_policy: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build executive director panel summarizing synthetic universe posture.
    
    Args:
        realism_summary: Output from summarize_synthetic_realism_for_global_health
        consistency_view: Output from build_synthetic_real_consistency_view
        scenario_policy: Output from derive_synthetic_scenario_policy
    
    Returns:
        Director panel with:
            - status_light: GREEN | YELLOW | RED
            - realism_ok: Boolean
            - consistency_status: String
            - scenarios_needing_review: List of scenario names
            - headline: Short neutral sentence
    """
    realism_ok = realism_summary.get("realism_ok", False)
    realism_status = realism_summary.get("status", "UNKNOWN")
    consistency_status = consistency_view.get("consistency_status", "UNKNOWN")
    policy_status = scenario_policy.get("status", "UNKNOWN")
    
    # Aggregate scenarios needing review
    scenarios_needing_review = set()
    scenarios_needing_review.update(realism_summary.get("scenarios_needing_review", []))
    scenarios_needing_review.update(scenario_policy.get("scenarios_needing_tuning", []))
    
    # Determine status light
    status_light = _determine_status_light(
        realism_ok=realism_ok,
        realism_status=realism_status,
        consistency_status=consistency_status,
        policy_status=policy_status,
    )
    
    # Build headline
    headline = _build_headline(
        status_light=status_light,
        realism_ok=realism_ok,
        consistency_status=consistency_status,
        review_count=len(scenarios_needing_review),
    )
    
    return {
        "label": SAFETY_LABEL,
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status_light": status_light.value,
        "realism_ok": realism_ok,
        "consistency_status": consistency_status,
        "scenarios_needing_review": sorted(scenarios_needing_review),
        "headline": headline,
        "realism_status": realism_status,
        "policy_status": policy_status,
        "recommended_scenarios": scenario_policy.get("scenarios_recommended_for_rfl_experiments", []),
    }


def _determine_status_light(
    realism_ok: bool,
    realism_status: str,
    consistency_status: str,
    policy_status: str,
) -> StatusLight:
    """
    Determine status light based on all health indicators.
    
    Rules:
        - RED: BLOCK status in any component
        - YELLOW: WARN/ATTENTION status or not OK
        - GREEN: All OK
    """
    # RED: Any BLOCK
    if realism_status == "BLOCK" or policy_status == "BLOCK":
        return StatusLight.RED
    
    # RED: MISALIGNED consistency
    if consistency_status == "MISALIGNED":
        return StatusLight.RED
    
    # YELLOW: Any WARN/ATTENTION or not OK
    if not realism_ok or realism_status == "WARN" or policy_status == "ATTENTION":
        return StatusLight.YELLOW
    
    # YELLOW: PARTIAL consistency
    if consistency_status == "PARTIAL":
        return StatusLight.YELLOW
    
    # GREEN: All OK
    return StatusLight.GREEN


def _build_headline(
    status_light: StatusLight,
    realism_ok: bool,
    consistency_status: str,
    review_count: int,
) -> str:
    """Build a short neutral headline summarizing synthetic universe posture."""
    if status_light == StatusLight.GREEN:
        return f"Synthetic universe operational: {review_count} scenarios under review, consistency {consistency_status.lower()}."
    
    elif status_light == StatusLight.YELLOW:
        return f"Synthetic universe requires attention: {review_count} scenarios need review, consistency {consistency_status.lower()}."
    
    else:  # RED
        return f"Synthetic universe blocked: {review_count} scenarios require immediate review, consistency {consistency_status.lower()}."


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def build_complete_governance_view(
    synthetic_timeline: Dict[str, Any],
    realism_summary: Dict[str, Any],
    real_topology_health: Dict[str, Any],
    real_metric_health: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build complete governance view combining all components.
    
    This is the main entry point for generating a full governance report.
    """
    # Build consistency view
    consistency_view = build_synthetic_real_consistency_view(
        synthetic_timeline=synthetic_timeline,
        real_topology_health=real_topology_health,
        real_metric_health=real_metric_health,
    )
    
    # Derive policy
    scenario_policy = derive_synthetic_scenario_policy(
        consistency_view=consistency_view,
        realism_timeline=synthetic_timeline,
    )
    
    # Build director panel
    director_panel = build_synthetic_director_panel(
        realism_summary=realism_summary,
        consistency_view=consistency_view,
        scenario_policy=scenario_policy,
    )
    
    return {
        "label": SAFETY_LABEL,
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "realism_summary": realism_summary,
        "consistency_view": consistency_view,
        "scenario_policy": scenario_policy,
        "director_panel": director_panel,
    }


def format_director_panel(panel: Dict[str, Any]) -> str:
    """Format director panel as human-readable text."""
    lines = [
        f"\n{SAFETY_LABEL}",
        "",
        "=" * 60,
        "SYNTHETIC UNIVERSE DIRECTOR PANEL",
        "=" * 60,
        "",
    ]
    
    # Status light
    status_light = panel.get("status_light", "UNKNOWN")
    lines.append(f"Status Light: [{status_light}]")
    lines.append("")
    
    # Headline
    headline = panel.get("headline", "")
    lines.append("HEADLINE")
    lines.append("-" * 40)
    lines.append(f"  {headline}")
    lines.append("")
    
    # Key metrics
    lines.append("KEY METRICS")
    lines.append("-" * 40)
    lines.append(f"  Realism OK:        {panel.get('realism_ok', False)}")
    lines.append(f"  Consistency:       {panel.get('consistency_status', 'UNKNOWN')}")
    lines.append(f"  Realism Status:    {panel.get('realism_status', 'UNKNOWN')}")
    lines.append(f"  Policy Status:     {panel.get('policy_status', 'UNKNOWN')}")
    lines.append("")
    
    # Scenarios needing review
    needing_review = panel.get("scenarios_needing_review", [])
    if needing_review:
        lines.append("SCENARIOS NEEDING REVIEW")
        lines.append("-" * 40)
        for scenario in needing_review:
            lines.append(f"  - {scenario}")
        lines.append("")
    
    # Recommended scenarios
    recommended = panel.get("recommended_scenarios", [])
    if recommended:
        lines.append("RECOMMENDED FOR RFL EXPERIMENTS")
        lines.append("-" * 40)
        for scenario in recommended:
            lines.append(f"  - {scenario}")
        lines.append("")
    
    lines.append("=" * 60)
    lines.append(f"{SAFETY_LABEL}")
    lines.append("")
    
    return "\n".join(lines)


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    # Example usage with mock data
    from experiments.synthetic_uplift.scenario_governance import (
        build_realism_envelope_timeline,
        summarize_synthetic_realism_for_global_health,
    )
    
    # Mock timeline
    mock_timeline = build_realism_envelope_timeline([
        {
            "scenario_name": "synthetic_test",
            "envelope_pass": True,
            "violated_checks": [],
            "timestamp": "2025-01-01T00:00:00Z",
        }
    ])
    
    # Mock realism summary
    mock_summary = summarize_synthetic_realism_for_global_health(mock_timeline)
    
    # Mock real health data
    mock_real_topology = {
        "envelope_breach_rate": 0.05,
        "has_temporal_drift": False,
    }
    
    mock_real_metrics = {
        "variance_level": "moderate",
        "correlation_level": "low",
    }
    
    # Build complete view
    view = build_complete_governance_view(
        synthetic_timeline=mock_timeline,
        realism_summary=mock_summary,
        real_topology_health=mock_real_topology,
        real_metric_health=mock_real_metrics,
    )
    
    # Print director panel
    print(format_director_panel(view["director_panel"]))

