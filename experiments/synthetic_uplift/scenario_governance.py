#!/usr/bin/env python3
"""
==============================================================================
PHASE II — SYNTHETIC TEST DATA ONLY
==============================================================================

Scenario Governance & Envelope Analytics
------------------------------------------

This module provides governance objects and analytics for synthetic scenarios:

    1. Scenario Realism Snapshots - Point-in-time capture of scenario state
    2. Realism Envelope Timeline - Historical tracking of envelope compliance
    3. Global Health Summary - Aggregated health status for synthetic region

Must NOT:
    - Produce claims about real uplift
    - Mix synthetic and real data
    - Imply empirical validity

==============================================================================
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set

from experiments.synthetic_uplift.noise_models import SAFETY_LABEL


# ==============================================================================
# CONSTANTS
# ==============================================================================

SCHEMA_VERSION = "governance_v1.0"

# Critical scenarios that should never repeatedly breach envelope
CRITICAL_SCENARIOS = {
    "synthetic_null_uplift",
    "synthetic_positive_uplift",
    "synthetic_negative_uplift",
}


# ==============================================================================
# HEALTH STATUS
# ==============================================================================

class HealthStatus(Enum):
    """Health status for synthetic region."""
    OK = "OK"
    WARN = "WARN"
    BLOCK = "BLOCK"


# ==============================================================================
# TASK 1: SCENARIO REALISM SNAPSHOT
# ==============================================================================

def build_scenario_realism_snapshot(
    scenario_name: str,
    config: Dict[str, Any],
    envelope_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a realism snapshot for a scenario.
    
    Args:
        scenario_name: Name of the scenario (must start with synthetic_)
        config: Scenario configuration/parameters dict
        envelope_result: Result from envelope check (violations, pass/fail)
    
    Returns:
        Snapshot dictionary with:
            - schema_version: Version of snapshot schema
            - scenario_name: Name of the scenario
            - drift_mode: Detected drift mode (none/sinusoidal/linear/step)
            - envelope_pass: Whether envelope check passed
            - violated_checks: List of envelope rule IDs that were violated
            - rare_event_profile: Summarized rare event configuration
            - timestamp: When snapshot was taken
    """
    if not scenario_name.startswith("synthetic_"):
        raise ValueError(f"Scenario must start with 'synthetic_': {scenario_name}")
    
    # Extract drift mode
    drift_mode = _extract_drift_mode(config)
    
    # Extract envelope pass status and violations
    envelope_pass, violated_checks = _extract_envelope_status(envelope_result)
    
    # Build rare event profile
    rare_event_profile = _build_rare_event_profile(config)
    
    # Build snapshot
    snapshot = {
        "label": SAFETY_LABEL,
        "schema_version": SCHEMA_VERSION,
        "scenario_name": scenario_name,
        "drift_mode": drift_mode,
        "envelope_pass": envelope_pass,
        "violated_checks": violated_checks,
        "rare_event_profile": rare_event_profile,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_hash": _compute_config_hash(config),
    }
    
    return snapshot


def _extract_drift_mode(config: Dict[str, Any]) -> str:
    """Extract the drift mode from scenario config."""
    # Check parameters.drift first
    params = config.get("parameters", config)
    drift = params.get("drift", {})
    drift_mode = drift.get("mode", "none")
    
    # Map cyclical to sinusoidal for consistency
    mode_map = {
        "cyclical": "sinusoidal",
        "monotonic": "linear",
        "shock": "step",
    }
    
    return mode_map.get(drift_mode, drift_mode)


def _extract_envelope_status(
    envelope_result: Dict[str, Any],
) -> tuple[bool, List[str]]:
    """Extract pass status and violated check IDs from envelope result."""
    # Handle different envelope result formats
    
    # Format 1: Direct violations list
    if "violations" in envelope_result:
        violations = envelope_result["violations"]
        violated_checks = []
        for v in violations:
            if isinstance(v, dict):
                # Build check ID from parameter
                param = v.get("parameter", "unknown")
                bound_type = v.get("bound_type", "unknown")
                check_id = f"{param}:{bound_type}"
                violated_checks.append(check_id)
            elif isinstance(v, str):
                violated_checks.append(v)
        
        passed = envelope_result.get("passed", len(violations) == 0)
        return passed, violated_checks
    
    # Format 2: Simple pass/fail
    if "passed" in envelope_result:
        return envelope_result["passed"], []
    
    if "envelope_pass" in envelope_result:
        return envelope_result["envelope_pass"], envelope_result.get("violated_checks", [])
    
    # Default: assume passed if no violations found
    return True, []


def _build_rare_event_profile(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build a summary of rare event configuration."""
    params = config.get("parameters", config)
    rare_events = params.get("rare_events", [])
    
    if not rare_events:
        return {
            "count": 0,
            "types": [],
            "has_catastrophic": False,
            "has_burst": False,
            "has_recovery": False,
            "total_trigger_probability": 0.0,
        }
    
    types = []
    has_catastrophic = False
    has_burst = False
    has_recovery = False
    total_trigger_prob = 0.0
    
    for event in rare_events:
        event_type = event.get("type", "unknown")
        types.append(event_type)
        
        if "catastrophic" in event_type.lower():
            has_catastrophic = True
        if "burst" in event_type.lower() or "outlier" in event_type.lower():
            has_burst = True
        if "recovery" in event_type.lower():
            has_recovery = True
        
        total_trigger_prob += event.get("trigger_probability", 0.0)
    
    return {
        "count": len(rare_events),
        "types": list(set(types)),
        "has_catastrophic": has_catastrophic,
        "has_burst": has_burst,
        "has_recovery": has_recovery,
        "total_trigger_probability": total_trigger_prob,
    }


def _compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute a stable hash of the configuration."""
    # Serialize deterministically
    serialized = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


# ==============================================================================
# TASK 2: REALISM ENVELOPE TIMELINE
# ==============================================================================

def build_realism_envelope_timeline(
    snapshots: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build a timeline of envelope compliance across multiple snapshots.
    
    Args:
        snapshots: Sequence of realism snapshots from build_scenario_realism_snapshot
    
    Returns:
        Timeline dictionary with:
            - per_scenario: Dict mapping scenario name to:
                - times_passed: Count of envelope passes
                - times_failed: Count of envelope failures
                - violated_checks_history: List of violated checks over time
                - pass_rate: Ratio of passes to total
            - global:
                - scenario_count: Total unique scenarios
                - total_snapshots: Total number of snapshots
                - envelope_breach_rate: Global breach rate
                - scenarios_with_repeated_breaches: List of scenarios with >1 breach
    """
    if not snapshots:
        return _empty_timeline()
    
    # Track per-scenario statistics
    per_scenario: Dict[str, Dict[str, Any]] = {}
    
    for snapshot in snapshots:
        scenario_name = snapshot.get("scenario_name", "unknown")
        envelope_pass = snapshot.get("envelope_pass", True)
        violated_checks = snapshot.get("violated_checks", [])
        timestamp = snapshot.get("timestamp", "")
        
        if scenario_name not in per_scenario:
            per_scenario[scenario_name] = {
                "times_passed": 0,
                "times_failed": 0,
                "violated_checks_history": [],
                "snapshots": [],
            }
        
        record = per_scenario[scenario_name]
        
        if envelope_pass:
            record["times_passed"] += 1
        else:
            record["times_failed"] += 1
        
        if violated_checks:
            record["violated_checks_history"].append({
                "timestamp": timestamp,
                "checks": violated_checks,
            })
        
        record["snapshots"].append({
            "timestamp": timestamp,
            "passed": envelope_pass,
            "violated_checks": violated_checks,
        })
    
    # Compute pass rates
    for scenario_name, record in per_scenario.items():
        total = record["times_passed"] + record["times_failed"]
        record["pass_rate"] = record["times_passed"] / total if total > 0 else 1.0
    
    # Compute global statistics
    total_snapshots = len(snapshots)
    scenario_count = len(per_scenario)
    
    total_breaches = sum(r["times_failed"] for r in per_scenario.values())
    envelope_breach_rate = total_breaches / total_snapshots if total_snapshots > 0 else 0.0
    
    scenarios_with_repeated_breaches = [
        name for name, record in per_scenario.items()
        if record["times_failed"] > 1
    ]
    
    return {
        "label": SAFETY_LABEL,
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "per_scenario": {
            name: {
                "times_passed": record["times_passed"],
                "times_failed": record["times_failed"],
                "violated_checks_history": record["violated_checks_history"],
                "pass_rate": record["pass_rate"],
            }
            for name, record in per_scenario.items()
        },
        "global": {
            "scenario_count": scenario_count,
            "total_snapshots": total_snapshots,
            "envelope_breach_rate": envelope_breach_rate,
            "scenarios_with_repeated_breaches": sorted(scenarios_with_repeated_breaches),
        },
    }


def _empty_timeline() -> Dict[str, Any]:
    """Return an empty timeline structure."""
    return {
        "label": SAFETY_LABEL,
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "per_scenario": {},
        "global": {
            "scenario_count": 0,
            "total_snapshots": 0,
            "envelope_breach_rate": 0.0,
            "scenarios_with_repeated_breaches": [],
        },
    }


# ==============================================================================
# TASK 3: GLOBAL HEALTH HOOK
# ==============================================================================

def summarize_synthetic_realism_for_global_health(
    timeline: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Summarize synthetic realism for global health monitoring.
    
    Args:
        timeline: Timeline from build_realism_envelope_timeline
    
    Returns:
        Health summary with:
            - realism_ok: Boolean indicating overall health
            - envelope_breach_rate: Global breach rate
            - scenarios_needing_review: List of scenarios requiring attention
            - status: OK | WARN | BLOCK
                - BLOCK: Critical scenarios repeatedly breach envelope
                - WARN: Non-critical scenarios breach or any single breach
                - OK: All scenarios within envelope
    """
    global_stats = timeline.get("global", {})
    per_scenario = timeline.get("per_scenario", {})
    
    envelope_breach_rate = global_stats.get("envelope_breach_rate", 0.0)
    repeated_breaches = set(global_stats.get("scenarios_with_repeated_breaches", []))
    
    # Identify scenarios needing review
    scenarios_needing_review = []
    
    for name, record in per_scenario.items():
        if record.get("times_failed", 0) > 0:
            scenarios_needing_review.append({
                "scenario": name,
                "times_failed": record["times_failed"],
                "pass_rate": record.get("pass_rate", 0.0),
                "is_critical": name in CRITICAL_SCENARIOS,
            })
    
    # Sort by severity (critical first, then by failure count)
    scenarios_needing_review.sort(
        key=lambda x: (-int(x["is_critical"]), -x["times_failed"])
    )
    
    # Determine status
    status = _determine_health_status(
        envelope_breach_rate=envelope_breach_rate,
        repeated_breaches=repeated_breaches,
        per_scenario=per_scenario,
    )
    
    # Overall realism OK if no breaches
    realism_ok = envelope_breach_rate == 0.0
    
    return {
        "label": SAFETY_LABEL,
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "realism_ok": realism_ok,
        "envelope_breach_rate": envelope_breach_rate,
        "scenarios_needing_review": [s["scenario"] for s in scenarios_needing_review],
        "scenarios_needing_review_details": scenarios_needing_review,
        "status": status.value,
        "critical_scenarios_monitored": sorted(CRITICAL_SCENARIOS),
    }


def _determine_health_status(
    envelope_breach_rate: float,
    repeated_breaches: Set[str],
    per_scenario: Dict[str, Any],
) -> HealthStatus:
    """
    Determine the health status based on envelope compliance.
    
    Rules:
        - BLOCK: Any critical scenario has repeated breaches
        - WARN: Any breach occurred (single or non-critical repeated)
        - OK: No breaches
    """
    # Check for critical scenarios with repeated breaches
    critical_repeated = repeated_breaches & CRITICAL_SCENARIOS
    if critical_repeated:
        return HealthStatus.BLOCK
    
    # Check for any breaches
    if envelope_breach_rate > 0:
        return HealthStatus.WARN
    
    return HealthStatus.OK


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def build_governance_report(
    scenario_configs: Dict[str, Dict[str, Any]],
    envelope_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build a complete governance report from scenario configs and envelope results.
    
    Args:
        scenario_configs: Dict mapping scenario name to config
        envelope_results: Dict mapping scenario name to envelope check result
    
    Returns:
        Complete governance report with snapshots, timeline, and health summary
    """
    # Build snapshots for all scenarios
    snapshots = []
    for name in scenario_configs:
        config = scenario_configs[name]
        envelope_result = envelope_results.get(name, {"passed": True})
        
        snapshot = build_scenario_realism_snapshot(name, config, envelope_result)
        snapshots.append(snapshot)
    
    # Build timeline
    timeline = build_realism_envelope_timeline(snapshots)
    
    # Build health summary
    health_summary = summarize_synthetic_realism_for_global_health(timeline)
    
    return {
        "label": SAFETY_LABEL,
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "snapshots": snapshots,
        "timeline": timeline,
        "health_summary": health_summary,
    }


def load_and_build_governance_report() -> Dict[str, Any]:
    """
    Load registry and build governance report.
    
    This is the main entry point for generating a governance report
    from the current scenario registry.
    """
    import json
    from pathlib import Path
    
    from experiments.synthetic_uplift.realism_envelope import RealismEnvelopeChecker
    
    # Load registry
    registry_path = Path(__file__).parent / "scenario_registry.json"
    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)
    
    scenarios = registry.get("scenarios", {})
    
    # Run envelope checks
    checker = RealismEnvelopeChecker()
    envelope_results = {}
    
    for name, scenario in scenarios.items():
        params = scenario.get("parameters", {})
        violations = checker.check_scenario(name, params)
        
        envelope_results[name] = {
            "passed": not any(v.severity == "error" for v in violations),
            "violations": [v.to_dict() for v in violations],
        }
    
    # Build configs dict
    scenario_configs = {
        name: scenario for name, scenario in scenarios.items()
    }
    
    return build_governance_report(scenario_configs, envelope_results)


# ==============================================================================
# CLI SUPPORT
# ==============================================================================

def format_governance_report(report: Dict[str, Any]) -> str:
    """Format governance report as human-readable text."""
    lines = [
        f"\n{SAFETY_LABEL}",
        "",
        "=" * 60,
        "SCENARIO GOVERNANCE REPORT",
        "=" * 60,
        "",
    ]
    
    # Health summary
    health = report.get("health_summary", {})
    status = health.get("status", "UNKNOWN")
    realism_ok = health.get("realism_ok", False)
    breach_rate = health.get("envelope_breach_rate", 0.0)
    
    lines.append("HEALTH STATUS")
    lines.append("-" * 40)
    lines.append(f"  Status:        [{status}]")
    lines.append(f"  Realism OK:    {realism_ok}")
    lines.append(f"  Breach Rate:   {breach_rate:.2%}")
    lines.append("")
    
    # Scenarios needing review
    needing_review = health.get("scenarios_needing_review_details", [])
    if needing_review:
        lines.append("SCENARIOS NEEDING REVIEW")
        lines.append("-" * 40)
        for s in needing_review:
            critical = " [CRITICAL]" if s.get("is_critical") else ""
            lines.append(f"  {s['scenario']}{critical}")
            lines.append(f"    Failed: {s['times_failed']} | Pass Rate: {s['pass_rate']:.2%}")
        lines.append("")
    
    # Timeline summary
    timeline = report.get("timeline", {})
    global_stats = timeline.get("global", {})
    
    lines.append("TIMELINE SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  Scenarios:     {global_stats.get('scenario_count', 0)}")
    lines.append(f"  Snapshots:     {global_stats.get('total_snapshots', 0)}")
    lines.append(f"  Repeated:      {len(global_stats.get('scenarios_with_repeated_breaches', []))}")
    lines.append("")
    
    lines.append("=" * 60)
    lines.append(f"{SAFETY_LABEL}")
    lines.append("")
    
    return "\n".join(lines)


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    report = load_and_build_governance_report()
    print(format_governance_report(report))

