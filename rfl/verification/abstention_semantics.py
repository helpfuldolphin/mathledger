"""Abstention semantics and analytics module.

Provides semantic categorization, validation, aggregation, and governance
functions for abstention records and epistemic risk analysis.

Phase V additions:
- build_epistemic_drift_timeline: Analyzes epistemic risk trends over time
- summarize_abstention_for_global_console: Summarizes abstention for governance dashboard
- compose_abstention_with_uplift_decision: Combines epistemic evaluation with uplift decisions
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Schema versions
ABSTENTION_TAXONOMY_VERSION = "1.0.0"
EPISTEMIC_PROFILE_SCHEMA_VERSION = "1.0.0"
DRIFT_TIMELINE_SCHEMA_VERSION = "1.0.0"
GLOBAL_CONSOLE_SCHEMA_VERSION = "1.0.0"
UPLIFT_DECISION_SCHEMA_VERSION = "1.0.0"


class AbstentionType(Enum):
    """Canonical abstention type enumeration."""

    ABSTAIN_TIMEOUT = "abstain_timeout"
    ABSTAIN_CRASH = "abstain_crash"
    ABSTAIN_INVALID = "abstain_invalid"
    ABSTAIN_BUDGET = "abstain_budget"
    ABSTAIN_RESOURCE = "abstain_resource"
    ABSTAIN_ORACLE = "abstain_oracle"
    ABSTAIN_LEAN_ERROR = "abstain_lean_error"
    ABSTAIN_UNKNOWN = "abstain_unknown"


class SemanticCategory(Enum):
    """High-level semantic categories for abstentions."""

    TIMEOUT_RELATED = "timeout_related"
    CRASH_RELATED = "crash_related"
    INVALID_RELATED = "invalid_related"
    RESOURCE_RELATED = "resource_related"
    ORACLE_RELATED = "oracle_related"
    UNKNOWN_RELATED = "unknown_related"


# Mapping from AbstentionType to SemanticCategory
ABSTENTION_TREE: Dict[AbstentionType, SemanticCategory] = {
    AbstentionType.ABSTAIN_TIMEOUT: SemanticCategory.TIMEOUT_RELATED,
    AbstentionType.ABSTAIN_CRASH: SemanticCategory.CRASH_RELATED,
    AbstentionType.ABSTAIN_INVALID: SemanticCategory.INVALID_RELATED,
    AbstentionType.ABSTAIN_BUDGET: SemanticCategory.RESOURCE_RELATED,
    AbstentionType.ABSTAIN_RESOURCE: SemanticCategory.RESOURCE_RELATED,
    AbstentionType.ABSTAIN_ORACLE: SemanticCategory.ORACLE_RELATED,
    AbstentionType.ABSTAIN_LEAN_ERROR: SemanticCategory.CRASH_RELATED,
    AbstentionType.ABSTAIN_UNKNOWN: SemanticCategory.UNKNOWN_RELATED,
}


def categorize(abstention_type: AbstentionType) -> SemanticCategory:
    """Get semantic category for an abstention type."""
    return ABSTENTION_TREE.get(abstention_type, SemanticCategory.UNKNOWN_RELATED)


# ─────────────────────────────────────────────────────────────────────────────
# Phase V: Epistemic Drift Timeline
# ─────────────────────────────────────────────────────────────────────────────


def build_epistemic_drift_timeline(
    profiles: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build epistemic drift timeline from a sequence of epistemic profiles.

    Analyzes risk band patterns over time to detect drift and volatility.

    Args:
        profiles: Sequence of epistemic abstention profiles, each with
                 'epistemic_risk_band' field (LOW/MEDIUM/HIGH)

    Returns:
        {
            "schema_version": "1.0.0",
            "drift_index": float,  # 0.0 = stable, 1.0 = volatile
            "risk_band": "STABLE" | "DRIFTING" | "VOLATILE",
            "change_points": List[Dict],  # Transitions detected
            "summary_text": str
        }
    """
    if not profiles:
        return {
            "schema_version": DRIFT_TIMELINE_SCHEMA_VERSION,
            "drift_index": 0.0,
            "risk_band": "STABLE",
            "change_points": [],
            "summary_text": "No profiles available for drift analysis",
        }

    if len(profiles) == 1:
        risk_band = profiles[0].get("epistemic_risk_band", "LOW")
        return {
            "schema_version": DRIFT_TIMELINE_SCHEMA_VERSION,
            "drift_index": 0.0,
            "risk_band": "STABLE",
            "change_points": [],
            "summary_text": f"Single profile with {risk_band} risk band (baseline established)",
        }

    # Map risk bands to numeric values
    risk_map = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    risk_values = [
        risk_map.get(p.get("epistemic_risk_band", "LOW"), 0) for p in profiles
    ]

    # Compute variance (normalized to 0-1 range)
    mean_risk = sum(risk_values) / len(risk_values)
    variance = sum((v - mean_risk) ** 2 for v in risk_values) / len(risk_values)
    # Max variance for 3-band system: (0-1)^2 + (1-1)^2 + (2-1)^2 = 1.33
    max_variance = 1.33
    drift_index = min(1.0, variance / max_variance) if max_variance > 0 else 0.0

    # Classify drift band
    if drift_index <= 0.2:
        drift_band = "STABLE"
    elif drift_index <= 0.6:
        drift_band = "DRIFTING"
    else:
        drift_band = "VOLATILE"

    # Detect change points (transitions between risk bands)
    change_points: List[Dict[str, Any]] = []
    for i in range(1, len(risk_values)):
        prev_risk = risk_values[i - 1]
        curr_risk = risk_values[i]
        change_magnitude = abs(curr_risk - prev_risk) / 2.0  # Normalize to 0-1

        if change_magnitude >= 0.3:  # Significant transition
            prev_band = profiles[i - 1].get("epistemic_risk_band", "LOW")
            curr_band = profiles[i].get("epistemic_risk_band", "LOW")
            slice_name = profiles[i].get("slice_name", f"slice_{i}")

            change_points.append(
                {
                    "slice_name": slice_name,
                    "transition": f"{prev_band} → {curr_band}",
                    "change_magnitude": round(change_magnitude, 3),
                    "index": i,
                }
            )

    # Generate summary text
    if change_points:
        summary_text = (
            f"Drift analysis: {drift_band} pattern (drift_index={drift_index:.2f}). "
            f"Detected {len(change_points)} significant transition(s)."
        )
    else:
        summary_text = (
            f"Drift analysis: {drift_band} pattern (drift_index={drift_index:.2f}). "
            f"No significant transitions detected."
        )

    return {
        "schema_version": DRIFT_TIMELINE_SCHEMA_VERSION,
        "drift_index": round(drift_index, 3),
        "risk_band": drift_band,
        "change_points": change_points,
        "summary_text": summary_text,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase V: Global Console Adapter
# ─────────────────────────────────────────────────────────────────────────────


def summarize_abstention_for_global_console(
    profile: Dict[str, Any],
    storyline: Dict[str, Any],
    drift_timeline: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Summarize abstention and epistemic risk for global governance console.

    Combines epistemic profile, storyline, and drift timeline into a unified
    console-friendly summary.

    Args:
        profile: Epistemic abstention profile (from build_epistemic_abstention_profile)
        storyline: Abstention storyline (from build_abstention_storyline)
        drift_timeline: Drift timeline (from build_epistemic_drift_timeline)

    Returns:
        {
            "schema_version": "1.0.0",
            "abstention_status_light": "GREEN" | "YELLOW" | "RED",
            "epistemic_risk": "LOW" | "MEDIUM" | "HIGH",
            "storyline_snapshot": Dict,
            "drift_band": str,
            "headline": str
        }
    """
    epistemic_risk = profile.get("epistemic_risk_band", "LOW")
    drift_band = drift_timeline.get("risk_band", "STABLE")
    global_trend = storyline.get("global_epistemic_trend", "STABLE")

    # Determine status light (multi-factor logic)
    if epistemic_risk == "HIGH" or drift_band == "VOLATILE":
        status_light = "RED"
    elif (
        epistemic_risk == "MEDIUM"
        or drift_band == "DRIFTING"
        or global_trend == "DEGRADING"
    ):
        status_light = "YELLOW"
    else:
        status_light = "GREEN"

    # Extract storyline snapshot
    storyline_snapshot = {
        "trend": global_trend,
        "story": storyline.get("story", ""),
    }

    # Generate headline
    headline = (
        f"Abstention epistemic risk: {epistemic_risk}. "
        f"Drift pattern: {drift_band}. "
        f"Trend: {global_trend}."
    )

    return {
        "schema_version": GLOBAL_CONSOLE_SCHEMA_VERSION,
        "abstention_status_light": status_light,
        "epistemic_risk": epistemic_risk,
        "storyline_snapshot": storyline_snapshot,
        "drift_band": drift_band,
        "headline": headline,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase V: Uplift Decision Composition
# ─────────────────────────────────────────────────────────────────────────────


def compose_abstention_with_uplift_decision(
    epistemic_eval: Dict[str, Any],
    uplift_eval: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compose abstention epistemic evaluation with uplift evaluation.

    Epistemic gate has veto power: if epistemic status is BLOCK, final status is BLOCK.
    Otherwise, uses upgrade rules to combine both evaluations.

    Args:
        epistemic_eval: Epistemic evaluation dict with 'status' (OK/WARN/BLOCK),
                       'blocking_slices', 'reasons'
        uplift_eval: Uplift evaluation dict with 'uplift_safety_decision' (PASS/WARN/BLOCK),
                     'decision_rationale', 'risk_band'

    Returns:
        {
            "schema_version": "1.0.0",
            "final_status": "OK" | "WARN" | "BLOCK",
            "uplift_ok": bool,  # True if not blocked
            "epistemic_upgrade_applied": bool,
            "blocking_slices": List[str],
            "reasons": List[str],
            "advisory": Optional[str]
        }
    """
    epistemic_status = epistemic_eval.get("status", "OK")
    uplift_status = uplift_eval.get("uplift_safety_decision", "PASS")

    # Map uplift status to epistemic status format
    uplift_status_map = {"PASS": "OK", "WARN": "WARN", "BLOCK": "BLOCK"}
    uplift_status_normalized = uplift_status_map.get(uplift_status, "OK")

    # Upgrade rules: epistemic gate has veto power
    epistemic_upgrade_applied = False
    if epistemic_status == "BLOCK":
        final_status = "BLOCK"
        epistemic_upgrade_applied = True
    elif uplift_status_normalized == "BLOCK":
        # Uplift gate can also block (even if epistemic is OK)
        final_status = "BLOCK"
        epistemic_upgrade_applied = False
    elif epistemic_status == "WARN" and uplift_status_normalized == "OK":
        final_status = "WARN"
        epistemic_upgrade_applied = True
    elif epistemic_status == "WARN" and uplift_status_normalized == "WARN":
        final_status = "WARN"
    elif epistemic_status == "OK" and uplift_status_normalized == "WARN":
        final_status = "WARN"
    else:  # Both OK
        final_status = "OK"

    # Combine blocking slices
    epistemic_blocking = epistemic_eval.get("blocking_slices", [])
    uplift_blocking = uplift_eval.get("blocking_slices", [])
    blocking_slices = sorted(list(set(epistemic_blocking + uplift_blocking)))

    # Combine reasons
    epistemic_reasons = epistemic_eval.get("reasons", [])
    uplift_reasons = uplift_eval.get("decision_rationale", [])
    reasons = epistemic_reasons + [
        f"Uplift gate: {r}" for r in uplift_reasons if r not in epistemic_reasons
    ]

    # Generate advisory message if epistemic upgrade was applied
    advisory = None
    if epistemic_upgrade_applied and final_status in ["WARN", "BLOCK"]:
        advisory = (
            f"Epistemic gate {final_status.lower()}ed uplift decision "
            f"(original uplift status: {uplift_status_normalized})"
        )

    return {
        "schema_version": UPLIFT_DECISION_SCHEMA_VERSION,
        "final_status": final_status,
        "uplift_ok": final_status != "BLOCK",
        "epistemic_upgrade_applied": epistemic_upgrade_applied,
        "blocking_slices": blocking_slices,
        "reasons": reasons,
        "advisory": advisory,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase V: Abstention Storyline (helper for global console)
# ─────────────────────────────────────────────────────────────────────────────


def build_abstention_storyline(
    profiles: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build abstention storyline from epistemic profiles.

    Creates a narrative summary of epistemic risk trends.

    Args:
        profiles: Sequence of epistemic abstention profiles

    Returns:
        {
            "schema_version": "1.0.0",
            "global_epistemic_trend": "IMPROVING" | "STABLE" | "DEGRADING",
            "story": str
        }
    """
    if not profiles:
        return {
            "schema_version": "1.0.0",
            "global_epistemic_trend": "STABLE",
            "story": "No profiles available for storyline analysis",
        }

    risk_map = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    risk_values = [
        risk_map.get(p.get("epistemic_risk_band", "LOW"), 0) for p in profiles
    ]

    # Determine trend
    if len(risk_values) < 2:
        trend = "STABLE"
    else:
        first_half = risk_values[: len(risk_values) // 2]
        second_half = risk_values[len(risk_values) // 2 :]
        first_avg = sum(first_half) / len(first_half) if first_half else 0
        second_avg = sum(second_half) / len(second_half) if second_half else 0

        if second_avg < first_avg - 0.1:
            trend = "IMPROVING"
        elif second_avg > first_avg + 0.1:
            trend = "DEGRADING"
        else:
            trend = "STABLE"

    # Generate story
    story = (
        f"Epistemic risk trend: {trend}. "
        f"Analyzed {len(profiles)} profile(s). "
        f"Current risk band: {profiles[-1].get('epistemic_risk_band', 'LOW')}."
    )

    return {
        "schema_version": "1.0.0",
        "global_epistemic_trend": trend,
        "story": story,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase V: Epistemic Abstention Profile (stub for integration)
# ─────────────────────────────────────────────────────────────────────────────


def build_epistemic_abstention_profile(
    snapshot: Dict[str, Any],
    verifier_noise_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build epistemic abstention profile from health snapshot.

    This is a stub implementation for integration purposes.
    Full implementation would analyze timeout_rate, crash_rate, invalid_rate
    and correlate with verifier noise stats.

    Args:
        snapshot: Abstention health snapshot
        verifier_noise_stats: Optional verifier noise statistics

    Returns:
        {
            "schema_version": "1.0.0",
            "slice_name": str,
            "epistemic_risk_band": "LOW" | "MEDIUM" | "HIGH",
            "timeout_rate": float,
            "crash_rate": float,
            "invalid_rate": float,
            "verifier_noise_correlation": Optional[float]
        }
    """
    # Extract rates from snapshot
    by_category = snapshot.get("by_category", {})
    timeout_rate = by_category.get("timeout_related", 0.0) / 100.0
    crash_rate = by_category.get("crash_related", 0.0) / 100.0
    invalid_rate = by_category.get("invalid_related", 0.0) / 100.0

    # Determine risk band (simplified logic)
    if invalid_rate > 0.7 or crash_rate > 0.5:
        risk_band = "HIGH"
    elif invalid_rate > 0.4 or crash_rate > 0.3 or timeout_rate > 0.6:
        risk_band = "MEDIUM"
    else:
        risk_band = "LOW"

    # Compute verifier noise correlation if stats provided
    verifier_noise_correlation = None
    if verifier_noise_stats:
        # Stub: would compute actual correlation
        verifier_noise_correlation = 0.0

    return {
        "schema_version": EPISTEMIC_PROFILE_SCHEMA_VERSION,
        "slice_name": snapshot.get("slice_name", "unknown"),
        "epistemic_risk_band": risk_band,
        "timeout_rate": round(timeout_rate, 3),
        "crash_rate": round(crash_rate, 3),
        "invalid_rate": round(invalid_rate, 3),
        "verifier_noise_correlation": verifier_noise_correlation,
    }

