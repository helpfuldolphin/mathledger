# PHASE II — NOT USED IN PHASE I
"""
Confusability Risk Map & Family Dynamics Engine

This module provides family-level risk analysis and drift detection for
confusability contracts. It enables governance monitoring without making
uplift claims or using prescriptive language.

=============================================================================
RISK LEVEL DEFINITIONS
=============================================================================

Risk levels are assigned based on confusability and difficulty:

| Confusability | Difficulty | Risk Level |
|---------------|------------|------------|
| >= 0.7        | hard       | HIGH       |
| >= 0.7        | medium     | MEDIUM     |
| >= 0.7        | easy       | MEDIUM     |
| 0.4-0.7       | hard       | MEDIUM     |
| 0.4-0.7       | medium     | LOW        |
| 0.4-0.7       | easy       | LOW        |
| < 0.4         | any        | LOW        |

Risk reflects structural proximity to targets, NOT performance quality.
Higher risk = more structurally similar to targets = harder to distinguish.

=============================================================================
LANGUAGE CONSTRAINTS
=============================================================================

All human-readable strings in this module must be:
- Neutral and descriptive
- NO prescriptive verbs: "fix", "change", "modify", "improve", "correct"
- NO uplift claims: "better", "worse", "improvement", "degradation"
- NO value judgments: "good", "bad", "poor", "excellent"

Example allowed: "Risk level increased from LOW to HIGH"
Example forbidden: "Risk worsened - needs fixing"

=============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# Schema version for risk snapshots
RISK_SCHEMA_VERSION = "1.0.0"

# Risk level definitions
RISK_LEVELS = ["LOW", "MEDIUM", "HIGH"]

# Risk level ordering for comparisons
RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


# =============================================================================
# RISK LEVEL COMPUTATION
# =============================================================================

def compute_family_risk_level(
    avg_confusability: float,
    difficulty_band: str,
) -> str:
    """
    Compute risk level for a family based on confusability and difficulty.
    
    Risk Matrix:
    - HIGH: High confusability (>=0.7) + hard difficulty
    - MEDIUM: High confusability + non-hard OR moderate confusability + hard
    - LOW: Low-moderate confusability + non-hard
    
    Args:
        avg_confusability: Average confusability of family members [0, 1]
        difficulty_band: "easy", "medium", or "hard"
        
    Returns:
        "LOW", "MEDIUM", or "HIGH"
    """
    if avg_confusability >= 0.7:
        if difficulty_band == "hard":
            return "HIGH"
        else:
            return "MEDIUM"
    elif avg_confusability >= 0.4:
        if difficulty_band == "hard":
            return "MEDIUM"
        else:
            return "LOW"
    else:
        return "LOW"


# =============================================================================
# TASK 1: FAMILY RISK SNAPSHOT
# =============================================================================

@dataclass
class FamilyRiskEntry:
    """Risk entry for a single family."""
    fingerprint: str
    members_count: int
    avg_confusability: float
    difficulty_band: str
    risk_level: str  # "LOW" | "MEDIUM" | "HIGH"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "avg_confusability": round(self.avg_confusability, 6),
            "difficulty_band": self.difficulty_band,
            "fingerprint": self.fingerprint,
            "members_count": self.members_count,
            "risk_level": self.risk_level,
        }


def build_family_risk_snapshot(contract: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a family risk snapshot from a confusability contract.
    
    For each family, computes:
    - fingerprint
    - members_count
    - avg_confusability
    - difficulty_band
    - risk_level (LOW/MEDIUM/HIGH)
    
    Global fields:
    - high_risk_family_count
    - medium_risk_family_count
    - low_risk_family_count
    - schema_version
    - summary_notes (neutral descriptions only)
    
    Args:
        contract: Confusability contract dict (v1.1.0 format)
        
    Returns:
        Risk snapshot dict with families and global summary
    """
    families = contract.get("families", {})
    slice_name = contract.get("slice_name", "unknown")
    
    risk_entries: Dict[str, Dict[str, Any]] = {}
    high_count = 0
    medium_count = 0
    low_count = 0
    
    # Process each family
    for fingerprint in sorted(families.keys()):  # Sorted for determinism
        fam_data = families[fingerprint]
        
        members = fam_data.get("members", [])
        avg_conf = fam_data.get("avg_confusability", 0.0)
        diff_band = fam_data.get("difficulty_band", "easy")
        
        risk_level = compute_family_risk_level(avg_conf, diff_band)
        
        entry = FamilyRiskEntry(
            fingerprint=fingerprint,
            members_count=len(members),
            avg_confusability=avg_conf,
            difficulty_band=diff_band,
            risk_level=risk_level,
        )
        
        risk_entries[fingerprint] = entry.to_dict()
        
        if risk_level == "HIGH":
            high_count += 1
        elif risk_level == "MEDIUM":
            medium_count += 1
        else:
            low_count += 1
    
    # Build summary notes (neutral language only)
    summary_notes = _build_summary_notes(
        high_count, medium_count, low_count, len(families)
    )
    
    return {
        "families": risk_entries,
        "high_risk_family_count": high_count,
        "low_risk_family_count": low_count,
        "medium_risk_family_count": medium_count,
        "schema_version": RISK_SCHEMA_VERSION,
        "slice_name": slice_name,
        "summary_notes": summary_notes,
        "total_family_count": len(families),
    }


def _build_summary_notes(
    high: int,
    medium: int,
    low: int,
    total: int,
) -> str:
    """
    Build neutral summary notes for a risk snapshot.
    
    Uses only descriptive language, no value judgments.
    """
    if total == 0:
        return "No families present in contract."
    
    parts = []
    
    if high > 0:
        parts.append(f"{high} HIGH-risk {'family' if high == 1 else 'families'}")
    if medium > 0:
        parts.append(f"{medium} MEDIUM-risk {'family' if medium == 1 else 'families'}")
    if low > 0:
        parts.append(f"{low} LOW-risk {'family' if low == 1 else 'families'}")
    
    distribution = ", ".join(parts) if parts else "No families categorized"
    
    return f"Distribution: {distribution} out of {total} total."


# =============================================================================
# TASK 2: MULTI-CONTRACT FAMILY DRIFT ANALYZER
# =============================================================================

@dataclass
class FamilyDriftResult:
    """Result of comparing two family risk snapshots."""
    families_new: List[str]  # Fingerprints of new families
    families_removed: List[str]  # Fingerprints of removed families
    families_increased_risk: List[Dict[str, Any]]  # Families with higher risk
    families_decreased_risk: List[Dict[str, Any]]  # Families with lower risk
    families_unchanged: List[str]  # Fingerprints with same risk
    net_risk_trend: str  # "IMPROVING" | "STABLE" | "DEGRADING"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "families_decreased_risk": self.families_decreased_risk,
            "families_increased_risk": self.families_increased_risk,
            "families_new": sorted(self.families_new),
            "families_removed": sorted(self.families_removed),
            "families_unchanged": sorted(self.families_unchanged),
            "net_risk_trend": self.net_risk_trend,
        }


def compare_family_risk(
    old_snapshot: Dict[str, Any],
    new_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare two family risk snapshots to detect drift.
    
    Analyzes:
    - families_new: Families present in new but not old
    - families_removed: Families present in old but not new
    - families_increased_risk: Families where risk level increased
    - families_decreased_risk: Families where risk level decreased
    - net_risk_trend: Overall trend (IMPROVING/STABLE/DEGRADING)
    
    All comparisons are deterministic and keyed by fingerprint.
    
    Args:
        old_snapshot: Previous risk snapshot
        new_snapshot: Current risk snapshot
        
    Returns:
        Drift analysis dict
    """
    old_families = old_snapshot.get("families", {})
    new_families = new_snapshot.get("families", {})
    
    old_keys = set(old_families.keys())
    new_keys = set(new_families.keys())
    
    # Detect new and removed families
    families_new = sorted(new_keys - old_keys)
    families_removed = sorted(old_keys - new_keys)
    
    # Analyze common families for risk changes
    common_keys = old_keys & new_keys
    
    families_increased_risk: List[Dict[str, Any]] = []
    families_decreased_risk: List[Dict[str, Any]] = []
    families_unchanged: List[str] = []
    
    for fingerprint in sorted(common_keys):
        old_risk = old_families[fingerprint].get("risk_level", "LOW")
        new_risk = new_families[fingerprint].get("risk_level", "LOW")
        
        old_order = RISK_ORDER.get(old_risk, 0)
        new_order = RISK_ORDER.get(new_risk, 0)
        
        if new_order > old_order:
            families_increased_risk.append({
                "fingerprint": fingerprint,
                "old_risk": old_risk,
                "new_risk": new_risk,
            })
        elif new_order < old_order:
            families_decreased_risk.append({
                "fingerprint": fingerprint,
                "old_risk": old_risk,
                "new_risk": new_risk,
            })
        else:
            families_unchanged.append(fingerprint)
    
    # Compute net risk trend
    net_risk_trend = _compute_net_risk_trend(
        len(families_increased_risk),
        len(families_decreased_risk),
        len(families_new),
        len(families_removed),
        old_snapshot.get("high_risk_family_count", 0),
        new_snapshot.get("high_risk_family_count", 0),
    )
    
    result = FamilyDriftResult(
        families_new=families_new,
        families_removed=families_removed,
        families_increased_risk=families_increased_risk,
        families_decreased_risk=families_decreased_risk,
        families_unchanged=families_unchanged,
        net_risk_trend=net_risk_trend,
    )
    
    return result.to_dict()


def _compute_net_risk_trend(
    increased_count: int,
    decreased_count: int,
    new_count: int,
    removed_count: int,
    old_high_risk: int,
    new_high_risk: int,
) -> str:
    """
    Compute the net risk trend between snapshots.
    
    Uses neutral terminology:
    - IMPROVING: Net decrease in risk indicators
    - STABLE: No significant change
    - DEGRADING: Net increase in risk indicators
    
    Note: "IMPROVING" and "DEGRADING" refer to structural risk metrics,
    not performance or quality judgments.
    """
    # Primary indicator: high-risk family count change
    high_risk_delta = new_high_risk - old_high_risk
    
    # Secondary indicator: risk level transitions
    risk_transition_delta = increased_count - decreased_count
    
    # Combined score (positive = more risk, negative = less risk)
    combined_score = high_risk_delta * 2 + risk_transition_delta
    
    if combined_score > 0:
        return "DEGRADING"
    elif combined_score < 0:
        return "IMPROVING"
    else:
        return "STABLE"


# =============================================================================
# TASK 3: CONFUSABILITY GOVERNANCE SIGNAL
# =============================================================================

@dataclass
class GovernanceSignal:
    """Global health governance signal for confusability."""
    confusability_ok: bool
    high_risk_family_count: int
    risk_trend: Optional[str]  # "IMPROVING" | "STABLE" | "DEGRADING" | None
    status: str  # "OK" | "ATTENTION" | "HOT"
    summary: str  # Neutral description
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "confusability_ok": self.confusability_ok,
            "high_risk_family_count": self.high_risk_family_count,
            "status": self.status,
            "summary": self.summary,
        }
        if self.risk_trend is not None:
            result["risk_trend"] = self.risk_trend
        return result


def summarize_confusability_for_global_health(
    risk_snapshot: Dict[str, Any],
    drift: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate a governance signal for global health monitoring.
    
    Produces:
    - confusability_ok: Boolean indicating if confusability is within bounds
    - high_risk_family_count: Number of HIGH-risk families
    - risk_trend: Trend from drift analysis (if provided)
    - status: OK | ATTENTION | HOT
    - summary: Neutral descriptive summary
    
    Status levels:
    - OK: No HIGH-risk families, trend stable/improving
    - ATTENTION: Some HIGH-risk families OR degrading trend
    - HOT: Multiple HIGH-risk families AND degrading trend
    
    Args:
        risk_snapshot: Family risk snapshot
        drift: Optional drift analysis from compare_family_risk
        
    Returns:
        Governance signal dict
    """
    high_risk_count = risk_snapshot.get("high_risk_family_count", 0)
    total_families = risk_snapshot.get("total_family_count", 0)
    slice_name = risk_snapshot.get("slice_name", "unknown")
    
    # Get trend if drift provided
    risk_trend: Optional[str] = None
    if drift is not None:
        risk_trend = drift.get("net_risk_trend")
    
    # Determine status
    status = _determine_governance_status(high_risk_count, risk_trend)
    
    # Determine if confusability is OK
    confusability_ok = (status == "OK")
    
    # Build neutral summary
    summary = _build_governance_summary(
        slice_name, high_risk_count, total_families, risk_trend, status
    )
    
    signal = GovernanceSignal(
        confusability_ok=confusability_ok,
        high_risk_family_count=high_risk_count,
        risk_trend=risk_trend,
        status=status,
        summary=summary,
    )
    
    return signal.to_dict()


def _determine_governance_status(
    high_risk_count: int,
    risk_trend: Optional[str],
) -> str:
    """
    Determine governance status from risk indicators.
    
    Logic:
    - HOT: Multiple HIGH-risk families AND degrading trend
    - ATTENTION: Any HIGH-risk families OR degrading trend
    - OK: No HIGH-risk families, stable/improving or no trend data
    """
    is_degrading = (risk_trend == "DEGRADING")
    
    if high_risk_count >= 2 and is_degrading:
        return "HOT"
    elif high_risk_count > 0 or is_degrading:
        return "ATTENTION"
    else:
        return "OK"


def _build_governance_summary(
    slice_name: str,
    high_risk_count: int,
    total_families: int,
    risk_trend: Optional[str],
    status: str,
) -> str:
    """
    Build a neutral governance summary.
    
    Uses only descriptive language, no value judgments or prescriptive verbs.
    """
    parts = [f"Slice '{slice_name}':"]
    
    if total_families == 0:
        parts.append("No families present.")
    else:
        parts.append(f"{high_risk_count} of {total_families} families at HIGH risk.")
    
    if risk_trend is not None:
        parts.append(f"Trend: {risk_trend}.")
    
    parts.append(f"Status: {status}.")
    
    return " ".join(parts)


# =============================================================================
# FORBIDDEN LANGUAGE CHECK (for testing)
# =============================================================================

# Words forbidden in human-readable output
FORBIDDEN_WORDS = frozenset([
    # Prescriptive verbs
    "fix", "change", "modify", "improve", "correct", "adjust", "update",
    "must", "should", "need", "require",
    # Value judgments
    "better", "worse", "good", "bad", "poor", "excellent", "great", "terrible",
    # Uplift claims
    "improvement", "degradation", "regression", "progress",
    "uplift", "reward", "penalty", "score",
])


def check_forbidden_language(text: str) -> List[str]:
    """
    Check if text contains forbidden language.
    
    Returns list of found forbidden words (empty if clean).
    """
    text_lower = text.lower()
    words = set(text_lower.split())
    
    # Also check for word boundaries in continuous text
    found = []
    for forbidden in FORBIDDEN_WORDS:
        if forbidden in words or f" {forbidden}" in text_lower or f"{forbidden} " in text_lower:
            found.append(forbidden)
    
    return sorted(found)


# =============================================================================
# PHASE IV: CURRICULUM-COUPLED RISK CONTROL
# =============================================================================

# Difficulty band ordering for averaging
DIFFICULTY_BAND_ORDER = {"easy": 0, "medium": 1, "hard": 2}


def _average_difficulty_band(bands: List[str]) -> str:
    """
    Compute average difficulty band from a list of bands.
    
    Returns the band that represents the average (rounds to nearest).
    """
    if not bands:
        return "easy"
    
    total = sum(DIFFICULTY_BAND_ORDER.get(b, 0) for b in bands)
    avg_order = total / len(bands)
    
    # Round to nearest
    if avg_order < 0.5:
        return "easy"
    elif avg_order < 1.5:
        return "medium"
    else:
        return "hard"


def _determine_slice_confusability_status(
    high_risk_fraction: float,
    avg_difficulty: str,
) -> str:
    """
    Determine slice confusability status from risk fraction and difficulty.
    
    Logic:
    - HOT: High risk fraction (>=0.5) OR (high risk fraction >=0.3 AND hard difficulty)
    - ATTENTION: Moderate risk fraction (0.2-0.5) OR (any high risk AND medium difficulty)
    - OK: Low risk fraction (<0.2) and non-hard difficulty
    """
    if high_risk_fraction >= 0.5:
        return "HOT"
    elif high_risk_fraction >= 0.3 and avg_difficulty == "hard":
        return "HOT"
    elif high_risk_fraction >= 0.2:
        return "ATTENTION"
    elif high_risk_fraction > 0 and avg_difficulty == "hard":
        return "ATTENTION"
    else:
        return "OK"


def build_slice_confusability_view(
    contract: Dict[str, Any],
    risk_snapshot: Dict[str, Any],
    curriculum_manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build slice-level confusability governance view.
    
    This bridges curriculum structure with confusability geometry by:
    1. Mapping families to the slice (via contract)
    2. Computing aggregate risk metrics per slice
    3. Determining slice-level status
    
    Args:
        contract: Confusability contract dict (v1.1.0 format)
        risk_snapshot: Family risk snapshot from build_family_risk_snapshot
        curriculum_manifest: Optional curriculum metadata (for future extensibility)
        
    Returns:
        Dict with slice_name and computed metrics:
        {
            "slice_name": str,
            "high_risk_family_fraction": float [0, 1],
            "average_difficulty_band": "easy" | "medium" | "hard",
            "slice_confusability_status": "OK" | "ATTENTION" | "HOT",
            "total_families": int,
            "high_risk_families": int,
        }
    """
    slice_name = contract.get("slice_name", "unknown")
    families = contract.get("families", {})
    risk_families = risk_snapshot.get("families", {})
    
    if not families:
        return {
            "slice_name": slice_name,
            "high_risk_family_fraction": 0.0,
            "average_difficulty_band": "easy",
            "slice_confusability_status": "OK",
            "total_families": 0,
            "high_risk_families": 0,
        }
    
    # Collect families in this slice
    high_risk_count = 0
    difficulty_bands = []
    
    for fingerprint in families.keys():
        if fingerprint in risk_families:
            risk_entry = risk_families[fingerprint]
            if risk_entry.get("risk_level") == "HIGH":
                high_risk_count += 1
            difficulty_bands.append(risk_entry.get("difficulty_band", "easy"))
    
    total_families = len(families)
    high_risk_fraction = high_risk_count / total_families if total_families > 0 else 0.0
    avg_difficulty = _average_difficulty_band(difficulty_bands)
    status = _determine_slice_confusability_status(high_risk_fraction, avg_difficulty)
    
    return {
        "slice_name": slice_name,
        "high_risk_family_fraction": round(high_risk_fraction, 6),
        "average_difficulty_band": avg_difficulty,
        "slice_confusability_status": status,
        "total_families": total_families,
        "high_risk_families": high_risk_count,
    }


def summarize_decoy_confusability_for_uplift(
    slice_views: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Summarize decoy confusability for uplift/MAAS governance.
    
    Determines if decoys are acceptable for uplift runs based on slice-level
    confusability status. Uses neutral language only.
    
    Rules:
    - BLOCK: Any slice has status "HOT"
    - ATTENTION: Any slice has status "ATTENTION" (but no HOT)
    - OK: All slices have status "OK"
    
    Args:
        slice_views: Dict mapping slice_name -> slice_confusability_view
        
    Returns:
        {
            "decoy_ok_for_uplift": bool,
            "slices_needing_review": List[str],
            "status": "OK" | "ATTENTION" | "BLOCK",
            "hot_slices": List[str],
            "attention_slices": List[str],
        }
    """
    hot_slices = []
    attention_slices = []
    
    for slice_name, view in slice_views.items():
        status = view.get("slice_confusability_status", "OK")
        if status == "HOT":
            hot_slices.append(slice_name)
        elif status == "ATTENTION":
            attention_slices.append(slice_name)
    
    # Determine overall status
    if hot_slices:
        overall_status = "BLOCK"
        decoy_ok = False
    elif attention_slices:
        overall_status = "ATTENTION"
        decoy_ok = True  # Can proceed but needs review
    else:
        overall_status = "OK"
        decoy_ok = True
    
    slices_needing_review = sorted(hot_slices + attention_slices)
    
    return {
        "decoy_ok_for_uplift": decoy_ok,
        "slices_needing_review": slices_needing_review,
        "status": overall_status,
        "hot_slices": sorted(hot_slices),
        "attention_slices": sorted(attention_slices),
    }


def build_confusability_director_panel(
    risk_snapshot: Dict[str, Any],
    family_drift: Optional[Dict[str, Any]],
    slice_views: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build director-level confusability panel for high-level governance.
    
    Provides a single-view summary of confusability posture across all slices.
    Uses neutral, descriptive language only.
    
    Args:
        risk_snapshot: Family risk snapshot
        family_drift: Optional drift analysis from compare_family_risk
        slice_views: Dict mapping slice_name -> slice_confusability_view
        
    Returns:
        {
            "status_light": "GREEN" | "YELLOW" | "RED",
            "high_risk_family_count": int,
            "net_risk_trend": Optional[str],
            "slices_hot": List[str],
            "slices_attention": List[str],
            "headline": str,  # Neutral descriptive sentence
        }
    """
    high_risk_count = risk_snapshot.get("high_risk_family_count", 0)
    
    # Get trend from drift if available
    net_risk_trend = None
    if family_drift is not None:
        net_risk_trend = family_drift.get("net_risk_trend")
    
    # Collect slice statuses
    slices_hot = []
    slices_attention = []
    
    for slice_name, view in slice_views.items():
        status = view.get("slice_confusability_status", "OK")
        if status == "HOT":
            slices_hot.append(slice_name)
        elif status == "ATTENTION":
            slices_attention.append(slice_name)
    
    # Determine status light
    if slices_hot or (high_risk_count >= 5 and net_risk_trend == "DEGRADING"):
        status_light = "RED"
    elif slices_attention or high_risk_count >= 3 or net_risk_trend == "DEGRADING":
        status_light = "YELLOW"
    else:
        status_light = "GREEN"
    
    # Build neutral headline
    headline = _build_director_headline(
        status_light, high_risk_count, net_risk_trend, len(slices_hot), len(slices_attention)
    )
    
    return {
        "status_light": status_light,
        "high_risk_family_count": high_risk_count,
        "net_risk_trend": net_risk_trend,
        "slices_hot": sorted(slices_hot),
        "slices_attention": sorted(slices_attention),
        "headline": headline,
    }


def _build_director_headline(
    status_light: str,
    high_risk_count: int,
    net_risk_trend: Optional[str],
    hot_slice_count: int,
    attention_slice_count: int,
) -> str:
    """
    Build neutral headline for director panel.
    
    Uses only descriptive language, no value judgments.
    """
    parts = []
    
    if status_light == "RED":
        parts.append("Confusability status: RED.")
    elif status_light == "YELLOW":
        parts.append("Confusability status: YELLOW.")
    else:
        parts.append("Confusability status: GREEN.")
    
    if high_risk_count > 0:
        parts.append(f"{high_risk_count} HIGH-risk {'family' if high_risk_count == 1 else 'families'} present.")
    
    if net_risk_trend is not None:
        parts.append(f"Trend: {net_risk_trend}.")
    
    if hot_slice_count > 0:
        parts.append(f"{hot_slice_count} slice{'s' if hot_slice_count != 1 else ''} at HOT status.")
    elif attention_slice_count > 0:
        parts.append(f"{attention_slice_count} slice{'s' if attention_slice_count != 1 else ''} at ATTENTION status.")
    
    return " ".join(parts) if parts else "No confusability data available."


# =============================================================================
# PHASE IV FOLLOW-UP: DRIFT GOVERNANCE & UPLIFT PRE-SCREEN
# =============================================================================

def build_decoy_family_drift_governor(
    old_snapshot: Dict[str, Any],
    new_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build decoy family drift governor to classify drift severity.
    
    Classifies drift strictly by delta in risk bands:
    - NONE: No risk level changes
    - MINOR: Risk changes within adjacent bands (LOW↔MEDIUM, MEDIUM↔HIGH)
    - MAJOR: Risk changes across non-adjacent bands (LOW↔HIGH)
    
    Uses neutral, descriptive language only.
    
    Args:
        old_snapshot: Previous family risk snapshot
        new_snapshot: Current family risk snapshot
        
    Returns:
        {
            "drift_severity": "NONE" | "MINOR" | "MAJOR",
            "families_with_changed_risk": List[Dict],  # {fingerprint, old_risk, new_risk}
            "new_high_risk_families": List[str],  # Fingerprints that became HIGH
            "neutral_notes": List[str],  # Descriptive notes about drift
        }
    """
    old_families = old_snapshot.get("families", {})
    new_families = new_snapshot.get("families", {})
    
    # Find common families and track risk changes
    common_fingerprints = set(old_families.keys()) & set(new_families.keys())
    
    families_with_changed_risk = []
    new_high_risk_families = []
    max_severity = "NONE"
    
    for fingerprint in sorted(common_fingerprints):
        old_risk = old_families[fingerprint].get("risk_level", "LOW")
        new_risk = new_families[fingerprint].get("risk_level", "LOW")
        
        if old_risk != new_risk:
            families_with_changed_risk.append({
                "fingerprint": fingerprint,
                "old_risk": old_risk,
                "new_risk": new_risk,
            })
            
            # Track new HIGH-risk families
            if new_risk == "HIGH" and old_risk != "HIGH":
                new_high_risk_families.append(fingerprint)
            
            # Determine severity of this change
            severity = _classify_risk_change_severity(old_risk, new_risk)
            if _severity_order(severity) > _severity_order(max_severity):
                max_severity = severity
    
    # Check for new families that are HIGH risk
    new_fingerprints = set(new_families.keys()) - set(old_families.keys())
    for fingerprint in new_fingerprints:
        if new_families[fingerprint].get("risk_level") == "HIGH":
            new_high_risk_families.append(fingerprint)
    
    # Build neutral notes
    neutral_notes = _build_drift_governor_notes(
        max_severity, len(families_with_changed_risk), len(new_high_risk_families)
    )
    
    return {
        "drift_severity": max_severity,
        "families_with_changed_risk": families_with_changed_risk,
        "new_high_risk_families": sorted(new_high_risk_families),
        "neutral_notes": neutral_notes,
    }


def _classify_risk_change_severity(old_risk: str, new_risk: str) -> str:
    """
    Classify the severity of a risk level change.
    
    Returns:
        "NONE": No change
        "MINOR": Adjacent band change (LOW↔MEDIUM, MEDIUM↔HIGH)
        "MAJOR": Non-adjacent band change (LOW↔HIGH)
    """
    if old_risk == new_risk:
        return "NONE"
    
    risk_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    old_order = risk_order.get(old_risk, 0)
    new_order = risk_order.get(new_risk, 0)
    
    delta = abs(new_order - old_order)
    
    if delta == 1:
        return "MINOR"
    elif delta == 2:
        return "MAJOR"
    else:
        return "NONE"  # Should not happen with valid risk levels


def _severity_order(severity: str) -> int:
    """Get ordering for severity levels."""
    return {"NONE": 0, "MINOR": 1, "MAJOR": 2}.get(severity, 0)


def _build_drift_governor_notes(
    severity: str,
    changed_count: int,
    new_high_count: int,
) -> List[str]:
    """
    Build neutral notes about drift.
    
    Uses only descriptive language, no value judgments.
    """
    notes = []
    
    if severity == "NONE":
        notes.append("No risk level changes detected.")
    elif severity == "MINOR":
        notes.append("Minor risk level changes detected (adjacent bands).")
    elif severity == "MAJOR":
        notes.append("Major risk level changes detected (non-adjacent bands).")
    
    if changed_count > 0:
        notes.append(f"{changed_count} {'family' if changed_count == 1 else 'families'} with changed risk levels.")
    
    if new_high_count > 0:
        notes.append(f"{new_high_count} {'family' if new_high_count == 1 else 'families'} became HIGH risk.")
    
    return notes


def build_decoy_uplift_prescreen(
    drift_governor: Dict[str, Any],
    slice_view: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build uplift pre-screen filter based on drift and slice status.
    
    Provides advisory guidance on whether to proceed with uplift runs.
    Uses neutral, descriptive language only.
    
    Status logic:
    - BLOCK: Major drift AND slice HOT
    - ATTENTION: Moderate drift OR slice ATTENTION (or minor drift + slice HOT)
    - OK: Otherwise
    
    This is advisory only - does not enforce blocking.
    
    Args:
        drift_governor: Output from build_decoy_family_drift_governor
        slice_view: Output from build_slice_confusability_view
        
    Returns:
        {
            "status": "OK" | "ATTENTION" | "BLOCK",
            "advisory_ok": bool,  # True if OK, False if BLOCK/ATTENTION
            "drift_severity": str,
            "slice_status": str,
            "advisory_notes": List[str],  # Neutral descriptive notes
        }
    """
    drift_severity = drift_governor.get("drift_severity", "NONE")
    slice_status = slice_view.get("slice_confusability_status", "OK")
    
    # Determine overall status
    if drift_severity == "MAJOR" and slice_status == "HOT":
        overall_status = "BLOCK"
        advisory_ok = False
    elif drift_severity == "MAJOR" or slice_status == "HOT":
        overall_status = "ATTENTION"
        advisory_ok = True  # Can proceed but review recommended
    elif drift_severity == "MINOR" and slice_status == "ATTENTION":
        overall_status = "ATTENTION"
        advisory_ok = True
    elif drift_severity == "MINOR" or slice_status == "ATTENTION":
        overall_status = "ATTENTION"
        advisory_ok = True
    else:
        overall_status = "OK"
        advisory_ok = True
    
    # Build advisory notes
    advisory_notes = _build_prescreen_advisory_notes(
        overall_status, drift_severity, slice_status
    )
    
    return {
        "status": overall_status,
        "advisory_ok": advisory_ok,
        "drift_severity": drift_severity,
        "slice_status": slice_status,
        "advisory_notes": advisory_notes,
    }


def _build_prescreen_advisory_notes(
    status: str,
    drift_severity: str,
    slice_status: str,
) -> List[str]:
    """
    Build neutral advisory notes for pre-screen.
    
    Uses only descriptive language, no prescriptive verbs.
    """
    notes = []
    
    notes.append(f"Pre-screen status: {status}.")
    
    if drift_severity != "NONE":
        notes.append(f"Drift severity: {drift_severity}.")
    
    if slice_status != "OK":
        notes.append(f"Slice confusability status: {slice_status}.")
    
    if status == "BLOCK":
        notes.append("Combination of major drift and HOT slice status indicates review recommended.")
    elif status == "ATTENTION":
        notes.append("Moderate drift or slice concerns present; review recommended before proceeding.")
    else:
        notes.append("No blocking conditions detected.")
    
    return notes


# =============================================================================
# PHASE V: DECOY-TOPOLOGY COHERENCE GRID
# =============================================================================

# Coherence band thresholds
COHERENCE_BAND_THRESHOLDS = {
    "COHERENT": 0.75,
    "PARTIAL": 0.45,
    # MISALIGNED is < 0.45
}

# Coherence band ordering
COHERENCE_BAND_ORDER = {"MISALIGNED": 0, "PARTIAL": 1, "COHERENT": 2}


def _normalize_confusability_contribution(
    drift_severity: str,
    slice_status: str,
) -> float:
    """
    Normalize confusability contribution to coherence score [0, 1].
    
    Higher score = more coherent (less drift, better status).
    """
    # Drift severity contribution (0.0 = MAJOR, 0.5 = MINOR, 1.0 = NONE)
    drift_scores = {"NONE": 1.0, "MINOR": 0.5, "MAJOR": 0.0}
    drift_score = drift_scores.get(drift_severity, 0.5)
    
    # Slice status contribution (0.0 = HOT, 0.5 = ATTENTION, 1.0 = OK)
    status_scores = {"OK": 1.0, "ATTENTION": 0.5, "HOT": 0.0}
    status_score = status_scores.get(slice_status, 0.5)
    
    # Average the two contributions
    return (drift_score + status_score) / 2.0


def _normalize_topology_contribution(
    topology_status: Optional[str],
) -> float:
    """
    Normalize topology status contribution to coherence score [0, 1].
    
    Assumes topology_status values: "STABLE", "DRIFTING", "UNSTABLE", None
    """
    if topology_status is None:
        return 0.5  # Neutral if unknown
    
    topology_scores = {
        "STABLE": 1.0,
        "DRIFTING": 0.5,
        "UNSTABLE": 0.0,
    }
    return topology_scores.get(topology_status, 0.5)


def _normalize_semantic_contribution(
    semantic_alignment: Optional[float],
) -> float:
    """
    Normalize semantic alignment contribution to coherence score [0, 1].
    
    If semantic_alignment is provided as float [0, 1], use directly.
    If None, return neutral 0.5.
    """
    if semantic_alignment is None:
        return 0.5
    
    # Clamp to [0, 1] range
    return max(0.0, min(1.0, float(semantic_alignment)))


def build_confusability_topology_coherence_map(
    slice_drift_governors: Dict[str, Dict[str, Any]],
    slice_views: Dict[str, Dict[str, Any]],
    topology_statuses: Optional[Dict[str, str]] = None,
    semantic_alignments: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Build confusability-topology coherence map.
    
    Integrates:
    - Decoy family drift (from drift governors)
    - Slice topology status
    - Semantic alignment signals
    
    Computes coherence scores per slice and global index.
    
    Args:
        slice_drift_governors: Dict mapping slice_name -> drift_governor output
        slice_views: Dict mapping slice_name -> slice_confusability_view
        topology_statuses: Optional dict mapping slice_name -> topology status
        semantic_alignments: Optional dict mapping slice_name -> alignment score [0, 1]
        
    Returns:
        {
            "slice_coherence_scores": Dict[str, float],
            "global_coherence_index": float,
            "coherence_band": "COHERENT" | "PARTIAL" | "MISALIGNED",
            "root_incoherence_causes": List[str],
        }
    """
    if topology_statuses is None:
        topology_statuses = {}
    if semantic_alignments is None:
        semantic_alignments = {}
    
    slice_coherence_scores = {}
    all_slices = set(slice_drift_governors.keys()) | set(slice_views.keys())
    
    # Compute coherence score for each slice
    for slice_name in sorted(all_slices):
        drift_governor = slice_drift_governors.get(slice_name, {})
        slice_view = slice_views.get(slice_name, {})
        topology_status = topology_statuses.get(slice_name)
        semantic_alignment = semantic_alignments.get(slice_name)
        
        # Get contributions
        drift_severity = drift_governor.get("drift_severity", "NONE")
        slice_status = slice_view.get("slice_confusability_status", "OK")
        
        confusability_contrib = _normalize_confusability_contribution(
            drift_severity, slice_status
        )
        topology_contrib = _normalize_topology_contribution(topology_status)
        semantic_contrib = _normalize_semantic_contribution(semantic_alignment)
        
        # Average normalized contributions
        coherence_score = (
            confusability_contrib + topology_contrib + semantic_contrib
        ) / 3.0
        
        slice_coherence_scores[slice_name] = round(coherence_score, 6)
    
    # Compute global coherence index (average of slice scores)
    if slice_coherence_scores:
        global_coherence_index = sum(slice_coherence_scores.values()) / len(slice_coherence_scores)
        global_coherence_index = round(global_coherence_index, 6)
    else:
        global_coherence_index = 0.5  # Neutral if no slices
    
    # Determine coherence band
    coherence_band = _determine_coherence_band(global_coherence_index)
    
    # Identify root incoherence causes
    root_causes = _identify_root_incoherence_causes(
        slice_coherence_scores, slice_drift_governors, slice_views
    )
    
    return {
        "slice_coherence_scores": slice_coherence_scores,
        "global_coherence_index": global_coherence_index,
        "coherence_band": coherence_band,
        "root_incoherence_causes": root_causes,
    }


def _determine_coherence_band(coherence_index: float) -> str:
    """
    Determine coherence band from global index.
    
    Thresholds:
    - > 0.75: COHERENT
    - 0.45-0.75: PARTIAL
    - < 0.45: MISALIGNED
    """
    if coherence_index > COHERENCE_BAND_THRESHOLDS["COHERENT"]:
        return "COHERENT"
    elif coherence_index >= COHERENCE_BAND_THRESHOLDS["PARTIAL"]:
        return "PARTIAL"
    else:
        return "MISALIGNED"


def _identify_root_incoherence_causes(
    slice_coherence_scores: Dict[str, float],
    slice_drift_governors: Dict[str, Dict[str, Any]],
    slice_views: Dict[str, Dict[str, Any]],
) -> List[str]:
    """
    Identify root causes of incoherence.
    
    Returns neutral, descriptive notes about what drives low coherence.
    """
    causes = []
    threshold = COHERENCE_BAND_THRESHOLDS["PARTIAL"]
    
    # Find slices with low coherence
    low_coherence_slices = [
        name for name, score in slice_coherence_scores.items()
        if score < threshold
    ]
    
    if low_coherence_slices:
        causes.append(
            f"{len(low_coherence_slices)} slice{'s' if len(low_coherence_slices) != 1 else ''} "
            f"with coherence below {threshold} threshold."
        )
    
    # Check for major drift
    major_drift_slices = []
    for slice_name, governor in slice_drift_governors.items():
        if governor.get("drift_severity") == "MAJOR":
            major_drift_slices.append(slice_name)
    
    if major_drift_slices:
        causes.append(
            f"{len(major_drift_slices)} slice{'s' if len(major_drift_slices) != 1 else ''} "
            f"with MAJOR drift severity."
        )
    
    # Check for HOT slices
    hot_slices = []
    for slice_name, view in slice_views.items():
        if view.get("slice_confusability_status") == "HOT":
            hot_slices.append(slice_name)
    
    if hot_slices:
        causes.append(
            f"{len(hot_slices)} slice{'s' if len(hot_slices) != 1 else ''} "
            f"at HOT confusability status."
        )
    
    if not causes:
        causes.append("No significant incoherence drivers identified.")
    
    return causes


def build_confusability_drift_horizon_predictor(
    drift_history: List[Dict[str, Any]],
    threshold: float = 0.45,
) -> Dict[str, Any]:
    """
    Predict when misalignment would cross a threshold using drift history.
    
    Uses 5-10 previous decoy drift snapshots to estimate trajectory.
    
    Args:
        drift_history: List of drift governor outputs (most recent last)
        threshold: Coherence threshold to predict crossing (default 0.45)
        
    Returns:
        {
            "horizon_estimate": Optional[int],  # Steps until threshold crossing, or None
            "confidence": float,  # [0, 1] confidence in estimate
            "trajectory": "IMPROVING" | "STABLE" | "DEGRADING",
            "current_coherence": float,
            "prediction_notes": List[str],
        }
    """
    if len(drift_history) < 2:
        return {
            "horizon_estimate": None,
            "confidence": 0.0,
            "trajectory": "STABLE",
            "current_coherence": 0.5,
            "prediction_notes": ["Insufficient history for prediction."],
        }
    
    # Extract coherence trend from history
    # For each snapshot, estimate coherence from drift severity
    coherence_estimates = []
    for governor in drift_history:
        severity = governor.get("drift_severity", "NONE")
        # Map severity to coherence estimate (NONE=1.0, MINOR=0.5, MAJOR=0.0)
        severity_scores = {"NONE": 1.0, "MINOR": 0.5, "MAJOR": 0.0}
        coherence_estimates.append(severity_scores.get(severity, 0.5))
    
    current_coherence = coherence_estimates[-1]
    
    # Compute trend (simple linear regression on last N points)
    n_points = min(len(coherence_estimates), 10)
    recent_estimates = coherence_estimates[-n_points:]
    
    if len(recent_estimates) < 2:
        return {
            "horizon_estimate": None,
            "confidence": 0.0,
            "trajectory": "STABLE",
            "current_coherence": current_coherence,
            "prediction_notes": ["Insufficient data points for trend analysis."],
        }
    
    # Simple linear trend: y = mx + b
    x_values = list(range(len(recent_estimates)))
    y_values = recent_estimates
    
    # Compute slope (m)
    n = len(x_values)
    x_mean = sum(x_values) / n
    y_mean = sum(y_values) / n
    
    numerator = sum((x_values[i] - x_mean) * (y_values[i] - y_mean) for i in range(n))
    denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
    
    if abs(denominator) < 1e-10:
        slope = 0.0
    else:
        slope = numerator / denominator
    
    # Determine trajectory
    if slope > 0.01:
        trajectory = "IMPROVING"
    elif slope < -0.01:
        trajectory = "DEGRADING"
    else:
        trajectory = "STABLE"
    
    # Predict horizon
    horizon_estimate = None
    confidence = 0.0
    
    if trajectory == "DEGRADING" and current_coherence > threshold:
        # Estimate steps until crossing threshold
        if abs(slope) > 1e-10:
            steps_to_threshold = (current_coherence - threshold) / abs(slope)
            horizon_estimate = max(1, int(round(steps_to_threshold)))
            
            # Confidence based on data quality and trend strength
            # More points and stronger trend = higher confidence
            trend_strength = abs(slope) * n_points
            confidence = min(1.0, 0.3 + (trend_strength * 0.7))
        else:
            confidence = 0.1
    elif trajectory == "IMPROVING" and current_coherence < threshold:
        # Moving away from threshold (improving)
        if abs(slope) > 1e-10:
            steps_to_threshold = (threshold - current_coherence) / abs(slope)
            horizon_estimate = max(1, int(round(steps_to_threshold)))
            confidence = min(1.0, 0.3 + (abs(slope) * n_points * 0.7))
        else:
            confidence = 0.1
    else:
        # Already past threshold or stable
        confidence = 0.5 if trajectory == "STABLE" else 0.3
    
    # Build prediction notes
    prediction_notes = _build_horizon_prediction_notes(
        horizon_estimate, confidence, trajectory, current_coherence, threshold
    )
    
    return {
        "horizon_estimate": horizon_estimate,
        "confidence": round(confidence, 6),
        "trajectory": trajectory,
        "current_coherence": round(current_coherence, 6),
        "prediction_notes": prediction_notes,
    }


def _build_horizon_prediction_notes(
    horizon_estimate: Optional[int],
    confidence: float,
    trajectory: str,
    current_coherence: float,
    threshold: float,
) -> List[str]:
    """
    Build neutral prediction notes.
    
    Uses only descriptive language, no prescriptive verbs.
    """
    notes = []
    
    notes.append(f"Current coherence: {current_coherence:.3f}.")
    notes.append(f"Trajectory: {trajectory}.")
    
    if horizon_estimate is not None:
        notes.append(
            f"Estimated {horizon_estimate} step{'s' if horizon_estimate != 1 else ''} "
            f"until threshold ({threshold}) crossing."
        )
    else:
        notes.append("No threshold crossing predicted in near term.")
    
    notes.append(f"Confidence: {confidence:.2f}.")
    
    return notes


def build_global_coherence_console_tile(
    coherence_map: Dict[str, Any],
    drift_history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Build global console tile for coherence monitoring.
    
    Provides high-level summary of coherence status across all slices.
    
    Args:
        coherence_map: Output from build_confusability_topology_coherence_map
        drift_history: Optional history for horizon prediction
        
    Returns:
        {
            "status_light": "GREEN" | "YELLOW" | "RED",
            "coherence_band": "COHERENT" | "PARTIAL" | "MISALIGNED",
            "slices_at_risk": List[str],
            "dominant_coherence_drivers": List[str],
            "headline": str,
        }
    """
    coherence_band = coherence_map.get("coherence_band", "PARTIAL")
    global_index = coherence_map.get("global_coherence_index", 0.5)
    slice_scores = coherence_map.get("slice_coherence_scores", {})
    root_causes = coherence_map.get("root_incoherence_causes", [])
    
    # Determine status light
    if coherence_band == "COHERENT":
        status_light = "GREEN"
    elif coherence_band == "PARTIAL":
        status_light = "YELLOW"
    else:  # MISALIGNED
        status_light = "RED"
    
    # Identify slices at risk (low coherence)
    threshold = COHERENCE_BAND_THRESHOLDS["PARTIAL"]
    slices_at_risk = [
        slice_name for slice_name, score in slice_scores.items()
        if score < threshold
    ]
    slices_at_risk = sorted(slices_at_risk)
    
    # Extract dominant drivers from root causes
    dominant_drivers = root_causes[:3]  # Top 3 causes
    
    # Build headline
    headline = _build_coherence_console_headline(
        status_light, coherence_band, global_index, len(slices_at_risk)
    )
    
    return {
        "status_light": status_light,
        "coherence_band": coherence_band,
        "slices_at_risk": slices_at_risk,
        "dominant_coherence_drivers": dominant_drivers,
        "headline": headline,
    }


def _build_coherence_console_headline(
    status_light: str,
    coherence_band: str,
    global_index: float,
    at_risk_count: int,
) -> str:
    """
    Build neutral headline for coherence console.
    
    Uses only descriptive language, no value judgments.
    """
    parts = []
    
    parts.append(f"Coherence status: {status_light} ({coherence_band}).")
    parts.append(f"Global coherence index: {global_index:.3f}.")
    
    if at_risk_count > 0:
        parts.append(
            f"{at_risk_count} slice{'s' if at_risk_count != 1 else ''} "
            f"below coherence threshold."
        )
    else:
        parts.append("All slices above coherence threshold.")
    
    return " ".join(parts)

