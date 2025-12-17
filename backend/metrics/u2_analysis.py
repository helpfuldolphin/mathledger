"""
U2 Evidence Analysis Module (D3 Phase VI).

This module provides evidence quality analysis, phase-portrait visualization,
envelope forecasting, and director panel integration for the Bootstrap Evidence Steward.

GOVERNANCE CONTRACT:
- All functions are read-only and side-effect free
- No uplift interpretation or governance decisions
- All outputs are deterministic and JSON-serializable
- No forbidden words in neutral explanations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


# =============================================================================
# FORBIDDEN WORDS (Governance Contract)
# =============================================================================

GOVERNANCE_SUMMARY_FORBIDDEN_WORDS = [
    "significant", "insignificant", "significant", "p-value", "pvalue",
    "reject", "accept", "hypothesis", "null", "alternative",
    "uplift", "improvement", "degradation", "better", "worse",
    "good", "bad", "should", "must", "need", "required",
    "fail", "success", "correct", "incorrect", "right", "wrong",
]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PairedDeltaResult:
    """
    Result of paired bootstrap delta analysis.
    
    Attributes:
        delta: Mean difference (rfl - baseline)
        ci_lower: Lower bound of confidence interval
        ci_upper: Upper bound of confidence interval
        n_baseline: Sample size for baseline
        n_rfl: Sample size for RFL
        metric_path: Dot-notation path to metric
        seed: Random seed used
        n_bootstrap: Number of bootstrap replicates
        method: Bootstrap method ("BCa" or "percentile")
        analysis_id: SHA-256 hash of analysis parameters
    """
    delta: float
    ci_lower: float
    ci_upper: float
    n_baseline: int
    n_rfl: int
    metric_path: str
    seed: int
    n_bootstrap: int
    method: str
    analysis_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with stable key ordering."""
        return {
            "delta": float(self.delta),
            "ci_lower": float(self.ci_lower),
            "ci_upper": float(self.ci_upper),
            "n_baseline": self.n_baseline,
            "n_rfl": self.n_rfl,
            "metric_path": self.metric_path,
            "seed": self.seed,
            "n_bootstrap": self.n_bootstrap,
            "method": self.method,
            "analysis_id": self.analysis_id,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        import json
        return json.dumps(self.to_dict(), sort_keys=True)


# =============================================================================
# EVIDENCE PACK & QUALITY SNAPSHOT
# =============================================================================

def build_evidence_pack(results: List[PairedDeltaResult]) -> Dict[str, Any]:
    """
    Build evidence pack from multiple PairedDeltaResult objects.
    
    Args:
        results: List of PairedDeltaResult objects
        
    Returns:
        Evidence pack dictionary with schema_version and results
    """
    return {
        "schema_version": "1.0.0",
        "results": [r.to_dict() for r in results],
        "analysis_count": len(results),
    }


def build_evidence_quality_snapshot(pack: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build evidence quality snapshot from evidence pack.
    
    Args:
        pack: Evidence pack from build_evidence_pack
        
    Returns:
        Quality snapshot with methods_by_metric, sample_size_bounds, etc.
    """
    results = pack.get("results", [])
    
    # Group by metric path
    methods_by_metric: Dict[str, List[str]] = {}
    sample_sizes: List[int] = []
    
    for result in results:
        metric_path = result.get("metric_path", "unknown")
        method = result.get("method", "unknown")
        
        if metric_path not in methods_by_metric:
            methods_by_metric[metric_path] = []
        if method not in methods_by_metric[metric_path]:
            methods_by_metric[metric_path].append(method)
        
        sample_sizes.append(result.get("n_baseline", 0))
        sample_sizes.append(result.get("n_rfl", 0))
    
    # Find metrics with multiple methods
    metrics_with_multiple_methods = [
        metric for metric, methods in methods_by_metric.items()
        if len(methods) > 1
    ]
    
    return {
        "schema_version": "1.0.0",
        "methods_by_metric": methods_by_metric,
        "sample_size_bounds": {
            "min": min(sample_sizes) if sample_sizes else 0,
            "max": max(sample_sizes) if sample_sizes else 0,
        },
        "metrics_with_multiple_methods": sorted(metrics_with_multiple_methods),
        "analysis_count": len(results),
    }


def evaluate_evidence_readiness(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate evidence readiness for governance review.
    
    Args:
        snapshot: Quality snapshot from build_evidence_quality_snapshot
        
    Returns:
        Readiness evaluation with ready_for_governance_review, status, etc.
    """
    methods_by_metric = snapshot.get("methods_by_metric", {})
    sample_size_bounds = snapshot.get("sample_size_bounds", {})
    min_sample = sample_size_bounds.get("min", 0)
    
    weak_points: List[str] = []
    
    # Check for multiple methods
    metrics_with_multiple = snapshot.get("metrics_with_multiple_methods", [])
    if not metrics_with_multiple:
        weak_points.append("no metrics with multiple methods")
    
    # Check sample sizes
    if min_sample < 50:
        weak_points.append("minimum sample size below 50")
    
    # Determine status
    if len(weak_points) == 0:
        status = "OK"
        ready_for_governance_review = True
    elif len(weak_points) <= 1:
        status = "ATTENTION"
        ready_for_governance_review = True
    else:
        status = "WEAK"
        ready_for_governance_review = False
    
    return {
        "ready_for_governance_review": ready_for_governance_review,
        "weak_points": sorted(weak_points),
        "status": status,
    }


def classify_evidence_quality_level(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classify evidence quality into TIER_1, TIER_2, or TIER_3.
    
    Args:
        snapshot: Quality snapshot from build_evidence_quality_snapshot
        
    Returns:
        Quality tier classification with requirements_met and requirements_missing
    """
    methods_by_metric = snapshot.get("methods_by_metric", {})
    sample_size_bounds = snapshot.get("sample_size_bounds", {})
    metrics_with_multiple = snapshot.get("metrics_with_multiple_methods", [])
    
    metric_count = len(methods_by_metric)
    min_sample = sample_size_bounds.get("min", 0)
    max_sample = sample_size_bounds.get("max", 0)
    
    requirements_met: List[str] = []
    requirements_missing: List[str] = []
    
    # TIER_3 requirements
    if metric_count >= 2:
        requirements_met.append("multiple_metrics")
    else:
        requirements_missing.append("multiple_metrics")
    
    if len(metrics_with_multiple) >= 1:
        requirements_met.append("multiple_methods_per_metric")
    else:
        requirements_missing.append("multiple_methods_per_metric")
    
    if min_sample >= 100 and max_sample >= 200:
        requirements_met.append("strong_sample_sizes")
    else:
        requirements_missing.append("strong_sample_sizes")
    
    # Determine tier
    if len(requirements_missing) == 0:
        quality_tier = "TIER_3"
    elif metric_count >= 2 and min_sample >= 50:
        quality_tier = "TIER_2"
    else:
        quality_tier = "TIER_1"
    
    return {
        "quality_tier": quality_tier,
        "requirements_met": sorted(requirements_met),
        "requirements_missing": sorted(requirements_missing),
    }


def evaluate_evidence_for_promotion(
    quality_snapshot: Dict[str, Any],
    readiness_eval: Dict[str, Any],
    adversarial_health: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate evidence for promotion based on quality, readiness, and adversarial health.
    
    Args:
        quality_snapshot: Quality snapshot from build_evidence_quality_snapshot
        readiness_eval: Readiness evaluation from evaluate_evidence_readiness
        adversarial_health: Adversarial health status from D2
        
    Returns:
        Promotion evaluation with promotion_ok, status, blocking_reasons, notes
    """
    readiness_status = readiness_eval.get("status", "WEAK")
    adversarial_status = adversarial_health.get("status", "UNKNOWN")
    
    blocking_reasons: List[str] = []
    notes: List[str] = []
    
    if readiness_status == "WEAK":
        blocking_reasons.append("evidence readiness status is WEAK")
        notes.append("evidence pack does not meet readiness criteria")
    
    if adversarial_status == "BLOCK":
        blocking_reasons.append("adversarial health status is BLOCK")
        notes.append("adversarial health check indicates blocking condition")
    
    # Determine promotion status
    if len(blocking_reasons) > 0:
        promotion_ok = False
        status = "BLOCK"
    elif readiness_status == "ATTENTION" or adversarial_status == "WARN":
        promotion_ok = True
        status = "WARN"
    else:
        promotion_ok = True
        status = "OK"
    
    return {
        "promotion_ok": promotion_ok,
        "status": status,
        "blocking_reasons": sorted(blocking_reasons),
        "notes": sorted(notes),
        "analysis_count": quality_snapshot.get("analysis_count", 0),
    }


# =============================================================================
# EVIDENCE QUALITY TIMELINE
# =============================================================================

def build_evidence_quality_timeline(snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build evidence quality timeline from sequence of quality snapshots.
    
    Args:
        snapshots: List of snapshot dicts with run_id, quality_tier, analysis_count
        
    Returns:
        Timeline dictionary with snapshots and quality_trend
    """
    if not snapshots:
        return {
            "schema_version": "1.0.0",
            "snapshots": [],
            "quality_trend": "STABLE",
        }
    
    # Extract tier values for trend calculation
    tier_order = {"TIER_1": 1, "TIER_2": 2, "TIER_3": 3}
    tier_values = [
        tier_order.get(s.get("quality_tier", "TIER_1"), 1)
        for s in snapshots
    ]
    
    # Calculate trend
    if len(tier_values) < 2:
        quality_trend = "STABLE"
    else:
        diffs = np.diff(tier_values)
        improving_count = np.sum(diffs > 0)
        degrading_count = np.sum(diffs < 0)
        total_transitions = improving_count + degrading_count
        
        if total_transitions == 0:
            quality_trend = "STABLE"
        elif improving_count > degrading_count:
            quality_trend = "IMPROVING"
        elif degrading_count > improving_count:
            quality_trend = "DEGRADING"
        else:
            quality_trend = "OSCILLATING"
    
    return {
        "schema_version": "1.0.0",
        "snapshots": snapshots,
        "quality_trend": quality_trend,
    }


# =============================================================================
# PHASE PORTRAIT (PHASE VI)
# =============================================================================

def build_evidence_phase_portrait(quality_timeline: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build evidence phase portrait from quality timeline.
    
    Creates geometric phase-space representation of evidence quality evolution.
    
    Args:
        quality_timeline: Timeline from build_evidence_quality_timeline
        
    Returns:
        Phase portrait with phase_points, trajectory_class, neutral_notes
    """
    snapshots = quality_timeline.get("snapshots", [])
    
    if not snapshots:
        return {
            "phase_points": [],
            "trajectory_class": "STABLE",
            "neutral_notes": ["no timeline data available"],
        }
    
    # Map tiers to numeric values
    tier_order = {"TIER_1": 1, "TIER_2": 2, "TIER_3": 3}
    
    # Build phase points: [run_index, tier_value]
    phase_points: List[List[int]] = []
    tier_values: List[int] = []
    
    for idx, snapshot in enumerate(snapshots):
        tier = snapshot.get("quality_tier", "TIER_1")
        tier_value = tier_order.get(tier, 1)
        phase_points.append([idx, tier_value])
        tier_values.append(tier_value)
    
    # Determine trajectory class
    if len(tier_values) < 2:
        trajectory_class = "STABLE"
    else:
        diffs = np.diff(tier_values)
        improving_transitions = np.sum(diffs > 0)
        degrading_transitions = np.sum(diffs < 0)
        total_transitions = improving_transitions + degrading_transitions
        
        if total_transitions == 0:
            trajectory_class = "STABLE"
        else:
            # Calculate ratios
            p_improve = improving_transitions / total_transitions if total_transitions > 0 else 0.0
            p_degrade = degrading_transitions / total_transitions if total_transitions > 0 else 0.0
            
            # Check for oscillation: both types present and neither dominates strongly
            if p_improve >= 0.3 and p_degrade >= 0.3:
                # Check if sequence shows alternation
                has_alternation = False
                for i in range(len(diffs) - 1):
                    if (diffs[i] > 0 and diffs[i + 1] < 0) or (diffs[i] < 0 and diffs[i + 1] > 0):
                        has_alternation = True
                        break
                
                if has_alternation:
                    trajectory_class = "OSCILLATING"
                elif p_degrade > 0.6:
                    # Degrading strongly dominates
                    trajectory_class = "DEGRADING"
                elif p_improve > 0.6:
                    # Improving strongly dominates
                    trajectory_class = "IMPROVING"
                else:
                    # Balanced but no clear alternation
                    trajectory_class = "OSCILLATING"
            elif p_degrade > 0.6:
                # Degrading strongly dominates
                trajectory_class = "DEGRADING"
            elif p_improve > 0.6:
                # Improving strongly dominates
                trajectory_class = "IMPROVING"
            elif improving_transitions > degrading_transitions:
                trajectory_class = "IMPROVING"
            elif degrading_transitions > improving_transitions:
                trajectory_class = "DEGRADING"
            else:
                trajectory_class = "STABLE"
    
    # Build neutral notes
    neutral_notes: List[str] = []
    if trajectory_class == "IMPROVING":
        neutral_notes.append("trajectory shows consistent quality increases")
    elif trajectory_class == "DEGRADING":
        neutral_notes.append("trajectory shows consistent quality decreases")
    elif trajectory_class == "OSCILLATING":
        neutral_notes.append("alternating improve/degrade cycles detected")
    else:
        neutral_notes.append("trajectory shows stable quality levels")
    
    neutral_notes = sorted(neutral_notes)
    
    return {
        "phase_points": phase_points,
        "trajectory_class": trajectory_class,
        "neutral_notes": neutral_notes,
    }


# =============================================================================
# REGRESSION WATCHDOG
# =============================================================================

def evaluate_evidence_quality_regression(timeline: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate evidence quality regression from timeline.
    
    Args:
        timeline: Timeline from build_evidence_quality_timeline
        
    Returns:
        Regression evaluation with status (OK, ATTENTION, BLOCK), regression_detected, notes
    """
    snapshots = timeline.get("snapshots", [])
    
    if len(snapshots) < 2:
        return {
            "status": "OK",
            "regression_detected": False,
            "notes": ["insufficient data for regression analysis"],
        }
    
    tier_order = {"TIER_1": 1, "TIER_2": 2, "TIER_3": 3}
    tier_values = [
        tier_order.get(s.get("quality_tier", "TIER_1"), 1)
        for s in snapshots
    ]
    
    # Check recent runs (last 2)
    recent_below_tier3 = all(tv < 3 for tv in tier_values[-2:])
    
    # Count transitions
    diffs = np.diff(tier_values)
    degrading_transitions = np.sum(diffs < 0)
    improving_transitions = np.sum(diffs > 0)
    
    # Count tier drops
    tier_3_to_lower = sum(
        1 for i in range(len(tier_values) - 1)
        if tier_values[i] == 3 and tier_values[i + 1] < 3
    )
    
    # Check for oscillating pattern (alternating up/down, but only adjacent-tier oscillations)
    is_oscillating = False
    if len(diffs) >= 2:
        # Check if all transitions are between adjacent tiers (|diff| == 1)
        all_adjacent = all(abs(d) <= 1 for d in diffs)
        if all_adjacent:
            alternations = sum(
                1 for i in range(len(diffs) - 1)
                if (diffs[i] > 0 and diffs[i + 1] < 0) or (diffs[i] < 0 and diffs[i + 1] > 0)
            )
            if alternations >= 1 and improving_transitions >= 1 and degrading_transitions >= 1:
                is_oscillating = True
    
    notes: List[str] = []
    regression_detected = False
    
    # Priority 1: Recent runs consistently below TIER_3 (not oscillating)
    if recent_below_tier3 and len(snapshots) >= 3 and not is_oscillating:
        status = "BLOCK"
        regression_detected = True
        notes.append("recent runs consistently below TIER_3")
    # Priority 2: Multiple tier drops from TIER_3 (BLOCK unless oscillating)
    elif tier_3_to_lower >= 2 and not is_oscillating:
        status = "BLOCK"
        regression_detected = True
        notes.append("multiple transitions from TIER_3 to lower tiers")
    # Priority 3: Consistent degrading pattern
    elif degrading_transitions >= 3 and improving_transitions == 0:
        status = "BLOCK"
        regression_detected = True
        notes.append("consistent degrading pattern with no recovery")
    # Priority 4: Oscillating pattern (treat as ATTENTION, not BLOCK, even with tier drops)
    elif is_oscillating:
        status = "ATTENTION"
        regression_detected = True
        notes.append("oscillating pattern detected")
    # Priority 5: Degrading with some recovery
    elif degrading_transitions >= 2 and improving_transitions >= 1:
        status = "ATTENTION"
        regression_detected = True
        notes.append("degrading pattern with intermittent recovery")
    # Priority 6: Single tier drop
    elif tier_3_to_lower >= 1:
        status = "ATTENTION"
        regression_detected = True
        notes.append("single transition from TIER_3 to lower tier")
    else:
        status = "OK"
        notes.append("no significant regression detected")
    
    return {
        "status": status,
        "regression_detected": regression_detected,
        "notes": sorted(notes),
    }


# =============================================================================
# ENVELOPE FORECAST (PHASE VI)
# =============================================================================

def forecast_evidence_envelope(
    phase_portrait: Dict[str, Any],
    regression_watchdog: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Forecast evidence quality envelope based on phase portrait and regression analysis.
    
    Args:
        phase_portrait: Phase portrait from build_evidence_phase_portrait
        regression_watchdog: Regression evaluation from evaluate_evidence_quality_regression
        
    Returns:
        Forecast with predicted_band, confidence, cycles_until_risk, neutral_explanation
    """
    trajectory = phase_portrait.get("trajectory_class", "STABLE")
    regression_status = regression_watchdog.get("status", "OK")
    phase_points = phase_portrait.get("phase_points", [])
    
    if not phase_points:
        return {
            "predicted_band": "LOW",
            "confidence": 0.0,
            "cycles_until_risk": 0,
            "neutral_explanation": ["insufficient data for envelope forecast"],
        }
    
    # Calculate average recent tier
    recent_points = phase_points[-3:] if len(phase_points) >= 3 else phase_points
    avg_recent_tier = np.mean([p[1] for p in recent_points]) if recent_points else 2.0
    
    neutral_explanation: List[str] = []
    
    # Decision tree for predicted band
    # BLOCK regression always results in LOW band
    if regression_status == "BLOCK":
        predicted_band = "LOW"
        confidence = 0.8
        cycles_until_risk = 0
        neutral_explanation.append("regression watchdog indicates blocking condition")
    # Prioritize trajectory class for oscillating patterns (when not BLOCK)
    elif trajectory == "OSCILLATING":
        predicted_band = "MEDIUM"
        confidence = 0.6
        cycles_until_risk = 2
        neutral_explanation.append("evidence quality shows oscillating pattern")
    elif regression_status == "ATTENTION":
        if trajectory == "DEGRADING":
            predicted_band = "LOW"
            confidence = 0.7
            cycles_until_risk = 1
            neutral_explanation.append("evidence quality shows attention points with degrading trajectory")
        else:
            predicted_band = "MEDIUM"
            confidence = 0.65
            cycles_until_risk = 2
            neutral_explanation.append("evidence quality shows attention points")
    else:  # OK status
        if trajectory == "IMPROVING" and avg_recent_tier >= 2.5:
            predicted_band = "HIGH"
            confidence = 0.75
            cycles_until_risk = 5
            neutral_explanation.append("evidence quality is improving with high recent quality")
        elif trajectory == "STABLE" and avg_recent_tier >= 2.5:
            predicted_band = "HIGH"
            confidence = 0.8
            cycles_until_risk = 4
            neutral_explanation.append("evidence quality is stable with high recent quality")
        elif trajectory == "IMPROVING":
            predicted_band = "MEDIUM"
            confidence = 0.7
            cycles_until_risk = 3
            neutral_explanation.append("evidence quality is improving")
        elif trajectory == "STABLE":
            predicted_band = "MEDIUM"
            confidence = 0.75
            cycles_until_risk = 3
            neutral_explanation.append("evidence quality is stable")
        elif trajectory == "OSCILLATING":
            predicted_band = "MEDIUM"
            confidence = 0.6
            cycles_until_risk = 2
            neutral_explanation.append("evidence quality shows oscillating pattern")
        else:  # DEGRADING
            predicted_band = "LOW"
            confidence = 0.5
            cycles_until_risk = 1
            neutral_explanation.append("evidence quality is degrading")
    
    # Adjust confidence based on data points
    if len(phase_points) < 3:
        confidence = max(0.0, confidence - 0.3)
        neutral_explanation.append("limited data points reduce forecast confidence")
    
    # Round confidence to 2 decimal places
    confidence = round(confidence, 2)
    
    return {
        "predicted_band": predicted_band,
        "confidence": confidence,
        "cycles_until_risk": cycles_until_risk,
        "neutral_explanation": sorted(neutral_explanation),
    }


# =============================================================================
# DIRECTOR PANEL V2 (PHASE VI)
# =============================================================================

def build_evidence_director_panel_v2(
    quality_tier_info: Dict[str, Any],
    promotion_eval: Dict[str, Any],
    phase_portrait: Optional[Dict[str, Any]] = None,
    forecast: Optional[Dict[str, Any]] = None,
    regression_watchdog: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build enhanced director panel v2 with phase portrait and forecast integration.
    
    Args:
        quality_tier_info: Quality tier classification from classify_evidence_quality_level
        promotion_eval: Promotion evaluation from evaluate_evidence_for_promotion
        phase_portrait: Optional phase portrait from build_evidence_phase_portrait
        forecast: Optional forecast from forecast_evidence_envelope
        regression_watchdog: Optional regression evaluation from evaluate_evidence_quality_regression
        
    Returns:
        Director panel with status_light, trajectory_class, regression_status, headline, flags
    """
    quality_tier = quality_tier_info.get("quality_tier", "TIER_1")
    promotion_ok = promotion_eval.get("promotion_ok", False)
    promotion_status = promotion_eval.get("status", "OK")
    analysis_count = promotion_eval.get("analysis_count", 0)
    
    # Extract from optional inputs
    trajectory_class = phase_portrait.get("trajectory_class", "STABLE") if phase_portrait else "STABLE"
    regression_status = regression_watchdog.get("status", "OK") if regression_watchdog else "OK"
    regression_detected = regression_watchdog.get("regression_detected", False) if regression_watchdog else False
    
    # Determine status light
    # GREEN: promotion OK, no blocking regression, status OK or WARN (but promotion_ok=True)
    if promotion_ok and not regression_detected and regression_status == "OK":
        status_light = "GREEN"
    # YELLOW: promotion OK but has warnings/attention, or regression attention
    elif promotion_ok and (promotion_status == "WARN" or regression_status == "ATTENTION"):
        status_light = "YELLOW"
    # RED: blocking conditions
    else:
        status_light = "RED"
    
    # Build flags
    flags: List[str] = []
    if regression_detected:
        flags.append("regression_detected")
    if trajectory_class == "OSCILLATING":
        flags.append("oscillating_trajectory")
    if forecast and forecast.get("predicted_band") == "LOW":
        flags.append("low_predicted_band")
    if forecast and forecast.get("cycles_until_risk", 999) <= 1:
        flags.append("imminent_risk")
    
    flags = sorted(flags)
    
    # Build headline
    headline_parts: List[str] = []
    headline_parts.append(f"Evidence pack has {quality_tier} quality")
    if trajectory_class != "STABLE":
        headline_parts.append(f"with {trajectory_class.lower()} trajectory")
    if regression_detected:
        headline_parts.append("and regression detected")
    
    headline = "; ".join(headline_parts) + "."
    
    return {
        "status_light": status_light,
        "quality_tier": quality_tier,
        "trajectory_class": trajectory_class,
        "regression_status": regression_status,
        "analysis_count": analysis_count,
        "evidence_ok": promotion_ok,
        "headline": headline,
        "flags": flags,
    }

