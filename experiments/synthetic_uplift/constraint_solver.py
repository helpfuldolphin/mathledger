#!/usr/bin/env python3
"""
==============================================================================
PHASE II — SYNTHETIC TEST DATA ONLY
==============================================================================

Realism Constraint Solver
--------------------------

This module implements adaptive constraint solving to self-correct synthetic
universes toward realism envelopes.

Features:
    1. Parameter adjustment planning based on envelope violations
    2. Cross-scenario similarity analysis and clustering
    3. Director realism tile for executive summary

The solver is advisory-only and deterministic. It adjusts only synthetic
parameters, never real metrics.

SHADOW-MODE GUARANTEES:
    - No enforcement: All outputs are advisory recommendations
    - No influence on upstream governance: Solver operates in read-only mode
    - Advisory only: No automatic parameter modifications

Must NOT:
    - Produce claims about real uplift
    - Mix synthetic and real data
    - Enforce adjustments (advisory only)

==============================================================================
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


# ==============================================================================
# CONSTANTS
# ==============================================================================

SAFETY_LABEL = "PHASE II — SYNTHETIC TEST DATA ONLY"
SCHEMA_VERSION = "constraint_solver_v1.0"

# Rounding precision for parameter adjustments
PARAMETER_PRECISION = 4

# Source path constants for window schema audit trail
SOURCE_PATH_DIRECT = "DIRECT"
SOURCE_PATH_METRICS_NESTED = "METRICS_NESTED"
SOURCE_PATH_SUMMARY_NESTED = "SUMMARY_NESTED"
SOURCE_PATH_MISSING = "MISSING"

# Valid source paths (must never emit unknown values)
VALID_SOURCE_PATHS = {
    SOURCE_PATH_DIRECT,
    SOURCE_PATH_METRICS_NESTED,
    SOURCE_PATH_SUMMARY_NESTED,
    SOURCE_PATH_MISSING,
}


def _coerce_source_path(source_path: str) -> str:
    """
    Coerce source_path to valid value, falling back to MISSING if unknown.
    
    This ensures that unknown path labels are never emitted. Used for defensive
    programming and audit trail integrity.
    
    Args:
        source_path: Source path string
    
    Returns:
        Valid source path string (guaranteed to be in VALID_SOURCE_PATHS)
    
    Note:
        This function should never receive an unknown source_path in normal operation.
        If it does, it falls back to MISSING for safety.
    """
    if source_path in VALID_SOURCE_PATHS:
        return source_path
    # Unknown source_path detected - assert for debugging, but always fallback to MISSING
    # This should never happen in normal operation, but we handle it defensively
    import warnings
    warnings.warn(
        f"Unknown source_path '{source_path}' detected. Falling back to MISSING. "
        f"Valid paths are: {VALID_SOURCE_PATHS}",
        RuntimeWarning,
        stacklevel=2,
    )
    return SOURCE_PATH_MISSING


# ==============================================================================
# ENVELOPE BOUNDS (Self-contained definition)
# ==============================================================================

@dataclass
class EnvelopeBounds:
    """
    Bounds for the realism envelope.
    
    These are NOT empirical bounds - they are reasonable ranges for
    synthetic data to remain plausible for stress-testing.
    """
    
    # Variance bounds
    max_per_cycle_sigma: float = 0.15
    max_per_item_sigma: float = 0.10
    
    # Correlation bounds
    min_correlation_rho: float = 0.0
    max_correlation_rho: float = 0.9
    
    # Drift amplitude bounds
    max_sinusoidal_amplitude: float = 0.25
    max_linear_slope: float = 0.005  # Per cycle
    max_step_delta: float = 0.40
    min_drift_period: int = 20  # Cycles
    
    # Rare event bounds
    max_rare_event_probability: float = 0.10  # Per cycle
    min_rare_event_duration: int = 1
    max_rare_event_magnitude: float = 0.70
    max_rare_events_per_scenario: int = 5
    
    # Probability bounds
    min_probability: float = 0.05
    max_probability: float = 0.95
    max_probability_spread: float = 0.60  # Max diff between baseline and RFL


DEFAULT_BOUNDS = EnvelopeBounds()


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def round_to_precision(value: float, precision: int = PARAMETER_PRECISION) -> float:
    """Round value to specified decimal precision."""
    return round(value, precision)


def clamp_probability(value: float, bounds: EnvelopeBounds = DEFAULT_BOUNDS) -> float:
    """Clamp probability to valid range [min_prob, max_prob]."""
    return round_to_precision(max(bounds.min_probability, min(bounds.max_probability, value)))


def clamp_correlation(value: float, bounds: EnvelopeBounds = DEFAULT_BOUNDS) -> float:
    """Clamp correlation to valid range [min_rho, max_rho]."""
    return round_to_precision(max(bounds.min_correlation_rho, min(bounds.max_correlation_rho, value)))


# ==============================================================================
# TASK 1: REALISM CONSTRAINT SOLVER
# ==============================================================================

def solve_realism_constraints(
    ratchet: Dict[str, Any],
    calibration_console: Dict[str, Any],
    scenario_configs: Dict[str, Dict[str, Any]],
    envelope_bounds: EnvelopeBounds = DEFAULT_BOUNDS,
) -> Dict[str, Any]:
    """
    Solve realism constraints and produce parameter adjustment plans.
    
    SHADOW-MODE: This function is advisory-only. It does not enforce
    adjustments or modify upstream governance.
    
    Args:
        ratchet: Output from build_synthetic_realism_ratchet
        calibration_console: Output from build_scenario_calibration_console
        scenario_configs: Dict mapping scenario name to full config
        envelope_bounds: Envelope bounds to target
    
    Returns:
        Constraint solution with:
            - parameter_adjustment_plan: Dict of scenario → adjusted parameters
            - expected_effects: Dict of scenario → neutral description
            - confidence: Overall confidence score [0, 1]
    """
    slices_to_recalibrate = calibration_console.get("slices_to_recalibrate", [])
    retention_scores = ratchet.get("scenario_retention_score", {})
    stability_classes = ratchet.get("stability_class", {})
    
    adjustment_plan = {}
    expected_effects = {}
    confidence_scores = []
    
    for scenario_name in slices_to_recalibrate:
        if scenario_name not in scenario_configs:
            continue
        
        config = scenario_configs[scenario_name]
        params = config.get("parameters", {})
        
        # Analyze violations and compute adjustments
        adjustments, effect, confidence = _compute_scenario_adjustments(
            scenario_name=scenario_name,
            params=params,
            retention_score=retention_scores.get(scenario_name, 1.0),
            stability=stability_classes.get(scenario_name, "STABLE"),
            bounds=envelope_bounds,
        )
        
        if adjustments:
            adjustment_plan[scenario_name] = adjustments
            expected_effects[scenario_name] = effect
            confidence_scores.append(confidence)
    
    # Overall confidence is average of individual confidences
    overall_confidence = (
        sum(confidence_scores) / len(confidence_scores)
        if confidence_scores
        else 1.0
    )
    
    return {
        "label": SAFETY_LABEL,
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parameter_adjustment_plan": adjustment_plan,
        "expected_effects": expected_effects,
        "confidence": round_to_precision(overall_confidence),
        "scenarios_adjusted": len(adjustment_plan),
        "adjustment_rationale": _build_adjustment_rationale(adjustment_plan, expected_effects),
    }


def _compute_scenario_adjustments(
    scenario_name: str,
    params: Dict[str, Any],
    retention_score: float,
    stability: str,
    bounds: EnvelopeBounds,
) -> Tuple[Dict[str, Any], str, float]:
    """
    Compute parameter adjustments for a single scenario.
    
    Returns:
        (adjustments_dict, effect_description, confidence)
    """
    adjustments = {}
    effect_parts = []
    confidence = 1.0
    
    # Adjust probabilities if out of range
    probs = params.get("probabilities", {})
    prob_adjustments = _adjust_probabilities(probs, bounds)
    if prob_adjustments:
        adjustments["probabilities"] = prob_adjustments
        effect_parts.append("probability ranges normalized")
        confidence *= 0.9  # High confidence for probability adjustments
    
    # Adjust drift if out of bounds
    drift = params.get("drift", {})
    drift_adjustments = _adjust_drift(drift, bounds)
    if drift_adjustments:
        adjustments["drift"] = drift_adjustments
        effect_parts.append("drift parameters constrained")
        confidence *= 0.8  # Medium confidence for drift adjustments
    
    # Adjust correlation if out of bounds
    corr = params.get("correlation", {})
    corr_adjustments = _adjust_correlation(corr, bounds)
    if corr_adjustments:
        adjustments["correlation"] = corr_adjustments
        effect_parts.append("correlation coefficient bounded")
        confidence *= 0.85  # Medium-high confidence
    
    # Adjust variance if out of bounds
    variance = params.get("variance", {})
    variance_adjustments = _adjust_variance(variance, bounds)
    if variance_adjustments:
        adjustments["variance"] = variance_adjustments
        effect_parts.append("variance parameters reduced")
        confidence *= 0.75  # Lower confidence for variance adjustments
    
    # Adjust rare events if excessive
    rare_events = params.get("rare_events", [])
    rare_adjustments = _adjust_rare_events(rare_events, bounds)
    if rare_adjustments:
        adjustments["rare_events"] = rare_adjustments
        effect_parts.append("rare event parameters moderated")
        confidence *= 0.7  # Lower confidence for rare event adjustments
    
    # Build effect description
    if effect_parts:
        effect = f"Adjustments applied to {', '.join(effect_parts)}."
    else:
        effect = "No adjustments required."
    
    # Reduce confidence for unstable scenarios
    if stability == "SHARP_DRIFT":
        confidence *= 0.6
    elif stability == "SOFT_DRIFT":
        confidence *= 0.8
    
    return adjustments, effect, round_to_precision(confidence)


def _adjust_probabilities(
    probs: Dict[str, Any],
    bounds: EnvelopeBounds,
) -> Dict[str, Any]:
    """Adjust probabilities to be within bounds."""
    adjustments = {}
    
    for mode, class_probs in probs.items():
        if not isinstance(class_probs, dict):
            continue
        
        adjusted = {}
        needs_adjustment = False
        
        for cls, prob in class_probs.items():
            if isinstance(prob, (int, float)):
                # Clamp to [min_prob, max_prob]
                clamped = clamp_probability(prob, bounds)
                adjusted[cls] = clamped
                if clamped != prob:
                    needs_adjustment = True
        
        if needs_adjustment:
            adjustments[mode] = adjusted
    
    return adjustments


def _adjust_drift(
    drift: Dict[str, Any],
    bounds: EnvelopeBounds,
) -> Dict[str, Any]:
    """Adjust drift parameters to be within bounds."""
    adjustments = {}
    mode = drift.get("mode", "none")
    
    if mode in ("cyclical", "sinusoidal"):
        amplitude = drift.get("amplitude", 0.0)
        if abs(amplitude) > bounds.max_sinusoidal_amplitude:
            adjustments["amplitude"] = round_to_precision(
                bounds.max_sinusoidal_amplitude * (1.0 if amplitude > 0 else -1.0)
            )
        
        period = drift.get("period", 100)
        if period < bounds.min_drift_period:
            adjustments["period"] = bounds.min_drift_period
    
    elif mode in ("linear", "monotonic"):
        slope = drift.get("slope", 0.0)
        if abs(slope) > bounds.max_linear_slope:
            adjustments["slope"] = round_to_precision(
                bounds.max_linear_slope * (1.0 if slope > 0 else -1.0)
            )
    
    elif mode in ("shock", "step"):
        delta = drift.get("shock_delta", drift.get("delta", 0.0))
        if abs(delta) > bounds.max_step_delta:
            adjustments["shock_delta"] = round_to_precision(
                bounds.max_step_delta * (1.0 if delta > 0 else -1.0)
            )
    
    return adjustments


def _adjust_correlation(
    corr: Dict[str, Any],
    bounds: EnvelopeBounds,
) -> Dict[str, Any]:
    """Adjust correlation to be within bounds."""
    adjustments = {}
    rho = corr.get("rho", 0.0)
    
    if rho < bounds.min_correlation_rho:
        adjustments["rho"] = bounds.min_correlation_rho
    elif rho > bounds.max_correlation_rho:
        adjustments["rho"] = bounds.max_correlation_rho
    
    return adjustments


def _adjust_variance(
    variance: Dict[str, Any],
    bounds: EnvelopeBounds,
) -> Dict[str, Any]:
    """Adjust variance parameters to be within bounds."""
    adjustments = {}
    
    per_cycle = variance.get("per_cycle_sigma", 0.0)
    if per_cycle > bounds.max_per_cycle_sigma:
        adjustments["per_cycle_sigma"] = round_to_precision(bounds.max_per_cycle_sigma)
    
    per_item = variance.get("per_item_sigma", 0.0)
    if per_item > bounds.max_per_item_sigma:
        adjustments["per_item_sigma"] = round_to_precision(bounds.max_per_item_sigma)
    
    return adjustments


def _adjust_rare_events(
    rare_events: List[Dict[str, Any]],
    bounds: EnvelopeBounds,
) -> List[Dict[str, Any]]:
    """Adjust rare events to be within bounds."""
    if not isinstance(rare_events, list):
        return []
    
    # Check count
    if len(rare_events) > bounds.max_rare_events_per_scenario:
        # Keep only first N events
        rare_events = rare_events[:bounds.max_rare_events_per_scenario]
    
    adjusted = []
    changes_made = False
    
    for event in rare_events:
        adj_event = event.copy()
        
        # Adjust probability
        prob = event.get("trigger_probability", 0.0)
        if prob > bounds.max_rare_event_probability:
            adj_event["trigger_probability"] = round_to_precision(
                bounds.max_rare_event_probability
            )
            changes_made = True
        
        # Adjust magnitude
        magnitude = abs(event.get("magnitude", 0.0))
        if magnitude > bounds.max_rare_event_magnitude:
            sign = 1.0 if event.get("magnitude", 0.0) >= 0 else -1.0
            adj_event["magnitude"] = round_to_precision(
                bounds.max_rare_event_magnitude * sign
            )
            changes_made = True
        
        # Adjust duration
        duration = event.get("duration", 1)
        if duration < bounds.min_rare_event_duration:
            adj_event["duration"] = bounds.min_rare_event_duration
            changes_made = True
        
        adjusted.append(adj_event)
    
    # Only return if changes were made
    if changes_made or len(adjusted) < len(rare_events):
        return adjusted
    
    return []


def _build_adjustment_rationale(
    adjustment_plan: Dict[str, Any],
    expected_effects: Dict[str, Any],
) -> str:
    """Build neutral rationale for adjustments."""
    if not adjustment_plan:
        return "No parameter adjustments required. All scenarios within envelope bounds."
    
    rationale = f"Parameter adjustments computed for {len(adjustment_plan)} scenario(s). "
    rationale += "Adjustments target envelope compliance while preserving scenario characteristics. "
    rationale += "All adjustments are advisory and maintain deterministic behavior."
    
    return rationale


# ==============================================================================
# TASK 2: CROSS-SCENARIO REALISM COUPLING MAP
# ==============================================================================

def build_cross_scenario_coupling_map(
    scenario_configs: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build pairwise similarity map and detect outliers using deterministic clustering.
    
    SHADOW-MODE: This function is read-only and does not modify scenarios.
    
    Args:
        scenario_configs: Dict mapping scenario name to full config
    
    Returns:
        Coupling map with:
            - pairwise_similarity: Dict of (scenario1, scenario2) → similarity score
            - scenario_clusters: Dict of cluster_id → list of scenarios
            - outliers: List of outlier scenario names
            - similarity_matrix: Full N×N similarity matrix
    """
    scenarios = list(scenario_configs.keys())
    n = len(scenarios)
    
    if n == 0:
        return {
            "label": SAFETY_LABEL,
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "pairwise_similarity": {},
            "scenario_clusters": {},
            "outliers": [],
            "similarity_matrix": {},
            "cluster_count": 0,
            "outlier_count": 0,
        }
    
    # Compute pairwise similarities
    pairwise_similarity = {}
    similarity_matrix = {}
    
    for i, s1 in enumerate(scenarios):
        similarity_matrix[s1] = {}
        for j, s2 in enumerate(scenarios):
            if i == j:
                similarity = 1.0
            elif i < j:
                similarity = _compute_scenario_similarity(
                    scenario_configs[s1],
                    scenario_configs[s2],
                )
                pairwise_similarity[(s1, s2)] = similarity
            else:
                # Use symmetric value
                similarity = similarity_matrix[s2][s1]
            
            similarity_matrix[s1][s2] = round_to_precision(similarity)
    
    # Deterministic clustering
    clusters, outliers = _cluster_scenarios(
        scenarios=scenarios,
        similarity_matrix=similarity_matrix,
    )
    
    return {
        "label": SAFETY_LABEL,
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pairwise_similarity": {
            f"{s1}:{s2}": round_to_precision(sim)
            for (s1, s2), sim in pairwise_similarity.items()
        },
        "scenario_clusters": clusters,
        "outliers": sorted(outliers),
        "similarity_matrix": {
            s1: {s2: sim for s2, sim in row.items()}
            for s1, row in similarity_matrix.items()
        },
        "cluster_count": len(clusters),
        "outlier_count": len(outliers),
    }


def _compute_scenario_similarity(
    config1: Dict[str, Any],
    config2: Dict[str, Any],
) -> float:
    """
    Compute similarity score between two scenarios [0, 1].
    
    Based on:
        - Variance profiles
        - Drift modes
        - Correlation structures
        - Probability distributions
    """
    params1 = config1.get("parameters", {})
    params2 = config2.get("parameters", {})
    
    # Variance similarity
    var1 = params1.get("variance", {})
    var2 = params2.get("variance", {})
    var_sim = _variance_similarity(var1, var2)
    
    # Drift similarity
    drift1 = params1.get("drift", {})
    drift2 = params2.get("drift", {})
    drift_sim = _drift_similarity(drift1, drift2)
    
    # Correlation similarity
    corr1 = params1.get("correlation", {})
    corr2 = params2.get("correlation", {})
    corr_sim = _correlation_similarity(corr1, corr2)
    
    # Probability similarity
    probs1 = params1.get("probabilities", {})
    probs2 = params2.get("probabilities", {})
    prob_sim = _probability_similarity(probs1, probs2)
    
    # Weighted average
    similarity = (
        var_sim * 0.2 +
        drift_sim * 0.3 +
        corr_sim * 0.2 +
        prob_sim * 0.3
    )
    
    return round_to_precision(similarity)


def _variance_similarity(var1: Dict[str, Any], var2: Dict[str, Any]) -> float:
    """Compute variance profile similarity."""
    sigma1_cycle = var1.get("per_cycle_sigma", 0.0)
    sigma1_item = var1.get("per_item_sigma", 0.0)
    sigma2_cycle = var2.get("per_cycle_sigma", 0.0)
    sigma2_item = var2.get("per_item_sigma", 0.0)
    
    # Normalize differences (max sigma is ~0.15)
    cycle_diff = abs(sigma1_cycle - sigma2_cycle) / 0.15
    item_diff = abs(sigma1_item - sigma2_item) / 0.10
    
    # Similarity is inverse of normalized difference
    similarity = 1.0 - min(1.0, (cycle_diff + item_diff) / 2.0)
    return round_to_precision(similarity)


def _drift_similarity(drift1: Dict[str, Any], drift2: Dict[str, Any]) -> float:
    """Compute drift mode similarity."""
    mode1 = drift1.get("mode", "none")
    mode2 = drift2.get("mode", "none")
    
    # Mode match
    if mode1 == mode2:
        mode_sim = 1.0
        
        # If same mode, compare parameters
        if mode1 in ("cyclical", "sinusoidal"):
            amp1 = drift1.get("amplitude", 0.0)
            amp2 = drift2.get("amplitude", 0.0)
            period1 = drift1.get("period", 100)
            period2 = drift2.get("period", 100)
            
            amp_diff = abs(amp1 - amp2) / 0.25  # Normalize by max
            period_diff = abs(period1 - period2) / 200.0  # Normalize
            
            param_sim = 1.0 - min(1.0, (amp_diff + period_diff) / 2.0)
            mode_sim = (mode_sim + param_sim) / 2.0
        
        elif mode1 in ("linear", "monotonic"):
            slope1 = drift1.get("slope", 0.0)
            slope2 = drift2.get("slope", 0.0)
            slope_diff = abs(slope1 - slope2) / 0.005  # Normalize by max
            param_sim = 1.0 - min(1.0, slope_diff)
            mode_sim = (mode_sim + param_sim) / 2.0
        
    else:
        # Different modes
        if mode1 == "none" or mode2 == "none":
            mode_sim = 0.3  # Some similarity if one has no drift
        else:
            mode_sim = 0.1  # Low similarity for different drift modes
    
    return round_to_precision(mode_sim)


def _correlation_similarity(corr1: Dict[str, Any], corr2: Dict[str, Any]) -> float:
    """Compute correlation structure similarity."""
    rho1 = corr1.get("rho", 0.0)
    rho2 = corr2.get("rho", 0.0)
    
    # Normalized difference
    rho_diff = abs(rho1 - rho2) / 0.9  # Max rho is 0.9
    similarity = 1.0 - min(1.0, rho_diff)
    
    return round_to_precision(similarity)


def _probability_similarity(probs1: Dict[str, Any], probs2: Dict[str, Any]) -> float:
    """Compute probability distribution similarity."""
    # Extract all probabilities
    all_probs1 = []
    all_probs2 = []
    
    for mode in ("baseline", "rfl"):
        if mode in probs1 and isinstance(probs1[mode], dict):
            all_probs1.extend(probs1[mode].values())
        if mode in probs2 and isinstance(probs2[mode], dict):
            all_probs2.extend(probs2[mode].values())
    
    if not all_probs1 or not all_probs2:
        return 0.5  # Default similarity
    
    # Compare mean and spread
    mean1 = sum(all_probs1) / len(all_probs1)
    mean2 = sum(all_probs2) / len(all_probs2)
    
    spread1 = max(all_probs1) - min(all_probs1) if all_probs1 else 0.0
    spread2 = max(all_probs2) - min(all_probs2) if all_probs2 else 0.0
    
    mean_diff = abs(mean1 - mean2) / 0.9  # Normalize
    spread_diff = abs(spread1 - spread2) / 0.9  # Normalize
    
    similarity = 1.0 - min(1.0, (mean_diff + spread_diff) / 2.0)
    return round_to_precision(similarity)


def _cluster_scenarios(
    scenarios: List[str],
    similarity_matrix: Dict[str, Dict[str, float]],
    similarity_threshold: float = 0.7,
) -> Tuple[Dict[int, List[str]], List[str]]:
    """
    Deterministic clustering of scenarios based on similarity.
    
    Uses a simple distance-based approach:
        - Scenarios with similarity >= threshold are in same cluster
        - Outliers are scenarios with no strong connections
    
    Returns:
        (clusters_dict, outliers_list)
    """
    n = len(scenarios)
    if n == 0:
        return {}, []
    
    # Build adjacency: scenarios that are similar enough
    adjacency = defaultdict(set)
    for i, s1 in enumerate(scenarios):
        for j, s2 in enumerate(scenarios):
            if i != j and similarity_matrix[s1][s2] >= similarity_threshold:
                adjacency[s1].add(s2)
    
    # Find connected components (clusters)
    visited = set()
    clusters = {}
    cluster_id = 0
    
    for scenario in scenarios:
        if scenario in visited:
            continue
        
        # BFS to find connected component
        cluster = []
        queue = [scenario]
        visited.add(scenario)
        
        while queue:
            current = queue.pop(0)
            cluster.append(current)
            
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        # Only create cluster if it has multiple scenarios or strong connections
        if len(cluster) > 1 or any(len(adjacency[s]) > 0 for s in cluster):
            clusters[cluster_id] = sorted(cluster)
            cluster_id += 1
    
    # Identify outliers: scenarios with low average similarity to others
    outliers = []
    for scenario in scenarios:
        similarities = [
            similarity_matrix[scenario][other]
            for other in scenarios
            if other != scenario
        ]
        
        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            if avg_similarity < 0.3:  # Low similarity threshold for outliers
                outliers.append(scenario)
    
    return clusters, outliers


# ==============================================================================
# TASK 3: DIRECTOR REALISM TILE
# ==============================================================================

def build_director_realism_tile(
    constraint_solution: Dict[str, Any],
    coupling_map: Dict[str, Any],
    calibration_console: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build director realism tile for executive summary.
    
    SHADOW-MODE: This function is read-only and provides advisory status only.
    
    Args:
        constraint_solution: Output from solve_realism_constraints
        coupling_map: Output from build_cross_scenario_coupling_map
        calibration_console: Output from build_scenario_calibration_console
    
    Returns:
        Director tile with:
            - status_light: GREEN / YELLOW / RED
            - scenario_outliers: List of outlier names
            - expected_convergence_cycles: Estimated cycles to convergence
            - headline: Neutral summary
    """
    calibration_status = calibration_console.get("calibration_status", "OK")
    realism_pressure = calibration_console.get("realism_pressure", 0.0)
    confidence = constraint_solution.get("confidence", 1.0)
    outliers = coupling_map.get("outliers", [])
    scenarios_adjusted = constraint_solution.get("scenarios_adjusted", 0)
    
    # Determine status light
    status_light = _determine_tile_status_light(
        calibration_status=calibration_status,
        realism_pressure=realism_pressure,
        confidence=confidence,
        outlier_count=len(outliers),
    )
    
    # Estimate convergence cycles
    expected_cycles = _estimate_convergence_cycles(
        scenarios_adjusted=scenarios_adjusted,
        realism_pressure=realism_pressure,
        confidence=confidence,
    )
    
    # Build headline
    headline = _build_tile_headline(
        status_light=status_light,
        scenarios_adjusted=scenarios_adjusted,
        outlier_count=len(outliers),
        expected_cycles=expected_cycles,
    )
    
    return {
        "label": SAFETY_LABEL,
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status_light": status_light,
        "scenario_outliers": sorted(outliers),
        "expected_convergence_cycles": expected_cycles,
        "headline": headline,
        "calibration_status": calibration_status,
        "realism_pressure": round_to_precision(realism_pressure),
        "adjustment_confidence": round_to_precision(confidence),
        "scenarios_adjusted": scenarios_adjusted,
    }


def _determine_tile_status_light(
    calibration_status: str,
    realism_pressure: float,
    confidence: float,
    outlier_count: int,
) -> str:
    """Determine status light for director tile."""
    # RED: Critical conditions
    if calibration_status == "BLOCK" or realism_pressure > 0.7 or confidence < 0.5:
        return "RED"
    
    # YELLOW: Moderate issues
    if calibration_status == "ATTENTION" or realism_pressure > 0.3 or outlier_count > 3:
        return "YELLOW"
    
    # GREEN: Everything else
    return "GREEN"


def _estimate_convergence_cycles(
    scenarios_adjusted: int,
    realism_pressure: float,
    confidence: float,
) -> int:
    """
    Estimate cycles needed for convergence to envelope.
    
    Heuristic: Based on number of adjustments, pressure, and confidence.
    """
    if scenarios_adjusted == 0:
        return 0
    
    # Base cycles per scenario
    base_cycles = 50
    
    # Adjust for pressure (higher pressure = more cycles)
    pressure_multiplier = 1.0 + realism_pressure
    
    # Adjust for confidence (lower confidence = more cycles)
    confidence_multiplier = 2.0 - confidence
    
    estimated = int(
        scenarios_adjusted * base_cycles * pressure_multiplier * confidence_multiplier
    )
    
    # Cap at reasonable maximum
    return min(estimated, 500)


def _build_tile_headline(
    status_light: str,
    scenarios_adjusted: int,
    outlier_count: int,
    expected_cycles: int,
) -> str:
    """Build neutral headline for director tile."""
    if status_light == "GREEN":
        return (
            f"Synthetic universe calibration operational: "
            f"{scenarios_adjusted} scenario(s) adjusted, {outlier_count} outlier(s) identified, "
            f"expected convergence in {expected_cycles} cycles."
        )
    elif status_light == "YELLOW":
        return (
            f"Synthetic universe calibration requires attention: "
            f"{scenarios_adjusted} scenario(s) adjusted, {outlier_count} outlier(s) identified, "
            f"expected convergence in {expected_cycles} cycles."
        )
    else:  # RED
        return (
            f"Synthetic universe calibration blocked: "
            f"{scenarios_adjusted} scenario(s) require adjustment, {outlier_count} outlier(s) detected, "
            f"convergence uncertain."
        )


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def build_complete_constraint_analysis(
    ratchet: Dict[str, Any],
    calibration_console: Dict[str, Any],
    scenario_configs: Dict[str, Dict[str, Any]],
    envelope_bounds: EnvelopeBounds = DEFAULT_BOUNDS,
) -> Dict[str, Any]:
    """
    Build complete constraint analysis combining solver, coupling map, and tile.
    
    This is the main entry point for generating a full constraint analysis.
    
    SHADOW-MODE: All outputs are advisory-only.
    """
    # Solve constraints
    constraint_solution = solve_realism_constraints(
        ratchet=ratchet,
        calibration_console=calibration_console,
        scenario_configs=scenario_configs,
        envelope_bounds=envelope_bounds,
    )
    
    # Build coupling map
    coupling_map = build_cross_scenario_coupling_map(scenario_configs)
    
    # Build director tile
    director_tile = build_director_realism_tile(
        constraint_solution=constraint_solution,
        coupling_map=coupling_map,
        calibration_console=calibration_console,
    )
    
    return {
        "label": SAFETY_LABEL,
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "constraint_solution": constraint_solution,
        "coupling_map": coupling_map,
        "director_tile": director_tile,
    }


def format_director_tile(tile: Dict[str, Any]) -> str:
    """Format director tile as human-readable text."""
    lines = [
        f"\n{SAFETY_LABEL}",
        "",
        "=" * 60,
        "DIRECTOR REALISM TILE",
        "=" * 60,
        "",
    ]
    
    # Status light
    status_light = tile.get("status_light", "UNKNOWN")
    lines.append(f"Status Light: [{status_light}]")
    lines.append("")
    
    # Headline
    headline = tile.get("headline", "")
    lines.append("HEADLINE")
    lines.append("-" * 40)
    lines.append(f"  {headline}")
    lines.append("")
    
    # Key metrics
    lines.append("KEY METRICS")
    lines.append("-" * 40)
    lines.append(f"  Scenarios Adjusted:  {tile.get('scenarios_adjusted', 0)}")
    lines.append(f"  Outliers:            {len(tile.get('scenario_outliers', []))}")
    lines.append(f"  Expected Cycles:     {tile.get('expected_convergence_cycles', 0)}")
    lines.append(f"  Confidence:          {tile.get('adjustment_confidence', 0.0):.1%}")
    lines.append(f"  Realism Pressure:    {tile.get('realism_pressure', 0.0):.1%}")
    lines.append("")
    
    # Outliers
    outliers = tile.get("scenario_outliers", [])
    if outliers:
        lines.append("SCENARIO OUTLIERS")
        lines.append("-" * 40)
        for outlier in outliers:
            lines.append(f"  - {outlier}")
        lines.append("")
    
    lines.append("=" * 60)
    lines.append(f"{SAFETY_LABEL}")
    lines.append("")
    
    return "\n".join(lines)


# ==============================================================================
# TASK 1: RATCHET INTEGRATION STUB
# ==============================================================================

def analyze_realism_from_ratchet(
    ratchet: Dict[str, Any],
    console: Dict[str, Any],
    scenario_configs: Dict[str, Dict[str, Any]],
    envelope_bounds: EnvelopeBounds = DEFAULT_BOUNDS,
) -> Dict[str, Any]:
    """
    Convenience entrypoint for analyzing realism constraints from ratchet/console.
    
    This is a thin wrapper around build_complete_constraint_analysis() that
    keeps all logic in the constraint_solver module.
    
    SHADOW-MODE: This function is advisory-only and does not enforce adjustments.
    
    Args:
        ratchet: Output from build_synthetic_realism_ratchet
        console: Output from build_scenario_calibration_console
        scenario_configs: Dict mapping scenario name to full config
        envelope_bounds: Envelope bounds to target (defaults to DEFAULT_BOUNDS)
    
    Returns:
        Complete constraint analysis with:
            - constraint_solution: Parameter adjustment plans
            - coupling_map: Cross-scenario similarity and clustering
            - director_tile: Executive summary tile
    """
    return build_complete_constraint_analysis(
        ratchet=ratchet,
        calibration_console=console,
        scenario_configs=scenario_configs,
        envelope_bounds=envelope_bounds,
    )


# ==============================================================================
# FIRST LIGHT REALISM ANNEX
# ==============================================================================

# NOTE: First Light Evidence Annex (Not a Gate)
# ----------------------------------------------
# This annex is a compact snapshot for the First Light evidence pack only.
# It is NOT a gate or escalation mechanism. All escalation decisions remain
# in the upstream governance systems (ratchet, calibration console, etc.).
# This annex provides auditors with a quick "one-pager" view of realism
# constraint status, but does not influence any control flow or decisions.

def build_first_light_realism_annex(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compress realism analysis into a compact tile for the First Light evidence pack.
    
    This is a First Light evidence annex only, not a gate. All escalation decisions
    remain elsewhere. This annex provides auditors with a compact view of realism
    constraint status for interpretation purposes only.
    
    SHADOW-MODE: This function is read-only and provides advisory status only.
    No effect on ratchet or uplift decisions.
    
    Args:
        analysis: Output from build_complete_constraint_analysis()
    
    Returns:
        Compact annex tile with:
            - schema_version: "1.0.0"
            - status_light: GREEN / YELLOW / RED
            - global_pressure: Realism pressure value [0.0, 1.0]
            - outliers: List of outlier scenario names
            - overall_confidence: Adjustment confidence score [0.0, 1.0]
    
    Interpretation Patterns:
        - GREEN + low pressure (< 0.3) + high confidence (> 0.7):
          "Twin realism looks consistent with scenario assumptions."
        
        - RED + high pressure (> 0.7) + high confidence (> 0.7):
          "Model realism likely out of spec; treat as high-priority investigation."
        
        - RED + high pressure + low confidence (< 0.5):
          "Realism constraints indicate issues, but confidence in adjustments is low;
           investigate both the realism violations and the adjustment uncertainty."
        
        - YELLOW + moderate pressure (0.3-0.7) + moderate confidence (0.5-0.7):
          "Realism constraints show moderate divergence; monitor and review."
    
    See docs/system_law/Realism_PhaseX_Spec.md for detailed interpretation guidance.
    """
    director_tile = analysis.get("director_tile", {})
    coupling_map = analysis.get("coupling_map", {})
    
    return {
        "schema_version": "1.0.0",
        "status_light": director_tile.get("status_light", "UNKNOWN"),
        "global_pressure": round_to_precision(director_tile.get("realism_pressure", 0.0)),
        "outliers": coupling_map.get("outliers", []),
        "overall_confidence": round_to_precision(director_tile.get("adjustment_confidence", 0.0)),
    }


# ==============================================================================
# CALIBRATION EXPERIMENT REALISM CARD
# ==============================================================================

# NOTE: Calibration Experiment Realism Card (P5 Dashboard)
# --------------------------------------------------------
# This card shows, per calibration experiment (CAL-EXP), how "realistic" the
# Twin looks relative to the Real adapter. It is a compact dashboard card for
# per-experiment realism assessment.
#
# This is purely observational; no gating semantics. All escalation decisions
# remain in upstream governance systems.

def build_cal_exp_realism_card(
    cal_id: str,
    annex: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a compact realism card for a calibration experiment.
    
    Shows per CAL-EXP how "realistic" the Twin looks relative to the Real adapter.
    This is a dashboard card for per-experiment realism assessment.
    
    SHADOW-MODE: This function is read-only and provides advisory status only.
    No effect on calibration decisions or gating.
    
    Args:
        cal_id: Calibration experiment identifier
        annex: Output from build_first_light_realism_annex() or equivalent structure
    
    Returns:
        Compact card with:
            - schema_version: "1.0.0"
            - cal_id: Calibration experiment identifier
            - status_light: GREEN / YELLOW / RED
            - global_pressure: Realism pressure value [0.0, 1.0]
            - overall_confidence: Adjustment confidence score [0.0, 1.0]
    """
    return {
        "schema_version": "1.0.0",
        "cal_id": cal_id,
        "status_light": annex.get("status_light", "UNKNOWN"),
        "global_pressure": round_to_precision(annex.get("global_pressure", 0.0)),
        "overall_confidence": round_to_precision(annex.get("overall_confidence", 0.0)),
    }


# ==============================================================================
# TASK 2: EVIDENCE ATTACHMENT
# ==============================================================================

def attach_realism_to_evidence(
    evidence: Dict[str, Any],
    analysis: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach realism constraint analysis to evidence payload.
    
    SHADOW-MODE: Non-mutating function that creates a new evidence dict
    with realism constraints attached. Does not modify the input evidence.
    
    Args:
        evidence: Evidence payload dictionary
        analysis: Output from build_complete_constraint_analysis()
    
    Returns:
        New evidence dictionary with realism constraints attached under:
            evidence["governance"]["realism_constraints"]
        
        Includes:
            - director_tile: Executive summary (status_light, headline, etc.)
            - coupling_summary: Summary of coupling map (outliers, cluster_count)
            - top_adjustments: Top 5 adjustment recommendations (by confidence)
    """
    # Create non-mutating copy
    new_evidence = dict(evidence)
    
    # Ensure governance structure exists
    if "governance" not in new_evidence:
        new_evidence["governance"] = {}
    
    # Create governance copy
    governance = dict(new_evidence["governance"])
    
    # Extract components from analysis
    director_tile = analysis.get("director_tile", {})
    coupling_map = analysis.get("coupling_map", {})
    constraint_solution = analysis.get("constraint_solution", {})
    
    # Build coupling summary (exclude large similarity matrix)
    coupling_summary = {
        "outliers": coupling_map.get("outliers", []),
        "outlier_count": coupling_map.get("outlier_count", 0),
        "cluster_count": coupling_map.get("cluster_count", 0),
        "scenario_count": len(coupling_map.get("similarity_matrix", {})),
    }
    
    # Extract top adjustments (by confidence, limit to 5)
    adjustment_plan = constraint_solution.get("parameter_adjustment_plan", {})
    expected_effects = constraint_solution.get("expected_effects", {})
    top_adjustments = []
    
    # Sort scenarios by confidence (if available) or by number of adjustments
    scenarios_with_adjustments = list(adjustment_plan.keys())
    for scenario in scenarios_with_adjustments[:5]:  # Top 5
        top_adjustments.append({
            "scenario": scenario,
            "adjustments": adjustment_plan[scenario],
            "expected_effect": expected_effects.get(scenario, "No description available"),
        })
    
    # Build realism constraints payload
    realism_constraints = {
        "label": SAFETY_LABEL,
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "director_tile": {
            "status_light": director_tile.get("status_light", "UNKNOWN"),
            "headline": director_tile.get("headline", ""),
            "scenarios_adjusted": director_tile.get("scenarios_adjusted", 0),
            "outliers": director_tile.get("scenario_outliers", []),
            "expected_convergence_cycles": director_tile.get("expected_convergence_cycles", 0),
            "realism_pressure": director_tile.get("realism_pressure", 0.0),
            "adjustment_confidence": director_tile.get("adjustment_confidence", 0.0),
        },
        "coupling_summary": coupling_summary,
        "top_adjustments": top_adjustments,
        "overall_confidence": constraint_solution.get("confidence", 1.0),
        "adjustment_rationale": constraint_solution.get("adjustment_rationale", ""),
        "first_light_annex": build_first_light_realism_annex(analysis),
    }
    
    # Attach to governance
    governance["realism_constraints"] = realism_constraints
    new_evidence["governance"] = governance
    
    return new_evidence


# ==============================================================================
# REALISM VS DIVERGENCE CONSISTENCY CHECK
# ==============================================================================

# NOTE: Realism-Divergence Consistency Check
# ------------------------------------------
# This helper checks consistency between realism pressure (from realism cards)
# and divergence windows (from calibration experiments). It helps reviewers
# identify when realism pressure aligns with high divergence, or when there
# is a conflict (e.g., GREEN realism status but persistently high divergence).
#
# This is purely observational; no gating semantics. All escalation decisions
# remain in upstream governance systems.

def _extract_divergence_rate(window: Dict[str, Any]) -> Tuple[Optional[float], str]:
    """
    Extract divergence rate from a window dictionary, supporting multiple key paths.
    
    Supports the following key paths (in order of precedence):
        - divergence_rate (direct key) → "DIRECT"
        - metrics.divergence_rate (nested under metrics) → "METRICS_NESTED"
        - summary.divergence_rate (nested under summary) → "SUMMARY_NESTED"
    
    Args:
        window: Window dictionary from calibration experiment
    
    Returns:
        Tuple of (divergence_rate, source_path) where:
            - divergence_rate: float [0.0, 1.0] or None if not found
            - source_path: "DIRECT" | "METRICS_NESTED" | "SUMMARY_NESTED" | "MISSING"
    """
    # Try direct key first
    if "divergence_rate" in window:
        value = window["divergence_rate"]
        if isinstance(value, (int, float)):
            return (float(value), SOURCE_PATH_DIRECT)
    
    # Try nested under metrics
    if "metrics" in window and isinstance(window["metrics"], dict):
        if "divergence_rate" in window["metrics"]:
            value = window["metrics"]["divergence_rate"]
            if isinstance(value, (int, float)):
                return (float(value), SOURCE_PATH_METRICS_NESTED)
    
    # Try nested under summary
    if "summary" in window and isinstance(window["summary"], dict):
        if "divergence_rate" in window["summary"]:
            value = window["summary"]["divergence_rate"]
            if isinstance(value, (int, float)):
                return (float(value), SOURCE_PATH_SUMMARY_NESTED)
    
    return (None, SOURCE_PATH_MISSING)


def summarize_realism_vs_divergence(
    realism_card: Dict[str, Any],
    cal_exp_windows: List[Dict[str, Any]],
    *,
    low_divergence_threshold: float = 0.5,
    high_divergence_threshold: float = 0.7,
    persistent_high_divergence_threshold: float = 0.9,
    persistent_window_count: int = 3,
    strict_window_extraction: bool = False,
) -> Dict[str, Any]:
    """
    Check consistency between realism card and divergence windows.
    
    Compares realism pressure (from realism card) with divergence patterns
    (from calibration experiment windows) to identify consistency, tension,
    or conflict.
    
    SHADOW-MODE: This function is read-only and provides advisory status only.
    No effect on calibration decisions or gating.
    
    Args:
        realism_card: Calibration experiment realism card from build_cal_exp_realism_card()
        cal_exp_windows: List of window dictionaries from calibration experiment.
            Each window should have divergence_rate accessible via:
                - divergence_rate (direct key)
                - metrics.divergence_rate (nested under metrics)
                - summary.divergence_rate (nested under summary)
        low_divergence_threshold: Threshold for low divergence (default: 0.5)
        high_divergence_threshold: Threshold for high divergence (default: 0.7)
        persistent_high_divergence_threshold: Threshold for persistent high divergence (default: 0.9)
        persistent_window_count: Number of windows required for persistent high divergence (default: 3)
        strict_window_extraction: If True, set status to INCONCLUSIVE if any window is MISSING (default: False)
    
    Returns:
        Consistency summary with:
            - schema_version: "1.0.0"
            - consistency_status: "CONSISTENT" | "TENSION" | "CONFLICT" | "INCONCLUSIVE"
            - advisory_notes: List of 0-3 neutral advisory notes
            - windows_analyzed: Number of windows successfully analyzed
            - high_divergence_window_count: Number of windows with high divergence
            - divergence_rate_sources: Dict counting windows by source_path
            - windows_dropped_missing_rate: Count of windows with MISSING divergence_rate
            - strict_mode_contract: Dict with enabled, missing_windows_count, status_when_missing
    
    Rules (deterministic):
        - INCONCLUSIVE: (if strict_window_extraction=True and any window is MISSING)
        - CONSISTENT: (status_light GREEN and final_divergence_rate < low_divergence_threshold)
                      OR (status_light RED and final_divergence_rate >= high_divergence_threshold)
        - CONFLICT: status_light GREEN but divergence persistently high
                    (>= persistent_window_count windows with divergence_rate >= persistent_high_divergence_threshold)
        - TENSION: otherwise
    
    Advisory Notes:
        - Generated based on consistency status and patterns
        - Maximum 3 notes
        - Neutral, descriptive language only
    """
    status_light = realism_card.get("status_light", "UNKNOWN")
    global_pressure = realism_card.get("global_pressure", 0.0)
    
    # Extract divergence rates from windows using normalization helper
    if not cal_exp_windows:
        # No windows available - default to TENSION
        return {
            "schema_version": "1.0.0",
            "consistency_status": "TENSION",
            "advisory_notes": [
                "No divergence windows available for comparison."
            ],
            "windows_analyzed": 0,
            "high_divergence_window_count": 0,
            "divergence_rate_sources": {},
            "windows_dropped_missing_rate": 0,
            "strict_mode_contract": {
                "enabled": strict_window_extraction,
                "missing_windows_count": 0,
                "status_when_missing": "INCONCLUSIVE",
            },
        }
    
    # Extract rates and track sources
    divergence_rates = []
    source_counts = {
        SOURCE_PATH_DIRECT: 0,
        SOURCE_PATH_METRICS_NESTED: 0,
        SOURCE_PATH_SUMMARY_NESTED: 0,
        SOURCE_PATH_MISSING: 0,
    }
    
    for window in cal_exp_windows:
        rate, source_path_raw = _extract_divergence_rate(window)
        # Coerce to ensure valid source_path (defensive programming)
        source_path = _coerce_source_path(source_path_raw)
        source_counts[source_path] = source_counts.get(source_path, 0) + 1
        
        if rate is not None:
            divergence_rates.append(rate)
    
    windows_analyzed = len(divergence_rates)
    windows_dropped_missing_rate = source_counts[SOURCE_PATH_MISSING]
    
    # Build strict_mode_contract
    strict_mode_contract = {
        "enabled": strict_window_extraction,
        "missing_windows_count": windows_dropped_missing_rate,
        "status_when_missing": "INCONCLUSIVE",
    }
    
    # Check strict mode: if any window is MISSING, set to INCONCLUSIVE
    if strict_window_extraction and windows_dropped_missing_rate > 0:
        return {
            "schema_version": "1.0.0",
            "consistency_status": "INCONCLUSIVE",
            "advisory_notes": [
                f"Strict mode enabled: {windows_dropped_missing_rate} window(s) missing divergence_rate. "
                "Cannot determine consistency status."
            ],
            "windows_analyzed": windows_analyzed,
            "high_divergence_window_count": sum(
                1 for dr in divergence_rates 
                if dr >= persistent_high_divergence_threshold
            ) if divergence_rates else 0,
            "divergence_rate_sources": {k: v for k, v in source_counts.items() if v > 0},
            "windows_dropped_missing_rate": windows_dropped_missing_rate,
            "strict_mode_contract": strict_mode_contract,
        }
    
    if not divergence_rates:
        # Windows present but no divergence_rate fields - default to TENSION
        return {
            "schema_version": "1.0.0",
            "consistency_status": "TENSION",
            "advisory_notes": [
                "Divergence windows present but no divergence_rate values found."
            ],
            "windows_analyzed": 0,
            "high_divergence_window_count": 0,
            "divergence_rate_sources": {k: v for k, v in source_counts.items() if v > 0},
            "windows_dropped_missing_rate": windows_dropped_missing_rate,
            "strict_mode_contract": strict_mode_contract,
        }
    
    final_divergence_rate = divergence_rates[-1] if divergence_rates else 0.0
    high_divergence_window_count = sum(
        1 for dr in divergence_rates 
        if dr >= persistent_high_divergence_threshold
    )
    
    # Determine consistency status
    consistency_status = "TENSION"  # Default
    
    # CONSISTENT: GREEN + low divergence OR RED + high divergence
    if status_light == "GREEN" and final_divergence_rate < low_divergence_threshold:
        consistency_status = "CONSISTENT"
    elif status_light == "RED" and final_divergence_rate >= high_divergence_threshold:
        consistency_status = "CONSISTENT"
    
    # CONFLICT: GREEN but persistently high divergence
    if status_light == "GREEN" and high_divergence_window_count >= persistent_window_count:
        consistency_status = "CONFLICT"
    
    # Generate advisory notes (0-3 notes, neutral language)
    advisory_notes = []
    
    if consistency_status == "CONSISTENT":
        if status_light == "GREEN":
            advisory_notes.append(
                "Realism status GREEN aligns with low final divergence rate."
            )
        else:  # RED
            advisory_notes.append(
                "Realism status RED aligns with high final divergence rate."
            )
    elif consistency_status == "CONFLICT":
        advisory_notes.append(
            f"Realism status GREEN but {high_divergence_window_count} windows show "
            f"high divergence (>= {persistent_high_divergence_threshold}). "
            "This suggests a potential mismatch between "
            "realism assessment and observed divergence patterns."
        )
        if global_pressure < 0.3:
            advisory_notes.append(
                "Low realism pressure combined with high divergence may indicate "
                "divergence patterns not captured by realism constraints."
            )
    else:  # TENSION
        if status_light == "GREEN" and final_divergence_rate >= low_divergence_threshold:
            advisory_notes.append(
                "Realism status GREEN but final divergence rate is moderate to high. "
                "Monitor for trends."
            )
        elif status_light == "RED" and final_divergence_rate < high_divergence_threshold:
            advisory_notes.append(
                "Realism status RED but final divergence rate is low to moderate. "
                "Review realism constraints and divergence metrics."
            )
        elif status_light == "YELLOW":
            advisory_notes.append(
                "Realism status YELLOW indicates moderate divergence; "
                "monitor and review as needed."
            )
    
    # Limit to 3 notes
    advisory_notes = advisory_notes[:3]
    
    return {
        "schema_version": "1.0.0",
        "consistency_status": consistency_status,
        "advisory_notes": advisory_notes,
        "windows_analyzed": windows_analyzed,
            "high_divergence_window_count": high_divergence_window_count,
            "divergence_rate_sources": {k: v for k, v in source_counts.items() if v > 0},
            "windows_dropped_missing_rate": windows_dropped_missing_rate,
            "strict_mode_contract": strict_mode_contract,
        }


# ==============================================================================
# CALIBRATION EXPERIMENT CARDS EVIDENCE ROLL-UP
# ==============================================================================

def attach_realism_cards_to_evidence(
    evidence: Dict[str, Any],
    cards: List[Dict[str, Any]],
    cal_exp_windows_map: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """
    Attach calibration experiment realism cards to evidence payload.
    
    Stores multiple calibration experiment cards under:
        evidence["governance"]["realism_cards"]
    
    Optionally includes divergence_consistency per card if cal_exp_windows_map
    is provided. The map should be: {cal_id: [window1, window2, ...]}
    
    SHADOW-MODE: Non-mutating function that creates a new evidence dict
    with realism cards attached. Purely observational; no gating semantics.
    Does not modify the input evidence.
    
    Args:
        evidence: Evidence payload dictionary
        cards: List of calibration experiment realism cards from build_cal_exp_realism_card()
        cal_exp_windows_map: Optional dictionary mapping cal_id to list of window dictionaries.
            If provided, divergence_consistency will be computed and attached to each card.
    
    Returns:
        New evidence dictionary with realism cards attached under:
            evidence["governance"]["realism_cards"]
        
        Structure:
            {
                "label": SAFETY_LABEL,
                "schema_version": "1.0.0",
                "generated_at": ISO timestamp,
                "cards": [card1, card2, ...],
                "card_count": len(cards),
            }
        
        Each card may optionally include:
            - divergence_consistency: Output from summarize_realism_vs_divergence()
    """
    # Create non-mutating copy
    new_evidence = dict(evidence)
    
    # Ensure governance structure exists
    if "governance" not in new_evidence:
        new_evidence["governance"] = {}
    
    # Create governance copy
    governance = dict(new_evidence["governance"])
    
    # Enhance cards with divergence consistency if windows map provided
    enhanced_cards = []
    for card in cards:
        enhanced_card = dict(card)
        
        # Add divergence consistency if windows available
        if cal_exp_windows_map is not None:
            cal_id = card.get("cal_id")
            if cal_id and cal_id in cal_exp_windows_map:
                windows = cal_exp_windows_map[cal_id]
                consistency = summarize_realism_vs_divergence(card, windows)
                enhanced_card["divergence_consistency"] = consistency
        
        enhanced_cards.append(enhanced_card)
    
    # Build realism cards payload
    realism_cards = {
        "label": SAFETY_LABEL,
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cards": enhanced_cards,
        "card_count": len(enhanced_cards),
    }
    
    # Attach to governance
    governance["realism_cards"] = realism_cards
    new_evidence["governance"] = governance
    
    return new_evidence


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    # Example usage with mock data
    mock_ratchet = {
        "scenario_retention_score": {
            "scenario_a": 0.8,
            "scenario_b": 0.6,
        },
        "stability_class": {
            "scenario_a": "STABLE",
            "scenario_b": "SOFT_DRIFT",
        },
    }
    
    mock_console = {
        "slices_to_recalibrate": ["scenario_a", "scenario_b"],
        "calibration_status": "ATTENTION",
        "realism_pressure": 0.4,
    }
    
    mock_configs = {
        "scenario_a": {
            "parameters": {
                "probabilities": {
                    "baseline": {"class_a": 0.75},
                    "rfl": {"class_a": 0.80},
                },
                "drift": {"mode": "none"},
                "correlation": {"rho": 0.5},
                "variance": {"per_cycle_sigma": 0.10, "per_item_sigma": 0.05},
            }
        },
        "scenario_b": {
            "parameters": {
                "probabilities": {
                    "baseline": {"class_a": 0.50},
                    "rfl": {"class_a": 0.55},
                },
                "drift": {"mode": "cyclical", "amplitude": 0.30, "period": 10},
                "correlation": {"rho": 0.95},
                "variance": {"per_cycle_sigma": 0.20, "per_item_sigma": 0.15},
            }
        },
    }
    
    # Build complete analysis
    analysis = build_complete_constraint_analysis(
        ratchet=mock_ratchet,
        calibration_console=mock_console,
        scenario_configs=mock_configs,
    )
    
    # Print director tile
    print(format_director_tile(analysis["director_tile"]))
    
    # Print adjustment plan summary
    print("\n" + "=" * 60)
    print("ADJUSTMENT PLAN SUMMARY")
    print("=" * 60)
    solution = analysis["constraint_solution"]
    for scenario, adjustments in solution["parameter_adjustment_plan"].items():
        print(f"\n{scenario}:")
        print(f"  Effects: {solution['expected_effects'][scenario]}")
        print(f"  Adjustments: {json.dumps(adjustments, indent=4)}")

