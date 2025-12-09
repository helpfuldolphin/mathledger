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

Must NOT:
    - Produce claims about real uplift
    - Mix synthetic and real data
    - Enforce adjustments (advisory only)

==============================================================================
"""

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Set, Tuple

from experiments.synthetic_uplift.noise_models import SAFETY_LABEL
from experiments.synthetic_uplift.realism_envelope import DEFAULT_BOUNDS, EnvelopeBounds


# ==============================================================================
# CONSTANTS
# ==============================================================================

SCHEMA_VERSION = "constraint_solver_v1.0"

# Rounding precision for parameter adjustments
PARAMETER_PRECISION = 4


# ==============================================================================
# ENUMS
# ==============================================================================

class AdjustmentConfidence(Enum):
    """Confidence level for parameter adjustments."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def round_to_precision(value: float, precision: int = PARAMETER_PRECISION) -> float:
    """Round value to specified decimal precision."""
    return round(value, precision)


def clamp_probability(value: float) -> float:
    """Clamp probability to valid range [0.05, 0.95]."""
    return round_to_precision(max(0.05, min(0.95, value)))


def clamp_correlation(value: float) -> float:
    """Clamp correlation to valid range [0.0, 0.9]."""
    return round_to_precision(max(0.0, min(0.9, value)))


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
        for cls, prob in class_probs.items():
            if isinstance(prob, (int, float)):
                # Clamp to [min_prob, max_prob]
                adjusted[cls] = clamp_probability(prob)
                if adjusted[cls] != prob:
                    adjustments[mode] = adjusted
                    break
        
        if mode in adjustments:
            # Fill in remaining classes
            for cls, prob in class_probs.items():
                if cls not in adjustments[mode]:
                    adjustments[mode][cls] = clamp_probability(prob)
    
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
        delta = drift.get("shock_delta", 0.0)
        if abs(delta) > bounds.max_step_delta:
            adjustments["shock_delta"] = round_to_precision(
                bounds.max_step_delta * (1.0 if delta > 0 else -1.0)
            )
    
    return adjustments if adjustments else {}


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
    for event in rare_events:
        adj_event = event.copy()
        
        # Adjust probability
        prob = event.get("trigger_probability", 0.0)
        if prob > bounds.max_rare_event_probability:
            adj_event["trigger_probability"] = round_to_precision(
                bounds.max_rare_event_probability
            )
        
        # Adjust magnitude
        magnitude = abs(event.get("magnitude", 0.0))
        if magnitude > bounds.max_rare_event_magnitude:
            sign = 1.0 if event.get("magnitude", 0.0) >= 0 else -1.0
            adj_event["magnitude"] = round_to_precision(
                bounds.max_rare_event_magnitude * sign
            )
        
        # Adjust duration
        duration = event.get("duration", 1)
        if duration < bounds.min_rare_event_duration:
            adj_event["duration"] = bounds.min_rare_event_duration
        
        adjusted.append(adj_event)
    
    # Only return if changes were made
    if adjusted != rare_events:
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
# MAIN
# ==============================================================================

if __name__ == "__main__":
    # Example usage
    from experiments.synthetic_uplift.realism_ratchet import (
        build_synthetic_realism_ratchet,
        build_scenario_calibration_console,
    )
    from experiments.synthetic_uplift.synthetic_real_governance import (
        build_synthetic_real_consistency_view,
    )
    from experiments.synthetic_uplift.scenario_governance import (
        build_realism_envelope_timeline,
        summarize_synthetic_realism_for_global_health,
    )
    import json
    from pathlib import Path
    
    # Load registry
    registry_path = Path(__file__).parent / "scenario_registry.json"
    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)
    
    scenario_configs = registry.get("scenarios", {})
    
    # Mock data for ratchet/console
    mock_timeline = build_realism_envelope_timeline([
        {
            "scenario_name": "synthetic_test",
            "envelope_pass": True,
            "violated_checks": [],
            "timestamp": "2025-01-01T00:00:00Z",
        }
    ])
    
    mock_summary = summarize_synthetic_realism_for_global_health(mock_timeline)
    mock_consistency = build_synthetic_real_consistency_view(
        synthetic_timeline=mock_timeline,
        real_topology_health={"envelope_breach_rate": 0.05},
        real_metric_health={},
    )
    
    from experiments.synthetic_uplift.synthetic_real_governance import derive_synthetic_scenario_policy
    mock_policy = derive_synthetic_scenario_policy(mock_consistency, mock_timeline)
    
    ratchet = build_synthetic_realism_ratchet(mock_consistency, mock_summary)
    console = build_scenario_calibration_console(ratchet, mock_policy)
    
    # Build complete analysis
    analysis = build_complete_constraint_analysis(
        ratchet=ratchet,
        calibration_console=console,
        scenario_configs=scenario_configs,
    )
    
    # Print director tile
    print(format_director_tile(analysis["director_tile"]))

